<#
.SYNOPSIS
  Windows/CUDA counterpart of the Mac H40 runner for the Gemma 4 26B-A4B MTP lane.

.DESCRIPTION
  Runs the frozen 48-token request through one named arm and writes a receipt that
  carries everything needed to compare two runs honestly: the emitted token ids and
  their gate verdict, the decode wall split into prefill / assistant / verifier, alpha,
  expert-cache misses, and the host state (free RAM, pagefile, GPU clocks and
  temperature, hard-fault counters) before and after.

  Matched to the Mac harness in PROTOCOL, not in mechanism. The Mac's macOS-specific
  parts -- vm_stat pressure levels, wired-memory ceilings, F_RDADVISE, lsof port
  observation -- have no Windows equivalent and are not faked here. What carries over is
  the part that makes a number mean something: one deterministic request, greedy
  decoding, a token-id gate, an idle-state admission check before the run, and host
  facts recorded beside every measurement.

.NOTES
  Two expectation files, deliberately:

    expected-48-token-ids.mac.json      the frozen Metal/Mac reference. ALWAYS compared
                                        and reported, NEVER the gate -- Windows agreeing
                                        with it is a finding, not an assumption.
    expected-48-token-ids.windows.json  this lane's own plain-decode output, written by
                                        `-Establish`. This IS the gate: MTP is lossless
                                        against the target it actually runs, so every
                                        speculative arm must reproduce it exactly.

  Sequential runs on this laptop drift thermally (a phantom 1.8x win was once
  manufactured that way), so a single run is a receipt, not evidence. Use paired
  alternating arms with cooling between them for any comparison.
#>
[CmdletBinding()]
param(
    # Absolute path to a prebuilt binary. Never builds -- a harness that builds cannot
    # say which source produced the number it reports.
    [string]$Binary = (Join-Path $PSScriptRoot '..\..\..\target\release\camelid.exe'),
    [string]$Model = (Join-Path $PSScriptRoot '..\..\..\models\google_gemma-4-26B-A4B-it-Q4_0.hot'),
    [string]$Cghost = (Join-Path $PSScriptRoot '..\..\..\models\google_gemma-4-26B-A4B-it-Q4_0.cghost'),
    [string]$Assistant = (Join-Path $PSScriptRoot '..\..\..\models\gemma-4-26B-A4B-it-assistant'),
    [string]$Arm = 'mtp-k8',
    [string]$Request = (Join-Path $PSScriptRoot 'request-48-plain.json'),
    [string]$ReceiptRoot = (Join-Path $PSScriptRoot 'runs'),
    [string]$Label = '',
    [int]$ExpertCacheMib = 64,
    # Pre-spawn floor. The touched set is ~9.6 GiB on the 26B row and free RAM at load
    # moves steady decode by more than most changes under test, so this is recorded even
    # when it passes.
    [int]$MinFreeMiB = 6500,
    # Idle gate. The GPU sits near 54 C idle on this box; anything materially above that
    # is a previous run still cooling.
    [int]$MaxIdleTempC = 60,
    # Write this lane's own plain-decode ids as the Windows expectation. Only meaningful
    # with -Arm plain.
    [switch]$Establish,
    # Enable CAMELID_SSER_PROFILE. Adds instrumentation to the measured path -- use it to
    # explain a number, not to produce one.
    [switch]$SserProfile,
    # Proceed even though other heavy processes are running. Records the fact.
    [switch]$AllowBusyBox
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$EXIT_PASS = 0
$EXIT_CHILD = 1
$EXIT_ADMISSION = 2
$EXIT_GATE = 3

function Resolve-Required {
    param([string]$Path, [string]$What)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$What not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).ProviderPath
}

function Get-HostSnapshot {
    param([string]$Phase)
    $os = Get-CimInstance Win32_OperatingSystem
    $mem = Get-CimInstance Win32_PerfRawData_PerfOS_Memory
    $page = @(Get-CimInstance Win32_PageFileUsage)
    $gpu = 'unavailable'
    try {
        $gpu = (& nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits) -join ' '
    } catch {
        $gpu = 'unavailable'
    }
    $fields = @('0', '0', '0', '0', '0', '0', '0')
    if ($gpu -ne 'unavailable') {
        $fields = ($gpu -split ',') | ForEach-Object { $_.Trim() }
    }
    $pageAllocated = 0
    $pageCurrent = 0
    $pagePeak = 0
    foreach ($p in $page) {
        $pageAllocated += [int]$p.AllocatedBaseSize
        $pageCurrent += [int]$p.CurrentUsage
        $pagePeak += [int]$p.PeakUsage
    }
    return [ordered]@{
        phase                  = $Phase
        timestamp              = (Get-Date).ToString('o')
        # `available` is free + standby, i.e. what a new allocation can actually take
        # without paging. That is the Windows analogue of the Mac harness's
        # "reclaimable", and it is the figure the floor gates on. `free_physical` is
        # kept beside it because the gap between them IS the page cache, and the page
        # cache is what makes a cold run cold.
        available_mib          = [int]$mem.AvailableMBytes
        free_physical_mib      = [int]([double]$os.FreePhysicalMemory / 1024.0)
        total_physical_mib     = [int]([double]$os.TotalVisibleMemorySize / 1024.0)
        # Commit charge, the number that actually predicts pagefile growth.
        committed_mib          = [int](([double]$os.TotalVirtualMemorySize - [double]$os.FreeVirtualMemory) / 1024.0)
        pagefile_allocated_mib = $pageAllocated
        pagefile_current_mib   = $pageCurrent
        pagefile_peak_mib      = $pagePeak
        # Cumulative raw counters. Their DELTA across a run is the Windows stand-in for
        # the Mac's swap-in/out gate; it is system-wide and includes ordinary mapped
        # file faults, so it is weaker evidence than vm_stat's. Recorded, not gated.
        hard_faults_pages_in   = [int64]$mem.PagesInputPersec
        hard_faults_pages_out  = [int64]$mem.PagesOutputPersec
        page_reads             = [int64]$mem.PageReadsPersec
        gpu_clock_sm_mhz       = [int]$fields[0]
        gpu_clock_mem_mhz      = [int]$fields[1]
        gpu_temp_c             = [int]$fields[2]
        gpu_power_w            = [double]$fields[3]
        gpu_util_pct           = [int]$fields[4]
        gpu_mem_used_mib       = [int]$fields[5]
        gpu_mem_total_mib      = [int]$fields[6]
    }
}

function Get-BusyProcesses {
    $names = @('camelid', 'cargo', 'rustc', 'llama-server', 'llama-cli', 'llama-bench', 'llama-completion')
    $busy = @()
    foreach ($name in $names) {
        try {
            foreach ($p in (Get-Process -Name $name -ErrorAction Stop)) {
                $busy += ('{0}({1})' -f $p.ProcessName, $p.Id)
            }
        } catch {
            # Not running. The only expected outcome.
        }
    }
    # Comma-wrapped: a bare `return @()` unrolls to $null on the way out, and the caller
    # then trips StrictMode asking an absent object for `.Count`.
    return , $busy
}

# `ConvertFrom-Json` hands a JSON array down the pipeline as ONE object, so `@(...)`
# around it yields an array containing the array. The cast unrolls it properly.
function Read-IdFile {
    param([string]$Path)
    return [int[]](Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
}

# PowerShell 5.1's `Out-File -Encoding utf8` emits a BOM, and `serde_json` rejects a
# leading BOM outright -- so an expectation file written that way would fail the gate it
# is supposed to define, with a parse error rather than a mismatch.
function Write-Utf8NoBom {
    param([string]$Path, [string]$Text)
    [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
}

function Compare-Ids {
    param($Got, $Want)
    if ($null -eq $Want) { return $null }
    $got = @($Got)
    $want = @($Want)
    $limit = [Math]::Min($got.Count, $want.Count)
    for ($i = 0; $i -lt $limit; $i++) {
        if ($got[$i] -ne $want[$i]) {
            return [ordered]@{
                exact_match      = $false
                first_divergence = $i
                got              = $got[$i]
                expected         = $want[$i]
                got_count        = $got.Count
                expected_count   = $want.Count
            }
        }
    }
    if ($got.Count -ne $want.Count) {
        return [ordered]@{
            exact_match      = $false
            first_divergence = $limit
            got              = $null
            expected         = $null
            got_count        = $got.Count
            expected_count   = $want.Count
        }
    }
    return [ordered]@{
        exact_match      = $true
        first_divergence = $null
        got              = $null
        expected         = $null
        got_count        = $got.Count
        expected_count   = $want.Count
    }
}

# --- resolve inputs -------------------------------------------------------------

$Binary = Resolve-Required $Binary 'binary'
$Model = Resolve-Required $Model 'model'
$Cghost = Resolve-Required $Cghost 'cghost'
$Request = Resolve-Required $Request 'request fixture'
$armPath = Resolve-Required (Join-Path $PSScriptRoot ('arms\{0}.json' -f $Arm)) ('arm "{0}"' -f $Arm)
$armSpec = Get-Content -LiteralPath $armPath -Raw | ConvertFrom-Json
if ($armSpec.mtp) {
    $Assistant = Resolve-Required $Assistant 'MTP assistant directory'
}

$macExpectPath = Join-Path $PSScriptRoot 'expected-48-token-ids.mac.json'
$winExpectPath = Join-Path $PSScriptRoot 'expected-48-token-ids.windows.json'
$macExpect = $null
if (Test-Path -LiteralPath $macExpectPath) {
    $macExpect = Read-IdFile $macExpectPath
}
$gateFile = $null
if ((Test-Path -LiteralPath $winExpectPath) -and (-not $Establish)) {
    $gateFile = (Resolve-Path -LiteralPath $winExpectPath).ProviderPath
}

$repoRoot = (& git -C $PSScriptRoot rev-parse --show-toplevel)
$repoCommit = (& git -C $PSScriptRoot rev-parse HEAD)
$sourceDirty = ((& git -C $repoRoot status --porcelain -- src Cargo.toml Cargo.lock build.rs) -join "`n")

$stamp = (Get-Date).ToString('yyyyMMddTHHmmssZ')
$runName = $Arm
if ($Label -ne '') { $runName = '{0}-{1}' -f $Arm, $Label }
$runDir = Join-Path $ReceiptRoot ('{0}-{1}' -f $stamp, $runName)
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$stdoutPath = Join-Path $runDir 'stdout.txt'
$stderrPath = Join-Path $runDir 'stderr.txt'
$receiptPath = Join-Path $runDir 'receipt.json'
$verdictPath = Join-Path $runDir 'verdict.json'

# --- admission ------------------------------------------------------------------

$before = Get-HostSnapshot 'before'
$busy = Get-BusyProcesses
$refusals = @()
if ($busy.Count -gt 0 -and -not $AllowBusyBox) {
    $refusals += ('other heavy processes running: {0} (pass -AllowBusyBox to measure anyway)' -f ($busy -join ', '))
}
if ($before.available_mib -lt $MinFreeMiB) {
    $refusals += ('available RAM {0} MiB is below the {1} MiB floor' -f $before.available_mib, $MinFreeMiB)
}
if ($before.gpu_temp_c -gt $MaxIdleTempC) {
    $refusals += ('GPU is at {0} C, above the {1} C idle gate -- let it cool' -f $before.gpu_temp_c, $MaxIdleTempC)
}
if ($refusals.Count -gt 0) {
    Write-Host '[h40] ADMISSION REFUSED'
    foreach ($r in $refusals) { Write-Host ('  - {0}' -f $r) }
    Write-Utf8NoBom $verdictPath (([ordered]@{
                arm         = $Arm
                admitted    = $false
                refusals    = $refusals
                host_before = $before
            }) | ConvertTo-Json -Depth 8)
    exit $EXIT_ADMISSION
}

# --- environment ----------------------------------------------------------------

$savedEnv = @{}
$armEnv = @{}
foreach ($prop in $armSpec.env.PSObject.Properties) {
    $armEnv[$prop.Name] = [string]$prop.Value
}
if ($SserProfile) { $armEnv['CAMELID_SSER_PROFILE'] = '1' }
foreach ($key in $armEnv.Keys) {
    $savedEnv[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
    [Environment]::SetEnvironmentVariable($key, $armEnv[$key], 'Process')
}

# --- run ------------------------------------------------------------------------

$cliArgs = @(
    'gemma4-cuda-generate',
    ('"{0}"' -f $Model),
    '--cghost', ('"{0}"' -f $Cghost),
    '--expert-cache-mib', $ExpertCacheMib,
    '--request-json', ('"{0}"' -f $Request),
    '--receipt', ('"{0}"' -f $receiptPath)
)
if ($armSpec.mtp) {
    $cliArgs += @('--mtp-assistant', ('"{0}"' -f $Assistant), '--mtp-draft-k', $armSpec.draft_k)
}
if ($null -ne $gateFile) {
    $cliArgs += @('--expect-token-ids', ('"{0}"' -f $gateFile))
}

Write-Host ('[h40] arm {0}: {1}' -f $Arm, $armSpec.description)
Write-Host ('[h40] available {0} MiB (free {1}) | GPU {2} C, {3}/{4} MHz | pagefile {5}/{6} MiB' -f `
        $before.available_mib, $before.free_physical_mib, $before.gpu_temp_c, `
        $before.gpu_clock_sm_mhz, $before.gpu_clock_mem_mhz, `
        $before.pagefile_current_mib, $before.pagefile_allocated_mib)
if ($null -eq $gateFile) {
    Write-Host '[h40] NOTE: no Windows expectation file -- this run is UNGATED. Establish one with `-Arm plain -Establish`.'
}

$runStart = Get-Date
$proc = Start-Process -FilePath $Binary -ArgumentList $cliArgs -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$wallSeconds = ((Get-Date) - $runStart).TotalSeconds
$exitCode = $proc.ExitCode

foreach ($key in $savedEnv.Keys) {
    [Environment]::SetEnvironmentVariable($key, $savedEnv[$key], 'Process')
}

$after = Get-HostSnapshot 'after'

# --- verdict --------------------------------------------------------------------

$receipt = $null
if (Test-Path -LiteralPath $receiptPath) {
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
}
if ($null -eq $receipt) {
    Write-Host ('[h40] child exited {0} without writing a receipt; see {1}' -f $exitCode, $stderrPath)
    Get-Content -LiteralPath $stderrPath -Tail 30 | ForEach-Object { Write-Host ('  | {0}' -f $_) }
    exit $EXIT_CHILD
}

$ids = @($receipt.token_ids)
$vsMac = Compare-Ids $ids $macExpect
$winExpect = $null
if ($null -ne $gateFile) {
    $winExpect = Read-IdFile $gateFile
}
$vsWindows = Compare-Ids $ids $winExpect

if ($Establish) {
    # Built by hand rather than through ConvertTo-Json: that cmdlet renders a
    # single-element array as a bare scalar, which would silently produce an
    # expectation file no run can ever match.
    Write-Utf8NoBom $winExpectPath ('[' + ($ids -join ', ') + ']')
    Write-Host ('[h40] established Windows expectation ({0} ids) -> {1}' -f $ids.Count, $winExpectPath)
}

$verdict = [ordered]@{
    schema           = 'camelid.gemma4.h40.v1'
    arm              = $Arm
    arm_description  = $armSpec.description
    label            = $Label
    admitted         = $true
    binary           = $Binary
    binary_sha256    = (Get-FileHash -LiteralPath $Binary -Algorithm SHA256).Hash.ToLower()
    binary_mtime     = (Get-Item -LiteralPath $Binary).LastWriteTimeUtc.ToString('o')
    harness_commit   = $repoCommit
    # Runtime sources only. A dirty `frontend/` belongs to somebody else and does not
    # make this number unreproducible; a dirty `src/` does.
    source_dirty     = $sourceDirty
    request_sha256   = (Get-FileHash -LiteralPath $Request -Algorithm SHA256).Hash.ToLower()
    env              = $armEnv
    exit_code        = $exitCode
    harness_wall_s   = $wallSeconds
    busy_processes   = $busy
    gate_file        = $gateFile
    vs_windows_expected = $vsWindows
    vs_mac_expected  = $vsMac
    host_before      = $before
    host_after       = $after
    host_delta       = [ordered]@{
        available_mib         = $after.available_mib - $before.available_mib
        free_physical_mib     = $after.free_physical_mib - $before.free_physical_mib
        pagefile_current_mib  = $after.pagefile_current_mib - $before.pagefile_current_mib
        hard_faults_pages_in  = $after.hard_faults_pages_in - $before.hard_faults_pages_in
        hard_faults_pages_out = $after.hard_faults_pages_out - $before.hard_faults_pages_out
        page_reads            = $after.page_reads - $before.page_reads
    }
    receipt          = $receipt
}
Write-Utf8NoBom $verdictPath ($verdict | ConvertTo-Json -Depth 12)

# --- report ---------------------------------------------------------------------

Write-Host ''
Write-Host ('[h40] {0} tokens | load {1:N1} s | decode {2:N2} s | {3:N2} tok/s decode-only' -f `
        $ids.Count, $receipt.load_secs, $receipt.decode_only_secs, $receipt.decode_tokens_per_second)
if ($null -ne $receipt.steady_tokens_per_second) {
    # The whole-run average includes the arena warm-up after prefill, which on a 48-token
    # request is half the run. Quote both or neither.
    Write-Host ('[h40] steady {0:N2} tok/s (2nd half of decode forwards)' -f $receipt.steady_tokens_per_second)
}
if ($null -ne $receipt.mtp) {
    Write-Host ('[h40] alpha {0:N2} over {1} rounds | acceptance {2:N1}%' -f `
            $receipt.mtp.alpha, $receipt.mtp.rounds, ($receipt.mtp.acceptance_rate * 100))
    Write-Host ('[h40] prefill {0:N0} ms | assistant {1:N0} ms | verifier {2:N0} ms ({3:N0} ms/round)' -f `
            $receipt.mtp.prefill_ms, $receipt.mtp.assistant_ms, $receipt.mtp.verify_ms, $receipt.mtp.verify_ms_per_round)
}
if ($null -ne $receipt.expert_cache) {
    $ec = $receipt.expert_cache
    Write-Host ('[h40] expert cache {0}/{1} resident | prefill {2} misses | decode {3} misses = {4:N1}/token, {5:P1} hit' -f `
            $ec.resident_experts, $ec.capacity, $ec.prefill_misses, $ec.decode_misses, `
            $ec.decode_misses_per_token, $ec.decode_hit_rate)
    # One miss is one ~3.19 MiB record off the .cghost at 1.3-1.9 GB/s, so this is the
    # floor a storage-bound round cannot beat.
    Write-Host ('[h40] decode storage: {0:N0} MiB, i.e. >= {1:N0} ms at 1.9 GB/s' -f `
            $ec.decode_miss_mib, ($ec.decode_miss_mib / 1900.0 * 1000.0))
}
Write-Host ('[h40] host delta: available {0:+#;-#;0} MiB | pagefile {1:+#;-#;0} MiB | hard-fault pages in {2}' -f `
        $verdict.host_delta.available_mib, $verdict.host_delta.pagefile_current_mib, `
        $verdict.host_delta.hard_faults_pages_in)

if ($null -ne $vsMac) {
    if ($vsMac.exact_match) {
        Write-Host ('[h40] vs Mac reference: IDENTICAL ({0} ids)' -f $vsMac.got_count)
    } else {
        $at = $vsMac.first_divergence
        $from = [Math]::Max(0, $at - 3)
        Write-Host ('[h40] vs Mac reference: diverges at index {0}; {1} vs {2} ids' -f `
                $at, $vsMac.got_count, $vsMac.expected_count)
        Write-Host ('[h40]   windows[{0}..] {1}' -f $from, (($ids[$from..([Math]::Min($ids.Count - 1, $at + 5))]) -join ' '))
        Write-Host ('[h40]   mac[{0}..]     {1}' -f $from, (($macExpect[$from..([Math]::Min($macExpect.Count - 1, $at + 5))]) -join ' '))
    }
}

$status = $EXIT_PASS
if ($exitCode -ne 0) {
    $status = $EXIT_GATE
    Write-Host ('[h40] FAIL: child exited {0}' -f $exitCode)
} elseif ($null -ne $vsWindows -and -not $vsWindows.exact_match) {
    $status = $EXIT_GATE
    Write-Host '[h40] FAIL: token ids differ from the Windows expectation'
} elseif ($null -ne $gateFile) {
    Write-Host '[h40] PASS: exact against the Windows expectation'
}
Write-Host ('[h40] receipt -> {0}' -f $verdictPath)
exit $status
