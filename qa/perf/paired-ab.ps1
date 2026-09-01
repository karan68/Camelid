<#
.SYNOPSIS
  Paired-alternating, thermally-gated A/B harness for the Gemma 4 CUDA ghost lane.

.DESCRIPTION
  Sequential GPU A/Bs on this laptop are worthless. A prior campaign measured a
  1.8x win that vanished entirely under paired measurement -- it was thermal
  drift between two runs, not a code change. This harness exists so that never
  happens again.

  What it enforces:
    * ABABAB alternation, >= 6 alternations (3 pairs) by default.
    * An idle gate before EVERY arm: the GPU must be at or below -IdleTempC and
      at or below -IdleClockMhz before the arm starts. It waits, it does not
      proceed hot.
    * Drift rejection: if the pre-arm temperature of the two halves of a pair
      differs by more than -MaxPairDriftC, that PAIR is discarded, not averaged.
    * A free-RAM floor. Host RAM at load moves steady decode by more than most
      of the optimisations under test, so a run started under memory pressure is
      not comparable to one that was not.
    * Reporting of the PAIRED delta with per-pair sign consistency -- 3 of 3
      pairs agreeing is the evidence, not the mean alone.
    * Token-ID identity between arms. Any exact-parity change that alters the
      128 token IDs has failed regardless of what it did to throughput.

.EXAMPLE
  # Compare the current binary against itself (harness self-test: expect ~0%).
  .\qa\perf\paired-ab.ps1 -ArmAName base -ArmBName base

.EXAMPLE
  # Compare two builds.
  .\qa\perf\paired-ab.ps1 `
      -ArmAExe .\target\release\camelid.exe `
      -ArmBExe .\qa\perf\bin\camelid-soa.exe `
      -ArmAName wire -ArmBName soa -Pairs 4
#>
[CmdletBinding()]
param(
  [string]$ArmAExe        = ".\target\release\camelid.exe",
  [string]$ArmBExe        = ".\target\release\camelid.exe",
  [string]$ArmAName       = "A",
  [string]$ArmBName       = "B",
  [hashtable]$ArmAEnv     = @{},
  [hashtable]$ArmBEnv     = @{},

  [string]$Model          = ".\models\google_gemma-4-26B-A4B-it-Q4_0.hot",
  [string]$Cghost         = ".\models\google_gemma-4-26B-A4B-it-Q4_0.cghost",
  [string]$Prompt         = "The capital of France is",
  [int]$MaxTokens         = 128,
  [int]$ExpertCacheMib    = 64,

  [int]$Pairs             = 3,
  [int]$IdleTempC         = 52,
  [int]$IdleClockMhz      = 600,
  [int]$MaxPairDriftC     = 3,
  [int]$MinFreeRamMib     = 4000,
  [int]$IdleTimeoutSec    = 420,
  [string]$OutDir         = ".\qa\perf\receipts",

  # Measure the FLOOR, not the peak. Back-to-back runs leave the 12 GiB .cghost hot in
  # the OS page cache, which is worth ~2x on this lane -- 14.36 vs 18.49 steady tok/s on
  # bit-identical counters. Real users on a 16 GiB machine with a browser open do not
  # have that cache. -ColdCache purges each artifact's cached pages before EVERY arm so
  # both sides start cold and the comparison reflects the regime users are actually in.
  [switch]$ColdCache
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------- helpers ----

function Get-GpuState {
  $raw = & nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader,nounits
  $p = ($raw -split ',') | ForEach-Object { $_.Trim() }
  [pscustomobject]@{ TempC = [int]$p[0]; ClockMhz = [int]$p[1]; MemUsedMib = [int]$p[2] }
}

function Get-FreeRamMib {
  [int]((Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue)
}

function Get-StandbyMib {
  [int]((Get-Counter '\Memory\Standby Cache Normal Priority Bytes').CounterSamples[0].CookedValue/1MB)
}

# Purge one file's cached pages. Opening a handle with FILE_FLAG_NO_BUFFERING makes the
# Windows cache manager drop that file's pages -- measured here as standby 6,652 -> 2,655 MiB
# with Available unchanged, i.e. the pages were released rather than merely reclaimed.
# No admin rights, no effect on any other process's working set.
function Clear-FileCache {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return }
  $NO_BUFFERING = [System.IO.FileOptions]0x20000000
  try {
    $fs = New-Object System.IO.FileStream($Path, [System.IO.FileMode]::Open, `
      [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite, 4096, $NO_BUFFERING)
    $buf = New-Object byte[] 4096
    $null = $fs.Read($buf, 0, 4096)
    $fs.Close()
  } catch {
    Write-Warning "Clear-FileCache could not purge $Path : $($_.Exception.Message)"
  }
}

function Wait-ForIdleGpu {
  param([int]$TempC, [int]$ClockMhz, [int]$TimeoutSec)
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  while ($true) {
    $s = Get-GpuState
    if ($s.TempC -le $TempC -and $s.ClockMhz -le $ClockMhz) { return $s }
    if ($sw.Elapsed.TotalSeconds -ge $TimeoutSec) {
      throw "GPU did not reach the idle gate ($TempC C / $ClockMhz MHz) within $TimeoutSec s; last was $($s.TempC) C / $($s.ClockMhz) MHz. Refusing to measure hot."
    }
    Start-Sleep -Seconds 5
  }
}

function Invoke-Arm {
  param([string]$Exe, [hashtable]$EnvVars, [string]$Label, [int]$Index)

  $pre = Wait-ForIdleGpu -TempC $IdleTempC -ClockMhz $IdleClockMhz -TimeoutSec $IdleTimeoutSec

  $standbyBefore = Get-StandbyMib
  if ($ColdCache) {
    # Purge in dependency order: the routed payload matters most, but the shadow carries
    # the common core read at load and the GGUF can back either.
    Clear-FileCache $Cghost
    Clear-FileCache $Model
    Clear-FileCache ($Model -replace '\.hot$', '.gguf')
    Start-Sleep -Seconds 2
  }
  $standbyAfter = Get-StandbyMib
  $freeRam = Get-FreeRamMib
  if ($freeRam -lt $MinFreeRamMib) {
    throw "Only $freeRam MiB of host RAM available (floor $MinFreeRamMib MiB). Free memory before measuring -- host RAM at load moves steady decode more than most changes under test."
  }

  # Env is applied to the child only, so the two arms cannot leak into each other.
  $childEnv = @{}
  foreach ($k in $EnvVars.Keys) { $childEnv[$k] = [string]$EnvVars[$k] }

  $argList = @(
    "gemma4-cuda-generate", $Model,
    "--cghost", $Cghost,
    "--expert-cache-mib", $ExpertCacheMib,
    "--prompt", $Prompt,
    "--max-tokens", $MaxTokens
  )

  $log = Join-Path $OutDir ("arm-{0}-{1:d2}.log" -f $Label, $Index)
  # Camelid writes its whole report to stderr. In PowerShell 5.1 a native command's
  # stderr comes back as ErrorRecords, so piping it into this scope would trip
  # $ErrorActionPreference = 'Stop' on the very first line. Redirect every stream to
  # the log INSIDE the child instead, and read the file back — the parent never sees
  # a native error stream at all.
  $script = @"
`$ErrorActionPreference = 'Continue'
$(($childEnv.GetEnumerator() | ForEach-Object { "`$env:$($_.Key) = '$($_.Value)'" }) -join "`n")
Set-Location '$((Get-Location).Path)'
& '$Exe' $(($argList | ForEach-Object { "'$_'" }) -join ' ') *> '$log'
exit `$LASTEXITCODE
"@
  $tmp = [System.IO.Path]::GetTempFileName() + ".ps1"
  Set-Content -Path $tmp -Value $script -Encoding utf8
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $tmp | Out-Null
  $armExit = $LASTEXITCODE
  Remove-Item $tmp -ErrorAction SilentlyContinue
  if (-not (Test-Path $log)) { throw "Arm '$Label' run $Index produced no log at $log" }
  $out = Get-Content -Path $log -Raw
  if ($armExit -ne 0) { throw "Arm '$Label' run $Index exited $armExit. See $log" }

  # PowerShell renders a native command's output through the host formatter, which
  # hard-wraps at the console width -- so every line we care about arrives split
  # across several. Collapse all whitespace runs first and match on the flat string.
  $flat = ($out -replace '\s+', ' ')

  # decode-only: X forwards/s all, Y forwards/s steady (...; Z s timed decode wall)
  $m = [regex]::Match($flat, 'decode-only:\s*([\d.]+)\s*forwards/s all,\s*([\d.]+)\s*forwards/s steady.*?([\d.]+)s timed decode wall')
  if (-not $m.Success) { throw "Arm '$Label' run $Index produced no decode-only line. See $log" }

  $ids = [regex]::Match($flat, 'token_ids:\s*\[([^\]]*)\]')
  $idHash = if ($ids.Success) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($ids.Groups[1].Value -replace '\s',''))
    (Get-FileHash -InputStream ([System.IO.MemoryStream]::new($bytes)) -Algorithm SHA256).Hash.Substring(0,16)
  } else { "unknown" }

  $counters = [regex]::Match($flat, 'SSER cache \(lifetime since load\):\s*(\d+) hits / (\d+) misses')

  [pscustomobject]@{
    Arm       = $Label
    Index     = $Index
    AllTps    = [double]$m.Groups[1].Value
    SteadyTps = [double]$m.Groups[2].Value
    WallSec   = [double]$m.Groups[3].Value
    PreTempC  = $pre.TempC
    PreClock  = $pre.ClockMhz
    FreeRam   = $freeRam
    StandbyBeforeMib = $standbyBefore
    StandbyAfterMib  = $standbyAfter
    ColdCache = [bool]$ColdCache
    TokenHash = $idHash
    Hits      = if ($counters.Success) { [int]$counters.Groups[1].Value } else { -1 }
    Misses    = if ($counters.Success) { [int]$counters.Groups[2].Value } else { -1 }
    Log       = $log
  }
}

# ------------------------------------------------------------------- run ----

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host ""
Write-Host "Paired A/B  |  $ArmAName  vs  $ArmBName" -ForegroundColor Cyan
Write-Host ("  gate: <= {0} C and <= {1} MHz idle, >= {2} MiB free RAM, pair drift <= {3} C" -f $IdleTempC, $IdleClockMhz, $MinFreeRamMib, $MaxPairDriftC)
Write-Host ("  cache: {0}" -f $(if ($ColdCache) { "COLD - artifact pages purged before every arm (measuring the floor)" } else { "warm - back-to-back runs keep the .cghost cached (measuring the peak)" })) `
  -ForegroundColor $(if ($ColdCache) { "Cyan" } else { "DarkGray" })
Write-Host ("  {0} pairs = {1} alternations, {2} tokens each" -f $Pairs, ($Pairs*2), $MaxTokens)
Write-Host ""

$results = @()
$kept    = @()

for ($i = 1; $i -le $Pairs; $i++) {
  Write-Host ("pair {0}/{1}" -f $i, $Pairs) -ForegroundColor Yellow

  # Alternate which arm leads, so any residual warm-up bias cancels across pairs.
  if ($i % 2 -eq 1) {
    $a = Invoke-Arm -Exe $ArmAExe -EnvVars $ArmAEnv -Label $ArmAName -Index $i
    $b = Invoke-Arm -Exe $ArmBExe -EnvVars $ArmBEnv -Label $ArmBName -Index $i
  } else {
    $b = Invoke-Arm -Exe $ArmBExe -EnvVars $ArmBEnv -Label $ArmBName -Index $i
    $a = Invoke-Arm -Exe $ArmAExe -EnvVars $ArmAEnv -Label $ArmAName -Index $i
  }
  $results += $a, $b

  $drift = [math]::Abs($a.PreTempC - $b.PreTempC)
  $delta = 100.0 * ($b.SteadyTps - $a.SteadyTps) / $a.SteadyTps
  $verdict = if ($drift -gt $MaxPairDriftC) { "DISCARDED (drift ${drift} C)" } else { "kept" }
  if ($drift -le $MaxPairDriftC) { $kept += [pscustomobject]@{ A = $a; B = $b; DeltaPct = $delta } }

  Write-Host ("    {0,-10} {1,6:N2} steady  ({2} C)   {3,-10} {4,6:N2} steady  ({5} C)   delta {6,7}%   {7}" -f `
    $a.Arm, $a.SteadyTps, $a.PreTempC, $b.Arm, $b.SteadyTps, $b.PreTempC, ("{0:+0.0;-0.0;0.0}" -f $delta), $verdict)
}

Write-Host ""
if ($kept.Count -eq 0) { throw "Every pair was discarded for thermal drift. Let the machine cool and retry." }

$deltas    = $kept.DeltaPct
$meanDelta = ($deltas | Measure-Object -Average).Average
$positive  = @($deltas | Where-Object { $_ -gt 0 }).Count
$sd        = if ($deltas.Count -gt 1) {
  [math]::Sqrt((($deltas | ForEach-Object { [math]::Pow($_ - $meanDelta, 2) }) | Measure-Object -Sum).Sum / ($deltas.Count - 1))
} else { [double]::NaN }

# Parity gate: an exact-parity change must not move the token IDs or the counters.
$hashes    = ($results.TokenHash | Sort-Object -Unique)
$parityOk  = ($hashes.Count -eq 1 -and $hashes[0] -ne "unknown")
$countersOk= (($results.Hits | Sort-Object -Unique).Count -eq 1) -and (($results.Misses | Sort-Object -Unique).Count -eq 1)

Write-Host "RESULT" -ForegroundColor Cyan
Write-Host ("  paired mean delta   {0,8}%  ({1} of {2} pairs positive)" -f ("{0:+0.00;-0.00;0.00}" -f $meanDelta), $positive, $deltas.Count)
if (-not [double]::IsNaN($sd)) {
  Write-Host ("  pair-to-pair sd     {0,7:N2}%" -f $sd)
  if ([math]::Abs($meanDelta) -lt $sd) {
    Write-Host "  NOT SEPARATED: the effect is smaller than the spread between pairs. More pairs, or no effect." -ForegroundColor Yellow
  }
}
Write-Host ("  token-ID identity   {0}" -f $(if ($parityOk) { "PASS (all arms $($hashes[0]))" } else { "FAIL -- arms disagree: $($hashes -join ', ')" })) `
  -ForegroundColor $(if ($parityOk) { "Green" } else { "Red" })
Write-Host ("  hit/miss identity   {0}" -f $(if ($countersOk) { "PASS" } else { "FAIL -- cache behaviour differs between arms" })) `
  -ForegroundColor $(if ($countersOk) { "Green" } else { "Red" })
Write-Host ""

$receipt = Join-Path $OutDir ("paired-{0}-vs-{1}-{2}.json" -f $ArmAName, $ArmBName, $(if ($ColdCache) { "cold" } else { "warm" }))
[pscustomobject]@{
  coldCache = [bool]$ColdCache
  armA = $ArmAName; armB = $ArmBName
  armAExe = $ArmAExe; armBExe = $ArmBExe
  armAEnv = $ArmAEnv; armBEnv = $ArmBEnv
  prompt = $Prompt; maxTokens = $MaxTokens; expertCacheMib = $ExpertCacheMib
  pairsRequested = $Pairs; pairsKept = $deltas.Count
  meanDeltaPct = $meanDelta; pairSdPct = $sd; positivePairs = $positive
  tokenIdentity = $parityOk; counterIdentity = $countersOk
  runs = $results
} | ConvertTo-Json -Depth 6 | Set-Content -Path $receipt -Encoding utf8
Write-Host "receipt: $receipt"
