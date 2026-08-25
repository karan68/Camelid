<#
.SYNOPSIS
  Alternating A/B over two H40 arms, with per-pair deltas and a sign-consistency check.

.DESCRIPTION
  A single run on this box is a receipt, not evidence: thermal drift once manufactured a
  phantom 1.8x win here. This alternates ABAB..., so drift that accumulates over the
  session lands on BOTH arms roughly equally and shows up as scatter rather than as a
  result. It reports the paired delta per pair and refuses to call a winner unless every
  pair agrees on the sign.

  Between runs it waits for the GPU to fall back under the idle gate, so arm B does not
  inherit arm A's heat.

.EXAMPLE
  .\paired-h40.ps1 -ArmA plain -ArmB plain-decodebulk -Pairs 3
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ArmA,
    [Parameter(Mandatory = $true)][string]$ArmB,
    [int]$Pairs = 3,
    [int]$MinFreeMiB = 4500,
    [int]$MaxIdleTempC = 60,
    [int]$CoolSeconds = 45,
    [string]$Label = 'paired'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

function Wait-ForCool {
    param([int]$MaxTempC, [int]$Budget)
    $deadline = (Get-Date).AddSeconds($Budget)
    while ((Get-Date) -lt $deadline) {
        $t = 0
        try { $t = [int]((& nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) | Select-Object -First 1) } catch { break }
        if ($t -le $MaxTempC) { return $t }
        Start-Sleep -Seconds 5
    }
    try { return [int]((& nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) | Select-Object -First 1) } catch { return -1 }
}

function Invoke-Arm {
    param([string]$Arm, [int]$Index)
    $temp = Wait-ForCool -MaxTempC $MaxIdleTempC -Budget $CoolSeconds
    Write-Host ('[paired] {0} run {1} (GPU {2} C)' -f $Arm, $Index, $temp)
    & (Join-Path $PSScriptRoot 'run-h40.ps1') -Arm $Arm -MinFreeMiB $MinFreeMiB `
        -MaxIdleTempC ($MaxIdleTempC + 5) -Label ('{0}-{1}{2}' -f $Label, $Arm, $Index) | Out-Null
    $exit = $LASTEXITCODE
    $dir = Get-ChildItem (Join-Path $PSScriptRoot 'runs') -Directory |
        Sort-Object Name -Descending | Select-Object -First 1
    $verdict = Get-Content (Join-Path $dir.FullName 'verdict.json') -Raw | ConvertFrom-Json
    # A refused run writes a verdict with `admitted = false` and none of the measurement
    # fields. Abort the whole comparison rather than reading through the gap: half a
    # paired series is not a weaker result, it is no result, and continuing would leave
    # the surviving arm looking like one.
    if (-not $verdict.admitted) {
        throw ("[paired] {0} run {1} was refused admission ({2}); aborting the series" -f `
                $Arm, $Index, ($verdict.refusals -join '; '))
    }
    return [ordered]@{
        arm     = $Arm
        exit    = $exit
        exact   = $verdict.vs_windows_expected.exact_match
        decode  = $verdict.receipt.decode_tokens_per_second
        steady  = $verdict.receipt.steady_tokens_per_second
        misses  = $verdict.receipt.expert_cache.decode_misses
        avail   = $verdict.host_before.available_mib
        temp    = $verdict.host_before.gpu_temp_c
        dir     = $dir.Name
    }
}

$rows = @()
for ($i = 1; $i -le $Pairs; $i++) {
    $rows += Invoke-Arm -Arm $ArmA -Index $i
    $rows += Invoke-Arm -Arm $ArmB -Index $i
}

Write-Host ''
# .NET alignment is a signed width -- positive right-aligns. `{1,>8}` is a parse error,
# not a right-align hint.
Write-Host ('{0,-22} {1,8} {2,8} {3,8} {4,7} {5,6} {6}' -f 'arm', 'decode', 'steady', 'misses', 'avail', 'gpuC', 'exact')
foreach ($r in $rows) {
    Write-Host ('{0,-22} {1,8:N2} {2,8:N2} {3,8} {4,7} {5,6} {6}' -f `
            $r.arm, $r.decode, $r.steady, $r.misses, $r.avail, $r.temp, $r.exact)
}

# Paired deltas: pair i is (A_i, B_i), the two runs closest in time and therefore in
# machine state. Comparing means across arms instead would let a monotone drift masquerade
# as an effect.
$deltas = @()
for ($i = 0; $i -lt $Pairs; $i++) {
    $a = $rows[$i * 2]
    $b = $rows[$i * 2 + 1]
    if ($a.steady -gt 0) {
        $deltas += (($b.steady - $a.steady) / $a.steady)
    }
}
Write-Host ''
if ($deltas.Count -eq 0) {
    Write-Host '[paired] no steady rates to compare (a speculative arm has none)'
    exit 0
}
$mean = ($deltas | Measure-Object -Average).Average
$positive = @($deltas | Where-Object { $_ -gt 0 }).Count
Write-Host ('[paired] steady delta {0} vs {1}: {2:P1} mean over {3} pairs ({4})' -f `
        $ArmB, $ArmA, $mean, $deltas.Count, (($deltas | ForEach-Object { '{0:P1}' -f $_ }) -join ', '))
if ($positive -eq $deltas.Count -or $positive -eq 0) {
    Write-Host '[paired] sign is consistent across every pair'
} else {
    Write-Host '[paired] SIGN FLIPS between pairs -- this is drift, not a result'
}
$notExact = @($rows | Where-Object { -not $_.exact }).Count
if ($notExact -gt 0) {
    Write-Host ('[paired] WARNING: {0} run(s) did not reproduce the expected token ids' -f $notExact)
    exit 3
}
