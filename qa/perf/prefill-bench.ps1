<#
.SYNOPSIS
  Prompt-processing (prefill) benchmark for the Gemma 4 CUDA ghost lane.

.DESCRIPTION
  Decode throughput was the only thing this lane had ever been measured on, and it is
  the half we were already competitive at. Prompt processing was never measured, and
  it is where the gap lives:

    measured 2026-08-24, RTX 3060 Laptop 6 GiB / 16 GiB RAM, Q4_0
      ~361-token prompt  -> 46.47 s   (7.8 tok/s)
      ~1441-token prompt -> 160.68 s  (9.0 tok/s)
    llama.cpp on comparable hardware: 577 t/s with --n-cpu-moe, 2400-3200 t/s GPU-resident

  Cause: `prefill_reusing_cache` runs one full 30-layer forward per prompt token
  (src/gemma4_runtime.rs:11908). Walking all 30 layers per token overwrites the
  ~889-slot expert arena roughly 3x per token, so the next token returns to layer 0
  and finds its experts already evicted. Serial ordering structurally defeats the cache.

  Prefill time is derived as (total generation wall) - (timed decode wall), both of
  which the harness already reports, with --max-tokens kept small so decode is noise.

.EXAMPLE
  .\qa\perf\prefill-bench.ps1
  .\qa\perf\prefill-bench.ps1 -Label after-chunked -Repeats 2
#>
[CmdletBinding()]
param(
  [string]$Exe        = ".\target\release\camelid.exe",
  [string]$Model      = ".\models\google_gemma-4-26B-A4B-it-Q4_0.hot",
  [string]$Cghost     = ".\models\google_gemma-4-26B-A4B-it-Q4_0.cghost",
  [int]$ExpertCacheMib= 64,
  [int]$MaxTokens     = 8,
  [int]$IdleTempC     = 52,
  [int]$Repeats       = 1,
  [string]$Label      = "baseline",
  # 0/1 = token-major (the original serial loop). >1 = layer-major chunked prefill.
  [int]$ChunkTokens   = 0,
  [switch]$ColdCache,
  [string]$OutDir     = ".\qa\perf\receipts"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# One sentence of ~46 tokens; repeat counts give roughly 3x / 12x / 48x that.
$unit = "The transformer architecture processes sequences using self-attention over learned key and value projections, which lets every position attend to every earlier position in a single step. "
$cases = [ordered]@{ "tiny" = 1; "small" = 3; "medium" = 12; "large" = 48 }

function Clear-FileCache {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return }
  try {
    $fs = New-Object System.IO.FileStream((Resolve-Path $Path).Path, [System.IO.FileMode]::Open, `
      [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite, 4096, [System.IO.FileOptions]0x20000000)
    $b = New-Object byte[] 4096; $null = $fs.Read($b, 0, 4096); $fs.Close()
  } catch {}
}

Write-Host ""
Write-Host "Prefill benchmark  |  label: $Label  |  cache: $(if ($ColdCache) { 'COLD' } else { 'warm' })" -ForegroundColor Cyan
Write-Host ""

$rows = @()
foreach ($name in $cases.Keys) {
  for ($rep = 1; $rep -le $Repeats; $rep++) {
    if ($ColdCache) { Clear-FileCache $Cghost; Clear-FileCache $Model; Start-Sleep -Seconds 2 }
    while ($true) {
      $t = [int](& nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
      if ($t -le $IdleTempC) { break }
      Start-Sleep -Seconds 5
    }
    $prompt = $unit * $cases[$name]
    $log = Join-Path $OutDir ("prefill-{0}-{1}-{2}.log" -f $Label, $name, $rep)

    $script = @"
`$ErrorActionPreference = 'Continue'
`$env:CAMELID_GEMMA4_CUDA_BATCH_EXPERT_COPIES = '1'
`$env:CAMELID_GEMMA4_SPECULATIVE = '0'
`$env:CAMELID_GEMMA4_GHOST_TIER_PREFILL = '1'
`$env:CAMELID_GEMMA4_PREFILL_CHUNK_TOKENS = '$ChunkTokens'
Set-Location '$((Get-Location).Path)'
& '$Exe' 'gemma4-cuda-generate' '$Model' '--cghost' '$Cghost' ``
  '--expert-cache-mib' '$ExpertCacheMib' '--max-tokens' '$MaxTokens' ``
  '--prompt' @'
$prompt
'@ *> '$log'
exit `$LASTEXITCODE
"@
    $tmp = [System.IO.Path]::GetTempFileName() + ".ps1"
    Set-Content -Path $tmp -Value $script -Encoding utf8
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $tmp | Out-Null
    Remove-Item $tmp -ErrorAction SilentlyContinue

    # Native output is host-formatted and hard-wrapped; flatten before matching.
    $flat = (Get-Content $log -Raw) -replace '\s+', ' '
    $g = [regex]::Match($flat, 'generated in ([\d.]+)s')
    $d = [regex]::Match($flat, '([\d.]+)s timed decode wall')
    $c = [regex]::Match($flat, '(\d+) hits / (\d+) misses')
    if (-not ($g.Success -and $d.Success)) { throw "no timing line in $log" }

    $total   = [double]$g.Groups[1].Value
    $decode  = [double]$d.Groups[1].Value
    $prefill = $total - $decode
    $hits    = [int]$c.Groups[1].Value
    $misses  = [int]$c.Groups[2].Value
    # 240 routed lookups per forward (30 layers x top-8). Decode contributes MaxTokens-1.
    $forwards      = [int](($hits + $misses) / 240)
    $promptTokens  = [math]::Max(1, $forwards - ($MaxTokens - 1))
    $rate          = if ($prefill -gt 0) { $promptTokens / $prefill } else { 0 }

    $rows += [pscustomobject]@{
      Case = $name; Rep = $rep; PreTempC = $t
      PromptTokens = $promptTokens; PrefillSec = [math]::Round($prefill, 2)
      PrefillTokPerSec = [math]::Round($rate, 2)
      Misses = $misses; MissesPerToken = [math]::Round($misses / $promptTokens, 1)
      Log = $log
    }
    Write-Host ("  {0,-6} rep{1}  {2,5} tok  prefill {3,7:N2}s  ->  {4,6:N2} tok/s   misses/tok {5,6:N1}" -f `
      $name, $rep, $promptTokens, $prefill, $rate, ($misses / $promptTokens))
  }
}

Write-Host ""
$receipt = Join-Path $OutDir ("prefill-{0}-{1}.json" -f $Label, $(if ($ColdCache) { "cold" } else { "warm" }))
[pscustomobject]@{
  label = $Label; coldCache = [bool]$ColdCache; maxTokens = $MaxTokens
  expertCacheMib = $ExpertCacheMib; rows = $rows
} | ConvertTo-Json -Depth 5 | Set-Content -Path $receipt -Encoding utf8
Write-Host "receipt: $receipt"
