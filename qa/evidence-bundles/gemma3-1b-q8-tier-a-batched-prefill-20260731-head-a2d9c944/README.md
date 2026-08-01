# Gemma 3 1B-It Q8_0 — Tier A batched windowed prefill, Phase 2 receipts

Row: `gemma_3_1b_it_q8_0` — `gemma-3-1b-it-Q8_0.gguf`, 1,069,306,368 B.
Branch `feat/gemma3-batched-prefill`, head `a2d9c944`, off `main` at `a5945f8a`.
Long-prompt TTFT campaign, Phase 2 — see `GEMMA3_METAL_CONDUCTOR.md` §17.

**No comparison with any other inference engine is made or implied here.** The pinned
llama.cpp build appears only as a *correctness* oracle for §3 and was deliberately run on its
CPU backend for a stable reduction order, which makes any timing comparison against it
meaningless. The two TTFT files below compare camelid to camelid: the same binary with one
env flag on and off.

## What Tier A is

Batch the weight-streaming half of prefill, opt-in behind `CAMELID_GEMMA3_BATCH_PREFILL`
(default OFF this phase), and hold it to raw bit-identity against the shipped token-by-token
lane. Attention stays per row. The bit-identity gate (G1) is in-source
(`gemma3_real_row_batched_prefill_kv_bit_identical`), not in this bundle.

## 1-2. `ttft-flag-off.json` / `ttft-flag-on.json` — the measurement

Same binary, one server alive at a time, started and killed by saved PID with death verified
(`ps -p`, `pgrep`, port). Streamed SSE, timing request-start to the first chunk carrying
non-empty content — never inferred from a non-streaming total. Prompts are supplied as exact
token-id arrays, so the tokenizer is out of the loop and N is exact. Every request uses a
distinct prompt window, so no two share a prefix; `prompt_cache_hit` was `false` on all 32.
Four interleaved rounds; round 0 is the cold column and is excluded from the mean.

Warm means (rounds 1-3), with the 1-minute load average each request ran under:

| N | flag OFF | sd | flag ON | sd | speedup |
|---:|---:|---:|---:|---:|---:|
| 600 | 7.772 s | 0.116 | 7.231 s | 0.105 | 1.07x |
| 1 200 | 16.314 s | 0.148 | 14.952 s | 0.361 | 1.09x |
| 2 366 | 34.138 s | 0.218 | 30.604 s | 0.180 | 1.12x |
| 2 400 | 35.162 s | 0.602 | 31.232 s | 0.270 | 1.13x |

Load ranged 3.05-4.53 across both columns — comparable, and recorded per request in the JSON.
The campaign's plan projected 2.0-3.6x for Tier A; §17e of the conductor explains the gap with
a mechanism (the batched GEMV's column tile is capped at 8, and its activation re-read is
chunk-invariant), not with a caveat.

## 3. `window-edge-tier-a-parity.json` — correctness with the flag ON

`scripts/chat-parity-gemma3.mjs --mode compare` against the Phase 1 oracle capture
(`../gemma3-1b-q8-window-edge-harness-20260731-head-a82dd41a/oracle-window-edge.json`,
llama.cpp `acd79d603`, CPU backend), new pack, depths 1/5/50, top-2 margins armed, token
identity scored on camelid's own `generated_token_ids`.

- **Cross-engine prompt tokenization identical 24/24.**
- **70/72 generation legs token-AND-text identical**, `all_pass: false`.
- **The failing set is bit-for-bit the Phase 1 baseline's**: `w-len-256` depth 50 diverging at
  generated index 13, and `w-len-513` depth 50 at index 5 — same items, same depths, same
  indices, same margins. Both are unanchored ladder items carrying the open-ended "name one
  item mentioned above" question, both pre-existing, and neither is a window item. They are not
  excused and they are not claimed as fixed.
- **All six anchored window items are clean at every depth**, as in Phase 1.

## 4. `batch-prefill-trace.txt` — where the time goes

The batched path's own per-command-buffer line (`CAMELID_RESIDENT_TRACE=1`) from the serve
process that produced §3. At 256-row chunks: `encode` ~10 ms, `commit_wait` 2.91-3.21 s,
`gpu_busy` 2.89-3.20 s. CPU-side encode is **0.04 ms/token** of a ~12 ms/token cost and
`commit_wait - gpu_busy` is ~0.01 ms/token — the GPU is busy for essentially the whole window,
so the residual is kernel time, not dispatch overhead or CPU starvation.

## Scope

- One row, one quant, one lane: `gemma-3-1b-it-Q8_0` on the macOS Metal GPU-resident lane.
- No claim above 2 433 prompt tokens.
- No decode-throughput claim: Tier A does not touch decode, and none was measured here.
- Timings are from a shared host whose load could not be brought below 2.0; read every absolute
  TTFT as an upper bound and the ratio as the result.

## Environment

Apple M4 Mac mini, 10 cores, 16 GiB RAM. Release build of this branch at `a2d9c944`. One
model-loading process alive at any time throughout.
