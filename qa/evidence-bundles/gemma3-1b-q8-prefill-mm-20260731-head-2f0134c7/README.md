# gemma-3-1b-it-Q8_0 — long-prompt TTFT campaign, Phase 3: the prefill GEMM

Row `gemma_3_1b_it_q8_0`, branch `feat/gemma3-batched-prefill`, head `2f0134c7`.
Host: 16 GB Apple M4 Mac mini, macOS. Full record: `GEMMA3_METAL_CONDUCTOR.md` §18.

Phase 3 replaces the batched-prefill GEMV with the tiled simdgroup-matrix kernel
`q8_0_block_wire_mm` — which is not new code: it is the shipped default prefill GEMM for
every other Q8_0 row on this host (`CAMELID_METAL_MM`, armed by the CLI fast stack).
gemma3 was excluded from it, not lacking it.

**This path is NOT bit-identical to the token-by-token lane** — the dequantized Q8_0 weight
and the activation panel stage in half and the MMA accumulates in tile order. It ships
opt-in behind `CAMELID_GEMMA3_PREFILL_MM` (default OFF), on top of
`CAMELID_GEMMA3_BATCH_PREFILL` (also default OFF), and its gate is the KV-equivalence
envelope published in §18a/§18b, not raw bit equality.

## Files

| file | what |
|---|---|
| `ttft-token-by-token.json` | TTFT sweep, both flags off — today's shipped lane |
| `ttft-batched-gemv.json` | TTFT sweep, `CAMELID_GEMMA3_BATCH_PREFILL=1` — Phase 2 |
| `ttft-batched-mm.json` | TTFT sweep, + `CAMELID_GEMMA3_PREFILL_MM=1` — Phase 3 |
| `gemm-probe.txt` | the three paths in ONE process (n=1200), and the chunk-width sweep |
| `g6-kv-envelope.txt` | gate G6: KV + final-hidden equivalence vs the shipped lane |
| `window-edge-mm-parity.json` | the window-edge pack vs the pinned oracle capture, MM armed |
| `SHA256SUMS` | checksums of every file above |

## Method, and its discards

One `camelid serve` alive at a time, PID saved, killed by that PID, death verified with
`ps -p` + `pgrep` + a port check between legs. Streamed SSE, timed from request start to the
first chunk carrying non-empty content — never inferred from a non-streaming total. Prompts
are exact token-id arrays (`camelid_prompt_token_ids`), so the tokenizer is out of the loop,
and every request takes a distinct prompt window so no two share a prefix
(`prompt_cache_hit` false on all 27 requests). Three rounds per leg; **round 0 is reported as
the cold column and excluded from the mean** — no other data is discarded. Every request
carries its own 1-minute load average, because this host runs other sessions' work and its
documented run-to-run spread is 6-15 %.

All three TTFT legs ran inside 20 minutes of each other at 1-minute load 2.50-3.73.

## Headline

| N | token-by-token | batched GEMV (Phase 2) | **batched MM (Phase 3)** | MM vs GEMV |
|---:|---:|---:|---:|---:|
| 600 | 8.734 s | 6.998 s | **1.304 s** | **5.36x** |
| 1 200 | 18.010 s | 14.450 s | **3.050 s** | **4.74x** |
| 2 400 | 38.174 s | 30.372 s | **7.573 s** | **4.01x** |

At the kernel, n=1200, one process: 17.860 -> 12.235 -> **2.538 ms/token**.

## What is disclosed rather than smoothed

- **The envelope pinned before measuring FAILED, twice.** §18b reproduces both failures with
  their numbers and explains the error (an absolute bound picked as if this row's tensors
  were O(1); they reach 1.4e2 and 3.3e4). The amendment is a relative bound, and the outlier
  factor is unchanged.
- **The KV gate's scalar half narrowed** from bit-identity's infinite separation to 4.3x
  against the weakest recorded window-mutation signature. The outlier half keeps 9.1x.
- **Today's token-by-token column is 8-12 % slower than the Phase 2 bundle's**, taken 2.5
  hours earlier; today's GEMV column is 3-4 % faster than that bundle's. Both are inside the
  documented host spread. Measured against Phase 2's own committed numbers the MM path is
  still 5.5x / 4.9x / 4.1x.
- **The two depth-50 window-edge failures are the Phase 1 ones**, unchanged, reported as
  failures and not excused.

No comparison with any other inference engine is made or implied; the pinned oracle capture
appears only as a correctness reference.
