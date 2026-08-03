# gemma-3-1b-it-Q8_0 — long-prompt TTFT campaign, Phase 4: batched windowed attention

Row `gemma_3_1b_it_q8_0`, branch `feat/gemma3-batched-prefill`, head `95429a42`.
Host: 16 GB Apple M4 Mac mini, macOS. Full record: `GEMMA3_METAL_CONDUCTOR.md` §19.

Phase 3 left prefill **79 % attention at 1 200 prompt tokens and 84 % at 2 400** — the weight
GEMMs had collapsed, but every row still ran its own `encode_attention` over its whole sliding
window. Phase 4 replaces that per-row loop with the attention-as-matmul chain the tree already
ships for non-windowed rows (`half_mm_batched_f16o` + `softmax_causal_rows` + `transpose_v16`),
and teaches two of those kernels the per-query-row **lower** mask bound they have never had. The
new MSL is one `window` uniform and about fifteen lines; `window = 0` is the disabled case, so
the 4 global gemma3 layers and every non-gemma3 caller run the identical arithmetic.

**This path is NOT bit-identical to the token-by-token lane** — Q/K/V and the score and
probability panels stage in half and the MMA accumulates in tile order. Its gate is the
KV-equivalence envelope published in §18a/§18b, **carried over unchanged**, plus an exact-count
mask gate. Decode is not on this path and is asserted unchanged.

This is also the phase that flips the campaign's three flags to **default ON** with `=0` as the
operator opt-out, for the gemma3 Q8_0 row only.

## Files

| file | what |
|---|---|
| `g6-kv-envelope.txt` | gate G6: KV + final-hidden equivalence vs the shipped token-by-token lane, n = 5/256/257/513/1024/2400 |
| `g6-kv-envelope-first-run-FAILED.txt` | the SAME gate's first run, which **failed** at n = 257/513/2400 and found a real bug. Kept deliberately — see below |
| `attn-probe.txt` | both attention paths in one process, same GEMM pinned on both sides, plus the dispatch grid's tile census |
| `gates.txt` | the decode bit-exactness gates, the Tier A bit-identity gates, the session-level oracle gate and the kernel-level mask gate, in run order |
| `window-mutation-phase4.json` | gate G7 re-run with Tier B in the tree — all 7 window mutants caught |
| `window-edge-attn-mm-parity.json` | the window-edge pack vs the pinned oracle capture, Tier B armed |
| `caprow-windowed-attn-mm.json` | the capability row's own 9/9 windowed claim, re-run under the NEW default |
| `caprow-sub512-attn-mm.json` | the capability row's own 15/15 sub-512 claim, re-run under the NEW default |
| `default-posture-trace.txt` | proof the shipped default actually reaches Tier B (`70 gemm=mm attn=mm`) |
| `ttft-attn-mm.json` | TTFT sweep, all three flags on — Phase 4 |
| `ttft-mm.json` | TTFT sweep, `CAMELID_GEMMA3_PREFILL_ATTN_MM=0` — Phase 3, measured in the same session |
| `ttft-token-by-token.json` | TTFT sweep, `CAMELID_GEMMA3_BATCH_PREFILL=0` — the pre-campaign lane, measured in the same session |
| `manifest.json` | machine-readable index of the above |
| `SHA256SUMS` | checksums of every file above |

## Method, and its discards

One `camelid serve` alive at a time, PID saved, killed by that PID, death verified with `ps -p`
plus a port check between legs. Streamed SSE, timed from request start to the first chunk
carrying non-empty content — never inferred from a non-streaming total. Prompts are exact token-id
arrays (`camelid_prompt_token_ids`), so the tokenizer is out of the loop, and every request takes
a distinct prompt window so no two share a prefix (`prompt_cache_hit` false on all requests).
Three rounds per leg; **round 0 is reported as the cold column and excluded from the mean** — no
other data is discarded. Every request carries its own 1-minute load average, because this host
runs other sessions' work and its documented run-to-run spread is 6-15 %.

The in-process probe and every gate ran as a single model-loading process at a time, never
concurrently, on a 16 GB machine that has crashed from concurrent model loads.

## Headline

TTFT, warm mean of rounds 1-2, streamed, prompts as exact token-id arrays:

| N | token-by-token | Phase 3 (per-row attention) | **Phase 4 (batched attention)** | vs Phase 3 | vs token-by-token |
|---:|---:|---:|---:|---:|---:|
| 600 | 8.078 s | 1.316 s | **0.520 s** | **2.53x** | **15.54x** |
| 1 200 | 17.453 s | 3.101 s | **0.971 s** | **3.19x** | **17.97x** |
| 2 400 | 37.475 s | 7.703 s | **1.962 s** | **3.93x** | **19.10x** |

Per-token prefill cost stopped growing with prompt length: 0.867 / 0.809 / 0.817 ms/token,
against Phase 3's 2.193 / 2.585 / 3.209. At the kernel, with the same GEMM pinned on both sides
so only attention differs, per-row 3.019 s -> batched 0.916 s at 1 200 tokens and 7.556 s ->
1.821 s at 2 400. The window cull drops 46.2 % of the score-tile grid at 2 400 tokens.

## Gate results

| gate | result |
|---|---|
| G6, KV envelope (bounds carried over from §18b, unmoved) | PASS at n = 5/256/257/513/1024/2400; worst cache 1.369e-3 relative (bound 2.0e-3), worst outlier ratio 5.8x (bound 8.0), worst final-hidden 2.557e-3 (bound 1.0e-2) |
| G5, the mask at the kernel | PASS; attended count exactly min(q+1, window) at windows 0/1/37/64/65/128, boundary pair exact, max &#124;P - cpu_ref&#124; 5.949e-5, q_offset blocking bit-identical |
| G7, 7-mutant window harness | PASS 7/7, survivors empty |
| Window-edge pack vs the pinned oracle | 70/72 legs, prompt tokenization 24/24 — the Phase 1/2/3 baseline exactly |
| Capability row's own claim, re-run under the NEW DEFAULT | windowed 9/9 all_pass true, sub-512 15/15 all_pass true |
| Default posture actually engages | `70 gemm=mm attn=mm`, zero `attn=row`, with no campaign env vars set |
| Decode, bit-exact | 50/50 greedy tokens at exactly 2.122e-4; split3 decode attention raw-bit identical |
| Tier A bit-identity | still bit-identical, synthetic and real row |
| fmt / clippy -D warnings / test --all-targets / scrub | clean; 60 green targets, 1 817 passed, 0 failed |

## What is disclosed rather than smoothed

- **The first G6 run FAILED, and it failed because the code was wrong.** It is committed here
  verbatim. `softmax_causal_rows` was handed `rows_per_block = k` (this chunk's row count) while
  the two GEMMs were handed the panel's true row pitch, so for any chunk narrower than the panel
  every head above head 0 read the wrong offset. It reproduced at exactly the lengths with a
  ragged last chunk — 257, 513, 2 400 — and passed at 5, 256 and 1 024. A sweep of round numbers
  would not have caught it, and nothing in the tree except the KV-equivalence gate would have
  named the layer, head, position and dim.
- **The KV gate's scalar half narrowed again**, from Phase 3's 7.1x / 14.3x / 44.6x separation
  against the weakest recorded window mutation to **5.7x / 8.6x / 22.9x** at n = 513 / 1 024 /
  2 400. The per-position outlier half still carries 25x-to-infinite. Both are real gates;
  neither is what bit-identity was.
- **The envelope was NOT widened.** Every bound is the one pinned and committed before Phase 3
  measured anything. It could not have been widened honestly: §18b already recorded the scalar
  bound sitting only 1.5x below the weakest recorded window mutation.
- **Two window-edge pack legs fail**, and they are the same two that have failed since Phase 1 —
  reported, not excused. See `window-edge-attn-mm-parity.json`.

## What this bundle does NOT claim

- No throughput or tokens-per-second number for this row is promoted to any shipped surface. The
  capability row's `performance_measured` stays `not_claimed_*`; these are campaign receipts, not
  a release claim.
- No comparison with any other inference engine is made or implied; the pinned oracle capture
  appears only as a correctness reference.
- Nothing here says the prefill is bit-identical to the token-by-token lane. It is not, by
  construction, and the numbers that quantify the distance are in `g6-kv-envelope.txt`.
- Context above 2 403 prompt tokens remains unmeasured for this row, exactly as before.

## Reproduce

```
# gates (each one process, never concurrent)
CAMELID_METAL_F32Y=1 CAMELID_METAL_WIRE=1 CAMELID_METAL_WIRE_NSG8=1 \
CAMELID_GEMMA3_GGUF=<models>/gemma-3-1b-it-Q8_0.gguf \
  cargo test --release --lib gemma3_real_row_prefill_attn_mm_kv -- --nocapture
CAMELID_METAL_F32Y=1 CAMELID_METAL_WIRE=1 CAMELID_METAL_WIRE_NSG8=1 \
CAMELID_GEMMA3_GGUF=<models>/gemma-3-1b-it-Q8_0.gguf \
  cargo test --release --lib gemma3_real_row_prefill_attn_probe -- --nocapture
CAMELID_METAL_F32Y=1 CAMELID_METAL_WIRE=1 CAMELID_METAL_WIRE_NSG8=1 \
  cargo test --release --lib windowed_attn_mm_mask -- --nocapture
CAMELID_METAL_F32Y=1 CAMELID_METAL_WIRE=1 CAMELID_METAL_WIRE_NSG8=1 \
CAMELID_GEMMA3_GGUF=<models>/gemma-3-1b-it-Q8_0.gguf \
CAMELID_GEMMA3_MUTATION_OUT=window-mutation-phase4.json \
  cargo test --release --lib gemma3_real_row_window_mutation -- --nocapture

# window-edge pack vs the pinned oracle capture (camelid serve only; the oracle is replayed)
node scripts/chat-parity-gemma3.mjs --mode compare --camelid http://127.0.0.1:8431 \
  --oracle qa/evidence-bundles/gemma3-1b-q8-window-edge-harness-20260731-head-a82dd41a/oracle-window-edge.json \
  --model-id "Gemma 3 1b It" --row-id gemma_3_1b_it_q8_0 \
  --lane-label gemma3_marker_chat_greedy_metal_resident_serve_attn_mm_prefill \
  --top-logprobs 2 --out window-edge-attn-mm-parity.json
```

## Environment

`CAMELID_STREAM_TIMING_DIAGNOSTICS=1` on every served leg. Flag postures per leg are in each
`ttft-*.json`'s `tag`, and in the Files table above. The oracle is the committed capture from
`qa/evidence-bundles/gemma3-1b-q8-window-edge-harness-20260731-head-a82dd41a/`; llama-server was
never running concurrently with a camelid server.
