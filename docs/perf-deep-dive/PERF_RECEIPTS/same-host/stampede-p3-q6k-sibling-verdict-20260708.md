# STAMPEDE P3 Lane B — Q6_K owner sibling: verdict + a routing discovery

## What was built
`q6_k_owner_prefill_tiled` (same flag `CAMELID_X86_KQUANT_MATMUL_OWNER`, same rows≥4 dispatch,
same 2D blocking as the Q4_K owner): hoists the per-cell 256-value 6-bit weight rebuild per
weight row (the default per-cell path re-runs the full scalar rebuild for EVERY cell) and uses
the in-tree bit-identity-proven AVX2 `aux32` lane kernel; the bits-sensitive per-cell f32 shape
(8 lane accumulators, `sums[l] += d·aux32[l]` per superblock, final left-fold) is kept verbatim
per `q6_k_wire_row_dot` — the shape KQUANT_RECON proved must not be restructured. Bitwise twin
tests (owner-vs-block-dot with engaged-check; scalar-vs-AVX2 aux32).

## Measured

| leg | prefill tok/s | vs Q4_K-owner-only (v2) | text |
|---|---|---|---|
| 3B Q4_K_M, owner on (v3 = Q4_K+Q6_K) | 22.19 | 0.99× (flat) | ≡ P0 receipt |
| Qwen3-4B Q4_K_M, owner on (v3) | 16.38 | 0.95× (flat/noise) | ≡ P0 receipt |

**Why flat, and why that's expected in hindsight:** `camelid inspect` shows the 3B Q4_K_M GGUF
carries 168 Q4K vs 29 Q6K tensors — the Q6K ones are per-layer `attn_v` (small) plus the lm head
(which executes at rows==1, outside the owner). Q6_K is ~4% of the batched prefill weight
stream on Q4_K_M rows; a kernel win on 4% is invisible end-to-end.

## The routing discovery (bench-design trap, receipt-worthy)
A pure-Q6_K validation model was minted locally (`llama-quantize --allow-requantize` from the
tinyllama Q8_0 → Q6_K). The A/B on it measured **flat prefill and the owner counter read 0**:
the response's `lane` field says `"experimental"` — a locally-requantized GGUF is not a
supported capability row, so serve routes it through the experimental/generic engine, which
never reaches the native K-quant block-dot kernels at all. **Local requants cannot exercise
native-lane kernels via serve**; any future K-quant kernel benchmark must use a supported row
(receipts `stampede-p3-q6k-owner-{off,on}-tinyllama-q6k-20260708.json` kept as the honest
negative; its decode swing is cold-page-cache noise on the freshly minted file).

## Verdict
Ship as COVERAGE: bitwise-correct, zero-risk (opt-in flag unchanged), completes the owner
across both K-quants used by supported rows, and removes the per-cell rebuild for any future
Q6_K-heavy supported row. No end-to-end speedup claim is made for current rows — the receipts
above say exactly that. The Lane B escalation ladder for real Q4_K_M prefill gains remains:
8-row interleaved repack, AVX-512 VNNI main side.
