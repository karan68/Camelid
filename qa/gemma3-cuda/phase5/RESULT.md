# Phase 5 — the gemma4 CUDA lane's Q4_0 mis-decode: root cause, fix, receipt

Phase 3 found the gemma4 CUDA-resident lane decoding `gemma-4-E2B-it-Q4_0` as
`passe dép oficialmenteynam shalthapp lenghtynam` where the CPU runtime said
`Paris`, and pinned admission to Q8_0 until the Q4_0 path earned a parity receipt.
This is that receipt. **The Q4_0 projections were never the defect.**

## Root cause: the tied head uploaded raw wire into kernels that read a repacked layout

`Gemma4CudaResident::load` picks a head lane from `token_embd`'s format and uploads
the weight. The Q8_0 arm repacked to SoA; the **Q4_K and Q6_K arms `clone_htod`'d the
raw GGUF wire**. Neither of those kernels reads the stock wire:

| Lane | What the GEMV indexes | What it was fed |
|---|---|---|
| `q8_gemv` | SoA split (`q8_wire_to_soa`) | ✅ SoA |
| `q4k_gemv` | quant-byte **swizzle** (`swz_q4k_blocks`) — each aux lane's four stride-8 bytes as one aligned `i32` for `__dp4a` | ❌ raw 144 B wire |
| `q6k_gemv` | **224 B padded** stride (`pad_q6k_blocks`) | ❌ raw 210 B wire |

Every other resident lane in the tree routes through `cuda_resident::repack_for_lane`,
which applies exactly these. The gemma4 head bypassed it.

A Q4_0-quantized gemma4 export carries a **Q4_K `token_embd`**. So the E2B Q4_0 row ran
`q4k_gemv` over unswizzled bytes: correctly-addressed but **wrongly-paired** nibbles.
The hidden states were fine and only the logits were garbage — which is exactly why the
lane produced fluent-looking nonsense rather than refusing.

The Q6_K arm was the same defect one step worse: a 210-vs-224 stride mismatch also reads
~22 MB past the end of the allocation.

### Why only this row was ever caught

Head lane is a property of `token_embd`, which no admission check looked at:

| Row | `token_embd` → head lane | Before |
|---|---|---|
| E4B Q8_0 — *the bring-up row* | Q8_0 → `Q8_0` | ✅ correct |
| E2B Q8_0 | Q8_0 → `Q8_0` | ✅ correct |
| **E2B Q4_0** | **Q4_K → `Q4K`** | ❌ **garbage** |
| 26B Q4_0 | Q6_K → `Q6K` | ❌ broken (never admitted — VRAM fit) |
| E4B NVFP4 | Q6_K → `Q6K` | ❌ broken (never admitted) |

**The only gemma4 row ever validated on this lane is the one whose head happened to be
Q8_0.** Every row with a K-quant head was broken.

Note also what did *not* catch this: `q4k_gemv` has a passing bit-parity unit test, and so
does `q4_0_gemv`. Both kernels were correct the whole time. Kernel parity is not row parity.

## The fix

All three head lanes now route through one function, `gemma4_runtime::gemma4_head_upload`,
mirroring `repack_for_lane`. Two kernel parameter comments that asserted "raw 210-byte Q6_K
wire" / "RAW 144-byte Q4_K wire" — contradicting their own `WIRE` constants, and the most
likely reason the head was wired this way — were corrected.

`gemma4_head_upload_matches_each_lane_gemv_layout` pins each lane's layout with explicit
index arithmetic rather than by calling the repack helpers back. Verified both directions:
**fails** with raw upload restored, **passes** with the fix.

## Receipt: greedy parity, CUDA-resident vs CPU gemma4 runtime

`gemma-4-E2B-it-Q4_0.gguf`, greedy, gemma chat template, one engine resident at a time.
Capture: `run-q4_0-parity.sh` → `q4_0-parity.json`.

| Prompt | Token-identical |
|---|---|
| Name the capital of France in one word. | ✅ |
| What color is the sky on a clear day? | ✅ |
| Name the largest ocean on Earth. | ✅ |
| What is 2 + 2? | ✅ |
| List three primary colors. | ✅ |

**5/5 legs token-identical (`all_pass: true`)** — full token streams, not just first-token
argmax. The in-tree `gemma4_cuda_matches_cpu_greedy` also passes on this row.

## End-to-end through serve, the surface the defect was reported on

`e2b-q4_0-cuda-admitted-health.json`:

```
selected_backend : gemma4_cuda_resident_runtime
decode_path      : gemma4_cuda_resident_decode
quant_type       : Q4_0
VRAM in use      : 1557 MiB     (vs 107 MiB context-only when it fell back)
```

| | Output |
|---|---|
| Phase 3, CUDA | `passe dép oficialmenteynam shalthapp lenghtynam` |
| **Phase 5, CUDA** | **`Paris`** |

Three further prompts answered coherently on the resident lane.

## Admission, relaxed — and narrowed

`gemma4_cuda_lane_admitted`'s quant arm is now `gemma4_projection_quant_admitted`, split
out so it is testable without a CUDA host. It keys on the **layer projections the lane
actually GEMVs**, admitting only formats with an end-to-end row receipt:

- **Admitted:** Q8_0 (E4B bring-up), **Q4_0 and Q4_1** (this receipt — the E2B Q4_0 file
  carries Q4_0 projections plus Q4_1 on four `ffn_down`, so both ran).
- **Still declined — NVFP4.** Lane-covered with a kernel parity test, but no gemma4 row
  has an end-to-end receipt. That gap is precisely what this defect was.
- **Still declined — K-quant projections.** The CPU wire lane serves them.

The old test was "does *any* tensor say Q8_0", which is also why this is a **narrowing**
as well as a relaxation: an E4B Q4_K_M row has a Q8_0 `token_embd`, so it passed, the plan
disclosed `gemma4_cuda_resident_runtime`, and the load site then refused it and served on
the CPU — the D20 disclosure defect this predicate exists to prevent. It now declines up
front with the tensor and format named.

## Scope

Measured on RTX 3060 Laptop (6 GB), Windows 11, `gemma-4-E2B-it-Q4_0.gguf` only. The Q6_K
head fix is proven by unit test and by inspection against `pad_q6k_blocks`, **not** by a
row receipt — no Q6_K-head gemma4 row fits this card (26B Q4_0, E4B NVFP4). Both remain
declined on quant or fit regardless. No throughput claim is made.
