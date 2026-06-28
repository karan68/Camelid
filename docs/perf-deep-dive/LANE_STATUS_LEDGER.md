# LANE_STATUS_LEDGER.md — Camelid speed-campaign lane verdicts

Authored 2026-06-28 from a code-grounded recon (not memory) so no agent relitigates a settled
lane. Governed by [BENCHMARK_TREATY.md](BENCHMARK_TREATY.md). HEAD `51018c00`, llama.cpp `acd79d6`,
host win i7-11800H (Ubuntu leg pending).

## The blunt read

CPU is where Camelid trails: a uniform ~0.62–0.85× per-kernel-throughput gap across prefill AND
decode. llama runs ONE tiled tinyBLAS GEMM for prefill projections + a single shared activation
quantize + an atomic chunk scheduler; Camelid runs bespoke per-role kernels that re-quantize the
activation per projection (attn_norm 3×, ffn_norm 2× per layer) and has no unified tiled owner.
**Decode is DRAM-bandwidth-bound (~33% of read roofline)** → every cheap ALU win there is a proven
dead end. **Prefill is compute-bound** → that is the one place a tiled GEMM can genuinely help.

## Lane verdicts

| Lane | Verdict | Evidence (code, not memory) |
|---|---|---|
| **1 — unified tiled Q8 PREFILL GEMM owner** | **v1 BUILT — bit-exact, size-scaling prefill win; does NOT match llama. v2 is the next lever.** | Shipped behind `CAMELID_X86_Q8_MATMUL_OWNER` (off\|ffn_down\|all, default-off): one role-agnostic hook in `linear_for_role_runtime_with_plan` covering all 7 projections, reusing the proven 4×4 AVX2 microkernel VERBATIM with a re-architected weight-resident loop nest (parallel over output-row bands, weight band L1/L2-resident, all input groups streamed against it). **Bit-identity unit test PASS** (`to_bits`); **e2e greedy parity PASS** (`first_divergent=-1`) on TinyLlama+Llama-3B. Measured prefill (focused 3-iter, win host, receipt `q8-prefill-owner-v1-20260628.json`): **−5% on 1.1B / neutral on 3B / +5% on 4B** — the win scales with weight size (small weights fit cache → only overhead; large weights exceed cache → residency bites). **It does NOT match llama** (still ~0.75× on 4B): the residual gap is llama's **AVX-512/VNNI register-tiled compute**, not the cache/bandwidth axis v1 addresses. **v2 = AVX-512/VNNI `dpbusd` microkernel** (bit-exact-preservable), but AVX-512 downclocks on this Tiger-Lake box → v2's fair evaluation needs the **Ubuntu host** (pending). Default-off; not promotable (Windows-only). |
| **2 — decode-overhead** | **DEAD (settled negative)** | Already profiled (`audit-workflow-result.json`); decode loop is alloc-clean, shared-quant, persistent pools. An 8-agent audit's 5 candidate micro-opts were ALL adversarially rejected (`confirmed_wins: []`). Decode is bandwidth-bound; overhead cuts cannot move tok/s. No new profiler needed. |
| **3 — T-MAC ternary (TQ2_0) LUT** | **DEAD (refuted)** | TQ2_0 is already fully shipped default-on (scalar+AVX2+prefill-tiling+decode, parity-certified `qa/ternary/tq2_0-bonsai-parity-receipt.json`). Its OWN receipt kills the LUT angle: the ternary dot is ~11% of decode and `AVX2+tiled == scalar throughput`. A LUT kernel optimizes a non-bottleneck. Not a beat on this host. |
| **4 — Q4_K AVX2** | **DEAD (no-op)** | Q4_K CPU decode already runs a bit-identical ggml-style AVX2 kernel default-on (`q4_k_dot_avx2`, refmath.rs:439, bit-identity test :826). Reimplementing duplicates + risks regressing a parity-certified path. |

## Already-settled negatives (do not re-run)

- Gated x86 packed-rows4/GEMM4 SIMD A/B (`CAMELID_X86_Q8_*`): −8…−11%, byte-identical → default-off.
- VNNI/AVX2/scalar packed-dot matrix: identical-throughput + byte-identical → decode is DRAM-bound.
- Prefill routing (layer-major, chunk 64/all/lm): <3% noise, parity-identical.
- Thread sensitivity: decode wants 1–2 threads, prefill scales to 16T; resolved by the shipped
  phase-adaptive prefill pool (already banked prefill 0.62×→0.80×, +24%).
- x86 software prefetch: NULL −0.8%. Q6_K AVX2 8-lane: −21% regression. AVX-512 decode: downclock.
- `target-cpu=x86-64-v3` + fat-LTO ("the +39% AVX2"): already shipped (compiler autovec).

## What "done" looks like

- **Prefill:** Lane 1 lands Q8 prefill at ~1.0× on both hosts, bit-exact → *matchable, the real work.*
- **Decode (Q4/Q8 + ternary):** bandwidth-tied at the ceiling → *match, not beat; stated as such.*
- Every claim labeled correctly; every number reproducible from the committed command on a host of record.
