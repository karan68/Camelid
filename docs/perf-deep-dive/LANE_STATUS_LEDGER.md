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
| **1 — unified tiled Q8 PREFILL GEMM owner** | **v1 BUILT — bit-exact, size-scaling prefill win; does NOT match llama. v2 is the next lever.** | Shipped behind `CAMELID_X86_Q8_MATMUL_OWNER` (off\|ffn_down\|all, default-off): one role-agnostic hook in `linear_for_role_runtime_with_plan` covering all 7 projections, reusing the proven 4×4 AVX2 microkernel VERBATIM with a re-architected weight-resident loop nest (parallel over output-row bands, weight band L1/L2-resident, all input groups streamed against it). **Bit-identity unit test PASS** (`to_bits`); **e2e greedy parity PASS** (`first_divergent=-1`) on TinyLlama+Llama-3B. Measured prefill (focused 3-iter, win host, receipt `q8-prefill-owner-v1-20260628.json`): **−5% on 1.1B / neutral on 3B / +5% on 4B** — the win scales with weight size (small weights fit cache → only overhead; large weights exceed cache → residency bites). **v2 = AVX-512/VNNI `dpbusd` microkernel** (`CAMELID_X86_Q8_MATMUL_OWNER_VNNI`, default-on, runtime-gated) = llama's tinyBLAS compute technique, **bit-exact** (single `dpbusd` per chunk-pair, weight band loaded once, reused across 4 input lanes; unit test PASS for both microkernels). **The Tiger-Lake AVX-512 downclock did NOT regress prefill** (sustained compute amortizes it, as llama proves running AVX-512 here). Measured median-of-5: prefill **+6.0% on 3B / +14.3% on 4B** over default → closes the gap **0.78×→0.85× (3B), 0.67×→0.81× (4B)**; e2e parity PASS. **v3 = wider 4x8 VNNI tile** (`CAMELID_X86_Q8_MATMUL_OWNER_4X8`, **default-on**) — bit-exact, and **SIGNIFICANT** once measured properly. A first cross-invocation A/B looked like a null (±10% thermal noise), so I **hardened the harness** (`camelid bench-owner-sweep`: load once, configs interleaved per round, PAIRED bootstrap-CI). That resolved it: 4x8 is **+3.3% (3B) / +3.8% (4B)** over 4x4 (CI excludes 1.0, 7-8/8 rounds). **Full owner stack (4x8, hardened, median-of-8 paired): +13.5% (3B) / +15.8% (4B)** over default → **0.78×→0.89× (3B), 0.67×→0.81× (4B)** of llama.cpp, every step significant. **v4(a)** (in-register VNNI reduction, replacing store+scalar) measured a **−3.5% REGRESSION** on both models (the hardened paired harness caught it; the in-register chain's `hadd` latency + AVX-512↔AVX2 transition lose to the store+scalar the CPU pipelines freely) → **reverted**. Tractable kernel levers now **exhausted**: cross-projection single-quant is dead-by-analysis (quant is ~0.1–0.5% of the matmul). Remaining: **v4(b) 2D cache blocking** — DE-RISKED via a prompt-length probe (`q8-prefill-owner-v4b-2dblock-derisk.json`): the owner IS memory-bound (vnni4x8 tok/s 30.3→27.4→23.3 at 393/753/1473 tok) and its advantage erodes (owner/off 1.139→1.106), so 2D blocking would recover **~+3% at ~1.5K tokens, more at long context** — modest for headline prompts, so **DEFERRED** (build only if long-context prefill becomes a priority). The last gap to llama is also partly the **bit-exact tax** (no-FMA, fixed reduction order). Owner default-off overall; **promotion gated on the Ubuntu/second host** (Tim provisioning — paused). Receipts `q8-prefill-owner-{v1,v2,v3,hardened,v4a-REVERTED,v4b-derisk}-*.json`. |
| **2 — decode-overhead** | **DEAD (settled negative)** | Already profiled (`audit-workflow-result.json`); decode loop is alloc-clean, shared-quant, persistent pools. An 8-agent audit's 5 candidate micro-opts were ALL adversarially rejected (`confirmed_wins: []`). Decode is bandwidth-bound; overhead cuts cannot move tok/s. No new profiler needed. |
| **3 — T-MAC ternary (TQ2_0) LUT** | **DEAD (refuted)** | TQ2_0 is already fully shipped default-on (scalar+AVX2+prefill-tiling+decode, parity-certified `qa/ternary/tq2_0-bonsai-parity-receipt.json`). Its OWN receipt kills the LUT angle: the ternary dot is ~11% of decode and `AVX2+tiled == scalar throughput`. A LUT kernel optimizes a non-bottleneck. Not a beat on this host. |
| **4 — Q4_K AVX2** | **DEAD (no-op)** | Q4_K CPU decode already runs a bit-identical ggml-style AVX2 kernel default-on (`q4_k_dot_avx2`, refmath.rs:439, bit-identity test :826). Reimplementing duplicates + risks regressing a parity-certified path. |

## Already-settled negatives (do not re-run)

- **Tree-speculation WIDTH on Apple Silicon (BARCHAN Gate 1, 2026-07-20, M4 / 3B Q8_0):** per-round
  verify cost is **linear in tree width** (R² = 0.997, N=5) at **38.24 ms per node** against a
  **35.72 ms** plain decode step — i.e. each verified row costs a full independent decode, and the batched verify amortizes the
  weight read **not at all**. k=15 vs k=1 = **9.25×** (KILL threshold 3×). Fixed per-round cost
  (intercept) ≈ 0, so this is NOT KV compaction / rollback / dispatch overhead — the PIVOT
  hypothesis is refuted too. Widening can never pay; the optimum is the *minimum* width (2 nodes,
  s_sync ≈ 1.03). Phases 3–4 dropped. Full curve: `BARCHAN_PHASE1_COST_CURVE.md`.
  Reopening requires first answering "why does one verified row cost a full decode?", which is a
  Q8 Metal *kernel* question — a lane `METAL_PARITY_RESULT.md` §4 already closed.
- **Q8_0 resident KV on Metal AS SHIPPED (`CAMELID_METAL_KV_DTYPE=q8`), M3/8 GiB, Llama-3.2-1B
  Q8_0, 2026-08-22:** do not re-run the plain "is q8 faster than the default" A/B — it is
  answered and the answer is *no, but not for the reason it looks like*. Against the
  zero-config F32 default, q8 decode is **not resolved** (0.871×, CI [0.731, 1.066]) and
  prefill is **4.48× worse** (significant). But enabling any compressed primary forfeits
  split-K decode, and against an F32 arm with split-K forced off — the apples-to-apples
  footing — q8 decode is **1.50× FASTER** (CI [1.468, 1.644]), rising monotonically with
  context depth (1.14× at 1k, 1.27× at 2k, 1.50× at 8k). So the KV-bandwidth win is real and
  the lane is **blocked, not dead**: the missing pieces are `attention_decode_splitk_kvq8`
  and a Q8 flash/matmul prefill. **Both were then built and measured (2026-08-22, same
  host).** PREFILL IS FIXED: the cause was one conjunct (`!self.kvq8` in `use_attn_mm`)
  excluding q8 from the simdgroup-matrix attention, NOT the Q8 dequant — q8/f16 prefill was
  1.00-1.07x at every depth while both trailed f32 by 1.8-4.4x, i.e. the two excluded
  formats landed on top of each other. Admitting q8 via a `kv_dequant_q8_to_h` staging pass
  takes prefill from **4.48x worse than the f32 default to 0.986x — parity** (CI [0.924,
  1.132]), a 4.5x improvement (CI [0.198, 0.252]); reproduced across four paired runs at
  0.222/0.220/0.426/0.398 vs pre-change q8. DECODE: `attention_decode_splitk_kvq8` was
  written and works, but **the composition hypothesis was WRONG** — predicted 1.5x-3.2x,
  measured **~+5%, and significant in only 2 of 4 paired runs** (1.059 SIG / 1.036 / 0.983 /
  1.106 SIG). Split-K wins mostly via memory-level parallelism on the KV read, the same
  resource Q8 already relieves, so the second lever has little left to pull. Treat it as a
  small real effect this host cannot resolve reliably, NOT as a headline number — an earlier
  revision of this entry quoted the single 8006 run-1 figure and overstated it. ES=2: also
  landed (`rope_rotate_batch_h` + `kv_scatter_batch_kvq8_h`; a fused `rope_scatter_qh_h_q8`
  is the WRONG SHAPE — one thread per RoPE pair cannot do a per-32-block amax, and the split
  path is one dispatch cheaper anyway). It produced **no resolvable throughput change at
  either depth** (q8-both/f32 prefill 0.986->1.000 at 8006, 0.990->0.984 at 2066, all
  unresolved) but made q8 output **token-identical to f32 in 4/4 rounds** at 2066 where es=4
  diverged at token 13 — keep it for fidelity and the dispatch saving, not for speed.
  Net: q8 matches the f32 default on prefill AND decode at 0.858 of its peak RSS (0.948 at
  2066). Do not re-run the compose experiment; do not re-derive the prefill cause; do not
  re-attempt a fused rope+Q8-scatter. Still open: `fit.rs KvDtype` has no Q8 variant.
  Note one depth already won outright even before the fix: at 4117 tokens q8
  beats the default **1.031×** (significant) despite the forfeit. **Quality is where the two
  compressed formats diverge:** on coherent English (`realtext` probe, 1460 tok) **f16 is
  token-identical to f32 in 3/3 rounds at every depth tested, q8 is NOT** — it diverges
  deterministically at generated token 50, though benignly (fluent alternative continuation).
  Filler-prompt parity numbers are worthless in both directions and must not be cited: flat
  distributions flip the argmax on noise, and long filler runs collapse into a repetition
  attractor (9 distinct tokens across 64 positions) where every arm agrees for free. Also
  settled: process peak RSS cannot measure this (it does not separate q8 from f16 at all);
  `vmmap` on `IOAccelerator (graphics)` shows f16→q8 saving 102.4 MB against 120 MB
  predicted. Receipt
  `PERF_RECEIPTS/same-host/metal-kv-q8-m3-20260822.json`, writeup `METAL_KV_Q8_RESULT.md`.
  One host only → nothing promotes.
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
