# SIROCCO — Lane K (kernel campaign)

**Machine:** RTX 3060 Laptop GPU (GA106, CC 8.6, 30 SM), 6 GB GDDR6, driver 576.83, CUDA 12.9, **WDDM** · Win11
**Target:** Llama-3.2-1B decode, Windows/CUDA · **Status:** the clean, correctness-preserving win is **shipped** ([PR #440](https://github.com/timtoole02/Camelid/pull/440), `main` @ `00569168`). Remaining upside is gated behind a precision-policy decision, not more engineering.

> **One-line result:** eight kernel experiments, **one shipped** (byte-identical `uint4` attention K-read, +~10% long-context), **seven rejected on measured evidence**. `main` only ever moved for the change that measurably earned it, byte-identically.

---

## 1. The mandate — why Lane K

SIROCCO Phase 0 (the [roofline receipt](qa/evidence-bundles/sirocco-phase0-roofline-20260712/README.md)) modeled the decode step as `t = B/BW_eff + C` and measured it. The verdict was decisive and **inverted the illustrative plan**, which had assumed an overhead-bound box (Lane A):

| quantity | measured | how |
|---|---|---|
| `BW_peak` (device triad) | **270.9 GB/s** | 94% of 288 nameplate |
| P1 decode (1B Q8_0, ctx≈0) | **126.4 tok/s**, 7.88 ms/tok | interleaved, monitored-boost |
| **`C` (host/loop overhead)** | **≈ 0** | `CAMELID_DECODE_TIME`: forward = step wall, sample/loop = 0 |
| **MBU@P1** | **0.615** | `B/(t·BW_peak)` |
| wire ratio (PC2) | 0.982 | API path not the bottleneck |

**`C ≈ 0` + MBU 0.615 ⇒ decode is kernel-bound, not overhead-bound.** The GEMV kernels are near the memory wall but leave ~35% on the floor; the host is not starving the GPU. That selected **Lane K (kernel campaign)** and ruled out Lane A/graphs (confirmed later: CUDA Graphs measured **−5%** at ctx≈0 — the empty-block fixed-shape attention tax exceeds any launch saving, and host cost is already ≈0).

---

## 2. Where the headroom actually is (regime map)

A follow-up decode-loop analysis (261 kernel launches/token; 148 non-GEMV moving only ~2.6 MB ≈ ~10 µs of real bandwidth) refined the picture. Headroom is **not** uniform:

| Regime | State | Why |
|---|---|---|
| **ctx≈0 Q8 (the 126 headline)** | 🪨 **nearly tapped out** | GEMVs at 84–101% of peak. Residual ~2.2 ms/token is ~130 tiny launches = per-kernel execution floors + the serial data-dependency chain (norm→QKV→rope→scatter→attn→O→FFN), **not** data movement or host launch (graphs already tried: −5%). Ceiling here is single digits, likely noise. |
| **long-ctx Q8 (4k–32k)** | 🌽 moderate, corner | Decode attention reads the f16 KV cache at ~10–20% of peak (PC3). +15–30% realistic at 2–8k; ~2× only at the 16–32k edge (atypical on a 6 GB/1B box). |
| **Q4 / K-quant models** | 🛢️ **fattest budget** | Q4_K_M decodes *slower* than Q8 despite half the bytes — the K-quant GEMVs are compute-bound. |

---

## 3. Scorecard

| # | Experiment | Regime | Outcome | Evidence |
|---|---|---|---|---|
| **1** | **`uint4` attention K-read** | long-ctx Q8 | ✅ **SHIPPED** — +~10% vs scalar, +5–10% vs the coalesced kernel, **byte-identical** | [PR #440](https://github.com/timtoole02/Camelid/pull/440) |
| 2 | Q8 GEMV `U`-unroll bump | ctx≈0 | ❌ refuted — GEMV already 84–102% of peak; U=4 optimal | microbench |
| 3 | CUDA Graphs default-on | ctx≈0 | ❌ −5% (empty-block attention tax; Lane A empty) | A/B |
| 4 | Coalesced attention default-on | long-ctx | ❌ breaks lossless spec-decode at ctx>512 | `splitk_spec_verify` self-skip |
| 5 | `uint4` weighted-V retile (`attention_decode`) | ctx≤512 | ❌ −4.7% (occupancy loss > load-width) | git-stash A/B, byte-identical |
| 6 | `uint4` split-K V (`attn_sk_partial`) | long-ctx | ❌ −18% (1/4-warp utilization) | git-stash A/B |
| 7 | Bit-exact dp4a `q6k_gemv` | Q4 models | ❌ regresses 0.34–0.68× (2-lane cap) | microbench, byte-identical |
| 8 | Token-parity (4-lane) dp4a `q6k` | Q4 models | ❌ fails token-parity (8/8 prompts diverge) | real-model verification |

---

## 4. The one win — `uint4` attention K-read (shipped)

**Change:** widen the scalar `u16` f16 K-cache read in the decode attention to **`uint4` (128-bit = 8 keys/load)**, accumulated in the **same d-order**, in `attention_decode`, `attention_decode_sw`, and `attn_sk_scores`. 45-line diff.

**Why it's byte-identical:** the resident module compiles `--fmad=false`, so preserving textual operation order preserves the bits. The load widens *within a thread*; the thread/warp count and the reduction order are unchanged. Alignment is exact (head_dim=64 → every position row is 16-byte-aligned; a `(head_dim & 7)==0` guard falls back to scalar otherwise).

**Correctness:** `splitk_spec_verify_bit_identical` **passes** (ran, not skipped), asserting the decode path is bit-identical to the *unchanged* `attention_batched`/`attention_tree_batched` spec-verify kernels across pc 512→4096 (incl. `attention_decode` at G=4). Since the verify kernels are untouched, this proves the read is byte-identical to the old scalar read ⇒ **decode == spec-verify ⇒ lossless speculative decode preserved.**

**Measurement** (ctx~1542 split-K, interleaved):

| | tok/s |
|---|---|
| scalar (pre-change) | ~84–86 |
| coalesced kernel (token-parity, breaks lossless) | 90.2 / 88.0 / 83.5 |
| **`uint4` (byte-identical)** | **94.7 / 93.9 / 91.6** |

It **strictly dominates** the previously-shipped opt-in coalesced kernel — faster *and* it preserves the lossless-spec guarantee the coalesced kernel breaks. It is a no-op at ctx≈0 (split-K inactive), so zero headline regression.

---

## 5. The rejections — and what each one taught

**#4 Coalesced attention default-on — the near-miss that mattered most.** It passed the obvious gates (load, ctx≈0, +5–8% long-ctx, *greedy token-identity*). Then the stricter `splitk_spec_verify` test caught that it re-associates the split-K dot, so **decode ≠ spec-verify at the bit level → lossless speculative decode breaks at ctx>512.** The greedy-token check only compared decode-vs-decode; it never checked decode-vs-spec-verify (the actual lossless property). **A naive default-flip would have shipped a silent correctness regression *and* disabled the test that guards it.** → *A speedup that passes greedy token-identity can still break the bit-identical spec-verify contract.*

**#5, #6 V-read vectorization — a dead end in every form.** Widening the attention **V** read (weighted-V −4.7%; split-K `attn_sk_partial` −18%) always loses, because the V read is *already coalesced* and vectorizing it drops the active thread/warp count (8× fewer threads at `attention_decode`; a quarter-warp at `attn_sk_partial`). Split-K's grid-level parallelism does **not** rescue within-block warp under-utilization. → *Only vectorize a read where you can widen the load **within** a thread without shrinking the thread/warp count. That is exactly why the K-dot (#1) won and the V-reads lost: the K-dot keeps one-thread-per-position and only cuts loads-per-thread.*

**#7, #8 dp4a-ize the K-quant GEMV — the fattest budget, gated by precision.** Q4_K_M decodes *slower* than Q8 despite fewer bytes; root cause is **measured**: `q6k_gemv`/`q5k_gemv` have **0 `__dp4a`** (pure scalar integer MAC) while `q4k_gemv` is dp4a-tuned. A perf-prototype dp4a `q6k` hit **1.74× at the lm_head shape** (49% → 86% of peak). But:
- The **byte-identical** version regresses (0.34–0.68×). Q6_K carries **four different scales per stride-32 weight group**, so a parity-preserving dp4a is capped at **2-lane** (q4k gets 4-lane only because its stride-8 weights share one scale). The 1.74× came from a 4-lane collapse that **breaks the parity anchor**.
- Relaxing to **token-parity** doesn't rescue it either: Q6_K is used for the **per-layer** `attn_v` + `ffn_down` projections (not just the lm_head), so the f32-tail-association difference **compounds across 16 layers** and flips greedy tokens (**8/8 prompts diverged** on the real Q4_K_M model; the kernel is integer-correct — it's genuine precision loss, not a bug).

→ *"Token-parity" is fragile for **per-layer** kernel changes — any f32-association change compounds across layers and flips greedy tokens. Only a **final** (lm_head) projection could tolerate it, and even then only probabilistically (which weakens the greedy-determinism Runnable invariant — recommend against).*

---

## 6. Principles (reusable, hardware-agnostic)

1. **Fix the denominator before optimizing.** Phase 0's `C≈0` + MBU 0.615 measurement redirected the whole campaign from the (empty) host-overhead lane to the kernel lane. Running the wrong lane is worse than running no lane.
2. **A perf-prototype that isn't parity-preserving over-states the win.** The dp4a `q6k` "1.74×" evaporated (or went negative) once the byte-identical / token-parity versions were actually built. Build the correctness-preserving kernel *before* trusting the number.
3. **Vectorize *within* a thread, not by dropping threads.** Widening a per-element read that is already coalesced trades away the parallelism that hides memory latency.
4. **Greedy token-identity ≠ bit-identity.** For any change touching the spec-verify / split-K path, the device-side `splitk_spec_verify_bit_identical` gate is the real backstop; the runtime greedy probe is blind above ctx~23 / G=1.
5. **Prove correctness at the level the invariant lives.** Byte-exact-vs-oracle (Runnable lane) is stronger than token-parity; token-parity is stronger than "usually identical." Don't silently downgrade the contract.

---

## 7. Current state & what remains

- **In `main`:** the `uint4` attention K-read (PR #440). Byte-identical, lossless-spec-preserving, +~10% long-context decode.
- **Exhausted:** the clean, correctness-preserving Lane K gains on this GPU. ctx≈0 Q8 is at the wall; the attention V-read avenue is a proven dead end.
- **Gated (not blocked by engineering):** the K-quant lm_head dp4a is a real ~1.74× kernel win, but capturing it requires **retiring the K-quant byte-exact-vs-oracle Runnable invariant** (or restricting to a probabilistic lm_head-only variant). That is a **precision-policy decision**, not a kernel tweak — and having built and measured both, the recommendation is that it is not worth trading greedy determinism for.
- **Untried (moderate, corner):** uncap `SPLITK_MAX` for long-context occupancy (+15–30% at 2–8k). Bit-safe by construction if the verify emulation's `n_splits` formula is mirrored in lockstep.

---

## 8. Reproduction

- **Phase 0 / G0:** [`qa/evidence-bundles/sirocco-phase0-roofline-20260712/`](qa/evidence-bundles/sirocco-phase0-roofline-20260712/README.md) — `roofline-receipt.json`, `triad.cu`, all three sweeps.
- **Shipped win:** [`qa/evidence-bundles/sirocco-laneK-attn-kvec-uint4-20260712/`](qa/evidence-bundles/sirocco-laneK-attn-kvec-uint4-20260712/README.md) — diff, `splitk_spec_verify` log, A/B, the design workflow.
- **Coalesced investigation:** [`qa/evidence-bundles/sirocco-laneK-coalesced-attn-20260712/`](qa/evidence-bundles/sirocco-laneK-coalesced-attn-20260712/README.md).
- **Kernels:** `src/cuda_resident.rs` — `q8_gemv` @230, `q4k_gemv` @455, `q6k_gemv` @779, `attention_decode` @1371, `attn_sk_scores` @2089 (`attn_sk_partial` @2207). Correctness gate: `splitk_spec_verify_bit_identical` (`src/cuda_resident/tests.rs`), `--ignored`, requires a CUDA device (does **not** run in CI).

_All measurements: RTX 3060 Laptop, monitored-boost (hardware clock pin admin-gated on WDDM); mem clock stable at 6000 MHz — the bandwidth governor — so bandwidth-bound tok/s are robust; SM clock floats and governs `C≈0`. A/Bs are interleaved (thermal-robust deltas)._
