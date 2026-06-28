# KQUANT_RECON.md — Phase 0 of the Q4_K_M decode conductor

Recon deliverable for `KQUANT_DECODE_CONDUCTOR.md` Phase 0. Every situation-map row below was
located **by symbol** (`git grep`), not by the line numbers in the conductor doc (those are a
prior). Line numbers recorded here are where the symbols resolve **on this branch**
(`feat/kquant-decode-conductor`, base `3812a1f7` = v0.1.8) and will drift.

**Bottom line: all 8 rows CONFIRMED. The Phase-1 gate (S2 + S7) is PASSED.** The certifiable
GPU engine exists and is fully wired; no Q4_K_M parity receipt exists yet. One residual unknown
(S2 caveat) is flagged below and is exactly what Phase 1's first action resolves.

---

## Environment recorded for the receipt

| Item | Value |
|---|---|
| Host | i7-11800H (8C/16T, Tiger Lake, AVX2/AVX-512/VNNI), ~16 GiB DDR4 |
| GPU | RTX 3060 Laptop 6 GiB (sm_86), Win11 |
| llama.cpp source pin | `acd79d6` — **CONFIRMED** present at `C:\Users\timto\llama.cpp` (HEAD = acd79d6 "jinja : add count/d/e filter aliases (#24606)") |
| llama.cpp binaries | `C:\Users\timto\tools\llama-cpp\` — `llama-bench.exe`, `llama-cli.exe` present (build provenance vs acd79d6 to be re-verified in Phase 1 via `--version`) |
| Branch / base | `feat/kquant-decode-conductor` from committed `3812a1f7` (v0.1.8); isolated worktree `C:\Users\timto\cam-kquant` so the uncommitted WARP work in the main checkout is untouched |

### Models

| Model | Role | Present? | Path / size / SHA-256 |
|---|---|---|---|
| Llama-3.2-3B-Instruct-Q4_K_M | **primary** | **NO** | only `Camelid/models/Llama-3.2-3B-Instruct-Q8_0.gguf` exists — Q4_K_M must be downloaded (~2 GB) |
| Qwen3-4B-Q4_K_M | secondary | **YES** | `C:\Users\timto\models\Qwen3-4B-Q4_K_M.gguf`, 2,497,280,256 B, SHA-256 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` |
| Qwen3-0.6B-Q4_K_M | extra | YES | `C:\Users\timto\models\Qwen3-0.6B-Q4_K_M.gguf` (396,704,416 B) |
| Qwen3-8B-Q4_K_M | extra | YES | `C:\Users\timto\models\Qwen3-8B-Q4_K_M.gguf` (5,027,783,488 B) |

**Phase-1 consequence:** the secondary model is ready to certify *now*; the primary needs a
download before its receipt can be minted.

---

## Situation map verdicts

| # | Claim | Verdict | Resolved location (this branch) |
|---|---|---|---|
| S1 | Scalar CPU K-quant oracles exist | **CONFIRMED** | `q4_k_wire_row_dot`, `q6_k_wire_row_dot` in `src/inference.rs` (consume the Q8_K activation) |
| S2 | CUDA fused K-quant GEMVs exist + wired | **CONFIRMED** (with caveat) | `q4k_gemv`/`q6k_gemv` + `launch_q4k_gemv`/`launch_q6k_gemv` in `src/cuda_resident.rs`; dispatch `ProjQuant::Q4K => launch_q4k_gemv(...)` / `Q6K => launch_q6k_gemv(...)` at ~`cuda_resident.rs:2260-2271`; `ProjQuant::{Q4K,Q6K}` enum + `needs_q8k()` at ~`:3174-3188`. Kernel header carries the full 8-lane f32 parity-anchor (lines ~440-583, self-reports "parity green" at unit level). |
| S3 | AVX2 Q4_K dot exists, NOT in LLM decode path | **CONFIRMED** | `q4_k_dot_avx2` + test `q4_k_avx2_bit_identical_to_scalar` exist **only** in `src/diffusion_gemma/refmath.rs`. No Q6_K AVX2 sibling anywhere. |
| S4 | Q8_0 fast path is the integration template | **CONFIRMED** | `q8_0_packed_rows4_dot` in `src/inference.rs`; `q8_0_runtime_packed_rows4_for_tensor` + `q8_0_runtime_packed_rows4_linear` in `src/tensor/mod.rs` |
| S5 | Decode dispatch has NO CPU K-quant arm | **CONFIRMED** | `linear_for_role_runtime_with_plan` (`src/inference.rs:6798`) tries only `try_x86_q8_*` arms, then falls through to `linear_with_diagnostic_layouts_with_plan` (f32 dequant / scalar). K-quant tensors get no packed matvec. |
| S6 | K-quant CPU tensors load wire-only for GPU path | **CONFIRMED** | `load_kquant_wire_linear` (`src/tensor/mod.rs:3297`) retains `q4_k_wire_bytes`/`q6_k_wire_bytes`, sets `data: Vec::new()`, `dtype: F32`, `source_type: Some(Q4K|Q6K)` |
| S7 | NO Q4_K_M parity receipt for a mainstream LLM | **CONFIRMED** | `qa/evidence-bundles/` has no q4_k/Q4_K_M bundle; the only K-quant-adjacent gemma4 bundles are **Q8_0** and **Q4_0** (`gemma4-26b-a4b-it-q4-0-...`), a different quant |
| S8 | K-quant is "gated/experimental", not Supported | **CONFIRMED** | `CAPABILITY_MATRIX.md:67`: "Q8_0 supported; broader K-quant decode (Q4_K/Q5_0/Q6_K) gated/experimental (wire_dequant.rs). SECONDARY axis." |

### S2 caveat (the one residual unknown)

S2 is confirmed at the **structural** level: the kernels exist, the `ProjQuant` dispatch routes
to them, the Q8_K activation signal (`needs_q8k()`) is wired, and the kernel header documents the
exact f32-lane parity discipline. What has **never been executed** is an end-to-end greedy decode
of a *full mixed Q4_K_M model* on this path. Per the conductor's own branch condition, if that
end-to-end run does not actually produce coherent tokens, the campaign re-scopes to "finish the
GPU path first." **Phase 1's first action is therefore a smoke decode** (a few tokens, coherence
check) *before* the full parity harness — cheap insurance against certifying an engine that
doesn't run.

---

## Primary-vs-secondary GGUF quant mix

The conductor requires a `gguf_dump` of the primary showing the Q4_K/Q6_K split. The **primary is
not on disk**, so the dump below is of the **present secondary** (`Qwen3-4B-Q4_K_M`), which is
sufficient to prove a real model exercises **both** kernels. The primary's mix will be dumped on
download as a Phase-1 precondition.

```
Qwen3-4B-Q4_K_M.gguf — tensor-type histogram
  Q4_K   count= 216   1.765 GB
  Q6_K   count=  37   0.725 GB
  F32    count= 145   0.001 GB   (norms only)

block-0 + global per-tensor mix
  blk.0.attn_q.weight      Q4_K     blk.0.attn_k.weight   Q4_K
  blk.0.attn_v.weight      Q6_K  ←  blk.0.attn_output.weight Q4_K
  blk.0.ffn_gate.weight    Q4_K     blk.0.ffn_up.weight   Q4_K
  blk.0.ffn_down.weight    Q6_K  ←
  token_embd.weight        Q6_K  ←  (tied: serves as lm_head; no separate output.weight)
  *_norm.weight            F32
```

**Confirms the conductor's prediction exactly:** `attn_v`, `ffn_down`, and the (tied) `lm_head`
are Q6_K; everything else dense is Q4_K. A single Qwen3-4B-Q4_K_M decode run therefore exercises
**both** the `q4k_gemv` and `q6k_gemv` lanes — both must be green for the receipt to mean
anything. (Qwen3-4B ties embeddings, so `token_embd` Q6_K doubles as the output projection; a
Llama-3.2-3B Q4_K_M will instead carry a separate Q6_K `output.weight` — to be confirmed on
download.)

---

## Gate decision

- **S2 CONFIRMED** → a certifiable engine exists. ✅
- **S7 CONFIRMED** → it is not yet certified. ✅

**→ Phase 1 may begin.** Ordered next actions:

1. Smoke-decode Qwen3-4B-Q4_K_M on the CUDA-resident path (resolve the S2 caveat) — STOP and
   triage to oracle order if tokens are incoherent; do **not** re-associate any reduction.
2. Build the full greedy parity harness vs llama.cpp `acd79d6` on the *same* GGUF, cache defeated.
3. Download Llama-3.2-3B-Instruct-Q4_K_M (primary), record SHA-256, dump its mix, repeat.
4. Emit + commit the `camelid.parity-receipt/v1` + `camelid.speed-receipt/v1` bundles; promote
   the two exact rows to Supported in `CAPABILITY_MATRIX.md`.

No row was refuted, so no re-scope is triggered at Phase 0.

---

## Phase 1 outcome (appended) — Qwen3-4B-Q4_K_M CERTIFIED

**Result: `all_pass = true`.** GPU-resident CUDA decode of `Qwen3-4B-Q4_K_M` is token-AND-text-
identical to llama.cpp `acd79d6` at 1/5/50 generated tokens on all 3 chat prompts (thinking-
disabled ChatML, greedy), plus cross-engine prompt-token parity. Bundle:
`qa/evidence-bundles/qwen3-4b-q4_k_m-windows-cuda-resident-parity-20260628T003317Z-head-0dccbf74/`.
The mixed model drove **both** `q4k_gemv` (Q4_K) and `q6k_gemv` (Q6_K) in one run; the 8-lane
f32 parity anchor holds end-to-end. The 'primary color' probe (a near-tie excluded from the
Q8_0 4B headline set) passed here at all token counts.

S2 caveat **resolved**: smoke decode produced coherent output (" Paris. The capital of Germany
is Berlin…"), 36/36 layers VRAM-resident, 19.44 tok/s median (4.92 GB peak).

### Two findings that reshape Phase 2 / add follow-ups

1. **CPU K-quant decode is BROKEN, not merely slow/scalar.** S5 said K-quant CPU tensors "fall
   to f32 dequant / scalar." Reality is stronger: with CUDA hidden the model **errors** —
   `matmul rhs-transposed ... blk.0.attn_q.weight ... no-row-major-data ... data_len=0`. K-quant
   2-D linears load **wire-only** (`load_kquant_wire_linear`, empty f32 `data`) and there is no
   CPU arm to read them, so there is no f32 fallback at all. ⇒ **Phase 2 must first give the CPU
   lane a way to READ K-quant tensors** (dequant-to-f32 fallback and/or the packed matvec), not
   just a *faster* path. (Silver lining: this same error is positive proof the passing GPU run
   was genuinely on the GPU.)

2. **The execution-plan disclosure mislabels the K-quant resident lane** (NEW follow-up, not in
   the conductor). `/api/capabilities` reports `selected_backend=cpu_reference`,
   `decode_path=safe_cpu_decode`, `quant_type=dense_or_other` ("non-validated row or quant;
   failing closed to safe path") for the loaded Q4_K_M model — yet `cuda_resident_active=true`
   and it ran GPU-resident. The planner classifier is Q8_0-centric. Parity-green output, wrong
   self-disclosure. Fix: teach the classifier to recognize Q4_K/Q6_K dense resident models.

### Speed (honest)

camelid Q4_K_M GPU-resident decode **19.44 tok/s** @ 4.92 GB; llama.cpp Q4_K_M **CPU** tg128
**12.35 tok/s**. Different backends — NOT a ratio. This box's llama.cpp has no `ggml-cuda.dll`,
so GPU-vs-GPU is blocked; Qwen3-4B-Q8_0 isn't on disk, so same-model Q8-vs-Q4 is deferred.

### Promotions (doc/ledger only)

`SUPPORT_MATRIX_v0.1.md` + `COMPATIBILITY.md`: added the `Qwen3-4B-Q4_K_M` row as
GPU-resident-parity-certified exact-row smoke with the caveats above and the bundle cited
(runtime support-contract recognition marked PENDING pending the disclosure fix). The
Llama-3.2-3B-Q4_K_M (primary) row remains unstarted — GGUF still needs download.
