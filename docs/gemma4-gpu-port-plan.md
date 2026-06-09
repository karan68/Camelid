# Gemma 4 GPU-resident decode — port plan

Goal: run gemma4 decode on the Metal GPU at the unified-memory bandwidth wall
(~120 GB/s on M4 → ~13–15 tok/s for the 8GB E4B Q8_0), up from the ~6 tok/s CPU
`sdot` ceiling. The win only materializes when the **whole** per-token graph runs
on GPU with no CPU readback until the final logits — a partial offload pays
~140 commit/wait round-trips/token and regresses (see
`gemma4-engine-status.md` "GPU port: scoped"). So we build and validate kernels
incrementally, then assemble the full resident graph last.

This is a multi-session effort. Each step below is independently committable and
parity-checked against the CPU reference (`src/gemma4_runtime.rs`, itself bit-exact
vs llama.cpp via `tests/gemma4_forward.rs`).

## Architecture facts (E4B-It Q8_0, from the GGUF)

- 42 layers, hidden 2560, ffn 10240, vocab 262144, 8 heads / 2 KV heads.
- **Per-layer-type head_dim**: sliding (local) = 256, global = 512.
  - sliding: q_dim 2048, kv_dim 512, rope_dim 256 (rotates full head, half=128).
  - global:  q_dim 4096, kv_dim 1024, rope_dim 512 (rotates full head, half=256).
- Sliding schedule: 5:1 (every 6th layer global), final layer forced global.
- `sliding_window = 512`; sliding layers attend only `[pos+1-512 .. pos]`.
- `shared_kv_layers = 18` → `first_kv_shared = 42 - 18 = 24`. Layers 24–41 skip
  their own K/V projection+scatter and read the last same-type layer's cache
  (last sliding layer's cache for sliding layers, last global for global).
- Dual RoPE θ: `rope_freq_base_global` vs `rope_freq_base_sliding`; RoPE pairing
  is split-half (pairing mode 1), freq = θ^(-2i/head_dim).
- `final_logit_softcapping = 30`.
- PLE (E-series): `per_layer_input_dim = 256`; per-layer-embedding stream injected
  after each layer's FFN (7-step; see CPU `step()`), uses F32 `ple_inp_gate` /
  `ple_proj` matrices + `post_norm` + scalar `ple_output_scale`.

## Reuse map (from the resident infra in src/metal.rs)

Reusable as-is (architecture-agnostic):
- **Q8 wire GEMV** `q8_0_block_linear_row_ksplit_f32y_wire` — f32 activations ×
  34-byte wire weights read **nocopy** (`q8_wire_nocopy_buffer`). Used for all 8
  matvecs (q/k/v/o/gate/up/down/logits). NOTE: f32 activations (no activation
  quant) — numerically the *original* CPU f32 path, not the sdot path; both pass
  the teacher-forced argmax test.
- **RMS norm** `rms_norm_f32` (full-width; weight applied as `normed*weight`,
  which is exactly gemma4's RMSNorm — no `1+w` fold). For the 5 per-layer norms +
  final norm.
- **RoPE** `rope_rotate_f32` pairing mode 1 — dual-θ / per-layer-type handled by
  computing per-layer cos/sin tables on CPU (cheap, head_dim/2 entries). NO new
  RoPE kernel needed.
- **KV scatter** `kv_scatter_f32`, **attention** `attention_decode_f32` (basic
  online-softmax variant — required because global head_dim=512 > the 128 cap on
  v2/splitk), **residual add** `residual_add_f32`, **argmax** + **embed gather**
  (sampling tail), **f32 dense GEMV** `linear_row_f32` (PLE matrices).

New kernels required:
1. **`gelu_mul_f32`** — GeGLU `gelu_tanh(gate)*up` (twin of `silu_mul_f32`).
   Ref: `inference::gemma4::geglu_into`. [STEP 1]
2. **`soft_cap_f32`** — `x <- cap*tanh(x/cap)` over logits. Ref:
   `inference::gemma4::soft_cap_in_place`. [STEP 1]
3. **per-head RMS norm** `rms_norm_per_head_f32` — normalize each head_dim chunk
   independently, optional weight (QK-norm uses q_norm/k_norm weights; V-norm is
   weightless). Ref: gemma `step()` q/k/v per-head `rms_norm`. [STEP 2]
4. **sliding-window attention** — add a `lo` (start position) param to the decode
   attention path so sliding layers attend `[lo..pos]`. Either a variant kernel or
   a scalar on `attention_decode_f32`. [STEP 3]

## Build order (each step: kernel + encode helper + `try_*` wrapper + parity test, committed)

- **STEP 1 — GeGLU + soft-cap kernels.** Smallest, fully self-contained; proves
  the add-a-gemma-kernel loop. Validate `try_gelu_mul`/`try_soft_cap` vs the CPU
  primitives over random vectors. ← start here
- **STEP 2 — per-head QK/V norm kernel.** Validate vs per-head CPU rms_norm.
- **STEP 3 — sliding-window decode attention.** Validate masked attention vs CPU
  for both window-clipped and full ranges, head_dim 256 and 512.
- **STEP 4 — Gemma4ResidentState scaffolding.** New struct (do NOT extend the
  Llama `ResidentDecodeState` — gemma's per-layer head_dim, cross-layer KV, and
  PLE diverge too far). Holds: per-tensor wire nocopy buffers (weights resident),
  per-layer KV cache sized to that layer's head_dim, ping-pong hidden buffers,
  gate/done events. Weights loaded as `wire_mmap::WirePages` (page-aligned, GPU
  reads nocopy — fits 16GB, no 2nd copy).
- **STEP 5 — single-layer resident forward** (no PLE, no KV sharing): norm → qkv →
  QK-norm → rope → scatter → attn → o → post-attn-norm → residual → ffn-norm →
  gate/up → geglu → down → post-ffw-norm → residual. Parity vs CPU `step()` for
  layer 0 at position 0. The hardest correctness milestone.
- **STEP 6 — cross-layer KV sharing + sliding window across all 42 layers.**
- **STEP 7 — PLE stream** (per-token `pli` at token start + per-layer 7-step
  injection on GPU with f32 GEMVs + geglu + norm + scale).
- **STEP 8 — logits + soft-cap + sampling tail**, end-to-end resident token.
- **STEP 9 — end-to-end parity** (`tests/gemma4_forward.rs` greedy decode must
  emit identical token ids) + **benchmark** vs the 6 tok/s CPU baseline. Gate the
  whole path behind `CAMELID_GEMMA4_GPU` (off by default until proven).

## CI / safety notes

- src/metal.rs is NOT module-gated: every new helper/struct touching Metal types
  needs its own `#[cfg(target_os = "macos")]`, and the non-macOS stubs need
  matching signatures, or ubuntu CI breaks (we can't cross-check locally — no
  rustup). Grep new fns for the cfg before pushing.
- New decode-attention asm/i8mm must stay off the M1-runner path (dotprod ok,
  i8mm not) — but the GPU kernels are MSL, so this only matters for any CPU
  reference helpers added alongside.
- The branch (`feat/gemma4-engine-support`) is local-only with pre-existing
  fmt/clippy debt; keep new code clean and don't bundle the debt fixes here.
