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
- **STEP 3 — sliding-window decode attention. DONE — no new kernel needed.**
  The existing `attention_decode_f32` already windows: attending to `[lo..=pos]`
  is `kv_base_offset += lo*position_stride` with `position_count = pos-lo+1`
  (sliding `lo = max(0, pos+1-512)`, global `lo = 0`). Locked in by
  `metal_sliding_window_attention_matches_cpu` (head_dim 256 windowed + 512 full).
- **STEP 4 — Gemma4ResidentState scaffolding. DONE (allocation only).**
  `Gemma4Metadata::layer_plan` (model.rs) is the single source of truth for
  per-layer-type dims, RoPE θ, sliding window, and cross-layer KV source
  resolution (unit-tested on the E4B 42-layer / 18-shared schedule).
  `metal::Gemma4ResidentState::new` allocates the per-layer KV cache (sized to
  each layer's head_dim, only for owning layers — shared layers hold `None`),
  ping-pong hidden buffers, and gate/done events, behind `gemma4_gpu_enabled()`
  (`CAMELID_GEMMA4_GPU`). STILL TODO here: resident WEIGHT buffers — load each
  Q8 tensor as `wire_mmap::WirePages` (page-aligned, GPU reads nocopy, fits
  16GB) + f32 norm/PLE buffers; wired alongside STEP 5's forward so they're
  validated by actually being consumed.
- **STEP 5 — single-layer resident forward** (no PLE, no KV sharing): norm → qkv →
  QK-norm → rope → scatter → attn → o → post-attn-norm → residual → ffn-norm →
  gate/up → geglu → down → post-ffw-norm → residual. Parity vs CPU `step()` for
  layer 0 at position 0. The hardest correctness milestone — sliced:
  - **5a DONE** — `encode_gemma4_q8_matmul` (f32 act × 34-byte wire Q8, always wire
    f32y, NOT gated on CAMELID_METAL_WIRE) + `try_gemma4_q8_matmul_f32y`, validated
    vs CPU f32×dequant. The 8×/layer GEMV workhorse. Reuse for the chain:
    `encode_rms_norm_f32` (full norm), `encode_binary` (GeGLU via gelu_mul_pipeline
    / residual via residual_add_pipeline). Metal's default compute encoder is
    SERIAL, so dependent dispatches chain in one encoder with no manual barriers
    (confirmed: `encode_ffn_block`).
  - **5b DONE** — `encode_gemma4_ffn` (rms_norm → gate/up GEMV → GeGLU → down GEMV
    → post_ffw_norm → residual) as one serial command buffer, no readback;
    `try_gemma4_ffn` + `metal_gemma4_ffn_matches_cpu` validate the whole sub-graph
    vs CPU. First composed gemma GPU sub-graph — proves dependent dispatches chain
    correctly without manual barriers.
  - **5c DONE** — `encode_gemma4_attention` (rms_norm → qkv GEMV → per-head QK/V
    norm → RoPE → KV scatter → windowed decode attn → o GEMV → post_attn_norm →
    residual) + `encode_rms_norm_per_head` helper. `try_gemma4_attention` (prefilled
    cache) + `metal_gemma4_attention_matches_cpu` validate the whole sub-graph vs a
    full CPU attention reference (head_dim 256, GQA 2:1). Passed first try.
  - **5b + 5c together cover every op in a gemma layer.** The full-layer chain (5d)
    is mechanical composition — attention(in→mid) then ffn(mid→out) in one encoder —
    but a 40-arg wrapper is ugly, so it folds into STEP 6 with a proper per-layer
    weight-bundle struct (`Gemma4ResidentLayer`). Done there alongside the resident
    weight residency + multi-layer orchestration.
- **STEP 6 — full-layer chain + multi-layer orchestration.**
  - **6a DONE** — `Gemma4ResidentLayer` weight bundle (6 norms + 7 wire weight
    buffers + dims/eps, `from_wire` ctor) + `encode_gemma4_layer` =
    attention(in→mid) + ffn(mid→out) in one serial command buffer.
    `try_gemma4_layer` + `metal_gemma4_layer_matches_cpu` validate the full layer
    vs the combined CPU chain. A complete gemma layer runs on GPU. Completes 5d.
  - **6b DONE** — `owns_kv` branch in `encode_gemma4_attention` (threaded through
    `encode_gemma4_layer` + try_* wrappers): shared layers skip K/V
    projection+norm+rope+scatter and run q-only attention against the source layer's
    cache. Validated by `metal_gemma4_attention_shared_matches_cpu` (shared path) and
    `metal_gemma4_two_layers_shared_kv_matches_cpu` (two layers, one command buffer,
    ping-pong + persistent shared cache, layer 1 reads layer 0's scattered token).
    Multi-layer orchestration + cross-layer KV sharing proven end-to-end.
- **STEP 7 — PLE stream. DONE (per-layer inject).** `encode_gemma4_ple`:
  `gated = ple_inp_gate·h` (f32 GEMV) → `gelu(gated)·pli` (gelu_mul) →
  `proj = ple_proj·gated` (f32 GEMV) → `h = (h + rms_norm(proj, post_norm)) *
  output_scale`. New `scale_f32` kernel + `encode_linear_transposed_f32`
  (output-major f32 GEMV). Validated by `metal_gemma4_ple_matches_cpu`. The
  per-token `pli` (per_layer_token_embd Q8 gather + per_layer_model_proj f32 matvec
  + norms) is computed on CPU once per token (depends only on the input embedding)
  and passed in — wired in the runtime (STEP 9).
- **STEP 8 — logits + soft-cap. DONE.** `encode_gemma4_head`:
  `normf = rms_norm(h, output_norm)` → `logits = token_embd · normf` (tied
  vocab-major Q8 embedding as output projection, one wire GEMV) →
  `cap·tanh(logits/cap)` in place. New `encode_soft_cap_f32` helper. Validated by
  `metal_gemma4_head_matches_cpu` (logits + greedy-argmax agreement). Greedy
  sampling reads logits back + argmaxes on CPU (the one end-of-token readback);
  GPU `argmax_f32_greedy` is a later fast-path option.
- **STEP 9 — real runtime + end-to-end parity & benchmark.**
  - **9a DONE** — `try_gemma4_forward` drives a whole token forward on the GPU in
    ONE command buffer: h0 → N layers (`encode_gemma4_layer` + optional
    `encode_gemma4_ple`, ping-pong, cross-KV cache resolution) → `encode_gemma4_head`
    → logits. Per-token CPU data via `Gemma4TokenLayerInput` (cos/sin, pli,
    window_start); PLE via `Gemma4ResidentPle`. Validated
    (`metal_gemma4_forward_matches_composed_pieces`) against the same pipeline built
    from the individually-validated `try_*` pieces. The GPU forward is complete.
  - **9b (real-model wiring + the payoff) — IN PROGRESS**:
    1. **Weight residency DONE** — `Gemma4ResidentLayer::from_wire_pages` builds a
       layer's weights from per-tensor `WirePages` via `q8_wire_nocopy_buffer` (GPU
       reads in place, no 8GB copy). Validated identical to `from_wire`
       (`metal_gemma4_layer_from_wire_pages_matches_copy`). token_embd +
       per_layer_token_embd will load the same way in the runtime.
    1b. **Still needed — stateful resident forward**: a `Gemma4ResidentModel` holding
       resident weights + token_embd + PERSISTENT caches (allocated once, scattered
       per token) + a `forward_token(h0, inputs, position) -> logits`. Refactor of
       `try_gemma4_forward` to stateful (no per-token weight/embd copy — required for
       both correctness across tokens AND a valid benchmark; `try_gemma4_forward`
       copies the 0.7GB token_embd per call).
    2. **Per-token CPU prep** (reuse `step()` math): embedding gather × √hidden → h0;
       `pli` (per_layer_token_embd Q8 gather + per_layer_model_proj f32 matvec +
       per_layer_proj_norm); per-layer cos/sin from `rope_freq_base_at` + position;
       window_start per `layer_plan`.
    3. **Drive** `try_gemma4_forward` per token (or add `Gemma4ResidentState::
       forward_token`), readback logits, argmax. Gate on `gemma4_gpu_enabled()`.
    4. **Validate**: `tests/gemma4_forward.rs` greedy decode emits identical token
       ids; then decode **benchmark** (target ~13–15 tok/s vs the ~6 CPU baseline).
    Risk: the WirePages path is memory-critical on the 16GB box (8GB anonymous,
    single-copy); verify residency before trusting the bench. Off by default.

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
