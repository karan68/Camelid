# DiffusionGemma lane recon (experimental/diffusiongemma)

> [!WARNING]
> **Active development — recon/evidence-only. Not supported.** This document
> describes the `experimental/diffusiongemma` lane. Nothing here is a support
> claim; the public support truth is [`COMPATIBILITY.md`](../../COMPATIBILITY.md)
> and [`STATUS.md`](../../STATUS.md). The lane's mission — a Rust-native
> DiffusionGemma runtime — is earned by parity receipts, not claims; until the
> Phase 5 multi-canvas parity artifact exists, the only claim made anywhere is
> "in progress."

This recon supersedes the architecture-level notes in
[`DIFFUSIONGEMMA_26B_A4B_RECON.md`](DIFFUSIONGEMMA_26B_A4B_RECON.md) where the
two disagree: that document predates llama.cpp's diffusion runtime and states no
oracle exists. One now does (see §2), which is what unblocked this lane.

All facts below are verified against one of two sources, named per section:

- **the tracked GGUF** — tensor inventory + metadata extracted by
  `camelid inspect` (unmodified; the metadata layer already represents every
  quant type in the file). Artifacts: `target/dg-recon-20260611T160428Z/
  {tensor-inventory.json, metadata.json, quant-coverage.json, inspect-raw.json,
  make-inventory.py}`.
- **the pinned reference** — llama.cpp source at the pin (§2). File/line
  citations are against that commit.

Credit: the reference runtime, model graph, and sampler recon'd here are the
work of the llama.cpp / ggml authors (ggml-org/llama.cpp PR #24423). Parity for
this lane is defined against that implementation at the pinned commit.

## 1. Tracked artifact

One file is the tracked row; nothing inherits from resemblance:

| | |
|---|---|
| File | `diffusiongemma-26B-A4B-it-Q4_K_M.gguf` (16,806,810,336 bytes) |
| Source | `unsloth/diffusiongemma-26B-A4B-it-GGUF` |
| SHA256 | recorded in `target/dg-recon-20260611T160428Z/llamacpp-pin.json` |
| Why this quant | smallest quant published in the official repo (Q5_K_M 19.2 / Q6_K 22.6 / Q8_0 26.9 / BF16 50.5 GB); Q8_0 of a 26B exceeds the dev-machine envelope |

## 2. Pinned reference runtime

llama.cpp master cannot generate from DiffusionGemma. The reference is
**ggml-org/llama.cpp PR #24423** at pinned commit
**`7ea238c68b34ec8c99c28a68b9beed5b150cabef`** (head repo
`danielhanchen/llama.cpp`, branch `diffusion-visual-updates`; PR open at pin
time, so the pin is the commit SHA, never the branch). Local checkout + Metal
build: `/Volumes/Untitled/reference/llama.cpp-dg`. Full build/invocation/
determinism details: `target/dg-recon-20260611T160428Z/llamacpp-pin.json`.

Three binaries matter:

- `llama-diffusion-cli` — end-to-end generation (EB sampler auto-enabled for
  canvas models, `examples/diffusion/diffusion-cli.cpp:310`).
- `llama-diffusion-gemma-eval` — **the per-step logits oracle**: feeds golden
  token ids `[prompt | canvas]` through one forward and dumps raw f32
  canvas-row logits; optional previous-step logits input enables
  self-conditioning. Flash attention off by default. This is the Phase 2/3
  comparator; no instrumentation patch is needed.
- `llama-diffusion-gemma-server` — not used by this lane yet.

## 3. What the GGUF actually is (verified, `metadata.json`)

- `general.architecture = "diffusion-gemma"` (llama.cpp arch string,
  `src/llama-arch.cpp:61`); `general.size_label = "128x2.6B"`; file type 15
  (Q4_K_M).
- 30 layers, `n_embd` 2816, context 262144, **`diffusion.canvas_length = 256`**.
- MoE: 128 experts, 8 used, expert FFN 704; dense (shared-expert) FFN 2112.
- Attention: 16 Q heads; sliding-window pattern **5 SWA : 1 global**
  (`sliding_window_pattern`, global at layers 5,11,17,23,29). Per-layer-type
  dims: SWA layers head dim 256, 8 KV heads, rope base 10000; global layers
  head dim 512, 2 KV heads, rope base 1e6 + proportional rope via a shared
  `rope_freqs.weight` [256]. Sliding window 1024. QK-norm per head; RMS eps 1e-6.
- `final_logit_softcapping = 30.0`. `attention.causal = false`.
- Head is **tied** to `token_embd.weight` (Q6K, [2816, 262144]) — no
  `output.weight` in the file.
- Tokenizer: `tokenizer.ggml.model = "gemma4"`, vocab 262144, merges 514906,
  BOS 2 / EOS 1 / PAD 0 / UNK 3 / **MASK 4**, `add_bos_token = true`,
  `add_space_prefix = false`, chat template 17.5k chars (tool-calling macros).
  Phase 1 owns the details; nothing is inherited from the SPM-LLaMA lane.
- **No `diffusion.eb_*` keys in the file** — the reference sampler runs on its
  in-code defaults (§6).
- **No vision/video/audio tensors in this file at all** (inventory class
  `vision_video` is empty). The text-only scope is satisfied by the artifact
  itself; there is nothing to skip at load time.

Tensor inventory: 692 tensors, **zero unclassified**
(`tensor-inventory.json`; classes: embedding 1, attention 115, attention_norm
120, ffn_norm 150, dense_ffn 90, moe_router 60, moe_experts 90, region_scalar
60, self_conditioning 4, rope 1, final_norm 1). Notable shapes: fused
`ffn_gate_up_exps` Q4K [2816, 1408, 128] + `ffn_down_exps` Q8_0 [704, 2816,
128] + per-expert `ffn_down_exps.scale` F32 [128]; global layers have **no
`attn_v`** (V reuses the K projection, `src/models/gemma4-common.h:48-50`).

Quant types present: Q4K (194), Q5_0 (33), Q6K (14), Q8_0 (28), F32 (423).
Camelid already eager-decodes all five; the missing piece is the
**lazy/file-backed path for Q4K/Q5_0 (+ confirm Q6K CPU)** plus
reference-block parity tests — that is the whole of Phase 0.5
(`quant-coverage.json`).

## 4. Forward pass (pinned source: `src/models/diffusion-gemma.cpp`)

The launch-day "encoder-decoder with cross-attention" description is
**virtual, not structural**. There is one shared Gemma-4 backbone (shared with
the AR `gemma4` arch via `src/models/gemma4-common.h`); no second weight
stack, no dedicated cross-attention projections. A single no-cache
bidirectional forward over `[prompt | canvas]` reproduces the two-pass
(encoder prefill + decoder denoise) result. Exactly three things are
region-aware, split at `P = n_tokens − canvas_length` (header comment,
`diffusion-gemma.cpp:9-16`):

1. **Input embedding**: prompt rows = `embed × √n_embd`; canvas rows =
   `rms_norm_noscale(embed × √n_embd [+ self-conditioning])` (lines 321-384).
2. **Per-layer scalar**: prompt rows × `enc_layer_output_scale`, canvas rows ×
   `layer_output_scale` (lines 473-487). This is the entire "encoder mode".
3. **Attention mask** (lines 21-86): prompt queries causal over prompt only
   (SWA-clipped on sliding layers, never see canvas); canvas queries
   bidirectional over all canvas + prompt ("cross-attention" is just canvas
   queries hitting prompt keys in self-attention; on sliding layers the canvas
   sees only the last `n_swa−1` prompt positions).

Layer body (identical to AR gemma4, `gemma4-common.h`): RMS attn_norm → Q
proj + per-head q-norm + rope (proportional rope via `rope_freqs` on global
layers) → K proj + k-norm + rope; V = v_proj or raw K proj when absent, then
**no-scale RMS v-norm** → attention (no pre-attn scaling,
`f_attention_scale = 1.0`) → post_attention_norm → residual. FFN: dense
shared-expert MLP (gelu, `post_ffw_norm_1`) **plus** 128-expert MoE — router
input is the *unnormed* post-attention residual: `rms_norm_noscale(attn_out)
× 1/√n_embd × ffn_gate_inp.scale → ffn_gate_inp.weight → softmax top-8`
(`gemma4-common.h:92-106`); expert output → `post_ffw_norm_2`; dense + MoE
summed, → `post_ffw_norm`, residual. Final: output_norm → tied head →
logit softcap tanh(z/30)·30.

**Self-conditioning** (lines 321-365; tensors `self_cond_{pre_norm,gate,up,
down}`): previous step's raw canvas logits → `softmax(logits / prev_t)` →
matmul with the (dequantized, transposed) embedding → × √n_embd → gated MLP
(`pre_norm → gelu(gate)·up → down`) → added to the canvas embedding before its
final no-scale RMS norm. Gated to zero on the first step (`sc_use = 0`), which
is the "zero-SC exactness forward" the eval binary exercises by default.

Three execution phases (`llama_diffusion_set_phase`, lines 730-742): UNIFIED
(one `[prompt|canvas]` pass, the parity-simplest), PREFILL (prompt only, writes
an F32 prompt-KV store), DECODE (canvas only, prepends cached prompt K/V).
The cached path matches unified only **to F32 round-off**
(`diffusion-gemma-eval.cpp:112`) — the lane's parity contract therefore pins
**UNIFIED** mode.

## 5. Loader contract (what Camelid must enforce)

From `load_arch_hparams`/`load_arch_tensors` (lines 166-277): requires
`diffusion.canvas_length > 0`; `n_embd_head_k == n_embd_head_v` (both
per-layer-type); `attn_v` optional per layer (absent ⇒ V = K proj); fused
`ffn_gate_up_exps` preferred over split gate/up (this file fuses); per-expert
scale loads from `ffn_down_exps.scale`; `rope_freqs` is a single shared
tensor; head tied when `output.weight` absent; 30 layers ⇒ type 26B-A4B.

## 6. Entropy-Bound sampler (pinned source: `examples/diffusion/diffusion.{h,cpp}`)

The real DiffusionGemma decode loop (`diffusion_generate_entropy_bound`,
`diffusion.cpp:442-711`), auto-enabled when `diffusion.canvas_length > 0`:

- **Canvas init is RANDOM TOKENS, not mask tokens**: uniform draw over the
  whole vocab per position (line 473). The `<mask>` token (id 4) exists in the
  vocab but plays no role in this sampler.
- **Defaults** (in-code; the GGUF carries no `diffusion.eb_*` overrides):
  `max_denoising_steps 48, t_max 0.8 → t_min 0.4 (linear in step), entropy
  bound 0.1, stability_threshold 1, confidence_threshold 0.005, seed 0`.
- **Per step** at temperature `t`: one forward over the working canvas; per
  position compute argmax, entropy of `softmax(raw/t)`, and one multinomial
  sample (inverse-CDF with pre-drawn uniform `u[pos]`); sort positions by
  entropy ascending and **accept** while the cumulative entropy of strictly
  earlier positions ≤ 0.1 (the lowest-entropy position always passes); rebuild
  the working canvas = sampled token where accepted, **fresh random token**
  where not (renoise); the *output* canvas is the **argmax** canvas.
- **Self-conditioning**: this step's raw canvas logits feed the next step at
  `1/prev_t`; off at step 0.
- **Adaptive stop**: argmax canvas unchanged for `stability_threshold` steps
  AND mean entropy < 0.005.
- **Determinism**: one `std::mt19937(seed)`; all randomness pre-drawn
  single-threaded in fixed order (canvas init, then per step `u[pos]` +
  renoise tokens; lines 467-473, 587-591). Host path is seed-reproducible.
  The CUDA `gpu_sample_reduce` path is explicitly **not** bit-exact (reduction
  order; `diffusion.cpp:637-638`) and must be off for parity runs. Residual
  cross-backend FP nondeterminism of the forward itself remains — every parity
  artifact must name the backend and comparison level. This caveat travels
  with every downstream parity claim.

## 7. Multi-canvas (block-autoregressive) loop (`diffusion-cli.cpp:413-485`)

Lives in the CLI, not the library: per block, run the EB denoiser over
`[prefix | 256-token canvas]`; trim the finished **argmax** canvas at the
first end-of-generation token or repetition loop (a token recurring at stride
1-2 for ≥ 6 steps); append the trimmed canvas to the prefix; repeat until an
end token, the block budget (`-n` → ⌈n/256⌉ blocks), or the ubatch limit.
The published "re-encoded and appended to the KV cache" is, in the reference,
simply the next block's forward over the grown prefix (re-prefill; in cached
mode a new PREFILL of the full prefix). The RNG stream continues across blocks
(no reseed). Phase 5's gate is parity over ≥ 2 blocks of exactly this loop.

## 8. Phase 0 gate status

- `tensor-inventory.json` + `metadata.json` exist; **zero** unclassified
  tensors; vision/video disposition: not present in the file.
- Reference runtime pinned, built, invocation + determinism knobs recorded
  (`llamacpp-pin.json`).
- Quant-coverage delta enumerated (`quant-coverage.json`) ⇒ Phase 0.5 scope.
- This document.

Next phase: **0.5 — quantization coverage** (lazy K-quant/Q5_0 load +
dequant parity vs llama.cpp on same-file blocks).
