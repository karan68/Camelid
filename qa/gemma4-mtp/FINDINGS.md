# Gemma 4 26B-A4B MTP assistant — architecture findings (2026-08-24)

Work toward MTP on the Windows/CUDA ghost lane. The assistant forward is being brought up
against the committed BF16 oracle **before** any kernel work, because it is self-contained
and target-free: the oracle needs no 26B weights at all.

**Status: NOT YET EXACT.** Recurrent-hidden cosine **0.547**, magnitude 215.64 vs the
oracle's 220.88 (within 2.4%). Up from 0.176 at first run. Five real semantic errors found
and fixed; at least one remains. Do not build on this until it reaches parity.

## Provenance — everything below is hash-verified, not inferred

| artifact | sha256 (prefix) | source |
|---|---|---|
| `model.safetensors` | `c082cc58…` | `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant` @ `9537141506fe…` |
| `config.json` | `23d2bc4a…` | same repo/revision |
| `modeling_gemma4_assistant.py` | `a77e6767…` | `huggingface/transformers` @ `0c92811846…` |
| `modeling_gemma4.py` | `51d9c119…` | same revision |
| `modeling_rope_utils.py` | (fetched) | same revision |

The first four match `qa/evidence-bundles/gemma4-26b-mtp-assistant-oracle/manifest.json`
exactly. **Fetch the reference source instead of reverse-engineering from config** — doing
the latter cost several wrong guesses before the sources settled every question in minutes.

## The five corrections (each one was measurably wrong)

1. **★ RMSNorm is `normed * weight`, NOT Gemma 2/3's `normed * (1 + weight)`.** Gemma 4
   changed the convention (`Gemma4RMSNorm.forward`). This was the single biggest error —
   it corrupts every norm in the model. Fixing it moved cosine 0.20 → 0.48 and brought the
   magnitude from 295 to within 2% of the oracle.
2. **★ `self.scaling = 1.0`** — attention does NOT scale by `head_dim**-0.5`. Gemma 4 relies
   on `q_norm` instead. Note the diagnostic harness has a `scaling is None` fallback to
   `head_dim**-0.5` that is never taken; do not copy it.
3. **`layer_scalar` multiplies the ENTIRE layer output**, residual included, as the last
   statement of the decoder layer (`hidden_states *= self.layer_scalar`). Not a branch scale.
   Values here: 0.297 / 0.516 / 0.535 / 0.426.
4. **Proportional RoPE (full-attention layer only)**: `rope_angles = int(0.25 * 512 // 2) = 64`
   frequencies `1/(1e6^(2i/512))`, then **192 zeros** padding to `head_dim/2`. So 3/4 of the
   dims are NOT rotated, but cos/sin remain full `head_dim` length and `rotate_half` still
   spans all 512. Sliding layers use plain default RoPE, θ=1e4, fully rotated.
5. **Sliding window is the LAST 1024 of 1031** (`[7, 1031)`), despite the double flip in
   `create_attention_masks`. Measured decisively: 0.547 windowed vs 0.161 unwindowed. With
   `scaling = 1.0` the softmax is extremely peaked, so 7 positions out of 1031 swing the
   result enormously — window boundaries are not a rounding detail here.

## Architecture, confirmed

- 4 layers: 3 × `sliding_attention` (16 q-heads × 256, 8 KV heads), 1 × `full_attention`
  (16 q-heads × 512, 2 KV heads). GQA maps q-head `h` to kv-head `h // n_rep`.
- **No `k_proj`/`v_proj` anywhere.** All 4 layers are `is_kv_shared_layer`; K/V arrive from
  the host and are consumed RAW — no RoPE, no norm applied to them.
- Shared-KV contract: sliding ← host layer 28, full ← host layer 29, `position_id` = the
  shared-KV logical length (1031), i.e. the still-unforwarded bonus token's position.
- `pre_projection` [1024, 5632] consumes `concat(target_scaled_embedding,
  target_final_normalized_hidden)` — **embedding first**.
- `post_projection` [2816, 1024] produces `last_hidden_state`, the recurrence input.
- `logits = lm_head(h_1024)` — a plain tied matmul, because `use_ordered_embeddings: False`.
  The `Gemma4AssistantMaskedEmbedder` centroid path (`num_centroids` 2048,
  `centroid_intermediate_top_k` 32) exists but is **inactive for this checkpoint**.
- Layer order: `input_layernorm → attn → post_attention_layernorm → +residual →
  pre_feedforward_layernorm → mlp → post_feedforward_layernorm → +residual → *layer_scalar`.
- `hidden_activation: gelu_pytorch_tanh`, `rms_norm_eps: 1e-6`.

## What is still wrong

Cosine 0.547 with the magnitude essentially correct. A correct-magnitude / wrong-direction
error is the signature of a rotation or an indexing/ordering fault, not a scale bug. Prime
suspects, in order:
1. The reference runs **BF16 end to end**, casting after every op (`type_as`), while this
   harness is f32 throughout. With `scaling = 1.0` and a very peaked softmax, BF16 rounding
   of the scores is not obviously second-order — worth testing before anything else.
2. The bidirectional mask may contribute more than the plain window crop modelled here.
3. RoPE frequency indexing on the sliding layers.

**The efficient way to finish this is per-stage comparison, not more end-to-end guessing.**
`generate_stage_diagnostics.py` in the oracle bundle emits every intermediate
(`layer.N.input_norm`, `.q_proj`, `.q_norm`, `.q_rope`, `.attention_scores`,
`.attention_probs`, `.attention_context`, `.o_proj`, …). Its output is NOT committed — only
the generator is. Running it needs torch 2.13 + transformers 5.16.0.dev0 on arm64, which
this box is not. Either run it on the Mac and commit the JSON, or relax the generator's
pinned-version `require()` calls and run it here.

## Files

- `mtp_inputs.py` — port of the oracle's deterministic BF16 bit-pattern generator.
  **All 6 inputs reproduce the oracle's recorded sha256 bit-exact**, so the harness is
  verified on its input side.
- `mtp_forward.py` — numpy reference forward (safetensors reader + 4-layer forward).
  Superseded in places by the corrections above; treat FINDINGS.md as authoritative.

Related: `qa/evidence-bundles/gemma4-26b-mtp-assistant-oracle/`,
`qa/gemma4/oracle/google_gemma-4-26B-A4B-it-Q4_0.PARITY-STATUS.md`.
