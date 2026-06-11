# DiffusionGemma 26B-A4B — architecture recon (recognized, fail-closed)

> [!NOTE]
> This is a design/recon note, not the public support ledger. For current
> support truth use [`COMPATIBILITY.md`](../../COMPATIBILITY.md) and
> [`STATUS.md`](../../STATUS.md). DiffusionGemma is **recognized and fails
> closed** with a typed blocker; it is **not** a supported runtime row, and
> nothing here is a support claim.

## What it is

`google/diffusiongemma-26B-A4B-it` — config `architectures:
["DiffusionGemmaForBlockDiffusion"]`, `model_type: diffusion_gemma`. It is built
on the Gemma 4 26B-A4B MoE foundation (the row Camelid *does* support, as Q4_0
QAT over the two-Mac distributed lane), but the generation paradigm is
fundamentally different — from Google's model card:

- **Discrete text diffusion** — generation is block-autoregressive *multi-canvas
  sampling*, not token-by-token autoregression. The model iteratively denoises a
  block of tokens (a "canvas") in parallel with a diffusion sampler, appends the
  finished canvas to the KV cache, then denoises the next canvas.
- **Encoder-decoder** — an autoregressive encoder prefills the prompt and builds
  the KV cache; a decoder applies **bidirectional attention** over the canvas and
  reaches the prompt context via **cross-attention**.
- **Entropy-Bound (EB) sampler** — the recommended diffusion sampler.
- **Multimodal** — interleaved text + image (variable aspect/resolution) + video
  inputs → text output.

GGUF quants published (unsloth): BF16 50.5 GB, Q8_0 26.9 GB, Q6_K 22.6 GB,
Q5_K_M 19.2 GB, Q4_K_M 16.8 GB.

## Why Camelid fails closed on it (today)

Camelid is a **decoder-only autoregressive** engine: causal attention, a
per-layer KV cache, and greedy next-token decode. DiffusionGemma needs, none of
which exist here:

1. A **diffusion decode loop** — multi-canvas iterative denoising, not
   next-token argmax.
2. **Bidirectional decoder attention** over a denoising canvas (Camelid attention
   is strictly causal).
3. An **encoder-decoder split with cross-attention** (Camelid is decoder-only).
4. The **Entropy-Bound diffusion sampler**.
5. **Multimodal** image/video towers (Camelid fails closed on multimodal input).
6. **No autoregressive reference comparator** — llama.cpp's autoregressive greedy
   path cannot produce an oracle for diffusion sampling, so even a from-scratch
   implementation could not be validated to Camelid's token-parity standard
   without a different, diffusion-aware reference and a determinism contract.

Memory/quant also rule out the easy paths: Q8_0 is 26.9 GB (same envelope block
as the regular 26B A4B Q8_0), and the smaller files are K-quants
(Q4_K_M/Q5_K_M/Q6_K) — formats Camelid's wire lane does not implement (it
supports Q8_0, Q4_0, and Q6_K read-in-place wire blocks).

Recognition + blocker live in `LlamaModelConfig::from_gguf` (any
`general.architecture` containing `diffusion` → typed
`UnsupportedModelArchitecture` naming the paradigm mismatch). Locked by
`tests/gemma4_metadata.rs::diffusiongemma_architecture_fails_closed` across the
`diffusion_gemma` / `diffusiongemma` / `gemma-diffusion` spellings.

## The honest path to real support (a genuine new frontier)

This is a substantial new lane, not a row addition. In rough order:

1. **Decide the determinism/parity contract first.** Diffusion sampling is
   iterative and sampler-dependent; define what "reproducible" and "parity"
   mean (e.g. fixed EB-sampler schedule + seedless greedy-equivalent denoising)
   and what reference produces the oracle. Without this there is no claimable
   support, by repo doctrine.
2. **Encoder path** — reuse the gemma4 autoregressive forward for the prompt
   encoder + KV cache (largely existing).
3. **Decoder path** — new: bidirectional attention over the canvas + cross-
   attention to the encoder cache.
4. **Diffusion sampler** — the multi-canvas denoising loop + EB sampler.
5. **K-quant wire kernels** (Q4_K_M/Q5_K_M) or run Q8_0 distributed (26.9 GB →
   two-Mac).
6. **Multimodal** — out of scope until the text path is proven; stays
   fail-closed.

Until 1–4 exist and are proven against a diffusion-aware oracle, DiffusionGemma
stays recognized-and-blocked — which is the correct, honest state for it now.
