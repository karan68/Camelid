# Phase 4 finding: the CUDA windowed lane is NOT token-identical to the reference

**Verdict: the gemma3 CUDA-resident lane must not be promoted.** It routes, it is
GPU-resident, and it is fast — and it produces a different token stream from both
the in-tree CPU reference and the pinned llama.cpp oracle on one of five sub-512
prompts. That is disqualifying on its own terms, independent of how large the
difference is.

## What was measured

Comparator: `llama.cpp 9632 (acd79d603)`, CPU backend, `-ngl 0 -ctk f32 -ctv f32
-fa off --no-repack`, greedy — replayed from PR #560's committed capture
`qa/evidence-bundles/gemma3-1b-q8-gpu-resident-parity-20260730-head-6eaf9053/oracle-short.json`.
Pack: the committed 5-prompt sub-512 gate pack. One engine resident at a time.

| Lane | Depths | Result |
|---|---|---|
| CUDA resident windowed (`cuda_resident_windowed_runtime`) | 1 / 5 / 50 | **10 / 15 legs** token-and-text identical |
| CPU runnable bridge, same host, same oracle (control) | 1 / 5 | **10 / 10 legs, `all_pass: true`** |

## Why this is a defect and not an artifact

Three things had to be ruled out, and all three were:

1. **Not a host artifact.** The runnable bridge on THIS Windows box reproduces the
   Mac-captured oracle exactly at depths 1 and 5. The capture travels.
2. **Not a near-tie.** At the divergent position the CUDA lane ranks `The` at
   −0.4486 and `This` at −1.1775 — a **0.729 nat** top-2 gap. The repo's soft
   near-tie line is 0.33 nats, and the most contested flip in the entire #560
   bundle was 0.4471 nats and was labelled its weakest attribution. This is more
   than either.
3. **Not FP reassociation in the attention kernel.** At these prompt lengths
   (16–19 tokens) `launch_attention` resolves to G = 1, so the weighted-V
   accumulation is a straight sequential sum — the reassociated path is not even
   taken. Nor is f16-KV rounding a candidate: that perturbs logits at the ~1e-3
   level, three orders of magnitude below the observed gap.

## The divergence

CUDA agrees with the runnable lane EXACTLY on 4 of 5 prompts at depths 1 and 5.
It diverges on one, from the very first generated token:

| Prompt | Depths diverging | Reference | CUDA |
|---|---|---|---|
| "What color is the sky on a clear day?" | **1, 5, 50** | `This` | `The` |
| "What is the capital of France?" | 50 only | …`the *only* capital`… | …`the capital of *all* of France`… |
| "Name the largest ocean on Earth." | 50 only | `165.25 million` | `165 million` |

A depth-1 divergence cannot be drift accumulation. The two depth-50-only rows may
be downstream of ordinary long-horizon divergence and are NOT independently
attributed here.

## What is ruled OUT as the cause so far

- **The window mask itself.** At 16–19 prompt tokens the window (512) never clips:
  `start = 0` on every sliding layer, so `attention_decode_sw` is arithmetically
  identical to the full-causal kernel. The bug is in something else that the
  windowed session turns on.
- **Schedule derivation.** The runnable reference builds its per-layer RoPE base
  from `cfg.gemma3.rope_freq_base_at(i)` — the SAME `Gemma3Metadata` accessor this
  lane uses. Both lanes read one source of truth.
- **Missing tensors.** The row carries `attn_q_norm`, `attn_k_norm`,
  `post_attention_norm` and `post_ffw_norm` on every block, and the engine refuses
  a gemma3 layer whose sandwich norms are unbound, so they are bound.

## What is NOT yet ruled out

The remaining candidates are all inside the per-layer forward, and black-box
probing cannot separate them:

- dual-θ table selection reaching the wrong layers (mapping, not derivation)
- the QK-norm application on the gemma3 geometry
- sandwich-norm placement relative to the residual
- the GeGLU activation
- the embedding scale
- a prefill/decode asymmetry: prefill runs token-by-token through
  `forward_token_hidden` (new in this campaign), decode through `forward_token`

## Next step

Localize in-process rather than over HTTP: a CUDA twin of
`metal.rs::gemma3_real_row_resident_forward_matches_runnable_oracle`, comparing
the resident forward against the runnable oracle layer by layer on the failing
prompt. A per-layer comparison names the defect in one run; every further
black-box probe only re-confirms that one exists.

## Scope of this finding

Sub-512 pack only. The windowed (>512) pack was NOT run — running it before the
sub-512 divergence is understood would only produce a second unexplained result.
No throughput claim is made or implied here.
