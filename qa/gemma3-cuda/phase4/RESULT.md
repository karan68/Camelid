# Phase 4 result — gemma3 on the CUDA resident windowed lane

Supersedes `FINDING-cuda-diverges.md`, whose structural diagnosis was **wrong**.
That file is kept for the audit trail; this one is the verdict.

Comparator throughout: `llama.cpp 9632 (acd79d603)`, CPU backend,
`-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack`, greedy — replayed from PR #560's
committed captures. One engine resident at a time.

## Headline

| Pack | Legs | Result |
|---|---|---|
| **Above the window** — 606 / 1205 / 2403 prompt tokens, depths 1/5/50 | 9 | **9/9 token-AND-text identical, `all_pass: true`**, tokenization 3/3 |
| Below the window — 5-prompt gate pack, depths 1/5/50 | 15 | **10/15** token and text; 5 legs across 3 prompts diverge |

Prompt tokenization is identical to the oracle on every prompt in both packs
(5/5 and 3/3), so both lanes provably saw the same input.

The above-the-window pack is the one that matters. Below 512 a full-causal
forward and a correctly-windowed forward agree by construction, so that pack
cannot distinguish a live mask from a dead one. Above it they cannot agree by
accident: #560 measured the maskless runnable bridge diverging at generated index
2 on the same 606-token prompt and never resynchronising. This lane matches the
oracle at 4.69x the window.

## The sub-512 divergences, correctly diagnosed

Five legs across three prompts:

| Prompt | Depths diverging | Reference | CUDA |
|---|---|---|---|
| "What color is the sky on a clear day?" | 1, 5, 50 | `This` | `The` |
| "What is the capital of France?" | 50 | …`the *only* capital`… | …`the capital of *all* of France`… |
| "Name the largest ocean on Earth." | 50 | `165.25 million` | `165 million` |

The first is the informative one: a divergence at depth 1 cannot be drift
accumulation. The two depth-50-only rows share a long correct prefix with the
oracle and are ordinary long-horizon divergence downstream of the same numerics;
they are NOT independently attributed.

A control run pins the attribution: the CPU runnable bridge on this same host,
against this same replayed oracle, is **10/10 at depths 1 and 5 (`all_pass:
true`)**. So the capture travels to this host and the in-tree CPU reference
reproduces it exactly — CUDA diverges from both.

**It is not structural.** A per-layer hidden-state trace of both lanes on that
exact prompt (`cuda-layers.tsv` / `cpu-layers.tsv`, captured via the
`CAMELID_LAYER_DUMP` instrument, keyed on POSITION) shows:

- worst relative L2 difference across all 494 (position, layer) pairs: **1.89%**
- at the final position, per-layer difference grows smoothly **0.15% (layer 0) →
  ~0.67% (layer 23)**
- **no step change at any layer**

A wrong QK-norm, sandwich-norm placement, GeGLU, embed scale or dual-theta mapping
produces a step change at the layer that carries it. There is none. The forward is
structurally correct; what remains is accumulated numerical difference.

Source of that difference, by design and not specific to gemma3: the CUDA resident
engine quantizes activations to Q8_0 per GEMV and stores KV as f16, while the
runnable oracle is f32 throughout. Those lanes are not numerically equivalent, and
that is a property of the CUDA engine that every row on it shares.

### A reasoning error worth recording

An earlier pass measured CUDA's top-2 margin at that position (`The` −0.4486,
`This` −1.1775, gap **0.729 nats**) and concluded "too large to be a near-tie,
therefore a bug". That inference was invalid: 0.729 nats is CUDA's margin in its
OWN distribution and says nothing about the reference's margin. A sub-1% shift in
a 1152-wide hidden state, projected through a 262k-row `lm_head`, moves individual
logits by exactly this order. The per-layer trace, not the logit gap, is what
settles structure.

## What is claimed

- The windowed forward on CUDA is token-and-text identical to the pinned oracle at
  606 / 1205 / 2403 prompt tokens, depths 1/5/50, zero flips.
- Below the window, **10 of 15** legs are identical. The 5 that are not are
  enumerated above and attributed to lane numerics rather than structure, on the
  evidence of the per-layer trace. This pack does NOT pass, and the row is not
  promoted on the strength of it.

The asymmetry is worth stating plainly rather than smoothing over: this lane is
**perfect on the hard pack and imperfect on the easy one**. That is what a
structurally-correct forward with a numerically-different arithmetic lane looks
like — the windowed pack's prompts are answerable from one sentence and their
continuations are not close calls, while the short pack contains at least one
genuinely borderline next-token choice. It is not evidence that the window works
and the rest does not.

## What is NOT claimed

- **No throughput number.** The lane is obviously much faster than the CPU bridge
  on this host, but no measurement phase has run and none is quoted here.
- **No bit-exactness** with the f32 runnable lane. It is not bit-exact by
  construction.
- **The sub-512 flip is disclosed, not adjudicated.** Confirming it is a genuine
  near-tie from the ORACLE's side needs a live llama.cpp to re-score the position;
  the replayed capture cannot be probed. Not run.
- No claim for any other gemma3 size or quant, and none for context above 2,403
  prompt tokens.
- The instrument (`CAMELID_LAYER_DUMP`) is a debugging lane: it forces a sync and
  a D2H per layer. It is never set in production or in any gate.
