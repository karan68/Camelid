# Phase 5 — the gemma4 Q4_0 mis-decode

**The root cause, the fix and its primary receipt now live on `main`**, landed
independently as PR #569 ("gemma4 CUDA: route the tied head through its lane's
GPU layout"). See `qa/gemma4-cuda-head-repack/`.

This directory previously carried a duplicate of that write-up and capture,
because the investigation ran in the same worktree as this campaign and its
changes were swept into this branch's commits before it had its own PR. The
duplicates are removed rather than kept in parallel: two copies of one receipt
is exactly the drift this repo's gates exist to prevent, and `main`'s is the
reviewed one.

What remains here is only what belongs to THIS campaign:

| File | Why it is here and not in #569 |
|---|---|
| `q4_0-parity-reverify.json` | An INDEPENDENT re-run of #569's harness on this campaign's branch and host, before relaxing the admission gate to admit Q4_0. 5/5 token-identical, `all_pass`. Kept because this campaign should not admit a quant on someone else's receipt alone. |
| `e2b-q4_0-cuda-admitted-health.json` | The admission decision, which IS this campaign's code: `/v1/health` showing `gemma4_cuda_resident_runtime` with the plan reason naming Q8_0/Q4_0/Q4_1 as receipted. #569 fixed the decode; this shows the gate letting it through. |

Both are scrubbed of operator paths (see the scrub commit).
