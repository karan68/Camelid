# 2026-05-06 — Overnight 8B Q8 gap-lane triage

Scope: diagnostic/performance evidence and guardrail alignment only. This does not promote Llama 3 8B 1024/2048 support and does not widen API/capabilities/frontend claims.

Starting state:

- Local main/head at start of this slice: `97ac0a3` (`Keep quickstart 8B context boundary red`), clean and tracking `origin/main`.
- Canonical Ubuntu active-run check at `2026-05-07T02:20:46Z` found no active high-CPU long 8B parity/perf run to duplicate; only older idle `backendinference serve` processes and a stale completed diagnostic watcher were visible.

Evidence reviewed on the canonical Ubuntu lane:

- Remote artifact root: `/home/ubuntu/work/Camelid-gap-lane-20260507T005918Z/target/gap-lane-patched-20260507T010345Z-head-3907e6c8bef9-dirty/`
- Unit gates captured there:
  - `q8-tests.log`: 9 Q8/file-backed reader tests passed, including batch chunk-read reuse and parallel-reader guardrails.
  - `prefill-chunk-tests.log`: `prefill_layer_major_chunk_token_count_has_separate_headroom_default` passed.
- 8B diagnostic prompt-pack outputs captured there:
  - `llama3-8b-context-1024/pack/summary.json`: PASS, 881 prompt tokens in 1024 context, prompt tokens/generated tokens/generated text all matched.
  - `llama3-8b-context-2048/pack/summary.json`: PASS, 1910 prompt tokens in 2048 context, prompt tokens/generated tokens/generated text all matched.
  - `progress.log`: 1024 run `2026-05-07T01:04:35Z`–`01:08:07Z`; 2048 run `2026-05-07T01:08:07Z`–`01:16:18Z`.

Local follow-up in this slice:

- Documented `BACKENDINFERENCE_PREFILL_LAYER_MAJOR_CHUNK_TOKENS`, the layer-major-only chunk-size knob added for 8B long-prefill headroom/comparison work.
- Kept the wording explicit that this is a runtime/performance knob only, not support evidence or an 8B 1024/2048 promotion signal.

Claim boundary:

- The reviewed Ubuntu artifact root is useful diagnostic/performance evidence, but it is tagged `dirty` and is not a copied/scrubbed public promotion bundle from current local head.
- Current support surfaces must keep Llama 3 8B 1024/2048 red/not-promoted until row-specific PASS artifacts and docs/API/frontend alignment land together in a deliberate promotion slice.
