# Camelid Status

Last updated: 2026-05-01

`STATUS.md` is Camelid's current release-evidence checkpoint. It answers three questions: **what can Camelid prove today, what changed recently, and what still blocks the next support move?** Treat it as a snapshot, not a diary. Detailed historical run logs, older validation slices, and superseded tactical notes now live in [`STATUS_ARCHIVE_2026-04.md`](STATUS_ARCHIVE_2026-04.md). Read it as evidence and blocker state, not as aspiration.

Executive summary: runtime capability improved at the blocker seam, but the public support boundary did not move.

## Release ledger snapshot

Camelid follows the same four-lane release ledger across the README, compatibility matrix, API capability reporting, and frontend readiness copy. If another surface sounds broader, treat it as stale and bring it back to this ledger. The purpose of this file is not to sound ambitious; it is to record exactly what the current evidence can defend, no more and no less.

- **Supported generation gate:** TinyLlama 1.1B Chat Q8_0 remains the only supported end-to-end generation lane.
- **Evidence-only lane:** Llama 3.2 1B Instruct Q8_0 remains narrow evidence only.
- **Acceptance target:** Llama 3.2 3B Instruct Q8_0 remains the exact WebUI target. The exact GGUF now loads through `/api/models/load` with low backend RSS after streaming metadata parsing, but the guarded first-chat retry is still blocked before the first generated token under host free-page pressure.
- **Groundwork-only lane:** Llama 3 8B Instruct Q8_0 remains below supported generation until lazy or on-demand Q8 execution and bounded parity and memory evidence exist.

Two standing rules apply to every row:

- **Support rule:** Nothing inherits support across model size, quantization, tokenizer lane, API surface, or frontend state.
- **Credit rule:** Visible llama.cpp / ggml acknowledgement and the MIT notice remain part of any parity-backed release claim.

For the formal support ledger, see [`COMPATIBILITY.md`](COMPATIBILITY.md). For sequencing, see [`ROADMAP.md`](ROADMAP.md).

## What improved without changing the support line

Recent work improved the blocker seam without changing the release ledger:

- TinyLlama Q8_0 remains the trusted supported gate.
- Llama 3.2 1B Q8_0 remains informative evidence only.
- Llama 3.2 3B Q8_0 now has successful metadata/API load behavior with low backend RSS after streaming metadata parsing, and file-backed lazy-Q8 recovery materially reduced the earlier eager dense-load spike.
- Llama 3 8B Q8_0 remains groundwork-only until lazy or on-demand execution is wired through attention, FFN, and output projection and then validated with bounded artifacts.

Bottom line: the engineering seam moved forward, but no new support claim was earned.

## Current support evidence

The sections below summarize the artifact-backed boundary for each tracked row. They are intentionally narrower than "what might be close." If a supporting artifact is not called out here or in the linked files, Camelid should not imply the claim elsewhere.

### TinyLlama 1.1B Chat Q8_0

Status: **supported current gate**

Current evidence boundary:

- Five-prompt, 50-token parity audit against known-good llama-server.
- Prompt token IDs, generated token arrays, and generated text match.
- The token-major `output.weight` interpretation remains a protected correctness guardrail.

Representative artifacts:

- `target/edge-prompt-audit-fixed-20260428T1530/short.json`
- `target/edge-prompt-audit-fixed-20260428T1530/trailing-spaces.json`
- `target/edge-prompt-audit-fixed-20260428T1530/special-chars.json`
- `target/edge-prompt-audit-fixed-20260428T1530/longer.json`
- `target/edge-prompt-dequant-default-20260428T1604/multiline-long-default-50.json`
- `target/chat-parity-postfix-50-token-audit.json`

### Llama 3.2 1B Instruct Q8_0

Status: **evidence only / not a supported gate**

Current evidence boundary:

- One compact-header `hello` prompt matches llama.cpp through five deterministic generated tokens.
- Prompt IDs and generated IDs match for `[9906,0,2650,649,358]` / `"Hello! How can I"`.

Representative artifacts:

- `target/autonomous-small-model-parity-20260429T134615Z-head-9049492/llama32-1b-q8-chat-parity-5tok.json`
- `target/qa-small-model-parity-20260429T1338Z-head-35bfd58/`

### Llama 3.2 3B Instruct Q8_0

Status: **acceptance target / blocked before first token**

Current evidence boundary:

- The exact tracked GGUF is present locally.
- Metadata and `/api/models/load` behavior work with low backend RSS after streaming metadata parsing.
- Recent file-backed lazy-Q8 recovery materially reduced the older eager dense-load spike.
- The guarded first-chat retry still stopped before any generated token under host free-page pressure.

Representative artifacts:

- `target/llama32-3b-streaming-metadata-20260430T233604Z/`
- `target/llama32-3b-nocache-rowread-20260430T233844Z/`

Promotion remains blocked until Camelid has bounded prompt-token, first-token, short-generation, memory, API, and WebUI evidence for this exact row.

### Llama 3 8B Instruct Q8_0

Status: **groundwork only / generation blocked**

Current evidence boundary:

- Metadata, config, tokenizer, and chat-template handling are fixture-guarded.
- Independent tokenizer reference fixtures exist.
- Q8_0 retained-block loading and serial row/all-row dot groundwork exist.
- Generation remains blocked until lazy or on-demand Q8 execution is wired through attention, FFN, and output projection and then validated with bounded artifacts.

Representative artifacts:

- `target/backend-small-model-readiness-20260429T131209Z/`
- `target/perf-cron-20260429T122814Z-single-row-adapter-head-da53871/`

## Latest promotion-relevant work

### Docs professionalism pass

The top-level documentation set was tightened for executive readability, hierarchy, and release consistency without changing support truth. `README.md`, `COMPATIBILITY.md`, `ROADMAP.md`, and `STATUS.md` remain the public sources of truth. The README now pairs the front-door support ledger with a clearer reading order, while visible llama.cpp / ggml acknowledgement and the MIT notice remain intact wherever reference tooling and parity evidence depend on them. Recon and planning docs continue to carry explicit note banners.

### Llama 3.2 3B lazy-Q8 recovery

Recent backend work kept the support contract unchanged while improving the 3B execution seam:

- streaming metadata parsing moved `/api/models/load` to low backend RSS for the exact 3B artifact
- file-backed Q8 linear handling reduced the older eager dense-load spike
- a no-cache/row-read follow-up still stopped before any generated token under host free-page pressure

This is useful blocker-reduction evidence, not a support promotion.

## Next blocking work

In order of importance:

1. Preserve the TinyLlama Q8_0 supported gate.
2. Finish lazy or on-demand Q8 execution for larger LLaMA-family rows.
3. Capture bounded first-token and short-generation evidence for Llama 3.2 3B Q8_0.
4. Keep docs, `/api/capabilities`, and frontend readiness copy aligned with the exact-row support contract.

## Validation note

This file is intentionally a snapshot, not a diary. When a change materially affects support or its blockers:

- add the current evidence summary here
- keep the detailed run log and older slices in `STATUS_ARCHIVE_2026-04.md` or later archives
- update `COMPATIBILITY.md`, `ROADMAP.md`, and user-visible readiness copy in the same change window when support language changes
