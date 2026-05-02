# Camelid Status

Last updated: 2026-05-01

`STATUS.md` is Camelid's current release-evidence checkpoint. It records what Camelid can prove today, what moved recently, and what still blocks the next support change. Treat it as a briefing memo, not a diary. Detailed historical run logs, older validation slices, and superseded tactical notes now live in [`STATUS_ARCHIVE_2026-04.md`](STATUS_ARCHIVE_2026-04.md).

Use this file to answer three practical questions: what is supported now, what changed recently, and what still blocks the next support move?

Executive summary: runtime capability improved at the 3B/8B blocker seam, but the public support boundary did not move.

## Release ledger snapshot

Camelid follows the same four-lane release ledger across the README, compatibility matrix, API capability reporting, and frontend readiness copy. If another surface sounds broader, treat it as stale and bring it back to this ledger. The purpose of this file is simple: record exactly what the current evidence can defend, no more and no less.

Reading rule for the matrix: each row should answer three questions in plain English — what is validated now, what gates are still missing, and what exact blocker prevents promotion to the next release label.

For a fast read, the current answer is:

- **Supported generation gate:** TinyLlama 1.1B Chat Q8_0 remains the only supported end-to-end generation lane.
- **Evidence-only lane:** Llama 3.2 1B Instruct Q8_0 remains narrow evidence only.
- **Acceptance target:** Llama 3.2 3B Instruct Q8_0 remains the exact WebUI target. The exact tracked GGUF now loads successfully through `/api/models/load` with low backend RSS, and fresh Ubuntu backend-only artifacts now include repeat 5-token generation plus a bounded 50-token success for that same exact row, but this is still not a supported row. Support remains frozen until prompt-token parity, broader prompt/chat-template parity, API evidence, and WebUI evidence exist.
- **Groundwork-only lane with backend evidence:** Llama 3 8B Instruct Q8_0 still sits below supported generation, but it now has repeat bounded Ubuntu backend-only 5-token generation, prompt-cache validation, a fresh 50-token backend/API artifact, raw `hello` prompt-token parity, basic API smoke evidence, and a clearly passed memory gate for the exact tracked Q8_0 GGUF. Support remains frozen until broader prompt/chat-template parity, WebUI readiness, and performance/portability evidence exists.
- **Explicit non-claim:** no Llama 3-family row is a supported generation lane today.

Two standing rules apply to every row:

- **Support rule:** Nothing inherits support across model size, quantization, tokenizer lane, API surface, or frontend state.
- **Credit rule:** Visible llama.cpp / ggml acknowledgement and the MIT notice remain part of any parity-backed release claim.

For the formal support ledger, see [`COMPATIBILITY.md`](COMPATIBILITY.md). For sequencing, see [`ROADMAP.md`](ROADMAP.md).

Bottom line for reviewers: Camelid has one supported generation lane, one narrow evidence lane, one blocked acceptance target, and one larger-model groundwork lane. Recent work improved the blocker seam, but it did not earn broader release language.

## What improved without changing the support line

Recent work improved the blocker seam without changing the release ledger:

- TinyLlama Q8_0 remains the trusted supported gate.
- Llama 3.2 1B Q8_0 remains informative evidence only.
- Llama 3.2 3B Q8_0 now loads successfully through `/api/models/load` with low backend RSS after streaming metadata parsing, file-backed lazy-Q8 recovery materially reduced the earlier eager dense-load spike, and fresh Ubuntu backend-only artifacts now cover repeat 5-token generation plus a bounded 50-token run for that same exact row. That is blocker-seam progress, not a support change.
- Llama 3 8B Q8_0 remains groundwork-only in release terms, but the lane now has repeat bounded Ubuntu backend-only 5-token generation, prompt-cache validation, a fresh 50-token backend/API artifact, raw `hello` prompt-token parity, basic API smoke evidence, and an explicit memory gate on top of the earlier lazy/file-backed Q8 execution work.

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

Status: **acceptance target / backend evidence only**

Current evidence boundary:

- The exact tracked GGUF is present locally.
- The exact tracked GGUF loads successfully through `/api/models/load` with low backend RSS after streaming metadata parsing.
- Recent file-backed lazy-Q8 recovery materially reduced the older eager dense-load spike.
- Fresh Ubuntu backend-only `/v1/completions` artifacts now cover repeat 5-token generation for `hello` and one bounded 50-token run.
- This row is still not supported; prompt-token parity, broader prompt/chat-template parity, API evidence, and WebUI evidence are still missing.

Representative artifacts:

- `target/llama32-3b-streaming-metadata-20260430T233604Z/`
- `target/llama32-3b-nocache-rowread-20260430T233844Z/`
- `target/ubuntu-llama32-3b-q8-first-token-20260501T210715Z/`
- `target/ubuntu-crossboard-20260501T234138Z/llama32-3b-q8/`
- `target/ubuntu-50tok-20260501T234603Z/llama32-3b-q8/50tok.json`

Promotion remains blocked until Camelid has prompt-token parity, broader prompt/chat-template parity, API, WebUI, and promotion-quality exact-row evidence for this exact row.

### Llama 3 8B Instruct Q8_0

Status: **groundwork only / backend evidence only**

Current evidence boundary:

- Metadata, config, tokenizer, and chat-template handling are fixture-guarded.
- Independent tokenizer reference fixtures exist.
- Lazy/file-backed Q8 execution is now good enough for repeat bounded Ubuntu backend-only generation on the exact tracked Q8_0 GGUF.
- At current head `268f6fb`, `/api/models/load` succeeded and `/v1/completions` with prompt `hello`, `max_tokens=5`, `temperature=0` returned `, I'm a new`, establishing repeat bounded backend generation before the later current-head follow-up.
- Raw prompt-token parity for `hello` is now captured for this exact model SHA: Camelid returned `[128000, 15339]`, matching the prior llama.cpp `llama-tokenize --ids` reference for the same GGUF SHA.
- Prompt-cache validation and a fresh backend/API 50-token artifact now exist on Ubuntu; see `target/ubuntu-8b-50tok-fast-20260501T235312Z/50tok.json`.
- `/api/models/load`, `/api/models/tokenizer/encode`, `/v1/models`, and `/v1/completions` all responded in the current evidence bundle, with sampled backend RSS staying bounded below 400 MiB and no swap/OOM/runaway signature.
- Camelid still does not claim 8B supported generation, broader prompt-pack parity, chat-template parity, WebUI readiness, performance envelope, or portable packaging for this row.

Representative artifacts:

- `target/backend-small-model-readiness-20260429T131209Z/`
- `target/perf-cron-20260429T122814Z-single-row-adapter-head-da53871/`
- `target/ubuntu-llama3-8b-q8-first-token-20260501T2152Z/`
- `target/ubuntu-llama3-8b-q8-validation-20260501T235936Z/`
- `target/ubuntu-8b-50tok-fast-20260501T235312Z/50tok.json`
- `target/ubuntu-llama3-8b-q8-current-head-20260502T000207Z/`

## Latest promotion-relevant work

### Docs professionalism pass

The top-level documentation set was tightened for executive readability, hierarchy, and release consistency without changing support truth. `README.md`, `COMPATIBILITY.md`, `ROADMAP.md`, and `STATUS.md` remain the public sources of truth. The README now pairs the front-door support ledger with a clearer reading order, while visible llama.cpp / ggml acknowledgement and the MIT notice remain intact wherever reference tooling and parity evidence depend on them. Recon and planning docs continue to carry explicit note banners.

### Llama 3.2 3B lazy-Q8 recovery

Recent backend work kept the support contract unchanged while improving the 3B execution seam:

- streaming metadata parsing moved `/api/models/load` to low backend RSS for the exact 3B artifact
- file-backed Q8 linear handling reduced the older eager dense-load spike
- fresh Ubuntu backend-only artifacts now cover repeat 5-token generation plus a bounded 50-token run for the exact tracked 3B row

This is useful blocker-reduction and backend evidence, not a support promotion.

### Llama 3 8B backend-evidence groundwork

Recent backend work also converted the first bounded 8B runtime artifact into stronger backend-only evidence without widening the release boundary:

- the exact tracked `Meta-Llama-3-8B-Instruct-Q8_0.gguf` loaded successfully on Ubuntu
- repeat bounded backend-only `/v1/completions` 5-token generation succeeded for prompt `hello`
- current-head raw `hello` prompt-token parity matched `[128000, 15339]` for the exact same model SHA
- prompt-cache validation and a fresh 50-token backend/API artifact were captured at `target/ubuntu-8b-50tok-fast-20260501T235312Z/50tok.json`
- the current-head memory gate stayed bounded below 400 MiB RSS with no swap, OOM, timeout, or runaway retained-RSS signature

This is promising backend evidence, but still not a support promotion.

### Concise commit trail

- `268f6fb` — cached prompt follow-up generation bug fixed.
- `462008c` — stronger 8B backend evidence recorded across docs, API compatibility reporting, and frontend-facing support surfaces.

## Next blocking work

In order of importance:

1. Preserve the TinyLlama Q8_0 supported gate.
2. Convert the new Llama 3.2 3B Q8_0 repeat 5-token and bounded 50-token backend wins into prompt-token parity plus API/WebUI evidence for the exact target row.
3. Extend Llama 3 8B Q8_0 from the current backend-only slice into broader prompt-pack/chat-template parity, WebUI readiness, and performance evidence without changing support language early.
4. Keep docs, `/api/capabilities`, and frontend readiness copy aligned with the exact-row support contract.

## Validation note

This file is intentionally a snapshot, not a diary. When a change materially affects support or its blockers:

- add the current evidence summary here
- keep the detailed run log and older slices in `STATUS_ARCHIVE_2026-04.md` or later archives
- update `COMPATIBILITY.md`, `ROADMAP.md`, and user-visible readiness copy in the same change window when support language changes
