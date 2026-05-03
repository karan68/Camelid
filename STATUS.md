# Camelid Status

Last updated: 2026-05-02

`STATUS.md` is Camelid's current release-evidence checkpoint. It records what Camelid can prove today, what moved recently, and what still blocks the next support change. Treat it as a briefing memo, not a diary. Detailed historical run logs, older validation slices, and superseded tactical notes now live in [`STATUS_ARCHIVE_2026-04.md`](STATUS_ARCHIVE_2026-04.md).

Use this file to answer three practical questions: what is supported now, what changed recently, and what still blocks the next support move?

Executive summary: Camelid now has full API + frontend end-to-end smoke for the exact Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B Instruct Q8_0 rows. The previous 3B JSON-shaped broader prompt-pack blocker is resolved, and the 8B long-timeout three-prompt parity rerun passed. The public support boundary moved only for those exact rows and only for the validated local-chat/parity envelopes; broad Llama-family support, longer contexts, and portability remain outside the support claim.

## Release ledger snapshot

Camelid follows the same four-lane release ledger across the README, compatibility matrix, API capability reporting, and frontend readiness copy. If another surface sounds broader, treat it as stale and bring it back to this ledger. The purpose of this file is simple: record exactly what the current evidence can defend, no more and no less.

Reading rule for the matrix: each row should answer three questions in plain English — what is validated now, what gates are still missing, and what exact blocker prevents promotion to the next release label.

For a fast read, the current answer is:

- **Supported generation gates:** TinyLlama 1.1B Chat Q8_0 remains supported, and the exact Llama 3.2 1B/3B plus Llama 3 8B Instruct Q8_0 rows are now smoke-supported for short local chat/parity after exact-row load, completion, chat-completion, frontend smoke, and parity evidence.
- **Scope boundary:** Llama support is exact-row only: model version/size, Instruct variant, Q8_0 quantization, loaded runtime readiness, and the tested smoke/parity envelope all matter.
- **8B promotion:** Llama 3 8B Instruct Q8_0 now has end-to-end generation parity artifacts: compact parity, a long-timeout three-prompt 5-token Ubuntu parity run, API/frontend smoke, and bounded-memory evidence all agree for the exact tracked Q8_0 GGUF.
- **Explicit non-claim:** no broad Llama-family support exists today; neighboring variants remain unsupported unless they have their own exact row and evidence.

Two standing rules apply to every row:

- **Support rule:** Nothing inherits support across model size, quantization, tokenizer lane, API surface, or frontend state.
- **Credit rule:** Visible llama.cpp / ggml acknowledgement and the MIT notice remain part of any parity-backed release claim.

For the formal support ledger, see [`COMPATIBILITY.md`](COMPATIBILITY.md). For sequencing, see [`ROADMAP.md`](ROADMAP.md).

Bottom line for reviewers: Camelid has the original TinyLlama supported gate plus three exact Llama Q8_0 short-chat/parity smoke rows: Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B. That is a real end-to-end support expansion, but it is not a broad Llama-family claim.

## What changed in this support line

Recent work moved the exact-row release ledger in a narrow, evidence-backed way:

- TinyLlama Q8_0 remains the trusted supported gate.
- Llama 3.2 1B Q8_0 moved from evidence-only to supported exact-row smoke after compact parity, broader prompt-pack parity, `/api/models/load`, `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence aligned.
- Llama 3.2 3B Q8_0 moved from acceptance target to supported exact-row smoke after exact-row load, compact prompt-token/1-token/5-token/50-token parity, `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence aligned.
- The 3B broader JSON-shaped prompt divergence is now resolved: a post-Q8-dot rerun of the three-prompt, 50-token pack matched llama.cpp for prompt tokens, generated token IDs, and generated text.
- Llama 3 8B Q8_0 moved from groundwork-only to supported exact-row smoke after the long-timeout Ubuntu three-prompt 5-token parity run, API/frontend smoke, and bounded memory evidence aligned.

Bottom line: the engineering seam and product surface now agree for exact 1B/3B/8B short chat/parity; the support language stays intentionally narrow.

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

Status: **supported exact-row smoke**

Current evidence boundary:

- Compact-header `hello` matches llama.cpp through the completed bounded run on Ubuntu.
- Prompt token IDs, generated token IDs, and generated text all match for the compact bounded response.
- The broader five-prompt parity pack also passed for this exact 1B row.
- `/api/models/load`, `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence are aligned with `/api/capabilities`.
- The support claim is limited to this exact 1B Instruct Q8_0 row and short local-chat smoke; neighboring Llama rows, other quantizations, longer contexts, and broader chat-template behavior remain outside the claim.

Representative artifacts:

- `target/autonomous-small-model-parity-20260429T134615Z-head-9049492/llama32-1b-q8-chat-parity-5tok.json`
- `target/qa-small-model-parity-20260429T1338Z-head-35bfd58/`
- `target/parity-50tok-20260502T031820Z/llama32-1b-50tok/report.json`
- `target/qa-discord-20260502T1832Z/PROMOTION_QA_SUMMARY.md`
- `target/qa-discord-20260502T1832Z/llama32-1b-broad-canonical/summary.json`
- `target/qa-discord-20260502T1832Z/llama32-1b-api-smoke/summary.json`
- `target/qa-discord-20260502T1832Z/llama32-1b-webui-smoke/summary-node22b.txt`

### Llama 3.2 3B Instruct Q8_0

Status: **supported exact-row smoke**

Current evidence boundary:

- The exact tracked GGUF is present locally.
- The exact tracked GGUF loads successfully through `/api/models/load` with low backend RSS after streaming metadata parsing.
- Recent file-backed lazy-Q8 recovery materially reduced the older eager dense-load spike.
- The Ubuntu compact-header `hello` harness matches llama.cpp for prompt tokens plus deterministic 1-token, 5-token, and bounded 50-token generation.
- `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence are aligned with `/api/capabilities` for this exact row.
- The support claim is limited to this exact 3B Instruct Q8_0 row and the validated local-chat/parity envelope; longer contexts, stronger memory/performance evidence, portable packaging, and broader chat-template coverage remain follow-up gates.

Representative artifacts:

- `target/llama32-3b-streaming-metadata-20260430T233604Z/summary.md`
- `target/llama32-3b-nocache-rowread-20260430T233844Z/summary.md`
- `target/ubuntu-llama32-3b-q8-first-token-20260501T210715Z/summary.md`
- `target/ubuntu-llama32-3b-q8-first-token-20260501T210715Z/load-response.json`
- `target/ubuntu-llama32-3b-q8-first-token-20260501T210715Z/completion-response.json`
- `target/ubuntu-llama32-3b-q8-first-token-20260501T210715Z/required-forward-trace.log`
- `target/ubuntu-llama32-3b-q8-first-token-20260501T210715Z/meminfo-samples.log`
- `target/parity-20260502T030911Z/llama32-3b-1tok/report.json`
- `target/parity-20260502T030911Z/llama32-3b-5tok/report.json`
- `target/parity-50tok-20260502T031820Z/llama32-3b-50tok/report.json`

Expansion beyond the current supported row remains blocked until Camelid has longer-context coverage plus stronger memory/performance, portability, and broader WebUI/chat-template evidence for this exact row.

Latest broader-prompt result:

- `target/camelid-regression-q8dot-20260502T232633Z/llama32-3b-compact/summary.json` passed the compact three-prompt 5-token pack after the file-backed Q8_0 dot-parity fix.
- `target/camelid-llama32-3b-broad-50-after-q8dot-clean-20260502T233427Z/pack/summary.json` passed the broader three-prompt 50-token pack, including the former JSON-shaped blocker prompt `answer with valid JSON for {"ok":true,"value":2}`.
- For that broader pack, prompt tokens, generated token IDs, and generated text all matched llama.cpp; every prompt reports `first_generated_token_diff_index=-1` and `first_generated_text_diff_index=-1`.

### Llama 3 8B Instruct Q8_0

Status: **supported exact-row smoke**

Current evidence boundary:

- Metadata, config, tokenizer, and chat-template handling are fixture-guarded.
- Independent tokenizer reference fixtures exist.
- Lazy/file-backed Q8 execution is now good enough for repeat bounded parity on the exact tracked Q8_0 GGUF.
- The Ubuntu compact-header `hello` harness matches llama.cpp for prompt tokens and deterministic generation at 1, 5, and bounded 50-token lengths on this exact row.
- The long-timeout Ubuntu three-prompt 5-token parity run passed for `hello`, alpacas, and JSON with prompt tokens, generated token IDs, and generated text all matching llama.cpp.
- `/api/models/load`, `/v1/completions`, `/v1/chat/completions`, and frontend smoke passed for this exact row.
- The support claim is limited to this exact Llama 3 8B Instruct Q8_0 row and tested short smoke/parity envelope; neighboring Llama rows, other quantizations, longer contexts, and broader chat-template behavior remain outside the claim.

Representative artifacts:

- `target/backend-small-model-readiness-20260429T131209Z/report.json`
- `target/perf-cron-20260429T122814Z-single-row-adapter-head-da53871/summary.md`
- `target/ubuntu-llama3-8b-q8-first-token-20260501T2152Z/summary.md`
- `target/ubuntu-llama3-8b-q8-validation-20260501T235936Z/targeted-lazy-q8-tests.log`
- `target/ubuntu-llama3-8b-q8-current-head-20260502T000207Z/validation-summary.json`
- `target/ubuntu-llama3-8b-q8-current-head-20260502T000207Z/first-token.completion-summary.json`
- `target/ubuntu-llama3-8b-q8-current-head-20260502T000207Z/first-token.meminfo-samples.log`
- `target/acceptance-llama3-8b-broader-5tok-longtimeout-20260503T010536Z/summary.json`
- `target/private-four-llama-e2e-20260502T212751Z-head-c5e6d7e/matrix-summary.json`
- `target/ubuntu-llama3-8b-q8-current-head-20260502T000207Z/first-token.required-forward-trace.log`
- `target/parity-20260502T030911Z/llama3-8b-1tok/report.json`
- `target/parity-20260502T030911Z/llama3-8b-5tok/report.json`
- `target/parity-50tok-20260502T031820Z/llama3-8b-50tok/report.json`
- `target/ubuntu-llama3-8b-q8-current-head-20260502T000207Z/short-5tok.completion-summary.json`
- `target/ubuntu-llama3-8b-q8-current-head-20260502T000207Z/short-5tok.meminfo-samples.log`
- `target/ubuntu-llama3-8b-q8-current-head-20260502T000207Z/short-5tok.required-forward-trace.log`

## Latest promotion-relevant work

### Docs professionalism pass

The top-level documentation set was tightened for executive readability, hierarchy, and release consistency without changing support truth. `README.md`, `COMPATIBILITY.md`, `ROADMAP.md`, and `STATUS.md` remain the public sources of truth. The README now pairs the front-door support ledger with a clearer reading order, while visible llama.cpp / ggml acknowledgement and the MIT notice remain intact wherever reference tooling and parity evidence depend on them. Recon and planning docs continue to carry explicit note banners.


### Full frontend/API end-to-end smoke

Fresh end-to-end validation artifact: `target/e2e-docs-20260502T2130Z-r3/`.

- Llama 3.2 1B Instruct Q8_0 loaded as the exact supported compatibility row, reported `generation_ready=true`, unlocked WebUI chat as `contract_supported=true`, and returned `"Hello"` from `/v1/chat/completions` in 8.49s.
- Llama 3.2 3B Instruct Q8_0 loaded as the exact supported compatibility row, reported `generation_ready=true`, unlocked WebUI chat as `contract_supported=true`, and returned `"Hello"` from `/v1/chat/completions` in 24.24s.
- Llama 3 8B Instruct Q8_0 loaded and generated through the same frontend/API smoke path; after the later parity promotion it is now an exact supported compatibility row. The smoke returned `"Hello"` in 55.51s.

### Llama 3.2 3B exact-row smoke promotion

Recent backend and frontend work aligned the 3B execution seam with the user-visible support contract:

- streaming metadata parsing moved `/api/models/load` to low backend RSS for the exact 3B artifact
- file-backed Q8 linear handling reduced the older eager dense-load spike
- compact prompt-token, 1-token, 5-token, and bounded 50-token parity passed for the exact tracked 3B row
- `/v1/completions`, `/v1/chat/completions`, and frontend smoke now pass under the exact supported compatibility row

This is a support promotion only for the exact 3B Instruct Q8_0 short-chat smoke row.

### Llama 3 8B exact-row smoke promotion

Recent backend work converted the 8B runtime artifacts into an exact-row support promotion:

- the exact tracked `Meta-Llama-3-8B-Instruct-Q8_0.gguf` loaded successfully on Ubuntu
- repeat bounded backend-only `/v1/completions` first-token probes returned `,` for prompt `hello`
- current-head raw `hello` prompt-token parity matched `[128000, 15339]` for the exact same model SHA
- a short deterministic 5-token backend slice returned `, I'm a new`
- the long-timeout Ubuntu three-prompt 5-token parity run passed for `hello`, alpacas, and JSON: `target/acceptance-llama3-8b-broader-5tok-longtimeout-20260503T010536Z/summary.json`
- `/v1/completions`, `/v1/chat/completions`, and frontend smoke passed in `target/private-four-llama-e2e-20260502T212751Z-head-c5e6d7e/matrix-summary.json`
- the current-head memory gate stayed bounded: first-token sampled RSS roughly `6,220 -> 378,520 KiB`; 5-token sampled RSS roughly `6,076 -> 396,912 KiB`; no swap, OOM, timeout, or runaway retained-RSS signature appeared

This is a support promotion only for the exact Llama 3 8B Instruct Q8_0 row and tested short smoke/parity envelope.

## Latest downloaded Llama-family matrix

Latest Ubuntu downloaded-model matrix: `target/downloaded-llama-matrix-20260502T231000Z/summary.json`.

Downloaded GGUF rows covered by this sweep:

- `tinyllama-1.1b-chat-v1.0.Q8_0.gguf`
- `Llama-3.2-1B-Instruct-Q8_0.gguf`
- `Llama-3.2-3B-Instruct-Q8_0.gguf`
- `Meta-Llama-3-8B-Instruct-Q8_0.gguf`

Results:

- **TinyLlama 1.1B Chat Q8_0:** `hello` and the alpacas prompt matched llama.cpp; the JSON-shaped prompt diverged despite matching prompt tokens (`endpoint` vs `function` wording in the generated text).
- **Llama 3.2 1B Instruct Q8_0:** the three-prompt Llama 3 pack passed completely; prompt tokens, generated token IDs, and generated text all matched llama.cpp.
- **Llama 3.2 3B Instruct Q8_0:** the earlier downloaded matrix captured the now-fixed JSON-shaped prompt divergence; the post-Q8-dot clean rerun at `target/camelid-llama32-3b-broad-50-after-q8dot-clean-20260502T233427Z/pack/summary.json` supersedes that 3B result and passes all three prompts for prompt tokens, generated token IDs, and generated text.
- **Llama 3 8B Instruct Q8_0:** the long-timeout rerun at `target/acceptance-llama3-8b-broader-5tok-longtimeout-20260503T010536Z/summary.json` passed `hello`, alpacas, and JSON for prompt-token, generated-token, and generated-text parity at 5 tokens, clearing the earlier client-timeout blocker for the exact 8B row.

The downloaded-model matrix still disproves a broad inherited “perfect Llama-family parity” claim. Camelid should claim only the exact supported rows and envelopes backed by passing artifacts: TinyLlama Q8_0, Llama 3.2 1B Q8_0, Llama 3.2 3B Q8_0, and Llama 3 8B Q8_0.

## Next blocking work

In order of importance:

1. Preserve the TinyLlama Q8_0 supported gate and the exact Llama 3.2 1B/3B short-chat smoke gates.
2. Preserve and publish the Llama 3.2 1B broader prompt-pack win as exact-row evidence, without lending it to neighboring rows.
3. Preserve the Llama 3.2 3B broader prompt-pack win in docs, API, and regression evidence without lending it to neighboring rows.
4. Preserve the Llama 3 8B exact-row promotion in docs, API, frontend readiness, and regression evidence without lending it to neighboring rows.
5. Keep docs, `/api/capabilities`, and frontend readiness copy aligned with the exact-row support contract.

### Qwen prerequisite note

Qwen remains future work, not a runtime-support lane. Before Camelid promotes any Qwen wording beyond planning, the first exact-row prerequisite is one chosen GGUF with tokenizer/chat-template fixtures, llama.cpp token-reference checks, and bounded load plus prompt-token parity evidence on that same row. Until those artifacts exist, Qwen should stay out of support/readiness language and out of runtime-promotion scheduling.

## Validation note

This file is intentionally a snapshot, not a diary. When a change materially affects support or its blockers:

- add the current evidence summary here
- keep the detailed run log and older slices in `STATUS_ARCHIVE_2026-04.md` or later archives
- update `COMPATIBILITY.md`, `ROADMAP.md`, and user-visible readiness copy in the same change window when support language changes
