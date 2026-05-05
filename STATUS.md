# Camelid Status

Last updated: 2026-05-05

`STATUS.md` is Camelid's current release-evidence checkpoint. It records what Camelid can prove today, what moved recently, and what still blocks the next support change. Treat it as a briefing memo, not a diary. Detailed historical run logs, older validation slices, and superseded tactical notes now live in [`STATUS_ARCHIVE_2026-04.md`](STATUS_ARCHIVE_2026-04.md).

Use this file to answer three practical questions: what is supported now, what changed recently, and what still blocks the next support move?

Executive summary: Camelid now has full API + frontend end-to-end smoke for the exact Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B Instruct Q8_0 rows, refreshed on the reopened Ubuntu validation lane. The previous 3B JSON-shaped broader prompt-pack blocker is resolved, the 8B broader three-prompt 50-token rerun passed, and the first bounded 8B 512-context plus compact chat-template-shapes packs now have passing public summaries. The public support boundary moved only for those exact rows and only for the validated local-chat/parity envelopes; broad Llama-family support, larger contexts, arbitrary template execution, and portability remain outside the support claim.

## Release ledger snapshot

Camelid follows the same four-lane release ledger across the README, compatibility matrix, API capability reporting, and frontend readiness copy. If another surface sounds broader, treat it as stale and bring it back to this ledger. The purpose of this file is simple: record exactly what the current evidence can defend, no more and no less.

Reading rule for the matrix: each row should answer three questions in plain English — what is validated now, what gates are still missing, and what exact blocker prevents promotion to the next release label.

For a fast read, the current answer is:

- **Supported generation gates:** TinyLlama 1.1B Chat Q8_0 remains supported, and the exact Llama 3.2 1B/3B plus Llama 3 8B Instruct Q8_0 rows are now smoke-supported for short local chat/parity after exact-row load, completion, chat-completion, frontend smoke, and parity evidence.
- **Scope boundary:** Llama support is exact-row only: model version/size, Instruct variant, Q8_0 quantization, loaded runtime readiness, and the tested smoke/parity envelope all matter.
- **8B promotion:** Llama 3 8B Instruct Q8_0 now has end-to-end generation parity artifacts: compact parity, a three-prompt 50-token Ubuntu parity run, the bounded compact chat-template-shapes pack, API/frontend smoke, and bounded-memory evidence all agree for the exact tracked Q8_0 GGUF.
- **Explicit non-claim:** no broad Llama-family support exists today; neighboring variants remain unsupported unless they have their own exact row and evidence.

Two standing rules apply to every row:

- **Support rule:** Nothing inherits support across model size, quantization, tokenizer lane, API surface, or frontend state.
- **Credit rule:** Visible llama.cpp / ggml acknowledgement and the MIT notice remain part of any parity-backed release claim.

For the formal support ledger, see [`COMPATIBILITY.md`](COMPATIBILITY.md). For sequencing, see [`ROADMAP.md`](ROADMAP.md).

Bottom line for reviewers: Camelid has the original TinyLlama supported gate plus three exact Llama Q8_0 short-chat/parity smoke rows: Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B. That is a real end-to-end support expansion, but it is not a broad Llama-family claim.

## Durable evidence anchors

- `qa/evidence-bundles/four-row-public-20260503T024327Z/manifest.json` plus `qa/evidence-bundles/four-row-public-20260503T024327Z/SHA256SUMS` are the committed carry-forward row bundles/checksums for the public smoke boundary.
- `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json` is the committed Ubuntu perf/portability summary for the current four-row sweep.
- `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/manifest.json` plus its per-row manifests/checksums are the durable current-head citation target for exact rerun tracks, blocker notes, and command files.
- `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json` is the sanitized API + frontend smoke summary for all four exact rows on a clean public checkout; `qa/evidence-bundles/four-row-api-only-20260504T230722Z-head-13a465608fbf/manifest.json` is the narrower API-only predecessor.
- `qa/evidence-bundles/full-support-normalized-wp1-20260505T032406Z-head-bcf9e647d6fd/manifest.json` is the current-head normalized TinyLlama/1B/3B API/WebUI smoke bundle from the reopened Ubuntu lane; all three rows passed and the public bundle checksum file verifies with `SHA256SUMS` sha256 `ce87c02fba64fcd78efe10c01b030435d185bc785f06a4d9df4cbd04048da283`.
- `qa/evidence-bundles/full-support-normalized-wp2-8b-watchdog-20260505T041404Z-head-83c21f0cbf5a/manifest.json` is the current-public-head normalized Llama 3 8B Instruct Q8_0 API/WebUI/RSS smoke bundle from the reopened Ubuntu lane; the exact row passed load/models/capabilities/completions/chat/timing-summary/frontend smoke, frontend chat generated `Hello`, max sampled backend RSS was `283372 KiB`, and `SHA256SUMS` sha256 is `83334a9083806081569322978db273044753515a195359d0b4326cf6352367da`.
- `qa/evidence-bundles/8b-checkmark-current-head-20260505T052647Z-head-864e07b51f36/manifest.json` is the latest public-main Llama 3 8B Instruct Q8_0 checkmark refresh at head `864e07b51f36`; API/WebUI/RSS smoke passed load/models/capabilities/completions/chat/timing-summary/frontend smoke, frontend chat generated `Hello`, max sampled backend RSS was `286056 KiB`, and `SHA256SUMS` sha256 is `0774c12816651c6f330f072141ec2de83c958bb857cb771df57716755724b2cf`. This preserves exact-row smoke support only.
- `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json` records the reopened-lane pass for the first bounded 8B 512-context pack.
- `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json` records the reopened-lane pass for the bounded 8B broader three-prompt 50-token pack.
- `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/manifest.json` records the reopened-lane pass for the bounded 8B compact chat-template-shapes pack.
- `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/manifest.json` records the clean-main reopened-lane exact 8B API/WebUI smoke pass for the completion-diagnostics API path: `/v1/completions` carries response-local timing diagnostics, the promotion smoke timing summary passed, frontend chat generated `Hello`, and RSS was sampled around the smoke window. This is an exact-row API evidence slice, not a new broad support claim. The earlier patched-tree predecessor remains at `qa/evidence-bundles/llama3-8b-api-webui-rss-20260505T014408Z-head-8cef7af4d6c6/manifest.json` for audit history.
- `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/manifest.json` records exact 8B retained-block lazy-Q8 hot-path cost probes for representative FFN tensors and `output.weight`, including the guarded swapped logical row/column interpretation. This is measurement evidence only; it does not promote broad 8B/full-context/arbitrary-template support or performance portability.
- `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-helper-validated-20260505T0350Z-head-e22307f2f90b/manifest.json` validates the reusable hot-path bundle helper on clean public `main` at `e22307f2f90b` and repeats the exact-row retained-block measurements (`35.78 ms`, `35.73 ms`, `320.24 ms` for the checked tensors). This is helper/measurement evidence only.
- Raw `target/` paths below are drill-down artifacts only; they are not the sole public evidence anchor.

## What changed in this support line

Recent work moved the exact-row release ledger in a narrow, evidence-backed way:

- TinyLlama Q8_0 remains the trusted supported gate.
- Llama 3.2 1B Q8_0 moved from evidence-only to supported exact-row smoke after compact parity, broader prompt-pack parity, `/api/models/load`, `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence aligned.
- Llama 3.2 3B Q8_0 moved from acceptance target to supported exact-row smoke after exact-row load, compact prompt-token/1-token/5-token/50-token parity, `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence aligned.
- The 3B broader JSON-shaped prompt divergence is now resolved: a post-Q8-dot rerun of the three-prompt, 50-token pack matched llama.cpp for prompt tokens, generated token IDs, and generated text.
- Llama 3 8B Q8_0 moved from groundwork-only to supported exact-row smoke after Ubuntu three-prompt parity, API/frontend smoke, and bounded memory evidence aligned; the current public broader-pack rerun is the bounded three-prompt 50-token pass.

Bottom line: the engineering seam and product surface now agree for exact 1B/3B/8B short chat/parity; the support language stays intentionally narrow.

## Repo-health verification pass

A fresh local repo-health pass ran on 2026-05-04 to keep the public tree and CI contract honest before heavier model work resumes.

Verified locally on the current tree:

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --all-targets --all-features`
- `cargo doc --no-deps --all-features`
- `bash scripts/check-public-scrub.sh`
- `cd frontend && npm ci && npm run build && npm run smoke:model-state`

Result: all of the above passed locally. The CI workflow was also tightened so the Rust job now enforces clippy and docs generation in addition to format and tests, and the frontend job now runs the support-contract/model-state smoke gate that protects exact-row chat unlock behavior. This keeps the GitHub gate aligned with the documented validation contract.

## Current support evidence

The sections below summarize the artifact-backed boundary for each tracked row. They are intentionally narrower than "what might be close." If a supporting artifact is not called out here or in the linked files, Camelid should not imply the claim elsewhere.

### TinyLlama 1.1B Chat Q8_0

Status: **supported current gate**

Current evidence boundary:

- Five-prompt, 50-token parity audit against known-good llama-server.
- Prompt token IDs, generated token arrays, and generated text match.
- The token-major `output.weight` interpretation remains a protected correctness guardrail.

Representative durable evidence:

- `qa/evidence-bundles/four-row-public-20260503T024327Z/tinyllama_1_1b_chat_q8_0.bundle.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/manifest.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/SHA256SUMS`
- `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/tinyllama_1_1b_chat_q8_0/manifest.json`

The older five 50-token source JSONs remain listed under that current-head row manifest's `broader-parity` carry-forward track instead of standing alone as the release citation.

### Llama 3.2 1B Instruct Q8_0

Status: **supported exact-row smoke**

Current evidence boundary:

- Compact-header `hello` matches llama.cpp through the completed bounded run on Ubuntu.
- Prompt token IDs, generated token IDs, and generated text all match for the compact bounded response.
- The broader five-prompt parity pack also passed for this exact 1B row.
- `/api/models/load`, `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence are aligned with `/api/capabilities`.
- The support claim is limited to this exact 1B Instruct Q8_0 row and short local-chat smoke; neighboring Llama rows, other quantizations, longer contexts, and broader chat-template behavior remain outside the claim.

Representative durable evidence:

- `qa/evidence-bundles/four-row-public-20260503T024327Z/llama32_1b_instruct_q8_0.bundle.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/manifest.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/SHA256SUMS`
- `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json`
- `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/llama32_1b_instruct_q8_0/manifest.json`

### Llama 3.2 3B Instruct Q8_0

Status: **supported exact-row smoke**

Current evidence boundary:

- The exact tracked GGUF is present locally.
- The exact tracked GGUF loads successfully through `/api/models/load` with low backend RSS after streaming metadata parsing.
- Recent file-backed lazy-Q8 recovery materially reduced the older eager dense-load spike.
- The Ubuntu compact-header `hello` harness matches llama.cpp for prompt tokens plus deterministic 1-token, 5-token, and bounded 50-token generation.
- `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence are aligned with `/api/capabilities` for this exact row.
- The support claim is limited to this exact 3B Instruct Q8_0 row and the validated local-chat/parity envelope; longer contexts, stronger memory/performance evidence, portable packaging, and broader chat-template coverage remain follow-up gates.

Representative durable evidence:

- `qa/evidence-bundles/four-row-public-20260503T024327Z/llama32_3b_instruct_q8_0.bundle.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/manifest.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/SHA256SUMS`
- `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json`
- `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/llama32_3b_instruct_q8_0/manifest.json`

Selected source artifacts recorded by those committed files:

- `target/ubuntu-followup-20260502T015231Z/llama32_3b-50tok.json` preserves compact bounded parity inside the carry-forward bundle.
- `target/camelid-llama32-3b-broad-50-after-q8dot-clean-20260502T233427Z/pack/summary.json` is the post-Q8-dot broader three-prompt clean rerun called out by the current-head row manifest notes.

Expansion beyond the current supported row remains blocked until Camelid has longer-context coverage plus stronger memory/performance, portability, and broader WebUI/chat-template evidence for this exact row.

### Llama 3 8B Instruct Q8_0

Status: **supported exact-row smoke**

Current evidence boundary:

- Metadata, config, tokenizer, and chat-template handling are fixture-guarded.
- Independent tokenizer reference fixtures exist.
- Lazy/file-backed Q8 execution is now good enough for repeat bounded parity on the exact tracked Q8_0 GGUF.
- The Ubuntu compact-header `hello` harness matches llama.cpp for prompt tokens and deterministic generation at 1, 5, and bounded 50-token lengths on this exact row.
- The Ubuntu three-prompt 50-token parity run passed for `hello`, alpacas, and JSON with prompt tokens, generated token IDs, and generated text all matching the known-good reference.
- The first bounded 512-context pack now passes on the reopened Ubuntu lane: prompt tokens, generated token IDs, and generated text matched for `qa/prompt-packs/llama3-context-512-smoke.json` at `58acf592345c69c1b684544124cd23804e2899f1`.
- The bounded compact chat-template-shapes pack now passes on the reopened Ubuntu lane: all 4 prompts in `qa/prompt-packs/llama3-chat-template-shapes.json` matched prompt tokens, generated token IDs, and generated text at `d13541ad8d7e87426cddd0d0a13e292f39c73f31`.
- `/api/models/load`, `/v1/completions`, `/v1/chat/completions`, and frontend smoke passed for this exact row.
- A clean-main reopened-lane API/WebUI/RSS smoke at `aee469b` validated that `/v1/completions` exposes `backendinference.timings_ms` like chat completions, allowing the promotion smoke bundle to summarize response-local timings for both endpoints. The exact 8B smoke passed all API/frontend steps and generated `Hello` in the frontend smoke; this is recorded in `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/manifest.json`.
- The current-public-head watchdog refresh at `83c21f0cbf5a` repeats the normalized 8B API/WebUI/RSS smoke after the CI evidence-checksum gate landed; it is published at `qa/evidence-bundles/full-support-normalized-wp2-8b-watchdog-20260505T041404Z-head-83c21f0cbf5a/manifest.json` and remains exact-row smoke evidence only.
- The latest public-main checkmark refresh at `864e07b51f36` repeats the exact 8B API/WebUI/RSS smoke on current `origin/main`; it is published at `qa/evidence-bundles/8b-checkmark-current-head-20260505T052647Z-head-864e07b51f36/manifest.json` and remains exact-row smoke evidence only.
- A retained-block lazy-Q8 hot-path probe at `723a665` measured representative exact-row 8B costs without widening support: logical `[14336,4096]` FFN dots were about 36.7 ms each in serial microbench mode, and swapped logical `output.weight` `[128256,4096]` was about 328 ms while avoiding about 2.0 GiB f32 materialization. This narrows optimization targets; it is not production throughput, portability, or full-support evidence.
- The support claim is limited to this exact Llama 3 8B Instruct Q8_0 row and tested smoke/parity envelope; neighboring Llama rows, other quantizations, larger contexts, and broader chat-template behavior remain outside the claim.

Representative durable evidence:

- `qa/evidence-bundles/four-row-public-20260503T024327Z/llama3_8b_instruct_q8_0.bundle.json` (the committed pre-promotion guarded-WebUI carry-forward slice)
- `qa/evidence-bundles/four-row-public-20260503T024327Z/manifest.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/SHA256SUMS`
- `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json`
- `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/llama3_8b_instruct_q8_0/manifest.json`
- `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json`
- `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json`
- `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/manifest.json`
- `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/manifest.json`
- `qa/evidence-bundles/llama3-8b-api-webui-rss-20260505T014408Z-head-8cef7af4d6c6/manifest.json` (historical patched-tree predecessor)
- `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/manifest.json`
- `qa/validation-notes/2026-05-05-lazy-q8-hotpath-costs.md`
- `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-helper-validated-20260505T0350Z-head-e22307f2f90b/manifest.json`
- `qa/validation-notes/2026-05-05-lazy-q8-hotpath-helper-validation.md`
- `qa/validation-notes/2026-05-05-8b-clean-main-api-webui-rss.md`
- `qa/validation-notes/2026-05-05-8b-broader-50tok.md`
- `qa/validation-notes/2026-05-04-8b-context-512-rerun.md`
- `qa/validation-notes/2026-05-05-8b-chat-template-shapes.md`
- `qa/validation-notes/2026-05-05-8b-api-webui-rss.md`
- `qa/validation-notes/2026-05-03-ubuntu-toolchain-and-8b-context.md`

Selected source artifacts recorded by those committed files:

- `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json` is the public sanitized broader three-prompt 50-token parity pass for the exact 8B row; the older `target/acceptance-llama3-8b-broader-5tok-longtimeout-20260503T010536Z/summary.json` remains historical carry-forward evidence only.
- `target/ubuntu-llama3-8b-q8-current-head-20260502T000207Z/validation-summary.json` is the bounded-RSS short-slice summary carried forward beside the current-head blocker note.

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
- the Ubuntu three-prompt 50-token parity run passed for `hello`, alpacas, and JSON: `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json`
- the reopened-lane first 512-context pack passed with prompt-token, generated-token, and generated-text parity; public summary/checksums live at `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/`
- the reopened-lane compact chat-template-shapes pack passed all 4 checked shapes with prompt-token, generated-token, and generated-text parity; public summary/checksums live at `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/`
- `/v1/completions`, `/v1/chat/completions`, and frontend smoke are preserved in the sanitized carry-forward bundle at `qa/evidence-bundles/four-row-public-20260503T024327Z/llama3_8b_instruct_q8_0.bundle.json`; the newer reopened-lane API + frontend smoke summary at `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json` refreshes exact-row WebUI/API readiness for the four tracked rows, but broader/full-support evidence is still outstanding
- clean-main API timing diagnostics for `/v1/completions` passed the exact 8B API/WebUI smoke and response-local timing summary at `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/manifest.json`; sampled backend RSS for that smoke moved `6,316 -> 283,352 KiB`, which is useful smoke-window evidence but not peak memory proof
- retained-block lazy-Q8 hot-path measurement at `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/manifest.json` shows FFN logical `[14336,4096]` serial all-row dots around 36.7 ms and swapped `output.weight` logical `[128256,4096]` around 328 ms; this is optimization-grounding evidence only, not a wider support/performance claim
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
- **Llama 3 8B Instruct Q8_0:** the reopened-lane rerun at `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json` passed `hello`, alpacas, and JSON for prompt-token, generated-token, and generated-text parity at 50 tokens, clearing the earlier client-timeout blocker for the exact 8B row.

The downloaded-model matrix still disproves a broad inherited “perfect Llama-family parity” claim. Camelid should claim only the exact supported rows and envelopes backed by passing artifacts: TinyLlama Q8_0, Llama 3.2 1B Q8_0, Llama 3.2 3B Q8_0, and Llama 3 8B Q8_0.

Public evidence packaging note: sanitized carry-forward bundle manifests/checksums for the four-row smoke slices now live under `qa/evidence-bundles/four-row-public-20260503T024327Z/` and `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/`. They intentionally preserve the blocked public-head rerun state instead of overstating it.

Current-head durable execution note: the exact-row normalized rerun scaffold is now checked in at `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/`. Its per-row manifests/commands give docs, API, and frontend a stable current-head citation target while broader context coverage and stronger perf/portability evidence are still outstanding. The exact 8B broader three-prompt 50-token rerun is separately captured at `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/`, the first exact 8B 512-context rerun is separately captured at `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/`, the bounded exact 8B compact chat-template-shapes rerun is separately captured at `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/`, the historical patched-tree exact 8B API/WebUI/RSS timing smoke is separately captured at `qa/evidence-bundles/llama3-8b-api-webui-rss-20260505T014408Z-head-8cef7af4d6c6/`, the clean-main exact 8B API/WebUI/RSS timing rerun is captured at `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/manifest.json`, the latest public-main exact 8B checkmark refresh is captured at `qa/evidence-bundles/8b-checkmark-current-head-20260505T052647Z-head-864e07b51f36/manifest.json`, the measurement-only retained-block lazy-Q8 hot-path probe is captured at `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/manifest.json`, and the clean-main helper-validation repeat is captured at `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-helper-validated-20260505T0350Z-head-e22307f2f90b/manifest.json`.

## Next blocking work

In order of importance:

1. Preserve the TinyLlama Q8_0 supported gate and the exact Llama 3.2 1B/3B short-chat smoke gates.
2. Preserve and publish the Llama 3.2 1B broader prompt-pack win as exact-row evidence, without lending it to neighboring rows.
3. Preserve the Llama 3.2 3B broader prompt-pack win in docs, API, and regression evidence without lending it to neighboring rows.
4. Preserve the Llama 3 8B exact-row promotion in docs, API, frontend readiness, and regression evidence without lending it to neighboring rows.
5. Keep docs, `/api/capabilities`, and frontend readiness copy aligned with the exact-row support contract.

Current operator update: Tim has reopened the approved Ubuntu validation lane. Promotion-grade Llama-family reruns for 1B/3B/8B should resume there from clean public `main` checkouts while preserving any dirty remote worktrees. Fresh reopened-lane API-only and API + frontend smoke summaries now live at `qa/evidence-bundles/four-row-api-only-20260504T230722Z-head-13a465608fbf/manifest.json` and `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`; the exact 8B broader three-prompt 50-token pack has a passing public summary at `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json`; the first exact 8B 512-context pack has a passing public summary at `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json`; and the bounded exact 8B compact chat-template-shapes pack has a passing public summary at `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/manifest.json`. This does not widen support by itself: broader/full support still requires normalized parity, memory/perf, broader context/template coverage, and durable-bundle evidence. See `qa/validation-notes/2026-05-04-validation-lane-reopened.md`, `qa/validation-notes/2026-05-04-8b-context-512-rerun.md`, `qa/validation-notes/2026-05-05-8b-broader-50tok.md`, and `qa/validation-notes/2026-05-05-8b-chat-template-shapes.md`.

### Qwen prerequisite note

Qwen remains future work, not a runtime-support lane. Before Camelid promotes any Qwen wording beyond planning, the first exact-row prerequisite is one chosen GGUF with tokenizer/chat-template fixtures, llama.cpp token-reference checks, and bounded load plus prompt-token parity evidence on that same row. Until those artifacts exist, Qwen should stay out of support/readiness language and out of runtime-promotion scheduling.

## Validation note

This file is intentionally a snapshot, not a diary. When a change materially affects support or its blockers:

- add the current evidence summary here
- keep the detailed run log and older slices in `STATUS_ARCHIVE_2026-04.md` or later archives
- update `COMPATIBILITY.md`, `ROADMAP.md`, and user-visible readiness copy in the same change window when support language changes
