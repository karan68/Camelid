# Camelid Status

Last updated: 2026-05-04

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

## Durable evidence anchors

- `qa/evidence-bundles/four-row-public-20260503T024327Z/manifest.json` plus `qa/evidence-bundles/four-row-public-20260503T024327Z/SHA256SUMS` are the committed carry-forward row bundles/checksums for the public smoke boundary.
- `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json` is the committed Ubuntu perf/portability summary for the current four-row sweep.
- `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/manifest.json` plus its per-row manifests/checksums are the durable current-head citation target for exact rerun tracks, blocker notes, and command files.
- Raw `target/` paths below are drill-down artifacts only; they are not the sole public evidence anchor.

## What changed in this support line

Recent work moved the exact-row release ledger in a narrow, evidence-backed way:

- TinyLlama Q8_0 remains the trusted supported gate.
- Llama 3.2 1B Q8_0 moved from evidence-only to supported exact-row smoke after compact parity, broader prompt-pack parity, `/api/models/load`, `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence aligned.
- Llama 3.2 3B Q8_0 moved from acceptance target to supported exact-row smoke after exact-row load, compact prompt-token/1-token/5-token/50-token parity, `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence aligned.
- The 3B broader JSON-shaped prompt divergence is now resolved: a post-Q8-dot rerun of the three-prompt, 50-token pack matched llama.cpp for prompt tokens, generated token IDs, and generated text.
- Llama 3 8B Q8_0 moved from groundwork-only to supported exact-row smoke after the long-timeout Ubuntu three-prompt 5-token parity run, API/frontend smoke, and bounded memory evidence aligned.

Bottom line: the engineering seam and product surface now agree for exact 1B/3B/8B short chat/parity; the support language stays intentionally narrow.

## Repo-health verification pass

A fresh local repo-health pass ran on 2026-05-04 to keep the public tree and CI contract honest before heavier model work resumes.

Verified locally on the current tree:

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --all-targets --all-features`
- `cargo doc --no-deps --all-features`
- `bash scripts/check-public-scrub.sh`
- `cd frontend && npm ci && npm run build`

Result: all of the above passed locally. The CI workflow was also tightened so the Rust job now enforces clippy and docs generation in addition to format and tests, keeping the GitHub gate aligned with the documented validation contract.

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
- The long-timeout Ubuntu three-prompt 5-token parity run passed for `hello`, alpacas, and JSON with prompt tokens, generated token IDs, and generated text all matching llama.cpp.
- `/api/models/load`, `/v1/completions`, `/v1/chat/completions`, and frontend smoke passed for this exact row.
- The support claim is limited to this exact Llama 3 8B Instruct Q8_0 row and tested short smoke/parity envelope; neighboring Llama rows, other quantizations, longer contexts, and broader chat-template behavior remain outside the claim.

Representative durable evidence:

- `qa/evidence-bundles/four-row-public-20260503T024327Z/llama3_8b_instruct_q8_0.bundle.json` (the committed pre-promotion guarded-WebUI carry-forward slice)
- `qa/evidence-bundles/four-row-public-20260503T024327Z/manifest.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/SHA256SUMS`
- `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json`
- `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/llama3_8b_instruct_q8_0/manifest.json`
- `qa/validation-notes/2026-05-03-ubuntu-toolchain-and-8b-context.md`

Selected source artifacts recorded by those committed files:

- `target/acceptance-llama3-8b-broader-5tok-longtimeout-20260503T010536Z/summary.json` is the broader three-prompt 5-token parity pass referenced by the row bundle.
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
- the long-timeout Ubuntu three-prompt 5-token parity run passed for `hello`, alpacas, and JSON: `target/acceptance-llama3-8b-broader-5tok-longtimeout-20260503T010536Z/summary.json`
- `/v1/completions`, `/v1/chat/completions`, and frontend smoke are preserved in the sanitized carry-forward bundle at `qa/evidence-bundles/four-row-public-20260503T024327Z/llama3_8b_instruct_q8_0.bundle.json`; that bundle remains a pre-promotion guarded-WebUI slice from source smoke commit `c5e6d7e`, so a fresh current-head contract-supported rerun is still outstanding
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

Public evidence packaging note: sanitized carry-forward bundle manifests/checksums for the four-row smoke slices now live under `qa/evidence-bundles/four-row-public-20260503T024327Z/` and `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/`. They intentionally preserve the blocked public-head rerun state instead of overstating it.

Current-head durable execution note: the exact-row normalized rerun scaffold is now checked in at `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/`. Its per-row manifests/commands give docs, API, and frontend a stable current-head citation target while the Ubuntu reruns for longer context, broader template coverage, and stronger perf/portability evidence are still outstanding.

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
