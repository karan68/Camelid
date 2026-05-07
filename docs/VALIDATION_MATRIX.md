# Validation Matrix

Last updated: 2026-05-07

Run the smallest meaningful validation lane for your change. If a change affects support language, readiness behavior, or exact-row claims, update docs and evidence references together.

## Current support-surface alignment rule

Every public surface should tell the same four-row story:

- TinyLlama 1.1B Chat Q8_0 is the supported current gate, with checked 512-context/template/RSS evidence.
- Llama 3.2 1B and 3B Instruct Q8_0 are exact-row smoke-supported through checked 512/1024/2048 bounded context packs.
- Llama 3 8B Instruct Q8_0 is exact-row smoke-supported through the checked 512 bounded context pack only.
- Llama 3 8B 1024/2048 remain red unless fresh PASS artifacts and synchronized docs/API/frontend alignment exist; broad/full 8B support still requires model-native/larger context beyond checked packs, arbitrary-template, throughput, and portability evidence.

Next-family public language is locked to planning/validation, not support:

- Mistral 7B Instruct: “In active validation for `Mistral-7B-Instruct-v0.3.Q8_0.gguf`; not supported yet.”
- Mixtral 8x7B Instruct: “Planned first MoE exact-row candidate for `Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf`; not supported yet.”
- Qwen 2.5 7B Instruct: “Planned exact-row candidate for `Qwen2.5-7B-Instruct-Q8_0.gguf`; not supported yet.”
- Gemma 2 9B Instruct: “Planned exact-row candidate for `gemma-2-9b-it-Q8_0.gguf`; not supported yet.”

First promotion for any of those rows requires row-specific source/SHA/license, tokenizer/template references, bounded load/readiness, parity, API/WebUI, RSS/timing, scrubbed manifest, and checksum evidence. Mixtral also needs MoE expert-routing proof or typed unsupported behavior before any runtime claim.

| Change type | Minimum expected checks | Extra checks when relevant | Notes |
| --- | --- | --- | --- |
| Docs-only | `git diff --check`<br>`bash scripts/check-public-scrub.sh` | n/a | Keep support language synchronized with `README.md`, `COMPATIBILITY.md`, `STATUS.md`, and UI copy when claims change. |
| Frontend-only copy/layout | `cd frontend && npm ci && npm run build` | `npm run smoke` or `npm run smoke:tiny` when chat/model-load/readiness surfaces change | Do not loosen readiness gates or support wording without matching evidence/docs updates. |
| Backend-only non-inference changes | `cargo fmt --all -- --check`<br>`cargo clippy --all-targets --all-features -- -D warnings`<br>`cargo test --all-targets --all-features`<br>`cargo doc --no-deps --all-features`<br>`bash scripts/check-public-scrub.sh` | frontend build if API shape or delivery may be affected | Good default lane for parser, API, CLI, and non-runtime refactors. |
| Inference/tokenizer/runtime changes | Standard backend gate above | targeted parity, readiness, or smoke artifacts for the affected exact row(s) | Do not broaden support from seam evidence alone. |
| Frontend + backend readiness/chat-path changes | Standard backend gate + `cd frontend && npm ci && npm run build` | frontend smoke against the affected exact row(s) | Required when `/v1/health`, `/api/capabilities`, model loading, or WebUI chat gating changes. |
| Support-contract / compatibility-row changes | Validation appropriate to the underlying code/docs change | fresh evidence bundles and synchronized updates to public sources of truth | A support claim is a release decision, not a wording tweak. |
| QA / evidence-publication changes | Validate the producing scripts or manifests you changed | scrub/publication checks and updated artifact references | Keep public bundle paths, manifests, and summaries internally consistent. |

## Public vs maintainer-only validation

Public contributor expectations stop at local reproducible checks plus public artifact references.

The following may still be maintainer-only workflows rather than baseline contributor requirements:

- promotion-grade reruns on the approved Ubuntu validation lane
- SSH-backed remote execution
- private operator recovery/debug procedures

Public docs may reference those workflows at a high level, but should not depend on unpublished infrastructure details.

## When in doubt

- choose the smallest lane that could realistically catch your change
- if a claim gets stronger, the evidence must get stronger too
- if code, docs, frontend copy, and compatibility rows disagree, the task is not finished
