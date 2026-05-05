# Camelid Full-Support Blocker Matrix

Last updated: 2026-05-05

This is the owner/TPM matrix for the four-row target Tim actually cares about:

1. TinyLlama 1.1B Chat Q8_0
2. Llama 3.2 1B Instruct Q8_0
3. Llama 3.2 3B Instruct Q8_0
4. Llama 3 8B Instruct Q8_0

It is intentionally stricter than the public support ledger. `COMPATIBILITY.md`, `STATUS.md`, `README.md`, `/api/capabilities`, and the WebUI may honestly describe exact-row smoke support where that is all the evidence supports. This file tracks what is still missing before anyone should call the entire four-row set “fully supported.”

## Evidence gates

A row is **full-support green** only when all of these are green on the exact GGUF row being claimed:

- **Parity artifacts:** prompt-token parity plus deterministic generated token/text parity against llama.cpp or the row’s accepted reference across the agreed prompt pack and token budget.
- **API completions/chat:** `/api/models/load`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions` captured on current head for the exact row.
- **WebUI readiness:** frontend smoke proves runtime readiness and support-contract readiness agree; supported rows unlock normal chat, blocked rows fail closed.
- **Memory/perf envelope:** bounded RSS/materialization behavior plus enough latency/throughput evidence to describe the supported envelope without handwaving.
- **Docs/API/frontend agreement:** `COMPATIBILITY.md`, `STATUS.md`, `README.md`, `/api/capabilities`, and WebUI copy all say the same exact thing.
- **Commit/push status:** the evidence-backed support boundary is committed and pushed; any local artifact paths needed for review are either durable or explicitly reproducible.

## Execution-target guardrail

- For this four-row push, promotion-grade Llama-family parity, API, WebUI, and memory/perf reruns should execute on the canonical Ubuntu validation host recorded in private operator notes, not on ad hoc local runtimes.
- As of 2026-05-04, Tim has reopened the approved Ubuntu validation lane. Use fresh clean public `main` checkouts for promotion-grade reruns, preserve any dirty remote worktrees, and do not publish private host addresses in repo docs.
- Do not spend that lane on local Mac llama runtimes unless Tim explicitly redirects the work; keep the Mac for docs, frontend, evidence normalization, privacy scrub, or lightweight guardrail checks.

## Current full-support blocker matrix

| Row | Current product claim | Parity artifacts | API completions/chat | WebUI readiness | Memory/perf envelope | Docs/API/frontend agreement | Commit/push status | Full-support blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | Supported current gate | Documented PASS for the TinyLlama gate, including 5-prompt/50-token parity. Cited artifacts are under gitignored `target/` and are not present in this public worktree checkout. | Fresh reopened-lane exact-row API smoke is captured in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`. | Fresh reopened-lane frontend smoke is captured in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`. | Documented as measured for the current gate; exact support-grade current-head perf artifact is not present in this checkout. | Public docs/API/frontend agree this is supported. | Current public checkout through `83c21f0` preserves reopened-lane API/WebUI smoke evidence plus the normalized WP1 public bundle for this row. | Mostly evidence-packaging/recency for full-support normalization: preserve current-head parity/perf artifacts in a durable location before using it as part of the four-row full-support release. |
| Llama 3.2 1B Instruct Q8_0 | Supported exact-row smoke, not broad/full support | Documented PASS for compact parity and the broader prompt pack. Cited artifacts are under gitignored `target/` and are not present in this checkout. | Fresh reopened-lane exact-row API smoke is captured in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`. | Fresh reopened-lane frontend smoke unlocks supported WebUI chat for this exact row in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`. | Only bounded smoke runs are claimed. Stronger memory/perf, longer-context, and portability evidence are still missing. | Public docs/API/frontend agree on **short local-chat smoke** only. | Current public checkout through `83c21f0` preserves 1B exact-row smoke support, the reopened-lane API/WebUI refresh, and the normalized WP1 public bundle. | Missing full-support expansion evidence: longer context, broader chat-template behavior, stronger memory/perf, portability, and durable current-head artifacts. |
| Llama 3.2 3B Instruct Q8_0 | Supported exact-row smoke, not broad/full support | Documented PASS for compact prompt-token/1-token/5-token/50-token parity and the post-Q8-dot broader 3-prompt/50-token pack. Cited artifacts are under gitignored `target/` and are not present in this checkout. | Fresh reopened-lane exact-row API smoke is captured in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`. | Fresh reopened-lane frontend smoke unlocks supported WebUI chat for this exact row in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`. | Only bounded smoke/perf follow-up is claimed; stronger memory/perf envelope, longer contexts, and portability are missing. | Public docs/API/frontend agree on **short local-chat smoke** only; this is intentionally not a broad Llama-family claim. | Current public checkout through `83c21f0` preserves 3B exact-row smoke support, the reopened-lane API/WebUI refresh, and the normalized WP1 public bundle. | Missing full-support expansion evidence: longer context, broader chat-template acceptance, stronger memory/perf, portability, and durable current-head artifacts. |
| Llama 3 8B Instruct Q8_0 | Supported exact-row smoke, not broad/full support | Documented PASS for compact prompt-token/1-token/5-token/bounded-50-token parity, the three-prompt 50-token Ubuntu pack, the first bounded 512-context pack, and the bounded compact chat-template-shapes pack. | Fresh reopened-lane exact-row API smoke is captured in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`; the clean-main completion-diagnostics API/WebUI/RSS timing smoke passed at `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/manifest.json`, and the current-public-head normalized refresh passed at `qa/evidence-bundles/full-support-normalized-wp2-8b-watchdog-20260505T041404Z-head-83c21f0cbf5a/manifest.json`. | Fresh reopened-lane frontend smoke unlocks supported WebUI chat for this exact row in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`; the clean-main 8B smoke and the current-public-head normalized refresh also generated `Hello` with WebUI chat enabled. | Bounded compact/backend memory evidence exists, the first 512-context pack passed with timed-process max RSS `17262740 KiB`, the clean-main API/WebUI smoke sampled backend RSS `6316 -> 283352 KiB`, the current-public-head normalized refresh sampled max backend RSS `283372 KiB`, and the retained-block lazy-Q8 hot-path probe measured representative FFN/output tensor costs without claiming production throughput; support-grade performance envelope, portability evidence, and broader context buckets are still missing. | Public docs/API/frontend now agree on **short local-chat/parity smoke** plus one bounded broader 50-token pack, one bounded 512-context pack, one compact chat-template-shapes pack, one clean-main completion-diagnostics API timing slice, one current-public-head normalized API/WebUI/RSS smoke refresh, and one measurement-only lazy-Q8 hot-path cost probe; this remains intentionally narrower than broad Llama-family support. | Current public checkout through `83c21f0` preserves exact-row 8B API/WebUI/timing evidence, the current-public-head normalized API/WebUI/RSS refresh, and measurement-only lazy-Q8 hot-path evidence. | Missing full-support expansion evidence: broader context, broader chat-template acceptance, stronger memory/perf, portability, and durable full-support artifacts. Current note: the earlier first bounded longer-context timeout and compact template-shape gap are cleared for checked packs only, and the hot-path probe is optimization-grounding evidence only; see `qa/validation-notes/2026-05-05-8b-broader-50tok.md`, `qa/validation-notes/2026-05-04-8b-context-512-rerun.md`, `qa/validation-notes/2026-05-05-8b-chat-template-shapes.md`, `qa/validation-notes/2026-05-05-8b-api-webui-rss.md`, `qa/validation-notes/2026-05-05-8b-clean-main-api-webui-rss.md`, `qa/validation-notes/2026-05-05-lazy-q8-hotpath-costs.md`, `qa/validation-notes/2026-05-05-lazy-q8-hotpath-helper-validation.md`, and `qa/validation-notes/2026-05-05-wp2-8b-normalized-api-webui-watchdog.md`. |

## Immediate work packets

### WP0 — Make evidence durable before more claims

- Create one non-ambiguous artifact root per run, for example `target/full-support-YYYYMMDDTHHMMSSZ-head-<sha>/`.
- Generate the normalized current-head scaffold first so every row has the same manifest/command shape before Ubuntu reruns: `node scripts/prepare-full-support-bundle.mjs --out-dir target/full-support-$(date -u +%Y%m%dT%H%M%SZ)-head-$(git rev-parse --short=12 HEAD)`. Use `node scripts/prepare-full-support-bundle.mjs --help` for the side-effect-free option summary.
- For each row, save: parity report(s), model-promotion smoke bundle, frontend smoke stdout/stderr/summary, memory samples, command lines, current `git rev-parse HEAD`, and model SHA256.
- If artifacts stay under gitignored `target/`, publish only the exact artifact manifest/checksums in docs; do not pretend reviewers can fetch local paths from GitHub.
- Current sanitized carry-forward bundle/checksum roots are `qa/evidence-bundles/four-row-public-20260503T024327Z/`, `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/`, `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/`, `qa/evidence-bundles/full-support-normalized-wp1-20260505T032406Z-head-bcf9e647d6fd/`, `qa/evidence-bundles/full-support-normalized-wp2-8b-watchdog-20260505T041404Z-head-83c21f0cbf5a/`, `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/`, `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/`, `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/`, `qa/evidence-bundles/llama3-8b-api-webui-rss-20260505T014408Z-head-8cef7af4d6c6/`, `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/`, `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/`, and `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-helper-validated-20260505T0350Z-head-e22307f2f90b/`; they improve reviewability but do **not** replace the remaining full-support parity/perf/portability reruns.

### WP1 — Normalize current-head evidence shape for TinyLlama/1B/3B

Run the same promotion bundle shape for the three rows already claimed as supported/smoke-supported so the release story is not lopsided.

Template:

```bash
OUT="target/full-support-$(date -u +%Y%m%dT%H%M%SZ)"
node scripts/model-promotion-smoke-bundle.mjs \
  --model "$CAMELID_MODEL_DIR/<exact-gguf>" \
  --model-id <exact-row-id> \
  --out-dir "$OUT/<row>/api-webui" \
  --expect-compatibility-row <capability-row-id> \
  --expect-compatibility-status <expected-status> \
  --expect-contract-supported true \
  --expect-webui-chat enabled
```

### WP2 — Preserve the cleared 8B parity/context blockers

- Keep the failed alpacas/client-timeout case and the successful long-timeout rerun side by side in `STATUS.md` so the promotion history stays auditable.
- Treat `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json` as the current public exact-row 8B broader-prompt 50-token parity artifact for this smoke promotion; the older 5-token target artifact is historical carry-forward only.
- Treat `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json` as the exact-row first 512-context pack artifact, not a broader context-support artifact.
- Do **not** lend the 8B row to neighboring Llama versions, sizes, quantizations, larger contexts, or broad chat-template behavior.

### WP3 — Expand beyond 8B smoke only with a new full-support bundle

Required before widening the public support ledger beyond exact-row smoke:

- Fresh `/api/models/load`, `/v1/models`, `/v1/completions`, `/v1/chat/completions` artifacts saved for the exact 8B row. Current note: the clean-main completion-diagnostics smoke at `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/manifest.json` validates the API timing-summary path after the patch landed; it is still an exact-row smoke slice, not broader/full support evidence.
- WebUI smoke shows exact-row contract-supported readiness and normal chat unlock on current head.
- Memory/perf envelope documents bounded RSS/materialization, latency, token budget, host class, model SHA, and no OOM/swap/runaway retained-RSS signature.
- Broader context, broader chat-template behavior, stronger performance/portability evidence, and synchronized docs/API/frontend updates land in the same commit.
- Current state: remote validation is available again on the approved Ubuntu lane; do not substitute local Mac llama-server/reference workloads unless Tim explicitly authorizes that lane.

## Current repo/artifact observations

- Current public checkout preserves exact-row smoke guardrails and reopened-lane evidence. The normalized TinyLlama/1B/3B API/WebUI smoke bundle is published at `qa/evidence-bundles/full-support-normalized-wp1-20260505T032406Z-head-bcf9e647d6fd/manifest.json`; the normalized 8B API/WebUI/RSS refresh is published at `qa/evidence-bundles/full-support-normalized-wp2-8b-watchdog-20260505T041404Z-head-83c21f0cbf5a/manifest.json`; the completion-diagnostics API evidence has a clean-main rerun at `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/manifest.json`, the retained-block lazy-Q8 hot-path probe is published at `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/manifest.json`, and the helper-validation repeat is published at `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-helper-validated-20260505T0350Z-head-e22307f2f90b/manifest.json` as measurement evidence only.
- Local `target/` has 2026-05-05 normalized scaffold/audit outputs including `target/full-support-20260505T025614Z-head-5eaec2ffe489/manifest.json`, `target/evidence-bundle-privacy-audit-20260505T025551Z.json`, and `target/evidence-bundle-privacy-audit-post-harness-help.json`; the privacy audits report `finding_count: 0` for committed `qa/evidence-bundles`. These are local watchdog artifacts only because `/target/` is gitignored.
- The public checkout’s `target/` directory still does **not** contain the cited raw parity/promotion artifacts. That is expected because `/target/` is gitignored. Sanitized carry-forward manifests/checksums now live under `qa/evidence-bundles/four-row-public-20260503T024327Z/`, `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/`, `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/`, `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/`, `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/`, and `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/`, but reviewers still cannot fetch the private raw target tree itself from GitHub.
- Reopened-lane API-only and API + frontend smoke summaries now live under `qa/evidence-bundles/four-row-api-only-20260504T230722Z-head-13a465608fbf/` and `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/`. They refresh exact-row API/WebUI readiness evidence only; they do not close parity, broader context/template, or performance blockers.
- Reopened-lane 8B broader 50-token, 512-context, compact chat-template-shapes, patched-tree API/WebUI/RSS timing, clean-main API/WebUI/RSS timing, retained-block lazy-Q8 hot-path measurement, and helper-validation repeat summaries now live under their dedicated `qa/evidence-bundles/llama3-8b-*` directories. They close only those exact checked 8B slices; the hot-path probes are optimization-grounding measurement evidence, not production performance or portability claims.
- Larger-model remote runtime validation is available again on Tim’s approved Ubuntu validation lane. Use clean public checkouts for new runs and preserve dirty worktrees.
- The older `projects/backendinference` worktree is behind public `origin/main` and locally dirty; use the public `projects/Camelid` worktree for release docs/commits unless deliberately recovering old local artifacts.
- The earlier Ubuntu rebuild failure was an environment/toolchain-selection issue, not a new Cargo.lock mystery: bare distro cargo was too old, while current Camelid head now has a verified Rust/Cargo floor of `1.87+`. Use the checked-in rustup wrapper/toolchain files for Ubuntu validation; details are in `qa/validation-notes/2026-05-03-ubuntu-toolchain-and-8b-context.md`.
