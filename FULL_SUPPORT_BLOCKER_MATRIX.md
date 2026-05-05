# Camelid Full-Support Blocker Matrix

Last updated: 2026-05-04

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
| TinyLlama 1.1B Chat Q8_0 | Supported current gate | Documented PASS for the TinyLlama gate, including 5-prompt/50-token parity. Cited artifacts are under gitignored `target/` and are not present in this public worktree checkout. | Fresh reopened-lane exact-row API smoke is captured in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`. | Fresh reopened-lane frontend smoke is captured in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`. | Documented as measured for the current gate; exact support-grade current-head perf artifact is not present in this checkout. | Public docs/API/frontend agree this is supported. | Public repo is pushed at `027b301` with reopened-lane API/WebUI and 8B context evidence recorded. | Mostly evidence-packaging/recency for full-support normalization: preserve current-head parity/perf artifacts in a durable location before using it as part of the four-row full-support release. |
| Llama 3.2 1B Instruct Q8_0 | Supported exact-row smoke, not broad/full support | Documented PASS for compact parity and the broader prompt pack. Cited artifacts are under gitignored `target/` and are not present in this checkout. | Fresh reopened-lane exact-row API smoke is captured in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`. | Fresh reopened-lane frontend smoke unlocks supported WebUI chat for this exact row in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`. | Only bounded smoke runs are claimed. Stronger memory/perf, longer-context, and portability evidence are still missing. | Public docs/API/frontend agree on **short local-chat smoke** only. | Public repo is pushed at `027b301` with 1B smoke support preserved. | Missing full-support expansion evidence: longer context, broader chat-template behavior, stronger memory/perf, portability, and durable current-head artifacts. |
| Llama 3.2 3B Instruct Q8_0 | Supported exact-row smoke, not broad/full support | Documented PASS for compact prompt-token/1-token/5-token/50-token parity and the post-Q8-dot broader 3-prompt/50-token pack. Cited artifacts are under gitignored `target/` and are not present in this checkout. | Fresh reopened-lane exact-row API smoke is captured in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`. | Fresh reopened-lane frontend smoke unlocks supported WebUI chat for this exact row in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`. | Only bounded smoke/perf follow-up is claimed; stronger memory/perf envelope, longer contexts, and portability are missing. | Public docs/API/frontend agree on **short local-chat smoke** only; this is intentionally not a broad Llama-family claim. | Public repo is pushed at `027b301` with 3B smoke support preserved. | Missing full-support expansion evidence: longer context, broader chat-template acceptance, stronger memory/perf, portability, and durable current-head artifacts. |
| Llama 3 8B Instruct Q8_0 | Supported exact-row smoke, not broad/full support | Documented PASS for compact prompt-token/1-token/5-token/bounded-50-token parity, the long-timeout three-prompt 5-token Ubuntu pack, and the first bounded 512-context pack. | Fresh reopened-lane exact-row API smoke is captured in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`. | Fresh reopened-lane frontend smoke unlocks supported WebUI chat for this exact row in `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`. | Bounded compact/backend memory evidence exists, and the first 512-context pack passed with timed-process max RSS `17262740 KiB`; support-grade performance envelope, portability evidence, and broader context buckets are still missing. | Public docs/API/frontend now agree on **short local-chat/parity smoke** plus one bounded 512-context pack only; this remains intentionally narrower than broad Llama-family support. | Public repo is pushed at `027b301` with the exact-row 8B smoke promotion and first 512-context evidence preserved. | Missing full-support expansion evidence: broader context, broader chat-template acceptance, stronger memory/perf, portability, and durable current-head artifacts. Current note: the earlier first bounded longer-context timeout is cleared for one pack only; see `qa/validation-notes/2026-05-04-8b-context-512-rerun.md`. |

## Immediate work packets

### WP0 — Make evidence durable before more claims

- Create one non-ambiguous artifact root per run, for example `target/full-support-YYYYMMDDTHHMMSSZ-head-<sha>/`.
- Generate the normalized current-head scaffold first so every row has the same manifest/command shape before Ubuntu reruns: `node scripts/prepare-full-support-bundle.mjs --out-dir target/full-support-$(date -u +%Y%m%dT%H%M%SZ)-head-$(git rev-parse --short=12 HEAD)`.
- For each row, save: parity report(s), model-promotion smoke bundle, frontend smoke stdout/stderr/summary, memory samples, command lines, current `git rev-parse HEAD`, and model SHA256.
- If artifacts stay under gitignored `target/`, publish only the exact artifact manifest/checksums in docs; do not pretend reviewers can fetch local paths from GitHub.
- Current sanitized carry-forward bundle/checksum roots are `qa/evidence-bundles/four-row-public-20260503T024327Z/`, `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/`, `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/`, and `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/`; they improve reviewability but do **not** replace the remaining full-support parity/perf/portability reruns.

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
- Treat `target/acceptance-llama3-8b-broader-5tok-longtimeout-20260503T010536Z/summary.json` as the exact-row 8B broader-prompt parity artifact for this smoke promotion.
- Treat `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json` as the exact-row first 512-context pack artifact, not a broader context-support artifact.
- Do **not** lend the 8B row to neighboring Llama versions, sizes, quantizations, larger contexts, or broad chat-template behavior.

### WP3 — Expand beyond 8B smoke only with a new full-support bundle

Required before widening the public support ledger beyond exact-row smoke:

- Fresh `/api/models/load`, `/v1/models`, `/v1/completions`, `/v1/chat/completions` artifacts saved for the exact 8B row.
- WebUI smoke shows exact-row contract-supported readiness and normal chat unlock on current head.
- Memory/perf envelope documents bounded RSS/materialization, latency, token budget, host class, model SHA, and no OOM/swap/runaway retained-RSS signature.
- Broader context, broader chat-template behavior, stronger performance/portability evidence, and synchronized docs/API/frontend updates land in the same commit.
- Current state: remote validation is available again on the approved Ubuntu lane; do not substitute local Mac llama-server/reference workloads unless Tim explicitly authorizes that lane.

## Current repo/artifact observations

- Current public worktree: `027b3017883662e79c576f1d65a3da9fd71cc3a1 Record 8B context-512 validation evidence` is on `main` and aligned with `origin/main`; the exact-row smoke guardrails and reopened-lane evidence remain preserved in that landed state.
- Local `target/` has fresh 2026-05-04 doc/debug build outputs and `target/evidence-bundle-privacy-audit-20260504-watchdog.json`, which reports `finding_count: 0` for committed `qa/evidence-bundles`. These are local watchdog artifacts only because `/target/` is gitignored.
- The public checkout’s `target/` directory still does **not** contain the cited raw parity/promotion artifacts. That is expected because `/target/` is gitignored. Sanitized carry-forward manifests/checksums now live under `qa/evidence-bundles/four-row-public-20260503T024327Z/`, `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/`, and `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/`, but reviewers still cannot fetch the private raw target tree itself from GitHub.
- Reopened-lane API-only and API + frontend smoke summaries now live under `qa/evidence-bundles/four-row-api-only-20260504T230722Z-head-13a465608fbf/` and `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/`. They refresh exact-row API/WebUI readiness evidence only; they do not close parity, broader context/template, or performance blockers.
- A reopened-lane 8B 512-context summary now lives under `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/`; it closes only that one exact 8B context pack.
- Larger-model remote runtime validation is available again on Tim’s approved Ubuntu validation lane. Use clean public checkouts for new runs and preserve dirty worktrees.
- The older `projects/backendinference` worktree is behind public `origin/main` and locally dirty; use the public `projects/Camelid` worktree for release docs/commits unless deliberately recovering old local artifacts.
- The earlier Ubuntu rebuild failure was an environment/toolchain-selection issue, not a new Cargo.lock mystery: bare distro cargo was too old, while current Camelid head now has a verified Rust/Cargo floor of `1.87+`. Use the checked-in rustup wrapper/toolchain files for Ubuntu validation; details are in `qa/validation-notes/2026-05-03-ubuntu-toolchain-and-8b-context.md`.
