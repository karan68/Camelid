# Camelid Full-Support Blocker Matrix

Last updated: 2026-05-02

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
- Do not spend that lane on local Mac llama runtimes unless Tim explicitly redirects the work; keep the Mac for docs, frontend, or lightweight guardrail checks.

## Current full-support blocker matrix

| Row | Current product claim | Parity artifacts | API completions/chat | WebUI readiness | Memory/perf envelope | Docs/API/frontend agreement | Commit/push status | Full-support blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | Supported current gate | Documented PASS for the TinyLlama gate, including 5-prompt/50-token parity. Cited artifacts are under gitignored `target/` and are not present in this public worktree checkout. | Supported path documented; needs a fresh current-head real-GGUF API bundle if the four-row push requires reviewable recency evidence. | Supported gate documented; needs fresh current-head WebUI capture if we want all four rows to share the same evidence shape. | Documented as measured for the current gate; exact current-head artifact is not present in this checkout. | Public docs/API/frontend agree this is supported. | Public repo is pushed at `9d091ce`, which already includes the exact-row 8B smoke promotion. | Mostly evidence-packaging/recency: regenerate or preserve current-head parity/API/WebUI/perf artifacts in a durable location before using it as part of the four-row full-support release. |
| Llama 3.2 1B Instruct Q8_0 | Supported exact-row smoke, not broad/full support | Documented PASS for compact parity and the broader prompt pack. Cited artifacts are under gitignored `target/` and are not present in this checkout. | Documented exact-row load, completions, and chat-completions smoke. Need a current-head promotion bundle with saved request/response artifacts for release review. | Documented frontend smoke unlocks supported WebUI chat for exact row. Need saved current-head smoke artifact if promoting beyond smoke. | Only bounded smoke runs are claimed. Stronger memory/perf, longer-context, and portability evidence are still missing. | Public docs/API/frontend agree on **short local-chat smoke** only. | Public repo is pushed at `9d091ce` with 1B smoke support preserved. | Missing full-support expansion evidence: longer context, broader chat-template behavior, stronger memory/perf, portability, and durable current-head artifacts. |
| Llama 3.2 3B Instruct Q8_0 | Supported exact-row smoke, not broad/full support | Documented PASS for compact prompt-token/1-token/5-token/50-token parity and the post-Q8-dot broader 3-prompt/50-token pack. Cited artifacts are under gitignored `target/` and are not present in this checkout. | Documented exact-row load, `/v1/completions`, `/v1/chat/completions`, and five-prompt API smoke. Need a current-head promotion bundle saved for review if expanding the claim. | Documented frontend smoke unlocks supported WebUI chat for exact row. Need saved current-head smoke artifact if promoting beyond smoke. | Only bounded smoke/perf follow-up is claimed; stronger memory/perf envelope, longer contexts, and portability are missing. | Public docs/API/frontend agree on **short local-chat smoke** only; this is intentionally not a broad Llama-family claim. | Public repo is pushed at `9d091ce` with 3B smoke support preserved. | Missing full-support expansion evidence: longer context, broader chat-template acceptance, stronger memory/perf, portability, and durable current-head artifacts. |
| Llama 3 8B Instruct Q8_0 | Supported exact-row smoke, not broad/full support | Documented PASS for compact prompt-token/1-token/5-token/bounded-50-token parity plus the long-timeout three-prompt 5-token Ubuntu pack. | Documented exact-row load, `/v1/completions`, `/v1/chat/completions`, and API smoke. Need a current-head promotion bundle saved for review if expanding the claim. | Documented frontend smoke unlocks supported WebUI chat for the exact row. Need saved current-head smoke artifact if promoting beyond smoke. | Bounded compact/backend memory evidence exists, but a support-grade performance envelope, portability evidence, and longer-context evidence are still missing. | Public docs/API/frontend now agree on **short local-chat/parity smoke** only; this remains intentionally narrower than broad Llama-family support. | Public repo is pushed at `9d091ce` with the exact-row 8B smoke promotion landed. | Missing full-support expansion evidence: longer context, broader chat-template acceptance, stronger memory/perf, portability, and durable current-head artifacts. |

## Immediate work packets

### WP0 — Make evidence durable before more claims

- Create one non-ambiguous artifact root per run, for example `target/full-support-YYYYMMDDTHHMMSSZ/<row>/`.
- For each row, save: parity report(s), model-promotion smoke bundle, frontend smoke stdout/stderr/summary, memory samples, command lines, current `git rev-parse HEAD`, and model SHA256.
- If artifacts stay under gitignored `target/`, publish only the exact artifact manifest/checksums in docs; do not pretend reviewers can fetch local paths from GitHub.

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

### WP2 — Preserve the cleared 8B parity blocker

- Keep the failed alpacas/client-timeout case and the successful long-timeout rerun side by side in `STATUS.md` so the promotion history stays auditable.
- Treat `target/acceptance-llama3-8b-broader-5tok-longtimeout-20260503T010536Z/summary.json` as the exact-row 8B broader-prompt parity artifact for this smoke promotion.
- Do **not** lend the 8B row to neighboring Llama versions, sizes, quantizations, longer contexts, or broad chat-template behavior.

### WP3 — Expand beyond 8B smoke only with a new full-support bundle

Required before widening the public support ledger beyond exact-row smoke:

- Fresh `/api/models/load`, `/v1/models`, `/v1/completions`, `/v1/chat/completions` artifacts saved for the exact 8B row.
- WebUI smoke shows exact-row contract-supported readiness and normal chat unlock on current head.
- Memory/perf envelope documents bounded RSS/materialization, latency, token budget, host, model SHA, and no OOM/swap/runaway retained-RSS signature.
- Longer context, broader chat-template behavior, stronger performance/portability evidence, and synchronized docs/API/frontend updates land in the same commit.

## Current repo/artifact observations

- Current public worktree: based on pushed `9d091ce Promote exact Llama 3 8B smoke row`; the guardrails from `713c744` remain preserved in that landed state.
- The public checkout’s `target/` directory does **not** contain the cited parity/promotion artifacts. That is expected because `/target/` is gitignored, but it means local artifact paths in docs are not reviewable from GitHub by themselves.
- The older `projects/backendinference` worktree is behind public `origin/main` and locally dirty; use the public `projects/Camelid` worktree for release docs/commits unless deliberately recovering old local artifacts.
