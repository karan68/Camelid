# Validation note — Ubuntu validation lane reopened

Date: 2026-05-04

Tim reopened the approved Ubuntu validation lane for Camelid promotion-grade runtime evidence. Do not publish private host addresses, SSH commands, key paths, or local operator-only details in the public repo.

Execution guardrails:

- Use clean public `main` checkouts for new validation runs.
- Preserve existing dirty remote worktrees; do not reset or overwrite them just to run current-head validation.
- Use `scripts/with-rustup-cargo.sh` or an equivalent rustup-managed toolchain on Ubuntu hosts whose distro `/usr/bin/cargo` is too old for the checked-in Rust floor.
- Generate full-support scaffolds on the reopened lane with `node scripts/prepare-full-support-bundle.mjs ...`; if Tim pauses the lane again, pass `--validation-host-status blocked_by_operator_shutdown` so generated runtime commands remain blocked.
- Keep claims exact-row only. A reopened host is not evidence; only passing artifacts can move docs, API, or frontend language.

Current promotion posture:

- TinyLlama remains the supported current gate.
- Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B Instruct Q8_0 remain supported exact-row smoke lanes only.
- Broader/full support still needs normalized current-head parity, API/WebUI, memory/perf, context, and durable-bundle evidence per exact row.
- The first 8B 512-context timeout has a passing rerun at `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json`; broader context, performance/portability, and full-support normalization remain blockers.
