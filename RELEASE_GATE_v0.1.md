# Camelid v0.1 Release Gate

Date: 2026-05-31

Branch: `release/v0.1-evidence`

Release candidate SHA: release branch HEAD; record exact SHA when cutting rc1

Tag status: no tag created.

## Gate Summary

Current status: not ready to tag.

The documentation posture is now v0.1-safe, but the release gate has blocking failures. This file records the commands that ran, their results, and the unresolved release blockers.

## Required Lightweight Gates

Run these from the release checkout:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
node scripts/check-public-evidence-claims.mjs
bash scripts/check-public-scrub.sh
cd frontend && npm ci && npm run build && npm run smoke:model-state
```

## Gate Results

| Gate | Command | Status | Notes |
| --- | --- | --- | --- |
| Branch/SHA | `git status --short --branch && git rev-parse HEAD` | PASS | Confirmed `release/v0.1-evidence` at the release branch HEAD. Multiple untracked release-lane files already exist and are treated as other lanes' work unless named in this document. |
| Rust format | `cargo fmt --all -- --check` | FAIL | Existing Rust formatting drift in `src/api/mod.rs`, `src/cluster.rs`, `src/distributed.rs`, `src/main.rs`, `tests/api_vertical_slice.rs`, and `tests/distributed_tests.rs`. This docs/QA lane did not auto-format source code. |
| Rust clippy | `CARGO_TARGET_DIR='/Volumes/SSK Drive/OpenClaw/cargo-targets/Camelid-v0.1-evidence' cargo clippy --all-targets --all-features -- -D warnings` | FAIL | Clippy reported unused assignment in `src/inference.rs`, redundant closure and needless borrow in `src/api/mod.rs`, manual slice-size calculation in `src/cluster.rs`, `map_or` simplifications in `src/inference.rs`, range-loop warnings in `src/tensor/mod.rs`, and one too-many-arguments lint in `src/inference.rs`. |
| Rust tests | `CARGO_TARGET_DIR='/Volumes/SSK Drive/OpenClaw/cargo-targets/Camelid-v0.1-evidence' cargo test --all-targets --all-features` | FAIL | 305 passed, 5 failed, 1 ignored. Failures are Metal tests in `src/metal.rs`: `metal_linear_row_transposed_matches_cpu_for_small_dense_dot_rows`, `metal_q8_0_encoded_linear_row_matches_cpu_for_small_rows`, `metal_q8_0_encoded_linear_rows_matches_cpu_for_small_rows`, `metal_q8_0_block_linear_row_matches_cpu_for_small_rows`, and `metal_q8_0_block_two_linear_rows_matches_cpu_for_small_rows`. |
| Public evidence claims | `node scripts/check-public-evidence-claims.mjs` | PASS | Checked 96 manifest files and 49 summary files. |
| Public scrub | `bash scripts/check-public-scrub.sh` | PASS | No public scrub violations reported. |
| Frontend install/build/model-state smoke | `cd frontend && npm ci && npm run build && npm run smoke:model-state` | FAIL | `npm ci` and `npm run build` passed. `npm run smoke:model-state` failed because the smoke expects only the TinyLlama/1B/3B/8B hardening row ids, while current capabilities data also includes `mistral_7b_instruct_v0_3_q8_0`. |

## Comparator and Evidence Gates

| Gate | Status | Required before tag |
| --- | --- | --- |
| v0.1 evidence bundle | PARTIAL / BLOCKED | Dry-run bundle `qa/evidence-bundles/v0.1/dryrun-release-captain/` proves harness output shape only. Real Camelid/comparator benchmark entries are still required or must be explicitly deferred. |
| llama.cpp baseline | BLOCKED | Run a pinned same-host baseline or explicitly defer with rationale. |
| MLX-LM baseline | PARTIAL | Memory comparison evidence exists; v0.1 speed baseline must be run or explicitly deferred. |
| Ollama baseline | BLOCKED | Run baseline or explicitly defer with rationale. |
| Support matrix | Out of scope for this lane | Owned by another lane; do not edit here. |
| Correctness matrix | Out of scope for this lane | Owned by another lane; do not edit here. |

## Current Blocking Failures

- Rust formatting gate is red.
- Clippy gate is red.
- Rust test gate is red on five Metal tests.
- Frontend model-state smoke is red due to the Mistral row expectation mismatch.
- Only a dry-run v0.1 evidence bundle exists; real benchmark evidence is missing.
- Comparator baseline decisions are unresolved.

## Tag Rule

Do not create `v0.1.0-rc1` or `v0.1.0` from this lane until:

- lightweight gates pass or have documented non-blocking failures
- comparator baseline status is resolved
- a fresh v0.1 evidence bundle exists or is explicitly deferred by the release captain
- release docs and README contain no unsupported performance, model-family, UI, or distributed claims
- release captain signs off
- Tim approves any final `v0.1.0` tag
