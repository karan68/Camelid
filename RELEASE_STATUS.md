# Camelid v0.1 Release Status

Last updated: 2026-05-31

Branch: `release/v0.1-evidence`

Current release SHA: release branch HEAD; record exact SHA when cutting rc1

Release target: `v0.1.0-rc1`

Release posture: evidence release candidate in progress. No tag is allowed until all release gates below pass and the release captain signs off.

## Latest Release Captain Update

Camelid v0.1 update:

Shipped:

- Created clean release worktree on `release/v0.1-evidence` from `origin/main`.
- Added this release status ledger and gate checklist.
- Spawned scoped lanes for benchmark harness, correctness/support docs, comparator baselines, public docs, and QA gate.
- Created durable isolated cron release-captain job `86d81dd7-ace2-4739-a3d8-8bc15de73e23` on a 30-minute cadence with no public delivery until release evidence is ready.
- Generated dry-run harness bundle `qa/evidence-bundles/v0.1/dryrun-release-captain/` to prove bundle layout; it is skipped/dry-run evidence only, not benchmark evidence.

Evidence:

- Primary checkout was inspected before release edits and found dirty; no release edits were made there.
- Release worktree starts clean at source SHA `b6a769301415d0f91cba7b9a043f9f925fa4b884`.
- Durable continuation is enabled through cron job `86d81dd7-ace2-4739-a3d8-8bc15de73e23`; the isolated job has no pinned session key.
- Dry-run harness output contains the required files: `machine.json`, `model_manifest.json`, `commands.md`, `raw_logs/`, `results.json`, `results.csv`, and `summary.md`.

Blocker/Risk:

- Existing README and status language currently contain broad performance/UI/distributed phrasing that is not v0.1-safe until backed by fresh evidence or rewritten.
- Existing evidence bundles cite many historical SHAs; v0.1 needs a fresh bundle tied to this release branch SHA before an rc tag.
- The current v0.1 bundle is dry-run only: all four entries are `skipped`, so the real benchmark gate remains blocked.

Next:

- Run real Camelid/llama.cpp/Ollama/MLX comparator entries or explicitly defer them with release-captain rationale.
- Run lightweight harness self-tests and repo QA.
- Decide which comparator baselines can be run immediately on local macOS versus explicitly deferred.

Need Tim:

- No decision needed yet. Final `v0.1.0` tag remains approval-gated; `v0.1.0-rc1` is allowed only if release gates pass.

## Current Checkout

- Primary repo checkout inspected: `/Users/timtoole/.openclaw/workspace/projects/Camelid`
- Primary checkout state at start: `main`, SHA `1b207f953ad8d40abcd833bf4d4677b22d44b334`, behind `origin/main` by 17 commits, with existing uncommitted work.
- Release worktree: `/Users/timtoole/.openclaw/workspace/projects/Camelid-v0.1-evidence`
- Release worktree state at start: clean branch `release/v0.1-evidence` from `origin/main` at the release branch HEAD
- Preservation rule: the dirty primary checkout is not modified by this release lane.

## Release Captain Update Format

Camelid v0.1 update:

Shipped:

Evidence:

Blocker/Risk:

Next:

Need Tim:

## v0.1 Blockers

- Benchmark harness must generate a complete evidence bundle under `qa/evidence-bundles/v0.1/<timestamp>/`.
- llama.cpp baseline must be pinned, reproducible, and separated by backend mode.
- Ollama baseline must exist or be explicitly deferred with release-captain rationale.
- MLX baseline must exist or be explicitly deferred with release-captain rationale.
- Correctness and support matrices must cite exact-row evidence only.
- README and release docs must remove unsupported speed, model-family, UI, and distributed claims.
- Final QA gate must run on this release branch and record commands, machine, SHA, timestamps, pass/fail, and notes.
- `v0.1.0-rc1` may be created only after gates pass. Final `v0.1.0` requires Tim approval.

## Evidence Bundle Contract

Every benchmark result must record:

- Camelid commit SHA
- Comparator commit or version
- Model name
- Model path
- Model SHA256 hash
- Quantization
- Prompt
- Context size
- Max generated tokens
- Thread count
- Batch settings
- Runtime flags
- Environment variables
- Hardware details
- OS version
- Raw command
- Raw output
- Timing data
- Memory data
- Pass/fail status

## Release Gate Checklist

- [ ] Repo builds cleanly.
- [ ] Tests pass or failures are documented as non-release blockers.
- [ ] Benchmark harness runs from a clean checkout.
- [ ] llama.cpp baseline exists.
- [ ] Ollama baseline exists or is explicitly deferred with reason.
- [ ] MLX baseline exists or is explicitly deferred with reason.
- [ ] Correctness matrix exists.
- [ ] Support matrix exists.
- [ ] README is updated and does not overclaim.
- [ ] Release notes exist.
- [ ] Evidence bundle exists.
- [ ] Public docs contain no unsupported performance claims.
- [ ] Release Captain signs off.

## Lane Ownership

- Release Captain: release scope, evidence standards, final checklist, tag decision.
- Benchmark Harness: repeatable matrix runner and evidence bundle schema.
- Correctness and Parity: exact support matrix and parity proof boundaries.
- Apple Silicon Performance: macOS arm64 evidence only, clearly separated by runtime/backend mode.
- llama.cpp Comparator: pinned llama.cpp build and benchmark baseline.
- Ollama Comparator: practical user-facing benchmark baseline.
- MLX Comparator: Apple Silicon MLX market-context baseline.
- Distributed Mac Mini: included only if stable; otherwise explicitly excluded.
- Documentation: public README and release documents.
- QA and Release Gate: final commands, pass/fail ledger, and tag readiness.
