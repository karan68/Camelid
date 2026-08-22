# Camelid Phase 1 runtime comparator

This internal harness implements the first benchmark-system phase: a local,
informational, same-host base/head comparison over Camelid's hidden
`bench-generate` command.

It does not set a regression threshold, run in CI, compare external runtimes,
exercise either agent loop, or create public evidence.

## Safety and validity

- Source SHAs, `Cargo.lock`, model bytes, prompt bytes, and built binaries are
  hashed or checked before measurement.
- Base and head build into separate target directories.
- The planner owns balanced arm order; a campaign cannot submit all-base then
  all-head ordering.
- Every process block gets a unique marker at the front of the prompt, shared by
  the paired base/head arms.
- Phase 1 supports only `cpu_deterministic`, asserted by `--deterministic` plus
  the absence of Camelid's structured GPU offload status.
- Cross-arm output-token divergence invalidates the performance result.
- Invalid and unfavorable samples remain in the sealed local bundle.
- Numeric verdicts remain `INCONCLUSIVE_NOISE` until a later calibration phase
  approves practical margins and sample counts.

## Commands

Print the source-manifest digest for the exact harness code and schemas:

```sh
node tools/bench/system/cli.mjs digest
```

Copy `examples/campaign.phase1.json` outside the tracked tree, replace every
placeholder, and pin the digest above. Audit the resolved plan without building:

```sh
node tools/bench/system/cli.mjs plan --config <campaign.json> --out <plan.json>
```

Run the complete local campaign. This builds both arms serially unless a local
ablation campaign explicitly supplies `--prepared` identities:

```sh
node tools/bench/system/cli.mjs run --config <campaign.json> --out-root <output-root>
```

The output directory is refused if it already contains files. A complete bundle
contains the resolved plan, prepared binary identities, raw stdout/stderr,
materialized block prompts, per-sample records, comparison, summary, manifest,
and `SHA256SUMS`.

Builds set `CARGO_NET_OFFLINE=true` when the campaign network policy is `deny`.
Provision the pinned toolchain and dependency cache before starting; a missing
crate is a preparation failure, not permission to resolve mutable dependencies
during measurement.

One lock file serializes campaigns under the selected output root. The harness
does not remove a stale lock automatically. After an interrupted controller,
prove the recorded PID and all campaign-owned children are gone before manually
removing the lock. A run that fails after creating its directory writes
`failure.json` with state `INCOMPLETE`; reruns use a new campaign ID.

## Self-tests

```sh
node tools/bench/system/test-schemas.mjs
node tools/bench/system/test-bench-generate-parser.mjs
node tools/bench/system/test-stats.mjs
node tools/bench/system/test-planner.mjs
node tools/bench/system/test-prepare.mjs
node tools/bench/system/test-process-runner.mjs
node tools/bench/system/test-runtime-adapter.mjs
node tools/bench/system/test-bundle.mjs
node tools/bench/system/test-cli.mjs
node tools/bench/system/test-safety.mjs
node tools/bench/test-v0.1-benchmark-harness.mjs
```

The existing `validation-scripts` CI job runs the same set through
`scripts/test-benchmark-system-phase1.mjs`.
