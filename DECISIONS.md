# DECISIONS.md — distributed parity lane

Authoritative record of binding decisions for the distributed parity lane. Each entry is
dated and justified. Companion to `DISTRIBUTED_RECON.md`.

## D1 — Topology: pipeline parallelism by contiguous layer block (2026-06-13)

Cut the decoder layer stack into contiguous blocks; each node loads only its block's
weights and owns only its block's KV. One hidden-state vector walks the stack node→node,
one network hop per block per token. Coordinator holds embeddings-in + final-norm/output-
projection/sampling-out and lives on the fastest node; shards hold a contiguous layer block
and its KV only.

**Wire:** raw little-endian f32 activations, row-major, length-prefixed framed TCP,
synchronous request/response per hop per token. The scalar absolute position travels with
the activation (needed for RoPE and the KV write offset; positions are not reconstructable
on a shard that skipped earlier layers).

**Why this and not tensor parallelism:** the goal is to make memory add up while keeping the
math exactly sequential — sequential math is what makes token-identity (the gate) provable.
Tensor parallelism splits within each layer, forcing a network sync per layer (latency
multiplier) and a far larger numeric-divergence surface. **Tensor parallelism is rejected
for this lane.**

**Split ratio by RAM-fit, not speed.** A ~10× Mac↔Pi compute gap cannot be balanced; do not
try. Mac gets the large block (+coordinator); each Pi gets a contiguous tail block sized
under its 8 GB ceiling.

This decision matches what the repo already implements (`distribute-master`/
`distribute-worker`, `src/cluster.rs` framed activation protocol, `forward_layer_range_from_hidden`).
It is recorded here as binding, not as new construction.

## D2 — Naming: use `camelid` / `CAMELID_*`, NOT `backendinference` (2026-06-13)

The spec's operating rule #2 says "names stay on `backendinference`" and references
`BACKENDINFERENCE_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES`. **This is stale.** The actual,
current package identifiers are:

- crate/binary: `camelid` (`Cargo.toml:2`), subcommands under `camelid …`.
- env var: `CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES` (`src/api/mod.rs:57`).
- `backendinference` / `BACKENDINFERENCE_*` is **legacy branding that the repo's own public
  scrub CI actively forbids** (`scripts/check-public-scrub.sh:46-48`, branding_pattern
  `backendinference|BackendInference|backend inference`).

Following rule #2 literally would reintroduce branding that CI rejects and that an earlier
rename deliberately removed. The **intent** of rule #2 — "keep current package identifiers,
no rename in this lane" — is satisfied by using `camelid` / `CAMELID_*`, and is *violated*
by introducing `backendinference`. **Decision: use `camelid` / `CAMELID_*` everywhere.** The
memory-cap discipline the spec invokes is enforced via
`CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES`.

## D3 — Reuse existing infrastructure; the lane's deliverable is the receipt, not plumbing (2026-06-13)

Recon (see `DISTRIBUTED_RECON.md`) found the transport, shard servers, coordinator,
per-token pipeline, and layer-range forward already exist and are tested. Gemma 4 already
has a passing distributed parity gate vs a llama.cpp oracle. The genuinely missing work is:

1. A bitwise in-process chained-partition parity test for **Llama** (Phase 1), with the
   execution lane pinned (see D4).
2. A distributed parity **receipt** for the Llama path in the spec's artifact schema,
   built on the existing `camelid.parity-receipt/v1` framework (`src/receipt/`), not a new
   one (Phases 2–4).
3. The cluster frontend tab with a standing experimental banner (Phase 5).

**Decision: do not re-implement working code.** Retrofit parity + receipts onto the
existing Llama distributed path and lift the generic `cluster.rs` protocol (add the
versioning + FNV checksum hygiene Gemma 4 already proved) rather than inventing a parallel
stack. Pending user confirmation of this re-scope (the spec was written as if greenfield).

## D4 — Pin the parity reference lane (2026-06-13, RESOLVED at Phase 1 gate)

`forward_layer_range_from_hidden` routes `seq_len==1` decode to the GPU-resident path
(`src/inference.rs:2110-2117`), which is a different implementation from the CPU chunk path.
Token-identity must be asserted against a **single, named** reference lane, not "whatever
the engine happens to pick." **Decision (provisional): the parity reference is single-node
`camelid` greedy (temp 0, seed 0) on the same exact GGUF, with the distributed run forced
onto the same execution lane (CPU vs resident) as the reference.** Finalize at the Phase 1
gate once the in-process split test exists. Never paper over a divergence with a tolerance
window (operating rule #6); a single differing f32 bit is a finding to document, not smooth.

**RESOLVED:** the reference lane is the **CPU chunk path** (`forward_layer_range_from_hidden`
with resident paths disabled via `set_resident_paths_disabled(true)`), driven identically by
both the single-node reference and the split so the test isolates loop-cut state leaks from
kernel differences. Phase 1 gate **PASSED** 2026-06-13: `tests/distributed_llama_parity.rs`
proves 8 greedy steps (prefill + 7 decode) bitwise-identical between full `[0,22)` and chained
`[0,11)+[11,22)` on TinyLlama 1.1B Chat Q8_0 (`fmt`/`clippy -D warnings` clean; test-only, so
the single-node 50-token gate is unchanged). Open follow-up: a separate gate may later validate
the GPU-resident lane across a split, but the distributed pipeline's parity reference stays the
CPU lane.

## D6 — Phase 3 driver: a same-binary `parity_node` example (2026-06-13, PASS)

Phase 3 (two Macs over LAN) runs through `examples/parity_node.rs`, a single binary with
`worker` and `coordinator` modes. Rationale: the existing `distribute-master`/`-worker`
CLI does not pin the CPU lane or emit a `DistributedParityReceipt`, both of which the gate
requires. The example reuses the library's public session API, `src/cluster.rs` wire, and
`src/receipt`. **The same binary must run on every node** — a different build would defeat
the parity claim — so it is built once and copied to each node.

Transport hardening (not parity relaxation): the worker resets its KV cache per connection
(a persistent worker serves many runs); the coordinator resets its KV per run, uses bounded
connect-retry + whole-run retry, and the worker survives a single bad run. mini2's worker
runs under `caffeinate` (App Nap / Wi-Fi power-save otherwise stalled `accept()`).

**Result PASS 2026-06-13:** two consecutive runs token-identical, mac-m4 `[0,11)` → mini2
`[11,22)`, TinyLlama 1.1B Q8_0 (byte-identical GGUF, sha `a4c9bb1d…`); deterministic
`receipt_id 33b79d8d…`; artifacts `qa/distributed/{two-mac-tinyllama-q8.json,
cluster-topology.json}` + `CLUSTER_BENCH.md` (honest latency, ~5.5 tok/s, capability not
speed).

## D7 — Phase 4 prep + Pi blocker (2026-06-13, BLOCKED on hardware)

Phase 4 (a model too big for any single node, across Mac+Mac+Pi, parity vs a llama.cpp
oracle) is blocked at the hardware boundary, reported not faked (operating rule #5):
- Pi at `192.168.86.27` unreachable (off / off-network); other Pis don't resolve; the Pis
  run NanoCamelid, not camelid.
- No rust cross target here (Homebrew toolchain, no `rustup`), so cross-compiling camelid
  for aarch64-Linux isn't available locally. **Plan: build `parity_node` natively on the Pi**
  (its own aarch64-Linux rustc) — `build.rs` is confirmed portable (x86 AMX shim gated to
  `linux+x86_64`; Accelerate to macOS; Metal to `cfg(macos)`).
- Both Macs are 16 GB (32 GB aggregate). The only local model too big for any node is
  Mixtral-8x7B Q8_0 (46 GB, MoE — camelid support uncertain), which needs ≥3 nodes → needs
  the Pi.

Prep landed so the Pi run is turnkey once a Pi is online: `parity_node` now enforces a
per-node materialization cap (`--max-weight-bytes` / `CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES`)
and **refuses to start with a typed error** when a shard's slice exceeds it (verified: the
[11,22) TinyLlama shard reports ~2.2 GB and refuses at cap=1000). User chose to bring the
Pis online first; awaiting reachable Pi address(es).

## D5 — Branch / naming for the lane (2026-06-13)

Work proceeds on `feat/distributed-parity-lane` off `origin/main`. Single-node TinyLlama
1.1B Chat Q8_0 gate and the full validation suite (`fmt`/`clippy -D warnings`/`test`/`doc`)
must stay green before every commit (operating rule #3).
