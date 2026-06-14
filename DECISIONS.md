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

## D7 — Phase 4 (heterogeneous Mac+Pi): PORTABILITY SOLVED, Pi inference BLOCKED (2026-06-13)

Reported, not faked (operating rule #5). Two distinct results:

**RESOLVED — camelid is portable to aarch64-Linux (the spec's assumed hard blocker).**
- All 3 Pis found: `camelid1`/`camelid2`/`camelid3` on the local LAN (IPs redacted) — Pi 5, 16 GB,
  aarch64 Linux (kernel 6.12 rpi-2712), running NanoCamelid. Operator SSH key path redacted,
  user `tooleman`. (An older ssh-config LAN entry now points at a different device.)
- `build.rs` portable (x86 AMX gated to linux+x86_64; Accelerate macOS; Metal cfg(macos)).
  No rust cross-target locally (Homebrew, no rustup), so built **natively on camelid1**:
  installed rustup (1.96), rsynced source, `cargo build --release --example parity_node`
  succeeded in **2m10s** with zero errors. The Linux binary runs (prints usage; clean `ldd`,
  no Metal/Accelerate).

**RESOLVED — the Pi runs camelid inference; the earlier "crash" was an SSH self-kill.** The
launch failures were `pkill -f parity_node` matching the *SSH command's own cmdline* (which
contains "parity_node") and killing its own session before the launch ran — not a code crash.
Launching without that pkill, the Pi worker loads and listens fine (TinyLlama [11,22) ~2.2 GB
f32 est; llama-2-13b [20,40) 26 GB f32 est / ~6.5 GB Q8 resident, within a 40 GB cap).

**RESULT — cross-ARM parity is EXPERIMENTAL-DIVERGENT (a real finding, recorded).** Run
`hetero-mac-pi-tinyllama-q8`: Mac (Apple) coordinator [0,11) + reference, Pi (Cortex) worker
[11,22) + head, TinyLlama Q8_0 (byte-identical GGUF). The heterogeneous output is
**token-identical to the all-Apple single-node reference for the first 24 generated tokens,
then diverges at generated token 25** (`first_divergent_generated_token_index=25`,
`generated_tokens_match=false`). Cross-platform IEEE-754 differences (FMA contraction /
accumulation order, Apple vs ARM Cortex) compound until a greedy argmax flips. Sealed receipt
`qa/distributed/hetero-mac-pi-tinyllama-q8.json` (`receipt_id 11bbe0e1…`, `validated:false`) —
documented, NOT hidden behind a tolerance (operating rule #6). The lane stays **experimental**
for heterogeneous Apple+Cortex configs. parity_node now emits the receipt on divergence
instead of aborting.

**Too-big (llama-2-13b) capability evidence + reference gap.** Each shard loads only its slice
(Pi [20,40) ~6.5 GB Q8 within a 40 GB f32-est cap; full model's 52 GB f32 est would exceed it
→ "fits no single capped node"). But the **single-node reference OOM'd the 16 GB Mac** loading
the full 13 GB model — which is itself evidence the full model exceeds one 16 GB node. So a
same-engine reference can't be computed on-device for 13b; completing the 13b parity verdict
needs the **llama.cpp oracle** reference (spec-sanctioned for models that fit nowhere whole) —
a documented next step (add an oracle/external-reference mode to parity_node).

Phase 4 prep landed: per-node materialization cap (`--max-weight-bytes` /
`CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES`) with a typed refuse-to-start (verified:
TinyLlama [11,22) ~2.2 GB refuses at cap=1000).

## D5 — Branch / naming for the lane (2026-06-13)

Work proceeds on `feat/distributed-parity-lane` off `origin/main`. Single-node TinyLlama
1.1B Chat Q8_0 gate and the full validation suite (`fmt`/`clippy -D warnings`/`test`/`doc`)
must stay green before every commit (operating rule #3).

## D6 — `camelid chat` terminal mode (2026-06-13)

Work proceeds on `feat/terminal-chat` off `origin/main`. Full validation suite
(`fmt`/`clippy -D warnings`/`test`/`doc`) stays green before every commit (operating rule #3).
Recon detail backing these decisions: `RECON_CHAT.md`.

**Decision A — compatibility source: PROGRAMMATIC.** The supported set is structured data at
`GET /api/capabilities` → `CapabilitiesResponse.model_compatibility: Vec<ModelCompatibilityTarget>`
(and the Rust `capabilities_response_with_plan`). The picker filters rows by
`status.starts_with("supported")` at runtime — no hardcoded list, no second prose parser of
`COMPATIBILITY.md`. The "prose-only → stop for confirmation" boundary did not trigger. The
capabilities ledger (supported rows) and `curated_catalog()` (8 pullable rows) are two lists
joined on `model_compatibility.id == CatalogItem.catalog_id`; supported rows with no catalog
entry render `[supported · no pull alias]`.

**Decision B — architecture: Option B (HTTP client over a child-process `serve`).** Justification:
the engine already ships an audited OpenAI-compatible SSE lane at `/v1/chat/completions`; driving
it over HTTP makes terminal output provably identical to the validated lane (constraints 2/4/5)
and avoids Option A's re-plumbed streaming + token-for-token parity-test burden. The usual cost
of Option B (an HTTP client) is **zero here**: the repo has no HTTP-client dep but already
hand-rolls a blocking HTTP/1.1 client over `std::net::TcpStream` in `src/receipt/verify.rs`
(`http_json`/`parse_http_response`); `chat` reuses that pattern and extends it with a
line-buffered `data:` SSE reader. Control-plane actions reuse pub Rust in-process —
`catalog::run_pull`/`curated_catalog()` for pull, a `models/<filename>` fs check for availability;
only load/capabilities/health/generation go over HTTP. The `chat` command body is fully
synchronous (blocking client + blocking line editor + `std::process::Child`); no tokio in the
chat path.

- **Spawn-vs-attach:** probe `GET /v1/health` on `--addr` (default `127.0.0.1:8181`); attach if
  it answers, else spawn `camelid serve --addr <addr> --no-open` as a child and poll health until
  ready. Tear the child down on exit **only if we spawned it** (an attached server is left alone).

**History reset on model switch.** Switching the active model mid-session clears conversation
history (a different model = a different context window; carrying history is a footgun) and prints
a one-line notice. The `--system`/`/system` system prompt is **re-applied** across a switch
(it is a session-level instruction, not model-specific context).

**Constraint #4 gate (repo wins).** The backend's only generation gate is the typed
architecture error (`unsupported_runtime`/`generation_ready=false`). A recognized-arch
non-ledger GGUF would generate, same as the engine + React frontend today. `chat` reuses that
typed gate verbatim and does not invent new per-file matching (which would violate "reuse, don't
reimplement"); the picker's supported-only selection keeps the normal path from reaching a
non-supported row. Full rationale in `RECON_CHAT.md` §8.

**Line editor dependency: `rustyline`.** Chosen over `reedline` for a smaller, mature, sync API
(line editing + in-session history) that fits the synchronous chat loop with no async glue. It is
the single new dependency this feature adds.

**Ctrl-C semantics.** A SIGINT handler flips a cancel flag. During `rustyline` readline the
terminal is raw, so Ctrl-C arrives as a byte → `Interrupted` → an idle hint, not a quit. During
a stream the terminal is cooked, so Ctrl-C raises SIGINT → the read loop aborts cleanly and the
**entire in-flight turn (the user message and the partial reply) is discarded**, so history stays
coherent. Ctrl-D / `/exit` quit cleanly; a spawned server is torn down by `ServerHandle`'s Drop.

**Phase 8 evidence.** SSE/de-chunk parser and ledger-derived picker are covered by bin unit
tests (`cargo test --bin camelid chat::`). The supported-path and unsupported-gate end-to-end
checks live in `scripts/chat-terminal-smoke.sh`, gated on `CAMELID_CHAT_SUPPORTED_GGUF` /
`CAMELID_CHAT_UNSUPPORTED_GGUF` (no-op when unset, like the parity scripts) so they never block
`cargo test`.

## D7 — `camelid chat` full-screen TUI (2026-06-13)

`chat` grew a full-screen ratatui front end (the default on an interactive terminal). Both UIs
share one UI-agnostic `session::Session` core (state, sampling settings, request shape, save/load
— no I/O); `inline.rs` is the line REPL (now also the `--plain`/non-TTY fallback that the smoke
scripts drive) and `tui.rs` is the ratatui app.

- **Concurrency:** the redraw loop runs on the main thread; each generation streams on a
  background thread that forwards deltas over an `mpsc` channel, polled non-blocking each frame.
  Ctrl-C (a key event in raw mode) flips the shared `session::CANCEL` flag → the worker aborts.
- **Reuse intact:** the worker calls the same `Client::chat_stream` SSE lane; `Client` is now
  `Clone` (just a `SocketAddr`). No second generation path.
- **Deps:** `ratatui = "0.29"` (+ its `crossterm`). Only network/HTTP stays hand-rolled.
- **Downloads in the TUI:** selecting a not-downloaded picker row (or `/pull`) suspends the
  alt-screen, runs the existing `catalog::run_pull` with visible `curl` progress, then re-enters
  — so the audited pull path is reused and progress isn't swallowed by the alt-screen.
- **Front-end choice:** TUI when stdin+stdout are both TTYs and `--plain` is unset; else inline.
  The `--model` unsupported gate runs before either UI (typed error + exit 1, no screen takeover).
- **New options:** `/set` (temperature, top_p, top_k, max_tokens, seed, stream), `/system`,
  `/reset`, `/retry`, `/save`/`/load` (session JSON), `/info`, `/tokens`, `/pull`, plus CLI
  `--top-p/--top-k/--seed/--plain`. Mascot redesigned to the dromedary line-art.
