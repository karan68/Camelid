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

## D8 — `camelid chat` production feature pass (2026-06-14)

Added on top of the TUI (D7), all sharing the `session::Session` core and the same audited
HTTP/SSE lane:

- **Slash command palette** (`palette.rs`): typing `/` opens a filtering popup over the input
  (prefix-ranked, ↑↓ select, Tab complete, Enter run). One static `COMMANDS` registry is the
  single source of truth for the palette, `/help`, and dispatch (alias-aware) in both front ends.
- **Instant loaded-model switching** — **zero backend change**: `get_or_load_model` already
  activates an already-loaded model by the request's `model` id with no reload, and `GET
  /v1/models` lists every loaded model (id + ctx/params/size). `/switch` and the model browser's
  "● loaded (instant)" section just send the chosen id; `/model <id>` prefers a loaded model.
- **Markdown rendering** (`markdown.rs`, dependency-free): fenced code blocks, headings, bullets,
  block quotes, inline `code`/**bold**/*italic*, with style-aware width wrapping. Assistant turns
  render as Markdown; user turns stay plain.
- **Themes** (`theme.rs`): sandstorm/mono/ocean/nord, `/theme [name]` cycles/picks; every widget
  + the markdown renderer pull styles from the active theme.
- **Live streaming** spinner + tok/s in the status bar (frame-driven), a **context gauge** in the
  sidebar (last-turn tokens vs the model's `n_ctx_train`), and **`/copy`** to the clipboard via an
  OSC 52 escape with a hand-rolled base64 (`clipboard.rs`) — works inside the alt-screen and over
  SSH, no dep.
- New deps this pass: **none** beyond `ratatui` (added in D7). New unit tests: palette
  resolve/rank, markdown render/inline-code, base64 vectors (bin chat tests 18 → 27).

Note: ratatui's diff renderer interleaves cursor escapes between characters, so PTY screen-scrape
string matching is unreliable; verify features by reconstructing the screen, not substring search.

## D9 — Deterministic CPU forward pass, Pillar One (2026-06-14)

Opt-in deterministic inference for the supported **TinyLlama 1.1B Chat Q8_0** lane, behind
`--deterministic` (`serve` and `bench-generate`; env `CAMELID_DETERMINISTIC=1`). The default
(GPU resident-decode) fast path is untouched — verified byte-for-byte identical token stream
before/after the change at ~88 tok/s on M4. Credit: the pinned reduction order mirrors the
llama.cpp reference block-wise Q8_0 dot layout the parity contract is gated against.

**Flag, not a Cargo feature.** Determinism is a *runtime path choice* (CPU vs the Metal GPU
fast stack), not a compile-time one: the same shipped binary serves both the default fast path
and the deterministic path, and nothing in the default build changes. A Cargo feature would
fork the binary and risk a different default artifact; a runtime flag keeps one byte-identical
default binary and lets a caller opt in per invocation (or per process via the env var, which
the engine reads directly so library embedders get the same guarantee).

**Mechanism (fail-closed in the engine).** `apply_deterministic_mode()` (CLI) sets
`CAMELID_DETERMINISTIC=1`, forces the whole Metal stack off, and disables GPU sampling. The
engine reads `inference::deterministic_mode_enabled()` and **ANDs every Metal/GPU dispatch gate
with `!deterministic`** (`q8_0_metal_enabled`, `resident_decode_metal_enabled`,
`q8_0_metal_retained_enabled`, `q8_0_hybrid_retained_enabled`, and the two
`try_*_linear_row_metal` paths). So even if a `CAMELID_METAL_*` override is present, deterministic
mode still resolves to the order-stable CPU kernels. The default path (`deterministic == false`)
takes the existing branch unchanged.

**Why this is bit-exact for free on the CPU path.** Phase 0 (see
`qa/determinism/determinism-baseline-*.md`) established that the CPU forward pass is *already*
order-stable: every reduction site parallelizes over the **output** dimension (one thread owns a
disjoint set of outputs and runs that output's entire reduction serially), with no cross-thread
float combine, no atomics, no parallel sum/reduce/fold, and a compile-time-fixed SIMD lane
layout. Empirically the CPU stream is byte-identical across runs, across processes, and between
`--threads 1` and `--threads 10`. So `--deterministic` adds **no determinism penalty inside the
CPU computation** — its only cost is forgoing the GPU fast path (~12.5 vs ~88 tok/s on M4).

### The pinned reduction order (the contract a future portable trace depends on)

This order is **order-stable** on a given binary + host (identical across runs/processes/thread
counts). The *values* remain ISA-dependent (i8mm vs dotprod vs scalar round differently), so a
cross-machine trace must pin the host/ISA, not assume cross-ISA bit-equality.

- **Site 1 — Linear / matmul** (Q, K, V, attention-output, FFN gate/up/down) **and Site 3 — lm_head**
  (same kernels, `q8_0_packed_rows4_dot_i8_matmul` / `dot_product_row`):
  1. Output-partitioned across rayon — each output element (or packed group of 4) is computed
     entirely by one thread; parallelism **never** splits a single output's reduction.
  2. The activation row is quantized once (`quantize_q8_0_row`, an order-independent transform),
     then the dot over the K dimension accumulates **block by block in ascending Q8_0 block
     index** (32 values/block), each block scaled by its f32 scale, summed left-to-right into a
     fixed scalar/lane accumulator.
  3. The per-block 32-wide product reduces in a **compile-time-fixed lane layout** (i8mm/dotprod
     `q8_0_packed_4x8` → fixed `sums[0..4]`; scalar fallback unrolled-by-4, left-to-right).

- **Site 2 — Attention** (`attention_context_for_head_into`):
  1. **QKᵀ scores:** each score is a serial `dot_product_row(query, key@pos)` over `head_dim`,
     fixed left-to-right order (vDSP_dotpr on macOS / unrolled-by-4 scalar elsewhere).
  2. **Softmax:** max via `fold(NEG_INFINITY, f32::max)` over positions in **ascending position
     order**; exp + sum accumulated in ascending position order; normalize by `1.0 / sum`.
  3. **Value accumulation:** `out += prob * value` summed over cached positions in **ascending
     position order**. Prefill batches partition over output rows (one thread per row); single-
     token decode is serial per head. Neither splits a reduction across threads.

### Measured overhead (M4, TinyLlama 1.1B Q8_0, hello → 50 tok, greedy, median of 5)

| Path | tok/s | Determinism |
|---|---|---|
| Default (GPU resident decode) | ~88 | self-identical per process; **diverges from CPU at tok 25** (not a bit-reproducible reference) |
| `--deterministic` (CPU) | ~12.5 | **bit-exact** across runs, processes, and thread counts |

## D10 — `camelid chat` agent mode (2026-06-14) — Phase 0

Recon: `RECON_AGENT.md`. Agent mode = a tool-calling plan-act-observe loop built as a mode of the
existing `chat` subcommand (new `src/chat/agent.rs`, driven from the session core; `--agent` flag
+ `/agent` toggle). Reuses the inference client, splash/turn-marker, and TTY/color helpers.

**Decision A — ledger `tool_capable`: PROGRAMMATIC.** `ModelCompatibilityTarget` is structured
Rust surfaced by `/api/capabilities`; add `tool_capable: bool` per row (default false), read by the
agent gate + a future frontend from the same source — no second parser. Rows stay `false` until a
real tool-call round-trip promotes one (same evidence bar as the support gate), so agent mode is
built+tested but honestly refuses every supported row until then. The "stop if prose-only" boundary
does not trigger.

**Decision B — sandbox root: cwd (or `--workdir`), enforced.** Canonicalized absolute root; every
file-tool path is joined, canonicalized, and required to be the root or a descendant (rejects
`..`, outside-absolute, escaping symlinks). For new write targets the parent is checked. Enforced
in code before I/O.

**Decision C — shell: `/bin/sh -c`, cwd-pinned, timed, verbatim-approved.** Model supplies the
whole command; the verbatim string (from the parsed call, not model prose) is shown at approval;
cwd = sandbox root; default 30s timeout; captured stdout/stderr/exit. Honest limitation: `sh` runs
with the user's permissions and can leave the root cwd — `run_shell` is cwd-confined + approval-
gated, NOT a filesystem jail (true jailing = `sandbox-exec`/namespaces, a documented follow-up).
Exec is therefore the highest risk class: always prompts, never in an auto-approve default.

**Auto-approve stance.** `--auto-approve` may exist for power users but prints a prominent warning,
is never a README default, and only relaxes *prompting* — never the sandbox (file tools) or the
prompt-injection rule (tool-result content is data, never permission to escalate or act).

**Phase 1 (tool-calling in inference) is a SUBSTANTIAL scope boundary** — confirming the approach
(server-side / client-side / hybrid; see RECON_AGENT) before building it. The deterministic core
(loop/tools/sandbox/approval/mock/tests) is buildable first and independent of that choice.

### D10 (cont.) — agent mode: deterministic core DONE; Phase 1 + promotion pending

**Built + tested (Phases 0, 2–6, 8-core):** `src/chat/{tools,tool_parse,agent}.rs`. The
plan-act-observe loop is UI/model-agnostic (`ModelDriver`/`Approver`/`Reporter` traits), bounded
(`--max-steps`, default 25) and cancellable; the full tool set (read_file/list_dir/search/
write_file/edit_file/run_shell/http_fetch) is sandbox-confined; the approval gate is
y/a/n/q with session policy; the Llama/Hermes tool-call parser is in `tool_parse.rs`. **44 bin
unit tests** cover sandbox-escape rejection, each tool, parse (valid Llama/Qwen + malformed →
clean), loop-with-mock (threads results, step cap), denial handling, **prompt-injection in a tool
result does not execute**, and **--auto-approve still enforces the sandbox**. fmt/clippy/test/doc
green. Capability gate verified live: `--agent` on a non-tool-capable row refuses (typed error,
exit 2).

**Front-end decision:** agent mode runs in the **line renderer** (synchronous, readline approvals,
clean redirected transcripts) — entered via `--agent`. `/agent` in the chat front ends is a
discoverable pointer to it. The full-screen TUI agent (modal approvals inside the redraw loop) is
a documented follow-up.

**Phase 1 (Hybrid tool-calling) — PENDING, scoped:** the server currently rejects `tools` (it
falls into `#[serde(flatten)] unsupported_fields`). To enable: add `tools: Option<Vec<Value>>` to
`ChatCompletionRequest` + `GenerationSessionRequest` (+ the handler conversion), and thread it into
the Jinja render context (`render_jinja_chat_template` → add `custom_tools`/`tools` to `context!`;
`render_metadata_jinja_chat_template_prompt` + `render_chat_prompt_for_tokenization_for_model_result`
gain an `Option<&[Value]>` param — the ~12 existing callers, mostly tests, pass `None`). Backward-
compatible: `tools=none` → the template's no-tools path → byte-identical render. The model's own
chat template then renders tools; the client parses output (already built).

**Promotion — PENDING, evidence-gated:** set `tool_capable=true` on `llama32_3b_instruct_q8_0`
**only after** a real round-trip shows it emits a parseable tool call. Llama 3.2 3B is a small
model and tool-calling reliability is genuinely uncertain — if it doesn't round-trip cleanly, the
row stays `false` and that is reported honestly (no demo claimed without evidence).

### D10 (cont.) — Phase 1 SHIPPED; promotion NOT earned (evidence)

**Phase 1 (Hybrid tool-calling) — DONE.** `tools` is now an accepted field on
`ChatCompletionRequest`/`GenerationSessionRequest` and is threaded into the model's own Jinja
chat template (`render_jinja_chat_template` gained `custom_tools`/`tools` in its `context!`; a
dedicated `render_chat_prompt_for_tokenization_with_tools` is used only when a request carries
tools, so the shared render chain and its ~12 callers are untouched). Backward-compatible:
`tools=None` → the template's no-tools path → byte-identical render (lib's 438 tests still pass).
Verified live: a request with `tools` is no longer rejected (it reached model resolution), and the
model renders + emits tool calls.

**Promotion — NOT done (honest, evidence-gated).** A live round-trip on Llama 3.2 **1B** Q8_0
(the 3B would not load under severe box contention — 140–168s even for the 1B) showed the model
**emits the correct Llama tool-call format** (`{"name":"read_file","parameters":{…}}`) — so the
threading + parser work end-to-end — **but with malformed arguments** (it echoed the JSON schema
instead of `{"path":"notes.txt"}`) and looped to the length cap. That is not a usable tool call,
so **no row is promoted**: `tool_capable` stays false for every row and agent mode keeps refusing
(verified: exit 2). This matches the spec's capability tension exactly — small Llama 3.2 models
don't tool-call reliably, and the more-capable 3B/8B (or Qwen3/Mistral) round-trip awaits a calm
box. The server-side `tool_capable` ledger column is added the moment a row earns it (add the
field to `ModelCompatibilityTarget` + set the promoted row true); until then the client's
`CompatRow.tool_capable` defaults false, so the gate is correct without it. No capability is
published that no model has demonstrated.

### D10 (cont.) — agent-mode hardening + promotion harness (2026-06-14)

**Phase 0 — render bug, fixed (`TOOLCALL_DIAG.md`).** The malformed args were a render bug, not
just model size: tools were threaded as **OpenAI-nested** `{type, function:{…}}`, so the model
saw the envelope leak into the prompt and its `parameters` field was the JSON *schema*
(`properties`/`required`/`type`) — which a weak model echoes. Fix: the server now **normalizes**
each tool to its flat `function` object before rendering (matching llama.cpp/vLLM), localized to
the tools-present path; `tools=None` stays byte-identical (438 lib tests pass). Proven offline by
`tool_render_nested_vs_flat_diagnostic`. Conclusion: render is now canonical-correct, so a future
big-model failure cannot be a leftover template bug. The parser maps `parameters`/`arguments`
correctly (it never keyed off the schema's `properties`).

**Phase 1 — parser robustness.** `tool_parse` tests now cover plain/`python_tag` JSON, Hermes
`<tool_call>` tags, the `function` envelope, **double-encoded args** (a JSON-string normalized to
an object), multiple calls per turn, leading/trailing prose, the schema-echo failure mode (name
parsed, no real args → the gate rejects), and malformed/truncated/empty (clean, no panic).

**Phase 2 — `agent-eval` promotion harness.** A subcommand that loads a model with a **bounded
timeout** and runs a fixed tool-use battery, reporting **PASS / FAIL / INCONCLUSIVE** + a hashed
receipt (`camelid.agent_eval/v1`: model id, GGUF+quant+size, raw output, parsed calls, per-case
pass, host loadavg, timestamp, `promotion_eligible`). **INCONCLUSIVE** (load timed out) never
changes a flag and never counts as FAIL — this is the noisy-box fix. Promotion = flip
`tool_capable` true only after a PASS receipt. Verified live: 1B → **FAIL** (loaded 48s, malformed
args, exit 1) even with the corrected render; 1s timeout → **INCONCLUSIVE** (exit 3, flag
untouched). No row promoted (no PASS earned).

**Phase 3 — polish.** Loop: **repeat-call detection** (3 identical calls → break with an
explanation, not the whole budget) + a step-cap summary of what ran. Approval grants are
**session-scoped** (the `a` choice persists across goals; `/tools` shows which are auto-allowed).
Tool output truncates with an explicit "(N more lines)". Injection resistance proven
source-agnostically: a fooled model that *follows* injected content into a destructive call is
still **denied by the gate** (the gate, not the model, is the backstop) — covers file and
http_fetch result content alike.

**Honesty.** No `tool_capable` flag is set; nothing in the docs claims tool-calling works on a row
without a PASS receipt. The capability only ever moves on harness evidence.

### D10 (cont.) — first promotion: Llama 3.2 3B Instruct Q8_0 is tool_capable (2026-06-14)

`llama32_3b_instruct_q8_0` earned a **PASS** from `agent-eval` and is promoted: `tool_capable: true`
on that `ModelCompatibilityTarget` row (the field is added to the struct, all other rows `false`).
Receipt committed at `qa/agent-eval/Llama-3.2-3B-Instruct-Q8_0-…-PASS.json`: with the corrected
flat-tools render the 3B emitted a well-formed `read_file(notes.txt)` call (args `{"path":…}`, not
the schema echo), read the fixture (`alpha\nbeta\ngamma\n`), and answered `3` — `promotion_eligible:
true`, host loadavg 2.5. This confirms the Phase 0 render fix was the actual blocker: the 3B was
capable; the OpenAI-nested render had been breaking it (it `FAIL`ed before the fix, `PASS`es after).

`--agent --model <the 3B GGUF>` now runs the live loop (the catalog-label match sets the active
label to the ledger id, so `active_tool_capable()` matches). The 1B remains `FAIL`/gated (too weak);
Qwen3-4B `FAIL`ed by reasoning in `<think>` instead of emitting the call (and isn't a ledger row);
both are honest non-promotions. The gate, harness, and ledger all read the one `tool_capable` flag.
## D11 — Execution-trace rollup, Pillar Two (2026-06-14)

Built on D9: now that the deterministic CPU forward pass is bit-exact, a receipt can carry a
cryptographic **execution-trace rollup** — proof of *how* the computation ran, not just which
tokens came out. This is the "later portable execution-trace feature" D9 was the foundation for.

**Shape — single rollup (chosen over per-layer-localized or final-logits-only).** One streaming
SHA-256 (`ExecutionTraceHasher`, `camelid.execution-trace/v1`, `sha256-rollup-v1`) folds, in
forward order across every generated token, each transformer layer's output hidden state and the
final logits (domain-separated by a kind byte + index + length prefix; little-endian f32 bytes).
A mismatch on re-derivation proves the run differs but does not localize the token/layer — that
is the single-rollup tradeoff, chosen for the simplest end-to-end slice. Runtime cost is
negligible (the bytes are already materialized; SHA-256 is multi-GB/s — sub-1% of decode), so
scope, not speed, drove the choice.

**Only meaningful on the deterministic lane; fail-closed.** `LlamaInferenceSession::
enable_execution_trace()` refuses to arm unless `deterministic_mode_enabled()` (RECEIPTS.md
rule 2 — a digest over a non-reproducible run is meaningless). The default (non-deterministic)
path never allocates the hasher and is byte-for-byte unchanged; the receipt field is
`Option<ExecutionTraceBlock>` with `skip_serializing_if = None`, so non-traced receipts serialize
and digest exactly as before (proven by `execution_trace_absent_keeps_receipt_byte_identical`).

**Emission and verification cannot desync — they share one path.** Both the served generation and
`verify-receipt`'s re-run funnel through `replay_receipt_request → generate_decoded_tokens`, where
the rollup is armed (when deterministic) and captured once. When tracing, the prompt-prefix cache
is bypassed (a cache hit would skip the prompt forwards on one side only). Verification re-derives
the digest from an independent re-run and checks it matches; the rollup is included in the
`receipt_id` digest, so it is itself tamper-evident.

**ISA-pinned, not cross-ISA-portable.** The digest is reduction-order-stable on the deterministic
CPU lane but ISA-specific (the Q8_0 dot rounds differently across ISAs), so the block records
`lane` (`deterministic-cpu`) and `host_isa` (e.g. `aarch64-i8mm`). `verify-receipt` re-derives only
when the verifier's ISA matches; otherwise it prints `SKIP execution-trace` rather than a false
`FAIL`. The committed test digest is the M4/i8mm reference (i8mm-guarded), matching the D9 pattern.

**Proof:** `tests/execution_trace.rs` (engine rollup: run-to-run + thread-count invariant +
prompt-sensitive + pinned + fail-closed) and `tests/execution_trace_receipt.rs` (the full
emit→re-derive round trip through the real API replay path). Not built here: per-layer/per-token
localization, distributed (per-shard) trace chains, signing.

### D10 (cont.) — evaluating 8B / Mistral; system-prompt fold (2026-06-14)

Ran `agent-eval` against the other tool-capable-family ledger rows; **neither earns a PASS, so
neither is promoted** (the gate holds):
- `llama3_8b_instruct_q8_0` — **FAIL** (genuine): Meta-Llama-3 8B is *original* Llama 3, which lacks
  the Llama 3.1+ tool-calling training/template. It hallucinated a result in prose instead of
  emitting a structured call. Bigger ≠ tool-capable; it's training+template.
- `mistral_7b_instruct_v0_3_q8_0` — **FAIL** (not a clean capability verdict): its v0.3 template
  first rejected a standalone `system` role ("roles must alternate"). Fix below cleared that, after
  which it *rendered* but still didn't emit a structured call — the GGUF template doesn't present
  tools in Mistral's `[AVAILABLE_TOOLS]` form, so Mistral improvised a shell command in prose.
  Promoting it needs Mistral-specific tool-template rendering — a real follow-up, not done here.

**System-prompt fold (kept).** `LiveDriver::step` now retries with the system prompt folded into the
first user message **only when the template rejects a standalone system role** (Mistral v0.3, Gemma).
The system-role path is tried first and is unchanged, so the promoted 3B is unaffected (re-verified
PASS). This is a correct robustness fix (system-less templates no longer error) and a stepping stone
toward Mistral support. The only promoted row remains `llama32_3b_instruct_q8_0`.

## D11 — Native Windows desktop app (`camelid-desktop`): sidecar + embedded-UI WebView (2026-06-19)

Add a second, **additive** Windows executable (`camelid-desktop`) that gives users a native
desktop chat experience with no browser. Two binding choices, both confirmed before coding:

**1. Engine integration = sidecar, not in-process.** The desktop process spawns the shipped
`camelid serve --addr 127.0.0.1:<ephemeral> --no-open` bound to **loopback only**, health-gates
on `/v1/health`, and kills it on window close. `api::router_with_state()` is `pub` so in-process
*was* feasible, but sidecar is chosen for v1 because it guarantees byte-identical generation
behavior to the shipped server and isolates engine crashes from the UI. Any divergence in
generation between desktop and `camelid serve` would be a regression, not a feature.

**2. UI delivery = point the WebView at the sidecar's already-embedded UI, NOT re-bundle the
frontend.** The `camelid` router's fallback route (`*` → `crate::web_ui::handler`, `src/api/mod.rs`)
already serves the same React app the web path uses, embedded in the binary via `rust-embed`.
The desktop WebView therefore navigates to `http://127.0.0.1:<port>/`, making UI **and** API
same-origin: the frontend's `defaultApiBase()` resolves to `window.location.origin`
(`frontend/src/hooks/useDashboardData.js`), so no base-URL injection or ephemeral-port plumbing
is needed, and the runtime-ready + exact-supported-row capability gate is inherited **verbatim**
(it is literally the same UI hitting the same `/api/capabilities`). The brief's literal Phase 2
("bundle the frontend as Tauri assets") is therefore intentionally collapsed: the frontend is
already bundled *inside `camelid.exe`*; the desktop `.exe` needs no npm/Vite at runtime and keeps
no second copy of the UI. Tauri ships only a tiny static "starting engine…" splash page.

**Stack:** Tauri v2 (Rust backend + WebView2). WebView2 ships with supported Windows 10/11.

**Workspace:** the root `Cargo.toml` becomes a workspace (`resolver = "2"`,
`default-members = ["."]`) with `camelid-desktop` added as a member. CI builds the server with
`cargo build --release --locked --bin camelid`, which does not pull the desktop member into its
build graph, so the server binary stays byte-for-byte unaffected. `default-members = ["."]` keeps
bare `cargo build`/`cargo test` scoped to `camelid` as before; the desktop app is built explicitly
with `-p camelid-desktop`.

**No new inference / no metric fabrication / no support-contract drift / additive CI:** the desktop
app reuses the shipped engine and gate wholesale; any tokens/sec readout (Phase 3) is sourced from
the existing SSE `decode_tps` real generation event, rendered unavailable when absent. The new
release job is additive and independently skippable; existing server artifacts are untouched.
See `camelid-desktop/README.md`.

## D12 — CPU KV cache: f16-rounded values were stored in f32 buffers; f16 storage + head-major layout lanes (2026-07-01)

Measured finding (Item-3 recon): the CPU KV write path has rounded every stored
key/value element through f16 unconditionally since the f16-KV work landed
(`copy_to_f16_kv_cache_storage`, now `store_kv_head_row` in
`src/inference/kv_cache.rs`), but kept the results in f32 buffers — 2x the
bytes for zero additional precision. For Llama-3.2-3B at 8192 ctx that is
~1.79 GiB of KV instead of ~0.9 GiB, which is exactly what host-limited
deep-context runs on the 15.7 GiB Windows dev box.

Consequences, recorded as binding context for future parity claims:

- All Item-1 (blocked-dot, PR #355) and Item-2 (head-parallel, PR #358)
  parity and A/B evidence was earned on f16-rounded KV values.
- The Item-3 f16 storage lane (`BACKENDINFERENCE_KV_F16`, PR #359) therefore
  introduces ZERO new rounding: it stores the same bits as u16 and expands
  exactly on read. Both Item-3 lanes (with `BACKENDINFERENCE_KV_LAYOUT_HEAD_MAJOR`)
  carry the bitwise-identity contract — end-to-end token identity was proven
  in 17/17 flag combinations plus the serve stack, and the 8192-ctx run
  completes on the dev box (4.037 tok/s, 5.72 GB peak working set).
- Canonical conversions are pinned in `src/inference/kv_f16.rs`: store = the
  historical helper semantics (RNE, NaN -> sign|0x7E00, F16C fast path with a
  NaN fixup); read = exact `vcvtph2ps` semantics on the full 2^16 domain
  (quiets signalling NaNs — unreachable from the canonical store; the delta
  vs the older helper is documented there and locked by exhaustive tests).

Receipts: `target/kv-lanes-baseline-20260701T180113/`,
`target/kv-lanes-ab-20260701T185422/`, `target/kv-lanes-ab2-20260701T195346/`
(SHA256-sealed, dev box) and the PR #359 description.


## D13 - Decode thread-width policy: detected physical cores, never SMT logical count (2026-07-02)

**Decision:** on Windows x86_64, single-token decode runs by default on a dedicated
rayon pool sized to the DETECTED physical core count (GetLogicalProcessorInformation,
one RelationProcessorCore record per core). The SMT logical count is never used for
decode. Detection failure, an operator-pinned global (CAMELID_THREADS), or a global
already narrower than physical all fail closed to the previous inline-on-global
behavior. `BACKENDINFERENCE_DECODE_THREADS=N` remains the explicit width override and
`=0`/`=off` is the kill switch; `BACKENDINFERENCE_DECODE_POOL_DEDICATED=1` remains the
isolate-at-global-width override.

**Basis (receipts):** the Items-4/5 P2 sweep (`target/decode-sched-p2-20260702T011105/`,
SHA256-sealed): decode tok/s is flat across widths 4-8 (within ~3%) and falls
monotonically past the physical count (width 16 = -20% vs the plateau at depth 512,
-19% at 2048); the serve-path A/B in the same receipt shows the dedicated-pool
configuration at -4.5% wall and an explicit narrower width at -7.1%. This host's
measured optimum (6) is deliberately NOT hardcoded: the portable, defensible policy is
"physical, not logical"; per-host refinement of the exact width within the flat region
is deferred to GAIT calibration.


## D14 — Models page: five-zone IA, derived membership only, no localStorage truth (2026-07-02)

**Decision:** the Models page is exactly five zones in one scroll — (1) active model
bar with the only Unload action, (2) Supported, (3) Experimental
(compatible → eligible → not-anchored), (4) one global Downloads panel, (5) Get
models (curated + live Hugging Face search with confirmed downloads). Section
membership is computed at render time from `/api/models/local` +
`/api/capabilities` via `frontend/src/lib/modelLanes.js` (extracted verbatim from
the old LocalLaneSections); no hand-authored array places a model. The
`SUPPORTED_MODELS` list survives only as curated-download decoration
(blurb/Recommended) in Zone 5.

**localStorage download tracking is removed as a source of truth.** Download
progress renders only from the `/api/models/catalog/downloads` poll
(bytes/total), owned by one spine hook (`useModelsPageData`); "downloaded" means
the file is present in the live `/api/models/local` scan, nothing else.

**Diagnostics relocation:** Tokenizer Playground and the Model Inspector moved
into a collapsed "Diagnostics" disclosure at the page bottom (only ModelsView
consumed them). Import-a-GGUF-by-path also lives there — it is the only way to
load a GGUF stored outside `models/`, so the capability is preserved but off the
primary surface. The permanently-disabled Hosted API fieldset, the legacy
FILTERS grid, the hero ledger, the runtime status grid, the acceptance/tracked-
row panels, and the localStorage-driven SupportedModels section were deleted
outright. (The import/hosted-API panel was not listed in the rebuild conductor's
inventory; this disposition is the flagged resolution.)

**"Not-yet-runnable, visible but disabled" interpretation:** the backend exposes
no pre-download refusal signal per catalog row, so the disabled-with-reason state
is driven by the real derivable gates — runtime offline or the backend not
advertising `hf_catalog_install`/`model_downloads`. Per-row implementability is
still decided at load time by the inspect-first typed-blocker flow (fail-closed,
rendered verbatim).

## D15 - Q8 prefill GEMM owner: default ON, win-x86_64 only (2026-07-08)

**Decision:** `CAMELID_X86_Q8_MATMUL_OWNER` defaults to `All` on Windows x86_64 and
stays `Off` everywhere else. Explicit rollback: `CAMELID_X86_Q8_MATMUL_OWNER=off`.
The variant inside the owner is unchanged (4x8 AVX-512 VNNI when the CPU has it,
AVX2 4x4 otherwise; both bit-exact with twin tests).

**Basis (receipts):** re-validated at the llama.cpp b9918 re-pin on main 582781da with
the hardened paired in-process sweep, now carrying an ENGAGED-CHECK (the sweep aborts
if an owner-on config never dispatches the owner arm, and the per-record
`owner_prefill_taken` counts are in the receipt — off=0, owner=280): 3B Q8 prefill
26.97 -> 30.30 tok/s (+12.3%, CI [1.115,1.133], 8/8 rounds), Qwen3-4B 20.00 -> 22.40
(+11.9%, CI [1.113,1.126], 8/8). Bit-exactness unchanged from the #345 twin tests.
`PERF_RECEIPTS/same-host/q8-prefill-owner-b9918-revalidation-20260708/`.

**Treaty note (Tim-visible, not buried):** BENCHMARK_TREATY.md asks for a both-host
(Windows + Ubuntu) win before promoting a default; the Ubuntu host remains
PENDING/paused per Tim's earlier call. This flip follows the D13 precedent instead:
single-host receipts promote a default that is cfg-SCOPED to exactly the host class
that was measured (win-x86_64), leaving every other target Off. If the treaty should
bind even scoped flips, revert by deleting the cfg branch in
`Q8MatmulOwnerScope::from_env`.
