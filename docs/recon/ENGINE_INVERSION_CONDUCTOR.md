# API Engine Inversion — Kill the Orphan-Decode Hazard

Mission status ledger. Brief: conductor prompt of 2026-07-09. Camelid pin `ffada00f`
(main, 2026-07-08). Comparator pin: llama.cpp `c15c5c77a426` `tools/server/`
(read-only reference; not vendored).

## Problem (one paragraph)

`AppState::generation_lock` (src/api/mod.rs) is acquired in the HTTP handler's async
frame, but the decode runs inside `tokio::task::spawn_blocking`, which cannot be
aborted. Any event that drops the handler future (client disconnect, the server's own
generation timeout) releases the lock while the blocking decode keeps running against
the shared CUDA-resident KV state; the next request then decodes concurrently with the
orphan — the exact corruption class the lock exists to prevent.

## Phase ledger

| Phase | Scope | Status | Receipts |
|---|---|---|---|
| 0 | Repro harness P0-T1/T2/T3 + R1 lane recon | **DONE — GATE 0 GO** | qa/evidence-bundles/engine-inversion-gate0-orphan-repro-20260709T134304Z-head-ffada00f/ + docs/recon/ENGINE_INVERSION_R1_LANE_RECON.md |
| 1 | Cancellation plumbing (token + deadline + guard rides with compute) | IMPLEMENTED — Gate 1 validation running | P0 tests flipped to PASS; suite/fmt/clippy/parity/perf pending |
| 2 | Engine inversion (engine worker thread, bounded queue, lock removal) | pending Gate 1 | — |
| 3 | Streaming over events (no per-token spawn_blocking) | pending Gate 2 | — |
| 4 | Re-certification (parity, receipts, perf, compat) | pending Gate 3 | — |
| 5 | Multi-slot recon memo (expected KILL/defer) | pending | — |

## Phase 0 verified source anchors (at `ffada00f`, pre-instrumentation line numbers)

- Lock + rationale: `src/api/mod.rs:105-118` (`generation_lock`, field at :112).
- Guard acquisitions before decode: `llama_server_completion` :2198,
  `completions` :6951, `chat_completions` :7332 (guard dropped when frame ends, :7347).
- Additional lock sites (Phase 2 scope): receipt-replay probe :7696, replay
  session-prep :7735 — both must route through the engine queue.
- Detaching timeout: `generate_decoded_tokens_blocking` :9020-9053 —
  `tokio::time::timeout(timeout, handle)` returns 503 on `Err(_)` and drops the
  `JoinHandle` only; the `spawn_blocking` decode (:9026) is detached, not stopped.
- Streaming guard held inside `async_stream`: `stream_completion` :10551, guard moved
  into the generator at :10574; per-token `spawn_blocking` step
  (`generate_stream_step_blocking` :9116, `StreamGenerationStepRequest` :9069).
- No cancellation anywhere: `generate_token_ids` :9537 has no per-step stop check.
- Test-sleep hook (deterministic repro lever): `generation_step_test_sleep_duration`
  :9265, env `CAMELID_TEST_GENERATION_STEP_SLEEP_MS`, `#[cfg(test)]`-gated.
- D5 (wrong invariant proven): `generation_lock_serializes_decoding` :12000 —
  verifies guards serialize, cannot see compute outliving its guard.

## Phase 0 instrumentation + tests (this branch)

- `decode_probe` (`#[cfg(test)]`, src/api/mod.rs): ACTIVE/MAX_SEEN atomics with a
  RAII guard entered at the top of both blocking decode closures (non-streaming
  decode + streaming step). Test-only; compiled out of release.
- `generation_timeout_must_not_orphan_decode` (P0-T2, deterministic): request A hits
  the server timeout (503) with the decode still sleeping on the blocking pool;
  request B acquires the freed lock and decodes. Asserts the DESIRED invariant
  (lock never free while a decode is live; max one concurrent decode) — FAILS on
  the pin; that failure is the Gate 0 receipt.
- `client_disconnect_must_not_orphan_decode` (P0-T1): handler-shaped task aborted
  mid-decode (mechanically identical to hyper dropping the handler future on TCP
  disconnect: the future is dropped at its await point, releasing the guard).
  Fidelity note: an in-process future drop IS the disconnect mechanism — hyper's
  connection task drops the service future; no TCP-layer behavior reaches the
  handler other than that drop.
- `stream_disconnect_must_not_orphan_decode_step` (P0-T3): real `stream_completion`
  response body driven by a reader task, aborted mid token step (client hangup →
  body dropped → generator + guard dropped between polls); orphan step overlaps
  request B.

All three take `test_support::env_lock`, drain the probe before asserting (so an
expected failure cannot leak an orphan or a probe underflow into sibling tests).

## Gate ledger

- GATE 0 (GO/KILL): **GO** (2026-07-09). All three triggers reproduce: each test
  fails on the pin with `orphans_at_next_acquire = 1` — the lock was acquired while
  the previous request's decode was still live on the blocking pool. Receipt:
  qa/evidence-bundles/engine-inversion-gate0-orphan-repro-20260709T134304Z-head-ffada00f/.
- R1 disposition: gemma4-Cuda and runnable-CUDA are OWN-SERIALIZED (whole-decode
  Mutex acquired inside spawn_blocking); CPU lane variants are unserialized but
  per-call isolated; dg lane on CUDA builds is SHARED-AND-UNSERIALIZED **among its
  own requests only** (process-global Engine, per-kernel-op locking) — does NOT
  touch main-engine state, so the mission proceeds; fix filed as a separate task
  outside this mission (see ENGINE_INVERSION_R1_LANE_RECON.md).
- GATE 1..4: pending.

## Phase 1 implementation record (2026-07-09)

- `GenerationCancel` (token + engine-armed deadline) and `gen_guard` added to
  `PreparedGeneration`; `CancelOnDrop` held in every generation handler frame and
  inside the SSE generator.
- `generate_decoded_tokens_blocking`: guard moved INTO the blocking closure; the
  old detaching `tokio::time::timeout` replaced by an engine-side per-step deadline
  check + unconditional `handle.await`. Returns the guard so multi-choice keeps the
  lock across all choices (unchanged coverage). `mark_healthy` no longer fires on a
  cancelled/timed-out decode (same coverage as the old timeout branch).
- `generate_token_ids`: cooperative stop check at the top of every step (covers
  speculative rounds); timeout payload byte-identical to pre-inversion
  (`generated_tokens` stays null on the non-streaming path).
- Streaming: guard rides each step via `StreamGenerationStepRequest`/
  `TimedGenerationStep`; held by the generator only BETWEEN steps.
- Replay path deliberately keeps its prior unlocked decode (pre-existing
  documented decision; Phase 2 routes it through the engine queue). NOTE: this is
  a pre-existing concurrent-decode hole (replay decode vs live decode) — recorded
  here so Phase 2 closes it explicitly.
- Behavior notes (documented, not observable by a well-behaved client): the
  timeout 503 now returns after the decode observes the deadline (≤1 step late)
  instead of racing it; a decode that completes all tokens just past the deadline
  now returns its result instead of a 503; a decode wedged INSIDE a single forward
  no longer produces a 503-with-orphan — the request waits (honest: the lock
  cannot be safely freed while compute may still touch KV state).

## Phase 1 design note (locked during Phase 0 test authoring)

Cancellation alone is not sufficient for T1/T3: on future-drop the guard frees
immediately while the (cancelled) decode still has ≤1 step to run — request B could
still overlap that tail. Guard/compute equivalence therefore requires the
`OwnedMutexGuard` to travel INTO the blocking closure (non-streaming: moved into the
decode closure; streaming: rides `StreamGenerationStepRequest`/`TimedGenerationStep`
through each step), so the lock is held by the compute itself, not the abandonable
async frame. `CancelOnDrop` then bounds the orphan tail to ≤1 step, and the lock
stays contended until that tail exits. The Phase 0 tests assert exactly this strong
invariant (`orphans_at_next_acquire == 0`).
