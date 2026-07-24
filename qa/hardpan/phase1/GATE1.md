# GATE 1 — `run_shell` survives a real build

**Gate:** Phase 1's exit criterion. A real `cargo build` on a cold target dir, driven through the
**real** `run_shell` (`validate()` + `execute()`) on the Windows box, returns `ToolOutcome::Ok` with
its output intact and inside the timeout.
**Host:** Windows 11 · rustc 1.95.0 · `campaign/hardpan-phase0` @ W1+W2+W4 applied
**Fixes under test:** W1 (pipe drain) · W2 (job-object tree teardown) · W4 (system32 interpreter)

## The gate command

```
$ cargo build --color never 2>&1        (cwd = a fresh, cold, dependency-free crate)
```

The crate is generated with 800 unused-variable bindings so `cargo build` emits **184,123 bytes** of
real compiler output — **2.8× the 64 KiB pipe buffer** — without failing the build. This is the exact
shape of W1: a successful long-output command. Measured raw output size:

```
raw cargo build output bytes = 184123
exceeds 64 KiB pipe buffer   = True
warning line count           = 800
```

## Result — PASS

```
$ cargo build --color never 2>&1   (cwd = fresh cold crate)
outcome  = ToolOutcome::Ok
elapsed  = 461 ms  (timeout budget 180000 ms)
text_len = 16416 bytes (clipped to 16384 for the model)
--- head ---
exit: 0
stdout:
   Compiling hardpan_gate1 v0.0.0 (C:\...\.tmp0tf4Wg)
warning: unused variable: `gate1_unused_0`
 --> main.rs:2:9
   ...
```

The tool **drained all 184 KiB** (that is what averts the wedge), then `clip()` trimmed the captured
text to 16 KiB for the model — the model sees a correct, progressing build that `Ok`s in under half a
second. Full transcript: [`gate1-transcript.txt`](gate1-transcript.txt).

Driven by the `#[ignore]` gate test `gate1_real_cold_cargo_build_through_run_shell`:

```
cargo test --release --lib -- --ignored --nocapture gate1_
test ... gate1_real_cold_cargo_build_through_run_shell ... ok
```

## W1 / W2 rows — diff against Phase 0

The Phase 0 `#[ignore]` probes were replaced by asserting regression tests. Same commands, before vs
after the Phase 1 fixes:

| row | Phase 0 (pre-fix) | Now (post-fix) |
|-----|-------------------|----------------|
| **W1** `type big.txt` (413,696 B) | `Err("command timed out after 10s")`, **0 payload**, 10,075 ms | `Ok`, payload intact, **72 ms** |
| **W1-stderr** `type big.txt 1>&2` | (per-pipe wedge, same as W1) | `Ok`, payload intact, **64 ms** |
| **W2** `ping -n …` past timeout | **1 orphaned grandchild** survived | tree torn down, **0 orphans** |
| **GATE 1** real cold `cargo build` (184,123 B) | would wedge → 180 s timeout, output discarded | `Ok`, **461 ms** |

Live "now" run:

```
[W1-stderr] elapsed_ms=64 outcome=ToolOutcome::Ok text_bytes=16416
[W1]        payload=413696 elapsed_ms=72 outcome=ToolOutcome::Ok text_bytes=16416
[W2]        orphaned_grandchild_pids = []
test result: ok. 3 passed; 0 failed; finished in 4.71s
```

## Gates
- `cargo fmt` clean · `clippy --all-targets --all-features -D warnings` clean
- Full lib suite green; `run_shell` set (7 tests) green in 4.68 s
- Negative controls (per-fix): W1 reverted → both drain tests re-wedge at 15 s; W2 reverted → the
  same tree-teardown test runs 274.79 s (orphan survives *and* wedges the drain-thread join). See
  the per-finding receipts.
