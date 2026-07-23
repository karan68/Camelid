# W1 receipt — `run_shell` now drains its pipes

**Finding:** W1 · `src/chat/tools.rs` `run_shell` · pipes never drained → child emitting >64 KiB
wedges in `write()`, is false-timed-out, output discarded.
**Fix:** lift the per-pipe reader-thread pattern from `run_windows_command` into `run_shell`; join
after normal exit and on the kill path so no thread leaks.
**Host:** Windows 11 · rustc 1.95.0 · `campaign/hardpan-phase0`

## Before (Phase 0 probe, real `run_shell` via validate()+execute())

```
[W1] payload_bytes         = 417792
[W1] shell_timeout_secs    = 10
[W1] elapsed_ms            = 10075        <- full deadline burned
[W1] outcome              = ToolOutcome::Err
[W1] returned_text        = "command timed out after 10s"  (27 bytes)
[W1] captured_any_payload = false         <- every byte discarded
control (32,640 B, under the buffer): Ok in 87 ms, payload captured
```

## After (asserting regression tests)

```
running 2 tests
test chat::tools::tests::run_shell_drains_more_than_a_pipe_buffer ... ok
test chat::tools::tests::run_shell_drains_more_than_a_pipe_buffer_on_stderr ... ok
test result: ok. 2 passed; 0 failed; finished in 0.08s
```

413,696 B now returns `ToolOutcome::Ok` with the payload intact, in <100 ms.

## Negative control — the regression tests have teeth

With the drain reverted (reader handles forced to `None`, pipes never read) and nothing else changed:

```
[W1]        payload=413696 elapsed_ms=15023 outcome=ToolOutcome::Err text_bytes=27
            panicked: a command emitting 413696 bytes must succeed, got Err("command timed out after 15s")
[W1-stderr] elapsed_ms=15042 outcome=ToolOutcome::Err text_bytes=27
            panicked: stderr past one pipe buffer must not wedge the child
test result: FAILED. 0 passed; 2 failed; finished in 15.05s
```

The stderr leg re-wedges independently, confirming the 64 KiB quota is **per pipe** — both readers
are load-bearing.

## Notes
- The Phase 0 `#[ignore]` probes `hardpan_w1_*` were replaced by these two asserting tests; the
  helpers (`oversized_payload`, `run_shell_cat`) are cross-platform, so the drain is exercised on
  Linux CI too (where the same wedge exists — W1 is not Windows-specific, per amendment A3).
- Gates: `cargo fmt` clean · `clippy --all-targets --all-features -D warnings` clean · full lib
  suite green.
