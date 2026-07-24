# W2 receipt — `run_shell` timeout now tears down the whole process tree

**Finding:** W2 · `src/chat/tools.rs` `run_shell` · no job object → a timeout's `child.kill()` reaps
only `cmd.exe`; every descendant (rustc, node, a CUDA process) survives as an orphan.
**Scope:** job object only. The `CREATE_NO_WINDOW` half was **dropped at GATE 0** (A6) — `run_shell`
is not registered on any console-less surface, so the flag would be pure regression risk.
**Fix:** assign the child to a kill-on-close `win_job::JobObject` right after spawn (behind
`#[cfg(windows)]`), and `terminate()` it on the timeout path before `child.kill()`. Mirrors
`run_windows_command`. Unix path untouched.
**Host:** Windows 11 · rustc 1.95.0 · `campaign/hardpan-phase0`

## Before (Phase 0 probe)

```
[W2] shell_timeout_secs       = 5
[W2] outcome                  = ToolOutcome::Err
[W2] orphaned_grandchild_pids = [16080]
[W2] orphan_count             = 1        <- ping survived the timeout
```

## After (asserting regression test)

```
running 1 test
test chat::tools::tests::run_shell_timeout_tears_down_the_process_tree ... ok
test result: ok. 1 passed; finished in 4.76s
[W2] orphaned_grandchild_pids = []
```

`run_shell_timeout_tears_down_the_process_tree` runs `ping -n 271 127.0.0.1` (271 s of work) under a
3 s timeout and asserts no `ping` with that unique count survives. Orphans are attributed by
**command line**, not image name, so the test cannot flake on an unrelated `ping`, and cleanup targets
exactly what it created (by PID, never a blanket image kill — the box also runs a desktop sidecar).

## Negative control — and a coupling worth recording

Reverting **just** the job object (`_job = None`, nothing else changed) did not merely resurrect the
orphan — it made the same test run for **274.79 s** instead of 4.76 s:

```
test ... has been running for over 60 seconds
test ... ok
test result: ok. 1 passed; finished in 274.79s     <- ~271 s = the ping's full lifetime
```

**Why:** W1's drain threads `read_to_end` the child's pipes, and the orphaned grandchild *inherited
the write end*. On the timeout path `run_shell` joins those threads, and `read_to_end` cannot return
EOF until every write end closes — which, with the grandchild orphaned, is only when the ping exits
on its own 271 s later. So without the job object the orphan **both survives and wedges teardown**;
`run_shell` hangs for the grandchild's entire lifetime.

The job object's `terminate()` closes all inherited write ends at once, so the drain threads EOF
immediately and the join is prompt — hence 4.76 s vs 274.79 s (a **57× difference** on the same
test). This is why W2 lands directly after W1: W1's drain makes prompt tree-teardown load-bearing,
not just tidy. (`run_windows_command` has carried both halves together since it was written, for the
same reason.)

The negative control left no leak: the test kills any survivor by PID before asserting, and a
post-run sweep for `ping -n 271` found nothing.

## Notes
- Best-effort, exactly like `run_windows_command`: if `JobObject::new()`/`assign()` fails, the
  `child.kill()` backstop still reaps the direct child (its descendants may then escape — the same
  accepted risk the sibling documents).
- Gates: `cargo fmt` clean · `clippy --all-targets --all-features -D warnings` clean · all 7
  `run_shell` tests green in 4.68 s.
