# GATE 2 — PowerShell encoding and exit codes both provably correct

**Gate:** Phase 2's exit criterion. (1) `run_windows_command` with a non-ASCII command and non-ASCII
output round-trips byte-identical; (2) a deliberately failing native command returns
`ToolOutcome::Err` with the child's real exit code. Both as `#[cfg(windows)]` unit tests beside the
existing `run_windows_command` tests.
**Finding:** W3(a) encoding + W3(b) exit codes · `src/chat/tools.rs` `run_windows_command`
**Fix (one commit, shared stdin preamble):** the command rides in as base64 inside a pure-ASCII
preamble that decodes identically under any code page, sets the child's output side to UTF-8
(no-BOM), decodes and `Invoke-Expression`s the real command, then re-raises `$LASTEXITCODE`:

```
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$c = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('<b64>'))
Invoke-Expression $c
if ($LASTEXITCODE -ne $null) { exit $LASTEXITCODE }
```

Still stdin delivery — **not** `-EncodedCommand`, which would reintroduce the ~32 KiB command-line
ceiling the stdin design was chosen to avoid.
**Host:** Windows 11 · rustc 1.95.0 · PowerShell 5.1.26100.8925 · `campaign/hardpan-phase0`

## Before → after (identical commands, Phase 0 probe vs GATE 2 test)

| leg | Phase 0 (pre-fix) | Now (post-fix, asserted) |
|-----|-------------------|--------------------------|
| **(a) output** `éü—✓日€` built from codepoints in the child | `ef bf bd ef bf bd 2d ef bf bd 3f 3f` (U+FFFD ×3, `-`, `??`) | byte-identical `c3 a9 c3 bc e2 80 94 e2 9c 93 e6 97 a5 e2 82 ac` |
| **(a) input** same 6 codepoints in the command text | child received **16** codepoints (CP437 mojibake) | child receives exactly `U+00E9 U+00FC U+2014 U+2713 U+65E5 U+20AC` |
| **(b)** `cmd /c exit 3; Write-Output done` | `exit: 0` → **`Ok`** (the defect) | `exit: 3` → **`Err`**, `done` still captured |
| **(b)** `cmd /c exit 42` | `exit: 1` (flattened) | `exit: 42` (true code) |
| **(b)** `Write-Output ok` | `exit: 0` → `Ok` | unchanged — no false failures |
| **(b)** `throw 'boom'` | `exit: 1` → `Err` | unchanged |

Documented residual (asserted so a change is noticed): `$LASTEXITCODE` tracks only the **last**
native command, so `cmd /c exit 3; cmd /c exit 0` reports 0 before and after the fix.

## GATE 2 test run — PASS

```
test chat::tools::tests::base64_ascii_matches_known_vectors ............... ok
test chat::tools::tests::run_windows_command_round_trips_non_ascii ........ ok
test chat::tools::tests::run_windows_command_propagates_native_exit_codes . ok
```

And every pre-existing `run_windows_command` behavior survives the wrapper:

```
test ... run_windows_command_is_exec_and_runs_under_sandboxed_mode ... ok
test ... quoting_survives_stdin_transport ... ok      <- quoting now rides base64: fully transparent
test ... multiline_command_survives_stdin ... ok      <- IEX executes the multi-line string whole
test ... timeout_hard_kills_a_hung_command ... ok
test ... run_windows_command_cwd_escape_is_refused ... ok
test ... run_windows_command_refused_when_shell_disabled ... ok
test result: ok. 9 passed; 0 failed; finished in 5.14s
```

Full lib suite: 1140 passed, 0 failed.

## Negative evidence

No re-break flip was run for W3 (unlike W1/W2). The teeth are documented differently: the GATE 2
assertions check the exact values Phase 0 **measured as wrong on the identical commands** —
`exit: 3` where pre-fix printed `exit: 0`; 6 exact codepoints where pre-fix delivered 16; clean UTF-8
bytes where pre-fix returned `ef bf bd`. Running these assertions against the pre-fix tree is the
Phase 0 record itself (`qa/hardpan/REPRO.md`, W3 rows); they fail it by inspection.

## Decisions recorded
- **`pwsh.exe` preference: skipped, deliberately.** The interpreter stays Windows PowerShell 5.1 by
  absolute System32 path. Reasons: 5.1 ships on every Windows install, so behavior is uniform; the
  preamble makes 5.1 UTF-8-correct, so pwsh would add no capability; and the execution host has no
  pwsh installed (A8), so a preference branch would ship untested — receipts-or-it-didn't-happen
  applies to interpreter branches too.
- **Fix viability was pre-validated in Phase 0** (probe 2: FIX-1..3, EXIT-1..6) before any product
  code changed; the shipped preamble is byte-for-byte the probed one.
- **A11 (W10) filed:** `win_uia.rs::run_ps` shares W3(a)'s output leg (embedded scripts are ASCII so
  its input leg is safe; UIA output can be non-ASCII). Not fixed inline per campaign rule 5b — the
  preamble pattern transplants directly if promoted to a follow-up.

## Gates
`cargo fmt` clean · `clippy --all-targets --all-features -D warnings` clean · full lib suite
1140/1140 · the 9-test `run_windows_command` set green in 5.14 s.
