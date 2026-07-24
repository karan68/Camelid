# W4 receipt — `run_shell` resolves `cmd` through System32

**Finding:** W4 · `src/chat/tools.rs:1831` used bare `Command::new("cmd")` while every other Windows
exec path resolves through `system32()`. **LOW — hardening only.**
**Fix:** `Command::new(system32("cmd.exe"))`. One line. No behavior change.
**Host:** Windows 11 · rustc 1.95.0 · `campaign/hardpan-phase0`

## This is a hardening null, not a vulnerability fix

There is no before/after behavioral delta to capture, and that is the honest receipt. std's Windows
process search (`sys/process/windows.rs`) consults, in order: child-`PATH` (only if the builder
changed `PATH` — this one does not) → application directory → **System32** → Windows dir → parent
`PATH`. **System32 precedes the parent `PATH`, and the current directory is never searched.** So bare
`"cmd"` already resolved to `%SystemRoot%\System32\cmd.exe` in every realistic case. Routing through
`system32()` makes that explicit instead of resting on a std implementation detail and on nobody ever
calling `.env("PATH", …)` on this builder — matching the discipline the neighbouring
`run_windows_command` comment already states.

Functional proof it still works: `run_shell_runs_in_root_and_captures` (`cmd /C dir /b`) and all 7
`run_shell` tests pass unchanged.

## The `cmd /C` quoting decision

**Decision: keep `cmd /C <command>`; do NOT switch `run_shell` to stdin delivery.**

The conductor asked whether `run_shell` should feed its command over stdin the way
`run_windows_command` does, to sidestep quoting. It should not:

- `run_shell`'s contract is **one shell command line**, symmetric with the `/bin/sh -c <command>`
  Unix arm directly above it. Feeding `cmd` a script over stdin changes its exit-code and echo
  semantics and would diverge the two platforms for no benefit.
- The quoting mismatch (std applies CRT-style quoting to the `command` arg; `cmd` does not use CRT
  parsing) is **not exploitable**. std only ever emits a `\` immediately before a `"`, and `"` is an
  illegal Windows filename character — so a mangled path errors out rather than escaping the cwd pin.
  Verified in Phase 0 (the W4 refutation attempt tried and failed to turn this into a cwd escape).
- `run_windows_command` chose stdin **because** PowerShell command-line quoting is genuinely lossy;
  that rationale does not transfer to `cmd /C`.

So the mismatch is a correctness footnote for pathological command strings, not a security gap, and
the fix is the interpreter-path hardening alone.

## Notes
- Gates: `cargo fmt` clean · `clippy --all-targets --all-features -D warnings` clean · all 7
  `run_shell` tests green in 4.73 s.
