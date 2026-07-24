# HARDPAN — Windows agent-terminal parity (evidence bundle)

Campaign summary, 2026-07-23. Branch `campaign/hardpan-phase0` off `origin/main` b4e3a905.
Full reproduction table and gate rulings: `qa/hardpan/REPRO.md`. Per-phase receipts:
`qa/hardpan/phase1/` (W1, W2, W4 + GATE 1), `qa/hardpan/phase2/` (W3 + GATE 2),
`qa/hardpan/phase3/` + `qa/hardpan/MANUAL_CHECKLIST.md` (W6, W5, W8 + GATE 3).

| Finding | Exit | One-line receipt |
|---------|------|------------------|
| W1 pipe drain | **FIXED** | 417 KB: timeout/0 bytes → Ok/72 ms; negative control re-wedges |
| W2 job object | **FIXED** (console half dropped at GATE 0) | orphaned grandchild → 0 orphans; control also showed the 57× teardown-wedge coupling with W1 |
| W3 encoding + exit codes | **FIXED** | 6 codepoints in = 6 out (was 16); `exit 3` then success → Err/3 (was Ok/0) |
| W4 bare `cmd` | **FIXED** (hardening null — no security delta claimed) | `system32("cmd.exe")`, quoting decision recorded |
| W5 paste | **FIXED** (+A12 deviation: pasted-Enter guard) | 3-line paste: 2 goals → 0 goals, 2 newlines absorbed; clipboard read round-trips é€ + CRLF |
| W6 teardown guard | **FIXED** | RAII guard + chaining hook, both TUIs; hook chain proven by test |
| W7 `/copy` OSC 52 | **STRUCK at GATE 0** | did not reproduce — OSC 52 honoured in 2/2 real-console runs |
| W8 AltGr | **FIXED** | German layout `@ € [ ]`: 4/4 dropped → 4/4 inserted |
| W9 LineWriter | **NULL** (this bundle) | 8× the writes = 1.07× the time; ratatui writes diffs and already flushes once per frame |

`hardpan-w9-null-speed-receipt.json` is the W9 measurement record — a passing outcome per the
conductor ("a null result is a passing outcome") and the SIROCCO/STAMPEDE precedent.

Remaining owed: the ☐ CERT rows of `qa/hardpan/MANUAL_CHECKLIST.md` (real-TUI legs), authorized
by Tim to land post-merge. Amendment log (A1–A12, including the `win_uia.rs` W10 follow-up) lives
at the foot of `qa/hardpan/REPRO.md`.
