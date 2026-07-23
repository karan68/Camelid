# HARDPAN GATE 3 — manual checklist (CERT)

**Scope:** Phase 3 terminal fixes — W6 (teardown guard), W5 (paste), W8 (AltGr). W7 was struck at
GATE 0; there is no `/copy` row.
**Instrument:** `qa/hardpan/input-probe/` (now mirroring the POST-fix predicates) plus the driver
scripts described per row. CERT rows marked ☐ are run by Tim on the Windows box against the real
`camelid` binary; rows marked ✅ AUTOMATED carry committed instrument receipts.

> **Merge-ahead note (2026-07-23).** Tim authorized finishing and merging without a further
> sign-off pause ("don't ask my permission — finish this and merge it"). The automated instrument
> rows below were captured before merge; the remaining ☐ rows are owed as post-merge CERT, at
> Tim's convenience, against the merged binary. Any red row reopens its finding — merged is not
> sealed.

| # | Row | Status | Evidence |
|---|-----|--------|----------|
| 1 | Multi-line paste creates **exactly one** goal (terminal-injected paste; interior Enters become text) | ✅ AUTOMATED (instrument) + ☐ CERT real TUI | [`phase3/w5-paste-postfix.txt`](phase3/w5-paste-postfix.txt): same 3-line clipboard as Phase 0 → `goals_fired = 0`, `pasted_newlines = 2`, 204 chars inserted. Phase 0 pre-fix: 2 goals. |
| 2 | `Ctrl+V` inserts clipboard text (hosts that pass the chord through) | ✅ AUTOMATED (module) + ☐ CERT real TUI | `gate3_clipboard_read_returns_real_content` — round-trips `HARDPAN-GATE3-CLIPBOARD-é€-multi\r\nline` byte-exact through the real Win32 clipboard, incl. non-ASCII + CRLF. |
| 3 | AltGr characters reach the input box on a non-US layout | ✅ AUTOMATED (instrument) + ☐ CERT real TUI | [`phase3/w8-altgr-postfix.txt`](phase3/w8-altgr-postfix.txt): German layout, AltGr `@ € [ ]` → 4/4 `ALTGR-INSERT`, 0 dropped. Phase 0 pre-fix: 4/4 dropped. Layout restored to `en-US` (verified). Plus the pure-predicate unit test over every modifier shape. |
| 4 | A deliberately panicking build restores the console (no wedged raw-mode window) | ✅ AUTOMATED (unit) + ☐ CERT real TUI | `panic_hook_chains_to_previous` (chained hook still prints the payload) + `guard_restores_once_and_only_when_armed`; the guard's restore sequence is visible in the test log (`[?1049l[?2004l`). Real-TUI leg: run `camelid agent`, force a panic on the TUI thread (e.g. a debug-injected `panic!` build), confirm the prompt comes back cooked with the payload visible. |
| 5 | Mouse scroll and `Tab` sidebar still work | ☐ CERT real TUI | Untouched code paths (`Event::Mouse` arm, `KeyCode::Tab` arm precede the changed guard); no automated instrument. |
| 6 | `--plain` and `camelid agent exec` unchanged | ☐ CERT real TUI | Phase 3 touched only `agent_tui.rs`/`tui.rs` full-screen paths + shared modules; the plain/exec surfaces don't enter raw mode. |
| 7 | `/copy` — **struck** | — | W7 struck at GATE 0: OSC 52 writes are honoured on this host (receipts in REPRO.md); `clipboard.rs` untouched. |

## How to re-run the automated rows

```bash
# Row 1 + 3 (real console required; drivers add/remove the German layout safely):
cargo run --release --manifest-path qa/hardpan/input-probe/Cargo.toml -- probe.log 25
# Row 2:
cargo test --release --lib -- --ignored --nocapture gate3_clipboard
# Row 4:
cargo test --release --lib -- term_guard
```

The Phase 0 (pre-fix) counterparts of rows 1 and 3 are recorded in `REPRO.md` — same drivers, same
clipboard, same chords, opposite outcomes.
