# HARDPAN GATE 3 — manual checklist (CERT)

**Scope:** Phase 3 terminal fixes — W6 (teardown guard), W5 (paste), W8 (AltGr). W7 was struck at
GATE 0; there is no `/copy` row.
**Instrument:** `qa/hardpan/input-probe/` (now mirroring the POST-fix predicates) plus the driver
scripts described per row. CERT rows marked ☐ are run by Tim on the Windows box against the real
`camelid` binary; rows marked ✅ AUTOMATED carry committed instrument receipts.

> **Merge-ahead note (2026-07-23).** Tim authorized finishing and merging without a further
> sign-off pause ("don't ask my permission — finish this and merge it"). PR #496 merged at main
> `4796d718` after CI green. **Post-merge CERT executed 2026-07-23 against the merged binary**
> (receipts below, `phase3/cert-real-*.txt`): every model-free row now has a real-binary or
> real-code live receipt. The only legs still owed to a human eyeball are the two that need a
> loaded model — free RAM (2.1 GB) was below this box's hard models+3 GB floor, and that rule is
> not bent — plus the mouse-scroll glance. Any red row reopens its finding.

| # | Row | Status | Evidence |
|---|-----|--------|----------|
| 1 | Multi-line paste creates **exactly one** goal (terminal-injected paste; interior Enters become text) | ✅ AUTOMATED (instrument) · ☐ agent-TUI leg **RAM-blocked** | [`phase3/w5-paste-postfix.txt`](phase3/w5-paste-postfix.txt): same 3-line clipboard as Phase 0 → `goals_fired = 0`, `pasted_newlines = 2`, 204 chars inserted. Phase 0 pre-fix: 2 goals. The agent TUI requires a loaded model (`agent mode needs a model`), and 2.1 GB free < model+3 GB floor — owed when RAM allows. |
| 2 | `Ctrl+V` inserts clipboard text (hosts that pass the chord through) | ✅ AUTOMATED (module, real clipboard) · ☐ agent-TUI leg **RAM-blocked** | `gate3_clipboard_read_returns_real_content` — round-trips `HARDPAN-GATE3-CLIPBOARD-é€-multi\r\nline` byte-exact through the real Win32 clipboard, incl. non-ASCII + CRLF. Same RAM block as row 1 for the in-TUI leg. |
| 3 | AltGr characters reach the input box on a non-US layout | ✅ **REAL BINARY** + instrument | [`phase3/cert-real-binary-run.txt`](phase3/cert-real-binary-run.txt) + [`cert-real-binary-altgr-snap.txt`](phase3/cert-real-binary-altgr-snap.txt): the merged `camelid chat` TUI, German layout, AltGr Q/E/8/9 → the input box holds the contiguous run `@€[]` (screen-buffer scrape, line `│@€[]`). Both front ends share `term_guard::char_inserts`, so this certifies the one predicate the agent TUI also uses. Layout restored to `en-US`. Plus the instrument flip ([`w8-altgr-postfix.txt`](phase3/w8-altgr-postfix.txt): pre-fix 4/4 dropped → post-fix 4/4 insert) and the pure-predicate unit test. |
| 4 | A deliberately panicking build restores the console (no wedged raw-mode window) | ✅ **REAL CODE, LIVE** | [`phase3/cert-real-panic-restore.txt`](phase3/cert-real-panic-restore.txt): `gate3_panic_restore_live` in a real console — alt screen entered, deliberate panic with the terminal armed, restore sequence emitted **before** the payload printed, and the real console input mode measured byte-identical before raw mode and after the hook (`0x1f7 == 0x1f7`). Plus the chained-hook and idempotency unit tests. |
| 5 | Mouse scroll and `Tab` sidebar still work | ✅ **REAL BINARY** (Tab) · ☐ mouse-scroll glance owed | [`phase3/cert-real-binary-run.txt`](phase3/cert-real-binary-run.txt): Tab hides the sidebar and Tab restores it (scrape-verified, `sidebar_before=True / after=False / restored=True`). Ctrl+D then quit with exit 0 and the spawned server was reaped — no stray processes. Mouse scroll is an untouched code path; a 5-second eyeball remains owed. |
| 6 | `--plain` and `camelid agent exec` unchanged | ✅ **REAL BINARY** (modelless) | Modelless `camelid chat --agent --plain` exits fast and plainly: one line (`agent mode needs a model — pass --model <gguf>`), exit 2, cooked console, no raw-mode entry; malformed-flag errors equally plain. The with-model legs inherit the row-1 RAM block. |
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
