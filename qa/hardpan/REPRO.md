# HARDPAN Phase 0 — reproduction table

**Campaign:** HARDPAN — Windows Agent-Terminal Parity
**Gate:** GATE 0 (Tim signs this table before any product code lands)
**Base:** `origin/main` @ `b4e3a905`, worktree branch `campaign/hardpan-phase0`
**Host:** Windows 11 Home 26220.8925 · x86_64 · rustc **1.95.0** (pinned by `rust-toolchain.toml`)
**Pinned deps as resolved:** crossterm **0.28.1** (via `ratatui` 0.29.0 — `cargo tree -i crossterm`), ratatui **0.29.0**
**Console:** ANSI code page 1252, OEM code page 437; Windows Terminal 1.24.11911.0 installed; `HKCU:\Console\%%Startup` absent (default delegation)

> **Read this first.** Seven of the nine findings reproduce. **W7 does not reproduce and is proposed
> for striking.** **W9's premise is contradicted** by both measurement and the ratatui source, and is
> proposed for downgrade to a documented null. Three more findings (W1, W2, W3) reproduce but are
> **materially narrower or differently caused** than the conductor states. Every correction is
> evidenced below. Nothing here was taken on the strength of the conductor document.

---

## Summary

| ID | Conductor's claim | Verdict | Evidence |
|----|-------------------|---------|----------|
| **W1** | `run_shell` never drains pipes → false timeout, output discarded | **CONFIRMED (live)** | 417,792 B → `Err("command timed out after 10s")`, 0 payload bytes, full 10,075 ms burned. Control at 32,640 B → `Ok` in **87 ms**. |
| **W2** | No job object → orphans; no `CREATE_NO_WINDOW` | **CONFIRMED (live) / half CONTESTED** | 1 orphaned `PING.EXE` grandchild survived the timeout. The `CREATE_NO_WINDOW` half is likely **unreachable** — see A6. |
| **W3(a)** | UTF-8 mangled via ANSI code page | **CONFIRMED (live), wrong mechanism** | Child settles on **OEM 437**, not ANSI 1252. Input leg: 6 codepoints arrive as **16**. Output leg irreversibly lossy. |
| **W3(b)** | Failing native command returns `Ok` → "build passed" | **CONFIRMED (live), but narrower** | True **only when the failure is not the last statement**. `cmd /c exit 3` alone → `Err`. Real code flattened 3/42 → **1**. |
| **W4** | Bare `cmd`; hardening only | **REASONED — hardening null** | std searches **System32 before parent PATH**; threat-model delta ≈ 0. See A7. |
| **W5** | Bracketed paste dead on Windows → partial goals | **CONFIRMED (live)** | 3-line paste → `Event::Paste = 0`, **2 Enter events = 2 goals fired**. Second-order `?`-bail **not** reproduced. |
| **W6** | No panic/early-return guard wedges the console | **REASONED, not reproduced** (by design) | No panic hook in `src/`. Deliberately not reproduced — see the W6 row. |
| **W7** | conhost ignores OSC 52 → `/copy` lies | **DOES NOT REPRODUCE → propose STRIKE** | OSC 52 **honoured**, clipboard updated, in 2/2 real-console runs. |
| **W8** | AltGr characters silently swallowed | **CONFIRMED (live)** | `@ € [ ]` all arrive as `CONTROL\|ALT` → **4/4 dropped** by the guard. |
| **W9** | 1 KiB `LineWriter` costs many console writes | **PREMISE CONTRADICTED → propose null** | 8× the writes cost **1.07×** the time. ratatui writes *diffs* and already flushes once per frame. |

---

## How to re-run everything

Exec-surface probes (W1, W2, W3) drive the **real** functions through `validate()` + `execute()`:

```bash
cargo test --release --lib -- --ignored --nocapture hardpan_
```

They live in `src/chat/tools.rs` (`mod tests`), are `#[cfg(windows)]` + `#[ignore]`, and **assert
nothing about correctness** — the observation is the deliverable, so CI stays green on a tree that
still has the defect. Phase 1/2 convert them into asserting regression tests.

Terminal probes (W5, W8) need a **real console** — under a redirected stdout the paste path and
console write costs do not exist:

```bash
cargo run --release --manifest-path qa/hardpan/input-probe/Cargo.toml -- input.log 25
```

`qa/hardpan/input-probe/` is a standalone crate, detached from the workspace (empty `[workspace]`
table), pinned to `crossterm =0.28.1` — the exact version camelid resolves. It reimplements the two
TUI predicates verbatim (`agent_tui.rs:718` and `:767`) and prints the verdict beside each event.

---

## W1 — `run_shell` never drains its pipes  ·  **CONFIRMED**

**Anchor:** `src/chat/tools.rs:1818-1867`; the sole read is `wait_with_output()` at `:1868`, reached
only after the child exits.

**Command:** `type big.txt` on a 417,792-byte file, 10 s shell timeout.

| Metric | >64 KiB payload | Control (32,640 B) |
|---|---|---|
| outcome | `ToolOutcome::Err` | `ToolOutcome::Ok` |
| returned text | `command timed out after 10s` (27 B) | 16,416 B |
| payload captured | **false** | true |
| elapsed | **10,075 ms** (full deadline) | **87 ms** |

The only difference is crossing the 64 KiB pipe buffer. A successful command is reported to the model
as a timeout with every byte discarded.

**Corrections to the conductor (all A1/A3/A4/A5 below):** the std anchor, the API name, the
"Windows-specific" framing, the `cargo build` exemplar, and the value of the recovered output are all
wrong or overstated. The **defect and its consequence are exactly as described.**

---

## W2 — no job object, no `CREATE_NO_WINDOW`  ·  **job half CONFIRMED, console half CONTESTED**

**Anchor:** `src/chat/tools.rs:1856` (`child.kill()` reaps only `cmd.exe`).
**Verified:** `src/chat/shell_sandbox.rs` contains **no** `creation_flags`, `JobObject`, `win_job`, or
`CREATE_NO_WINDOW` — nothing upstream supplies what `run_shell` omits.

**Command:** `ping -n 120 127.0.0.1` (a grandchild under `cmd /C`), 5 s timeout.

```
elapsed_ms                = 5047
outcome                   = ToolOutcome::Err
orphaned_grandchild_pids  = [16080]
orphan_count              = 1
```

The orphan **survives**; it is not killed by the read ends closing. The probe cleans up by PID only —
never by image name.

**Contested half → needs your call (A6):** adding `CREATE_NO_WINDOW` may be a pure regression. The
adversarial pass argues `run_shell` is never registered on any console-less surface (desktop spawns
`serve`, whose agent loop pins `ShellSandbox::Disabled` + `WorkspaceReadOnly`; the subagent worker
hardcodes `yolo: false` + an approver that always denies Exec). If so the flag would only mint a
hidden console per invocation on the surfaces where `run_shell` *does* run. **Not independently
re-verified by me** — flagged, not accepted.

---

## W3(a) — PowerShell encoding  ·  **CONFIRMED, mechanism wrong, scope narrower**

**Anchors:** `src/chat/tools.rs:2192` (raw UTF-8 in), `:2231` (`from_utf8_lossy` out).
**Vector:** `U+00E9 U+00FC U+2014 U+2713 U+65E5 U+20AC` = `c3 a9 c3 bc e2 80 94 e2 9c 93 e6 97 a5 e2 82 ac`

```
output_leg_got_hex        = ef bf bd ef bf bd 2d ef bf bd 3f 3f      (want: the 16 bytes above)
output_leg_matches        = false
input_leg_got_codepoints  = U+251C U+2310 U+251C U+255D U+0393 U+00C7 U+00F6 U+0393
                            U+00A3 U+00F4 U+00B5 U+00F9 U+00D1 U+0393 U+00E9 U+00BC
input_leg_matches         = false
child_encodings           = Out=IBM437 In=IBM437
```

Both legs are corrupted. **Six** codepoints arrive as **sixteen**. The output leg is *irreversibly*
lossy: `—` → `-` and `日`/`€` → `?` are valid ASCII, so `from_utf8_lossy` never flags them.

**Two corrections.** (1) It is the **OEM** code page (437), not the ANSI page (1252) the conductor
names. (2) The stated reason `win_console::init()` cannot help — "these handles are pipes" — is
wrong; a `CREATE_NO_WINDOW` child gets a **fresh, windowless console** that simply does not inherit
the parent's code page. The conclusion (it doesn't help) stands; the mechanism does not.

**Scope narrower than stated:** a pure byte **pass-through** of native tool output round-trips
byte-transparently, because CP437 is single-byte and the decode/re-encode is lossless at byte level.
So `cargo`/`git`/`rustc` output is *not* corrupted. The damage is to text PowerShell **interprets** —
string length, comparison, regex, path handling — where the child genuinely sees mojibake.

**Phase 2 fix pre-validated (forward-looking, no product code changed).** The conductor's ASCII
preamble + base64 payload was measured end-to-end:

```
FIX-1 output leg : c3 a9 c3 bc e2 80 94 e2 9c 93 e6 97 a5 e2 82 ac   -> byte-exact
FIX-2 input leg  : U+00E9 U+00FC U+2014 U+2713 U+65E5 U+20AC          -> exact
FIX-3 encodings  : Out=utf-8  In=IBM437   (In no longer matters: base64 is ASCII)
```

This confirms the conductor's reasoning for *why* base64 is airtight where setting `InputEncoding`
alone is not. **`pwsh.exe` is not installed on this host (A8)**, so the "prefer PowerShell 7" option
cannot be tested here and would ship as an untested fallback.

---

## W3(b) — swallowed exit codes  ·  **CONFIRMED, but only in one shape**

| command | reported | outcome |
|---|---|---|
| `cmd /c exit 3; Write-Output done` | `exit: 0` | **`ToolOutcome::Ok`** ← the defect |
| `cmd /c exit 3` | `exit: 1` | `ToolOutcome::Err` |
| `cmd /c exit 42` | `exit: 1` | `ToolOutcome::Err` |
| `Write-Output ok` | `exit: 0` | `ToolOutcome::Ok` |
| `throw 'boom'` | `exit: 1` | `ToolOutcome::Err` |

**The conductor overstates this.** "A failed `cargo build` yields `status.success() == true`" is
**false** when the failing command is the last statement — PowerShell exits 1 and the tool correctly
reports `Err`. The real defect is narrower and still serious: **any successful statement after a
failing native command erases the failure entirely**, and the true exit code is flattened to 1 in
every case.

Propagation was validated forward: `exit $LASTEXITCODE` restores 3 and 42, keeps a pure-cmdlet
success at 0, and keeps `throw` non-zero. **Residual limit to document, not a regression:**
`$LASTEXITCODE` tracks only the *last* native command, so `cmd /c exit 3; cmd /c exit 0` reports 0
both before and after the fix.

---

## W4 — bare `cmd`  ·  **REASONED (not reproduced), and a hardening null**

Per the conductor, W4 is reasoned rather than reproduced: there is no live exploit to capture.

**Stronger than the conductor states — this is a *null*, not a low-severity item.** std searches, in
order: child-`PATH` → **application directory** → `GetSystemDirectoryW` → `GetWindowsDirectoryW` →
parent-`PATH`. **System32 precedes parent-`PATH`**, and child-`PATH` is only consulted when the
builder changed `PATH` (`grep` finds no `.env(`/`.envs(`/`env_clear()` in `tools.rs` or
`shell_sandbox.rs`). The current directory is never searched. So `Command::new("cmd")` at
`tools.rs:1831` **already resolves to `%SystemRoot%\System32\cmd.exe`** in every case except an
attacker-writable application directory — where an attacker could already replace `camelid.exe`
itself. Routing through `system32()` is still worth doing for consistency with the neighbouring
comment, but the commit must not claim a security delta.

The `cmd /C` quoting concern is real but degrades into *errors and garbled literals*, not injection:
std only emits `\` immediately before `"`, and `"` is an illegal Windows filename character.

---

## W5 — bracketed paste is dead on Windows  ·  **CONFIRMED (live)**

Source-verified: in crossterm 0.28.1, `Event::Paste` is constructed **only** in
`src/event/sys/unix/parse.rs` (3 sites). It cannot occur on Windows.

Live, pasting a **3-line** goal into a real console via Ctrl+V:

```
event_paste_count = 0
enter_key_count   = 2      <- each maps to KeyCode::Enter => submit()
chars_inserted    = 204
```

Each Enter arrived as a genuine `KeyCode::Enter`, so **one paste fires two goals**: the first line
launches, the second collides with the "a goal is already running" guard, the third is left in the
box. Exactly as described.

**One half struck:** the second-order claim (that `EnableBracketedPaste` fails and the `?` at
`agent_tui.rs:214` bails with raw mode already on) **does not reproduce** — the probe measured
`EnableBracketedPaste = Ok(())`. Making that call non-fatal is still cheap insurance, but it fixes
nothing observable here.

---

## W6 — no panic or early-return guard  ·  **REASONED, deliberately not reproduced**

Per the conductor, W6 is reasoned rather than reproduced: deliberately panicking a TUI to wedge a
console is not a receipt worth collecting, and it risks the operator's session.

Verified statically: **no `set_hook` anywhere in `src/`**; `agent_tui.rs:206-227` enables raw mode,
the alternate screen, mouse capture, and bracketed paste, then relies on straight-line teardown.
`tui.rs:74-88` has the same shape.

**Both named triggers look weaker than stated.** The `?` at `:214` does not fire here (measured
above). And the agent/tool/HTTP work runs on a **spawned worker thread** (`agent_tui.rs:470`), so its
panics do not bypass teardown. Counter-note worth keeping: a hook would not help the vectors that
actually wedge a console (Ctrl+Break, kill-by-PID, stack-overflow abort) because those do not unwind.
W6 remains worth fixing as **cheap, correct hygiene and a safety net for Phase 3 testing** — which is
exactly the role the conductor assigns it — but it should not be sold as fixing an observed failure.

---

## W7 — `/copy` reports success on a clipboard it never wrote  ·  **DOES NOT REPRODUCE — propose STRIKE**

The finding's user-visible consequence rests on one premise: *"conhost ignores OSC 52 outright."*
**That premise is false on this host.** Two independent runs in real console windows, emitting
byte-for-byte the sequence `clipboard.rs:8` builds (`ESC ] 52 ; c ; <base64> BEL`):

| run | launched via | `osc52_honoured` | clipboard after |
|---|---|---|---|
| A | default terminal | **true** | `HARDPAN-OSC52-default` |
| B | `conhost.exe` explicitly | **true** | `HARDPAN-OSC52-forced-conhost` |

The clipboard was seeded with a sentinel first and verified changed. Windows Terminal has honoured
OSC 52 writes by default since v1.2 (2020) and conhost itself gained it in 2025 — this box runs a
2026 build with WT 1.24.11911.0 installed.

**What remains true is a code fact, not a defect:** `copy()` returns `true` because the *write*
succeeded, so it cannot in principle know whether the terminal acted. That is worth fixing if the
Win32 clipboard path lands anyway for W5's Ctrl+V paste — but on this host `/copy` **tells the
truth**, and per the conductor's own rule ("anything that does not reproduce is struck from scope"),
**W7's headline consequence should be struck.**

Note this also removes the stated justification for adding `Win32_System_DataExchange` +
`Win32_System_Memory` to `Cargo.toml`. Those features are still needed **if** you want the Ctrl+V
*read* path for W5 — which is now the stronger reason to add them.

---

## W8 — AltGr characters are silently swallowed  ·  **CONFIRMED (live)**

Source-verified: `agent_tui.rs:718` computes `ctrl = key.modifiers.contains(CONTROL)` and `:767`
inserts only `Char(c) if !ctrl` — the guard tests CONTROL **alone** and ignores ALT. crossterm
`parse.rs:82-83` derives CONTROL from `LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED` and ALT from
`LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED`, with no AltGr special case. Identical guard at `tui.rs:626`
and `:712`.

Live, with the German layout added temporarily via `Set-WinUserLanguageList` (**restored to `en-US`
immediately afterwards, verified**) and AltGr chords synthesised as LCtrl+RAlt via `SendInput`:

```
KEY Char('@') mods=CONTROL|ALT  -> DROPPED  (guard `if !ctrl` at agent_tui.rs:767)
KEY Char('€') mods=CONTROL|ALT  -> DROPPED
KEY Char('[') mods=CONTROL|ALT  -> DROPPED
KEY Char(']') mods=CONTROL|ALT  -> DROPPED
KEY Char('a') mods=(none)       -> insert_char('a')      <- control, still works
chars_dropped_by_guard = 4
```

**4 of 4 AltGr characters lost.** A user on a German keyboard cannot type `@`, `€`, `[`, or `]` into
the goal box. The counter-hypothesis that Windows Terminal/ConPTY would strip the modifiers is
**false on this host**.

---

## W9 — 1 KiB `LineWriter` on the renderer  ·  **PREMISE CONTRADICTED — propose documented null**

Source-verified: `LineWriter::new` does use a 1024-byte buffer (std 1.95.0,
`io/buffered/linewriter.rs:90`) and `Stdout` is a `LineWriter`.

**Measured** in a real console (200 frames × 8 KiB, same bytes both ways):

| delivery | writes/frame | total | per frame |
|---|---|---|---|
| 8 × 1 KiB (LineWriter shape) | 8 | 139 ms | 0.695 ms |
| 1 × 8 KiB (BufWriter shape) | 1 | 130 ms | 0.650 ms |

**8× the syscalls cost 1.07× the time.** Console cost here tracks *bytes*, not call count.

Two independent source facts make the real-world delta smaller still: ratatui's `Terminal::flush`
(`terminal.rs:198-205`) writes only the **diff** between frames, not a full frame — in the agent's
steady state that is the spinner plus a few streamed characters, far under 1 KiB. And `Terminal::draw`
always ends in an `execute!`, which is queue-**then-flush**, so any frame whose diff is under 1024
bytes is **already exactly one write today**.

Recommendation: keep the `BufWriter` change only if it is free, and land the null explicitly — the
SIROCCO/STAMPEDE precedent. **Do not report W9 as a win.**

---

## Amendment log

| id | anchor | consequence | phase |
|----|--------|-------------|-------|
| **A1** | conductor cites "std 1.85 `sys/pal/windows/pipe.rs:61`" | Toolchain is **1.95.0**; the path is now `sys/process/windows/child_pipe.rs:56`. Constant `64 * 1024` unchanged. API is **`NtCreateNamedPipeFile`** (`child_pipe.rs:116`), not `CreateNamedPipeW` — std explicitly says it does *not* use `CreatePipe`. Cosmetic, but every W1 citation needs correcting. | 1 |
| **A2** | `tools.rs:2137` | A `CREATE_NO_WINDOW` child gets a **fresh windowless console** at OEM CP437 — not "no console at all", and not the ANSI page. Verified: `CreateNoWindow=false` → 65001. Conclusion unchanged, mechanism wrong. | 2 |
| **A3** | `tools.rs:1818` | W1 is **not Windows-specific** — `run_shell` is one cross-platform fn and Linux's default pipe capacity is also 64 KiB. The fix should say so. | 1 |
| **A4** | `src/main.rs:301` | Default `shell_timeout` is **30 s**, so `cargo build`/`npm ci` false-timeout with or without W1. Three of the four cited exemplars don't isolate the defect. `git log` in this repo *does*: 1,536,819 bytes. | 1 |
| **A5** | `tools.rs:258`, `:2458`, `:205` | `MAX_OUTPUT_BYTES = 16 KiB` and a second `clipped()` layer cap what the model can ever see. W1's fix buys a correct **outcome**, not "every byte of output". | 1 |
| **A6** | `workspace_bridge.rs:404`, `subagent.rs:590` | W2's `CREATE_NO_WINDOW` half may be **unreachable** (`run_shell` not registered on console-less surfaces) and could be a pure regression. Needs a decision before Phase 1. | 1 |
| **A7** | `tools.rs:1831` | Bare `cmd` is a **hardening null**: std searches System32 **before** parent-`PATH`, and never the cwd. Do not claim a security delta. | 1 |
| **A8** | Phase 2 | **`pwsh.exe` is not installed on this host.** The "prefer PowerShell 7" option cannot be validated here; if taken, it ships untested. | 2 |
| **A9** | `ratatui-0.29.0/src/terminal/terminal.rs:198-205`, `backend/crossterm.rs:211` | W9's premise is contradicted at the source: ratatui writes **diffs only** and `execute!` already flushes once per frame. | 4 |
| **A10** | `qa/hardpan/input-probe/` | New evidence instrument added (standalone, workspace-detached, `crossterm =0.28.1`). Reusable as the Phase 3 CERT harness. | 0 |

---

## What GATE 0 is asking you to sign

1. **Seven findings reproduce**: W1, W2 (job half), W3(a), W3(b), W5, W8 — plus W4/W6 accepted as reasoned-only per the conductor's own carve-out.
2. **Strike W7.** Its premise is false on this host; `/copy` currently tells the truth.
3. **Downgrade W9** to a measured null before Phase 3 spends effort on it.
4. **Decide A6** — does the `CREATE_NO_WINDOW` half of W2 stay in scope?
5. **Accept the narrowed scope of W3(b)** — the bug is "a success after a failure erases it", not "failures always report success".
