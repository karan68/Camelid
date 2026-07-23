//! HARDPAN Phase 0 — Windows terminal input probe (evidence instrument).
//!
//! Answers two contested findings with observation instead of argument:
//!
//! * **W5** — does a multi-line paste arrive as `Event::Paste`, or as key events
//!   containing Enter? Each Enter that reaches `on_key` calls `submit()`
//!   (agent_tui.rs:759), so the Enter count IS the count of goals that would fire.
//! * **W8** — what modifiers ride along with an AltGr-produced character? The
//!   agent TUI computes `ctrl = key.modifiers.contains(CONTROL)`
//!   (agent_tui.rs:718) and inserts only `Char(c) if !ctrl` (agent_tui.rs:767),
//!   so a Char carrying CONTROL is silently dropped.
//!
//! The probe reimplements those two predicates verbatim and prints the verdict
//! next to each event. It writes a log file and always restores the terminal.
//!
//! Run in a REAL console window (never a redirected pipe):
//!   cargo run --release -- <logfile> [seconds]

use std::io::Write;
use std::time::{Duration, Instant};

use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use crossterm::{execute, tty::IsTty};

/// Mirror of `agent_tui.rs:718` — the guard the real TUI applies.
fn tui_ctrl(key: &KeyEvent) -> bool {
    key.modifiers.contains(KeyModifiers::CONTROL)
}

/// Mirror of `agent_tui.rs:767` — `KeyCode::Char(c) if !ctrl => insert_char(c)`.
fn tui_would_insert(key: &KeyEvent) -> bool {
    matches!(key.code, KeyCode::Char(_)) && !tui_ctrl(key)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let log_path = args.next().unwrap_or_else(|| "hardpan-input.log".into());
    let secs: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(25);

    let mut log: Vec<String> = Vec::new();
    macro_rules! rec {
        ($($a:tt)*) => {{ let s = format!($($a)*); println!("{s}\r"); log.push(s); }};
    }

    rec!("=== HARDPAN input probe (crossterm 0.28.1) ===");
    rec!("stdout_is_tty        = {}", std::io::stdout().is_tty());
    rec!("window_secs          = {secs}");

    // W5 second-order: does EnableBracketedPaste fail on this host? The real TUI
    // puts this behind `?` at agent_tui.rs:214, AFTER raw mode + alternate screen
    // are already active, so a failure here would strand the console.
    let raw = enable_raw_mode();
    rec!("enable_raw_mode      = {raw:?}");
    let bp = execute!(std::io::stdout(), EnableBracketedPaste);
    rec!("EnableBracketedPaste = {bp:?}");
    rec!("  (Err here => the `?` at agent_tui.rs:214 would bail with raw mode ON)");
    rec!("");
    rec!("--- events (paste multi-line text; press AltGr chars; ESC to finish) ---");

    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut enters = 0usize;
    let mut pastes = 0usize;
    let mut chars_inserted = 0usize;
    let mut chars_dropped = 0usize;
    let mut dropped_detail: Vec<String> = Vec::new();

    while Instant::now() < deadline {
        let left = deadline.saturating_duration_since(Instant::now());
        match event::poll(left.min(Duration::from_millis(200))) {
            Ok(true) => {}
            Ok(false) => continue,
            Err(e) => {
                rec!("poll error: {e}");
                break;
            }
        }
        let ev = match event::read() {
            Ok(e) => e,
            Err(e) => {
                rec!("read error: {e}");
                break;
            }
        };
        match ev {
            // On Windows this arm is expected to be unreachable: crossterm
            // 0.28.1 constructs Event::Paste only in sys/unix/parse.rs.
            Event::Paste(ref s) => {
                pastes += 1;
                rec!("PASTE  len={} lines={} :: {:?}", s.len(), s.lines().count(), s);
            }
            Event::Key(k) => {
                // Windows delivers both Press and Release records; the TUI acts
                // on Press only, so count that way.
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                let verdict = match k.code {
                    KeyCode::Enter => {
                        enters += 1;
                        "-> on_key: KeyCode::Enter => submit()  [FIRES A GOAL]".to_string()
                    }
                    KeyCode::Char(c) => {
                        if tui_would_insert(&k) {
                            chars_inserted += 1;
                            format!("-> on_key: insert_char({c:?})")
                        } else {
                            chars_dropped += 1;
                            let d = format!(
                                "DROPPED Char({c:?}) mods={:?} (guard `if !ctrl` at agent_tui.rs:767)",
                                k.modifiers
                            );
                            dropped_detail.push(d.clone());
                            format!("-> {d}")
                        }
                    }
                    _ => "-> (not a Char/Enter)".to_string(),
                };
                rec!("KEY    {:?} mods={:?} kind={:?}  {}", k.code, k.modifiers, k.kind, verdict);
                if k.code == KeyCode::Esc {
                    rec!("(ESC pressed - finishing)");
                    break;
                }
            }
            other => rec!("OTHER  {other:?}"),
        }
    }

    let _ = execute!(std::io::stdout(), DisableBracketedPaste);
    let _ = disable_raw_mode();

    let mut tail = Vec::new();
    tail.push(String::new());
    tail.push("=== SUMMARY ===".into());
    tail.push(format!("event_paste_count      = {pastes}   (W5: >0 means bracketed paste works here)"));
    tail.push(format!("enter_key_count        = {enters}   (W5: goals that would fire from one paste)"));
    tail.push(format!("chars_inserted         = {chars_inserted}"));
    tail.push(format!("chars_dropped_by_guard = {chars_dropped}   (W8: AltGr characters lost)"));
    for d in &dropped_detail {
        tail.push(format!("  {d}"));
    }
    for line in &tail {
        println!("{line}");
    }
    log.extend(tail);

    if let Ok(mut f) = std::fs::File::create(&log_path) {
        let _ = f.write_all(log.join("\r\n").as_bytes());
    }
    println!("\nlog written to {log_path}");
}
