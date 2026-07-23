//! HARDPAN — Windows terminal input probe (evidence instrument).
//!
//! Phase 0 ran this with the PRE-fix predicates and captured both defects live:
//! a 3-line paste arrived as keystrokes whose 2 Enters each fired a goal (W5),
//! and AltGr characters carrying CONTROL|ALT were 4/4 dropped (W8). The probe
//! now mirrors the POST-fix Phase 3 predicates, so re-running the same drivers
//! shows the flip — this is the GATE 3 CERT instrument:
//!
//! * **W5** — an Enter processed while more input is already queued
//!   (`event::poll(0)`) is a pasted newline, not a submit. Mirrors the agent
//!   TUI run loop's `paste_burst`.
//! * **W8** — a `Char` carrying CONTROL|ALT is an AltGr composition and
//!   inserts; CONTROL alone stays a chord. Mirrors `term_guard::char_inserts`.
//!
//! The probe prints the verdict next to each event, writes a log file, and
//! always restores the terminal.
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

/// Mirror of `term_guard::char_inserts` — the POST-W8 shared guard: a Char
/// carrying CONTROL|ALT is an AltGr composition (insert); CONTROL alone is a
/// chord (drop).
fn tui_would_insert(key: &KeyEvent) -> bool {
    matches!(key.code, KeyCode::Char(_))
        && (!key.modifiers.contains(KeyModifiers::CONTROL)
            || key.modifiers.contains(KeyModifiers::ALT))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let log_path = args.next().unwrap_or_else(|| "hardpan-input.log".into());
    let secs: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(25);

    let mut log: Vec<String> = Vec::new();
    macro_rules! rec {
        ($($a:tt)*) => {{ let s = format!($($a)*); println!("{s}\r"); log.push(s); }};
    }

    rec!("=== HARDPAN input probe (crossterm 0.28.1, POST-Phase-3 predicates) ===");
    rec!("stdout_is_tty        = {}", std::io::stdout().is_tty());
    rec!("window_secs          = {secs}");

    let raw = enable_raw_mode();
    rec!("enable_raw_mode      = {raw:?}");
    // Mirrors the post-W5 setup: non-fatal, on its own write.
    let bp = execute!(std::io::stdout(), EnableBracketedPaste);
    rec!("EnableBracketedPaste = {bp:?}  (non-fatal in the real TUI since W5)");
    rec!("");
    rec!("--- events (paste multi-line text; press AltGr chars; ESC to finish) ---");

    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut pastes = 0usize;
    let mut goals_fired = 0usize;
    let mut pasted_newlines = 0usize;
    let mut chars_inserted = 0usize;
    let mut chars_dropped = 0usize;
    let mut dropped_detail: Vec<String> = Vec::new();
    let mut altgr_inserted: Vec<String> = Vec::new();

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
            // On Windows this arm stays unreachable in crossterm 0.28.1.
            Event::Paste(ref s) => {
                pastes += 1;
                rec!("PASTE  len={} lines={} :: {:?}", s.len(), s.lines().count(), s);
            }
            Event::Key(k) => {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                let verdict = match k.code {
                    KeyCode::Enter => {
                        // Mirror of the agent TUI run loop (post-W5): an Enter
                        // glued to more queued input is a pasted newline.
                        let burst = event::poll(Duration::ZERO).unwrap_or(false);
                        if burst {
                            pasted_newlines += 1;
                            "-> paste_burst: insert '\\n'  [NO goal fires]".to_string()
                        } else {
                            goals_fired += 1;
                            "-> on_key: Enter => submit()  [FIRES A GOAL]".to_string()
                        }
                    }
                    KeyCode::Char(c) => {
                        if tui_would_insert(&k) {
                            chars_inserted += 1;
                            if k.modifiers.contains(KeyModifiers::CONTROL)
                                && k.modifiers.contains(KeyModifiers::ALT)
                            {
                                let d = format!("ALTGR-INSERT Char({c:?}) mods={:?}", k.modifiers);
                                altgr_inserted.push(d.clone());
                                format!("-> {d}  [W8 fix live]")
                            } else {
                                format!("-> on_key: insert_char({c:?})")
                            }
                        } else {
                            chars_dropped += 1;
                            let d = format!(
                                "DROPPED Char({c:?}) mods={:?} (CONTROL-only chord)",
                                k.modifiers
                            );
                            dropped_detail.push(d.clone());
                            format!("-> {d}")
                        }
                    }
                    _ => "-> (not a Char/Enter)".to_string(),
                };
                rec!(
                    "KEY    {:?} mods={:?} kind={:?}  {}",
                    k.code,
                    k.modifiers,
                    k.kind,
                    verdict
                );
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
    tail.push("=== SUMMARY (post-Phase-3 predicates) ===".into());
    tail.push(format!("event_paste_count      = {pastes}"));
    tail.push(format!(
        "goals_fired            = {goals_fired}   (W5: a multi-line paste must fire at most 1)"
    ));
    tail.push(format!(
        "pasted_newlines        = {pasted_newlines}   (W5: interior Enters absorbed as text)"
    ));
    tail.push(format!("chars_inserted         = {chars_inserted}"));
    for d in &altgr_inserted {
        tail.push(format!("  {d}"));
    }
    tail.push(format!(
        "chars_dropped_by_guard = {chars_dropped}   (W8: must be 0 for AltGr; CONTROL-only chords only)"
    ));
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
