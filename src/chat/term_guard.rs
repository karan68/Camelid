//! Terminal teardown safety for the full-screen TUIs (HARDPAN W6).
//!
//! Both `agent_tui` and `tui` enable raw mode, the alternate screen, and mouse
//! capture, then relied on straight-line teardown at the end of `run`. An early
//! `?` between setup and teardown — or a panic on the TUI thread — left the
//! console in raw mode on the alternate buffer with QuickEdit cleared. On Unix
//! the operator types `reset`; cmd.exe has no `reset`, so on Windows the window
//! is dead and gets closed (the W6 finding's severity).
//!
//! Two layers, one shared implementation:
//! - [`TerminalGuard`] — an RAII guard armed right after raw mode goes on; its
//!   `Drop` restores the terminal on every exit path, including early `?`
//!   returns and unwinding panics.
//! - [`install_panic_hook`] — a chaining panic hook that restores the terminal
//!   BEFORE the previous hook prints the payload, so the message lands on the
//!   main screen buffer with echo on instead of invisibly inside the alternate
//!   screen. It restores only for panics on the thread that armed the guard:
//!   worker-thread panics (the agent loop runs on a spawned worker) are
//!   non-fatal today and must not yank a live TUI out of its alternate screen.
//!
//! Restoration is idempotent (an atomic active flag), so hook-then-Drop is
//! fine, and `run_suspended`'s own teardown/re-entry (tui.rs) is unaffected.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use ratatui::crossterm::cursor::Show;
use ratatui::crossterm::event::{DisableBracketedPaste, DisableMouseCapture};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{disable_raw_mode, LeaveAlternateScreen};

/// Whether a TUI currently owns the terminal (armed and not yet restored).
static TUI_ACTIVE: AtomicBool = AtomicBool::new(false);
/// The thread that armed the guard — the only thread whose panics restore.
static TUI_THREAD: Mutex<Option<std::thread::ThreadId>> = Mutex::new(None);

/// Restore the terminal to a usable state. Idempotent — the first caller wins,
/// later calls are no-ops. Best-effort by design: a failing write during
/// teardown must not mask the original error or panic payload.
pub(crate) fn restore_terminal() {
    if !TUI_ACTIVE.swap(false, Ordering::SeqCst) {
        return;
    }
    // Same order as the straight-line teardown this replaces: cooked mode
    // first, then leave the alternate screen, release the mouse, and bring the
    // cursor back.
    let _ = disable_raw_mode();
    let mut out = std::io::stdout();
    let _ = execute!(out, LeaveAlternateScreen, DisableMouseCapture, Show);
    // Separate write: on a host where this one command is unsupported, it must
    // not short-circuit the cursor restore above.
    let _ = execute!(out, DisableBracketedPaste);
}

/// RAII terminal guard. Arm it immediately after `enable_raw_mode` succeeds so
/// every later fallible setup step — and everything inside the run loop — is
/// covered by `Drop`.
pub(crate) struct TerminalGuard;

impl TerminalGuard {
    pub(crate) fn arm() -> Self {
        *TUI_THREAD
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(std::thread::current().id());
        TUI_ACTIVE.store(true, Ordering::SeqCst);
        TerminalGuard
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_terminal();
    }
}

/// Install the chaining panic hook (once per process; both front ends call it,
/// later calls are no-ops). The previous hook — default or user-installed — is
/// captured and invoked after restoration, so panic output is never lost.
pub(crate) fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let tui_thread = *TUI_THREAD
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if tui_thread == Some(std::thread::current().id()) {
                restore_terminal();
            }
            previous(info);
        }));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// W6 — the hook chains: a hook installed before ours still runs, so panic
    /// output is not lost. Probed with a worker-thread panic (which must NOT
    /// restore, only chain).
    #[test]
    fn panic_hook_chains_to_previous() {
        use std::sync::atomic::AtomicUsize;
        static PREV_RAN: AtomicUsize = AtomicUsize::new(0);
        let earlier = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            PREV_RAN.fetch_add(1, Ordering::SeqCst);
            earlier(info);
        }));
        install_panic_hook();
        let before = PREV_RAN.load(Ordering::SeqCst);
        let _ = std::thread::spawn(|| panic!("hardpan w6 chain probe")).join();
        assert!(
            PREV_RAN.load(Ordering::SeqCst) > before,
            "the pre-existing hook must still run after install_panic_hook"
        );
    }

    /// W6 — restore is gated on the armed flag (no-op when idle) and arming /
    /// dropping flips it exactly once.
    #[test]
    fn guard_restores_once_and_only_when_armed() {
        // Not armed: restore must be a no-op and must not panic.
        restore_terminal();
        let guard = TerminalGuard::arm();
        assert!(TUI_ACTIVE.load(Ordering::SeqCst), "arm sets the flag");
        drop(guard);
        assert!(
            !TUI_ACTIVE.load(Ordering::SeqCst),
            "drop restores and clears the flag"
        );
    }
}
