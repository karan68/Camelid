//! Read-only Win32 clipboard access for the agent TUI's Ctrl+V paste path (W5).
//!
//! Read-only on purpose. HARDPAN GATE 0 struck W7 — OSC 52 clipboard *writes*
//! are honoured by both Windows Terminal and current conhost on this project's
//! supported hosts, so `clipboard.rs` stays the copy mechanism and is not
//! touched. *Reading*, however, has no terminal-side path that reaches the app
//! as an event on Windows (crossterm 0.28 never constructs `Event::Paste`
//! there), so paste needs the real Win32 API.
//!
//! SAFETY discipline follows `win_console.rs`: every `unsafe` block carries the
//! invariant it relies on.

use windows_sys::Win32::System::DataExchange::{CloseClipboard, GetClipboardData, OpenClipboard};
use windows_sys::Win32::System::Memory::{GlobalLock, GlobalSize, GlobalUnlock};

/// `CF_UNICODETEXT` — a fixed Win32 ABI value, written out here rather than
/// imported so one constant doesn't pull in the whole `Win32_System_Ole`
/// feature.
const CF_UNICODETEXT: u32 = 13;

/// Read the clipboard as text (`CF_UNICODETEXT`), or `None` if the clipboard is
/// held by another process, empty, or holds no text. Never blocks.
pub fn read_text() -> Option<String> {
    // SAFETY: OpenClipboard(null) opens the clipboard for the current task; a
    // zero return means another process holds it — give up rather than wait.
    let opened = unsafe { OpenClipboard(std::ptr::null_mut()) };
    if opened == 0 {
        return None;
    }
    // From here every path must CloseClipboard — hence the inner closure.
    let text = (|| {
        // SAFETY: the clipboard is open (checked above). The returned handle is
        // owned by the clipboard, not us: it is locked below, never freed.
        let handle = unsafe { GetClipboardData(CF_UNICODETEXT) };
        if handle.is_null() {
            return None;
        }
        // SAFETY: `handle` is a live HGLOBAL from GetClipboardData; GlobalLock
        // pins it and yields its address, or null on failure.
        let ptr = unsafe { GlobalLock(handle) } as *const u16;
        if ptr.is_null() {
            return None;
        }
        // SAFETY: while locked, the allocation is valid for GlobalSize(handle)
        // bytes. CF_UNICODETEXT is NUL-terminated by contract, but a
        // misbehaving producer may omit the NUL — bound the scan by the
        // allocation size instead of trusting the contract.
        let max_units = unsafe { GlobalSize(handle) } / 2;
        let mut len = 0usize;
        // SAFETY: every read is inside the locked allocation (len < max_units).
        while len < max_units && unsafe { *ptr.add(len) } != 0 {
            len += 1;
        }
        // SAFETY: [ptr, ptr + len) was verified in-bounds by the scan above.
        let units = unsafe { std::slice::from_raw_parts(ptr, len) };
        let s = String::from_utf16_lossy(units);
        // SAFETY: `handle` is locked (above); this releases only our pin.
        unsafe { GlobalUnlock(handle) };
        Some(s)
    })();
    // SAFETY: opened by us above and not yet closed on any path.
    unsafe { CloseClipboard() };
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: read_text must never crash or hang, whatever the clipboard holds
    /// (including empty or non-text, and headless CI window stations where the
    /// open itself may fail — every shape returns cleanly).
    #[test]
    fn read_text_never_panics() {
        let _ = read_text();
    }

    /// GATE 3 evidence probe — round-trips a known value through the REAL
    /// clipboard (mutates global state, so #[ignore]d; run explicitly):
    ///   cargo test --release --lib -- --ignored --nocapture gate3_clipboard
    #[test]
    #[ignore = "mutates the real clipboard — run with --ignored --nocapture"]
    fn gate3_clipboard_read_returns_real_content() {
        let marker = "HARDPAN-GATE3-CLIPBOARD-\u{00E9}\u{20AC}-multi\r\nline";
        // Seed via PowerShell (the same surface a user copies from), then read
        // through the module under test.
        let status = std::process::Command::new("powershell.exe")
            .args(["-NoProfile", "-Command", "Set-Clipboard -Value ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('SEFSRFBBTi1HQVRFMy1DTElQQk9BUkQtw6nigqwtbXVsdGkNCmxpbmU=')))"])
            .status()
            .expect("spawn powershell");
        assert!(status.success(), "Set-Clipboard failed");
        std::thread::sleep(std::time::Duration::from_millis(300));
        let got = read_text().expect("clipboard should hold text");
        println!("[GATE3-W5] clipboard_read = {got:?}");
        assert_eq!(got.trim_end(), marker, "read must match what was copied");
    }
}
