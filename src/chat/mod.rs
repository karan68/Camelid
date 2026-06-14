//! `camelid chat` — an interactive terminal chat client for the local Camelid
//! engine.
//!
//! Two front ends share one [`session::Session`] core (state, sampling, request
//! shape — no I/O):
//! - [`tui`]: a full-screen ratatui app (scrollable chat, status bar, sidebar,
//!   modal picker) — the default on an interactive terminal.
//! - [`inline`]: a scrollback-friendly line REPL — used for `--plain`, pipes,
//!   and non-TTY contexts (the lane the smoke scripts and tests drive).
//!
//! Both stream `/v1/chat/completions` over the same audited HTTP/SSE client, so
//! terminal output matches the validated lane. The picker is derived from the
//! `/api/capabilities` ledger at runtime (supported rows only); pointing
//! `--model` at an unsupported GGUF is refused with the engine's typed error.
//! See `DECISIONS.md` D6 and `RECON_CHAT.md`.

mod banner;
mod client;
mod clipboard;
mod inline;
mod markdown;
mod models;
mod palette;
mod server;
mod session;
mod theme;
mod tui;

use std::io::IsTerminal;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use client::Client;
use server::ServerHandle;
use session::{LoadResult, Session, Settings};

pub(crate) const VERSION: &str = match option_env!("CAMELID_GIT_DESCRIBE") {
    Some(describe) => describe,
    None => env!("CARGO_PKG_VERSION"),
};

/// Parsed `camelid chat` flags.
pub struct ChatOptions {
    pub model: Option<PathBuf>,
    pub addr: SocketAddr,
    pub system: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
    pub no_stream: bool,
    pub models_dir: PathBuf,
    /// Force the inline line REPL instead of the full-screen TUI.
    pub plain: bool,
}

/// Entry point for the `Chat` subcommand. Returns a process exit code (0 = ok,
/// non-zero for the typed unsupported-state backstop) so the caller can exit
/// after this function's `ServerHandle` has torn down any spawned server.
pub fn run_chat(opts: ChatOptions) -> anyhow::Result<i32> {
    install_sigint_handler();

    let client = Client::new(opts.addr);
    let server = ServerHandle::ensure(opts.addr, &client)?;
    let spawned = server.spawned();

    let settings = Settings {
        temperature: opts.temperature,
        top_p: opts.top_p,
        top_k: opts.top_k,
        max_tokens: opts.max_tokens,
        seed: opts.seed,
        stream: !opts.no_stream,
    };
    let mut session = Session::new(client, opts.models_dir, settings, opts.system);

    // --model backstop: load + classify before any UI, so an unsupported GGUF
    // exits with the typed error and no screen takeover. Loading a cold GGUF can
    // take several seconds, so give feedback before the UI takes the screen.
    if let Some(model) = &opts.model {
        eprintln!("Loading {} …", model.display());
        match session.load_model_file(model, None, None)? {
            LoadResult::Loaded => {}
            LoadResult::Unsupported(message) => {
                eprintln!("{message}");
                return Ok(1);
            }
        }
    }

    // Full-screen TUI when we have a real terminal on both ends and the user did
    // not ask for plain mode; otherwise the inline REPL.
    let interactive = std::io::stdout().is_terminal() && std::io::stdin().is_terminal();
    if interactive && !opts.plain {
        tui::run(&mut session, opts.addr, spawned)?;
    } else {
        inline::run(&mut session, opts.addr, spawned)?;
    }
    Ok(0)
}

extern "C" fn on_sigint(_signal: libc::c_int) {
    session::CANCEL.store(true, Ordering::SeqCst);
}

/// Install a SIGINT handler that flips the cancel flag (used by the inline
/// stream loop). The TUI runs in raw mode where Ctrl-C arrives as a key event,
/// so it cancels through its event loop instead.
fn install_sigint_handler() {
    unsafe {
        libc::signal(libc::SIGINT, on_sigint as *const () as libc::sighandler_t);
    }
}
