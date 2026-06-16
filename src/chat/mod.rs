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

mod agent;
mod agent_eval;
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
mod tool_parse;
mod tools;
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
    /// Enter agent mode (tool-calling loop) instead of plain chat.
    pub agent: bool,
    /// Sandbox root for agent tools (default: cwd).
    pub workdir: Option<PathBuf>,
    pub max_steps: usize,
    pub auto_approve: bool,
    pub allow_net: bool,
    pub shell_timeout: u64,
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
    // take several seconds, so give feedback before the UI takes the screen. A
    // known supported GGUF is labeled with its ledger id (so posture + the agent
    // tool-capable gate match), exactly like the picker.
    if let Some(model) = &opts.model {
        eprintln!("Loading {} …", model.display());
        let label = catalog_label_for(model);
        let posture = label.as_ref().map(|_| "supported");
        match session.load_model_file(model, label.as_deref(), posture)? {
            LoadResult::Loaded => {}
            LoadResult::Unsupported(message) => {
                eprintln!("{message}");
                return Ok(1);
            }
        }
    }

    // Agent mode: a tool-calling loop (line renderer), gated to tool-capable rows.
    if opts.agent {
        if !session.has_model() {
            eprintln!("agent mode needs a model — pass --model <gguf>");
            return Ok(2);
        }
        let cfg = agent::AgentConfig {
            workdir: opts.workdir.unwrap_or_else(|| PathBuf::from(".")),
            max_steps: opts.max_steps,
            auto_approve: opts.auto_approve,
            allow_net: opts.allow_net,
            shell_timeout: std::time::Duration::from_secs(opts.shell_timeout),
            max_tokens: opts.max_tokens,
            temperature: opts.temperature,
        };
        return agent::run_agent(&mut session, opts.addr, cfg);
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

/// If `model`'s filename matches a curated-catalog row, return that catalog id
/// (= the ledger row id) so a `--model`-loaded supported GGUF carries its ledger
/// identity (posture + agent tool-capable gate).
fn catalog_label_for(model: &std::path::Path) -> Option<String> {
    let name = model.file_name()?.to_str()?;
    camelid::api::curated_catalog()
        .into_iter()
        .find(|item| item.filename == name)
        .map(|item| item.catalog_id.to_string())
}

/// Parsed `camelid agent-eval` flags.
pub struct AgentEvalOptions {
    pub model: PathBuf,
    pub addr: SocketAddr,
    pub load_timeout: u64,
    pub max_steps: usize,
    pub max_tokens: u32,
    pub receipt_dir: PathBuf,
}

/// Entry for the `agent-eval` subcommand: the tool-capability promotion harness.
/// Returns PASS(0) / FAIL(1) / INCONCLUSIVE(3).
pub fn run_agent_eval(opts: AgentEvalOptions) -> anyhow::Result<i32> {
    agent_eval::run(agent_eval::EvalConfig {
        addr: opts.addr,
        model: opts.model,
        load_timeout: opts.load_timeout,
        max_steps: opts.max_steps,
        max_tokens: opts.max_tokens,
        receipt_dir: opts.receipt_dir,
    })
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
