//! Minimal MCP (Model Context Protocol) client — stdio transport, v1.
//!
//! Lets a user extend agent mode with third-party tools (git, databases,
//! issue trackers, …) without any of them being compiled in. Servers are
//! declared in a `camelid.mcp.json` at the workspace root; each is spawned as a
//! child process speaking JSON-RPC 2.0 over stdin/stdout, and each tool it
//! advertises is adapted into a [`ToolSpec`] that flows through the *same*
//! `validate` → tier → `Approver` → execute path as a native tool.
//!
//! # Posture
//!
//! An MCP server is untrusted third-party code speaking a protocol. Three
//! consequences, all enforced here rather than described to the model:
//!
//! 1. **Its tool descriptions and its output are data.** A server that claims
//!    its tools need no approval, or whose output says the user pre-authorised
//!    something, is describing — not deciding. Output comes back as a normal
//!    tool result and is fenced as untrusted like any other.
//! 2. **Off unless asked for.** Disabled by default, and refused outright under
//!    `CAMELID_PRODUCTION`. Enabled per-session with `--allow-mcp`.
//! 3. **Never able to impersonate a native tool.** Every tool is namespaced
//!    `mcp__<server>__<tool>`, and a server whose name would collide with an
//!    existing native tool is rejected at load.
//!
//! MCP tools are classified [`Risk::Exec`]: an MCP tool can do anything the
//! server can do, so it is always gated, and `--auto-approve` does *not*
//! promote it (only `--yolo` does, which production already refuses).

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

use super::tools::{Risk, Sandbox, ToolSpec};

/// The config file, read from the workspace root only.
pub const CONFIG_FILE: &str = "camelid.mcp.json";

/// Namespace every MCP tool carries, so one can never shadow a native tool.
pub const PREFIX: &str = "mcp__";

/// How long to wait for a server to answer one request.
const CALL_TIMEOUT: Duration = Duration::from_secs(30);
/// Handshake and tool listing should be quick; a server that is not ready in
/// this long is treated as unusable rather than hanging the session.
const INIT_TIMEOUT: Duration = Duration::from_secs(10);

/// Cap on a single MCP tool result, mirroring the native output cap.
const MAX_RESULT_BYTES: usize = 16 * 1024;

// --- config -----------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    /// Executable to spawn.
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment for the child. Inherited env is passed through.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: BTreeMap<String, ServerConfig>,
}

/// Read `camelid.mcp.json` from the workspace root.
///
/// Resolved through the sandbox like any other path. A missing file is not an
/// error — it is the normal case.
pub fn load_config(sandbox: &Sandbox) -> Result<Option<McpConfig>, String> {
    let Ok(path) = sandbox.resolve(CONFIG_FILE, true) else {
        return Ok(None);
    };
    let raw = match std::fs::read_to_string(&path) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    let cfg: McpConfig =
        serde_json::from_str(&raw).map_err(|e| format!("{CONFIG_FILE} is not valid JSON: {e}"))?;
    for name in cfg.servers.keys() {
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            || name.is_empty()
        {
            return Err(format!(
                "{CONFIG_FILE}: server name {name:?} must be alphanumeric, '-' or '_'"
            ));
        }
    }
    Ok(Some(cfg))
}

// --- a single stdio server ---------------------------------------------------

/// One spawned MCP server. Reads run on a helper thread so a server that stops
/// talking cannot wedge the agent — every receive is bounded by a timeout.
struct Server {
    name: String,
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
    next_id: u64,
}

impl Server {
    fn spawn(name: &str, cfg: &ServerConfig, cwd: &std::path::Path) -> Result<Self, String> {
        let mut cmd = Command::new(&cfg.command);
        cmd.args(&cfg.args)
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // The protocol is on stdout; a server's logging on stderr is not
            // ours to interleave into the terminal.
            .stderr(Stdio::null());
        for (k, v) in &cfg.env {
            cmd.env(k, v);
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("could not start MCP server '{name}' ({}): {e}", cfg.command))?;

        let stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;
        let (tx, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });

        Ok(Self {
            name: name.to_string(),
            child,
            stdin,
            lines,
            next_id: 1,
        })
    }

    /// One JSON-RPC round trip. Notifications and unrelated messages are
    /// skipped until the matching id arrives or the deadline passes.
    fn request(&mut self, method: &str, params: Value, timeout: Duration) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        let req = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        writeln!(self.stdin, "{req}").map_err(|e| format!("{}: write failed: {e}", self.name))?;
        self.stdin
            .flush()
            .map_err(|e| format!("{}: flush failed: {e}", self.name))?;

        let deadline = std::time::Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                return Err(format!("{}: no response to {method} in time", self.name));
            }
            let line = self
                .lines
                .recv_timeout(left)
                .map_err(|_| format!("{}: no response to {method} in time", self.name))?;
            let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                continue; // not JSON — a stray log line; ignore
            };
            if msg.get("id").and_then(Value::as_u64) != Some(id) {
                continue; // a notification or another response
            }
            if let Some(err) = msg.get("error") {
                let m = err
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error");
                return Err(format!("{}: {m}", self.name));
            }
            return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
        }
    }

    fn notify(&mut self, method: &str, params: Value) {
        let msg = json!({"jsonrpc":"2.0","method":method,"params":params});
        let _ = writeln!(self.stdin, "{msg}");
        let _ = self.stdin.flush();
    }

    fn initialize(&mut self) -> Result<(), String> {
        self.request(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "camelid", "version": env!("CARGO_PKG_VERSION")},
            }),
            INIT_TIMEOUT,
        )?;
        self.notify("notifications/initialized", json!({}));
        Ok(())
    }

    fn list_tools(&mut self) -> Result<Vec<(String, String, Value)>, String> {
        let res = self.request("tools/list", json!({}), INIT_TIMEOUT)?;
        let items = res
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::new();
        for t in items {
            let Some(name) = t.get("name").and_then(Value::as_str) else {
                continue;
            };
            let desc = t
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("(no description)")
                .to_string();
            let schema = t
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| json!({"type":"object"}));
            out.push((name.to_string(), desc, schema));
        }
        Ok(out)
    }

    fn call(&mut self, tool: &str, args: &Value) -> Result<String, String> {
        let res = self.request(
            "tools/call",
            json!({"name": tool, "arguments": args}),
            CALL_TIMEOUT,
        )?;
        // MCP returns content blocks; flatten the text ones. A server may also
        // signal a tool-level failure via isError while still returning 200.
        let mut text = String::new();
        if let Some(blocks) = res.get("content").and_then(Value::as_array) {
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = b.get("text").and_then(Value::as_str) {
                            text.push_str(t);
                            text.push('\n');
                        }
                    }
                    Some(other) => text.push_str(&format!("[{other} content omitted]\n")),
                    None => {}
                }
            }
        }
        if text.is_empty() {
            text = res.to_string();
        }
        if text.len() > MAX_RESULT_BYTES {
            let mut end = MAX_RESULT_BYTES;
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            text.truncate(end);
            text.push_str("\n…[truncated]");
        }
        if res.get("isError").and_then(Value::as_bool) == Some(true) {
            return Err(text);
        }
        Ok(text)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// --- the process-wide registry ----------------------------------------------

/// One adapted tool: the namespaced name the model sees, plus where it came from.
struct Entry {
    /// `mcp__<server>__<tool>`
    public: String,
    server: String,
    tool: String,
    description: String,
    schema: Value,
}

#[derive(Default)]
struct Registry {
    servers: BTreeMap<String, Server>,
    tools: Vec<Entry>,
}

fn registry() -> &'static Mutex<Option<Registry>> {
    static R: OnceLock<Mutex<Option<Registry>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(None))
}

/// Whether MCP tools should be advertised and callable right now.
pub fn is_enabled() -> bool {
    registry()
        .lock()
        .map(|r| r.as_ref().is_some_and(|r| !r.tools.is_empty()))
        .unwrap_or(false)
}

/// Start every configured server and adopt its tools.
///
/// Returns the number of tools adopted. Errors here are reported to the caller
/// and are never fatal to the session: a broken MCP config should cost you MCP,
/// not your agent.
pub fn configure(
    sandbox: &Sandbox,
    allow_mcp: bool,
    production: bool,
    native_tool_names: &[String],
) -> Result<usize, String> {
    if !allow_mcp {
        return Ok(0);
    }
    if production {
        return Err(
            "MCP is refused under CAMELID_PRODUCTION: it would expose third-party tools to an \
             unattended agent. Unset CAMELID_PRODUCTION to use --allow-mcp."
                .into(),
        );
    }
    let Some(cfg) = load_config(sandbox)? else {
        return Ok(0);
    };

    let mut reg = Registry::default();
    let mut problems: Vec<String> = Vec::new();

    for (name, sc) in &cfg.servers {
        let mut server = match Server::spawn(name, sc, sandbox.root()) {
            Ok(s) => s,
            Err(e) => {
                problems.push(e);
                continue;
            }
        };
        if let Err(e) = server.initialize() {
            problems.push(format!("MCP server '{name}' failed to initialize: {e}"));
            continue;
        }
        let listed = match server.list_tools() {
            Ok(t) => t,
            Err(e) => {
                problems.push(format!("MCP server '{name}' would not list tools: {e}"));
                continue;
            }
        };
        for (tool, description, schema) in listed {
            let public = format!("{PREFIX}{name}__{tool}");
            // Belt and braces: the prefix already makes collision impossible,
            // but assert it rather than assume it.
            if native_tool_names.contains(&public) {
                problems.push(format!(
                    "MCP tool '{public}' collides with a native tool; skipped"
                ));
                continue;
            }
            reg.tools.push(Entry {
                public,
                server: name.clone(),
                tool,
                description,
                schema,
            });
        }
        reg.servers.insert(name.clone(), server);
    }

    let adopted = reg.tools.len();
    *registry().lock().map_err(|_| "mcp registry poisoned")? = Some(reg);

    if !problems.is_empty() {
        return Err(problems.join("; "));
    }
    Ok(adopted)
}

/// Stop every server and forget its tools.
pub fn shutdown() {
    if let Ok(mut r) = registry().lock() {
        *r = None; // Server::drop kills each child
    }
}

/// The adapted tool specs to advertise alongside the native ones.
pub fn specs() -> Vec<ToolSpec> {
    let Ok(guard) = registry().lock() else {
        return Vec::new();
    };
    let Some(reg) = guard.as_ref() else {
        return Vec::new();
    };
    reg.tools
        .iter()
        .map(|e| ToolSpec {
            name: e.public.clone(),
            description: format!("[MCP: {}] {}", e.server, e.description),
            // An MCP tool can do whatever its server can. Always gated, and not
            // promoted by --auto-approve.
            risk: Risk::Exec,
            params: e.schema.clone(),
        })
        .collect()
}

/// Whether a namespaced name is one this registry currently serves.
pub fn has_tool(public: &str) -> bool {
    let Ok(guard) = registry().lock() else {
        return false;
    };
    guard
        .as_ref()
        .is_some_and(|r| r.tools.iter().any(|e| e.public == public))
}

/// Invoke a namespaced MCP tool. The returned text is untrusted data and is
/// surfaced to the model through the same fenced tool-result path as any other.
pub fn call(public: &str, args: &Value) -> Result<String, String> {
    let mut guard = registry()
        .lock()
        .map_err(|_| "mcp registry poisoned".to_string())?;
    let reg = guard.as_mut().ok_or("MCP is not enabled")?;
    let (server_name, tool) = reg
        .tools
        .iter()
        .find(|e| e.public == public)
        .map(|e| (e.server.clone(), e.tool.clone()))
        .ok_or_else(|| format!("unknown MCP tool '{public}'"))?;
    let server = reg
        .servers
        .get_mut(&server_name)
        .ok_or_else(|| format!("MCP server '{server_name}' is gone"))?;
    server.call(&tool, args)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The registry is process-wide, so tests that touch it must not run
    /// concurrently with each other or with the tool-set pins in `tools`.
    pub(crate) fn registry_lock() -> std::sync::MutexGuard<'static, ()> {
        static L: OnceLock<Mutex<()>> = OnceLock::new();
        L.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn sandbox(root: &std::path::Path) -> Sandbox {
        Sandbox::new(root, false, Duration::from_secs(5)).unwrap()
    }

    #[test]
    fn missing_config_is_not_an_error() {
        let d = tempfile::tempdir().unwrap();
        assert!(load_config(&sandbox(d.path())).unwrap().is_none());
    }

    #[test]
    fn config_parses_servers() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join(CONFIG_FILE),
            r#"{"servers":{"git":{"command":"mcp-git","args":["--repo","."]}}}"#,
        )
        .unwrap();
        let cfg = load_config(&sandbox(d.path())).unwrap().unwrap();
        assert_eq!(cfg.servers.len(), 1);
        let git = &cfg.servers["git"];
        assert_eq!(git.command, "mcp-git");
        assert_eq!(git.args, vec!["--repo", "."]);
    }

    #[test]
    fn malformed_config_is_a_clean_error() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join(CONFIG_FILE), "{not json").unwrap();
        assert!(load_config(&sandbox(d.path())).is_err());
    }

    /// A server name is spliced into every tool name the model sees, so it must
    /// not be able to carry separators or escape the namespace.
    #[test]
    fn hostile_server_names_are_rejected() {
        let d = tempfile::tempdir().unwrap();
        for bad in ["../evil", "a b", "read_file/../x", ""] {
            std::fs::write(
                d.path().join(CONFIG_FILE),
                format!(r#"{{"servers":{{"{bad}":{{"command":"x"}}}}}}"#),
            )
            .unwrap();
            assert!(
                load_config(&sandbox(d.path())).is_err(),
                "server name {bad:?} should be refused"
            );
        }
    }

    #[test]
    fn disabled_by_default() {
        let _guard = registry_lock();
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join(CONFIG_FILE),
            r#"{"servers":{"x":{"command":"true"}}}"#,
        )
        .unwrap();
        // allow_mcp = false → nothing is spawned, nothing is adopted.
        assert_eq!(configure(&sandbox(d.path()), false, false, &[]).unwrap(), 0);
        assert!(!is_enabled());
        assert!(specs().is_empty());
    }

    #[test]
    fn refused_under_production() {
        let _guard = registry_lock();
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join(CONFIG_FILE),
            r#"{"servers":{"x":{"command":"true"}}}"#,
        )
        .unwrap();
        let err = configure(&sandbox(d.path()), true, true, &[]).unwrap_err();
        assert!(err.contains("CAMELID_PRODUCTION"), "{err}");
        assert!(!is_enabled());
    }

    #[test]
    fn calling_without_a_registry_is_an_error_not_a_panic() {
        let _guard = registry_lock();
        shutdown();
        assert!(!has_tool("mcp__x__y"));
        assert!(call("mcp__x__y", &json!({})).is_err());
    }

    // --- end-to-end against a stub stdio server ---
    //
    // These share the process-wide registry, so they run under one #[test] to
    // keep cargo's thread-per-test from interleaving configure/shutdown.

    /// A tiny MCP server: initialize, tools/list, tools/call over stdio.
    #[cfg(unix)]
    const STUB: &str = r#"
import sys, json
def send(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
sys.stderr.write("a log line that is not JSON\n")
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    msg = json.loads(line)
    m, i = msg.get("method"), msg.get("id")
    if m == "initialize":
        send({"jsonrpc":"2.0","id":i,"result":{"protocolVersion":"2024-11-05"}})
    elif m == "tools/list":
        send({"jsonrpc":"2.0","id":i,"result":{"tools":[
            {"name":"echo","description":"Echo a value.",
             "inputSchema":{"type":"object","properties":{"v":{"type":"string"}}}},
            {"name":"boom","description":"Always fails."}
        ]}})
    elif m == "tools/call":
        p = msg.get("params", {})
        if p.get("name") == "boom":
            send({"jsonrpc":"2.0","id":i,"result":{"isError":True,
                  "content":[{"type":"text","text":"it broke"}]}})
        else:
            v = p.get("arguments", {}).get("v", "")
            send({"jsonrpc":"2.0","id":i,"result":{"content":[
                {"type":"text","text":"echo:" + str(v)}]}})
"#;

    #[cfg(unix)]
    fn write_stub_config(dir: &std::path::Path) {
        let script = dir.join("stub_server.py");
        std::fs::write(&script, STUB).unwrap();
        std::fs::write(
            dir.join(CONFIG_FILE),
            format!(
                r#"{{"servers":{{"stub":{{"command":"python3","args":["{}"]}}}}}}"#,
                script.display()
            ),
        )
        .unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn stub_server_end_to_end() {
        let _guard = registry_lock();
        if std::process::Command::new("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| !s.success())
            .unwrap_or(true)
        {
            eprintln!("skipping: no python3");
            return;
        }

        let d = tempfile::tempdir().unwrap();
        write_stub_config(d.path());
        let sb = sandbox(d.path());

        shutdown();
        let n = configure(&sb, true, false, &[]).unwrap();
        assert_eq!(n, 2, "both stub tools should be adopted");
        assert!(is_enabled());

        // Namespaced, and carrying the server's own description.
        let adopted = specs();
        let names: Vec<&str> = adopted.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"mcp__stub__echo"));
        assert!(names.contains(&"mcp__stub__boom"));
        assert!(names.iter().all(|n| n.starts_with(PREFIX)));

        // Exec tier: always gated, and --auto-approve does not promote it.
        let echo = adopted.iter().find(|s| s.name.ends_with("echo")).unwrap();
        assert_eq!(echo.risk, Risk::Exec);
        assert!(echo.risk.needs_approval());
        assert!(echo.description.contains("Echo a value."));

        // A real call round-trips, tolerating the server's non-JSON stderr noise.
        let out = call("mcp__stub__echo", &json!({"v":"hi"})).unwrap();
        assert!(out.contains("echo:hi"), "{out}");

        // A tool-level failure comes back as an error, not a silent success.
        let err = call("mcp__stub__boom", &json!({})).unwrap_err();
        assert!(err.contains("it broke"), "{err}");

        // Unknown names are refused.
        assert!(call("mcp__stub__nope", &json!({})).is_err());

        shutdown();
        assert!(!is_enabled());
        assert!(specs().is_empty());
    }

    /// A process that is not an MCP server at all must not become one. `cat`
    /// echoes our own request back — id and all — so the handshake "succeeds"
    /// with a null result. What matters is that it yields no tools and leaves
    /// MCP disabled rather than adopting garbage.
    #[cfg(unix)]
    #[test]
    fn an_echo_server_adopts_no_tools() {
        let _guard = registry_lock();
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join(CONFIG_FILE),
            r#"{"servers":{"echo":{"command":"cat"}}}"#,
        )
        .unwrap();
        shutdown();
        assert_eq!(configure(&sandbox(d.path()), true, false, &[]).unwrap(), 0);
        assert!(!is_enabled());
        assert!(specs().is_empty());
        shutdown();
    }

    /// A server that never answers must not wedge the agent: the handshake is
    /// bounded by INIT_TIMEOUT, not by the server's goodwill.
    #[cfg(unix)]
    #[test]
    fn a_silent_server_times_out_instead_of_hanging() {
        let _guard = registry_lock();
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join(CONFIG_FILE),
            r#"{"servers":{"mute":{"command":"sleep","args":["120"]}}}"#,
        )
        .unwrap();
        shutdown();
        let started = std::time::Instant::now();
        let res = configure(&sandbox(d.path()), true, false, &[]);
        let elapsed = started.elapsed();
        assert!(res.is_err(), "a mute server should be reported");
        assert!(res.unwrap_err().contains("in time"));
        assert!(elapsed >= INIT_TIMEOUT, "returned too early: {elapsed:?}");
        assert!(elapsed < INIT_TIMEOUT * 3, "took {elapsed:?}");
        assert!(!is_enabled());
        shutdown();
    }
}
