//! The agent tool set: sandboxed file/search/shell/network tools, their
//! JSON-schema specs, and the security-critical path resolution.
//!
//! Every tool is confined to a canonical working-directory root (Decision B):
//! a path is joined to the root, canonicalized (resolving symlinks), and
//! required to stay inside the root before any I/O — enforced here in code, not
//! in a prompt. Tool *results* are untrusted data; the loop never treats them as
//! instructions (constraint 6). `run_shell` is cwd-pinned + approval-gated, not a
//! filesystem jail (Decision C / DECISIONS D9).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

/// Risk class — drives the approval gate (Phase 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Risk {
    Read,
    Write,
    Exec,
    Network,
}

impl Risk {
    pub fn label(self) -> &'static str {
        match self {
            Risk::Read => "read",
            Risk::Write => "write",
            Risk::Exec => "exec",
            Risk::Network => "network",
        }
    }
    /// Read-only tools may run without prompting (configurable); the rest gate.
    pub fn needs_approval(self) -> bool {
        self != Risk::Read
    }
}

/// A tool advertised to the model: name, description, JSON-schema params.
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub risk: Risk,
    pub params: Value,
}

/// A tool call the model emitted (already parsed to name + JSON args).
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub args: Value,
}

/// The result of running a tool — text the model consumes as data.
#[derive(Debug, Clone)]
pub enum ToolOutcome {
    Ok(String),
    Err(String),
}

impl ToolOutcome {
    pub fn text(&self) -> &str {
        match self {
            ToolOutcome::Ok(s) | ToolOutcome::Err(s) => s,
        }
    }
    pub fn is_err(&self) -> bool {
        matches!(self, ToolOutcome::Err(_))
    }
}

/// The enforced sandbox: a canonical root + the network/shell policy.
pub struct Sandbox {
    root: PathBuf,
    allow_net: bool,
    shell_timeout: Duration,
}

const MAX_READ_BYTES: usize = 64 * 1024;
const MAX_OUTPUT_BYTES: usize = 16 * 1024;
const MAX_SEARCH_HITS: usize = 100;

impl Sandbox {
    /// Build a sandbox rooted at `root` (canonicalized). Fails if the root does
    /// not resolve to a real directory.
    pub fn new(root: &Path, allow_net: bool, shell_timeout: Duration) -> anyhow::Result<Self> {
        let root = std::fs::canonicalize(root)
            .map_err(|e| anyhow::anyhow!("workdir {} is not accessible: {e}", root.display()))?;
        anyhow::ensure!(
            root.is_dir(),
            "workdir {} is not a directory",
            root.display()
        );
        Ok(Self {
            root,
            allow_net,
            shell_timeout,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a user/model-supplied path against the root and confirm it stays
    /// inside. `must_exist=false` resolves the parent (for write targets that
    /// don't exist yet). This is the path-escape backstop (constraint 5).
    pub fn resolve(&self, raw: &str, must_exist: bool) -> Result<PathBuf, String> {
        if raw.trim().is_empty() {
            return Err("empty path".into());
        }
        let candidate = {
            let p = Path::new(raw);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                self.root.join(p)
            }
        };
        let canon = if must_exist {
            std::fs::canonicalize(&candidate).map_err(|e| format!("cannot access {raw}: {e}"))?
        } else {
            let parent = candidate
                .parent()
                .ok_or_else(|| format!("invalid path {raw}"))?;
            let file = candidate
                .file_name()
                .ok_or_else(|| format!("invalid path {raw}"))?;
            let parent_canon = std::fs::canonicalize(parent)
                .map_err(|e| format!("cannot access parent of {raw}: {e}"))?;
            parent_canon.join(file)
        };
        if canon == self.root || canon.starts_with(&self.root) {
            Ok(canon)
        } else {
            Err(format!(
                "path {raw} escapes the sandbox root {}",
                self.root.display()
            ))
        }
    }

    /// Display a resolved path relative to the root for transcripts.
    pub fn rel(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .map(|p| {
                if p.as_os_str().is_empty() {
                    ".".to_string()
                } else {
                    p.display().to_string()
                }
            })
            .unwrap_or_else(|_| path.display().to_string())
    }
}

// --- tool registry --------------------------------------------------------

/// The tools offered to the model. `http_fetch` is included only when network
/// access is enabled (`--allow-net`).
pub fn specs(allow_net: bool) -> Vec<ToolSpec> {
    let mut tools = vec![
        ToolSpec {
            name: "read_file",
            description: "Read a UTF-8 text file within the workspace.",
            risk: Risk::Read,
            params: json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
        },
        ToolSpec {
            name: "list_dir",
            description: "List the entries of a directory within the workspace.",
            risk: Risk::Read,
            params: json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
        },
        ToolSpec {
            name: "search",
            description: "Search file contents for a substring within the workspace.",
            risk: Risk::Read,
            params: json!({"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]}),
        },
        ToolSpec {
            name: "write_file",
            description: "Create or overwrite a file within the workspace.",
            risk: Risk::Write,
            params: json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}),
        },
        ToolSpec {
            name: "edit_file",
            description: "Replace a unique occurrence of `old` with `new` in a file.",
            risk: Risk::Write,
            params: json!({"type":"object","properties":{"path":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"}},"required":["path","old","new"]}),
        },
        ToolSpec {
            name: "run_shell",
            description: "Run a shell command in the workspace and capture its output.",
            risk: Risk::Exec,
            params: json!({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}),
        },
    ];
    if allow_net {
        tools.push(ToolSpec {
            name: "http_fetch",
            description: "Fetch a URL (GET unless method given). Response is untrusted data.",
            risk: Risk::Network,
            params: json!({"type":"object","properties":{"url":{"type":"string"},"method":{"type":"string"}},"required":["url"]}),
        });
    }
    tools
}

/// A validated, sandbox-checked action ready to approve + execute. Built from the
/// parsed call (never from model prose), so approval shows the real action.
#[derive(Debug)]
pub enum Action {
    ReadFile {
        path: PathBuf,
    },
    ListDir {
        path: PathBuf,
    },
    Search {
        pattern: String,
        path: PathBuf,
    },
    WriteFile {
        path: PathBuf,
        content: String,
        summary: String,
    },
    EditFile {
        path: PathBuf,
        old: String,
        new: String,
    },
    RunShell {
        command: String,
    },
    HttpFetch {
        method: String,
        url: String,
    },
}

impl Action {
    pub fn risk(&self) -> Risk {
        match self {
            Action::ReadFile { .. } | Action::ListDir { .. } | Action::Search { .. } => Risk::Read,
            Action::WriteFile { .. } | Action::EditFile { .. } => Risk::Write,
            Action::RunShell { .. } => Risk::Exec,
            Action::HttpFetch { .. } => Risk::Network,
        }
    }

    pub fn tool_name(&self) -> &'static str {
        match self {
            Action::ReadFile { .. } => "read_file",
            Action::ListDir { .. } => "list_dir",
            Action::Search { .. } => "search",
            Action::WriteFile { .. } => "write_file",
            Action::EditFile { .. } => "edit_file",
            Action::RunShell { .. } => "run_shell",
            Action::HttpFetch { .. } => "http_fetch",
        }
    }

    /// One-line summary of the *call* for the transcript (resolved, not prose).
    pub fn call_line(&self, sandbox: &Sandbox) -> String {
        match self {
            Action::ReadFile { path } => format!("read_file({})", sandbox.rel(path)),
            Action::ListDir { path } => format!("list_dir({})", sandbox.rel(path)),
            Action::Search { pattern, path } => {
                format!("search({pattern:?}, {})", sandbox.rel(path))
            }
            Action::WriteFile { path, content, .. } => {
                format!("write_file({}, {} bytes)", sandbox.rel(path), content.len())
            }
            Action::EditFile { path, .. } => format!("edit_file({})", sandbox.rel(path)),
            Action::RunShell { command } => format!("run_shell({command})"),
            Action::HttpFetch { method, url } => format!("http_fetch({method} {url})"),
        }
    }

    /// The full, verbatim approval text — exactly what will happen.
    pub fn approval_detail(&self, sandbox: &Sandbox) -> String {
        match self {
            Action::WriteFile { path, summary, .. } => {
                format!("write_file → {}\n{summary}", sandbox.rel(path))
            }
            Action::EditFile { path, old, new } => format!(
                "edit_file → {}\n  - {}\n  + {}",
                sandbox.rel(path),
                first_line(old),
                first_line(new),
            ),
            Action::RunShell { command } => format!(
                "run_shell in {}:\n  $ {command}",
                sandbox.rel(sandbox.root())
            ),
            Action::HttpFetch { method, url } => format!("http_fetch:\n  {method} {url}"),
            other => other.call_line(sandbox),
        }
    }

    /// Execute the (already approved) action.
    pub fn execute(&self, sandbox: &Sandbox) -> ToolOutcome {
        match self {
            Action::ReadFile { path } => read_file(path),
            Action::ListDir { path } => list_dir(path),
            Action::Search { pattern, path } => search(pattern, path),
            Action::WriteFile { path, content, .. } => write_file(path, content),
            Action::EditFile { path, old, new } => edit_file(path, old, new),
            Action::RunShell { command } => run_shell(sandbox, command),
            Action::HttpFetch { method, url } => http_fetch(sandbox, method, url),
        }
    }
}

#[derive(Deserialize)]
struct PathArg {
    path: String,
}

/// Validate a parsed tool call against the schema + sandbox. Returns a typed
/// error string (→ tool-error result the model can recover from) rather than
/// panicking, for unknown tools, bad args, or sandbox escapes.
pub fn validate(call: &ToolCall, sandbox: &Sandbox) -> Result<Action, String> {
    let args = &call.args;
    let str_arg = |key: &str| -> Result<String, String> {
        args.get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("{} requires a string `{key}`", call.name))
    };
    match call.name.as_str() {
        "read_file" => {
            let a: PathArg = parse_args(args, &call.name)?;
            Ok(Action::ReadFile {
                path: sandbox.resolve(&a.path, true)?,
            })
        }
        "list_dir" => {
            let a: PathArg = parse_args(args, &call.name)?;
            Ok(Action::ListDir {
                path: sandbox.resolve(&a.path, true)?,
            })
        }
        "search" => {
            let pattern = str_arg("pattern")?;
            let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
            Ok(Action::Search {
                pattern,
                path: sandbox.resolve(path, true)?,
            })
        }
        "write_file" => {
            let path_raw = str_arg("path")?;
            let content = str_arg("content")?;
            let path = sandbox.resolve(&path_raw, false)?;
            let summary = write_summary(&path, &content);
            Ok(Action::WriteFile {
                path,
                content,
                summary,
            })
        }
        "edit_file" => {
            let path = sandbox.resolve(&str_arg("path")?, true)?;
            Ok(Action::EditFile {
                path,
                old: str_arg("old")?,
                new: str_arg("new")?,
            })
        }
        "run_shell" => Ok(Action::RunShell {
            command: str_arg("command")?,
        }),
        "http_fetch" => {
            if !sandbox.allow_net {
                return Err("network tools are disabled (start with --allow-net)".into());
            }
            let method = args
                .get("method")
                .and_then(Value::as_str)
                .unwrap_or("GET")
                .to_ascii_uppercase();
            Ok(Action::HttpFetch {
                method,
                url: str_arg("url")?,
            })
        }
        other => Err(format!("unknown tool `{other}`")),
    }
}

fn parse_args<T: for<'de> Deserialize<'de>>(args: &Value, name: &str) -> Result<T, String> {
    serde_json::from_value(args.clone()).map_err(|e| format!("{name} has invalid arguments: {e}"))
}

// --- execution ------------------------------------------------------------

fn read_file(path: &Path) -> ToolOutcome {
    match std::fs::read(path) {
        Ok(bytes) => {
            let truncated = bytes.len() > MAX_READ_BYTES;
            let slice = &bytes[..bytes.len().min(MAX_READ_BYTES)];
            let mut text = String::from_utf8_lossy(slice).into_owned();
            if truncated {
                text.push_str(&format!("\n…[truncated at {MAX_READ_BYTES} bytes]"));
            }
            ToolOutcome::Ok(text)
        }
        Err(e) => ToolOutcome::Err(format!("read failed: {e}")),
    }
}

fn list_dir(path: &Path) -> ToolOutcome {
    let mut entries = Vec::new();
    let read = match std::fs::read_dir(path) {
        Ok(r) => r,
        Err(e) => return ToolOutcome::Err(format!("list failed: {e}")),
    };
    for entry in read.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let suffix = match entry.file_type() {
            Ok(t) if t.is_dir() => "/",
            _ => "",
        };
        entries.push(format!("{name}{suffix}"));
    }
    entries.sort();
    ToolOutcome::Ok(if entries.is_empty() {
        "(empty)".into()
    } else {
        entries.join("\n")
    })
}

fn search(pattern: &str, root: &Path) -> ToolOutcome {
    let needle = pattern.to_lowercase();
    let mut hits = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if hits.len() >= MAX_SEARCH_HITS {
            break;
        }
        let Ok(read) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in read.flatten() {
            let path = entry.path();
            let ft = entry.file_type();
            if matches!(&ft, Ok(t) if t.is_dir()) {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name == ".git" || name == "target" || name == "node_modules" {
                    continue;
                }
                stack.push(path);
                continue;
            }
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            if bytes.len() > MAX_READ_BYTES * 8 {
                continue;
            }
            let text = String::from_utf8_lossy(&bytes);
            for (n, line) in text.lines().enumerate() {
                if line.to_lowercase().contains(&needle) {
                    hits.push(format!("{}:{}: {}", path.display(), n + 1, line.trim()));
                    if hits.len() >= MAX_SEARCH_HITS {
                        break;
                    }
                }
            }
        }
    }
    ToolOutcome::Ok(if hits.is_empty() {
        format!("no matches for {pattern:?}")
    } else {
        hits.join("\n")
    })
}

fn write_file(path: &Path, content: &str) -> ToolOutcome {
    match std::fs::write(path, content) {
        Ok(()) => ToolOutcome::Ok(format!(
            "wrote {} bytes to {}",
            content.len(),
            path.display()
        )),
        Err(e) => ToolOutcome::Err(format!("write failed: {e}")),
    }
}

fn edit_file(path: &Path, old: &str, new: &str) -> ToolOutcome {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return ToolOutcome::Err(format!("read failed: {e}")),
    };
    let count = content.matches(old).count();
    if count == 0 {
        return ToolOutcome::Err("`old` text not found in file".into());
    }
    if count > 1 {
        return ToolOutcome::Err(format!(
            "`old` text is not unique ({count} occurrences); include more context"
        ));
    }
    let updated = content.replacen(old, new, 1);
    match std::fs::write(path, &updated) {
        Ok(()) => ToolOutcome::Ok(format!("edited {}", path.display())),
        Err(e) => ToolOutcome::Err(format!("write failed: {e}")),
    }
}

fn run_shell(sandbox: &Sandbox, command: &str) -> ToolOutcome {
    // `/bin/sh -c <command>`, cwd pinned to the sandbox root, with a timeout.
    let mut child = match Command::new("/bin/sh")
        .arg("-c")
        .arg(command)
        .current_dir(&sandbox.root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return ToolOutcome::Err(format!("spawn failed: {e}")),
    };
    let deadline = std::time::Instant::now() + sandbox.shell_timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return ToolOutcome::Err(format!(
                        "command timed out after {}s",
                        sandbox.shell_timeout.as_secs()
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return ToolOutcome::Err(format!("wait failed: {e}")),
        }
    }
    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => return ToolOutcome::Err(format!("output failed: {e}")),
    };
    let mut text = String::new();
    let code = output.status.code().unwrap_or(-1);
    text.push_str(&format!("exit: {code}\n"));
    let stdout = clip(&String::from_utf8_lossy(&output.stdout));
    let stderr = clip(&String::from_utf8_lossy(&output.stderr));
    if !stdout.is_empty() {
        text.push_str(&format!("stdout:\n{stdout}\n"));
    }
    if !stderr.is_empty() {
        text.push_str(&format!("stderr:\n{stderr}\n"));
    }
    if output.status.success() {
        ToolOutcome::Ok(text)
    } else {
        ToolOutcome::Err(text)
    }
}

fn http_fetch(sandbox: &Sandbox, method: &str, url: &str) -> ToolOutcome {
    if !sandbox.allow_net {
        return ToolOutcome::Err("network disabled".into());
    }
    // Reuse curl (already a dependency for `pull`); no auto-injected credentials.
    let output = Command::new("curl")
        .args(["-sS", "--max-time", "30", "-X", method, url])
        .current_dir(&sandbox.root)
        .stdin(Stdio::null())
        .output();
    match output {
        Ok(o) if o.status.success() => ToolOutcome::Ok(clip(&String::from_utf8_lossy(&o.stdout))),
        Ok(o) => ToolOutcome::Err(format!(
            "fetch failed: {}",
            clip(&String::from_utf8_lossy(&o.stderr))
        )),
        Err(e) => ToolOutcome::Err(format!("could not run curl: {e}")),
    }
}

// --- helpers --------------------------------------------------------------

fn clip(s: &str) -> String {
    if s.len() <= MAX_OUTPUT_BYTES {
        s.trim_end().to_string()
    } else {
        format!("{}\n…[truncated]", &s[..MAX_OUTPUT_BYTES])
    }
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").to_string()
}

fn write_summary(path: &Path, content: &str) -> String {
    let new_lines = content.lines().count();
    match std::fs::read_to_string(path) {
        Ok(existing) => format!(
            "  overwrite: {} lines → {} lines",
            existing.lines().count(),
            new_lines
        ),
        Err(_) => format!("  create: {new_lines} lines"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox(dir: &Path) -> Sandbox {
        Sandbox::new(dir, false, Duration::from_secs(5)).unwrap()
    }

    fn call(name: &str, args: Value) -> ToolCall {
        ToolCall {
            name: name.into(),
            args,
        }
    }

    #[test]
    fn read_file_happy_path() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello\nworld\n").unwrap();
        let sb = sandbox(dir.path());
        let action = validate(&call("read_file", json!({"path":"a.txt"})), &sb).unwrap();
        let out = action.execute(&sb);
        assert!(matches!(out, ToolOutcome::Ok(ref s) if s.contains("hello")));
    }

    #[test]
    fn read_file_rejects_sandbox_escape() {
        let dir = tempfile::tempdir().unwrap();
        let sb = sandbox(dir.path());
        let err =
            validate(&call("read_file", json!({"path":"../../etc/passwd"})), &sb).unwrap_err();
        assert!(err.contains("escapes") || err.contains("cannot access"));
        // absolute outside-root is refused too
        let err2 = validate(&call("read_file", json!({"path":"/etc/passwd"})), &sb).unwrap_err();
        assert!(err2.contains("escapes") || err2.contains("cannot access"));
    }

    #[test]
    fn write_then_edit_within_sandbox() {
        let dir = tempfile::tempdir().unwrap();
        let sb = sandbox(dir.path());
        let w = validate(
            &call(
                "write_file",
                json!({"path":"out.txt","content":"one\ntwo\n"}),
            ),
            &sb,
        )
        .unwrap();
        assert_eq!(w.risk(), Risk::Write);
        assert!(matches!(w.execute(&sb), ToolOutcome::Ok(_)));
        let e = validate(
            &call(
                "edit_file",
                json!({"path":"out.txt","old":"two","new":"three"}),
            ),
            &sb,
        )
        .unwrap();
        assert!(matches!(e.execute(&sb), ToolOutcome::Ok(_)));
        let body = std::fs::read_to_string(dir.path().join("out.txt")).unwrap();
        assert!(body.contains("three") && !body.contains("two"));
    }

    #[test]
    fn edit_non_unique_is_a_clean_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("d.txt"), "x x x").unwrap();
        let sb = sandbox(dir.path());
        let e = validate(
            &call("edit_file", json!({"path":"d.txt","old":"x","new":"y"})),
            &sb,
        )
        .unwrap();
        assert!(e.execute(&sb).is_err());
    }

    #[test]
    fn unknown_tool_and_bad_args_are_errors_not_panics() {
        let dir = tempfile::tempdir().unwrap();
        let sb = sandbox(dir.path());
        assert!(validate(&call("frobnicate", json!({})), &sb).is_err());
        assert!(validate(&call("read_file", json!({})), &sb).is_err());
    }

    #[test]
    fn http_fetch_offered_only_with_net() {
        assert!(specs(false).iter().all(|t| t.name != "http_fetch"));
        assert!(specs(true).iter().any(|t| t.name == "http_fetch"));
        let dir = tempfile::tempdir().unwrap();
        let sb = sandbox(dir.path()); // allow_net = false
        assert!(validate(&call("http_fetch", json!({"url":"http://x"})), &sb).is_err());
    }

    #[test]
    fn run_shell_runs_in_root_and_captures() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("marker.txt"), "x").unwrap();
        let sb = sandbox(dir.path());
        let a = validate(&call("run_shell", json!({"command":"ls"})), &sb).unwrap();
        assert_eq!(a.risk(), Risk::Exec);
        let out = a.execute(&sb);
        assert!(matches!(out, ToolOutcome::Ok(ref s) if s.contains("marker.txt")));
    }
}
