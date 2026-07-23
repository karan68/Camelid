use std::io::{self, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use serde_json::{json, Value};

use super::client::Client;

const CONTROL_TIMEOUT: Duration = Duration::from_secs(30);

pub struct WorkspaceCliOptions {
    pub addr: SocketAddr,
    pub json: bool,
    pub timeout: Duration,
    pub action: WorkspaceCliAction,
}

pub enum WorkspaceCliAction {
    Ask {
        workspace: PathBuf,
        goal: String,
        thread_id: Option<String>,
        max_steps: usize,
        max_tokens: u32,
        temperature: f32,
    },
    Threads {
        workspace: PathBuf,
    },
    Show {
        workspace: PathBuf,
        thread_id: String,
    },
    Compact {
        workspace: PathBuf,
        thread_id: String,
        undo: bool,
    },
    Delete {
        workspace: PathBuf,
        thread_id: String,
    },
}

pub fn run(options: WorkspaceCliOptions) -> anyhow::Result<i32> {
    anyhow::ensure!(
        options.addr.ip().is_loopback(),
        "Workspace CLI connects only to a loopback Camelid server"
    );
    anyhow::ensure!(
        !options.timeout.is_zero() && options.timeout <= Duration::from_secs(24 * 60 * 60),
        "timeout_seconds must be between 1 and 86400"
    );
    if let WorkspaceCliAction::Ask {
        max_steps,
        max_tokens,
        temperature,
        ..
    } = &options.action
    {
        anyhow::ensure!(
            (1..=32).contains(max_steps),
            "max_steps must be between 1 and 32"
        );
        anyhow::ensure!(
            (1..=1024).contains(max_tokens),
            "max_tokens must be between 1 and 1024"
        );
        anyhow::ensure!(
            temperature.is_finite() && (0.0..=2.0).contains(temperature),
            "temperature must be finite and between 0 and 2"
        );
    }
    let token = crate::workspace_auth::load_token(options.addr).map_err(|error| {
        anyhow::anyhow!(
            "Workspace CLI credential is unavailable for {}: {error}. Start `camelid serve` on that address first",
            options.addr
        )
    })?;
    let client = Client::new(options.addr);
    match options.action {
        WorkspaceCliAction::Ask {
            workspace,
            goal,
            thread_id,
            max_steps,
            max_tokens,
            temperature,
        } => ask(
            &client,
            &token,
            options.json,
            options.timeout,
            workspace,
            goal,
            thread_id,
            max_steps,
            max_tokens,
            temperature,
        ),
        WorkspaceCliAction::Threads { workspace } => {
            let path = workspace_query("/api/agent/workspace/threads", &workspace);
            let body = request(&client, &token, "GET", &path, None, &[200])?;
            if options.json {
                println!("{body}");
            } else {
                render_threads(&body);
            }
            Ok(0)
        }
        WorkspaceCliAction::Show {
            workspace,
            thread_id,
        } => {
            let path = workspace_query(
                &format!(
                    "/api/agent/workspace/threads/{}",
                    encode_component(&thread_id)
                ),
                &workspace,
            );
            let body = request(&client, &token, "GET", &path, None, &[200])?;
            if options.json {
                println!("{body}");
            } else {
                render_thread(&body);
            }
            Ok(0)
        }
        WorkspaceCliAction::Compact {
            workspace,
            thread_id,
            undo,
        } => {
            let path = workspace_query(
                &format!(
                    "/api/agent/workspace/threads/{}/compact",
                    encode_component(&thread_id)
                ),
                &workspace,
            );
            let method = if undo { "DELETE" } else { "POST" };
            let body = request(&client, &token, method, &path, None, &[200])?;
            if options.json {
                println!("{body}");
            } else if undo {
                println!("Restored the previous compaction state for {thread_id}.");
            } else {
                let archived = body["archived_turns"].as_u64().unwrap_or(0);
                println!("Compacted {archived} turn(s) in {thread_id}.");
            }
            Ok(0)
        }
        WorkspaceCliAction::Delete {
            workspace,
            thread_id,
        } => {
            let path = workspace_query(
                &format!(
                    "/api/agent/workspace/threads/{}",
                    encode_component(&thread_id)
                ),
                &workspace,
            );
            request(&client, &token, "DELETE", &path, None, &[204])?;
            if options.json {
                println!("{}", json!({"deleted": true, "thread_id": thread_id}));
            } else {
                println!("Deleted {thread_id}.");
            }
            Ok(0)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn ask(
    client: &Client,
    token: &str,
    json_output: bool,
    timeout: Duration,
    workspace: PathBuf,
    goal: String,
    thread_id: Option<String>,
    max_steps: usize,
    max_tokens: u32,
    temperature: f32,
) -> anyhow::Result<i32> {
    let mut payload = json!({
        "workspace": workspace,
        "goal": goal,
        "max_steps": max_steps,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "allow_writes": false
    });
    if let Some(thread_id) = thread_id {
        payload["thread_id"] = json!(thread_id);
    }
    let created = request(
        client,
        token,
        "POST",
        "/api/agent/workspace/sessions",
        Some(&payload),
        &[201],
    )?;
    let session_id = created["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Workspace session response omitted id"))?
        .to_string();
    if !json_output {
        eprintln!("Workspace thread: {session_id}");
    }

    let cancel = AtomicBool::new(false);
    let mut state = AskState::default();
    let events_path = format!(
        "/api/agent/workspace/sessions/{}/events",
        encode_component(&session_id)
    );
    let stream = client.workspace_events(&events_path, token, &cancel, timeout, |event| {
        if json_output {
            println!("{event}");
        }
        state.observe(event, json_output)
    });
    if let Err(error) = stream {
        let _ = cancel_session(client, token, &session_id);
        return Err(error);
    }
    if state.saw_delta && !json_output {
        println!();
    } else if !json_output {
        if let Some(answer) = &state.answer {
            println!("{answer}");
        }
    }
    if let Some(error) = state.error {
        let _ = cancel_session(client, token, &session_id);
        anyhow::bail!(error);
    }
    let outcome = state
        .outcome
        .ok_or_else(|| anyhow::anyhow!("Workspace event stream ended before a terminal event"))?;
    Ok(outcome_exit_code(&outcome))
}

fn cancel_session(client: &Client, token: &str, session_id: &str) -> anyhow::Result<()> {
    let path = format!(
        "/api/agent/workspace/sessions/{}",
        encode_component(session_id)
    );
    request(client, token, "DELETE", &path, None, &[200, 204]).map(|_| ())
}

fn request(
    client: &Client,
    token: &str,
    method: &str,
    path: &str,
    body: Option<&Value>,
    expected: &[u16],
) -> anyhow::Result<Value> {
    let (status, response) =
        client.workspace_request(method, path, body, token, CONTROL_TIMEOUT)?;
    if expected.contains(&status) {
        return Ok(response);
    }
    let code = response
        .pointer("/error/code")
        .and_then(Value::as_str)
        .unwrap_or("workspace_request_failed");
    let message = response
        .pointer("/error/message")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("Workspace request returned HTTP {status}"));
    anyhow::bail!("{code}: {message}")
}

#[derive(Default)]
struct AskState {
    saw_delta: bool,
    answer: Option<String>,
    outcome: Option<String>,
    error: Option<String>,
}

impl AskState {
    fn observe(&mut self, event: &Value, json_output: bool) -> bool {
        match event["event"].as_str() {
            Some("model.delta") => {
                if let Some(content) = event["content"].as_str() {
                    self.saw_delta = true;
                    if !json_output {
                        print!("{content}");
                        let _ = io::stdout().flush();
                    }
                }
            }
            Some("model.answer") => {
                self.answer = event["content"].as_str().map(str::to_string);
            }
            Some("tool.call") if !json_output => {
                if let Some(detail) = event["detail"].as_str() {
                    eprintln!("  -> {detail}");
                }
            }
            Some("memory.compacted") if !json_output => {
                eprintln!("  context compacted");
            }
            Some("session.notice") if !json_output => {
                if let Some(content) = event["content"].as_str() {
                    eprintln!("  {content}");
                }
            }
            Some("approval.required") => {
                self.error = Some(
                    "Workspace unexpectedly requested approval; the CLI cancelled the read-only session"
                        .to_string(),
                );
                return true;
            }
            Some("session.error") => {
                self.error = event["message"].as_str().map(str::to_string);
                return true;
            }
            Some("session.finished") => {
                self.outcome = event["outcome"].as_str().map(str::to_string);
                return true;
            }
            _ => {}
        }
        false
    }
}

fn outcome_exit_code(outcome: &str) -> i32 {
    match outcome {
        "answered" => 0,
        "aborted" | "step_capped" | "repeated" => 3,
        _ => 1,
    }
}

fn workspace_query(endpoint: &str, workspace: &std::path::Path) -> String {
    format!(
        "{endpoint}?workspace={}",
        encode_component(&workspace.to_string_lossy())
    )
}

fn encode_component(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.as_bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(char::from(*byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn render_threads(body: &Value) {
    let threads = body["threads"].as_array().cloned().unwrap_or_default();
    if threads.is_empty() {
        println!("No saved Workspace conversations.");
        return;
    }
    for thread in threads {
        println!(
            "{}  {}  ({} turn{})",
            thread["id"].as_str().unwrap_or("unknown"),
            thread["title"].as_str().unwrap_or("Untitled"),
            thread["turn_count"].as_u64().unwrap_or(0),
            if thread["turn_count"].as_u64() == Some(1) {
                ""
            } else {
                "s"
            }
        );
    }
}

fn render_thread(body: &Value) {
    println!(
        "# {}\n",
        body["thread"]["title"]
            .as_str()
            .unwrap_or("Workspace conversation")
    );
    for turn in body["turns"].as_array().cloned().unwrap_or_default() {
        println!("You: {}", turn["user_text"].as_str().unwrap_or_default());
        let answer = turn["assistant_text"].as_str().unwrap_or_default();
        if !answer.is_empty() {
            println!("Camelid: {answer}");
        }
        let outcome = turn["terminal_outcome"].as_str().unwrap_or_default();
        if outcome != "answered" {
            println!("[{outcome}]");
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_encoding_preserves_no_path_or_parameter_separators() {
        assert_eq!(
            encode_component(r"C:\work & notes"),
            "C%3A%5Cwork%20%26%20notes"
        );
        assert_eq!(encode_component("café"), "caf%C3%A9");
        assert!(!encode_component("x&other=y").contains('&'));
        assert!(!encode_component("../../thread").contains('/'));
    }

    #[test]
    fn terminal_outcomes_have_stable_exit_codes() {
        assert_eq!(outcome_exit_code("answered"), 0);
        assert_eq!(outcome_exit_code("driver_error"), 1);
        assert_eq!(outcome_exit_code("step_capped"), 3);
        assert_eq!(outcome_exit_code("aborted"), 3);
    }

    #[test]
    fn invalid_limits_fail_before_credential_discovery() {
        let error = run(WorkspaceCliOptions {
            addr: "127.0.0.1:8181".parse().unwrap(),
            json: false,
            timeout: Duration::ZERO,
            action: WorkspaceCliAction::Threads {
                workspace: PathBuf::from("."),
            },
        })
        .unwrap_err();
        assert!(error.to_string().contains("timeout_seconds"));
    }

    #[test]
    fn event_observer_captures_answer_and_terminal_state() {
        let mut state = AskState::default();
        assert!(!state.observe(&json!({"event":"model.answer","content":"done"}), true));
        assert!(state.observe(
            &json!({"event":"session.finished","outcome":"answered"}),
            true
        ));
        assert_eq!(state.answer.as_deref(), Some("done"));
        assert_eq!(state.outcome.as_deref(), Some("answered"));
    }

    #[test]
    fn approval_event_fails_closed() {
        let mut state = AskState::default();
        assert!(state.observe(
            &json!({"event":"approval.required","tool":"write_file"}),
            true
        ));
        assert!(state.error.unwrap().contains("cancelled"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn authenticated_client_reaches_the_live_workspace_router() {
        let token = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = crate::api::router_with_workspace_cli_token_for_tests(token);
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let token = token.to_string();
        let response = tokio::task::spawn_blocking(move || {
            Client::new(addr).workspace_request(
                "GET",
                "/api/agent/workspace/threads?workspace=definitely-missing",
                None,
                &token,
                Duration::from_secs(5),
            )
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.0, 400);
        assert_eq!(response.1["error"]["code"], "workspace_root_not_accessible");
        server.abort();
    }
}
