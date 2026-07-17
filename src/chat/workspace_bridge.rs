//! Synchronous bridge between the UI-agnostic agent loop and an external
//! controller such as the Web Workspace API.
//!
//! The agent loop remains the sole tool-execution owner. This module only
//! transports rendered events and approval decisions over bounded channels.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{net::SocketAddr, path::PathBuf};

use serde::{Deserialize, Serialize};

use super::agent::{
    run_loop, AgentConfig, AgentMsg, Approver, Decision, LiveDriver, LoopEnd, Policy, Reporter,
};
use super::audit::NoopSink;
use super::client::Client;
use super::shell_sandbox::ShellSandbox;
use super::tools::{self, Action, Sandbox, ToolOutcome, ToolProfile};

const APPROVAL_POLL: Duration = Duration::from_millis(25);
const DEFAULT_APPROVAL_TIMEOUT: Duration = Duration::from_secs(5 * 60);

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub(crate) enum WorkspaceEvent {
    #[serde(rename = "session.started")]
    Started { workspace: String, model_id: String },
    #[serde(rename = "model.delta")]
    ModelDelta { content: String },
    #[serde(rename = "model.answer")]
    ModelAnswer { content: String },
    #[serde(rename = "tool.call")]
    ToolCall { detail: String },
    #[serde(rename = "approval.required")]
    ApprovalRequired {
        approval_id: String,
        tool: String,
        risk: String,
        detail: String,
    },
    #[serde(rename = "tool.result")]
    ToolResult {
        tool: String,
        outcome: &'static str,
        content: String,
    },
    #[serde(rename = "session.notice")]
    Notice { content: String },
    #[serde(rename = "session.finished")]
    Finished { outcome: &'static str },
    #[serde(rename = "session.error")]
    Error { message: String },
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WorkspaceDecisionKind {
    AllowOnce,
    AlwaysTool,
    Deny,
    Abort,
}

#[derive(Debug)]
pub(crate) struct WorkspaceDecision {
    pub approval_id: String,
    pub decision: WorkspaceDecisionKind,
}

pub(crate) struct WorkspaceBridgeWorker {
    pub reporter: WorkspaceReporter,
    pub approver: WorkspaceApprover,
    pub cancel: Arc<AtomicBool>,
}

pub(crate) struct WorkspaceBridgeClient {
    pub events: Receiver<WorkspaceEvent>,
    decisions: SyncSender<WorkspaceDecision>,
    cancel: Arc<AtomicBool>,
    pending_approval: Arc<Mutex<Option<String>>>,
}

pub(crate) struct WorkspaceRunConfig {
    pub addr: SocketAddr,
    pub workspace: PathBuf,
    pub goal: String,
    pub model_id: String,
    pub family: String,
    pub max_steps: usize,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl WorkspaceBridgeClient {
    #[cfg(test)]
    pub fn try_decide(
        &self,
        approval_id: String,
        decision: WorkspaceDecisionKind,
    ) -> Result<(), &'static str> {
        if self
            .pending_approval
            .lock()
            .map_err(|_| "the approval state is unavailable")?
            .as_deref()
            != Some(approval_id.as_str())
        {
            return Err("the approval is stale or does not belong to this session");
        }
        match self.decisions.try_send(WorkspaceDecision {
            approval_id,
            decision,
        }) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err("a decision is already pending"),
            Err(TrySendError::Disconnected(_)) => Err("the workspace session has ended"),
        }
    }

    #[cfg(test)]
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    pub fn into_parts(self) -> (Receiver<WorkspaceEvent>, WorkspaceBridgeControl) {
        (
            self.events,
            WorkspaceBridgeControl {
                decisions: self.decisions,
                cancel: self.cancel,
                pending_approval: self.pending_approval,
            },
        )
    }
}

#[derive(Clone)]
pub(crate) struct WorkspaceBridgeControl {
    decisions: SyncSender<WorkspaceDecision>,
    cancel: Arc<AtomicBool>,
    pending_approval: Arc<Mutex<Option<String>>>,
}

impl WorkspaceBridgeControl {
    pub fn try_decide(
        &self,
        approval_id: String,
        decision: WorkspaceDecisionKind,
    ) -> Result<(), &'static str> {
        if self
            .pending_approval
            .lock()
            .map_err(|_| "the approval state is unavailable")?
            .as_deref()
            != Some(approval_id.as_str())
        {
            return Err("the approval is stale or does not belong to this session");
        }
        match self.decisions.try_send(WorkspaceDecision {
            approval_id,
            decision,
        }) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err("a decision is already pending"),
            Err(TrySendError::Disconnected(_)) => Err("the workspace session has ended"),
        }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
}

pub(crate) fn bridge(capacity: usize) -> (WorkspaceBridgeWorker, WorkspaceBridgeClient) {
    bridge_with_timeout(capacity, DEFAULT_APPROVAL_TIMEOUT)
}

fn bridge_with_timeout(
    capacity: usize,
    approval_timeout: Duration,
) -> (WorkspaceBridgeWorker, WorkspaceBridgeClient) {
    let capacity = capacity.max(1);
    let (event_tx, event_rx) = sync_channel(capacity);
    let (decision_tx, decision_rx) = sync_channel(1);
    let cancel = Arc::new(AtomicBool::new(false));
    let pending_approval = Arc::new(Mutex::new(None));
    (
        WorkspaceBridgeWorker {
            reporter: WorkspaceReporter {
                events: event_tx.clone(),
            },
            approver: WorkspaceApprover {
                events: event_tx,
                decisions: decision_rx,
                cancel: Arc::clone(&cancel),
                pending_approval: Arc::clone(&pending_approval),
                approval_timeout,
            },
            cancel: Arc::clone(&cancel),
        },
        WorkspaceBridgeClient {
            events: event_rx,
            decisions: decision_tx,
            cancel,
            pending_approval,
        },
    )
}

#[derive(Clone)]
pub(crate) struct WorkspaceReporter {
    events: SyncSender<WorkspaceEvent>,
}

impl WorkspaceReporter {
    fn send(&self, event: WorkspaceEvent) {
        // A bounded blocking send provides backpressure without unbounded memory.
        // A dropped receiver ends delivery; the agent loop remains cancellable.
        let _ = self.events.send(event);
    }

    fn model_delta(&self, content: &str) {
        self.send(WorkspaceEvent::ModelDelta {
            content: content.to_string(),
        });
    }
}

impl Reporter for WorkspaceReporter {
    fn model_text(&mut self, text: &str) {
        self.send(WorkspaceEvent::ModelAnswer {
            content: text.to_string(),
        });
    }

    fn tool_call(&mut self, line: &str) {
        self.send(WorkspaceEvent::ToolCall {
            detail: line.to_string(),
        });
    }

    fn tool_result(&mut self, name: &str, outcome: &ToolOutcome) {
        self.send(WorkspaceEvent::ToolResult {
            tool: name.to_string(),
            outcome: if outcome.is_err() { "error" } else { "ok" },
            content: outcome.text().to_string(),
        });
    }

    fn notice(&mut self, text: &str) {
        self.send(WorkspaceEvent::Notice {
            content: text.to_string(),
        });
    }
}

pub(crate) struct WorkspaceApprover {
    events: SyncSender<WorkspaceEvent>,
    decisions: Receiver<WorkspaceDecision>,
    cancel: Arc<AtomicBool>,
    pending_approval: Arc<Mutex<Option<String>>>,
    approval_timeout: Duration,
}

impl WorkspaceApprover {
    fn clear_pending(&self) {
        if let Ok(mut pending) = self.pending_approval.lock() {
            *pending = None;
        }
    }
}

impl Approver for WorkspaceApprover {
    fn approve(&mut self, action: &Action, sandbox: &Sandbox) -> Decision {
        let approval_id = uuid::Uuid::new_v4().to_string();
        let Ok(mut pending) = self.pending_approval.lock() else {
            return Decision::Abort;
        };
        *pending = Some(approval_id.clone());
        drop(pending);
        let event = WorkspaceEvent::ApprovalRequired {
            approval_id: approval_id.clone(),
            tool: action.tool_name().to_string(),
            risk: action.risk().label().to_string(),
            detail: action.approval_detail(sandbox),
        };
        if self.events.send(event).is_err() {
            self.clear_pending();
            return Decision::Abort;
        }

        let deadline = Instant::now() + self.approval_timeout;
        loop {
            if self.cancel.load(Ordering::Acquire) {
                self.clear_pending();
                return Decision::Abort;
            }
            if Instant::now() >= deadline {
                self.clear_pending();
                let _ = self.events.send(WorkspaceEvent::Notice {
                    content: "approval timed out; the session was aborted".to_string(),
                });
                return Decision::Abort;
            }
            match self.decisions.recv_timeout(APPROVAL_POLL) {
                Ok(decision) if decision.approval_id == approval_id => {
                    self.clear_pending();
                    return match decision.decision {
                        WorkspaceDecisionKind::AllowOnce => Decision::Once,
                        WorkspaceDecisionKind::AlwaysTool => Decision::AlwaysTool,
                        WorkspaceDecisionKind::Deny => Decision::No,
                        WorkspaceDecisionKind::Abort => Decision::Abort,
                    };
                }
                Ok(_) => continue,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => {
                    self.clear_pending();
                    return Decision::Abort;
                }
            }
        }
    }
}

pub(crate) fn run_live(
    config: WorkspaceRunConfig,
    mut worker: WorkspaceBridgeWorker,
) -> Result<LoopEnd, String> {
    let sandbox = match Sandbox::new(&config.workspace, false, Duration::from_secs(30)) {
        Ok(sandbox) => sandbox.with_shell_mode(ShellSandbox::Disabled),
        Err(error) => {
            let message = error.to_string();
            worker.reporter.send(WorkspaceEvent::Error {
                message: message.clone(),
            });
            worker.reporter.send(WorkspaceEvent::Finished {
                outcome: "driver_error",
            });
            return Err(message);
        }
    };
    worker.reporter.send(WorkspaceEvent::Started {
        workspace: sandbox.root().display().to_string(),
        model_id: config.model_id.clone(),
    });

    let tools = tools::specs_for(ToolProfile::WorkspaceFiles, false, ShellSandbox::Disabled);
    let mut history = vec![
        AgentMsg::System(super::agent::system_prompt(&sandbox, &tools)),
        AgentMsg::User(config.goal),
    ];
    let mut driver = LiveDriver::with(
        Client::new(config.addr),
        config.model_id,
        config.family,
        config.max_tokens,
        config.temperature,
    );
    let delta_reporter = worker.reporter.clone();
    driver.set_delta_sink(Some(Box::new(move |delta| {
        delta_reporter.model_delta(delta);
    })));
    let agent_config = AgentConfig {
        workdir: config.workspace,
        max_steps: config.max_steps,
        auto_approve: false,
        yolo: false,
        allow_net: false,
        allow_fs: false,
        shell_timeout: Duration::from_secs(30),
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        audit: Box::new(NoopSink),
        shell_sandbox: ShellSandbox::Disabled,
        tool_profile: ToolProfile::WorkspaceFiles,
    };
    let end = run_loop(
        &mut driver,
        &mut worker.approver,
        &mut worker.reporter,
        &sandbox,
        &agent_config,
        worker.cancel.as_ref(),
        &mut Policy::default(),
        &mut history,
    );
    let outcome = match end {
        LoopEnd::Answered => "answered",
        LoopEnd::Aborted => "aborted",
        LoopEnd::StepCapped => "step_capped",
        LoopEnd::Repeated => "repeated",
        LoopEnd::DriverError => "driver_error",
    };
    worker.reporter.send(WorkspaceEvent::Finished { outcome });
    Ok(end)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::thread;

    use serde_json::{json, Value};

    use super::*;
    use crate::chat::agent::{
        run_loop, AgentConfig, AgentMsg, LoopEnd, ModelDriver, ModelStep, Policy,
    };
    use crate::chat::audit::NoopSink;
    use crate::chat::shell_sandbox::ShellSandbox;
    use crate::chat::tools::{ToolCall, ToolProfile, ToolSpec};

    struct ScriptedDriver {
        steps: Vec<ModelStep>,
        next: usize,
    }

    impl ModelDriver for ScriptedDriver {
        fn step(
            &mut self,
            _history: &[AgentMsg],
            _tools: &[ToolSpec],
        ) -> Result<ModelStep, String> {
            let step = self
                .steps
                .get(self.next)
                .ok_or_else(|| "script exhausted".to_string())?;
            self.next += 1;
            Ok(match step {
                ModelStep::Text(text) => ModelStep::Text(text.clone()),
                ModelStep::Calls(calls) => ModelStep::Calls(calls.clone()),
            })
        }
    }

    fn call(name: &str, args: Value) -> ToolCall {
        ToolCall {
            name: name.to_string(),
            args,
        }
    }

    fn config(root: &std::path::Path) -> AgentConfig {
        AgentConfig {
            workdir: root.to_path_buf(),
            max_steps: 4,
            auto_approve: false,
            yolo: false,
            allow_net: false,
            allow_fs: false,
            shell_timeout: Duration::from_secs(5),
            max_tokens: 64,
            temperature: 0.0,
            audit: Box::new(NoopSink),
            shell_sandbox: ShellSandbox::Disabled,
            tool_profile: ToolProfile::WorkspaceFiles,
        }
    }

    fn run_write_loop(
        root: std::path::PathBuf,
        worker: WorkspaceBridgeWorker,
    ) -> thread::JoinHandle<LoopEnd> {
        thread::spawn(move || {
            let sandbox = Sandbox::new(&root, false, Duration::from_secs(5)).unwrap();
            let mut driver = ScriptedDriver {
                steps: vec![
                    ModelStep::Calls(vec![call(
                        "write_file",
                        json!({"path":"result.txt","content":"approved"}),
                    )]),
                    ModelStep::Text("done".to_string()),
                ],
                next: 0,
            };
            let mut reporter = worker.reporter;
            let mut approver = worker.approver;
            let mut history = vec![AgentMsg::User("write the result".to_string())];
            run_loop(
                &mut driver,
                &mut approver,
                &mut reporter,
                &sandbox,
                &config(&root),
                worker.cancel.as_ref(),
                &mut Policy::default(),
                &mut history,
            )
        })
    }

    fn next_approval(client: &WorkspaceBridgeClient) -> String {
        loop {
            match client.events.recv_timeout(Duration::from_secs(2)).unwrap() {
                WorkspaceEvent::ApprovalRequired { approval_id, .. } => return approval_id,
                _ => continue,
            }
        }
    }

    #[test]
    fn write_waits_for_matching_approval_before_execution() {
        let root = tempfile::tempdir().unwrap();
        let (worker, client) = bridge(16);
        let join = run_write_loop(root.path().to_path_buf(), worker);
        let approval_id = next_approval(&client);
        assert!(!root.path().join("result.txt").exists());

        client
            .try_decide(approval_id, WorkspaceDecisionKind::AllowOnce)
            .unwrap();
        assert_eq!(join.join().unwrap(), LoopEnd::Answered);
        assert_eq!(
            std::fs::read_to_string(root.path().join("result.txt")).unwrap(),
            "approved"
        );
    }

    #[test]
    fn denied_write_never_executes() {
        let root = tempfile::tempdir().unwrap();
        let (worker, client) = bridge(16);
        let join = run_write_loop(root.path().to_path_buf(), worker);
        let approval_id = next_approval(&client);
        client
            .try_decide(approval_id, WorkspaceDecisionKind::Deny)
            .unwrap();

        assert_eq!(join.join().unwrap(), LoopEnd::Answered);
        assert!(!root.path().join("result.txt").exists());
    }

    #[test]
    fn cancellation_while_approval_is_pending_aborts_without_writing() {
        let root = tempfile::tempdir().unwrap();
        let (worker, client) = bridge(16);
        let join = run_write_loop(root.path().to_path_buf(), worker);
        let _approval_id = next_approval(&client);
        client.cancel();

        assert_eq!(join.join().unwrap(), LoopEnd::Aborted);
        assert!(!root.path().join("result.txt").exists());
    }

    #[test]
    fn read_only_calls_do_not_request_approval() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("note.txt"), "hello").unwrap();
        let (mut worker, client) = bridge(16);
        let sandbox = Sandbox::new(root.path(), false, Duration::from_secs(5)).unwrap();
        let mut driver = ScriptedDriver {
            steps: vec![
                ModelStep::Calls(vec![call("read_file", json!({"path":"note.txt"}))]),
                ModelStep::Text("done".to_string()),
            ],
            next: 0,
        };
        let mut history = vec![AgentMsg::User("read note.txt".to_string())];
        let end = run_loop(
            &mut driver,
            &mut worker.approver,
            &mut worker.reporter,
            &sandbox,
            &config(root.path()),
            &AtomicBool::new(false),
            &mut Policy::default(),
            &mut history,
        );
        assert_eq!(end, LoopEnd::Answered);

        let events = client.events.try_iter().collect::<Vec<_>>();
        assert!(events
            .iter()
            .all(|event| !matches!(event, WorkspaceEvent::ApprovalRequired { .. })));
        assert!(events.iter().any(|event| matches!(
            event,
            WorkspaceEvent::ToolResult { tool, outcome: "ok", .. } if tool == "read_file"
        )));
    }

    #[test]
    fn workspace_profile_rejects_an_unadvertised_exec_tool() {
        let root = tempfile::tempdir().unwrap();
        let (mut worker, client) = bridge(16);
        let sandbox = Sandbox::new(root.path(), false, Duration::from_secs(5)).unwrap();
        let mut driver = ScriptedDriver {
            steps: vec![
                ModelStep::Calls(vec![call("run_shell", json!({"command":"echo unsafe"}))]),
                ModelStep::Text("stopped".to_string()),
            ],
            next: 0,
        };
        let mut history = vec![AgentMsg::User("run a command".to_string())];
        let end = run_loop(
            &mut driver,
            &mut worker.approver,
            &mut worker.reporter,
            &sandbox,
            &config(root.path()),
            &AtomicBool::new(false),
            &mut Policy::default(),
            &mut history,
        );
        assert_eq!(end, LoopEnd::Answered);
        let events = client.events.try_iter().collect::<Vec<_>>();
        assert!(events
            .iter()
            .all(|event| !matches!(event, WorkspaceEvent::ApprovalRequired { .. })));
        assert!(events.iter().any(|event| matches!(
            event,
            WorkspaceEvent::ToolResult { outcome: "error", content, .. }
                if content.contains("not available in this agent mode")
        )));
    }

    #[test]
    fn stale_approval_id_is_rejected_before_it_reaches_the_worker() {
        let root = tempfile::tempdir().unwrap();
        let (worker, client) = bridge(16);
        let join = run_write_loop(root.path().to_path_buf(), worker);
        let approval_id = next_approval(&client);
        assert_eq!(
            client.try_decide("not-current".to_string(), WorkspaceDecisionKind::AllowOnce),
            Err("the approval is stale or does not belong to this session")
        );
        client
            .try_decide(approval_id, WorkspaceDecisionKind::Deny)
            .unwrap();
        assert_eq!(join.join().unwrap(), LoopEnd::Answered);
        assert!(!root.path().join("result.txt").exists());
    }

    #[test]
    fn approval_timeout_aborts_without_writing() {
        let root = tempfile::tempdir().unwrap();
        let (worker, client) = bridge_with_timeout(16, Duration::from_millis(40));
        let join = run_write_loop(root.path().to_path_buf(), worker);
        let _approval_id = next_approval(&client);
        assert_eq!(join.join().unwrap(), LoopEnd::Aborted);
        assert!(!root.path().join("result.txt").exists());
    }
}
