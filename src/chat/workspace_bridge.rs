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
    run_loop, AgentConfig, AgentMsg, Approver, ContextBudgetUsage, Decision, LiveDriver, LoopEnd,
    ModelDriver, ModelStep, ModelStepMetrics, Policy, Reporter,
};
use super::audit::NoopSink;
use super::client::Client;
use super::shell_sandbox::ShellSandbox;
use super::tools::{Action, Sandbox, ToolOutcome, ToolProfile};
use super::workspace_memory::MemoryContext;
use crate::dial::{self, DialTier, ReviewOutcome};

const APPROVAL_POLL: Duration = Duration::from_millis(25);
const DEFAULT_APPROVAL_TIMEOUT: Duration = Duration::from_secs(5 * 60);
pub(crate) const WORKSPACE_CONTEXT_BUDGET_TOKENS: u32 = 4_096;
const WORKSPACE_MODEL_STEP_TIMEOUT: Duration = Duration::from_secs(90);
/// Wall-clock budget a whole turn must still fit inside for a review pass to be
/// worth starting. A turn that has already run this long is one the caller has
/// waited on long enough; the draft is returned instead of doubling the wait.
const WORKSPACE_REVIEW_TURN_BUDGET: Duration = Duration::from_secs(10 * 60);
/// Minimum slice of [`WORKSPACE_REVIEW_TURN_BUDGET`] that must remain before a
/// review is started, so a review is never begun only to be abandoned.
const WORKSPACE_REVIEW_TIME_FLOOR: Duration = Duration::from_secs(20);

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub(crate) enum WorkspaceEvent {
    #[serde(rename = "session.started")]
    Started { workspace: String, model_id: String },
    #[serde(rename = "turn.started")]
    TurnStarted { turn_index: u32 },
    #[serde(rename = "memory.updated")]
    MemoryUpdated {
        prompt_tokens: u32,
        generation_tokens: u32,
        budget_total: u32,
        system_tokens_estimate: u32,
        tool_definition_tokens_estimate: u32,
        message_tokens_estimate: u32,
        recent_memory_tokens_estimate: u32,
        retrieved_memory_tokens_estimate: u32,
        evidence_memory_tokens_estimate: u32,
        tool_result_tokens_estimate: u32,
    },
    #[serde(rename = "memory.compacted")]
    MemoryCompacted {
        compacted_through_turn: Option<u32>,
        archived_turns: u32,
        compaction_count: u32,
        trigger_tokens: u32,
        budget_total: u32,
    },
    #[serde(rename = "model.delta")]
    ModelDelta { content: String },
    #[serde(rename = "model.timing")]
    ModelTiming {
        total_ms: u64,
        ttft_ms: Option<u64>,
        output_tokens: Option<u32>,
    },
    #[serde(rename = "model.answer")]
    ModelAnswer { content: String },
    /// The answer that was just streamed is a draft that the dial is about to review.
    #[serde(rename = "dial.draft_ready")]
    DialDraftReady { tier: String },
    #[serde(rename = "dial.review_started")]
    DialReviewStarted,
    /// `changed` is false when the reviewer declined, so the draft above stands as the answer.
    #[serde(rename = "dial.review_finished")]
    DialReviewFinished { changed: bool },
    /// A guard declined to spend a second pass; the draft above stands as the answer.
    #[serde(rename = "dial.review_skipped")]
    DialReviewSkipped { reason: String },
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
    pub delivery_failed: Arc<AtomicBool>,
}

pub(crate) struct WorkspaceBridgeClient {
    pub events: Receiver<WorkspaceEvent>,
    decisions: SyncSender<WorkspaceDecision>,
    cancel: Arc<AtomicBool>,
    pending_approval: Arc<Mutex<Option<String>>>,
}

#[derive(Clone)]
pub(crate) struct WorkspaceRunConfig {
    pub addr: SocketAddr,
    pub workspace: PathBuf,
    pub goal: String,
    pub client_message_id: String,
    pub turn_index: u32,
    pub memory: MemoryContext,
    pub model_id: String,
    pub family: String,
    pub max_steps: usize,
    pub max_tokens: u32,
    pub temperature: f32,
    /// Optional session-scoped semantic index. When present, each turn gets a
    /// bounded set of relevant workspace excerpts before the model runs.
    pub semantic_retriever: Option<Arc<super::semantic_search::WorkspaceSemanticRetriever>>,
    /// Effort tier for this turn. `None` means the caller did not ask for a
    /// tier, which behaves exactly as the tiers that do not review.
    pub dial_tier: Option<DialTier>,
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
    let delivery_failed = Arc::new(AtomicBool::new(false));
    let pending_approval = Arc::new(Mutex::new(None));
    (
        WorkspaceBridgeWorker {
            reporter: WorkspaceReporter {
                events: event_tx.clone(),
                delivery_failed: Arc::clone(&delivery_failed),
            },
            approver: WorkspaceApprover {
                events: event_tx,
                decisions: decision_rx,
                cancel: Arc::clone(&cancel),
                delivery_failed: Arc::clone(&delivery_failed),
                pending_approval: Arc::clone(&pending_approval),
                approval_timeout,
            },
            cancel: Arc::clone(&cancel),
            delivery_failed,
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
    delivery_failed: Arc<AtomicBool>,
}

impl WorkspaceReporter {
    fn send(&self, event: WorkspaceEvent) {
        // A bounded blocking send provides backpressure without unbounded memory.
        // A dropped receiver ends delivery; the agent loop remains cancellable.
        if self.events.send(event).is_err() {
            self.delivery_failed.store(true, Ordering::Release);
        }
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

    fn context_budget(&mut self, usage: ContextBudgetUsage) {
        self.send(WorkspaceEvent::MemoryUpdated {
            prompt_tokens: usage.prompt_tokens,
            generation_tokens: usage.generation_tokens,
            budget_total: usage.budget_tokens,
            system_tokens_estimate: usage.system_tokens_estimate,
            tool_definition_tokens_estimate: usage.tool_definition_tokens_estimate,
            message_tokens_estimate: usage.message_tokens_estimate,
            recent_memory_tokens_estimate: usage.recent_memory_tokens_estimate,
            retrieved_memory_tokens_estimate: usage.retrieved_memory_tokens_estimate,
            evidence_memory_tokens_estimate: usage.evidence_memory_tokens_estimate,
            tool_result_tokens_estimate: usage.tool_result_tokens_estimate,
        });
    }

    fn model_timing(&mut self, metrics: ModelStepMetrics) {
        self.send(WorkspaceEvent::ModelTiming {
            total_ms: metrics.total_ms,
            ttft_ms: metrics.ttft_ms,
            output_tokens: metrics.output_tokens,
        });
    }
}

pub(crate) struct WorkspaceApprover {
    events: SyncSender<WorkspaceEvent>,
    decisions: Receiver<WorkspaceDecision>,
    cancel: Arc<AtomicBool>,
    delivery_failed: Arc<AtomicBool>,
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
            self.delivery_failed.store(true, Ordering::Release);
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
    let turn_started = Instant::now();
    let review_tier = config.dial_tier.filter(|tier| tier.wants_review());
    let review_params = review_tier.map(|tier| ReviewParams {
        addr: config.addr,
        model_id: config.model_id.clone(),
        family: config.family.clone(),
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        tier,
    });
    let review_goal = review_params.is_some().then(|| config.goal.clone());
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
        workspace: sandbox.root_display(),
        model_id: config.model_id.clone(),
    });
    worker.reporter.send(WorkspaceEvent::TurnStarted {
        turn_index: config.turn_index,
    });

    let system = super::agent::workspace_system_prompt(&sandbox);
    let mut history = vec![AgentMsg::System(system)];
    if let Some(retriever) = config.semantic_retriever.as_ref() {
        worker.reporter.notice(&format!(
            "retrieving semantically relevant workspace excerpts with {}",
            retriever.model_id()
        ));
        match retriever.retrieve_context(&config.goal, 5) {
            Ok(Some(context)) => history.push(AgentMsg::Memory(context)),
            Ok(None) => {}
            Err(error) => worker
                .reporter
                .notice(&format!("semantic retrieval was unavailable: {error}")),
        }
    }
    if let Some(memory) = render_relevant_memory(&config.memory.relevant) {
        history.push(AgentMsg::Memory(memory));
    }
    if let Some(memory) = render_evidence_memory(&config.memory.evidence) {
        history.push(AgentMsg::Memory(memory));
    }
    if let Some(memory) = render_recent_memory(&config.memory.recent) {
        history.push(AgentMsg::Memory(memory));
    }
    history.push(AgentMsg::User(config.goal));
    let mut driver = LiveDriver::with(
        Client::new(config.addr),
        config.model_id,
        config.family,
        config.max_tokens,
        config.temperature,
    );
    driver.set_context_budget(Some(WORKSPACE_CONTEXT_BUDGET_TOKENS));
    driver.set_native_tool_history(true);
    driver.set_stream_control(Arc::clone(&worker.cancel), WORKSPACE_MODEL_STEP_TIMEOUT);
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
        tool_profile: ToolProfile::WorkspaceReadOnly,
        ctx_budget: None,
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
    // A review only ever runs over an answer that exists, and only ever adds a
    // second answer on top of it. Nothing below can turn a good turn into a bad
    // one: every path that is not a strictly better answer keeps the draft.
    if matches!(end, LoopEnd::Answered) {
        if let (Some(params), Some(goal)) = (review_params, review_goal) {
            if let Some(draft) = last_assistant_text(&history) {
                let tier = params.tier;
                let max_tokens = params.max_tokens;
                let mut review_driver = LiveDriver::with(
                    Client::new(params.addr),
                    params.model_id,
                    params.family,
                    params.max_tokens,
                    params.temperature,
                );
                review_driver.set_context_budget(Some(WORKSPACE_CONTEXT_BUDGET_TOKENS));
                review_driver
                    .set_stream_control(Arc::clone(&worker.cancel), WORKSPACE_MODEL_STEP_TIMEOUT);
                if let Some(revised) = run_review_pass(
                    &mut worker,
                    &mut review_driver,
                    tier,
                    max_tokens,
                    &goal,
                    &draft,
                    turn_started.elapsed(),
                ) {
                    worker.reporter.model_text(&revised);
                }
            }
        }
    }
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

/// Everything a review pass needs from a [`WorkspaceRunConfig`] whose other
/// fields the agent loop has already consumed.
struct ReviewParams {
    addr: SocketAddr,
    model_id: String,
    family: String,
    max_tokens: u32,
    temperature: f32,
    tier: DialTier,
}

fn last_assistant_text(history: &[AgentMsg]) -> Option<String> {
    history.iter().rev().find_map(|message| match message {
        AgentMsg::Assistant(text) => Some(text.clone()),
        _ => None,
    })
}

/// Reviews `draft` with a second pass over `driver` and returns a revision only
/// when the reviewer supplied one.
///
/// Every guard, every transport failure and every reply this module cannot read
/// keeps the draft, so the worst outcome of a review is the answer the caller
/// would have received without one. Exactly one of `dial.review_finished` or
/// `dial.review_skipped` follows every `dial.draft_ready`.
fn run_review_pass(
    worker: &mut WorkspaceBridgeWorker,
    driver: &mut dyn ModelDriver,
    tier: DialTier,
    max_tokens: u32,
    goal: &str,
    draft: &str,
    elapsed: Duration,
) -> Option<String> {
    worker.reporter.send(WorkspaceEvent::DialDraftReady {
        tier: tier.as_str().to_string(),
    });
    let skip = |worker: &mut WorkspaceBridgeWorker, reason: &str| -> Option<String> {
        worker.reporter.send(WorkspaceEvent::DialReviewSkipped {
            reason: reason.to_string(),
        });
        None
    };

    if !dial::review_is_worth_attempting(draft) {
        return skip(worker, "empty_draft");
    }
    if !dial::review_fits_time_budget(
        elapsed,
        WORKSPACE_REVIEW_TURN_BUDGET,
        WORKSPACE_REVIEW_TIME_FLOOR,
    ) {
        return skip(worker, "time_budget");
    }
    if worker.cancel.load(Ordering::Relaxed) {
        return skip(worker, "cancelled");
    }

    let review_history = vec![
        AgentMsg::System(dial::review_instruction().to_string()),
        AgentMsg::User(dial::review_request(goal, draft)),
    ];

    // The driver's own accounting decides whether a second pass fits; this
    // module never re-derives it. A count the driver cannot produce proceeds,
    // because an overflowing prompt is refused by the server on its own and
    // that refusal lands on the same keep-the-draft path.
    if let Ok(Some(projected)) = driver.prompt_tokens(&review_history, &[]) {
        if !dial::review_fits_context_budget(projected, WORKSPACE_CONTEXT_BUDGET_TOKENS, max_tokens)
        {
            return skip(worker, "context_budget");
        }
    }

    worker.reporter.send(WorkspaceEvent::DialReviewStarted);
    let finished = |worker: &mut WorkspaceBridgeWorker, changed: bool| {
        worker
            .reporter
            .send(WorkspaceEvent::DialReviewFinished { changed });
    };

    // No tools are offered to the reviewer, so a review can never re-run the
    // work the draft already did.
    let reply = match driver.step(&review_history, &[]) {
        Ok(ModelStep::Text(text)) => text,
        Ok(ModelStep::Calls(_)) | Err(_) => {
            finished(worker, false);
            return None;
        }
    };
    match dial::interpret_review(draft, &reply) {
        ReviewOutcome::Unchanged => {
            finished(worker, false);
            None
        }
        ReviewOutcome::Revised(revised) => {
            finished(worker, true);
            Some(revised)
        }
    }
}

fn render_relevant_memory(relevant: &[super::workspace_memory::StoredTurn]) -> Option<String> {
    if relevant.is_empty() {
        return None;
    }
    let mut rendered = String::from("Relevant earlier conversation excerpts:\n");
    for turn in relevant {
        rendered.push_str(&format!(
            "- Earlier user: {}\n  Earlier assistant: {}\n",
            turn.user_text, turn.assistant_text
        ));
    }
    Some(rendered)
}

fn render_recent_memory(recent: &[super::workspace_memory::StoredTurn]) -> Option<String> {
    if recent.is_empty() {
        return None;
    }
    let mut rendered = String::from("Recent conversation excerpts:\n");
    for turn in recent {
        rendered.push_str(&format!(
            "- Earlier user: {}\n  Earlier assistant: {}\n",
            turn.user_text, turn.assistant_text
        ));
    }
    Some(rendered)
}

fn render_evidence_memory(evidence: &[super::workspace_memory::StoredEvidence]) -> Option<String> {
    if evidence.is_empty() {
        return None;
    }
    let mut rendered = String::from("Evidence recorded for selected earlier turns:\n");
    for entry in evidence {
        rendered.push_str(&format!(
            "- Tool: {}\n  Call: {}\n  Observation: {}\n  SHA-256: {}\n",
            entry.tool, entry.detail, entry.observation, entry.observation_sha256
        ));
    }
    Some(rendered)
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
            tool_profile: ToolProfile::Full,
            ctx_budget: None,
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
        let mut read_only_config = config(root.path());
        read_only_config.tool_profile = ToolProfile::WorkspaceReadOnly;
        let end = run_loop(
            &mut driver,
            &mut worker.approver,
            &mut worker.reporter,
            &sandbox,
            &read_only_config,
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
        let mut read_only_config = config(root.path());
        read_only_config.tool_profile = ToolProfile::WorkspaceReadOnly;
        let end = run_loop(
            &mut driver,
            &mut worker.approver,
            &mut worker.reporter,
            &sandbox,
            &read_only_config,
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

    /// A reviewer whose reply is fixed, and which records exactly what it was
    /// asked. Recording the tool list is the point: a reviewer that is offered
    /// tools could re-run the work the draft already did.
    struct RecordingReviewer {
        reply: Result<ModelStep, String>,
        projected_tokens: Result<Option<u32>, String>,
        seen_histories: Vec<Vec<AgentMsg>>,
        seen_tool_counts: Vec<usize>,
    }

    impl RecordingReviewer {
        fn answering(text: &str) -> Self {
            Self {
                reply: Ok(ModelStep::Text(text.to_string())),
                projected_tokens: Ok(None),
                seen_histories: Vec::new(),
                seen_tool_counts: Vec::new(),
            }
        }

        fn failing(error: &str) -> Self {
            Self {
                reply: Err(error.to_string()),
                projected_tokens: Ok(None),
                seen_histories: Vec::new(),
                seen_tool_counts: Vec::new(),
            }
        }

        fn projecting(text: &str, tokens: u32) -> Self {
            let mut reviewer = Self::answering(text);
            reviewer.projected_tokens = Ok(Some(tokens));
            reviewer
        }

        fn steps(&self) -> usize {
            self.seen_tool_counts.len()
        }
    }

    impl ModelDriver for RecordingReviewer {
        fn step(&mut self, history: &[AgentMsg], tools: &[ToolSpec]) -> Result<ModelStep, String> {
            self.seen_histories.push(history.to_vec());
            self.seen_tool_counts.push(tools.len());
            match &self.reply {
                Ok(ModelStep::Text(text)) => Ok(ModelStep::Text(text.clone())),
                Ok(ModelStep::Calls(calls)) => Ok(ModelStep::Calls(calls.clone())),
                Err(error) => Err(error.clone()),
            }
        }

        fn prompt_tokens(
            &mut self,
            _history: &[AgentMsg],
            _tools: &[ToolSpec],
        ) -> Result<Option<u32>, String> {
            self.projected_tokens.clone()
        }
    }

    /// Drains the events a review produced. The draft itself was already
    /// reported by the agent loop, so only the dial events are of interest.
    fn dial_events(client: &WorkspaceBridgeClient) -> Vec<WorkspaceEvent> {
        let mut events = Vec::new();
        while let Ok(event) = client.events.recv_timeout(Duration::from_millis(50)) {
            events.push(event);
        }
        events
    }

    fn review(
        reviewer: &mut RecordingReviewer,
        draft: &str,
        elapsed: Duration,
    ) -> (Option<String>, Vec<WorkspaceEvent>) {
        let (mut worker, client) = bridge(16);
        let revised = run_review_pass(
            &mut worker,
            reviewer,
            DialTier::High,
            64,
            "explain the retry policy",
            draft,
            elapsed,
        );
        (revised, dial_events(&client))
    }

    /// Test 7: a reviewer that declines leaves the draft standing, and says so.
    #[test]
    fn a_declined_review_keeps_the_draft() {
        let mut reviewer = RecordingReviewer::answering(dial::REVIEW_DECLINE_MARKER);
        let (revised, events) = review(&mut reviewer, "the draft answer", Duration::ZERO);

        assert_eq!(revised, None);
        assert_eq!(
            events,
            vec![
                WorkspaceEvent::DialDraftReady {
                    tier: "high".to_string()
                },
                WorkspaceEvent::DialReviewStarted,
                WorkspaceEvent::DialReviewFinished { changed: false },
            ]
        );
    }

    /// Test 8: a substantive reply replaces the draft and is reported as a
    /// change, so a caller can always tell which answer it is looking at.
    #[test]
    fn a_substantive_review_revises_the_draft() {
        let mut reviewer = RecordingReviewer::answering("retries stop after three attempts");
        let (revised, events) = review(&mut reviewer, "retries never stop", Duration::ZERO);

        assert_eq!(
            revised.as_deref(),
            Some("retries stop after three attempts")
        );
        assert_eq!(
            events.last(),
            Some(&WorkspaceEvent::DialReviewFinished { changed: true })
        );
    }

    /// Test 9: the reviewer is asked exactly once, and both the task and the
    /// draft reach it. A review that lost the draft would be reviewing nothing.
    #[test]
    fn the_reviewer_is_asked_once_with_the_task_and_the_draft() {
        let mut reviewer = RecordingReviewer::answering(dial::REVIEW_DECLINE_MARKER);
        let (_, _) = review(&mut reviewer, "retries never stop", Duration::ZERO);

        assert_eq!(reviewer.steps(), 1);
        let history = &reviewer.seen_histories[0];
        assert!(matches!(history[0], AgentMsg::System(_)));
        let AgentMsg::User(request) = &history[1] else {
            panic!("the review request must be the user turn: {history:?}");
        };
        assert!(request.contains("explain the retry policy"), "{request}");
        assert!(request.contains("retries never stop"), "{request}");
    }

    /// Test 14: the reviewer is offered no tools, so a review can never re-run
    /// the work the draft already did.
    #[test]
    fn the_reviewer_is_offered_no_tools() {
        let mut reviewer = RecordingReviewer::answering(dial::REVIEW_DECLINE_MARKER);
        let (_, _) = review(&mut reviewer, "the draft answer", Duration::ZERO);

        assert_eq!(reviewer.seen_tool_counts, vec![0]);
    }

    /// Test 12: a reviewer that fails or times out keeps the draft. The step
    /// timeout surfaces as an error from the driver, so this is the same path.
    #[test]
    fn a_failed_review_keeps_the_draft() {
        let mut reviewer = RecordingReviewer::failing("model step timed out after 90s");
        let (revised, events) = review(&mut reviewer, "the draft answer", Duration::ZERO);

        assert_eq!(revised, None);
        assert_eq!(
            events.last(),
            Some(&WorkspaceEvent::DialReviewFinished { changed: false })
        );
    }

    /// A reviewer that answers with tool calls is not answering; the draft
    /// stands rather than the turn losing its answer.
    #[test]
    fn a_reviewer_that_calls_tools_keeps_the_draft() {
        let mut reviewer = RecordingReviewer {
            reply: Ok(ModelStep::Calls(vec![call("read_file", json!({}))])),
            projected_tokens: Ok(None),
            seen_histories: Vec::new(),
            seen_tool_counts: Vec::new(),
        };
        let (revised, events) = review(&mut reviewer, "the draft answer", Duration::ZERO);

        assert_eq!(revised, None);
        assert_eq!(
            events.last(),
            Some(&WorkspaceEvent::DialReviewFinished { changed: false })
        );
    }

    /// Test 10: a review that would not fit the context budget is never asked
    /// for, and the reason is reported rather than left to be guessed.
    #[test]
    fn a_review_over_the_context_budget_is_skipped_before_the_model_runs() {
        let mut reviewer = RecordingReviewer::projecting(
            "a revision that must never be produced",
            WORKSPACE_CONTEXT_BUDGET_TOKENS - 63,
        );
        let (revised, events) = review(&mut reviewer, "the draft answer", Duration::ZERO);

        assert_eq!(revised, None);
        assert_eq!(reviewer.steps(), 0);
        assert_eq!(
            events,
            vec![
                WorkspaceEvent::DialDraftReady {
                    tier: "high".to_string()
                },
                WorkspaceEvent::DialReviewSkipped {
                    reason: "context_budget".to_string()
                },
            ]
        );
    }

    /// The same projection one token smaller does fit, so the guard above is
    /// the budget boundary and not a review that never runs.
    #[test]
    fn a_review_that_exactly_fits_the_context_budget_still_runs() {
        let mut reviewer = RecordingReviewer::projecting(
            dial::REVIEW_DECLINE_MARKER,
            WORKSPACE_CONTEXT_BUDGET_TOKENS - 64,
        );
        let (_, events) = review(&mut reviewer, "the draft answer", Duration::ZERO);

        assert_eq!(reviewer.steps(), 1);
        assert_eq!(
            events.last(),
            Some(&WorkspaceEvent::DialReviewFinished { changed: false })
        );
    }

    /// Test 11: a turn that has already spent its budget returns the draft
    /// rather than doubling the wait the caller has already borne.
    #[test]
    fn a_review_is_skipped_when_the_turn_budget_is_spent() {
        let mut reviewer = RecordingReviewer::answering("a revision that must never be produced");
        let (revised, events) = review(
            &mut reviewer,
            "the draft answer",
            WORKSPACE_REVIEW_TURN_BUDGET,
        );

        assert_eq!(revised, None);
        assert_eq!(reviewer.steps(), 0);
        assert_eq!(
            events.last(),
            Some(&WorkspaceEvent::DialReviewSkipped {
                reason: "time_budget".to_string()
            })
        );
    }

    /// Test 13: a turn cancelled before the review starts spends nothing more.
    #[test]
    fn a_cancelled_turn_never_starts_a_review() {
        let mut reviewer = RecordingReviewer::answering("a revision that must never be produced");
        let (mut worker, client) = bridge(16);
        worker.cancel.store(true, Ordering::Relaxed);
        let revised = run_review_pass(
            &mut worker,
            &mut reviewer,
            DialTier::High,
            64,
            "explain the retry policy",
            "the draft answer",
            Duration::ZERO,
        );

        assert_eq!(revised, None);
        assert_eq!(reviewer.steps(), 0);
        assert_eq!(
            dial_events(&client).last(),
            Some(&WorkspaceEvent::DialReviewSkipped {
                reason: "cancelled".to_string()
            })
        );
    }

    /// An empty draft is nothing to review, and asking anyway would invite the
    /// reviewer to answer the task itself with no tools and no context.
    #[test]
    fn an_empty_draft_is_never_reviewed() {
        let mut reviewer = RecordingReviewer::answering("a revision that must never be produced");
        let (revised, events) = review(&mut reviewer, "   \n  ", Duration::ZERO);

        assert_eq!(revised, None);
        assert_eq!(reviewer.steps(), 0);
        assert_eq!(
            events.last(),
            Some(&WorkspaceEvent::DialReviewSkipped {
                reason: "empty_draft".to_string()
            })
        );
    }

    /// Test 6: a review is never itself reviewed. One pass per turn is the
    /// whole cost bound, so the reviewer is asked exactly once no matter what
    /// it replies.
    #[test]
    fn a_revision_is_never_reviewed_again() {
        let mut reviewer = RecordingReviewer::answering("a revised answer");
        let (revised, _) = review(&mut reviewer, "the draft answer", Duration::ZERO);

        assert_eq!(revised.as_deref(), Some("a revised answer"));
        assert_eq!(reviewer.steps(), 1);
    }

    /// The tiers that do not review must not pay for a driver, a prompt or a
    /// decision. Only the reviewing tiers reach [`run_review_pass`] at all.
    #[test]
    fn only_the_reviewing_tiers_ask_for_a_review() {
        assert!(!DialTier::Low.wants_review());
        assert!(!DialTier::Medium.wants_review());
        assert!(DialTier::High.wants_review());
        assert!(DialTier::Ultra.wants_review());
    }
}
