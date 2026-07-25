//! Local durable remote-agent host boundary (Phase 2; no relay).

#![cfg_attr(not(test), allow(dead_code))]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use camelid_agent_events::{AgentEvent, AgentEventError, AgentEventSink, ToolRisk};
use camelid_remote_protocol::{
    canonical_json, decode_command, decode_message, ApprovalDecision as ProtocolDecision, Command,
};
use camelid_remote_store::{
    AcceptApprovalDecision, AcceptCancelTurn, AcceptDecision, AcceptStartTurn, AcceptTurn,
    CommandResult, CompleteTurn, ExpireApproval, PendingApproval, RemoteStore, StoreError,
};
use uuid::Uuid;

use super::agent::{
    self, AgentConfig, AgentMsg, ApprovalScope, Approver, Decision, LoopEnd, ModelDriver, Policy,
    Reporter,
};
use super::tools::{Action, Sandbox, ToolProfile};

#[derive(Clone)]
pub struct LocalRemoteHost {
    store: Arc<Mutex<RemoteStore>>,
    session_id: Uuid,
    device_id: Uuid,
    device_key: [u8; 32],
    runtime: camelid_agent_runtime::AgentRuntime<AgentMsg>,
    clock: Arc<AtomicU64>,
    hub: Arc<EventHub>,
    approval_waiters: Arc<Mutex<HashMap<Uuid, Sender<()>>>>,
    approval_timeout: Duration,
}

pub struct HostIdentity<'a> {
    pub canonical_root: &'a str,
    pub model_id: &'a str,
    pub model_sha256: &'a str,
    pub capability_snapshot_json: &'a str,
}

pub struct AcceptedRemoteTurn {
    pub turn_id: Uuid,
    text: String,
}

pub enum StartTurnAcceptance {
    Accepted(AcceptedRemoteTurn),
    Duplicate(CommandResult),
}

impl LocalRemoteHost {
    pub fn new(
        store: RemoteStore,
        session_id: Uuid,
        device_id: Uuid,
        device_key: [u8; 32],
        identity: HostIdentity<'_>,
        first_timestamp: u64,
    ) -> Result<Self, StoreError> {
        Self::from_shared(
            Arc::new(Mutex::new(store)),
            session_id,
            device_id,
            device_key,
            identity,
            first_timestamp,
        )
    }

    pub fn from_shared(
        store: Arc<Mutex<RemoteStore>>,
        session_id: Uuid,
        device_id: Uuid,
        device_key: [u8; 32],
        identity: HostIdentity<'_>,
        first_timestamp: u64,
    ) -> Result<Self, StoreError> {
        let context = store
            .lock()
            .map_err(|_| StoreError::Unavailable)?
            .load_session_context(
                session_id,
                identity.canonical_root,
                identity.model_id,
                identity.model_sha256,
                identity.capability_snapshot_json,
            )?;
        let transcript: Vec<AgentMsg> =
            serde_json::from_str(&context.transcript_json).map_err(|_| StoreError::Invalid)?;
        let plan: Vec<camelid_agent_runtime::PlanStep> =
            serde_json::from_str(&context.plan_json).map_err(|_| StoreError::Invalid)?;
        let runtime = camelid_agent_runtime::AgentRuntime::default();
        runtime
            .transcript()
            .replace(transcript)
            .map_err(|_| StoreError::Unavailable)?;
        runtime
            .plan()
            .replace(plan)
            .map_err(|_| StoreError::Unavailable)?;
        Ok(Self {
            store,
            session_id,
            device_id,
            device_key,
            runtime,
            clock: Arc::new(AtomicU64::new(first_timestamp)),
            hub: Arc::new(EventHub::default()),
            approval_waiters: Arc::new(Mutex::new(HashMap::new())),
            approval_timeout: Duration::from_secs(5 * 60),
        })
    }

    pub fn for_device(&self, device_id: Uuid, device_key: [u8; 32]) -> Self {
        let mut host = self.clone();
        host.device_id = device_id;
        host.device_key = device_key;
        host
    }

    #[cfg(test)]
    fn set_approval_timeout(&mut self, timeout: Duration) {
        self.approval_timeout = timeout;
    }

    pub fn subscribe(&self) -> Result<Receiver<camelid_remote_store::StoredEvent>, StoreError> {
        self.hub.subscribe()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_message(
        &self,
        message: &[u8],
        driver: &mut dyn ModelDriver,
        reporter: &mut dyn Reporter,
        sandbox: &Sandbox,
        config: &AgentConfig,
    ) -> Result<LoopEnd, StoreError> {
        match self.accept_start_message(message)? {
            StartTurnAcceptance::Accepted(turn) => {
                self.run_accepted_turn(turn, driver, reporter, sandbox, config)
            }
            StartTurnAcceptance::Duplicate(_) => Err(StoreError::Conflict),
        }
    }

    pub fn accept_start_message(&self, message: &[u8]) -> Result<StartTurnAcceptance, StoreError> {
        let envelope = decode_message(message).map_err(|_| StoreError::Invalid)?;
        if envelope.device_id != self.device_id || envelope.session_id != Some(self.session_id) {
            return Err(StoreError::Conflict);
        }
        match decode_command(&envelope).map_err(|_| StoreError::Invalid)? {
            Command::StartTurn {
                command_id,
                turn_id,
                text,
            } => self.accept_start_turn(command_id, turn_id, &text),
            _ => Err(StoreError::Invalid),
        }
    }

    pub fn approval_message(&self, message: &[u8]) -> Result<(), StoreError> {
        let envelope = decode_message(message).map_err(|_| StoreError::Invalid)?;
        if envelope.device_id != self.device_id || envelope.session_id != Some(self.session_id) {
            return Err(StoreError::Conflict);
        }
        match decode_command(&envelope).map_err(|_| StoreError::Invalid)? {
            Command::ApprovalDecision {
                command_id,
                turn_id,
                call_id,
                approval_id,
                action_digest,
                decision,
            } => self.approval_decision(
                command_id,
                turn_id,
                call_id,
                approval_id,
                &action_digest,
                decision,
            ),
            _ => Err(StoreError::Invalid),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn approval_decision(
        &self,
        command_id: Uuid,
        turn_id: Uuid,
        call_id: Uuid,
        approval_id: Uuid,
        action_digest: &str,
        decision: ProtocolDecision,
    ) -> Result<(), StoreError> {
        let decision_token = protocol_decision_token(decision);
        let request_digest = format!(
            "sha256:{}",
            hex_digest(
                &canonical_json(&serde_json::json!({
                    "command_id": command_id,
                    "turn_id": turn_id,
                    "call_id": call_id,
                    "approval_id": approval_id,
                    "action_digest": action_digest,
                    "decision": decision_token,
                }))
                .map_err(|_| StoreError::Invalid)?,
            )
        );
        let now = self.now();
        let mut store = self.store.lock().map_err(|_| StoreError::Unavailable)?;
        if !store.device_authorized(self.device_id, &self.device_key)? {
            return Err(StoreError::Conflict);
        }
        let accepted = store.accept_approval_decision(AcceptApprovalDecision {
            device_id: self.device_id,
            command_id,
            request_digest: &request_digest,
            session_id: self.session_id,
            turn_id,
            call_id,
            approval_id,
            action_digest,
            decision: decision_token,
            created_at_unix_ms: now,
        })?;
        drop(store);
        if matches!(accepted, AcceptDecision::Applied) {
            self.wake_approval(approval_id);
        }
        Ok(())
    }

    pub fn cancel_message(&self, message: &[u8]) -> Result<(), StoreError> {
        let envelope = decode_message(message).map_err(|_| StoreError::Invalid)?;
        if envelope.device_id != self.device_id || envelope.session_id != Some(self.session_id) {
            return Err(StoreError::Conflict);
        }
        match decode_command(&envelope).map_err(|_| StoreError::Invalid)? {
            Command::CancelTurn {
                command_id,
                turn_id,
            } => self.cancel_turn(command_id, turn_id),
            _ => Err(StoreError::Invalid),
        }
    }

    pub fn cancel_turn(&self, command_id: Uuid, turn_id: Uuid) -> Result<(), StoreError> {
        let request_digest = format!(
            "sha256:{}",
            hex_digest(
                &canonical_json(&serde_json::json!({
                    "command_id": command_id,
                    "turn_id": turn_id,
                }))
                .map_err(|_| StoreError::Invalid)?,
            )
        );
        let now = self.now();
        let mut store = self.store.lock().map_err(|_| StoreError::Unavailable)?;
        if !store.device_authorized(self.device_id, &self.device_key)? {
            return Err(StoreError::Conflict);
        }
        let event = store.accept_cancel_turn(AcceptCancelTurn {
            device_id: self.device_id,
            command_id,
            request_digest: &request_digest,
            session_id: self.session_id,
            turn_id,
            created_at_unix_ms: now,
        })?;
        drop(store);
        if let Some(event) = event {
            self.runtime.cancel().request();
            self.wake_all_approvals();
            self.hub.broadcast(event);
        }
        Ok(())
    }

    pub fn cancel_locally(&self) -> Result<(), StoreError> {
        let event = self
            .store
            .lock()
            .map_err(|_| StoreError::Unavailable)?
            .cancel_active_turn_locally(self.session_id, self.now())?;
        if let Some(event) = event {
            self.runtime.cancel().request();
            self.wake_all_approvals();
            self.hub.broadcast(event);
        }
        Ok(())
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn run_turn(
        &self,
        command_id: Uuid,
        turn_id: Uuid,
        text: &str,
        driver: &mut dyn ModelDriver,
        reporter: &mut dyn Reporter,
        sandbox: &Sandbox,
        config: &AgentConfig,
        decision: Decision,
    ) -> Result<LoopEnd, StoreError> {
        let StartTurnAcceptance::Accepted(turn) =
            self.accept_start_turn(command_id, turn_id, text)?
        else {
            return Err(StoreError::Conflict);
        };
        self.run_accepted_turn_with_approval(
            turn,
            driver,
            reporter,
            sandbox,
            config,
            ApprovalMode::Immediate(decision),
        )
    }

    pub fn run_accepted_turn(
        &self,
        turn: AcceptedRemoteTurn,
        driver: &mut dyn ModelDriver,
        reporter: &mut dyn Reporter,
        sandbox: &Sandbox,
        config: &AgentConfig,
    ) -> Result<LoopEnd, StoreError> {
        self.run_accepted_turn_with_approval(
            turn,
            driver,
            reporter,
            sandbox,
            config,
            ApprovalMode::Remote,
        )
    }

    fn accept_start_turn(
        &self,
        command_id: Uuid,
        turn_id: Uuid,
        text: &str,
    ) -> Result<StartTurnAcceptance, StoreError> {
        let request_digest = format!(
            "sha256:{}",
            hex_digest(
                &canonical_json(&serde_json::json!({
                    "command_id": command_id,
                    "turn_id": turn_id,
                    "text": text,
                }))
                .map_err(|_| StoreError::Invalid)?,
            )
        );
        let now = self.now();
        let mut store = self.store.lock().map_err(|_| StoreError::Unavailable)?;
        if !store.device_authorized(self.device_id, &self.device_key)? {
            return Err(StoreError::Conflict);
        }
        let accepted = store.accept_start_turn(AcceptStartTurn {
            device_id: self.device_id,
            command_id,
            request_digest: &request_digest,
            session_id: self.session_id,
            turn_id,
            user_text: text,
            created_at_unix_ms: now,
        })?;
        drop(store);
        match accepted {
            AcceptTurn::Duplicate(result) => Ok(StartTurnAcceptance::Duplicate(result)),
            AcceptTurn::Accepted { events } => {
                for event in events {
                    self.hub.broadcast(event);
                }
                Ok(StartTurnAcceptance::Accepted(AcceptedRemoteTurn {
                    turn_id,
                    text: text.into(),
                }))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_accepted_turn_with_approval(
        &self,
        turn: AcceptedRemoteTurn,
        driver: &mut dyn ModelDriver,
        reporter: &mut dyn Reporter,
        sandbox: &Sandbox,
        config: &AgentConfig,
        approval_mode: ApprovalMode,
    ) -> Result<LoopEnd, StoreError> {
        if config.tool_profile != ToolProfile::RemoteV1 {
            return Err(StoreError::Invalid);
        }
        let turn_id = turn.turn_id;
        self.runtime.cancel().reset();
        self.runtime
            .transcript()
            .push(AgentMsg::User(turn.text))
            .map_err(|_| StoreError::Unavailable)?;
        let mut sink = DurableSink {
            store: Arc::clone(&self.store),
            session_id: self.session_id,
            turn_id,
            clock: Arc::clone(&self.clock),
            hub: Arc::clone(&self.hub),
        };
        let mut approver = DurableApprover {
            store: Arc::clone(&self.store),
            session_id: self.session_id,
            turn_id,
            device_id: self.device_id,
            mode: approval_mode,
            clock: Arc::clone(&self.clock),
            hub: Arc::clone(&self.hub),
            waiters: Arc::clone(&self.approval_waiters),
            timeout: self.approval_timeout,
        };
        let mut policy = Policy::default();
        let end = agent::run_loop_with_runtime_events(
            driver,
            &mut approver,
            reporter,
            sandbox,
            config,
            &mut policy,
            &self.runtime,
            &mut sink,
        );
        let transcript = self
            .runtime
            .transcript()
            .snapshot()
            .map_err(|_| StoreError::Unavailable)?;
        let answer = transcript.iter().rev().find_map(|message| match message {
            AgentMsg::Assistant(text) => Some(text.as_str()),
            _ => None,
        });
        let transcript_json =
            serde_json::to_string(&transcript).map_err(|_| StoreError::Invalid)?;
        let plan = super::plan::get_in(self.runtime.plan()).map_err(|_| StoreError::Unavailable)?;
        let plan_json = serde_json::to_string(&plan).map_err(|_| StoreError::Invalid)?;
        let outcome = match end {
            LoopEnd::Answered => "completed",
            LoopEnd::Aborted => "aborted",
            LoopEnd::StepCapped => "step_capped",
            LoopEnd::Repeated => "repeated",
            LoopEnd::DriverError => "driver_error",
        };
        let mut store = self.store.lock().map_err(|_| StoreError::Unavailable)?;
        let event = store.complete_turn(CompleteTurn {
            session_id: self.session_id,
            turn_id,
            outcome,
            assistant_text: answer,
            transcript_json: &transcript_json,
            plan_json: &plan_json,
            finished_at_unix_ms: self.now(),
        })?;
        drop(store);
        self.hub.broadcast(event);
        Ok(end)
    }

    pub fn replay(&self, after: u64) -> Result<Vec<camelid_remote_store::StoredEvent>, StoreError> {
        self.replay_limit(after, 256)
    }

    pub fn replay_limit(
        &self,
        after: u64,
        limit: u16,
    ) -> Result<Vec<camelid_remote_store::StoredEvent>, StoreError> {
        self.store
            .lock()
            .map_err(|_| StoreError::Unavailable)?
            .replay(self.session_id, after, limit)
    }

    pub fn session_head(&self) -> Result<camelid_remote_store::SessionHead, StoreError> {
        self.store
            .lock()
            .map_err(|_| StoreError::Unavailable)?
            .session_head(self.session_id)
    }

    fn now(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    fn wake_approval(&self, approval_id: Uuid) {
        let sender = self
            .approval_waiters
            .lock()
            .ok()
            .and_then(|mut waiters| waiters.remove(&approval_id));
        if let Some(sender) = sender {
            let _ = sender.send(());
        }
    }

    fn wake_all_approvals(&self) {
        let senders = self
            .approval_waiters
            .lock()
            .map(|mut waiters| {
                waiters
                    .drain()
                    .map(|(_, sender)| sender)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for sender in senders {
            let _ = sender.send(());
        }
    }
}

struct DurableSink {
    store: Arc<Mutex<RemoteStore>>,
    session_id: Uuid,
    turn_id: Uuid,
    clock: Arc<AtomicU64>,
    hub: Arc<EventHub>,
}

impl AgentEventSink for DurableSink {
    fn emit(&mut self, event: AgentEvent) -> Result<(), AgentEventError> {
        let payload = event_payload(&event)?;
        let mut store = self
            .store
            .lock()
            .map_err(|_| AgentEventError::SinkUnavailable)?;
        let now = self.clock.fetch_add(1, Ordering::Relaxed);
        let stored = match &event {
            AgentEvent::ApprovalRequired {
                approval_id,
                record,
            } => {
                let approval = record.approval_record().ok_or(AgentEventError::Invalid)?;
                let detail =
                    serde_json::to_string(approval).map_err(|_| AgentEventError::Invalid)?;
                store
                    .insert_pending_approval(
                        PendingApproval {
                            approval_id: *approval_id,
                            session_id: self.session_id,
                            turn_id: self.turn_id,
                            call_id: record.call_id(),
                            action_digest: record
                                .action_digest()
                                .ok_or(AgentEventError::Invalid)?,
                            tool: record.tool(),
                            risk: risk_token(record.risk()),
                            detail_json: &detail,
                            created_at_unix_ms: now,
                        },
                        &payload,
                    )
                    .map_err(|_| AgentEventError::SinkUnavailable)?
            }
            _ => store
                .append_event(
                    self.session_id,
                    Some(self.turn_id),
                    event_name(&event),
                    &payload,
                    now,
                )
                .map_err(|_| AgentEventError::SinkUnavailable)?,
        };
        drop(store);
        self.hub.broadcast(stored);
        Ok(())
    }
}

#[derive(Default)]
struct EventHub {
    subscribers: Mutex<Vec<Sender<camelid_remote_store::StoredEvent>>>,
}

impl EventHub {
    fn subscribe(&self) -> Result<Receiver<camelid_remote_store::StoredEvent>, StoreError> {
        let (sender, receiver) = mpsc::channel();
        self.subscribers
            .lock()
            .map_err(|_| StoreError::Unavailable)?
            .push(sender);
        Ok(receiver)
    }

    fn broadcast(&self, event: camelid_remote_store::StoredEvent) {
        if let Ok(mut subscribers) = self.subscribers.lock() {
            subscribers.retain(|subscriber| subscriber.send(event.clone()).is_ok());
        }
    }
}

#[derive(Clone, Copy)]
enum ApprovalMode {
    #[cfg(test)]
    Immediate(Decision),
    Remote,
}

struct DurableApprover {
    store: Arc<Mutex<RemoteStore>>,
    session_id: Uuid,
    turn_id: Uuid,
    device_id: Uuid,
    mode: ApprovalMode,
    clock: Arc<AtomicU64>,
    hub: Arc<EventHub>,
    waiters: Arc<Mutex<HashMap<Uuid, Sender<()>>>>,
    timeout: Duration,
}

impl Approver for DurableApprover {
    fn approve(&mut self, _action: &Action, _sandbox: &Sandbox) -> Decision {
        Decision::No
    }

    fn approve_scoped(
        &mut self,
        _action: &Action,
        _sandbox: &Sandbox,
        scope: ApprovalScope<'_>,
    ) -> Decision {
        match self.mode {
            #[cfg(test)]
            ApprovalMode::Immediate(decision) => self.settle_immediate(scope, decision),
            ApprovalMode::Remote => self.wait_for_remote(scope),
        }
    }
}

impl DurableApprover {
    #[cfg(test)]
    fn settle_immediate(&self, scope: ApprovalScope<'_>, decision: Decision) -> Decision {
        let token = match decision {
            Decision::Once => "allow_once",
            Decision::No | Decision::AlwaysTool => "deny",
            Decision::Abort => "abort_turn",
            Decision::Expired => "expired",
            Decision::InvalidatedByCancel => "invalidated_by_cancel",
        };
        let result = self
            .store
            .lock()
            .map_err(|_| StoreError::Unavailable)
            .and_then(|mut store| {
                store.settle_approval(camelid_remote_store::SettleApproval {
                    approval_id: scope.approval_id,
                    session_id: self.session_id,
                    turn_id: self.turn_id,
                    call_id: scope.call_id,
                    action_digest: scope.action_digest,
                    decision: token,
                    device_id: if matches!(
                        decision,
                        Decision::Expired | Decision::InvalidatedByCancel
                    ) {
                        None
                    } else {
                        Some(self.device_id)
                    },
                    settled_at_unix_ms: self.clock.fetch_add(1, Ordering::Relaxed),
                })
            });
        if result.is_err() {
            Decision::Abort
        } else if decision == Decision::AlwaysTool {
            Decision::No
        } else {
            decision
        }
    }

    fn wait_for_remote(&self, scope: ApprovalScope<'_>) -> Decision {
        let (sender, receiver) = mpsc::channel();
        let registered = self
            .waiters
            .lock()
            .map(|mut waiters| match waiters.entry(scope.approval_id) {
                std::collections::hash_map::Entry::Occupied(_) => false,
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(sender);
                    true
                }
            })
            .unwrap_or(false);
        if !registered {
            return Decision::Abort;
        }
        if let Some(decision) = self.read_decision(scope) {
            self.remove_waiter(scope.approval_id);
            return decision;
        }
        match receiver.recv_timeout(self.timeout) {
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.remove_waiter(scope.approval_id);
                self.read_decision(scope).unwrap_or(Decision::Abort)
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.remove_waiter(scope.approval_id);
                self.expire_or_read(scope)
            }
        }
    }

    fn expire_or_read(&self, scope: ApprovalScope<'_>) -> Decision {
        let result = self
            .store
            .lock()
            .map_err(|_| StoreError::Unavailable)
            .and_then(|mut store| {
                store.expire_approval(ExpireApproval {
                    session_id: self.session_id,
                    turn_id: self.turn_id,
                    call_id: scope.call_id,
                    approval_id: scope.approval_id,
                    action_digest: scope.action_digest,
                    expired_at_unix_ms: self.clock.fetch_add(1, Ordering::Relaxed),
                })
            });
        match result {
            Ok(event) => {
                self.hub.broadcast(event);
                Decision::Expired
            }
            Err(StoreError::Conflict) => self.read_decision(scope).unwrap_or(Decision::Abort),
            Err(_) => Decision::Abort,
        }
    }

    fn read_decision(&self, scope: ApprovalScope<'_>) -> Option<Decision> {
        self.store
            .lock()
            .ok()?
            .approval_decision(
                self.session_id,
                self.turn_id,
                scope.call_id,
                scope.approval_id,
                scope.action_digest,
            )
            .ok()?
            .as_deref()
            .and_then(decision_from_token)
    }

    fn remove_waiter(&self, approval_id: Uuid) {
        if let Ok(mut waiters) = self.waiters.lock() {
            waiters.remove(&approval_id);
        }
    }
}

fn protocol_decision_token(decision: ProtocolDecision) -> &'static str {
    match decision {
        ProtocolDecision::AllowOnce => "allow_once",
        ProtocolDecision::Deny => "deny",
        ProtocolDecision::AbortTurn => "abort_turn",
    }
}

fn decision_from_token(token: &str) -> Option<Decision> {
    match token {
        "allow_once" => Some(Decision::Once),
        "deny" => Some(Decision::No),
        "abort_turn" => Some(Decision::Abort),
        "expired" => Some(Decision::Expired),
        "invalidated_by_cancel" | "invalidated_by_restart" => Some(Decision::InvalidatedByCancel),
        _ => None,
    }
}

fn event_name(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::ModelDelta { .. } => "model.delta",
        AgentEvent::ModelAnswer { .. } => "model.answer",
        AgentEvent::PlanUpdated { .. } => "plan.updated",
        AgentEvent::ToolCall { .. } => "tool.call",
        AgentEvent::ApprovalRequired { .. } => "approval.required",
        AgentEvent::ApprovalSettled { .. } => "approval.settled",
        AgentEvent::ToolResult { .. } => "tool.result",
        AgentEvent::Notice { .. } => "session.notice",
        AgentEvent::Timing { .. } => "model.timing",
    }
}

fn event_payload(event: &AgentEvent) -> Result<serde_json::Value, AgentEventError> {
    match event {
        AgentEvent::ApprovalRequired {
            approval_id,
            record,
        } => {
            let mut authority =
                serde_json::to_value(record.approval_record().ok_or(AgentEventError::Invalid)?)
                    .map_err(|_| AgentEventError::Invalid)?;
            let object = authority.as_object_mut().ok_or(AgentEventError::Invalid)?;
            object.insert("call_id".into(), serde_json::json!(record.call_id()));
            object.insert(
                "action_digest".into(),
                serde_json::json!(record.action_digest().ok_or(AgentEventError::Invalid)?),
            );
            object.insert("detail".into(), serde_json::json!(record.detail()));
            Ok(serde_json::json!({
                "approval_id": approval_id,
                "record": authority,
            }))
        }
        AgentEvent::ApprovalSettled { settlement } => {
            serde_json::to_value(settlement).map_err(|_| AgentEventError::Invalid)
        }
        AgentEvent::ModelDelta { content }
        | AgentEvent::ModelAnswer { content }
        | AgentEvent::Notice { content } => Ok(serde_json::json!({"content": content})),
        AgentEvent::PlanUpdated { steps } => Ok(serde_json::json!({"steps": steps})),
        AgentEvent::ToolCall { record } => {
            serde_json::to_value(record).map_err(|_| AgentEventError::Invalid)
        }
        AgentEvent::ToolResult { record } => {
            serde_json::to_value(record).map_err(|_| AgentEventError::Invalid)
        }
        AgentEvent::Timing { metrics } => {
            serde_json::to_value(metrics).map_err(|_| AgentEventError::Invalid)
        }
    }
}

fn risk_token(risk: ToolRisk) -> &'static str {
    match risk {
        ToolRisk::Read => "read",
        ToolRisk::Write => "write",
        ToolRisk::Exec => "exec",
        ToolRisk::Network => "network",
        ToolRisk::Plan => "plan",
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::Duration;

    use camelid_remote_store::SessionState;
    use serde_json::json;

    use super::*;
    use crate::chat::agent::{ContextBudgetUsage, ModelStep, ModelStepMetrics};
    use crate::chat::audit::NoopSink;
    use crate::chat::shell_sandbox::ShellSandbox;
    use crate::chat::tools::{ToolCall, ToolOutcome, ToolSpec};

    struct ScriptDriver {
        steps: Vec<ModelStep>,
        index: usize,
    }

    struct CancellingDriver {
        cancel: camelid_agent_runtime::CancellationToken,
    }

    struct BlockingDriver {
        entered: std::sync::mpsc::Sender<()>,
        release: std::sync::mpsc::Receiver<()>,
    }

    impl ModelDriver for CancellingDriver {
        fn step(
            &mut self,
            _history: &[AgentMsg],
            _tools: &[ToolSpec],
        ) -> Result<ModelStep, String> {
            self.cancel.request();
            Ok(ModelStep::Text("partial answer must be discarded".into()))
        }
    }

    impl ModelDriver for BlockingDriver {
        fn step(
            &mut self,
            _history: &[AgentMsg],
            _tools: &[ToolSpec],
        ) -> Result<ModelStep, String> {
            self.entered.send(()).map_err(|error| error.to_string())?;
            self.release.recv().map_err(|error| error.to_string())?;
            Ok(ModelStep::Text("cancelled partial answer".into()))
        }
    }

    impl ModelDriver for ScriptDriver {
        fn step(
            &mut self,
            _history: &[AgentMsg],
            _tools: &[ToolSpec],
        ) -> Result<ModelStep, String> {
            let step = self.steps.get(self.index).ok_or("script exhausted")?;
            self.index += 1;
            Ok(match step {
                ModelStep::Text(text) => ModelStep::Text(text.clone()),
                ModelStep::Calls(calls) => ModelStep::Calls(calls.clone()),
            })
        }

        fn take_step_metrics(&mut self) -> Option<ModelStepMetrics> {
            None
        }
    }

    #[derive(Default)]
    struct SilentReporter;

    impl Reporter for SilentReporter {
        fn model_text(&mut self, _text: &str) {}
        fn tool_call(&mut self, _line: &str) {}
        fn tool_result(&mut self, _name: &str, _outcome: &ToolOutcome) {}
        fn notice(&mut self, _text: &str) {}
        fn context_budget(&mut self, _usage: ContextBudgetUsage) {}
    }

    fn config(root: &Path) -> AgentConfig {
        AgentConfig {
            workdir: root.to_path_buf(),
            max_steps: 4,
            auto_approve: false,
            yolo: false,
            allow_net: false,
            allow_fs: false,
            shell_timeout: Duration::from_secs(5),
            max_tokens: 128,
            temperature: 0.0,
            audit: Box::new(NoopSink),
            shell_sandbox: ShellSandbox::Disabled,
            tool_profile: ToolProfile::RemoteV1,
            ctx_budget: None,
        }
    }

    fn identity<'a>(root: &'a str, capability_snapshot_json: &'a str) -> HostIdentity<'a> {
        HostIdentity {
            canonical_root: root,
            model_id: "model",
            model_sha256: "sha256:model",
            capability_snapshot_json,
        }
    }

    fn host(root: &Path) -> (tempfile::TempDir, LocalRemoteHost, Uuid) {
        let data = tempfile::tempdir().unwrap();
        let mut store = RemoteStore::open(&data.path().join("remote.sqlite3")).unwrap();
        let session_id = Uuid::new_v4();
        let device_id = Uuid::new_v4();
        let device_key = [9_u8; 32];
        store
            .create_session(
                session_id,
                &root.display().to_string(),
                "model",
                "sha256:model",
                "{}",
                1,
            )
            .unwrap();
        store
            .register_device(device_id, "Phone", &device_key, 2)
            .unwrap();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 3)
            .unwrap();
        (
            data,
            LocalRemoteHost::new(
                store,
                session_id,
                device_id,
                device_key,
                identity(&root.display().to_string(), "{}"),
                10,
            )
            .unwrap(),
            session_id,
        )
    }

    fn write_driver(path: &str) -> ScriptDriver {
        ScriptDriver {
            steps: vec![
                ModelStep::Calls(vec![ToolCall {
                    name: "write_file".into(),
                    args: json!({"path": path, "content": "durable content"}),
                }]),
                ModelStep::Text("done".into()),
            ],
            index: 0,
        }
    }

    fn read_write_verify_driver(path: &str) -> ScriptDriver {
        ScriptDriver {
            steps: vec![
                ModelStep::Calls(vec![ToolCall {
                    name: "read_file".into(),
                    args: json!({"path": path}),
                }]),
                ModelStep::Calls(vec![ToolCall {
                    name: "write_file".into(),
                    args: json!({"path": path, "content": "durable content"}),
                }]),
                ModelStep::Calls(vec![ToolCall {
                    name: "read_file".into(),
                    args: json!({"path": path}),
                }]),
                ModelStep::Text("verified".into()),
            ],
            index: 0,
        }
    }

    fn answer_driver() -> ScriptDriver {
        ScriptDriver {
            steps: vec![ModelStep::Text("done".into())],
            index: 0,
        }
    }

    fn start_message(
        device_id: Uuid,
        session_id: Uuid,
        command_id: Uuid,
        turn_id: Uuid,
    ) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "protocol": "camelid.remote/v1",
            "message_id": Uuid::new_v4(),
            "kind": "command",
            "host_id": Uuid::new_v4(),
            "device_id": device_id,
            "session_id": session_id,
            "sent_at_unix_ms": 10,
            "payload": {
                "command": "start_turn",
                "command_id": command_id,
                "turn_id": turn_id,
                "text": "write through protocol"
            }
        }))
        .unwrap()
    }

    fn cancel_message(
        device_id: Uuid,
        session_id: Uuid,
        command_id: Uuid,
        turn_id: Uuid,
    ) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "protocol": "camelid.remote/v1",
            "message_id": Uuid::new_v4(),
            "kind": "command",
            "host_id": Uuid::new_v4(),
            "device_id": device_id,
            "session_id": session_id,
            "sent_at_unix_ms": 11,
            "payload": {
                "command": "cancel_turn",
                "command_id": command_id,
                "turn_id": turn_id
            }
        }))
        .unwrap()
    }

    fn approval_message(
        device_id: Uuid,
        session_id: Uuid,
        command_id: Uuid,
        turn_id: Uuid,
        required: &camelid_remote_store::StoredEvent,
        decision: &str,
    ) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "protocol": "camelid.remote/v1",
            "message_id": Uuid::new_v4(),
            "kind": "command",
            "host_id": Uuid::new_v4(),
            "device_id": device_id,
            "session_id": session_id,
            "sent_at_unix_ms": 12,
            "payload": {
                "command": "approval_decision",
                "command_id": command_id,
                "turn_id": turn_id,
                "call_id": required.payload["record"]["call_id"],
                "approval_id": required.payload["approval_id"],
                "action_digest": required.payload["record"]["action_digest"],
                "decision": decision
            }
        }))
        .unwrap()
    }

    fn receive_event(
        receiver: &Receiver<camelid_remote_store::StoredEvent>,
        event_type: &str,
    ) -> camelid_remote_store::StoredEvent {
        loop {
            let event = receiver.recv_timeout(Duration::from_secs(5)).unwrap();
            if event.event_type == event_type {
                return event;
            }
        }
    }

    #[test]
    fn local_host_approved_write_is_durable_replayable_and_idempotent() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(workspace.path().join("approved.txt"), "before").unwrap();
        let (_data, host, _session_id) = host(workspace.path());
        let command_id = Uuid::new_v4();
        let turn_id = Uuid::new_v4();
        let mut reporter = SilentReporter;
        assert_eq!(
            host.run_turn(
                command_id,
                turn_id,
                "write the file",
                &mut read_write_verify_driver("approved.txt"),
                &mut reporter,
                &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
                &config(workspace.path()),
                Decision::Once,
            )
            .unwrap(),
            LoopEnd::Answered
        );
        assert_eq!(
            std::fs::read_to_string(workspace.path().join("approved.txt")).unwrap(),
            "durable content"
        );
        let events = host.replay(0).unwrap();
        let names = events
            .iter()
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "user.message",
                "turn.accepted",
                "tool.call",
                "tool.result",
                "tool.call",
                "approval.required",
                "approval.settled",
                "tool.result",
                "tool.call",
                "tool.result",
                "model.answer",
                "turn.finished",
            ]
        );
        assert!(host
            .run_turn(
                command_id,
                turn_id,
                "write the file",
                &mut write_driver("second.txt"),
                &mut reporter,
                &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
                &config(workspace.path()),
                Decision::Once,
            )
            .is_err());
        assert_eq!(host.replay(0).unwrap().len(), 12);
        assert!(!workspace.path().join("second.txt").exists());
    }

    #[test]
    fn local_host_denied_write_never_mutates_and_turn_remains_reusable() {
        let workspace = tempfile::tempdir().unwrap();
        let (_data, host, _session_id) = host(workspace.path());
        let mut reporter = SilentReporter;
        assert_eq!(
            host.run_turn(
                Uuid::new_v4(),
                Uuid::new_v4(),
                "deny the write",
                &mut write_driver("denied.txt"),
                &mut reporter,
                &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
                &config(workspace.path()),
                Decision::No,
            )
            .unwrap(),
            LoopEnd::Answered
        );
        assert!(!workspace.path().join("denied.txt").exists());
        assert_eq!(
            host.run_turn(
                Uuid::new_v4(),
                Uuid::new_v4(),
                "allow next",
                &mut write_driver("next.txt"),
                &mut reporter,
                &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
                &config(workspace.path()),
                Decision::Once,
            )
            .unwrap(),
            LoopEnd::Answered
        );
        assert!(workspace.path().join("next.txt").exists());
    }

    #[test]
    fn local_host_rejects_revoked_device_before_accepting_a_turn() {
        let workspace = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        let mut store = RemoteStore::open(&data.path().join("remote.sqlite3")).unwrap();
        let session_id = Uuid::new_v4();
        let device_id = Uuid::new_v4();
        let key = [3_u8; 32];
        store
            .create_session(session_id, "workspace", "model", "sha256:model", "{}", 1)
            .unwrap();
        store
            .register_device(device_id, "Lost phone", &key, 2)
            .unwrap();
        store.revoke_device(device_id, 3).unwrap();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 4)
            .unwrap();
        let host = LocalRemoteHost::new(
            store,
            session_id,
            device_id,
            key,
            identity("workspace", "{}"),
            10,
        )
        .unwrap();
        assert!(host
            .run_turn(
                Uuid::new_v4(),
                Uuid::new_v4(),
                "write",
                &mut write_driver("blocked.txt"),
                &mut SilentReporter,
                &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
                &config(workspace.path()),
                Decision::Once,
            )
            .is_err());
        assert!(host.replay(0).unwrap().is_empty());
        assert!(!workspace.path().join("blocked.txt").exists());
    }

    #[test]
    fn local_host_restores_only_identity_bound_typed_context() {
        let workspace = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        let path = data.path().join("remote.sqlite3");
        let session_id = Uuid::new_v4();
        let device_id = Uuid::new_v4();
        let key = [6_u8; 32];
        let root = workspace.path().display().to_string();
        let transcript = vec![
            AgentMsg::User("remember this".into()),
            AgentMsg::Assistant("stored".into()),
        ];
        let plan = vec![camelid_agent_runtime::PlanStep {
            status: camelid_agent_runtime::PlanStatus::InProgress,
            text: "continue safely".into(),
        }];
        let mut store = RemoteStore::open(&path).unwrap();
        store
            .create_session(
                session_id,
                &root,
                "model",
                "sha256:model",
                "{\"tools\":[\"write_file\"]}",
                1,
            )
            .unwrap();
        store.register_device(device_id, "Phone", &key, 2).unwrap();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 3)
            .unwrap();
        let turn_id = Uuid::new_v4();
        let AcceptTurn::Accepted { .. } = store
            .accept_start_turn(AcceptStartTurn {
                device_id,
                command_id: Uuid::new_v4(),
                request_digest: "sha256:start",
                session_id,
                turn_id,
                user_text: "remember this",
                created_at_unix_ms: 4,
            })
            .unwrap()
        else {
            panic!("expected accepted")
        };
        store
            .complete_turn(CompleteTurn {
                session_id,
                turn_id,
                outcome: "completed",
                assistant_text: Some("stored"),
                transcript_json: &serde_json::to_string(&transcript).unwrap(),
                plan_json: &serde_json::to_string(&plan).unwrap(),
                finished_at_unix_ms: 5,
            })
            .unwrap();
        drop(store);

        let restored = LocalRemoteHost::new(
            RemoteStore::open(&path).unwrap(),
            session_id,
            device_id,
            key,
            identity(&root, "{\"tools\":[\"write_file\"]}"),
            10,
        )
        .unwrap();
        assert_eq!(restored.runtime.transcript().snapshot().unwrap().len(), 2);
        assert_eq!(restored.runtime.plan().snapshot().unwrap(), plan);
        drop(restored);

        assert!(matches!(
            LocalRemoteHost::new(
                RemoteStore::open(&path).unwrap(),
                session_id,
                device_id,
                key,
                identity(&root, "{\"tools\":[]}"),
                10,
            ),
            Err(StoreError::Conflict)
        ));
        assert!(matches!(
            LocalRemoteHost::new(
                RemoteStore::open(&path).unwrap(),
                session_id,
                device_id,
                key,
                identity("different-root", "{\"tools\":[\"write_file\"]}"),
                10,
            ),
            Err(StoreError::Conflict)
        ));
        assert!(matches!(
            LocalRemoteHost::new(
                RemoteStore::open(&path).unwrap(),
                session_id,
                device_id,
                key,
                HostIdentity {
                    canonical_root: &root,
                    model_id: "different-model",
                    model_sha256: "sha256:model",
                    capability_snapshot_json: "{\"tools\":[\"write_file\"]}",
                },
                10,
            ),
            Err(StoreError::Conflict)
        ));
        assert!(matches!(
            LocalRemoteHost::new(
                RemoteStore::open(&path).unwrap(),
                session_id,
                device_id,
                key,
                HostIdentity {
                    canonical_root: &root,
                    model_id: "model",
                    model_sha256: "sha256:different",
                    capability_snapshot_json: "{\"tools\":[\"write_file\"]}",
                },
                10,
            ),
            Err(StoreError::Conflict)
        ));
    }

    #[test]
    fn local_host_abort_settles_pending_approval_without_mutation() {
        let workspace = tempfile::tempdir().unwrap();
        let (_data, host, _session_id) = host(workspace.path());
        assert_eq!(
            host.run_turn(
                Uuid::new_v4(),
                Uuid::new_v4(),
                "abort",
                &mut write_driver("aborted.txt"),
                &mut SilentReporter,
                &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
                &config(workspace.path()),
                Decision::Abort,
            )
            .unwrap(),
            LoopEnd::Aborted
        );
        assert!(!workspace.path().join("aborted.txt").exists());
        let events = host.replay(0).unwrap();
        assert!(events
            .iter()
            .any(|event| event.event_type == "approval.settled"));
        assert_eq!(events.last().unwrap().event_type, "turn.finished");
    }

    #[test]
    fn local_host_database_failure_before_approval_causes_no_mutation() {
        let workspace = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        let mut store = RemoteStore::open(&data.path().join("remote.sqlite3")).unwrap();
        let session_id = Uuid::new_v4();
        let device_id = Uuid::new_v4();
        let key = [4_u8; 32];
        store
            .create_session(session_id, "workspace", "model", "sha256:model", "{}", 1)
            .unwrap();
        store.register_device(device_id, "Phone", &key, 2).unwrap();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 3)
            .unwrap();
        store
            .execute_batch_for_test(
                "CREATE TRIGGER reject_approval_event BEFORE INSERT ON remote_events
             WHEN NEW.event_type = 'approval.required'
             BEGIN SELECT RAISE(ABORT, 'injected persistence failure'); END;",
            )
            .unwrap();
        let host = LocalRemoteHost::new(
            store,
            session_id,
            device_id,
            key,
            identity("workspace", "{}"),
            10,
        )
        .unwrap();
        let result = host.run_turn(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "write",
            &mut write_driver("not-written.txt"),
            &mut SilentReporter,
            &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
            &config(workspace.path()),
            Decision::Once,
        );
        assert_eq!(result.unwrap(), LoopEnd::DriverError);
        assert!(!workspace.path().join("not-written.txt").exists());
        let events = host.replay(0).unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event_type == "approval.required")
                .count(),
            0
        );
        assert_eq!(events.last().unwrap().event_type, "turn.finished");
        assert_eq!(events.last().unwrap().payload["outcome"], "driver_error");
    }

    #[test]
    fn local_host_cancellation_during_model_step_discards_partial_answer() {
        let workspace = tempfile::tempdir().unwrap();
        let (_data, host, _session_id) = host(workspace.path());
        let mut driver = CancellingDriver {
            cancel: host.runtime.cancel().clone(),
        };
        assert_eq!(
            host.run_turn(
                Uuid::new_v4(),
                Uuid::new_v4(),
                "cancel during model",
                &mut driver,
                &mut SilentReporter,
                &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
                &config(workspace.path()),
                Decision::No,
            )
            .unwrap(),
            LoopEnd::Aborted
        );
        let events = host.replay(0).unwrap();
        assert!(!events
            .iter()
            .any(|event| event.event_type == "model.answer"));
        assert_eq!(events.last().unwrap().payload["outcome"], "aborted");
    }

    #[test]
    fn local_host_protocol_cancel_is_durable_idempotent_and_interrupts_inference() {
        let workspace = tempfile::tempdir().unwrap();
        let (_data, host, session_id) = host(workspace.path());
        let host = Arc::new(host);
        let command_id = Uuid::new_v4();
        let turn_id = Uuid::new_v4();
        let (entered_sender, entered_receiver) = std::sync::mpsc::channel();
        let (release_sender, release_receiver) = std::sync::mpsc::channel();
        let thread_host = Arc::clone(&host);
        let root = workspace.path().to_path_buf();
        let worker = std::thread::spawn(move || {
            thread_host.run_turn(
                Uuid::new_v4(),
                turn_id,
                "cancel from phone",
                &mut BlockingDriver {
                    entered: entered_sender,
                    release: release_receiver,
                },
                &mut SilentReporter,
                &Sandbox::new(&root, false, Duration::from_secs(5)).unwrap(),
                &config(&root),
                Decision::No,
            )
        });
        entered_receiver.recv().unwrap();
        let message = cancel_message(host.device_id, session_id, command_id, turn_id);
        host.cancel_message(&message).unwrap();
        host.cancel_message(&message).unwrap();
        release_sender.send(()).unwrap();
        assert_eq!(worker.join().unwrap().unwrap(), LoopEnd::Aborted);
        let events = host.replay(0).unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event_type == "session.state_changed")
                .count(),
            1
        );
        assert!(!events
            .iter()
            .any(|event| event.event_type == "model.answer"));
        assert_eq!(events.last().unwrap().event_type, "turn.finished");
        assert_eq!(events.last().unwrap().payload["outcome"], "aborted");

        let (next_entered_sender, next_entered_receiver) = std::sync::mpsc::channel();
        let (next_release_sender, next_release_receiver) = std::sync::mpsc::channel();
        let next_host = Arc::clone(&host);
        let next_root = workspace.path().to_path_buf();
        let next_worker = std::thread::spawn(move || {
            next_host.run_turn(
                Uuid::new_v4(),
                Uuid::new_v4(),
                "new turn",
                &mut BlockingDriver {
                    entered: next_entered_sender,
                    release: next_release_receiver,
                },
                &mut SilentReporter,
                &Sandbox::new(&next_root, false, Duration::from_secs(5)).unwrap(),
                &config(&next_root),
                Decision::No,
            )
        });
        next_entered_receiver.recv().unwrap();
        host.cancel_message(&message).unwrap();
        next_release_sender.send(()).unwrap();
        assert_eq!(next_worker.join().unwrap().unwrap(), LoopEnd::Answered);
    }

    #[test]
    fn local_host_protocol_approval_commits_before_waking_and_executes_once() {
        let workspace = tempfile::tempdir().unwrap();
        let (_data, host, session_id) = host(workspace.path());
        let receiver = host.subscribe().unwrap();
        let host = Arc::new(host);
        let turn_id = Uuid::new_v4();
        let start = start_message(host.device_id, session_id, Uuid::new_v4(), turn_id);
        let root = workspace.path().to_path_buf();
        let thread_host = Arc::clone(&host);
        let worker = std::thread::spawn(move || {
            thread_host.run_message(
                &start,
                &mut write_driver("remote-approved.txt"),
                &mut SilentReporter,
                &Sandbox::new(&root, false, Duration::from_secs(5)).unwrap(),
                &config(&root),
            )
        });
        let required = receive_event(&receiver, "approval.required");
        let decision = approval_message(
            host.device_id,
            session_id,
            Uuid::new_v4(),
            turn_id,
            &required,
            "allow_once",
        );
        host.approval_message(&decision).unwrap();
        host.approval_message(&decision).unwrap();
        assert_eq!(worker.join().unwrap().unwrap(), LoopEnd::Answered);
        assert_eq!(
            std::fs::read_to_string(workspace.path().join("remote-approved.txt")).unwrap(),
            "durable content"
        );
        let events = host.replay(0).unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event_type == "approval.settled")
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event_type == "tool.result")
                .count(),
            1
        );
    }

    #[test]
    fn local_host_protocol_approval_timeout_aborts_without_mutation() {
        let workspace = tempfile::tempdir().unwrap();
        let (_data, mut host, session_id) = host(workspace.path());
        host.set_approval_timeout(Duration::from_millis(10));
        let turn_id = Uuid::new_v4();
        let start = start_message(host.device_id, session_id, Uuid::new_v4(), turn_id);
        assert_eq!(
            host.run_message(
                &start,
                &mut write_driver("expired.txt"),
                &mut SilentReporter,
                &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
                &config(workspace.path()),
            )
            .unwrap(),
            LoopEnd::Aborted
        );
        assert!(!workspace.path().join("expired.txt").exists());
        let events = host.replay(0).unwrap();
        assert!(events
            .iter()
            .any(|event| event.event_type == "approval.expired"));
        assert_eq!(events.last().unwrap().event_type, "turn.finished");
        assert_eq!(events.last().unwrap().payload["outcome"], "aborted");
    }

    #[test]
    fn local_host_protocol_entry_validates_envelope_before_state_changes() {
        let workspace = tempfile::tempdir().unwrap();
        let (_data, host, session_id) = host(workspace.path());
        let command_id = Uuid::new_v4();
        let turn_id = Uuid::new_v4();
        let message = start_message(host.device_id, session_id, command_id, turn_id);
        assert_eq!(
            host.run_message(
                &message,
                &mut answer_driver(),
                &mut SilentReporter,
                &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
                &config(workspace.path()),
            )
            .unwrap(),
            LoopEnd::Answered
        );
        assert!(host
            .replay(0)
            .unwrap()
            .iter()
            .any(|event| event.event_type == "model.answer"));

        let before = host.replay(0).unwrap().len();
        let wrong_session = start_message(
            host.device_id,
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
        );
        assert!(host
            .run_message(
                &wrong_session,
                &mut answer_driver(),
                &mut SilentReporter,
                &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
                &config(workspace.path()),
            )
            .is_err());
        assert!(host
            .run_message(
                b"not-json",
                &mut answer_driver(),
                &mut SilentReporter,
                &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
                &config(workspace.path()),
            )
            .is_err());
        assert_eq!(host.replay(0).unwrap().len(), before);
        assert!(!workspace.path().join("wrong.txt").exists());
        assert!(!workspace.path().join("malformed.txt").exists());
    }

    #[test]
    fn local_host_start_acceptance_is_durable_and_retries_return_stored_status() {
        let workspace = tempfile::tempdir().unwrap();
        let (_data, host, session_id) = host(workspace.path());
        let command_id = Uuid::new_v4();
        let turn_id = Uuid::new_v4();
        let message = start_message(host.device_id, session_id, command_id, turn_id);
        let StartTurnAcceptance::Accepted(turn) = host.accept_start_message(&message).unwrap()
        else {
            panic!("expected accepted start")
        };
        let StartTurnAcceptance::Duplicate(duplicate) =
            host.accept_start_message(&message).unwrap()
        else {
            panic!("expected accepted duplicate")
        };
        assert_eq!(duplicate.status, "accepted");

        assert_eq!(
            host.run_accepted_turn(
                turn,
                &mut answer_driver(),
                &mut SilentReporter,
                &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
                &config(workspace.path()),
            )
            .unwrap(),
            LoopEnd::Answered
        );
        let StartTurnAcceptance::Duplicate(duplicate) =
            host.accept_start_message(&message).unwrap()
        else {
            panic!("expected completed duplicate")
        };
        assert_eq!(duplicate.status, "applied");
    }

    #[test]
    fn local_host_dropped_live_subscriber_recovers_every_event_by_replay() {
        let workspace = tempfile::tempdir().unwrap();
        let (_data, host, _session_id) = host(workspace.path());
        let receiver = host.subscribe().unwrap();
        drop(receiver);
        assert_eq!(
            host.run_turn(
                Uuid::new_v4(),
                Uuid::new_v4(),
                "continue without viewer",
                &mut write_driver("viewerless.txt"),
                &mut SilentReporter,
                &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
                &config(workspace.path()),
                Decision::Once,
            )
            .unwrap(),
            LoopEnd::Answered
        );
        assert!(workspace.path().join("viewerless.txt").exists());
        let replay = host.replay(0).unwrap();
        assert_eq!(replay.len(), 8);
        assert!(replay
            .windows(2)
            .all(|pair| pair[1].sequence == pair[0].sequence + 1));
    }

    #[test]
    fn local_host_restart_replays_exact_committed_events_after_transport_loss() {
        let workspace = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        let database = data.path().join("remote.sqlite3");
        let session_id = Uuid::new_v4();
        let device_id = Uuid::new_v4();
        let device_key = [7_u8; 32];
        let root = workspace.path().display().to_string();
        let mut store = RemoteStore::open(&database).unwrap();
        store
            .create_session(session_id, &root, "model", "sha256:model", "{}", 1)
            .unwrap();
        store
            .register_device(device_id, "Phone", &device_key, 2)
            .unwrap();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 3)
            .unwrap();
        let host = LocalRemoteHost::new(
            store,
            session_id,
            device_id,
            device_key,
            identity(&root, "{}"),
            10,
        )
        .unwrap();
        assert_eq!(
            host.run_turn(
                Uuid::new_v4(),
                Uuid::new_v4(),
                "commit before relay loss",
                &mut answer_driver(),
                &mut SilentReporter,
                &Sandbox::new(workspace.path(), false, Duration::from_secs(5)).unwrap(),
                &config(workspace.path()),
                Decision::No,
            )
            .unwrap(),
            LoopEnd::Answered
        );
        let committed = host.replay(0).unwrap();
        assert!(!committed.is_empty());
        drop(host);

        let restarted = LocalRemoteHost::new(
            RemoteStore::open(&database).unwrap(),
            session_id,
            device_id,
            device_key,
            identity(&root, "{}"),
            20,
        )
        .unwrap();
        assert_eq!(restarted.replay(0).unwrap(), committed);
    }
}
