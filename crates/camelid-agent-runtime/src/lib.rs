//! Session-owned mutable state for one Camelid agent runtime.
//!
//! This crate contains no model, tool, filesystem, or approval execution. It
//! replaces process-global mutable state with independently constructible state
//! holders that the existing CLI/TUI and the future remote host can compose.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use serde::{Deserialize, Serialize};

pub const MAX_PLAN_STEPS: usize = 20;
pub const MAX_PLAN_STEP_CHARS: usize = 160;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RuntimeStateError {
    #[error("agent runtime state is unavailable")]
    Unavailable,
}

#[derive(Clone, Default)]
pub struct CancellationToken {
    requested: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn request(&self) {
        self.requested.store(true, Ordering::Release);
    }

    pub fn reset(&self) {
        self.requested.store(false, Ordering::Release);
    }

    pub fn is_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    pub fn as_atomic(&self) -> &AtomicBool {
        &self.requested
    }

    pub fn as_atomic_arc(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.requested)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Pending,
    InProgress,
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanStep {
    pub status: PlanStatus,
    pub text: String,
}

#[derive(Default)]
pub struct PlanState {
    steps: Mutex<Vec<PlanStep>>,
}

impl PlanState {
    pub fn replace(&self, mut steps: Vec<PlanStep>) -> Result<Vec<PlanStep>, RuntimeStateError> {
        normalize_plan(&mut steps);
        lock(&self.steps)?.clone_from(&steps);
        Ok(steps)
    }

    pub fn snapshot(&self) -> Result<Vec<PlanStep>, RuntimeStateError> {
        Ok(lock(&self.steps)?.clone())
    }

    pub fn clear(&self) -> Result<(), RuntimeStateError> {
        lock(&self.steps)?.clear();
        Ok(())
    }

    pub fn complete_all(&self) -> Result<usize, RuntimeStateError> {
        let mut steps = lock(&self.steps)?;
        let mut changed = 0;
        for step in steps.iter_mut() {
            if step.status != PlanStatus::Done {
                step.status = PlanStatus::Done;
                changed += 1;
            }
        }
        Ok(changed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointRecord {
    pub relative_path: String,
    pub backup: Option<PathBuf>,
    pub tool: String,
    pub post_hash: Option<u64>,
}

#[derive(Default)]
pub struct CheckpointStore {
    records: Mutex<Vec<CheckpointRecord>>,
}

impl CheckpointStore {
    pub fn push(&self, record: CheckpointRecord) -> Result<(), RuntimeStateError> {
        lock(&self.records)?.push(record);
        Ok(())
    }

    pub fn snapshot(&self) -> Result<Vec<CheckpointRecord>, RuntimeStateError> {
        Ok(lock(&self.records)?.clone())
    }

    pub fn latest(&self) -> Result<Option<CheckpointRecord>, RuntimeStateError> {
        Ok(lock(&self.records)?.last().cloned())
    }

    pub fn pop_latest(&self) -> Result<Option<CheckpointRecord>, RuntimeStateError> {
        Ok(lock(&self.records)?.pop())
    }

    pub fn clear(&self) -> Result<(), RuntimeStateError> {
        lock(&self.records)?.clear();
        Ok(())
    }
}

pub struct TranscriptState<Message> {
    messages: Mutex<Vec<Message>>,
}

impl<Message> Default for TranscriptState<Message> {
    fn default() -> Self {
        Self {
            messages: Mutex::new(Vec::new()),
        }
    }
}

impl<Message: Clone> TranscriptState<Message> {
    pub fn push(&self, message: Message) -> Result<(), RuntimeStateError> {
        lock(&self.messages)?.push(message);
        Ok(())
    }

    pub fn replace(&self, messages: Vec<Message>) -> Result<(), RuntimeStateError> {
        *lock(&self.messages)? = messages;
        Ok(())
    }

    pub fn snapshot(&self) -> Result<Vec<Message>, RuntimeStateError> {
        Ok(lock(&self.messages)?.clone())
    }

    pub fn clear(&self) -> Result<(), RuntimeStateError> {
        lock(&self.messages)?.clear();
        Ok(())
    }
}

#[derive(Clone)]
pub struct AgentRuntime<Message> {
    inner: Arc<AgentRuntimeInner<Message>>,
}

struct AgentRuntimeInner<Message> {
    cancel: CancellationToken,
    plan: PlanState,
    checkpoints: CheckpointStore,
    transcript: TranscriptState<Message>,
}

impl<Message> Default for AgentRuntime<Message> {
    fn default() -> Self {
        Self {
            inner: Arc::new(AgentRuntimeInner {
                cancel: CancellationToken::default(),
                plan: PlanState::default(),
                checkpoints: CheckpointStore::default(),
                transcript: TranscriptState::default(),
            }),
        }
    }
}

impl<Message> AgentRuntime<Message> {
    pub fn cancel(&self) -> &CancellationToken {
        &self.inner.cancel
    }

    pub fn plan(&self) -> &PlanState {
        &self.inner.plan
    }

    pub fn checkpoints(&self) -> &CheckpointStore {
        &self.inner.checkpoints
    }

    pub fn transcript(&self) -> &TranscriptState<Message> {
        &self.inner.transcript
    }
}

fn normalize_plan(steps: &mut Vec<PlanStep>) {
    steps.truncate(MAX_PLAN_STEPS);
    for step in steps {
        let text = step.text.trim();
        step.text = if text.chars().count() > MAX_PLAN_STEP_CHARS {
            text.chars().take(MAX_PLAN_STEP_CHARS).collect::<String>() + "..."
        } else {
            text.to_string()
        };
    }
}

fn lock<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>, RuntimeStateError> {
    mutex.lock().map_err(|_| RuntimeStateError::Unavailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(status: PlanStatus, text: &str) -> PlanStep {
        PlanStep {
            status,
            text: text.into(),
        }
    }

    #[test]
    fn two_runtimes_share_no_mutable_state() {
        let first = AgentRuntime::<String>::default();
        let second = AgentRuntime::<String>::default();

        first.cancel().request();
        first
            .plan()
            .replace(vec![step(PlanStatus::InProgress, "first plan")])
            .unwrap();
        first
            .checkpoints()
            .push(CheckpointRecord {
                relative_path: "first.txt".into(),
                backup: None,
                tool: "write_file".into(),
                post_hash: Some(1),
            })
            .unwrap();
        first.transcript().push("first message".into()).unwrap();

        assert!(first.cancel().is_requested());
        assert!(!second.cancel().is_requested());
        assert_eq!(first.plan().snapshot().unwrap().len(), 1);
        assert!(second.plan().snapshot().unwrap().is_empty());
        assert_eq!(first.checkpoints().snapshot().unwrap().len(), 1);
        assert!(second.checkpoints().snapshot().unwrap().is_empty());
        assert_eq!(first.transcript().snapshot().unwrap(), ["first message"]);
        assert!(second.transcript().snapshot().unwrap().is_empty());
    }

    #[test]
    fn cancellation_clones_share_only_their_runtime_flag() {
        let first = CancellationToken::default();
        let first_clone = first.clone();
        let second = CancellationToken::default();
        first_clone.request();
        assert!(first.is_requested());
        assert!(!second.is_requested());
        first.reset();
        assert!(!first_clone.is_requested());
    }

    #[test]
    fn runtime_clones_share_one_session_without_leaking_to_another() {
        let runtime = AgentRuntime::<String>::default();
        let worker = runtime.clone();
        let other = AgentRuntime::<String>::default();

        worker.cancel().request();
        worker
            .plan()
            .replace(vec![step(PlanStatus::InProgress, "shared plan")])
            .unwrap();
        worker.transcript().push("shared message".into()).unwrap();

        assert!(runtime.cancel().is_requested());
        assert_eq!(runtime.plan().snapshot().unwrap()[0].text, "shared plan");
        assert_eq!(runtime.transcript().snapshot().unwrap(), ["shared message"]);
        assert!(!other.cancel().is_requested());
        assert!(other.plan().snapshot().unwrap().is_empty());
        assert!(other.transcript().snapshot().unwrap().is_empty());
    }

    #[test]
    fn plans_are_bounded_normalized_and_completed_idempotently() {
        let plan = PlanState::default();
        let long = format!("  {}  ", "x".repeat(MAX_PLAN_STEP_CHARS + 20));
        let stored = plan
            .replace(
                (0..MAX_PLAN_STEPS + 5)
                    .map(|_| step(PlanStatus::Pending, &long))
                    .collect(),
            )
            .unwrap();
        assert_eq!(stored.len(), MAX_PLAN_STEPS);
        assert!(stored[0].text.chars().count() <= MAX_PLAN_STEP_CHARS + 3);
        assert!(!stored[0].text.starts_with(' '));
        assert_eq!(plan.complete_all().unwrap(), MAX_PLAN_STEPS);
        assert_eq!(plan.complete_all().unwrap(), 0);
        assert!(plan
            .snapshot()
            .unwrap()
            .iter()
            .all(|step| step.status == PlanStatus::Done));
    }

    #[test]
    fn checkpoint_and_transcript_operations_are_lifo_and_replaceable() {
        let runtime = AgentRuntime::<u32>::default();
        runtime
            .checkpoints()
            .push(CheckpointRecord {
                relative_path: "a".into(),
                backup: Some(PathBuf::from("backup-a")),
                tool: "edit_file".into(),
                post_hash: Some(7),
            })
            .unwrap();
        assert_eq!(
            runtime
                .checkpoints()
                .latest()
                .unwrap()
                .unwrap()
                .relative_path,
            "a"
        );
        assert_eq!(
            runtime
                .checkpoints()
                .pop_latest()
                .unwrap()
                .unwrap()
                .post_hash,
            Some(7)
        );
        assert!(runtime.checkpoints().latest().unwrap().is_none());

        runtime.transcript().push(1).unwrap();
        runtime.transcript().replace(vec![2, 3]).unwrap();
        assert_eq!(runtime.transcript().snapshot().unwrap(), [2, 3]);
    }
}
