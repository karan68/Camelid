//! Structured, UI-independent events emitted by one Camelid agent runtime.
//!
//! Authority never comes from display strings. A gated call can be created
//! only from a validated protocol [`ApprovalRecord`], and its digest is computed
//! by the constructor. Renderers may display `detail`; command processors must
//! branch only on typed fields and digests.

use camelid_agent_runtime::PlanStep;
use camelid_remote_protocol::{ApprovalRecord, ApprovalRisk, ProtocolError};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const MAX_TOOL_NAME_BYTES: usize = 128;
const MAX_DETAIL_BYTES: usize = 1024 * 1024;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AgentEventError {
    #[error("invalid agent event")]
    Invalid,
    #[error("agent event is too large")]
    TooLarge,
    #[error("agent event sink is unavailable")]
    SinkUnavailable,
}

impl From<ProtocolError> for AgentEventError {
    fn from(error: ProtocolError) -> Self {
        match error {
            ProtocolError::MessageTooLarge { .. } => Self::TooLarge,
            _ => Self::Invalid,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolRisk {
    Read,
    Write,
    Exec,
    Network,
    Plan,
}

impl From<ApprovalRisk> for ToolRisk {
    fn from(risk: ApprovalRisk) -> Self {
        match risk {
            ApprovalRisk::Write => Self::Write,
            ApprovalRisk::Exec => Self::Exec,
            ApprovalRisk::Network => Self::Network,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalTier {
    Auto,
    Confirm,
    Deny,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidatedToolCall {
    call_id: Uuid,
    tool: String,
    risk: ToolRisk,
    approval_tier: ApprovalTier,
    detail: String,
    approval_record: Option<ApprovalRecord>,
    action_digest: Option<String>,
}

impl ValidatedToolCall {
    pub fn ungated(
        call_id: Uuid,
        tool: impl Into<String>,
        risk: ToolRisk,
        approval_tier: ApprovalTier,
        detail: impl Into<String>,
    ) -> Result<Self, AgentEventError> {
        if approval_tier == ApprovalTier::Confirm {
            return Err(AgentEventError::Invalid);
        }
        let tool = tool.into();
        let detail = detail.into();
        validate_display(&tool, &detail)?;
        Ok(Self {
            call_id,
            tool,
            risk,
            approval_tier,
            detail,
            approval_record: None,
            action_digest: None,
        })
    }

    pub fn gated(
        call_id: Uuid,
        approval_record: ApprovalRecord,
        detail: impl Into<String>,
    ) -> Result<Self, AgentEventError> {
        approval_record.validate()?;
        let action_digest = approval_record.digest()?;
        let tool = approval_record.tool.clone();
        let risk = approval_record.risk.into();
        let detail = detail.into();
        validate_display(&tool, &detail)?;
        Ok(Self {
            call_id,
            tool,
            risk,
            approval_tier: ApprovalTier::Confirm,
            detail,
            approval_record: Some(approval_record),
            action_digest: Some(action_digest),
        })
    }

    pub fn call_id(&self) -> Uuid {
        self.call_id
    }

    pub fn tool(&self) -> &str {
        &self.tool
    }

    pub fn risk(&self) -> ToolRisk {
        self.risk
    }

    pub fn approval_tier(&self) -> ApprovalTier {
        self.approval_tier
    }

    /// Human display only. Never parse this back into an action.
    pub fn detail(&self) -> &str {
        &self.detail
    }

    pub fn approval_record(&self) -> Option<&ApprovalRecord> {
        self.approval_record.as_ref()
    }

    pub fn action_digest(&self) -> Option<&str> {
        self.action_digest.as_deref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    AllowOnce,
    Deny,
    AbortTurn,
    Expired,
    InvalidatedByCancel,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalSettlement {
    pub approval_id: Uuid,
    pub call_id: Uuid,
    pub action_digest: String,
    pub decision: ApprovalDecision,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResultRecord {
    pub call_id: Uuid,
    pub tool: String,
    pub is_error: bool,
    /// Untrusted model data, never authority.
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelTiming {
    pub total_ms: u64,
    pub ttft_ms: Option<u64>,
    pub output_tokens: Option<u32>,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum AgentEvent {
    ModelDelta {
        content: String,
    },
    ModelAnswer {
        content: String,
    },
    PlanUpdated {
        steps: Vec<PlanStep>,
    },
    ToolCall {
        record: ValidatedToolCall,
    },
    ApprovalRequired {
        approval_id: Uuid,
        record: ValidatedToolCall,
    },
    ApprovalSettled {
        settlement: ApprovalSettlement,
    },
    ToolResult {
        record: ToolResultRecord,
    },
    Notice {
        content: String,
    },
    Timing {
        metrics: ModelTiming,
    },
}

pub trait AgentEventSink {
    fn emit(&mut self, event: AgentEvent) -> Result<(), AgentEventError>;
}

fn validate_display(tool: &str, detail: &str) -> Result<(), AgentEventError> {
    if tool.is_empty()
        || tool.len() > MAX_TOOL_NAME_BYTES
        || !tool
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(AgentEventError::Invalid);
    }
    if detail.len() > MAX_DETAIL_BYTES {
        return Err(AgentEventError::TooLarge);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use camelid_remote_protocol::{ApprovalAction, ResolvedTarget, APPROVAL_RECORD_SCHEMA};

    use super::*;

    fn edit_record(new: &str) -> ApprovalRecord {
        ApprovalRecord {
            schema: APPROVAL_RECORD_SCHEMA.into(),
            tool: "edit_file".into(),
            risk: ApprovalRisk::Write,
            action: ApprovalAction::EditFile {
                target: ResolvedTarget {
                    canonical_native: "/work/src/lib.rs".into(),
                    workspace_display: "src/lib.rs".into(),
                },
                old: "old".into(),
                new: new.into(),
            },
        }
    }

    #[test]
    fn gated_call_binds_validated_record_and_computed_digest() {
        let record = edit_record("new");
        let expected = record.digest().unwrap();
        let call =
            ValidatedToolCall::gated(Uuid::nil(), record, "model prose is display only").unwrap();
        assert_eq!(call.approval_tier(), ApprovalTier::Confirm);
        assert_eq!(call.tool(), "edit_file");
        assert_eq!(call.risk(), ToolRisk::Write);
        assert_eq!(call.action_digest(), Some(expected.as_str()));
        assert!(call.approval_record().is_some());
        assert_eq!(call.detail(), "model prose is display only");
    }

    #[test]
    fn executable_field_change_changes_the_gated_digest() {
        let first =
            ValidatedToolCall::gated(Uuid::nil(), edit_record("new"), "same detail").unwrap();
        let second =
            ValidatedToolCall::gated(Uuid::nil(), edit_record("different"), "same detail").unwrap();
        assert_ne!(first.action_digest(), second.action_digest());
    }

    #[test]
    fn display_prose_cannot_create_confirm_authority() {
        assert!(matches!(
            ValidatedToolCall::ungated(
                Uuid::nil(),
                "edit_file",
                ToolRisk::Write,
                ApprovalTier::Confirm,
                "approve whatever the model said"
            ),
            Err(AgentEventError::Invalid)
        ));
    }

    #[test]
    fn invalid_record_and_unbounded_display_fail_closed() {
        let mut invalid = edit_record("new");
        invalid.tool = "read_file".into();
        assert!(ValidatedToolCall::gated(Uuid::nil(), invalid, "detail").is_err());
        assert!(matches!(
            ValidatedToolCall::ungated(
                Uuid::nil(),
                "read_file",
                ToolRisk::Read,
                ApprovalTier::Auto,
                "x".repeat(MAX_DETAIL_BYTES + 1)
            ),
            Err(AgentEventError::TooLarge)
        ));
    }

    #[derive(Default)]
    struct MemorySink(Vec<AgentEvent>);

    impl AgentEventSink for MemorySink {
        fn emit(&mut self, event: AgentEvent) -> Result<(), AgentEventError> {
            self.0.push(event);
            Ok(())
        }
    }

    #[test]
    fn sink_receives_typed_events_without_render_parsing() {
        let call =
            ValidatedToolCall::gated(Uuid::nil(), edit_record("new"), "rendered detail").unwrap();
        let mut sink = MemorySink::default();
        sink.emit(AgentEvent::ToolCall { record: call }).unwrap();
        assert!(matches!(sink.0.as_slice(), [AgentEvent::ToolCall { .. }]));
    }
}
