//! Versioned application protocol for remote agent control.
//!
//! Phase 0 deliberately exposes no transport or host command. This module pins
//! the plaintext contract that will eventually live inside the authenticated
//! Noise channel: bounded parsing, strict privilege-bearing commands, forward-
//! compatible event envelopes, pairing QR validation, and canonical approval
//! digests. The relay never imports or parses these types.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use url::Url;
use uuid::Uuid;

pub const PROTOCOL: &str = "camelid.remote/v1";
pub const APPROVAL_RECORD_SCHEMA: &str = "camelid.approval-record/v1";
pub const MAX_NOISE_RECORD_BYTES: usize = 65_535;
pub const NOISE_TAG_BYTES: usize = 16;
pub const MAX_TRANSPORT_PLAINTEXT_BYTES: usize = MAX_NOISE_RECORD_BYTES - NOISE_TAG_BYTES;
pub const CHUNK_HEADER_BYTES: usize = 64;
pub const MAX_CHUNK_DATA_BYTES: usize = MAX_TRANSPORT_PLAINTEXT_BYTES - CHUNK_HEADER_BYTES;
pub const MAX_MESSAGE_CHUNKS: usize = 18;
pub const MAX_INNER_MESSAGE_BYTES: usize = 1_114_112;
pub const MAX_TURN_TEXT_BYTES: usize = 4 * 1024;
pub const MAX_APPROVAL_RECORD_BYTES: usize = 1024 * 1024;
pub const MAX_REPLAY_EVENTS: u16 = 256;
pub const MAX_SESSION_CATALOG_ENTRIES: u16 = 64;

const MAX_RELAY_URL_BYTES: usize = 2 * 1024;
const MAX_EVENT_TOKEN_BYTES: usize = 64;
const MAX_RESULT_MESSAGE_BYTES: usize = 2 * 1024;
const MAX_PATH_BYTES: usize = 8 * 1024;
const MAX_SHELL_LAYERS: usize = 16;
const MAX_SHELL_LAYER_BYTES: usize = 64;
const MAX_SHELL_NOTE_BYTES: usize = 1024;
const MAX_METHOD_BYTES: usize = 32;
const MAX_DEVICE_LABEL_BYTES: usize = 128;
const MAX_PAIRING_CAPABILITIES: usize = 16;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ProtocolError {
    #[error("message_too_large: {actual} bytes exceeds {limit}")]
    MessageTooLarge { actual: usize, limit: usize },
    #[error("invalid_message: {0}")]
    InvalidMessage(String),
    #[error("unsupported_protocol")]
    UnsupportedProtocol,
    #[error("invalid_message: expected {expected} message")]
    UnexpectedKind { expected: &'static str },
    #[error("invalid_message: non-canonical number")]
    NonCanonicalNumber,
    #[error("invalid_message: invalid transport chunk")]
    InvalidChunk,
}

fn invalid(message: impl Into<String>) -> ProtocolError {
    ProtocolError::InvalidMessage(message.into())
}

fn bounded_serde_error(error: serde_json::Error) -> ProtocolError {
    let text = error.to_string();
    let mut end = text.len().min(256);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    invalid(text[..end].to_string())
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MessageKind {
    Command,
    CommandResult,
    EventBatch,
    ReplayRequest,
    ReplayEnd,
    SessionCatalogRequest,
    SessionCatalog,
    Ping,
    Pong,
    Error,
}

/// Decrypted application envelope. Unknown top-level fields are accepted so a
/// v1 peer can receive additive metadata. The finite `kind` vocabulary remains
/// strict because an unknown privileged message must never acquire semantics by
/// accident.
#[derive(Clone, Deserialize, Serialize, PartialEq)]
pub struct RemoteMessage {
    pub protocol: String,
    pub message_id: Uuid,
    pub kind: MessageKind,
    pub host_id: Uuid,
    pub device_id: Uuid,
    pub session_id: Option<Uuid>,
    pub sent_at_unix_ms: u64,
    pub payload: Value,
}

pub fn decode_message(input: &[u8]) -> Result<RemoteMessage, ProtocolError> {
    if input.len() > MAX_INNER_MESSAGE_BYTES {
        return Err(ProtocolError::MessageTooLarge {
            actual: input.len(),
            limit: MAX_INNER_MESSAGE_BYTES,
        });
    }
    let message: RemoteMessage = serde_json::from_slice(input).map_err(bounded_serde_error)?;
    if message.protocol != PROTOCOL {
        return Err(ProtocolError::UnsupportedProtocol);
    }
    Ok(message)
}

const CHUNK_MAGIC: &[u8; 4] = b"CMR1";
const CHUNK_VERSION: u8 = 1;

/// Split one canonical inner message into records that fit the Noise 65,535-byte
/// limit. The fixed header is itself encrypted and authenticated by Noise. Its
/// whole-message digest additionally prevents an implementation from accepting
/// an omitted, duplicated, reordered, or cross-message chunk sequence.
pub fn encode_chunks(message_id: Uuid, message: &[u8]) -> Result<Vec<Vec<u8>>, ProtocolError> {
    if message.is_empty() || message.len() > MAX_INNER_MESSAGE_BYTES {
        return Err(ProtocolError::MessageTooLarge {
            actual: message.len(),
            limit: MAX_INNER_MESSAGE_BYTES,
        });
    }
    let chunk_count = message.len().div_ceil(MAX_CHUNK_DATA_BYTES);
    if chunk_count > MAX_MESSAGE_CHUNKS {
        return Err(ProtocolError::MessageTooLarge {
            actual: message.len(),
            limit: MAX_INNER_MESSAGE_BYTES,
        });
    }
    let chunk_count_u16 = u16::try_from(chunk_count).map_err(|_| ProtocolError::InvalidChunk)?;
    let total_u32 = u32::try_from(message.len()).map_err(|_| ProtocolError::InvalidChunk)?;
    let digest: [u8; 32] = Sha256::digest(message).into();

    message
        .chunks(MAX_CHUNK_DATA_BYTES)
        .enumerate()
        .map(|(index, data)| {
            let index = u16::try_from(index).map_err(|_| ProtocolError::InvalidChunk)?;
            let mut chunk = Vec::with_capacity(CHUNK_HEADER_BYTES + data.len());
            chunk.extend_from_slice(CHUNK_MAGIC);
            chunk.push(CHUNK_VERSION);
            chunk.push(0);
            chunk.extend_from_slice(&[0, 0]);
            chunk.extend_from_slice(message_id.as_bytes());
            chunk.extend_from_slice(&index.to_be_bytes());
            chunk.extend_from_slice(&chunk_count_u16.to_be_bytes());
            chunk.extend_from_slice(&total_u32.to_be_bytes());
            chunk.extend_from_slice(&digest);
            chunk.extend_from_slice(data);
            Ok(chunk)
        })
        .collect()
}

struct DecodedChunk<'a> {
    message_id: Uuid,
    index: u16,
    count: u16,
    total: usize,
    digest: [u8; 32],
    data: &'a [u8],
}

fn decode_chunk(frame: &[u8]) -> Result<DecodedChunk<'_>, ProtocolError> {
    if !(CHUNK_HEADER_BYTES..=MAX_TRANSPORT_PLAINTEXT_BYTES).contains(&frame.len())
        || &frame[..4] != CHUNK_MAGIC
        || frame[4] != CHUNK_VERSION
        || frame[5..8] != [0, 0, 0]
    {
        return Err(ProtocolError::InvalidChunk);
    }
    let message_id = Uuid::from_slice(&frame[8..24]).map_err(|_| ProtocolError::InvalidChunk)?;
    let index = u16::from_be_bytes([frame[24], frame[25]]);
    let count = u16::from_be_bytes([frame[26], frame[27]]);
    let total = u32::from_be_bytes([frame[28], frame[29], frame[30], frame[31]]) as usize;
    let digest = frame[32..64]
        .try_into()
        .map_err(|_| ProtocolError::InvalidChunk)?;
    let data = &frame[CHUNK_HEADER_BYTES..];
    if count == 0
        || usize::from(count) > MAX_MESSAGE_CHUNKS
        || index >= count
        || total == 0
        || total > MAX_INNER_MESSAGE_BYTES
        || data.is_empty()
        || data.len() > MAX_CHUNK_DATA_BYTES
    {
        return Err(ProtocolError::InvalidChunk);
    }
    Ok(DecodedChunk {
        message_id,
        index,
        count,
        total,
        digest,
        data,
    })
}

#[derive(Default)]
pub struct ChunkReassembler {
    message_id: Option<Uuid>,
    count: u16,
    next_index: u16,
    total: usize,
    digest: [u8; 32],
    message: Vec<u8>,
}

impl ChunkReassembler {
    pub fn push(&mut self, frame: &[u8]) -> Result<Option<Vec<u8>>, ProtocolError> {
        let chunk = decode_chunk(frame)?;
        match self.message_id {
            None => {
                if chunk.index != 0 {
                    return Err(ProtocolError::InvalidChunk);
                }
                self.message_id = Some(chunk.message_id);
                self.count = chunk.count;
                self.total = chunk.total;
                self.digest = chunk.digest;
                self.message = Vec::with_capacity(chunk.total);
            }
            Some(message_id)
                if message_id != chunk.message_id
                    || self.count != chunk.count
                    || self.total != chunk.total
                    || self.digest != chunk.digest =>
            {
                return Err(ProtocolError::InvalidChunk);
            }
            Some(_) => {}
        }
        if chunk.index != self.next_index
            || self.message.len().saturating_add(chunk.data.len()) > self.total
        {
            return Err(ProtocolError::InvalidChunk);
        }
        self.message.extend_from_slice(chunk.data);
        self.next_index += 1;
        if self.next_index != self.count {
            return Ok(None);
        }
        if self.message.len() != self.total
            || <[u8; 32]>::from(Sha256::digest(&self.message)) != self.digest
        {
            return Err(ProtocolError::InvalidChunk);
        }
        let message = std::mem::take(&mut self.message);
        *self = Self::default();
        Ok(Some(message))
    }
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, tag = "command", rename_all = "snake_case")]
pub enum Command {
    StartTurn {
        command_id: Uuid,
        turn_id: Uuid,
        text: String,
    },
    ApprovalDecision {
        command_id: Uuid,
        turn_id: Uuid,
        call_id: Uuid,
        approval_id: Uuid,
        action_digest: String,
        decision: ApprovalDecision,
    },
    CancelTurn {
        command_id: Uuid,
        turn_id: Uuid,
    },
    CreateSession {
        command_id: Uuid,
        session_id: Uuid,
    },
    ActivateSession {
        command_id: Uuid,
        session_id: Uuid,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    AllowOnce,
    Deny,
    AbortTurn,
}

pub fn decode_command(message: &RemoteMessage) -> Result<Command, ProtocolError> {
    if message.kind != MessageKind::Command {
        return Err(ProtocolError::UnexpectedKind {
            expected: "command",
        });
    }
    let command: Command =
        serde_json::from_value(message.payload.clone()).map_err(bounded_serde_error)?;
    command.validate()?;
    Ok(command)
}

impl Command {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        match self {
            Self::StartTurn { text, .. } => {
                if text.trim().is_empty() {
                    return Err(invalid("start_turn text is empty"));
                }
                if text.len() > MAX_TURN_TEXT_BYTES {
                    return Err(ProtocolError::MessageTooLarge {
                        actual: text.len(),
                        limit: MAX_TURN_TEXT_BYTES,
                    });
                }
            }
            Self::ApprovalDecision { action_digest, .. } => {
                validate_sha256(action_digest)?;
            }
            Self::CancelTurn { .. } | Self::CreateSession { .. } | Self::ActivateSession { .. } => {
            }
        }
        Ok(())
    }
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReplayRequest {
    pub after_sequence: u64,
    pub limit: u16,
}

impl ReplayRequest {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if !(1..=MAX_REPLAY_EVENTS).contains(&self.limit) {
            return Err(invalid(format!(
                "replay limit must be between 1 and {MAX_REPLAY_EVENTS}"
            )));
        }
        Ok(())
    }
}

pub fn decode_replay_request(message: &RemoteMessage) -> Result<ReplayRequest, ProtocolError> {
    if message.kind != MessageKind::ReplayRequest {
        return Err(ProtocolError::UnexpectedKind {
            expected: "replay_request",
        });
    }
    let request: ReplayRequest =
        serde_json::from_value(message.payload.clone()).map_err(bounded_serde_error)?;
    request.validate()?;
    Ok(request)
}

#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RemoteEvent {
    pub sequence: u64,
    pub event_id: Uuid,
    pub turn_id: Option<Uuid>,
    pub event: String,
    pub created_at_unix_ms: u64,
    pub payload: Value,
}

impl RemoteEvent {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.sequence == 0 {
            return Err(invalid("event sequence starts at one"));
        }
        validate_token("event", &self.event, MAX_EVENT_TOKEN_BYTES)
    }
}

#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EventBatch {
    pub events: Vec<RemoteEvent>,
}

impl EventBatch {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.events.is_empty() || self.events.len() > usize::from(MAX_REPLAY_EVENTS) {
            return Err(invalid("event batch size is invalid"));
        }
        for event in &self.events {
            event.validate()?;
        }
        if !self
            .events
            .windows(2)
            .all(|pair| pair[1].sequence == pair[0].sequence + 1)
        {
            return Err(invalid("event batch sequence is not contiguous"));
        }
        Ok(())
    }
}

pub fn decode_event_batch(message: &RemoteMessage) -> Result<EventBatch, ProtocolError> {
    if message.kind != MessageKind::EventBatch {
        return Err(ProtocolError::UnexpectedKind {
            expected: "event_batch",
        });
    }
    let batch: EventBatch =
        serde_json::from_value(message.payload.clone()).map_err(bounded_serde_error)?;
    batch.validate()?;
    Ok(batch)
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RemoteSessionState {
    Armed,
    Idle,
    Running,
    WaitingApproval,
    Cancelling,
    Failed,
    Closed,
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReplayEnd {
    pub last_sequence: u64,
    pub has_more: bool,
    pub session_state: RemoteSessionState,
}

pub fn decode_replay_end(message: &RemoteMessage) -> Result<ReplayEnd, ProtocolError> {
    if message.kind != MessageKind::ReplayEnd {
        return Err(ProtocolError::UnexpectedKind {
            expected: "replay_end",
        });
    }
    serde_json::from_value(message.payload.clone()).map_err(bounded_serde_error)
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SessionCatalogCursor {
    pub updated_at_unix_ms: u64,
    pub history_id: Uuid,
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SessionCatalogRequest {
    pub cursor: Option<SessionCatalogCursor>,
    pub limit: u16,
    pub revision: Option<String>,
}

impl SessionCatalogRequest {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.limit == 0 || self.limit > MAX_SESSION_CATALOG_ENTRIES {
            return Err(invalid("session catalog limit is invalid"));
        }
        if let Some(revision) = self.revision.as_deref() {
            validate_sha256(revision)?;
            if self.cursor.is_none() {
                return Err(invalid("session catalog revision requires a cursor"));
            }
        }
        if self.cursor.is_some() && self.revision.is_none() {
            return Err(invalid("session catalog cursor requires a revision"));
        }
        Ok(())
    }
}

pub fn decode_session_catalog_request(
    message: &RemoteMessage,
) -> Result<SessionCatalogRequest, ProtocolError> {
    if message.kind != MessageKind::SessionCatalogRequest {
        return Err(ProtocolError::UnexpectedKind {
            expected: "session_catalog_request",
        });
    }
    if message.session_id.is_some() {
        return Err(invalid("session catalog request must be host scoped"));
    }
    let request: SessionCatalogRequest =
        serde_json::from_value(message.payload.clone()).map_err(bounded_serde_error)?;
    request.validate()?;
    Ok(request)
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionHistorySource {
    Remote,
    AgentSaved,
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SessionSummary {
    pub history_id: Uuid,
    pub source: SessionHistorySource,
    pub title: String,
    pub state: RemoteSessionState,
    pub canonical_root: String,
    pub model_id: String,
    pub model_sha256: Option<String>,
    pub created_at_unix_ms: u64,
    pub updated_at_unix_ms: u64,
    pub last_event_sequence: u64,
    pub active: bool,
    pub continuable: bool,
    pub refusal_code: Option<String>,
}

impl SessionSummary {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        validate_text("session title", &self.title, 256)?;
        validate_text("canonical root", &self.canonical_root, MAX_PATH_BYTES)?;
        validate_token("model id", &self.model_id, 256)?;
        if let Some(model_sha256) = self.model_sha256.as_deref() {
            validate_sha256(model_sha256)?;
        }
        if self.continuable && self.model_sha256.is_none() {
            return Err(invalid(
                "continuable session requires a model artifact digest",
            ));
        }
        if let Some(code) = self.refusal_code.as_deref() {
            validate_token("refusal code", code, MAX_EVENT_TOKEN_BYTES)?;
            if self.continuable {
                return Err(invalid("continuable session has a refusal code"));
            }
        }
        if !self.continuable && self.refusal_code.is_none() {
            return Err(invalid("non-continuable session requires a refusal code"));
        }
        Ok(())
    }
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SessionCatalog {
    pub active_session_id: Uuid,
    pub revision: String,
    pub sessions: Vec<SessionSummary>,
    pub next_cursor: Option<SessionCatalogCursor>,
}

impl SessionCatalog {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        validate_sha256(&self.revision)?;
        if self.sessions.len() > usize::from(MAX_SESSION_CATALOG_ENTRIES) {
            return Err(invalid("session catalog is too large"));
        }
        for summary in &self.sessions {
            summary.validate()?;
        }
        if !self.sessions.windows(2).all(|pair| {
            pair[0].updated_at_unix_ms > pair[1].updated_at_unix_ms
                || (pair[0].updated_at_unix_ms == pair[1].updated_at_unix_ms
                    && pair[0].history_id.as_bytes() < pair[1].history_id.as_bytes())
        }) {
            return Err(invalid("session catalog ordering is invalid"));
        }
        Ok(())
    }
}

pub fn decode_session_catalog(message: &RemoteMessage) -> Result<SessionCatalog, ProtocolError> {
    if message.kind != MessageKind::SessionCatalog {
        return Err(ProtocolError::UnexpectedKind {
            expected: "session_catalog",
        });
    }
    if message.session_id.is_some() {
        return Err(invalid("session catalog must be host scoped"));
    }
    let catalog: SessionCatalog =
        serde_json::from_value(message.payload.clone()).map_err(bounded_serde_error)?;
    catalog.validate()?;
    Ok(catalog)
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CommandStatus {
    Accepted,
    Applied,
    Rejected,
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CommandResult {
    pub command_id: Uuid,
    pub status: CommandStatus,
    pub code: String,
    pub message: String,
    pub current_event_sequence: u64,
}

impl CommandResult {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        validate_token("code", &self.code, MAX_EVENT_TOKEN_BYTES)?;
        validate_text("message", &self.message, MAX_RESULT_MESSAGE_BYTES)
    }
}

pub fn decode_command_result(message: &RemoteMessage) -> Result<CommandResult, ProtocolError> {
    if message.kind != MessageKind::CommandResult {
        return Err(ProtocolError::UnexpectedKind {
            expected: "command_result",
        });
    }
    let result: CommandResult =
        serde_json::from_value(message.payload.clone()).map_err(bounded_serde_error)?;
    result.validate()?;
    Ok(result)
}

/// QR payload shown only after an explicit local pairing action. Its capability
/// fields are redacted from `Debug` so an innocent diagnostic cannot print a
/// live route or one-time pairing secret.
#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairingQr {
    pub v: u8,
    pub relay_url: String,
    pub route_id: String,
    pub host_id: Uuid,
    pub host_noise_public: String,
    pub pairing_secret: String,
    pub expires_at_unix_ms: u64,
}

impl fmt::Debug for PairingQr {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairingQr")
            .field("v", &self.v)
            .field("relay_url", &self.relay_url)
            .field("route_id", &"[redacted]")
            .field("host_id", &self.host_id)
            .field("host_noise_public", &"[redacted]")
            .field("pairing_secret", &"[redacted]")
            .field("expires_at_unix_ms", &self.expires_at_unix_ms)
            .finish()
    }
}

impl PairingQr {
    pub fn decode(input: &[u8]) -> Result<Self, ProtocolError> {
        if input.len() > MAX_RELAY_URL_BYTES + 512 {
            return Err(ProtocolError::MessageTooLarge {
                actual: input.len(),
                limit: MAX_RELAY_URL_BYTES + 512,
            });
        }
        let qr: Self = serde_json::from_slice(input).map_err(bounded_serde_error)?;
        qr.validate()?;
        Ok(qr)
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.v != 1 {
            return Err(invalid("unsupported QR version"));
        }
        validate_relay_url(&self.relay_url)?;
        validate_base64url("route_id", &self.route_id, 22)?;
        validate_base64url("host_noise_public", &self.host_noise_public, 43)?;
        validate_base64url("pairing_secret", &self.pairing_secret, 22)?;
        if self.expires_at_unix_ms == 0 {
            return Err(invalid("pairing expiry is missing"));
        }
        Ok(())
    }
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairRequest {
    pub pairing_secret: String,
    pub device_label: String,
    pub app_protocol_version: u16,
    pub supported_capabilities: Vec<String>,
}

impl fmt::Debug for PairRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairRequest")
            .field("pairing_secret", &"[redacted]")
            .field("device_label", &self.device_label)
            .field("app_protocol_version", &self.app_protocol_version)
            .field("supported_capabilities", &self.supported_capabilities)
            .finish()
    }
}

impl PairRequest {
    pub fn decode(input: &[u8]) -> Result<Self, ProtocolError> {
        if input.len() > 4096 {
            return Err(ProtocolError::MessageTooLarge {
                actual: input.len(),
                limit: 4096,
            });
        }
        let request: Self = serde_json::from_slice(input).map_err(bounded_serde_error)?;
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        validate_base64url("pairing_secret", &self.pairing_secret, 22)?;
        validate_text("device_label", &self.device_label, MAX_DEVICE_LABEL_BYTES)?;
        if self.device_label.trim().is_empty() {
            return Err(invalid("device_label is empty"));
        }
        if self.app_protocol_version != 1 {
            return Err(invalid("unsupported app protocol version"));
        }
        validate_capabilities(&self.supported_capabilities)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairResponse {
    pub v: u8,
    pub host_id: Uuid,
    pub device_id: Uuid,
    pub session_id: Uuid,
    pub supported_capabilities: Vec<String>,
}

impl PairResponse {
    pub fn decode(input: &[u8]) -> Result<Self, ProtocolError> {
        if input.len() > 4096 {
            return Err(ProtocolError::MessageTooLarge {
                actual: input.len(),
                limit: 4096,
            });
        }
        let response: Self = serde_json::from_slice(input).map_err(bounded_serde_error)?;
        response.validate()?;
        Ok(response)
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.v != 1 {
            return Err(invalid("unsupported pairing response version"));
        }
        validate_capabilities(&self.supported_capabilities)
    }
}

fn validate_capabilities(capabilities: &[String]) -> Result<(), ProtocolError> {
    if capabilities.len() > MAX_PAIRING_CAPABILITIES {
        return Err(invalid("too many pairing capabilities"));
    }
    for (index, capability) in capabilities.iter().enumerate() {
        validate_token("supported_capability", capability, MAX_EVENT_TOKEN_BYTES)?;
        if capabilities[..index].contains(capability) {
            return Err(invalid("duplicate pairing capability"));
        }
    }
    Ok(())
}

fn validate_relay_url(raw: &str) -> Result<(), ProtocolError> {
    validate_text("relay_url", raw, MAX_RELAY_URL_BYTES)?;
    let parsed = Url::parse(raw).map_err(|_| invalid("relay_url is invalid"))?;
    if parsed.scheme() != "wss" {
        return Err(invalid("relay_url must use wss"));
    }
    if parsed.host_str().is_none() || !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(invalid("relay_url requires a host and no credentials"));
    }
    if parsed.fragment().is_some() {
        return Err(invalid("relay_url must not contain a fragment"));
    }
    Ok(())
}

fn validate_base64url(field: &str, value: &str, exact_len: usize) -> Result<(), ProtocolError> {
    if value.len() != exact_len
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(invalid(format!("{field} is not canonical base64url")));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalRisk {
    Write,
    Exec,
    Network,
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ApprovalRecord {
    pub schema: String,
    pub tool: String,
    pub risk: ApprovalRisk,
    pub action: ApprovalAction,
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum ApprovalAction {
    WriteFile {
        target: ResolvedTarget,
        content: String,
    },
    EditFile {
        target: ResolvedTarget,
        old: String,
        new: String,
    },
    RunShell {
        command: String,
        workdir: ResolvedTarget,
        timeout_ms: u64,
        enforcement: ShellEnforcement,
    },
    HttpFetch {
        method: String,
        url: String,
    },
    WebSearch {
        query: String,
    },
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResolvedTarget {
    pub canonical_native: String,
    pub workspace_display: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ShellMode {
    Sandboxed,
    Unrestricted,
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ShellEnforcement {
    pub platform: String,
    pub mode: ShellMode,
    pub enforced_layers: Vec<String>,
    pub note: Option<String>,
}

impl ApprovalAction {
    fn contract(&self) -> (&'static str, ApprovalRisk) {
        match self {
            Self::WriteFile { .. } => ("write_file", ApprovalRisk::Write),
            Self::EditFile { .. } => ("edit_file", ApprovalRisk::Write),
            Self::RunShell { .. } => ("run_shell", ApprovalRisk::Exec),
            Self::HttpFetch { .. } => ("http_fetch", ApprovalRisk::Network),
            Self::WebSearch { .. } => ("web_search", ApprovalRisk::Network),
        }
    }

    fn validate(&self) -> Result<(), ProtocolError> {
        match self {
            Self::WriteFile { target, .. } => target.validate(),
            Self::EditFile { target, .. } => target.validate(),
            Self::RunShell {
                command,
                workdir,
                timeout_ms,
                enforcement,
            } => {
                validate_nonempty("command", command)?;
                workdir.validate()?;
                if *timeout_ms == 0 {
                    return Err(invalid("shell timeout must be non-zero"));
                }
                enforcement.validate()
            }
            Self::HttpFetch { method, url } => {
                if method.is_empty()
                    || method.len() > MAX_METHOD_BYTES
                    || !method.bytes().all(|byte| byte.is_ascii_uppercase())
                {
                    return Err(invalid("HTTP method is not canonical uppercase ASCII"));
                }
                let parsed = Url::parse(url).map_err(|_| invalid("fetch URL is invalid"))?;
                if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
                    return Err(invalid(
                        "fetch URL must use http or https and include a host",
                    ));
                }
                if !parsed.username().is_empty() || parsed.password().is_some() {
                    return Err(invalid("fetch URL must not contain credentials"));
                }
                Ok(())
            }
            Self::WebSearch { query } => validate_nonempty("query", query),
        }
    }
}

impl ResolvedTarget {
    fn validate(&self) -> Result<(), ProtocolError> {
        validate_text("canonical_native", &self.canonical_native, MAX_PATH_BYTES)?;
        validate_text("workspace_display", &self.workspace_display, MAX_PATH_BYTES)?;
        if self.canonical_native.contains('\0') || self.workspace_display.contains('\0') {
            return Err(invalid("resolved target contains NUL"));
        }
        Ok(())
    }
}

impl ShellEnforcement {
    fn validate(&self) -> Result<(), ProtocolError> {
        validate_token("platform", &self.platform, MAX_EVENT_TOKEN_BYTES)?;
        if self.enforced_layers.len() > MAX_SHELL_LAYERS {
            return Err(invalid("too many shell enforcement layers"));
        }
        for layer in &self.enforced_layers {
            validate_token("enforced layer", layer, MAX_SHELL_LAYER_BYTES)?;
        }
        if let Some(note) = &self.note {
            validate_text("shell note", note, MAX_SHELL_NOTE_BYTES)?;
        }
        Ok(())
    }
}

impl ApprovalRecord {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.schema != APPROVAL_RECORD_SCHEMA {
            return Err(invalid("unsupported approval record schema"));
        }
        self.action.validate()?;
        let (tool, risk) = self.action.contract();
        if self.tool != tool || self.risk != risk {
            return Err(invalid("approval tool or risk does not match the action"));
        }
        let canonical = canonical_json(self)?;
        if canonical.len() > MAX_APPROVAL_RECORD_BYTES {
            return Err(ProtocolError::MessageTooLarge {
                actual: canonical.len(),
                limit: MAX_APPROVAL_RECORD_BYTES,
            });
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, ProtocolError> {
        self.validate()?;
        let canonical = canonical_json(self)?;
        let mut hasher = Sha256::new();
        hasher.update(canonical);
        Ok(format!("sha256:{:x}", hasher.finalize()))
    }
}

/// Canonical JSON used only for protocol identities and digests. Object keys
/// use RFC 8785's UTF-16 sort order; floating-point numbers are rejected so no
/// implementation-specific number rendering can change an authority digest.
pub fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, ProtocolError> {
    let value = serde_json::to_value(value).map_err(bounded_serde_error)?;
    let mut output = Vec::new();
    write_canonical(&value, &mut output)?;
    Ok(output)
}

fn write_canonical(value: &Value, output: &mut Vec<u8>) -> Result<(), ProtocolError> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(true) => output.extend_from_slice(b"true"),
        Value::Bool(false) => output.extend_from_slice(b"false"),
        Value::Number(number) if number.is_i64() || number.is_u64() => {
            output.extend_from_slice(number.to_string().as_bytes());
        }
        Value::Number(_) => return Err(ProtocolError::NonCanonicalNumber),
        Value::String(text) => {
            serde_json::to_writer(output, text).map_err(bounded_serde_error)?;
        }
        Value::Array(items) => {
            output.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                write_canonical(item, output)?;
            }
            output.push(b']');
        }
        Value::Object(object) => {
            let mut entries: Vec<_> = object.iter().collect();
            entries.sort_by_cached_key(|(key, _)| key.encode_utf16().collect::<Vec<_>>());
            output.push(b'{');
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                serde_json::to_writer(&mut *output, key).map_err(bounded_serde_error)?;
                output.push(b':');
                write_canonical(item, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), ProtocolError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(invalid("action_digest must be tagged sha256"));
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(
            "action_digest must contain 64 lowercase hex digits",
        ));
    }
    Ok(())
}

fn validate_nonempty(field: &str, value: &str) -> Result<(), ProtocolError> {
    if value.trim().is_empty() {
        return Err(invalid(format!("{field} is empty")));
    }
    Ok(())
}

fn validate_text(field: &str, value: &str, max_bytes: usize) -> Result<(), ProtocolError> {
    if value.len() > max_bytes {
        return Err(ProtocolError::MessageTooLarge {
            actual: value.len(),
            limit: max_bytes,
        });
    }
    if value.chars().any(|character| character == '\0') {
        return Err(invalid(format!("{field} contains NUL")));
    }
    Ok(())
}

fn validate_token(field: &str, value: &str, max_bytes: usize) -> Result<(), ProtocolError> {
    if value.is_empty()
        || value.len() > max_bytes
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'.' | b'-')
        })
    {
        return Err(invalid(format!("{field} is not a bounded protocol token")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn session_catalog_is_host_scoped_bounded_and_stably_ordered() {
        let active = Uuid::from_u128(1);
        let newer = SessionSummary {
            history_id: Uuid::from_u128(2),
            source: SessionHistorySource::Remote,
            title: "Fix parser".into(),
            state: RemoteSessionState::Idle,
            canonical_root: "/work".into(),
            model_id: "qwen3_4b_q4_k_m".into(),
            model_sha256: Some(format!("sha256:{}", "a".repeat(64))),
            created_at_unix_ms: 10,
            updated_at_unix_ms: 20,
            last_event_sequence: 8,
            active: true,
            continuable: true,
            refusal_code: None,
        };
        let older = SessionSummary {
            history_id: Uuid::from_u128(3),
            updated_at_unix_ms: 19,
            active: false,
            ..newer.clone()
        };
        let catalog = SessionCatalog {
            active_session_id: active,
            revision: format!("sha256:{}", "b".repeat(64)),
            sessions: vec![newer, older],
            next_cursor: None,
        };
        assert!(catalog.validate().is_ok());

        let mut reversed = catalog.clone();
        reversed.sessions.reverse();
        assert!(reversed.validate().is_err());

        let message = RemoteMessage {
            protocol: PROTOCOL.into(),
            message_id: Uuid::from_u128(4),
            kind: MessageKind::SessionCatalog,
            host_id: Uuid::from_u128(5),
            device_id: Uuid::from_u128(6),
            session_id: None,
            sent_at_unix_ms: 30,
            payload: serde_json::to_value(catalog).unwrap(),
        };
        assert_eq!(decode_session_catalog(&message).unwrap().sessions.len(), 2);
        let mut session_scoped = message;
        session_scoped.session_id = Some(active);
        assert!(decode_session_catalog(&session_scoped).is_err());
    }

    #[test]
    fn session_catalog_request_requires_consistent_cursor_revision() {
        let request = SessionCatalogRequest {
            cursor: None,
            limit: MAX_SESSION_CATALOG_ENTRIES,
            revision: None,
        };
        assert!(request.validate().is_ok());
        assert!(SessionCatalogRequest {
            limit: 0,
            ..request.clone()
        }
        .validate()
        .is_err());
        assert!(SessionCatalogRequest {
            cursor: Some(SessionCatalogCursor {
                updated_at_unix_ms: 10,
                history_id: Uuid::from_u128(7),
            }),
            ..request.clone()
        }
        .validate()
        .is_err());
        assert!(SessionCatalogRequest {
            revision: Some(format!("sha256:{}", "c".repeat(64))),
            ..request
        }
        .validate()
        .is_err());
    }

    const VALID_COMMAND: &[u8] =
        include_bytes!("../../../tests/fixtures/remote/v1/valid/start_turn_message.json");
    const VALID_QR: &[u8] =
        include_bytes!("../../../tests/fixtures/remote/v1/valid/pairing_qr.json");
    const VALID_APPROVAL: &str =
        include_str!("../../../tests/fixtures/remote/v1/valid/edit_file_approval_record.json");
    const INVALID_UNKNOWN_COMMAND_FIELD: &[u8] =
        include_bytes!("../../../tests/fixtures/remote/v1/invalid/start_turn_unknown_field.json");
    const INVALID_QR_SCHEME: &[u8] =
        include_bytes!("../../../tests/fixtures/remote/v1/invalid/pairing_qr_http.json");
    const INVALID_APPROVAL_AUTHORITY: &str =
        include_str!("../../../tests/fixtures/remote/v1/invalid/approval_mismatched_tool.json");
    const SCHEMAS: &[(&str, &str)] = &[
        (
            "inner-message",
            include_str!("../../../tests/fixtures/remote/v1/schema/inner-message.schema.json"),
        ),
        (
            "command",
            include_str!("../../../tests/fixtures/remote/v1/schema/command.schema.json"),
        ),
        (
            "event",
            include_str!("../../../tests/fixtures/remote/v1/schema/event.schema.json"),
        ),
        (
            "pairing-qr",
            include_str!("../../../tests/fixtures/remote/v1/schema/pairing-qr.schema.json"),
        ),
        (
            "approval-record",
            include_str!("../../../tests/fixtures/remote/v1/schema/approval-record.schema.json"),
        ),
        (
            "relay-envelope",
            include_str!("../../../tests/fixtures/remote/v1/schema/relay-envelope.schema.json"),
        ),
    ];

    #[test]
    fn valid_start_turn_fixture_round_trips() {
        let message = decode_message(VALID_COMMAND).expect("valid message fixture");
        let command = decode_command(&message).expect("valid command fixture");
        assert!(matches!(command, Command::StartTurn { .. }));
        let encoded = canonical_json(&message).expect("canonical message");
        assert!(decode_message(&encoded).expect("round trip") == message);
    }

    #[test]
    fn commands_reject_unknown_fields_and_oversized_text() {
        let message = decode_message(VALID_COMMAND).expect("valid fixture");
        let mut unknown = message.clone();
        unknown.payload["future_authority"] = json!(true);
        assert!(decode_command(&unknown).is_err());

        let mut oversized = message;
        oversized.payload["text"] = json!("x".repeat(MAX_TURN_TEXT_BYTES + 1));
        assert!(matches!(
            decode_command(&oversized),
            Err(ProtocolError::MessageTooLarge { .. })
        ));
    }

    #[test]
    fn committed_invalid_fixtures_fail_closed() {
        let message = decode_message(INVALID_UNKNOWN_COMMAND_FIELD).expect("valid envelope");
        assert!(decode_command(&message).is_err());
        assert!(PairingQr::decode(INVALID_QR_SCHEME).is_err());

        let approval: ApprovalRecord =
            serde_json::from_str(INVALID_APPROVAL_AUTHORITY).expect("valid fixture JSON");
        assert!(approval.validate().is_err());
    }

    #[test]
    fn envelope_accepts_additive_fields_but_rejects_unknown_kinds() {
        let mut value: Value = serde_json::from_slice(VALID_COMMAND).expect("fixture JSON");
        value["future_metadata"] = json!({"ignored": true});
        assert!(decode_message(&serde_json::to_vec(&value).unwrap()).is_ok());
        value["kind"] = json!("execute_without_approval");
        assert!(decode_message(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn oversized_message_is_rejected_before_json_parsing() {
        let input = vec![b' '; MAX_INNER_MESSAGE_BYTES + 1];
        assert!(matches!(
            decode_message(&input),
            Err(ProtocolError::MessageTooLarge { actual, limit })
                if actual == input.len() && limit == MAX_INNER_MESSAGE_BYTES
        ));
    }

    #[test]
    fn maximum_inner_message_round_trips_through_ordered_chunks() {
        let message = vec![b'x'; MAX_INNER_MESSAGE_BYTES];
        let chunks = encode_chunks(Uuid::nil(), &message).expect("bounded message");
        assert_eq!(chunks.len(), MAX_MESSAGE_CHUNKS);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.len() <= MAX_TRANSPORT_PLAINTEXT_BYTES));

        let mut reassembler = ChunkReassembler::default();
        let mut completed = None;
        for chunk in chunks {
            completed = reassembler.push(&chunk).expect("valid chunk");
        }
        assert_eq!(completed.expect("complete message"), message);
    }

    #[test]
    fn chunks_fail_closed_on_oversize_reorder_duplicate_and_tamper() {
        assert!(encode_chunks(Uuid::nil(), &[]).is_err());
        assert!(encode_chunks(Uuid::nil(), &vec![0; MAX_INNER_MESSAGE_BYTES + 1]).is_err());

        let chunks = encode_chunks(Uuid::nil(), &vec![b'x'; MAX_CHUNK_DATA_BYTES + 1]).unwrap();
        let mut reordered = ChunkReassembler::default();
        assert!(reordered.push(&chunks[1]).is_err());

        let mut duplicated = ChunkReassembler::default();
        assert!(duplicated.push(&chunks[0]).unwrap().is_none());
        assert!(duplicated.push(&chunks[0]).is_err());

        let mut tampered_chunks = chunks;
        let last = tampered_chunks.last_mut().unwrap();
        *last.last_mut().unwrap() ^= 1;
        let mut tampered = ChunkReassembler::default();
        assert!(tampered.push(&tampered_chunks[0]).unwrap().is_none());
        assert!(tampered.push(&tampered_chunks[1]).is_err());
    }

    #[test]
    fn pairing_qr_is_strict_bounded_and_redacted() {
        let qr = PairingQr::decode(VALID_QR).expect("valid QR fixture");
        let debug = format!("{qr:?}");
        assert!(!debug.contains(&qr.route_id));
        assert!(!debug.contains(&qr.host_noise_public));
        assert!(!debug.contains(&qr.pairing_secret));

        let mut invalid: Value = serde_json::from_slice(VALID_QR).unwrap();
        invalid["relay_url"] = json!("https://relay.example.test/v1/connect");
        assert!(PairingQr::decode(&serde_json::to_vec(&invalid).unwrap()).is_err());
        invalid["relay_url"] = json!("wss://relay.example.test/v1/connect");
        invalid["pairing_secret"] = json!("short");
        assert!(PairingQr::decode(&serde_json::to_vec(&invalid).unwrap()).is_err());
    }

    #[test]
    fn pairing_request_is_strict_bounded_and_redacted() {
        let request = PairRequest::decode(
            br#"{"pairing_secret":"AAAAAAAAAAAAAAAAAAAAAA","device_label":"Karan's phone","app_protocol_version":1,"supported_capabilities":["agent_events"]}"#,
        )
        .unwrap();
        assert!(!format!("{request:?}").contains(&request.pairing_secret));

        let mut duplicate = request.clone();
        duplicate.supported_capabilities.push("agent_events".into());
        assert!(duplicate.validate().is_err());
        let unknown = br#"{"pairing_secret":"AAAAAAAAAAAAAAAAAAAAAA","device_label":"Phone","app_protocol_version":1,"supported_capabilities":[],"admin":true}"#;
        assert!(PairRequest::decode(unknown).is_err());
    }

    #[test]
    fn event_batches_replay_end_and_command_results_are_strict() {
        let session_id = Uuid::from_u128(1);
        let base = |kind, payload| RemoteMessage {
            protocol: PROTOCOL.into(),
            message_id: Uuid::from_u128(2),
            kind,
            host_id: Uuid::from_u128(3),
            device_id: Uuid::from_u128(4),
            session_id: Some(session_id),
            sent_at_unix_ms: 1,
            payload,
        };
        let first = RemoteEvent {
            sequence: 4,
            event_id: Uuid::from_u128(5),
            turn_id: None,
            event: "session.notice".into(),
            created_at_unix_ms: 1,
            payload: json!({"content":"ready"}),
        };
        let second = RemoteEvent {
            sequence: 5,
            ..first.clone()
        };
        let batch = base(
            MessageKind::EventBatch,
            serde_json::to_value(EventBatch {
                events: vec![first.clone(), second],
            })
            .unwrap(),
        );
        assert_eq!(decode_event_batch(&batch).unwrap().events.len(), 2);
        let gap = base(
            MessageKind::EventBatch,
            serde_json::to_value(EventBatch {
                events: vec![
                    first.clone(),
                    RemoteEvent {
                        sequence: 6,
                        ..first
                    },
                ],
            })
            .unwrap(),
        );
        assert!(decode_event_batch(&gap).is_err());

        let end = base(
            MessageKind::ReplayEnd,
            json!({"last_sequence":5,"has_more":false,"session_state":"idle"}),
        );
        assert_eq!(
            decode_replay_end(&end).unwrap().session_state,
            RemoteSessionState::Idle
        );
        let result = base(
            MessageKind::CommandResult,
            json!({
                "command_id": Uuid::from_u128(6),
                "status": "applied",
                "code": "ok",
                "message": "done",
                "current_event_sequence": 5
            }),
        );
        assert_eq!(decode_command_result(&result).unwrap().code, "ok");
    }

    #[test]
    fn pairing_response_is_strict_and_bounded() {
        let response = PairResponse {
            v: 1,
            host_id: Uuid::from_u128(7),
            device_id: Uuid::from_u128(8),
            session_id: Uuid::from_u128(9),
            supported_capabilities: vec!["agent_events".into()],
        };
        let encoded = serde_json::to_vec(&response).unwrap();
        assert_eq!(PairResponse::decode(&encoded).unwrap(), response);
        let mut duplicate = response.clone();
        duplicate.supported_capabilities.push("agent_events".into());
        assert!(duplicate.validate().is_err());
        let unknown = json!({
            "v": 1,
            "host_id": Uuid::from_u128(10),
            "device_id": Uuid::from_u128(11),
            "session_id": Uuid::from_u128(12),
            "supported_capabilities": [],
            "persistent_approval": true
        });
        assert!(PairResponse::decode(&serde_json::to_vec(&unknown).unwrap()).is_err());
    }

    #[test]
    fn approval_fixture_digest_is_stable_and_every_field_is_bound() {
        let record: ApprovalRecord = serde_json::from_str(VALID_APPROVAL).expect("fixture JSON");
        let digest = record.digest().expect("valid approval record");
        assert_eq!(
            digest,
            "sha256:995876c30076bb24a3273e215dcd0839eceef86fcb97bce4c9a1a8038fab6fdd"
        );

        let mut changed = record.clone();
        if let ApprovalAction::EditFile { new, .. } = &mut changed.action {
            new.push('!');
        }
        assert_ne!(changed.digest().unwrap(), digest);
    }

    #[test]
    fn approval_record_rejects_mismatched_authority_and_floats() {
        let mut record: ApprovalRecord = serde_json::from_str(VALID_APPROVAL).unwrap();
        record.tool = "read_file".into();
        assert!(record.validate().is_err());
        assert_eq!(
            canonical_json(&json!({"not_authority": 1.5})),
            Err(ProtocolError::NonCanonicalNumber)
        );
    }

    #[test]
    fn canonical_json_sorts_utf16_keys_and_ignores_input_order() {
        let first = json!({"z": 1, "a": {"two": 2, "one": 1}});
        let second = json!({"a": {"one": 1, "two": 2}, "z": 1});
        assert_eq!(
            canonical_json(&first).unwrap(),
            canonical_json(&second).unwrap()
        );
        assert_eq!(
            String::from_utf8(canonical_json(&first).unwrap()).unwrap(),
            r#"{"a":{"one":1,"two":2},"z":1}"#
        );
    }

    #[test]
    fn unknown_event_tokens_remain_non_privileged_and_replayable() {
        let event = RemoteEvent {
            sequence: 9,
            event_id: Uuid::nil(),
            turn_id: None,
            event: "future.observation".into(),
            created_at_unix_ms: 1,
            payload: json!({"new": true}),
        };
        assert!(event.validate().is_ok());
    }

    #[test]
    fn replay_and_command_result_bounds_are_strict() {
        assert!(ReplayRequest {
            after_sequence: 41,
            limit: MAX_REPLAY_EVENTS,
        }
        .validate()
        .is_ok());
        assert!(ReplayRequest {
            after_sequence: 41,
            limit: 0,
        }
        .validate()
        .is_err());

        let mut result = CommandResult {
            command_id: Uuid::nil(),
            status: CommandStatus::Rejected,
            code: "stale_approval".into(),
            message: "The approval is no longer pending".into(),
            current_event_sequence: 42,
        };
        assert!(result.validate().is_ok());
        result.code = "Allow Anyway".into();
        assert!(result.validate().is_err());
        result.code = "invalid_message".into();
        result.message = "x".repeat(MAX_RESULT_MESSAGE_BYTES + 1);
        assert!(result.validate().is_err());
    }

    #[test]
    fn schema_corpus_is_parseable_versioned_and_named() {
        for (name, source) in SCHEMAS {
            let schema: Value = serde_json::from_str(source).expect("schema JSON");
            assert_eq!(
                schema["$schema"],
                "https://json-schema.org/draft/2020-12/schema"
            );
            let id = schema["$id"].as_str().expect("schema $id");
            assert!(id.ends_with(&format!("/{name}.schema.json")));
            assert_eq!(schema["type"], "object");
        }

        let manifest: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/remote/v1/manifest.json"
        ))
        .expect("manifest JSON");
        assert_eq!(manifest["protocol"], PROTOCOL);
        assert_eq!(
            manifest["approval_fixture_digest"],
            "sha256:995876c30076bb24a3273e215dcd0839eceef86fcb97bce4c9a1a8038fab6fdd"
        );
        assert_eq!(manifest["schema_count"], SCHEMAS.len());
    }
}
