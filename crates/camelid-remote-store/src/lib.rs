//! Authoritative durable state for one Camelid remote agent host.
//!
//! Every event is committed with its sequence before callers may broadcast it.
//! Commands are idempotent by `(device_id, command_id, request_digest)`, and
//! approval settlement is a single conditional update from `pending`.

use std::path::Path;
use std::time::Duration;

use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub const SCHEMA_VERSION: i64 = 3;
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("remote persistence is unavailable")]
    Unavailable,
    #[error("remote persistence schema is newer than this Camelid build")]
    NewerSchema,
    #[error("remote persistence state conflict")]
    Conflict,
    #[error("remote persistence record is invalid")]
    Invalid,
}

impl From<rusqlite::Error> for StoreError {
    fn from(_: rusqlite::Error) -> Self {
        Self::Unavailable
    }
}

impl From<serde_json::Error> for StoreError {
    fn from(_: serde_json::Error) -> Self {
        Self::Invalid
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Armed,
    Idle,
    Running,
    WaitingApproval,
    Cancelling,
    Failed,
    Closed,
}

impl SessionState {
    fn token(self) -> &'static str {
        match self {
            Self::Armed => "armed",
            Self::Idle => "idle",
            Self::Running => "running",
            Self::WaitingApproval => "waiting_approval",
            Self::Cancelling => "cancelling",
            Self::Failed => "failed",
            Self::Closed => "closed",
        }
    }

    fn may_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Armed, Self::Idle)
                | (Self::Armed, Self::Closed)
                | (Self::Idle, Self::Running)
                | (Self::Idle, Self::Closed)
                | (Self::Running, Self::WaitingApproval)
                | (Self::Running, Self::Idle)
                | (Self::Running, Self::Cancelling)
                | (Self::Running, Self::Failed)
                | (Self::WaitingApproval, Self::Running)
                | (Self::WaitingApproval, Self::Cancelling)
                | (Self::WaitingApproval, Self::Idle)
                | (Self::WaitingApproval, Self::Failed)
                | (Self::Cancelling, Self::Idle)
                | (Self::Cancelling, Self::Failed)
                | (Self::Failed, Self::Closed)
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredEvent {
    pub sequence: u64,
    pub event_id: Uuid,
    pub turn_id: Option<Uuid>,
    pub event_type: String,
    pub payload: Value,
    pub created_at_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResult {
    pub status: String,
    pub response_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredHostIdentity {
    pub host_id: Uuid,
    pub noise_public: [u8; 32],
    pub secret_reference: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredRelayBinding {
    pub relay_url: String,
    pub route_id: String,
    pub capability_secret_reference: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredDevice {
    pub device_id: Uuid,
    pub label: String,
    pub created_at_unix_ms: u64,
    pub last_seen_at_unix_ms: Option<u64>,
    pub revoked_at_unix_ms: Option<u64>,
}

pub enum BeginCommand {
    New,
    Duplicate(CommandResult),
}

pub struct PendingApproval<'a> {
    pub approval_id: Uuid,
    pub session_id: Uuid,
    pub turn_id: Uuid,
    pub call_id: Uuid,
    pub action_digest: &'a str,
    pub tool: &'a str,
    pub risk: &'a str,
    pub detail_json: &'a str,
    pub created_at_unix_ms: u64,
}

pub struct SettleApproval<'a> {
    pub approval_id: Uuid,
    pub session_id: Uuid,
    pub turn_id: Uuid,
    pub call_id: Uuid,
    pub action_digest: &'a str,
    pub decision: &'a str,
    pub device_id: Option<Uuid>,
    pub settled_at_unix_ms: u64,
}

pub struct AcceptStartTurn<'a> {
    pub device_id: Uuid,
    pub command_id: Uuid,
    pub request_digest: &'a str,
    pub session_id: Uuid,
    pub turn_id: Uuid,
    pub user_text: &'a str,
    pub created_at_unix_ms: u64,
}

pub enum AcceptTurn {
    Accepted { events: [StoredEvent; 2] },
    Duplicate(CommandResult),
}

pub struct CompleteTurn<'a> {
    pub session_id: Uuid,
    pub turn_id: Uuid,
    pub outcome: &'a str,
    pub assistant_text: Option<&'a str>,
    pub transcript_json: &'a str,
    pub plan_json: &'a str,
    pub finished_at_unix_ms: u64,
}

pub struct AcceptCancelTurn<'a> {
    pub device_id: Uuid,
    pub command_id: Uuid,
    pub request_digest: &'a str,
    pub session_id: Uuid,
    pub turn_id: Uuid,
    pub created_at_unix_ms: u64,
}

pub struct AcceptApprovalDecision<'a> {
    pub device_id: Uuid,
    pub command_id: Uuid,
    pub request_digest: &'a str,
    pub session_id: Uuid,
    pub turn_id: Uuid,
    pub call_id: Uuid,
    pub approval_id: Uuid,
    pub action_digest: &'a str,
    pub decision: &'a str,
    pub created_at_unix_ms: u64,
}

pub enum AcceptDecision {
    Applied,
    Duplicate(CommandResult),
}

pub struct AcceptCreateSession<'a> {
    pub device_id: Uuid,
    pub command_id: Uuid,
    pub request_digest: &'a str,
    pub session_id: Uuid,
    pub canonical_root: &'a str,
    pub model_id: &'a str,
    pub model_sha256: &'a str,
    pub capability_snapshot_json: &'a str,
    pub created_at_unix_ms: u64,
}

pub struct AcceptActivateSession<'a> {
    pub device_id: Uuid,
    pub command_id: Uuid,
    pub request_digest: &'a str,
    pub session_id: Uuid,
    pub canonical_root: &'a str,
    pub model_id: &'a str,
    pub model_sha256: &'a str,
    pub capability_snapshot_json: &'a str,
    pub activated_at_unix_ms: u64,
}

pub enum AcceptSessionSwitch {
    Applied(ActiveSession),
    Duplicate(CommandResult),
}

pub struct ExpireApproval<'a> {
    pub session_id: Uuid,
    pub turn_id: Uuid,
    pub call_id: Uuid,
    pub approval_id: Uuid,
    pub action_digest: &'a str,
    pub expired_at_unix_ms: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct SessionContext {
    pub transcript_json: String,
    pub plan_json: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionHead {
    pub state: SessionState,
    pub last_event_sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveSession {
    pub session_id: Uuid,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredSessionSummary {
    pub session_id: Uuid,
    pub state: SessionState,
    pub last_event_sequence: u64,
    pub updated_at_unix_ms: u64,
    pub capability_snapshot_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredSessionCatalogEntry {
    pub session_id: Uuid,
    pub title: String,
    pub state: SessionState,
    pub canonical_root: String,
    pub model_id: String,
    pub model_sha256: String,
    pub capability_snapshot_json: String,
    pub created_at_unix_ms: u64,
    pub updated_at_unix_ms: u64,
    pub last_event_sequence: u64,
}

pub struct RemoteStore {
    connection: Connection,
}

impl RemoteStore {
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        let connection = Connection::open(path)?;
        connection.busy_timeout(BUSY_TIMEOUT)?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        migrate(&connection)?;
        Ok(Self { connection })
    }

    #[cfg(feature = "test-hooks")]
    pub fn execute_batch_for_test(&self, sql: &str) -> Result<(), StoreError> {
        self.connection.execute_batch(sql)?;
        Ok(())
    }

    pub fn create_session(
        &mut self,
        session_id: Uuid,
        canonical_root: &str,
        model_id: &str,
        model_sha256: &str,
        capability_snapshot_json: &str,
        created_at_unix_ms: u64,
    ) -> Result<(), StoreError> {
        serde_json::from_str::<Value>(capability_snapshot_json)?;
        self.connection.execute(
            "INSERT INTO remote_sessions (
                session_id, canonical_root, model_id, model_sha256,
                capability_snapshot_json, state, transcript_json, plan_json,
                created_at, updated_at, next_event_sequence
             ) VALUES (?1, ?2, ?3, ?4, ?5, 'armed', '[]', '[]', ?6, ?6, 1)",
            params![
                session_id.to_string(),
                canonical_root,
                model_id,
                model_sha256,
                capability_snapshot_json,
                to_i64(created_at_unix_ms)?,
            ],
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_and_activate_session(
        &mut self,
        session_id: Uuid,
        canonical_root: &str,
        model_id: &str,
        model_sha256: &str,
        capability_snapshot_json: &str,
        created_at_unix_ms: u64,
    ) -> Result<ActiveSession, StoreError> {
        serde_json::from_str::<Value>(capability_snapshot_json)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let active_turns: i64 = transaction.query_row(
            "SELECT count(*) FROM remote_turns WHERE finished_at IS NULL",
            [],
            |row| row.get(0),
        )?;
        if active_turns != 0 {
            return Err(StoreError::Conflict);
        }
        let now = to_i64(created_at_unix_ms)?;
        transaction.execute(
            "INSERT INTO remote_sessions (
                session_id, canonical_root, model_id, model_sha256,
                capability_snapshot_json, state, transcript_json, plan_json,
                created_at, updated_at, next_event_sequence
             ) VALUES (?1, ?2, ?3, ?4, ?5, 'idle', '[]', '[]', ?6, ?6, 1)",
            params![
                session_id.to_string(),
                canonical_root,
                model_id,
                model_sha256,
                capability_snapshot_json,
                now,
            ],
        )?;
        let generation = next_active_generation(&transaction)?;
        transaction.execute(
            "INSERT INTO remote_active_session (singleton, session_id, generation, activated_at)
             VALUES (1, ?1, ?2, ?3)
             ON CONFLICT(singleton) DO UPDATE SET
                session_id = excluded.session_id,
                generation = excluded.generation,
                activated_at = excluded.activated_at",
            params![session_id.to_string(), to_i64(generation)?, now],
        )?;
        transaction.commit()?;
        Ok(ActiveSession {
            session_id,
            generation,
        })
    }

    pub fn active_session(&self) -> Result<Option<ActiveSession>, StoreError> {
        self.connection
            .query_row(
                "SELECT session_id, generation FROM remote_active_session WHERE singleton = 1",
                [],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
            )
            .optional()?
            .map(|(session_id, generation)| {
                Ok(ActiveSession {
                    session_id: Uuid::parse_str(&session_id).map_err(|_| StoreError::Invalid)?,
                    generation: to_u64(generation)?,
                })
            })
            .transpose()
    }

    pub fn activate_session(
        &mut self,
        session_id: Uuid,
        activated_at_unix_ms: u64,
    ) -> Result<ActiveSession, StoreError> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let eligible: i64 = transaction.query_row(
            "SELECT count(*) FROM remote_sessions
             WHERE session_id = ?1 AND state IN ('armed','idle','failed')",
            [session_id.to_string()],
            |row| row.get(0),
        )?;
        let active_turns: i64 = transaction.query_row(
            "SELECT count(*) FROM remote_turns WHERE finished_at IS NULL",
            [],
            |row| row.get(0),
        )?;
        if eligible != 1 || active_turns != 0 {
            return Err(StoreError::Conflict);
        }
        let generation = next_active_generation(&transaction)?;
        transaction.execute(
            "INSERT INTO remote_active_session (singleton, session_id, generation, activated_at)
             VALUES (1, ?1, ?2, ?3)
             ON CONFLICT(singleton) DO UPDATE SET
                session_id = excluded.session_id,
                generation = excluded.generation,
                activated_at = excluded.activated_at",
            params![
                session_id.to_string(),
                to_i64(generation)?,
                to_i64(activated_at_unix_ms)?,
            ],
        )?;
        transaction.commit()?;
        Ok(ActiveSession {
            session_id,
            generation,
        })
    }

    pub fn accept_create_session(
        &mut self,
        request: AcceptCreateSession<'_>,
    ) -> Result<AcceptSessionSwitch, StoreError> {
        let capability_snapshot = serde_json::from_str::<Value>(request.capability_snapshot_json)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        if let Some(result) = existing_session_command(
            &transaction,
            request.device_id,
            request.command_id,
            request.request_digest,
        )? {
            return Ok(AcceptSessionSwitch::Duplicate(result));
        }
        ensure_no_unfinished_turn(&transaction)?;
        let now = to_i64(request.created_at_unix_ms)?;
        transaction.execute(
            "INSERT INTO remote_sessions (
                session_id, canonical_root, model_id, model_sha256,
                capability_snapshot_json, state, transcript_json, plan_json,
                     created_at, updated_at, next_event_sequence
                 ) VALUES (?1, ?2, ?3, ?4, ?5, 'idle', '[]', '[]', ?6, ?6, 3)",
            params![
                request.session_id.to_string(),
                request.canonical_root,
                request.model_id,
                request.model_sha256,
                request.capability_snapshot_json,
                now,
            ],
        )?;
        for (sequence, event_type, payload) in [
            (1_i64, "host.capabilities", capability_snapshot),
            (2_i64, "session.armed", serde_json::json!({"state":"idle"})),
        ] {
            transaction.execute(
                "INSERT INTO remote_events (
                    session_id, sequence, event_id, turn_id, event_type, payload_json, created_at
                 ) VALUES (?1, ?2, ?3, NULL, ?4, ?5, ?6)",
                params![
                    request.session_id.to_string(),
                    sequence,
                    Uuid::new_v4().to_string(),
                    event_type,
                    serde_json::to_string(&payload)?,
                    now,
                ],
            )?;
        }
        let generation = next_active_generation(&transaction)?;
        set_active_session(&transaction, request.session_id, generation, now)?;
        let response = serde_json::json!({
            "code": "session_created",
            "session_id": request.session_id,
            "generation": generation,
            "current_event_sequence": 2,
        });
        insert_session_command(
            &transaction,
            request.device_id,
            request.command_id,
            request.session_id,
            "create_session",
            request.request_digest,
            &response,
            now,
        )?;
        transaction.commit()?;
        Ok(AcceptSessionSwitch::Applied(ActiveSession {
            session_id: request.session_id,
            generation,
        }))
    }

    pub fn accept_activate_session(
        &mut self,
        request: AcceptActivateSession<'_>,
    ) -> Result<AcceptSessionSwitch, StoreError> {
        serde_json::from_str::<Value>(request.capability_snapshot_json)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        if let Some(result) = existing_session_command(
            &transaction,
            request.device_id,
            request.command_id,
            request.request_digest,
        )? {
            return Ok(AcceptSessionSwitch::Duplicate(result));
        }
        ensure_no_unfinished_turn(&transaction)?;
        let eligible: i64 = transaction.query_row(
            "SELECT count(*) FROM remote_sessions
             WHERE session_id = ?1 AND canonical_root = ?2 AND model_id = ?3
               AND model_sha256 = ?4 AND capability_snapshot_json = ?5
               AND state IN ('armed','idle','failed')",
            params![
                request.session_id.to_string(),
                request.canonical_root,
                request.model_id,
                request.model_sha256,
                request.capability_snapshot_json,
            ],
            |row| row.get(0),
        )?;
        if eligible != 1 {
            return Err(StoreError::Conflict);
        }
        let now = to_i64(request.activated_at_unix_ms)?;
        transaction.execute(
            "UPDATE remote_sessions SET state = 'idle', updated_at = ?2
             WHERE session_id = ?1 AND state IN ('armed','failed')",
            params![request.session_id.to_string(), now],
        )?;
        let generation = next_active_generation(&transaction)?;
        set_active_session(&transaction, request.session_id, generation, now)?;
        let head_sequence: i64 = transaction.query_row(
            "SELECT next_event_sequence - 1 FROM remote_sessions WHERE session_id = ?1",
            [request.session_id.to_string()],
            |row| row.get(0),
        )?;
        let response = serde_json::json!({
            "code": "session_activated",
            "session_id": request.session_id,
            "generation": generation,
            "current_event_sequence": to_u64(head_sequence)?,
        });
        insert_session_command(
            &transaction,
            request.device_id,
            request.command_id,
            request.session_id,
            "activate_session",
            request.request_digest,
            &response,
            now,
        )?;
        transaction.commit()?;
        Ok(AcceptSessionSwitch::Applied(ActiveSession {
            session_id: request.session_id,
            generation,
        }))
    }

    pub fn reusable_session(
        &self,
        canonical_root: &str,
        model_id: &str,
        model_sha256: &str,
        capability_snapshot_json: &str,
    ) -> Result<Option<(Uuid, SessionState)>, StoreError> {
        serde_json::from_str::<Value>(capability_snapshot_json)?;
        self.connection
            .query_row(
                "SELECT session_id, state FROM remote_sessions
                 WHERE canonical_root = ?1 AND model_id = ?2 AND model_sha256 = ?3
                   AND capability_snapshot_json = ?4 AND state IN ('armed','idle','failed')
                 ORDER BY updated_at DESC LIMIT 1",
                params![
                    canonical_root,
                    model_id,
                    model_sha256,
                    capability_snapshot_json,
                ],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?
            .map(|(session_id, state)| {
                Ok((
                    Uuid::parse_str(&session_id).map_err(|_| StoreError::Invalid)?,
                    parse_session_state(&state)?,
                ))
            })
            .transpose()
    }

    pub fn rearm_session(
        &mut self,
        session_id: Uuid,
        expected: SessionState,
        updated_at_unix_ms: u64,
    ) -> Result<(), StoreError> {
        if !matches!(expected, SessionState::Armed | SessionState::Failed) {
            return Err(StoreError::Invalid);
        }
        let changed = self.connection.execute(
            "UPDATE remote_sessions SET state = 'idle', updated_at = ?1
             WHERE session_id = ?2 AND state = ?3",
            params![
                to_i64(updated_at_unix_ms)?,
                session_id.to_string(),
                expected.token(),
            ],
        )?;
        if changed != 1 {
            return Err(StoreError::Conflict);
        }
        Ok(())
    }

    pub fn initialize_host_identity(
        &mut self,
        host_id: Uuid,
        noise_public: &[u8; 32],
        secret_reference: &str,
        created_at_unix_ms: u64,
    ) -> Result<(), StoreError> {
        if secret_reference.trim().is_empty() || secret_reference.len() > 1024 {
            return Err(StoreError::Invalid);
        }
        self.connection.execute(
            "INSERT INTO remote_meta (
                singleton, host_id, host_noise_public, host_secret_reference, created_at
             ) VALUES (1, ?1, ?2, ?3, ?4)",
            params![
                host_id.to_string(),
                noise_public,
                secret_reference,
                to_i64(created_at_unix_ms)?,
            ],
        )?;
        Ok(())
    }

    pub fn host_identity(&self) -> Result<StoredHostIdentity, StoreError> {
        self.optional_host_identity()?
            .ok_or(StoreError::Unavailable)
    }

    pub fn optional_host_identity(&self) -> Result<Option<StoredHostIdentity>, StoreError> {
        let (host_id, public, secret_reference): (String, Vec<u8>, String) = match self
            .connection
            .query_row(
                "SELECT host_id, host_noise_public, host_secret_reference
                 FROM remote_meta WHERE singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?
        {
            Some(identity) => identity,
            None => return Ok(None),
        };
        Ok(Some(StoredHostIdentity {
            host_id: Uuid::parse_str(&host_id).map_err(|_| StoreError::Invalid)?,
            noise_public: public.try_into().map_err(|_| StoreError::Invalid)?,
            secret_reference,
        }))
    }

    pub fn relay_binding(&self) -> Result<Option<StoredRelayBinding>, StoreError> {
        self.connection
            .query_row(
                "SELECT relay_url, route_id, capability_secret_reference
                 FROM remote_relay WHERE singleton = 1",
                [],
                |row| {
                    Ok(StoredRelayBinding {
                        relay_url: row.get(0)?,
                        route_id: row.get(1)?,
                        capability_secret_reference: row.get(2)?,
                    })
                },
            )
            .optional()
            .map_err(StoreError::from)
    }

    pub fn set_relay_binding(
        &mut self,
        relay_url: &str,
        route_id: &str,
        capability_secret_reference: &str,
        updated_at_unix_ms: u64,
    ) -> Result<(), StoreError> {
        if relay_url.is_empty()
            || relay_url.len() > 2048
            || route_id.len() != 22
            || capability_secret_reference.is_empty()
            || capability_secret_reference.len() > 1024
        {
            return Err(StoreError::Invalid);
        }
        self.connection.execute(
            "INSERT INTO remote_relay (
                singleton, relay_url, route_id, capability_secret_reference, updated_at
             ) VALUES (1, ?1, ?2, ?3, ?4)
             ON CONFLICT(singleton) DO UPDATE SET
                relay_url = excluded.relay_url,
                route_id = excluded.route_id,
                capability_secret_reference = excluded.capability_secret_reference,
                updated_at = excluded.updated_at",
            params![
                relay_url,
                route_id,
                capability_secret_reference,
                to_i64(updated_at_unix_ms)?,
            ],
        )?;
        Ok(())
    }

    pub fn authorized_device_count(&self) -> Result<usize, StoreError> {
        let count: i64 = self.connection.query_row(
            "SELECT count(*) FROM remote_devices WHERE revoked_at IS NULL",
            [],
            |row| row.get(0),
        )?;
        count.try_into().map_err(|_| StoreError::Invalid)
    }

    pub fn devices(&self) -> Result<Vec<StoredDevice>, StoreError> {
        let mut statement = self.connection.prepare(
            "SELECT device_id, label, created_at, last_seen_at, revoked_at
             FROM remote_devices ORDER BY created_at ASC, device_id ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, Option<i64>>(3)?,
                row.get::<_, Option<i64>>(4)?,
            ))
        })?;
        rows.map(|row| {
            let (device_id, label, created_at, last_seen_at, revoked_at) = row?;
            Ok(StoredDevice {
                device_id: Uuid::parse_str(&device_id).map_err(|_| StoreError::Invalid)?,
                label,
                created_at_unix_ms: to_u64(created_at)?,
                last_seen_at_unix_ms: last_seen_at.map(to_u64).transpose()?,
                revoked_at_unix_ms: revoked_at.map(to_u64).transpose()?,
            })
        })
        .collect()
    }

    pub fn revoke_all_devices(&mut self, revoked_at_unix_ms: u64) -> Result<Vec<Uuid>, StoreError> {
        let device_ids = self
            .devices()?
            .into_iter()
            .filter(|device| device.revoked_at_unix_ms.is_none())
            .map(|device| device.device_id)
            .collect::<Vec<_>>();
        if device_ids.is_empty() {
            return Ok(device_ids);
        }
        let changed = self.connection.execute(
            "UPDATE remote_devices SET revoked_at = ?1 WHERE revoked_at IS NULL",
            [to_i64(revoked_at_unix_ms)?],
        )?;
        if changed != device_ids.len() {
            return Err(StoreError::Conflict);
        }
        Ok(device_ids)
    }

    pub fn load_session_context(
        &self,
        session_id: Uuid,
        canonical_root: &str,
        model_id: &str,
        model_sha256: &str,
        capability_snapshot_json: &str,
    ) -> Result<SessionContext, StoreError> {
        serde_json::from_str::<Value>(capability_snapshot_json)?;
        self.connection
            .query_row(
                "SELECT transcript_json, plan_json FROM remote_sessions
                 WHERE session_id = ?1 AND canonical_root = ?2 AND model_id = ?3
                   AND model_sha256 = ?4 AND capability_snapshot_json = ?5
                   AND state IN ('armed','idle','failed')",
                params![
                    session_id.to_string(),
                    canonical_root,
                    model_id,
                    model_sha256,
                    capability_snapshot_json,
                ],
                |row| {
                    Ok(SessionContext {
                        transcript_json: row.get(0)?,
                        plan_json: row.get(1)?,
                    })
                },
            )
            .optional()?
            .ok_or(StoreError::Conflict)
    }

    pub fn session_head(&self, session_id: Uuid) -> Result<SessionHead, StoreError> {
        let (state, next_sequence): (String, i64) = self.connection.query_row(
            "SELECT state, next_event_sequence FROM remote_sessions WHERE session_id = ?1",
            [session_id.to_string()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let state = parse_session_state(&state)?;
        Ok(SessionHead {
            state,
            last_event_sequence: to_u64(next_sequence)?.saturating_sub(1),
        })
    }

    pub fn latest_session_summary(&self) -> Result<Option<StoredSessionSummary>, StoreError> {
        self.connection
            .query_row(
                "SELECT session_id, state, next_event_sequence, updated_at, capability_snapshot_json
                 FROM remote_sessions ORDER BY updated_at DESC, session_id DESC LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                },
            )
            .optional()?
            .map(|(session_id, state, next_sequence, updated_at, capability_snapshot_json)| {
                Ok(StoredSessionSummary {
                    session_id: Uuid::parse_str(&session_id).map_err(|_| StoreError::Invalid)?,
                    state: parse_session_state(&state)?,
                    last_event_sequence: to_u64(next_sequence)?.saturating_sub(1),
                    updated_at_unix_ms: to_u64(updated_at)?,
                    capability_snapshot_json,
                })
            })
            .transpose()
    }

    pub fn active_session_summary(&self) -> Result<Option<StoredSessionSummary>, StoreError> {
        self.connection
            .query_row(
                "SELECT session.session_id, session.state, session.next_event_sequence,
                        session.updated_at, session.capability_snapshot_json
                 FROM remote_active_session AS active
                 JOIN remote_sessions AS session ON session.session_id = active.session_id
                 WHERE active.singleton = 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                },
            )
            .optional()?
            .map(
                |(session_id, state, next_sequence, updated_at, capability_snapshot_json)| {
                    Ok(StoredSessionSummary {
                        session_id: Uuid::parse_str(&session_id)
                            .map_err(|_| StoreError::Invalid)?,
                        state: parse_session_state(&state)?,
                        last_event_sequence: to_u64(next_sequence)?.saturating_sub(1),
                        updated_at_unix_ms: to_u64(updated_at)?,
                        capability_snapshot_json,
                    })
                },
            )
            .transpose()
    }

    pub fn list_session_catalog(
        &self,
        canonical_root: &str,
        cursor: Option<(u64, Uuid)>,
        limit: u16,
    ) -> Result<Vec<StoredSessionCatalogEntry>, StoreError> {
        if canonical_root.is_empty() || limit == 0 || limit > 65 {
            return Err(StoreError::Invalid);
        }
        let cursor_time = cursor.map(|value| to_i64(value.0)).transpose()?;
        let cursor_id = cursor.map(|value| value.1.to_string());
        let mut statement = self.connection.prepare(
            "SELECT s.session_id,
                    COALESCE((
                        SELECT json_extract(e.payload_json, '$.content')
                        FROM remote_events e
                        WHERE e.session_id = s.session_id AND e.event_type = 'user.message'
                        ORDER BY e.sequence ASC LIMIT 1
                    ), 'New agent session'),
                    s.state, s.canonical_root, s.model_id, s.model_sha256,
                    s.capability_snapshot_json, s.created_at, s.updated_at, s.next_event_sequence
             FROM remote_sessions s
             WHERE s.canonical_root = ?1
               AND (?2 IS NULL OR s.updated_at < ?2 OR (s.updated_at = ?2 AND s.session_id > ?3))
             ORDER BY s.updated_at DESC, s.session_id ASC
             LIMIT ?4",
        )?;
        let rows = statement.query_map(
            params![canonical_root, cursor_time, cursor_id, i64::from(limit)],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, i64>(9)?,
                ))
            },
        )?;
        rows.map(|row| {
            let (
                session_id,
                title,
                state,
                canonical_root,
                model_id,
                model_sha256,
                capability_snapshot_json,
                created_at,
                updated_at,
                next_sequence,
            ) = row?;
            Ok(StoredSessionCatalogEntry {
                session_id: Uuid::parse_str(&session_id).map_err(|_| StoreError::Invalid)?,
                title,
                state: parse_session_state(&state)?,
                canonical_root,
                model_id,
                model_sha256,
                capability_snapshot_json,
                created_at_unix_ms: to_u64(created_at)?,
                updated_at_unix_ms: to_u64(updated_at)?,
                last_event_sequence: to_u64(next_sequence)?.saturating_sub(1),
            })
        })
        .collect()
    }

    pub fn session_catalog_entry(
        &self,
        canonical_root: &str,
        session_id: Uuid,
    ) -> Result<Option<StoredSessionCatalogEntry>, StoreError> {
        let row = self
            .connection
            .query_row(
                "SELECT s.session_id,
                        COALESCE((
                            SELECT json_extract(e.payload_json, '$.content')
                            FROM remote_events e
                            WHERE e.session_id = s.session_id AND e.event_type = 'user.message'
                            ORDER BY e.sequence ASC LIMIT 1
                        ), 'New agent session'),
                        s.state, s.canonical_root, s.model_id, s.model_sha256,
                        s.capability_snapshot_json, s.created_at, s.updated_at,
                        s.next_event_sequence
                 FROM remote_sessions s
                 WHERE s.canonical_root = ?1 AND s.session_id = ?2",
                params![canonical_root, session_id.to_string()],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                        row.get::<_, i64>(7)?,
                        row.get::<_, i64>(8)?,
                        row.get::<_, i64>(9)?,
                    ))
                },
            )
            .optional()?;
        row.map(
            |(
                session_id,
                title,
                state,
                canonical_root,
                model_id,
                model_sha256,
                capability_snapshot_json,
                created_at,
                updated_at,
                next_sequence,
            )| {
                Ok(StoredSessionCatalogEntry {
                    session_id: Uuid::parse_str(&session_id).map_err(|_| StoreError::Invalid)?,
                    title,
                    state: parse_session_state(&state)?,
                    canonical_root,
                    model_id,
                    model_sha256,
                    capability_snapshot_json,
                    created_at_unix_ms: to_u64(created_at)?,
                    updated_at_unix_ms: to_u64(updated_at)?,
                    last_event_sequence: to_u64(next_sequence)?.saturating_sub(1),
                })
            },
        )
        .transpose()
    }

    pub fn ensure_session_bootstrap_events(
        &mut self,
        session_id: Uuid,
        capability_snapshot: &Value,
        created_at_unix_ms: u64,
    ) -> Result<Vec<StoredEvent>, StoreError> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: i64 = transaction.query_row(
            "SELECT count(*) FROM remote_events
             WHERE session_id = ?1 AND event_type IN ('host.capabilities','session.armed')",
            [session_id.to_string()],
            |row| row.get(0),
        )?;
        if existing != 0 {
            return Ok(Vec::new());
        }
        let first_sequence: i64 = transaction.query_row(
            "UPDATE remote_sessions
             SET next_event_sequence = next_event_sequence + 2, updated_at = ?2
             WHERE session_id = ?1 AND state = 'idle'
             RETURNING next_event_sequence - 2",
            params![session_id.to_string(), to_i64(created_at_unix_ms)?],
            |row| row.get(0),
        )?;
        let payloads = [
            capability_snapshot.clone(),
            serde_json::json!({"state":"idle"}),
        ];
        let event_types = ["host.capabilities", "session.armed"];
        let mut events = Vec::with_capacity(2);
        for offset in 0..2_i64 {
            let event_id = Uuid::new_v4();
            let index = offset as usize;
            let sequence = first_sequence + offset;
            transaction.execute(
                "INSERT INTO remote_events (
                    session_id, sequence, event_id, turn_id, event_type, payload_json, created_at
                 ) VALUES (?1, ?2, ?3, NULL, ?4, ?5, ?6)",
                params![
                    session_id.to_string(),
                    sequence,
                    event_id.to_string(),
                    event_types[index],
                    serde_json::to_string(&payloads[index])?,
                    to_i64(created_at_unix_ms)?,
                ],
            )?;
            events.push(StoredEvent {
                sequence: to_u64(sequence)?,
                event_id,
                turn_id: None,
                event_type: event_types[index].into(),
                payload: payloads[index].clone(),
                created_at_unix_ms,
            });
        }
        transaction.commit()?;
        Ok(events)
    }

    pub fn register_device(
        &mut self,
        device_id: Uuid,
        label: &str,
        noise_static_public: &[u8; 32],
        created_at_unix_ms: u64,
    ) -> Result<(), StoreError> {
        if label.trim().is_empty() || label.len() > 128 {
            return Err(StoreError::Invalid);
        }
        self.connection.execute(
            "INSERT INTO remote_devices (device_id, label, noise_static_public, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                device_id.to_string(),
                label,
                noise_static_public,
                to_i64(created_at_unix_ms)?
            ],
        )?;
        Ok(())
    }

    pub fn device_authorized(
        &self,
        device_id: Uuid,
        noise_static_public: &[u8; 32],
    ) -> Result<bool, StoreError> {
        Ok(self
            .connection
            .query_row(
                "SELECT 1 FROM remote_devices
             WHERE device_id = ?1 AND noise_static_public = ?2 AND revoked_at IS NULL",
                params![device_id.to_string(), noise_static_public],
                |_| Ok(()),
            )
            .optional()?
            .is_some())
    }

    pub fn authorized_device_for_key(
        &self,
        noise_static_public: &[u8; 32],
    ) -> Result<Option<Uuid>, StoreError> {
        self.connection
            .query_row(
                "SELECT device_id FROM remote_devices
                 WHERE noise_static_public = ?1 AND revoked_at IS NULL",
                params![noise_static_public],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(|value| Uuid::parse_str(&value).map_err(|_| StoreError::Invalid))
            .transpose()
    }

    pub fn mark_device_seen(
        &mut self,
        device_id: Uuid,
        seen_at_unix_ms: u64,
    ) -> Result<(), StoreError> {
        let changed = self.connection.execute(
            "UPDATE remote_devices SET last_seen_at = ?1
             WHERE device_id = ?2 AND revoked_at IS NULL",
            params![to_i64(seen_at_unix_ms)?, device_id.to_string()],
        )?;
        if changed != 1 {
            return Err(StoreError::Conflict);
        }
        Ok(())
    }

    pub fn revoke_device(
        &mut self,
        device_id: Uuid,
        revoked_at_unix_ms: u64,
    ) -> Result<(), StoreError> {
        let changed = self.connection.execute(
            "UPDATE remote_devices SET revoked_at = ?1
             WHERE device_id = ?2 AND revoked_at IS NULL",
            params![to_i64(revoked_at_unix_ms)?, device_id.to_string()],
        )?;
        if changed != 1 {
            return Err(StoreError::Conflict);
        }
        Ok(())
    }

    pub fn create_turn(
        &mut self,
        turn_id: Uuid,
        session_id: Uuid,
        command_id: Uuid,
        user_text: &str,
        created_at_unix_ms: u64,
    ) -> Result<(), StoreError> {
        if user_text.trim().is_empty() || user_text.len() > 4096 {
            return Err(StoreError::Invalid);
        }
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let active: i64 = transaction.query_row(
            "SELECT count(*) FROM remote_turns WHERE session_id = ?1 AND finished_at IS NULL",
            [session_id.to_string()],
            |row| row.get(0),
        )?;
        if active != 0 {
            return Err(StoreError::Conflict);
        }
        transaction.execute(
            "INSERT INTO remote_turns (turn_id, session_id, command_id, user_text, state, created_at)
             VALUES (?1, ?2, ?3, ?4, 'accepted', ?5)",
            params![turn_id.to_string(), session_id.to_string(), command_id.to_string(), user_text, to_i64(created_at_unix_ms)?],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn accept_start_turn(
        &mut self,
        request: AcceptStartTurn<'_>,
    ) -> Result<AcceptTurn, StoreError> {
        if request.user_text.trim().is_empty() || request.user_text.len() > 4096 {
            return Err(StoreError::Invalid);
        }
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<(String, String, String)> = transaction
            .query_row(
                "SELECT request_digest, status, response_json FROM remote_commands
                 WHERE device_id = ?1 AND command_id = ?2",
                params![
                    request.device_id.to_string(),
                    request.command_id.to_string()
                ],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        if let Some((digest, status, response_json)) = existing {
            if digest != request.request_digest {
                return Err(StoreError::Conflict);
            }
            return Ok(AcceptTurn::Duplicate(CommandResult {
                status,
                response_json,
            }));
        }
        let now = to_i64(request.created_at_unix_ms)?;
        let first_sequence: i64 = transaction
            .query_row(
                "UPDATE remote_sessions
                 SET state = 'running', next_event_sequence = next_event_sequence + 2, updated_at = ?2
                 WHERE session_id = ?1 AND state = 'idle'
                 RETURNING next_event_sequence - 2",
                params![request.session_id.to_string(), now],
                |row| row.get(0),
            )
            .optional()?
            .ok_or(StoreError::Conflict)?;
        let active: i64 = transaction.query_row(
            "SELECT count(*) FROM remote_turns WHERE session_id = ?1 AND finished_at IS NULL",
            [request.session_id.to_string()],
            |row| row.get(0),
        )?;
        if active != 0 {
            return Err(StoreError::Conflict);
        }
        let response = serde_json::json!({
            "code": "ok",
            "turn_id": request.turn_id,
            "current_event_sequence": first_sequence + 1,
        });
        transaction.execute(
            "INSERT INTO remote_commands (
                device_id, command_id, session_id, command_type, request_digest,
                status, response_json, created_at, finished_at
             ) VALUES (?1, ?2, ?3, 'start_turn', ?4, 'accepted', ?5, ?6, NULL)",
            params![
                request.device_id.to_string(),
                request.command_id.to_string(),
                request.session_id.to_string(),
                request.request_digest,
                serde_json::to_string(&response)?,
                now,
            ],
        )?;
        transaction.execute(
            "INSERT INTO remote_turns (
                turn_id, session_id, command_id, user_text, state, created_at, started_at
             ) VALUES (?1, ?2, ?3, ?4, 'running', ?5, ?5)",
            params![
                request.turn_id.to_string(),
                request.session_id.to_string(),
                request.command_id.to_string(),
                request.user_text,
                now,
            ],
        )?;
        let payloads = [
            serde_json::json!({"content": request.user_text}),
            serde_json::json!({"turn_id": request.turn_id}),
        ];
        let event_types = ["user.message", "turn.accepted"];
        let mut events = Vec::with_capacity(2);
        for offset in 0..2_i64 {
            let event_id = Uuid::new_v4();
            let sequence = first_sequence + offset;
            let index = offset as usize;
            transaction.execute(
                "INSERT INTO remote_events (
                    session_id, sequence, event_id, turn_id, event_type, payload_json, created_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    request.session_id.to_string(),
                    sequence,
                    event_id.to_string(),
                    request.turn_id.to_string(),
                    event_types[index],
                    serde_json::to_string(&payloads[index])?,
                    now,
                ],
            )?;
            events.push(StoredEvent {
                sequence: to_u64(sequence)?,
                event_id,
                turn_id: Some(request.turn_id),
                event_type: event_types[index].into(),
                payload: payloads[index].clone(),
                created_at_unix_ms: request.created_at_unix_ms,
            });
        }
        transaction.commit()?;
        let events = events.try_into().map_err(|_| StoreError::Unavailable)?;
        Ok(AcceptTurn::Accepted { events })
    }

    pub fn accept_cancel_turn(
        &mut self,
        request: AcceptCancelTurn<'_>,
    ) -> Result<Option<StoredEvent>, StoreError> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<String> = transaction.query_row(
            "SELECT request_digest FROM remote_commands WHERE device_id = ?1 AND command_id = ?2",
            params![request.device_id.to_string(), request.command_id.to_string()],
            |row| row.get(0),
        ).optional()?;
        if let Some(digest) = existing {
            if digest != request.request_digest {
                return Err(StoreError::Conflict);
            }
            return Ok(None);
        }
        let active: i64 = transaction.query_row(
            "SELECT count(*) FROM remote_turns
             WHERE session_id = ?1 AND turn_id = ?2 AND finished_at IS NULL",
            params![request.session_id.to_string(), request.turn_id.to_string()],
            |row| row.get(0),
        )?;
        if active != 1 {
            return Err(StoreError::Conflict);
        }
        let now = to_i64(request.created_at_unix_ms)?;
        let sequence: i64 = transaction.query_row(
            "UPDATE remote_sessions
             SET state = 'cancelling', next_event_sequence = next_event_sequence + 1, updated_at = ?2
             WHERE session_id = ?1 AND state IN ('running','waiting_approval')
             RETURNING next_event_sequence - 1",
            params![request.session_id.to_string(), now],
            |row| row.get(0),
        ).optional()?.ok_or(StoreError::Conflict)?;
        transaction.execute(
            "UPDATE remote_approvals
             SET state = 'settled', decision = 'invalidated_by_cancel', settled_at = ?2
             WHERE session_id = ?1 AND state = 'pending'",
            params![request.session_id.to_string(), now],
        )?;
        let response = serde_json::json!({"code":"ok","state":"cancelling"});
        transaction.execute(
            "INSERT INTO remote_commands (
                device_id, command_id, session_id, command_type, request_digest,
                status, response_json, created_at, finished_at
             ) VALUES (?1, ?2, ?3, 'cancel_turn', ?4, 'applied', ?5, ?6, ?6)",
            params![
                request.device_id.to_string(),
                request.command_id.to_string(),
                request.session_id.to_string(),
                request.request_digest,
                serde_json::to_string(&response)?,
                now
            ],
        )?;
        let event_id = Uuid::new_v4();
        let payload = serde_json::json!({"state":"cancelling"});
        transaction.execute(
            "INSERT INTO remote_events (
                session_id, sequence, event_id, turn_id, event_type, payload_json, created_at
             ) VALUES (?1, ?2, ?3, ?4, 'session.state_changed', ?5, ?6)",
            params![
                request.session_id.to_string(),
                sequence,
                event_id.to_string(),
                request.turn_id.to_string(),
                serde_json::to_string(&payload)?,
                now
            ],
        )?;
        transaction.commit()?;
        Ok(Some(StoredEvent {
            sequence: to_u64(sequence)?,
            event_id,
            turn_id: Some(request.turn_id),
            event_type: "session.state_changed".into(),
            payload,
            created_at_unix_ms: request.created_at_unix_ms,
        }))
    }

    pub fn cancel_active_turn_locally(
        &mut self,
        session_id: Uuid,
        created_at_unix_ms: u64,
    ) -> Result<Option<StoredEvent>, StoreError> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let turn_id: Option<String> = transaction
            .query_row(
                "SELECT turn_id FROM remote_turns
                 WHERE session_id = ?1 AND finished_at IS NULL",
                [session_id.to_string()],
                |row| row.get(0),
            )
            .optional()?;
        let Some(turn_id) = turn_id else {
            return Ok(None);
        };
        let now = to_i64(created_at_unix_ms)?;
        let sequence: i64 = transaction
            .query_row(
                "UPDATE remote_sessions
                 SET state = 'cancelling', next_event_sequence = next_event_sequence + 1,
                     updated_at = ?2
                 WHERE session_id = ?1 AND state IN ('running','waiting_approval')
                 RETURNING next_event_sequence - 1",
                params![session_id.to_string(), now],
                |row| row.get(0),
            )
            .optional()?
            .ok_or(StoreError::Conflict)?;
        transaction.execute(
            "UPDATE remote_approvals
             SET state = 'settled', decision = 'invalidated_by_cancel', settled_at = ?2
             WHERE session_id = ?1 AND state = 'pending'",
            params![session_id.to_string(), now],
        )?;
        let turn_id = Uuid::parse_str(&turn_id).map_err(|_| StoreError::Invalid)?;
        let event_id = Uuid::new_v4();
        let payload = serde_json::json!({"state":"cancelling"});
        transaction.execute(
            "INSERT INTO remote_events (
                session_id, sequence, event_id, turn_id, event_type, payload_json, created_at
             ) VALUES (?1, ?2, ?3, ?4, 'session.state_changed', ?5, ?6)",
            params![
                session_id.to_string(),
                sequence,
                event_id.to_string(),
                turn_id.to_string(),
                serde_json::to_string(&payload)?,
                now,
            ],
        )?;
        transaction.commit()?;
        Ok(Some(StoredEvent {
            sequence: to_u64(sequence)?,
            event_id,
            turn_id: Some(turn_id),
            event_type: "session.state_changed".into(),
            payload,
            created_at_unix_ms,
        }))
    }

    pub fn accept_approval_decision(
        &mut self,
        request: AcceptApprovalDecision<'_>,
    ) -> Result<AcceptDecision, StoreError> {
        if !matches!(request.decision, "allow_once" | "deny" | "abort_turn") {
            return Err(StoreError::Invalid);
        }
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<(String, String, String)> = transaction
            .query_row(
                "SELECT request_digest, status, response_json FROM remote_commands
                 WHERE device_id = ?1 AND command_id = ?2",
                params![
                    request.device_id.to_string(),
                    request.command_id.to_string()
                ],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        if let Some((digest, status, response_json)) = existing {
            if digest != request.request_digest {
                return Err(StoreError::Conflict);
            }
            return Ok(AcceptDecision::Duplicate(CommandResult {
                status,
                response_json,
            }));
        }
        let settled_at = to_i64(request.created_at_unix_ms)?;
        settle_approval_in(
            &transaction,
            &SettleApproval {
                approval_id: request.approval_id,
                session_id: request.session_id,
                turn_id: request.turn_id,
                call_id: request.call_id,
                action_digest: request.action_digest,
                decision: request.decision,
                device_id: Some(request.device_id),
                settled_at_unix_ms: request.created_at_unix_ms,
            },
        )?;
        let response = serde_json::json!({"code":"ok","decision":request.decision});
        transaction.execute(
            "INSERT INTO remote_commands (
                device_id, command_id, session_id, command_type, request_digest,
                status, response_json, created_at, finished_at
             ) VALUES (?1, ?2, ?3, 'approval_decision', ?4, 'applied', ?5, ?6, ?6)",
            params![
                request.device_id.to_string(),
                request.command_id.to_string(),
                request.session_id.to_string(),
                request.request_digest,
                serde_json::to_string(&response)?,
                settled_at,
            ],
        )?;
        transaction.commit()?;
        Ok(AcceptDecision::Applied)
    }

    pub fn expire_approval(
        &mut self,
        request: ExpireApproval<'_>,
    ) -> Result<StoredEvent, StoreError> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        settle_approval_in(
            &transaction,
            &SettleApproval {
                approval_id: request.approval_id,
                session_id: request.session_id,
                turn_id: request.turn_id,
                call_id: request.call_id,
                action_digest: request.action_digest,
                decision: "expired",
                device_id: None,
                settled_at_unix_ms: request.expired_at_unix_ms,
            },
        )?;
        let sequence: i64 = transaction.query_row(
            "UPDATE remote_sessions SET next_event_sequence = next_event_sequence + 1
             WHERE session_id = ?1 RETURNING next_event_sequence - 1",
            [request.session_id.to_string()],
            |row| row.get(0),
        )?;
        let event_id = Uuid::new_v4();
        let payload = serde_json::json!({
            "approval_id": request.approval_id,
            "call_id": request.call_id,
            "action_digest": request.action_digest,
        });
        transaction.execute(
            "INSERT INTO remote_events (
                session_id, sequence, event_id, turn_id, event_type, payload_json, created_at
             ) VALUES (?1, ?2, ?3, ?4, 'approval.expired', ?5, ?6)",
            params![
                request.session_id.to_string(),
                sequence,
                event_id.to_string(),
                request.turn_id.to_string(),
                serde_json::to_string(&payload)?,
                to_i64(request.expired_at_unix_ms)?,
            ],
        )?;
        transaction.commit()?;
        Ok(StoredEvent {
            sequence: to_u64(sequence)?,
            event_id,
            turn_id: Some(request.turn_id),
            event_type: "approval.expired".into(),
            payload,
            created_at_unix_ms: request.expired_at_unix_ms,
        })
    }

    pub fn approval_decision(
        &self,
        session_id: Uuid,
        turn_id: Uuid,
        call_id: Uuid,
        approval_id: Uuid,
        action_digest: &str,
    ) -> Result<Option<String>, StoreError> {
        self.connection
            .query_row(
                "SELECT decision FROM remote_approvals
                 WHERE session_id = ?1 AND turn_id = ?2 AND call_id = ?3
                   AND approval_id = ?4 AND action_digest = ?5 AND state = 'settled'",
                params![
                    session_id.to_string(),
                    turn_id.to_string(),
                    call_id.to_string(),
                    approval_id.to_string(),
                    action_digest,
                ],
                |row| row.get(0),
            )
            .optional()
            .map_err(StoreError::from)
    }

    pub fn finish_turn(
        &mut self,
        turn_id: Uuid,
        outcome: &str,
        assistant_text: Option<&str>,
        finished_at_unix_ms: u64,
    ) -> Result<(), StoreError> {
        if !matches!(
            outcome,
            "completed" | "aborted" | "step_capped" | "repeated" | "driver_error" | "interrupted"
        ) {
            return Err(StoreError::Invalid);
        }
        let changed = self.connection.execute(
            "UPDATE remote_turns SET state = 'finished', outcome = ?1, assistant_text = ?2, finished_at = ?3
             WHERE turn_id = ?4 AND finished_at IS NULL",
            params![outcome, assistant_text, to_i64(finished_at_unix_ms)?, turn_id.to_string()],
        )?;
        if changed != 1 {
            return Err(StoreError::Conflict);
        }
        Ok(())
    }

    pub fn complete_turn(&mut self, request: CompleteTurn<'_>) -> Result<StoredEvent, StoreError> {
        if !matches!(
            request.outcome,
            "completed" | "aborted" | "step_capped" | "repeated" | "driver_error" | "interrupted"
        ) {
            return Err(StoreError::Invalid);
        }
        serde_json::from_str::<Value>(request.transcript_json)?;
        serde_json::from_str::<Value>(request.plan_json)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let now = to_i64(request.finished_at_unix_ms)?;
        let command_id: String = transaction
            .query_row(
            "UPDATE remote_turns SET state = 'finished', outcome = ?1, assistant_text = ?2, finished_at = ?3
             WHERE turn_id = ?4 AND session_id = ?5 AND finished_at IS NULL
             RETURNING command_id",
                params![request.outcome, request.assistant_text, now, request.turn_id.to_string(), request.session_id.to_string()],
                |row| row.get(0),
            )
            .optional()?
            .ok_or(StoreError::Conflict)?;
        let allowed_state = if request.outcome == "aborted" {
            "state IN ('running','cancelling')"
        } else {
            "state = 'running'"
        };
        let sql = format!(
            "UPDATE remote_sessions
             SET state = 'idle', next_event_sequence = next_event_sequence + 1, updated_at = ?2,
                 transcript_json = ?3, plan_json = ?4
             WHERE session_id = ?1 AND {allowed_state}
             RETURNING next_event_sequence - 1"
        );
        let sequence: i64 = transaction
            .query_row(
                &sql,
                params![
                    request.session_id.to_string(),
                    now,
                    request.transcript_json,
                    request.plan_json
                ],
                |row| row.get(0),
            )
            .optional()?
            .ok_or(StoreError::Conflict)?;
        let event_id = Uuid::new_v4();
        let payload = serde_json::json!({"outcome": request.outcome});
        transaction.execute(
            "INSERT INTO remote_events (
                session_id, sequence, event_id, turn_id, event_type, payload_json, created_at
             ) VALUES (?1, ?2, ?3, ?4, 'turn.finished', ?5, ?6)",
            params![
                request.session_id.to_string(),
                sequence,
                event_id.to_string(),
                request.turn_id.to_string(),
                serde_json::to_string(&payload)?,
                now
            ],
        )?;
        let command_response = serde_json::json!({
            "code": request.outcome,
            "turn_id": request.turn_id,
            "current_event_sequence": sequence,
        });
        let changed = transaction.execute(
            "UPDATE remote_commands
             SET status = 'applied', response_json = ?1, finished_at = ?2
             WHERE session_id = ?3 AND command_id = ?4
               AND command_type = 'start_turn' AND status = 'accepted'",
            params![
                serde_json::to_string(&command_response)?,
                now,
                request.session_id.to_string(),
                command_id,
            ],
        )?;
        if changed != 1 {
            return Err(StoreError::Conflict);
        }
        transaction.commit()?;
        Ok(StoredEvent {
            sequence: to_u64(sequence)?,
            event_id,
            turn_id: Some(request.turn_id),
            event_type: "turn.finished".into(),
            payload,
            created_at_unix_ms: request.finished_at_unix_ms,
        })
    }

    pub fn transition_session(
        &mut self,
        session_id: Uuid,
        expected: SessionState,
        next: SessionState,
        updated_at_unix_ms: u64,
    ) -> Result<(), StoreError> {
        if !expected.may_transition_to(next) {
            return Err(StoreError::Invalid);
        }
        let changed = self.connection.execute(
            "UPDATE remote_sessions SET state = ?1, updated_at = ?2
             WHERE session_id = ?3 AND state = ?4",
            params![
                next.token(),
                to_i64(updated_at_unix_ms)?,
                session_id.to_string(),
                expected.token(),
            ],
        )?;
        if changed != 1 {
            return Err(StoreError::Conflict);
        }
        Ok(())
    }

    pub fn append_event(
        &mut self,
        session_id: Uuid,
        turn_id: Option<Uuid>,
        event_type: &str,
        payload: &Value,
        created_at_unix_ms: u64,
    ) -> Result<StoredEvent, StoreError> {
        validate_token(event_type)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let sequence: i64 = transaction.query_row(
            "UPDATE remote_sessions
             SET next_event_sequence = next_event_sequence + 1, updated_at = ?2
             WHERE session_id = ?1
             RETURNING next_event_sequence - 1",
            params![session_id.to_string(), to_i64(created_at_unix_ms)?],
            |row| row.get(0),
        )?;
        let event_id = Uuid::new_v4();
        let payload_json = serde_json::to_string(payload)?;
        transaction.execute(
            "INSERT INTO remote_events (
                session_id, sequence, event_id, turn_id, event_type, payload_json, created_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session_id.to_string(),
                sequence,
                event_id.to_string(),
                turn_id.map(|id| id.to_string()),
                event_type,
                payload_json,
                to_i64(created_at_unix_ms)?,
            ],
        )?;
        transaction.commit()?;
        Ok(StoredEvent {
            sequence: to_u64(sequence)?,
            event_id,
            turn_id,
            event_type: event_type.into(),
            payload: payload.clone(),
            created_at_unix_ms,
        })
    }

    pub fn replay(
        &self,
        session_id: Uuid,
        after_sequence: u64,
        limit: u16,
    ) -> Result<Vec<StoredEvent>, StoreError> {
        if limit == 0 || limit > 256 {
            return Err(StoreError::Invalid);
        }
        let mut statement = self.connection.prepare(
            "SELECT sequence, event_id, turn_id, event_type, payload_json, created_at
             FROM remote_events
             WHERE session_id = ?1 AND sequence > ?2
             ORDER BY sequence ASC LIMIT ?3",
        )?;
        let rows = statement.query_map(
            params![
                session_id.to_string(),
                to_i64(after_sequence)?,
                i64::from(limit),
            ],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            },
        )?;
        rows.map(|row| {
            let (sequence, event_id, turn_id, event_type, payload_json, created_at) = row?;
            Ok(StoredEvent {
                sequence: to_u64(sequence)?,
                event_id: Uuid::parse_str(&event_id).map_err(|_| StoreError::Invalid)?,
                turn_id: turn_id
                    .map(|id| Uuid::parse_str(&id).map_err(|_| StoreError::Invalid))
                    .transpose()?,
                event_type,
                payload: serde_json::from_str(&payload_json)?,
                created_at_unix_ms: to_u64(created_at)?,
            })
        })
        .collect()
    }

    pub fn begin_command(
        &mut self,
        device_id: Uuid,
        command_id: Uuid,
        session_id: Uuid,
        command_type: &str,
        request_digest: &str,
        created_at_unix_ms: u64,
    ) -> Result<BeginCommand, StoreError> {
        let existing: Option<(String, String, String)> = self
            .connection
            .query_row(
                "SELECT request_digest, status, response_json FROM remote_commands
                 WHERE device_id = ?1 AND command_id = ?2",
                params![device_id.to_string(), command_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        if let Some((stored_digest, status, response_json)) = existing {
            if stored_digest != request_digest {
                return Err(StoreError::Conflict);
            }
            return Ok(BeginCommand::Duplicate(CommandResult {
                status,
                response_json,
            }));
        }
        self.connection.execute(
            "INSERT INTO remote_commands (
                device_id, command_id, session_id, command_type, request_digest,
                status, response_json, created_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, 'accepted', '{}', ?6)",
            params![
                device_id.to_string(),
                command_id.to_string(),
                session_id.to_string(),
                command_type,
                request_digest,
                to_i64(created_at_unix_ms)?,
            ],
        )?;
        Ok(BeginCommand::New)
    }

    pub fn finish_command(
        &mut self,
        device_id: Uuid,
        command_id: Uuid,
        status: &str,
        response: &Value,
        finished_at_unix_ms: u64,
    ) -> Result<(), StoreError> {
        if !matches!(status, "applied" | "rejected") {
            return Err(StoreError::Invalid);
        }
        let changed = self.connection.execute(
            "UPDATE remote_commands SET status = ?1, response_json = ?2, finished_at = ?3
             WHERE device_id = ?4 AND command_id = ?5 AND finished_at IS NULL",
            params![
                status,
                serde_json::to_string(response)?,
                to_i64(finished_at_unix_ms)?,
                device_id.to_string(),
                command_id.to_string(),
            ],
        )?;
        if changed != 1 {
            return Err(StoreError::Conflict);
        }
        Ok(())
    }

    pub fn insert_pending_approval(
        &mut self,
        approval: PendingApproval<'_>,
        event_payload: &Value,
    ) -> Result<StoredEvent, StoreError> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let changed = transaction.execute(
            "UPDATE remote_sessions SET state = 'waiting_approval', updated_at = ?2
             WHERE session_id = ?1 AND state = 'running'",
            params![
                approval.session_id.to_string(),
                to_i64(approval.created_at_unix_ms)?
            ],
        )?;
        if changed != 1 {
            return Err(StoreError::Conflict);
        }
        transaction.execute(
            "INSERT INTO remote_approvals (
                approval_id, session_id, turn_id, call_id, action_digest,
                tool, risk, detail_json, state, created_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'pending', ?9)",
            params![
                approval.approval_id.to_string(),
                approval.session_id.to_string(),
                approval.turn_id.to_string(),
                approval.call_id.to_string(),
                approval.action_digest,
                approval.tool,
                approval.risk,
                approval.detail_json,
                to_i64(approval.created_at_unix_ms)?,
            ],
        )?;
        let sequence: i64 = transaction.query_row(
            "UPDATE remote_sessions SET next_event_sequence = next_event_sequence + 1
             WHERE session_id = ?1 RETURNING next_event_sequence - 1",
            [approval.session_id.to_string()],
            |row| row.get(0),
        )?;
        let event_id = Uuid::new_v4();
        transaction.execute(
            "INSERT INTO remote_events (
                session_id, sequence, event_id, turn_id, event_type, payload_json, created_at
             ) VALUES (?1, ?2, ?3, ?4, 'approval.required', ?5, ?6)",
            params![
                approval.session_id.to_string(),
                sequence,
                event_id.to_string(),
                approval.turn_id.to_string(),
                serde_json::to_string(event_payload)?,
                to_i64(approval.created_at_unix_ms)?,
            ],
        )?;
        transaction.commit()?;
        Ok(StoredEvent {
            sequence: to_u64(sequence)?,
            event_id,
            turn_id: Some(approval.turn_id),
            event_type: "approval.required".into(),
            payload: event_payload.clone(),
            created_at_unix_ms: approval.created_at_unix_ms,
        })
    }

    pub fn settle_approval(&mut self, request: SettleApproval<'_>) -> Result<(), StoreError> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        settle_approval_in(&transaction, &request)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn recover_interrupted(&mut self, updated_at_unix_ms: u64) -> Result<usize, StoreError> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let updated_at = to_i64(updated_at_unix_ms)?;
        transaction.execute(
                        "UPDATE remote_commands
                         SET status = 'rejected', response_json = '{\"code\":\"interrupted\"}', finished_at = ?1
                         WHERE command_type = 'start_turn' AND status = 'accepted'
                             AND EXISTS (
                                 SELECT 1 FROM remote_turns
                                 WHERE remote_turns.session_id = remote_commands.session_id
                                     AND remote_turns.command_id = remote_commands.command_id
                                     AND remote_turns.finished_at IS NULL
                             )",
                        [updated_at],
                )?;
        transaction.execute(
            "UPDATE remote_approvals
             SET state = 'settled', decision = 'invalidated_by_restart', settled_at = ?1
             WHERE state = 'pending'",
            [updated_at],
        )?;
        transaction.execute(
            "UPDATE remote_turns
             SET state = 'finished', outcome = 'interrupted', finished_at = ?1
             WHERE finished_at IS NULL AND session_id IN (
                 SELECT session_id FROM remote_sessions
                 WHERE state IN ('running', 'waiting_approval', 'cancelling')
             )",
            [updated_at],
        )?;
        let changed = transaction.execute(
            "UPDATE remote_sessions SET state = 'failed', updated_at = ?1
             WHERE state IN ('running', 'waiting_approval', 'cancelling')",
            [updated_at],
        )?;
        transaction.commit()?;
        Ok(changed)
    }
}

fn settle_approval_in(
    transaction: &rusqlite::Transaction<'_>,
    request: &SettleApproval<'_>,
) -> Result<(), StoreError> {
    if !matches!(
        request.decision,
        "allow_once" | "deny" | "abort_turn" | "expired" | "invalidated_by_cancel"
    ) {
        return Err(StoreError::Invalid);
    }
    let changed = transaction.execute(
        "UPDATE remote_approvals
         SET state = 'settled', decision = ?1, decided_by_device = ?2, settled_at = ?3
         WHERE approval_id = ?4 AND session_id = ?5 AND turn_id = ?6 AND call_id = ?7
           AND action_digest = ?8 AND state = 'pending'",
        params![
            request.decision,
            request.device_id.map(|id| id.to_string()),
            to_i64(request.settled_at_unix_ms)?,
            request.approval_id.to_string(),
            request.session_id.to_string(),
            request.turn_id.to_string(),
            request.call_id.to_string(),
            request.action_digest,
        ],
    )?;
    if changed != 1 {
        return Err(StoreError::Conflict);
    }
    let next_state = match request.decision {
        "allow_once" | "deny" => "running",
        "abort_turn" | "expired" | "invalidated_by_cancel" => "cancelling",
        _ => return Err(StoreError::Invalid),
    };
    let changed = transaction.execute(
        "UPDATE remote_sessions SET state = ?1, updated_at = ?2
         WHERE session_id = ?3 AND state = 'waiting_approval'",
        params![
            next_state,
            to_i64(request.settled_at_unix_ms)?,
            request.session_id.to_string(),
        ],
    )?;
    if changed != 1 {
        return Err(StoreError::Conflict);
    }
    Ok(())
}

fn migrate(connection: &Connection) -> Result<(), StoreError> {
    let mut version: i64 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if version > SCHEMA_VERSION {
        return Err(StoreError::NewerSchema);
    }
    if version == 0 {
        connection.execute_batch(include_str!("schema_v1.sql"))?;
        connection.pragma_update(None, "user_version", 1)?;
        version = 1;
    }
    if version == 1 {
        connection.execute_batch(
            "BEGIN IMMEDIATE;
             CREATE TABLE remote_relay (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                relay_url TEXT NOT NULL,
                route_id TEXT NOT NULL,
                capability_secret_reference TEXT NOT NULL,
                updated_at INTEGER NOT NULL
             ) STRICT;
             PRAGMA user_version = 2;
             COMMIT;",
        )?;
        version = 2;
    }
    if version == 2 {
        connection.execute_batch(
            "BEGIN IMMEDIATE;
             CREATE TABLE remote_active_session (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                session_id TEXT NOT NULL UNIQUE REFERENCES remote_sessions(session_id),
                generation INTEGER NOT NULL CHECK(generation >= 1),
                activated_at INTEGER NOT NULL
             ) STRICT;
             CREATE UNIQUE INDEX remote_one_unfinished_turn
             ON remote_turns ((1)) WHERE finished_at IS NULL;
             PRAGMA user_version = 3;
             COMMIT;",
        )?;
    }
    Ok(())
}

fn next_active_generation(transaction: &rusqlite::Transaction<'_>) -> Result<u64, StoreError> {
    let current: Option<i64> = transaction
        .query_row(
            "SELECT generation FROM remote_active_session WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    to_u64(current.unwrap_or(0))?
        .checked_add(1)
        .ok_or(StoreError::Invalid)
}

fn ensure_no_unfinished_turn(transaction: &rusqlite::Transaction<'_>) -> Result<(), StoreError> {
    let active_turns: i64 = transaction.query_row(
        "SELECT count(*) FROM remote_turns WHERE finished_at IS NULL",
        [],
        |row| row.get(0),
    )?;
    if active_turns != 0 {
        return Err(StoreError::Conflict);
    }
    Ok(())
}

fn set_active_session(
    transaction: &rusqlite::Transaction<'_>,
    session_id: Uuid,
    generation: u64,
    activated_at: i64,
) -> Result<(), StoreError> {
    transaction.execute(
        "INSERT INTO remote_active_session (singleton, session_id, generation, activated_at)
         VALUES (1, ?1, ?2, ?3)
         ON CONFLICT(singleton) DO UPDATE SET
            session_id = excluded.session_id,
            generation = excluded.generation,
            activated_at = excluded.activated_at",
        params![session_id.to_string(), to_i64(generation)?, activated_at],
    )?;
    Ok(())
}

fn existing_session_command(
    transaction: &rusqlite::Transaction<'_>,
    device_id: Uuid,
    command_id: Uuid,
    request_digest: &str,
) -> Result<Option<CommandResult>, StoreError> {
    let existing: Option<(String, String, String)> = transaction
        .query_row(
            "SELECT request_digest, status, response_json FROM remote_commands
             WHERE device_id = ?1 AND command_id = ?2",
            params![device_id.to_string(), command_id.to_string()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    match existing {
        Some((stored_digest, _, _)) if stored_digest != request_digest => Err(StoreError::Conflict),
        Some((_, status, response_json)) => Ok(Some(CommandResult {
            status,
            response_json,
        })),
        None => Ok(None),
    }
}

#[allow(clippy::too_many_arguments)]
fn insert_session_command(
    transaction: &rusqlite::Transaction<'_>,
    device_id: Uuid,
    command_id: Uuid,
    session_id: Uuid,
    command_type: &str,
    request_digest: &str,
    response: &Value,
    now: i64,
) -> Result<(), StoreError> {
    transaction.execute(
        "INSERT INTO remote_commands (
            device_id, command_id, session_id, command_type, request_digest,
            status, response_json, created_at, finished_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, 'applied', ?6, ?7, ?7)",
        params![
            device_id.to_string(),
            command_id.to_string(),
            session_id.to_string(),
            command_type,
            request_digest,
            serde_json::to_string(response)?,
            now,
        ],
    )?;
    Ok(())
}

fn validate_token(token: &str) -> Result<(), StoreError> {
    if token.is_empty()
        || token.len() > 64
        || !token.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-')
        })
    {
        return Err(StoreError::Invalid);
    }
    Ok(())
}

fn parse_session_state(state: &str) -> Result<SessionState, StoreError> {
    match state {
        "armed" => Ok(SessionState::Armed),
        "idle" => Ok(SessionState::Idle),
        "running" => Ok(SessionState::Running),
        "waiting_approval" => Ok(SessionState::WaitingApproval),
        "cancelling" => Ok(SessionState::Cancelling),
        "failed" => Ok(SessionState::Failed),
        "closed" => Ok(SessionState::Closed),
        _ => Err(StoreError::Invalid),
    }
}

fn to_i64(value: u64) -> Result<i64, StoreError> {
    value.try_into().map_err(|_| StoreError::Invalid)
}

fn to_u64(value: i64) -> Result<u64, StoreError> {
    value.try_into().map_err(|_| StoreError::Invalid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open() -> (tempfile::TempDir, RemoteStore, Uuid) {
        let directory = tempfile::tempdir().unwrap();
        let mut store = RemoteStore::open(&directory.path().join("remote.sqlite3")).unwrap();
        let session_id = Uuid::new_v4();
        store
            .create_session(session_id, "/work", "model", "sha256:model", "{}", 1)
            .unwrap();
        (directory, store, session_id)
    }

    #[test]
    fn events_commit_monotonic_sequence_and_replay_after_reopen() {
        let (directory, mut store, session_id) = open();
        assert_eq!(
            store
                .append_event(session_id, None, "session.armed", &serde_json::json!({}), 2)
                .unwrap()
                .sequence,
            1
        );
        assert_eq!(
            store
                .append_event(
                    session_id,
                    None,
                    "session.state_changed",
                    &serde_json::json!({"state":"idle"}),
                    3
                )
                .unwrap()
                .sequence,
            2
        );
        drop(store);
        let reopened = RemoteStore::open(&directory.path().join("remote.sqlite3")).unwrap();
        let replay = reopened.replay(session_id, 0, 256).unwrap();
        assert_eq!(
            replay
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            [1, 2]
        );
        assert_eq!(reopened.replay(session_id, 1, 1).unwrap().len(), 1);
        assert_eq!(
            reopened.session_head(session_id).unwrap(),
            SessionHead {
                state: SessionState::Armed,
                last_event_sequence: 2,
            }
        );
    }

    #[test]
    fn latest_session_summary_uses_latest_update_and_current_cursor() {
        let (_directory, mut store, first_session_id) = open();
        store
            .append_event(
                first_session_id,
                None,
                "session.armed",
                &serde_json::json!({}),
                2,
            )
            .unwrap();
        let second_session_id = Uuid::new_v4();
        store
            .create_session(
                second_session_id,
                "/other",
                "model",
                "sha256:model",
                "{}",
                3,
            )
            .unwrap();
        store
            .transition_session(
                second_session_id,
                SessionState::Armed,
                SessionState::Idle,
                4,
            )
            .unwrap();

        assert_eq!(
            store.latest_session_summary().unwrap(),
            Some(StoredSessionSummary {
                session_id: second_session_id,
                state: SessionState::Idle,
                last_event_sequence: 0,
                updated_at_unix_ms: 4,
                capability_snapshot_json: "{}".into(),
            })
        );
    }

    #[test]
    fn session_catalog_is_workspace_scoped_titled_and_cursor_stable() {
        let (_directory, mut store, first_session_id) = open();
        store
            .append_event(
                first_session_id,
                None,
                "user.message",
                &serde_json::json!({"content":"First task"}),
                30,
            )
            .unwrap();
        let second_session_id = Uuid::from_u128(2);
        store
            .create_session(
                second_session_id,
                "/work",
                "model",
                "sha256:model",
                "{}",
                20,
            )
            .unwrap();
        store
            .append_event(
                second_session_id,
                None,
                "user.message",
                &serde_json::json!({"content":"Second task"}),
                30,
            )
            .unwrap();
        let other_session_id = Uuid::from_u128(3);
        store
            .create_session(
                other_session_id,
                "/other",
                "model",
                "sha256:model",
                "{}",
                40,
            )
            .unwrap();

        let first_page = store.list_session_catalog("/work", None, 1).unwrap();
        assert_eq!(first_page.len(), 1);
        assert_eq!(first_page[0].title, "Second task");
        assert_eq!(first_page[0].session_id, second_session_id);
        let second_page = store
            .list_session_catalog(
                "/work",
                Some((first_page[0].updated_at_unix_ms, first_page[0].session_id)),
                2,
            )
            .unwrap();
        assert_eq!(second_page.len(), 1);
        assert_eq!(second_page[0].title, "First task");
        assert_eq!(second_page[0].session_id, first_session_id);
        assert!(!first_page
            .iter()
            .chain(second_page.iter())
            .any(|entry| entry.session_id == other_session_id));
    }

    #[test]
    fn active_session_create_switch_and_restart_are_atomic() {
        let (directory, mut store, first_session_id) = open();
        assert_eq!(store.active_session().unwrap(), None);
        assert_eq!(
            store.activate_session(first_session_id, 2).unwrap(),
            ActiveSession {
                session_id: first_session_id,
                generation: 1,
            }
        );
        let second_session_id = Uuid::new_v4();
        assert_eq!(
            store
                .create_and_activate_session(
                    second_session_id,
                    "/work",
                    "model",
                    "sha256:model",
                    "{}",
                    3,
                )
                .unwrap(),
            ActiveSession {
                session_id: second_session_id,
                generation: 2,
            }
        );
        assert!(store.session_head(first_session_id).is_ok());
        drop(store);
        let mut reopened = RemoteStore::open(&directory.path().join("remote.sqlite3")).unwrap();
        assert_eq!(
            reopened.active_session().unwrap(),
            Some(ActiveSession {
                session_id: second_session_id,
                generation: 2,
            })
        );
        let turn_id = Uuid::new_v4();
        assert!(matches!(
            reopened
                .accept_start_turn(AcceptStartTurn {
                    device_id: Uuid::new_v4(),
                    command_id: Uuid::new_v4(),
                    request_digest: "sha256:start",
                    session_id: second_session_id,
                    turn_id,
                    user_text: "work",
                    created_at_unix_ms: 4,
                })
                .unwrap(),
            AcceptTurn::Accepted { .. }
        ));
        assert!(matches!(
            reopened.activate_session(first_session_id, 5),
            Err(StoreError::Conflict)
        ));
        assert_eq!(
            reopened.active_session().unwrap().unwrap().session_id,
            second_session_id
        );
    }

    #[test]
    fn session_switch_commands_are_atomic_idempotent_and_digest_bound() {
        let (_directory, mut store, first_session_id) = open();
        store.activate_session(first_session_id, 2).unwrap();
        let device_id = Uuid::new_v4();
        let create_command = Uuid::new_v4();
        let second_session_id = Uuid::new_v4();
        let create = || AcceptCreateSession {
            device_id,
            command_id: create_command,
            request_digest: "sha256:create",
            session_id: second_session_id,
            canonical_root: "/work",
            model_id: "model",
            model_sha256: "sha256:model",
            capability_snapshot_json: "{}",
            created_at_unix_ms: 3,
        };
        assert!(matches!(
            store.accept_create_session(create()).unwrap(),
            AcceptSessionSwitch::Applied(ActiveSession { session_id, .. })
                if session_id == second_session_id
        ));
        assert_eq!(store.replay(second_session_id, 0, 256).unwrap().len(), 2);
        assert_eq!(
            store
                .session_head(second_session_id)
                .unwrap()
                .last_event_sequence,
            2
        );
        assert!(matches!(
            store.accept_create_session(create()).unwrap(),
            AcceptSessionSwitch::Duplicate(_)
        ));
        assert!(matches!(
            store.accept_create_session(AcceptCreateSession {
                request_digest: "sha256:changed",
                ..create()
            }),
            Err(StoreError::Conflict)
        ));

        let activate_command = Uuid::new_v4();
        let activate = || AcceptActivateSession {
            device_id,
            command_id: activate_command,
            request_digest: "sha256:activate",
            session_id: first_session_id,
            canonical_root: "/work",
            model_id: "model",
            model_sha256: "sha256:model",
            capability_snapshot_json: "{}",
            activated_at_unix_ms: 4,
        };
        assert!(matches!(
            store.accept_activate_session(activate()).unwrap(),
            AcceptSessionSwitch::Applied(ActiveSession { session_id, .. })
                if session_id == first_session_id
        ));
        assert!(matches!(
            store.accept_activate_session(activate()).unwrap(),
            AcceptSessionSwitch::Duplicate(_)
        ));
    }

    #[test]
    fn session_bootstrap_events_are_durable_and_idempotent() {
        let (_directory, mut store, session_id) = open();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 2)
            .unwrap();
        let capabilities = serde_json::json!({
            "workspace": "/work",
            "model_id": "model",
            "tools": ["read_file"]
        });
        let events = store
            .ensure_session_bootstrap_events(session_id, &capabilities, 3)
            .unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "host.capabilities");
        assert_eq!(events[0].payload, capabilities);
        assert_eq!(events[1].event_type, "session.armed");
        assert!(store
            .ensure_session_bootstrap_events(session_id, &serde_json::json!({}), 4)
            .unwrap()
            .is_empty());
        assert_eq!(store.replay(session_id, 0, 256).unwrap(), events);
    }

    #[test]
    fn command_ids_are_idempotent_but_conflicting_reuse_fails() {
        let (_directory, mut store, session_id) = open();
        let device = Uuid::new_v4();
        let command = Uuid::new_v4();
        assert!(matches!(
            store
                .begin_command(device, command, session_id, "start_turn", "sha256:a", 2)
                .unwrap(),
            BeginCommand::New
        ));
        store
            .finish_command(
                device,
                command,
                "applied",
                &serde_json::json!({"code":"ok"}),
                3,
            )
            .unwrap();
        let BeginCommand::Duplicate(result) = store
            .begin_command(device, command, session_id, "start_turn", "sha256:a", 4)
            .unwrap()
        else {
            panic!("expected duplicate")
        };
        assert_eq!(result.status, "applied");
        assert!(matches!(
            store.begin_command(
                device,
                command,
                session_id,
                "start_turn",
                "sha256:different",
                4
            ),
            Err(StoreError::Conflict)
        ));
    }

    #[test]
    fn approval_insert_and_event_are_atomic_and_first_settlement_wins() {
        let (_directory, mut store, session_id) = open();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 2)
            .unwrap();
        store
            .transition_session(session_id, SessionState::Idle, SessionState::Running, 3)
            .unwrap();
        let turn_id = Uuid::new_v4();
        let call_id = Uuid::new_v4();
        let approval_id = Uuid::new_v4();
        let digest = "sha256:approval";
        let event = store
            .insert_pending_approval(
                PendingApproval {
                    approval_id,
                    session_id,
                    turn_id,
                    call_id,
                    action_digest: digest,
                    tool: "write_file",
                    risk: "write",
                    detail_json: "{}",
                    created_at_unix_ms: 4,
                },
                &serde_json::json!({"approval_id":approval_id}),
            )
            .unwrap();
        assert_eq!(event.sequence, 1);
        store
            .settle_approval(SettleApproval {
                approval_id,
                session_id,
                turn_id,
                call_id,
                action_digest: digest,
                decision: "allow_once",
                device_id: Some(Uuid::new_v4()),
                settled_at_unix_ms: 5,
            })
            .unwrap();
        assert!(matches!(
            store.settle_approval(SettleApproval {
                approval_id,
                session_id,
                turn_id,
                call_id,
                action_digest: digest,
                decision: "allow_once",
                device_id: None,
                settled_at_unix_ms: 6
            }),
            Err(StoreError::Conflict)
        ));
        assert!(matches!(
            store.settle_approval(SettleApproval {
                approval_id: Uuid::new_v4(),
                session_id,
                turn_id,
                call_id,
                action_digest: digest,
                decision: "allow_once",
                device_id: None,
                settled_at_unix_ms: 6
            }),
            Err(StoreError::Conflict)
        ));
    }

    #[test]
    fn approval_decision_command_is_atomic_idempotent_and_exactly_scoped() {
        let (_directory, mut store, session_id) = open();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 2)
            .unwrap();
        store
            .transition_session(session_id, SessionState::Idle, SessionState::Running, 3)
            .unwrap();
        let device_id = Uuid::new_v4();
        let command_id = Uuid::new_v4();
        let turn_id = Uuid::new_v4();
        let call_id = Uuid::new_v4();
        let approval_id = Uuid::new_v4();
        let action_digest = "sha256:approval";
        store
            .insert_pending_approval(
                PendingApproval {
                    approval_id,
                    session_id,
                    turn_id,
                    call_id,
                    action_digest,
                    tool: "write_file",
                    risk: "write",
                    detail_json: "{}",
                    created_at_unix_ms: 4,
                },
                &serde_json::json!({}),
            )
            .unwrap();
        let request = || AcceptApprovalDecision {
            device_id,
            command_id,
            request_digest: "sha256:command",
            session_id,
            turn_id,
            call_id,
            approval_id,
            action_digest,
            decision: "allow_once",
            created_at_unix_ms: 5,
        };
        assert!(matches!(
            store.accept_approval_decision(request()).unwrap(),
            AcceptDecision::Applied
        ));
        assert_eq!(
            store
                .approval_decision(session_id, turn_id, call_id, approval_id, action_digest)
                .unwrap()
                .as_deref(),
            Some("allow_once")
        );
        assert!(matches!(
            store.accept_approval_decision(request()).unwrap(),
            AcceptDecision::Duplicate(_)
        ));
        assert!(matches!(
            store.accept_approval_decision(AcceptApprovalDecision {
                request_digest: "sha256:other",
                ..request()
            }),
            Err(StoreError::Conflict)
        ));
        assert!(matches!(
            store.accept_approval_decision(AcceptApprovalDecision {
                command_id: Uuid::new_v4(),
                approval_id: Uuid::new_v4(),
                ..request()
            }),
            Err(StoreError::Conflict)
        ));
    }

    #[test]
    fn approval_timeout_aborts_and_records_expiry_before_late_decisions() {
        let (_directory, mut store, session_id) = open();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 2)
            .unwrap();
        store
            .transition_session(session_id, SessionState::Idle, SessionState::Running, 3)
            .unwrap();
        let turn_id = Uuid::new_v4();
        let call_id = Uuid::new_v4();
        let approval_id = Uuid::new_v4();
        let action_digest = "sha256:approval";
        store
            .insert_pending_approval(
                PendingApproval {
                    approval_id,
                    session_id,
                    turn_id,
                    call_id,
                    action_digest,
                    tool: "write_file",
                    risk: "write",
                    detail_json: "{}",
                    created_at_unix_ms: 4,
                },
                &serde_json::json!({}),
            )
            .unwrap();
        let expired = store
            .expire_approval(ExpireApproval {
                session_id,
                turn_id,
                call_id,
                approval_id,
                action_digest,
                expired_at_unix_ms: 5,
            })
            .unwrap();
        assert_eq!(expired.event_type, "approval.expired");
        assert_eq!(
            store
                .approval_decision(session_id, turn_id, call_id, approval_id, action_digest)
                .unwrap()
                .as_deref(),
            Some("expired")
        );
        assert!(matches!(
            store.accept_approval_decision(AcceptApprovalDecision {
                device_id: Uuid::new_v4(),
                command_id: Uuid::new_v4(),
                request_digest: "sha256:late",
                session_id,
                turn_id,
                call_id,
                approval_id,
                action_digest,
                decision: "allow_once",
                created_at_unix_ms: 6,
            }),
            Err(StoreError::Conflict)
        ));
        assert_eq!(
            store
                .replay(session_id, 0, 256)
                .unwrap()
                .last()
                .unwrap()
                .event_type,
            "approval.expired"
        );
    }

    #[test]
    fn restart_invalidates_pending_work_and_never_resumes_it() {
        let (_directory, mut store, session_id) = open();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 2)
            .unwrap();
        let approval_id = Uuid::new_v4();
        let turn_id = Uuid::new_v4();
        let call_id = Uuid::new_v4();
        let AcceptTurn::Accepted { .. } = store
            .accept_start_turn(AcceptStartTurn {
                device_id: Uuid::new_v4(),
                command_id: Uuid::new_v4(),
                request_digest: "sha256:start",
                session_id,
                turn_id,
                user_text: "work",
                created_at_unix_ms: 3,
            })
            .unwrap()
        else {
            panic!("expected accepted")
        };
        store
            .insert_pending_approval(
                PendingApproval {
                    approval_id,
                    session_id,
                    turn_id,
                    call_id,
                    action_digest: "sha256:x",
                    tool: "write_file",
                    risk: "write",
                    detail_json: "{}",
                    created_at_unix_ms: 4,
                },
                &serde_json::json!({}),
            )
            .unwrap();
        assert_eq!(store.recover_interrupted(5).unwrap(), 1);
        assert!(matches!(
            store.settle_approval(SettleApproval {
                approval_id,
                session_id,
                turn_id,
                call_id,
                action_digest: "sha256:x",
                decision: "allow_once",
                device_id: None,
                settled_at_unix_ms: 6
            }),
            Err(StoreError::Conflict)
        ));
        assert!(matches!(
            store.finish_turn(turn_id, "completed", None, 6),
            Err(StoreError::Conflict)
        ));
        let (state, outcome, finished_at): (String, String, i64) = store
            .connection
            .query_row(
                "SELECT state, outcome, finished_at FROM remote_turns WHERE turn_id = ?1",
                [turn_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            (state.as_str(), outcome.as_str(), finished_at),
            ("finished", "interrupted", 5)
        );
    }

    #[test]
    fn newer_schema_fails_closed_without_overwrite() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("newer.sqlite3");
        let connection = Connection::open(&path).unwrap();
        connection.pragma_update(None, "user_version", 99).unwrap();
        drop(connection);
        assert!(matches!(
            RemoteStore::open(&path),
            Err(StoreError::NewerSchema)
        ));
        let connection = Connection::open(path).unwrap();
        assert_eq!(
            connection
                .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
                .unwrap(),
            99
        );
    }

    #[test]
    fn schema_one_migrates_to_relay_binding_without_plaintext_capability() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("schema-one.sqlite3");
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch(include_str!("schema_v1.sql"))
            .unwrap();
        connection.pragma_update(None, "user_version", 1).unwrap();
        drop(connection);

        let mut store = RemoteStore::open(&path).unwrap();
        assert_eq!(store.relay_binding().unwrap(), None);
        store
            .set_relay_binding(
                "wss://relay.example.test",
                "abcdefghijklmnopqrstuv",
                "dpapi-file:v1:11111111-1111-4111-8111-111111111111",
                7,
            )
            .unwrap();
        assert_eq!(
            store.relay_binding().unwrap(),
            Some(StoredRelayBinding {
                relay_url: "wss://relay.example.test".into(),
                route_id: "abcdefghijklmnopqrstuv".into(),
                capability_secret_reference: "dpapi-file:v1:11111111-1111-4111-8111-111111111111"
                    .into(),
            })
        );
        let database_bytes = std::fs::read(path).unwrap();
        assert!(!database_bytes
            .windows(b"secret-host-bearer".len())
            .any(|window| window == b"secret-host-bearer"));
    }

    #[test]
    fn host_identity_stores_public_key_and_secret_reference_once() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = RemoteStore::open(&directory.path().join("identity.sqlite3")).unwrap();
        assert_eq!(store.optional_host_identity().unwrap(), None);
        let host_id = Uuid::new_v4();
        store
            .initialize_host_identity(host_id, &[7_u8; 32], "os-secret://camelid/host", 1)
            .unwrap();
        assert_eq!(
            store.host_identity().unwrap(),
            StoredHostIdentity {
                host_id,
                noise_public: [7_u8; 32],
                secret_reference: "os-secret://camelid/host".into(),
            }
        );
        assert!(matches!(
            store.initialize_host_identity(Uuid::new_v4(), &[8_u8; 32], "other", 2),
            Err(StoreError::Unavailable)
        ));
    }

    #[test]
    fn session_state_machine_rejects_impossible_and_stale_transitions() {
        let (_directory, mut store, session_id) = open();
        assert!(matches!(
            store.transition_session(
                session_id,
                SessionState::Armed,
                SessionState::WaitingApproval,
                2
            ),
            Err(StoreError::Invalid)
        ));
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 2)
            .unwrap();
        assert!(matches!(
            store.transition_session(session_id, SessionState::Armed, SessionState::Idle, 3),
            Err(StoreError::Conflict)
        ));
    }

    #[test]
    fn replay_rejects_zero_and_overlarge_batches() {
        let (_directory, store, session_id) = open();
        assert!(matches!(
            store.replay(session_id, 0, 0),
            Err(StoreError::Invalid)
        ));
        assert!(matches!(
            store.replay(session_id, 0, 257),
            Err(StoreError::Invalid)
        ));
    }

    #[test]
    fn device_authorization_binds_key_and_revocation() {
        let (_directory, mut store, _session_id) = open();
        let device = Uuid::new_v4();
        let key = [7_u8; 32];
        store.register_device(device, "Phone", &key, 2).unwrap();
        assert_eq!(store.authorized_device_for_key(&key).unwrap(), Some(device));
        assert!(store.device_authorized(device, &key).unwrap());
        assert!(!store.device_authorized(device, &[8_u8; 32]).unwrap());
        store.mark_device_seen(device, 3).unwrap();
        assert_eq!(store.devices().unwrap()[0].last_seen_at_unix_ms, Some(3));
        store.revoke_device(device, 4).unwrap();
        assert_eq!(store.authorized_device_for_key(&key).unwrap(), None);
        assert!(!store.device_authorized(device, &key).unwrap());
        assert!(matches!(
            store.mark_device_seen(device, 5),
            Err(StoreError::Conflict)
        ));
        assert!(matches!(
            store.revoke_device(device, 5),
            Err(StoreError::Conflict)
        ));
    }

    #[test]
    fn device_inventory_and_revoke_all_are_durable_and_idempotent() {
        let (_directory, mut store, _session_id) = open();
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        store
            .register_device(first, "Phone", &[1_u8; 32], 2)
            .unwrap();
        store
            .register_device(second, "Tablet", &[2_u8; 32], 3)
            .unwrap();
        assert_eq!(
            store
                .devices()
                .unwrap()
                .iter()
                .map(|device| (device.device_id, device.label.as_str()))
                .collect::<Vec<_>>(),
            vec![(first, "Phone"), (second, "Tablet")]
        );
        assert_eq!(store.revoke_all_devices(4).unwrap(), vec![first, second]);
        assert!(store.revoke_all_devices(5).unwrap().is_empty());
        assert_eq!(store.authorized_device_count().unwrap(), 0);
    }

    #[test]
    fn local_cancel_invalidates_pending_approval_before_broadcast_event() {
        let (_directory, mut store, session_id) = open();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 2)
            .unwrap();
        let turn_id = Uuid::new_v4();
        let call_id = Uuid::new_v4();
        let approval_id = Uuid::new_v4();
        let AcceptTurn::Accepted { .. } = store
            .accept_start_turn(AcceptStartTurn {
                device_id: Uuid::new_v4(),
                command_id: Uuid::new_v4(),
                request_digest: "sha256:start",
                session_id,
                turn_id,
                user_text: "goal",
                created_at_unix_ms: 3,
            })
            .unwrap()
        else {
            panic!("expected accepted")
        };
        store
            .insert_pending_approval(
                PendingApproval {
                    approval_id,
                    session_id,
                    turn_id,
                    call_id,
                    action_digest: "sha256:approval",
                    tool: "write_file",
                    risk: "write",
                    detail_json: "{}",
                    created_at_unix_ms: 4,
                },
                &serde_json::json!({}),
            )
            .unwrap();
        let event = store
            .cancel_active_turn_locally(session_id, 5)
            .unwrap()
            .unwrap();
        assert_eq!(event.event_type, "session.state_changed");
        assert_eq!(
            store
                .approval_decision(session_id, turn_id, call_id, approval_id, "sha256:approval")
                .unwrap()
                .as_deref(),
            Some("invalidated_by_cancel")
        );
    }

    #[test]
    fn only_one_unfinished_turn_exists_and_terminal_settlement_is_idempotency_safe() {
        let (_directory, mut store, session_id) = open();
        let first = Uuid::new_v4();
        store
            .create_turn(first, session_id, Uuid::new_v4(), "goal", 2)
            .unwrap();
        assert!(matches!(
            store.create_turn(Uuid::new_v4(), session_id, Uuid::new_v4(), "other", 3),
            Err(StoreError::Conflict)
        ));
        store
            .finish_turn(first, "completed", Some("answer"), 4)
            .unwrap();
        assert!(matches!(
            store.finish_turn(first, "completed", Some("answer"), 5),
            Err(StoreError::Conflict)
        ));
        store
            .create_turn(Uuid::new_v4(), session_id, Uuid::new_v4(), "next", 6)
            .unwrap();
    }

    #[test]
    fn start_turn_atomically_deduplicates_state_turn_and_initial_events() {
        let (_directory, mut store, session_id) = open();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 2)
            .unwrap();
        let device_id = Uuid::new_v4();
        let command_id = Uuid::new_v4();
        let turn_id = Uuid::new_v4();
        let request = || AcceptStartTurn {
            device_id,
            command_id,
            request_digest: "sha256:start",
            session_id,
            turn_id,
            user_text: "do the work",
            created_at_unix_ms: 3,
        };
        let AcceptTurn::Accepted { events } = store.accept_start_turn(request()).unwrap() else {
            panic!("expected accepted")
        };
        assert_eq!(events[0].event_type, "user.message");
        assert_eq!(events[1].event_type, "turn.accepted");
        let AcceptTurn::Duplicate(result) = store.accept_start_turn(request()).unwrap() else {
            panic!("expected duplicate")
        };
        assert_eq!(result.status, "accepted");
        assert_eq!(store.replay(session_id, 0, 256).unwrap().len(), 2);
        assert!(matches!(
            store.accept_start_turn(AcceptStartTurn {
                request_digest: "sha256:other",
                ..request()
            }),
            Err(StoreError::Conflict)
        ));
        let finished = store
            .complete_turn(CompleteTurn {
                session_id,
                turn_id,
                outcome: "completed",
                assistant_text: Some("done"),
                transcript_json: "[]",
                plan_json: "[]",
                finished_at_unix_ms: 4,
            })
            .unwrap();
        let AcceptTurn::Duplicate(result) = store.accept_start_turn(request()).unwrap() else {
            panic!("expected completed duplicate")
        };
        assert_eq!(result.status, "applied");
        assert_eq!(
            serde_json::from_str::<Value>(&result.response_json).unwrap()["current_event_sequence"],
            finished.sequence
        );
    }

    #[test]
    fn settled_context_reopens_only_for_matching_root_and_model_identity() {
        let (directory, mut store, session_id) = open();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 2)
            .unwrap();
        let turn_id = Uuid::new_v4();
        let AcceptTurn::Accepted { .. } = store
            .accept_start_turn(AcceptStartTurn {
                device_id: Uuid::new_v4(),
                command_id: Uuid::new_v4(),
                request_digest: "sha256:start",
                session_id,
                turn_id,
                user_text: "goal",
                created_at_unix_ms: 3,
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
                assistant_text: Some("answer"),
                transcript_json: "[{\"role\":\"user\"}]",
                plan_json: "[]",
                finished_at_unix_ms: 4,
            })
            .unwrap();
        drop(store);
        let reopened = RemoteStore::open(&directory.path().join("remote.sqlite3")).unwrap();
        let context = reopened
            .load_session_context(session_id, "/work", "model", "sha256:model", "{}")
            .unwrap();
        assert_eq!(context.transcript_json, "[{\"role\":\"user\"}]");
        assert!(matches!(
            reopened.load_session_context(session_id, "/other", "model", "sha256:model", "{}"),
            Err(StoreError::Conflict)
        ));
        assert!(matches!(
            reopened.load_session_context(session_id, "/work", "other", "sha256:model", "{}"),
            Err(StoreError::Conflict)
        ));
        assert!(matches!(
            reopened.load_session_context(session_id, "/work", "model", "sha256:other", "{}"),
            Err(StoreError::Conflict)
        ));
        assert!(matches!(
            reopened.load_session_context(
                session_id,
                "/work",
                "model",
                "sha256:model",
                "{\"tools\":[]}"
            ),
            Err(StoreError::Conflict)
        ));
    }

    #[test]
    fn reusable_session_requires_exact_identity_and_explicit_rearm() {
        let (_directory, mut store, session_id) = open();
        assert_eq!(
            store
                .reusable_session("/work", "model", "sha256:model", "{}")
                .unwrap(),
            Some((session_id, SessionState::Armed))
        );
        assert_eq!(
            store
                .reusable_session("/other", "model", "sha256:model", "{}")
                .unwrap(),
            None
        );
        store
            .rearm_session(session_id, SessionState::Armed, 2)
            .unwrap();
        assert_eq!(
            store.session_head(session_id).unwrap().state,
            SessionState::Idle
        );
        assert!(matches!(
            store.rearm_session(session_id, SessionState::Idle, 3),
            Err(StoreError::Invalid)
        ));
    }

    #[test]
    fn interrupted_session_must_recover_to_failed_before_rearm() {
        let (_directory, mut store, session_id) = open();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 2)
            .unwrap();
        let turn_id = Uuid::new_v4();
        let AcceptTurn::Accepted { .. } = store
            .accept_start_turn(AcceptStartTurn {
                device_id: Uuid::new_v4(),
                command_id: Uuid::new_v4(),
                request_digest: "sha256:start",
                session_id,
                turn_id,
                user_text: "goal",
                created_at_unix_ms: 3,
            })
            .unwrap()
        else {
            panic!("expected accepted turn")
        };
        assert_eq!(store.recover_interrupted(4).unwrap(), 1);
        assert_eq!(
            store
                .reusable_session("/work", "model", "sha256:model", "{}")
                .unwrap(),
            Some((session_id, SessionState::Failed))
        );
        store
            .rearm_session(session_id, SessionState::Failed, 5)
            .unwrap();
        assert_eq!(
            store.session_head(session_id).unwrap().state,
            SessionState::Idle
        );
    }

    #[test]
    fn cancel_turn_is_durable_idempotent_and_scoped_to_the_active_turn() {
        let (_directory, mut store, session_id) = open();
        store
            .transition_session(session_id, SessionState::Armed, SessionState::Idle, 2)
            .unwrap();
        let turn_id = Uuid::new_v4();
        let device_id = Uuid::new_v4();
        let AcceptTurn::Accepted { .. } = store
            .accept_start_turn(AcceptStartTurn {
                device_id,
                command_id: Uuid::new_v4(),
                request_digest: "sha256:start",
                session_id,
                turn_id,
                user_text: "goal",
                created_at_unix_ms: 3,
            })
            .unwrap()
        else {
            panic!("expected accepted")
        };
        let command_id = Uuid::new_v4();
        let request = || AcceptCancelTurn {
            device_id,
            command_id,
            request_digest: "sha256:cancel",
            session_id,
            turn_id,
            created_at_unix_ms: 4,
        };
        let event = store
            .accept_cancel_turn(request())
            .unwrap()
            .expect("first cancel event");
        assert_eq!(event.event_type, "session.state_changed");
        assert!(store.accept_cancel_turn(request()).unwrap().is_none());
        assert!(matches!(
            store.accept_cancel_turn(AcceptCancelTurn {
                request_digest: "sha256:other",
                ..request()
            }),
            Err(StoreError::Conflict)
        ));
        assert!(matches!(
            store.accept_cancel_turn(AcceptCancelTurn {
                command_id: Uuid::new_v4(),
                turn_id: Uuid::new_v4(),
                ..request()
            }),
            Err(StoreError::Conflict)
        ));
    }
}
