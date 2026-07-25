BEGIN IMMEDIATE;

CREATE TABLE remote_meta (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    host_id TEXT NOT NULL UNIQUE,
    host_noise_public BLOB NOT NULL,
    host_secret_reference TEXT NOT NULL,
    created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE remote_devices (
    device_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    noise_static_public BLOB NOT NULL UNIQUE,
    created_at INTEGER NOT NULL,
    last_seen_at INTEGER,
    revoked_at INTEGER,
    push_capability_id TEXT
) STRICT;

CREATE TABLE remote_sessions (
    session_id TEXT PRIMARY KEY,
    canonical_root TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    capability_snapshot_json TEXT NOT NULL CHECK(json_valid(capability_snapshot_json)),
    state TEXT NOT NULL CHECK(state IN ('armed','idle','running','waiting_approval','cancelling','failed','closed')),
    transcript_json TEXT NOT NULL CHECK(json_valid(transcript_json)),
    plan_json TEXT NOT NULL CHECK(json_valid(plan_json)),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    closed_at INTEGER,
    next_event_sequence INTEGER NOT NULL CHECK(next_event_sequence >= 1)
) STRICT;

CREATE TABLE remote_turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES remote_sessions(session_id) ON DELETE CASCADE,
    command_id TEXT NOT NULL,
    user_text TEXT NOT NULL,
    state TEXT NOT NULL,
    outcome TEXT,
    assistant_text TEXT,
    created_at INTEGER NOT NULL,
    started_at INTEGER,
    finished_at INTEGER
) STRICT;

CREATE TABLE remote_events (
    session_id TEXT NOT NULL REFERENCES remote_sessions(session_id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    turn_id TEXT,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    created_at INTEGER NOT NULL,
    PRIMARY KEY(session_id, sequence)
) STRICT;

CREATE TABLE remote_commands (
    device_id TEXT NOT NULL,
    command_id TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES remote_sessions(session_id) ON DELETE CASCADE,
    command_type TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('accepted','applied','rejected')),
    response_json TEXT NOT NULL CHECK(json_valid(response_json)),
    created_at INTEGER NOT NULL,
    finished_at INTEGER,
    PRIMARY KEY(device_id, command_id)
) STRICT;

CREATE TABLE remote_approvals (
    approval_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES remote_sessions(session_id) ON DELETE CASCADE,
    turn_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    action_digest TEXT NOT NULL,
    tool TEXT NOT NULL,
    risk TEXT NOT NULL,
    detail_json TEXT NOT NULL CHECK(json_valid(detail_json)),
    state TEXT NOT NULL CHECK(state IN ('pending','settled')),
    decision TEXT,
    decided_by_device TEXT,
    created_at INTEGER NOT NULL,
    settled_at INTEGER,
    UNIQUE(session_id, turn_id, call_id, approval_id, action_digest)
) STRICT;

CREATE INDEX remote_events_replay ON remote_events(session_id, sequence);
CREATE INDEX remote_approvals_pending ON remote_approvals(session_id, state);

COMMIT;