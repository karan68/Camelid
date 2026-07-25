# Camelid Remote Agent Control

**Status:** Phase 0 protocol/crypto, Phase 1 runtime/event, Phase 2 durable local authority, and
Phase 3 relay/transport development gates are complete on the implementation branch. Phase 4 is
in progress: a development-only `camelid agent host` command and Android session/approval UI now
exist and pass their local development gates. Phase 5 has an initial development-only local
Remote view: it reads live host-owned status and can revoke a device or emergency-disable the
running host through the same-origin loopback API. Local QR creation/cancel/expiry/confirmation,
exact capability display, physical Android pair/revoke/re-pair, restart/replay, and one read-only
remote turn now have development evidence. Android can also browse durable agent histories, create
a new remote session, explicitly activate an older remote session, replay its transcript, and
continue it while the host retains exactly one active execution session. Retention controls, the
remaining physical lifecycle matrix, push, platform qualification, and release gates remain open.
No production relay deployment, implementation capability, support promotion, or shipped claim exists.
**Decision posture:** Works-anywhere, end-to-end encrypted, local-authority design.
**Last code audit:** 2026-07-25 on branch `feat/remote-agent-control` at HEAD
`5a85a98fd6e7056e3b7a597b44655cc711f39224` plus the uncommitted development changes recorded in
the Phase 4 receipts.
**Applies to:** the full CLI/TUI coding agent, not the read-only Web Workspace product.

## 1. Purpose

This document defines the end-to-end design for controlling a Camelid coding-agent
session from a phone while preserving Camelid's local execution and evidence-first
principles.

It is written to be sufficient context for a new implementation session. It records:

- verified behavior in the current repository;
- corrections to overly broad descriptions of current security boundaries;
- binding product and architecture decisions for the proposed feature;
- protocol, cryptography, persistence, lifecycle, relay, host, and mobile designs;
- exact repository seams to reuse and seams that must not be reused;
- implementation slices, tests, evidence gates, release criteria, and rollback;
- rejected alternatives and unresolved external deployment choices.

This document is not evidence that remote control has shipped. Until implementation and
the gates in this document pass, public capability surfaces must not advertise remote
control.

### 1.1 Claim labels

The document uses these labels deliberately:

- **VERIFIED**: observed in the repository at the audited commit, normally with a source
  path or symbol.
- **BINDING DESIGN**: the architecture this project will implement unless this document is
  amended before code lands.
- **GATE**: behavior that must be proven by executable tests or evidence before promotion.
- **DEFERRED**: intentionally outside the first release.
- **OPEN EXTERNAL**: cannot be decided from this repository alone, such as hosted relay
  ownership or production domain selection.

When a paragraph has no label, it explains or derives consequences from nearby verified
facts and binding decisions.

## 2. Product Thesis

**BINDING DESIGN:** The phone is a controller and synchronized view. The computer running
Camelid remains the sole authority for inference, filesystem access, tool validation,
approval settlement, cancellation, checkpoints, and session state.

The model and source tree do not move to the relay. The relay routes opaque encrypted
frames and generic wake notifications. It cannot decrypt prompts, responses, tool calls,
approval details, file contents, or tool results.

The intended workflow is:

1. On the computer, the user selects an exact tool-capable model and canonical workspace,
   reviews the enabled capability profile, and explicitly arms remote control.
2. Camelid loads the GGUF locally and keeps one local agent host running.
3. The user pairs a phone by scanning a short-lived QR code displayed by Camelid.
4. The phone may disconnect, lock, switch networks, or be killed without owning or ending
   the agent session.
5. The user starts a turn or watches an existing turn from the phone.
6. If a validated action requires approval, Camelid records it locally, sends an encrypted
   event, and waits.
7. The phone shows the exact validated action. The user can allow it once, deny it, or
   abort the turn.
8. Camelid verifies that the decision matches the current session, turn, call, approval,
   and action digest before executing anything.
9. Work continues locally. Reconnecting clients replay locally persisted events by
   sequence number.

### 2.1 Meaning of local-first

For this feature, local-first means:

- model weights and inference remain on the user's Camelid host;
- source files and tool execution remain on that host;
- the authoritative transcript, event log, device grants, and approval state remain on
  that host;
- the hosted relay receives only opaque ciphertext plus unavoidable routing metadata;
- direct private-network and self-hosted-relay deployment remain protocol-compatible;
- remote access is opt-in and removable without changing ordinary Chat, Web Workspace,
  the OpenAI-compatible API, or local CLI/TUI agent behavior.

It does not mean that a works-anywhere connection can avoid all intermediaries. A phone on
cellular data and a computer behind NAT need a rendezvous path unless the user supplies a
VPN or opens an inbound route. The design makes the intermediary blind rather than
pretending it does not exist.

### 2.2 Non-goals for v1

The first release will not provide:

- cloud-hosted inference or cloud-cloned repositories;
- model execution on the phone;
- remote selection of arbitrary host folders;
- remote model installation, loading, unloading, or deletion;
- multiple simultaneous agent sessions in one Camelid process;
- shared/team sessions or multiple writers;
- remote `--yolo` or `--auto-approve`;
- remote `--allow-fs`;
- remote persistent approval grants (`AlwaysTool`);
- MCP, subagents, GUI input, UI Automation, screenshot, system inspection, or arbitrary
  computer-control tools;
- remote shell by default;
- attachments, voice, or remote file browsing independent of agent observations;
- guaranteed execution while the host is asleep, powered off, or disconnected;
- delivery of commands while the host is offline;
- cryptographic concealment of traffic timing, message length, relay account identity, or
  push-delivery metadata;
- a claim that existing shell execution is a complete filesystem jail.

## 3. Current Camelid: Verified Architecture

This section describes what exists before this project.

### 3.1 Agent entry and model gate

**VERIFIED:** `src/main.rs` exposes:

- `camelid chat --agent`, which chooses the full-screen agent TUI on an interactive
  terminal and the inline renderer for `--plain`, pipes, or non-TTY operation;
- `camelid agent exec`, a headless one-shot agent with tri-state exit behavior;
- agent flags for workspace root, steps, tokens, network, filesystem scope, MCP,
  shell mode, and unattended approval posture.

**VERIFIED:** `src/chat/mod.rs::run_chat`:

- ensures or attaches to a local `camelid serve` process;
- loads the requested GGUF;
- requires the active compatibility-ledger row to be `tool_capable` before entering agent
  mode;
- builds `AgentConfig` and dispatches to `agent_tui::run`, `agent::run_agent`, or
  `agent::run_exec`.

An arbitrary GGUF does not gain agent authority merely because the model can produce text.
The exact row must have earned the `tool_capable` capability.

### 3.2 Agent loop

**VERIFIED:** `src/chat/agent.rs::run_loop` is a bounded, model- and UI-agnostic
plan-act-observe loop. Its primary boundaries are:

- `ModelDriver`: obtains the next model step and prompt metrics;
- `Approver`: decides one already-validated gated action;
- `Reporter`: reports model text, tool calls, tool results, notices, context usage, and
  timing;
- `Sandbox`: canonical root and execution policy;
- `ApprovalPolicy`: the single tier decision point for whether an action auto-runs,
  asks, or is denied.

For each model-emitted tool call, the loop:

1. parses the call into a name and JSON arguments;
2. validates it against the active tool profile and sandbox;
3. converts it into a typed `Action`;
4. reports the validated call;
5. obtains the effective approval tier;
6. asks the `Approver` only when that tier is `Confirm`;
7. executes through the audited execution path only after approval;
8. fences the result as untrusted data before replaying it to the model;
9. stops repeated identical calls with identical results and enforces a step cap.

**VERIFIED:** `RunOutcome` maps loop termination to one shared process contract:

- completed -> status `completed`, exit `0`;
- driver failure -> status `failed`, exit `1`;
- abort, step cap, or repeated no-progress -> status `inconclusive`, exit `3`.

### 3.3 Tool risk and approval

**VERIFIED:** `src/chat/tools.rs` classifies actions as `Read`, `Write`, `Exec`,
`Network`, or `Plan`.

Default tiers are:

| Risk | Default tier | Current meaning |
|---|---|---|
| Read | Auto | Run after validation without a prompt |
| Plan | Auto | Changes only the visible process-global plan |
| Write | Confirm | Ask before file mutation |
| Network | Confirm | Ask before Camelid's explicit network tool |
| Exec | Confirm | Ask before process, MCP, subagent, or GUI execution |

`--auto-approve` may promote write/network confirmation but not exec. The stronger
`--today-is-a-good-day-to-die`/`--yolo` may promote exec. Both are refused when
`CAMELID_PRODUCTION` is set.

**VERIFIED:** the `AlwaysTool` interactive decision is session-scoped. Saved sessions record
the historical grant names for display, but resume does not restore their authority.

### 3.4 Native file confinement

**VERIFIED:** `src/chat/tools.rs::Sandbox::resolve`:

- canonicalizes the configured workspace root;
- resolves relative paths against that root;
- canonicalizes existing targets or the parent of a new write target;
- resolves symlinks before checking containment;
- rejects a path that is neither the root nor a descendant;
- widens file-tool access only after explicit `--allow-fs`.

**VERIFIED:** model-driven writes to `.camelid/` state are separately refused. This
prevents a model from forging its saved sessions or checkpoint state through native write/edit
tools.

This is a strong boundary for native file tools. It is not automatically inherited by a
general child process.

### 3.5 Shell boundary: required correction

**VERIFIED:** `src/chat/shell_sandbox.rs` and the original D10 decision make the
following narrower claim:

- `disabled`: shell tools are not registered;
- `unrestricted`: command is working-directory-pinned and wall-clock-timed but otherwise
  unconfined;
- Linux `sandboxed`: supported x86_64/aarch64 builds install seccomp, rlimits, and
  `NO_NEW_PRIVS`; root launches can drop UID/GID and may chroot only when the workspace is a
  usable root filesystem containing `/bin/sh`;
- Windows `sandboxed`: working-directory pin plus wall-clock timeout, with approval as the
  primary backstop; there is no seccomp or UID drop;
- macOS and unsupported platforms fail closed in sandboxed mode.

Consequences:

- On Windows, an approved PowerShell or shell command can access paths outside the
  workspace using the user's OS permissions.
- On typical unprivileged Linux runs, seccomp blocks selected network and privilege
  syscalls, but an ordinary project directory cannot be chrooted. A command may read host
  paths permitted to the user.
- `cwd-pin` is not a filesystem jail.

**BINDING DESIGN:** Remote UI and capability events must describe the enforcement actually
applied. They must never collapse this into “workspace jailed” for shell actions.

### 3.6 Network boundary: required correction

**VERIFIED:** `--allow-net` controls advertisement and validation of Camelid's explicit
`web_search` and `http_fetch` actions.

It is not a universal network firewall:

- a permitted Windows shell command may use the network;
- unrestricted shell may use the network;
- an enabled MCP server is third-party code and may use the network;
- external audit webhooks use the network;
- the proposed remote transport itself necessarily uses the network.

**BINDING DESIGN:** Remote v1 will state “Camelid network tools disabled/enabled” rather than
“all network access disabled.” Shell capability and its platform enforcement are shown
separately.

### 3.7 TUI controller pattern

**VERIFIED:** `src/chat/agent_tui.rs` already proves that execution and presentation can be
separated:

- one `Engine` owns driver, sandbox, configuration, policy, and transcript;
- the loop runs on a background thread;
- a channel-backed reporter emits typed-in-process UI events;
- a channel-backed approver sends a validated action to the redraw loop and waits on a
  reply channel;
- the UI can cancel the turn and resolve a pending approval without executing the action
  itself.

Remote control should generalize this ownership pattern. It should not screen-scrape the
terminal.

### 3.8 Persistence and process-global state

**VERIFIED:** `src/chat/agent_session.rs` stores agent transcripts under
`.camelid/sessions`. Resume:

- replays prior transcript entries as context;
- never re-executes historical actions;
- rebuilds the current system policy instead of trusting a saved system message;
- requires the same model identity and a still-live `tool_capable` capability;
- does not restore approval grants.

**VERIFIED:** some full-agent state is process-global rather than session-owned:

- `src/chat/session.rs::CANCEL` is a process-global atomic;
- `src/chat/plan.rs` stores one process-global plan;
- `src/chat/checkpoint.rs` stores one process-global checkpoint log;
- `src/chat/mcp.rs` stores one process-global MCP registry;
- subagent configuration is process-global.

**BINDING DESIGN:** Remote v1 supports many durable, replayable histories but exactly one active
execution session per Camelid process. Dormant histories cannot accept commands until an explicit,
durable activation transaction makes one authoritative. Parallel execution sessions remain blocked
until the remaining globals are moved into session-owned runtimes and tested for isolation.

### 3.9 Web Workspace is not the remote host

**VERIFIED:** Web Workspace is intentionally read-only and exposes only `read_file`,
`list_dir`, and literal `search`.

**VERIFIED:** `WorkspaceSessionManager` has browser-specific lifecycle semantics:

- one active Workspace session;
- a turn waits for one SSE consumer before execution begins;
- the event stream can be claimed only once;
- failing to claim within 30 seconds aborts the turn;
- dropping the SSE stream requests cancellation;
- the SSE sequence starts at zero for that claimed stream;
- there is no durable replay cursor for a new viewer.

These are valid fail-closed browser semantics and wrong for mobile supervision.

**BINDING DESIGN:** Reuse the agent loop and the concept of an external bridge. Do not
extend `WorkspaceSessionManager` into the remote runtime and do not widen Web Workspace's
tool profile.

## 4. Architecture Overview

The works-anywhere topology has four logical components:

```text
Mobile app
    |  TLS to relay; Noise-encrypted payload remains opaque to relay
    v
Blind relay and generic push gateway
    ^
    |  outbound TLS/WebSocket from the host; no inbound router port required
    |
Camelid remote agent host
    |-- local durable event/session/device store
    |-- local agent runtime and approval authority
    |-- local camelid serve / engine worker
    |-- local GGUF, workspace, tools, checkpoints
```

### 4.1 Component responsibilities

#### Camelid remote agent host

The host:

- is explicitly armed by the local operator;
- selects and pins one canonical workspace;
- selects and validates one exact `tool_capable` model;
- snapshots the enabled capability profile before accepting remote commands;
- owns the agent runtime and all mutable session state;
- validates every model tool call locally;
- persists commands, events, pending approvals, and terminal outcomes locally;
- performs end-to-end pairing and session encryption;
- opens only outbound relay connections in works-anywhere mode;
- supports event replay after mobile or relay disconnects;
- rejects duplicate, stale, unauthorized, and invalid commands;
- sends generic push hints with no sensitive content;
- remains useful with the relay removed by supporting a direct transport later.

#### Blind relay

The relay:

- accepts authenticated host and mobile WebSocket connections;
- routes binary Noise handshake and transport frames by opaque route ID;
- applies strict size, rate, connection, and TTL limits;
- never receives end-to-end private keys;
- never parses inner Camelid protocol messages;
- does not store session transcripts or source data;
- stores only routing/account/device-push metadata needed to operate the service;
- emits generic APNs/FCM wake notifications when presented with a valid notification
  capability;
- exposes no tool, file, shell, model, or agent API of its own;
- fails closed when a route, token, or connection is invalid.

#### Mobile app

The app:

- generates and protects its device key material in Keychain/Keystore-backed storage;
- pairs by scanning a local QR code;
- verifies the host identity pinned in that QR code;
- opens end-to-end encrypted sessions through the relay;
- keeps only a cached view and replay cursor, never authoritative execution state;
- displays the exact session capability snapshot;
- displays validated tool actions and action digests;
- sends idempotent prompts, decisions, and cancellation requests;
- treats push as a hint and reconnects for authoritative state;
- supports app kill, backgrounding, network roaming, and replay;
- provides device-local biometric/app-lock protection before high-impact decisions.

#### Local Camelid engine

The existing local server and engine worker:

- load and execute the GGUF;
- preserve the compatibility and model-transition gates;
- remain the sole inference path;
- are never exposed directly to the relay or phone;
- are called by the local host through the existing audited local client path.

### 4.2 Authority matrix

| State or action | Authoritative owner | Relay authority | Mobile authority |
|---|---|---|---|
| Model weights and execution | Camelid host | None | None |
| Canonical workspace | Camelid host/local operator | None | Read display only |
| Tool capability profile | Camelid host/local operator | None | Read display only |
| Tool validation | Camelid host | None | None |
| Pending approval identity | Camelid host | None | May answer current item |
| Tool execution | Camelid host | None | None |
| Transcript/event sequence | Camelid host | None | Cached projection |
| Cancellation state | Camelid host | None | May request cancellation |
| Device authorization | Camelid host | Routes only | Own-key possession |
| Push delivery token | Relay push subsystem | Delivery only | Registers/revokes token |
| Relay route availability | Relay | Routing only | No content authority |

## 5. Binding Product Decisions

### 5.1 Works-anywhere uses a blind relay

The primary product path uses an optional hosted or self-hosted relay so neither the host
nor phone requires an inbound public port. Direct LAN/VPN transport is a later compatible
transport, not the primary v1 onboarding path.

### 5.2 Host authority never migrates

Disconnecting the phone, killing the app, or restarting the relay does not transfer
execution ownership. The agent host continues until it reaches an approval, finishes,
fails, is locally stopped, or hits its existing bounds.

### 5.3 Many histories, one active execution session in v1

One Camelid process owns one armed remote host, many durable histories, and exactly one active
multi-turn execution session. Dormant histories are replay-only until explicit activation. Session
creation and activation are atomic, command-deduplicated authority changes and are refused while a
turn is unfinished. One turn may run at a time. A prompt submitted while a turn is running returns
`session_busy`; v1 does not queue or steer mid-turn prompts.

Several paired devices may observe, but only the first valid decision for a pending
approval settles it. New prompts are serialized through the command log.

The host-scoped session catalog is revision-pinned, bounded, workspace-scoped, ordered by update
time and stable UUID tie-break, and refreshed after committed event bursts. A phone maintains a
separate replay projection and cursor per history. CLI saved-agent sessions may be listed and
replayed, but legacy saves lacking an exact model artifact digest are not continuable and never
restore historical approval grants.

### 5.4 Local operator chooses dangerous scope

The phone cannot widen capabilities. The local operator chooses model, workspace, shell
mode, and explicit network-tool access when arming the host.

Remote commands cannot enable:

- filesystem access outside the canonical root;
- persistent auto-approval;
- unattended/yolo execution;
- MCP or subagents;
- GUI/system control;
- model lifecycle operations;
- shell if it was not locally armed;
- network tools if they were not locally armed.

### 5.5 Remote approval choices are narrow

Remote v1 offers:

- `allow_once`;
- `deny`;
- `abort_turn`.

It does not offer `always_tool`. Persistent or session-wide grants require a local
operator in v1.

### 5.6 Push is not state transport

Push notifications contain no prompt, filename, command, diff, approval detail, or model
output. The message is equivalent to “Camelid needs attention.” The app reconnects and
decrypts authoritative events before displaying details.

APNs and FCM are best effort. Missing a push cannot lose an approval or event because those
remain in the host's local store.

### 5.7 Host offline means unavailable

The relay does not queue user commands for later host execution in v1. If the host is
offline, the mobile app reports that fact. This avoids delayed execution of an old command
after the security context, workspace, branch, or operator intent has changed.

### 5.8 Remote does not silently attach to a TUI

The remote host is a dedicated headless session owner. It does not inject keystrokes into
an arbitrary existing terminal or scrape terminal output. Local observation/control may be
added through a structured local UI that attaches to the same host event log.

## 6. Capability Profile

### 6.1 Remote-safe v1 profile

**BINDING DESIGN:** Introduce a dedicated profile rather than reusing `Full` or
`WorkspaceReadOnly`.

Default enabled tools:

- `read_file`;
- `list_dir`;
- literal `search`;
- `update_plan`;
- `write_file`;
- `edit_file`.

Local opt-in tools:

- `run_shell`, only when the local operator chooses a shell mode other than `disabled`;
- `web_search` and `http_fetch`, only with local `--allow-net`.

Always unavailable in remote v1:

- `run_windows_command`;
- `inspect_system`;
- `spawn_subagent` and `check_subagent_status`;
- `type_text`, `press_keys`, mouse tools, `ui_inspect`, `ui_click`, and `screenshot`;
- every `mcp__*` tool;
- unrestricted file access.

### 6.2 Capability snapshot

The host records and emits a capability snapshot before accepting a turn:

```json
{
  "workspace": "C:\\work\\project",
  "model_id": "exact-ledger-row",
  "model_artifact_sha256": "sha256:...",
  "tools": ["read_file", "list_dir", "search", "update_plan", "write_file", "edit_file"],
  "file_scope": "canonical_workspace",
  "shell": {
    "enabled": false,
    "mode": "disabled",
    "enforced_layers": [],
    "note": null
  },
  "camelid_network_tools": false,
  "mcp": false,
  "subagents": false,
  "gui_control": false,
  "persistent_approval_grants": false
}
```

The actual wire uses platform-neutral slash-separated paths for display only where possible;
the host continues using canonical native paths internally. The snapshot is descriptive and
cannot grant authority. Host-side configuration remains authoritative.

### 6.3 Approval display requirements

The mobile approval card must show:

- host name and connection state;
- session and workspace display name;
- tool and risk class;
- exact validated target or command detail;
- shell enforcement note when the action is shell-backed;
- action digest, available in a details disclosure;
- turn age and approval age;
- choices allowed by the current protocol.

The card must not reduce an approval to a model-written prose summary. It renders fields
derived from the validated `Action`.

## 7. Session and Turn State Machines

### 7.1 Host connection state

```text
disabled -> enrolling -> connecting -> online
                         |              |
                         v              v
                       errored <---- reconnecting
```

- `disabled`: no relay connection and no remote control.
- `enrolling`: establishing or refreshing the host's relay route.
- `connecting`: outbound WebSocket/TLS connection in progress.
- `online`: relay route available.
- `reconnecting`: bounded exponential reconnect after transport loss.
- `errored`: operator-visible terminal configuration/authentication error.

Remote transport state does not determine agent turn state.

### 7.2 Session state

```text
armed -> idle -> running -> idle
                  |  |
                  |  +-> waiting_approval -> running
                  +----> cancelling -> idle
                  +----> failed
idle/failed -> closed
```

- `armed`: local model/workspace/capability checks passed; no remote turn yet.
- `idle`: accepts one new `start_turn` command.
- `running`: model/tool loop active.
- `waiting_approval`: one validated action is pending.
- `cancelling`: cooperative cancellation requested but not settled.
- `failed`: runtime unavailable until explicit local recovery or safe reset.
- `closed`: terminal session; rejects new commands.

An answered, aborted, step-capped, or repeated turn normally returns a reusable session to
`idle`. A host-level invariant or persistence failure moves it to `failed`.

### 7.3 Turn state

```text
accepted -> started -> running -> completed
                         |  |-> aborted
                         |  |-> step_capped
                         |  |-> repeated
                         |  +-> driver_error
                         +----> waiting_approval -> running
```

Turn outcomes retain Camelid's existing meaning. `step_capped`, `repeated`, and `aborted`
are inconclusive, not definitive failures.

### 7.4 Approval state

```text
pending -> allowed_once
        -> denied
        -> aborted
        -> expired
        -> invalidated_by_cancel
```

Only `pending` accepts a decision. An approval is scoped to:

- `session_id`;
- `turn_id`;
- `call_id`;
- `approval_id`;
- `action_digest`.

Every field must match. First valid settlement wins atomically. Duplicates return the
stored command result; conflicting or late decisions return `stale_approval`.

### 7.5 Approval timeout

The current bridge default is five minutes. Remote v1 keeps a five-minute host-side timeout
unless measured usability evidence justifies a change. Timeout aborts the turn and records an
`approval.expired` event. It never auto-allows.

The mobile countdown is informational; the host clock and state are authoritative.

## 8. Durable Local State

### 8.1 Location

Remote-control state is system-owned application data, not repository content. Proposed
default database:

- Windows: `%LOCALAPPDATA%\camelid\remote-control.sqlite3`;
- Linux: `$XDG_DATA_HOME/camelid/remote-control.sqlite3` with standard fallback;
- macOS: the normal Camelid application-data directory;
- tests: `CAMELID_REMOTE_DB` may select an isolated path.

It must not be stored under the controlled workspace. Agent file tools cannot mutate it.

### 8.2 Database ownership

**BINDING DESIGN:** Add a separate SQLite module instead of overloading Workspace memory.
The store uses bundled SQLite through the existing `rusqlite` dependency, WAL mode, foreign
keys, immediate write transactions, a busy timeout, explicit schema versioning, and
fail-closed handling for unknown newer schemas.

### 8.3 Proposed schema v1

Logical schema; exact SQL lands with migrations and tests.

#### `remote_meta`

| Column | Purpose |
|---|---|
| `schema_version` | Exact database schema version |
| `host_id` | Stable random host identifier |
| `host_noise_public` | Host static Noise public key |
| `encrypted_host_noise_secret` | Host secret protected by OS credential storage, not plaintext DB |
| `created_at` | Unix milliseconds |

Private host keys should live in Windows Credential Manager, macOS Keychain, or a
permission-restricted secret file on Linux. The database stores a reference when the platform
store supports one. Exact platform adapters are a security gate, not an excuse to put an
unencrypted private key in SQLite.

#### `remote_devices`

| Column | Purpose |
|---|---|
| `device_id` | Host-assigned stable UUID |
| `label` | User-visible device label, bounded |
| `noise_static_public` | Authorized mobile static key |
| `created_at` | Pairing time |
| `last_seen_at` | Last authenticated connection |
| `revoked_at` | Null while authorized |
| `push_capability_id` | Opaque relay notification capability reference |

Device revocation is local and immediate for new handshakes. An already-open encrypted
connection is closed when its device is revoked.

#### `remote_sessions`

| Column | Purpose |
|---|---|
| `session_id` | UUID |
| `canonical_root` | Host-native canonical workspace |
| `model_id` | Exact compatibility row/model identity |
| `model_sha256` | Exact GGUF artifact hash when available from the loaded row |
| `capability_snapshot_json` | Informational immutable session snapshot |
| `state` | Session state token |
| `transcript_json` | Serialized `AgentMsg` transcript after settled boundaries |
| `plan_json` | Session-owned plan after runtime refactor |
| `created_at`, `updated_at`, `closed_at` | Lifecycle timestamps |
| `next_event_sequence` | Atomic next sequence allocator |

No approval grants are persisted as authority.

#### `remote_turns`

| Column | Purpose |
|---|---|
| `turn_id` | UUID |
| `session_id` | Owning session |
| `command_id` | Idempotent mobile command |
| `user_text` | Local plaintext user prompt |
| `state`, `outcome` | Lifecycle and terminal outcome |
| `assistant_text` | Final complete answer only |
| `created_at`, `started_at`, `finished_at` | Timestamps |

Partial model deltas are events, not final assistant text.

#### `remote_events`

| Column | Purpose |
|---|---|
| `session_id`, `sequence` | Composite primary key and replay cursor |
| `event_id` | UUID for diagnostics |
| `turn_id` | Nullable session-level event scope |
| `event_type` | Stable protocol token |
| `payload_json` | Local plaintext structured payload |
| `created_at` | Unix milliseconds |

Event append and sequence allocation happen in one transaction. Broadcast occurs only after
commit. A live subscriber can miss a broadcast but recover from the committed sequence.

#### `remote_commands`

| Column | Purpose |
|---|---|
| `device_id`, `command_id` | Unique idempotency key |
| `session_id` | Target session |
| `command_type` | `start_turn`, `approval_decision`, or `cancel_turn` |
| `request_digest` | Detect conflicting reuse of an ID |
| `status` | accepted, applied, rejected |
| `response_json` | Stable response returned to retries |
| `created_at`, `finished_at` | Timestamps |

Reusing a command ID with a different request digest is a protocol violation, not a retry.

#### `remote_approvals`

| Column | Purpose |
|---|---|
| `approval_id` | UUID primary key |
| `session_id`, `turn_id`, `call_id` | Scope |
| `action_digest` | Digest over canonical validated approval record |
| `tool`, `risk`, `detail` | Local authoritative display data |
| `state`, `decision` | Settlement |
| `decided_by_device` | Nullable for local/timeout settlement |
| `created_at`, `settled_at` | Timestamps |

The insert of a pending approval and its `approval.required` event is atomic before the
agent waits.

### 8.4 Retention

Initial conservative policy:

- device grants remain until revoked;
- open session transcript and event history remain until explicit local deletion;
- closed sessions default to 30-day retention, configurable locally;
- command deduplication rows remain at least as long as their session;
- no relay transcript retention exists because the relay never receives plaintext events.

Retention must be implemented with explicit local controls and tests before automatic deletion
is enabled. v1 may retain until explicit deletion if automatic pruning cannot be proven safe.

### 8.5 Event coalescing

Persisting one row per generated token would create unnecessary write load. The host coalesces
`model.delta` text into bounded chunks using both:

- a maximum UTF-8 byte size, proposed 8 KiB;
- a short maximum flush interval, proposed 50 ms.

The final `model.answer` carries the authoritative complete answer. Delta loss must never
change the final transcript. Coalescing constants require a load/replay test before they become
binding implementation constants.

## 9. End-to-End Protocol

### 9.1 Layering

There are three layers:

1. Relay routing envelope: minimal metadata the relay needs.
2. Noise handshake/transport frame: end-to-end encrypted between mobile and host.
3. Camelid remote message: typed JSON or CBOR inside the encrypted transport.

**BINDING DESIGN:** v1 inner messages use canonical JSON for inspectability and test fixtures.
Binary attachments do not exist in v1. A future binary encoding requires a protocol-version
change or negotiated capability.

Phase 0 pins Noise's 65,535-byte record maximum. Each decrypted transport record starts with a
fixed 64-byte encrypted chunk header containing protocol version, message UUID, index/count,
total length, and whole-message SHA-256. At most 18 strictly ordered chunks reconstruct one inner
message of at most 1,114,112 bytes. Duplicate, omitted, reordered, cross-message, overlong, or
digest-mismatched chunks close the connection. A turn prompt remains bounded to 4 KiB of UTF-8,
replay batches to 256 events, and a canonical approval record to 1 MiB. Oversized approvals fail
closed rather than asking the user to approve truncated content. These constants are versioned in
the fixture manifest and may change only with compatible negotiation or a protocol-version change.

### 9.2 Relay-visible envelope

Conceptual shape:

```json
{
  "relay_protocol": "camelid.relay/v1",
  "route_id": "opaque-128-bit-id",
  "connection_id": "relay-generated",
  "frame_type": "handshake",
  "payload_base64": "opaque-bytes"
}
```

The relay validates only envelope size, route, connection authorization, and rate limits. It
does not inspect or log payload bytes beyond operational byte counts.

Production transport should use binary WebSocket frames rather than base64 JSON; the JSON above
defines semantics, not an inefficient required encoding.

### 9.3 Inner message envelope

Every decrypted message has:

```json
{
  "protocol": "camelid.remote/v1",
  "message_id": "uuid",
  "kind": "command|command_result|event_batch|replay_request|replay_end|ping|pong|error",
  "host_id": "uuid",
  "device_id": "uuid",
  "session_id": "uuid-or-null",
  "sent_at_unix_ms": 1780000000000,
  "payload": {}
}
```

The Noise channel authenticates and orders frames for one connection. Application IDs still
exist for idempotency, replay diagnostics, and durable command semantics.

**BINDING DESIGN:** Inner-envelope fields are additive within v1, but `kind` remains a closed
vocabulary. Unknown event names advance the replay cursor without privileged side effects.
Privilege-bearing command payloads reject unknown fields and carry an explicit `command`
discriminator. This lets old clients tolerate new observational metadata without letting an old
host silently ignore part of a command that may affect authority.

### 9.4 Commands

#### `start_turn`

```json
{
  "command": "start_turn",
  "command_id": "client-uuid",
  "turn_id": "client-uuid",
  "text": "Run the focused tests and fix the failure"
}
```

Rules:

- session must be `armed`, `idle`, or reusable after a prior terminal turn;
- no turn may already be active;
- text must be non-empty valid UTF-8 and bounded to 4 KiB in v1;
- command and turn IDs must be valid UUIDs;
- duplicate command ID with identical request returns the stored result;
- duplicate ID with different content returns `idempotency_conflict`;
- host records `user.message` before starting the worker;
- no queued or steering semantics in v1.

#### `approval_decision`

```json
{
  "command": "approval_decision",
  "command_id": "client-uuid",
  "turn_id": "uuid",
  "call_id": "uuid",
  "approval_id": "uuid",
  "action_digest": "sha256:hex",
  "decision": "allow_once|deny|abort_turn"
}
```

Rules:

- all scope fields and digest must match the current pending row;
- settlement uses an atomic conditional update from `pending`;
- `allow_once` resolves the in-memory approver only after durable settlement succeeds;
- `deny` returns a denied tool result to the model;
- `abort_turn` settles the approval and sets cancellation;
- replayed, stale, or conflicting decisions cannot execute an action.

#### `cancel_turn`

```json
{
  "command": "cancel_turn",
  "command_id": "client-uuid",
  "turn_id": "uuid"
}
```

Rules:

- idempotent;
- sets the session-owned cancellation token;
- invalidates a pending approval before releasing its waiter;
- returns `accepted` while cancellation is cooperative;
- terminal truth arrives through `turn.finished` and session status events;
- does not claim immediate cancellation of a child process until process-tree teardown is
  observed.

### 9.5 Command result

```json
{
  "command_id": "client-uuid",
  "status": "accepted|applied|rejected",
  "code": "ok|session_busy|stale_approval|idempotency_conflict|capability_denied|...",
  "message": "bounded human-readable text",
  "current_event_sequence": 42
}
```

Machine logic branches on `code`, never the human message.

### 9.6 Events

Every event is durable before broadcast:

```json
{
  "sequence": 42,
  "event_id": "uuid",
  "turn_id": "uuid-or-null",
  "event": "approval.required",
  "created_at_unix_ms": 1780000000000,
  "payload": {}
}
```

Required v1 event vocabulary:

- `host.capabilities`;
- `session.armed`;
- `session.state_changed`;
- `turn.accepted`;
- `turn.started`;
- `user.message`;
- `model.delta`;
- `model.timing`;
- `model.answer`;
- `plan.updated`;
- `tool.call`;
- `approval.required`;
- `approval.settled`;
- `approval.expired`;
- `tool.result`;
- `session.notice`;
- `turn.finished`;
- `session.error`;
- `device.revoked`.

Unknown event types are ignored by clients after retaining their sequence position. New fields
are additive. A client must never interpret an unknown event as approval or success.

### 9.7 Tool call event

`tool.call` is emitted after validation, before approval/execution:

```json
{
  "call_id": "uuid",
  "tool": "edit_file",
  "risk": "write",
  "approval_tier": "confirm",
  "detail": "edit_file -> src/lib.rs\n  - old\n  + new",
  "action_digest": "sha256:hex"
}
```

Raw model prose does not populate this event. Invalid calls emit a tool-error result with no
action approval.

### 9.8 Approval record and digest

The current audit digest covers raw model arguments. Remote approval needs a separate digest
over the validated action the host intends to execute.

**BINDING DESIGN:** Add a serializable `ApprovalRecord` produced from `Action` after sandbox
resolution. It includes:

- schema identifier `camelid.approval-record/v1`;
- tool name and risk;
- canonical or workspace-relative resolved target as appropriate;
- exact command/method/URL or bounded write/edit representation;
- execution timeout and shell enforcement where applicable;
- stable action-specific fields in lexicographic JSON object order.

`action_digest = SHA-256(canonical_json(ApprovalRecord))`, tagged `sha256:`.

For write content, the complete content must remain available to the encrypted mobile approval
view. The compact card may show a bounded preview, but an approval cannot silently omit the rest;
the user must be able to expand the complete proposed content before allowing it. Protocol size
limits may reject an excessively large approval rather than approve on a truncated payload.

### 9.9 Replay

After every connection or app foreground transition, mobile sends:

```json
{
  "kind": "replay_request",
  "session_id": "uuid",
  "after_sequence": 42,
  "limit": 256
}
```

The host returns ordered `event_batch` messages and a final:

```json
{
  "kind": "replay_end",
  "session_id": "uuid",
  "last_sequence": 96,
  "has_more": false,
  "session_state": "waiting_approval"
}
```

Rules:

- sequence is allocated by the host database, not the connection;
- batches are capped by event count and encrypted frame bytes;
- client applies only strictly increasing sequence numbers;
- duplicate events are ignored;
- a gap triggers another replay request, never speculative UI state;
- live broadcasts may race replay, so the client buffers them and applies by sequence after
  replay catches up;
- approval cards are reconstructed from authoritative pending/settled events;
- replay suppresses notifications, sounds, command auto-send, and other live-only side effects.

### 9.10 Status snapshot

Replay end carries current session state. A separate read request may obtain a bounded snapshot
for diagnostics, but snapshots never replace the event log for history.

## 10. Cryptography and Pairing

### 10.1 Goals

The cryptographic design must provide:

- relay-blind confidentiality and integrity;
- mutual host/device authentication after pairing;
- QR-pinned host identity during first pairing;
- forward secrecy for transport connections;
- revocable per-device grants;
- replay-resistant ordered transport;
- no shared static API key copied between devices;
- interoperable Rust and iOS/Android implementations;
- test vectors and an independent security review before promotion.

### 10.2 Noise construction

**BINDING DESIGN (amended 2026-07-24):** Use one pinned Rust Noise implementation for the
host and both mobile native modules. The selected development dependency is `snow` 0.10.0,
compiled with only Curve25519, ChaChaPoly, BLAKE2, and OS randomness.
The suite remains:

```text
Noise_IK_25519_ChaChaPoly_BLAKE2s
```

The shared core is deliberate. The Phase 0 ecosystem review found no maintained Swift/Kotlin
pair for this exact suite: current Swift libraries omit BLAKE2s, the Kotlin candidate omits IK
and is explicitly unaudited, and the Java reference implementation supports the suite but has
no release and has been inactive for years. Changing the hash solely to match weak mobile
packages would not improve the security boundary. Thin Swift and Kotlin bindings therefore
own no handshake logic and call the same reviewed Rust state machine used by the host.

`snow` is maintained, tracks Noise revision 34, supports the exact suite, and is widely used,
but its own documentation states that it has not received a formal audit. It is therefore a
candidate, not security evidence. Remote control cannot be promoted beyond development until
the shared wrapper, dependency configuration, protocol use, key handling, and authorization
boundary receive the independent review required by Phase 5. Dependency findings and pins are
recorded in `docs/architecture/REMOTE_AGENT_CONTROL_CRYPTO_REVIEW.md`.

Why IK fits this topology:

- the mobile initiator knows the host static public key from the QR code;
- the initiator's static key is encrypted in the first handshake message;
- the host learns and authenticates the device key during pairing;
- later connections authenticate both existing static identities;
- Noise supplies transcript-bound key derivation, authenticated encryption, ordering, and
  forward-secret ephemeral contributions.

**GATE:** Before production code depends on the shared core, add standalone Swift and Kotlin
binding tests that exchange a complete IK handshake and transport messages with the Rust host,
prove transcript identity, reconnect with fresh ephemeral keys, exercise explicit rekey, and
reject tampering and a wrong host key. Bindings are not accepted merely because they compile.

### 10.3 Host key material

On first remote setup, the host creates one Curve25519 Noise static keypair using the OS CSPRNG.
The private key is protected by the platform credential store where available. It is never sent
to the relay, embedded in a QR code, logged, or exported by ordinary diagnostics.

Rotating the host static key revokes every paired device and route binding. The UI must state
that consequence before rotation.

### 10.4 Device key material

The phone creates one static keypair per Camelid host. Private key storage must be backed by:

- iOS Keychain with device-only accessibility and biometric/app-lock policy where supported;
- Android Keystore-backed protection or an encrypted key wrapper whose key is Keystore-backed.

The app must not put private keys in AsyncStorage, unencrypted SQLite, logs, crash reports, or
cloud backup.

### 10.5 Pairing QR

The QR payload contains no private key and is valid for one short-lived pairing:

```json
{
  "v": 1,
  "relay_url": "wss://relay.example.invalid/v1/connect",
  "route_id": "base64url-128-bit-random",
  "host_id": "uuid",
  "host_noise_public": "base64url-32-bytes",
  "pairing_secret": "base64url-128-bit-random",
  "expires_at_unix_ms": 1780000300000
}
```

`example.invalid` is intentionally not a production-domain assumption. Hosted deployment URL is
**OPEN EXTERNAL**.

Pairing constraints:

- QR creation requires explicit local action;
- expiry is proposed at five minutes;
- pairing secret is single-use and invalidated after success, timeout, cancel, or host restart;
- only one active pairing secret exists per host in v1;
- the host public key pins the responder and defeats relay MITM;
- possession of a photographed live QR can authorize a device until expiry, so the local UI
  shows pairing progress and device label before final acceptance;
- local operator can require an explicit final confirmation displaying both device label and a
  short authentication fingerprint.
- the secret-bearing QR payload is returned only by an explicit same-origin local action with
  `Cache-Control: no-store`; it is absent from status polling and retained by the Web UI only in
  component memory;
- an active offer is not silently replaced: the operator cancels it or waits for expiry;
- relay reconnect invalidates an awaiting connection-bound confirmation, while an unclaimed offer
  remains valid only until its original expiry;
- if durable registration succeeds but encrypted response delivery fails, the grant remains and
  the device must reconnect through the authorized-device path; pairing is not repeated to create
  a second grant.

### 10.6 Initial pairing flow

1. Host opens an authenticated relay route and displays the QR.
2. Mobile scans and validates version, expiry, key lengths, and URL scheme.
3. Mobile creates a fresh static device keypair.
4. Relay connects mobile's opaque stream to the host route.
5. Mobile initiates Noise IK using the QR-pinned host public key.
6. Inside the encrypted channel, mobile sends `pair.request` with the one-time secret, bounded
   device label, app protocol version, and supported capabilities.
7. Host compares the secret in constant time, checks expiry and unused state, and shows the
   local confirmation.
8. Local operator confirms. Host persists the device public key and appends a local device event.
9. Host responds with assigned `device_id`, host metadata, and a relay push-registration
   capability.
10. Pairing secret is destroyed and cannot be reused.

If local confirmation is omitted in the earliest CLI prototype, that prototype is development
only and cannot be promoted as safe phone pairing.

### 10.7 Reconnection flow

1. Mobile reconnects to the route using relay routing credentials.
2. A new Noise IK handshake uses the same authorized static keys and fresh ephemeral keys.
3. Host rejects unrecognized or revoked initiator static keys.
4. Mobile verifies the same host public key it paired with.
5. Both sides enter transport mode and exchange inner protocol messages.
6. Mobile requests replay after its last committed sequence.

### 10.8 Relay authentication versus content authentication

Relay routing credentials are random capabilities used only to prevent arbitrary route abuse.
They are not content encryption keys and cannot authorize a Camelid command. Compromise of a
relay token may permit connection attempts or generic push spam, but Noise authentication and
host command validation still reject forged content.

### 10.9 Rekey and limits

Noise transport must rekey before library-specified nonce exhaustion and under a conservative
message/byte threshold. Exact thresholds follow the selected audited implementation and are
pinned by tests; they are not guessed in code comments.

Frames have strict maximum ciphertext size:

- 65,535 bytes per Noise record;
- replay and large inner messages split across authenticated ordered chunks;
- complete write approval may span chunks but retains one canonical whole-record digest;
- malformed, oversized, duplicate, omitted, cross-message, or out-of-order chunks close the
  connection.

### 10.10 Cryptographic non-claims

E2EE does not hide:

- relay account and route metadata;
- client IP addresses from the relay;
- connection timing and frame sizes;
- push token from APNs/FCM and the push gateway;
- plaintext from the local Camelid process or paired phone;
- data the local LLM provider would receive (Camelid uses the local model in this design);
- a compromised host or compromised phone.

## 11. Relay Design

### 11.1 Deployment shape

The relay is a separate service/crate and does not run inside the local inference API. It can
be hosted by the project or self-hosted with the same wire protocol.

**OPEN EXTERNAL:** production operator, domain, account model, region, abuse policy, and funding
are not determined by the repository and must not be fabricated in code or docs.

### 11.2 Relay API surface

Proposed minimal surface:

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/v1/routes/enroll` | Host obtains/refreshes an opaque route capability |
| `GET` upgrade | `/v1/routes/:route_id/host` | Host outbound WebSocket |
| `GET` upgrade | `/v1/routes/:route_id/device` | Device WebSocket routed to host |
| `POST` | `/v1/push/register` | Device registers push token using notification capability |
| `POST` | `/v1/push/notify` | Host requests generic wake hint |
| `DELETE` | `/v1/push/:capability` | Revoke push registration |
| `GET` | `/healthz` | Non-sensitive service health |

The relay API has its own version and typed errors. It never exposes Camelid agent routes.

### 11.3 Routing

- route IDs are at least 128 bits of CSPRNG entropy;
- host route capability and device routing capabilities are separate;
- one host connection per route in v1;
- multiple authorized device connections may be forwarded, subject to a small cap;
- unknown routes return indistinguishable not-found/unauthorized behavior where practical;
- relay forwards binary frames without parsing Noise or Camelid payloads;
- no command is persisted for an offline host;
- if host is absent, device receives an explicit relay-level unavailable status;
- backpressure is bounded; a slow connection is disconnected rather than buffered without limit.

### 11.4 Relay metadata

Allowed operational metadata:

- account/route identifiers;
- connection timestamps and byte counters;
- rate-limit counters;
- coarse error classes;
- push token and platform needed for delivery;
- capability revocation and expiry;
- abuse/security audit records that contain no ciphertext payload dump.

Forbidden logs/storage:

- Noise payload bytes;
- decrypted application messages (not available by design);
- QR pairing secret;
- host or device private keys;
- prompts, responses, filenames, commands, diffs, or tool outputs;
- full authorization headers or bearer capabilities.

### 11.5 Rate and resource limits

The implementation must define and test bounded:

- connections per account, route, and source IP;
- frame bytes and frames per second;
- handshake duration;
- idle timeout and keepalive;
- concurrent devices per host;
- push requests per device and time window;
- route lifetime and credential refresh;
- global memory per socket and total process backpressure.

Default numbers are selected from load testing, not invented in this design document.

### 11.6 Push gateway

The relay stores platform push tokens because APNs/FCM require them. The host receives a random
notification capability after pairing. Presenting that capability may request only a fixed
generic notification category:

- `approval_required`;
- `turn_finished`;
- `host_attention`.

The relay maps each category to generic copy. It accepts no caller-supplied notification body.
This prevents encrypted content from leaking through push payloads or lock-screen previews.

Push acknowledgements do not settle approvals. The app reconnects and reads host state.

### 11.7 Relay availability

Relay outage makes remote transport unavailable but does not stop local execution. The host
uses bounded exponential reconnect with jitter. It surfaces status locally and records no false
“online” state.

Development hosts require an explicit WebSocket ping interval through
`--relay-keepalive-ms`/`CAMELID_REMOTE_KEEPALIVE_MS`. A supported production default remains gated
on relay load and idle-timeout testing. Ping failure uses the same bounded reconnect path and
invalidates connection-bound Noise/pairing state before devices establish a fresh session.
Relay operators likewise provide `CAMELID_RELAY_KEEPALIVE_MS`. The relay sends Ping frames to both
host and device sockets so React Native clients remain live without needing a control-frame API.

## 12. Host Runtime Design

### 12.1 New command

Proposed CLI:

```text
camelid agent host \
  --model <exact-tool-capable.gguf> \
  --workdir <canonical-project> \
  --relay <wss/https relay base> \
  --shell-sandbox disabled \
  --reconnect-initial-ms <measured-value> \
  --reconnect-max-ms <measured-value> \
  --reconnect-jitter-percent <0..50> \
  [--allow-net]
```

Binding defaults:

- shell disabled;
- network tools disabled;
- file scope confined;
- MCP off;
- subagents off;
- GUI/system control off;
- one active execution session;
- no unattended approval mode.

Exact flag spelling may change during CLI review, but a remote host must be a first-class
subcommand rather than hidden environment magic.

The development CLI uses `--relay-url` and requires the three reconnect values above (also
available through `CAMELID_REMOTE_RECONNECT_*`). It deliberately has no production reconnect
defaults before load testing. Initial pairing renders a scannable terminal QR; the one-time secret
is not also printed as plaintext. Existing sessions restore the exact session ID and protected
route bearer. A configured relay may restore opaque route credentials from a mode-`0600` Unix
state file selected by `CAMELID_RELAY_STATE`; configured persistent relay state fails closed on
platforms without that permission adapter.

### 12.2 Runtime ownership refactor

Introduce a session-owned runtime before remote execution:

```rust
pub struct AgentRuntime {
    pub cancel: Arc<AtomicBool>,
    pub plan: PlanState,
    pub checkpoints: CheckpointStore,
    pub transcript: Vec<AgentMsg>,
    pub policy: ApprovalPolicy,
    pub sandbox: Sandbox,
    pub config: AgentConfig,
}
```

The exact fields may use interior synchronization, but ownership must be per runtime.

Refactor targets:

- replace direct `session::CANCEL` use in the remote lane with runtime cancellation;
- make plan state injectable instead of process-global;
- make checkpoint log injectable instead of process-global;
- keep MCP and subagents unavailable in v1 rather than pretending their process-global state
  is isolated;
- preserve local CLI/TUI behavior through adapters and regression tests.

Do not perform a broad parallel-execution rewrite merely to ship v1. Extract only the state remote
v1 uses, retain one-active-session enforcement, and leave MCP/subagent isolation as a later gate.

### 12.3 Structured reporter

The current `Reporter` interface loses identifiers and structured action fields. Add a structured
event boundary without making UI text the protocol.

Proposed direction:

```rust
pub trait AgentEventSink {
    fn emit(&mut self, event: AgentEvent) -> Result<(), EventSinkError>;
}

pub enum AgentEvent {
    ModelDelta { content: String },
    ModelAnswer { content: String },
    PlanUpdated { steps: Vec<Step> },
    ToolCall { record: ToolCallRecord },
    ApprovalRequired { record: ApprovalRecord },
    ApprovalSettled { record: ApprovalSettlement },
    ToolResult { record: ToolResultRecord },
    Notice { content: String },
    Timing { metrics: ModelStepMetrics },
}
```

Existing `Reporter` implementations can be adapters during migration. The remote protocol never
parses `call_line`, approval prose, or terminal rendering back into data.

### 12.4 Remote approver

The remote approver:

1. receives a validated `Action` and generates `call_id`/`approval_id`;
2. derives `ApprovalRecord` and action digest;
3. atomically persists pending approval plus event;
4. broadcasts the committed sequence;
5. requests generic push;
6. waits on a bounded session-owned decision channel while observing cancellation and timeout;
7. returns only after durable settlement;
8. fails closed on database, channel, device, or state mismatch.

Unlike `WorkspaceApprover`, subscriber disconnect does not abort the turn. Approval timeout or an
explicit cancel does.

### 12.5 Event store and hub

`RemoteEventStore` owns SQLite append/replay. `RemoteEventHub` owns best-effort live notification
to current encrypted connections.

Order:

```text
runtime event -> SQLite transaction/sequence commit -> live hub broadcast -> mobile
```

Never broadcast first. A crash after commit is replayable; a crash before commit never produces
an authoritative event.

### 12.6 Command processor

`RemoteCommandProcessor` is the sole consumer of decrypted mobile commands. It:

- validates protocol/version/schema/size;
- binds the Noise initiator key to a non-revoked device;
- begins or retrieves the idempotency row;
- checks session and turn state;
- invokes the session owner through typed methods;
- commits a stable command result;
- sends that result to the caller;
- never executes `Action` directly.

### 12.7 Host restart

v1 restart contract:

- paired devices and closed/idle history remain locally stored;
- an in-flight turn is marked interrupted/inconclusive during recovery;
- pending approvals become invalidated and cannot be answered after restart;
- no tool call resumes automatically;
- the local operator must re-arm the model/workspace host;
- mobile may replay history after the host returns;
- continuing an old transcript requires the existing model identity and live capability checks;
- approval grants remain absent.

Automatic resurrection of an in-flight tool call is explicitly rejected.

### 12.8 Local UI

The first implementation uses CLI output and a fake mobile client. Product integration later adds
a Remote section to the existing same-origin Web/Tauri UI with:

- enable/disable remote host;
- relay/direct connection status;
- QR pairing and expiry;
- pending local confirmation;
- paired device list with last seen and revoke;
- exact session capability snapshot;
- active session/turn state;
- local stop and emergency disable;
- retention/delete controls;
- honest shell enforcement text.

The local UI never displays relay connectivity as proof that the agent is healthy.

## 13. Mobile Application Design

### 13.1 Technology decision

**BINDING DESIGN:** Target both iOS and Android from one React Native application, with native
modules for secure key storage, push notifications, QR scanning, and the selected Noise
implementation. A mobile web page is not the v1 security surface because browser storage,
background lifecycle, push behavior, and durable device-key protection differ materially.

Expo may be used only with a development build/custom native modules; an assumption that stock
Expo Go supplies the required cryptography or key protection is not acceptable. Final framework
and library versions are selected and pinned during the cross-language crypto gate.

### 13.2 Screens

#### Host list

- paired hosts and honest online/offline/connecting status;
- active session state and attention indicator;
- no source content in OS-level widgets by default;
- revoke/remove local pairing action.

#### Pair host

- QR scanner;
- parsed host/relay/fingerprint preview;
- expiry and invalid-payload errors;
- optional manual code is deferred until it can preserve host-key pinning securely;
- local device label before pairing confirmation.

#### Session

- ordered user/assistant transcript reconstructed from replay;
- streaming model answer;
- plan state;
- structured tool cards and bounded results;
- running/waiting/cancelling/terminal status;
- composer only while session accepts a turn;
- Stop command with settlement status;
- reconnect and replay progress rather than an empty transcript.

#### Approval

- exact validated action detail;
- complete write/edit content available before allowing;
- action age and expiration;
- Allow Once, Deny, Abort Turn;
- biometric/app-lock check before Allow Once when configured;
- disabled controls after settlement or disconnect until state refresh;
- no optimistic “approved” state before host command result/event.

#### Security/device settings

- host fingerprint;
- local key-protection status;
- push enablement;
- app lock/biometric requirement;
- clear local cached transcript without revoking host grant;
- revoke device through host when connected;
- clear pairing locally.

### 13.3 Mobile state model

The app stores:

- host public identity and route metadata;
- device secret key in secure storage;
- host-assigned device ID;
- last applied event sequence per session;
- bounded encrypted-at-rest or OS-protected display cache if implemented;
- push token registration state;
- non-sensitive UI preferences.

The app does not store:

- host model credentials;
- filesystem credentials;
- approval authority independent of its device key;
- a queue of commands to execute when the host later appears.

### 13.4 Reconciliation reducer

One pure reducer applies events by sequence. Replay and live events use the same path. Side
effects are outside the reducer and suppressed during replay.

Required properties:

- ignore duplicate sequence;
- reject/apply no event across a gap until replay fills it;
- preserve settled approvals as non-actionable cards;
- final `model.answer` replaces/coalesces streaming projection for the turn;
- terminal event wins over late deltas;
- cancellation remains pending until authoritative state settles;
- unknown events advance sequence but do not mutate privileged state;
- app relaunch reconstructs from host replay, not optimistic local bubbles.

### 13.5 Mobile lifecycle

- entering background is a transport boundary;
- WebSocket continuity is not assumed on iOS or Android;
- foreground opens a fresh Noise session and requests replay;
- push wakes the user's attention, not a permanent background socket;
- exponential reconnect is bounded and stops when the OS suspends the app;
- network switches cannot duplicate commands because all commands are idempotent;
- app kill loses only transient UI, not host state.

### 13.6 Accessibility and safety

- approval actions have text labels and screen-reader descriptions;
- risk is not communicated by color alone;
- long commands/content wrap and can be selected/copied;
- workspace and host identity remain visible on approval screens;
- destructive-looking approvals are not reduced to lock-screen actions in v1;
- notification actions cannot approve tools without opening the authenticated app.

## 14. Repository Code Design

This is the proposed ownership map. Exact filenames may be adjusted to fit code review, but
responsibilities must not collapse back into `api/mod.rs` or `workspace.rs`.

### 14.1 Core Camelid additions

```text
src/chat/agent_runtime.rs
    Session-owned cancel/plan/checkpoint/transcript/policy state.

src/chat/agent_events.rs
    Structured AgentEvent, ToolCallRecord, ApprovalRecord, settlement records.

src/chat/remote_host.rs
    One-session host lifecycle and worker ownership.

src/chat/remote_approver.rs
    Durable approval creation/wait/settlement adapter.

src/chat/remote_store.rs
    SQLite schema, migrations, transactions, command dedupe, replay.

src/chat/remote_protocol.rs
    Inner v1 message/event schemas and strict validation.

src/chat/remote_crypto.rs
    Noise adapter only; no business state.

src/chat/remote_transport.rs
    Outbound relay WebSocket, reconnect, frame bounds, keepalive.

src/chat/remote_pairing.rs
    QR payload, one-time secret state, local confirmation, device registry.

src/chat/remote_push.rs
    Generic push capability registration/request.
```

### 14.2 Existing files expected to change

```text
src/main.rs
    Add `agent host` CLI and explicit remote flags.

src/chat/mod.rs
    Register modules and dispatch host command.

src/chat/agent.rs
    Emit structured validated events and accept session-owned runtime seams.

src/chat/tools.rs
    Add remote-safe tool policy and canonical ApprovalRecord conversion.

src/chat/agent_tui.rs
src/chat/agent.rs (inline)
    Adapt existing renderers to structured event/runtime seams without behavior drift.

src/api/mod.rs or a narrow src/api/remote.rs
    Loopback-only local management status/pairing UI routes. No relay-facing agent API.

frontend/src/...
    Local Remote settings/operations view after protocol proof.

DOCS.md
DECISIONS.md
COMPATIBILITY.md
STATUS.md
ledger/camelid-ledger.json
    Updated only at the appropriate implementation/promotion gates.
```

### 14.3 Relay service

Proposed separate workspace member or separate repository:

```text
services/camelid-relay/
    Cargo.toml
    src/main.rs
    src/routes.rs
    src/auth.rs
    src/router.rs
    src/push.rs
    src/limits.rs
    src/metrics.rs
```

It must not depend on Camelid inference or chat modules. Sharing only relay wire types is
acceptable through a tiny schema crate if that does not expose inner protocol plaintext.

### 14.4 Mobile app

Proposed separate package or repository:

```text
mobile/
    app/ or src/screens/
    src/crypto/
    src/protocol/
    src/sync/
    src/storage/
    src/push/
    src/components/approval/
    native/ secure-store and Noise adapters as required
```

Whether mobile lives in this monorepo is **OPEN EXTERNAL** until release ownership and CI cost
are agreed. Protocol fixtures must be shared regardless of repository location.

### 14.5 Dependency policy

New dependencies require an explicit audit for:

- maintained release and compatible license;
- Rust 1.89 compatibility or an intentional toolchain decision;
- mobile platform support and cross-language interoperability;
- no telemetry or hidden hosted service;
- supply-chain footprint and transitive native code;
- constant-time/security posture where relevant;
- reproducible lockfiles and advisory scanning.

Likely capability needs include WebSocket client/server, futures utilities, Noise, secure random,
base64url QR representation, QR generation/scanning, and platform secure storage. This document
does not fabricate final crate/package versions before the dependency gate.

## 15. Error Contract

All inner protocol errors have stable machine codes and bounded human messages. Proposed v1
codes:

- `unsupported_protocol`;
- `invalid_message`;
- `message_too_large`;
- `device_not_authorized`;
- `device_revoked`;
- `host_not_armed`;
- `session_not_found`;
- `session_busy`;
- `session_closed`;
- `turn_not_found`;
- `turn_not_active`;
- `capability_denied`;
- `stale_approval`;
- `approval_expired`;
- `idempotency_conflict`;
- `persistence_unavailable`;
- `model_identity_mismatch`;
- `model_not_tool_capable`;
- `workspace_unavailable`;
- `internal_error`.

Errors never contain private keys, bearer capabilities, full stack traces, unbounded tool output,
or raw database errors. Local debug logs may contain bounded diagnostics but must pass the same
secret and path review as other durable evidence.

## 16. Threat Model

### 16.1 Assets

- source code and uncommitted changes;
- prompts, model output, and tool observations;
- local filesystem and shell authority;
- model/provider credentials and MCP credentials;
- host and device private keys;
- approval decisions;
- session transcript and event history;
- relay routing and push capabilities.

### 16.2 Adversaries

- passive network observer;
- malicious or compromised relay;
- attacker who guesses/scans relay route IDs;
- attacker with a stolen relay routing token but no device private key;
- attacker with a photographed live pairing QR;
- stolen unlocked or compromised phone;
- compromised host account;
- prompt injection in repository, shell output, fetched page, or model response;
- replaying or racing approval/command frames;
- denial-of-service client sending oversized or rapid frames;
- confused user approving the wrong host/workspace/action.

### 16.3 Required defenses

- TLS to relay plus end-to-end Noise encryption;
- QR-pinned host static identity;
- per-device static key and local grant registry;
- single-use high-entropy pairing secret and local confirmation;
- strict schema/size/rate/time bounds;
- command idempotency and conflicting-ID rejection;
- sequence-based replay and gap handling;
- approval binding to session/turn/call/id/digest;
- first-settlement-wins transaction;
- local-only capability widening;
- validated `Action` as approval source;
- tool results remain untrusted model data;
- host-side canonical path checks remain unchanged for native file tools;
- honest shell enforcement and default-disabled remote shell;
- device revocation closes active connections;
- generic push copy only;
- no relay plaintext logs or command queue;
- secure key storage and mobile app lock;
- emergency local disable that drops route and active device connections.

### 16.4 Out of protection scope

The design cannot protect against:

- an attacker already running code as the Camelid host user;
- a malicious OS/kernel on host or phone;
- a user deliberately approving a harmful validated action;
- data the agent intentionally sends through an approved network/shell/MCP action;
- shoulder-surfing or photographing a live pairing QR before it expires;
- traffic analysis at the relay;
- denial of service against the relay or host connection;
- host sleep/power loss.

## 17. Rejected Designs

### 17.1 Publicly tunnel the existing Camelid API

Rejected because current Workspace authorization is based on loopback binding and browser
same-origin properties, not remote device identity. Exposing or header-rewriting it would bypass
the intended boundary and expose unrelated model-management/API surfaces.

### 17.2 Reuse Web Workspace as the full mobile agent

Rejected because Workspace is deliberately read-only, viewer-claimed, single-stream, and
disconnect-cancelling. Widening it would mix two product contracts and regress the read-only
surface.

### 17.3 Raw terminal, tmux, SSH, or screen scraping

Rejected as the product protocol because terminal bytes do not reliably encode tool identity,
approval scope, replay, idempotency, or durable turn state. SSH/Tailscale may remain an expert
operator fallback, not Camelid's mobile architecture.

### 17.4 Plaintext cloud transcript/command relay

Rejected because the relay would possess source, prompts, outputs, and approval details, which
violates the chosen local-authority/privacy posture.

### 17.5 Static shared API key

Rejected because one copied secret grants every device equivalent authority, has poor revocation
and attribution, and is likely to leak through screenshots/configuration. Per-device keys and
grants are required.

### 17.6 Custom ad hoc cryptography

Rejected. Do not invent a signature string, PBKDF pairing scheme, or encryption envelope when
Noise and reviewed AEAD/key-agreement implementations exist.

### 17.7 Relay queues commands while host is offline

Rejected for v1. Delayed commands could execute after branch, workspace, model, capability, or
operator intent changed. Offline command semantics require a separate design with expiry and
fresh host confirmation.

### 17.8 Parallel active sessions before state ownership refactor

Rejected. Process-global cancel, plan, checkpoint, MCP, and subagent state would permit
cross-session interference. Many dormant histories with one atomically selected active session are
permitted; concurrent active execution sessions are not.

## 18. Implementation Plan and Gates

No slice widens public claims before its gate passes. Each slice should be a reviewable commit or
small PR with focused tests.

### Phase 0: truth and protocol fixtures

**Status (2026-07-24): COMPLETE for the development gate.** The versioned schemas, strict Rust
decoder/chunker, canonical approval digest, fixed-suite shared Rust core, generated Swift/Kotlin
bindings, dependency review, Linux Swift/Kotlin execution, and Android-optimized Kotlin source
compilation pass. This is not iOS/Android device qualification and not a security audit. Platform
secure-storage adapters, mobile packaging, and real-device evidence remain Phase 4 work; independent
security review remains a Phase 5 release blocker.

Work:

- land this document;
- correct user-facing shell/network descriptions where they overstate confinement;
- define JSON schemas/fixtures for inner messages, events, approval record, relay envelope, and QR;
- add protocol version/unknown-field/unknown-event rules;
- choose candidate Noise libraries and produce a cross-language spike outside production paths.

Gate:

- docs and fixtures contain no shipped claim;
- malformed/oversized fixture tests fail closed;
- Rust and mobile spike complete Noise IK handshake, exchange encrypted fixture, reject tamper,
  reject wrong host key, and reconnect with fresh transport keys;
- dependency/security review recorded.

### Phase 1: session-owned runtime seam

**Status (2026-07-24): COMPLETE for the development gate.** TUI, inline, and headless agent
front ends use independently constructible runtime cancellation, plan, checkpoint, and transcript
state. The remote-only loop accepts only the dedicated `RemoteV1` profile and emits typed events;
confirm-tier records are constructed from validated `Action` fields and bind complete executable
content to a canonical digest. Existing CLI/TUI behavior remains on the same validation,
approval-policy, execution, and audit path. No host/store/relay command is enabled.

Work:

- introduce `AgentRuntime` cancellation, plan, checkpoint, transcript, and policy ownership needed
  by remote v1;
- add structured agent events and approval records;
- adapt inline and TUI renderers;
- preserve existing tool validation/execution path and CLI behavior;
- do not enable networking or remote host yet.

Gate:

- existing chat/agent tests remain green;
- TUI and inline approval behavior unchanged;
- two test runtimes do not share cancel/plan/checkpoints;
- approval digest changes when any executable validated field changes;
- raw model prose cannot populate validated action fields.

### Phase 2: local durable host, no relay

**Status (2026-07-24): COMPLETE for the development gate.** A standalone bundled-SQLite store
owns schema versioning, device grants, one-session/one-turn state, command idempotency, monotonic
events, approval settlement, replay, cancellation, and restart recovery. The local host consumes
real `camelid.remote/v1` command envelopes and the Phase 1 structured loop. Its integration harness
proves read-before-write, durable allow-once approval, read-after-write verification, denial,
abort/timeout/cancel, persistence failure, subscriber loss/replay, restart invalidation, and exact
root/model/hash/capability-bound hydration. No relay-facing socket, public host command, mobile
product, capability row, or public claim exists.

Work:

- implement SQLite schema/migrations;
- implement one-session host and state machines;
- append-before-broadcast event store;
- implement idempotent commands, approval settlement, cancellation, replay;
- build a CLI `remote-test-client` or integration harness over loopback/in-process channels;
- no phone or cloud.

Gate scenarios:

- start turn -> read -> write approval -> allow once -> verify -> answer;
- deny write causes no mutation;
- stale/wrong digest decision causes no mutation;
- duplicate allow command cannot execute twice;
- conflicting reused command ID is rejected;
- disconnect/reconnect replays exact ordered history;
- subscriber absence does not stop execution;
- event broadcast loss recovers from SQLite;
- cancellation during model stream discards partial final answer;
- cancellation during approval invalidates it;
- host crash/restart marks active work inconclusive and never resumes a tool;
- model/root/capability identity mismatch refuses resume;
- database failure aborts before an unrecorded approval/execution transition.

### Phase 3: relay and end-to-end transport

**Development gate status: COMPLETE (2026-07-24).** The executable receipt is
`docs/architecture/REMOTE_AGENT_CONTROL_PHASE3_RECEIPT.md`. This is internal development
evidence only; the Phase 4 mobile and Phase 5 packaging, deployment, and independent security
review gates remain open.

Work:

- implement bounded blind relay;
- host outbound connection/reconnect;
- QR pairing, local confirmation, device registry/revocation;
- Noise transport and strict frame limits;
- generic push capability API with a fake push provider for tests;
- no mobile product UI yet.

Gate scenarios:

- relay test cannot decode known plaintext from captured frames;
- malicious relay frame modification is rejected;
- relay substitution with wrong host key fails pairing;
- expired/reused pairing secret fails;
- revoked device cannot reconnect and an open connection is closed;
- stolen relay route token without device key cannot send a command;
- host reconnect preserves session and replay;
- relay restart loses no authoritative host event;
- slow-client and oversized-frame backpressure stays bounded;
- host offline rejects commands instead of queuing them;
- generic push provider receives no sensitive content.

### Phase 4: mobile application

**Foundation status: IN PROGRESS (2026-07-25).** The local executable receipt is
`docs/architecture/REMOTE_AGENT_CONTROL_PHASE4_FOUNDATION.md`; Android details are in
`docs/architecture/REMOTE_AGENT_CONTROL_PHASE4_ANDROID_RECEIPT.md`. Mobile protocol, replay,
storage, approval-gate, QR pairing transport, Android Rust/Keystore linking, an arm64 development
APK, API 36 emulator instrumentation, fresh-IK reconnect, replay request, and authenticated chunk
transport pass. The Windows-first host CLI now performs tool-capable model admission, DPAPI host
identity and relay-bearer protection, durable session/command/event authority, authenticated
per-device dispatch, replay, approval, cancellation, and restart-safe session/route restoration.
Android now has host, pairing, session, activity, settings, and exact-action approval surfaces;
its reducer buffers live/replay races and reconnects with fresh IK on foreground. Its History view
negotiates a host-scoped catalog, isolates per-session replay projections, creates new remote
sessions, and explicitly activates continuable remote histories while preserving one active
execution authority. iOS, push, local
host-management UI, production relay/operator decisions, independent security review, and every
physical-device scenario below remain open.

Work:

- iOS and Android development builds;
- secure key storage adapters;
- QR pairing and host fingerprint;
- Noise transport and protocol client;
- replay reducer;
- session, tool, and approval UI;
- generic push registration;
- app lock/biometric gate;
- lifecycle/reconnect handling.

Gate scenarios on real devices:

- pair, revoke, re-pair;
- Wi-Fi -> cellular transition;
- lock/unlock during model run;
- kill/relaunch during streaming;
- kill/relaunch during pending approval;
- duplicate notification and duplicate decision;
- stale approval card after another device/local operator settles;
- long command/write preview without truncation-based approval;
- accessibility and small-screen overflow;
- push disabled/dropped still recovers pending state on app open;
- lost phone revoked from host blocks future connection.

### Phase 5: product integration

Work:

- local Remote settings/operations view in Web/Tauri UI;
- relay/self-host configuration;
- device list and emergency disable;
- retention/delete controls;
- capability contract entry marked preview/partial only after real end-to-end evidence;
- packaging, update compatibility, support docs, privacy policy, and threat-model review.

Gate:

- clean-machine install and pairing on supported host platforms;
- Windows, macOS, and Linux host qualification is separate and evidence-backed;
- iOS and Android real-device receipts;
- full Rust/frontend/mobile/relay CI;
- external security review or documented independent review of cryptographic and authorization
  boundaries;
- relay privacy/log audit;
- rollback rehearsal;
- public claims exactly match evidence.

### Phase 6: later capabilities

Deferred until v1 is stable:

- direct LAN/Tailscale transport using the same Noise/application protocol;
- self-hosted relay packaging;
- multiple parallel execution sessions after full state isolation;
- remote persistent grants with stronger policy and step-up authentication;
- attachments and encrypted blob transfer;
- mid-turn steering/queued prompts;
- optional remote-safe MCP policies;
- team sharing;
- durable offline command dispatch with fresh-host confirmation;
- background host service/autostart;
- session schedules.

## 19. Validation Matrix

### 19.1 Unit tests

- state transition exhaustiveness;
- event sequence atomicity;
- command idempotency and digest conflict;
- approval scope/digest matching;
- capability profile cannot widen from remote input;
- QR parser bounds and expiry;
- device registry/revocation;
- protocol unknown event/field behavior;
- frame size and replay pagination;
- delta coalescing preserves final answer;
- migration and unknown-newer-schema refusal;
- secret redaction in errors/logs.

### 19.2 Integration tests

- real `run_loop` with scripted driver and local durable bridge;
- local model API with canned or gated exact model where necessary;
- relay host/device sockets through network fault injection;
- disconnect at every command/approval persistence boundary;
- concurrent duplicate decisions and first-writer settlement;
- relay restart and host reconnect;
- database busy/failure;
- push fake provider;
- cross-language crypto fixtures.

### 19.3 Security tests

- route enumeration resistance;
- wrong/malformed device key;
- handshake replay/tamper;
- pairing secret brute-force rate bounds;
- revoked-device live connection teardown;
- stolen relay capability cannot authenticate inner protocol;
- malicious payload length/decompression avoidance (v1 has no compression);
- JSON depth/string/array bounds;
- action digest canonicalization ambiguity;
- approval content chunk reorder/omission;
- local path/symlink escape regression;
- shell capability disclosure on each host platform;
- prompt injection cannot settle approval or widen capability.

### 19.4 Reliability tests

- app background/foreground;
- mobile process death;
- host process death;
- relay process death;
- packet loss, duplication, delay, and reordering at the relay harness;
- slow phone and bounded backpressure;
- event log retention boundary;
- clock skew affects display/expiry safely but not Noise transport identity;
- push loss and duplication;
- network transition and DNS/TLS failure.

### 19.5 Performance budgets

Measure before setting release budgets:

- relay frame latency and memory per connection;
- host event-commit latency;
- model delta coalescing write rate;
- replay time and bytes for long sessions;
- mobile cold-open to authoritative state;
- approval round-trip excluding user think time;
- idle host/relay/mobile resource use.

No latency or scale SLA is claimed in this design.

## 20. Observability and Privacy

### 20.1 Host logs

Allowed:

- state transitions and stable IDs;
- event sequence and event type;
- bounded timing and byte counts;
- action digest, tool name, risk, and settlement outcome;
- relay connection error class;
- device ID/label in local-only administration logs.

Default-redacted:

- prompt/answer text;
- file content and tool output;
- full path and shell command;
- keys, pairing secrets, route tokens, notification capabilities;
- Noise frames.

### 20.2 Relay metrics

Allowed aggregate metrics:

- active routes/connections;
- frame/byte counts and rejected frame classes;
- reconnect rates;
- push request/result counts;
- bounded latency histograms;
- abuse/rate-limit counters.

No model, workspace, prompt, tool, approval, or source-code dimensions exist at the relay.

### 20.3 Mobile diagnostics

Crash and analytics reporting is off by default unless separately designed and disclosed. A debug
export requires explicit user action, redacts secrets, and never includes private keys or raw
encrypted transport frames. Production mobile logs must not print decrypted protocol payloads.

## 21. Release Contract

Remote control may be advertised only after:

- the exact capability is present in structured Camelid capability output;
- relay, host, iOS, and Android versions are mutually compatible under a documented matrix;
- a clean end-to-end receipt proves local model execution, local file mutation, remote approval,
  reconnect replay, cancellation, and relay ciphertext blindness;
- public docs state the shell and network limitations accurately;
- device revocation and emergency disable are verified;
- no unrelated Camelid surface is remotely exposed;
- the project has a support and security-reporting path for the relay/mobile components.

Initial capability wording should be narrow, for example:

> Preview: control one explicitly armed local Camelid agent from paired iOS/Android devices
> through an end-to-end encrypted relay. Inference, source files, tool validation, and execution
> remain on the host. Relay availability is required for the hosted works-anywhere path. Shell
> and network authority depend on the locally selected capability profile and platform-specific
> enforcement.

Do not claim “the relay sees nothing,” “all commands are workspace-jailed,” “network is always
off,” “zero metadata,” or “production-ready” without evidence supporting those exact statements.

## 22. Rollback and Emergency Disable

### 22.1 User rollback

The local operator can:

- stop the remote host;
- revoke one or all devices;
- rotate the host key (revokes all pairings);
- disable push capability;
- delete local remote session history;
- return to ordinary local CLI/TUI agent behavior.

Emergency disable closes relay connections, cancels the active remote turn, invalidates pending
approvals, and refuses new commands. It does not delete source files or silently undo completed
agent writes.

### 22.2 Code rollback

The feature remains additive. Removing:

- the `agent host` command;
- remote host/store/crypto/transport modules;
- local Remote UI;
- relay/mobile packages;
- the remote capability entry;

must leave ordinary Chat, Web Workspace, model management, local CLI/TUI agent, API inference,
and compatibility gates unchanged.

### 22.3 Protocol rollback

Relay can disable a vulnerable protocol version globally at route enrollment while local Camelid
continues to work. Host and mobile reject unsupported versions explicitly. No server-side fallback
may silently downgrade E2EE or authorization.

## 23. Open External Decisions

These items require project/product/operator decisions and must not be guessed by an implementation
agent:

1. Hosted relay operator, domain, regions, account model, and funding.
2. Whether relay and mobile live in this monorepo or separate repositories.
3. Production mobile signing accounts, bundle identifiers, and store ownership.
4. APNs/FCM credentials and privacy-policy ownership.
5. Supported host OS versions for the first public preview.
6. Whether users must sign into a Camelid account or use anonymous high-entropy relay routes in v1.
7. Retention defaults and legal/privacy obligations for relay routing metadata.
8. External security review scope and responsible disclosure process across all components.
9. Whether a branded hosted relay is free, paid, rate-limited, or self-host only.

The architecture remains valid with either an account-backed or anonymous-capability relay, but
the relay abuse/recovery design differs. That choice needs its own amendment before production
relay authentication is implemented.

## 24. Implementation Rules for Future Sessions

An implementation session starting from this document must:

1. Work in the real Git checkout, not a stale extracted workspace.
2. Re-read current code before editing; paths and symbols here are anchored to the audited commit.
3. Check whether upstream has advanced the agent/runtime/session architecture.
4. Preserve unrelated user changes and active branches.
5. Implement one phase at a time and run its focused executable gate immediately after the first
   edit.
6. Keep verified current behavior, proposed design, and shipped claims separate.
7. Never expose the current loopback management API through a tunnel as a shortcut.
8. Never parse terminal rendering or human prose into protocol authority.
9. Never execute before durable approval settlement.
10. Never broadcast an event before its sequence is committed.
11. Never restore approval grants from disk or relay state.
12. Never introduce remote shell/network/MCP/GUI scope without a local capability gate and honest
    platform disclosure.
13. Never invent cryptography; use the selected audited Noise implementation and pinned
    cross-language fixtures.
14. Never put private keys or pairing secrets in SQLite plaintext, logs, QR diagnostics, crash
    reports, or relay storage.
15. Keep exactly one active execution session; dormant histories may be replayed but cannot accept
  commands until explicit durable activation.
16. Treat push as a hint and replay as truth.
17. Treat relay outage as remote unavailability, not permission to queue delayed commands.
18. Keep all queues, frames, strings, event batches, retries, and timeouts bounded.
19. Fail closed on unknown schema versions, identity mismatch, stale approval, database failure,
    and capability ambiguity.
20. Update this document and `DECISIONS.md` when a binding choice changes.

## 25. Reference Projects and Lessons

These sources informed the design; they are references, not dependencies or proof that Camelid
already implements their behavior.

### Amp

- Remote-controllable local threads plus cloud executors demonstrate the importance of separating
  control plane from execution location.
- Durable cloud-backed threads make multi-device UI possible but allow the service to store
  transcript/tool data. Camelid chooses a blind relay and host-authoritative store instead.
- Reference: <https://ampcode.com/news/neo>
- Security boundary: <https://ampcode.com/security>

### Claude Code Remote Control and Channels

- Outbound-only host connections avoid inbound ports.
- Vendor-stored transcript enables synchronization but is incompatible with Camelid's chosen blind
  relay posture.
- Permission relay demonstrates exact request IDs, authenticated senders, and first-answer-wins.
- References:
  - <https://code.claude.com/docs/en/remote-control>
  - <https://code.claude.com/docs/en/channels-reference>

### Happy

- Open-source mobile/CLI/server separation demonstrates client-side encryption, paired devices,
  structured provider adapters, durable sequence/replay work, and the operational complexity of
  reconnect/RPC routing.
- Its documented message-loss and state-reconciliation fixes reinforce that persistent sequence
  truth must not depend on WebSocket delivery.
- Reference: <https://github.com/slopus/happy>

### Tactic Remote

- tmux persistence and mobile lifecycle handling show why execution cannot depend on a phone
  connection.
- Its terminal-oriented architecture is useful for compatibility but not the protocol selected for
  Camelid.
- Cloudflare TLS is transport encryption, not device-to-device E2EE; Camelid requires an inner
  cryptographic channel blind to the relay.
- References:
  - <https://tacticremote.com/blog/2026-02-28-tmux-architecture-and-session-persistence>
  - <https://tacticremote.com/docs/setup/security-model-and-trust-boundaries>

### Codex app-server

- `thread -> turn -> item`, typed server requests, authoritative completion events, bounded queues,
  capability negotiation, resume, steering, cancellation, and approvals are the strongest public
  structured-agent reference reviewed.
- Its remote-control implementation also confirms the host-enrollment, pairing, device-list, and
  outbound-relay shape, though Camelid's trust/storage choices remain independent.
- Reference: <https://github.com/openai/codex/tree/main/codex-rs/app-server>

### Agent Client Protocol

- ACP demonstrates standardized initialization, capability negotiation, sessions, prompts,
  cancellation, updates, terminal/filesystem methods, and permission requests.
- ACP does not by itself supply Camelid's relay, E2EE pairing, push, or durable host event store.
- Reference: <https://agentclientprotocol.com/>

## 26. Final Decision Summary

Camelid will pursue a works-anywhere mobile remote-control architecture with these defining
properties:

- local model, local files, local tools, local authority;
- optional blind relay reached through outbound connections;
- end-to-end Noise encryption between paired phone and host;
- local durable, monotonically sequenced event and command store;
- reconnect by replay cursor, independent of viewer lifetime;
- many durable histories with one active execution session in v1;
- local-only capability widening;
- dedicated remote-safe tool profile;
- exact validated-action approvals bound to a cryptographic digest;
- no remote yolo, unrestricted filesystem, persistent grants, MCP, subagents, or GUI control;
- generic push hints only;
- no offline delayed command queue;
- structured protocol, never terminal scraping;
- evidence-gated implementation and honest platform-specific shell/network claims.

The next implementation action is to continue Phase 4/5 without promotion: link and qualify the
iOS XCFramework/Keychain path on macOS, implement generic APNs/FCM push registration, add the local
host-management/emergency-disable UI, and run the full physical-device lifecycle matrix. Do not
add a public capability row or support claim before the production relay/operator decision,
independent security review, and clean end-to-end release receipt.