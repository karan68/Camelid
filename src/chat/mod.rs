//! `camelid chat` — an interactive terminal chat client for the local Camelid
//! engine.
//!
//! Two front ends share one [`session::Session`] core (state, sampling, request
//! shape — no I/O):
//! - [`tui`]: a full-screen ratatui app (scrollable chat, status bar, sidebar,
//!   modal picker) — the default on an interactive terminal.
//! - [`inline`]: a scrollback-friendly line REPL — used for `--plain`, pipes,
//!   and non-TTY contexts (the lane the smoke scripts and tests drive).
//!
//! Both stream `/v1/chat/completions` over the same audited HTTP/SSE client, so
//! terminal output matches the validated lane. The picker is derived from the
//! `/api/capabilities` ledger at runtime (supported rows only); pointing
//! `--model` at an unsupported GGUF is refused with the engine's typed error.
//! See `DECISIONS.md` D6 and `RECON_CHAT.md`.

pub(crate) mod agent;
mod agent_bench;
mod agent_eval;
mod agent_orchestration;
mod agent_session;
mod agent_syscap;
mod agent_tui;
mod audit;
mod banner;
mod checkpoint;
mod client;
mod clipboard;
mod inline;
mod markdown;
mod mcp;
mod models;
mod palette;
mod plan;
pub(crate) mod remote_control;
mod remote_host;
mod remote_identity;
mod remote_pairing;
mod remote_transport;
mod server;
mod session;
mod shell_sandbox;
mod subagent;
mod term_guard;
mod theme;
mod tool_parse;
mod tools;
mod tui;
#[cfg(windows)]
mod win_clipboard;
#[cfg(windows)]
mod win_console;
#[cfg(windows)]
mod win_input;
#[cfg(windows)]
mod win_job;
#[cfg(windows)]
mod win_uia;
pub(crate) mod workspace_bridge;
mod workspace_cli;
pub(crate) mod workspace_memory;

pub use workspace_cli::{run as run_workspace_cli, WorkspaceCliAction, WorkspaceCliOptions};

use std::collections::{HashMap, VecDeque};
use std::io::IsTerminal;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc, Mutex};

use sha2::{Digest, Sha256};
use uuid::Uuid;

use camelid_remote_protocol::{
    decode_command, decode_message, decode_replay_request, decode_session_catalog_request,
    encode_chunks, Command, CommandResult as ProtocolCommandResult, CommandStatus, EventBatch,
    MessageKind, RemoteEvent, RemoteMessage, RemoteSessionState, ReplayEnd, SessionCatalog,
    SessionCatalogCursor, SessionHistorySource, SessionSummary, PROTOCOL,
};
use camelid_remote_store::{
    AcceptActivateSession, AcceptCreateSession, AcceptSessionSwitch, RemoteStore, SessionHead,
    SessionState, StoredEvent, StoredSessionCatalogEntry,
};

const MAX_HOST_SESSION_HISTORIES: usize = 1024;

use client::Client;
use server::{RemoteHostServerHandle, ServerHandle};
use session::{LoadResult, Session, Settings};

pub(crate) const VERSION: &str = match option_env!("CAMELID_GIT_DESCRIBE") {
    Some(describe) => describe,
    None => env!("CARGO_PKG_VERSION"),
};

#[derive(Debug)]
pub struct RemoteHostOptions {
    pub model: PathBuf,
    pub addr: SocketAddr,
    pub workdir: PathBuf,
    pub relay_url: String,
    pub db_path: Option<PathBuf>,
    pub max_steps: usize,
    pub max_tokens: u32,
    pub allow_net: bool,
    pub allow_shell: bool,
    pub shell_timeout: u64,
    pub models_dir: PathBuf,
    pub reconnect_initial_ms: u64,
    pub reconnect_max_ms: u64,
    pub reconnect_jitter_percent: u8,
    pub relay_keepalive_ms: u64,
}

#[derive(Debug)]
pub struct RemoteAdminOptions {
    pub db_path: Option<PathBuf>,
}

pub fn list_remote_devices(options: RemoteAdminOptions) -> anyhow::Result<()> {
    let store = open_remote_admin_store(&options)?;
    let devices = store
        .devices()
        .map_err(|error| anyhow::anyhow!("remote device registry unavailable: {error}"))?;
    let output = devices
        .into_iter()
        .map(|device| {
            serde_json::json!({
                "device_id": device.device_id,
                "label": device.label,
                "status": if device.revoked_at_unix_ms.is_some() { "revoked" } else { "authorized" },
                "created_at_unix_ms": device.created_at_unix_ms,
                "last_seen_at_unix_ms": device.last_seen_at_unix_ms,
                "revoked_at_unix_ms": device.revoked_at_unix_ms,
            })
        })
        .collect::<Vec<_>>();
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

pub fn revoke_remote_device(options: RemoteAdminOptions, device_id: Uuid) -> anyhow::Result<()> {
    let mut store = open_remote_admin_store(&options)?;
    store
        .revoke_device(device_id, unix_time_ms()?)
        .map_err(|error| anyhow::anyhow!("device {device_id} could not be revoked: {error}"))?;
    println!("revoked remote device {device_id}");
    Ok(())
}

pub fn disable_remote_devices(options: RemoteAdminOptions) -> anyhow::Result<()> {
    let mut store = open_remote_admin_store(&options)?;
    let revoked = store
        .revoke_all_devices(unix_time_ms()?)
        .map_err(|error| anyhow::anyhow!("remote devices could not be disabled: {error}"))?;
    println!("disabled {} remote device(s)", revoked.len());
    Ok(())
}

fn open_remote_admin_store(options: &RemoteAdminOptions) -> anyhow::Result<RemoteStore> {
    let database_path = match options.db_path.as_ref() {
        Some(path) => path.clone(),
        None => remote_data_root()?.join("remote-control.sqlite3"),
    };
    anyhow::ensure!(
        database_path.is_file(),
        "remote authority database {} does not exist",
        database_path.display()
    );
    RemoteStore::open(&database_path)
        .map_err(|error| anyhow::anyhow!("remote state unavailable: {error}"))
}

struct RemoteConnection {
    device_id: Uuid,
    device_key: [u8; 32],
    host: remote_host::LocalRemoteHost,
    reassembler: camelid_remote_protocol::ChunkReassembler,
}

struct TurnJob {
    connection_id: Uuid,
    command_id: Uuid,
    turn: remote_host::AcceptedRemoteTurn,
    host: remote_host::LocalRemoteHost,
}

struct TurnCompletion {
    connection_id: Uuid,
    command_id: Uuid,
    result: Result<agent::LoopEnd, camelid_remote_store::StoreError>,
    head: Result<SessionHead, camelid_remote_store::StoreError>,
}

struct RemoteReporter;

impl agent::Reporter for RemoteReporter {
    fn model_text(&mut self, _: &str) {}
    fn tool_call(&mut self, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &tools::ToolOutcome) {}
    fn notice(&mut self, text: &str) {
        eprintln!("remote agent: {text}");
    }
}

pub async fn run_remote_host(options: RemoteHostOptions) -> anyhow::Result<i32> {
    init_terminal();
    anyhow::ensure!(options.max_steps > 0, "--max-steps must be at least 1");
    anyhow::ensure!(options.max_tokens > 0, "--max-tokens must be at least 1");
    anyhow::ensure!(
        options.relay_keepalive_ms > 0,
        "--relay-keepalive-ms must be at least 1"
    );
    anyhow::ensure!(
        options.shell_timeout > 0,
        "--shell-timeout must be at least 1"
    );
    let reconnect_policy = remote_transport::ReconnectPolicy::new(
        std::time::Duration::from_millis(options.reconnect_initial_ms),
        std::time::Duration::from_millis(options.reconnect_max_ms),
        options.reconnect_jitter_percent,
    )
    .map_err(|error| anyhow::anyhow!("invalid relay reconnect policy: {error}"))?;
    let cancellation = tokio_util::sync::CancellationToken::new();
    let signal_cancellation = cancellation.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        signal_cancellation.cancel();
    });
    let canonical_workdir = std::fs::canonicalize(&options.workdir).map_err(|error| {
        anyhow::anyhow!(
            "remote workspace {} is unavailable: {error}",
            options.workdir.display()
        )
    })?;
    anyhow::ensure!(
        canonical_workdir.is_dir(),
        "remote workspace is not a directory"
    );
    let canonical_model = std::fs::canonicalize(&options.model).map_err(|error| {
        anyhow::anyhow!(
            "remote model {} is unavailable: {error}",
            options.model.display()
        )
    })?;

    let (remote_management, mut remote_management_commands) = remote_control::channel();
    let client = Client::new(options.addr);
    let _server = RemoteHostServerHandle::start(
        options.addr,
        &client,
        options.models_dir.clone(),
        remote_management.clone(),
    )
    .await?;
    let mut session = Session::new(
        client,
        options.models_dir.clone(),
        Settings {
            temperature: 0.0,
            top_p: None,
            top_k: None,
            max_tokens: options.max_tokens,
            seed: None,
            stream: false,
            enable_thinking: false,
        },
        None,
    );
    eprintln!("Loading {} ...", canonical_model.display());
    let label = catalog_label_for(&canonical_model);
    match session.load_model_file(
        &canonical_model,
        label.as_deref(),
        label.as_ref().map(|_| "supported"),
    )? {
        LoadResult::Loaded => {}
        LoadResult::Unsupported(message) => anyhow::bail!(message),
    }
    anyhow::ensure!(
        session.active_tool_capable(),
        "remote host requires an exact tool-capable compatibility-ledger model"
    );

    let data_root = remote_data_root()?;
    std::fs::create_dir_all(&data_root)?;
    let database_path = options
        .db_path
        .clone()
        .unwrap_or_else(|| data_root.join("remote-control.sqlite3"));
    if let Some(parent) = database_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut store = RemoteStore::open(&database_path)
        .map_err(|error| anyhow::anyhow!("remote state unavailable: {error}"))?;
    let secret_store = remote_identity::ProtectedFileSecretStore::new(data_root.join("secrets"));
    let now = unix_time_ms()?;
    let host_identity = remote_identity::load_or_create(&mut store, &secret_store, now)
        .map_err(|error| anyhow::anyhow!("remote host identity unavailable: {error}"))?;

    let model_sha256 = format!("sha256:{}", sha256_file(&canonical_model)?);
    let model_id = session.active_label.clone();
    let remote_shell_mode = if options.allow_shell {
        shell_sandbox::ShellSandbox::Sandboxed
    } else {
        shell_sandbox::ShellSandbox::Disabled
    };
    let shell_snapshot = if options.allow_shell {
        let enforced = shell_sandbox::describe_sandboxed(&canonical_workdir)
            .map_err(|error| anyhow::anyhow!("remote shell cannot be enforced: {error}"))?;
        serde_json::json!({
            "enabled": true,
            "mode": enforced.mode.as_str(),
            "enforced_layers": enforced.layers,
            "note": enforced.note,
        })
    } else {
        serde_json::json!({
            "enabled": false,
            "mode": "disabled",
            "enforced_layers": [],
            "note": null,
        })
    };
    let enabled_tools = tools::specs_for(
        tools::ToolProfile::RemoteV1,
        options.allow_net,
        remote_shell_mode,
    )
    .into_iter()
    .map(|tool| tool.name)
    .collect::<Vec<_>>();
    let capability_snapshot_value = serde_json::json!({
        "profile": "remote_v1",
        "workspace": canonical_workdir.display().to_string(),
        "model_id": model_id,
        "model_artifact_sha256": model_sha256,
        "tools": enabled_tools,
        "file_scope": "canonical_workspace",
        "shell": shell_snapshot,
        "camelid_network_tools": options.allow_net,
        "mcp": false,
        "subagents": false,
        "gui_control": false,
        "persistent_approval_grants": false,
        "max_steps": options.max_steps,
        "max_tokens": options.max_tokens,
    });
    let capability_snapshot = serde_json::to_string(&capability_snapshot_value)?;
    store
        .recover_interrupted(now)
        .map_err(|error| anyhow::anyhow!("interrupted remote work recovery failed: {error}"))?;
    let canonical_root = canonical_workdir.display().to_string();
    let persisted_active = store
        .active_session()
        .map_err(|error| anyhow::anyhow!("active remote session lookup failed: {error}"))?
        .and_then(|active| {
            store
                .session_catalog_entry(&canonical_root, active.session_id)
                .ok()
                .flatten()
        })
        .filter(|entry| {
            entry.model_id == model_id
                && entry.model_sha256 == model_sha256
                && entry.capability_snapshot_json == capability_snapshot
                && matches!(
                    entry.state,
                    SessionState::Armed | SessionState::Idle | SessionState::Failed
                )
        })
        .map(|entry| (entry.session_id, entry.state));
    let reusable = match persisted_active {
        Some(active) => Some(active),
        None => store
            .reusable_session(
                &canonical_root,
                &model_id,
                &model_sha256,
                &capability_snapshot,
            )
            .map_err(|error| anyhow::anyhow!("remote session lookup failed: {error}"))?,
    };
    let mut session_id = match reusable {
        Some((session_id, SessionState::Idle)) => session_id,
        Some((session_id, state @ (SessionState::Armed | SessionState::Failed))) => {
            store
                .rearm_session(session_id, state, now + 1)
                .map_err(|error| {
                    anyhow::anyhow!("remote session could not be re-armed: {error}")
                })?;
            session_id
        }
        Some(_) => anyhow::bail!("remote session is not safely reusable"),
        None => {
            let session_id = Uuid::new_v4();
            store
                .create_session(
                    session_id,
                    &canonical_root,
                    &model_id,
                    &model_sha256,
                    &capability_snapshot,
                    now,
                )
                .and_then(|()| {
                    store.transition_session(
                        session_id,
                        SessionState::Armed,
                        SessionState::Idle,
                        now + 1,
                    )
                })
                .map_err(|error| anyhow::anyhow!("remote session could not be armed: {error}"))?;
            session_id
        }
    };
    store
        .ensure_session_bootstrap_events(session_id, &capability_snapshot_value, now + 2)
        .map_err(|error| anyhow::anyhow!("remote bootstrap events failed: {error}"))?;
    store
        .activate_session(session_id, now + 3)
        .map_err(|error| anyhow::anyhow!("remote session activation failed: {error}"))?;
    let stored_identity = store
        .host_identity()
        .map_err(|error| anyhow::anyhow!("remote host identity unavailable: {error}"))?;
    anyhow::ensure!(
        stored_identity.host_id == host_identity.host_id
            && stored_identity.noise_public == host_identity.public_key,
        "remote host identity metadata does not match protected key material"
    );
    let enrollment_token = std::env::var("CAMELID_RELAY_ENROLLMENT_TOKEN").ok();
    let endpoint = remote_transport::HostRelayEndpoint::new(&options.relay_url)
        .map_err(|error| anyhow::anyhow!("invalid relay endpoint: {error}"))?;
    let stored_binding = store
        .relay_binding()
        .map_err(|error| anyhow::anyhow!("relay binding lookup failed: {error}"))?;
    let restored = stored_binding
        .as_ref()
        .filter(|binding| binding.relay_url == options.relay_url)
        .and_then(|binding| {
            secret_store
                .load_bytes(&binding.capability_secret_reference)
                .ok()
                .and_then(|bytes| String::from_utf8(bytes).ok())
                .map(|host_capability| remote_transport::RelayEnrollment {
                    route_id: binding.route_id.clone(),
                    host_capability,
                })
        });
    let (enrollment, mut relay, route_replaced) = match restored {
        Some(enrollment) => match remote_transport::HostRelaySocket::connect(
            &endpoint,
            &enrollment.route_id,
            &enrollment.host_capability,
        )
        .await
        {
            Ok(relay) => (enrollment, relay, false),
            Err(_) => {
                let enrollment = enroll_and_persist_relay(
                    &endpoint,
                    enrollment_token.as_deref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "CAMELID_RELAY_ENROLLMENT_TOKEN is required to replace a stale route"
                        )
                    })?,
                    &options.relay_url,
                    &mut store,
                    &secret_store,
                    stored_binding.as_ref(),
                )
                .await?;
                let relay = remote_transport::HostRelaySocket::connect(
                    &endpoint,
                    &enrollment.route_id,
                    &enrollment.host_capability,
                )
                .await
                .map_err(|error| anyhow::anyhow!("relay host connection failed: {error}"))?;
                (enrollment, relay, true)
            }
        },
        None => {
            let enrollment = enroll_and_persist_relay(
                &endpoint,
                enrollment_token.as_deref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "CAMELID_RELAY_ENROLLMENT_TOKEN is required for first enrollment"
                    )
                })?,
                &options.relay_url,
                &mut store,
                &secret_store,
                stored_binding.as_ref(),
            )
            .await?;
            let relay = remote_transport::HostRelaySocket::connect(
                &endpoint,
                &enrollment.route_id,
                &enrollment.host_capability,
            )
            .await
            .map_err(|error| anyhow::anyhow!("relay host connection failed: {error}"))?;
            (enrollment, relay, true)
        }
    };
    drop(enrollment_token);
    let authorized_devices = store
        .authorized_device_count()
        .map_err(|error| anyhow::anyhow!("device registry unavailable: {error}"))?;
    let store = Arc::new(Mutex::new(store));
    remote_management
        .activate(Arc::clone(&store))
        .map_err(|_| anyhow::anyhow!("remote management state could not be activated"))?;

    let pairing = remote_pairing::PairingCoordinator::new(Arc::clone(&store), stored_identity);
    let mut noise = remote_transport::AuthorizedNoiseSessions::new(
        host_identity.private_key(),
        Arc::clone(&store),
    );
    if authorized_devices == 0 || route_replaced {
        eprintln!("No usable paired device. Create a pairing offer from the local Remote view.");
    } else {
        eprintln!("Existing relay route and paired devices restored.");
    }

    let sandbox = tools::Sandbox::new(
        &canonical_workdir,
        options.allow_net,
        std::time::Duration::from_secs(options.shell_timeout),
    )?
    .with_shell_mode(remote_shell_mode);
    let history_sandbox =
        tools::Sandbox::new(&canonical_workdir, false, std::time::Duration::from_secs(1))?
            .with_shell_mode(shell_sandbox::ShellSandbox::Disabled);
    let config = agent::AgentConfig {
        workdir: canonical_workdir.clone(),
        max_steps: options.max_steps,
        auto_approve: false,
        yolo: false,
        allow_net: options.allow_net,
        allow_fs: false,
        shell_timeout: std::time::Duration::from_secs(options.shell_timeout),
        max_tokens: options.max_tokens,
        temperature: 0.0,
        audit: audit::sink_from_config(None),
        shell_sandbox: sandbox.shell_mode(),
        tool_profile: tools::ToolProfile::RemoteV1,
        ctx_budget: Some(
            session
                .active_ctx
                .unwrap_or(agent::AGENT_VALIDATED_CTX)
                .min(agent::AGENT_VALIDATED_CTX),
        ),
    };
    let mut driver = agent::LiveDriver::new(&session, options.max_tokens, 0.0);
    driver.set_context_budget(config.ctx_budget);
    let (turn_sender, turn_receiver) = mpsc::channel::<TurnJob>();
    let (completion_sender, mut completion_receiver) =
        tokio::sync::mpsc::unbounded_channel::<TurnCompletion>();
    std::thread::Builder::new()
        .name("camelid-remote-model".into())
        .spawn(move || {
            let mut reporter = RemoteReporter;
            while let Ok(job) = turn_receiver.recv() {
                let result = job.host.run_accepted_turn(
                    job.turn,
                    &mut driver,
                    &mut reporter,
                    &sandbox,
                    &config,
                );
                let head = job.host.session_head();
                let _ = completion_sender.send(TurnCompletion {
                    connection_id: job.connection_id,
                    command_id: job.command_id,
                    result,
                    head,
                });
            }
        })?;

    let mut base_host: Option<remote_host::LocalRemoteHost> = None;
    let mut events: Option<mpsc::Receiver<StoredEvent>> = None;
    let mut connections: HashMap<Uuid, RemoteConnection> = HashMap::new();
    let mut active_turn_device: Option<Uuid> = None;
    let mut tick = tokio::time::interval(std::time::Duration::from_millis(20));
    let keepalive_interval = std::time::Duration::from_millis(options.relay_keepalive_ms);
    let mut next_keepalive = tokio::time::Instant::now() + keepalive_interval;
    loop {
        tokio::select! {
            received = relay.receive() => {
                let frame = match received {
                    Ok(frame) => frame,
                    Err(_) => {
                        connections.clear();
                        if let Ok(Some(remote_pairing::PairingStatus::AwaitingConfirmation {
                            connection_id,
                            ..
                        })) = pairing.status() {
                            let _ = pairing.cancel_connection(connection_id);
                            let _ = remote_management.publish_pairing(None);
                        }
                        noise.reset_for_reconnect();
                        let reconnected = remote_transport::HostRelaySocket::connect_with_backoff(
                            &endpoint,
                            &enrollment.route_id,
                            &enrollment.host_capability,
                            reconnect_policy,
                            &cancellation,
                        )
                        .await;
                        relay = match reconnected {
                            Ok(relay) => relay,
                            Err(_) if cancellation.is_cancelled() => return Ok(0),
                            Err(error) => {
                                return Err(anyhow::anyhow!("relay reconnect failed: {error}"));
                            }
                        };
                        next_keepalive = tokio::time::Instant::now() + keepalive_interval;
                        eprintln!("Relay connection restored; devices must establish fresh Noise sessions.");
                        continue;
                    }
                };
                let connection_id = frame.connection_id;
                if let std::collections::hash_map::Entry::Vacant(entry) =
                    connections.entry(connection_id)
                {
                    match noise.accept_first_record(frame.clone()) {
                        Ok(handshake) if handshake.initial_payload.is_empty() => {
                            relay.send(handshake.response).await
                                .map_err(|error| anyhow::anyhow!("Noise response failed: {error}"))?;
                            let (device_id, device_key) = noise
                                .authenticated_device(connection_id)
                                .ok_or_else(|| anyhow::anyhow!("authenticated device identity disappeared"))?;
                            store
                                .lock()
                                .map_err(|_| anyhow::anyhow!("remote device registry unavailable"))?
                                .mark_device_seen(device_id, unix_time_ms()?)
                                .map_err(|error| anyhow::anyhow!("remote device last-seen update failed: {error}"))?;
                            let host = match base_host.as_ref() {
                                Some(host) => host.for_device(device_id, device_key),
                                None => {
                                    let host = remote_host::LocalRemoteHost::from_shared(
                                        Arc::clone(&store),
                                        session_id,
                                        device_id,
                                        device_key,
                                        remote_host::HostIdentity {
                                            canonical_root: &canonical_workdir.display().to_string(),
                                            model_id: &model_id,
                                            model_sha256: &model_sha256,
                                            capability_snapshot_json: &capability_snapshot,
                                        },
                                        unix_time_ms()?,
                                    ).map_err(|error| anyhow::anyhow!("remote host state unavailable: {error}"))?;
                                    events = Some(host.subscribe().map_err(|error| anyhow::anyhow!("remote events unavailable: {error}"))?);
                                    base_host = Some(host.clone());
                                    host
                                }
                            };
                            entry.insert(RemoteConnection {
                                device_id,
                                device_key,
                                host,
                                reassembler: camelid_remote_protocol::ChunkReassembler::default(),
                            });
                            eprintln!("Authenticated remote device connected.");
                            continue;
                        }
                        Ok(_) => {}
                        Err(_) => {}
                    }
                    let accepting_pairing = matches!(
                        pairing.status(),
                        Ok(Some(remote_pairing::PairingStatus::Offered { .. }))
                    );
                    if accepting_pairing {
                        let received = noise.receive_pairing_first_record(frame);
                        if let Ok(received) = received {
                            let pending = pairing.receive_authenticated_request(
                                &received.pairing_payload,
                                connection_id,
                                received.device_noise_public,
                                &received.handshake_hash,
                                unix_time_ms()?,
                            );
                            if let (
                                Ok(pending),
                                Ok(Some(remote_pairing::PairingStatus::AwaitingConfirmation {
                                    expires_at_unix_ms,
                                    ..
                                })),
                            ) = (pending, pairing.status())
                            {
                                remote_management.publish_pairing(Some(
                                    remote_control::RemotePairingStatus::AwaitingConfirmation {
                                        confirmation_id: pending.confirmation_id,
                                        expires_at_unix_ms,
                                        device_label: pending.device_label,
                                        authentication_fingerprint: pending.authentication_fingerprint,
                                    },
                                )).map_err(|_| anyhow::anyhow!("pairing status unavailable"))?;
                                continue;
                            }
                        }
                    }
                    noise.disconnect(connection_id);
                    let _ = pairing.cancel_connection(connection_id);
                    let _ = relay.disconnect_device(connection_id).await;
                    continue;
                }

                let plaintext = match noise.open(frame) {
                    Ok(plaintext) => plaintext,
                    Err(_) => {
                        connections.remove(&connection_id);
                        noise.disconnect(connection_id);
                        let _ = relay.disconnect_device(connection_id).await;
                        continue;
                    }
                };
                let complete = {
                    let connection = connections.get_mut(&connection_id).expect("checked above");
                    connection.reassembler.push(&plaintext)
                };
                let message = match complete {
                    Ok(Some(message)) => message,
                    Ok(None) => continue,
                    Err(_) => {
                        connections.remove(&connection_id);
                        noise.disconnect(connection_id);
                        let _ = relay.disconnect_device(connection_id).await;
                        continue;
                    }
                };
                let envelope = match decode_message(&message) {
                    Ok(envelope) => envelope,
                    Err(_) => {
                        connections.remove(&connection_id);
                        noise.disconnect(connection_id);
                        let _ = relay.disconnect_device(connection_id).await;
                        continue;
                    }
                };
                let connection = connections.get(&connection_id).expect("checked above");
                if envelope.host_id != host_identity.host_id
                    || envelope.device_id != connection.device_id
                {
                    connections.remove(&connection_id);
                    noise.disconnect(connection_id);
                    let _ = relay.disconnect_device(connection_id).await;
                    continue;
                }
                let host = connection.host.clone();
                let device_id = connection.device_id;
                match envelope.kind {
                    MessageKind::SessionCatalogRequest => {
                        let request = match decode_session_catalog_request(&envelope) {
                            Ok(request) => request,
                            Err(_) => {
                                connections.remove(&connection_id);
                                noise.disconnect(connection_id);
                                let _ = relay.disconnect_device(connection_id).await;
                                continue;
                            }
                        };
                        let catalog = build_session_catalog(
                            &SessionCatalogContext {
                                store: &store,
                                sandbox: &history_sandbox,
                                host_id: host_identity.host_id,
                                canonical_root: &canonical_root,
                                active_session_id: session_id,
                                model_id: &model_id,
                                model_sha256: &model_sha256,
                                capability_snapshot: &capability_snapshot,
                            },
                            &request,
                        )?;
                        send_host_remote_payload(
                            &mut relay,
                            &mut noise,
                            connection_id,
                            host_identity.host_id,
                            device_id,
                            MessageKind::SessionCatalog,
                            serde_json::to_value(catalog)?,
                        )
                        .await?;
                    }
                    MessageKind::ReplayRequest => {
                        let Some(replay_session_id) = envelope.session_id else {
                            connections.remove(&connection_id);
                            noise.disconnect(connection_id);
                            let _ = relay.disconnect_device(connection_id).await;
                            continue;
                        };
                        let request = decode_replay_request(&envelope)
                            .map_err(|_| anyhow::anyhow!("invalid replay request"))?;
                        let history = store
                            .lock()
                            .map_err(|_| anyhow::anyhow!("remote history unavailable"))?
                            .session_catalog_entry(&canonical_root, replay_session_id)
                            .map_err(|error| anyhow::anyhow!("remote history unavailable: {error}"))?;
                        let (replay, last_sequence, replay_state) = if replay_session_id == session_id {
                            let replay = host.replay_limit(request.after_sequence, request.limit)
                                .map_err(|error| anyhow::anyhow!("replay failed: {error}"))?;
                            let head = host.session_head()
                                .map_err(|error| anyhow::anyhow!("session head unavailable: {error}"))?;
                            (replay, head.last_event_sequence, remote_session_state(head.state))
                        } else if history.is_some() {
                            let store = store
                                .lock()
                                .map_err(|_| anyhow::anyhow!("remote history unavailable"))?;
                            let replay = store
                                .replay(replay_session_id, request.after_sequence, request.limit)
                                .map_err(|error| anyhow::anyhow!("history replay failed: {error}"))?;
                            let head = store
                                .session_head(replay_session_id)
                                .map_err(|error| anyhow::anyhow!("history head unavailable: {error}"))?;
                            (replay, head.last_event_sequence, remote_session_state(head.state))
                        } else if let Some(saved) = find_saved_agent_history(
                            &history_sandbox,
                            host_identity.host_id,
                            replay_session_id,
                        )? {
                            let last_sequence = saved.events.last().map_or(0, |event| event.sequence);
                            let replay = saved
                                .events
                                .into_iter()
                                .filter(|event| event.sequence > request.after_sequence)
                                .take(usize::from(request.limit))
                                .collect::<Vec<_>>();
                            (replay, last_sequence, RemoteSessionState::Closed)
                        } else {
                            connections.remove(&connection_id);
                            noise.disconnect(connection_id);
                            let _ = relay.disconnect_device(connection_id).await;
                            continue;
                        };
                        if !replay.is_empty() {
                            send_event_batches(
                                &mut relay,
                                &mut noise,
                                connection_id,
                                host_identity.host_id,
                                device_id,
                                replay_session_id,
                                &replay,
                            ).await?;
                        }
                        let returned = replay.last().map_or(request.after_sequence, |event| event.sequence);
                        send_remote_payload(
                            &mut relay,
                            &mut noise,
                            connection_id,
                            host_identity.host_id,
                            device_id,
                            replay_session_id,
                            MessageKind::ReplayEnd,
                            serde_json::to_value(ReplayEnd {
                                last_sequence,
                                has_more: returned < last_sequence,
                                session_state: replay_state,
                            })?,
                        ).await?;
                    }
                    MessageKind::Command => {
                        if envelope.session_id != Some(session_id) {
                            connections.remove(&connection_id);
                            noise.disconnect(connection_id);
                            let _ = relay.disconnect_device(connection_id).await;
                            continue;
                        }
                        let command = decode_command(&envelope)
                            .map_err(|_| anyhow::anyhow!("invalid remote command"))?;
                        let command_id = command_id(&command);
                        match command {
                            Command::StartTurn { .. } => {
                                match host.accept_start_message(&message) {
                                    Ok(remote_host::StartTurnAcceptance::Accepted(turn)) => {
                                        active_turn_device = Some(device_id);
                                        turn_sender.send(TurnJob {
                                            connection_id,
                                            command_id,
                                            turn,
                                            host: host.clone(),
                                        }).map_err(|_| anyhow::anyhow!("remote model worker stopped"))?;
                                        send_command_result(
                                            &mut relay, &mut noise, connection_id,
                                            host_identity.host_id, device_id, session_id, command_id,
                                            CommandStatus::Accepted, "accepted", "turn accepted for local execution",
                                            host.session_head().map(|head| head.last_event_sequence).unwrap_or(0),
                                        ).await?;
                                    }
                                    Ok(remote_host::StartTurnAcceptance::Duplicate(result)) => {
                                        send_stored_command_result(
                                            &mut relay, &mut noise, connection_id,
                                            host_identity.host_id, device_id, session_id,
                                            command_id, &result,
                                        ).await?;
                                    }
                                    Err(_) => {
                                        send_command_result(
                                            &mut relay, &mut noise, connection_id,
                                            host_identity.host_id, device_id, session_id, command_id,
                                            CommandStatus::Rejected, "session_busy", "session cannot accept this turn",
                                            host.session_head().map(|head| head.last_event_sequence).unwrap_or(0),
                                        ).await?;
                                    }
                                }
                            }
                            Command::ApprovalDecision { .. } => {
                                let result = host.approval_message(&message);
                                let (status, code, text) = if result.is_ok() {
                                    (CommandStatus::Applied, "applied", "approval decision applied")
                                } else {
                                    (CommandStatus::Rejected, "rejected", "approval decision rejected")
                                };
                                send_command_result(
                                    &mut relay, &mut noise, connection_id,
                                    host_identity.host_id, device_id, session_id, command_id,
                                    status, code, text,
                                    host.session_head().map(|head| head.last_event_sequence).unwrap_or(0),
                                ).await?;
                            }
                            Command::CancelTurn { .. } => {
                                let result = host.cancel_message(&message);
                                let (status, code, text) = if result.is_ok() {
                                    (CommandStatus::Applied, "applied", "turn cancellation applied")
                                } else {
                                    (CommandStatus::Rejected, "rejected", "turn cancellation rejected")
                                };
                                send_command_result(
                                    &mut relay, &mut noise, connection_id,
                                    host_identity.host_id, device_id, session_id, command_id,
                                    status, code, text,
                                    host.session_head().map(|head| head.last_event_sequence).unwrap_or(0),
                                ).await?;
                            }
                            Command::CreateSession {
                                command_id,
                                session_id: requested_session_id,
                            } => {
                                let request_digest = session_switch_digest(
                                    "create_session",
                                    command_id,
                                    requested_session_id,
                                )?;
                                let result = store
                                    .lock()
                                    .map_err(|_| anyhow::anyhow!("remote session store unavailable"))?
                                    .accept_create_session(AcceptCreateSession {
                                        device_id,
                                        command_id,
                                        request_digest: &request_digest,
                                        session_id: requested_session_id,
                                        canonical_root: &canonical_root,
                                        model_id: &model_id,
                                        model_sha256: &model_sha256,
                                        capability_snapshot_json: &capability_snapshot,
                                        created_at_unix_ms: unix_time_ms()?,
                                    });
                                match result {
                                    Ok(AcceptSessionSwitch::Applied(_)) => {
                                        let (next_host, next_events) = rebind_runtime_session(
                                            &store,
                                            &mut connections,
                                            requested_session_id,
                                            &canonical_root,
                                            &model_id,
                                            &model_sha256,
                                            &capability_snapshot,
                                            unix_time_ms()?,
                                        )?;
                                        send_command_result(
                                            &mut relay, &mut noise, connection_id,
                                            host_identity.host_id, device_id, session_id, command_id,
                                            CommandStatus::Applied, "session_created", "new agent session created",
                                            2,
                                        ).await?;
                                        session_id = requested_session_id;
                                        base_host = Some(next_host);
                                        events = Some(next_events);
                                        broadcast_session_catalog(
                                            &mut relay,
                                            &mut noise,
                                            &connections,
                                            &store,
                                            &history_sandbox,
                                            &canonical_root,
                                            session_id,
                                            &model_id,
                                            &model_sha256,
                                            &capability_snapshot,
                                            host_identity.host_id,
                                        ).await?;
                                    }
                                    Ok(AcceptSessionSwitch::Duplicate(result)) => {
                                        send_stored_command_result(
                                            &mut relay, &mut noise, connection_id,
                                            host_identity.host_id, device_id, session_id,
                                            command_id, &result,
                                        ).await?;
                                    }
                                    Err(_) => {
                                        send_command_result(
                                            &mut relay, &mut noise, connection_id,
                                            host_identity.host_id, device_id, session_id, command_id,
                                            CommandStatus::Rejected, "session_switch_rejected", "new session could not be created",
                                            host.session_head().map(|head| head.last_event_sequence).unwrap_or(0),
                                        ).await?;
                                    }
                                }
                            }
                            Command::ActivateSession {
                                command_id,
                                session_id: requested_session_id,
                            } => {
                                let request_digest = session_switch_digest(
                                    "activate_session",
                                    command_id,
                                    requested_session_id,
                                )?;
                                let result = store
                                    .lock()
                                    .map_err(|_| anyhow::anyhow!("remote session store unavailable"))?
                                    .accept_activate_session(AcceptActivateSession {
                                        device_id,
                                        command_id,
                                        request_digest: &request_digest,
                                        session_id: requested_session_id,
                                        canonical_root: &canonical_root,
                                        model_id: &model_id,
                                        model_sha256: &model_sha256,
                                        capability_snapshot_json: &capability_snapshot,
                                        activated_at_unix_ms: unix_time_ms()?,
                                    });
                                match result {
                                    Ok(AcceptSessionSwitch::Applied(_)) => {
                                        let head = store
                                            .lock()
                                            .map_err(|_| anyhow::anyhow!("remote session store unavailable"))?
                                            .session_head(requested_session_id)
                                            .map_err(|error| anyhow::anyhow!("activated session head unavailable: {error}"))?;
                                        let (next_host, next_events) = rebind_runtime_session(
                                            &store,
                                            &mut connections,
                                            requested_session_id,
                                            &canonical_root,
                                            &model_id,
                                            &model_sha256,
                                            &capability_snapshot,
                                            unix_time_ms()?,
                                        )?;
                                        send_command_result(
                                            &mut relay, &mut noise, connection_id,
                                            host_identity.host_id, device_id, session_id, command_id,
                                            CommandStatus::Applied, "session_activated", "agent session activated",
                                            head.last_event_sequence,
                                        ).await?;
                                        session_id = requested_session_id;
                                        base_host = Some(next_host);
                                        events = Some(next_events);
                                        broadcast_session_catalog(
                                            &mut relay,
                                            &mut noise,
                                            &connections,
                                            &store,
                                            &history_sandbox,
                                            &canonical_root,
                                            session_id,
                                            &model_id,
                                            &model_sha256,
                                            &capability_snapshot,
                                            host_identity.host_id,
                                        ).await?;
                                    }
                                    Ok(AcceptSessionSwitch::Duplicate(result)) => {
                                        send_stored_command_result(
                                            &mut relay, &mut noise, connection_id,
                                            host_identity.host_id, device_id, session_id,
                                            command_id, &result,
                                        ).await?;
                                    }
                                    Err(_) => {
                                        send_command_result(
                                            &mut relay, &mut noise, connection_id,
                                            host_identity.host_id, device_id, session_id, command_id,
                                            CommandStatus::Rejected, "session_switch_rejected", "agent session could not be activated",
                                            host.session_head().map(|head| head.last_event_sequence).unwrap_or(0),
                                        ).await?;
                                    }
                                }
                            }
                        }
                    }
                    MessageKind::Ping => {
                        if envelope.session_id != Some(session_id) {
                            connections.remove(&connection_id);
                            noise.disconnect(connection_id);
                            let _ = relay.disconnect_device(connection_id).await;
                            continue;
                        }
                        send_remote_payload(
                            &mut relay, &mut noise, connection_id,
                            host_identity.host_id, device_id, session_id,
                            MessageKind::Pong, serde_json::json!({}),
                        ).await?;
                    }
                    _ => {
                        connections.remove(&connection_id);
                        noise.disconnect(connection_id);
                        let _ = relay.disconnect_device(connection_id).await;
                    }
                }
            }
            completion = completion_receiver.recv() => {
                let Some(completion) = completion else {
                    anyhow::bail!("remote model worker stopped");
                };
                active_turn_device = None;
                if let Some(connection) = connections.get(&completion.connection_id) {
                    let (status, code, message) = match completion.result {
                        Ok(agent::LoopEnd::Answered) => (CommandStatus::Applied, "completed", "turn completed"),
                        Ok(agent::LoopEnd::Aborted) => (CommandStatus::Applied, "aborted", "turn aborted"),
                        Ok(agent::LoopEnd::StepCapped) => (CommandStatus::Applied, "step_capped", "turn reached its step limit"),
                        Ok(agent::LoopEnd::Repeated) => (CommandStatus::Applied, "repeated", "turn stopped after repeated work"),
                        Ok(agent::LoopEnd::DriverError) => (CommandStatus::Rejected, "driver_error", "local model driver failed"),
                        Err(_) => (CommandStatus::Rejected, "rejected", "turn was rejected by local authority"),
                    };
                    send_command_result(
                        &mut relay, &mut noise, completion.connection_id,
                        host_identity.host_id, connection.device_id, session_id,
                        completion.command_id, status, code, message,
                        completion.head.map(|head| head.last_event_sequence).unwrap_or(0),
                    ).await?;
                }
            }
            control = remote_management_commands.recv() => {
                let Some(control) = control else {
                    continue;
                };
                match control {
                    remote_control::RemoteManagementCommand::CreatePairingOffer { reply } => {
                        let result = pairing
                            .create_offer(
                                &pairing_relay_url(&options.relay_url),
                                &enrollment.route_id,
                                unix_time_ms()?,
                            )
                            .and_then(|offer| {
                                let qr_payload = serde_json::to_string(&offer)
                                    .map_err(|_| remote_pairing::PairingError::Unavailable)?;
                                remote_management
                                    .publish_pairing(Some(remote_control::RemotePairingStatus::Offered {
                                        expires_at_unix_ms: offer.expires_at_unix_ms,
                                    }))
                                    .map_err(|_| remote_pairing::PairingError::Unavailable)?;
                                Ok(remote_control::RemotePairingOffer {
                                    qr_payload,
                                    expires_at_unix_ms: offer.expires_at_unix_ms,
                                })
                            })
                            .map_err(|_| remote_control::RemoteManagementError::Rejected);
                        let _ = reply.send(result);
                    }
                    remote_control::RemoteManagementCommand::ConfirmPairing {
                        confirmation_id,
                        accepted,
                        reply,
                    } => {
                        let pending = pairing.status()
                            .ok()
                            .flatten()
                            .and_then(|status| match status {
                                remote_pairing::PairingStatus::AwaitingConfirmation {
                                    confirmation_id: expected_id,
                                    connection_id,
                                    authentication_fingerprint,
                                    ..
                                } if expected_id == confirmation_id => Some((connection_id, authentication_fingerprint)),
                                _ => None,
                            });
                        let result = match pending {
                            Some((connection_id, fingerprint)) if accepted => {
                                match remote_transport::finish_pairing_after_confirmation(
                                    &mut noise,
                                    &pairing,
                                    confirmation_id,
                                    connection_id,
                                    &fingerprint,
                                    true,
                                    session_id,
                                    unix_time_ms()?,
                                ) {
                                    Ok(accepted_pairing) => {
                                        if remote_management.publish_pairing(None).is_err() {
                                            Err(remote_control::RemoteManagementError::Unavailable)
                                        } else {
                                            match relay.send(accepted_pairing.response).await {
                                                Ok(()) => Ok(()),
                                                Err(_) => Err(remote_control::RemoteManagementError::Rejected),
                                            }
                                        }
                                    }
                                    Err(_) => Err(remote_control::RemoteManagementError::Rejected),
                                }
                            }
                            Some((connection_id, fingerprint)) => {
                                let rejected = pairing.confirm(
                                    confirmation_id,
                                    connection_id,
                                    &fingerprint,
                                    false,
                                    unix_time_ms()?,
                                );
                                noise.reject_pairing(connection_id);
                                let _ = relay.disconnect_device(connection_id).await;
                                let _ = remote_management.publish_pairing(None);
                                if matches!(rejected, Err(remote_pairing::PairingError::Rejected)) {
                                    Ok(())
                                } else {
                                    Err(remote_control::RemoteManagementError::Rejected)
                                }
                            }
                            None => Err(remote_control::RemoteManagementError::Rejected),
                        };
                        let _ = reply.send(result);
                    }
                    remote_control::RemoteManagementCommand::CancelPairing { reply } => {
                        let result = pairing.cancel()
                            .map_err(|_| remote_control::RemoteManagementError::Rejected);
                        match result {
                            Ok(connection_id) => {
                                if let Some(connection_id) = connection_id {
                                    noise.reject_pairing(connection_id);
                                    let _ = relay.disconnect_device(connection_id).await;
                                }
                                let _ = remote_management.publish_pairing(None);
                                let _ = reply.send(Ok(()));
                            }
                            Err(error) => {
                                let _ = reply.send(Err(error));
                            }
                        }
                    }
                    remote_control::RemoteManagementCommand::RevokeDevice { device_id, reply } => {
                        let result = noise.revoke_device(device_id, unix_time_ms()?)
                            .map_err(|_| remote_control::RemoteManagementError::Rejected);
                        match result {
                            Ok(connection_ids) => {
                                let mut cancellation_failed = false;
                                if active_turn_device == Some(device_id) {
                                    if let Some(host) = base_host.as_ref() {
                                        if host.cancel_locally().is_err() {
                                            cancellation_failed = true;
                                        }
                                    }
                                    active_turn_device = None;
                                }
                                for connection_id in connection_ids {
                                    connections.remove(&connection_id);
                                    let _ = relay.disconnect_device(connection_id).await;
                                }
                                let _ = reply.send(if cancellation_failed {
                                    Err(remote_control::RemoteManagementError::Rejected)
                                } else {
                                    Ok(())
                                });
                            }
                            Err(error) => {
                                let _ = reply.send(Err(error));
                            }
                        }
                    }
                    remote_control::RemoteManagementCommand::EmergencyDisable { reply } => {
                        if let Ok(Some(connection_id)) = pairing.cancel() {
                            noise.reject_pairing(connection_id);
                            let _ = relay.disconnect_device(connection_id).await;
                        }
                        let _ = remote_management.publish_pairing(None);
                        let result = store.lock()
                            .map_err(|_| remote_control::RemoteManagementError::Unavailable)
                            .and_then(|mut store| {
                                unix_time_ms()
                                    .map_err(|_| remote_control::RemoteManagementError::Rejected)
                                    .and_then(|now| store.revoke_all_devices(now).map_err(|_| remote_control::RemoteManagementError::Rejected))
                            });
                        match result {
                            Ok(_) => {
                                let mut cancellation_failed = false;
                                if let Some(host) = base_host.as_ref() {
                                    if host.cancel_locally().is_err() {
                                        cancellation_failed = true;
                                    }
                                }
                                active_turn_device = None;
                                let connection_ids = connections.keys().copied().collect::<Vec<_>>();
                                connections.clear();
                                noise.reset_for_reconnect();
                                for connection_id in connection_ids {
                                    let _ = relay.disconnect_device(connection_id).await;
                                }
                                let _ = reply.send(if cancellation_failed {
                                    Err(remote_control::RemoteManagementError::Rejected)
                                } else {
                                    Ok(())
                                });
                            }
                            Err(error) => {
                                let _ = reply.send(Err(error));
                            }
                        }
                    }
                }
            }
            _ = tick.tick() => {
                if tokio::time::Instant::now() >= next_keepalive {
                    if relay.ping().await.is_err() {
                        connections.clear();
                        if let Ok(Some(remote_pairing::PairingStatus::AwaitingConfirmation {
                            connection_id,
                            ..
                        })) = pairing.status() {
                            let _ = pairing.cancel_connection(connection_id);
                            let _ = remote_management.publish_pairing(None);
                        }
                        noise.reset_for_reconnect();
                        relay = match remote_transport::HostRelaySocket::connect_with_backoff(
                            &endpoint,
                            &enrollment.route_id,
                            &enrollment.host_capability,
                            reconnect_policy,
                            &cancellation,
                        )
                        .await {
                            Ok(relay) => relay,
                            Err(_) if cancellation.is_cancelled() => return Ok(0),
                            Err(error) => return Err(anyhow::anyhow!("relay reconnect failed: {error}")),
                        };
                        eprintln!("Relay connection restored after keepalive failure; devices must establish fresh Noise sessions.");
                    }
                    next_keepalive = tokio::time::Instant::now() + keepalive_interval;
                }
                let pairing_was_active = pairing.status()
                    .map(|status| status.is_some())
                    .unwrap_or(false);
                if let Ok(connection_id) = pairing.expire(unix_time_ms()?) {
                    if pairing_was_active && pairing.status().is_ok_and(|status| status.is_none()) {
                        if let Some(connection_id) = connection_id {
                            noise.reject_pairing(connection_id);
                            let _ = relay.disconnect_device(connection_id).await;
                        }
                        let _ = remote_management.publish_pairing(None);
                    }
                }
                let active_turn_revoked = active_turn_device_revoked(
                    &store,
                    active_turn_device,
                ).map_err(|error| {
                    anyhow::anyhow!("active remote device check failed: {error}")
                })?;
                let revoked = noise
                    .disconnect_revoked()
                    .map_err(|error| anyhow::anyhow!("device revocation check failed: {error}"))?;
                if active_turn_revoked {
                    if let Some(host) = base_host.as_ref() {
                        host.cancel_locally().map_err(|error| {
                            anyhow::anyhow!("local revocation cancellation failed: {error}")
                        })?;
                    }
                }
                for (connection_id, _) in revoked {
                    connections.remove(&connection_id);
                    let _ = relay.disconnect_device(connection_id).await;
                }
                let mut catalog_changed = false;
                if let Some(receiver) = events.as_ref() {
                    while let Ok(event) = receiver.try_recv() {
                        catalog_changed = true;
                        let recipients = connections
                            .iter()
                            .map(|(connection_id, connection)| (*connection_id, connection.device_id))
                            .collect::<Vec<_>>();
                        for (connection_id, device_id) in recipients {
                            if send_event_batch(
                                &mut relay, &mut noise, connection_id,
                                host_identity.host_id, device_id, session_id,
                                std::slice::from_ref(&event),
                            ).await.is_err() {
                                connections.remove(&connection_id);
                                noise.disconnect(connection_id);
                            }
                        }
                    }
                }
                if catalog_changed && !connections.is_empty() {
                    broadcast_session_catalog(
                        &mut relay,
                        &mut noise,
                        &connections,
                        &store,
                        &history_sandbox,
                        &canonical_root,
                        session_id,
                        &model_id,
                        &model_sha256,
                        &capability_snapshot,
                        host_identity.host_id,
                    ).await?;
                }
            }
            _ = cancellation.cancelled() => {
                if let Ok(Some(connection_id)) = pairing.cancel() {
                    noise.reject_pairing(connection_id);
                    let _ = relay.disconnect_device(connection_id).await;
                }
                let _ = remote_management.publish_pairing(None);
                if let Some(host) = base_host.as_ref() {
                    host.cancel_locally().map_err(|error| {
                        anyhow::anyhow!("local shutdown cancellation failed: {error}")
                    })?;
                }
                return Ok(0);
            }
        }
    }
}

fn active_turn_device_revoked(
    store: &Arc<Mutex<RemoteStore>>,
    active_turn_device: Option<Uuid>,
) -> Result<bool, camelid_remote_store::StoreError> {
    let Some(active_turn_device) = active_turn_device else {
        return Ok(false);
    };
    let devices = store
        .lock()
        .map_err(|_| camelid_remote_store::StoreError::Unavailable)?
        .devices()?;
    Ok(!devices.iter().any(|device| {
        device.device_id == active_turn_device && device.revoked_at_unix_ms.is_none()
    }))
}

struct SessionCatalogContext<'a> {
    store: &'a Arc<Mutex<RemoteStore>>,
    sandbox: &'a tools::Sandbox,
    host_id: Uuid,
    canonical_root: &'a str,
    active_session_id: Uuid,
    model_id: &'a str,
    model_sha256: &'a str,
    capability_snapshot: &'a str,
}

fn build_session_catalog(
    context: &SessionCatalogContext<'_>,
    request: &camelid_remote_protocol::SessionCatalogRequest,
) -> anyhow::Result<SessionCatalog> {
    let mut entries = Vec::new();
    let mut cursor = None;
    loop {
        let page = context
            .store
            .lock()
            .map_err(|_| anyhow::anyhow!("remote session catalog unavailable"))?
            .list_session_catalog(context.canonical_root, cursor, 65)
            .map_err(|error| anyhow::anyhow!("remote session catalog unavailable: {error}"))?;
        if page.is_empty() {
            break;
        }
        if entries.len().saturating_add(page.len()) > MAX_HOST_SESSION_HISTORIES {
            anyhow::bail!("remote session catalog exceeds the supported history limit");
        }
        cursor = page
            .last()
            .map(|entry| (entry.updated_at_unix_ms, entry.session_id));
        entries.extend(page);
        if entries.len() % 65 != 0 {
            break;
        }
    }

    let mut summaries = entries
        .iter()
        .map(|entry| {
            session_catalog_summary(
                entry,
                context.active_session_id,
                context.model_id,
                context.model_sha256,
                context.capability_snapshot,
            )
        })
        .collect::<Vec<_>>();
    summaries.extend(
        saved_agent_histories(context.sandbox, context.host_id)?
            .into_iter()
            .map(|saved| saved.summary),
    );
    summaries.sort_by(|left, right| {
        right
            .updated_at_unix_ms
            .cmp(&left.updated_at_unix_ms)
            .then_with(|| left.history_id.as_bytes().cmp(right.history_id.as_bytes()))
    });
    if summaries.len() > MAX_HOST_SESSION_HISTORIES {
        anyhow::bail!("combined agent session catalog exceeds the supported history limit");
    }

    let revision_material = summaries
        .iter()
        .map(|entry| {
            serde_json::json!({
                "history_id": entry.history_id,
                "source": entry.source,
                "state": entry.state,
                "updated_at_unix_ms": entry.updated_at_unix_ms,
                "last_event_sequence": entry.last_event_sequence,
            })
        })
        .collect::<Vec<_>>();
    let revision = format!(
        "sha256:{:x}",
        Sha256::digest(serde_json::to_vec(&revision_material)?)
    );
    if request
        .revision
        .as_deref()
        .is_some_and(|expected| expected != revision)
    {
        anyhow::bail!("remote session catalog changed during pagination");
    }

    let start = match request.cursor.as_ref() {
        Some(cursor) => summaries
            .iter()
            .position(|entry| {
                entry.updated_at_unix_ms == cursor.updated_at_unix_ms
                    && entry.history_id == cursor.history_id
            })
            .map(|index| index + 1)
            .ok_or_else(|| anyhow::anyhow!("remote session catalog cursor is stale"))?,
        None => 0,
    };
    let end = start
        .saturating_add(usize::from(request.limit))
        .min(summaries.len());
    let sessions = summaries[start..end].to_vec();
    let next_cursor = (end < summaries.len()).then(|| SessionCatalogCursor {
        updated_at_unix_ms: summaries[end - 1].updated_at_unix_ms,
        history_id: summaries[end - 1].history_id,
    });
    Ok(SessionCatalog {
        active_session_id: context.active_session_id,
        revision,
        sessions,
        next_cursor,
    })
}

fn session_catalog_summary(
    entry: &StoredSessionCatalogEntry,
    active_session_id: Uuid,
    model_id: &str,
    model_sha256: &str,
    capability_snapshot: &str,
) -> SessionSummary {
    let active = entry.session_id == active_session_id;
    let identity_matches = entry.model_id == model_id
        && entry.model_sha256 == model_sha256
        && entry.capability_snapshot_json == capability_snapshot;
    let state_continuable = matches!(
        entry.state,
        SessionState::Armed | SessionState::Idle | SessionState::Failed
    );
    let continuable = identity_matches && state_continuable;
    let refusal_code = if continuable {
        None
    } else if !identity_matches {
        Some("model_or_capability_identity_mismatch".into())
    } else {
        Some("session_state_not_continuable".into())
    };
    SessionSummary {
        history_id: entry.session_id,
        source: SessionHistorySource::Remote,
        title: truncate_utf8(&entry.title, 256),
        state: remote_session_state(entry.state),
        canonical_root: entry.canonical_root.clone(),
        model_id: entry.model_id.clone(),
        model_sha256: Some(entry.model_sha256.clone()),
        created_at_unix_ms: entry.created_at_unix_ms,
        updated_at_unix_ms: entry.updated_at_unix_ms,
        last_event_sequence: entry.last_event_sequence,
        active,
        continuable,
        refusal_code,
    }
}

struct SavedAgentHistory {
    summary: SessionSummary,
    events: Vec<StoredEvent>,
}

fn saved_agent_histories(
    sandbox: &tools::Sandbox,
    host_id: Uuid,
) -> anyhow::Result<Vec<SavedAgentHistory>> {
    let mut histories = Vec::new();
    for saved_id in agent_session::list(sandbox) {
        let path =
            agent_session::path_for(sandbox, &saved_id).map_err(|error| anyhow::anyhow!(error))?;
        let raw = std::fs::read(&path)?;
        let saved =
            agent_session::load(sandbox, &saved_id).map_err(|error| anyhow::anyhow!(error))?;
        let history_id = deterministic_history_id(
            host_id,
            &sandbox.root().display().to_string(),
            &saved_id,
            &raw,
        );
        let updated_at_unix_ms = std::fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
            .and_then(|duration| u64::try_from(duration.as_millis()).ok())
            .unwrap_or(0);
        let events = saved_agent_events(history_id, updated_at_unix_ms, &saved.transcript);
        let title = saved
            .transcript
            .iter()
            .find_map(|message| match message {
                agent::AgentMsg::User(text) => Some(truncate_utf8(text, 256)),
                _ => None,
            })
            .unwrap_or_else(|| saved.id.clone());
        histories.push(SavedAgentHistory {
            summary: SessionSummary {
                history_id,
                source: SessionHistorySource::AgentSaved,
                title,
                state: RemoteSessionState::Closed,
                canonical_root: saved.workspace,
                model_id: saved.model_id,
                model_sha256: None,
                created_at_unix_ms: updated_at_unix_ms,
                updated_at_unix_ms,
                last_event_sequence: events.last().map_or(0, |event| event.sequence),
                active: false,
                continuable: false,
                refusal_code: Some("model_artifact_identity_unavailable".into()),
            },
            events,
        });
    }
    Ok(histories)
}

fn saved_agent_events(
    history_id: Uuid,
    created_at_unix_ms: u64,
    transcript: &[agent::AgentMsg],
) -> Vec<StoredEvent> {
    let mut events = Vec::new();
    let mut pending_calls = VecDeque::new();
    for message in transcript {
        match message {
            agent::AgentMsg::User(content) => push_saved_event(
                &mut events,
                history_id,
                created_at_unix_ms,
                "user.message",
                serde_json::json!({"content": content}),
            ),
            agent::AgentMsg::Assistant(content) => push_saved_event(
                &mut events,
                history_id,
                created_at_unix_ms,
                "model.answer",
                serde_json::json!({"content": content}),
            ),
            agent::AgentMsg::ToolCalls(calls) => {
                for call in calls {
                    let call_id = deterministic_child_id(history_id, events.len() as u64 + 1);
                    pending_calls.push_back((call.name.clone(), call_id));
                    push_saved_event(
                        &mut events,
                        history_id,
                        created_at_unix_ms,
                        "tool.call",
                        serde_json::json!({
                            "call_id": call_id,
                            "tool": call.name,
                            "detail": serde_json::to_string(&call.args).unwrap_or_else(|_| "{}".into()),
                        }),
                    );
                }
            }
            agent::AgentMsg::ToolResult { name, outcome } => {
                let call_id = pending_calls
                    .iter()
                    .position(|(pending_name, _)| pending_name == name)
                    .and_then(|index| pending_calls.remove(index))
                    .map(|(_, call_id)| call_id)
                    .unwrap_or_else(|| deterministic_child_id(history_id, events.len() as u64 + 1));
                push_saved_event(
                    &mut events,
                    history_id,
                    created_at_unix_ms,
                    "tool.result",
                    serde_json::json!({
                        "call_id": call_id,
                        "content": outcome.text(),
                        "is_error": outcome.is_err(),
                    }),
                );
            }
            agent::AgentMsg::System(_)
            | agent::AgentMsg::Memory(_)
            | agent::AgentMsg::Summary(_) => {}
        }
    }
    events
}

fn push_saved_event(
    events: &mut Vec<StoredEvent>,
    history_id: Uuid,
    created_at_unix_ms: u64,
    event_type: &str,
    payload: serde_json::Value,
) {
    let sequence = events.len() as u64 + 1;
    events.push(StoredEvent {
        sequence,
        event_id: deterministic_child_id(history_id, sequence),
        turn_id: None,
        event_type: event_type.into(),
        payload,
        created_at_unix_ms,
    });
}

fn deterministic_history_id(host_id: Uuid, root: &str, saved_id: &str, raw: &[u8]) -> Uuid {
    let mut digest = Sha256::new();
    digest.update(host_id.as_bytes());
    digest.update(root.as_bytes());
    digest.update(saved_id.as_bytes());
    digest.update(raw);
    uuid_from_digest(digest.finalize().into())
}

fn deterministic_child_id(history_id: Uuid, sequence: u64) -> Uuid {
    let mut digest = Sha256::new();
    digest.update(history_id.as_bytes());
    digest.update(sequence.to_be_bytes());
    uuid_from_digest(digest.finalize().into())
}

fn uuid_from_digest(mut bytes: [u8; 32]) -> Uuid {
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes[..16].try_into().expect("fixed digest prefix"))
}

fn find_saved_agent_history(
    sandbox: &tools::Sandbox,
    host_id: Uuid,
    history_id: Uuid,
) -> anyhow::Result<Option<SavedAgentHistory>> {
    Ok(saved_agent_histories(sandbox, host_id)?
        .into_iter()
        .find(|history| history.summary.history_id == history_id))
}

fn truncate_utf8(value: &str, maximum_bytes: usize) -> String {
    if value.len() <= maximum_bytes {
        return value.to_string();
    }
    let mut end = maximum_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_string()
}

#[allow(clippy::too_many_arguments)]
fn rebind_runtime_session(
    store: &Arc<Mutex<RemoteStore>>,
    connections: &mut HashMap<Uuid, RemoteConnection>,
    session_id: Uuid,
    canonical_root: &str,
    model_id: &str,
    model_sha256: &str,
    capability_snapshot: &str,
    first_timestamp: u64,
) -> anyhow::Result<(remote_host::LocalRemoteHost, mpsc::Receiver<StoredEvent>)> {
    let first = connections
        .values()
        .next()
        .ok_or_else(|| anyhow::anyhow!("session switch requires an authenticated device"))?;
    let base = remote_host::LocalRemoteHost::from_shared(
        Arc::clone(store),
        session_id,
        first.device_id,
        first.device_key,
        remote_host::HostIdentity {
            canonical_root,
            model_id,
            model_sha256,
            capability_snapshot_json: capability_snapshot,
        },
        first_timestamp,
    )
    .map_err(|error| anyhow::anyhow!("active remote session could not be loaded: {error}"))?;
    let events = base
        .subscribe()
        .map_err(|error| anyhow::anyhow!("remote events unavailable: {error}"))?;
    for connection in connections.values_mut() {
        connection.host = base.for_device(connection.device_id, connection.device_key);
        connection.reassembler = camelid_remote_protocol::ChunkReassembler::default();
    }
    Ok((base, events))
}

#[allow(clippy::too_many_arguments)]
async fn broadcast_session_catalog(
    relay: &mut remote_transport::HostRelaySocket,
    noise: &mut remote_transport::AuthorizedNoiseSessions,
    connections: &HashMap<Uuid, RemoteConnection>,
    store: &Arc<Mutex<RemoteStore>>,
    sandbox: &tools::Sandbox,
    canonical_root: &str,
    active_session_id: Uuid,
    model_id: &str,
    model_sha256: &str,
    capability_snapshot: &str,
    host_id: Uuid,
) -> anyhow::Result<()> {
    let catalog = build_session_catalog(
        &SessionCatalogContext {
            store,
            sandbox,
            host_id,
            canonical_root,
            active_session_id,
            model_id,
            model_sha256,
            capability_snapshot,
        },
        &camelid_remote_protocol::SessionCatalogRequest {
            cursor: None,
            limit: camelid_remote_protocol::MAX_SESSION_CATALOG_ENTRIES,
            revision: None,
        },
    )?;
    let payload = serde_json::to_value(catalog)?;
    for (connection_id, connection) in connections {
        send_host_remote_payload(
            relay,
            noise,
            *connection_id,
            host_id,
            connection.device_id,
            MessageKind::SessionCatalog,
            payload.clone(),
        )
        .await?;
    }
    Ok(())
}

fn command_id(command: &Command) -> Uuid {
    match command {
        Command::StartTurn { command_id, .. }
        | Command::ApprovalDecision { command_id, .. }
        | Command::CancelTurn { command_id, .. }
        | Command::CreateSession { command_id, .. }
        | Command::ActivateSession { command_id, .. } => *command_id,
    }
}

fn session_switch_digest(
    command: &str,
    command_id: Uuid,
    session_id: Uuid,
) -> anyhow::Result<String> {
    let canonical = camelid_remote_protocol::canonical_json(&serde_json::json!({
        "command": command,
        "command_id": command_id,
        "session_id": session_id,
    }))?;
    Ok(format!("sha256:{:x}", Sha256::digest(canonical)))
}

fn pairing_relay_url(relay_url: &str) -> String {
    format!("{}/v1/connect", relay_url.trim_end_matches('/'))
}

#[cfg(test)]
mod remote_pairing_qr_tests {
    use super::{partition_event_batches, StoredEvent, Uuid};

    fn event(sequence: u64, content_bytes: usize) -> StoredEvent {
        StoredEvent {
            sequence,
            event_id: Uuid::new_v4(),
            turn_id: None,
            event_type: "model.delta".into(),
            payload: serde_json::json!({"content":"x".repeat(content_bytes)}),
            created_at_unix_ms: sequence,
        }
    }

    #[test]
    fn replay_partition_preserves_order_and_respects_message_bytes() {
        let events = vec![event(1, 600_000), event(2, 600_000), event(3, 16)];
        let batches = partition_event_batches(&events).unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(
            batches
                .iter()
                .flat_map(|batch| batch.iter().map(|event| event.sequence))
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn replay_partition_refuses_one_oversized_event() {
        let events = vec![event(1, camelid_remote_protocol::MAX_INNER_MESSAGE_BYTES)];
        assert!(partition_event_batches(&events).is_err());
    }
}

#[cfg(test)]
mod remote_admin_tests {
    use std::sync::{Arc, Mutex};

    use camelid_remote_store::RemoteStore;
    use uuid::Uuid;

    use super::{
        active_turn_device_revoked, disable_remote_devices, revoke_remote_device,
        RemoteAdminOptions,
    };

    #[test]
    fn external_revocation_cancels_even_without_a_live_noise_session() {
        let directory = tempfile::tempdir().unwrap();
        let database_path = directory.path().join("remote.sqlite3");
        let device_id = Uuid::new_v4();
        let mut store = RemoteStore::open(&database_path).unwrap();
        store
            .register_device(device_id, "phone", &[7; 32], 1)
            .unwrap();
        let shared = Arc::new(Mutex::new(store));

        assert!(!active_turn_device_revoked(&shared, Some(device_id)).unwrap());
        RemoteStore::open(&database_path)
            .unwrap()
            .revoke_device(device_id, 2)
            .unwrap();
        assert!(active_turn_device_revoked(&shared, Some(device_id)).unwrap());
    }

    #[test]
    fn admin_revoke_and_disable_share_the_host_authority_database() {
        let directory = tempfile::tempdir().unwrap();
        let database_path = directory.path().join("remote.sqlite3");
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        let mut store = RemoteStore::open(&database_path).unwrap();
        store.register_device(first, "first", &[1; 32], 1).unwrap();
        store
            .register_device(second, "second", &[2; 32], 2)
            .unwrap();
        drop(store);

        revoke_remote_device(
            RemoteAdminOptions {
                db_path: Some(database_path.clone()),
            },
            first,
        )
        .unwrap();
        disable_remote_devices(RemoteAdminOptions {
            db_path: Some(database_path.clone()),
        })
        .unwrap();

        let devices = RemoteStore::open(&database_path)
            .unwrap()
            .devices()
            .unwrap();
        assert!(devices
            .iter()
            .all(|device| device.revoked_at_unix_ms.is_some()));
    }
}

async fn enroll_and_persist_relay(
    endpoint: &remote_transport::HostRelayEndpoint,
    enrollment_token: &str,
    relay_url: &str,
    store: &mut RemoteStore,
    secrets: &remote_identity::ProtectedFileSecretStore,
    previous: Option<&camelid_remote_store::StoredRelayBinding>,
) -> anyhow::Result<remote_transport::RelayEnrollment> {
    let enrollment = remote_transport::enroll_route(endpoint, enrollment_token)
        .await
        .map_err(|error| anyhow::anyhow!("relay enrollment failed: {error}"))?;
    let secret_reference = format!("dpapi-file:v1:{}", Uuid::new_v4());
    secrets
        .store_bytes(&secret_reference, enrollment.host_capability.as_bytes())
        .map_err(|error| anyhow::anyhow!("relay capability protection failed: {error}"))?;
    if let Err(error) = store.set_relay_binding(
        relay_url,
        &enrollment.route_id,
        &secret_reference,
        unix_time_ms()?,
    ) {
        let _ = secrets.delete_bytes(&secret_reference);
        return Err(anyhow::anyhow!("relay binding persistence failed: {error}"));
    }
    if let Some(previous) = previous {
        if previous.capability_secret_reference != secret_reference {
            let _ = secrets.delete_bytes(&previous.capability_secret_reference);
        }
    }
    Ok(enrollment)
}

fn remote_session_state(state: SessionState) -> RemoteSessionState {
    match state {
        SessionState::Armed => RemoteSessionState::Armed,
        SessionState::Idle => RemoteSessionState::Idle,
        SessionState::Running => RemoteSessionState::Running,
        SessionState::WaitingApproval => RemoteSessionState::WaitingApproval,
        SessionState::Cancelling => RemoteSessionState::Cancelling,
        SessionState::Failed => RemoteSessionState::Failed,
        SessionState::Closed => RemoteSessionState::Closed,
    }
}

fn remote_event(event: &StoredEvent) -> RemoteEvent {
    RemoteEvent {
        sequence: event.sequence,
        event_id: event.event_id,
        turn_id: event.turn_id,
        event: event.event_type.clone(),
        created_at_unix_ms: event.created_at_unix_ms,
        payload: event.payload.clone(),
    }
}

#[allow(clippy::too_many_arguments)]
async fn send_command_result(
    relay: &mut remote_transport::HostRelaySocket,
    noise: &mut remote_transport::AuthorizedNoiseSessions,
    connection_id: Uuid,
    host_id: Uuid,
    device_id: Uuid,
    session_id: Uuid,
    command_id: Uuid,
    status: CommandStatus,
    code: &str,
    message: &str,
    current_event_sequence: u64,
) -> anyhow::Result<()> {
    send_remote_payload(
        relay,
        noise,
        connection_id,
        host_id,
        device_id,
        session_id,
        MessageKind::CommandResult,
        serde_json::to_value(ProtocolCommandResult {
            command_id,
            status,
            code: code.into(),
            message: message.into(),
            current_event_sequence,
        })?,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn send_stored_command_result(
    relay: &mut remote_transport::HostRelaySocket,
    noise: &mut remote_transport::AuthorizedNoiseSessions,
    connection_id: Uuid,
    host_id: Uuid,
    device_id: Uuid,
    session_id: Uuid,
    command_id: Uuid,
    result: &camelid_remote_store::CommandResult,
) -> anyhow::Result<()> {
    let payload: serde_json::Value = serde_json::from_str(&result.response_json)
        .map_err(|_| anyhow::anyhow!("stored command result is invalid"))?;
    let status = match result.status.as_str() {
        "accepted" => CommandStatus::Accepted,
        "applied" => CommandStatus::Applied,
        "rejected" => CommandStatus::Rejected,
        _ => anyhow::bail!("stored command status is invalid"),
    };
    let code = payload
        .get("code")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("ok");
    let sequence = payload
        .get("current_event_sequence")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    send_command_result(
        relay,
        noise,
        connection_id,
        host_id,
        device_id,
        session_id,
        command_id,
        status,
        code,
        "idempotent command result",
        sequence,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn send_event_batch(
    relay: &mut remote_transport::HostRelaySocket,
    noise: &mut remote_transport::AuthorizedNoiseSessions,
    connection_id: Uuid,
    host_id: Uuid,
    device_id: Uuid,
    session_id: Uuid,
    events: &[StoredEvent],
) -> anyhow::Result<()> {
    let batch = EventBatch {
        events: events.iter().map(remote_event).collect(),
    };
    batch.validate()?;
    send_remote_payload(
        relay,
        noise,
        connection_id,
        host_id,
        device_id,
        session_id,
        MessageKind::EventBatch,
        serde_json::to_value(batch)?,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn send_event_batches(
    relay: &mut remote_transport::HostRelaySocket,
    noise: &mut remote_transport::AuthorizedNoiseSessions,
    connection_id: Uuid,
    host_id: Uuid,
    device_id: Uuid,
    session_id: Uuid,
    events: &[StoredEvent],
) -> anyhow::Result<()> {
    for batch in partition_event_batches(events)? {
        send_event_batch(
            relay,
            noise,
            connection_id,
            host_id,
            device_id,
            session_id,
            batch,
        )
        .await?;
    }
    Ok(())
}

fn partition_event_batches(events: &[StoredEvent]) -> anyhow::Result<Vec<&[StoredEvent]>> {
    const ENVELOPE_RESERVE_BYTES: usize = 2048;
    let limit = camelid_remote_protocol::MAX_INNER_MESSAGE_BYTES - ENVELOPE_RESERVE_BYTES;
    let mut batches = Vec::new();
    let mut start = 0;
    while start < events.len() {
        let mut end = start;
        while end < events.len()
            && end - start < usize::from(camelid_remote_protocol::MAX_REPLAY_EVENTS)
        {
            let candidate = EventBatch {
                events: events[start..=end].iter().map(remote_event).collect(),
            };
            if serde_json::to_vec(&candidate)?.len() > limit {
                break;
            }
            end += 1;
        }
        if end == start {
            anyhow::bail!("one remote event exceeds the inner message limit");
        }
        batches.push(&events[start..end]);
        start = end;
    }
    Ok(batches)
}

#[allow(clippy::too_many_arguments)]
async fn send_remote_payload(
    relay: &mut remote_transport::HostRelaySocket,
    noise: &mut remote_transport::AuthorizedNoiseSessions,
    connection_id: Uuid,
    host_id: Uuid,
    device_id: Uuid,
    session_id: Uuid,
    kind: MessageKind,
    payload: serde_json::Value,
) -> anyhow::Result<()> {
    send_scoped_remote_payload(
        relay,
        noise,
        connection_id,
        host_id,
        device_id,
        Some(session_id),
        kind,
        payload,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn send_host_remote_payload(
    relay: &mut remote_transport::HostRelaySocket,
    noise: &mut remote_transport::AuthorizedNoiseSessions,
    connection_id: Uuid,
    host_id: Uuid,
    device_id: Uuid,
    kind: MessageKind,
    payload: serde_json::Value,
) -> anyhow::Result<()> {
    send_scoped_remote_payload(
        relay,
        noise,
        connection_id,
        host_id,
        device_id,
        None,
        kind,
        payload,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn send_scoped_remote_payload(
    relay: &mut remote_transport::HostRelaySocket,
    noise: &mut remote_transport::AuthorizedNoiseSessions,
    connection_id: Uuid,
    host_id: Uuid,
    device_id: Uuid,
    session_id: Option<Uuid>,
    kind: MessageKind,
    payload: serde_json::Value,
) -> anyhow::Result<()> {
    let message_id = Uuid::new_v4();
    let message = RemoteMessage {
        protocol: PROTOCOL.into(),
        message_id,
        kind,
        host_id,
        device_id,
        session_id,
        sent_at_unix_ms: unix_time_ms()?,
        payload,
    };
    let encoded = serde_json::to_vec(&message)?;
    for chunk in encode_chunks(message_id, &encoded)? {
        let sealed = noise
            .seal(connection_id, &chunk)
            .map_err(|error| anyhow::anyhow!("Noise seal failed: {error}"))?;
        relay
            .send(sealed)
            .await
            .map_err(|error| anyhow::anyhow!("relay send failed: {error}"))?;
    }
    Ok(())
}

fn remote_data_root() -> anyhow::Result<PathBuf> {
    #[cfg(windows)]
    {
        let root = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .ok_or_else(|| anyhow::anyhow!("LOCALAPPDATA is unavailable"))?;
        Ok(root.join("Camelid").join("remote"))
    }
    #[cfg(not(windows))]
    {
        if let Some(root) = std::env::var_os("XDG_DATA_HOME") {
            return Ok(PathBuf::from(root).join("camelid").join("remote"));
        }
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or_else(|| anyhow::anyhow!("HOME is unavailable"))?;
        Ok(home
            .join(".local")
            .join("share")
            .join("camelid")
            .join("remote"))
    }
}

fn unix_time_ms() -> anyhow::Result<u64> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}

fn sha256_file(path: &std::path::Path) -> anyhow::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    std::io::copy(&mut file, &mut digest)?;
    Ok(format!("{:x}", digest.finalize()))
}

/// Parsed `camelid chat` flags.
pub struct ChatOptions {
    pub model: Option<PathBuf>,
    pub addr: SocketAddr,
    pub system: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
    pub no_stream: bool,
    pub models_dir: PathBuf,
    /// Force the inline line REPL instead of the full-screen TUI.
    pub plain: bool,
    /// Enter agent mode (tool-calling loop) instead of plain chat.
    pub agent: bool,
    /// Sandbox root for agent tools (default: cwd).
    pub workdir: Option<PathBuf>,
    pub max_steps: usize,
    pub auto_approve: bool,
    /// `--yolo` (unattended): auto-approve EXEC tools too so the agent runs a
    /// whole task without prompting. Refused under production.
    pub yolo: bool,
    pub allow_net: bool,
    /// `--allow-fs`: agent file tools may read/write anywhere on disk (still
    /// approval-gated), not just under the workspace root.
    pub allow_fs: bool,
    /// `--allow-mcp`: load MCP servers from `camelid.mcp.json` and offer their
    /// tools. Off by default; refused under production.
    pub allow_mcp: bool,
    pub shell_timeout: u64,
    /// Opt-in thinking mode (`chat --enable-thinking`): the model emits its own
    /// `<think>…</think>` reasoning. NOT parity-locked (leading-trace lane only).
    pub enable_thinking: bool,
    /// Audit webhook URL (`--audit-webhook` / `CAMELID_AUDIT_WEBHOOK`). When unset,
    /// the agent uses the no-op sink and emits nothing.
    pub audit_webhook: Option<String>,
    /// `run_shell` confinement: `disabled` | `sandboxed` (default) | `unrestricted`.
    pub shell_sandbox: String,
    /// Headless one-shot (`camelid agent exec`): run this goal to completion and
    /// exit, instead of opening a REPL. Implies `agent` + `plain`.
    pub exec_goal: Option<String>,
}

/// Entry point for the `Chat` subcommand. Returns a process exit code (0 = ok,
/// non-zero for the typed unsupported-state backstop) so the caller can exit
/// after this function's `ServerHandle` has torn down any spawned server.
pub fn run_chat(opts: ChatOptions) -> anyhow::Result<i32> {
    init_terminal();
    install_sigint_handler();

    let client = Client::new(opts.addr);
    let server = ServerHandle::ensure(opts.addr, &client)?;
    let spawned = server.spawned();

    let settings = Settings {
        temperature: opts.temperature,
        top_p: opts.top_p,
        top_k: opts.top_k,
        max_tokens: opts.max_tokens,
        seed: opts.seed,
        stream: !opts.no_stream,
        enable_thinking: opts.enable_thinking,
    };
    let mut session = Session::new(client, opts.models_dir, settings, opts.system);

    // --model backstop: load + classify before any UI, so an unsupported GGUF
    // exits with the typed error and no screen takeover. Loading a cold GGUF can
    // take several seconds, so give feedback before the UI takes the screen. A
    // known supported GGUF is labeled with its ledger id (so posture + the agent
    // tool-capable gate match), exactly like the picker.
    if let Some(model) = &opts.model {
        eprintln!("Loading {} …", model.display());
        let label = catalog_label_for(model);
        let posture = label.as_ref().map(|_| "supported");
        match session.load_model_file(model, label.as_deref(), posture)? {
            LoadResult::Loaded => {}
            LoadResult::Unsupported(message) => {
                eprintln!("{message}");
                return Ok(1);
            }
        }
    }

    // Agent mode: a tool-calling loop (line renderer), gated to tool-capable rows.
    if opts.agent {
        if !session.has_model() {
            eprintln!("agent mode needs a model — pass --model <gguf>");
            return Ok(2);
        }
        let shell_sandbox = match opts.shell_sandbox.parse::<shell_sandbox::ShellSandbox>() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{e}");
                return Ok(2);
            }
        };
        let cfg = agent::AgentConfig {
            workdir: opts.workdir.unwrap_or_else(|| PathBuf::from(".")),
            max_steps: opts.max_steps,
            auto_approve: opts.auto_approve,
            yolo: opts.yolo,
            allow_net: opts.allow_net,
            allow_fs: opts.allow_fs,
            shell_timeout: std::time::Duration::from_secs(opts.shell_timeout),
            max_tokens: opts.max_tokens,
            temperature: opts.temperature,
            audit: audit::sink_from_config(opts.audit_webhook.as_deref()),
            shell_sandbox,
            tool_profile: tools::ToolProfile::Full,
            // The smaller of what the model was trained for and what the agent
            // lane is validated to; falls back to the validated ceiling when the
            // server has not reported a context length.
            ctx_budget: Some(
                session
                    .active_ctx
                    .unwrap_or(agent::AGENT_VALIDATED_CTX)
                    .min(agent::AGENT_VALIDATED_CTX),
            ),
        };
        // MCP servers, if the user opted in. A broken MCP config costs you MCP,
        // not your session, so problems are reported and the agent still runs.
        if opts.allow_mcp {
            match tools::Sandbox::new(&cfg.workdir, cfg.allow_net, cfg.shell_timeout) {
                Ok(sb) => {
                    let native: Vec<String> = tools::specs(cfg.allow_net, shell_sandbox)
                        .into_iter()
                        .map(|t| t.name)
                        .collect();
                    match mcp::configure(&sb, true, agent::is_production(), &native) {
                        Ok(0) => eprintln!(
                            "--allow-mcp: no MCP tools loaded (no {} at the workspace root?)",
                            mcp::CONFIG_FILE
                        ),
                        Ok(n) => eprintln!("MCP: {n} tool(s) loaded — each is approval-gated"),
                        Err(e) => eprintln!("MCP: {e}"),
                    }
                }
                Err(e) => eprintln!("MCP: workspace unavailable: {e}"),
            }
        }

        // Headless one-shot: no REPL, tri-state exit, answer on stdout.
        if let Some(goal) = opts.exec_goal.as_deref() {
            let code = agent::run_exec(&mut session, opts.addr, cfg, goal);
            mcp::shutdown();
            return code;
        }

        // Full-screen TUI agent on a real terminal (default); the line renderer
        // is the fallback for --plain, pipes, and non-TTY runs (smoke/tests).
        let interactive = std::io::stdout().is_terminal() && std::io::stdin().is_terminal();
        let code = if interactive && !opts.plain {
            agent_tui::run(&mut session, opts.addr, cfg)
        } else {
            agent::run_agent(&mut session, opts.addr, cfg)
        };
        mcp::shutdown();
        return code;
    }

    // Full-screen TUI when we have a real terminal on both ends and the user did
    // not ask for plain mode; otherwise the inline REPL.
    let interactive = std::io::stdout().is_terminal() && std::io::stdin().is_terminal();
    if interactive && !opts.plain {
        tui::run(&mut session, opts.addr, spawned)?;
    } else {
        inline::run(&mut session, opts.addr, spawned)?;
    }
    Ok(0)
}

/// Resolve an exact supported artifact filename to its compatibility row so a
/// `--model` load carries the same ledger identity used by agent admission.
fn catalog_label_for(model: &std::path::Path) -> Option<String> {
    let name = model.file_name()?.to_str()?;
    crate::api::supported_compatibility_row_id_for_filename(name).map(str::to_string)
}

#[cfg(test)]
mod remote_model_identity_tests {
    use super::{catalog_label_for, pairing_relay_url};

    #[test]
    fn non_catalog_qwen_q4km_resolves_to_its_tool_capable_compatibility_row() {
        assert_eq!(
            catalog_label_for(std::path::Path::new("Qwen3-4B-Q4_K_M.gguf")).as_deref(),
            Some("qwen3_4b_q4_k_m")
        );
    }

    #[test]
    fn pairing_qr_uses_the_relay_device_connect_base() {
        assert_eq!(
            pairing_relay_url("wss://relay.example.test/"),
            "wss://relay.example.test/v1/connect"
        );
    }
}

#[cfg(test)]
mod remote_session_catalog_tests {
    use super::*;

    #[test]
    fn catalog_pagination_is_revision_pinned_and_identity_gated() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = RemoteStore::open(&directory.path().join("remote.sqlite3")).unwrap();
        let active_session_id = Uuid::from_u128(10);
        let dormant_session_id = Uuid::from_u128(11);
        let model_sha256 = format!("sha256:{}", "a".repeat(64));
        store
            .create_session(
                active_session_id,
                "/work",
                "qwen3_4b_q4_k_m",
                &model_sha256,
                "{}",
                10,
            )
            .unwrap();
        store
            .append_event(
                active_session_id,
                None,
                "user.message",
                &serde_json::json!({"content":"Active task"}),
                30,
            )
            .unwrap();
        store
            .create_session(
                dormant_session_id,
                "/work",
                "other_model",
                &model_sha256,
                "{}",
                20,
            )
            .unwrap();
        let store = Arc::new(Mutex::new(store));
        let sandbox =
            tools::Sandbox::new(directory.path(), false, std::time::Duration::from_secs(1))
                .unwrap();
        let context = SessionCatalogContext {
            store: &store,
            sandbox: &sandbox,
            host_id: Uuid::from_u128(9),
            canonical_root: "/work",
            active_session_id,
            model_id: "qwen3_4b_q4_k_m",
            model_sha256: &model_sha256,
            capability_snapshot: "{}",
        };

        let first = build_session_catalog(
            &context,
            &camelid_remote_protocol::SessionCatalogRequest {
                cursor: None,
                limit: 1,
                revision: None,
            },
        )
        .unwrap();
        assert_eq!(first.sessions.len(), 1);
        assert!(first.sessions[0].active);
        assert!(first.sessions[0].continuable);
        let second = build_session_catalog(
            &context,
            &camelid_remote_protocol::SessionCatalogRequest {
                cursor: first.next_cursor,
                limit: 1,
                revision: Some(first.revision.clone()),
            },
        )
        .unwrap();
        assert_eq!(second.sessions[0].history_id, dormant_session_id);
        assert!(!second.sessions[0].continuable);
        assert_eq!(
            second.sessions[0].refusal_code.as_deref(),
            Some("model_or_capability_identity_mismatch")
        );

        store
            .lock()
            .unwrap()
            .append_event(
                dormant_session_id,
                None,
                "session.notice",
                &serde_json::json!({"message":"changed"}),
                40,
            )
            .unwrap();
        assert!(build_session_catalog(
            &context,
            &camelid_remote_protocol::SessionCatalogRequest {
                cursor: second.next_cursor,
                limit: 1,
                revision: Some(first.revision),
            },
        )
        .is_err());
    }

    #[test]
    fn saved_agent_history_is_deterministic_replay_only_and_ignores_grants() {
        let directory = tempfile::tempdir().unwrap();
        let sandbox =
            tools::Sandbox::new(directory.path(), false, std::time::Duration::from_secs(1))
                .unwrap();
        agent_session::save(
            &sandbox,
            &agent_session::SavedAgentSession {
                id: "saved-task".into(),
                model_id: "qwen3_4b_q4_k_m".into(),
                tool_capable: true,
                workspace: directory.path().display().to_string(),
                transcript: vec![
                    agent::AgentMsg::System("policy".into()),
                    agent::AgentMsg::User("Inspect Cargo.toml".into()),
                    agent::AgentMsg::ToolCalls(vec![tools::ToolCall {
                        name: "read_file".into(),
                        args: serde_json::json!({"path":"Cargo.toml"}),
                    }]),
                    agent::AgentMsg::ToolResult {
                        name: "read_file".into(),
                        outcome: tools::ToolOutcome::Ok("package = camelid".into()),
                    },
                    agent::AgentMsg::Assistant("The package is camelid.".into()),
                ],
                plan: Vec::new(),
                grants: vec!["write_file".into()],
            },
        )
        .unwrap();
        let host_id = Uuid::from_u128(77);
        let first = saved_agent_histories(&sandbox, host_id).unwrap();
        let second = saved_agent_histories(&sandbox, host_id).unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].summary.history_id, second[0].summary.history_id);
        assert_eq!(first[0].summary.source, SessionHistorySource::AgentSaved);
        assert!(!first[0].summary.continuable);
        assert_eq!(
            first[0].summary.refusal_code.as_deref(),
            Some("model_artifact_identity_unavailable")
        );
        assert_eq!(first[0].summary.model_sha256, None);
        assert_eq!(
            first[0]
                .events
                .iter()
                .map(|event| event.event_type.as_str())
                .collect::<Vec<_>>(),
            ["user.message", "tool.call", "tool.result", "model.answer"]
        );
        assert!(!format!("{:?}", first[0].events).contains("write_file"));
    }
}

/// Parsed `camelid agent-eval` flags.
pub struct AgentEvalOptions {
    pub model: PathBuf,
    pub addr: SocketAddr,
    pub load_timeout: u64,
    pub max_steps: usize,
    pub max_tokens: u32,
    pub receipt_dir: PathBuf,
}

/// Entry for the `agent-eval` subcommand: the tool-capability promotion harness.
/// Returns PASS(0) / FAIL(1) / INCONCLUSIVE(3).
pub fn run_agent_eval(opts: AgentEvalOptions) -> anyhow::Result<i32> {
    agent_eval::run(agent_eval::EvalConfig {
        addr: opts.addr,
        model: opts.model,
        load_timeout: opts.load_timeout,
        max_steps: opts.max_steps,
        max_tokens: opts.max_tokens,
        receipt_dir: opts.receipt_dir,
    })
}

/// Parsed `camelid agent-syscap-eval` flags.
pub struct AgentSyscapOptions {
    pub receipt_dir: PathBuf,
}

/// Entry for the `agent-syscap-eval` subcommand: the Phase-1 Windows
/// system-control gate. Returns PASS(0) / FAIL(1) / INCONCLUSIVE(3) and emits a
/// sealed `camelid.agent-syscap-receipt/v1`.
pub fn run_agent_syscap_eval(opts: AgentSyscapOptions) -> anyhow::Result<i32> {
    agent_syscap::run(agent_syscap::SyscapConfig {
        receipt_dir: opts.receipt_dir,
    })
}

/// Entry for the hidden `__subagent` worker subcommand: run one scoped agent loop
/// described by `task_file` and write its result file. Returns 0/1/3.
pub fn run_subagent_worker(task_file: &std::path::Path) -> anyhow::Result<i32> {
    subagent::run_worker(task_file)
}

/// Parsed `camelid agent-orchestration-eval` flags.
pub struct AgentOrchestrationOptions {
    pub receipt_dir: PathBuf,
    pub model: Option<PathBuf>,
    pub addr: SocketAddr,
    pub load_timeout: u64,
}

/// Entry for the `agent-orchestration-eval` subcommand: the orchestration gate.
/// Without `--model` it runs the canned rung-2 mechanics battery; with `--model`
/// it runs the rung-3 real-model round-trip. Returns 0/1/3.
pub fn run_agent_orchestration_eval(opts: AgentOrchestrationOptions) -> anyhow::Result<i32> {
    agent_orchestration::run(agent_orchestration::OrchestrationConfig {
        receipt_dir: opts.receipt_dir,
        model: opts.model,
        addr: opts.addr,
        load_timeout: opts.load_timeout,
    })
}

/// Parsed `camelid agent-orchestration-bench` flags.
pub struct AgentOrchestrationBenchOptions {
    pub receipt_dir: PathBuf,
    pub model: Option<PathBuf>,
    pub addr: SocketAddr,
    pub load_timeout: u64,
}

/// Entry for the `agent-orchestration-bench` subcommand: the rung-4 wall-clock
/// measurement (concurrent vs sequential subagents) → sealed bench receipt.
pub fn run_agent_orchestration_bench(opts: AgentOrchestrationBenchOptions) -> anyhow::Result<i32> {
    agent_bench::run(agent_bench::BenchConfig {
        receipt_dir: opts.receipt_dir,
        model: opts.model,
        addr: opts.addr,
        load_timeout: opts.load_timeout,
    })
}

extern "C" fn on_sigint(_signal: libc::c_int) {
    session::CANCEL.store(true, Ordering::SeqCst);
}

/// Install a SIGINT handler that flips the cancel flag (used by the inline
/// stream loop). The TUI runs in raw mode where Ctrl-C arrives as a key event,
/// so it cancels through its event loop instead.
fn install_sigint_handler() {
    unsafe {
        libc::signal(libc::SIGINT, on_sigint as *const () as libc::sighandler_t);
    }
}

/// Prepare the terminal for the line-mode renderers (inline + agent). On Windows
/// this enables ANSI escape processing and a UTF-8 code page so colors and glyphs
/// render the way they do on macOS/Linux; the full-screen TUI already gets this
/// from crossterm. A no-op on Unix, where terminals handle ANSI + UTF-8 natively.
#[cfg(windows)]
fn init_terminal() {
    win_console::init();
}
#[cfg(not(windows))]
fn init_terminal() {}
