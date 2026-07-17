use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};

use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use super::{
    api_error, capabilities_response_with_plan, curated_catalog, AppState, LoadedModel,
    NON_CATALOG_SUPPORTED_ARTIFACTS,
};
use crate::chat::agent::LoopEnd;
use crate::chat::workspace_bridge::{
    bridge, run_live, WorkspaceBridgeControl, WorkspaceBridgeWorker, WorkspaceDecisionKind,
    WorkspaceEvent, WorkspaceRunConfig,
};

const EVENT_BACKLOG: usize = 128;
const EVENT_STREAM_BUFFER: usize = 128;
const DEFAULT_MAX_STEPS: usize = 12;
const MAX_STEPS: usize = 32;
const DEFAULT_MAX_TOKENS: u32 = 800;
const MAX_TOKENS: u32 = 4096;
const MAX_GOAL_BYTES: usize = 16 * 1024;

#[derive(Clone, Default)]
pub(super) struct WorkspaceSessionManager {
    active: Arc<Mutex<Option<Arc<ActiveWorkspaceSession>>>>,
}

struct ActiveWorkspaceSession {
    id: String,
    workspace: PathBuf,
    model_id: String,
    state: StdMutex<WorkspaceSessionState>,
    events: StdMutex<Option<std::sync::mpsc::Receiver<WorkspaceEvent>>>,
    worker: StdMutex<Option<WorkspaceBridgeWorker>>,
    run_config: StdMutex<Option<WorkspaceRunConfig>>,
    control: WorkspaceBridgeControl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkspaceSessionState {
    WaitingForEvents,
    Running,
    Cancelling,
    Finished,
    Cancelled,
    Failed,
}

impl WorkspaceSessionState {
    fn as_str(self) -> &'static str {
        match self {
            Self::WaitingForEvents => "waiting_for_events",
            Self::Running => "running",
            Self::Cancelling => "cancelling",
            Self::Finished => "finished",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        }
    }

    fn blocks_model_transition(self) -> bool {
        matches!(
            self,
            Self::WaitingForEvents | Self::Running | Self::Cancelling
        )
    }

    fn after_cancel_request(self) -> Self {
        match self {
            Self::WaitingForEvents => Self::Cancelled,
            Self::Running => Self::Cancelling,
            other => other,
        }
    }

    fn after_worker_exit(self, result: &Result<LoopEnd, String>) -> Self {
        if self == Self::Cancelling {
            return Self::Cancelled;
        }
        match result {
            Ok(LoopEnd::Aborted) => Self::Cancelled,
            Ok(LoopEnd::DriverError) | Err(_) => Self::Failed,
            Ok(LoopEnd::Answered | LoopEnd::StepCapped | LoopEnd::Repeated) => Self::Finished,
        }
    }
}

impl WorkspaceSessionManager {
    pub(super) async fn blocks_model_transition(&self) -> bool {
        let active = self.active.lock().await.clone();
        active.is_some_and(|session| {
            session
                .state
                .lock()
                .map(|state| state.blocks_model_transition())
                .unwrap_or(true)
        })
    }

    fn active_state(active: &Option<Arc<ActiveWorkspaceSession>>) -> Option<WorkspaceSessionState> {
        active.as_ref().map(|session| {
            session
                .state
                .lock()
                .map(|state| *state)
                .unwrap_or(WorkspaceSessionState::Running)
        })
    }
}

#[derive(Debug, Deserialize)]
pub(super) struct CreateWorkspaceSessionRequest {
    workspace: PathBuf,
    goal: String,
    #[serde(default)]
    max_steps: Option<usize>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct WorkspaceSessionResponse {
    id: String,
    workspace: String,
    model_id: String,
    state: &'static str,
    max_steps: usize,
    max_tokens: u32,
}

#[derive(Debug, Serialize)]
struct WorkspaceSessionStatusResponse {
    id: String,
    workspace: String,
    model_id: String,
    state: &'static str,
}

#[derive(Debug, Deserialize)]
pub(super) struct WorkspaceDecisionRequest {
    approval_id: String,
    decision: WorkspaceDecisionKind,
}

#[derive(Debug, Serialize)]
struct WorkspaceEventEnvelope {
    sequence: u64,
    session_id: String,
    #[serde(flatten)]
    event: WorkspaceEvent,
}

struct CancelStreamOnDrop(WorkspaceBridgeControl);

impl Drop for CancelStreamOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

pub(super) async fn create_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CreateWorkspaceSessionRequest>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }

    let goal = request.goal.trim();
    if goal.is_empty() || goal.len() > MAX_GOAL_BYTES {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_workspace_goal",
            format!("goal must contain 1 to {MAX_GOAL_BYTES} UTF-8 bytes"),
            Some("goal"),
        );
    }
    let max_steps = request.max_steps.unwrap_or(DEFAULT_MAX_STEPS);
    if !(1..=MAX_STEPS).contains(&max_steps) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_workspace_limits",
            format!("max_steps must be between 1 and {MAX_STEPS}"),
            Some("max_steps"),
        );
    }
    let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    if !(1..=MAX_TOKENS).contains(&max_tokens) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_workspace_limits",
            format!("max_tokens must be between 1 and {MAX_TOKENS}"),
            Some("max_tokens"),
        );
    }
    let temperature = request.temperature.unwrap_or(0.0);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_workspace_limits",
            "temperature must be finite and between 0 and 2".to_string(),
            Some("temperature"),
        );
    }

    let workspace = match std::fs::canonicalize(&request.workspace) {
        Ok(path) if path.is_dir() => path,
        _ => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "workspace_root_not_accessible",
                "workspace must name an accessible local directory".to_string(),
                Some("workspace"),
            )
        }
    };

    let (model, family) = match active_tool_capable_model(&state).await {
        Ok(value) => value,
        Err(response) => return response,
    };

    let mut active = state.workspace_sessions.active.lock().await;
    if let Some(existing_state) = WorkspaceSessionManager::active_state(&active) {
        let blocks_replacement = existing_state.blocks_model_transition();
        if blocks_replacement {
            return api_error(
                StatusCode::CONFLICT,
                "workspace_session_already_active",
                "finish or cancel the current Workspace session before starting another"
                    .to_string(),
                None,
            );
        }
        *active = None;
    }

    let id = format!("workspace-{}", uuid::Uuid::new_v4());
    let (worker, client) = bridge(EVENT_BACKLOG);
    let (events, control) = client.into_parts();
    let run_config = WorkspaceRunConfig {
        addr: state.serve_addr,
        workspace: workspace.clone(),
        goal: goal.to_string(),
        model_id: model.id.clone(),
        family,
        max_steps,
        max_tokens,
        temperature,
    };
    let session = Arc::new(ActiveWorkspaceSession {
        id: id.clone(),
        workspace: workspace.clone(),
        model_id: model.id.clone(),
        state: StdMutex::new(WorkspaceSessionState::WaitingForEvents),
        events: StdMutex::new(Some(events)),
        worker: StdMutex::new(Some(worker)),
        run_config: StdMutex::new(Some(run_config)),
        control,
    });
    *active = Some(session);

    (
        StatusCode::CREATED,
        Json(WorkspaceSessionResponse {
            id,
            workspace: workspace.display().to_string(),
            model_id: model.id,
            state: WorkspaceSessionState::WaitingForEvents.as_str(),
            max_steps,
            max_tokens,
        }),
    )
        .into_response()
}

pub(super) async fn session_events(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let session = match find_session(&state, &id).await {
        Ok(session) => session,
        Err(response) => return response,
    };

    let events = match session.events.lock() {
        Ok(mut events) => events.take(),
        Err(_) => None,
    };
    let worker = match session.worker.lock() {
        Ok(mut worker) => worker.take(),
        Err(_) => None,
    };
    let run_config = match session.run_config.lock() {
        Ok(mut config) => config.take(),
        Err(_) => None,
    };
    let (Some(events), Some(worker), Some(run_config)) = (events, worker, run_config) else {
        return api_error(
            StatusCode::CONFLICT,
            "workspace_event_stream_already_claimed",
            "this Workspace session already has an event consumer".to_string(),
            None,
        );
    };
    if let Ok(mut status) = session.state.lock() {
        *status = WorkspaceSessionState::Running;
    }

    let worker_session = Arc::clone(&session);
    std::thread::Builder::new()
        .name("camelid-workspace-agent".to_string())
        .spawn(move || {
            let result = run_live(run_config, worker);
            if let Ok(mut status) = worker_session.state.lock() {
                *status = status.after_worker_exit(&result);
            }
        })
        .expect("spawn Workspace agent thread");

    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(EVENT_STREAM_BUFFER);
    let forward_control = session.control.clone();
    std::thread::Builder::new()
        .name("camelid-workspace-events".to_string())
        .spawn(move || {
            while let Ok(event) = events.recv() {
                if event_tx.try_send(event).is_err() {
                    forward_control.cancel();
                    break;
                }
            }
        })
        .expect("spawn Workspace event forwarder");

    let session_id = session.id.clone();
    let disconnect_guard = CancelStreamOnDrop(session.control.clone());
    let stream = async_stream::stream! {
        let _disconnect_guard = disconnect_guard;
        let mut sequence = 0_u64;
        while let Some(event) = event_rx.recv().await {
            sequence += 1;
            let terminal = matches!(event, WorkspaceEvent::Finished { .. } | WorkspaceEvent::Error { .. });
            let envelope = WorkspaceEventEnvelope {
                sequence,
                session_id: session_id.clone(),
                event,
            };
            match serde_json::to_string(&envelope) {
                Ok(json) => yield Ok::<Event, std::convert::Infallible>(
                    Event::default().event("workspace").id(sequence.to_string()).data(json)
                ),
                Err(_) => continue,
            }
            if terminal {
                break;
            }
        }
    };
    Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(std::time::Duration::from_secs(10))
                .text("ping"),
        )
        .into_response()
}

pub(super) async fn decide(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Json(request): Json<WorkspaceDecisionRequest>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let session = match find_session(&state, &id).await {
        Ok(session) => session,
        Err(response) => return response,
    };
    match session
        .control
        .try_decide(request.approval_id, request.decision)
    {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(message) => api_error(
            StatusCode::CONFLICT,
            "workspace_approval_not_pending",
            message.to_string(),
            Some("approval_id"),
        ),
    }
}

pub(super) async fn session_status(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let session = match find_session(&state, &id).await {
        Ok(session) => session,
        Err(response) => return response,
    };
    let status = session
        .state
        .lock()
        .map(|state| state.as_str())
        .unwrap_or("error");
    Json(WorkspaceSessionStatusResponse {
        id: session.id.clone(),
        workspace: session.workspace.display().to_string(),
        model_id: session.model_id.clone(),
        state: status,
    })
    .into_response()
}

pub(super) async fn cancel_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let active = state.workspace_sessions.active.lock().await;
    let Some(session) = active.as_ref() else {
        return workspace_not_found();
    };
    if session.id != id {
        return workspace_not_found();
    }
    session.control.cancel();
    if let Ok(mut status) = session.state.lock() {
        *status = status.after_cancel_request();
    }
    StatusCode::NO_CONTENT.into_response()
}

async fn find_session(state: &AppState, id: &str) -> Result<Arc<ActiveWorkspaceSession>, Response> {
    let active = state.workspace_sessions.active.lock().await;
    active
        .as_ref()
        .filter(|session| session.id == id)
        .cloned()
        .ok_or_else(workspace_not_found)
}

fn workspace_not_found() -> Response {
    api_error(
        StatusCode::NOT_FOUND,
        "workspace_session_not_found",
        "Workspace session was not found".to_string(),
        None,
    )
}

fn authorize(state: &AppState, headers: &HeaderMap) -> Option<Response> {
    if state.serve_addr.ip().is_loopback() && local_management_request_allowed(headers) {
        return None;
    }
    Some(api_error(
        StatusCode::FORBIDDEN,
        "local_management_forbidden",
        "Workspace is available only from Camelid's loopback web UI".to_string(),
        None,
    ))
}

fn local_management_request_allowed(headers: &HeaderMap) -> bool {
    let host_is_loopback = headers
        .get("host")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<axum::http::uri::Authority>().ok())
        .is_some_and(|authority| loopback_host(authority.host()));
    if !host_is_loopback {
        return false;
    }
    let origin = headers
        .get("origin")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<axum::http::Uri>().ok());
    if let Some(origin) = &origin {
        if !matches!(origin.scheme_str(), Some("http") | Some("https"))
            || !origin.host().is_some_and(loopback_host)
        {
            return false;
        }
    }
    headers
        .get("sec-fetch-site")
        .and_then(|value| value.to_str().ok())
        == Some("same-origin")
        || origin.is_some()
}

fn loopback_host(host: &str) -> bool {
    let host = host.trim_matches(['[', ']']);
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

async fn active_tool_capable_model(state: &AppState) -> Result<(LoadedModel, String), Response> {
    let active_id = state.active_model_id.read().await.clone().ok_or_else(|| {
        api_error(
            StatusCode::CONFLICT,
            "model_not_loaded",
            "load a tool-capable model before starting Workspace".to_string(),
            None,
        )
    })?;
    let model = state
        .loaded_models
        .read()
        .await
        .get(&active_id)
        .cloned()
        .ok_or_else(|| {
            api_error(
                StatusCode::CONFLICT,
                "model_not_loaded",
                "the active model is no longer loaded".to_string(),
                None,
            )
        })?;
    let filename = model
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let row = tool_capable_row_for_filename(filename);
    match row {
        Some((_, family)) => Ok((model, family.to_string())),
        None => Err(api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "model_not_tool_capable",
            "the active exact model row has not earned tool-capable status".to_string(),
            None,
        )),
    }
}

fn tool_capable_row_for_filename(filename: &str) -> Option<(&'static str, &'static str)> {
    let row_id = curated_catalog()
        .iter()
        .find(|item| item.filename == filename)
        .map(|item| item.catalog_id)
        .or_else(|| {
            NON_CATALOG_SUPPORTED_ARTIFACTS
                .iter()
                .find(|(artifact, _)| *artifact == filename)
                .map(|(_, row_id)| *row_id)
        });
    row_id.and_then(|row_id| {
        capabilities_response_with_plan(None)
            .model_compatibility
            .into_iter()
            .find(|row| row.id == row_id && row.tool_capable && row.status.starts_with("supported"))
            .map(|row| (row.id, row.family))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn management_authorization_is_loopback_and_browser_scoped() {
        let mut headers = HeaderMap::new();
        headers.insert("host", "127.0.0.1:8181".parse().unwrap());
        headers.insert("sec-fetch-site", "same-origin".parse().unwrap());
        assert!(local_management_request_allowed(&headers));

        headers.insert("origin", "https://attacker.example".parse().unwrap());
        assert!(!local_management_request_allowed(&headers));

        headers.insert("origin", "http://localhost:4173".parse().unwrap());
        assert!(local_management_request_allowed(&headers));

        headers.remove("origin");
        headers.remove("sec-fetch-site");
        assert!(!local_management_request_allowed(&headers));
    }

    #[test]
    fn session_state_blocks_model_transitions_only_while_active() {
        assert!(WorkspaceSessionState::WaitingForEvents.blocks_model_transition());
        assert!(WorkspaceSessionState::Running.blocks_model_transition());
        assert!(WorkspaceSessionState::Cancelling.blocks_model_transition());
        assert!(!WorkspaceSessionState::Finished.blocks_model_transition());
        assert!(!WorkspaceSessionState::Cancelled.blocks_model_transition());
        assert!(!WorkspaceSessionState::Failed.blocks_model_transition());
    }

    #[test]
    fn cancellation_stays_blocking_until_a_running_worker_exits() {
        let requested = WorkspaceSessionState::Running.after_cancel_request();
        assert_eq!(requested, WorkspaceSessionState::Cancelling);
        assert!(requested.blocks_model_transition());
        assert_eq!(
            requested.after_worker_exit(&Ok(LoopEnd::Aborted)),
            WorkspaceSessionState::Cancelled
        );
        assert_eq!(
            WorkspaceSessionState::WaitingForEvents.after_cancel_request(),
            WorkspaceSessionState::Cancelled
        );
    }

    #[test]
    fn disconnect_abort_and_driver_failure_have_truthful_terminal_states() {
        assert_eq!(
            WorkspaceSessionState::Running.after_worker_exit(&Ok(LoopEnd::Aborted)),
            WorkspaceSessionState::Cancelled
        );
        assert_eq!(
            WorkspaceSessionState::Running.after_worker_exit(&Ok(LoopEnd::DriverError)),
            WorkspaceSessionState::Failed
        );
        assert_eq!(
            WorkspaceSessionState::Running.after_worker_exit(&Err("startup failed".to_string())),
            WorkspaceSessionState::Failed
        );
    }

    #[test]
    fn manager_reads_terminal_and_active_session_states_without_guessing() {
        let make_session = |state| {
            let (worker, client) = bridge(1);
            let (events, control) = client.into_parts();
            Arc::new(ActiveWorkspaceSession {
                id: "session-test".to_string(),
                workspace: PathBuf::from("."),
                model_id: "model-test".to_string(),
                state: StdMutex::new(state),
                events: StdMutex::new(Some(events)),
                worker: StdMutex::new(Some(worker)),
                run_config: StdMutex::new(None),
                control,
            })
        };

        let running = Some(make_session(WorkspaceSessionState::Running));
        assert_eq!(
            WorkspaceSessionManager::active_state(&running),
            Some(WorkspaceSessionState::Running)
        );
        assert!(WorkspaceSessionManager::active_state(&running)
            .unwrap()
            .blocks_model_transition());

        let finished = Some(make_session(WorkspaceSessionState::Finished));
        assert_eq!(
            WorkspaceSessionManager::active_state(&finished),
            Some(WorkspaceSessionState::Finished)
        );
        assert!(!WorkspaceSessionManager::active_state(&finished)
            .unwrap()
            .blocks_model_transition());
    }

    #[test]
    fn every_earned_tool_capable_artifact_resolves_by_exact_filename() {
        let expected = [
            ("ornith-1.0-9b-Q4_K_M.gguf", "ornith_1_0_9b_q4_k_m"),
            ("ornith-1.0-9b-Q8_0.gguf", "Ornith 1.0 9B"),
            (
                "Llama-3.2-3B-Instruct-Q8_0.gguf",
                "llama32_3b_instruct_q8_0",
            ),
            ("Qwen3-4B-Q8_0.gguf", "qwen3_4b_instruct_q8_0"),
            ("Qwen3-4B-Q4_K_M.gguf", "qwen3_4b_q4_k_m"),
        ];
        for (filename, row_id) in expected {
            assert_eq!(
                tool_capable_row_for_filename(filename).map(|row| row.0),
                Some(row_id),
                "missing exact tool-capable mapping for {filename}"
            );
        }
        assert_eq!(
            tool_capable_row_for_filename("neighboring-model.gguf"),
            None
        );
    }
}
