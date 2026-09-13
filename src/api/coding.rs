//! Local same-origin HTTP controller for durable coding runs.
use super::{api_error, workspace, AppState};
use crate::chat::coding::{Config, Manager, Run};
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Response, Sse,
    },
    Json,
};
use serde::Deserialize;
use serde_json::json;
use std::{convert::Infallible, path::PathBuf, sync::Arc, time::Duration};

pub(super) type CodingSessionManager = Manager;
fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), Box<Response>> {
    if state.serve_addr.ip().is_loopback() && workspace::local_management_request_allowed(headers) {
        Ok(())
    } else {
        Err(Box::new(api_error(
            StatusCode::FORBIDDEN,
            "coding_local_only",
            "Coding requires Camelid's local, same-origin interface.".into(),
            None,
        )))
    }
}
fn failure(error: impl ToString) -> Response {
    api_error(
        StatusCode::CONFLICT,
        "coding_session_error",
        error.to_string(),
        None,
    )
}
fn run(state: &AppState, id: &str) -> Result<Arc<Run>, Box<Response>> {
    state
        .coding_sessions
        .get(id, Arc::new(state.changes.clone()))
        .map_err(|error| Box::new(failure(error)))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Create {
    workspace: PathBuf,
    goal: String,
    message_id: String,
    model_id: String,
    #[serde(default)]
    project_id: String,
    #[serde(default)]
    instructions: String,
    #[serde(default)]
    references: String,
    #[serde(default)]
    allow_commands: bool,
    max_steps: Option<usize>,
    max_tokens: Option<u32>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Message {
    message: String,
    message_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Control {
    action: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Decision {
    approved: bool,
}

pub(super) async fn list(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authorize(&state, &headers) {
        return *e;
    }
    let tool_capable_model = workspace::active_tool_capable_model(&state)
        .await
        .ok()
        .map(|(model, _)| model.id);
    let result =
        tokio::task::spawn_blocking(move || state.coding_sessions.list(Arc::new(state.changes)))
            .await;
    match result {
        Ok(Ok(sessions)) => Json(json!({"tool_capable_model":tool_capable_model,"sessions":sessions.iter().map(|s| json!({"id":s.id,"title":s.title,"phase":s.phase,"updated_at":s.updated_at,"workspace":s.config.workspace,"project_id":s.config.project_id,"model_id":s.config.model_id})).collect::<Vec<_>>()})).into_response(),
        Ok(Err(e)) => failure(e), Err(e) => failure(e),
    }
}
pub(super) async fn create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<Create>,
) -> Response {
    if let Err(e) = authorize(&state, &headers) {
        return *e;
    }
    if request.instructions.len() > 12000
        || request.references.len() > 96000
        || request.project_id.len() > 100
        || request.goal.len() > 16000
        || request.goal.trim().is_empty()
    {
        return failure("Project context or goal exceeds the coding-session limits.");
    }
    let steps = request.max_steps.unwrap_or(32);
    let tokens = request.max_tokens.unwrap_or(2048);
    if !(1..=64).contains(&steps) || !(128..=4096).contains(&tokens) {
        return failure("Use 1–64 steps and 128–4096 reply tokens.");
    }
    let workspace = match std::fs::canonicalize(&request.workspace) {
        Ok(p) if p.is_dir() => p,
        _ => return failure("Choose an accessible local project folder."),
    };
    let _transition = state.model_transition.lock().await;
    if state.workspace_sessions.blocks_model_transition().await {
        return failure("Wait for the active Workspace turn to finish.");
    }
    let (model, family) = match workspace::active_tool_capable_model(&state).await {
        Ok(model) => model,
        Err(e) => return e,
    };
    if model.id != request.model_id {
        return failure("The selected model changed. Refresh before starting Code.");
    }
    let Some(model_context) = model.llama_config.as_ref().map(|c| c.context_length) else {
        return failure("The model's effective context budget is unavailable.");
    };
    let max_tokens = tokens
        .min(state.server_limits.max_generation_tokens)
        .min(model_context / 2);
    let context_tokens = model_context
        .min(crate::chat::agent::AGENT_VALIDATED_CTX)
        .min((state.server_limits.max_prompt_tokens as u32).saturating_add(max_tokens));
    if context_tokens < 1024 || max_tokens < 128 {
        return failure("The current context limits are too small for coding tools.");
    }
    let config = Config {
        addr: state.serve_addr,
        workspace,
        model_id: model.id,
        model_sha256: model.lane.gguf_sha256,
        family,
        context_tokens,
        max_tokens,
        max_steps: steps,
        allow_commands: request.allow_commands,
        project_id: request.project_id,
        instructions: request.instructions,
        references: request.references,
    };
    match state.coding_sessions.create(
        config,
        request.goal,
        request.message_id,
        Arc::new(state.changes.clone()),
    ) {
        Ok(run) => Json(run.snapshot()).into_response(),
        Err(e) => failure(e),
    }
}
pub(super) async fn get(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authorize(&state, &headers) {
        return *e;
    }
    match run(&state, &id) {
        Ok(run) => Json(run.snapshot()).into_response(),
        Err(e) => *e,
    }
}
pub(super) async fn message(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(request): Json<Message>,
) -> Response {
    if let Err(e) = authorize(&state, &headers) {
        return *e;
    }
    let run = match run(&state, &id) {
        Ok(r) => r,
        Err(e) => return *e,
    };
    let _transition = state.model_transition.lock().await;
    if state.workspace_sessions.blocks_model_transition().await {
        return failure("Wait for Workspace to finish.");
    }
    let (model, _) = match workspace::active_tool_capable_model(&state).await {
        Ok(m) => m,
        Err(e) => return e,
    };
    let mut config = run.snapshot().config;
    if model.id != config.model_id || model.lane.gguf_sha256 != config.model_sha256 {
        return failure(
            "Load the exact model artifact used by this coding session before continuing it.",
        );
    }
    if std::fs::canonicalize(&config.workspace).ok().as_ref() != Some(&config.workspace) {
        return failure("The saved project folder is no longer the same workspace.");
    }
    let Some(model_context) = model.llama_config.as_ref().map(|c| c.context_length) else {
        return failure("The model's context budget is unavailable.");
    };
    config.addr = state.serve_addr;
    config.max_tokens = config
        .max_tokens
        .min(state.server_limits.max_generation_tokens)
        .min(model_context / 2);
    config.context_tokens = model_context
        .min(crate::chat::agent::AGENT_VALIDATED_CTX)
        .min((state.server_limits.max_prompt_tokens as u32).saturating_add(config.max_tokens));
    if config.context_tokens < 1024 || config.max_tokens < 128 {
        return failure("The current context limits are too small for coding tools.");
    }
    match state
        .coding_sessions
        .follow_up(&run, request.message, request.message_id, config)
    {
        Ok(()) => Json(run.snapshot()).into_response(),
        Err(e) => failure(e),
    }
}
pub(super) async fn control(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(request): Json<Control>,
) -> Response {
    if let Err(e) = authorize(&state, &headers) {
        return *e;
    }
    let run = match run(&state, &id) {
        Ok(r) => r,
        Err(e) => return *e,
    };
    match run.control(&request.action) {
        Ok(()) => Json(run.snapshot()).into_response(),
        Err(e) => failure(e),
    }
}
pub(super) async fn decide(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, approval)): Path<(String, String)>,
    Json(request): Json<Decision>,
) -> Response {
    if let Err(e) = authorize(&state, &headers) {
        return *e;
    }
    let run = match run(&state, &id) {
        Ok(r) => r,
        Err(e) => return *e,
    };
    match run.decide(&approval, request.approved) {
        Ok(()) => Json(json!({"accepted":true})).into_response(),
        Err(e) => failure(e),
    }
}
pub(super) async fn remove(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authorize(&state, &headers) {
        return *e;
    }
    match state
        .coding_sessions
        .remove(&id, Arc::new(state.changes.clone()))
    {
        Ok(()) => Json(json!({"removed":true})).into_response(),
        Err(e) => failure(e),
    }
}
pub(super) async fn events(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authorize(&state, &headers) {
        return *e;
    }
    let run = match run(&state, &id) {
        Ok(r) => r,
        Err(e) => return *e,
    };
    let mut receiver = run.subscribe();
    // Each SSE frame is a complete versioned snapshot. Gaps, reconnects, and
    // an expired event backlog need no action replay and no guessed deltas.
    let stream = async_stream::stream! {
        loop {
            let snapshot = run.snapshot();
            let terminal = !snapshot.phase.active();
            yield Ok::<Event, Infallible>(Event::default().event("coding").id(snapshot.seq.to_string()).json_data(&snapshot).unwrap_or_else(|_| Event::default().event("error").data("Could not encode coding state")));
            if terminal { break; }
            if receiver.changed().await.is_err() { break; }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn local_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("host", "127.0.0.1:8181".parse().unwrap());
        headers.insert("origin", "http://127.0.0.1:8181".parse().unwrap());
        headers
    }
    #[test]
    fn coding_rejects_cross_origin_and_non_loopback_surfaces() {
        let state = AppState::default();
        let mut headers = local_headers();
        assert!(authorize(&state, &headers).is_ok());
        headers.insert("origin", "https://unrelated.example".parse().unwrap());
        assert_eq!(
            authorize(&state, &headers).unwrap_err().status(),
            StatusCode::FORBIDDEN
        );
        let lan = AppState {
            serve_addr: "192.0.2.1:8181".parse().unwrap(),
            ..state
        };
        assert_eq!(
            authorize(&lan, &local_headers()).unwrap_err().status(),
            StatusCode::FORBIDDEN
        );
    }
    #[tokio::test]
    async fn create_fails_closed_without_a_loaded_certified_model() {
        let root = tempfile::tempdir().unwrap();
        let request = Create {
            workspace: root.path().into(),
            goal: "Inspect files".into(),
            message_id: "a".repeat(32),
            model_id: "unloaded".into(),
            project_id: String::new(),
            instructions: String::new(),
            references: String::new(),
            allow_commands: false,
            max_steps: None,
            max_tokens: None,
        };
        let state = AppState::default();
        let response = create(State(state.clone()), local_headers(), Json(request)).await;
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert!(!state.coding_sessions.busy());
    }
}
