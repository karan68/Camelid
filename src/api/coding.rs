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
    #[serde(default)]
    project: crate::chat::coding_project::ProjectSettings,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Message {
    message: String,
    message_id: String,
    mode: Option<String>,
    run_id: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Control {
    action: String,
    run_id: Option<String>,
    enabled: Option<bool>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Decision {
    approved: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CreateFolder {
    parent: PathBuf,
    name: String,
}

/// Explicit user setup action, independent of model/agent execution.
pub(super) async fn create_folder(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CreateFolder>,
) -> Response {
    if let Err(e) = authorize(&state, &headers) {
        return *e;
    }
    match tokio::task::spawn_blocking(move || create_project_folder(request)).await {
        Ok(Ok(path)) => (
            StatusCode::CREATED,
            Json(json!({"path": workspace::simplify_path(&path)})),
        )
            .into_response(),
        Ok(Err((status, message))) => {
            api_error(status, "coding_folder_error", message.into(), None)
        }
        Err(_) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "coding_folder_error",
            "Could not create the folder.".into(),
            None,
        ),
    }
}

fn create_project_folder(request: CreateFolder) -> Result<PathBuf, (StatusCode, &'static str)> {
    let name = request.name.trim();
    if name.is_empty()
        || name.len() > 255
        || matches!(name, "." | "..")
        || name.ends_with('.')
        || name
            .chars()
            .any(|c| c.is_control() || "<>:\"/\\|?*".contains(c))
        || name.eq_ignore_ascii_case(".git")
        || name.eq_ignore_ascii_case(".camelid")
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "Use one folder name without path separators, reserved characters, or a trailing dot.",
        ));
    }
    if !request.parent.is_absolute() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Choose an existing parent folder first.",
        ));
    }
    let parent = std::fs::canonicalize(request.parent).map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            "The parent folder is not accessible.",
        )
    })?;
    if !parent.is_dir() {
        return Err((StatusCode::BAD_REQUEST, "The parent path is not a folder."));
    }
    if parent.components().any(|c| {
        c.as_os_str()
            .to_str()
            .is_some_and(|s| s.eq_ignore_ascii_case(".git") || s.eq_ignore_ascii_case(".camelid"))
    }) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Choose a project location outside .git and .camelid folders.",
        ));
    }
    let path = parent.join(name);
    // create_dir fails if any entry already occupies the name, including a
    // symlink. Never reuse, overwrite, or recursively create a supplied path.
    std::fs::create_dir(&path).map_err(|e| match e.kind() {
        std::io::ErrorKind::AlreadyExists => (StatusCode::CONFLICT, "A file or folder with that name already exists. Choose another name or open the existing folder."),
        std::io::ErrorKind::PermissionDenied => (StatusCode::FORBIDDEN, "You do not have permission to create a folder here."),
        _ => (StatusCode::BAD_REQUEST, "Could not create that folder. Check the name and parent folder."),
    })?;
    Ok(path)
}

pub(super) async fn list(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authorize(&state, &headers) {
        return *e;
    }
    let execution_engine = match state.coding_sessions.engine() {
        Ok(value) => value,
        Err(error) => return failure(error),
    };
    let tool_capable_model = workspace::active_tool_capable_model(&state)
        .await
        .ok()
        .map(|(model, _)| model.id);
    let result =
        tokio::task::spawn_blocking(move || state.coding_sessions.list(Arc::new(state.changes)))
            .await;
    match result {
        Ok(Ok(sessions)) => Json(json!({"execution_engine":execution_engine,"tool_capable_model":tool_capable_model,"sessions":sessions.iter().map(|s| json!({"id":s.id,"title":s.title,"phase":s.phase,"updated_at":s.updated_at,"workspace":s.config.workspace,"project_id":s.config.project_id,"model_id":s.config.model_id})).collect::<Vec<_>>()})).into_response(),
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
        project: request.project,
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
        Ok(run) => match tokio::task::spawn_blocking(move || run.observed_snapshot()).await {
            Ok(snapshot) => Json(snapshot).into_response(),
            Err(error) => failure(error),
        },
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
    if let Some(mode) = request.mode.as_deref().filter(|mode| *mode != "follow_up") {
        return match request
            .run_id
            .as_deref()
            .ok_or_else(|| "Bind active input to the current run_id.".to_string())
            .and_then(|run_id| run.input(run_id, request.message_id, request.message, mode))
        {
            Ok(()) => Json(run.snapshot()).into_response(),
            Err(error) => failure(error),
        };
    }
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
    let result = if request.action == "auto_approve_files" {
        match (request.run_id, request.enabled) {
            (Some(run_id), Some(enabled)) => run.set_auto_approve_files(&run_id, enabled),
            _ => {
                Err("Provide the current run_id and an enabled boolean for file approvals.".into())
            }
        }
    } else if request.run_id.is_some() || request.enabled.is_some() {
        Err("File approval settings require the auto_approve_files action.".into())
    } else {
        run.control(&request.action)
    };
    match result {
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
            let observed = run.clone();
            let snapshot = match tokio::task::spawn_blocking(move || observed.observed_snapshot()).await {
                Ok(snapshot) => snapshot,
                Err(_) => {
                    yield Ok(Event::default().event("error").data("Could not inspect current project evidence"));
                    break;
                }
            };
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

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum ProjectAction {
    Settings {
        run_id: String,
        settings: crate::chat::coding_project::ProjectSettings,
    },
    SaveWorkflow {
        name: String,
        notes: String,
    },
    RestoreCheckpoint {
        run_id: String,
        checkpoint_id: String,
    },
}
pub(super) async fn project_action(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(request): Json<ProjectAction>,
) -> Response {
    if let Err(error) = authorize(&state, &headers) {
        return *error;
    }
    let run = match run(&state, &id) {
        Ok(run) => run,
        Err(error) => return *error,
    };
    let result = tokio::task::spawn_blocking(move || {
        match request {
            ProjectAction::Settings {
                run_id,
                mut settings,
            } => {
                state.coding_sessions.bind_project(&mut settings)?;
                run.settings(&run_id, settings)?;
            }
            ProjectAction::SaveWorkflow { name, notes } => run.save_workflow(name, notes)?,
            ProjectAction::RestoreCheckpoint {
                run_id,
                checkpoint_id,
            } => run.restore_checkpoint(&run_id, &checkpoint_id)?,
        }
        Ok::<_, String>(run.snapshot())
    })
    .await;
    match result {
        Ok(Ok(snapshot)) => Json(snapshot).into_response(),
        Ok(Err(error)) => failure(error),
        Err(error) => failure(error),
    }
}
pub(super) async fn preview(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(error) = authorize(&state, &headers) {
        return *error;
    }
    let run = match run(&state, &id) {
        Ok(run) => run,
        Err(error) => return *error,
    };
    match tokio::task::spawn_blocking(move || run.preview()).await {
        Ok(Ok(html)) => Json(json!({"html":html,"kind":"static_html","limitations":"Static preview with an opaque frame origin. External scripts, stylesheets, fetch requests, and module imports are restricted. Opening this preview does not count as a passing check."})).into_response(),
        Ok(Err(error)) => failure(error), Err(error) => failure(error)
    }
}

pub(super) async fn preview_server_status(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(error) = authorize(&state, &headers) {
        return *error;
    }
    let run = match run(&state, &id) {
        Ok(run) => run,
        Err(error) => return *error,
    };
    match tokio::task::spawn_blocking(move || run.preview_status()).await {
        Ok(status) => Json(status).into_response(),
        Err(error) => failure(error),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PreviewRequest {
    action: PreviewAction,
    #[serde(default)]
    entry: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum PreviewAction {
    Start,
    Open,
    Stop,
}

pub(super) async fn preview_server_action(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(request): Json<PreviewRequest>,
) -> Response {
    if let Err(error) = authorize(&state, &headers) {
        return *error;
    }
    let run = match run(&state, &id) {
        Ok(run) => run,
        Err(error) => return *error,
    };
    let action = match request.action {
        PreviewAction::Start => "start",
        PreviewAction::Open => "open",
        PreviewAction::Stop => "stop",
    };
    match tokio::task::spawn_blocking(move || run.manage_preview(action, &request.entry)).await {
        Ok(Ok(status)) => Json(status).into_response(),
        Ok(Err(error)) => failure(error),
        Err(error) => failure(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    fn local_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("host", "127.0.0.1:8181".parse().unwrap());
        headers.insert("origin", "http://127.0.0.1:8181".parse().unwrap());
        headers
    }
    #[tokio::test]
    async fn managed_preview_api_rejects_foreign_origin_before_session_lookup() {
        let state = AppState::default();
        let mut headers = local_headers();
        headers.insert("origin", "https://unrelated.example".parse().unwrap());
        assert_eq!(
            preview_server_status(
                State(state.clone()),
                headers.clone(),
                Path("missing".into())
            )
            .await
            .status(),
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            preview_server_action(
                State(state),
                headers,
                Path("missing".into()),
                Json(PreviewRequest {
                    action: PreviewAction::Start,
                    entry: "index.html".into()
                })
            )
            .await
            .status(),
            StatusCode::FORBIDDEN
        );
    }
    #[tokio::test]
    async fn active_input_and_reconnect_preserve_identity_authority_and_evidence() {
        let temp = tempfile::tempdir().unwrap();
        // Hold model I/O at a real loopback listener so the lead remains active
        // without inference or a race against an unavailable model endpoint.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let state = AppState {
            coding_sessions: crate::chat::coding::Manager::new(temp.path().join("sessions")),
            ..AppState::default()
        };
        let config = Config {
            addr: listener.local_addr().unwrap(),
            workspace: std::fs::canonicalize(temp.path()).unwrap(),
            model_id: "fixture".into(),
            model_sha256: "a".repeat(64),
            family: "qwen3".into(),
            context_tokens: 4096,
            max_tokens: 512,
            max_steps: 4,
            allow_commands: false,
            project_id: String::new(),
            instructions: String::new(),
            references: String::new(),
            project: Default::default(),
        };
        let run = state
            .coding_sessions
            .create(
                config,
                "Inspect the fixture".into(),
                "a".repeat(32),
                Arc::new(state.changes.clone()),
            )
            .unwrap();
        let snapshot = run.snapshot();
        for (mode, key, text) in [
            ("steer", "b", "Preserve count_open(tasks)"),
            ("queue", "c", "Summarize the checks afterward"),
        ] {
            let request = || {
                Json(Message {
                    message: text.into(),
                    message_id: key.repeat(32),
                    mode: Some(mode.into()),
                    run_id: Some(snapshot.run_id.clone()),
                })
            };
            let mut foreign = local_headers();
            foreign.insert("origin", "https://unrelated.example".parse().unwrap());
            assert_eq!(
                message(
                    State(state.clone()),
                    foreign,
                    Path(snapshot.id.clone()),
                    request()
                )
                .await
                .status(),
                StatusCode::FORBIDDEN
            );
            for _ in 0..2 {
                assert_eq!(
                    message(
                        State(state.clone()),
                        local_headers(),
                        Path(snapshot.id.clone()),
                        request()
                    )
                    .await
                    .status(),
                    StatusCode::OK
                );
            }
        }
        let incoming = run.snapshot().incoming;
        assert_eq!(incoming.len(), 2);
        assert_eq!(incoming[0].id, "b".repeat(32));
        assert_eq!(incoming[0].text, "Preserve count_open(tasks)");
        assert_eq!(incoming[1].mode, "queue");
        run.control("stop").unwrap();
        drop(listener);
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while run.snapshot().phase.active() {
            assert!(
                std::time::Instant::now() < deadline,
                "Fixture worker did not stop"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Restore a deterministic historical check receipt, then change its
        // file externally. Both GET and the terminal SSE frame must expose
        // stale evidence; reconnect must never replace it with a raw pass.
        let root = std::fs::canonicalize(temp.path()).unwrap();
        std::fs::write(root.join("answer.txt"), "verified version").unwrap();
        let files =
            crate::chat::coding_project::fingerprints(&root, ["answer.txt".into()].into_iter())
                .unwrap();
        let saved_path = temp
            .path()
            .join("sessions")
            .join(format!("{}.json", snapshot.id));
        let mut saved: Value =
            serde_json::from_slice(&std::fs::read(&saved_path).unwrap()).unwrap();
        saved["snapshot"]["reviews"] =
            json!([{"id":"d".repeat(32),"path":"answer.txt","status":"applied"}]);
        saved["snapshot"]["checks"] = json!([{"id":"e".repeat(32),"run_id":snapshot.run_id,"revision":0,"time":1,"command":"fixture check","status":"passed","output":"Deterministic test receipt","files":files}]);
        std::fs::write(&saved_path, serde_json::to_vec(&saved).unwrap()).unwrap();
        std::fs::write(root.join("answer.txt"), "later manual edit").unwrap();
        let restored = AppState {
            coding_sessions: crate::chat::coding::Manager::new(temp.path().join("sessions")),
            ..state
        };
        let response = get(
            State(restored.clone()),
            local_headers(),
            Path(snapshot.id.clone()),
        )
        .await;
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["checks"][0]["status"], "stale");
        let response = events(State(restored), local_headers(), Path(snapshot.id)).await;
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let stream = String::from_utf8(bytes.to_vec()).unwrap();
        let data = stream
            .lines()
            .find_map(|line| line.strip_prefix("data:"))
            .unwrap();
        let value: Value = serde_json::from_str(data).unwrap();
        assert_eq!(value["checks"][0]["status"], "stale");
    }
    #[tokio::test]
    async fn project_folder_creation_requires_local_authority_without_a_model() {
        let root = tempfile::tempdir().unwrap();
        let mut headers = local_headers();
        headers.insert("origin", "https://unrelated.example".parse().unwrap());
        let request = || CreateFolder {
            parent: root.path().into(),
            name: "Tiny Tasks".into(),
        };
        let denied = create_folder(State(AppState::default()), headers, Json(request())).await;
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);
        assert!(!root.path().join("Tiny Tasks").exists());
        let created =
            create_folder(State(AppState::default()), local_headers(), Json(request())).await;
        assert_eq!(created.status(), StatusCode::CREATED);
        let bytes = axum::body::to_bytes(created.into_body(), 4096)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            value["path"],
            workspace::simplify_path(
                &std::fs::canonicalize(root.path().join("Tiny Tasks")).unwrap()
            )
        );
        let duplicate =
            create_folder(State(AppState::default()), local_headers(), Json(request())).await;
        assert_eq!(duplicate.status(), StatusCode::CONFLICT);
    }
    #[test]
    fn project_folder_creation_does_not_overwrite_or_escape_parent() {
        let root = tempfile::tempdir().unwrap();
        for name in [
            "",
            "..",
            "../escape",
            "child/nested",
            "child\\nested",
            ".git",
            ".camelid",
            "x:y",
        ] {
            assert_eq!(
                create_project_folder(CreateFolder {
                    parent: root.path().into(),
                    name: name.into()
                })
                .unwrap_err()
                .0,
                StatusCode::BAD_REQUEST
            );
        }
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
        let file = root.path().join("existing");
        std::fs::write(&file, "preserve me").unwrap();
        assert_eq!(
            create_project_folder(CreateFolder {
                parent: root.path().into(),
                name: "existing".into()
            })
            .unwrap_err()
            .0,
            StatusCode::CONFLICT
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "preserve me");
        assert!(create_project_folder(CreateFolder {
            parent: file,
            name: "child".into()
        })
        .is_err());
        let protected = root.path().join(".git");
        std::fs::create_dir(&protected).unwrap();
        assert!(create_project_folder(CreateFolder {
            parent: protected.clone(),
            name: "child".into()
        })
        .is_err());
        assert!(!protected.join("child").exists());
        #[cfg(unix)]
        {
            let outside = tempfile::tempdir().unwrap();
            let link = root.path().join("linked");
            std::os::unix::fs::symlink(outside.path(), &link).unwrap();
            assert_eq!(
                create_project_folder(CreateFolder {
                    parent: root.path().into(),
                    name: "linked".into()
                })
                .unwrap_err()
                .0,
                StatusCode::CONFLICT
            );
            assert_eq!(std::fs::read_link(link).unwrap(), outside.path());
        }
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
    async fn file_approval_mode_requires_local_authority() {
        let mut headers = local_headers();
        headers.insert("origin", "https://unrelated.example".parse().unwrap());
        let response = control(
            State(AppState::default()),
            headers,
            Path("a".repeat(32)),
            Json(Control {
                action: "auto_approve_files".into(),
                run_id: Some("b".repeat(32)),
                enabled: Some(true),
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert!(serde_json::from_value::<Control>(
            json!({"action":"auto_approve_files","run_id":"b".repeat(32),"enabled":"true"})
        )
        .is_err());
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
            project: Default::default(),
        };
        let state = AppState::default();
        let response = create(State(state.clone()), local_headers(), Json(request)).await;
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert!(!state.coding_sessions.busy());
    }
}
