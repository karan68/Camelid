use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
};
use serde_json::{json, Value};
use tower::ServiceExt;

fn workspace_request(uri: &str, body: Value, authorized: bool) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header("host", "127.0.0.1:8181")
        .header("content-type", "application/json");
    if authorized {
        builder = builder.header("sec-fetch-site", "same-origin");
    }
    builder.body(Body::from(body.to_string())).unwrap()
}

async fn response_json(response: axum::response::Response) -> Value {
    serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap()
}

#[tokio::test]
async fn workspace_create_rejects_missing_browser_provenance_before_path_access() {
    let app = camelid::api::router();
    let response = app
        .oneshot(workspace_request(
            "/api/agent/workspace/sessions",
            json!({
                "workspace": "definitely-missing-workspace-root",
                "goal": "inspect files"
            }),
            false,
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert_eq!(
        response_json(response).await["error"]["code"],
        "local_management_forbidden"
    );
}

#[tokio::test]
async fn workspace_create_rejects_an_inaccessible_root_with_a_typed_error() {
    let app = camelid::api::router();
    let response = app
        .oneshot(workspace_request(
            "/api/agent/workspace/sessions",
            json!({
                "workspace": "definitely-missing-workspace-root",
                "goal": "inspect files"
            }),
            true,
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response_json(response).await["error"]["code"],
        "workspace_root_not_accessible"
    );
}

#[tokio::test]
async fn workspace_create_requires_a_loaded_tool_capable_model_after_root_validation() {
    let root = tempfile::tempdir().unwrap();
    let app = camelid::api::router();
    let response = app
        .oneshot(workspace_request(
            "/api/agent/workspace/sessions",
            json!({
                "workspace": root.path(),
                "goal": "inspect files"
            }),
            true,
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert_eq!(
        response_json(response).await["error"]["code"],
        "model_not_loaded"
    );
}

#[tokio::test]
async fn workspace_decision_for_an_unknown_session_is_typed_not_found() {
    let app = camelid::api::router();
    let response = app
        .oneshot(workspace_request(
            "/api/agent/workspace/sessions/missing/decisions",
            json!({
                "approval_id": "missing",
                "decision": "deny"
            }),
            true,
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        response_json(response).await["error"]["code"],
        "workspace_session_not_found"
    );
}
