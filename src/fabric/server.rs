//! A resident HTTP front door for [`super::Fabric::dispatch`].
//!
//! `fabric run` sends exactly one request per process invocation. This keeps a
//! fabric observed and routable for as long as a real client needs it, without
//! reimplementing placement or forwarding: every request still goes through
//! [`super::Fabric::dispatch`], so the routing and forwarding guarantees are
//! identical to the CLI path.
//!
//! Streaming is refused here for the same reason [`super::forward`] refuses it:
//! an SSE body is a different response shape than the JSON this proxy returns.

use std::net::SocketAddr;
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::Value;

use super::policy::{RouteError, RouteMode, RouteRequest};
use super::{DispatchError, Fabric, ForwardError};

/// Optional header a client sends to request affinity to a specific node.
const STICKY_HEADER: &str = "x-camelid-fabric-sticky";

/// Everything a request needs beyond which nodes exist.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    /// Default placement mode; a client can still request affinity to a
    /// specific node via [`STICKY_HEADER`] regardless of this default.
    pub mode: RouteMode,
    /// Budget for the generation itself, which can legitimately take minutes.
    pub forward_timeout: Duration,
}

#[derive(Clone)]
struct ServerState {
    fabric: Fabric,
    config: ServeConfig,
}

/// Build the router without binding a socket, so tests can drive it directly.
pub fn router(fabric: Fabric, config: ServeConfig) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(ServerState { fabric, config })
}

/// Bind `addr` and serve until the process is killed or the listener errs.
pub async fn serve(addr: SocketAddr, fabric: Fabric, config: ServeConfig) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    serve_on(listener, fabric, config).await
}

/// Serve on an already-bound listener.
///
/// Split from [`serve`] so a test can bind port 0, read back the real port,
/// and only then start serving — `serve` alone never exposes what it bound.
pub async fn serve_on(
    listener: tokio::net::TcpListener,
    fabric: Fabric,
    config: ServeConfig,
) -> std::io::Result<()> {
    axum::serve(listener, router(fabric, config)).await
}

async fn chat_completions(
    State(state): State<ServerState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let model = body.get("model").and_then(Value::as_str);
    let sticky = headers
        .get(STICKY_HEADER)
        .and_then(|value| value.to_str().ok());
    let request = RouteRequest::new(state.config.mode)
        .with_model(model)
        .with_sticky(sticky);

    match state.fabric.dispatch(
        "/v1/chat/completions",
        &body,
        &request,
        state.config.forward_timeout,
    ) {
        Ok((decision, answer)) => {
            let status = StatusCode::from_u16(answer.status).unwrap_or(StatusCode::BAD_GATEWAY);
            let mut response = (status, Json(answer.body)).into_response();
            let out = response.headers_mut();
            insert(out, "x-camelid-fabric-node", &decision.label);
            insert(
                out,
                "x-camelid-fabric-reason",
                &format!("{:?}", decision.reason),
            );
            if let Some(previous) = &decision.affinity_lost {
                insert(out, "x-camelid-fabric-affinity-lost", previous);
            }
            response
        }
        Err(DispatchError::Route(error)) => route_error(error),
        Err(DispatchError::Forward(error)) => forward_error(error),
    }
}

fn insert(headers: &mut HeaderMap, name: &'static str, value: &str) {
    // A label or reason string can never contain characters invalid in a
    // header value in practice, but a malformed one must not crash the
    // response — dropping the header is strictly better than losing the body.
    if let Ok(value) = HeaderValue::from_str(value) {
        headers.insert(name, value);
    }
}

fn route_error(error: RouteError) -> Response {
    error_response(StatusCode::SERVICE_UNAVAILABLE, &error.to_string())
}

fn forward_error(error: ForwardError) -> Response {
    // The node was reachable but refused unsupported input: that is the
    // caller's mistake, not the upstream's, so it is a 400 not a 502/503.
    let status = match &error {
        ForwardError::Unsupported(_) => StatusCode::BAD_REQUEST,
        ForwardError::Transport { .. } | ForwardError::Json { .. } => StatusCode::BAD_GATEWAY,
    };
    error_response(status, &error.to_string())
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({ "error": { "message": message, "type": "fabric_error" } })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::node::NodeSpec;
    use tower::ServiceExt;

    fn request(body: Value) -> axum::http::Request<axum::body::Body> {
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::new(body.to_string()))
            .expect("valid request")
    }

    async fn read_json(response: Response) -> (StatusCode, Value) {
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("collect body");
        (status, serde_json::from_slice(&bytes).expect("json body"))
    }

    #[tokio::test]
    async fn an_empty_fabric_answers_503_not_a_hang() {
        let fabric = Fabric::new(Vec::new());
        let router = router(
            fabric,
            ServeConfig {
                mode: RouteMode::Throughput,
                forward_timeout: Duration::from_millis(200),
            },
        );
        let response = router
            .oneshot(request(serde_json::json!({ "model": "m" })))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["type"], "fabric_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no nodes configured"));
    }

    #[tokio::test]
    async fn a_streaming_request_is_refused_with_400_before_any_probe() {
        // The node is unreachable on purpose: a 400 here proves the refusal
        // happened before dispatch ever tried to route the request.
        let fabric = Fabric::new(vec![NodeSpec {
            label: "dead".to_string(),
            host: "127.0.0.1".to_string(),
            port: 1,
        }]);
        let router = router(
            fabric,
            ServeConfig {
                mode: RouteMode::Throughput,
                forward_timeout: Duration::from_millis(200),
            },
        );
        let response = router
            .oneshot(request(serde_json::json!({ "model": "m", "stream": true })))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "fabric_error");
    }

    #[tokio::test]
    async fn a_malformed_body_is_rejected_before_it_reaches_the_fabric() {
        let fabric = Fabric::new(Vec::new());
        let router = router(
            fabric,
            ServeConfig {
                mode: RouteMode::Throughput,
                forward_timeout: Duration::from_millis(200),
            },
        );
        let bad = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::new("not json".to_string()))
            .expect("built request");
        let response = router.oneshot(bad).await.expect("router answers");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
