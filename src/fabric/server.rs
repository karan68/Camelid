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
//!
//! # Limits an operator has to know about
//!
//! * There is no authentication layer. Anything that can reach this address can
//!   drive every node in the fabric, so [`bind`] refuses a non-loopback address
//!   unless the risk is acknowledged explicitly.
//! * The token this proxy presents to its nodes is the one the fabric was built
//!   with; it authenticates the proxy to the nodes, never a client to the proxy.
//! * A dispatch runs on a blocking thread and blocking socket I/O is not
//!   cancellable, so a client that hangs up leaves its dispatch running until
//!   the node answers or `forward_timeout` expires.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::Value;

use super::policy::{RouteError, RouteMode, RouteRequest};
use super::{DispatchError, Fabric, ForwardError};
use crate::api::DEFAULT_MAX_REQUEST_BODY_BYTES;

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
    /// Shared rather than cloned: axum clones the state for every request, and
    /// a `Fabric` owns a `Vec<NodeSpec>`.
    fabric: Arc<Fabric>,
    config: ServeConfig,
}

/// Build the router without binding a socket, so tests can drive it directly.
pub fn router(fabric: Fabric, config: ServeConfig) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
        // Without this, axum's 2 MiB default would refuse conversations the
        // node behind the proxy accepts.
        .layer(DefaultBodyLimit::max(DEFAULT_MAX_REQUEST_BODY_BYTES))
        .with_state(ServerState {
            fabric: Arc::new(fabric),
            config,
        })
}

/// Bind `addr` for the proxy, refusing an exposure the operator did not ask for.
///
/// Binding goes through here rather than through [`tokio::net::TcpListener`]
/// directly so the guard cannot be skipped by forgetting to call it.
pub async fn bind(
    addr: SocketAddr,
    allow_unauthenticated_remote: bool,
) -> std::io::Result<tokio::net::TcpListener> {
    refuse_unauthenticated_remote(addr, allow_unauthenticated_remote)?;
    tokio::net::TcpListener::bind(addr).await
}

/// Refuse an unauthenticated listener that the network can reach. Pure.
///
/// This proxy has no auth layer at all, so a routable bind exposes every node
/// in the fabric, not just this process. `crate::api::server` refuses the same
/// shape of listener; this mirrors that refusal and its escape hatch.
fn refuse_unauthenticated_remote(
    addr: SocketAddr,
    allow_unauthenticated_remote: bool,
) -> std::io::Result<()> {
    if addr.ip().is_loopback() || allow_unauthenticated_remote {
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!(
            "refusing unauthenticated non-loopback listener {addr}; the fabric proxy has no \
             authentication of its own, so this would expose every configured node to the \
             network. Bind a loopback address, or explicitly acknowledge the risk with \
             --allow-unauthenticated-remote"
        ),
    ))
}

/// Serve on an already-bound listener.
///
/// Split from [`bind`] so a caller can read back the real port — which matters
/// when it asked for port 0 — before it starts serving.
pub async fn serve_on(
    listener: tokio::net::TcpListener,
    fabric: Fabric,
    config: ServeConfig,
) -> std::io::Result<()> {
    axum::serve(listener, router(fabric, config)).await
}

/// List every model the fabric can currently route to.
///
/// A client points at one address, so it has to be able to ask that address
/// what it can serve; the per-node listing is not reachable through the proxy.
/// The answer is the union across ready nodes, in the same shape a node
/// answers, minus the per-file `meta`: two nodes can serve one model id from
/// different files, and there is no honest way to merge that.
///
/// No ready node means an empty list rather than an error. "Nothing right now"
/// is the truthful answer to a discovery question, and a model picker can show
/// it without special-casing a failure.
async fn models(State(state): State<ServerState>) -> Response {
    let fabric = Arc::clone(&state.fabric);
    // Observing is blocking socket I/O against every node, same as a dispatch.
    let snapshots = match tokio::task::spawn_blocking(move || fabric.observe()).await {
        Ok(snapshots) => snapshots,
        Err(join_error) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("could not observe the fabric: {join_error}"),
            )
        }
    };

    let data: Vec<Value> = super::servable_models(&snapshots)
        .into_iter()
        .map(|id| {
            serde_json::json!({
                "id": id,
                "object": "model",
                "created": 0,
                "owned_by": "camelid",
            })
        })
        .collect();

    Json(serde_json::json!({ "object": "list", "data": data })).into_response()
}

async fn chat_completions(
    State(state): State<ServerState>,
    headers: HeaderMap,
    payload: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    let body = match payload {
        Ok(Json(body)) => body,
        Err(rejection) => return rejected_body(rejection),
    };
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);
    let sticky = headers
        .get(STICKY_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);

    // Fabric::dispatch is synchronous socket I/O (probes every node, then
    // forwards) and can legitimately run for the whole forward_timeout — up to
    // minutes for a real generation. Running it directly on an async worker
    // thread would starve every other in-flight request once concurrent
    // dispatches reach the runtime's worker count; spawn_blocking moves it onto
    // tokio's much larger blocking pool instead.
    let outcome = tokio::task::spawn_blocking(move || {
        let request = RouteRequest::new(state.config.mode)
            .with_model(model.as_deref())
            .with_sticky(sticky.as_deref());
        state.fabric.dispatch(
            "/v1/chat/completions",
            &body,
            &request,
            state.config.forward_timeout,
        )
    })
    .await;

    match outcome {
        Ok(Ok((decision, answer))) => {
            let status = StatusCode::from_u16(answer.status).unwrap_or(StatusCode::BAD_GATEWAY);
            let mut response = (status, Json(answer.body)).into_response();
            let out = response.headers_mut();
            insert(out, "x-camelid-fabric-node", &decision.label);
            insert(out, "x-camelid-fabric-reason", decision.reason.as_str());
            if let Some(previous) = &decision.affinity_lost {
                insert(out, "x-camelid-fabric-affinity-lost", previous);
            }
            response
        }
        Ok(Err(DispatchError::Route(error))) => route_error(error),
        Ok(Err(DispatchError::Forward(error))) => forward_error(error),
        Err(join_error) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("dispatch task did not complete: {join_error}"),
        ),
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

/// Answer a body axum refused in the same shape every other answer here uses.
///
/// Mirrors `crate::api::malformed_json_error`: the default `JsonRejection`
/// response is plain text, which would leave the rejection a client is most
/// likely to hit as the only one it cannot parse.
fn rejected_body(rejection: JsonRejection) -> Response {
    let status = if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
        StatusCode::PAYLOAD_TOO_LARGE
    } else {
        StatusCode::BAD_REQUEST
    };
    error_response(status, &rejection.to_string())
}

fn route_error(error: RouteError) -> Response {
    match &error {
        // Asking again will not make the model appear, and the nodes themselves
        // answer 404 `model_not_found` for exactly this. Answering 503 told
        // clients to retry, so an SDK spent its whole retry budget on a refusal
        // that could never change.
        //
        // Only when every node answered, though: a node still unaccounted for
        // may be the one that owns the model, and that refusal can clear.
        RouteError::ModelUnavailable { unobserved: 0, .. } => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": error.to_string(),
                    "type": "fabric_error",
                    "code": "model_not_found",
                    "param": "model",
                }
            })),
        )
            .into_response(),
        // These can genuinely clear on their own, so they stay retryable.
        RouteError::ModelUnavailable { .. }
        | RouteError::NoNodesConfigured
        | RouteError::AllNodesUnavailable { .. } => {
            error_response(StatusCode::SERVICE_UNAVAILABLE, &error.to_string())
        }
    }
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

    fn get_request(uri: &str) -> axum::http::Request<axum::body::Body> {
        axum::http::Request::builder()
            .method("GET")
            .uri(uri)
            .body(axum::body::Body::empty())
            .expect("valid request")
    }

    fn proxy(fabric: Fabric) -> Router {
        router(
            fabric,
            ServeConfig {
                mode: RouteMode::Throughput,
                forward_timeout: Duration::from_millis(200),
            },
        )
    }

    /// A client points at one address, so that address has to answer what it
    /// can serve. This used to 404, which left every model picker empty.
    #[tokio::test]
    async fn the_models_route_answers_a_list_even_with_nothing_to_serve() {
        let response = proxy(Fabric::new(Vec::new()))
            .oneshot(get_request("/v1/models"))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "list");
        assert_eq!(
            body["data"].as_array().expect("a data array").len(),
            0,
            "nothing ready is an empty list, not an error"
        );
    }

    /// The empty case above passes just as well against a route that can only
    /// ever answer nothing, so this is the one that proves it observes.
    #[tokio::test]
    async fn the_models_route_lists_what_a_ready_node_is_serving() {
        let node = StubModelNode::start("llama-3b");
        let fabric = Fabric::new(vec![node.spec("only")]).with_timeout(Duration::from_secs(2));
        let response = proxy(fabric)
            .oneshot(get_request("/v1/models"))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "list");
        let data = body["data"].as_array().expect("a data array");
        assert_eq!(data.len(), 1, "{body}");
        assert_eq!(data[0]["id"], "llama-3b");
        assert_eq!(data[0]["object"], "model");
        assert_eq!(data[0]["owned_by"], "camelid");
    }

    /// A model this fabric does not serve will not appear by asking again, and
    /// the nodes answer 404 `model_not_found` for it. Answering 503 told
    /// clients to retry: an OpenAI SDK spent its whole retry budget on a
    /// permanent refusal.
    #[tokio::test]
    async fn an_unservable_model_is_refused_as_not_found_not_as_try_again() {
        let node = StubModelNode::start("llama-3b");
        let fabric = Fabric::new(vec![node.spec("only")]).with_timeout(Duration::from_secs(2));
        let response = proxy(fabric)
            .oneshot(request(
                serde_json::json!({ "model": "no-such-model", "messages": [] }),
            ))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["param"], "model");
        // The message still names what the fabric *can* serve, which is the
        // part an operator acts on.
        let message = body["error"]["message"].as_str().expect("a message");
        assert!(message.contains("llama-3b"), "{message}");
    }

    /// The same refusal is *not* final when a node could not be consulted: the
    /// node that is down may be the one that owns the model, so a 404 would
    /// tell an SDK to give up on something a retry would find. This is the
    /// difference between a model that is absent and a fabric that cannot yet
    /// say.
    #[tokio::test]
    async fn a_model_missing_only_because_a_node_is_down_stays_retryable() {
        let node = StubModelNode::start("llama-3b");
        let fabric = Fabric::new(vec![
            node.spec("up"),
            // Port 1 is closed, so this node is observed as unreachable.
            NodeSpec {
                label: "down".to_string(),
                host: "127.0.0.1".to_string(),
                port: 1,
            },
        ])
        .with_timeout(Duration::from_millis(500));

        let response = proxy(fabric)
            .oneshot(request(
                serde_json::json!({ "model": "no-such-model", "messages": [] }),
            ))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;

        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "a node that never answered may be the one serving it: {body}"
        );
    }

    /// A fabric whose nodes are merely unreachable can recover on its own, so
    /// that refusal stays retryable.
    #[tokio::test]
    async fn an_unreachable_fabric_is_still_a_retryable_503() {
        let fabric = Fabric::new(vec![NodeSpec {
            label: "dead".to_string(),
            host: "127.0.0.1".to_string(),
            port: 1,
        }])
        .with_timeout(Duration::from_millis(200));
        let response = proxy(fabric)
            .oneshot(request(serde_json::json!({ "model": "m", "messages": [] })))
            .await
            .expect("router answers");
        let (status, _) = read_json(response).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    /// A node stub answering only `/v1/health`, which is all placement reads.
    struct StubModelNode {
        port: u16,
        shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl StubModelNode {
        fn start(model: &str) -> Self {
            use std::io::{Read, Write};
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
            let port = listener.local_addr().expect("addr").port();
            let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let flag = std::sync::Arc::clone(&shutdown);
            let body = format!(
                r#"{{"ok":true,"generation_ready":true,"active_model_id":"{model}","backend":"llama","version":"0.6.1","engine_queued_tasks":0,"engine_queue_depth":0}}"#
            );
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    if flag.load(std::sync::atomic::Ordering::SeqCst) {
                        break;
                    }
                    let Ok(mut stream) = stream else { continue };
                    // Read to the end of the head before answering: closing a
                    // socket that still holds unread bytes is an abortive close
                    // on Windows, and it discards the reply along with them.
                    let mut request = Vec::new();
                    let mut scratch = [0_u8; 1024];
                    while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                        match stream.read(&mut scratch) {
                            Ok(0) | Err(_) => break,
                            Ok(read) => request.extend_from_slice(&scratch[..read]),
                        }
                    }
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes());
                }
            });
            Self { port, shutdown }
        }

        fn spec(&self, label: &str) -> NodeSpec {
            NodeSpec {
                label: label.to_string(),
                host: "127.0.0.1".to_string(),
                port: self.port,
            }
        }
    }

    impl Drop for StubModelNode {
        fn drop(&mut self) {
            self.shutdown
                .store(true, std::sync::atomic::Ordering::SeqCst);
            let _ = std::net::TcpStream::connect(("127.0.0.1", self.port));
        }
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
        let (status, body) = read_json(response).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        // The rejection a client is most likely to hit must not be the one
        // error it cannot parse: axum's own JsonRejection body is plain text.
        assert_eq!(body["error"]["type"], "fabric_error");
        assert!(!body["error"]["message"].as_str().unwrap().is_empty());
    }

    /// A body the node behind the proxy would accept must not be refused by the
    /// proxy. Axum's default limit is 2 MiB, eight times stricter than the
    /// node's own default.
    #[tokio::test]
    async fn a_body_over_the_axum_default_still_reaches_the_fabric() {
        let fabric = Fabric::new(Vec::new());
        let router = router(
            fabric,
            ServeConfig {
                mode: RouteMode::Throughput,
                forward_timeout: Duration::from_millis(200),
            },
        );
        let body = serde_json::json!({ "model": "m", "pad": "x".repeat(4 * 1024 * 1024) });
        let response = router.oneshot(request(body)).await.expect("router answers");
        let (status, body) = read_json(response).await;
        // 503 means the body was read and placement ran; 413 would mean the
        // proxy refused a request the node would have served.
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["type"], "fabric_error");
    }

    #[tokio::test]
    async fn a_body_over_the_proxy_limit_is_refused_in_the_fabric_error_shape() {
        let fabric = Fabric::new(Vec::new());
        let router = router(
            fabric,
            ServeConfig {
                mode: RouteMode::Throughput,
                forward_timeout: Duration::from_millis(200),
            },
        );
        let oversize = crate::api::DEFAULT_MAX_REQUEST_BODY_BYTES + 1024;
        let body = serde_json::json!({ "model": "m", "pad": "x".repeat(oversize) });
        let response = router.oneshot(request(body)).await.expect("router answers");
        let (status, body) = read_json(response).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(body["error"]["type"], "fabric_error");
    }

    #[test]
    fn a_loopback_listener_needs_no_acknowledgement() {
        refuse_unauthenticated_remote("127.0.0.1:8282".parse().unwrap(), false)
            .expect("loopback is not exposed");
        refuse_unauthenticated_remote("[::1]:8282".parse().unwrap(), false)
            .expect("loopback is not exposed");
    }

    #[test]
    fn an_unauthenticated_non_loopback_listener_is_refused_by_default() {
        let error = refuse_unauthenticated_remote("0.0.0.0:8282".parse().unwrap(), false)
            .expect_err("every node in the fabric would be exposed");
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        // The refusal has to name the way out of it, or it is a dead end.
        assert!(
            error.to_string().contains("--allow-unauthenticated-remote"),
            "{error}"
        );
    }

    #[test]
    fn a_non_loopback_listener_starts_once_the_risk_is_acknowledged() {
        refuse_unauthenticated_remote("0.0.0.0:8282".parse().unwrap(), true)
            .expect("the operator accepted the exposure");
    }
}
