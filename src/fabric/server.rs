//! A resident HTTP front door for [`super::Fabric::dispatch`].
//!
//! `fabric run` sends exactly one request per process invocation. This keeps a
//! fabric observed and routable for as long as a real client needs it, without
//! reimplementing placement or forwarding: every request still goes through
//! [`super::Fabric::dispatch`], so the routing and forwarding guarantees are
//! identical to the CLI path.
//!
//! Streaming is relayed rather than interpreted: when a client asks for
//! `stream: true` the node's server-sent events are forwarded byte for byte, so
//! an event field this proxy has never heard of reaches the client unaltered.
//!
//! # Limits an operator has to know about
//!
//! * There is no authentication layer. Anything that can reach this address can
//!   drive every node in the fabric, so [`bind`] refuses a non-loopback address
//!   unless the risk is acknowledged explicitly.
//! * The token this proxy presents to its nodes is the one the fabric was built
//!   with; it authenticates the proxy to the nodes, never a client to the proxy.
//! * A non-streaming dispatch runs on a blocking thread and blocking socket I/O
//!   is not cancellable, so a client that hangs up leaves its dispatch running
//!   until the node answers or `forward_timeout` expires. A streaming dispatch
//!   does notice: its next send fails and the node's socket is dropped with it.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::Value;

use super::policy::{RouteDecision, RouteError, RouteMode, RouteRequest};
use super::{self as fabric, DispatchError, Fabric, ForwardError, StreamOutcome};
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
    ///
    /// On a streaming request it bounds two waits that mean the same thing —
    /// the wait for the node's response head, and any later gap in which the
    /// node sends nothing. A healthy stream resets the second with every token,
    /// so a long generation is never cut short by it.
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
    // A blank value is not a node name: taking it would silently switch this
    // request to affinity and then report affinity lost to a node called "".
    let sticky = headers
        .get(STICKY_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .map(str::to_string);
    let request = OwnedRequest { model, sticky };

    if fabric::wants_streaming(&body) {
        return stream_completion(state, body, request).await;
    }
    buffered_completion(state, body, request).await
}

/// The parts of a [`RouteRequest`] that outlive the borrow of the request body,
/// so placement can run on another thread.
struct OwnedRequest {
    model: Option<String>,
    sticky: Option<String>,
}

impl OwnedRequest {
    /// A client that names a node is asking for affinity to it, so the header
    /// settles the mode for its own request. Without this the header would be
    /// dead under the throughput default, since placement only consults a
    /// sticky label in [`RouteMode::Affinity`].
    fn as_route(&self, default_mode: RouteMode) -> RouteRequest<'_> {
        let mode = match self.sticky {
            Some(_) => RouteMode::Affinity,
            None => default_mode,
        };
        RouteRequest::new(mode)
            .with_model(self.model.as_deref())
            .with_sticky(self.sticky.as_deref())
    }
}

async fn buffered_completion(state: ServerState, body: Value, request: OwnedRequest) -> Response {
    // Fabric::dispatch is synchronous socket I/O (probes every node, then
    // forwards) and can legitimately run for the whole forward_timeout — up to
    // minutes for a real generation. Running it directly on an async worker
    // thread would starve every other in-flight request once concurrent
    // dispatches reach the runtime's worker count; spawn_blocking moves it onto
    // tokio's much larger blocking pool instead.
    let outcome = tokio::task::spawn_blocking(move || {
        state.fabric.dispatch(
            "/v1/chat/completions",
            &body,
            &request.as_route(state.config.mode),
            state.config.forward_timeout,
        )
    })
    .await;

    match outcome {
        Ok(Ok((decision, answer))) => {
            let status = StatusCode::from_u16(answer.status).unwrap_or(StatusCode::BAD_GATEWAY);
            let mut response = (status, Json(answer.body)).into_response();
            tag(response.headers_mut(), &decision);
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

/// What the blocking side reports once it has the node's response head.
///
/// Sent before any body byte, so the response status and the placement headers
/// are settled before the client is committed to reading a stream.
enum StreamStart {
    Streaming {
        decision: RouteDecision,
        status: u16,
        content_type: Option<String>,
    },
    /// The node answered outright instead of streaming.
    Buffered {
        decision: RouteDecision,
        answer: fabric::Forwarded,
    },
    Failed(DispatchError),
}

/// How many body pieces may sit between the node and a slow client.
///
/// Bounded on purpose: once it fills, the blocking reader stops reading its
/// socket, which is what pushes back on the node rather than buffering a
/// generation in memory.
const STREAM_CHANNEL_DEPTH: usize = 32;

async fn stream_completion(state: ServerState, body: Value, request: OwnedRequest) -> Response {
    let (start_tx, start_rx) = tokio::sync::oneshot::channel::<StreamStart>();
    let (chunk_tx, mut chunk_rx) =
        tokio::sync::mpsc::channel::<Result<Vec<u8>, String>>(STREAM_CHANNEL_DEPTH);

    // Same reasoning as the buffered path: this is blocking socket I/O for the
    // whole life of the generation, so it belongs on the blocking pool. It also
    // owns cancellation — every send below reports whether the client is still
    // there, and the socket to the node is dropped as soon as it is not.
    tokio::task::spawn_blocking(move || {
        let outcome = state.fabric.dispatch_streaming(
            "/v1/chat/completions",
            &body,
            &request.as_route(state.config.mode),
            state.config.forward_timeout,
            state.config.forward_timeout,
        );

        let (mut streaming, _placement) = match outcome {
            Err(error) => {
                let _ = start_tx.send(StreamStart::Failed(error));
                return;
            }
            Ok((StreamOutcome::Buffered(answer), placement)) => {
                let _ = start_tx.send(StreamStart::Buffered {
                    decision: placement.decision().clone(),
                    answer,
                });
                return;
            }
            Ok((StreamOutcome::Streaming(streaming), placement)) => {
                let start = StreamStart::Streaming {
                    decision: placement.decision().clone(),
                    status: streaming.status,
                    content_type: streaming.content_type.clone(),
                };
                if start_tx.send(start).is_err() {
                    return;
                }
                // Bound alongside the stream so the node stays reserved until
                // the last event is read — however this loop ends.
                (streaming, placement)
            }
        };

        loop {
            match streaming.next_chunk() {
                Ok(None) => return,
                Ok(Some(chunk)) => {
                    // A closed channel means the client hung up. Returning drops
                    // the node's socket with it, so the generation is not left
                    // running for nobody.
                    if chunk_tx.blocking_send(Ok(chunk)).is_err() {
                        return;
                    }
                }
                Err(error) => {
                    let _ = chunk_tx.blocking_send(Err(error.to_string()));
                    return;
                }
            }
        }
    });

    let start = match start_rx.await {
        Ok(start) => start,
        Err(_) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "dispatch task did not complete",
            )
        }
    };

    let (decision, status, content_type) = match start {
        StreamStart::Failed(DispatchError::Route(error)) => return route_error(error),
        StreamStart::Failed(DispatchError::Forward(error)) => return forward_error(error),
        StreamStart::Buffered { decision, answer } => {
            let status = StatusCode::from_u16(answer.status).unwrap_or(StatusCode::BAD_GATEWAY);
            let mut response = (status, Json(answer.body)).into_response();
            tag(response.headers_mut(), &decision);
            return response;
        }
        StreamStart::Streaming {
            decision,
            status,
            content_type,
        } => (decision, status, content_type),
    };

    let stream = async_stream::stream! {
        while let Some(chunk) = chunk_rx.recv().await {
            yield chunk;
        }
    };

    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    let out = response.headers_mut();
    // Only the node headers that describe the payload are relayed. Everything
    // else — Content-Length and the hop-by-hop set — describes the node's own
    // connection, and this response is framed independently of it.
    if let Some(content_type) = &content_type {
        insert(out, CONTENT_TYPE, content_type);
    }
    insert(out, CACHE_CONTROL, "no-cache");
    tag(out, &decision);
    response
}

/// Record which node served a request and why, on any answer shape.
fn tag(headers: &mut HeaderMap, decision: &RouteDecision) {
    insert(headers, "x-camelid-fabric-node", &decision.label);
    insert(headers, "x-camelid-fabric-reason", decision.reason.as_str());
    if let Some(previous) = &decision.affinity_lost {
        insert(headers, "x-camelid-fabric-affinity-lost", previous);
    }
}

fn insert<K>(headers: &mut HeaderMap, name: K, value: &str)
where
    K: axum::http::header::IntoHeaderName,
{
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
    async fn a_streaming_request_is_routed_rather_than_refused() {
        // The node is unreachable on purpose. A 503 — a placement failure —
        // proves the proxy tried to route the stream; the 400 this used to
        // answer would mean it had refused before looking at the fabric.
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
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
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

    /// The sticky header is documented as working whatever default the proxy
    /// was started with, and placement only reads a sticky label in affinity
    /// mode — so the header has to settle the mode for its own request.
    #[test]
    fn naming_a_node_asks_for_affinity_whatever_the_default_mode_is() {
        let pinned = OwnedRequest {
            model: None,
            sticky: Some("warm".to_string()),
        };
        let route = pinned.as_route(RouteMode::Throughput);
        assert_eq!(route.mode, RouteMode::Affinity);
        assert_eq!(route.sticky, Some("warm"));

        // Without one, the proxy's configured default is what stands.
        let plain = OwnedRequest {
            model: None,
            sticky: None,
        };
        assert_eq!(
            plain.as_route(RouteMode::Throughput).mode,
            RouteMode::Throughput
        );
        assert_eq!(
            plain.as_route(RouteMode::Affinity).mode,
            RouteMode::Affinity
        );
    }
}
