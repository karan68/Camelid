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
//! # Two credentials, in opposite directions
//!
//! [`ClientAuth`] is what a client must present to *this* proxy. The bearer the
//! [`Fabric`] was built with is what this proxy presents to *its nodes*. They
//! are configured separately and are not interchangeable: a fabric whose nodes
//! share one key must not thereby accept that key from the network.
//!
//! The check itself is [`crate::api::authenticate`] — the engine's, not a
//! second one. A client sees the same 401 whichever of the two it is talking
//! to, and there is only one constant-time comparison in the crate to get right.
//!
//! # Limits an operator has to know about
//!
//! * Without a [`ClientAuth`], anything that can reach this address can drive
//!   every node in the fabric, so [`bind`] refuses a non-loopback address
//!   unless a key is configured or the risk is acknowledged explicitly.
//! * Authentication is all-or-nothing: there is one key, it is not per-client,
//!   and nothing here revokes or rotates it while the proxy is running.
//! * A non-streaming dispatch runs on a blocking thread and blocking socket I/O
//!   is not cancellable, so a client that hangs up leaves its dispatch running
//!   until the node answers or `forward_timeout` expires. A streaming dispatch
//!   does notice: its next send fails and the node's socket is dropped with it.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{middleware, Json, Router};
use serde_json::Value;

use super::policy::{RouteDecision, RouteError, RouteMode, RouteRequest};
use super::{self as fabric, DispatchError, Fabric, ForwardError, StreamOutcome};
use crate::api::{ApiAuth, DEFAULT_MAX_REQUEST_BODY_BYTES};

/// Optional header a client sends to request affinity to a specific node.
const STICKY_HEADER: &str = "x-camelid-fabric-sticky";

/// The credential a client must present to this proxy.
///
/// Wraps the engine's own [`ApiAuth`] so a key is validated, and later compared,
/// by exactly the rules the engine applies to its own.
#[derive(Clone, Debug)]
pub struct ClientAuth {
    inner: ApiAuth,
}

impl ClientAuth {
    /// Accept every client. Safe only on a loopback listener, which is why
    /// [`bind`] refuses anything else without an acknowledgement.
    pub fn none() -> Self {
        Self {
            inner: ApiAuth::new(None),
        }
    }

    /// Require a key, read from a value or a file.
    ///
    /// Fails rather than starting unauthenticated: a proxy that silently
    /// ignored an unreadable key file would be open to whatever can reach it.
    pub fn resolve(
        api_key: Option<String>,
        api_key_file: Option<PathBuf>,
    ) -> std::io::Result<Self> {
        Ok(Self {
            inner: ApiAuth::new(crate::api::resolve_api_key(api_key, api_key_file)?),
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.inner.enabled()
    }
}

impl Default for ClientAuth {
    fn default() -> Self {
        Self::none()
    }
}

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
    /// What a client must present to be served at all.
    pub auth: ClientAuth,
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
    // Applied last, so it is the outermost layer: an unauthenticated request is
    // refused before anything here reads its body or observes the fabric.
    let auth = config.auth.inner.clone();
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
        .layer(middleware::from_fn_with_state(
            auth,
            crate::api::authenticate,
        ))
}

/// Bind `addr` for the proxy, refusing an exposure the operator did not ask for.
///
/// Binding goes through here rather than through [`tokio::net::TcpListener`]
/// directly so the guard cannot be skipped by forgetting to call it.
pub async fn bind(
    addr: SocketAddr,
    auth: &ClientAuth,
    allow_unauthenticated_remote: bool,
) -> std::io::Result<tokio::net::TcpListener> {
    refuse_unauthenticated_remote(addr, auth.is_enabled(), allow_unauthenticated_remote)?;
    tokio::net::TcpListener::bind(addr).await
}

/// Refuse an unauthenticated listener that the network can reach. Pure.
///
/// An unauthenticated routable bind exposes every node in the fabric, not just
/// this process. `crate::api::server` refuses the same shape of listener on the
/// same three conditions; this mirrors that refusal and its escape hatch.
fn refuse_unauthenticated_remote(
    addr: SocketAddr,
    authenticated: bool,
    allow_unauthenticated_remote: bool,
) -> std::io::Result<()> {
    if addr.ip().is_loopback() || authenticated || allow_unauthenticated_remote {
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!(
            "refusing unauthenticated non-loopback listener {addr}; this would expose every \
             configured node to the network. Bind a loopback address, configure \
             --api-key/--api-key-file, or explicitly acknowledge the risk with \
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

    /// A proxy that accepts every client. Authentication has its own tests;
    /// everything else here is about what the proxy does once a request is in.
    fn open_config() -> ServeConfig {
        ServeConfig {
            mode: RouteMode::Throughput,
            forward_timeout: Duration::from_millis(200),
            auth: ClientAuth::none(),
        }
    }

    fn proxy(fabric: Fabric) -> Router {
        router(fabric, open_config())
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
        let router = router(fabric, open_config());
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
        let router = router(fabric, open_config());
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
        let router = router(fabric, open_config());
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
        let router = router(fabric, open_config());
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
        let router = router(fabric, open_config());
        let oversize = crate::api::DEFAULT_MAX_REQUEST_BODY_BYTES + 1024;
        let body = serde_json::json!({ "model": "m", "pad": "x".repeat(oversize) });
        let response = router.oneshot(request(body)).await.expect("router answers");
        let (status, body) = read_json(response).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(body["error"]["type"], "fabric_error");
    }

    #[test]
    fn a_loopback_listener_needs_no_acknowledgement() {
        refuse_unauthenticated_remote("127.0.0.1:8282".parse().unwrap(), false, false)
            .expect("loopback is not exposed");
        refuse_unauthenticated_remote("[::1]:8282".parse().unwrap(), false, false)
            .expect("loopback is not exposed");
    }

    #[test]
    fn an_unauthenticated_non_loopback_listener_is_refused_by_default() {
        let error = refuse_unauthenticated_remote("0.0.0.0:8282".parse().unwrap(), false, false)
            .expect_err("every node in the fabric would be exposed");
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        // The refusal has to name the way out of it, or it is a dead end.
        assert!(
            error.to_string().contains("--allow-unauthenticated-remote"),
            "{error}"
        );
        // ...including the way out that does not give up authentication.
        assert!(error.to_string().contains("--api-key"), "{error}");
    }

    #[test]
    fn a_non_loopback_listener_starts_once_the_risk_is_acknowledged() {
        refuse_unauthenticated_remote("0.0.0.0:8282".parse().unwrap(), false, true)
            .expect("the operator accepted the exposure");
    }

    /// Requiring a key is what the acknowledgement is an alternative to, so a
    /// key has to satisfy the guard on its own.
    #[test]
    fn a_key_lets_a_non_loopback_listener_start_without_acknowledging_anything() {
        refuse_unauthenticated_remote("0.0.0.0:8282".parse().unwrap(), true, false)
            .expect("the listener is authenticated");
    }

    #[test]
    fn a_resolved_key_reports_itself_as_enabled() {
        assert!(!ClientAuth::none().is_enabled());
        assert!(!ClientAuth::resolve(None, None)
            .expect("no key is not an error")
            .is_enabled());
        assert!(ClientAuth::resolve(Some("k".to_string()), None)
            .expect("a key resolves")
            .is_enabled());
    }

    /// A key that cannot be read has to stop the process, not quietly leave the
    /// proxy open to whatever can reach it.
    #[test]
    fn an_unusable_key_is_an_error_rather_than_an_open_proxy() {
        ClientAuth::resolve(Some(" ".to_string()), None).expect_err("blank is not a key");
        ClientAuth::resolve(None, Some(PathBuf::from("no-such-key-file")))
            .expect_err("an unreadable file is not a key");
        ClientAuth::resolve(Some("k".to_string()), Some(PathBuf::from("f")))
            .expect_err("two sources is ambiguous");
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
