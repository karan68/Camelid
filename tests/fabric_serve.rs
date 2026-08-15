//! End-to-end tests for the resident fabric proxy (`camelid fabric serve`).
//!
//! [`fabric_end_to_end.rs`] proves `Fabric::dispatch` itself; these tests prove
//! the HTTP front door around it — a real client, over a real socket, talking
//! to the real router bound by [`camelid::fabric::server::serve_on`], routed to
//! stub nodes on loopback.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use camelid::fabric::server::{serve_on, serve_on_until, ClientAuth, ServeConfig};
use camelid::fabric::{Fabric, NodeSpec, RouteMode, ENGINE_QUEUE_FULL_CODE};

const PROBE_TIMEOUT: Duration = Duration::from_secs(3);
const FORWARD_TIMEOUT: Duration = Duration::from_secs(5);

/// One request a stub received: enough to say which route it was, what
/// credential the proxy presented on it, and what it asked for.
#[derive(Debug, Clone)]
struct Received {
    path: String,
    /// The `Authorization` header verbatim, or `None` if the request carried none.
    authorization: Option<String>,
    body: String,
}

#[derive(Clone)]
struct StubConfig {
    health: String,
    completion: String,
    completion_status: u16,
    /// Held before answering `/v1/chat/completions`, to stand in for a real
    /// generation that takes a while. Health is never delayed.
    completion_delay: Duration,
    /// When set, every route but `/v1/health` answers 401 unless the request
    /// carries exactly this bearer token — the arrangement a node started with
    /// `CAMELID_API_KEY` actually presents to the proxy.
    required_key: Option<String>,
    /// Hang up on `/v1/chat/completions` after reading the request instead of
    /// answering it. The node has the request, so nothing may re-send it.
    hangs_up_on_completion: bool,
    /// Server-sent events answered to a request carrying `"stream": true`,
    /// instead of one JSON body.
    stream_events: Vec<String>,
    /// Gap left between events. A test that measures arrival times uses it to
    /// tell a relayed stream from one that was buffered and released at the end.
    stream_gap: Duration,
    /// Hang up after the events instead of writing the terminal chunk, which is
    /// what a node that dies mid-generation leaves on the wire.
    stream_truncated: bool,
    /// Events actually written to the socket. It stops advancing once the peer
    /// has gone, which is how a cancellation test observes the hang-up.
    events_written: Arc<AtomicUsize>,
    /// When set, `/v1/health` reports this live count rather than a fixed one,
    /// and a completion holds it up for its whole duration. A real engine's
    /// `engine_queue_depth` behaves this way, and a placement test is only
    /// meaningful against a node whose reported load actually moves.
    live_in_flight: Option<Arc<AtomicUsize>>,
    /// When set, `/v1/health` reports this model rather than the fixed one, so
    /// a test can load a different model while the proxy is running.
    live_model: Option<Arc<Mutex<String>>>,
}

impl StubConfig {
    fn ready(model: &str, in_flight: usize) -> Self {
        Self {
            health: format!(
                r#"{{"ok":true,"generation_ready":true,"active_model_id":"{model}",
                    "backend":"llama","version":"0.5.4",
                    "engine_queued_tasks":0,"engine_queue_depth":{in_flight}}}"#
            ),
            completion: format!(
                r#"{{"choices":[{{"message":{{"role":"assistant","content":"served by {model}"}}}}]}}"#
            ),
            completion_status: 200,
            completion_delay: Duration::ZERO,
            required_key: None,
            hangs_up_on_completion: false,
            stream_events: Vec::new(),
            stream_gap: Duration::ZERO,
            stream_truncated: false,
            events_written: Arc::new(AtomicUsize::new(0)),
            live_in_flight: None,
            live_model: None,
        }
    }

    /// A node that reports the load it is really carrying, and takes `delay`
    /// to answer a completion.
    fn reporting_live_load(model: &str, delay: Duration) -> Self {
        Self {
            completion_delay: delay,
            live_in_flight: Some(Arc::new(AtomicUsize::new(0))),
            ..Self::ready(model, 0)
        }
    }

    /// A node answering exactly what a real engine answers once its bounded
    /// queue turns a request away: the typed code, the `runtime_unavailable`
    /// envelope a 503 carries, and the message naming the bound.
    fn queue_full(model: &str) -> Self {
        Self::refusing_with(
            model,
            ENGINE_QUEUE_FULL_CODE,
            "the generation queue is full; retry shortly (depth is bounded by CAMELID_QUEUE_DEPTH)",
        )
    }

    /// A node refusing with a 503 that is *not* backpressure. A node with no
    /// model loaded answers one of these, and sending that request on would
    /// spend another node's time reaching the same refusal.
    fn refusing_with(model: &str, code: &str, message: &str) -> Self {
        Self {
            completion_status: 503,
            completion: format!(
                r#"{{"error":{{"message":"{message}","type":"runtime_unavailable","code":"{code}"}}}}"#
            ),
            ..Self::ready(model, 0)
        }
    }

    /// A node whose active model can be changed while the proxy is running, the
    /// way an operator loading a different model changes it.
    fn with_switchable_model(model: &str) -> Self {
        Self {
            live_model: Some(Arc::new(Mutex::new(model.to_string()))),
            ..Self::ready(model, 0)
        }
    }

    /// A node that answers a streaming request with server-sent events.
    fn streaming(model: &str, events: &[&str], gap: Duration) -> Self {
        Self {
            stream_events: events.iter().map(|event| (*event).to_string()).collect(),
            stream_gap: gap,
            ..Self::ready(model, 0)
        }
    }

    /// A node that dies part-way through a stream: the events it managed to
    /// produce, then a hang-up with no terminal chunk.
    fn dying_mid_stream(model: &str, events: &[&str], gap: Duration) -> Self {
        Self {
            stream_truncated: true,
            ..Self::streaming(model, events, gap)
        }
    }

    /// A node with an API key set: health stays open, generation does not.
    fn requiring_key(model: &str, key: &str) -> Self {
        Self {
            required_key: Some(key.to_string()),
            ..Self::ready(model, 0)
        }
    }

    fn refusing(model: &str) -> Self {
        Self {
            completion: r#"{"error":{"message":"engine queue full"}}"#.to_string(),
            completion_status: 503,
            ..Self::ready(model, 0)
        }
    }

    fn slow(model: &str, delay: Duration) -> Self {
        Self {
            completion_delay: delay,
            ..Self::ready(model, 0)
        }
    }

    /// A node that takes the request and then dies without answering, which is
    /// what a node crashing mid-generation leaves on the wire.
    fn hanging_up_on_completion(model: &str) -> Self {
        Self {
            hangs_up_on_completion: true,
            ..Self::ready(model, 0)
        }
    }
}

/// A stand-in Camelid node on loopback, identical in shape to the one in
/// `fabric_end_to_end.rs` (duplicated rather than shared, since each
/// integration test file is its own crate).
struct StubNode {
    port: u16,
    shutdown: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<Received>>>,
    thread: Option<JoinHandle<()>>,
}

impl StubNode {
    fn start(config: StubConfig) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();
        let shutdown = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let thread_shutdown = Arc::clone(&shutdown);
        let thread_requests = Arc::clone(&requests);
        let config = Arc::new(config);
        let thread = std::thread::spawn(move || {
            for stream in listener.incoming() {
                if thread_shutdown.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(mut stream) = stream else { continue };
                // Each connection gets its own thread: a real node answers a
                // health probe while it is generating for someone else, and a
                // serial accept loop would make the proxy look slower than it
                // is when several requests probe the same node at once.
                let config = Arc::clone(&config);
                let requests = Arc::clone(&thread_requests);
                std::thread::spawn(move || serve_once(&mut stream, &config, &requests));
            }
        });
        Self {
            port,
            shutdown,
            requests,
            thread: Some(thread),
        }
    }

    fn spec(&self, label: &str) -> NodeSpec {
        NodeSpec {
            label: label.to_string(),
            host: "127.0.0.1".to_string(),
            port: self.port,
        }
    }

    fn received(&self) -> Vec<Received> {
        self.requests.lock().expect("stub lock").clone()
    }

    /// Stop accepting, releasing the port so it refuses connections the way a
    /// machine that has gone does. What it already recorded stays readable.
    fn stop_listening(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Unblocks the accept loop so it observes the flag and drops the
        // listener; the connection itself is never served.
        let _ = TcpStream::connect(("127.0.0.1", self.port));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for StubNode {
    fn drop(&mut self) {
        self.stop_listening();
    }
}

fn serve_once(stream: &mut TcpStream, config: &StubConfig, requests: &Mutex<Vec<Received>>) {
    let Some(received) = read_request(stream) else {
        return;
    };

    // Mirrors `route_requires_auth` in the real server: `/v1/health` is exempt,
    // everything else is gated.
    let authorized = match &config.required_key {
        Some(key) => {
            received.path == "/v1/health"
                || received.authorization.as_deref() == Some(&format!("Bearer {key}"))
        }
        None => true,
    };

    let asked_to_stream = serde_json::from_str::<Value>(&received.body)
        .ok()
        .and_then(|body| body.get("stream").and_then(Value::as_bool))
        .unwrap_or(false);

    if authorized && asked_to_stream && !config.stream_events.is_empty() {
        requests.lock().expect("stub lock").push(received);
        serve_event_stream(stream, config);
        return;
    }

    if authorized && config.hangs_up_on_completion && received.path == "/v1/chat/completions" {
        // Recorded first: the point of this stub is that the node did get the
        // request, whatever the caller ends up seeing.
        requests.lock().expect("stub lock").push(received);
        return;
    }

    let (status, body) = match received.path.as_str() {
        _ if !authorized => (
            401_u16,
            r#"{"error":{"message":"provide Authorization: Bearer <key> or X-API-Key"}}"#
                .to_string(),
        ),
        "/v1/health" => (200_u16, health_body(config)),
        "/v1/chat/completions" => {
            // Counted for exactly as long as the node is working on it.
            let busy = config.live_in_flight.as_ref().map(|count| {
                count.fetch_add(1, Ordering::SeqCst);
                Arc::clone(count)
            });
            if !config.completion_delay.is_zero() {
                std::thread::sleep(config.completion_delay);
            }
            if let Some(count) = busy {
                count.fetch_sub(1, Ordering::SeqCst);
            }
            (config.completion_status, config.completion.clone())
        }
        _ => (404, "{}".to_string()),
    };

    // Record only after deciding, so a malformed request never poisons the log.
    requests.lock().expect("stub lock").push(received);

    let response = format!(
        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

/// What `/v1/health` answers: the configured body, carrying whatever load and
/// model the node is really reporting when it is tracking those.
fn health_body(config: &StubConfig) -> String {
    let mut body: Value = serde_json::from_str(&config.health).expect("stub health is json");
    if let Some(count) = &config.live_in_flight {
        body["engine_queue_depth"] = count.load(Ordering::SeqCst).into();
    }
    if let Some(model) = &config.live_model {
        body["active_model_id"] = Value::String(model.lock().expect("stub model lock").clone());
    }
    body.to_string()
}

/// Answer with chunked `text/event-stream`, one HTTP chunk per event, exactly
/// as an axum `Sse` response reaches the wire.
fn serve_event_stream(stream: &mut TcpStream, config: &StubConfig) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                Cache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\
                Connection: close\r\n\r\n";
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }
    let _ = stream.flush();

    for event in &config.stream_events {
        if !config.stream_gap.is_zero() {
            std::thread::sleep(config.stream_gap);
        }
        let frame = format!("{:x}\r\n{event}\r\n", event.len());
        // A failed write is the peer having gone; stop rather than keep
        // generating for nobody, which is what a real node's channel does.
        if stream.write_all(frame.as_bytes()).is_err() || stream.flush().is_err() {
            return;
        }
        config.events_written.fetch_add(1, Ordering::SeqCst);
    }
    if config.stream_truncated {
        return;
    }
    let _ = stream.write_all(b"0\r\n\r\n");
    let _ = stream.flush();
}

fn read_request(stream: &mut TcpStream) -> Option<Received> {
    stream.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
    let mut raw = Vec::new();
    let mut chunk = [0_u8; 1024];
    loop {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&chunk[..read]);
        if let Some(header_end) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&raw[..header_end]).to_string();
            let expected = head
                .lines()
                .find(|line| line.to_ascii_lowercase().starts_with("content-length:"))
                .and_then(|line| line.split(':').nth(1))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if raw.len() >= header_end + 4 + expected {
                let mut parts = head.lines().next()?.split_whitespace();
                parts.next()?; // method
                let body_start = header_end + 4;
                return Some(Received {
                    path: parts.next()?.to_string(),
                    authorization: authorization(&head),
                    body: String::from_utf8_lossy(&raw[body_start..body_start + expected])
                        .to_string(),
                });
            }
        }
    }
    None
}

/// The `Authorization` header from a request head, matched case-insensitively
/// the way a real server matches it.
fn authorization(head: &str) -> Option<String> {
    head.lines()
        .skip(1)
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.trim().eq_ignore_ascii_case("authorization"))
        .map(|(_, value)| value.trim().to_string())
}

fn fabric_of(specs: Vec<NodeSpec>) -> Fabric {
    Fabric::new(specs).with_timeout(PROBE_TIMEOUT)
}

/// A fabric that reuses an observation, as `fabric serve` builds one.
fn fabric_reusing_observations(specs: Vec<NodeSpec>, max_age: Duration) -> Fabric {
    fabric_of(specs).with_max_observation_age(max_age)
}

/// Health probes every node saw. The whole point of a freshness bound is that
/// this stops growing with the number of client requests.
fn health_probes(nodes: &[StubNode]) -> usize {
    nodes
        .iter()
        .map(|node| {
            node.received()
                .iter()
                .filter(|received| received.path == "/v1/health")
                .count()
        })
        .sum()
}

fn completions_served(nodes: &[StubNode]) -> Vec<usize> {
    nodes
        .iter()
        .map(|node| {
            node.received()
                .iter()
                .filter(|received| received.path == "/v1/chat/completions")
                .count()
        })
        .collect()
}

/// Bind the real proxy on an OS-assigned port and start serving it in the
/// background, returning the address a client should connect to.
async fn start_proxy(fabric: Fabric, mode: RouteMode) -> SocketAddr {
    start_proxy_with_auth(fabric, mode, ClientAuth::none()).await
}

async fn start_proxy_with_auth(fabric: Fabric, mode: RouteMode, auth: ClientAuth) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let addr = listener.local_addr().expect("proxy addr");
    let config = ServeConfig {
        mode,
        forward_timeout: FORWARD_TIMEOUT,
        auth,
        bound: addr,
    };
    tokio::spawn(async move {
        let _ = serve_on(listener, fabric, config).await;
    });
    addr
}

/// Send one POST over a real socket and return (status, parsed body, headers).
async fn post_chat(
    addr: SocketAddr,
    body: &Value,
    extra_headers: &[(&str, &str)],
) -> (u16, Value, Vec<(String, String)>) {
    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("connect to proxy");
    let payload = body.to_string();
    let mut request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        payload.len()
    );
    for (name, value) in extra_headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str("\r\n");
    request.push_str(&payload);
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.expect("read response");
    let text = String::from_utf8_lossy(&raw);
    let header_end = text.find("\r\n\r\n").expect("header terminator");
    let head = &text[..header_end];
    let body_text = &text[header_end + 4..];

    let mut lines = head.lines();
    let status: u16 = lines
        .next()
        .expect("status line")
        .split_whitespace()
        .nth(1)
        .expect("status code")
        .parse()
        .expect("numeric status");
    let headers = lines
        .filter_map(|line| line.split_once(": "))
        .map(|(name, value)| (name.to_ascii_lowercase(), value.to_string()))
        .collect();
    let json: Value = serde_json::from_str(body_text).expect("json body");
    (status, json, headers)
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_request_is_placed_and_the_backend_answer_returned_verbatim() {
    let node = StubNode::start(StubConfig::ready("model-alpha", 0));
    let fabric = fabric_of(vec![node.spec("solo")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "model-alpha" }), &[]).await;

    assert_eq!(status, 200);
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "served by model-alpha"
    );
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("solo"));
}

#[tokio::test(flavor = "multi_thread")]
async fn placement_through_the_proxy_is_scoped_to_the_serving_node() {
    let alpha = StubNode::start(StubConfig::ready("model-alpha", 0));
    let beta = StubNode::start(StubConfig::ready("model-beta", 0));
    let fabric = fabric_of(vec![alpha.spec("alpha"), beta.spec("beta")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "model-beta" }), &[]).await;

    assert_eq!(status, 200);
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "served by model-beta"
    );
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("beta"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_node_5xx_reaches_the_client_verbatim_not_as_a_proxy_error() {
    let node = StubNode::start(StubConfig::refusing("model-alpha"));
    let fabric = fabric_of(vec![node.spec("solo")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, _headers) =
        post_chat(addr, &serde_json::json!({ "model": "model-alpha" }), &[]).await;

    assert_eq!(status, 503);
    assert_eq!(body["error"]["message"], "engine queue full");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_proxys_bearer_reaches_the_node_on_every_request_it_makes() {
    // `fabric serve --bearer` is only worth having if the token survives the
    // whole resident path. The proxy has no auth of its own, so the client sends
    // none: this asserts the credential on the wire is the fabric's, presented
    // to the node.
    let node = StubNode::start(StubConfig::requiring_key("model-alpha", "s3cret"));
    let fabric = fabric_of(vec![node.spec("solo")]).with_bearer(Some("s3cret"));
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "model-alpha" }), &[]).await;

    assert_eq!(status, 200);
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "served by model-alpha"
    );
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("solo"));

    // The probe is authenticated as well as the forward. Health is exempt from
    // the node's auth today, so that buys nothing now — it means placement keeps
    // working if the exemption is ever tightened.
    let seen = node.received();
    assert!(
        seen.iter().any(|request| request.path == "/v1/health"),
        "the node was probed as well as forwarded to"
    );
    assert!(
        seen.iter()
            .any(|request| request.path == "/v1/chat/completions"),
        "the forward reached the node"
    );
    for request in &seen {
        assert_eq!(
            request.authorization.as_deref(),
            Some("Bearer s3cret"),
            "`{}` went out unauthenticated",
            request.path
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_proxy_without_the_bearer_relays_the_nodes_401_rather_than_looking_healthy() {
    // The defect the flag exists for, and the proof the gate in the test above
    // is live rather than a no-op that would pass whatever the proxy sent.
    // `/v1/health` is exempt, so an unauthenticated proxy observes this node as
    // ready and places onto it fine; only the forward is refused, and that
    // refusal is the node answering, so it must reach the client with its own
    // status rather than as a proxy error naming a dead node.
    let node = StubNode::start(StubConfig::requiring_key("model-alpha", "s3cret"));
    let fabric = fabric_of(vec![node.spec("solo")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "model-alpha" }), &[]).await;

    assert_eq!(status, 401);
    assert!(
        body["error"]["message"].is_string(),
        "the refusal must carry a message an operator can act on: {body}"
    );
    // Placement still happened, so the answer is the node's, not the proxy's.
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("solo"));
    assert!(
        node.received()
            .iter()
            .all(|request| request.authorization.is_none()),
        "an unconfigured proxy must not send an Authorization header"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn no_eligible_node_answers_503_with_a_fabric_error_shape() {
    let fabric = fabric_of(Vec::new());
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, _headers) =
        post_chat(addr, &serde_json::json!({ "model": "anything" }), &[]).await;

    assert_eq!(status, 503);
    assert_eq!(body["error"]["type"], "fabric_error");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streaming_request_is_routed_rather_than_refused() {
    // Port 9 is a dead node. A 503 (a placement failure) rather than a 400
    // proves the proxy tried to route the stream instead of rejecting it.
    let fabric = fabric_of(vec![NodeSpec {
        label: "dead".to_string(),
        host: "127.0.0.1".to_string(),
        port: 9,
    }]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, _headers) = post_chat(
        addr,
        &serde_json::json!({ "model": "m", "stream": true }),
        &[],
    )
    .await;

    assert_eq!(status, 503);
    assert_eq!(body["error"]["type"], "fabric_error");
}

/// One piece of a streamed body, with how long after the request it arrived.
struct Piece {
    at: Duration,
    text: String,
}

/// Send a streaming POST and read the body as it arrives, timing each piece.
///
/// Deliberately does not reuse [`post_chat`]: that reads to EOF before looking
/// at anything, which would make a buffered response indistinguishable from a
/// streamed one — the exact thing these tests exist to tell apart.
async fn post_chat_streaming(
    addr: SocketAddr,
    body: &Value,
) -> (u16, Vec<(String, String)>, Vec<Piece>) {
    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("connect to proxy");
    let payload = body.to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let started = Instant::now();
    let mut raw = Vec::new();
    let mut scratch = [0_u8; 4096];
    let mut head_end = None;
    let mut pieces: Vec<Piece> = Vec::new();
    let mut delivered = 0_usize;

    // A read error rather than a clean close is how an aborted body reaches the
    // client, and that abort is itself the thing a truncation test reads.
    while let Ok(read) = stream.read(&mut scratch).await {
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&scratch[..read]);
        if head_end.is_none() {
            head_end = raw
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|at| at + 4);
            delivered = head_end.unwrap_or(0);
        }
        if let Some(start) = head_end {
            if raw.len() > delivered.max(start) {
                let fresh = String::from_utf8_lossy(&raw[delivered..]).to_string();
                delivered = raw.len();
                pieces.push(Piece {
                    at: started.elapsed(),
                    text: fresh,
                });
            }
        }
    }

    let head_end = head_end.expect("header terminator");
    let head = String::from_utf8_lossy(&raw[..head_end - 4]).to_string();
    let mut lines = head.lines();
    let status: u16 = lines
        .next()
        .expect("status line")
        .split_whitespace()
        .nth(1)
        .expect("status code")
        .parse()
        .expect("numeric status");
    let headers = lines
        .filter_map(|line| line.split_once(": "))
        .map(|(name, value)| (name.to_ascii_lowercase(), value.to_string()))
        .collect();
    (status, headers, pieces)
}

/// Strip HTTP chunk framing that hyper adds on the way back out to the client.
fn dechunk(raw: &str) -> String {
    let mut out = String::new();
    let mut rest = raw;
    loop {
        let Some((size_line, tail)) = rest.split_once("\r\n") else {
            return out;
        };
        let Ok(size) = usize::from_str_radix(size_line.trim(), 16) else {
            return out;
        };
        if size == 0 || tail.len() < size {
            return out;
        }
        out.push_str(&tail[..size]);
        rest = tail[size..].strip_prefix("\r\n").unwrap_or("");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_body_reaches_the_client_verbatim() {
    let events = [
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let node = StubNode::start(StubConfig::streaming("m", &events, Duration::ZERO));
    let fabric = fabric_of(vec![node.spec("only")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, headers, pieces) =
        post_chat_streaming(addr, &serde_json::json!({ "model": "m", "stream": true })).await;

    assert_eq!(status, 200);
    assert_eq!(header(&headers, "content-type"), Some("text/event-stream"));
    // Placement is still reported, exactly as on a buffered answer.
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("only"));

    let body: String = pieces.iter().map(|piece| piece.text.as_str()).collect();
    assert_eq!(dechunk(&body), events.concat());
    // The control for the truncation test below: a stream the node finished is
    // framed as finished, and that is the only thing that says so.
    assert!(
        body.ends_with("0\r\n\r\n"),
        "a completed stream must end with the terminal chunk: {body:?}"
    );
}

/// A node that dies mid-generation must not reach the client looking like a
/// stream that finished. Chunked framing is what carries that distinction: a
/// complete body ends with the terminal chunk, an aborted one does not. The
/// proxy used to read the node's EOF as the end of the body and then frame its
/// own response as complete, so a half answer arrived indistinguishable from a
/// whole one.
#[tokio::test(flavor = "multi_thread")]
async fn a_node_dying_mid_stream_is_not_relayed_as_a_completed_one() {
    // The gap matters: it makes the proxy flush the head and the early events
    // before the node dies, which is what a real generation does. Without it
    // everything would still be in one write buffer and the whole response
    // would be discarded, which is a different (also safe) outcome.
    let events = ["data: one\n\n", "data: two\n\n"];
    let node = StubNode::start(StubConfig::dying_mid_stream(
        "m",
        &events,
        Duration::from_millis(150),
    ));
    let fabric = fabric_of(vec![node.spec("only")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, _headers, pieces) =
        post_chat_streaming(addr, &serde_json::json!({ "model": "m", "stream": true })).await;

    // The head was already sent before the node died, so the status stands.
    assert_eq!(status, 200);
    let body: String = pieces.iter().map(|piece| piece.text.as_str()).collect();
    // What the node did produce still reaches the client; only the claim that
    // this was all of it is withheld.
    assert!(
        dechunk(&body).starts_with("data: one\n\n"),
        "the events the node did produce must still arrive: {body:?}"
    );
    assert!(
        !body.ends_with("0\r\n\r\n"),
        "a truncated stream was framed as complete: {body:?}"
    );
}

/// The proxy's contract is that a client can ask for a specific node whatever
/// default mode the proxy was started with. Both nodes are idle and `alpha`
/// sorts first, so throughput placement takes `alpha` — the answer coming from
/// `beta` can only mean the header was honoured.
#[tokio::test(flavor = "multi_thread")]
async fn a_client_can_pin_a_node_even_when_the_proxy_defaults_to_throughput() {
    let alpha = StubNode::start(StubConfig::ready("shared", 0));
    let beta = StubNode::start(StubConfig::ready("shared", 0));
    let fabric = fabric_of(vec![alpha.spec("alpha"), beta.spec("beta")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let body = serde_json::json!({ "model": "shared" });
    let (status, _body, headers) =
        post_chat(addr, &body, &[("x-camelid-fabric-sticky", "beta")]).await;

    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("beta"));
    assert_eq!(
        header(&headers, "x-camelid-fabric-reason"),
        Some("Affinity")
    );

    // Same proxy, same nodes, no header: the configured default still decides.
    let (status, _body, headers) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("alpha"));
}

/// The load-bearing property: the proxy must relay events as they are produced.
/// If it buffered the body and released it at the end, every piece would arrive
/// at once, after the whole generation.
#[tokio::test(flavor = "multi_thread")]
async fn events_reach_the_client_as_they_are_produced_not_at_the_end() {
    let gap = Duration::from_millis(200);
    let events = ["data: one\n\n", "data: two\n\n", "data: [DONE]\n\n"];
    let node = StubNode::start(StubConfig::streaming("m", &events, gap));
    let fabric = fabric_of(vec![node.spec("only")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, _headers, pieces) =
        post_chat_streaming(addr, &serde_json::json!({ "model": "m", "stream": true })).await;
    assert_eq!(status, 200);

    let carrying: Vec<&Piece> = pieces
        .iter()
        .filter(|piece| piece.text.contains("data:"))
        .collect();
    assert!(
        carrying.len() >= 2,
        "expected several separately delivered pieces, got {}",
        carrying.len()
    );

    // The stub finishes at ~3 * 200ms. A buffered proxy would hand everything
    // over at that point, so a first piece well before it can only mean the
    // bytes were relayed while the node was still generating.
    let first = carrying[0].at;
    assert!(
        first < gap * 2,
        "first event arrived after {first:?}; it looks buffered rather than relayed"
    );
    let last = carrying[carrying.len() - 1].at;
    assert!(
        last - first > gap / 2,
        "every piece arrived within {:?} of the first; that is a buffered body",
        last - first
    );
}

/// A node that refuses a streaming request answers with JSON, not with events.
/// That reason has to reach the client as a readable body rather than as an
/// empty stream, or an operator has nothing to act on.
#[tokio::test(flavor = "multi_thread")]
async fn a_node_refusing_a_stream_is_relayed_as_a_readable_answer() {
    let events = ["data: never sent\n\n"];
    let mut config = StubConfig::streaming("m", &events, Duration::ZERO);
    config.required_key = Some("expected-key".to_string());
    let node = StubNode::start(config);
    // The fabric holds no bearer, so the node answers 401 to the forward.
    let fabric = fabric_of(vec![node.spec("only")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, headers) = post_chat(
        addr,
        &serde_json::json!({ "model": "m", "stream": true }),
        &[],
    )
    .await;

    assert_eq!(status, 401);
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("Authorization")),
        "the node's own reason must survive: {body}"
    );
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("only"));
}

/// A client that hangs up mid-stream must take the node's work with it.
/// The buffered path cannot do this — blocking socket I/O is not cancellable —
/// so this is a property the streaming path adds rather than inherits.
#[tokio::test(flavor = "multi_thread")]
async fn a_client_hanging_up_stops_the_node_generating() {
    let gap = Duration::from_millis(100);
    let events: Vec<String> = (0..40).map(|i| format!("data: {i}\n\n")).collect();
    let borrowed: Vec<&str> = events.iter().map(String::as_str).collect();
    let config = StubConfig::streaming("m", &borrowed, gap);
    let written = Arc::clone(&config.events_written);
    let node = StubNode::start(config);
    let fabric = fabric_of(vec![node.spec("only")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("connect to proxy");
    let payload = serde_json::json!({ "model": "m", "stream": true }).to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    // Read enough to know the stream is genuinely running, then hang up.
    let mut scratch = [0_u8; 1024];
    let _ = stream.read(&mut scratch).await.expect("read head");
    tokio::time::sleep(gap * 3).await;
    drop(stream);

    // A hang-up only surfaces once a write to the client round-trips, so a few
    // more events can escape first. What must be true is that generation then
    // *stops*: settle past that window, then check the count is still moving.
    tokio::time::sleep(gap * 5).await;
    let settled = written.load(Ordering::SeqCst);
    tokio::time::sleep(gap * 10).await;
    let later = written.load(Ordering::SeqCst);

    assert_eq!(
        settled,
        later,
        "the node was still generating {} events after the client left",
        later - settled
    );
    assert!(
        later < events.len(),
        "the node wrote all {} events despite the client leaving",
        events.len()
    );
}

/// A burst of requests for one model must spread across the nodes serving it.
///
/// `/v1/health` reports what a node has already accepted, so a request still on
/// its way is invisible there. Without the fabric counting its own outstanding
/// placements, every request in a burst observes the same idle fabric and the
/// label tie-break sends all of them to whichever node sorts first — measured at
/// 10 / 2 / 0 across three nodes, with one node never used at all.
///
/// The nodes here report the load they are really under, which is the only way
/// this test means anything.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_burst_of_requests_spreads_across_the_nodes_serving_the_model() {
    const REQUESTS: usize = 12;
    let labels = ["node-a", "node-b", "node-c"];
    let nodes: Vec<StubNode> = labels
        .iter()
        .map(|_| {
            StubNode::start(StubConfig::reporting_live_load(
                "shared-model",
                Duration::from_millis(300),
            ))
        })
        .collect();
    let specs = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| node.spec(labels[i]))
        .collect();
    let addr = start_proxy(fabric_of(specs), RouteMode::Throughput).await;

    let body = serde_json::json!({ "model": "shared-model" });
    let results =
        futures_util::future::join_all((0..REQUESTS).map(|_| post_chat(addr, &body, &[]))).await;
    for (status, _, _) in &results {
        assert_eq!(*status, 200);
    }

    let served: Vec<usize> = nodes
        .iter()
        .map(|node| {
            node.received()
                .iter()
                .filter(|r| r.path == "/v1/chat/completions")
                .count()
        })
        .collect();
    let total: usize = served.iter().sum();
    assert_eq!(total, REQUESTS, "every request must reach a node");

    // Deliberately not asserting an exact split: probes race, so an occasional
    // request lands a place either way. What must hold is that the work is
    // shared rather than piled onto whichever label sorts first.
    let busiest = served.iter().max().copied().unwrap_or(0);
    assert!(
        busiest <= REQUESTS / 2,
        "one node took {busiest} of {REQUESTS}: {served:?}"
    );
    assert!(
        served.iter().all(|count| *count > 0),
        "a node serving the model was never used: {served:?}"
    );
}

/// `Fabric::dispatch` is synchronous socket I/O that can legitimately run for
/// the whole `forward_timeout` — up to minutes for a real generation. If the
/// handler ran it directly on an async worker thread instead of
/// `spawn_blocking`, concurrent requests would serialize on that thread once
/// the runtime is down to one worker: this pins the test runtime to exactly
/// one worker thread, sends four concurrent requests to four independently
/// slow nodes, and asserts they complete together rather than one-at-a-time.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn concurrent_slow_requests_do_not_serialize_on_the_single_worker_thread() {
    let delay = Duration::from_millis(500);
    let nodes: Vec<StubNode> = (0..4)
        .map(|i| StubNode::start(StubConfig::slow(&format!("model-{i}"), delay)))
        .collect();
    let specs = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| node.spec(&format!("node-{i}")))
        .collect();
    let fabric = fabric_of(specs);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let started = std::time::Instant::now();
    let requests = (0..4).map(|i| {
        let model = format!("model-{i}");
        async move { post_chat(addr, &serde_json::json!({ "model": model }), &[]).await }
    });
    let results = futures_util::future::join_all(requests).await;
    let elapsed = started.elapsed();

    for (status, _, _) in &results {
        assert_eq!(*status, 200);
    }
    // Serialized on one worker thread: ~4 * 500ms = 2000ms. Concurrent via
    // spawn_blocking: ~500ms plus scheduling overhead. 1200ms sits well clear
    // of both, so neither CI jitter nor a slow stub can decide the outcome.
    assert!(
        elapsed < Duration::from_millis(1200),
        "four concurrent slow requests took {elapsed:?}; \
         they appear to have serialized on the async runtime"
    );
}

/// Send one GET over a real socket and return (status, raw body).
async fn get_raw(addr: SocketAddr, path: &str, extra_headers: &[(&str, &str)]) -> (u16, String) {
    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("connect to proxy");
    let mut request = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n");
    for (name, value) in extra_headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str("\r\n");
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.expect("read response");
    let text = String::from_utf8_lossy(&raw).into_owned();
    let header_end = text.find("\r\n\r\n").expect("header terminator");
    let status: u16 = text[..header_end]
        .lines()
        .next()
        .expect("status line")
        .split_whitespace()
        .nth(1)
        .expect("status code")
        .parse()
        .expect("numeric status");
    (status, text[header_end + 4..].to_string())
}

fn authenticated(key: &str) -> ClientAuth {
    ClientAuth::resolve(Some(key.to_string()), None).expect("a key resolves")
}

/// The whole point of the credential is that an unauthenticated caller cannot
/// reach the fabric at all. Refusing after placement would still have spent a
/// probe on every node, and told the caller which of them are up.
#[tokio::test(flavor = "multi_thread")]
async fn an_unauthenticated_request_never_reaches_a_node() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy_with_auth(
        fabric_of(vec![node.spec("node-a")]),
        RouteMode::Throughput,
        authenticated("s3cret"),
    )
    .await;

    let (status, body, _) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;
    assert_eq!(status, 401);
    assert_eq!(body["error"]["type"], "authentication_error");
    assert!(
        node.received().is_empty(),
        "the node was touched by a request that was never authenticated: {:?}",
        node.received()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn either_header_the_engine_accepts_is_accepted_here() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy_with_auth(
        fabric_of(vec![node.spec("node-a")]),
        RouteMode::Throughput,
        authenticated("s3cret"),
    )
    .await;
    let body = serde_json::json!({ "model": "shared-model" });

    let (bearer, _, _) = post_chat(addr, &body, &[("Authorization", "Bearer s3cret")]).await;
    assert_eq!(bearer, 200, "Authorization: Bearer must be accepted");

    let (api_key, _, _) = post_chat(addr, &body, &[("X-API-Key", "s3cret")]).await;
    assert_eq!(api_key, 200, "X-API-Key must be accepted");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_wrong_key_is_refused_like_no_key_at_all() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy_with_auth(
        fabric_of(vec![node.spec("node-a")]),
        RouteMode::Throughput,
        authenticated("s3cret"),
    )
    .await;

    let (status, _, _) = post_chat(
        addr,
        &serde_json::json!({ "model": "shared-model" }),
        &[("Authorization", "Bearer wrong")],
    )
    .await;
    assert_eq!(status, 401);
    assert!(node.received().is_empty(), "{:?}", node.received());
}

/// Discovery is behind the key too: an unauthenticated caller must not learn
/// which models the fabric is serving.
#[tokio::test(flavor = "multi_thread")]
async fn discovery_does_not_answer_an_unauthenticated_caller() {
    let node = StubNode::start(StubConfig::ready("private-model", 0));
    let addr = start_proxy_with_auth(
        fabric_of(vec![node.spec("node-a")]),
        RouteMode::Throughput,
        authenticated("s3cret"),
    )
    .await;

    let (refused, body) = get_raw(addr, "/v1/models", &[]).await;
    assert_eq!(refused, 401);
    assert!(!body.contains("private-model"), "{body}");

    let (served, listing) =
        get_raw(addr, "/v1/models", &[("Authorization", "Bearer s3cret")]).await;
    assert_eq!(served, 200);
    assert!(listing.contains("private-model"), "{listing}");
}

/// A `tracing` sink a test can read back, so the access log can be asserted as
/// an operator actually reads it rather than by calling the middleware directly.
#[derive(Clone)]
struct AccessLog(Arc<Mutex<Vec<u8>>>);

impl AccessLog {
    /// The line mentioning `needle`. Every test in this binary shares one sink,
    /// so a caller has to look for something only it could have sent.
    fn line_mentioning(&self, needle: &str) -> Option<String> {
        let captured = self.0.lock().expect("access log");
        String::from_utf8_lossy(&captured)
            .lines()
            .find(|line| line.contains(needle))
            .map(str::to_string)
    }
}

impl Write for AccessLog {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("access log").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Install the capturing subscriber, once: a global default can only be set
/// once per process, and these tests share one.
fn access_log() -> AccessLog {
    static CAPTURED: OnceLock<AccessLog> = OnceLock::new();
    CAPTURED
        .get_or_init(|| {
            let sink = AccessLog(Arc::new(Mutex::new(Vec::new())));
            let writer = sink.clone();
            tracing_subscriber::fmt()
                .with_writer(move || writer.clone())
                .with_max_level(tracing::Level::INFO)
                // Both off so the assertions match on the text an operator
                // greps, not on escape codes or a timestamp.
                .with_ansi(false)
                .without_time()
                .init();
            sink
        })
        .clone()
}

/// The access log is the only place this proxy says who called and which
/// request it was. Both are supplied by the serving stack rather than by the
/// middleware, so neither is proven by calling the middleware directly.
#[tokio::test(flavor = "multi_thread")]
async fn a_request_is_logged_with_the_caller_and_the_id_it_was_given() {
    let recorded = access_log();
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy(fabric_of(vec![node.spec("node-a")]), RouteMode::Throughput).await;

    let mine = "only-this-test-sends-this-id";
    let body = serde_json::json!({ "model": "shared-model" });
    let (status, answered, headers) = post_chat(addr, &body, &[("x-request-id", mine)]).await;

    assert_eq!(status, 200, "{answered}");
    assert_eq!(
        header(&headers, "x-request-id"),
        Some(mine),
        "the id a client sent must come back to it, or it cannot quote it"
    );

    let line = recorded
        .line_mentioning(mine)
        .expect("the request must be logged under the id it was given");
    assert!(
        line.contains("127.0.0.1:"),
        "the line must name the caller, not `-`: {line}"
    );
}

/// Being asked to stop is not the same as being killed. A proxy holds other
/// people's requests, and dropping them on a deploy turns a routine restart
/// into a client-visible failure.
///
/// The load-bearing claim is an *ordering* one: the stop must not complete
/// until the work in flight has. Asserting only that the request got its answer
/// would prove nothing here, because a connection axum has already accepted is
/// served on its own task and finishes either way while this process lives — it
/// is the caller returning early, and the process exiting under it, that loses
/// the request. So this measures how long the stop took to report itself done.
#[tokio::test(flavor = "multi_thread")]
async fn a_stop_finishes_the_work_in_flight_and_accepts_no_more() {
    let generating = Duration::from_millis(600);
    let node = StubNode::start(StubConfig {
        completion_delay: generating,
        ..StubConfig::ready("shared-model", 0)
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let addr = listener.local_addr().expect("proxy addr");
    let fabric = fabric_of(vec![node.spec("node-a")]);
    let config = ServeConfig {
        mode: RouteMode::Throughput,
        forward_timeout: FORWARD_TIMEOUT,
        auth: ClientAuth::none(),
        bound: addr,
    };

    let (stop, stop_asked) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(async move {
        serve_on_until(listener, fabric, config, async move {
            let _ = stop_asked.await;
        })
        .await
    });

    let body = serde_json::json!({ "model": "shared-model" });
    let inflight = tokio::spawn(async move { post_chat(addr, &body, &[]).await });

    // Long enough for the request to be on the node, short enough that it is
    // still there when the stop arrives.
    let settle = Duration::from_millis(200);
    tokio::time::sleep(settle).await;
    let asked_at = Instant::now();
    let _ = stop.send(());

    serving
        .await
        .expect("the server joins")
        .expect("a stop is not a failure");
    let took = asked_at.elapsed();

    // The request still had most of its generation left when the stop landed.
    let still_owed = generating - settle;
    assert!(
        took >= still_owed / 2,
        "the proxy called itself stopped after {took:?}, while it still owed a \
         request about {still_owed:?} of work"
    );

    let (status, answered, _) = inflight.await.expect("the in-flight request joins");
    assert_eq!(
        status, 200,
        "a request already on a node must be finished, not dropped: {answered}"
    );
    assert!(
        tokio::net::TcpStream::connect(addr).await.is_err(),
        "a stopped proxy must not still be taking work"
    );
}

/// A proxy started without a key is unchanged: this is what every other test in
/// this file relies on, and what a loopback operator gets by default.
#[tokio::test(flavor = "multi_thread")]
async fn a_proxy_without_a_key_serves_an_anonymous_client() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy(fabric_of(vec![node.spec("node-a")]), RouteMode::Throughput).await;

    let (status, _, _) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;
    assert_eq!(status, 200);
}

/// A load balancer needs one address to ask whether to send traffic here. Until
/// this route existed the answer was a bodyless 404 whatever the fabric was
/// doing, so nothing could tell a working proxy from a useless one.
#[tokio::test(flavor = "multi_thread")]
async fn health_reports_ready_while_a_node_can_serve() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy(fabric_of(vec![node.spec("node-a")]), RouteMode::Throughput).await;

    let (status, body) = get_raw(addr, "/v1/health", &[]).await;
    assert_eq!(status, 200, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("json body");
    assert_eq!(parsed["ok"], serde_json::json!(true));
    assert_eq!(parsed["ready"], serde_json::json!(true));
    assert_eq!(parsed["nodes"]["ready"], serde_json::json!(1));
    assert_eq!(parsed["models"], serde_json::json!(["shared-model"]));
    // The test binds loopback, so the fabric may be named.
    assert_eq!(parsed["node_detail"][0]["spec"]["label"], "node-a");
}

/// A spec for a port nothing listens on. Binding and dropping proves the port
/// was free and is now closed; a hardcoded number proves neither.
async fn closed_port_spec(label: &str) -> NodeSpec {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a free port");
    let addr = listener.local_addr().expect("port");
    drop(listener);
    NodeSpec {
        label: label.to_string(),
        host: addr.ip().to_string(),
        port: addr.port(),
    }
}

/// Ready is "some request can be served", not "every node is well". A fabric
/// with one node down still takes traffic, and a proxy that reported itself
/// unhealthy here would take a whole fabric out of rotation over one node.
#[tokio::test(flavor = "multi_thread")]
async fn health_stays_ready_while_any_node_can_serve() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let specs = vec![node.spec("node-a"), closed_port_spec("node-gone").await];
    let addr = start_proxy(fabric_of(specs), RouteMode::Throughput).await;

    let (status, body) = get_raw(addr, "/v1/health", &[]).await;
    assert_eq!(status, 200, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("json body");
    assert_eq!(parsed["ready"], serde_json::json!(true));
    assert_eq!(parsed["nodes"]["ready"], serde_json::json!(1));
    assert_eq!(parsed["nodes"]["unreachable"], serde_json::json!(1));
}

/// The status code has to change, not just a field in the body: a load balancer
/// acts on the code alone. A proxy with nothing behind it must stop inviting
/// traffic it can only refuse.
#[tokio::test(flavor = "multi_thread")]
async fn health_stops_inviting_traffic_when_no_node_answers() {
    let specs = vec![closed_port_spec("node-gone").await];
    let addr = start_proxy(fabric_of(specs), RouteMode::Throughput).await;

    let (status, body) = get_raw(addr, "/v1/health", &[]).await;
    assert_eq!(status, 503, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("json body");
    assert_eq!(parsed["ready"], serde_json::json!(false));
    // Liveness is unchanged: this process is fine, its fabric is not, and
    // restarting this process would not bring a node back.
    assert_eq!(parsed["ok"], serde_json::json!(true));
    assert_eq!(parsed["nodes"]["unreachable"], serde_json::json!(1));
    assert_eq!(parsed["models"], serde_json::json!([]));
}

/// Health needs no key even on a proxy that requires one everywhere else: the
/// engine's shared `authenticate` keeps `/v1/health` public so a probe can run
/// without credentials, and this proxy reuses that check rather than writing a
/// second one. `/v1/models` stays behind the key, and the contrast is the point
/// — it is why the health body withholds the fabric's detail off-box, which the
/// unit tests cover because this test can only bind loopback.
#[tokio::test(flavor = "multi_thread")]
async fn health_answers_a_probe_that_carries_no_key() {
    let node = StubNode::start(StubConfig::ready("private-model", 0));
    let addr = start_proxy_with_auth(
        fabric_of(vec![node.spec("node-a")]),
        RouteMode::Throughput,
        authenticated("s3cret"),
    )
    .await;

    let (health, body) = get_raw(addr, "/v1/health", &[]).await;
    assert_eq!(health, 200, "a probe with no key must still get an answer");
    let parsed: Value = serde_json::from_str(&body).expect("json body");
    assert_eq!(parsed["ready"], serde_json::json!(true));

    let (models, listing) = get_raw(addr, "/v1/models", &[]).await;
    assert_eq!(models, 401, "discovery stays behind the key");
    assert!(!listing.contains("private-model"), "{listing}");

    let (chat, _, _) = post_chat(addr, &serde_json::json!({ "model": "private-model" }), &[]).await;
    assert_eq!(chat, 401, "generation stays behind the key");
}

/// A load balancer polls this route forever. If each poll re-probed the fabric,
/// adding a health check would multiply node traffic by the polling rate, which
/// is exactly the cost the freshness bound exists to remove.
#[tokio::test(flavor = "multi_thread")]
async fn repeated_health_checks_share_one_observation() {
    let nodes = vec![
        StubNode::start(StubConfig::ready("shared-model", 0)),
        StubNode::start(StubConfig::ready("shared-model", 0)),
    ];
    let specs = vec![nodes[0].spec("node-a"), nodes[1].spec("node-b")];
    let addr = start_proxy(
        fabric_reusing_observations(specs, Duration::from_secs(30)),
        RouteMode::Throughput,
    )
    .await;

    for _ in 0..5 {
        let (status, _) = get_raw(addr, "/v1/health", &[]).await;
        assert_eq!(status, 200);
    }

    assert_eq!(
        health_probes(&nodes),
        nodes.len(),
        "five health checks should share one observation, one probe per node"
    );
}

/// The proxy observes the fabric before every placement. Without a freshness
/// bound that is a `/v1/health` per node per request: measured at exactly 2 per
/// request against two nodes, and 2.0 s per request when one node black-holes,
/// against nodes answering in 2 ms.
#[tokio::test(flavor = "multi_thread")]
async fn requests_inside_the_freshness_window_share_one_observation() {
    let nodes = vec![
        StubNode::start(StubConfig::ready("shared-model", 0)),
        StubNode::start(StubConfig::ready("shared-model", 0)),
    ];
    let specs = vec![nodes[0].spec("node-a"), nodes[1].spec("node-b")];
    let addr = start_proxy(
        fabric_reusing_observations(specs, Duration::from_secs(30)),
        RouteMode::Throughput,
    )
    .await;

    let body = serde_json::json!({ "model": "shared-model" });
    for _ in 0..4 {
        let (status, _, _) = post_chat(addr, &body, &[]).await;
        assert_eq!(status, 200);
    }

    assert_eq!(
        health_probes(&nodes),
        nodes.len(),
        "four requests should share one observation, one probe per node"
    );
    assert_eq!(
        completions_served(&nodes).iter().sum::<usize>(),
        4,
        "every request must still reach a node"
    );
}

/// The bound is a bound, not a cache that never expires. Two requests either
/// side of it must each take their own observation.
#[tokio::test(flavor = "multi_thread")]
async fn an_observation_past_the_bound_is_taken_again() {
    let max_age = Duration::from_millis(50);
    let nodes = vec![StubNode::start(StubConfig::ready("shared-model", 0))];
    let specs = vec![nodes[0].spec("node-a")];
    let addr = start_proxy(
        fabric_reusing_observations(specs, max_age),
        RouteMode::Throughput,
    )
    .await;

    let body = serde_json::json!({ "model": "shared-model" });
    let (first, _, _) = post_chat(addr, &body, &[]).await;
    assert_eq!(first, 200);
    tokio::time::sleep(max_age * 4).await;
    let (second, _, _) = post_chat(addr, &body, &[]).await;
    assert_eq!(second, 200);

    assert_eq!(
        health_probes(&nodes),
        2,
        "a request after the bound must observe the fabric again"
    );
}

/// Reusing an observation must not undo the placement fix: these nodes report a
/// fixed load, so the observation is identical for every request in the burst
/// and the spreading can only come from the reservations the fabric keeps
/// itself. Without those, the label tie-break sends the whole burst to one node.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_burst_still_spreads_while_one_observation_is_reused() {
    const REQUESTS: usize = 12;
    let labels = ["node-a", "node-b", "node-c"];
    let nodes: Vec<StubNode> = labels
        .iter()
        .map(|_| StubNode::start(StubConfig::slow("shared-model", Duration::from_millis(300))))
        .collect();
    let specs = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| node.spec(labels[i]))
        .collect();
    let addr = start_proxy(
        fabric_reusing_observations(specs, Duration::from_secs(30)),
        RouteMode::Throughput,
    )
    .await;

    let body = serde_json::json!({ "model": "shared-model" });
    let results =
        futures_util::future::join_all((0..REQUESTS).map(|_| post_chat(addr, &body, &[]))).await;
    for (status, _, _) in &results {
        assert_eq!(*status, 200);
    }

    assert_eq!(
        health_probes(&nodes),
        nodes.len(),
        "the burst must have been placed from a single observation, \
         or this proves nothing about spreading without a fresh one"
    );

    let served = completions_served(&nodes);
    assert_eq!(
        served.iter().sum::<usize>(),
        REQUESTS,
        "every request must reach a node"
    );
    let busiest = served.iter().max().copied().unwrap_or(0);
    assert!(
        busiest <= REQUESTS / 2,
        "one node took {busiest} of {REQUESTS}: {served:?}"
    );
    assert!(
        served.iter().all(|count| *count > 0),
        "a node serving the model was never used: {served:?}"
    );
}

/// A node can die inside the freshness window, and an observation naming it as
/// ready is then wrong. The failed forward must drop that observation, so the
/// node is refused as unroutable rather than chosen again for the rest of the
/// window: the request after the failure is a routing refusal (503), not a
/// second failed forward (502).
#[tokio::test(flavor = "multi_thread")]
async fn a_node_that_stops_answering_drops_the_observation_that_named_it() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let specs = vec![node.spec("node-a")];
    let addr = start_proxy(
        fabric_reusing_observations(specs, Duration::from_secs(30)),
        RouteMode::Throughput,
    )
    .await;

    let body = serde_json::json!({ "model": "shared-model" });
    let (served, _, _) = post_chat(addr, &body, &[]).await;
    assert_eq!(served, 200, "the node answers while it is up");

    drop(node);

    let (after_death, _, _) = post_chat(addr, &body, &[]).await;
    assert_eq!(
        after_death, 502,
        "the observation still named the node, so this request is spent on it"
    );

    let (next, refusal, _) = post_chat(addr, &body, &[]).await;
    assert_eq!(
        next, 503,
        "the failed forward must have dropped the observation, so this request \
         observes the fabric again and finds nothing to route to"
    );
    assert_eq!(refusal["error"]["type"], "fabric_error");
}

/// Reusing an observation may delay routing to a newly loaded model. It must
/// not turn that delay into a *permanent* refusal: `/v1/chat/completions`
/// answers 404 `model_not_found` for a model no node serves, and a 404 tells an
/// OpenAI SDK to stop retrying. Issued from a reused observation, that verdict
/// is about the fabric as it was, so the fabric has to look again before it
/// refuses rather than settle the question from memory.
#[tokio::test(flavor = "multi_thread")]
async fn a_model_loaded_since_the_observation_is_not_refused_as_permanently_absent() {
    let config = StubConfig::with_switchable_model("old-model");
    let model = Arc::clone(config.live_model.as_ref().expect("a switchable node"));
    let node = StubNode::start(config);
    // Far longer than the test runs, so nothing here depends on it expiring.
    let addr = start_proxy(
        fabric_reusing_observations(vec![node.spec("only")], Duration::from_secs(30)),
        RouteMode::Throughput,
    )
    .await;

    let (served, _, _) = post_chat(addr, &serde_json::json!({ "model": "old-model" }), &[]).await;
    assert_eq!(served, 200, "the first request takes the observation");

    // The operator loads something else, exactly as `/api/models/load` does.
    *model.lock().expect("stub model lock") = "new-model".to_string();

    let (status, body, _) =
        post_chat(addr, &serde_json::json!({ "model": "new-model" }), &[]).await;
    assert_ne!(
        status, 404,
        "a permanent refusal was settled from a stale observation: {body}"
    );
    assert_eq!(status, 200, "{body}");
}

// ---------------------------------------------------------------------------
// Failing over to another node
//
// A node can go between being observed and being sent to. Placement then names
// a machine that is not there, and without failover the request dies against it
// even when a sibling is serving the same model. These tests are built on a
// reused observation because that is the arrangement `fabric serve` runs in,
// and the window where the fabric's view can be wrong by construction.
// ---------------------------------------------------------------------------

/// A fabric as `fabric serve` builds one, with an explicit attempt budget so a
/// test can turn failover off without changing anything else.
fn fabric_with_attempts(specs: Vec<NodeSpec>, attempts: usize) -> Fabric {
    fabric_reusing_observations(specs, Duration::from_secs(30)).with_max_forward_attempts(attempts)
}

/// Two nodes, both serving the model, and a proxy holding one observation of
/// them. `node-a` wins the label tie-break, so it is where the next request
/// goes; stopping it is what makes that observation wrong.
async fn two_nodes_one_of_which_will_vanish(
    attempts: usize,
) -> (StubNode, StubNode, SocketAddr, Value) {
    let a = StubNode::start(StubConfig::ready("shared-model", 0));
    let b = StubNode::start(StubConfig::ready("shared-model", 0));
    let specs = vec![a.spec("node-a"), b.spec("node-b")];
    let addr = start_proxy(fabric_with_attempts(specs, attempts), RouteMode::Throughput).await;
    let body = serde_json::json!({ "model": "shared-model" });

    let (status, _, headers) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 200);
    assert_eq!(
        header(&headers, "x-camelid-fabric-node"),
        Some("node-a"),
        "the tie-break must settle on node-a, or stopping it proves nothing"
    );
    (a, b, addr, body)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_vanished_node_does_not_fail_a_request_another_node_can_serve() {
    let (mut a, _b, addr, body) = two_nodes_one_of_which_will_vanish(2).await;
    let before = completions_served(std::slice::from_ref(&a))[0];
    a.stop_listening();

    let (status, answer, headers) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 200, "a sibling serves the same model: {answer}");
    assert_eq!(
        header(&headers, "x-camelid-fabric-node"),
        Some("node-b"),
        "the answer must come from the node that is still there"
    );
    assert_eq!(
        header(&headers, "x-camelid-fabric-attempts"),
        Some("2"),
        "a failover must be visible to the client that was served by one"
    );
    assert_eq!(
        completions_served(std::slice::from_ref(&a))[0],
        before,
        "the vanished node must not have been sent the request twice"
    );
}

/// The ablation, run as a test: the same fabric with the budget at one is the
/// behaviour this change replaces, and it fails the request.
#[tokio::test(flavor = "multi_thread")]
async fn a_budget_of_one_fails_the_request_the_way_it_did_before() {
    let (mut a, b, addr, body) = two_nodes_one_of_which_will_vanish(1).await;
    a.stop_listening();

    let (status, refusal, _) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 502, "{refusal}");
    assert_eq!(
        completions_served(std::slice::from_ref(&b))[0],
        0,
        "with failover off the sibling must never be asked"
    );
}

/// The line the whole design rests on. This node reads the request and then
/// hangs up, so it may already be generating; that is a different failure from
/// never having been reached, and it must not be sent anywhere else.
#[tokio::test(flavor = "multi_thread")]
async fn a_node_that_took_the_request_is_not_asked_to_run_it_again_elsewhere() {
    let a = StubNode::start(StubConfig::hanging_up_on_completion("shared-model"));
    let b = StubNode::start(StubConfig::ready("shared-model", 0));
    let specs = vec![a.spec("node-a"), b.spec("node-b")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;

    let (status, refusal, headers) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;

    assert_eq!(status, 502, "{refusal}");
    // The refusal names the node it happened against, so an operator can act on
    // the one that is actually broken.
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("node-a"));
    assert_eq!(
        completions_served(std::slice::from_ref(&a))[0],
        1,
        "the node did receive the request, which is why it may not be re-sent"
    );
    assert_eq!(
        completions_served(std::slice::from_ref(&b))[0],
        0,
        "a request that may already be running was sent to a second node"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn failover_stops_at_the_attempt_budget_rather_than_walking_the_fabric() {
    let labels = ["node-a", "node-b", "node-c"];
    let mut nodes: Vec<StubNode> = labels
        .iter()
        .map(|_| StubNode::start(StubConfig::ready("shared-model", 0)))
        .collect();
    let specs: Vec<NodeSpec> = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| node.spec(labels[index]))
        .collect();

    // Both proxies take their observation while all three nodes are up, so the
    // only difference between them is the budget.
    let bounded = start_proxy(
        fabric_with_attempts(specs.clone(), 2),
        RouteMode::Throughput,
    )
    .await;
    let generous = start_proxy(fabric_with_attempts(specs, 3), RouteMode::Throughput).await;
    let body = serde_json::json!({ "model": "shared-model" });
    for proxy in [bounded, generous] {
        let (status, _, headers) = post_chat(proxy, &body, &[]).await;
        assert_eq!(status, 200);
        assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("node-a"));
    }

    // Two of the three go, leaving a node that only a third attempt reaches.
    nodes[0].stop_listening();
    nodes[1].stop_listening();
    let served_by_c = completions_served(&nodes[2..])[0];

    let (status, refusal, _) = post_chat(bounded, &body, &[]).await;
    assert_eq!(
        status, 502,
        "two attempts must not silently become three: {refusal}"
    );
    assert_eq!(
        completions_served(&nodes[2..])[0],
        served_by_c,
        "the third node is beyond the budget and must not have been asked"
    );

    // The positive control: the same situation with room for a third attempt
    // does reach it, so the refusal above is the budget and not the exclusion.
    let (status, answer, headers) = post_chat(generous, &body, &[]).await;
    assert_eq!(status, 200, "{answer}");
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("node-c"));
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("3"));
}

/// A failover proves the observation is wrong, so it must not survive the
/// request that discovered it — even though that request succeeded. Otherwise
/// every request in the rest of the window pays the same dial to the same
/// absent node before being served.
#[tokio::test(flavor = "multi_thread")]
async fn a_served_failover_still_drops_the_observation_that_named_the_vanished_node() {
    let (mut a, b, addr, body) = two_nodes_one_of_which_will_vanish(2).await;
    a.stop_listening();

    let (status, _, headers) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("2"));

    let probes_before = completions_served(std::slice::from_ref(&b))[0];
    let (status, _, headers) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 200);
    assert_eq!(
        header(&headers, "x-camelid-fabric-attempts"),
        Some("1"),
        "the next request must be placed from a fresh observation, which no \
         longer names the node that is gone"
    );
    assert_eq!(
        header(&headers, "x-camelid-fabric-node"),
        Some("node-b"),
        "and it must go straight to the node that is there"
    );
    assert_eq!(
        completions_served(std::slice::from_ref(&b))[0],
        probes_before + 1
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_vanished_node_does_not_fail_a_streaming_request() {
    let events = ["data: one\n\n", "data: [DONE]\n\n"];
    let mut a = StubNode::start(StubConfig::streaming(
        "shared-model",
        &events,
        Duration::ZERO,
    ));
    let b = StubNode::start(StubConfig::streaming(
        "shared-model",
        &events,
        Duration::ZERO,
    ));
    let specs = vec![a.spec("node-a"), b.spec("node-b")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;
    let body = serde_json::json!({ "model": "shared-model", "stream": true });

    let (status, headers, _) = post_chat_streaming(addr, &body).await;
    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("node-a"));

    a.stop_listening();

    let (status, headers, pieces) = post_chat_streaming(addr, &body).await;
    assert_eq!(status, 200, "the sibling can still stream");
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("node-b"));
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("2"));
    let framed: String = pieces.iter().map(|piece| piece.text.as_str()).collect();
    assert_eq!(
        dechunk(&framed),
        events.concat(),
        "the failover must relay the sibling's stream verbatim"
    );
}

/// The streaming half of the safety rule. Once the head is relayed the client
/// has been committed to one node's answer, so a node that dies part-way
/// through must end that stream rather than start a second one somewhere else.
#[tokio::test(flavor = "multi_thread")]
async fn a_node_dying_mid_stream_is_not_replayed_on_another_node() {
    // Two events with a gap, exactly as `a_node_dying_mid_stream_is_not_relayed
    // _as_a_completed_one` uses: it is what makes the proxy flush the head and
    // the first event before the node dies, so the client really is committed
    // to node-a's answer by the time it fails.
    let gap = Duration::from_millis(150);
    let a = StubNode::start(StubConfig::dying_mid_stream(
        "shared-model",
        &["data: one\n\n", "data: two\n\n"],
        gap,
    ));
    let b = StubNode::start(StubConfig::streaming(
        "shared-model",
        &["data: elsewhere\n\n"],
        gap,
    ));
    let specs = vec![a.spec("node-a"), b.spec("node-b")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;

    let (status, headers, pieces) = post_chat_streaming(
        addr,
        &serde_json::json!({ "model": "shared-model", "stream": true }),
    )
    .await;

    assert_eq!(status, 200, "the head arrived before the node died");
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("node-a"));
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("1"));
    let framed: String = pieces.iter().map(|piece| piece.text.as_str()).collect();
    assert!(
        dechunk(&framed).starts_with("data: one\n\n"),
        "what the node did produce must still arrive: {framed:?}"
    );
    assert!(
        !framed.ends_with("0\r\n\r\n"),
        "a truncated stream must not be framed as complete: {framed:?}"
    );
    assert_eq!(
        completions_served(std::slice::from_ref(&b))[0],
        0,
        "the request was replayed on a second node after the client had already \
         been given part of the first node's answer"
    );
}

/// Failing over past a gone node is a recovery, not the reason a request ends.
/// When the node it moves on to fails it for real, that second failure is what
/// the operator has to act on: reporting the first sends them to a node the
/// fabric already routed around, and says nothing about the one that is
/// actually broken.
#[tokio::test(flavor = "multi_thread")]
async fn the_failure_that_ended_the_request_is_the_one_reported() {
    let a = StubNode::start(StubConfig::ready("shared-model", 0));
    // `node-b` is reachable and answers, but with a body that is not JSON, so
    // it fails the request in a way failover must not move past.
    let b = StubNode::start(StubConfig {
        completion: "<html>gateway timeout</html>".to_string(),
        ..StubConfig::ready("shared-model", 0)
    });
    let specs = vec![a.spec("node-a"), b.spec("node-b")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;
    let body = serde_json::json!({ "model": "shared-model" });

    let (status, _, headers) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("node-a"));

    let mut a = a;
    a.stop_listening();

    let (status, refusal, _) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 502);
    let message = refusal["error"]["message"]
        .as_str()
        .expect("a message an operator can act on");
    assert!(
        message.contains("node-b"),
        "the node that actually failed the request must be named: {message}"
    );
    assert!(
        !message.contains("node-a"),
        "node-a was routed around successfully; naming it points at the wrong \
         node: {message}"
    );
}

/// A node at its queue bound rejected the request rather than running it, so
/// another node can still take it. Until now the client got that 503 while a
/// sibling sat idle — one busy node undoing the reason the fabric exists.
#[tokio::test(flavor = "multi_thread")]
async fn a_saturated_node_hands_the_request_to_one_that_is_not() {
    let nodes = vec![
        StubNode::start(StubConfig::queue_full("shared-model")),
        StubNode::start(StubConfig::ready("shared-model", 0)),
    ];
    // `a-full` sorts first, so deterministic placement reaches it first.
    let specs = vec![nodes[0].spec("a-full"), nodes[1].spec("b-ready")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;

    assert_eq!(status, 200, "{body}");
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("b-ready"));
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("2"));
    assert_eq!(
        completions_served(&nodes),
        vec![1, 1],
        "the full node was asked once, then the ready one"
    );
}

/// Every node full is a real answer, not a proxy failure. The client gets a
/// node's own refusal, with the code and message it sent, rather than
/// something this proxy invented on its behalf.
#[tokio::test(flavor = "multi_thread")]
async fn a_fabric_that_is_entirely_saturated_returns_a_node_s_own_refusal() {
    let nodes = vec![
        StubNode::start(StubConfig::queue_full("shared-model")),
        StubNode::start(StubConfig::queue_full("shared-model")),
    ];
    let specs = vec![nodes[0].spec("a-full"), nodes[1].spec("b-full")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;

    assert_eq!(status, 503);
    assert_eq!(body["error"]["code"], ENGINE_QUEUE_FULL_CODE);
    assert_ne!(
        body["error"]["type"], "fabric_error",
        "a node answered, so the node's answer is what stands: {body}"
    );
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("2"));
    assert_eq!(completions_served(&nodes), vec![1, 1]);
}

/// The count of nodes asked has to be the count of nodes asked. With one node
/// and a budget of two, placement runs out before a second node is reached, so
/// reporting the budget sends an operator looking for a node that was never
/// contacted.
#[tokio::test(flavor = "multi_thread")]
async fn a_lone_saturated_node_is_one_attempt_not_two() {
    let nodes = vec![StubNode::start(StubConfig::queue_full("shared-model"))];
    let specs = vec![nodes[0].spec("a-full")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;

    assert_eq!(status, 503, "{body}");
    assert_eq!(body["error"]["code"], ENGINE_QUEUE_FULL_CODE);
    assert_eq!(completions_served(&nodes), vec![1]);
    assert_eq!(
        header(&headers, "x-camelid-fabric-attempts"),
        Some("1"),
        "one node was asked, so the answer must not claim two"
    );
}

/// The switch that turns off failover turns this off too: one attempt means
/// the first node's answer is the answer.
#[tokio::test(flavor = "multi_thread")]
async fn one_attempt_returns_the_refusal_without_asking_another_node() {
    let nodes = vec![
        StubNode::start(StubConfig::queue_full("shared-model")),
        StubNode::start(StubConfig::ready("shared-model", 0)),
    ];
    let specs = vec![nodes[0].spec("a-full"), nodes[1].spec("b-ready")];
    let addr = start_proxy(fabric_with_attempts(specs, 1), RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;

    assert_eq!(status, 503, "{body}");
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("1"));
    assert_eq!(
        completions_served(&nodes),
        vec![1, 0],
        "the ready node must not be asked when failover is off"
    );
}

/// Only backpressure is re-placed. A node refusing for any other reason has
/// answered the request; asking a second node would spend its time reaching
/// the same refusal, and turn one node's 503 into two.
#[tokio::test(flavor = "multi_thread")]
async fn a_refusal_that_is_not_backpressure_is_relayed_untouched() {
    let nodes = vec![
        StubNode::start(StubConfig::refusing_with(
            "shared-model",
            "model_unavailable",
            "no model is loaded",
        )),
        StubNode::start(StubConfig::ready("shared-model", 0)),
    ];
    let specs = vec![nodes[0].spec("a-refusing"), nodes[1].spec("b-ready")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;

    assert_eq!(status, 503);
    assert_eq!(body["error"]["message"], "no model is loaded");
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("1"));
    assert_eq!(
        completions_served(&nodes),
        vec![1, 0],
        "a refusal that is not backpressure must not be sent on"
    );
}

/// A streaming request gets the same treatment, because the refusal arrives
/// buffered — before one event has been relayed, so nothing has been said.
#[tokio::test(flavor = "multi_thread")]
async fn a_saturated_node_hands_a_streaming_request_on_as_well() {
    let events = [
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}",
        "data: [DONE]",
    ];
    let nodes = [
        StubNode::start(StubConfig::queue_full("shared-model")),
        StubNode::start(StubConfig::streaming(
            "shared-model",
            &events,
            Duration::ZERO,
        )),
    ];
    let specs = vec![nodes[0].spec("a-full"), nodes[1].spec("b-ready")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;

    let (status, headers, pieces) = post_chat_streaming(
        addr,
        &serde_json::json!({ "model": "shared-model", "stream": true }),
    )
    .await;

    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("b-ready"));
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("2"));
    assert!(
        !pieces.is_empty(),
        "the replacement node's events must arrive"
    );
}
