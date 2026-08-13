//! End-to-end tests for the resident fabric proxy (`camelid fabric serve`).
//!
//! [`fabric_end_to_end.rs`] proves `Fabric::dispatch` itself; these tests prove
//! the HTTP front door around it — a real client, over a real socket, talking
//! to the real router bound by [`camelid::fabric::server::serve_on`], routed to
//! stub nodes on loopback.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use camelid::fabric::server::{serve_on, ServeConfig};
use camelid::fabric::{Fabric, NodeSpec, RouteMode};

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
            stream_events: Vec::new(),
            stream_gap: Duration::ZERO,
            stream_truncated: false,
            events_written: Arc::new(AtomicUsize::new(0)),
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
}

impl Drop for StubNode {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(("127.0.0.1", self.port));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
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

    let (status, body) = match received.path.as_str() {
        _ if !authorized => (
            401_u16,
            r#"{"error":{"message":"provide Authorization: Bearer <key> or X-API-Key"}}"#
                .to_string(),
        ),
        "/v1/health" => (200_u16, config.health.clone()),
        "/v1/chat/completions" => {
            if !config.completion_delay.is_zero() {
                std::thread::sleep(config.completion_delay);
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

/// Bind the real proxy on an OS-assigned port and start serving it in the
/// background, returning the address a client should connect to.
async fn start_proxy(fabric: Fabric, mode: RouteMode) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let addr = listener.local_addr().expect("proxy addr");
    let config = ServeConfig {
        mode,
        forward_timeout: FORWARD_TIMEOUT,
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
