//! End-to-end tests for the resident fabric proxy (`camelid fabric serve`).
//!
//! [`fabric_end_to_end.rs`] proves `Fabric::dispatch` itself; these tests prove
//! the HTTP front door around it — a real client, over a real socket, talking
//! to the real router bound by [`camelid::fabric::server::serve_on`], routed to
//! stub nodes on loopback.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use camelid::fabric::server::{serve_on, ServeConfig};
use camelid::fabric::{Fabric, NodeSpec, RouteMode};

const PROBE_TIMEOUT: Duration = Duration::from_secs(3);
const FORWARD_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone)]
struct StubConfig {
    health: String,
    completion: String,
    completion_status: u16,
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
        }
    }

    fn refusing(model: &str) -> Self {
        Self {
            completion: r#"{"error":{"message":"engine queue full"}}"#.to_string(),
            completion_status: 503,
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
    thread: Option<JoinHandle<()>>,
}

impl StubNode {
    fn start(config: StubConfig) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread_shutdown = Arc::clone(&shutdown);
        let thread = std::thread::spawn(move || {
            for stream in listener.incoming() {
                if thread_shutdown.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(mut stream) = stream else { continue };
                serve_once(&mut stream, &config);
            }
        });
        Self {
            port,
            shutdown,
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

fn serve_once(stream: &mut TcpStream, config: &StubConfig) {
    let Some(path) = read_request_path(stream) else {
        return;
    };
    let (status, body) = match path.as_str() {
        "/v1/health" => (200_u16, config.health.clone()),
        "/v1/chat/completions" => (config.completion_status, config.completion.clone()),
        _ => (404, "{}".to_string()),
    };
    let response = format!(
        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

fn read_request_path(stream: &mut TcpStream) -> Option<String> {
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
                return Some(parts.next()?.to_string());
            }
        }
    }
    None
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
async fn no_eligible_node_answers_503_with_a_fabric_error_shape() {
    let fabric = fabric_of(Vec::new());
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, _headers) =
        post_chat(addr, &serde_json::json!({ "model": "anything" }), &[]).await;

    assert_eq!(status, 503);
    assert_eq!(body["error"]["type"], "fabric_error");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streaming_request_is_refused_by_the_proxy_before_any_node_is_touched() {
    // Port 9 is a dead node; a 400 (not a 502/503) proves the proxy refused
    // before it ever tried to route the request.
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

    assert_eq!(status, 400);
    assert_eq!(body["error"]["type"], "fabric_error");
}
