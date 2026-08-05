//! Probing a node's `/v1/health` over HTTP/1.1.
//!
//! This is the only file in the fabric that touches a socket. The response
//! parsing is split out as pure functions over byte slices so the awkward parts
//! — chunked bodies, truncated frames, a node that answers 500 — are tested
//! without a server.
//!
//! It does not reuse `chat::client`: that module is private to `chat`, is keyed
//! on a resolved `SocketAddr` (fabric members are named hosts), and carries SSE,
//! bearer auth and tool-call handling that a health probe must not depend on.
//! The overlap is one request/response round trip, and duplicating that is
//! cheaper than widening a shipped module's boundary.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::node::{NodeReady, NodeSnapshot, NodeSpec, NodeStatus};

/// Refuse a body larger than this. A health response is a few KiB; anything at
/// this size means we are not talking to a Camelid engine.
const MAX_BODY_BYTES: usize = 1024 * 1024;

/// Default probe budget. Wi-Fi RTT on this fabric was measured at 3-13 ms, so
/// two seconds is generous for a healthy node and still fails a dead one fast.
pub const DEFAULT_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeError {
    Resolve(String),
    Connect(String),
    Io(String),
    Malformed(String),
    TooLarge,
    Status(u16),
    Json(String),
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resolve(detail) => write!(f, "cannot resolve host: {detail}"),
            Self::Connect(detail) => write!(f, "cannot connect: {detail}"),
            Self::Io(detail) => write!(f, "connection failed: {detail}"),
            Self::Malformed(detail) => write!(f, "malformed HTTP response: {detail}"),
            Self::TooLarge => write!(f, "response exceeded {MAX_BODY_BYTES} bytes"),
            Self::Status(code) => write!(f, "health endpoint answered HTTP {code}"),
            Self::Json(detail) => write!(f, "health payload was not readable: {detail}"),
        }
    }
}

/// The subset of `/v1/health` the fabric routes on.
///
/// Every field but `ok` defaults, so a node running a newer or older engine that
/// renames an unrelated field still probes successfully instead of dropping out
/// of the fabric.
#[derive(Debug, Clone, Deserialize)]
struct HealthPayload {
    #[serde(default)]
    ok: bool,
    #[serde(default)]
    generation_ready: bool,
    #[serde(default)]
    active_model_id: Option<String>,
    #[serde(default)]
    backend: String,
    #[serde(default)]
    version: String,
    #[serde(default)]
    engine_queued_tasks: usize,
    #[serde(default)]
    engine_queue_depth: usize,
}

/// Status line plus body of an HTTP/1.1 response.
#[derive(Debug, PartialEq, Eq)]
struct HttpResponse {
    status: u16,
    body: Vec<u8>,
}

/// Parse a whole HTTP/1.1 response. Pure; see the tests at the bottom.
fn parse_http_response(raw: &[u8]) -> Result<HttpResponse, ProbeError> {
    let split = find_header_end(raw)
        .ok_or_else(|| ProbeError::Malformed("no header terminator".to_string()))?;
    let head = std::str::from_utf8(&raw[..split.headers_end])
        .map_err(|_| ProbeError::Malformed("headers are not UTF-8".to_string()))?;

    let mut lines = head.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| ProbeError::Malformed("empty response".to_string()))?;
    let status = parse_status_line(status_line)?;

    let mut chunked = false;
    let mut content_length: Option<usize> = None;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        match name.as_str() {
            "transfer-encoding" => {
                chunked = value.to_ascii_lowercase().contains("chunked");
            }
            "content-length" => {
                content_length = value.parse::<usize>().ok();
            }
            _ => {}
        }
    }

    let rest = &raw[split.body_start..];
    let body = if chunked {
        dechunk(rest)?
    } else if let Some(len) = content_length {
        if len > MAX_BODY_BYTES {
            return Err(ProbeError::TooLarge);
        }
        if rest.len() < len {
            return Err(ProbeError::Malformed(format!(
                "body truncated: expected {len} bytes, got {}",
                rest.len()
            )));
        }
        rest[..len].to_vec()
    } else {
        rest.to_vec()
    };

    if body.len() > MAX_BODY_BYTES {
        return Err(ProbeError::TooLarge);
    }
    Ok(HttpResponse { status, body })
}

struct HeaderSplit {
    headers_end: usize,
    body_start: usize,
}

fn find_header_end(raw: &[u8]) -> Option<HeaderSplit> {
    raw.windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|at| HeaderSplit {
            headers_end: at,
            body_start: at + 4,
        })
}

fn parse_status_line(line: &str) -> Result<u16, ProbeError> {
    let mut parts = line.split(' ');
    let version = parts
        .next()
        .ok_or_else(|| ProbeError::Malformed("no status line".to_string()))?;
    if !version.starts_with("HTTP/") {
        return Err(ProbeError::Malformed(format!(
            "status line does not start with HTTP/: {line}"
        )));
    }
    let code = parts
        .next()
        .ok_or_else(|| ProbeError::Malformed("status line has no code".to_string()))?;
    code.parse::<u16>()
        .map_err(|_| ProbeError::Malformed(format!("status code `{code}` is not a number")))
}

/// Decode `Transfer-Encoding: chunked`.
fn dechunk(mut rest: &[u8]) -> Result<Vec<u8>, ProbeError> {
    let mut out = Vec::new();
    loop {
        let line_end = rest
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or_else(|| ProbeError::Malformed("chunk size line unterminated".to_string()))?;
        let size_text = std::str::from_utf8(&rest[..line_end])
            .map_err(|_| ProbeError::Malformed("chunk size is not UTF-8".to_string()))?;
        // A chunk extension (`1a;name=value`) is legal and ignorable.
        let size_text = size_text.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_text, 16).map_err(|_| {
            ProbeError::Malformed(format!("chunk size `{size_text}` is not hexadecimal"))
        })?;
        rest = &rest[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        if out.len() + size > MAX_BODY_BYTES {
            return Err(ProbeError::TooLarge);
        }
        if rest.len() < size + 2 {
            return Err(ProbeError::Malformed("chunk body truncated".to_string()));
        }
        out.extend_from_slice(&rest[..size]);
        rest = &rest[size + 2..];
    }
}

/// Turn a successful health payload into a routing status. Pure.
fn classify(payload: &HealthPayload) -> NodeStatus {
    if !payload.ok {
        return NodeStatus::NotReady {
            reason: "engine reported not ok".to_string(),
        };
    }
    if !payload.generation_ready {
        let reason = match &payload.active_model_id {
            Some(model) => format!("model `{model}` loaded but not ready to generate"),
            None => "no model loaded".to_string(),
        };
        return NodeStatus::NotReady { reason };
    }
    NodeStatus::Ready(NodeReady {
        active_model_id: payload.active_model_id.clone(),
        backend: payload.backend.clone(),
        version: payload.version.clone(),
        // `engine_queue_depth` is a gauge of jobs in flight, not a bound; see
        // the note on `NodeReady`.
        in_flight: payload.engine_queue_depth,
        waiting: payload.engine_queued_tasks,
    })
}

fn read_health(spec: &NodeSpec, timeout: Duration) -> Result<HealthPayload, ProbeError> {
    let deadline = Instant::now() + timeout;
    let authority = spec.authority();

    let addr = authority
        .to_socket_addrs()
        .map_err(|error| ProbeError::Resolve(error.to_string()))?
        .next()
        .ok_or_else(|| ProbeError::Resolve("host resolved to no addresses".to_string()))?;

    let mut stream = TcpStream::connect_timeout(&addr, timeout)
        .map_err(|error| ProbeError::Connect(error.to_string()))?;
    // Short socket reads keep the loop responsive to the overall deadline; a
    // single long read timeout would overshoot it on a stalled peer.
    stream
        .set_read_timeout(Some(Duration::from_millis(100)))
        .map_err(|error| ProbeError::Io(error.to_string()))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|error| ProbeError::Io(error.to_string()))?;

    let request = format!(
        "GET /v1/health HTTP/1.1\r\nHost: {authority}\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|error| ProbeError::Io(error.to_string()))?;

    let mut raw = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        if Instant::now() >= deadline {
            return Err(ProbeError::Io("probe exceeded its deadline".to_string()));
        }
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => {
                raw.extend_from_slice(&chunk[..read]);
                if raw.len() > MAX_BODY_BYTES {
                    return Err(ProbeError::TooLarge);
                }
            }
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                continue
            }
            Err(error) => return Err(ProbeError::Io(error.to_string())),
        }
    }

    let response = parse_http_response(&raw)?;
    if response.status != 200 {
        return Err(ProbeError::Status(response.status));
    }
    serde_json::from_slice::<HealthPayload>(&response.body)
        .map_err(|error| ProbeError::Json(error.to_string()))
}

/// Probe one node. Never fails: an unreachable node is a routing fact, not an
/// error the caller has to handle separately.
pub fn probe_node(spec: &NodeSpec, timeout: Duration) -> NodeSnapshot {
    let started = Instant::now();
    match read_health(spec, timeout) {
        Ok(payload) => NodeSnapshot {
            spec: spec.clone(),
            status: classify(&payload),
            latency: Some(started.elapsed()),
        },
        Err(error) => NodeSnapshot {
            spec: spec.clone(),
            status: NodeStatus::Unreachable {
                reason: error.to_string(),
            },
            latency: None,
        },
    }
}

/// Probe every node concurrently, one thread each.
///
/// Sequential probing would make a fabric's status cost the sum of its timeouts,
/// so a single dead node would stall the view of every live one.
pub fn probe_fabric(specs: &[NodeSpec], timeout: Duration) -> Vec<NodeSnapshot> {
    if specs.is_empty() {
        return Vec::new();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = specs
            .iter()
            .map(|spec| scope.spawn(move || probe_node(spec, timeout)))
            .collect();
        handles
            .into_iter()
            .zip(specs)
            .map(|(handle, spec)| {
                handle.join().unwrap_or_else(|_| NodeSnapshot {
                    spec: spec.clone(),
                    status: NodeStatus::Unreachable {
                        reason: "probe thread panicked".to_string(),
                    },
                    latency: None,
                })
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(ok: bool, ready: bool, model: Option<&str>) -> HealthPayload {
        HealthPayload {
            ok,
            generation_ready: ready,
            active_model_id: model.map(str::to_string),
            backend: "llama".to_string(),
            version: "0.5.4".to_string(),
            engine_queued_tasks: 1,
            engine_queue_depth: 4,
        }
    }

    #[test]
    fn a_content_length_body_is_read_exactly() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello-trailing-garbage";
        let response = parse_http_response(raw).expect("parses");
        assert_eq!(response.status, 200);
        assert_eq!(response.body, b"hello");
    }

    #[test]
    fn a_chunked_body_is_reassembled() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let response = parse_http_response(raw).expect("parses");
        assert_eq!(response.body, b"hello world");
    }

    #[test]
    fn chunk_extensions_are_ignored() {
        let raw =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5;name=value\r\nhello\r\n0\r\n\r\n";
        assert_eq!(parse_http_response(raw).expect("parses").body, b"hello");
    }

    #[test]
    fn a_body_without_framing_headers_is_read_to_end() {
        let raw = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{}";
        assert_eq!(parse_http_response(raw).expect("parses").body, b"{}");
    }

    #[test]
    fn header_names_are_matched_case_insensitively() {
        let raw = b"HTTP/1.1 200 OK\r\ncOnTeNt-LeNgTh: 2\r\n\r\nok";
        assert_eq!(parse_http_response(raw).expect("parses").body, b"ok");
    }

    #[test]
    fn a_non_http_greeting_is_refused_rather_than_guessed_at() {
        let raw = b"SSH-2.0-OpenSSH_9.0\r\n\r\n";
        assert!(matches!(
            parse_http_response(raw),
            Err(ProbeError::Malformed(_))
        ));
    }

    #[test]
    fn a_response_with_no_header_terminator_is_refused() {
        assert!(matches!(
            parse_http_response(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"),
            Err(ProbeError::Malformed(_))
        ));
    }

    #[test]
    fn a_truncated_body_is_refused_rather_than_silently_short() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 99\r\n\r\nshort";
        assert!(matches!(
            parse_http_response(raw),
            Err(ProbeError::Malformed(_))
        ));
    }

    #[test]
    fn a_truncated_chunk_is_refused() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n9\r\nshort\r\n";
        assert!(matches!(
            parse_http_response(raw),
            Err(ProbeError::Malformed(_))
        ));
    }

    #[test]
    fn an_oversized_content_length_is_refused_before_allocating() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 99999999\r\n\r\n";
        assert_eq!(parse_http_response(raw), Err(ProbeError::TooLarge));
    }

    #[test]
    fn a_non_200_is_reported_with_its_code() {
        let raw = b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n";
        assert_eq!(parse_http_response(raw).expect("parses").status, 503);
    }

    #[test]
    fn a_ready_engine_becomes_a_routable_node() {
        let status = classify(&payload(true, true, Some("llama-3b")));
        let ready = status.ready().expect("ready");
        assert_eq!(ready.active_model_id.as_deref(), Some("llama-3b"));
        assert_eq!(ready.in_flight, 4);
        assert_eq!(ready.waiting, 1);
    }

    #[test]
    fn an_idle_engine_is_routable_rather_than_read_as_full() {
        // Regression: an idle node reports 0/0. Reading `engine_queue_depth` as a
        // capacity bound made a healthy idle fabric look saturated.
        let idle = HealthPayload {
            ok: true,
            generation_ready: true,
            active_model_id: Some("llama-3b".to_string()),
            backend: "llama".to_string(),
            version: "0.5.4".to_string(),
            engine_queued_tasks: 0,
            engine_queue_depth: 0,
        };
        let status = classify(&idle);
        assert!(status.is_ready());
        assert_eq!(status.ready().expect("ready").in_flight, 0);
    }

    #[test]
    fn a_loaded_but_unready_engine_names_the_model_it_is_warming() {
        let status = classify(&payload(true, false, Some("llama-3b")));
        match status {
            NodeStatus::NotReady { reason } => assert!(reason.contains("llama-3b"), "{reason}"),
            other => panic!("expected NotReady, got {other:?}"),
        }
    }

    #[test]
    fn an_engine_with_no_model_is_not_ready_and_says_so() {
        match classify(&payload(true, false, None)) {
            NodeStatus::NotReady { reason } => assert_eq!(reason, "no model loaded"),
            other => panic!("expected NotReady, got {other:?}"),
        }
    }

    #[test]
    fn an_engine_reporting_not_ok_is_never_routable() {
        assert!(!classify(&payload(false, true, Some("llama-3b"))).is_ready());
    }

    #[test]
    fn a_health_payload_missing_optional_fields_still_parses() {
        // A node on a different engine version must not drop out of the fabric
        // because an unrelated field was renamed.
        let parsed: HealthPayload =
            serde_json::from_str(r#"{"ok":true,"generation_ready":true}"#).expect("parses");
        assert!(classify(&parsed).is_ready());
    }

    #[test]
    fn probing_an_empty_fabric_does_no_work() {
        assert!(probe_fabric(&[], DEFAULT_PROBE_TIMEOUT).is_empty());
    }

    #[test]
    fn an_unroutable_address_becomes_unreachable_not_an_error() {
        // Port 1 on loopback is closed; this exercises the real socket path.
        let spec = NodeSpec {
            label: "dead".to_string(),
            host: "127.0.0.1".to_string(),
            port: 1,
        };
        let snapshot = probe_node(&spec, Duration::from_millis(500));
        assert!(matches!(snapshot.status, NodeStatus::Unreachable { .. }));
        assert_eq!(snapshot.label(), "dead");
    }
}
