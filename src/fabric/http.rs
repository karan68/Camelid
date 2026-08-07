//! Minimal blocking HTTP/1.1 client shared by the fabric's probe and forward
//! paths.
//!
//! Response parsing is pure over byte slices, so the awkward parts — chunked
//! bodies, truncated frames, a node that answers 500 — are tested without a
//! server. Only [`request`] touches a socket.
//!
//! This deliberately does not reuse `chat::client`: that module is private to
//! `chat`, is keyed on a resolved `SocketAddr` where fabric members are named
//! hosts, and carries SSE, bearer auth and tool-call handling that neither a
//! health probe nor a request forward should depend on.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

/// Never spend the whole budget dialling; a forward budget is minutes long and
/// a node that has not accepted in five seconds is not about to.
const CONNECT_ATTEMPT_CAP: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HttpError {
    Resolve(String),
    Connect(String),
    Io(String),
    Malformed(String),
    TooLarge(usize),
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resolve(detail) => write!(f, "cannot resolve host: {detail}"),
            Self::Connect(detail) => write!(f, "cannot connect: {detail}"),
            Self::Io(detail) => write!(f, "connection failed: {detail}"),
            Self::Malformed(detail) => write!(f, "malformed HTTP response: {detail}"),
            Self::TooLarge(limit) => write!(f, "response exceeded {limit} bytes"),
        }
    }
}

/// Status line plus body of an HTTP/1.1 response.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct HttpResponse {
    pub(crate) status: u16,
    pub(crate) body: Vec<u8>,
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

fn parse_status_line(line: &str) -> Result<u16, HttpError> {
    if !line.starts_with("HTTP/") {
        return Err(HttpError::Malformed(format!(
            "status line does not start with HTTP/: {line}"
        )));
    }
    let code = line
        .split(' ')
        .nth(1)
        .ok_or_else(|| HttpError::Malformed("status line has no code".to_string()))?;
    code.parse::<u16>()
        .map_err(|_| HttpError::Malformed(format!("status code `{code}` is not a number")))
}

/// Decode `Transfer-Encoding: chunked`.
fn dechunk(mut rest: &[u8], max_body: usize) -> Result<Vec<u8>, HttpError> {
    let mut out = Vec::new();
    loop {
        let line_end = rest
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or_else(|| HttpError::Malformed("chunk size line unterminated".to_string()))?;
        let size_text = std::str::from_utf8(&rest[..line_end])
            .map_err(|_| HttpError::Malformed("chunk size is not UTF-8".to_string()))?;
        // A chunk extension (`1a;name=value`) is legal and ignorable.
        let size_text = size_text.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_text, 16).map_err(|_| {
            HttpError::Malformed(format!("chunk size `{size_text}` is not hexadecimal"))
        })?;
        rest = &rest[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        if out.len() + size > max_body {
            return Err(HttpError::TooLarge(max_body));
        }
        if rest.len() < size + 2 {
            return Err(HttpError::Malformed("chunk body truncated".to_string()));
        }
        out.extend_from_slice(&rest[..size]);
        rest = &rest[size + 2..];
    }
}

/// Parse a whole HTTP/1.1 response. Pure; see the tests at the bottom.
pub(crate) fn parse_response(raw: &[u8], max_body: usize) -> Result<HttpResponse, HttpError> {
    let split = find_header_end(raw)
        .ok_or_else(|| HttpError::Malformed("no header terminator".to_string()))?;
    let head = std::str::from_utf8(&raw[..split.headers_end])
        .map_err(|_| HttpError::Malformed("headers are not UTF-8".to_string()))?;

    let mut lines = head.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| HttpError::Malformed("empty response".to_string()))?;
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
            "transfer-encoding" => chunked = value.to_ascii_lowercase().contains("chunked"),
            "content-length" => content_length = value.parse::<usize>().ok(),
            _ => {}
        }
    }

    let rest = &raw[split.body_start..];
    let body = if chunked {
        dechunk(rest, max_body)?
    } else if let Some(len) = content_length {
        if len > max_body {
            return Err(HttpError::TooLarge(max_body));
        }
        if rest.len() < len {
            return Err(HttpError::Malformed(format!(
                "body truncated: expected {len} bytes, got {}",
                rest.len()
            )));
        }
        rest[..len].to_vec()
    } else {
        rest.to_vec()
    };

    if body.len() > max_body {
        return Err(HttpError::TooLarge(max_body));
    }
    Ok(HttpResponse { status, body })
}

/// Connect to the first address that accepts.
///
/// A fabric member is usually named rather than numbered, and one name commonly
/// resolves to several addresses — typically an AAAA ahead of an A. Trying only
/// the first would report a healthy node as offline whenever its leading address
/// is unroutable, so every address gets a turn until the deadline runs out.
fn connect_any(addrs: &[SocketAddr], deadline: Instant) -> Result<TcpStream, HttpError> {
    let mut last: Option<String> = None;
    for (index, addr) in addrs.iter().enumerate() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        // Share what is left between the addresses still untried, so one that
        // black-holes instead of refusing cannot starve the ones behind it.
        let untried = (addrs.len() - index) as u32;
        let attempt = (remaining / untried)
            .min(CONNECT_ATTEMPT_CAP)
            .max(Duration::from_millis(1));
        match TcpStream::connect_timeout(addr, attempt) {
            Ok(stream) => return Ok(stream),
            Err(error) => last = Some(format!("{addr}: {error}")),
        }
    }
    Err(HttpError::Connect(last.unwrap_or_else(|| {
        "no address accepted a connection within the timeout".to_string()
    })))
}

/// Perform one request/response round trip against a node.
///
/// `timeout` bounds the whole exchange, not each socket operation, so a peer
/// that dribbles bytes forever still fails on schedule.
pub(crate) fn request(
    host: &str,
    port: u16,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    timeout: Duration,
    max_body: usize,
) -> Result<HttpResponse, HttpError> {
    let deadline = Instant::now() + timeout;
    let authority = format!("{host}:{port}");

    let addrs: Vec<SocketAddr> = authority
        .to_socket_addrs()
        .map_err(|error| HttpError::Resolve(error.to_string()))?
        .collect();
    if addrs.is_empty() {
        return Err(HttpError::Resolve(
            "host resolved to no addresses".to_string(),
        ));
    }

    let mut stream = connect_any(&addrs, deadline)?;
    // Short socket reads keep the loop responsive to the overall deadline; one
    // long read timeout would overshoot it on a stalled peer.
    stream
        .set_read_timeout(Some(Duration::from_millis(100)))
        .map_err(|error| HttpError::Io(error.to_string()))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|error| HttpError::Io(error.to_string()))?;

    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: {authority}\r\nAccept: application/json\r\nConnection: close\r\n");
    if let Some(body) = body {
        head.push_str("Content-Type: application/json\r\n");
        head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    head.push_str("\r\n");

    stream
        .write_all(head.as_bytes())
        .map_err(|error| HttpError::Io(error.to_string()))?;
    if let Some(body) = body {
        stream
            .write_all(body)
            .map_err(|error| HttpError::Io(error.to_string()))?;
    }

    let mut raw = Vec::new();
    let mut chunk = [0_u8; 8192];
    loop {
        if Instant::now() >= deadline {
            return Err(HttpError::Io("request exceeded its deadline".to_string()));
        }
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => {
                raw.extend_from_slice(&chunk[..read]);
                if raw.len() > max_body {
                    return Err(HttpError::TooLarge(max_body));
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
            Err(error) => return Err(HttpError::Io(error.to_string())),
        }
    }

    parse_response(&raw, max_body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    const LIMIT: usize = 1024 * 1024;

    #[test]
    fn a_dead_leading_address_does_not_hide_a_live_one() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("binds");
        let live = listener.local_addr().expect("has an address");
        // Port 1 is closed; it stands in for an unroutable AAAA ahead of the A.
        let dead = SocketAddr::from(([127, 0, 0, 1], 1));

        let stream = connect_any(&[dead, live], Instant::now() + Duration::from_secs(2))
            .expect("the second address accepts");
        assert_eq!(stream.peer_addr().expect("connected"), live);
    }

    #[test]
    fn every_address_failing_reports_the_last_failure() {
        let dead = SocketAddr::from(([127, 0, 0, 1], 1));
        let error = connect_any(&[dead, dead], Instant::now() + Duration::from_secs(1))
            .expect_err("nothing is listening");
        assert!(matches!(error, HttpError::Connect(_)), "{error:?}");
    }

    #[test]
    fn a_content_length_body_is_read_exactly() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello-trailing-garbage";
        let response = parse_response(raw, LIMIT).expect("parses");
        assert_eq!(response.status, 200);
        assert_eq!(response.body, b"hello");
    }

    #[test]
    fn a_chunked_body_is_reassembled() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        assert_eq!(
            parse_response(raw, LIMIT).expect("parses").body,
            b"hello world"
        );
    }

    #[test]
    fn chunk_extensions_are_ignored() {
        let raw =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5;name=value\r\nhello\r\n0\r\n\r\n";
        assert_eq!(parse_response(raw, LIMIT).expect("parses").body, b"hello");
    }

    #[test]
    fn a_body_without_framing_headers_is_read_to_end() {
        let raw = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{}";
        assert_eq!(parse_response(raw, LIMIT).expect("parses").body, b"{}");
    }

    #[test]
    fn header_names_are_matched_case_insensitively() {
        let raw = b"HTTP/1.1 200 OK\r\ncOnTeNt-LeNgTh: 2\r\n\r\nok";
        assert_eq!(parse_response(raw, LIMIT).expect("parses").body, b"ok");
    }

    #[test]
    fn a_non_http_greeting_is_refused_rather_than_guessed_at() {
        let raw = b"SSH-2.0-OpenSSH_9.0\r\n\r\n";
        assert!(matches!(
            parse_response(raw, LIMIT),
            Err(HttpError::Malformed(_))
        ));
    }

    #[test]
    fn a_response_with_no_header_terminator_is_refused() {
        assert!(matches!(
            parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n", LIMIT),
            Err(HttpError::Malformed(_))
        ));
    }

    #[test]
    fn a_truncated_body_is_refused_rather_than_silently_short() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 99\r\n\r\nshort";
        assert!(matches!(
            parse_response(raw, LIMIT),
            Err(HttpError::Malformed(_))
        ));
    }

    #[test]
    fn a_truncated_chunk_is_refused() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n9\r\nshort\r\n";
        assert!(matches!(
            parse_response(raw, LIMIT),
            Err(HttpError::Malformed(_))
        ));
    }

    #[test]
    fn an_oversized_content_length_is_refused_before_allocating() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 99999999\r\n\r\n";
        assert_eq!(parse_response(raw, LIMIT), Err(HttpError::TooLarge(LIMIT)));
    }

    #[test]
    fn an_oversized_chunked_body_is_refused_mid_stream() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n";
        assert_eq!(parse_response(raw, 2), Err(HttpError::TooLarge(2)));
    }

    #[test]
    fn a_non_200_is_reported_with_its_code() {
        let raw = b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n";
        assert_eq!(parse_response(raw, LIMIT).expect("parses").status, 503);
    }

    #[test]
    fn a_closed_port_reports_connect_failure_not_a_panic() {
        let error = request(
            "127.0.0.1",
            1,
            "GET",
            "/v1/health",
            None,
            Duration::from_millis(500),
            LIMIT,
        )
        .expect_err("port 1 is closed");
        assert!(
            matches!(error, HttpError::Connect(_) | HttpError::Io(_)),
            "unexpected: {error:?}"
        );
    }

    #[test]
    fn an_unresolvable_host_reports_a_resolve_failure() {
        let error = request(
            "camelid-fabric-host-that-does-not-exist.invalid",
            8181,
            "GET",
            "/v1/health",
            None,
            Duration::from_millis(500),
            LIMIT,
        )
        .expect_err("`.invalid` never resolves");
        assert!(
            matches!(error, HttpError::Resolve(_)),
            "unexpected: {error:?}"
        );
    }
}
