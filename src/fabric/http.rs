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

/// Bounds the header block and a chunked body's size line, so a peer that never
/// terminates either cannot grow a buffer without bound.
const MAX_HEAD_BYTES: usize = 64 * 1024;
const MAX_CHUNK_SIZE_LINE: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HttpError {
    Resolve(String),
    Connect(String),
    Io(String),
    Malformed(String),
    TooLarge(usize),
    /// The request could not be written safely; refused before any socket work.
    InvalidRequest(String),
}

impl HttpError {
    /// Whether the peer provably never received any part of the request.
    ///
    /// Only resolution and dialling qualify. Every other variant is raised at or
    /// after the first write, and `write_all` does not say how many bytes left
    /// the socket, so the request may already be running on the peer. Answering
    /// "it may have arrived" whenever that is possible is what lets a caller
    /// re-send elsewhere without risking a second execution.
    ///
    /// [`HttpError::InvalidRequest`] is deliberately excluded: nothing was sent,
    /// but the request is malformed, so another peer would refuse it identically.
    pub(crate) fn peer_never_received_it(&self) -> bool {
        match self {
            Self::Resolve(_) | Self::Connect(_) => true,
            Self::Io(_) | Self::Malformed(_) | Self::TooLarge(_) | Self::InvalidRequest(_) => false,
        }
    }
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resolve(detail) => write!(f, "cannot resolve host: {detail}"),
            Self::Connect(detail) => write!(f, "cannot connect: {detail}"),
            Self::Io(detail) => write!(f, "connection failed: {detail}"),
            Self::Malformed(detail) => write!(f, "malformed HTTP response: {detail}"),
            Self::TooLarge(limit) => write!(f, "response exceeded {limit} bytes"),
            Self::InvalidRequest(detail) => write!(f, "cannot build request: {detail}"),
        }
    }
}

/// Status line plus body of an HTTP/1.1 response.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct HttpResponse {
    pub(crate) status: u16,
    pub(crate) body: Vec<u8>,
}

/// The response headers this client acts on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResponseHead {
    pub(crate) status: u16,
    pub(crate) chunked: bool,
    pub(crate) content_length: Option<usize>,
    /// Lower-cased and stripped of parameters, so `text/event-stream;
    /// charset=utf-8` compares equal to `text/event-stream`.
    pub(crate) content_type: Option<String>,
}

impl ResponseHead {
    pub(crate) fn is_event_stream(&self) -> bool {
        self.content_type.as_deref() == Some("text/event-stream")
    }
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

/// Where an incremental [`ChunkDecoder`] is inside a chunked body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkState {
    /// Reading the hexadecimal size line.
    Size,
    /// Owed this many more payload bytes for the current chunk.
    Data(usize),
    /// Consuming the CRLF that terminates a chunk's payload.
    Crlf,
    /// The zero-size chunk arrived; the body is complete.
    Done,
}

/// A chunked body decoded as it arrives.
///
/// [`dechunk`] needs the whole body up front. A streamed body's frames straddle
/// socket reads, so this keeps the frame state across pushes. When the response
/// is not chunked it is a pass-through, which is what a node answering with
/// `Connection: close` and no length uses.
struct ChunkDecoder {
    chunked: bool,
    /// Bytes received but not yet framed.
    raw: Vec<u8>,
    state: ChunkState,
}

impl ChunkDecoder {
    fn new(chunked: bool) -> Self {
        Self {
            chunked,
            raw: Vec::new(),
            state: ChunkState::Size,
        }
    }

    fn finished(&self) -> bool {
        self.chunked && self.state == ChunkState::Done
    }

    /// Frame everything buffered so far, appending decoded payload to `out`.
    ///
    /// `max_chunk` bounds a single chunk, so a peer claiming a huge size is
    /// refused before anything is allocated for it. The total body is not
    /// bounded here: a caller streaming to a client never holds it all.
    fn push(&mut self, bytes: &[u8], max_chunk: usize, out: &mut Vec<u8>) -> Result<(), HttpError> {
        if !self.chunked {
            out.extend_from_slice(bytes);
            return Ok(());
        }
        self.raw.extend_from_slice(bytes);
        loop {
            match self.state {
                ChunkState::Done => return Ok(()),
                ChunkState::Size => {
                    let Some(line_end) = self.raw.windows(2).position(|w| w == b"\r\n") else {
                        // A peer that never terminates the size line must not
                        // grow this buffer without bound.
                        if self.raw.len() > MAX_CHUNK_SIZE_LINE {
                            return Err(HttpError::Malformed(
                                "chunk size line unterminated".to_string(),
                            ));
                        }
                        return Ok(());
                    };
                    let size_text = std::str::from_utf8(&self.raw[..line_end])
                        .map_err(|_| HttpError::Malformed("chunk size is not UTF-8".to_string()))?;
                    // A chunk extension (`1a;name=value`) is legal and ignorable.
                    let size_text = size_text.split(';').next().unwrap_or("").trim();
                    let size = usize::from_str_radix(size_text, 16).map_err(|_| {
                        HttpError::Malformed(format!("chunk size `{size_text}` is not hexadecimal"))
                    })?;
                    if size > max_chunk {
                        return Err(HttpError::TooLarge(max_chunk));
                    }
                    self.raw.drain(..line_end + 2);
                    self.state = if size == 0 {
                        // Any trailer after the terminal chunk is ignored: the
                        // connection closes next either way.
                        ChunkState::Done
                    } else {
                        ChunkState::Data(size)
                    };
                }
                ChunkState::Data(owed) => {
                    if self.raw.is_empty() {
                        return Ok(());
                    }
                    let take = owed.min(self.raw.len());
                    out.extend_from_slice(&self.raw[..take]);
                    self.raw.drain(..take);
                    self.state = if take == owed {
                        ChunkState::Crlf
                    } else {
                        ChunkState::Data(owed - take)
                    };
                }
                ChunkState::Crlf => {
                    if self.raw.len() < 2 {
                        return Ok(());
                    }
                    if &self.raw[..2] != b"\r\n" {
                        return Err(HttpError::Malformed(
                            "chunk payload not terminated by CRLF".to_string(),
                        ));
                    }
                    self.raw.drain(..2);
                    self.state = ChunkState::Size;
                }
            }
        }
    }
}

/// Parse the status line and the headers this client acts on. Pure.
fn parse_head(head: &str) -> Result<ResponseHead, HttpError> {
    let mut lines = head.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| HttpError::Malformed("empty response".to_string()))?;
    let status = parse_status_line(status_line)?;

    let mut parsed = ResponseHead {
        status,
        chunked: false,
        content_length: None,
        content_type: None,
    };
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        match name.as_str() {
            "transfer-encoding" => {
                parsed.chunked = value.to_ascii_lowercase().contains("chunked");
            }
            "content-length" => parsed.content_length = value.parse::<usize>().ok(),
            "content-type" => {
                let media = value.split(';').next().unwrap_or("").trim();
                parsed.content_type = Some(media.to_ascii_lowercase());
            }
            _ => {}
        }
    }
    Ok(parsed)
}

/// Parse a whole HTTP/1.1 response. Pure; see the tests at the bottom.
pub(crate) fn parse_response(raw: &[u8], max_body: usize) -> Result<HttpResponse, HttpError> {
    let split = find_header_end(raw)
        .ok_or_else(|| HttpError::Malformed("no header terminator".to_string()))?;
    let head = std::str::from_utf8(&raw[..split.headers_end])
        .map_err(|_| HttpError::Malformed("headers are not UTF-8".to_string()))?;
    let head = parse_head(head)?;

    let rest = &raw[split.body_start..];
    let body = if head.chunked {
        dechunk(rest, max_body)?
    } else if let Some(len) = head.content_length {
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
    Ok(HttpResponse {
        status: head.status,
        body,
    })
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

/// Build the request head. Pure, so the header set — including whether an
/// `Authorization` line is present at all — is tested without a server.
///
/// A bearer token is written into the head verbatim, so one carrying a control
/// character could append headers of its own. That is refused rather than sent.
/// Neither the error nor anything else here repeats the token's value.
fn request_head(
    method: &str,
    path: &str,
    authority: &str,
    accept: &str,
    body: Option<&[u8]>,
    bearer: Option<&str>,
) -> Result<String, HttpError> {
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {authority}\r\nAccept: {accept}\r\nConnection: close\r\n"
    );
    if let Some(token) = bearer {
        if token.is_empty() {
            return Err(HttpError::InvalidRequest(
                "bearer token is empty; pass no token rather than an empty one".to_string(),
            ));
        }
        if token.chars().any(char::is_control) {
            return Err(HttpError::InvalidRequest(
                "bearer token contains a control character".to_string(),
            ));
        }
        head.push_str(&format!("Authorization: Bearer {token}\r\n"));
    }
    if let Some(body) = body {
        head.push_str("Content-Type: application/json\r\n");
        head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    head.push_str("\r\n");
    Ok(head)
}

/// Media type sent as `Accept` on a normal request, and on one that expects a
/// server-sent event stream back.
pub(crate) const ACCEPT_JSON: &str = "application/json";
pub(crate) const ACCEPT_EVENT_STREAM: &str = "text/event-stream";

/// Resolve, connect, and write one request, leaving the socket ready to read.
///
/// Shared by [`request`] and [`open_stream`] so both dial, authenticate and
/// frame a request exactly the same way.
#[allow(clippy::too_many_arguments)]
fn connect_and_send(
    host: &str,
    port: u16,
    method: &str,
    path: &str,
    accept: &str,
    body: Option<&[u8]>,
    bearer: Option<&str>,
    deadline: Instant,
    write_timeout: Duration,
) -> Result<TcpStream, HttpError> {
    let authority = format!("{host}:{port}");
    // Built before resolving, so a request that cannot be written safely is
    // refused without touching the network.
    let head = request_head(method, path, &authority, accept, body, bearer)?;

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
    // Short socket reads keep the caller's loop responsive to its own deadline;
    // one long read timeout would overshoot it on a stalled peer.
    stream
        .set_read_timeout(Some(Duration::from_millis(100)))
        .map_err(|error| HttpError::Io(error.to_string()))?;
    stream
        .set_write_timeout(Some(write_timeout))
        .map_err(|error| HttpError::Io(error.to_string()))?;

    stream
        .write_all(head.as_bytes())
        .map_err(|error| HttpError::Io(error.to_string()))?;
    if let Some(body) = body {
        stream
            .write_all(body)
            .map_err(|error| HttpError::Io(error.to_string()))?;
    }
    Ok(stream)
}

/// Perform one request/response round trip against a node.
///
/// `timeout` bounds the whole exchange, not each socket operation, so a peer
/// that dribbles bytes forever still fails on schedule.
///
/// `bearer` is sent as `Authorization: Bearer`, which is what a node started
/// with an API key requires on every route but `/v1/health`.
// One more parameter than clippy's threshold; every one of them is a distinct
// property of a single round trip, so bundling them would only move the list.
#[allow(clippy::too_many_arguments)]
pub(crate) fn request(
    host: &str,
    port: u16,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    bearer: Option<&str>,
    timeout: Duration,
    max_body: usize,
) -> Result<HttpResponse, HttpError> {
    let deadline = Instant::now() + timeout;
    let mut stream = connect_and_send(
        host,
        port,
        method,
        path,
        ACCEPT_JSON,
        body,
        bearer,
        deadline,
        timeout,
    )?;

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
            Err(error) if is_retryable(&error) => continue,
            Err(error) => return Err(HttpError::Io(error.to_string())),
        }
    }

    parse_response(&raw, max_body)
}

/// A socket read that timed out rather than failed: the 100ms read timeout
/// fires constantly while a node is still thinking, and is not an error.
fn is_retryable(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
    )
}

/// A response whose body is read incrementally rather than all at once.
///
/// The head has already been read and parsed; the body is delivered by
/// [`ResponseStream::next_chunk`] as it arrives. Nothing here interprets the
/// payload — a caller relaying server-sent events forwards the bytes verbatim,
/// so an event field this client has never heard of cannot be mangled.
pub(crate) struct ResponseStream {
    stream: TcpStream,
    head: ResponseHead,
    decoder: ChunkDecoder,
    /// Body bytes that arrived alongside the head, not yet framed.
    pending: Vec<u8>,
    /// Decoded payload handed to the caller so far, which a declared
    /// `Content-Length` is measured against.
    delivered: usize,
    /// How long the node may send nothing at all before it counts as wedged.
    idle_timeout: Duration,
    max_chunk: usize,
}

/// A body that stopped before its framing said it would.
fn truncated_body() -> HttpError {
    HttpError::Malformed("connection closed before the response body completed".to_string())
}

/// Send a request and read only its head, leaving the body to be streamed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn open_stream(
    host: &str,
    port: u16,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    bearer: Option<&str>,
    head_timeout: Duration,
    idle_timeout: Duration,
    max_chunk: usize,
) -> Result<ResponseStream, HttpError> {
    let deadline = Instant::now() + head_timeout;
    let mut stream = connect_and_send(
        host,
        port,
        method,
        path,
        ACCEPT_EVENT_STREAM,
        body,
        bearer,
        deadline,
        head_timeout,
    )?;

    let mut raw = Vec::new();
    let mut scratch = [0_u8; 8192];
    let split = loop {
        if let Some(split) = find_header_end(&raw) {
            break split;
        }
        if raw.len() > MAX_HEAD_BYTES {
            return Err(HttpError::TooLarge(MAX_HEAD_BYTES));
        }
        if Instant::now() >= deadline {
            return Err(HttpError::Io(
                "node sent no response head before the deadline".to_string(),
            ));
        }
        match stream.read(&mut scratch) {
            Ok(0) => {
                return Err(HttpError::Malformed(
                    "connection closed before the response head completed".to_string(),
                ))
            }
            Ok(read) => raw.extend_from_slice(&scratch[..read]),
            Err(error) if is_retryable(&error) => continue,
            Err(error) => return Err(HttpError::Io(error.to_string())),
        }
    };

    let head_text = std::str::from_utf8(&raw[..split.headers_end])
        .map_err(|_| HttpError::Malformed("headers are not UTF-8".to_string()))?;
    let head = parse_head(head_text)?;

    Ok(ResponseStream {
        decoder: ChunkDecoder::new(head.chunked),
        pending: raw[split.body_start..].to_vec(),
        delivered: 0,
        head,
        stream,
        idle_timeout,
        max_chunk,
    })
}

impl ResponseStream {
    pub(crate) fn head(&self) -> &ResponseHead {
        &self.head
    }

    /// The next piece of decoded body, or `None` once the body is complete.
    ///
    /// A node that dies mid-body raises rather than ending the stream. The
    /// earlier bytes have already reached the client and cannot be taken back,
    /// but relaying this as a clean end would frame a half-generation as a whole
    /// one — the distinction chunked framing exists to carry.
    pub(crate) fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, HttpError> {
        let mut out = Vec::new();
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            self.decoder.push(&pending, self.max_chunk, &mut out)?;
            if !out.is_empty() {
                self.delivered += out.len();
                return Ok(Some(out));
            }
        }
        if self.complete() == Some(true) {
            return Ok(None);
        }

        let deadline = Instant::now() + self.idle_timeout;
        let mut scratch = [0_u8; 8192];
        loop {
            if Instant::now() >= deadline {
                return Err(HttpError::Io(
                    "node sent nothing before the idle timeout".to_string(),
                ));
            }
            match self.stream.read(&mut scratch) {
                Ok(0) if self.complete() == Some(false) => return Err(truncated_body()),
                Ok(0) => return Ok(None),
                Ok(read) => {
                    self.decoder
                        .push(&scratch[..read], self.max_chunk, &mut out)?;
                    if !out.is_empty() {
                        self.delivered += out.len();
                        return Ok(Some(out));
                    }
                    if self.complete() == Some(true) {
                        return Ok(None);
                    }
                }
                Err(error) if is_retryable(&error) => continue,
                Err(error) => return Err(HttpError::Io(error.to_string())),
            }
        }
    }

    /// Read the rest of the body into memory, for a response that is not the
    /// stream the caller asked for and is therefore small and bounded.
    pub(crate) fn into_buffered(mut self, max_body: usize) -> Result<HttpResponse, HttpError> {
        let mut body = Vec::new();
        let pending = std::mem::take(&mut self.pending);
        self.decoder.push(&pending, self.max_chunk, &mut body)?;
        self.delivered = body.len();

        let deadline = Instant::now() + self.idle_timeout;
        let mut scratch = [0_u8; 8192];
        while self.complete() != Some(true) {
            if Instant::now() >= deadline {
                return Err(HttpError::Io(
                    "node sent nothing before the idle timeout".to_string(),
                ));
            }
            match self.stream.read(&mut scratch) {
                Ok(0) => break,
                Ok(read) => {
                    self.decoder
                        .push(&scratch[..read], self.max_chunk, &mut body)?;
                    self.delivered = body.len();
                    if body.len() > max_body {
                        return Err(HttpError::TooLarge(max_body));
                    }
                }
                Err(error) if is_retryable(&error) => continue,
                Err(error) => return Err(HttpError::Io(error.to_string())),
            }
        }

        // The one-shot path calls these same bytes malformed; the two readers
        // must not disagree about whether a half-delivered body is an answer.
        if self.complete() == Some(false) {
            return Err(truncated_body());
        }
        if body.len() > max_body {
            return Err(HttpError::TooLarge(max_body));
        }
        Ok(HttpResponse {
            status: self.head.status,
            body,
        })
    }

    /// Whether the body's own framing says it is complete.
    ///
    /// `None` when the response declares no framing at all: the connection
    /// closing is then the only thing that can end it, so an EOF there is a
    /// clean end rather than a truncation.
    fn complete(&self) -> Option<bool> {
        if self.head.chunked {
            return Some(self.decoder.finished());
        }
        self.head.content_length.map(|len| self.delivered >= len)
    }
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
    fn only_a_failure_before_the_first_write_counts_as_never_received() {
        // Exhaustive on purpose: a new variant must be classified deliberately,
        // because answering "never received" wrongly permits a second execution.
        for error in [
            HttpError::Resolve("no such host".to_string()),
            HttpError::Connect("refused".to_string()),
        ] {
            assert!(error.peer_never_received_it(), "{error:?}");
        }
        for error in [
            HttpError::Io("reset".to_string()),
            HttpError::Malformed("truncated".to_string()),
            HttpError::TooLarge(1),
            HttpError::InvalidRequest("bad token".to_string()),
        ] {
            assert!(!error.peer_never_received_it(), "{error:?}");
        }
    }

    #[test]
    fn a_dial_failure_from_a_real_socket_is_classified_as_never_received() {
        // Not a hand-built variant: this is the error the dialling path actually
        // produces against a closed port, which is the case failover turns on.
        let dead = SocketAddr::from(([127, 0, 0, 1], 1));
        let error = connect_any(&[dead], Instant::now() + Duration::from_secs(1))
            .expect_err("nothing is listening");
        assert!(error.peer_never_received_it(), "{error:?}");
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

    /// Feed a body one byte at a time — the hostile case for an incremental
    /// decoder, because every frame boundary lands mid-push.
    fn decode_byte_by_byte(raw: &[u8], chunked: bool) -> Result<Vec<u8>, HttpError> {
        let mut decoder = ChunkDecoder::new(chunked);
        let mut out = Vec::new();
        for byte in raw {
            decoder.push(&[*byte], LIMIT, &mut out)?;
        }
        Ok(out)
    }

    /// Frame payloads as a chunked body. Built rather than written out so a
    /// hand-counted size can never make a fixture disagree with itself.
    fn chunked(payloads: &[&str]) -> Vec<u8> {
        let mut raw = Vec::new();
        for payload in payloads {
            raw.extend_from_slice(format!("{:x}\r\n{payload}\r\n", payload.len()).as_bytes());
        }
        raw.extend_from_slice(b"0\r\n\r\n");
        raw
    }

    #[test]
    fn an_incrementally_decoded_body_matches_the_whole_body_decoder() {
        // The same bytes through both decoders: whatever `dechunk` produces for
        // a complete body, the streaming decoder must produce for the pieces.
        let raw = chunked(&[
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
            "data: [DONE]\n\n",
        ]);
        let whole = dechunk(&raw, LIMIT).expect("whole-body decode");
        let streamed = decode_byte_by_byte(&raw, true).expect("incremental decode");
        assert_eq!(streamed, whole);
        assert_eq!(
            String::from_utf8(streamed).expect("utf8"),
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n"
        );
    }

    #[test]
    fn a_chunk_frame_split_across_reads_is_reassembled() {
        let raw = b"5\r\nhello\r\n5\r\nworld\r\n0\r\n\r\n";
        // Every possible split point, so no single boundary is special-cased.
        for at in 0..raw.len() {
            let mut decoder = ChunkDecoder::new(true);
            let mut out = Vec::new();
            decoder
                .push(&raw[..at], LIMIT, &mut out)
                .expect("first half");
            decoder
                .push(&raw[at..], LIMIT, &mut out)
                .expect("second half");
            assert_eq!(out, b"helloworld", "split at {at}");
            assert!(decoder.finished(), "split at {at}");
        }
    }

    #[test]
    fn the_terminal_chunk_ends_the_body_and_a_trailer_is_ignored() {
        let mut decoder = ChunkDecoder::new(true);
        let mut out = Vec::new();
        decoder
            .push(b"3\r\nabc\r\n0\r\nX-Trailer: 1\r\n\r\n", LIMIT, &mut out)
            .expect("decodes");
        assert_eq!(out, b"abc");
        assert!(decoder.finished());
    }

    #[test]
    fn an_unchunked_body_passes_straight_through() {
        let mut decoder = ChunkDecoder::new(false);
        let mut out = Vec::new();
        decoder
            .push(b"data: raw\n\n", LIMIT, &mut out)
            .expect("passes");
        assert_eq!(out, b"data: raw\n\n");
        // Without chunk framing there is no terminal chunk; only EOF ends it.
        assert!(!decoder.finished());
    }

    #[test]
    fn a_chunk_larger_than_the_cap_is_refused_before_it_is_allocated() {
        let mut decoder = ChunkDecoder::new(true);
        let mut out = Vec::new();
        // 0x1000000 = 16 MiB claimed against a 1 MiB cap.
        let error = decoder
            .push(b"1000000\r\n", LIMIT, &mut out)
            .expect_err("an oversized chunk is refused");
        assert!(matches!(error, HttpError::TooLarge(LIMIT)), "{error:?}");
        assert!(out.is_empty(), "nothing may be emitted for a refused chunk");
    }

    #[test]
    fn a_size_line_that_never_terminates_is_refused_rather_than_buffered() {
        let mut decoder = ChunkDecoder::new(true);
        let mut out = Vec::new();
        let filler = vec![b'a'; MAX_CHUNK_SIZE_LINE + 1];
        assert!(matches!(
            decoder.push(&filler, LIMIT, &mut out),
            Err(HttpError::Malformed(_))
        ));
    }

    #[test]
    fn a_chunk_not_terminated_by_crlf_is_malformed() {
        let mut decoder = ChunkDecoder::new(true);
        let mut out = Vec::new();
        assert!(matches!(
            decoder.push(b"3\r\nabcXX", LIMIT, &mut out),
            Err(HttpError::Malformed(_))
        ));
    }

    #[test]
    fn a_content_type_is_matched_without_its_parameters() {
        let head = parse_head(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n\
             Transfer-Encoding: chunked",
        )
        .expect("parses");
        assert!(head.is_event_stream());
        assert!(head.chunked);

        let json = parse_head("HTTP/1.1 200 OK\r\nContent-Type: application/json").expect("parses");
        assert!(!json.is_event_stream());
    }

    #[test]
    fn a_bearer_token_becomes_an_authorization_header() {
        let head = request_head(
            "POST",
            "/v1/chat/completions",
            "node:8181",
            ACCEPT_JSON,
            Some(b"{}"),
            Some("s3cret"),
        )
        .expect("a well-formed token is sendable");
        assert!(
            head.contains("\r\nAuthorization: Bearer s3cret\r\n"),
            "{head}"
        );
        // The rest of the head must survive the new line unchanged.
        assert!(
            head.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"),
            "{head}"
        );
        assert!(head.contains("\r\nHost: node:8181\r\n"), "{head}");
        assert!(head.contains("\r\nContent-Length: 2\r\n"), "{head}");
        assert!(head.ends_with("\r\n\r\n"), "{head}");
    }

    #[test]
    fn no_token_means_no_authorization_header_at_all() {
        // Not an empty one: an unauthenticated node must see the same request it
        // saw before the fabric learned about bearer tokens.
        let head = request_head("GET", "/v1/health", "node:8181", ACCEPT_JSON, None, None)
            .expect("builds");
        assert!(
            !head.to_ascii_lowercase().contains("authorization"),
            "{head}"
        );
        assert_eq!(
            head,
            "GET /v1/health HTTP/1.1\r\nHost: node:8181\r\nAccept: application/json\r\n\
             Connection: close\r\n\r\n"
        );
    }

    #[test]
    fn a_token_carrying_a_control_character_is_refused_rather_than_written() {
        // Writing this verbatim would let the token inject headers of its own.
        // `s3cret` appears in no refusal wording, so finding it in the message
        // would mean the value leaked rather than merely the word "token".
        for token in ["s3cret\r\nX-Injected: 1", "s3cret\n", "s3cret\0"] {
            let error = request_head(
                "GET",
                "/v1/health",
                "node:8181",
                ACCEPT_JSON,
                None,
                Some(token),
            )
            .expect_err("a control character is not sendable");
            match &error {
                HttpError::InvalidRequest(detail) => {
                    assert!(
                        !detail.contains("s3cret"),
                        "the token must not be echoed: {detail}"
                    )
                }
                other => panic!("expected InvalidRequest, got {other:?}"),
            }
        }
    }

    #[test]
    fn an_empty_token_is_refused_rather_than_sent_as_a_bare_bearer() {
        // `Authorization: Bearer ` is rejected by the server anyway; failing here
        // says why instead of arriving as an unexplained 401.
        assert!(matches!(
            request_head(
                "GET",
                "/v1/health",
                "node:8181",
                ACCEPT_JSON,
                None,
                Some("")
            ),
            Err(HttpError::InvalidRequest(_))
        ));
    }

    #[test]
    fn an_unsendable_token_is_refused_before_a_socket_is_opened() {
        // Port 1 is closed, so a connect error here would prove we dialled first.
        let error = request(
            "127.0.0.1",
            1,
            "GET",
            "/v1/health",
            None,
            Some("bad\r\ntoken"),
            Duration::from_millis(500),
            LIMIT,
        )
        .expect_err("refused");
        assert!(
            matches!(error, HttpError::InvalidRequest(_)),
            "must refuse before dialling, got {error:?}"
        );
    }

    #[test]
    fn a_closed_port_reports_connect_failure_not_a_panic() {
        let error = request(
            "127.0.0.1",
            1,
            "GET",
            "/v1/health",
            None,
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

    /// Whether a request head and its declared body have both arrived.
    ///
    /// One read is not enough: [`request_head`] writes the head and the body
    /// separately, so they can land in separate segments.
    fn request_complete(raw: &[u8]) -> bool {
        let Some(split) = find_header_end(raw) else {
            return false;
        };
        let head = String::from_utf8_lossy(&raw[..split.headers_end]);
        let declared = head
            .lines()
            .filter_map(|line| line.split_once(':'))
            .find(|(name, _)| name.trim().eq_ignore_ascii_case("content-length"))
            .and_then(|(_, value)| value.trim().parse::<usize>().ok())
            .unwrap_or(0);
        raw.len() >= split.body_start + declared
    }

    /// Answer one connection with canned bytes and hang up, so a frame that
    /// stops early can be exercised without a stub node.
    fn canned_node(raw: &'static [u8]) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("binds");
        let port = listener.local_addr().expect("has an address").port();
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            // Every byte of the request has to be consumed before answering:
            // closing a socket that still holds unread bytes is an abortive
            // close on Windows, and it discards the reply along with them.
            let mut request = Vec::new();
            let mut scratch = [0_u8; 1024];
            while !request_complete(&request) {
                match stream.read(&mut scratch) {
                    Ok(0) | Err(_) => break,
                    Ok(read) => request.extend_from_slice(&scratch[..read]),
                }
            }
            let _ = stream.write_all(raw);
        });
        port
    }

    fn stream_from(port: u16) -> ResponseStream {
        open_stream(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            Some(b"{}"),
            None,
            Duration::from_secs(10),
            Duration::from_secs(10),
            LIMIT,
        )
        .expect("the canned node answers")
    }

    /// `parse_response` calls a chunked body that stops early malformed. The
    /// streaming reader must not disagree with it about the same bytes: ending
    /// the relay cleanly would frame a half-generation as a whole one.
    #[test]
    fn a_stream_cut_off_before_its_terminal_chunk_is_reported_not_ended_cleanly() {
        // `data: one\n\n` is 11 bytes = 0xb. No terminal chunk follows it.
        let port = canned_node(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
              Transfer-Encoding: chunked\r\n\r\nb\r\ndata: one\n\n\r\n",
        );
        let mut stream = stream_from(port);

        assert_eq!(
            stream.next_chunk().expect("the first event arrives"),
            Some(b"data: one\n\n".to_vec())
        );
        let error = stream
            .next_chunk()
            .expect_err("the node died before the body was complete");
        assert!(matches!(error, HttpError::Malformed(_)), "{error:?}");
    }

    /// The control for the test above: a stream that does reach its terminal
    /// chunk still ends without an error, so the refusal is not blanket.
    #[test]
    fn a_stream_that_reaches_its_terminal_chunk_ends_cleanly() {
        let port = canned_node(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
              Transfer-Encoding: chunked\r\n\r\nb\r\ndata: one\n\n\r\n0\r\n\r\n",
        );
        let mut stream = stream_from(port);

        assert_eq!(
            stream.next_chunk().expect("the first event arrives"),
            Some(b"data: one\n\n".to_vec())
        );
        assert_eq!(stream.next_chunk().expect("the body is complete"), None);
    }

    /// With neither chunked framing nor a declared length, the close *is* the
    /// framing, so EOF must stay a clean end.
    #[test]
    fn a_close_delimited_stream_still_ends_at_eof() {
        let port = canned_node(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
              Connection: close\r\n\r\ndata: one\n\n",
        );
        let mut stream = stream_from(port);

        assert_eq!(
            stream.next_chunk().expect("the event arrives"),
            Some(b"data: one\n\n".to_vec())
        );
        assert_eq!(stream.next_chunk().expect("the close ends it"), None);
    }

    #[test]
    fn a_buffered_body_shorter_than_its_declared_length_is_refused() {
        let port = canned_node(
            b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n\
              Content-Length: 99\r\n\r\n{\"error\":",
        );
        let error = stream_from(port)
            .into_buffered(LIMIT)
            .expect_err("the declared body never arrived");
        assert!(matches!(error, HttpError::Malformed(_)), "{error:?}");
    }

    #[test]
    fn a_buffered_body_that_matches_its_declared_length_is_returned() {
        let port = canned_node(
            b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n\
              Content-Length: 2\r\n\r\n{}",
        );
        let response = stream_from(port).into_buffered(LIMIT).expect("complete");
        assert_eq!(response.status, 503);
        assert_eq!(response.body, b"{}");
    }
}
