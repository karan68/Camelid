//! Gemma 4 distributed layer sharding over TCP — two-node pipeline inference.
//!
//! Honest claim boundary: this is **distributed layer sharding** (each node
//! holds a contiguous layer range and the hidden state crosses the wire at the
//! cut point), NOT shared memory. The win is memory headroom: a row whose
//! weights do not fit one machine's budget (e.g. 12B-it Q8_0 at 12.7 GB on a
//! 16 GB Mac) runs with ~half the weight bytes resident per node.
//!
//! Topology: the MASTER owns layers `[0, split)` plus tokenization and the
//! greedy loop; the WORKER owns layers `[split, block_count)` plus the output
//! head, and returns the greedy argmax token id (optionally full logits for
//! parity audits). PLE inputs are recomputed on each node from the token id —
//! they depend only on the token's embedding row, so the wire carries exactly
//! `(token, position, hidden_state)` per step.
//!
//! Determinism: each node runs the same `Gemma4Runtime::step_range` math as the
//! single-node runtime; the hidden state crosses the wire as raw little-endian
//! f32 (Apple Silicon ↔ Apple Silicon), so distributed greedy output is
//! bit-comparable to single-node output. Every packet carries an FNV-1a
//! checksum and the session opens with a version/model/range handshake, so a
//! mismatched master/worker pair fails closed instead of silently diverging.

use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::ops::Range;
use std::path::Path;

use crate::gemma4_runtime::{Gemma4Runtime, Gemma4StepOutput};
use crate::{BackendError, Result};

/// Wire protocol version. Bump on ANY change to the message layout.
pub const GEMMA4_WIRE_VERSION: u32 = 1;
const HELLO_MAGIC: u32 = 0xCA4E1147;
const STEP_MAGIC: u32 = 0xCA4E5701;
const RESP_MAGIC: u32 = 0xCA4E5702;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn io_err(context: &str, e: std::io::Error) -> BackendError {
    BackendError::InvalidModelMetadata(format!("gemma4 distributed {context}: {e}"))
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn write_f32s<W: Write>(w: &mut W, values: &[f32]) -> std::io::Result<()> {
    // Little-endian f32, written per value (no unsafe transmute).
    let mut buf = Vec::with_capacity(values.len() * 4);
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    w.write_all(&buf)
}

fn read_f32s<R: Read>(r: &mut R, count: usize) -> std::io::Result<Vec<f32>> {
    let mut buf = vec![0u8; count * 4];
    r.read_exact(&mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn f32s_checksum(values: &[f32]) -> u64 {
    let mut buf = Vec::with_capacity(values.len() * 4);
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fnv1a(&buf)
}

/// Identity both ends must agree on before any activation crosses the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4Handshake {
    pub wire_version: u32,
    pub block_count: u32,
    pub hidden: u32,
    pub worker_first_layer: u32,
    pub worker_last_layer: u32,
    pub model_file_len: u64,
    /// True when the master wants full logits back each step (parity audits).
    pub return_logits: bool,
}

impl Gemma4Handshake {
    fn write<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        write_u32(w, HELLO_MAGIC)?;
        write_u32(w, self.wire_version)?;
        write_u32(w, self.block_count)?;
        write_u32(w, self.hidden)?;
        write_u32(w, self.worker_first_layer)?;
        write_u32(w, self.worker_last_layer)?;
        write_u64(w, self.model_file_len)?;
        write_u32(w, self.return_logits as u32)?;
        w.flush()
    }

    fn read<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let magic = read_u32(r)?;
        if magic != HELLO_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad hello magic {magic:#x}"),
            ));
        }
        Ok(Self {
            wire_version: read_u32(r)?,
            block_count: read_u32(r)?,
            hidden: read_u32(r)?,
            worker_first_layer: read_u32(r)?,
            worker_last_layer: read_u32(r)?,
            model_file_len: read_u64(r)?,
            return_logits: read_u32(r)? != 0,
        })
    }
}

fn model_file_len(path: &Path) -> Result<u64> {
    Ok(std::fs::metadata(path)
        .map_err(|e| BackendError::Io {
            path: path.to_path_buf(),
            source: e,
        })?
        .len())
}

/// Run the worker: load layers `range` (+ output head) and serve one master
/// connection at a time, forever. Each accepted connection is one generation
/// session with fresh KV caches.
pub fn run_worker(model: &Path, addr: &str, range: Range<usize>) -> Result<()> {
    // Bind BEFORE the (slow) shard load so a master can connect immediately;
    // its handshake waits in the accept backlog until the weights are ready.
    let listener = TcpListener::bind(addr).map_err(|e| io_err("bind", e))?;
    run_worker_on_listener(model, listener, range)
}

/// Serve the worker protocol on an already-bound listener.
///
/// Callers that must know the port before the worker starts — tests binding
/// `127.0.0.1:0` for an ephemeral port — bind themselves and hand the listener
/// over. That removes both the hard-coded port that can collide with whatever
/// else is on the box and the window in which the address is unbound, so no
/// readiness poll is needed: the socket is listening before this is called.
pub fn run_worker_on_listener(
    model: &Path,
    listener: TcpListener,
    range: Range<usize>,
) -> Result<()> {
    let addr = listener.local_addr().map_err(|e| io_err("local_addr", e))?;
    let runtime = Gemma4Runtime::load_layer_range(model, Some(range.clone()))?;
    if runtime.local_layer_range().end != runtime.block_count() {
        return Err(BackendError::InvalidModelMetadata(format!(
            "gemma4 worker must own the tail (layers ..{}); got {:?}",
            runtime.block_count(),
            runtime.local_layer_range()
        )));
    }
    let file_len = model_file_len(model)?;
    eprintln!(
        "[gemma4-worker] serving layers {:?} of {} on {addr}",
        runtime.local_layer_range(),
        runtime.block_count()
    );
    for stream in listener.incoming() {
        let stream = stream.map_err(|e| io_err("accept", e))?;
        if let Err(e) = serve_session(&runtime, file_len, stream) {
            eprintln!("[gemma4-worker] session ended: {e}");
        }
    }
    Ok(())
}

fn serve_session(runtime: &Gemma4Runtime, file_len: u64, stream: TcpStream) -> Result<()> {
    stream.set_nodelay(true).ok();
    // Symmetric to the master: a worker blocked reading the next step must not
    // hold the session (and the serial accept loop behind it) open forever when
    // the master's host disappears.
    arm_keepalive(&stream);
    // Keepalive covers a master that vanishes; this covers one that stays up but
    // stops sending. `run_worker_on_listener` accepts strictly one session at a
    // time, so without a bound here a single wedged master makes the worker
    // unreachable to every future master until the process is restarted. The
    // budget is generous because an idle gap between steps is legitimate — a
    // master mid-generation but slow is not an error.
    stream
        .set_read_timeout(Some(WORKER_SESSION_IDLE_TIMEOUT))
        .ok();
    let peer = stream.peer_addr().map_err(|e| io_err("peer_addr", e))?;
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| io_err("clone", e))?);
    let mut writer = BufWriter::new(stream);

    let hello = Gemma4Handshake::read(&mut reader).map_err(|e| io_err("hello read", e))?;
    let expected = Gemma4Handshake {
        wire_version: GEMMA4_WIRE_VERSION,
        block_count: runtime.block_count() as u32,
        hidden: runtime.hidden_size() as u32,
        worker_first_layer: runtime.local_layer_range().start as u32,
        worker_last_layer: runtime.local_layer_range().end as u32,
        model_file_len: file_len,
        return_logits: hello.return_logits, // master's choice
    };
    if hello != expected {
        // Reject with the exact mismatch, then close.
        let msg = format!("handshake mismatch: master sent {hello:?}, worker expects {expected:?}");
        write_u32(&mut writer, RESP_MAGIC).ok();
        write_u32(&mut writer, 1).ok(); // status 1 = rejected
        let bytes = msg.as_bytes();
        write_u32(&mut writer, bytes.len() as u32).ok();
        writer.write_all(bytes).ok();
        writer.flush().ok();
        return Err(BackendError::InvalidModelMetadata(msg));
    }
    write_u32(&mut writer, RESP_MAGIC).map_err(|e| io_err("hello ack", e))?;
    write_u32(&mut writer, 0).map_err(|e| io_err("hello ack", e))?;
    writer.flush().map_err(|e| io_err("hello ack", e))?;
    eprintln!(
        "[gemma4-worker] session from {peer} (return_logits={})",
        hello.return_logits
    );

    let hidden = runtime.hidden_size();
    let (mut kc, mut vc) = runtime.empty_kv_caches();
    loop {
        let magic = match read_u32(&mut reader) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            // Idle timeout: drop this session so the accept loop is free again.
            // Not a failure of the worker, so return Ok — the master gets an
            // EOF on its next step and reports it from its own side.
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                eprintln!(
                    "[gemma4-worker] session from {peer} idle for \
                     {WORKER_SESSION_IDLE_TIMEOUT:?}; closing so the next master can connect"
                );
                return Ok(());
            }
            Err(e) => return Err(io_err("step read", e)),
        };
        if magic != STEP_MAGIC {
            return Err(BackendError::InvalidModelMetadata(format!(
                "gemma4 distributed: bad step magic {magic:#x}"
            )));
        }
        let token = read_u32(&mut reader).map_err(|e| io_err("step token", e))?;
        let pos = read_u32(&mut reader).map_err(|e| io_err("step pos", e))? as usize;
        let h_len = read_u32(&mut reader).map_err(|e| io_err("step h_len", e))? as usize;
        if h_len != hidden {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "gemma4 distributed: hidden {h_len} != expected {hidden}"
            )));
        }
        let h = read_f32s(&mut reader, h_len).map_err(|e| io_err("step h", e))?;
        let sent_checksum = read_u64(&mut reader).map_err(|e| io_err("step checksum", e))?;
        let computed = f32s_checksum(&h);
        if sent_checksum != computed {
            return Err(BackendError::InvalidModelMetadata(format!(
                "gemma4 distributed: activation checksum mismatch at pos {pos} \
                 (sent {sent_checksum:#x}, computed {computed:#x})"
            )));
        }

        let logits = match runtime.step_range(token, pos, Some(h), &mut kc, &mut vc)? {
            Gemma4StepOutput::Logits(logits) => logits,
            Gemma4StepOutput::Hidden(_) => {
                return Err(BackendError::InvalidModelMetadata(
                    "gemma4 worker did not own the final layer".into(),
                ))
            }
        };
        let (next, max_logit) = greedy_argmax(&logits);

        write_u32(&mut writer, RESP_MAGIC).map_err(|e| io_err("resp", e))?;
        write_u32(&mut writer, 0).map_err(|e| io_err("resp", e))?;
        write_u32(&mut writer, next).map_err(|e| io_err("resp", e))?;
        writer
            .write_all(&max_logit.to_le_bytes())
            .map_err(|e| io_err("resp", e))?;
        if hello.return_logits {
            write_u32(&mut writer, logits.len() as u32).map_err(|e| io_err("resp", e))?;
            write_f32s(&mut writer, &logits).map_err(|e| io_err("resp logits", e))?;
            write_u64(&mut writer, f32s_checksum(&logits)).map_err(|e| io_err("resp", e))?;
        } else {
            write_u32(&mut writer, 0).map_err(|e| io_err("resp", e))?;
        }
        writer.flush().map_err(|e| io_err("resp flush", e))?;
    }
}

fn greedy_argmax(logits: &[f32]) -> (u32, f32) {
    let mut best = 0usize;
    let mut best_v = f32::MIN;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best = i;
            best_v = v;
        }
    }
    (best as u32, best_v)
}

/// One step's reply from the worker.
pub struct WorkerStep {
    pub next_token: u32,
    pub max_logit: f32,
    pub logits: Option<Vec<f32>>,
}

/// Steady-state per-step read budget. A healthy loopback step is ~40ms and a
/// LAN step is a few hundred; this is the "the peer is gone" backstop, not a
/// performance bound.
const STEP_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Arm TCP keepalive so a *dead* peer is detected in ~35s by the kernel, even
/// while a read is inside the long [`FIRST_STEP_READ_TIMEOUT`] budget.
///
/// This is what makes that budget safe to grant. A read timeout alone cannot
/// tell "the worker is slowly paging in its shard" from "the worker's host fell
/// off the network" — both surface as EAGAIN, so the only way to stay
/// responsive to the second was to be intolerant of the first. Keepalive splits
/// them at the right layer: probes are answered by the peer's TCP stack, not by
/// its blocked application thread, so a busy-but-alive worker holds the
/// connection while a black-holed one fails `PROBE_COUNT` probes and errors the
/// socket out promptly instead of waiting out the full budget.
///
/// There is no serve-side request timeout above this (the router mounts only
/// tower-http `trace`/`cors`), so this is the only bound on a hung dial.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn arm_keepalive(stream: &TcpStream) {
    use std::os::unix::io::AsRawFd;
    /// Quiet time before the first probe.
    const IDLE_SECS: libc::c_int = 15;
    /// Gap between probes once they start.
    const INTERVAL_SECS: libc::c_int = 5;
    /// Unanswered probes before the kernel kills the connection.
    const PROBE_COUNT: libc::c_int = 4;

    let fd = stream.as_raw_fd();
    let set = |level: libc::c_int, name: libc::c_int, value: libc::c_int| {
        // SAFETY: `fd` is a live socket owned by `stream` for this call, and
        // every option below takes a c_int of exactly this size.
        unsafe {
            libc::setsockopt(
                fd,
                level,
                name,
                std::ptr::addr_of!(value).cast::<libc::c_void>(),
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
        }
    };
    set(libc::SOL_SOCKET, libc::SO_KEEPALIVE, 1);
    #[cfg(target_os = "macos")]
    set(libc::IPPROTO_TCP, libc::TCP_KEEPALIVE, IDLE_SECS);
    #[cfg(target_os = "linux")]
    set(libc::IPPROTO_TCP, libc::TCP_KEEPIDLE, IDLE_SECS);
    set(libc::IPPROTO_TCP, libc::TCP_KEEPINTVL, INTERVAL_SECS);
    set(libc::IPPROTO_TCP, libc::TCP_KEEPCNT, PROBE_COUNT);
}

/// Keepalive tuning is platform-specific; elsewhere the read timeouts stand
/// alone (the supported distributed deployment is macOS-to-macOS).
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn arm_keepalive(_stream: &TcpStream) {}

/// Budget for the FIRST step of a session.
///
/// The worker binds and answers the handshake before its shard is resident:
/// `load_layer_range` maps the GGUF lazily and warms it on a background thread
/// (that advisory blocks for minutes on macOS over USB, so it cannot run on the
/// load path). The whole cold-fault cost therefore lands on the first
/// `step_range` — measured at 171s for a 1.9GB tail shard off a 38MB/s USB
/// volume, against the 30s steady-state budget. Charging cold I/O to a
/// steady-state timeout turned a slow-but-correct run into a hard abort, so the
/// first step gets its own budget — bounded in practice by [`arm_keepalive`],
/// which drops a genuinely dead peer in ~35s rather than letting it sit here for
/// the full budget — and later steps tighten back to [`STEP_READ_TIMEOUT`].
///
/// Sized from measurement, not guessed: worst observed first token was 171s
/// before the warm-up was range-scoped and 57s after, so this is ~5x the
/// realistic worst case. It is deliberately NOT larger — the residual exposure
/// is a worker that handshakes and then goes mute (still ACKing, so keepalive
/// cannot see it), which costs one budget per request.
const FIRST_STEP_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// How long a worker waits for the next step before abandoning the session.
///
/// The accept loop is serial — one session at a time — so an abandoned-but-open
/// session locks out every future master. Generous enough that a legitimately
/// slow master is never dropped mid-generation; the master's own first step is
/// bounded by [`FIRST_STEP_READ_TIMEOUT`], so a healthy pair never idles here.
const WORKER_SESSION_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// Master-side connection to a gemma4 worker (one generation session).
pub struct Gemma4WorkerClient {
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
    /// Cleared once a step response has arrived, at which point the read
    /// timeout drops from [`FIRST_STEP_READ_TIMEOUT`] to [`STEP_READ_TIMEOUT`].
    cold: bool,
}

/// Why a connect attempt failed — permanent errors must not be retried.
enum ConnectFailure {
    /// The worker answered and refused: version/shape/model mismatch. Retrying
    /// a deterministic fail-closed rejection just burns the retry budget in
    /// `sleep` (10 attempts x 3s = 30s) and buries the real message.
    Permanent(BackendError),
    /// Dial or IO error — the LAN flap case the retry loop exists for.
    Transient(BackendError),
}

impl Gemma4WorkerClient {
    /// Connect with bounded retries and IO timeouts. The two-Mac hosts are
    /// dual-homed (ethernet + wifi) and outbound sessions can flap mid-
    /// handshake — a blocked read on a black-holed connection would otherwise
    /// hang a serve request forever.
    ///
    /// Worst-case wall time is bounded by the handshake running on
    /// [`STEP_READ_TIMEOUT`], not [`FIRST_STEP_READ_TIMEOUT`]: 10 attempts x
    /// (30s read + 3s backoff) ~= 5.5 minutes. Only a permanent rejection short-
    /// circuits earlier; the cold budget is armed after this returns and so is
    /// never multiplied by the retry count.
    pub fn connect(addr: &str, handshake: &Gemma4Handshake) -> Result<Self> {
        // The recorded flap windows last seconds, not milliseconds — spread
        // the attempts over ~30s so one bad window cannot fail a model load.
        const ATTEMPTS: usize = 10;
        let mut last_err = None;
        for attempt in 0..ATTEMPTS {
            match Self::connect_once(addr, handshake) {
                Ok(client) => return Ok(client),
                // A handshake rejection is the worker's considered answer, not a
                // flap: fail immediately with its message.
                Err(ConnectFailure::Permanent(e)) => return Err(e),
                Err(ConnectFailure::Transient(e)) => {
                    eprintln!(
                        "[gemma4-master] worker connect attempt {}/{ATTEMPTS} failed: {e}",
                        attempt + 1
                    );
                    last_err = Some(e);
                    if attempt + 1 < ATTEMPTS {
                        std::thread::sleep(std::time::Duration::from_secs(3));
                    }
                }
            }
        }
        Err(last_err.expect("at least one attempt"))
    }

    fn connect_once(
        addr: &str,
        handshake: &Gemma4Handshake,
    ) -> std::result::Result<Self, ConnectFailure> {
        use ConnectFailure::{Permanent, Transient};
        let transient = |ctx: &'static str| move |e| Transient(io_err(ctx, e));

        let sock_addr = addr
            .to_socket_addrs()
            // A malformed `host:port` is an operator typo, not a flap — the
            // most likely resolve failure here, and retrying it 10x buries the
            // message behind 30s of sleep.
            .map_err(|e| Permanent(io_err("resolve", e)))?
            .next()
            .ok_or_else(|| {
                Permanent(BackendError::InvalidModelMetadata(format!(
                    "gemma4 distributed: worker address {addr} resolved to nothing"
                )))
            })?;
        let stream = TcpStream::connect_timeout(&sock_addr, std::time::Duration::from_secs(10))
            .map_err(transient("connect"))?;
        // The handshake gets the STEADY budget, not the cold one: the worker
        // answers it straight off the accept without touching a weight byte, so
        // a slow ack means something is wrong, not that a shard is paging in.
        // Granting the cold budget here would be retried ATTEMPTS times by
        // `connect` (a hello-ack timeout is Transient), stacking to ~100
        // minutes against a worker that accepts and then never answers —
        // keepalive cannot bound that one, because such a peer is still ACKing.
        // The cold budget is armed below, once the handshake has succeeded.
        stream.set_read_timeout(Some(STEP_READ_TIMEOUT)).ok();
        stream.set_write_timeout(Some(STEP_READ_TIMEOUT)).ok();
        stream.set_nodelay(true).ok();
        arm_keepalive(&stream);
        let mut reader = BufReader::new(stream.try_clone().map_err(transient("clone"))?);
        let mut writer = BufWriter::new(stream);
        handshake.write(&mut writer).map_err(transient("hello"))?;
        let magic = read_u32(&mut reader).map_err(transient("hello ack"))?;
        let status = read_u32(&mut reader).map_err(transient("hello ack"))?;
        if magic != RESP_MAGIC {
            // Not our protocol — a retry cannot turn this into a gemma4 worker.
            return Err(Permanent(BackendError::InvalidModelMetadata(format!(
                "gemma4 distributed: bad hello ack magic {magic:#x}"
            ))));
        }
        if status != 0 {
            // Cap the peer-controlled length: the reject body is a diagnostic
            // string, and an unbounded u32 would let a hostile or confused peer
            // make us allocate 4GB before we even read it.
            const MAX_REJECT_MSG: usize = 64 * 1024;
            let len = (read_u32(&mut reader).map_err(transient("hello reject"))? as usize)
                .min(MAX_REJECT_MSG);
            let mut msg = vec![0u8; len];
            reader
                .read_exact(&mut msg)
                .map_err(transient("hello reject"))?;
            return Err(Permanent(BackendError::InvalidModelMetadata(
                String::from_utf8_lossy(&msg).into_owned(),
            )));
        }
        // Handshake done — now grant the first step its cold budget. This is
        // reached exactly once per session and is never retried, so the worst
        // case is one FIRST_STEP_READ_TIMEOUT, not ATTEMPTS of them.
        reader
            .get_ref()
            .set_read_timeout(Some(FIRST_STEP_READ_TIMEOUT))
            .ok();
        Ok(Self {
            reader,
            writer,
            cold: true,
        })
    }

    /// Send one (token, position, hidden) step and receive the worker's result.
    pub fn step(&mut self, token: u32, pos: usize, h: &[f32]) -> Result<WorkerStep> {
        write_u32(&mut self.writer, STEP_MAGIC).map_err(|e| io_err("step", e))?;
        write_u32(&mut self.writer, token).map_err(|e| io_err("step", e))?;
        write_u32(&mut self.writer, pos as u32).map_err(|e| io_err("step", e))?;
        write_u32(&mut self.writer, h.len() as u32).map_err(|e| io_err("step", e))?;
        write_f32s(&mut self.writer, h).map_err(|e| io_err("step h", e))?;
        write_u64(&mut self.writer, f32s_checksum(h)).map_err(|e| io_err("step", e))?;
        self.writer.flush().map_err(|e| io_err("step flush", e))?;

        let magic = read_u32(&mut self.reader).map_err(|e| {
            // Only a TIMEOUT on the first step means "still paging in". A reset
            // or EOF is the worker dying, and blaming a cold shard for it would
            // send the reader looking at disk speed instead of the worker log.
            let timed_out = matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            );
            if self.cold && timed_out {
                // Name the budget that was exceeded — "Resource temporarily
                // unavailable" alone points nowhere near a cold shard.
                return BackendError::InvalidModelMetadata(format!(
                    "gemma4 distributed: no response to the first step (budget \
                     {FIRST_STEP_READ_TIMEOUT:?}; keepalive may have dropped a dead peer \
                     sooner) — either the worker is still paging in its shard from a slow or \
                     remote model volume, or it is unreachable; check the worker log: {e}"
                ));
            }
            io_err("resp", e)
        })?;
        if self.cold {
            // The worker is answering, so its shard is resident: drop to the
            // steady-state budget for the rest of the session.
            self.reader
                .get_ref()
                .set_read_timeout(Some(STEP_READ_TIMEOUT))
                .ok();
            self.cold = false;
        }
        if magic != RESP_MAGIC {
            return Err(BackendError::InvalidModelMetadata(format!(
                "gemma4 distributed: bad resp magic {magic:#x}"
            )));
        }
        let status = read_u32(&mut self.reader).map_err(|e| io_err("resp", e))?;
        if status != 0 {
            return Err(BackendError::InvalidModelMetadata(
                "gemma4 distributed: worker rejected step".into(),
            ));
        }
        let next_token = read_u32(&mut self.reader).map_err(|e| io_err("resp", e))?;
        let mut b = [0u8; 4];
        self.reader
            .read_exact(&mut b)
            .map_err(|e| io_err("resp", e))?;
        let max_logit = f32::from_le_bytes(b);
        let logits_len = read_u32(&mut self.reader).map_err(|e| io_err("resp", e))? as usize;
        let logits = if logits_len > 0 {
            let values = read_f32s(&mut self.reader, logits_len).map_err(|e| io_err("resp", e))?;
            let sent = read_u64(&mut self.reader).map_err(|e| io_err("resp", e))?;
            let computed = f32s_checksum(&values);
            if sent != computed {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "gemma4 distributed: logits checksum mismatch at pos {pos}"
                )));
            }
            Some(values)
        } else {
            None
        };
        Ok(WorkerStep {
            next_token,
            max_logit,
            logits,
        })
    }
}

/// Per-step wire/timing measurements from a master generation run.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct Gemma4DistributedStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub activation_payload_bytes_per_step: usize,
    pub ttft_ms: f64,
    pub decode_tokens_per_s: f64,
    pub total_wire_round_trips: usize,
    pub local_step_ms_avg: f64,
    pub wire_step_ms_avg: f64,
}

/// Run the master: layers `[0, split)` locally, the rest on the worker. Returns
/// (decoded text, generated ids, stats).
pub fn run_master(
    model: &Path,
    worker_addr: &str,
    split: usize,
    prompt: &str,
    max_new: usize,
    return_logits: bool,
) -> Result<(String, Vec<u32>, Gemma4DistributedStats)> {
    let runtime = Gemma4Runtime::load_layer_range(model, Some(0..split))?;
    let handshake = Gemma4Handshake {
        wire_version: GEMMA4_WIRE_VERSION,
        block_count: runtime.block_count() as u32,
        hidden: runtime.hidden_size() as u32,
        worker_first_layer: split as u32,
        worker_last_layer: runtime.block_count() as u32,
        model_file_len: model_file_len(model)?,
        return_logits,
    };
    let mut client = Gemma4WorkerClient::connect(worker_addr, &handshake)?;

    let prompt_tokens = runtime.tokenizer().encode(prompt, true, true)?;
    let stop = runtime.stop_token_ids();
    let (mut kc, mut vc) = runtime.empty_kv_caches();
    let hidden = runtime.hidden_size();

    let mut stats = Gemma4DistributedStats {
        prompt_tokens: prompt_tokens.len(),
        activation_payload_bytes_per_step: hidden * 4 + 24,
        ..Default::default()
    };
    let mut local_ms = 0f64;
    let mut wire_ms = 0f64;

    let t_start = std::time::Instant::now();
    let mut last_next = 0u32;
    let feed = |token: u32,
                pos: usize,
                kc: &mut crate::gemma4_runtime::Gemma4KvCache,
                vc: &mut crate::gemma4_runtime::Gemma4KvCache,
                client: &mut Gemma4WorkerClient,
                local_ms: &mut f64,
                wire_ms: &mut f64|
     -> Result<u32> {
        let t0 = std::time::Instant::now();
        let h = match runtime.step_range(token, pos, None, kc, vc)? {
            Gemma4StepOutput::Hidden(h) => h,
            Gemma4StepOutput::Logits(_) => {
                return Err(BackendError::InvalidModelMetadata(
                    "gemma4 master unexpectedly owns the full model; use single-node".into(),
                ))
            }
        };
        *local_ms += t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        let reply = client.step(token, pos, &h)?;
        *wire_ms += t1.elapsed().as_secs_f64() * 1e3;
        Ok(reply.next_token)
    };

    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        last_next = feed(
            tok,
            pos,
            &mut kc,
            &mut vc,
            &mut client,
            &mut local_ms,
            &mut wire_ms,
        )?;
        stats.total_wire_round_trips += 1;
    }
    stats.ttft_ms = t_start.elapsed().as_secs_f64() * 1e3;

    let mut generated = Vec::new();
    let t_decode = std::time::Instant::now();
    // `pos` is the absolute sequence position of the token being fed back.
    for pos in prompt_tokens.len()..prompt_tokens.len() + max_new {
        if stop.contains(&last_next) {
            break;
        }
        generated.push(last_next);
        last_next = feed(
            last_next,
            pos,
            &mut kc,
            &mut vc,
            &mut client,
            &mut local_ms,
            &mut wire_ms,
        )?;
        stats.total_wire_round_trips += 1;
    }
    let decode_s = t_decode.elapsed().as_secs_f64();
    stats.generated_tokens = generated.len();
    stats.decode_tokens_per_s = if decode_s > 0.0 {
        generated.len() as f64 / decode_s
    } else {
        0.0
    };
    let steps = stats.total_wire_round_trips.max(1) as f64;
    stats.local_step_ms_avg = local_ms / steps;
    stats.wire_step_ms_avg = wire_ms / steps;

    let text = runtime.tokenizer().decode(&generated, true)?;
    Ok((text, generated, stats))
}

/// Persistent serve-lane distributed runtime: the master shard (layers
/// `[0, split)`) stays loaded for the life of the server; each generation
/// request opens a fresh worker session (the worker allocates fresh KV caches
/// per connection), runs the same per-step wire protocol as [`run_master`],
/// and closes the session. Greedy semantics (stop set, cumulative streaming
/// decode) mirror [`Gemma4Runtime::generate_greedy_streaming`] exactly, so
/// distributed serve output stays token-comparable to single-node serve.
///
/// Requests are serialized by the worker (it serves one session at a time);
/// concurrent requests queue on the worker's accept backlog.
pub struct Gemma4DistributedRuntime {
    runtime: Gemma4Runtime,
    worker_addr: String,
    handshake: Gemma4Handshake,
}

impl Gemma4DistributedRuntime {
    /// Load the master shard and validate the worker handshake once, so a
    /// misconfigured pair fails at load time rather than on the first request.
    /// The probe session is closed immediately; each request reconnects.
    pub fn connect(model: &Path, worker_addr: &str, split: usize) -> Result<Self> {
        let runtime = Gemma4Runtime::load_layer_range(model, Some(0..split))?;
        let handshake = Gemma4Handshake {
            wire_version: GEMMA4_WIRE_VERSION,
            block_count: runtime.block_count() as u32,
            hidden: runtime.hidden_size() as u32,
            worker_first_layer: split as u32,
            worker_last_layer: runtime.block_count() as u32,
            model_file_len: model_file_len(model)?,
            return_logits: false,
        };
        drop(Gemma4WorkerClient::connect(worker_addr, &handshake)?);
        Ok(Self {
            runtime,
            worker_addr: worker_addr.to_string(),
            handshake,
        })
    }

    pub fn tokenizer(&self) -> &crate::tokenizer::Tokenizer {
        self.runtime.tokenizer()
    }

    pub fn worker_addr(&self) -> &str {
        &self.worker_addr
    }

    pub fn split(&self) -> usize {
        self.handshake.worker_first_layer as usize
    }

    pub fn generate_greedy(&self, prompt: &str, max_new: usize) -> Result<(String, Vec<u32>)> {
        self.generate_greedy_streaming(prompt, max_new, |_| {})
    }

    /// Greedy decode over the wire with the same incremental-delta contract as
    /// [`Gemma4Runtime::generate_greedy_streaming`]: the delta is the
    /// newly-appended suffix of the cumulative decode (SentencePiece-safe).
    pub fn generate_greedy_streaming<F: FnMut(&str)>(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_delta: F,
    ) -> Result<(String, Vec<u32>)> {
        let mut client = Gemma4WorkerClient::connect(&self.worker_addr, &self.handshake)?;
        let prompt_tokens = self.runtime.tokenizer().encode(prompt, true, true)?;
        let stop = self.runtime.stop_token_ids();
        let (mut kc, mut vc) = self.runtime.empty_kv_caches();

        let feed = |token: u32,
                    pos: usize,
                    kc: &mut crate::gemma4_runtime::Gemma4KvCache,
                    vc: &mut crate::gemma4_runtime::Gemma4KvCache,
                    client: &mut Gemma4WorkerClient|
         -> Result<u32> {
            let h = match self.runtime.step_range(token, pos, None, kc, vc)? {
                Gemma4StepOutput::Hidden(h) => h,
                Gemma4StepOutput::Logits(_) => {
                    return Err(BackendError::InvalidModelMetadata(
                        "gemma4 master unexpectedly owns the full model; use single-node".into(),
                    ))
                }
            };
            Ok(client.step(token, pos, &h)?.next_token)
        };

        let mut last_next = 0u32;
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            last_next = feed(tok, pos, &mut kc, &mut vc, &mut client)?;
        }

        let mut generated = Vec::new();
        let mut emitted = String::new();
        for pos in prompt_tokens.len()..prompt_tokens.len() + max_new {
            if stop.contains(&last_next) {
                break;
            }
            generated.push(last_next);
            let full = self.runtime.tokenizer().decode(&generated, true)?;
            if let Some(delta) = full.strip_prefix(&emitted) {
                if !delta.is_empty() {
                    on_delta(delta);
                }
            }
            emitted = full;
            last_next = feed(last_next, pos, &mut kc, &mut vc, &mut client)?;
        }
        Ok((emitted, generated))
    }
}

// Gated as a whole, not per-test: the only test here is platform-specific, so
// on other targets the module would be empty and `use super::*` an unused
// import — a hard error under this repo's `-D warnings`.
#[cfg(all(test, any(target_os = "macos", target_os = "linux")))]
mod tests {
    use super::*;

    /// The long [`FIRST_STEP_READ_TIMEOUT`] is only defensible because keepalive
    /// bounds a dead peer. `setsockopt` reports failure only through a return
    /// code this code deliberately ignores, so assert the options actually
    /// landed — silently unarmed keepalive would leave a hung dial blocking a
    /// serve request for the full ten minutes, which is the exact regression
    /// this pairing exists to prevent.
    #[test]
    fn keepalive_is_actually_armed_on_the_socket() {
        use std::net::TcpListener;
        use std::os::unix::io::AsRawFd;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let stream = TcpStream::connect(listener.local_addr().expect("addr")).expect("connect");
        arm_keepalive(&stream);

        let get = |level: libc::c_int, name: libc::c_int| -> libc::c_int {
            let mut value: libc::c_int = 0;
            let mut len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
            // SAFETY: `value`/`len` match the c_int the options return.
            let rc = unsafe {
                libc::getsockopt(
                    stream.as_raw_fd(),
                    level,
                    name,
                    std::ptr::addr_of_mut!(value).cast::<libc::c_void>(),
                    &mut len,
                )
            };
            assert_eq!(rc, 0, "getsockopt({level}, {name}) failed");
            value
        };

        assert_ne!(
            get(libc::SOL_SOCKET, libc::SO_KEEPALIVE),
            0,
            "SO_KEEPALIVE never armed: a black-holed worker would hang the first \
             step for FIRST_STEP_READ_TIMEOUT"
        );
        #[cfg(target_os = "macos")]
        assert_eq!(get(libc::IPPROTO_TCP, libc::TCP_KEEPALIVE), 15, "idle secs");
        #[cfg(target_os = "linux")]
        assert_eq!(get(libc::IPPROTO_TCP, libc::TCP_KEEPIDLE), 15, "idle secs");
        assert_eq!(get(libc::IPPROTO_TCP, libc::TCP_KEEPINTVL), 5, "probe gap");
        assert_eq!(get(libc::IPPROTO_TCP, libc::TCP_KEEPCNT), 4, "probe count");
    }
}
