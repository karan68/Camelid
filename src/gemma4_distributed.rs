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

use crate::gemma4_runtime::{Gemma4GenerationOutcome, Gemma4Runtime, Gemma4StepOutput};
use crate::{BackendError, Result};

/// Wire protocol version. Bump on ANY change to the message layout.
///
/// v2 added [`BEAT_MAGIC`]. A v1 master would read a beat where it expects a
/// response and abort with "bad resp magic" — fail-closed but useless — so the
/// handshake catches the skew first and names the real cause. A mixed-version
/// pair refuses to run until both nodes are on the same build.
pub const GEMMA4_WIRE_VERSION: u32 = 2;
const HELLO_MAGIC: u32 = 0xCA4E1147;
const STEP_MAGIC: u32 = 0xCA4E5701;
const RESP_MAGIC: u32 = 0xCA4E5702;
/// Worker -> master liveness frame, valid ONLY between a step being read and its
/// response being written. Fixed 28 bytes, no length prefix.
const BEAT_MAGIC: u32 = 0xCA4E5703;

/// One heartbeat frame: magic | seq | elapsed_ms | fnv1a(seq ‖ elapsed_ms).
///
/// Built as one stack array and written with a single `write_all` so a beat is
/// one syscall-visible unit: it can only ever be partially written by a socket
/// error, never by interleaving, and that case is treated as a dead session
/// rather than retried.
///
/// `elapsed_ms` is DIAGNOSTIC ONLY. The master never makes a control decision
/// from a peer-supplied clock — it appears in log text and nowhere else.
fn encode_beat(seq: u64, elapsed_ms: u64) -> [u8; 28] {
    let mut frame = [0u8; 28];
    frame[0..4].copy_from_slice(&BEAT_MAGIC.to_le_bytes());
    frame[4..12].copy_from_slice(&seq.to_le_bytes());
    frame[12..20].copy_from_slice(&elapsed_ms.to_le_bytes());
    let checksum = fnv1a(&frame[4..20]);
    frame[20..28].copy_from_slice(&checksum.to_le_bytes());
    frame
}

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

/// The worker's socket writer plus the two flags the beat thread coordinates on.
///
/// Generic over `W` so the beat loop can be unit-tested against a `Vec<u8>`
/// without a socket; `serve_session` instantiates it at `BufWriter<TcpStream>`.
struct SessionOut<W: Write> {
    writer: W,
    /// True except while a `step_range` call is in flight. The beat loop's wait
    /// predicate reads it, so the beat thread cannot begin a write once it is set.
    beats_closed: bool,
    /// First write failure seen by either thread. A failed write may have left a
    /// partial frame on the wire, so the session is over — never cleared.
    failed: Option<String>,
}

/// Emit a beat every `interval` until the step finishes or a write fails.
///
/// Teardown latency is a scheduler hop rather than up to a full interval, which
/// is the whole reason this is a condvar rather than `sleep` + an `AtomicBool`.
/// `wait_timeout_while` re-checks the predicate under the lock, so there is no
/// lost wakeup and `seq` stays gap-free.
fn beat_loop<W: Write>(
    out: &std::sync::Mutex<SessionOut<W>>,
    cv: &std::sync::Condvar,
    started: std::time::Instant,
    interval: std::time::Duration,
) {
    let mut seq: u64 = 0;
    let Ok(mut guard) = out.lock() else {
        return; // compute thread panicked while holding the lock; nothing to do
    };
    loop {
        let Ok((next, timeout)) =
            cv.wait_timeout_while(guard, interval, |s: &mut SessionOut<W>| {
                !s.beats_closed && s.failed.is_none()
            })
        else {
            return;
        };
        guard = next;
        if !timeout.timed_out() {
            return; // predicate went false: the step is finishing
        }
        seq += 1;
        let frame = encode_beat(seq, started.elapsed().as_millis() as u64);
        // Hold the guard across write_all + flush so a beat is never truncated
        // by teardown, which must take this same lock to set `beats_closed`.
        if let Err(e) = guard
            .writer
            .write_all(&frame)
            .and_then(|()| guard.writer.flush())
        {
            guard.failed = Some(format!("heartbeat write at beat {seq}: {e}"));
            return;
        }
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
    // Bound writes too. A master that stops reading (but whose host is alive, so
    // keepalive stays quiet) would otherwise block a beat or response write
    // forever, holding the serial accept loop with it.
    stream.set_write_timeout(Some(WORKER_WRITE_TIMEOUT)).ok();
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
        // Reject with the exact mismatch, then close. Version skew is called out
        // first because it is now a real operational case — v2 added the
        // heartbeat frame, so updating one node of a two-Mac pair and not the
        // other lands here. The "handshake mismatch" substring is asserted by
        // tests/gemma4_distributed_parity.rs; keep it.
        let skew = if hello.wire_version == expected.wire_version {
            String::new()
        } else {
            format!(
                "wire version mismatch (master v{}, worker v{}): update both nodes to the same \
                 camelid build — ",
                hello.wire_version, expected.wire_version
            )
        };
        let msg =
            format!("{skew}handshake mismatch: master sent {hello:?}, worker expects {expected:?}");
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

    // From here on the writer is shared with the beat thread. The handshake
    // above wrote through it directly, before any beat thread can exist — no
    // new byte lives on the reject path, which is what keeps a v1 master's
    // reject-read working against a v2 worker.
    let out = std::sync::Mutex::new(SessionOut {
        writer,
        beats_closed: true,
        failed: None,
    });
    let cv = std::sync::Condvar::new();
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

        // Arm the beat window. Any write failure from a previous step means the
        // stream may carry a partial frame, so the session is already over.
        {
            let mut guard = out.lock().map_err(|_| poisoned())?;
            if let Some(e) = guard.failed.take() {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "gemma4 distributed: {e}"
                )));
            }
            guard.beats_closed = false;
        }
        let started = std::time::Instant::now();
        let stepped = std::thread::scope(|scope| {
            // A drop guard, NOT a plain assignment after the call: if step_range
            // panics, the unwind must still tell the beat thread to stop, or
            // `scope`'s implicit join deadlocks forever against a thread asleep
            // on the condvar. Deleting this in the name of simplification turns
            // a panicking step into a hung worker process. It is load-bearing.
            struct StopBeats<'a, W: Write>(
                &'a std::sync::Mutex<SessionOut<W>>,
                &'a std::sync::Condvar,
            );
            impl<W: Write> Drop for StopBeats<'_, W> {
                fn drop(&mut self) {
                    if let Ok(mut guard) = self.0.lock() {
                        guard.beats_closed = true;
                    }
                    self.1.notify_all();
                }
            }
            let _stop = StopBeats(&out, &cv);
            scope.spawn(|| beat_loop(&out, &cv, started, HEARTBEAT_INTERVAL));
            runtime.step_range(token, pos, Some(h), &mut kc, &mut vc)
        });
        // The beat thread is now provably joined, so nothing else can write.
        let logits = match stepped? {
            Gemma4StepOutput::Logits(logits) => logits,
            Gemma4StepOutput::Hidden(_) => {
                return Err(BackendError::InvalidModelMetadata(
                    "gemma4 worker did not own the final layer".into(),
                ))
            }
        };
        let (next, max_logit) = greedy_argmax(&logits);
        let mut guard = out.lock().map_err(|_| poisoned())?;
        if let Some(e) = guard.failed.take() {
            // Do NOT append a response to a possibly-desynced stream.
            return Err(BackendError::InvalidModelMetadata(format!(
                "gemma4 distributed: {e}"
            )));
        }
        let writer = &mut guard.writer;
        write_u32(writer, RESP_MAGIC).map_err(|e| io_err("resp", e))?;
        write_u32(writer, 0).map_err(|e| io_err("resp", e))?;
        write_u32(writer, next).map_err(|e| io_err("resp", e))?;
        writer
            .write_all(&max_logit.to_le_bytes())
            .map_err(|e| io_err("resp", e))?;
        if hello.return_logits {
            write_u32(writer, logits.len() as u32).map_err(|e| io_err("resp", e))?;
            write_f32s(writer, &logits).map_err(|e| io_err("resp logits", e))?;
            write_u64(writer, f32s_checksum(&logits)).map_err(|e| io_err("resp", e))?;
        } else {
            write_u32(writer, 0).map_err(|e| io_err("resp", e))?;
        }
        writer.flush().map_err(|e| io_err("resp flush", e))?;
    }
}

/// A poisoned session mutex means the compute thread panicked mid-step; the
/// stream state is unknown, so the session is over.
fn poisoned() -> BackendError {
    BackendError::InvalidModelMetadata(
        "gemma4 distributed: worker session lock poisoned by a panic mid-step".into(),
    )
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
#[derive(Debug)]
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
/// while a read is inside the long [`COLD_STEP_CEILING`] budget.
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
/// How often the worker emits a [`BEAT_MAGIC`] frame while a `step_range` call
/// is in flight.
///
/// The worker binds and answers the handshake before its shard is resident:
/// `load_layer_range` maps the GGUF lazily and warms it on a background thread
/// (that advisory blocks for minutes on macOS over USB, so it cannot run on the
/// load path). The whole cold-fault cost therefore lands on the first
/// `step_range` — measured at 171s for a 1.9GB tail shard off a 38MB/s USB
/// volume, and a 26B tail shard is ~13GB, so ~350s at the same rate.
///
/// A 6-minute cold start costs ~72 frames x 28 bytes ~= 2KB of wire traffic.
const HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// The master's read budget for ALL post-handshake reads.
///
/// This is the constant that makes the lane scale. It bounds SILENCE, not work:
/// every frame — heartbeat or response — re-arms it, so a worker that keeps
/// proving liveness is tolerated indefinitely while one that goes mute is caught
/// in at most four intervals. Unlike a first-step budget, it does not have to be
/// re-guessed per disk speed or per row size, which is what previously made 12B
/// and 26B unserviceable without raising a number that simultaneously widened
/// the mute-worker window.
///
/// = 4 x [`HEARTBEAT_INTERVAL`], i.e. three consecutive missed beats. The beat
/// thread sleeps on a condvar and does one 28-byte write on wake, so it is not
/// plausibly starved for 20s even by a rayon-saturated `step_range`.
const HEARTBEAT_SILENCE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

/// Wall-clock ceiling on the FIRST step of a session, checked after each beat.
///
/// A pure liveness backstop: there is still no timeout layer above this (the
/// router mounts only tower-http `trace`/`cors`), so without it a worker that
/// beats forever but never finishes would hang a serve request forever. Sized to
/// be unreachable by legitimate paging rather than tight — ~2.6x the projected
/// worst case (a 26B ~13GB tail shard at the measured 38MB/s is ~350s). If this
/// fires, the worker is broken, not slow.
const COLD_STEP_CEILING: std::time::Duration = std::time::Duration::from_secs(900);

/// Wall-clock ceiling on every step after the first response.
///
/// Not 30s: a resident shard can be evicted under memory pressure on a 16 GB
/// Mac, so a mid-generation step can legitimately go back to disk. In practice a
/// wedged step is caught by [`HEARTBEAT_SILENCE_TIMEOUT`] long before this.
const STEADY_STEP_CEILING: std::time::Duration = std::time::Duration::from_secs(300);

/// Worker-side `SO_SNDTIMEO`. Bounds a beat or response write against a master
/// that stopped reading but whose host is alive, which keepalive cannot see and
/// which would otherwise block the serial accept loop forever.
const WORKER_WRITE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// How long a worker waits for the next step before abandoning the session.
///
/// The accept loop is serial — one session at a time — so an abandoned-but-open
/// session locks out every future master. MUST be >= [`COLD_STEP_CEILING`]: the
/// gap between the handshake and the master's first step covers the MASTER's own
/// local `step_range` over layers `[0, split)`, whose shard can be just as cold
/// as the worker's. At 300s this dropped healthy 26B sessions before the first
/// token ever crossed the wire.
const WORKER_SESSION_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(900);

/// Master-side connection to a gemma4 worker (one generation session).
pub struct Gemma4WorkerClient {
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
    /// Cleared once a step response has arrived, at which point the read
    /// step is bounded by [`COLD_STEP_CEILING`] rather than [`STEADY_STEP_CEILING`].
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
    /// [`STEP_READ_TIMEOUT`], not a step ceiling: 10 attempts x
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
        // Handshake done. From here the read timeout bounds SILENCE, not work:
        // the worker heartbeats while computing, so every frame re-arms it. It
        // is armed once and never changed again — a slow worker is tolerated by
        // beating, not by widening this.
        reader
            .get_ref()
            .set_read_timeout(Some(HEARTBEAT_SILENCE_TIMEOUT))
            .ok();
        Ok(Self {
            reader,
            writer,
            cold: true,
        })
    }

    /// Describe a failed frame read, distinguishing "went silent" from "died".
    ///
    /// A timeout here means the worker stopped proving liveness, which under v2
    /// is a real fault rather than an ambiguous slow-disk case — the whole point
    /// of the heartbeat is that a busy worker keeps talking. A reset or EOF is
    /// the worker dying, and blaming silence for it would send the reader
    /// looking at disk speed instead of the worker log.
    fn silence_error(
        &self,
        e: std::io::Error,
        beats: u64,
        pos: usize,
        ceiling: std::time::Duration,
    ) -> BackendError {
        if matches!(
            e.kind(),
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
        ) {
            return BackendError::InvalidModelMetadata(format!(
                "gemma4 distributed: worker went silent for {HEARTBEAT_SILENCE_TIMEOUT:?} at pos \
                 {pos} after {beats} heartbeat(s) (step ceiling {ceiling:?}) — it stopped \
                 sending liveness frames, so it is wedged or unreachable rather than merely \
                 slow; check the worker log: {e}"
            ));
        }
        io_err("resp", e)
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

        // Consume heartbeats until the response arrives. The socket timeout
        // bounds the gap BETWEEN frames, so a worker that keeps beating is
        // tolerated for as long as the wall-clock ceiling allows, while one that
        // goes mute is caught in at most HEARTBEAT_SILENCE_TIMEOUT.
        let ceiling = if self.cold {
            COLD_STEP_CEILING
        } else {
            STEADY_STEP_CEILING
        };
        let deadline = std::time::Instant::now() + ceiling;
        let mut beats: u64 = 0;
        let magic = loop {
            let magic = read_u32(&mut self.reader)
                .map_err(|e| self.silence_error(e, beats, pos, ceiling))?;
            if magic != BEAT_MAGIC {
                break magic;
            }
            let seq = read_u64(&mut self.reader).map_err(|e| io_err("beat seq", e))?;
            let elapsed_ms = read_u64(&mut self.reader).map_err(|e| io_err("beat elapsed", e))?;
            let sent = read_u64(&mut self.reader).map_err(|e| io_err("beat checksum", e))?;
            let mut body = [0u8; 16];
            body[0..8].copy_from_slice(&seq.to_le_bytes());
            body[8..16].copy_from_slice(&elapsed_ms.to_le_bytes());
            if sent != fnv1a(&body) {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "gemma4 distributed: heartbeat checksum mismatch at pos {pos} \
                     (sent {sent:#x}) — the stream is desynced; refusing to continue"
                )));
            }
            beats += 1;
            // Exact +1. Beats are ordered on one TCP stream and the worker
            // increments `seq` only after a frame is fully written, so a gap
            // means lost or partial framing, not a slow worker. Fail closed
            // rather than attempt to resync.
            if seq != beats {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "gemma4 distributed: heartbeat sequence gap at pos {pos} \
                     (expected {beats}, got {seq}) — the stream is desynced"
                )));
            }
            if std::time::Instant::now() >= deadline {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "gemma4 distributed: worker still heartbeating after {ceiling:?} on a \
                     single step at pos {pos} (beat {seq}, worker-side elapsed {elapsed_ms}ms). \
                     It is alive but not finishing — a stuck worker, not a slow disk; check \
                     the worker log."
                )));
            }
        };
        if self.cold {
            // The worker produced a response, so its shard is resident: later
            // steps get the tighter wall-clock ceiling.
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

    pub fn generate_greedy_cancellable<C: FnMut() -> bool>(
        &self,
        prompt: &str,
        max_new: usize,
        should_cancel: C,
    ) -> Result<Gemma4GenerationOutcome> {
        self.generate_greedy_streaming_cancellable(prompt, max_new, |_| {}, should_cancel)
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

    /// Cancellation-aware distributed decode. A TCP step is indivisible; the
    /// signal is observed before the next prompt/decode step so the worker
    /// session and local shard are released together without interleaving a
    /// second request into either side.
    pub fn generate_greedy_streaming_cancellable<F: FnMut(&str), C: FnMut() -> bool>(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_delta: F,
        mut should_cancel: C,
    ) -> Result<Gemma4GenerationOutcome> {
        if should_cancel() {
            return Ok(Gemma4GenerationOutcome::Cancelled {
                generated_tokens: 0,
            });
        }
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
            if should_cancel() {
                return Ok(Gemma4GenerationOutcome::Cancelled {
                    generated_tokens: 0,
                });
            }
            last_next = feed(tok, pos, &mut kc, &mut vc, &mut client)?;
        }

        let mut generated = Vec::new();
        let mut emitted = String::new();
        for pos in prompt_tokens.len()..prompt_tokens.len() + max_new {
            if should_cancel() {
                return Ok(Gemma4GenerationOutcome::Cancelled {
                    generated_tokens: generated.len(),
                });
            }
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
            if should_cancel() {
                return Ok(Gemma4GenerationOutcome::Cancelled {
                    generated_tokens: generated.len(),
                });
            }
            last_next = feed(last_next, pos, &mut kc, &mut vc, &mut client)?;
        }
        Ok(Gemma4GenerationOutcome::Complete {
            text: emitted,
            token_ids: generated,
        })
    }
}

// Gated as a whole, not per-test: the only test here is platform-specific, so
// on other targets the module would be empty and `use super::*` an unused
// import — a hard error under this repo's `-D warnings`.
#[cfg(all(test, any(target_os = "macos", target_os = "linux")))]
mod tests {
    use super::*;

    /// The long [`COLD_STEP_CEILING`] is only defensible because keepalive
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
             step for the whole cold ceiling"
        );
        #[cfg(target_os = "macos")]
        assert_eq!(get(libc::IPPROTO_TCP, libc::TCP_KEEPALIVE), 15, "idle secs");
        #[cfg(target_os = "linux")]
        assert_eq!(get(libc::IPPROTO_TCP, libc::TCP_KEEPIDLE), 15, "idle secs");
        assert_eq!(get(libc::IPPROTO_TCP, libc::TCP_KEEPINTVL), 5, "probe gap");
        assert_eq!(get(libc::IPPROTO_TCP, libc::TCP_KEEPCNT), 4, "probe count");
    }
}

#[cfg(test)]
mod heartbeat_tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Condvar, Mutex};
    use std::time::{Duration, Instant};

    fn handshake_for(listener_peer: &mut TcpStream, accept_ok: bool) -> Gemma4Handshake {
        let h = Gemma4Handshake {
            wire_version: GEMMA4_WIRE_VERSION,
            block_count: 4,
            hidden: 8,
            worker_first_layer: 2,
            worker_last_layer: 4,
            model_file_len: 1234,
            return_logits: false,
        };
        if accept_ok {
            write_u32(listener_peer, RESP_MAGIC).unwrap();
            write_u32(listener_peer, 0).unwrap();
            listener_peer.flush().unwrap();
        }
        h
    }

    /// A beat encodes and verifies under the same checksum rule the master applies.
    #[test]
    fn beat_frame_round_trips_and_detects_corruption() {
        let frame = encode_beat(7, 4242);
        assert_eq!(
            u32::from_le_bytes(frame[0..4].try_into().unwrap()),
            BEAT_MAGIC
        );
        assert_eq!(u64::from_le_bytes(frame[4..12].try_into().unwrap()), 7);
        assert_eq!(u64::from_le_bytes(frame[12..20].try_into().unwrap()), 4242);
        assert_eq!(
            u64::from_le_bytes(frame[20..28].try_into().unwrap()),
            fnv1a(&frame[4..20]),
            "checksum must cover seq ‖ elapsed_ms exactly as the master recomputes it"
        );
        let mut corrupt = frame;
        corrupt[5] ^= 0xFF;
        assert_ne!(
            u64::from_le_bytes(corrupt[20..28].try_into().unwrap()),
            fnv1a(&corrupt[4..20]),
            "a flipped seq bit must not still satisfy the checksum"
        );
    }

    /// The beat loop emits gap-free 1-based sequence numbers and stops promptly.
    ///
    /// Gap-freeness is what lets the master use strict `seq == beats + 1` as a
    /// desync detector rather than merely a monotonicity check.
    #[test]
    fn beat_loop_emits_gap_free_sequence_then_stops_on_close() {
        let out = Mutex::new(SessionOut {
            writer: Vec::<u8>::new(),
            beats_closed: false,
            failed: None,
        });
        let cv = Condvar::new();
        let started = Instant::now();
        std::thread::scope(|scope| {
            let h = scope.spawn(|| beat_loop(&out, &cv, started, Duration::from_millis(10)));
            // Let several beats accumulate, then close the window.
            while out.lock().unwrap().writer.len() < 28 * 3 {
                std::thread::yield_now();
            }
            out.lock().unwrap().beats_closed = true;
            cv.notify_all();
            h.join().unwrap();
        });
        let buf = out.into_inner().unwrap().writer;
        assert_eq!(buf.len() % 28, 0, "only whole beat frames are ever written");
        let count = buf.len() / 28;
        assert!(count >= 3, "expected several beats, got {count}");
        for i in 0..count {
            let f = &buf[i * 28..(i + 1) * 28];
            assert_eq!(u32::from_le_bytes(f[0..4].try_into().unwrap()), BEAT_MAGIC);
            assert_eq!(
                u64::from_le_bytes(f[4..12].try_into().unwrap()),
                i as u64 + 1,
                "seq must be 1-based and gap-free"
            );
            assert_eq!(
                u64::from_le_bytes(f[20..28].try_into().unwrap()),
                fnv1a(&f[4..20])
            );
        }
    }

    /// A write failure is latched and stops the loop, so no frame is ever
    /// appended after a possibly-partial one.
    #[test]
    fn beat_loop_latches_write_failure_and_exits() {
        struct Failing;
        impl Write for Failing {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "peer gone",
                ))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let out = Mutex::new(SessionOut {
            writer: Failing,
            beats_closed: false,
            failed: None,
        });
        let cv = Condvar::new();
        beat_loop(&out, &cv, Instant::now(), Duration::from_millis(1));
        let guard = out.lock().unwrap();
        assert!(
            guard
                .failed
                .as_deref()
                .is_some_and(|m| m.contains("heartbeat write")),
            "write failure must be latched for the compute thread: {:?}",
            guard.failed
        );
    }

    /// The master must CONSUME beats and still return the real response.
    ///
    /// The end-to-end suites cannot cover this: they run warm, so steps take
    /// ~40ms and no beat is ever emitted. This drives the wire contract
    /// directly — several beats, then a response — which is the path that would
    /// silently break if the master ever stopped treating BEAT_MAGIC as a
    /// continue rather than a frame boundary.
    #[test]
    fn master_consumes_beats_then_reads_the_response() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let (mut peer, _) = listener.accept().unwrap();
            use std::io::Read as _;
            let mut hello = [0u8; 36];
            peer.read_exact(&mut hello).unwrap();
            write_u32(&mut peer, RESP_MAGIC).unwrap();
            write_u32(&mut peer, 0).unwrap();
            peer.flush().unwrap();
            // Read the step frame: magic|token|pos|h_len|h|checksum.
            let mut head = [0u8; 16];
            peer.read_exact(&mut head).unwrap();
            let h_len = u32::from_le_bytes(head[12..16].try_into().unwrap()) as usize;
            let mut body = vec![0u8; h_len * 4 + 8];
            peer.read_exact(&mut body).unwrap();
            // Three beats, then the response.
            for seq in 1..=3u64 {
                peer.write_all(&encode_beat(seq, seq * 1000)).unwrap();
                peer.flush().unwrap();
            }
            write_u32(&mut peer, RESP_MAGIC).unwrap();
            write_u32(&mut peer, 0).unwrap();
            write_u32(&mut peer, 4242).unwrap();
            peer.write_all(&1.5f32.to_le_bytes()).unwrap();
            write_u32(&mut peer, 0).unwrap();
            peer.flush().unwrap();
            std::thread::sleep(Duration::from_millis(200));
        });

        let handshake = Gemma4Handshake {
            wire_version: GEMMA4_WIRE_VERSION,
            block_count: 4,
            hidden: 8,
            worker_first_layer: 2,
            worker_last_layer: 4,
            model_file_len: 1234,
            return_logits: false,
        };
        let mut client =
            Gemma4WorkerClient::connect(&addr.to_string(), &handshake).expect("handshake ok");
        let step = client
            .step(1, 0, &[0.0f32; 8])
            .expect("beats must not break the response");
        assert_eq!(
            step.next_token, 4242,
            "response after beats must parse intact"
        );
        assert_eq!(step.max_logit, 1.5);
        let _ = worker.join();
    }

    /// A beat with a wrong sequence number is a desync and must fail closed
    /// rather than be tolerated as a slow worker.
    #[test]
    fn master_fails_closed_on_a_heartbeat_sequence_gap() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let (mut peer, _) = listener.accept().unwrap();
            use std::io::Read as _;
            let mut hello = [0u8; 36];
            peer.read_exact(&mut hello).unwrap();
            write_u32(&mut peer, RESP_MAGIC).unwrap();
            write_u32(&mut peer, 0).unwrap();
            peer.flush().unwrap();
            let mut head = [0u8; 16];
            peer.read_exact(&mut head).unwrap();
            let h_len = u32::from_le_bytes(head[12..16].try_into().unwrap()) as usize;
            let mut body = vec![0u8; h_len * 4 + 8];
            peer.read_exact(&mut body).unwrap();
            // seq jumps 1 -> 3: a frame was lost, so the stream is untrustworthy.
            peer.write_all(&encode_beat(1, 1000)).unwrap();
            peer.write_all(&encode_beat(3, 3000)).unwrap();
            peer.flush().unwrap();
            std::thread::sleep(Duration::from_millis(200));
        });

        let handshake = Gemma4Handshake {
            wire_version: GEMMA4_WIRE_VERSION,
            block_count: 4,
            hidden: 8,
            worker_first_layer: 2,
            worker_last_layer: 4,
            model_file_len: 1234,
            return_logits: false,
        };
        let mut client =
            Gemma4WorkerClient::connect(&addr.to_string(), &handshake).expect("handshake ok");
        let err = client
            .step(1, 0, &[0.0f32; 8])
            .expect_err("gap must fail closed");
        let msg = format!("{err}");
        assert!(
            msg.contains("sequence gap"),
            "must name the desync rather than resync: {msg}"
        );
        let _ = worker.join();
    }

    /// THE REGRESSION TEST for the mute-worker residual.
    ///
    /// A worker that completes the handshake and then goes silent must be caught
    /// in about one silence budget. Before v2 this cost a full 300s first-step
    /// budget per request, and keepalive could not see it because the peer's TCP
    /// stack keeps ACKing — which is exactly what this fake worker does by
    /// holding an open, idle socket.
    #[test]
    fn mute_worker_is_caught_in_one_silence_budget_not_a_step_ceiling() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let (mut peer, _) = listener.accept().unwrap();
            let mut hello = [0u8; 36];
            use std::io::Read as _;
            peer.read_exact(&mut hello).unwrap();
            let _ = handshake_for(&mut peer, true);
            // Now go mute, holding the socket open so keepalive stays quiet.
            std::thread::sleep(HEARTBEAT_SILENCE_TIMEOUT + Duration::from_secs(5));
        });

        let handshake = Gemma4Handshake {
            wire_version: GEMMA4_WIRE_VERSION,
            block_count: 4,
            hidden: 8,
            worker_first_layer: 2,
            worker_last_layer: 4,
            model_file_len: 1234,
            return_logits: false,
        };
        let mut client =
            Gemma4WorkerClient::connect(&addr.to_string(), &handshake).expect("handshake ok");
        let began = Instant::now();
        let err = client.step(1, 0, &[0.0f32; 8]).expect_err("must fail");
        let elapsed = began.elapsed();

        assert!(
            elapsed < HEARTBEAT_SILENCE_TIMEOUT + Duration::from_secs(10),
            "a mute worker must be caught in about one silence budget, took {elapsed:?}"
        );
        assert!(
            elapsed >= HEARTBEAT_SILENCE_TIMEOUT - Duration::from_secs(2),
            "must not fire before the silence budget elapses, took {elapsed:?}"
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("went silent"),
            "error must name silence, not a cold shard: {msg}"
        );
        let _ = worker.join();
    }
}
