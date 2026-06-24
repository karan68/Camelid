//! GAIT — per-(model × machine) execution-profile selection.
//!
//! The campaign goal is a flagless, measured, persisted execution configuration
//! discovered once per (model-fingerprint × machine-fingerprint) pair and reused
//! instantly forever after. This module is the **spine skeleton**: the two
//! fingerprints, their combined key, a fail-closed on-disk store of
//! `camelid.gait-receipt/v1` records, and a selector consulted at the execution
//! planner's decision site.
//!
//! This first slice is deliberately **parity-neutral**. The selector only runs
//! when the `CAMELID_GAIT` bring-up gate is set, and with an empty store (the
//! only state this slice can produce) it returns `None`, so the planner falls
//! through to the existing `requested_profile()` / `Auto` path unchanged. The
//! richer per-stage kernel/chunk configuration (the `MANAGED_ENV_KEYS` surface)
//! is **not** wired here — that is a later lane. Nothing in this module mutates
//! the environment or alters any decode/math path.
//!
//! The fingerprints intentionally hash *geometry*, not weights: two checkpoints
//! of the same shape and quantization share a gait, so a fine-tune reuses its
//! base model's profile.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::execution_plan::ExecutionProfile;
use crate::gguf::GgufFile;
use crate::receipt::{canonical_json, sha256_hex};

/// Schema identifier stamped into every v1 gait receipt. Mirrors the
/// `camelid.parity-receipt/v1` family so receipts are cited by fingerprint and
/// trivially checked for tampering.
pub const GAIT_RECEIPT_SCHEMA_V1: &str = "camelid.gait-receipt/v1";

/// Environment gate for the GAIT selector. Bring-up scaffold only: the end state
/// of the campaign has no user flag. When unset (the default), the planner is
/// byte-identical to today.
pub const GAIT_GATE_ENV: &str = "CAMELID_GAIT";

/// True when the GAIT selector is enabled for this process. Recognizes the usual
/// truthy spellings; anything else (including unset) is off.
pub fn gait_enabled() -> bool {
    matches!(
        std::env::var(GAIT_GATE_ENV).ok().as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

/// The inference-shaping geometry of a model, derived from GGUF metadata. Two
/// models with identical shape + quantization hash identically.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSig {
    pub arch: Option<String>,
    pub n_layers: Option<u32>,
    pub n_embd: Option<u32>,
    pub n_heads: Option<u32>,
    pub n_kv_heads: Option<u32>,
    pub head_dim: Option<u32>,
    pub n_ff: Option<u32>,
    pub vocab_size: Option<u32>,
    pub rope_dim: Option<u32>,
    pub sliding_window: Option<u32>,
    pub qk_norm: bool,
    /// Per-stage-class set of quantization types present (e.g. `attn_q -> {Q8_0}`,
    /// `ffn_down -> {Q4K}`). Ordered containers keep serialization deterministic.
    pub quant_classes: BTreeMap<String, BTreeSet<String>>,
}

impl ModelSig {
    /// Derive the signature directly from GGUF metadata + tensor descriptors.
    ///
    /// Reads the same architecture-prefixed keys the model loader uses
    /// (`{arch}.block_count`, `{arch}.attention.head_count`, …) but stays
    /// fail-closed: every field is optional and a missing/odd value is simply
    /// `None` rather than an error, so this never fails for an unsupported or
    /// partially-described model.
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let arch = gguf.architecture().map(|s| s.to_string());
        let mu = |suffix: &str| -> Option<u32> {
            arch.as_deref()
                .and_then(|a| gguf.metadata_u32(&format!("{a}.{suffix}")))
        };

        let mut quant_classes: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut qk_norm = false;
        let mut token_embd_vocab: Option<u32> = None;
        for tensor in &gguf.tensors {
            let class = tensor_class(&tensor.name);
            quant_classes
                .entry(class.to_string())
                .or_default()
                .insert(format!("{:?}", tensor.tensor_type));
            let lower = tensor.name.to_ascii_lowercase();
            if lower.contains("attn_q_norm") || lower.contains("attn_k_norm") {
                qk_norm = true;
            }
            if class == "token_embd" {
                // token_embd.weight is [n_embd, vocab]; the larger dim is vocab.
                if let Some(max) = tensor.dimensions.iter().copied().max() {
                    token_embd_vocab = u32::try_from(max).ok();
                }
            }
        }

        // Resolve every metadata-derived field before moving `arch` into the
        // struct, so the `mu` closure's borrow of `arch` has ended.
        let n_layers = mu("block_count");
        let n_embd = mu("embedding_length");
        let n_heads = mu("attention.head_count");
        let n_kv_heads = mu("attention.head_count_kv");
        let head_dim = mu("attention.key_length");
        let n_ff = mu("feed_forward_length");
        let vocab_size = mu("vocab_size").or(token_embd_vocab);
        let rope_dim = mu("rope.dimension_count");
        let sliding_window = mu("attention.sliding_window");

        ModelSig {
            arch,
            n_layers,
            n_embd,
            n_heads,
            n_kv_heads,
            head_dim,
            n_ff,
            vocab_size,
            rope_dim,
            sliding_window,
            qk_norm,
            quant_classes,
        }
    }

    /// Lowercase-hex SHA-256 of the canonical serialization. Stable across runs
    /// for identical inputs and independent of field declaration order.
    pub fn digest(&self) -> String {
        let value = serde_json::to_value(self).expect("ModelSig serializes to JSON");
        sha256_hex(canonical_json(&value).as_bytes())
    }
}

/// Classify a tensor by its inference stage from its GGUF name. Coarse but
/// deterministic; norm tensors are matched first so QK-norm weights do not fall
/// into the attention-projection buckets.
fn tensor_class(name: &str) -> &'static str {
    let n = name.to_ascii_lowercase();
    if n.contains("norm") {
        "norm"
    } else if n.contains("attn_qkv") {
        "attn_qkv"
    } else if n.contains("attn_q") {
        "attn_q"
    } else if n.contains("attn_k") {
        "attn_k"
    } else if n.contains("attn_v") {
        "attn_v"
    } else if n.contains("attn_output") || n.contains("attn_out") {
        "attn_output"
    } else if n.contains("ffn_gate") {
        "ffn_gate"
    } else if n.contains("ffn_up") {
        "ffn_up"
    } else if n.contains("ffn_down") {
        "ffn_down"
    } else if n.contains("token_embd") || n.contains("tok_embeddings") {
        "token_embd"
    } else if n.contains("lm_head") || n.contains("output.weight") {
        "output"
    } else {
        "other"
    }
}

/// The inference-relevant fingerprint of the host machine.
///
/// Captures the facts that change which gait is fastest: precise CPU ISA, the
/// physical/SMT topology, and cache sizes. Read-only detection — it shapes the
/// fingerprint, never decode behavior.
///
/// Three campaign inputs are deliberately still absent and tracked for later
/// lanes: per-core **EfficiencyClass** (P/E split — needs the variable-length
/// `GetLogicalProcessorInformationEx` walk, Lane D), **measured STREAM
/// bandwidth / DRAM latency** (Lane B calibration), and the **Windows power-plan
/// GUID** (Lane F drift detection). Their omission keeps this slice read-only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineSig {
    pub os: String,
    pub arch: String,
    pub logical_cores: usize,
    /// Physical core count (SMT siblings collapsed). `None` if the query failed.
    pub physical_cores: Option<usize>,
    /// True when any physical core exposes SMT siblings.
    pub smt: Option<bool>,
    /// Distinct per-core EfficiencyClass values (sorted). On a hybrid CPU the
    /// higher class is the P-core; a single value means a uniform topology.
    /// Empty when the query is unavailable. This is the input to the eventual
    /// P/E role partition.
    pub efficiency_classes: Vec<u8>,
    /// True when more than one EfficiencyClass is present (a hybrid CPU). `None`
    /// when efficiency classes could not be read.
    pub hybrid: Option<bool>,
    /// Sorted list of detected CPU ISA tokens (e.g. `avx2`, `avx512vnni`; `neon`
    /// on aarch64). Precise where the coarse `SimdCaps` was not.
    pub isa: Vec<String>,
    /// Per-core L1 data cache size in bytes (representative max), if known.
    pub l1d_bytes: Option<u32>,
    pub l2_bytes: Option<u32>,
    /// Shared L3 cache size in bytes, if known.
    pub l3_bytes: Option<u32>,
}

impl MachineSig {
    /// Probe the host cheaply (no CUDA context, unlike `HardwareProfile::detect`).
    pub fn detect() -> Self {
        let topo = detect_topology();
        let efficiency_classes = detect_efficiency_classes();
        let hybrid = if efficiency_classes.is_empty() {
            None
        } else {
            Some(efficiency_classes.len() > 1)
        };
        MachineSig {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            logical_cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            physical_cores: topo.physical_cores,
            smt: topo.smt,
            efficiency_classes,
            hybrid,
            isa: detect_isa(),
            l1d_bytes: topo.l1d_bytes,
            l2_bytes: topo.l2_bytes,
            l3_bytes: topo.l3_bytes,
        }
    }

    /// Build from an already-detected [`crate::capability::HardwareProfile`],
    /// reusing its coarse SIMD probe. Topology/cache fields are left `None` (the
    /// profile does not carry them); call [`MachineSig::detect`] for the full
    /// fingerprint.
    pub fn from_hardware(profile: &crate::capability::HardwareProfile) -> Self {
        let mut isa = Vec::new();
        if profile.simd.avx2 {
            isa.push("avx2".to_string());
        }
        if profile.simd.fma {
            isa.push("fma".to_string());
        }
        if profile.simd.avx512f {
            isa.push("avx512f".to_string());
        }
        if profile.simd.neon {
            isa.push("neon".to_string());
        }
        isa.sort();
        MachineSig {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            logical_cores: profile.cpu_logical_cores,
            physical_cores: None,
            smt: None,
            efficiency_classes: Vec::new(),
            hybrid: None,
            isa,
            l1d_bytes: None,
            l2_bytes: None,
            l3_bytes: None,
        }
    }

    /// Lowercase-hex SHA-256 of the canonical serialization.
    pub fn digest(&self) -> String {
        let value = serde_json::to_value(self).expect("MachineSig serializes to JSON");
        sha256_hex(canonical_json(&value).as_bytes())
    }
}

/// Precise CPU ISA tokens, runtime-detected. On x86-64 this resolves the
/// AVX-512 sub-features the coarse `SimdCaps { avx512f }` could not distinguish.
#[cfg(target_arch = "x86_64")]
fn detect_isa() -> Vec<String> {
    // `is_x86_feature_detected!` requires a direct string literal — it cannot be
    // driven by a macro metavariable, so each probe is spelled out.
    let mut isa = Vec::new();
    if std::is_x86_feature_detected!("avx2") {
        isa.push("avx2".to_string());
    }
    if std::is_x86_feature_detected!("fma") {
        isa.push("fma".to_string());
    }
    if std::is_x86_feature_detected!("avx512f") {
        isa.push("avx512f".to_string());
    }
    if std::is_x86_feature_detected!("avx512bw") {
        isa.push("avx512bw".to_string());
    }
    if std::is_x86_feature_detected!("avx512vl") {
        isa.push("avx512vl".to_string());
    }
    if std::is_x86_feature_detected!("avx512dq") {
        isa.push("avx512dq".to_string());
    }
    if std::is_x86_feature_detected!("avx512cd") {
        isa.push("avx512cd".to_string());
    }
    if std::is_x86_feature_detected!("avx512vbmi") {
        isa.push("avx512vbmi".to_string());
    }
    if std::is_x86_feature_detected!("avx512vnni") {
        isa.push("avx512vnni".to_string());
    }
    if std::is_x86_feature_detected!("avx512bf16") {
        isa.push("avx512bf16".to_string());
    }
    isa.sort();
    isa
}

#[cfg(target_arch = "aarch64")]
fn detect_isa() -> Vec<String> {
    let mut isa = Vec::new();
    if std::arch::is_aarch64_feature_detected!("neon") {
        isa.push("neon".to_string());
    }
    isa
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_isa() -> Vec<String> {
    Vec::new()
}

/// Physical topology + cache, collapsed from the OS processor map.
#[derive(Default)]
struct Topology {
    physical_cores: Option<usize>,
    smt: Option<bool>,
    l1d_bytes: Option<u32>,
    l2_bytes: Option<u32>,
    l3_bytes: Option<u32>,
}

/// Windows x86-64: walk the fixed-size `GetLogicalProcessorInformation` records
/// (the same proven API the Rayon sizing uses) for physical cores, SMT, and
/// L1d/L2/L3 cache sizes. Best-effort: any failure leaves the field `None`.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn detect_topology() -> Topology {
    use windows_sys::Win32::System::SystemInformation::{
        CacheData, CacheUnified, GetLogicalProcessorInformation,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION,
    };
    const RELATION_PROCESSOR_CORE: i32 = 0;
    const RELATION_CACHE: i32 = 2;
    const LTP_PC_SMT: u8 = 0x1;

    let mut topo = Topology::default();
    // SAFETY: standard two-call sizing pattern; union fields are read only for
    // the record relationship that defines them.
    unsafe {
        let mut len: u32 = 0;
        GetLogicalProcessorInformation(std::ptr::null_mut(), &mut len);
        if len == 0 {
            return topo;
        }
        let count = len as usize / std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>();
        if count == 0 {
            return topo;
        }
        let mut buf: Vec<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> = Vec::with_capacity(count);
        if GetLogicalProcessorInformation(buf.as_mut_ptr(), &mut len) == 0 {
            return topo;
        }
        buf.set_len(count);

        let mut physical = 0usize;
        let mut smt = false;
        let (mut l1d, mut l2, mut l3): (u32, u32, u32) = (0, 0, 0);
        for info in &buf {
            if info.Relationship == RELATION_PROCESSOR_CORE {
                physical += 1;
                if info.Anonymous.ProcessorCore.Flags & LTP_PC_SMT != 0 {
                    smt = true;
                }
            } else if info.Relationship == RELATION_CACHE {
                let cache = info.Anonymous.Cache;
                match cache.Level {
                    1 if cache.Type == CacheData || cache.Type == CacheUnified => {
                        l1d = l1d.max(cache.Size)
                    }
                    2 => l2 = l2.max(cache.Size),
                    3 => l3 = l3.max(cache.Size),
                    _ => {}
                }
            }
        }
        if physical > 0 {
            topo.physical_cores = Some(physical);
            topo.smt = Some(smt);
        }
        if l1d > 0 {
            topo.l1d_bytes = Some(l1d);
        }
        if l2 > 0 {
            topo.l2_bytes = Some(l2);
        }
        if l3 > 0 {
            topo.l3_bytes = Some(l3);
        }
    }
    topo
}

#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
fn detect_topology() -> Topology {
    Topology::default()
}

/// Windows x86-64: distinct per-core EfficiencyClass values (sorted), via the
/// variable-length `GetLogicalProcessorInformationEx` records walked by their
/// `Size` field. A hybrid CPU reports more than one class (the higher is the
/// P-core); a uniform CPU reports one. Best-effort: returns empty on any failure.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn detect_efficiency_classes() -> Vec<u8> {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationProcessorCore,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    let mut classes = BTreeSet::new();
    // SAFETY: standard two-call sizing; records are walked by their self-reported
    // `Size`, and union/field reads use `read_unaligned` since the byte buffer is
    // only 1-aligned. Only RelationProcessorCore records were requested.
    unsafe {
        let mut len: u32 = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, std::ptr::null_mut(), &mut len);
        if len == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; len as usize];
        let ok = GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            buf.as_mut_ptr() as *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
            &mut len,
        );
        if ok == 0 {
            return Vec::new();
        }
        let total = len as usize;
        let mut offset = 0usize;
        // Each record begins with Relationship (u32) + Size (u32).
        while offset + 8 <= total {
            let rec = buf.as_ptr().add(offset) as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
            let size = std::ptr::read_unaligned(std::ptr::addr_of!((*rec).Size)) as usize;
            if size == 0 || offset + size > total {
                break;
            }
            let relationship = std::ptr::read_unaligned(std::ptr::addr_of!((*rec).Relationship));
            if relationship == RelationProcessorCore {
                let eff = std::ptr::read_unaligned(std::ptr::addr_of!(
                    (*rec).Anonymous.Processor.EfficiencyClass
                ));
                classes.insert(eff);
            }
            offset += size;
        }
    }
    classes.into_iter().collect()
}

#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
fn detect_efficiency_classes() -> Vec<u8> {
    Vec::new()
}

/// The flagless selection key: `H(model_sig):H(machine_sig)`. No flags
/// participate.
pub fn gait_key(model: &ModelSig, machine: &MachineSig) -> String {
    format!("{}:{}", model.digest(), machine.digest())
}

/// One persisted execution profile per `gait_key`. The skeleton records only the
/// coarse [`ExecutionProfile`]; the measured per-stage kernel struct is added by
/// a later lane. `receipt_id` is the SHA-256 of the canonical body (every field
/// except itself), so the record is self-verifying.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaitReceipt {
    pub schema: String,
    pub receipt_id: String,
    pub gait_key: String,
    pub model_sig: ModelSig,
    pub machine_sig: MachineSig,
    pub recorded_profile: ExecutionProfile,
    /// Measured memory characteristics at calibration time — the roofline
    /// denominator and resonance target. Recorded in the body (so it is bound
    /// into `receipt_id`) but **deliberately not part of `gait_key`**: a noisy
    /// measured float would make the cache lookup miss every run. Absent for
    /// receipts written before a measurement was taken.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryMeasurement>,
}

impl GaitReceipt {
    /// Construct and seal a receipt for the given fingerprints + chosen profile.
    pub fn new(model_sig: ModelSig, machine_sig: MachineSig, recorded_profile: ExecutionProfile) -> Self {
        let gait_key = gait_key(&model_sig, &machine_sig);
        let mut receipt = GaitReceipt {
            schema: GAIT_RECEIPT_SCHEMA_V1.to_string(),
            receipt_id: String::new(),
            gait_key,
            model_sig,
            machine_sig,
            recorded_profile,
            memory: None,
        };
        receipt.seal();
        receipt
    }

    /// Attach a memory measurement and re-seal (the measurement is part of the
    /// canonical body).
    pub fn with_memory(mut self, memory: MemoryMeasurement) -> Self {
        self.memory = Some(memory);
        self.seal();
        self
    }

    /// Recompute the digest the `receipt_id` field should hold.
    pub fn compute_receipt_id(&self) -> String {
        let mut value = serde_json::to_value(self).expect("GaitReceipt serializes to JSON");
        if let serde_json::Value::Object(map) = &mut value {
            map.remove("receipt_id");
        }
        sha256_hex(canonical_json(&value).as_bytes())
    }

    /// Populate `receipt_id` from the canonical body. Call after every other
    /// field is final.
    pub fn seal(&mut self) {
        self.receipt_id = self.compute_receipt_id();
    }

    /// True when the stored `receipt_id` matches the recomputed digest.
    pub fn verify_self_digest(&self) -> bool {
        self.compute_receipt_id() == self.receipt_id
    }
}

/// Resolve the gait store root. Windows: `%LOCALAPPDATA%\Camelid\gait`. Other
/// platforms get an XDG-ish / temp fallback so the module compiles and is inert
/// everywhere. Returns `None` only if no base directory can be determined.
pub fn gait_dir() -> Option<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("XDG_DATA_HOME").map(PathBuf::from))
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share"))
        })
        .unwrap_or_else(std::env::temp_dir);
    Some(base.join("Camelid").join("gait"))
}

/// File name for a key. The key contains `:` (invalid on Windows), so it is
/// sanitized; the full key is also recorded inside the receipt and re-checked on
/// load.
fn key_filename(key: &str) -> String {
    format!("{}.gait.json", key.replace(':', "_"))
}

/// Persist a receipt under `dir`. Fail-closed: returns the error to the caller,
/// who logs and degrades rather than propagating.
pub fn store_in(dir: &Path, receipt: &GaitReceipt) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(key_filename(&receipt.gait_key));
    let json = serde_json::to_string_pretty(receipt)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&path, json)?;
    Ok(path)
}

/// Load a receipt for `key` from `dir`. Returns `None` on any failure — missing
/// file, parse error, schema mismatch, digest mismatch, or key mismatch — so a
/// corrupt or stale store can never alter behavior beyond "no cached profile".
pub fn load_from(dir: &Path, key: &str) -> Option<GaitReceipt> {
    let path = dir.join(key_filename(key));
    let text = std::fs::read_to_string(&path).ok()?;
    let receipt: GaitReceipt = serde_json::from_str(&text).ok()?;
    if receipt.schema != GAIT_RECEIPT_SCHEMA_V1 {
        return None;
    }
    if receipt.gait_key != key {
        return None;
    }
    if !receipt.verify_self_digest() {
        return None;
    }
    Some(receipt)
}

/// The selector consulted at the planner's decision site.
///
/// Returns `Some((profile, reason))` only when the gate is on AND a valid cached
/// receipt exists for this model on this machine. In every other case it returns
/// `None`, and the planner falls through to its existing default. In this
/// skeleton the store is always empty, so this always returns `None` — the gate
/// being on is, today, observably identical to it being off.
pub fn maybe_select_profile(gguf: &GgufFile) -> Option<(ExecutionProfile, String)> {
    if !gait_enabled() {
        return None;
    }
    let model_sig = ModelSig::from_gguf(gguf);
    let machine_sig = MachineSig::detect();
    let key = gait_key(&model_sig, &machine_sig);
    let dir = gait_dir()?;
    let receipt = load_from(&dir, &key)?;
    Some((
        receipt.recorded_profile,
        format!("gait: applied cached profile for {key}"),
    ))
}

/// Measured memory characteristics — the roofline denominator (sustained DRAM
/// bandwidth) and the latency the resonance tuning must hide. Measured at
/// calibration, not guessed, and never folded into the gait key.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MemoryMeasurement {
    /// Sustained STREAM-Triad bandwidth in GB/s (10^9 bytes).
    pub stream_triad_gbs: f64,
    /// Dependent-load (pointer-chase) DRAM latency in nanoseconds.
    pub dram_latency_ns: f64,
}

/// Measure sustained bandwidth and idle latency. Tens of milliseconds of work;
/// run once at calibration. Buffers are sized past any client LLC so the kernels
/// hit DRAM rather than cache.
///
/// Single-threaded probe. The bandwidth figure is only meaningful in an
/// optimized build (an unoptimized build is overhead-bound, not memory-bound) —
/// which is the build that runs calibration, so this matches the roofline the
/// decode path actually sees.
pub fn measure_memory() -> MemoryMeasurement {
    MemoryMeasurement {
        stream_triad_gbs: measure_stream_triad_gbs(),
        dram_latency_ns: measure_dram_latency_ns(),
    }
}

#[inline(never)]
fn triad(a: &mut [f64], b: &[f64], c: &[f64], scalar: f64) {
    for i in 0..a.len() {
        a[i] = b[i] + scalar * c[i];
    }
}

fn measure_stream_triad_gbs() -> f64 {
    // 32 MiB per array (> a 24-30 MiB client L3); three arrays touched per pass.
    const N: usize = 4 * 1024 * 1024;
    let mut a = vec![0.0f64; N];
    let b = vec![1.0f64; N];
    let c = vec![2.0f64; N];
    let scalar = 3.0;
    triad(&mut a, &b, &c, scalar); // warm caches/TLB
    let iters = 8;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        triad(&mut a, &b, &c, scalar);
    }
    let secs = start.elapsed().as_secs_f64();
    std::hint::black_box(a[N - 1]);
    // Triad moves 24 bytes per element (read b, read c, write a).
    let bytes = 24.0 * N as f64 * iters as f64;
    if secs > 0.0 {
        bytes / secs / 1e9
    } else {
        0.0
    }
}

fn measure_dram_latency_ns() -> f64 {
    // 64 MiB index ring (> LLC). A coprime stride builds one Hamiltonian cycle,
    // so each load depends on the previous and prefetchers cannot hide latency.
    const N: usize = 16 * 1024 * 1024;
    let mut next = vec![0u32; N];
    let stride = 9973usize; // odd prime -> coprime with the power-of-two N
    let mut idx = 0usize;
    for _ in 0..N {
        let nxt = (idx + stride) % N;
        next[idx] = nxt as u32;
        idx = nxt;
    }
    let mut p = 0usize;
    for _ in 0..(N / 8) {
        p = next[p] as usize; // warm
    }
    let start = std::time::Instant::now();
    for _ in 0..N {
        p = next[p] as usize;
    }
    let secs = start.elapsed().as_secs_f64();
    std::hint::black_box(p);
    secs * 1e9 / N as f64
}

/// Row-dot parity oracles for the K-quant formats.
///
/// The calibration tournament (Lane B) disqualifies any candidate kernel whose
/// row·input dot diverges from a reference. Q8_0 already has one
/// (`Q8_0TensorBlocks::dot_row_f32`); this provides the missing Q4_K and Q6_K
/// references. Each reuses the *validated* super-block `dequantize` path and
/// accumulates in f32 — it is a reference, not a fast path, so clarity and
/// faithfulness beat speed.
pub mod oracle {
    use crate::tensor::{
        Q4KBlock, Q6KBlock, QK_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
    };

    /// Reference dot of one quantized matrix row against `input`.
    ///
    /// `row_bytes` is the contiguous quantized bytes of a single row; `input`
    /// has one value per column and its length must be a positive multiple of
    /// the 256-element super-block. Returns `Err` on a shape/length mismatch.
    fn dot_row_kquant(
        row_bytes: &[u8],
        input: &[f32],
        block_bytes: usize,
        dequant: impl Fn(&[u8], &mut [f32; QK_K_BLOCK_SIZE]),
    ) -> Result<f32, String> {
        if input.is_empty() || input.len() % QK_K_BLOCK_SIZE != 0 {
            return Err(format!(
                "row-dot input width {} is not a positive multiple of {QK_K_BLOCK_SIZE}",
                input.len()
            ));
        }
        let n_blocks = input.len() / QK_K_BLOCK_SIZE;
        let need = n_blocks * block_bytes;
        if row_bytes.len() < need {
            return Err(format!(
                "row-dot needs {need} quantized bytes for {n_blocks} super-blocks, got {}",
                row_bytes.len()
            ));
        }
        let mut decoded = [0.0f32; QK_K_BLOCK_SIZE];
        let mut sum = 0.0f32;
        for block in 0..n_blocks {
            let chunk = &row_bytes[block * block_bytes..(block + 1) * block_bytes];
            dequant(chunk, &mut decoded);
            let base = block * QK_K_BLOCK_SIZE;
            for (j, &weight) in decoded.iter().enumerate() {
                sum += weight * input[base + j];
            }
        }
        Ok(sum)
    }

    /// Q4_K row·input reference dot.
    pub fn dot_row_q4k(row_bytes: &[u8], input: &[f32]) -> Result<f32, String> {
        dot_row_kquant(row_bytes, input, Q4_K_BLOCK_BYTES, |chunk, out| {
            let bytes: &[u8; Q4_K_BLOCK_BYTES] =
                chunk.try_into().expect("chunk sized to Q4_K_BLOCK_BYTES");
            Q4KBlock::from_bytes(bytes).dequantize(out);
        })
    }

    /// Q6_K row·input reference dot.
    pub fn dot_row_q6k(row_bytes: &[u8], input: &[f32]) -> Result<f32, String> {
        dot_row_kquant(row_bytes, input, Q6_K_BLOCK_BYTES, |chunk, out| {
            let bytes: &[u8; Q6_K_BLOCK_BYTES] =
                chunk.try_into().expect("chunk sized to Q6_K_BLOCK_BYTES");
            Q6KBlock::from_bytes(bytes).dequantize(out);
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // A deterministic, finite super-block: scale = 1.0 (f16 0x3C00), min = 0,
        // with varied scale/value bytes. Any byte pattern decodes without panic;
        // finite scales keep the result NaN-free so equality checks are valid.
        fn synth_block(n: usize, seed: u8) -> Vec<u8> {
            let mut b = vec![0u8; n];
            b[0] = 0x00;
            b[1] = 0x3C; // f16 1.0 scale
            b[2] = 0x00;
            b[3] = 0x3C; // f16 1.0 min
            for (i, byte) in b.iter_mut().enumerate().skip(4) {
                *byte = (i as u8).wrapping_mul(31).wrapping_add(seed);
            }
            b
        }

        fn naive_reference(
            row_bytes: &[u8],
            input: &[f32],
            block_bytes: usize,
            dequant: impl Fn(&[u8], &mut [f32; QK_K_BLOCK_SIZE]),
        ) -> f32 {
            let n_blocks = input.len() / QK_K_BLOCK_SIZE;
            let mut decoded = [0.0f32; QK_K_BLOCK_SIZE];
            let mut sum = 0.0f32;
            for block in 0..n_blocks {
                let chunk = &row_bytes[block * block_bytes..(block + 1) * block_bytes];
                dequant(chunk, &mut decoded);
                for j in 0..QK_K_BLOCK_SIZE {
                    sum += decoded[j] * input[block * QK_K_BLOCK_SIZE + j];
                }
            }
            sum
        }

        #[test]
        fn q4k_oracle_matches_direct_dequant_two_blocks() {
            let mut bytes = synth_block(Q4_K_BLOCK_BYTES, 7);
            bytes.extend(synth_block(Q4_K_BLOCK_BYTES, 19));
            let input: Vec<f32> = (0..2 * QK_K_BLOCK_SIZE)
                .map(|i| ((i % 13) as f32) * 0.5 - 3.0)
                .collect();
            let got = dot_row_q4k(&bytes, &input).expect("q4k dot");
            let want = naive_reference(&bytes, &input, Q4_K_BLOCK_BYTES, |c, o| {
                let a: &[u8; Q4_K_BLOCK_BYTES] = c.try_into().unwrap();
                Q4KBlock::from_bytes(a).dequantize(o);
            });
            assert_eq!(got.to_bits(), want.to_bits());
            assert!(got.is_finite());
        }

        #[test]
        fn q6k_oracle_matches_direct_dequant() {
            let bytes = synth_block(Q6_K_BLOCK_BYTES, 5);
            let input: Vec<f32> = (0..QK_K_BLOCK_SIZE).map(|i| (i as f32) * 0.01).collect();
            let got = dot_row_q6k(&bytes, &input).expect("q6k dot");
            let want = naive_reference(&bytes, &input, Q6_K_BLOCK_BYTES, |c, o| {
                let a: &[u8; Q6_K_BLOCK_BYTES] = c.try_into().unwrap();
                Q6KBlock::from_bytes(a).dequantize(o);
            });
            assert_eq!(got.to_bits(), want.to_bits());
        }

        #[test]
        fn rejects_misaligned_width() {
            assert!(dot_row_q4k(&[0u8; Q4_K_BLOCK_BYTES], &[1.0; 100]).is_err());
            assert!(dot_row_q4k(&[], &[]).is_err());
        }

        #[test]
        fn rejects_short_byte_buffer() {
            // One super-block of input but zero bytes of weights.
            assert!(dot_row_q6k(&[], &[1.0; QK_K_BLOCK_SIZE]).is_err());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};

    fn descriptor(name: &str, ty: GgufTensorType, dims: Vec<u64>) -> GgufTensorDescriptor {
        GgufTensorDescriptor {
            name: name.to_string(),
            dimensions: dims,
            tensor_type: ty,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: 0,
        }
    }

    fn sample_gguf() -> GgufFile {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufMetadataValue::String("llama".to_string()),
        );
        metadata.insert("llama.block_count".to_string(), GgufMetadataValue::U32(2));
        metadata.insert(
            "llama.embedding_length".to_string(),
            GgufMetadataValue::U32(16),
        );
        metadata.insert(
            "llama.attention.head_count".to_string(),
            GgufMetadataValue::U32(4),
        );
        metadata.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufMetadataValue::U32(2),
        );
        metadata.insert(
            "llama.feed_forward_length".to_string(),
            GgufMetadataValue::U32(32),
        );
        GgufFile {
            path: PathBuf::from("sample.gguf"),
            version: 3,
            tensor_count: 3,
            metadata_count: metadata.len() as i64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors: vec![
                descriptor("token_embd.weight", GgufTensorType::Q8_0, vec![16, 128]),
                descriptor("blk.0.attn_q.weight", GgufTensorType::Q8_0, vec![16, 16]),
                descriptor("blk.0.ffn_down.weight", GgufTensorType::Q4K, vec![32, 16]),
            ],
        }
    }

    #[test]
    fn model_sig_reads_geometry_and_quant_classes() {
        let sig = ModelSig::from_gguf(&sample_gguf());
        assert_eq!(sig.arch.as_deref(), Some("llama"));
        assert_eq!(sig.n_layers, Some(2));
        assert_eq!(sig.n_embd, Some(16));
        assert_eq!(sig.n_heads, Some(4));
        assert_eq!(sig.n_kv_heads, Some(2));
        assert_eq!(sig.n_ff, Some(32));
        // vocab falls back to the larger token_embd dimension.
        assert_eq!(sig.vocab_size, Some(128));
        assert_eq!(
            sig.quant_classes.get("attn_q"),
            Some(&BTreeSet::from(["Q8_0".to_string()]))
        );
        assert_eq!(
            sig.quant_classes.get("ffn_down"),
            Some(&BTreeSet::from(["Q4K".to_string()]))
        );
    }

    #[test]
    fn digests_are_deterministic_and_sensitive() {
        let a = ModelSig::from_gguf(&sample_gguf());
        let b = ModelSig::from_gguf(&sample_gguf());
        assert_eq!(a.digest(), b.digest());
        assert_eq!(a.digest().len(), 64);

        let mut c = a.clone();
        c.n_layers = Some(99);
        assert_ne!(a.digest(), c.digest());
    }

    #[test]
    fn machine_sig_detect_is_stable() {
        let a = MachineSig::detect();
        let b = MachineSig::detect();
        assert_eq!(a.digest(), b.digest());
        assert_eq!(a.os, std::env::consts::OS);
    }

    #[test]
    fn memory_measurement_is_sane() {
        let m = measure_memory();
        assert!(
            m.stream_triad_gbs > 0.5 && m.stream_triad_gbs < 5000.0,
            "implausible bandwidth: {} GB/s",
            m.stream_triad_gbs
        );
        assert!(
            m.dram_latency_ns > 0.0 && m.dram_latency_ns < 100_000.0,
            "implausible latency: {} ns",
            m.dram_latency_ns
        );
    }

    #[test]
    fn receipt_with_memory_reseals_and_round_trips() {
        let dir = std::env::temp_dir().join(format!("camelid_gait_mem_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let base = GaitReceipt::new(
            ModelSig::from_gguf(&sample_gguf()),
            MachineSig::detect(),
            ExecutionProfile::Auto,
        );
        let with_mem = base.clone().with_memory(MemoryMeasurement {
            stream_triad_gbs: 30.0,
            dram_latency_ns: 80.0,
        });
        // Attaching a measurement changes the sealed digest but not the key.
        assert_ne!(with_mem.receipt_id, base.receipt_id);
        assert_eq!(with_mem.gait_key, base.gait_key);
        assert!(with_mem.verify_self_digest());

        store_in(&dir, &with_mem).expect("store");
        let loaded = load_from(&dir, &with_mem.gait_key).expect("load");
        assert_eq!(loaded, with_mem);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn machine_sig_topology_is_coherent() {
        let m = MachineSig::detect();
        assert!(m.logical_cores >= 1);
        if let Some(p) = m.physical_cores {
            assert!(p >= 1 && p <= m.logical_cores);
        }
        // A per-core L2 never exceeds the shared L3, when both are known.
        if let (Some(l2), Some(l3)) = (m.l2_bytes, m.l3_bytes) {
            assert!(l2 <= l3);
        }
        // `hybrid` must agree with the number of distinct efficiency classes.
        if !m.efficiency_classes.is_empty() {
            assert_eq!(m.hybrid, Some(m.efficiency_classes.len() > 1));
        }
    }

    #[test]
    fn gait_key_combines_both_digests() {
        let model = ModelSig::from_gguf(&sample_gguf());
        let machine = MachineSig::detect();
        let key = gait_key(&model, &machine);
        assert_eq!(key, format!("{}:{}", model.digest(), machine.digest()));
    }

    #[test]
    fn receipt_round_trips_through_the_store() {
        let dir = std::env::temp_dir().join(format!("camelid_gait_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let receipt = GaitReceipt::new(
            ModelSig::from_gguf(&sample_gguf()),
            MachineSig::detect(),
            ExecutionProfile::Auto,
        );
        assert!(receipt.verify_self_digest());

        let path = store_in(&dir, &receipt).expect("store");
        assert!(path.exists());

        let loaded = load_from(&dir, &receipt.gait_key).expect("load");
        assert_eq!(loaded, receipt);
        assert!(loaded.verify_self_digest());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_misses_on_empty_store() {
        let dir = std::env::temp_dir().join(format!("camelid_gait_miss_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(load_from(&dir, "deadbeef:cafef00d").is_none());
    }

    #[test]
    fn selector_is_none_when_gate_disabled() {
        // With the gate unset, the selector must not touch the store at all.
        std::env::remove_var(GAIT_GATE_ENV);
        assert!(maybe_select_profile(&sample_gguf()).is_none());
    }
}
