//! Verifiable parity receipts.
//!
//! A receipt records that ONE request, replayed deterministically, produced
//! byte-identical output to the llama.cpp reference. Two rules govern this
//! module and every surface built on it:
//!
//! 1. A receipt proves one request matched the reference — it is NOT a support
//!    promotion. A green receipt does not move any row in the release ledger
//!    (`README.md` / `COMPATIBILITY.md` / `STATUS.md`), and no copy, field
//!    name, or log line may imply otherwise.
//! 2. Receipts are only meaningful for deterministic runs. A receipt for a
//!    sampled (non-greedy) run must be stamped `reproducible: false` and is
//!    never presented as verifiable; the verifier refuses to assert parity on
//!    non-reproducible receipts.
//!
//! The schema is versioned from day one (`camelid.parity-receipt/v1`). The
//! `receipt_id` is the SHA-256 of a canonical serialization of the receipt
//! body (sorted keys, no insignificant whitespace, `receipt_id` excluded), so
//! a receipt can be cited by fingerprint and trivially checked for tampering.

pub mod distributed;
pub mod verify;

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::gguf::{GgufFile, GgufMetadataValue};

/// Schema identifier stamped into every v1 receipt.
pub const RECEIPT_SCHEMA_V1: &str = "camelid.parity-receipt/v1";

/// Sentinel for "no divergence found" in `first_divergent_token_index`,
/// matching the convention of the chat-parity harness scripts.
pub const NO_DIVERGENCE: i64 = -1;

#[derive(Debug, thiserror::Error)]
pub enum ReceiptError {
    #[error("receipt serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("receipt digest mismatch: stored {stored}, computed {computed}")]
    DigestMismatch { stored: String, computed: String },

    #[error("receipt schema mismatch: expected {expected}, found {found}")]
    SchemaMismatch { expected: String, found: String },

    #[error("I/O error while reading {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
}

/// A verifiable record of one deterministic generation request.
///
/// Round-trips losslessly through `serde_json`. Field order here mirrors the
/// documented schema for readability of emitted files; the canonical digest
/// form is key-sorted and independent of declaration order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityReceipt {
    /// Always [`RECEIPT_SCHEMA_V1`] for receipts produced by this build.
    pub schema: String,
    /// SHA-256 of the canonical receipt body (every field except this one).
    pub receipt_id: String,
    /// RFC 3339 UTC timestamp of receipt creation.
    pub created_utc: String,
    pub lane: LaneIdentity,
    pub reference: ReferenceIdentity,
    pub request: ReceiptRequest,
    /// `false` for any run with nondeterministic sampling. Non-reproducible
    /// receipts are never presented as verifiable (rule 2).
    pub reproducible: bool,
    pub result: ReceiptResult,
    pub parity: ParityBlock,
    /// Deferred signing seam: an optional detached signature over
    /// `receipt_id`. Absent in v1; present-but-optional so adding it later is
    /// not a schema-breaking change. Key management is intentionally not
    /// implemented in this pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<SignatureBlock>,
}

/// Identity of the exact model lane the receipt is about. A receipt is only
/// meaningful for the exact GGUF bytes named here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaneIdentity {
    pub model_id: String,
    /// Lowercase hex SHA-256 of the exact GGUF file.
    pub gguf_sha256: String,
    pub gguf_filename: String,
    pub quantization: String,
    pub architecture: String,
    /// As reported by the loader (e.g. `llama_spm`, `gpt2_bpe`).
    pub tokenizer_kind: String,
    /// Lowercase hex SHA-256 of the tokenizer metadata/source if available.
    pub tokenizer_sha256: Option<String>,
    pub camelid_version: String,
    pub camelid_commit: String,
}

impl LaneIdentity {
    /// Capture the lane identity for a loaded GGUF. `gguf_sha256` is the
    /// streaming file hash computed once at load time; `tokenizer_kind` is
    /// the loader-reported summary (e.g. `llama_spm`) when the tokenizer
    /// loaded, with the raw `tokenizer.ggml.model` metadata as fallback.
    pub fn capture(
        model_id: &str,
        path: &Path,
        gguf: &GgufFile,
        tokenizer_kind: Option<&str>,
        gguf_sha256: String,
    ) -> Self {
        let tokenizer_kind = tokenizer_kind
            .map(ToOwned::to_owned)
            .or_else(|| match gguf.metadata.get("tokenizer.ggml.model") {
                Some(GgufMetadataValue::String(value)) => Some(value.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "unknown".to_string());
        Self {
            model_id: model_id.to_string(),
            gguf_sha256,
            gguf_filename: path
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| path.display().to_string()),
            quantization: quantization_label(gguf),
            architecture: gguf.architecture().unwrap_or("unknown").to_string(),
            tokenizer_kind,
            tokenizer_sha256: tokenizer_metadata_sha256(gguf),
            camelid_version: camelid_version(),
            camelid_commit: camelid_commit(),
        }
    }
}

/// Human label for the GGUF's quantization: `general.file_type` when present
/// (llama.cpp ftype naming), else the dominant tensor type by count.
pub fn quantization_label(gguf: &GgufFile) -> String {
    if let Some(label) = gguf
        .metadata
        .get("general.file_type")
        .and_then(file_type_label)
    {
        return label.to_string();
    }
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for tensor in &gguf.tensors {
        *counts
            .entry(format!("{:?}", tensor.tensor_type))
            .or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(label, _)| label)
        .unwrap_or_else(|| "unknown".to_string())
}

fn file_type_label(value: &GgufMetadataValue) -> Option<&'static str> {
    let id = match value {
        GgufMetadataValue::U32(v) => i64::from(*v),
        GgufMetadataValue::I32(v) => i64::from(*v),
        GgufMetadataValue::U64(v) => i64::try_from(*v).ok()?,
        GgufMetadataValue::I64(v) => *v,
        _ => return None,
    };
    Some(match id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        32 => "BF16",
        _ => return None,
    })
}

/// SHA-256 over the canonical JSON of all `tokenizer.*` GGUF metadata
/// entries, or `None` when the file carries no tokenizer metadata.
pub fn tokenizer_metadata_sha256(gguf: &GgufFile) -> Option<String> {
    let entries: BTreeMap<&String, &GgufMetadataValue> = gguf
        .metadata
        .iter()
        .filter(|(key, _)| key.starts_with("tokenizer."))
        .collect();
    if entries.is_empty() {
        return None;
    }
    let value = serde_json::to_value(&entries).ok()?;
    Some(sha256_hex(canonical_json(&value).as_bytes()))
}

/// Version of the running build: `git describe --tags --dirty` embedded at
/// compile time when available, else crate version (+ short commit).
pub fn camelid_version() -> String {
    if let Some(describe) = option_env!("CAMELID_GIT_DESCRIBE") {
        if !describe.is_empty() {
            return describe.to_string();
        }
    }
    match option_env!("CAMELID_GIT_COMMIT") {
        Some(commit) if !commit.is_empty() => {
            let short = &commit[..commit.len().min(12)];
            format!("{}+{short}", env!("CARGO_PKG_VERSION"))
        }
        _ => env!("CARGO_PKG_VERSION").to_string(),
    }
}

/// Commit embedded at compile time (`git rev-parse HEAD`), or `"unknown"`
/// for builds made outside a git checkout.
pub fn camelid_commit() -> String {
    match option_env!("CAMELID_GIT_COMMIT") {
        Some(commit) if !commit.is_empty() => commit.to_string(),
        _ => "unknown".to_string(),
    }
}

/// Identity of the reference engine the receipt compares against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceIdentity {
    pub tool: String,
    pub binary: String,
    pub version: Option<String>,
    pub commit: Option<String>,
}

/// The exact request that was (or is to be) replayed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReceiptRequest {
    pub endpoint: String,
    /// The exact input: a chat `messages` array or a raw `prompt` string.
    pub messages_or_prompt: Value,
    pub max_tokens: u32,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
}

/// What this Camelid build produced for the request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReceiptResult {
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub generated_text: String,
    pub completion_tokens: u32,
    pub finish_reason: String,
}

/// Comparison outcome against the reference engine.
///
/// When no reference was live at emit time, `compared_against_reference` is
/// `false` and every match field is `None` — the receipt is then a claim of
/// output for the verifier to check independently. Match fields are never
/// fabricated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityBlock {
    pub compared_against_reference: bool,
    pub prompt_tokens_match: Option<bool>,
    pub generated_tokens_match: Option<bool>,
    pub generated_text_match: Option<bool>,
    /// [`NO_DIVERGENCE`] (-1) means no divergence; `None` means not compared.
    pub first_divergent_token_index: Option<i64>,
}

impl ParityBlock {
    /// The honest placeholder for a receipt emitted with no live reference.
    pub fn not_compared() -> Self {
        Self {
            compared_against_reference: false,
            prompt_tokens_match: None,
            generated_tokens_match: None,
            generated_text_match: None,
            first_divergent_token_index: None,
        }
    }
}

/// Reserved for the deferred signing decision: a detached signature over
/// `receipt_id`. Not produced by v1 emitters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignatureBlock {
    pub algorithm: String,
    pub public_key: String,
    pub signature: String,
}

impl ParityReceipt {
    /// Canonical serialization of the receipt body: every field except
    /// `receipt_id`, recursively key-sorted, compact (no insignificant
    /// whitespace). This is the byte string the digest is computed over.
    pub fn canonical_body(&self) -> Result<String, ReceiptError> {
        let mut value = serde_json::to_value(self)?;
        if let Value::Object(map) = &mut value {
            map.remove("receipt_id");
        }
        Ok(canonical_json(&value))
    }

    /// Recompute the digest the `receipt_id` field should hold.
    pub fn compute_receipt_id(&self) -> Result<String, ReceiptError> {
        Ok(sha256_hex(self.canonical_body()?.as_bytes()))
    }

    /// Populate `receipt_id` from the canonical body. Call last, after every
    /// other field is final.
    pub fn seal(&mut self) -> Result<(), ReceiptError> {
        self.receipt_id = self.compute_receipt_id()?;
        Ok(())
    }

    /// Cheap tamper check: recompute the digest and confirm it matches the
    /// stored `receipt_id`. This is the first thing the verifier runs.
    pub fn verify_self_digest(&self) -> Result<(), ReceiptError> {
        let computed = self.compute_receipt_id()?;
        if computed == self.receipt_id {
            Ok(())
        } else {
            Err(ReceiptError::DigestMismatch {
                stored: self.receipt_id.clone(),
                computed,
            })
        }
    }
}

/// Deterministic canonical JSON: object keys recursively sorted, compact
/// separators. Implemented by explicit key sorting so it does not depend on
/// `serde_json`'s map backend (`preserve_order` may be enabled transitively).
pub fn canonical_json(value: &Value) -> String {
    serde_json::to_string(&sorted_value(value))
        .expect("re-serializing an already-parsed JSON value cannot fail")
}

fn sorted_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for key in keys {
                out.insert(key.clone(), sorted_value(&map[key]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted_value).collect()),
        other => other.clone(),
    }
}

/// Lowercase hex SHA-256 of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

/// Lowercase hex SHA-256 of a file, computed with a streaming reader so large
/// GGUFs are never held in memory.
pub fn sha256_file_hex(path: &Path) -> Result<String, ReceiptError> {
    let io_err = |source| ReceiptError::Io {
        path: path.to_path_buf(),
        source,
    };
    let file = std::fs::File::open(path).map_err(io_err)?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let read = reader.read(&mut buf).map_err(io_err)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(out, "{byte:02x}").expect("writing hex to a String cannot fail");
    }
    out
}

/// Current time as an RFC 3339 UTC timestamp (second precision).
pub fn rfc3339_utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is before the Unix epoch")
        .as_secs();
    rfc3339_utc_from_epoch_secs(secs as i64)
}

/// RFC 3339 UTC timestamp for the given seconds since the Unix epoch.
pub fn rfc3339_utc_from_epoch_secs(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let day_secs = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        day_secs / 3600,
        (day_secs % 3600) / 60,
        day_secs % 60,
    )
}

/// Days-since-epoch to (year, month, day) in the proleptic Gregorian
/// calendar. Howard Hinnant's `civil_from_days` algorithm.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_receipt() -> ParityReceipt {
        ParityReceipt {
            schema: RECEIPT_SCHEMA_V1.to_string(),
            receipt_id: String::new(),
            created_utc: rfc3339_utc_from_epoch_secs(1_780_000_000),
            lane: LaneIdentity {
                model_id: "tinyllama-q8".to_string(),
                gguf_sha256: "ab".repeat(32),
                gguf_filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf".to_string(),
                quantization: "Q8_0".to_string(),
                architecture: "llama".to_string(),
                tokenizer_kind: "llama_spm".to_string(),
                tokenizer_sha256: None,
                camelid_version: "0.1.0".to_string(),
                camelid_commit: "65a1c35".to_string(),
            },
            reference: ReferenceIdentity {
                tool: "llama.cpp".to_string(),
                binary: "llama-server".to_string(),
                version: Some("b4567".to_string()),
                commit: None,
            },
            request: ReceiptRequest {
                endpoint: "/v1/chat/completions".to_string(),
                messages_or_prompt: json!([{ "role": "user", "content": "Count to three." }]),
                max_tokens: 50,
                temperature: 0.0,
                top_p: None,
                top_k: None,
                seed: None,
                stop: vec![],
            },
            reproducible: true,
            result: ReceiptResult {
                prompt_token_ids: vec![1, 529, 29989],
                generated_token_ids: vec![29907, 650],
                generated_text: "One".to_string(),
                completion_tokens: 2,
                finish_reason: "length".to_string(),
            },
            parity: ParityBlock {
                compared_against_reference: true,
                prompt_tokens_match: Some(true),
                generated_tokens_match: Some(true),
                generated_text_match: Some(true),
                first_divergent_token_index: Some(NO_DIVERGENCE),
            },
            signature: None,
        }
    }

    #[test]
    fn receipt_round_trips_losslessly() {
        let mut receipt = sample_receipt();
        receipt.seal().expect("seal");
        let json = serde_json::to_string(&receipt).expect("serialize");
        let back: ParityReceipt = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(receipt, back);
    }

    #[test]
    fn not_compared_block_round_trips_with_null_match_fields() {
        let mut receipt = sample_receipt();
        receipt.parity = ParityBlock::not_compared();
        receipt.seal().expect("seal");
        let json = serde_json::to_string(&receipt).expect("serialize");
        assert!(json.contains("\"prompt_tokens_match\":null"));
        let back: ParityReceipt = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(receipt, back);
    }

    #[test]
    fn signature_field_is_absent_when_none() {
        let mut receipt = sample_receipt();
        receipt.seal().expect("seal");
        let json = serde_json::to_string(&receipt).expect("serialize");
        assert!(!json.contains("signature"));
    }

    #[test]
    fn canonical_body_excludes_receipt_id_and_sorts_keys() {
        let receipt = sample_receipt();
        let body = receipt.canonical_body().expect("canonical body");
        assert!(!body.contains("receipt_id"));
        // Top-level keys appear in sorted order.
        let positions: Vec<usize> = [
            "\"created_utc\"",
            "\"lane\"",
            "\"parity\"",
            "\"reference\"",
            "\"reproducible\"",
            "\"request\"",
            "\"result\"",
            "\"schema\"",
        ]
        .iter()
        .map(|key| body.find(key).expect("key present"))
        .collect();
        assert!(positions.windows(2).all(|w| w[0] < w[1]));
        // Compact: no insignificant whitespace.
        assert!(!body.contains(": "));
        assert!(!body.contains(", "));
    }

    #[test]
    fn seal_then_verify_self_digest_passes() {
        let mut receipt = sample_receipt();
        receipt.seal().expect("seal");
        assert_eq!(receipt.receipt_id.len(), 64);
        receipt
            .verify_self_digest()
            .expect("untouched receipt verifies");
    }

    #[test]
    fn mutating_one_field_changes_digest_and_fails_verification() {
        let mut receipt = sample_receipt();
        receipt.seal().expect("seal");
        let original_id = receipt.receipt_id.clone();

        let mut tampered = receipt.clone();
        // Mutate one byte of the receipt body.
        tampered.result.generated_text = "One!".to_string();
        let recomputed = tampered.compute_receipt_id().expect("recompute");
        assert_ne!(recomputed, original_id);
        assert!(matches!(
            tampered.verify_self_digest(),
            Err(ReceiptError::DigestMismatch { .. })
        ));

        receipt
            .verify_self_digest()
            .expect("original still verifies");
    }

    #[test]
    fn sha256_matches_known_test_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_file_streams_and_matches_buffer_hash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fixture.bin");
        // Larger than the 1 MiB streaming buffer to exercise multiple reads.
        let data: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &data).expect("write fixture");
        assert_eq!(
            sha256_file_hex(&path).expect("hash file"),
            sha256_hex(&data)
        );
    }

    #[test]
    fn rfc3339_known_values() {
        assert_eq!(rfc3339_utc_from_epoch_secs(0), "1970-01-01T00:00:00Z");
        assert_eq!(
            rfc3339_utc_from_epoch_secs(1_717_286_400),
            "2024-06-02T00:00:00Z"
        );
        assert_eq!(
            rfc3339_utc_from_epoch_secs(951_827_696),
            "2000-02-29T12:34:56Z"
        );
    }

    fn gguf_for_lane_tests(
        metadata: BTreeMap<String, GgufMetadataValue>,
        tensors: Vec<crate::gguf::GgufTensorDescriptor>,
    ) -> GgufFile {
        GgufFile {
            path: std::path::PathBuf::from("/models/lane-test.Q8_0.gguf"),
            version: 3,
            tensor_count: tensors.len() as i64,
            metadata_count: metadata.len() as i64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors,
        }
    }

    #[test]
    fn lane_capture_uses_file_type_architecture_and_tokenizer_metadata() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufMetadataValue::String("llama".to_string()),
        );
        metadata.insert("general.file_type".to_string(), GgufMetadataValue::U32(7));
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            GgufMetadataValue::String("llama".to_string()),
        );
        let gguf = gguf_for_lane_tests(metadata, Vec::new());

        let lane = LaneIdentity::capture(
            "lane-test",
            Path::new("/models/lane-test.Q8_0.gguf"),
            &gguf,
            Some("llama_spm"),
            "ab".repeat(32),
        );

        assert_eq!(lane.model_id, "lane-test");
        assert_eq!(lane.gguf_filename, "lane-test.Q8_0.gguf");
        assert_eq!(lane.quantization, "Q8_0");
        assert_eq!(lane.architecture, "llama");
        assert_eq!(lane.tokenizer_kind, "llama_spm");
        let tokenizer_sha = lane.tokenizer_sha256.expect("tokenizer metadata present");
        assert_eq!(tokenizer_sha.len(), 64);
        assert_eq!(lane.camelid_version, camelid_version());
        assert_eq!(lane.camelid_commit, camelid_commit());
    }

    #[test]
    fn lane_capture_without_tokenizer_metadata_yields_null_tokenizer_hash() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufMetadataValue::String("llama".to_string()),
        );
        let gguf = gguf_for_lane_tests(metadata, Vec::new());

        let lane = LaneIdentity::capture(
            "lane-test",
            Path::new("/models/lane-test.gguf"),
            &gguf,
            None,
            "cd".repeat(32),
        );

        assert_eq!(lane.tokenizer_sha256, None);
        assert_eq!(lane.tokenizer_kind, "unknown");
    }

    #[test]
    fn quantization_label_falls_back_to_dominant_tensor_type() {
        use crate::gguf::{GgufTensorDescriptor, GgufTensorType};
        let descriptor = |name: &str, tensor_type| GgufTensorDescriptor {
            name: name.to_string(),
            dimensions: vec![1],
            tensor_type,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: 0,
        };
        let gguf = gguf_for_lane_tests(
            BTreeMap::new(),
            vec![
                descriptor("blk.0.attn_q.weight", GgufTensorType::Q8_0),
                descriptor("blk.0.attn_k.weight", GgufTensorType::Q8_0),
                descriptor("blk.0.attn_norm.weight", GgufTensorType::F32),
            ],
        );
        assert_eq!(quantization_label(&gguf), "Q8_0");
        assert_eq!(
            quantization_label(&gguf_for_lane_tests(BTreeMap::new(), Vec::new())),
            "unknown"
        );
    }

    #[test]
    fn canonical_json_sorts_nested_objects() {
        let value = json!({ "b": { "z": 1, "a": [ { "y": 2, "x": 3 } ] }, "a": 0 });
        assert_eq!(
            canonical_json(&value),
            r#"{"a":0,"b":{"a":[{"x":3,"y":2}],"z":1}}"#
        );
    }
}
