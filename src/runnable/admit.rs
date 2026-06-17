//! Admission gate for the runnable lane.
//!
//! A GGUF runs iff **all three** coverage axes are covered:
//!
//! ```text
//! architecture graph  ×  quant dequant  ×  tokenizer
//! ```
//!
//! `admit` evaluates a parsed [`GgufFile`] against the v1 covered-set and either
//! returns an [`AdmissionOk`] summary or an [`AdmissionReject`] naming the offending
//! axis, the offending value, and (for the quant axis) the offending tensor. The
//! reject is `Serialize` so the refusal reason is machine-readable, per
//! `RUNNABLE_LANE_SPEC.md` principle #2.
//!
//! The covered-set here is **authoritative for the runnable lane** and is taken
//! verbatim from the spec — it intentionally differs from `model.rs`'s
//! optimized-lane architecture allowlist (see `BACKEND_ASKS.md` RA-4). In
//! particular the runnable set includes `gemma2` and excludes
//! `mistral`/`smollm3`/`gemma4`/`lfm2`.

use serde::Serialize;
use std::collections::BTreeSet;
use std::fmt;

use crate::error::BackendError;
use crate::gguf::{GgufFile, GgufTensorType};

/// v1 covered architectures (`general.architecture`).
pub const COVERED_ARCHITECTURES: &[&str] =
    &["llama", "qwen2", "qwen3", "gemma2", "gemma3", "phi3"];

/// v1 covered tokenizer models (`tokenizer.ggml.model`), grouped by family below.
/// SPM (sentencepiece/llama-style) + BPE (gpt2-style) are the two covered families.
const SPM_TOKENIZERS: &[&str] = &["llama", "gemma", "gemma4"];
const BPE_TOKENIZERS: &[&str] = &["gpt2"];

/// Sentinel used in a reject when the offending axis value is absent from metadata.
const ABSENT: &str = "<absent>";

/// The coverage axis a GGUF failed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionAxis {
    Architecture,
    Quant,
    Tokenizer,
}

impl AdmissionAxis {
    pub fn as_str(self) -> &'static str {
        match self {
            AdmissionAxis::Architecture => "architecture",
            AdmissionAxis::Quant => "quant",
            AdmissionAxis::Tokenizer => "tokenizer",
        }
    }
}

/// Which covered tokenizer family a model resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerFamily {
    /// SentencePiece-style (llama/gemma).
    Spm,
    /// GPT-2-style byte-level BPE (qwen/phi/gpt2).
    Bpe,
}

/// Structured admission rejection. Names the offending axis + value (+ tensor for
/// the quant axis) so the refusal is machine-readable, not just a string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdmissionReject {
    pub axis: AdmissionAxis,
    /// The offending value (architecture name, quant type, or tokenizer model);
    /// `"<absent>"` when the value was missing from metadata entirely.
    pub offending_value: String,
    /// For the quant axis, the first tensor carrying the unsupported quant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor: Option<String>,
    /// Human-readable single-line reason.
    pub message: String,
}

impl fmt::Display for AdmissionReject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for AdmissionReject {}

/// An admission rejection maps onto the existing `UnsupportedGguf` backend error so
/// callers that thread `BackendError` keep working; the structured form is preserved
/// in the message.
impl From<AdmissionReject> for BackendError {
    fn from(reject: AdmissionReject) -> Self {
        BackendError::UnsupportedGguf(reject.message)
    }
}

/// Summary of an admitted GGUF: the resolved architecture, tokenizer family, and the
/// distinct set of quant types present (handy for downstream dequant wiring).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdmissionOk {
    pub architecture: String,
    pub tokenizer: TokenizerFamily,
    pub quants: BTreeSet<GgufTensorType>,
}

/// Evaluate the three coverage axes against the v1 covered-set.
///
/// Axes are checked in a fixed order — architecture, tokenizer, then every tensor's
/// quant — so the reported failure is deterministic. The quant scan reports the
/// **first** tensor (in file order) carrying an unsupported type.
pub fn admit(file: &GgufFile) -> Result<AdmissionOk, AdmissionReject> {
    let architecture = check_architecture(file)?;
    let tokenizer = check_tokenizer(file)?;
    let quants = check_quants(file)?;
    Ok(AdmissionOk {
        architecture,
        tokenizer,
        quants,
    })
}

fn check_architecture(file: &GgufFile) -> Result<String, AdmissionReject> {
    match file.architecture() {
        Some(arch) if COVERED_ARCHITECTURES.contains(&arch) => Ok(arch.to_string()),
        Some(arch) => Err(AdmissionReject {
            axis: AdmissionAxis::Architecture,
            offending_value: arch.to_string(),
            tensor: None,
            message: format!(
                "unsupported architecture {arch:?}; runnable v1 covers {}",
                joined(COVERED_ARCHITECTURES)
            ),
        }),
        None => Err(AdmissionReject {
            axis: AdmissionAxis::Architecture,
            offending_value: ABSENT.to_string(),
            tensor: None,
            message: "missing general.architecture; runnable admission requires it".to_string(),
        }),
    }
}

fn check_tokenizer(file: &GgufFile) -> Result<TokenizerFamily, AdmissionReject> {
    match file.metadata_string("tokenizer.ggml.model") {
        Some(model) if SPM_TOKENIZERS.contains(&model) => Ok(TokenizerFamily::Spm),
        Some(model) if BPE_TOKENIZERS.contains(&model) => Ok(TokenizerFamily::Bpe),
        Some(model) => Err(AdmissionReject {
            axis: AdmissionAxis::Tokenizer,
            offending_value: model.to_string(),
            tensor: None,
            message: format!(
                "unsupported tokenizer.ggml.model {model:?}; runnable v1 covers SPM ({}) and BPE ({})",
                joined(SPM_TOKENIZERS),
                joined(BPE_TOKENIZERS)
            ),
        }),
        None => Err(AdmissionReject {
            axis: AdmissionAxis::Tokenizer,
            offending_value: ABSENT.to_string(),
            tensor: None,
            message: "missing tokenizer.ggml.model; runnable admission requires it".to_string(),
        }),
    }
}

/// A GGUF tensor quant type is covered iff the runnable lane has a dequant-to-f32
/// routine for it. K-quant *mix* recipes (Q4_K_M, Q5_K_M) are not distinct ggml
/// types — they appear on the wire as Q4K/Q5K/Q6K/Q8_0 tensors, all covered below.
fn is_covered_quant(tt: GgufTensorType) -> bool {
    matches!(
        tt,
        GgufTensorType::F32
            | GgufTensorType::F16
            | GgufTensorType::Q8_0
            | GgufTensorType::Q6K
            | GgufTensorType::Q5K
            | GgufTensorType::Q4K
            | GgufTensorType::Q4_0
    )
}

fn check_quants(file: &GgufFile) -> Result<BTreeSet<GgufTensorType>, AdmissionReject> {
    let mut seen = BTreeSet::new();
    for tensor in &file.tensors {
        if !is_covered_quant(tensor.tensor_type) {
            return Err(AdmissionReject {
                axis: AdmissionAxis::Quant,
                offending_value: format!("{:?}", tensor.tensor_type),
                tensor: Some(tensor.name.clone()),
                message: format!(
                    "unsupported quant {:?} in tensor {}; runnable v1 covers \
                     F32, F16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K",
                    tensor.tensor_type, tensor.name
                ),
            });
        }
        seen.insert(tensor.tensor_type);
    }
    Ok(seen)
}

fn joined(items: &[&str]) -> String {
    items.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufMetadataValue, GgufTensorDescriptor};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    /// Build a minimal in-set GGUF (llama / SPM / Q8_0) we can mutate per test.
    fn base_fixture() -> GgufFile {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("llama".into()),
        );
        metadata.insert(
            "tokenizer.ggml.model".into(),
            GgufMetadataValue::String("llama".into()),
        );
        GgufFile {
            path: PathBuf::from("/tmp/model.gguf"),
            version: 3,
            tensor_count: 2,
            metadata_count: metadata.len() as i64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors: vec![
                tensor("token_embd.weight", GgufTensorType::F32),
                tensor("blk.0.attn_q.weight", GgufTensorType::Q8_0),
            ],
        }
    }

    fn tensor(name: &str, tt: GgufTensorType) -> GgufTensorDescriptor {
        GgufTensorDescriptor {
            name: name.into(),
            dimensions: vec![32, 32],
            tensor_type: tt,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: 34,
        }
    }

    fn set_meta(file: &mut GgufFile, key: &str, value: &str) {
        file.metadata
            .insert(key.into(), GgufMetadataValue::String(value.into()));
    }

    #[test]
    fn accepts_in_set_llama_spm_q8() {
        let ok = admit(&base_fixture()).expect("in-set GGUF must admit");
        assert_eq!(ok.architecture, "llama");
        assert_eq!(ok.tokenizer, TokenizerFamily::Spm);
        assert!(ok.quants.contains(&GgufTensorType::Q8_0));
        assert!(ok.quants.contains(&GgufTensorType::F32));
    }

    #[test]
    fn accepts_each_covered_architecture() {
        for arch in COVERED_ARCHITECTURES {
            let mut file = base_fixture();
            set_meta(&mut file, "general.architecture", arch);
            // qwen/phi ship a BPE tokenizer; keep SPM here — admission is per-axis
            // independent, and the architecture axis is what we're exercising.
            assert!(
                admit(&file).is_ok(),
                "covered architecture {arch} must admit"
            );
        }
    }

    #[test]
    fn accepts_bpe_tokenizer() {
        let mut file = base_fixture();
        set_meta(&mut file, "general.architecture", "qwen3");
        set_meta(&mut file, "tokenizer.ggml.model", "gpt2");
        let ok = admit(&file).expect("qwen3 + gpt2-BPE must admit");
        assert_eq!(ok.tokenizer, TokenizerFamily::Bpe);
    }

    #[test]
    fn accepts_covered_kquants() {
        for tt in [
            GgufTensorType::F16,
            GgufTensorType::Q4_0,
            GgufTensorType::Q4K,
            GgufTensorType::Q5K,
            GgufTensorType::Q6K,
        ] {
            let mut file = base_fixture();
            file.tensors.push(tensor("blk.0.ffn_down.weight", tt));
            assert!(admit(&file).is_ok(), "covered quant {tt:?} must admit");
        }
    }

    #[test]
    fn rejects_unknown_architecture() {
        let mut file = base_fixture();
        set_meta(&mut file, "general.architecture", "mixtral");
        let reject = admit(&file).expect_err("unknown arch must reject");
        assert_eq!(reject.axis, AdmissionAxis::Architecture);
        assert_eq!(reject.offending_value, "mixtral");
        assert!(reject.tensor.is_none());
        assert!(reject.message.contains("mixtral"));
    }

    #[test]
    fn rejects_missing_architecture() {
        let mut file = base_fixture();
        file.metadata.remove("general.architecture");
        let reject = admit(&file).expect_err("missing arch must reject");
        assert_eq!(reject.axis, AdmissionAxis::Architecture);
        assert_eq!(reject.offending_value, "<absent>");
    }

    #[test]
    fn rejects_unknown_quant_naming_tensor() {
        let mut file = base_fixture();
        // Q3_K is an explicit v1 gap (deferred to v1.1).
        file.tensors
            .push(tensor("blk.12.ffn_down.weight", GgufTensorType::Q3K));
        let reject = admit(&file).expect_err("Q3_K must reject");
        assert_eq!(reject.axis, AdmissionAxis::Quant);
        assert_eq!(reject.offending_value, "Q3K");
        assert_eq!(reject.tensor.as_deref(), Some("blk.12.ffn_down.weight"));
        assert!(reject.message.contains("blk.12.ffn_down.weight"));
    }

    #[test]
    fn rejects_iquant_naming_tensor() {
        let mut file = base_fixture();
        // i-quants (IQ4_NL here) are an explicit v1 gap.
        file.tensors
            .push(tensor("blk.3.attn_k.weight", GgufTensorType::IQ4NL));
        let reject = admit(&file).expect_err("IQ4_NL must reject");
        assert_eq!(reject.axis, AdmissionAxis::Quant);
        assert_eq!(reject.offending_value, "IQ4NL");
        assert_eq!(reject.tensor.as_deref(), Some("blk.3.attn_k.weight"));
    }

    #[test]
    fn rejects_unknown_tokenizer() {
        let mut file = base_fixture();
        set_meta(&mut file, "tokenizer.ggml.model", "rwkv");
        let reject = admit(&file).expect_err("unknown tokenizer must reject");
        assert_eq!(reject.axis, AdmissionAxis::Tokenizer);
        assert_eq!(reject.offending_value, "rwkv");
        assert!(reject.tensor.is_none());
    }

    #[test]
    fn rejects_missing_tokenizer() {
        let mut file = base_fixture();
        file.metadata.remove("tokenizer.ggml.model");
        let reject = admit(&file).expect_err("missing tokenizer must reject");
        assert_eq!(reject.axis, AdmissionAxis::Tokenizer);
        assert_eq!(reject.offending_value, "<absent>");
    }

    #[test]
    fn architecture_axis_checked_before_quant() {
        // A file failing on multiple axes reports architecture first (fixed order).
        let mut file = base_fixture();
        set_meta(&mut file, "general.architecture", "mixtral");
        file.tensors
            .push(tensor("blk.0.ffn_down.weight", GgufTensorType::Q2K));
        let reject = admit(&file).expect_err("must reject");
        assert_eq!(reject.axis, AdmissionAxis::Architecture);
    }

    #[test]
    fn reject_serializes_to_machine_readable_json() {
        let mut file = base_fixture();
        file.tensors
            .push(tensor("blk.12.ffn_down.weight", GgufTensorType::Q3K));
        let reject = admit(&file).expect_err("Q3_K must reject");
        let json = serde_json::to_value(&reject).expect("reject serializes");
        assert_eq!(json["axis"], "quant");
        assert_eq!(json["offending_value"], "Q3K");
        assert_eq!(json["tensor"], "blk.12.ffn_down.weight");
    }

    #[test]
    fn reject_converts_to_backend_error() {
        let mut file = base_fixture();
        set_meta(&mut file, "general.architecture", "mixtral");
        let reject = admit(&file).expect_err("must reject");
        let err: BackendError = reject.into();
        assert!(matches!(err, BackendError::UnsupportedGguf(_)));
    }
}
