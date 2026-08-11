//! Inspection-only adapter for Hugging Face `tokenizer.json` files.
//!
//! This module intentionally owns no generation state. It validates the JSON
//! envelope, asks the upstream Hugging Face `tokenizers` crate to deserialize
//! the complete pipeline, and records deterministic encode/decode probes that
//! can be compared with an external Transformers reference later.

use std::{fs, path::Path};

use serde::Serialize;
use serde_json::{Map, Value};

/// Stable probe inputs for tokenizer parity evidence.
///
/// Special tokens are disabled so these observations cover only the serialized
/// normalizer, pre-tokenizer, model, and decoder pipeline. BOS/EOS and chat
/// rendering remain explicit future gates.
pub const HF_TOKENIZER_PARITY_PROBES: [&str; 3] = [
    "Hello, Camelid!",
    " leading space",
    "caf\u{e9} \u{65e5}\u{672c}\u{8a9e}\nline 2",
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HfTokenizerJsonSummary {
    pub format_version: String,
    pub model_type: String,
    pub vocab_size: usize,
    pub added_tokens: usize,
    pub special_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalizer_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pre_tokenizer_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_processor_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoder_type: Option<String>,
    pub probes: Vec<HfTokenizerProbeObservation>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HfTokenizerProbeObservation {
    pub input: String,
    pub token_ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub decoded: String,
    /// Informational only. Normalizers and decoders may intentionally change
    /// spacing or Unicode, so a mismatch does not fail tokenizer readiness.
    pub exact_round_trip: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HfTokenizerJsonBlockerCode {
    TokenizerJsonUnreadable,
    InvalidTokenizerJson,
    InvalidTokenizerJsonRoot,
    MissingTokenizerVersion,
    MissingTokenizerModel,
    MissingTokenizerModelType,
    TokenizerDeserializationFailed,
    EmptyTokenizerVocabulary,
    TokenizerProbeEncodeFailed,
    TokenizerProbeProducedNoTokens,
    TokenizerProbeDecodeFailed,
}

impl HfTokenizerJsonBlockerCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TokenizerJsonUnreadable => "tokenizer_json_unreadable",
            Self::InvalidTokenizerJson => "invalid_tokenizer_json",
            Self::InvalidTokenizerJsonRoot => "invalid_tokenizer_json_root",
            Self::MissingTokenizerVersion => "missing_tokenizer_version",
            Self::MissingTokenizerModel => "missing_tokenizer_model",
            Self::MissingTokenizerModelType => "missing_tokenizer_model_type",
            Self::TokenizerDeserializationFailed => "tokenizer_deserialization_failed",
            Self::EmptyTokenizerVocabulary => "empty_tokenizer_vocabulary",
            Self::TokenizerProbeEncodeFailed => "tokenizer_probe_encode_failed",
            Self::TokenizerProbeProducedNoTokens => "tokenizer_probe_produced_no_tokens",
            Self::TokenizerProbeDecodeFailed => "tokenizer_probe_decode_failed",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HfTokenizerJsonBlocker {
    pub code: HfTokenizerJsonBlockerCode,
    pub message: String,
}

impl HfTokenizerJsonBlocker {
    fn new(code: HfTokenizerJsonBlockerCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// Validate and inspect a local Hugging Face `tokenizer.json` without wiring it
/// into Camelid's runtime tokenizer or generation paths.
pub fn inspect_hf_tokenizer_json(
    path: impl AsRef<Path>,
) -> Result<HfTokenizerJsonSummary, HfTokenizerJsonBlocker> {
    let bytes = fs::read(path.as_ref()).map_err(|_| {
        HfTokenizerJsonBlocker::new(
            HfTokenizerJsonBlockerCode::TokenizerJsonUnreadable,
            "could not read required Hugging Face tokenizer.json",
        )
    })?;
    let value: Value = serde_json::from_slice(&bytes).map_err(|_| {
        HfTokenizerJsonBlocker::new(
            HfTokenizerJsonBlockerCode::InvalidTokenizerJson,
            "required Hugging Face tokenizer.json is not valid JSON",
        )
    })?;
    let root = value.as_object().ok_or_else(|| {
        HfTokenizerJsonBlocker::new(
            HfTokenizerJsonBlockerCode::InvalidTokenizerJsonRoot,
            "required Hugging Face tokenizer.json must contain a JSON object",
        )
    })?;

    let format_version = root
        .get("version")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            HfTokenizerJsonBlocker::new(
                HfTokenizerJsonBlockerCode::MissingTokenizerVersion,
                "required Hugging Face tokenizer.json is missing string field version",
            )
        })?
        .to_string();
    let model = root
        .get("model")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            HfTokenizerJsonBlocker::new(
                HfTokenizerJsonBlockerCode::MissingTokenizerModel,
                "required Hugging Face tokenizer.json is missing object field model",
            )
        })?;
    let model_type = component_type(model).ok_or_else(|| {
        HfTokenizerJsonBlocker::new(
            HfTokenizerJsonBlockerCode::MissingTokenizerModelType,
            "required Hugging Face tokenizer.json model is missing string field type",
        )
    })?;

    // Deserialize from bytes instead of from_file so upstream diagnostics cannot
    // accidentally expose a user's absolute local model path.
    let tokenizer = tokenizers::Tokenizer::from_bytes(&bytes).map_err(|_| {
        HfTokenizerJsonBlocker::new(
            HfTokenizerJsonBlockerCode::TokenizerDeserializationFailed,
            format!(
                "Hugging Face tokenizers could not deserialize tokenizer.json model type {model_type}"
            ),
        )
    })?;
    let vocab_size = tokenizer.get_vocab_size(true);
    if vocab_size == 0 {
        return Err(HfTokenizerJsonBlocker::new(
            HfTokenizerJsonBlockerCode::EmptyTokenizerVocabulary,
            "Hugging Face tokenizer.json contains an empty vocabulary",
        ));
    }

    let mut probes = Vec::with_capacity(HF_TOKENIZER_PARITY_PROBES.len());
    for input in HF_TOKENIZER_PARITY_PROBES {
        let encoding = tokenizer.encode(input, false).map_err(|_| {
            HfTokenizerJsonBlocker::new(
                HfTokenizerJsonBlockerCode::TokenizerProbeEncodeFailed,
                "Hugging Face tokenizer.json failed the deterministic encode probe",
            )
        })?;
        if encoding.get_ids().is_empty() {
            return Err(HfTokenizerJsonBlocker::new(
                HfTokenizerJsonBlockerCode::TokenizerProbeProducedNoTokens,
                "Hugging Face tokenizer.json produced no tokens for a non-empty parity probe",
            ));
        }
        let decoded = tokenizer.decode(encoding.get_ids(), false).map_err(|_| {
            HfTokenizerJsonBlocker::new(
                HfTokenizerJsonBlockerCode::TokenizerProbeDecodeFailed,
                "Hugging Face tokenizer.json failed the deterministic decode probe",
            )
        })?;
        probes.push(HfTokenizerProbeObservation {
            input: input.to_string(),
            token_ids: encoding.get_ids().to_vec(),
            tokens: encoding.get_tokens().to_vec(),
            exact_round_trip: decoded == input,
            decoded,
        });
    }

    Ok(HfTokenizerJsonSummary {
        format_version,
        model_type,
        vocab_size,
        added_tokens: root
            .get("added_tokens")
            .and_then(Value::as_array)
            .map_or(0, Vec::len),
        special_tokens: root
            .get("added_tokens")
            .and_then(Value::as_array)
            .map_or(0, |tokens| {
                tokens
                    .iter()
                    .filter(|token| token.get("special").and_then(Value::as_bool) == Some(true))
                    .count()
            }),
        normalizer_type: optional_component_type(root, "normalizer"),
        pre_tokenizer_type: optional_component_type(root, "pre_tokenizer"),
        post_processor_type: optional_component_type(root, "post_processor"),
        decoder_type: optional_component_type(root, "decoder"),
        probes,
    })
}

fn optional_component_type(root: &Map<String, Value>, key: &str) -> Option<String> {
    root.get(key)
        .and_then(Value::as_object)
        .and_then(component_type)
}

fn component_type(component: &Map<String, Value>) -> Option<String> {
    component
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_string)
}

#[cfg(test)]
pub(crate) const TEST_WORDLEVEL_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {"[UNK]": 0, "Hello": 1, ",": 2, "Camelid": 3, "!": 4, "leading": 5, "space": 6, "caf\u00e9": 7, "\u65e5\u672c\u8a9e": 8, "line": 9, "2": 10},
    "unk_token": "[UNK]"
  }
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_adapter_records_stable_wordlevel_encode_decode_observations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        fs::write(&path, TEST_WORDLEVEL_TOKENIZER_JSON).unwrap();

        let summary = inspect_hf_tokenizer_json(&path).unwrap();

        assert_eq!(summary.format_version, "1.0");
        assert_eq!(summary.model_type, "WordLevel");
        assert_eq!(summary.vocab_size, 11);
        assert_eq!(summary.added_tokens, 1);
        assert_eq!(summary.special_tokens, 1);
        assert_eq!(summary.pre_tokenizer_type.as_deref(), Some("Whitespace"));
        assert_eq!(summary.probes.len(), HF_TOKENIZER_PARITY_PROBES.len());
        assert_eq!(summary.probes[0].token_ids, [1, 2, 3, 4]);
        assert_eq!(summary.probes[0].tokens, ["Hello", ",", "Camelid", "!"]);
        assert_eq!(summary.probes[0].decoded, "Hello , Camelid !");
        assert!(!summary.probes[0].exact_round_trip);
    }

    #[test]
    fn invalid_json_has_a_typed_blocker() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        fs::write(&path, "not json").unwrap();

        let blocker = inspect_hf_tokenizer_json(&path).unwrap_err();

        assert_eq!(
            blocker.code,
            HfTokenizerJsonBlockerCode::InvalidTokenizerJson
        );
        assert_eq!(blocker.code.as_str(), "invalid_tokenizer_json");
    }

    #[test]
    fn structurally_missing_model_has_a_typed_blocker() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        fs::write(&path, r#"{"version":"1.0"}"#).unwrap();

        let blocker = inspect_hf_tokenizer_json(&path).unwrap_err();

        assert_eq!(
            blocker.code,
            HfTokenizerJsonBlockerCode::MissingTokenizerModel
        );
    }

    #[test]
    fn upstream_deserialization_failure_has_a_typed_blocker() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        fs::write(
            &path,
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":[],"unk_token":"[UNK]"}}"#,
        )
        .unwrap();

        let blocker = inspect_hf_tokenizer_json(&path).unwrap_err();

        assert_eq!(
            blocker.code,
            HfTokenizerJsonBlockerCode::TokenizerDeserializationFailed
        );
    }
}
