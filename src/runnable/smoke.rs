//! Smoke-admission: a per-model "does it actually run cleanly here" check.
//!
//! This is **distinct from Phase 5 oracle qualification**. Oracle qualification
//! proves the runnable graph for an (architecture, quant, tokenizer) combo is
//! numerically equivalent to HF transformers — that is a one-time, per-architecture
//! gate. Smoke-admission is per-*model-file*: it confirms THIS GGUF admits, loads,
//! and executes deterministically without blowing up, and that greedy decode is not
//! degenerate. It attests deterministic execution, NOT parity.
//!
//! Guardrail: smoke-admission runs ONLY on combos that are already oracle-qualified
//! (so a green smoke can never be mistaken for "this architecture is correct" — that
//! claim is the oracle's). Any GGUF on a non-anchored combo is refused with
//! `combo not yet anchored`.
//!
//! A pass emits a **runnable** receipt (`execution_lane = Runnable`, never copper),
//! whose parity block is `not_compared` — honest that no reference was consulted.

use std::collections::HashSet;

use crate::error::{BackendError, Result};
use crate::gguf::{read_metadata, GgufFile, GgufTensorType};
use crate::receipt::{
    rfc3339_utc_now, sha256_file_hex, ExecutionLane, LaneIdentity, ParityBlock, ParityReceipt,
    ReceiptRequest, ReceiptResult, ReferenceIdentity, RECEIPT_SCHEMA_V1,
};
use crate::tokenizer::Tokenizer;

use super::admit::{self, TokenizerFamily};
use super::RunnableModel;

const SMOKE_PROMPT: &str = "What is the capital of France?";
const GEN_TOKENS: usize = 24;
/// A working LM's logits sit in roughly [-30, 30]; anything past this is a blow-up.
const SANE_LOGIT_ABS: f32 = 200.0;
/// Minimum distinct tokens over the generation before we call it degenerate.
const MIN_DISTINCT: usize = 6;

/// The (architecture, tokenizer-family, quant) combos that have been oracle-qualified
/// (HF-anchored), plus phi3 which is implemented + coherence-validated with HF parity
/// pending a larger-RAM machine (allowed per the runnable-lane memory policy). Smoke
/// runs ONLY on these — everything else is "combo not yet anchored".
fn is_oracle_qualified(arch: &str, tok: TokenizerFamily, quant: &str) -> bool {
    matches!(
        (arch, tok, quant),
        ("llama", TokenizerFamily::Spm, "Q8_0")
            | ("qwen3", TokenizerFamily::Bpe, "Q8_0")
            | ("gemma3", TokenizerFamily::Spm, "Q8_0")
            | ("phi3", TokenizerFamily::Spm, "Q8_0")
    )
}

/// Result of a passing smoke-admission.
pub struct SmokeReport {
    pub architecture: String,
    pub quant: String,
    pub tokenizer: TokenizerFamily,
    pub prompt_token_count: usize,
    pub generated: Vec<u32>,
    pub generated_text: String,
    pub logit_min: f32,
    pub logit_max: f32,
    /// Runnable receipt (lane=runnable, never copper) attesting deterministic
    /// execution — not parity.
    pub receipt: ParityReceipt,
}

/// Run smoke-admission on a GGUF. `Ok` only when every check passes; the returned
/// report carries the runnable receipt.
pub fn smoke_admit(path: &str) -> Result<SmokeReport> {
    let gguf = read_metadata(path)?;

    // (1) covered-set admission gate (Phase 1).
    let admitted = admit::admit(&gguf).map_err(BackendError::from)?;
    let quant = headline_quant(&gguf);

    // Oracle-qualified guardrail.
    if !is_oracle_qualified(&admitted.architecture, admitted.tokenizer, &quant) {
        return Err(BackendError::UnsupportedGguf(format!(
            "combo not yet anchored: {}/{}/{:?}; smoke-admission runs only on \
             oracle-qualified combos (llama/Q8_0/SPM, qwen3/Q8_0/BPE, gemma3/Q8_0/SPM, \
             phi3/Q8_0/SPM)",
            admitted.architecture, quant, admitted.tokenizer
        )));
    }

    // (2) load: all tensors present, shapes consistent, dequant succeeds on every
    // weight (RunnableModel::load fails closed otherwise).
    let model = RunnableModel::load(path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;

    // (3) greedy forward sanity on a fixed tiny prompt: finite logits, sane range.
    let (text, add_special, parse_special) = build_prompt(&tok);
    let prompt = tok.encode(&text, add_special, parse_special)?;
    if prompt.is_empty() {
        return Err(BackendError::InvalidTensorData(
            "smoke: prompt tokenized to nothing".into(),
        ));
    }
    let logits = model.forward_logits(&prompt)?;
    if !logits.iter().all(|v| v.is_finite()) {
        return Err(BackendError::InvalidTensorData(
            "smoke: forward produced non-finite logits".into(),
        ));
    }
    let logit_min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let logit_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !(logit_min > -SANE_LOGIT_ABS && logit_max < SANE_LOGIT_ABS) {
        return Err(BackendError::InvalidTensorData(format!(
            "smoke: logit range [{logit_min:.1}, {logit_max:.1}] is out of sane bounds"
        )));
    }

    // (4) coherence: greedy decode, fail if degenerate.
    let generated = model.generate(&prompt, GEN_TOKENS)?;
    check_not_degenerate(&generated)?;
    let generated_text = tok.decode(&generated, true).unwrap_or_default();

    let receipt = build_runnable_receipt(
        path,
        &gguf,
        admitted.tokenizer,
        &text,
        &prompt,
        &generated,
        &generated_text,
    )?;

    Ok(SmokeReport {
        architecture: admitted.architecture,
        quant,
        tokenizer: admitted.tokenizer,
        prompt_token_count: prompt.len(),
        generated,
        generated_text,
        logit_min,
        logit_max,
        receipt,
    })
}

/// The headline quant: the most common quantized (non-F32) tensor type, e.g. `Q8_0`.
fn headline_quant(gguf: &GgufFile) -> String {
    use std::collections::HashMap;
    let mut counts: HashMap<GgufTensorType, usize> = HashMap::new();
    for t in &gguf.tensors {
        if t.tensor_type != GgufTensorType::F32 {
            *counts.entry(t.tensor_type).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(tt, _)| format!("{tt:?}"))
        .unwrap_or_else(|| "F32".to_string())
}

/// Build the smoke prompt. Instruction-tuned models degenerate on raw completion
/// prompts, so render the GGUF's own chat template when present (gives each model a
/// fair coherence test); fall back to a raw completion prompt otherwise.
fn build_prompt(tok: &Tokenizer) -> (String, bool, bool) {
    if let Some(tmpl) = tok.chat_template.as_deref() {
        if let Some(rendered) = render_chat_template(tmpl, tok) {
            // Chat templates emit their own BOS (`{{ bos_token }}`) and control
            // markers, so don't add BOS again and DO parse specials.
            return (rendered, false, true);
        }
    }
    (SMOKE_PROMPT.to_string(), true, false)
}

/// Render a single-user-turn chat prompt from the GGUF Jinja chat template.
/// Best-effort: returns None if the template uses constructs we don't support, in
/// which case the caller falls back to a raw prompt.
fn render_chat_template(tmpl: &str, tok: &Tokenizer) -> Option<String> {
    let mut env = minijinja::Environment::new();
    // Some templates guard with raise_exception; surface it as a render error so we
    // fall back rather than panic.
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_template("chat", tmpl).ok()?;
    let template = env.get_template("chat").ok()?;
    let bos = tok.token_text(tok.special.bos).unwrap_or("");
    let eos = tok.token_text(tok.special.eos).unwrap_or("");
    let ctx = minijinja::context! {
        messages => vec![minijinja::context!{ role => "user", content => SMOKE_PROMPT }],
        add_generation_prompt => true,
        bos_token => bos,
        eos_token => eos,
    };
    let rendered = template.render(ctx).ok()?;
    if rendered.trim().is_empty() {
        None
    } else {
        Some(rendered)
    }
}

/// Degeneracy check: too few distinct tokens, or a short repetition cycle in the
/// tail, both signal a broken/garbage run rather than a working LM.
fn check_not_degenerate(gen: &[u32]) -> Result<()> {
    let distinct: HashSet<u32> = gen.iter().copied().collect();
    if distinct.len() < MIN_DISTINCT {
        return Err(BackendError::InvalidTensorData(format!(
            "smoke: degenerate greedy output — only {} distinct tokens in {}",
            distinct.len(),
            gen.len()
        )));
    }
    if let Some(period) = tail_cycle_period(gen, 4) {
        return Err(BackendError::InvalidTensorData(format!(
            "smoke: degenerate greedy output — tail is a period-{period} repetition loop"
        )));
    }
    Ok(())
}

/// Detect a period-`p` (1..=max_period) repetition loop occupying the tail: the last
/// three full periods are all identical. Returns the smallest such period.
fn tail_cycle_period(gen: &[u32], max_period: usize) -> Option<usize> {
    let n = gen.len();
    for p in 1..=max_period {
        if n < 3 * p {
            continue;
        }
        let tail = &gen[n - 3 * p..];
        if tail[..p] == tail[p..2 * p] && tail[p..2 * p] == tail[2 * p..] {
            return Some(p);
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn build_runnable_receipt(
    path: &str,
    gguf: &GgufFile,
    tok_family: TokenizerFamily,
    prompt_text: &str,
    prompt: &[u32],
    generated: &[u32],
    generated_text: &str,
) -> Result<ParityReceipt> {
    let gguf_sha = sha256_file_hex(std::path::Path::new(path))
        .map_err(|e| BackendError::InvalidTensorData(format!("smoke: hash gguf: {e}")))?;
    let model_id = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let tok_kind = match tok_family {
        TokenizerFamily::Spm => "llama_spm",
        TokenizerFamily::Bpe => "gpt2_bpe",
    };
    let lane = LaneIdentity::capture(
        &model_id,
        std::path::Path::new(path),
        gguf,
        Some(tok_kind),
        gguf_sha,
    );
    let mut receipt = ParityReceipt {
        schema: RECEIPT_SCHEMA_V1.to_string(),
        receipt_id: String::new(),
        created_utc: rfc3339_utc_now(),
        lane,
        // No reference engine was consulted — smoke-admission is not a parity check.
        reference: ReferenceIdentity {
            tool: "none".to_string(),
            binary: "runnable-smoke-admission".to_string(),
            version: None,
            commit: None,
        },
        request: ReceiptRequest {
            endpoint: "runnable/smoke-admission".to_string(),
            messages_or_prompt: serde_json::json!(prompt_text),
            max_tokens: GEN_TOKENS as u32,
            temperature: 0.0,
            top_p: None,
            top_k: None,
            seed: None,
            stop: vec![],
        },
        reproducible: true,
        result: ReceiptResult {
            prompt_token_ids: prompt.to_vec(),
            generated_token_ids: generated.to_vec(),
            generated_text: generated_text.to_string(),
            completion_tokens: generated.len() as u32,
            finish_reason: "length".to_string(),
        },
        // Honest: no reference comparison happened.
        parity: ParityBlock::not_compared(),
        // The load-bearing distinction: this is a runnable-lane receipt.
        execution_lane: Some(ExecutionLane::Runnable),
        execution_trace: None,
        signature: None,
    };
    receipt
        .seal()
        .map_err(|e| BackendError::InvalidTensorData(format!("smoke: seal receipt: {e}")))?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_qualified_gate() {
        assert!(is_oracle_qualified("llama", TokenizerFamily::Spm, "Q8_0"));
        assert!(is_oracle_qualified("qwen3", TokenizerFamily::Bpe, "Q8_0"));
        assert!(is_oracle_qualified("gemma3", TokenizerFamily::Spm, "Q8_0"));
        assert!(is_oracle_qualified("phi3", TokenizerFamily::Spm, "Q8_0"));
        // Not anchored: a covered architecture we have not HF-anchored.
        assert!(!is_oracle_qualified("gemma2", TokenizerFamily::Spm, "Q8_0"));
        // Not anchored: anchored arch but a quant we did not anchor.
        assert!(!is_oracle_qualified("llama", TokenizerFamily::Spm, "Q4K"));
        // Not anchored: anchored arch but unexpected tokenizer family.
        assert!(!is_oracle_qualified("llama", TokenizerFamily::Bpe, "Q8_0"));
    }

    #[test]
    fn detects_repetition_loops() {
        // period-1
        assert_eq!(tail_cycle_period(&[1, 2, 3, 7, 7, 7], 4), Some(1));
        // period-2
        assert_eq!(tail_cycle_period(&[1, 2, 4, 5, 4, 5, 4, 5], 4), Some(2));
        // period-4 (gemma raw-prompt failure shape)
        assert_eq!(
            tail_cycle_period(&[9, 8, 7, 6, 9, 8, 7, 6, 9, 8, 7, 6], 4),
            Some(4)
        );
        // healthy, varied tail
        assert_eq!(tail_cycle_period(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4), None);
    }

    #[test]
    fn degenerate_low_distinct_rejected() {
        let gen = vec![5u32, 5, 5, 6, 5, 6, 5, 6];
        assert!(check_not_degenerate(&gen).is_err());
    }

    #[test]
    fn healthy_generation_accepted() {
        let gen: Vec<u32> = (100..130).collect();
        assert!(check_not_degenerate(&gen).is_ok());
    }
}
