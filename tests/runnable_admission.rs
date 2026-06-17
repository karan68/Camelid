//! Phase 1 / Gate 1 integration evidence for the runnable lane.
//!
//! Proves the admission gate against **real** GGUF files on disk: it parses a
//! known-good model, dumps its metadata + tensor table, and confirms admission.
//! Real-file tests are skipped (not failed) when the model is absent, so the suite
//! is portable off this dev box.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use camelid::gguf::{read_metadata, GgufTensorType};
use camelid::runnable::{admit, TokenizerFamily};

fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("models")
}

/// Read metadata, dump a compact metadata + tensor-table summary, and admit.
fn dump_and_admit(filename: &str, expect_tokenizer: TokenizerFamily) {
    let path = models_dir().join(filename);
    if !path.exists() {
        eprintln!("SKIP {filename}: not present at {}", path.display());
        return;
    }

    let file = read_metadata(&path).expect("known-good GGUF must parse");

    // --- metadata dump (key scalars) ---
    eprintln!("=== {filename} ===");
    eprintln!("  version={} tensors={}", file.version, file.tensor_count);
    eprintln!("  architecture={:?}", file.architecture());
    eprintln!(
        "  tokenizer.ggml.model={:?}",
        file.metadata_string("tokenizer.ggml.model")
    );

    // --- tensor table: count per quant type ---
    let mut by_type: BTreeMap<String, usize> = BTreeMap::new();
    for t in &file.tensors {
        *by_type.entry(format!("{:?}", t.tensor_type)).or_default() += 1;
    }
    for (ty, n) in &by_type {
        eprintln!("  tensor_type {ty:<6} x{n}");
    }
    if let Some(first) = file.tensors.first() {
        eprintln!(
            "  first tensor: {} {:?} dims={:?}",
            first.name, first.tensor_type, first.dimensions
        );
    }

    // --- admission ---
    let ok = admit(&file).expect("in-set model must admit");
    eprintln!(
        "  ADMITTED arch={} tokenizer={:?} quants={:?}",
        ok.architecture, ok.tokenizer, ok.quants
    );
    assert_eq!(ok.tokenizer, expect_tokenizer);
}

#[test]
fn tinyllama_q8_parses_dumps_and_admits() {
    // Known-validated TinyLlama 1.1B Q8_0 — the spec's Gate 1 reference model.
    // SPM tokenizer (tokenizer.ggml.model = "llama").
    dump_and_admit(
        "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        TokenizerFamily::Spm,
    );
}

#[test]
fn llama32_1b_q8_admits() {
    // Llama 3.2 ships a gpt2/llama-bpe tokenizer.
    dump_and_admit("Llama-3.2-1B-Instruct-Q8_0.gguf", TokenizerFamily::Bpe);
}

#[test]
fn qwen3_q8_admits_as_bpe() {
    dump_and_admit("Qwen3-1.7B-Q8_0.gguf", TokenizerFamily::Bpe);
}

#[test]
fn diffusiongemma_real_file_is_refused() {
    // A real on-disk out-of-set model. Empirically its `general.architecture` is
    // "diffusion-gemma" (out-of-set) AND it carries a Q5_0 tensor
    // (`self_cond_down.weight`, also out-of-set). Architecture is the first axis
    // checked, so the gate must refuse there — real evidence of a precise,
    // machine-readable refusal naming the offending axis + value.
    use camelid::runnable::AdmissionAxis;

    let path = models_dir().join("diffusiongemma-26B-A4B-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("SKIP diffusiongemma: not present");
        return;
    }
    let file = read_metadata(&path).expect("must parse");
    let arch = file.architecture().map(str::to_string);
    eprintln!("diffusiongemma architecture={arch:?}");

    // Sanity: this file does contain at least one out-of-set quant too, proving
    // the quant scan would also bite were architecture not checked first.
    let has_uncovered_quant = file
        .tensors
        .iter()
        .any(|t| !is_covered_quant(t.tensor_type));
    eprintln!("contains an uncovered quant: {has_uncovered_quant}");

    let reject = admit(&file).expect_err("out-of-set model must be refused");
    eprintln!("REFUSED: axis={:?} value={} :: {reject}", reject.axis, reject.offending_value);
    assert_eq!(reject.axis, AdmissionAxis::Architecture);
    assert_eq!(reject.offending_value, arch.unwrap());
}

/// Mirror of the gate's covered-quant predicate, for the sanity check above.
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
