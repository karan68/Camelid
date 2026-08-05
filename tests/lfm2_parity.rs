//! LFM2 / LFM2.5 short-conv bring-up: external parity anchor for the runnable
//! lane, with **llama.cpp as the reference oracle**.
//!
//! HARD GATE: the greedy (argmax) token sequence produced by the runnable
//! lane must match the pinned llama.cpp reference EXACTLY on frozen fixtures.
//!
//! Why llama.cpp and not HF transformers (the anchor `runnable_parity.rs`
//! uses): LFM2's short-conv graph is ported tensor-for-tensor from llama.cpp
//! `src/models/lfm2.cpp`, so llama.cpp is the graph this implementation claims
//! to reproduce. The fixtures are frozen token ids captured from
//! `llama-server /completion` with `return_tokens`, so the check runs offline
//! and the tokenizer is NOT a variable (both sides are fed the same prompt
//! ids).
//!
//! Scope of the claim: the greedy gate proves the LFM2 **forward graph** — the
//! double-gated short convolution, its rolling conv state, the QK-normed GQA
//! attention layers, and the per-layer conv/attention schedule. It does NOT
//! prove tokenizer parity, because it hands both sides the same prompt ids by
//! construction; `lfm2_tokenizer_matches_llamacpp` below covers that separately,
//! and chat-template fidelity is gated in
//! `api::tests::lfm2_renderer_matches_llamacpp_applied_template`.
//!
//! That split is load-bearing, not bookkeeping: this file passed 96/96 while
//! `Tokenizer::from_gguf` was refusing every LFM2 row outright, because a
//! frozen-ids test cannot see the tokenizer at all.
//!
//! Compute-heavy (CPU f32, 2.6B): run with `--release`. Skips if the model or
//! the fixtures are absent.

use std::path::{Path, PathBuf};

use camelid::receipt::sha256_file_hex;
use camelid::runnable::RunnableModel;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct PromptFixture {
    prompt_text: String,
    prompt_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Lfm2Fixtures {
    reference: String,
    gguf: String,
    /// SHA-256 of the exact GGUF the reference ran on. The test refuses to
    /// certify a DIFFERENT file under this receipt.
    gguf_sha256: String,
    max_new: usize,
    fixtures: Vec<PromptFixture>,
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// Tokenizer parity for the LFM2 row.
///
/// The greedy gate below feeds BOTH sides the same prompt ids, so it
/// deliberately does not measure tokenization. This test closes that hole using
/// the same frozen fixtures: llama.cpp's own `/tokenize` output for the raw
/// completion prompts (`add_special=false`) and for the chat-template prompts
/// (`add_special=true`, so they carry the leading BOS).
///
/// This is what proves `tokenizer.ggml.pre = "lfm2"` really is the `llama-bpe`
/// dialect llama.cpp resolves it to (`src/llama-vocab.cpp:2111-2123`) rather
/// than merely being accepted.
#[test]
fn lfm2_tokenizer_matches_llamacpp() {
    #[derive(Deserialize)]
    struct TemplateCase {
        name: String,
        applied_prompt: String,
        prompt_ids: Vec<u32>,
    }
    #[derive(Deserialize)]
    struct TemplateDoc {
        cases: Vec<TemplateCase>,
    }

    let fx_path = repo().join("tests/fixtures/lfm2_parity/lfm2.json");
    if !fx_path.exists() {
        eprintln!("SKIP: LFM2 parity fixtures absent");
        return;
    }
    let fx: Lfm2Fixtures =
        serde_json::from_str(&std::fs::read_to_string(&fx_path).unwrap()).expect("fixtures parse");
    let gguf_path = repo().join("models").join(&fx.gguf);
    if !gguf_path.exists() {
        eprintln!("SKIP: {} absent", gguf_path.display());
        return;
    }

    let gguf = camelid::gguf::read_metadata(gguf_path.to_str().unwrap()).expect("gguf metadata");
    // The blocker this pins: before `lfm2` was mapped to the llama-bpe dialect,
    // this constructor refused outright and every end-to-end LFM2 path died
    // here — with a parity-certified forward sitting behind it.
    let tok = camelid::tokenizer::Tokenizer::from_gguf(&gguf).expect("lfm2 tokenizer builds");

    // Raw completion prompts: llama.cpp tokenized these with add_special=false.
    for pf in &fx.fixtures {
        let ids = tok.encode(&pf.prompt_text, false, true).expect("encode");
        assert_eq!(
            ids, pf.prompt_ids,
            "tokenizer diverged from llama.cpp on {:?}",
            pf.prompt_text
        );
    }

    // Chat-template prompts: tokenized with add_special=true, so the reference
    // ids carry the leading BOS that LFM2's template asks for.
    let tpl_path = repo().join("tests/fixtures/lfm2_parity/lfm2-chat-template.json");
    if tpl_path.exists() {
        let doc: TemplateDoc = serde_json::from_str(&std::fs::read_to_string(&tpl_path).unwrap())
            .expect("chat-template fixture parses");
        for case in &doc.cases {
            let ids = tok
                .encode(&case.applied_prompt, true, true)
                .expect("encode");
            assert_eq!(
                ids, case.prompt_ids,
                "tokenizer diverged from llama.cpp on chat case {:?}",
                case.name
            );
        }
    }
}

#[test]
fn lfm2_matches_llamacpp_greedy() {
    let fx_path = repo().join("tests/fixtures/lfm2_parity/lfm2.json");
    if !fx_path.exists() {
        eprintln!("SKIP: LFM2 parity fixtures absent ({})", fx_path.display());
        return;
    }
    let fx: Lfm2Fixtures =
        serde_json::from_str(&std::fs::read_to_string(&fx_path).unwrap()).expect("fixtures parse");

    let gguf_path = repo().join("models").join(&fx.gguf);
    if !gguf_path.exists() {
        eprintln!("SKIP: {} absent", gguf_path.display());
        return;
    }
    let gguf_sha = sha256_file_hex(&gguf_path).expect("hash gguf");
    // Fail closed on a different file rather than silently certifying it: the
    // receipt is only meaningful for the bytes the reference actually ran.
    assert_eq!(
        gguf_sha, fx.gguf_sha256,
        "GGUF sha256 does not match the fixture's reference row"
    );

    let model = RunnableModel::load(gguf_path.to_str().unwrap()).expect("model loads");
    assert_eq!(model.architecture, "lfm2", "fixture/model arch mismatch");

    eprintln!("=== runnable lfm2 vs {} ===", fx.reference);
    let mut all_match = true;
    let mut prompt_records = Vec::new();

    for pf in &fx.fixtures {
        let cam_greedy = model
            .generate(&pf.prompt_ids, fx.max_new)
            .expect("generate");
        // The reference may stop early on EOS; compare over the overlap so a
        // shorter reference is not scored as a mismatch on length alone, but
        // require the reference to have produced something.
        assert!(
            !pf.greedy_ids.is_empty(),
            "fixture {:?} carries no reference tokens",
            pf.prompt_text
        );
        let n = pf.greedy_ids.len().min(cam_greedy.len());
        let matched = n == pf.greedy_ids.len() && cam_greedy[..n] == pf.greedy_ids[..n];
        all_match &= matched;

        let first_divergence = (0..n).find(|&i| cam_greedy[i] != pf.greedy_ids[i]);

        eprintln!(
            "  {:?}\n    ref greedy = {:?}\n    cam greedy = {:?}\n    match={matched} first_divergence={first_divergence:?}",
            pf.prompt_text, pf.greedy_ids, cam_greedy
        );

        prompt_records.push(json!({
            "prompt_text": pf.prompt_text,
            "prompt_ids": pf.prompt_ids,
            "ref_greedy": pf.greedy_ids,
            "cam_greedy": cam_greedy,
            "match": matched,
            "first_divergence": first_divergence,
        }));
    }

    eprintln!("  ALL greedy match = {all_match}");

    let artifact = json!({
        "lane": "runnable",
        "architecture": model.architecture,
        "quant": "Q8_0",
        "tokenizer": "gpt2_bpe",
        "gguf_filename": fx.gguf,
        "gguf_sha256": gguf_sha,
        "reference": fx.reference,
        "fixture_count": fx.fixtures.len(),
        "max_new": fx.max_new,
        "decode": "greedy/argmax",
        "proves": "lfm2 forward graph (short-conv + rolling conv state + QK-normed GQA + per-layer schedule)",
        "does_not_prove": "tokenizer parity (both sides are fed the same prompt ids by construction) — gated separately by lfm2_tokenizer_matches_llamacpp; chat-template fidelity — gated by api::tests::lfm2_renderer_matches_llamacpp_applied_template; and end-to-end serve, long context, sampling, tools, other quants, GPU lanes, other LFM2 variants — none of which are gated anywhere yet",
        "result": if all_match { "pass" } else { "fail" },
        "all_greedy_match": all_match,
        "prompts": prompt_records,
    });
    let out_dir = repo().join("qa").join("runnable");
    std::fs::create_dir_all(&out_dir).ok();
    let out_path = out_dir.join("lfm2-parity.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&artifact).unwrap())
        .expect("write artifact");
    eprintln!("  parity artifact -> {}", out_path.display());

    assert!(
        all_match,
        "runnable lfm2 greedy token sequence must match llama.cpp exactly"
    );
}
