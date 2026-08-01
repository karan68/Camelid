//! Tokenizer identity + throughput probe.
//!
//! `dump`  — encode every case in a case file under all four
//!           (add_special, parse_special) combinations and write the token ids
//!           to JSON. Diffing two dumps taken across a code change is the
//!           byte-identical-ids gate: tokenization identity is load-bearing for
//!           every parity receipt in this repo, so the diff must be empty.
//!
//! `bench` — time `encode` on a synthetic content ladder in both
//!           `parse_special` modes, in-process (no HTTP round trip).
//!
//! Usage:
//!   cargo run --release --example tokenizer_probe -- dump  <gguf> <cases.json> <out.json>
//!   cargo run --release --example tokenizer_probe -- bench <gguf> [reps]

use std::time::Instant;

use camelid::{gguf::read_metadata, tokenizer::Tokenizer};

const MODES: [(bool, bool); 4] = [(false, false), (false, true), (true, false), (true, true)];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: tokenizer_probe <dump|bench> <gguf> [...]");
        std::process::exit(2);
    }
    let mode = args[1].as_str();
    let gguf_path = &args[2];

    let load_start = Instant::now();
    let gguf = read_metadata(gguf_path).expect("read gguf metadata");
    let meta_ms = load_start.elapsed().as_secs_f64() * 1e3;
    let data_start_offset = gguf.data_start_offset;
    let build_start = Instant::now();
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("build tokenizer");
    let build_ms = build_start.elapsed().as_secs_f64() * 1e3;
    let load_ms = load_start.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "loaded {} vocab={} model={} load_ms={:.1} meta_ms={:.1} build_ms={:.1} header_bytes={}",
        gguf_path,
        tokenizer.tokens.len(),
        tokenizer.model.as_summary_model(),
        load_ms,
        meta_ms,
        build_ms,
        data_start_offset
    );

    match mode {
        "dump" => dump(&tokenizer, gguf_path, &args[3], &args[4]),
        "stats" => stats(&tokenizer),
        "bench" => {
            let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
            bench(&tokenizer, gguf_path, reps);
        }
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(2);
        }
    }
}

/// How many vocabulary entries the special-token matcher has to consider, and
/// how they distribute over first bytes — the shape that decides whether a
/// first-byte bucket is enough or a deeper structure is needed.
fn stats(tokenizer: &Tokenizer) {
    use std::collections::BTreeMap;
    let mut by_first: BTreeMap<u8, usize> = BTreeMap::new();
    let mut user_defined = 0usize;
    let mut control = 0usize;
    let mut pattern_bytes = 0usize;
    for token in &tokenizer.tokens {
        let is_control = match token.kind {
            camelid::tokenizer::TokenKind::UserDefined => false,
            camelid::tokenizer::TokenKind::Control => true,
            _ => continue,
        };
        let Some(&first) = token.text.as_bytes().first() else {
            continue;
        };
        if is_control {
            control += 1;
        } else {
            user_defined += 1;
        }
        pattern_bytes += token.text.len();
        *by_first.entry(first).or_default() += 1;
    }
    let total = user_defined + control;
    println!("specials total={total} user_defined={user_defined} control={control} pattern_bytes={pattern_bytes}");
    let mut buckets: Vec<_> = by_first.into_iter().collect();
    buckets.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    println!("distinct first bytes: {}", buckets.len());
    for (byte, n) in buckets.iter().take(8) {
        println!("  0x{byte:02x} {:?} -> {n}", *byte as char);
    }
}

fn dump(tokenizer: &Tokenizer, gguf_path: &str, cases_path: &str, out_path: &str) {
    let raw = std::fs::read_to_string(cases_path).expect("read cases");
    let parsed: serde_json::Value = serde_json::from_str(&raw).expect("parse cases");
    let cases = parsed["cases"].as_array().expect("cases array");

    let mut out = Vec::with_capacity(cases.len() * MODES.len());
    let mut total_ids: u64 = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap_or("");
        let text = case["text"].as_str().unwrap_or("");
        for (add_special, parse_special) in MODES {
            let ids = tokenizer
                .encode(text, add_special, parse_special)
                .unwrap_or_else(|err| {
                    panic!("encode failed for {name} ({add_special},{parse_special}): {err}")
                });
            total_ids += ids.len() as u64;
            out.push(serde_json::json!({
                "name": name,
                "add_special": add_special,
                "parse_special": parse_special,
                "ids": ids,
            }));
        }
    }

    let doc = serde_json::json!({
        "gguf": gguf_path,
        "vocab": tokenizer.tokens.len(),
        "tokenizer_model": tokenizer.model.as_summary_model(),
        "add_space_prefix": tokenizer.config.add_space_prefix,
        "cases": out.len(),
        "total_ids": total_ids,
        "results": out,
    });
    std::fs::write(out_path, serde_json::to_vec(&doc).expect("serialize")).expect("write dump");
    eprintln!("wrote {out_path}: {} results, {total_ids} ids", out.len());
}

/// Content ladder mirroring the /tokenize sweep in the Phase 0 write-up.
fn ladder() -> Vec<String> {
    let unit = "The quick brown fox jumps over the lazy dog near the river bank. ";
    [1usize, 2, 4, 10, 21, 42, 84, 165, 330]
        .iter()
        .map(|reps| unit.repeat(*reps))
        .collect()
}

fn bench(tokenizer: &Tokenizer, gguf_path: &str, reps: usize) {
    println!("# in-process encode, {gguf_path}, best of {reps}");
    println!("chars\ttokens\tplain_ms\tspecial_ms\tdelta_us_per_token");
    let mut rows = Vec::new();
    for text in ladder() {
        let mut best = [f64::MAX; 2];
        let mut ntok = 0usize;
        for (slot, parse_special) in [false, true].into_iter().enumerate() {
            for _ in 0..reps {
                let start = Instant::now();
                let ids = tokenizer
                    .encode(&text, false, parse_special)
                    .expect("encode");
                let ms = start.elapsed().as_secs_f64() * 1e3;
                if ms < best[slot] {
                    best[slot] = ms;
                }
                ntok = ids.len();
            }
        }
        let delta_us_per_token = if ntok > 0 {
            (best[1] - best[0]) * 1e3 / ntok as f64
        } else {
            0.0
        };
        println!(
            "{}\t{}\t{:.3}\t{:.3}\t{:.1}",
            text.len(),
            ntok,
            best[0],
            best[1],
            delta_us_per_token
        );
        rows.push(serde_json::json!({
            "chars": text.len(),
            "tokens": ntok,
            "plain_ms": best[0],
            "special_ms": best[1],
            "delta_us_per_token": delta_us_per_token,
        }));
    }
    eprintln!("{}", serde_json::to_string(&rows).expect("serialize rows"));
}
