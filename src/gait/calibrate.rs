//! The parity-gated calibration tournament (Lane B).
//!
//! On first encounter of a `gait_key`, calibration discovers the fastest
//! execution configuration for *this* model on *this* machine, then persists it.
//! The engine here owns the hard part — extracting the model's real stage
//! shapes, running candidates through a **parity gate** (a faster candidate that
//! diverges is disqualified, full stop), picking a winner only if it beats
//! baseline by a margin, computing the roofline %, and failing closed to the
//! proven baseline otherwise.
//!
//! The act of *timing real decode* is an injected seam (`trial`): production
//! passes a closure that runs the engine under a candidate's configuration and
//! returns its tok/s plus a parity token (the greedy-output digest, matching the
//! `qa/speed` methodology); tests pass deterministic stubs. This keeps the
//! selection logic fully testable without ever fabricating a throughput number.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{gait_dir, store_in, GaitReceipt, MachineSig, MemoryMeasurement, ModelSig};
use crate::execution_plan::ExecutionProfile;
use crate::gguf::GgufFile;

/// A representative tensor shape for one inference stage class — the geometry a
/// candidate kernel is timed against. Calibration tunes on the model's *actual*
/// shapes, not synthetic ones.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageShape {
    pub stage: String,
    pub dims: Vec<u64>,
    pub quant: String,
}

/// One representative shape per matmul stage class present in the model.
pub fn stage_shapes(gguf: &GgufFile) -> Vec<StageShape> {
    let mut by_class: BTreeMap<&'static str, StageShape> = BTreeMap::new();
    for tensor in &gguf.tensors {
        if tensor.dimensions.len() != 2 {
            continue; // GEMV/GEMM stages are rank-2 weights.
        }
        let class = super::tensor_class(&tensor.name);
        if !is_matmul_stage(class) {
            continue;
        }
        by_class.entry(class).or_insert_with(|| StageShape {
            stage: class.to_string(),
            dims: tensor.dimensions.clone(),
            quant: format!("{:?}", tensor.tensor_type),
        });
    }
    by_class.into_values().collect()
}

/// Total quantized weight bytes read per decoded token — the numerator of the
/// roofline ratio (achieved bytes/s ÷ measured DRAM bytes/s). Sums the matmul
/// weight tensors a decode step streams once.
pub fn decode_weight_bytes(gguf: &GgufFile) -> u64 {
    gguf.tensors
        .iter()
        .filter(|t| is_matmul_stage(super::tensor_class(&t.name)))
        .map(|t| t.n_bytes)
        .sum()
}

fn is_matmul_stage(class: &str) -> bool {
    matches!(
        class,
        "attn_q"
            | "attn_k"
            | "attn_v"
            | "attn_qkv"
            | "attn_output"
            | "ffn_gate"
            | "ffn_up"
            | "ffn_down"
            | "output"
    )
}

/// A configuration to evaluate. `profile` is what gets recorded and later
/// applied; `label` is for evidence/logs. (As the campaign matures, candidates
/// will also carry the per-stage `MANAGED_ENV_KEYS` struct.)
#[derive(Debug, Clone)]
pub struct Candidate {
    pub label: String,
    pub profile: ExecutionProfile,
}

/// The result of timing one candidate: its throughput and a parity token. Two
/// candidates are parity-equal iff their tokens are equal — the gate that
/// disqualifies a fast-but-divergent candidate.
#[derive(Debug, Clone)]
pub struct TrialResult {
    pub tokens_per_s: f64,
    pub parity_token: String,
}

/// Tournament knobs.
#[derive(Debug, Clone, Copy)]
pub struct TournamentConfig {
    /// A candidate must beat baseline by at least this factor to win; otherwise
    /// the tournament fails closed to baseline. Slower-but-correct always wins.
    pub min_speedup: f64,
}

impl Default for TournamentConfig {
    fn default() -> Self {
        Self { min_speedup: 1.05 }
    }
}

/// The evidence a calibration produced — recorded into the gait receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationOutcome {
    pub selected_profile: ExecutionProfile,
    pub reason: String,
    pub baseline_tokens_per_s: f64,
    pub selected_tokens_per_s: f64,
    pub speedup: f64,
    pub roofline_pct: f64,
    /// True when no candidate qualified and the proven baseline was kept.
    pub fell_back: bool,
    /// Candidate labels disqualified for diverging from baseline parity.
    pub parity_disqualified: Vec<String>,
}

fn roofline_pct(tokens_per_s: f64, weight_bytes_per_token: u64, stream_gbs: f64) -> f64 {
    if stream_gbs <= 0.0 || weight_bytes_per_token == 0 {
        return 0.0;
    }
    (tokens_per_s * weight_bytes_per_token as f64) / (stream_gbs * 1e9)
}

/// Run the tournament. `trial` times one candidate and returns its throughput +
/// parity token (or `None` if that candidate could not be run). The baseline is
/// timed first and is the parity reference; every candidate must reproduce its
/// parity token to be eligible. The fastest eligible candidate that clears
/// `min_speedup` wins; otherwise the result fails closed to baseline.
pub fn run_tournament(
    baseline: &Candidate,
    candidates: &[Candidate],
    config: &TournamentConfig,
    weight_bytes_per_token: u64,
    memory: &MemoryMeasurement,
    mut trial: impl FnMut(&Candidate) -> Option<TrialResult>,
) -> CalibrationOutcome {
    let gbs = memory.stream_triad_gbs;

    let base = trial(baseline);
    let base = match base {
        Some(r) if r.tokens_per_s > 0.0 => r,
        _ => {
            return CalibrationOutcome {
                selected_profile: baseline.profile.clone(),
                reason: "baseline trial failed; failing closed to baseline".to_string(),
                baseline_tokens_per_s: 0.0,
                selected_tokens_per_s: 0.0,
                speedup: 1.0,
                roofline_pct: 0.0,
                fell_back: true,
                parity_disqualified: Vec::new(),
            };
        }
    };

    let mut best: Option<(&Candidate, f64)> = None;
    let mut disqualified = Vec::new();
    for candidate in candidates {
        let Some(result) = trial(candidate) else {
            continue; // could not run this candidate — simply skip it
        };
        if result.parity_token != base.parity_token {
            disqualified.push(candidate.label.clone()); // PARITY GATE
            continue;
        }
        match best {
            Some((_, best_tok)) if result.tokens_per_s <= best_tok => {}
            _ => best = Some((candidate, result.tokens_per_s)),
        }
    }

    match best {
        Some((winner, tok)) if tok >= base.tokens_per_s * config.min_speedup => {
            let speedup = tok / base.tokens_per_s;
            CalibrationOutcome {
                selected_profile: winner.profile.clone(),
                reason: format!("gait: {} won at {speedup:.3}x over baseline", winner.label),
                baseline_tokens_per_s: super::round_sig6(base.tokens_per_s),
                selected_tokens_per_s: super::round_sig6(tok),
                speedup: super::round_sig6(speedup),
                roofline_pct: super::round_sig6(roofline_pct(tok, weight_bytes_per_token, gbs)),
                fell_back: false,
                parity_disqualified: disqualified,
            }
        }
        _ => CalibrationOutcome {
            selected_profile: baseline.profile.clone(),
            reason: "no parity-clean candidate beat baseline by margin; keeping baseline"
                .to_string(),
            baseline_tokens_per_s: super::round_sig6(base.tokens_per_s),
            selected_tokens_per_s: super::round_sig6(base.tokens_per_s),
            speedup: 1.0,
            roofline_pct: super::round_sig6(roofline_pct(
                base.tokens_per_s,
                weight_bytes_per_token,
                gbs,
            )),
            fell_back: true,
            parity_disqualified: disqualified,
        },
    }
}

/// End-to-end: fingerprint the model + machine, measure memory, run the
/// tournament, and persist a sealed gait receipt under `dir`. Returns the
/// outcome and the receipt path (`None` if the store write failed — fail-closed,
/// never panics). The caller supplies the timing `trial`.
pub fn calibrate_and_store(
    dir: &Path,
    gguf: &GgufFile,
    baseline: &Candidate,
    candidates: &[Candidate],
    config: &TournamentConfig,
    trial: impl FnMut(&Candidate) -> Option<TrialResult>,
) -> (CalibrationOutcome, Option<PathBuf>) {
    let model_sig = ModelSig::from_gguf(gguf);
    let machine_sig = MachineSig::detect();
    let memory = super::measure_memory();
    let weight_bytes = decode_weight_bytes(gguf);

    let outcome = run_tournament(baseline, candidates, config, weight_bytes, &memory, trial);

    let receipt = GaitReceipt::new(model_sig, machine_sig, outcome.selected_profile.clone())
        .with_memory(memory)
        .with_calibration(outcome.clone());
    let path = store_in(dir, &receipt).ok();
    (outcome, path)
}

/// Convenience: the default store directory for calibration output.
pub fn default_store_dir() -> Option<PathBuf> {
    gait_dir()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};
    use std::collections::BTreeMap as Map;
    use std::path::PathBuf;

    fn descriptor(name: &str, ty: GgufTensorType, dims: Vec<u64>, n_bytes: u64) -> GgufTensorDescriptor {
        GgufTensorDescriptor {
            name: name.to_string(),
            dimensions: dims,
            tensor_type: ty,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes,
        }
    }

    fn sample_gguf() -> GgufFile {
        let mut metadata = Map::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufMetadataValue::String("llama".to_string()),
        );
        GgufFile {
            path: PathBuf::from("sample.gguf"),
            version: 3,
            tensor_count: 4,
            metadata_count: metadata.len() as i64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors: vec![
                descriptor("token_embd.weight", GgufTensorType::Q8_0, vec![16, 128], 4096),
                descriptor("blk.0.attn_q.weight", GgufTensorType::Q8_0, vec![16, 16], 256),
                descriptor("blk.0.ffn_down.weight", GgufTensorType::Q4K, vec![32, 16], 288),
                descriptor("blk.0.attn_norm.weight", GgufTensorType::F32, vec![16], 64),
            ],
        }
    }

    fn cand(label: &str, profile: ExecutionProfile) -> Candidate {
        Candidate { label: label.to_string(), profile }
    }

    fn mem() -> MemoryMeasurement {
        MemoryMeasurement { stream_triad_gbs: 30.0, dram_latency_ns: 80.0 }
    }

    #[test]
    fn stage_shapes_are_matmul_only_and_deduped() {
        let shapes = stage_shapes(&sample_gguf());
        let stages: Vec<&str> = shapes.iter().map(|s| s.stage.as_str()).collect();
        // attn_q and ffn_down are matmul stages; token_embd and norm are not.
        assert!(stages.contains(&"attn_q"));
        assert!(stages.contains(&"ffn_down"));
        assert!(!stages.contains(&"token_embd"));
        assert!(!stages.contains(&"norm"));
    }

    #[test]
    fn decode_weight_bytes_sums_matmul_weights_only() {
        // attn_q (256) + ffn_down (288); token_embd and norm excluded.
        assert_eq!(decode_weight_bytes(&sample_gguf()), 256 + 288);
    }

    #[test]
    fn parity_gate_beats_raw_speed() {
        let baseline = cand("auto", ExecutionProfile::Auto);
        let candidates = vec![
            cand("fast_clean", ExecutionProfile::Experimental),
            cand("faster_diverges", ExecutionProfile::Debug),
            cand("slow_clean", ExecutionProfile::Safe),
        ];
        let outcome = run_tournament(
            &baseline,
            &candidates,
            &TournamentConfig::default(),
            1_000_000,
            &mem(),
            |c| {
                Some(match c.label.as_str() {
                    "auto" => TrialResult { tokens_per_s: 10.0, parity_token: "A".into() },
                    "fast_clean" => TrialResult { tokens_per_s: 13.0, parity_token: "A".into() },
                    // Faster, but DIVERGES — must be disqualified despite winning on speed.
                    "faster_diverges" => TrialResult { tokens_per_s: 20.0, parity_token: "B".into() },
                    "slow_clean" => TrialResult { tokens_per_s: 9.0, parity_token: "A".into() },
                    _ => unreachable!(),
                })
            },
        );
        assert!(!outcome.fell_back);
        assert_eq!(outcome.selected_profile, ExecutionProfile::Experimental);
        assert!((outcome.speedup - 1.3).abs() < 1e-9);
        assert!(outcome.parity_disqualified.contains(&"faster_diverges".to_string()));
        assert!(outcome.roofline_pct > 0.0);
    }

    #[test]
    fn fails_closed_when_nothing_beats_margin() {
        let baseline = cand("auto", ExecutionProfile::Auto);
        let candidates = vec![cand("marginal", ExecutionProfile::Experimental)];
        let outcome = run_tournament(
            &baseline,
            &candidates,
            &TournamentConfig::default(),
            1_000_000,
            &mem(),
            |c| {
                Some(match c.label.as_str() {
                    "auto" => TrialResult { tokens_per_s: 10.0, parity_token: "A".into() },
                    // Only +2%, below the 5% margin.
                    _ => TrialResult { tokens_per_s: 10.2, parity_token: "A".into() },
                })
            },
        );
        assert!(outcome.fell_back);
        assert_eq!(outcome.selected_profile, ExecutionProfile::Auto);
        assert_eq!(outcome.speedup, 1.0);
    }

    #[test]
    fn fails_closed_when_baseline_trial_fails() {
        let baseline = cand("auto", ExecutionProfile::Auto);
        let outcome = run_tournament(
            &baseline,
            &[cand("x", ExecutionProfile::Experimental)],
            &TournamentConfig::default(),
            0,
            &mem(),
            |_| None,
        );
        assert!(outcome.fell_back);
        assert_eq!(outcome.selected_profile, ExecutionProfile::Auto);
    }

    #[test]
    fn calibrate_and_store_persists_a_readable_receipt() {
        let dir = std::env::temp_dir().join(format!("camelid_gait_calib_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let baseline = cand("auto", ExecutionProfile::Auto);
        let candidates = vec![cand("fast_clean", ExecutionProfile::Experimental)];
        let (outcome, path) = calibrate_and_store(
            &dir,
            &sample_gguf(),
            &baseline,
            &candidates,
            &TournamentConfig::default(),
            |c| {
                Some(match c.label.as_str() {
                    "auto" => TrialResult { tokens_per_s: 10.0, parity_token: "A".into() },
                    _ => TrialResult { tokens_per_s: 14.0, parity_token: "A".into() },
                })
            },
        );
        assert_eq!(outcome.selected_profile, ExecutionProfile::Experimental);
        let path = path.expect("receipt stored");
        assert!(path.exists());

        // The stored receipt round-trips and carries the calibration evidence.
        let text = std::fs::read_to_string(&path).unwrap();
        let receipt: GaitReceipt = serde_json::from_str(&text).unwrap();
        assert!(receipt.verify_self_digest());
        assert_eq!(receipt.recorded_profile, ExecutionProfile::Experimental);
        assert_eq!(receipt.calibration.unwrap().selected_profile, ExecutionProfile::Experimental);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
