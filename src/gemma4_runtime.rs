//! Gemma 4 inference runtime — loads a gemma4 GGUF and generates text.
//!
//! The forward math is the one validated bit-for-bit against llama.cpp in
//! `tests/gemma4_forward.rs` (prompt "The capital of France is" → " Paris..."),
//! here driven by an **incremental KV cache**: each [`Gemma4Runtime::step`]
//! processes one token at one position, so the 8GB of Q8 weights are read once
//! per generated token (O(n)) rather than re-prefilled (O(n²)).
//!
//! Weights stay Q8_0 in memory (the model fits in ~8GB; full f32 would not fit a
//! 16GB box); matmuls dequantize on the fly via [`q8_matvec`]. Cross-layer KV
//! sharing: layers >= `first_kv_shared` reuse the last same-type layer's cache.

use crate::gguf::{read_metadata, GgufTensorType};
use crate::inference::gemma4::{gelu_tanh, soft_cap_in_place};
use crate::inference::{
    q4_0_wire_block_dequant, q4_0_wire_row_dot, q6_k_wire_block_dequant, q6_k_wire_row_dot,
    q8_0_wire_row_dot, quantize_q8_0_blocks, quantize_q8_k_blocks,
};
use crate::model::{Gemma4Binding, Gemma4Metadata, LlamaModelConfig};
use crate::tensor::{f16_bits_to_f32, Q8_0Block, TensorStore};
use crate::tokenizer::Tokenizer;
use crate::wire_mmap::GgufWireMmap;
use crate::{BackendError, Result};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

/// Q8_0 wire-block geometry (GGUF on-disk format): 32 quantized values per block,
/// stored as a 2-byte little-endian f16 scale followed by 32 i8 quants = 34 bytes.
const Q8_VALUES_PER_BLOCK: usize = 32;
const Q8_WIRE_BYTES_PER_BLOCK: usize = 34;

/// The wire quant formats the gemma4 CPU runtime reads in place. Q8_0 is the
/// proven baseline lane; Q4_0 and Q6_K are the QAT-row formats (all the QAT
/// linear weights are Q4_0; the tied token/per-layer embeddings are Q6_K).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WireFormat {
    Q8_0,
    Q4_0,
    Q6K,
}

impl WireFormat {
    #[inline]
    fn values_per_block(self) -> usize {
        match self {
            WireFormat::Q8_0 | WireFormat::Q4_0 => 32,
            WireFormat::Q6K => crate::inference::Q6_K_VALUES_PER_BLOCK,
        }
    }

    #[inline]
    fn bytes_per_block(self) -> usize {
        match self {
            WireFormat::Q8_0 => Q8_WIRE_BYTES_PER_BLOCK,
            WireFormat::Q4_0 => crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK,
            WireFormat::Q6K => crate::inference::Q6_K_WIRE_BYTES_PER_BLOCK,
        }
    }
}

/// A quantized weight read straight from the memory-mapped GGUF — no eager
/// decode and no second resident copy. The mmap pages fault in on first touch
/// (during the first generation) and stay in the OS page cache after, so
/// `load()` is ~instant instead of spending ~240s materializing 8GB of decoded
/// blocks up front. Dequant happens inline in the matmul — only the block
/// scale is decoded per block per pass (negligible next to the mul-adds it
/// scales). Any tensor type outside [`WireFormat`] fails closed at load.
struct WireQuant {
    mmap: Arc<GgufWireMmap>,
    offset: u64,
    element_count: usize,
    format: WireFormat,
}

impl WireQuant {
    fn new(store: &TensorStore, mmap: &Arc<GgufWireMmap>, name: &str) -> Result<Self> {
        let desc = store.descriptor(name)?;
        let format = match desc.tensor_type {
            GgufTensorType::Q8_0 => WireFormat::Q8_0,
            GgufTensorType::Q4_0 => WireFormat::Q4_0,
            GgufTensorType::Q6K => WireFormat::Q6K,
            other => {
                return Err(BackendError::UnsupportedTensorType(format!(
                    "tensor {name} is {other:?}; gemma4 wire load supports Q8_0, Q4_0, and Q6_K"
                )))
            }
        };
        let element_count = desc.dimensions.iter().product::<u64>() as usize;
        if !element_count.is_multiple_of(format.values_per_block()) {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {name} element count {element_count} is not block-aligned"
            )));
        }
        let byte_len = element_count / format.values_per_block() * format.bytes_per_block();
        if desc.n_bytes as usize != byte_len {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {name} {format:?} byte size {} != expected {byte_len}",
                desc.n_bytes
            )));
        }
        // Validate the whole tensor range lies inside the mapping once, so the
        // hot-path `bytes()` can index without re-checking.
        mmap.bytes(desc.absolute_offset, byte_len)?;
        Ok(Self {
            mmap: mmap.clone(),
            offset: desc.absolute_offset,
            element_count,
            format,
        })
    }

    /// The tensor's full wire-byte slice. Bounds were validated in `new`.
    #[inline]
    fn bytes(&self) -> &[u8] {
        let byte_len =
            self.element_count / self.format.values_per_block() * self.format.bytes_per_block();
        self.mmap
            .bytes(self.offset, byte_len)
            .expect("wire quant range validated at load")
    }

    #[inline]
    fn block_scale(bytes: &[u8], block: usize) -> f32 {
        let b = block * Q8_WIRE_BYTES_PER_BLOCK;
        f16_bits_to_f32(u16::from_le_bytes([bytes[b], bytes[b + 1]]))
    }

    /// y[o] = sum_i dequant(W[o*in + i]) * x[i]. Rows are block-aligned
    /// (in % 32 == 0). The activation `x` is quantized to Q8 once, then each
    /// output row is a Q8×Q8 NEON `sdot` against the weight row read in place
    /// from the wire bytes ([`q8_0_wire_row_dot`]) — the same fast i8 dot the
    /// Llama path uses, ~Nx the prior scalar f32 mul-add per block. Quantizing
    /// the activation mirrors what llama.cpp does for Q8_0 matmuls, so the
    /// bit-against-llama.cpp parity in `tests/gemma4_forward.rs` is preserved.
    fn matvec(&self, in_dim: usize, out_dim: usize, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), in_dim);
        debug_assert_eq!(
            in_dim % self.format.values_per_block(),
            0,
            "matvec assumes block-aligned rows"
        );
        match self.format {
            WireFormat::Q8_0 | WireFormat::Q4_0 => self.matvec_q(out_dim, &quantize_q8_0_blocks(x)),
            // Q6_K rows dot against Q8_K activations (the reference's K-quant
            // activation format) — used by the QAT tied embedding head.
            WireFormat::Q6K => self.matvec_q8k(out_dim, &quantize_q8_k_blocks(x)),
        }
    }

    /// [`matvec`] against an activation already quantized to Q8 blocks. Lets a
    /// caller that runs several projections off one activation (q/k/v share the
    /// pre-attention norm; gate/up share the pre-FFN norm) quantize it a single
    /// time instead of once per projection.
    ///
    /// Rows are processed in fixed chunks rather than one rayon task per row:
    /// the 262K-vocab output projection would otherwise spawn 262K tiny tasks
    /// per token and pay closure/steal overhead comparable to the ~48-block dot
    /// itself. Each row's dot is unchanged and rows land at fixed indices, so
    /// the result is bit-identical to the per-row version (greedy parity safe).
    fn matvec_q(&self, out_dim: usize, xq: &[Q8_0Block]) -> Vec<f32> {
        const ROW_CHUNK: usize = 64;
        let row_bytes = xq.len() * self.format.bytes_per_block();
        let bytes = self.bytes();
        let row_dot: fn(&[u8], &[Q8_0Block]) -> f32 = match self.format {
            WireFormat::Q8_0 => q8_0_wire_row_dot,
            WireFormat::Q4_0 => q4_0_wire_row_dot,
            WireFormat::Q6K => unreachable!("Q6_K matvec routes through matvec_q8k"),
        };
        let mut out = vec![0f32; out_dim];
        out.par_chunks_mut(ROW_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let base = chunk_idx * ROW_CHUNK;
                for (i, d) in dst.iter_mut().enumerate() {
                    let o = base + i;
                    *d = row_dot(&bytes[o * row_bytes..(o + 1) * row_bytes], xq);
                }
            });
        out
    }

    /// Dot a contiguous range of `out_count` output rows starting at
    /// `row_start`, against a pre-quantized activation — used to project a
    /// single MoE expert's matrix out of a 3D `[in_dim, rows, n_expert]` tensor
    /// (expert e occupies rows `e*rows_per_expert ..`). `in_dim` is implied by
    /// `xq.len() * values_per_block`; each row is `xq.len()` blocks wide.
    fn matvec_q_rows(&self, row_start: usize, out_count: usize, xq: &[Q8_0Block]) -> Vec<f32> {
        const ROW_CHUNK: usize = 64;
        let row_bytes = xq.len() * self.format.bytes_per_block();
        let bytes = self.bytes();
        let row_dot: fn(&[u8], &[Q8_0Block]) -> f32 = match self.format {
            WireFormat::Q8_0 => q8_0_wire_row_dot,
            WireFormat::Q4_0 => q4_0_wire_row_dot,
            WireFormat::Q6K => unreachable!("Q6_K rows route through matvec_q8k"),
        };
        let mut out = vec![0f32; out_count];
        out.par_chunks_mut(ROW_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let base = row_start + chunk_idx * ROW_CHUNK;
                for (i, d) in dst.iter_mut().enumerate() {
                    let o = base + i;
                    *d = row_dot(&bytes[o * row_bytes..(o + 1) * row_bytes], xq);
                }
            });
        out
    }

    /// Batched [`matvec_q`]: dot each output row against EACH of the `xqs`
    /// activations, reading the weight row from the wire bytes ONCE per row and
    /// reusing it across all `xqs`. For K activations this reads the whole weight
    /// matrix once instead of K times — the speculative-decode bandwidth win, since
    /// verifying K draft tokens then costs a single weight pass. The returned
    /// `out[k]` is bit-identical to `matvec_q(out_dim, xqs[k])` (same row_dot, same
    /// order), so greedy parity is preserved.
    fn matmul_q(&self, out_dim: usize, xqs: &[Vec<Q8_0Block>]) -> Vec<Vec<f32>> {
        const ROW_CHUNK: usize = 64;
        let k = xqs.len();
        if k == 0 {
            return Vec::new();
        }
        let row_bytes = xqs[0].len() * self.format.bytes_per_block();
        let bytes = self.bytes();
        let row_dot: fn(&[u8], &[Q8_0Block]) -> f32 = match self.format {
            WireFormat::Q8_0 => q8_0_wire_row_dot,
            WireFormat::Q4_0 => q4_0_wire_row_dot,
            WireFormat::Q6K => unreachable!("Q6_K matmul routes through matmul_q8k"),
        };
        // out[ki][o]; one Vec per activation. Chunk over output rows (the same fixed
        // chunking matvec_q uses) so each weight row is read once and dotted against
        // all k activations. We fill a flat [out_dim * k] buffer in row-chunk order,
        // then transpose into per-activation rows.
        let mut flat = vec![0f32; out_dim * k];
        flat.par_chunks_mut(ROW_CHUNK * k)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let base = chunk_idx * ROW_CHUNK;
                let rows = dst.len() / k;
                for r in 0..rows {
                    let o = base + r;
                    let w = &bytes[o * row_bytes..(o + 1) * row_bytes];
                    for (ki, xq) in xqs.iter().enumerate() {
                        dst[r * k + ki] = row_dot(w, xq);
                    }
                }
            });
        let mut out: Vec<Vec<f32>> = (0..k).map(|_| vec![0f32; out_dim]).collect();
        for o in 0..out_dim {
            for (ki, row) in out.iter_mut().enumerate() {
                row[o] = flat[o * k + ki];
            }
        }
        out
    }

    /// [`matvec`] for Q6_K rows against a Q8_K-quantized activation. Same fixed
    /// row chunking as [`Self::matvec_q`] (greedy-parity-safe ordering).
    fn matvec_q8k(&self, out_dim: usize, xq: &[crate::inference::Q8KBlock]) -> Vec<f32> {
        const ROW_CHUNK: usize = 64;
        let row_bytes = xq.len() * self.format.bytes_per_block();
        let bytes = self.bytes();
        let mut out = vec![0f32; out_dim];
        out.par_chunks_mut(ROW_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let base = chunk_idx * ROW_CHUNK;
                for (i, d) in dst.iter_mut().enumerate() {
                    let o = base + i;
                    *d = q6_k_wire_row_dot(&bytes[o * row_bytes..(o + 1) * row_bytes], xq);
                }
            });
        out
    }

    /// Batched [`matvec_q8k`]: each Q6_K output row is read once and dotted against
    /// every Q8_K activation in `xqs`. The QAT tied head over K verify positions in a
    /// single weight pass; `out[k]` is bit-identical to `matvec_q8k(out_dim, xqs[k])`.
    fn matmul_q8k(&self, out_dim: usize, xqs: &[Vec<crate::inference::Q8KBlock>]) -> Vec<Vec<f32>> {
        const ROW_CHUNK: usize = 64;
        let k = xqs.len();
        if k == 0 {
            return Vec::new();
        }
        let row_bytes = xqs[0].len() * self.format.bytes_per_block();
        let bytes = self.bytes();
        let mut flat = vec![0f32; out_dim * k];
        flat.par_chunks_mut(ROW_CHUNK * k)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let base = chunk_idx * ROW_CHUNK;
                let rows = dst.len() / k;
                for r in 0..rows {
                    let o = base + r;
                    let w = &bytes[o * row_bytes..(o + 1) * row_bytes];
                    for (ki, xq) in xqs.iter().enumerate() {
                        dst[r * k + ki] = q6_k_wire_row_dot(w, xq);
                    }
                }
            });
        let mut out: Vec<Vec<f32>> = (0..k).map(|_| vec![0f32; out_dim]).collect();
        for o in 0..out_dim {
            for (ki, row) in out.iter_mut().enumerate() {
                row[o] = flat[o * k + ki];
            }
        }
        out
    }

    /// Dequantize a contiguous element range [start, start+len) — used for
    /// row-major embedding lookups into vocab-major Q8 tables.
    fn dequantize_elements(&self, start: usize, len: usize) -> Result<Vec<f32>> {
        let end = start.checked_add(len).ok_or_else(|| {
            BackendError::InvalidTensorData("wire dequant range overflows usize".into())
        })?;
        if end > self.element_count {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "wire dequant range {start}..{end} exceeds element count {}",
                self.element_count
            )));
        }
        let bytes = self.bytes();
        let mut out = Vec::with_capacity(len);
        match self.format {
            WireFormat::Q8_0 => {
                const BV: usize = Q8_VALUES_PER_BLOCK;
                const BB: usize = Q8_WIRE_BYTES_PER_BLOCK;
                for e in start..end {
                    let block = e / BV;
                    let within = e % BV;
                    let scale = Self::block_scale(bytes, block);
                    let q = bytes[block * BB + 2 + within] as i8;
                    out.push(scale * q as f32);
                }
            }
            WireFormat::Q4_0 => {
                const BB: usize = crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK;
                let mut block = usize::MAX;
                let mut decoded = [0f32; 32];
                for e in start..end {
                    if e / 32 != block {
                        block = e / 32;
                        decoded = q4_0_wire_block_dequant(&bytes[block * BB..(block + 1) * BB]);
                    }
                    out.push(decoded[e % 32]);
                }
            }
            WireFormat::Q6K => {
                const BV: usize = crate::inference::Q6_K_VALUES_PER_BLOCK;
                const BB: usize = crate::inference::Q6_K_WIRE_BYTES_PER_BLOCK;
                let mut block = usize::MAX;
                let mut decoded = [0f32; BV];
                for e in start..end {
                    if e / BV != block {
                        block = e / BV;
                        decoded = q6_k_wire_block_dequant(&bytes[block * BB..(block + 1) * BB]);
                    }
                    out.push(decoded[e % BV]);
                }
            }
        }
        Ok(out)
    }
}

/// Greedy-decode stop set: the tokenizer's metadata-declared end ids (EOS/EOT/
/// EOM) plus any end-of-turn marker piece present in the vocab. Gemma 4 renamed
/// the marker from Gemma 3's `<end_of_turn>` to `<turn|>` (id 106; all of
/// E2B/E4B/12B), so a single hardcoded spelling misses the stop and the model
/// emits EOG ids forever. The metadata ids are the authoritative contract;
/// llama.cpp stops on the same set.
fn gemma4_stop_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    let sp = &tokenizer.special;
    let mut ids: Vec<u32> = [sp.eos, sp.eot, sp.eom].iter().flatten().copied().collect();
    for marker in ["<turn|>", "<end_of_turn>"] {
        if let Ok(tokens) = tokenizer.encode(marker, false, true) {
            if tokens.len() == 1 {
                ids.push(tokens[0]);
            }
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn f32_matvec(w: &[f32], in_dim: usize, out_dim: usize, x: &[f32]) -> Vec<f32> {
    (0..out_dim)
        .into_par_iter()
        .map(|o| {
            w[o * in_dim..(o + 1) * in_dim]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = (mss + eps).powf(-0.5);
    match weight {
        Some(w) => x.iter().zip(w).map(|(v, w)| v * inv * w).collect(),
        None => x.iter().map(|v| v * inv).collect(),
    }
}

/// Camelid's gemma4 KV cache is f32. The reference's DEFAULT cache is f16
/// (+ flash attention with an f16-rounded Q path), which flips near-tie argmax
/// positions relative to plain-f32 math — llama.cpp's own `-ctk/-ctv/-fa`
/// settings flip the same positions. Parity oracles are therefore captured with
/// the pinned comparator configuration `-ctk f32 -ctv f32 -fa off --no-repack`
/// (the plain-f32 numeric path this runtime implements); the oracle artifacts
/// record that configuration. `f32_to_f16_bits` (tensor module) remains
/// available for cache-precision experiments.
/// RoPE with optional per-frequency factors (GGUF `rope_freqs.weight`).
///
/// Gemma 4 applies the factor table on FULL-attention layers only ("proportional
/// rope", mirroring llama.cpp's `gemma4-iswa`: `freq_factors` is the layer's
/// `rope_freqs` when `!is_swa`, null otherwise). The shipped table is 1.0 for
/// pair indices 0..64 and 1e30 beyond — dividing the frequency by 1e30 zeroes
/// the rotation, so only the first 64 frequency pairs of a global head carry
/// position. Skipping the factors is numerically close on short prompts but is
/// NOT the reference math (it measurably shifts near-tie logits).
fn apply_rope(
    vec: &mut [f32],
    heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
    factors: Option<&[f32]>,
) {
    let half = head_dim / 2;
    for h in 0..heads {
        let base = h * head_dim;
        for i in 0..half {
            let mut freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
            if let Some(factors) = factors {
                freq /= factors[i];
            }
            let (s, c) = (position as f32 * freq).sin_cos();
            let (a, b) = (vec[base + i], vec[base + half + i]);
            vec[base + i] = a * c - b * s;
            vec[base + half + i] = b * c + a * s;
        }
    }
}

struct LayerWeights {
    attn_norm: Vec<f32>,
    attn_q: WireQuant,
    /// `None` on shared-KV layers in trimmed (QAT) exports — never read there.
    attn_k: Option<WireQuant>,
    attn_v: Option<WireQuant>, // None on V-less layers (V = K projection)
    attn_output: WireQuant,
    q_norm: Vec<f32>,
    k_norm: Option<Vec<f32>>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: WireQuant,
    ffn_up: WireQuant,
    ffn_down: WireQuant,
    post_ffw_norm: Vec<f32>,
    // PLE (E-series); inp_gate/proj are small F32 matrices in the GGUF.
    post_norm: Option<Vec<f32>>,
    ple_inp_gate: Option<Vec<f32>>,
    ple_proj: Option<Vec<f32>>,
    ple_output_scale: f32,
    /// Gemma 4 A4B (26B) sparse-expert branch; `None` on dense rows. When
    /// present, the FFN runs the two-branch MoE block (see `MoeWeights`).
    moe: Option<MoeWeights>,
}

/// Sparse 128-expert branch weights for one Gemma 4 A4B MoE layer. The dense
/// `ffn_gate/up/down` on [`LayerWeights`] are the parallel shared-expert MLP.
struct MoeWeights {
    /// Router matrix [n_embd, n_expert], F32, row-major (out=expert).
    gate_inp: Vec<f32>,
    /// Router input scale [n_embd], F32, elementwise.
    gate_inp_scale: Vec<f32>,
    /// Fused per-expert gate‖up, Q4_0 wire; row `e*2*n_ff_exp + o` is expert e
    /// output o (gate for o<n_ff_exp, up for o>=n_ff_exp), in_dim = n_embd.
    gate_up_exps: WireQuant,
    /// Per-expert down, Q4_0 wire; row `e*n_embd + o` is expert e output o,
    /// in_dim = n_ff_exp.
    down_exps: WireQuant,
    /// Per-expert down scale [n_expert], F32, scalar per expert.
    down_exps_scale: Vec<f32>,
    pre_norm_2: Vec<f32>,
    post_norm_1: Vec<f32>,
    post_norm_2: Vec<f32>,
    n_expert: usize,
    n_expert_used: usize,
    n_ff_exp: usize,
}

/// Per-phase CPU decode counters (µs), populated only when
/// `CAMELID_GEMMA4_CPU_TIMING=1`. Printed by `generate_greedy` as an average per
/// step: embedding+PLE prep, attention (proj/rope/scores/output), FFN(+PLE
/// injection), and the 262K-vocab output projection. Diagnostics only — no
/// effect on generated tokens.
static CPU_EMBED_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CPU_ATTN_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CPU_FFN_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CPU_OUTPROJ_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CPU_STEP_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn cpu_timing_enabled() -> bool {
    std::env::var("CAMELID_GEMMA4_CPU_TIMING").is_ok_and(|v| v == "1")
}

fn report_cpu_timing() {
    use std::sync::atomic::Ordering::Relaxed;
    let n = CPU_STEP_N.load(Relaxed).max(1);
    eprintln!(
        "[gemma4-cpu-timing] {n} steps: embed+pli {}us, attention {}us, ffn+ple {}us, output-proj {}us (avg/step)",
        CPU_EMBED_US.load(Relaxed) / n,
        CPU_ATTN_US.load(Relaxed) / n,
        CPU_FFN_US.load(Relaxed) / n,
        CPU_OUTPROJ_US.load(Relaxed) / n,
    );
}

/// A loaded Gemma 4 model ready to generate.
///
/// Supports loading a contiguous **layer range** for distributed layer sharding:
/// a shard holds weights only for `[first_layer, first_layer + layers.len())`,
/// computes its own PLE inputs from the token id (PLE depends only on the token,
/// never on upstream activations), and exchanges the hidden state at the cut
/// point. The full single-node runtime is the `0..block_count` special case.
pub struct Gemma4Runtime {
    config: LlamaModelConfig,
    g: Gemma4Metadata,
    tokenizer: Tokenizer,
    /// Global index of the first locally-loaded layer (0 on a full runtime).
    first_layer: usize,
    layers: Vec<LayerWeights>,
    token_embd: WireQuant,
    per_layer_token_embd: Option<WireQuant>,
    per_layer_model_proj: Option<Vec<f32>>, // BF16 -> f32
    per_layer_proj_norm: Option<Vec<f32>>,
    output_norm: Vec<f32>,
    /// GGUF `rope_freqs.weight` — per-frequency factors applied on FULL
    /// attention layers only (None when absent).
    rope_factors: Option<Vec<f32>>,
    first_kv_shared: usize,
    last_sliding_layer: usize,
    last_full_layer: usize,
}

/// One shard step's result: interior shards hand the hidden state to the next
/// shard; the tail shard (owning the final layer) produces logits.
pub enum Gemma4StepOutput {
    Hidden(Vec<f32>),
    Logits(Vec<f32>),
}

/// Per-layer incremental KV cache: `cache[local_layer][position]` is one
/// position's packed `[kv_heads * head_dim]` K (or V) row.
pub type Gemma4KvCache = Vec<Vec<Vec<f32>>>;

impl Gemma4Runtime {
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_layer_range(path, None)
    }

    /// Load only the given contiguous global layer range (None = all layers).
    /// Fails closed if the range would separate a KV-sharing layer from the
    /// cache it reads (the split must keep every shared layer on the same shard
    /// as its source layer).
    pub fn load_layer_range(path: &Path, range: Option<std::ops::Range<usize>>) -> Result<Self> {
        let gguf = read_metadata(path)?;
        let config = LlamaModelConfig::from_gguf(&gguf)?;
        let g = config.gemma4.clone().ok_or_else(|| {
            BackendError::UnsupportedModelArchitecture("not a gemma4 model".into())
        })?;
        let binding = Gemma4Binding::bind(&gguf, &config)?;
        let store = TensorStore::open(path, &gguf);
        let tokenizer = Tokenizer::from_gguf(&gguf)?;

        let block_count = config.block_count as usize;
        let range = range.unwrap_or(0..block_count);
        if range.start >= range.end || range.end > block_count {
            return Err(BackendError::InvalidModelMetadata(format!(
                "gemma4 layer range {range:?} is invalid for {block_count} layers"
            )));
        }
        // Cross-layer KV sharing constraint: every local layer must read a cache
        // owned by a layer in the same range.
        let plan = g.layer_plan(block_count, config.attention_head_count as usize);
        for l in range.clone() {
            let src = plan[l].kv_source_layer;
            if !range.contains(&src) {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "gemma4 layer range {range:?} separates layer {l} from its shared \
                     KV source layer {src}; choose a split that keeps the trailing \
                     shared-KV block together (first shared source is layer {})",
                    block_count - g.num_kv_shared_layers as usize
                )));
            }
        }

        // Memory-map the GGUF once. Q8 weights are referenced in place (no eager
        // decode); kick off background readahead so the first generation does not
        // pay the whole cold-fault cost serially. The advisory MUST run off the
        // loading thread: on macOS madvise(MADV_WILLNEED) over a USB-backed
        // volume blocks until the kernel has paged in the advised range —
        // observed live as a 12.7 GB 12B mapping stalling a serve-lane model
        // load for 10+ minutes while loading a half-model shard.
        let mmap = GgufWireMmap::map(path)?;
        {
            let mmap = mmap.clone();
            std::thread::spawn(move || mmap.advise_willneed());
        }
        let q8 = |name: &str| WireQuant::new(&store, &mmap, name);
        let f32t = |name: &str| -> Result<Vec<f32>> { Ok(store.load_cpu_f32(name)?.data) };

        let mut layers = Vec::with_capacity(range.len());
        for l in &binding.layers[range.clone()] {
            layers.push(LayerWeights {
                attn_norm: f32t(&l.attn_norm.name)?,
                attn_q: q8(&l.attn_q.name)?,
                attn_k: l.attn_k.as_ref().map(|d| q8(&d.name)).transpose()?,
                attn_v: l.attn_v.as_ref().map(|d| q8(&d.name)).transpose()?,
                attn_output: q8(&l.attn_output.name)?,
                q_norm: f32t(&l.attn_q_norm.name)?,
                k_norm: l.attn_k_norm.as_ref().map(|d| f32t(&d.name)).transpose()?,
                post_attn_norm: f32t(&l.post_attention_norm.name)?,
                ffn_norm: f32t(&l.ffn_norm.name)?,
                ffn_gate: q8(&l.ffn_gate.name)?,
                ffn_up: q8(&l.ffn_up.name)?,
                ffn_down: q8(&l.ffn_down.name)?,
                post_ffw_norm: f32t(&l.post_ffw_norm.name)?,
                post_norm: l.post_norm.as_ref().map(|d| f32t(&d.name)).transpose()?,
                ple_inp_gate: l.ple_inp_gate.as_ref().map(|d| f32t(&d.name)).transpose()?,
                ple_proj: l.ple_proj.as_ref().map(|d| f32t(&d.name)).transpose()?,
                ple_output_scale: l
                    .ple_output_scale
                    .as_ref()
                    .map(|d| f32t(&d.name))
                    .transpose()?
                    .and_then(|v| v.first().copied())
                    .unwrap_or(1.0),
                moe: l
                    .moe
                    .as_ref()
                    .map(|m| -> Result<MoeWeights> {
                        let moe_meta = config.moe.as_ref().ok_or_else(|| {
                            BackendError::InvalidModelMetadata(
                                "gemma4 MoE layer present but no expert metadata".into(),
                            )
                        })?;
                        let n_expert = moe_meta.expert_count as usize;
                        // 2*n_ff_exp = gate_up rows / n_expert; n_ff_exp halves it.
                        let gate_up = q8(&m.gate_up_exps.name)?;
                        let two_nff =
                            gate_up.element_count / (n_expert * config.embedding_length as usize);
                        Ok(MoeWeights {
                            gate_inp: f32t(&m.gate_inp.name)?,
                            gate_inp_scale: f32t(&m.gate_inp_scale.name)?,
                            gate_up_exps: gate_up,
                            down_exps: q8(&m.down_exps.name)?,
                            down_exps_scale: f32t(&m.down_exps_scale.name)?,
                            pre_norm_2: f32t(&m.pre_norm_2.name)?,
                            post_norm_1: f32t(&m.post_norm_1.name)?,
                            post_norm_2: f32t(&m.post_norm_2.name)?,
                            n_expert,
                            n_expert_used: moe_meta.expert_used_count as usize,
                            n_ff_exp: two_nff / 2,
                        })
                    })
                    .transpose()?,
            });
        }

        let first_kv_shared = config.block_count as usize - g.num_kv_shared_layers as usize;
        Ok(Self {
            tokenizer,
            first_layer: range.start,
            token_embd: q8(&binding.token_embedding.name)?,
            per_layer_token_embd: binding
                .per_layer_token_embd
                .as_ref()
                .map(|d| q8(&d.name))
                .transpose()?,
            per_layer_model_proj: binding
                .per_layer_model_proj
                .as_ref()
                .map(|d| f32t(&d.name))
                .transpose()?,
            per_layer_proj_norm: binding
                .per_layer_proj_norm
                .as_ref()
                .map(|d| f32t(&d.name))
                .transpose()?,
            output_norm: f32t(&binding.output_norm.name)?,
            rope_factors: binding
                .rope_freqs
                .as_ref()
                .map(|d| f32t(&d.name))
                .transpose()?,
            first_kv_shared,
            last_sliding_layer: (0..first_kv_shared)
                .rev()
                .find(|&l| g.is_sliding_layer(l))
                .unwrap_or(0),
            last_full_layer: (0..first_kv_shared)
                .rev()
                .find(|&l| !g.is_sliding_layer(l))
                .unwrap_or(0),
            layers,
            config,
            g,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Global layer range loaded on this shard.
    pub fn local_layer_range(&self) -> std::ops::Range<usize> {
        self.first_layer..self.first_layer + self.layers.len()
    }

    pub fn block_count(&self) -> usize {
        self.config.block_count as usize
    }

    pub fn hidden_size(&self) -> usize {
        self.config.embedding_length as usize
    }

    /// Greedy stop set for this model (metadata EOS/EOT/EOM + literal
    /// `<end_of_turn>` when present).
    pub fn stop_token_ids(&self) -> Vec<u32> {
        gemma4_stop_token_ids(&self.tokenizer)
    }

    /// Fresh per-LOCAL-layer KV caches for one sequence.
    pub fn empty_kv_caches(&self) -> (Gemma4KvCache, Gemma4KvCache) {
        (
            vec![Vec::new(); self.layers.len()],
            vec![Vec::new(); self.layers.len()],
        )
    }

    /// Process one token at absolute `pos`, appending its K/V to the per-layer
    /// caches (`kc`/`vc`; only non-shared layers store entries — shared layers read
    /// the last same-type layer's cache, already updated this step). Returns the
    /// next-token logits.
    fn step(
        &self,
        token: u32,
        pos: usize,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
    ) -> Result<Vec<f32>> {
        match self.step_range(token, pos, None, kc, vc)? {
            Gemma4StepOutput::Logits(logits) => Ok(logits),
            Gemma4StepOutput::Hidden(_) => Err(BackendError::InvalidModelMetadata(
                "step() requires a runtime that owns the final layer; use step_range \
                 on interior shards"
                    .into(),
            )),
        }
    }

    /// True when the batched [`Self::step_chunk`] forward is usable: single-node
    /// (this runtime owns every layer including the head) and no MoE layer. The
    /// speculative-decode lane needs the head shard; MoE rows are distributed-only.
    fn supports_chunk_forward(&self) -> bool {
        self.first_layer == 0
            && self.first_layer + self.layers.len() == self.config.block_count as usize
            && self.layers.iter().all(|lw| lw.moe.is_none())
    }

    /// Batched forward over `tokens` at consecutive positions `start_pos +
    /// 0..tokens.len()`, appending all K K/V rows to the caches and returning the
    /// next-token logits at EACH position. Numerically identical to calling
    /// [`Self::step`] once per token (same dots, same order) — the only difference is
    /// that each weight matrix is read ONCE for the whole chunk via [`matmul_q`]
    /// instead of once per token, which is the speculative-decode verify win.
    /// Requires [`Self::supports_chunk_forward`]; caller guarantees it.
    #[allow(clippy::needless_range_loop)]
    fn step_chunk(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
    ) -> Result<Vec<Vec<f32>>> {
        let kk = tokens.len();
        debug_assert!(kk > 0);
        let hidden = self.config.embedding_length as usize;
        let heads = self.config.attention_head_count as usize;
        let ple_dim = self.g.per_layer_input_dim as usize;
        let eps = self.config.rms_norm_epsilon;
        let n_local = self.layers.len();
        let block_count = self.config.block_count as usize;
        let ple_total = block_count * ple_dim;
        let win = self.g.sliding_window as usize;

        // Per-token scaled embedding (== step_range's h0) and the PLE per-layer input.
        let mut hs: Vec<Vec<f32>> = Vec::with_capacity(kk);
        // pli_tok[i][li] is layer li's per-layer input for token i.
        let mut pli_tok: Vec<Vec<Vec<f32>>> = Vec::with_capacity(kk);
        for &token in tokens {
            let h0: Vec<f32> = self
                .token_embd
                .dequantize_elements(token as usize * hidden, hidden)?
                .iter()
                .map(|v| v * (hidden as f32).sqrt())
                .collect();
            let pli: Vec<Vec<f32>> = if let (Some(te), Some(proj), Some(pn)) = (
                self.per_layer_token_embd.as_ref(),
                self.per_layer_model_proj.as_ref(),
                self.per_layer_proj_norm.as_ref(),
            ) {
                let local_span = n_local * ple_dim;
                let ti = te.dequantize_elements(token as usize * ple_total, local_span)?;
                let proj_local = &proj[0..local_span * hidden];
                let ctx = f32_matvec(proj_local, hidden, local_span, &h0);
                let proj_scale = (hidden as f32).powf(-0.5);
                let ple_embed_scale = (ple_dim as f32).sqrt();
                (0..n_local)
                    .map(|li| {
                        let ctx_l: Vec<f32> = (0..ple_dim)
                            .map(|d| ctx[li * ple_dim + d] * proj_scale)
                            .collect();
                        let ctx_n = rms_norm(&ctx_l, Some(pn), eps);
                        (0..ple_dim)
                            .map(|d| {
                                (ctx_n[d] + ti[li * ple_dim + d] * ple_embed_scale)
                                    * std::f32::consts::FRAC_1_SQRT_2
                            })
                            .collect()
                    })
                    .collect()
            } else {
                Vec::new()
            };
            hs.push(h0);
            pli_tok.push(pli);
        }

        for li in 0..n_local {
            let l = li; // single-node: global == local
            let lw = &self.layers[li];
            let sliding = self.g.is_sliding_layer(l);
            let head_dim = self.g.head_dim_at(l) as usize;
            let theta = self.g.rope_freq_base_at(l);
            let kv_heads = self.g.kv_heads_at(l) as usize;
            let ffn_dim = self.g.ffn_length_at(l) as usize;
            let q_dim = heads * head_dim;
            let kv_dim = kv_heads * head_dim;
            let rope_factors = if sliding {
                None
            } else {
                self.rope_factors.as_deref()
            };

            // --- attention projections, batched (one weight pass each) ---
            let xnq: Vec<Vec<Q8_0Block>> = hs
                .iter()
                .map(|h| quantize_q8_0_blocks(&rms_norm(h, Some(&lw.attn_norm), eps)))
                .collect();
            let mut q_rows = lw.attn_q.matmul_q(q_dim, &xnq);
            for q in q_rows.iter_mut() {
                for hh in 0..heads {
                    let s = &mut q[hh * head_dim..(hh + 1) * head_dim];
                    s.copy_from_slice(&rms_norm(s, Some(&lw.q_norm), eps));
                }
            }
            for (i, q) in q_rows.iter_mut().enumerate() {
                apply_rope(q, heads, head_dim, start_pos + i, theta, rope_factors);
            }

            if l < self.first_kv_shared {
                let mut k_rows = lw
                    .attn_k
                    .as_ref()
                    .expect("validate() guarantees owning layers bind attn_k")
                    .matmul_q(kv_dim, &xnq);
                let mut v_rows = match lw.attn_v.as_ref() {
                    Some(wv) => wv.matmul_q(kv_dim, &xnq),
                    None => k_rows.clone(),
                };
                for i in 0..kk {
                    for hh in 0..kv_heads {
                        let s = &mut k_rows[i][hh * head_dim..(hh + 1) * head_dim];
                        s.copy_from_slice(&rms_norm(
                            s,
                            Some(
                                lw.k_norm
                                    .as_deref()
                                    .expect("validate() guarantees owning layers bind attn_k_norm"),
                            ),
                            eps,
                        ));
                        let sv = &mut v_rows[i][hh * head_dim..(hh + 1) * head_dim];
                        sv.copy_from_slice(&rms_norm(sv, None, eps));
                    }
                    apply_rope(
                        &mut k_rows[i],
                        kv_heads,
                        head_dim,
                        start_pos + i,
                        theta,
                        rope_factors,
                    );
                }
                // Append all K rows in position order; query i (below) then reads the
                // cache only up to its own position, so causality holds.
                for i in 0..kk {
                    kc[li].push(std::mem::take(&mut k_rows[i]));
                    vc[li].push(std::mem::take(&mut v_rows[i]));
                }
            }

            let src_global = if l < self.first_kv_shared {
                l
            } else if sliding {
                self.last_sliding_layer
            } else {
                self.last_full_layer
            };
            let src = src_global - self.first_layer;
            let group = heads / self.g.kv_heads_at(src_global) as usize;

            // --- per-position attention (cheap; no big weight read) ---
            let mut attn_q_rows: Vec<Vec<Q8_0Block>> = Vec::with_capacity(kk);
            for i in 0..kk {
                let pos = start_pos + i;
                let lo = if sliding {
                    (pos + 1).saturating_sub(win)
                } else {
                    0
                };
                let q = &q_rows[i];
                let mut attn = vec![0f32; q_dim];
                for hh in 0..heads {
                    let kvh = hh / group;
                    let qh = &q[hh * head_dim..(hh + 1) * head_dim];
                    let mut scores: Vec<f32> = (lo..=pos)
                        .map(|p| {
                            let kp = &kc[src][p][kvh * head_dim..(kvh + 1) * head_dim];
                            qh.iter().zip(kp).map(|(a, b)| a * b).sum()
                        })
                        .collect();
                    let m = scores.iter().cloned().fold(f32::MIN, f32::max);
                    let mut den = 0f32;
                    for s in &mut scores {
                        *s = (*s - m).exp();
                        den += *s;
                    }
                    let out = &mut attn[hh * head_dim..(hh + 1) * head_dim];
                    for (idx, p) in (lo..=pos).enumerate() {
                        let w = scores[idx] / den;
                        let vp = &vc[src][p][kvh * head_dim..(kvh + 1) * head_dim];
                        for d in 0..head_dim {
                            out[d] += w * vp[d];
                        }
                    }
                }
                attn_q_rows.push(quantize_q8_0_blocks(&attn));
            }
            // o-projection batched, then residual + post-attn norm per token.
            let o_rows = lw.attn_output.matmul_q(hidden, &attn_q_rows);
            for i in 0..kk {
                let on = rms_norm(&o_rows[i], Some(&lw.post_attn_norm), eps);
                for (a, b) in hs[i].iter_mut().zip(&on) {
                    *a += b;
                }
            }

            // --- FFN (dense), batched ---
            let ffnq: Vec<Vec<Q8_0Block>> = hs
                .iter()
                .map(|h| quantize_q8_0_blocks(&rms_norm(h, Some(&lw.ffn_norm), eps)))
                .collect();
            let gate_rows = lw.ffn_gate.matmul_q(ffn_dim, &ffnq);
            let up_rows = lw.ffn_up.matmul_q(ffn_dim, &ffnq);
            let actq: Vec<Vec<Q8_0Block>> = (0..kk)
                .map(|i| {
                    let act: Vec<f32> = gate_rows[i]
                        .iter()
                        .zip(&up_rows[i])
                        .map(|(g, u)| gelu_tanh(*g) * u)
                        .collect();
                    quantize_q8_0_blocks(&act)
                })
                .collect();
            let mlp_rows = lw.ffn_down.matmul_q(hidden, &actq);
            for i in 0..kk {
                let ffn_out = rms_norm(&mlp_rows[i], Some(&lw.post_ffw_norm), eps);
                for (a, b) in hs[i].iter_mut().zip(&ffn_out) {
                    *a += b;
                }
                // PLE residual (per token, cheap f32 matvecs).
                if let (Some(ig), Some(pj), Some(pnn)) = (
                    lw.ple_inp_gate.as_ref(),
                    lw.ple_proj.as_ref(),
                    lw.post_norm.as_ref(),
                ) {
                    let mut gated = f32_matvec(ig, hidden, ple_dim, &hs[i]);
                    for (gv, pv) in gated.iter_mut().zip(&pli_tok[i][li]) {
                        *gv = gelu_tanh(*gv) * pv;
                    }
                    let proj = f32_matvec(pj, ple_dim, hidden, &gated);
                    let pnv = rms_norm(&proj, Some(pnn), eps);
                    for (a, b) in hs[i].iter_mut().zip(&pnv) {
                        *a += b;
                    }
                }
                if lw.ple_output_scale != 1.0 {
                    for v in hs[i].iter_mut() {
                        *v *= lw.ple_output_scale;
                    }
                }
            }
        }

        // --- head, batched over the K positions ---
        let vocab = self.config.vocab_size.unwrap() as usize;
        let lastq: Vec<Vec<f32>> = hs
            .iter()
            .map(|h| rms_norm(h, Some(&self.output_norm), eps))
            .collect();
        let mut logits_rows: Vec<Vec<f32>> = match self.token_embd.format {
            WireFormat::Q6K => {
                let xqs: Vec<Vec<crate::inference::Q8KBlock>> =
                    lastq.iter().map(|l| quantize_q8_k_blocks(l)).collect();
                self.token_embd.matmul_q8k(vocab, &xqs)
            }
            _ => {
                let xqs: Vec<Vec<Q8_0Block>> =
                    lastq.iter().map(|l| quantize_q8_0_blocks(l)).collect();
                self.token_embd.matmul_q(vocab, &xqs)
            }
        };
        if let Some(cap) = self.g.final_logit_softcapping {
            for logits in logits_rows.iter_mut() {
                soft_cap_in_place(logits, cap);
            }
        }
        Ok(logits_rows)
    }

    /// One token's forward over the locally-loaded layer range.
    ///
    /// `h_in` is the hidden state arriving from the upstream shard (`None` on
    /// the shard owning layer 0, which embeds the token itself). KV caches are
    /// indexed by LOCAL layer (length `self.layers.len()`). PLE inputs are
    /// recomputed locally from the token id — they depend only on the token's
    /// embedding row, never on upstream activations, so no extra wire traffic.
    /// Returns logits on the shard owning the final layer, otherwise the hidden
    /// state to forward.
    pub fn step_range(
        &self,
        token: u32,
        pos: usize,
        h_in: Option<Vec<f32>>,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
    ) -> Result<Gemma4StepOutput> {
        let hidden = self.config.embedding_length as usize;
        let heads = self.config.attention_head_count as usize;
        let ple_dim = self.g.per_layer_input_dim as usize;
        let eps = self.config.rms_norm_epsilon;
        let n_local = self.layers.len();
        let block_count = self.config.block_count as usize;
        // PLE tables are sized by the GLOBAL layer count.
        let ple_total = block_count * ple_dim;
        let win = self.g.sliding_window as usize;
        let is_tail = self.first_layer + n_local == block_count;

        let timing = cpu_timing_enabled();
        let t_start = std::time::Instant::now();

        // The scaled token embedding: the layer-0 input on the head shard, and
        // the PLE context source on every shard (PLE depends only on the token).
        let h0: Vec<f32> = self
            .token_embd
            .dequantize_elements(token as usize * hidden, hidden)?
            .iter()
            .map(|v| v * (hidden as f32).sqrt())
            .collect();
        let mut h = match h_in {
            Some(h_in) => {
                if h_in.len() != hidden {
                    return Err(BackendError::RuntimeShapeMismatch(format!(
                        "shard received hidden state of {} values, expected {hidden}",
                        h_in.len()
                    )));
                }
                h_in
            }
            None => {
                if self.first_layer != 0 {
                    return Err(BackendError::InvalidModelMetadata(
                        "interior shard requires the upstream hidden state".into(),
                    ));
                }
                h0.clone()
            }
        };

        // Per-layer input (token-identity + context) for the LOCAL layers only:
        // pli[li] belongs to global layer first_layer + li.
        let pli: Vec<Vec<f32>> = if let (Some(te), Some(proj), Some(pn)) = (
            self.per_layer_token_embd.as_ref(),
            self.per_layer_model_proj.as_ref(),
            self.per_layer_proj_norm.as_ref(),
        ) {
            let local_span = n_local * ple_dim;
            let ti = te.dequantize_elements(
                token as usize * ple_total + self.first_layer * ple_dim,
                local_span,
            )?;
            // proj is [ple_total rows x hidden] row-major: take the local rows.
            let proj_local = &proj[self.first_layer * ple_dim * hidden
                ..(self.first_layer * ple_dim + local_span) * hidden];
            let ctx = f32_matvec(proj_local, hidden, local_span, &h0);
            let proj_scale = (hidden as f32).powf(-0.5);
            let ple_embed_scale = (ple_dim as f32).sqrt();
            (0..n_local)
                .map(|li| {
                    let ctx_l: Vec<f32> = (0..ple_dim)
                        .map(|d| ctx[li * ple_dim + d] * proj_scale)
                        .collect();
                    let ctx_n = rms_norm(&ctx_l, Some(pn), eps);
                    (0..ple_dim)
                        .map(|d| {
                            (ctx_n[d] + ti[li * ple_dim + d] * ple_embed_scale)
                                * std::f32::consts::FRAC_1_SQRT_2
                        })
                        .collect()
                })
                .collect()
        } else {
            Vec::new()
        };

        let mut embed_us = t_start.elapsed().as_micros() as u64;
        let (mut attn_us, mut ffn_us) = (0u64, 0u64);

        for li in 0..n_local {
            let t_layer = std::time::Instant::now();
            let l = self.first_layer + li; // global layer index
            let lw = &self.layers[li];
            let sliding = self.g.is_sliding_layer(l);
            let head_dim = self.g.head_dim_at(l) as usize;
            let theta = self.g.rope_freq_base_at(l);
            // Per-layer geometry: 12B varies kv heads across layers, E2B varies
            // the FFN width. Never use the config scalars here.
            let kv_heads = self.g.kv_heads_at(l) as usize;
            let ffn_dim = self.g.ffn_length_at(l) as usize;
            let q_dim = heads * head_dim;
            let kv_dim = kv_heads * head_dim;

            // RoPE frequency factors apply on FULL attention layers only
            // (reference: gemma4-iswa attaches rope_freqs when !is_swa).
            let rope_factors = if sliding {
                None
            } else {
                self.rope_factors.as_deref()
            };

            let xn = rms_norm(&h, Some(&lw.attn_norm), eps);
            // q/k/v all project the same normed input — quantize it once.
            let xnq = quantize_q8_0_blocks(&xn);
            let mut q = lw.attn_q.matvec_q(q_dim, &xnq);
            for hh in 0..heads {
                let s = &mut q[hh * head_dim..(hh + 1) * head_dim];
                s.copy_from_slice(&rms_norm(s, Some(&lw.q_norm), eps));
            }
            apply_rope(&mut q, heads, head_dim, pos, theta, rope_factors);
            // Diagnostics: dump head-0 Q (post-norm/post-rope) for one layer for
            // cross-runtime attention bisection (CAMELID_GEMMA4_DUMP_ATTN=<layer>).
            if std::env::var("CAMELID_GEMMA4_DUMP_ATTN").ok().as_deref() == Some(&l.to_string()) {
                eprintln!(
                    "[attn] pos {pos} layer {l} q0..2 [{:.6}, {:.6}, {:.6}] q64..65 [{:.6}, {:.6}] q128..129 [{:.6}, {:.6}]",
                    q[0], q[1], q[2], q[64], q[65], q[128], q[129]
                );
            }

            if l < self.first_kv_shared {
                let mut k = lw
                    .attn_k
                    .as_ref()
                    .expect("validate() guarantees owning layers bind attn_k")
                    .matvec_q(kv_dim, &xnq);
                // V-less layers (12B full attention) reuse the raw K projection
                // as V — reference: `if v_proj is not present, use Kcur as Vcur`.
                // V then takes the usual weightless norm and never RoPE.
                let mut v = match lw.attn_v.as_ref() {
                    Some(wv) => wv.matvec_q(kv_dim, &xnq),
                    None => k.clone(),
                };
                for hh in 0..kv_heads {
                    let s = &mut k[hh * head_dim..(hh + 1) * head_dim];
                    s.copy_from_slice(&rms_norm(
                        s,
                        Some(
                            lw.k_norm
                                .as_deref()
                                .expect("validate() guarantees owning layers bind attn_k_norm"),
                        ),
                        eps,
                    ));
                    let sv = &mut v[hh * head_dim..(hh + 1) * head_dim];
                    sv.copy_from_slice(&rms_norm(sv, None, eps));
                }
                apply_rope(&mut k, kv_heads, head_dim, pos, theta, rope_factors);
                kc[li].push(k);
                vc[li].push(v);
            }
            // Global source layer, then LOCAL cache index (the load-time range
            // check guarantees the source lives on this shard).
            let src_global = if l < self.first_kv_shared {
                l
            } else if sliding {
                self.last_sliding_layer
            } else {
                self.last_full_layer
            };
            let src = src_global - self.first_layer;
            // GQA group against the cache actually read — the SOURCE layer's
            // geometry when KV is shared.
            let group = heads / self.g.kv_heads_at(src_global) as usize;
            let lo = if sliding {
                (pos + 1).saturating_sub(win)
            } else {
                0
            };
            let mut attn = vec![0f32; q_dim];
            for hh in 0..heads {
                let kvh = hh / group;
                let qh = &q[hh * head_dim..(hh + 1) * head_dim];
                let mut scores: Vec<f32> = (lo..=pos)
                    .map(|p| {
                        let kp = &kc[src][p][kvh * head_dim..(kvh + 1) * head_dim];
                        qh.iter().zip(kp).map(|(a, b)| a * b).sum()
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f32::MIN, f32::max);
                let mut den = 0f32;
                for s in &mut scores {
                    *s = (*s - m).exp();
                    den += *s;
                }
                let out = &mut attn[hh * head_dim..(hh + 1) * head_dim];
                for (idx, p) in (lo..=pos).enumerate() {
                    let w = scores[idx] / den;
                    let vp = &vc[src][p][kvh * head_dim..(kvh + 1) * head_dim];
                    for d in 0..head_dim {
                        out[d] += w * vp[d];
                    }
                }
            }
            let o = lw.attn_output.matvec(q_dim, hidden, &attn);
            let on = rms_norm(&o, Some(&lw.post_attn_norm), eps);
            for (a, b) in h.iter_mut().zip(&on) {
                *a += b;
            }
            attn_us += t_layer.elapsed().as_micros() as u64;
            let t_ffn = std::time::Instant::now();
            // Dense "shared expert" MLP branch (also the whole FFN on dense rows):
            // ffn_norm -> parallel GeGLU -> down. On dense rows this is followed
            // directly by post_ffw_norm + residual; on MoE rows it gets its own
            // post_norm_1 and is summed with the expert branch first.
            let xn = rms_norm(&h, Some(&lw.ffn_norm), eps);
            let xnq = quantize_q8_0_blocks(&xn);
            let gate = lw.ffn_gate.matvec_q(ffn_dim, &xnq);
            let up = lw.ffn_up.matvec_q(ffn_dim, &xnq);
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| gelu_tanh(*g) * u)
                .collect();
            let mut mlp = lw.ffn_down.matvec(ffn_dim, hidden, &act);

            let ffn_out = if let Some(moe) = lw.moe.as_ref() {
                // attn_out is the current `h` (post-attention residual) — every
                // branch reads it, so snapshot before any write.
                let attn_out = &h;
                // Dense branch keeps its own post-norm (post_norm_1).
                mlp = rms_norm(&mlp, Some(&moe.post_norm_1), eps);

                // Router runs on attn_out with its OWN weightless norm, scaled by
                // 1/sqrt(n_embd), then the elementwise gate_inp_scale.
                let mut r = rms_norm(attn_out, None, eps);
                let inv = 1.0f32 / (hidden as f32).sqrt();
                for (rv, sv) in r.iter_mut().zip(&moe.gate_inp_scale) {
                    *rv = *rv * inv * sv;
                }
                let logits = f32_matvec(&moe.gate_inp, hidden, moe.n_expert, &r);
                // softmax over all experts, then top-k by probability.
                let maxl = logits.iter().cloned().fold(f32::MIN, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let sum: f32 = probs.iter().sum();
                for p in probs.iter_mut() {
                    *p /= sum;
                }
                let mut idx: Vec<usize> = (0..moe.n_expert).collect();
                idx.sort_unstable_by(|&a, &b| {
                    probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b))
                });
                idx.truncate(moe.n_expert_used);
                // sum-normalize the selected weights (clamped), w_scale=1.
                let mut wsum: f32 = idx.iter().map(|&e| probs[e]).sum();
                wsum = wsum.max(6.103_515e-5);

                let cur_moe = rms_norm(attn_out, Some(&moe.pre_norm_2), eps);
                let cur_moe_q = quantize_q8_0_blocks(&cur_moe);
                let two_nff = 2 * moe.n_ff_exp;
                let mut moe_acc = vec![0f32; hidden];
                for &e in &idx {
                    let w = probs[e] / wsum;
                    // fused gate‖up for expert e: rows e*2nff .. +2nff, in_dim=n_embd.
                    let gate_up = moe
                        .gate_up_exps
                        .matvec_q_rows(e * two_nff, two_nff, &cur_moe_q);
                    let hexp: Vec<f32> = (0..moe.n_ff_exp)
                        .map(|o| gelu_tanh(gate_up[o]) * gate_up[o + moe.n_ff_exp])
                        .collect();
                    let hexp_q = quantize_q8_0_blocks(&hexp);
                    // down for expert e: rows e*n_embd .. +n_embd, in_dim=n_ff_exp.
                    let y = moe.down_exps.matvec_q_rows(e * hidden, hidden, &hexp_q);
                    let scale = moe.down_exps_scale[e] * w;
                    for (a, yv) in moe_acc.iter_mut().zip(&y) {
                        *a += yv * scale;
                    }
                }
                let cur_moe = rms_norm(&moe_acc, Some(&moe.post_norm_2), eps);

                // combine the two branches, then the shared post_ffw_norm.
                let mut combined = mlp;
                for (c, m) in combined.iter_mut().zip(&cur_moe) {
                    *c += m;
                }
                rms_norm(&combined, Some(&lw.post_ffw_norm), eps)
            } else {
                rms_norm(&mlp, Some(&lw.post_ffw_norm), eps)
            };
            for (a, b) in h.iter_mut().zip(&ffn_out) {
                *a += b;
            }
            if let (Some(ig), Some(pj), Some(pnn)) = (
                lw.ple_inp_gate.as_ref(),
                lw.ple_proj.as_ref(),
                lw.post_norm.as_ref(),
            ) {
                let mut gated = f32_matvec(ig, hidden, ple_dim, &h);
                for (gv, pv) in gated.iter_mut().zip(&pli[li]) {
                    *gv = gelu_tanh(*gv) * pv;
                }
                let proj = f32_matvec(pj, ple_dim, hidden, &gated);
                let pnv = rms_norm(&proj, Some(pnn), eps);
                for (a, b) in h.iter_mut().zip(&pnv) {
                    *a += b;
                }
            }
            // `layer_output_scale` multiplies the layer output UNCONDITIONALLY
            // when present (reference applies it outside the PLE block; the
            // dense 12B carries it on every layer with no PLE at all). 1.0 when
            // the tensor is absent.
            if lw.ple_output_scale != 1.0 {
                for v in h.iter_mut() {
                    *v *= lw.ple_output_scale;
                }
            }
            ffn_us += t_ffn.elapsed().as_micros() as u64;
            // Diagnostics only: per-layer hidden-state fingerprint for
            // cross-runtime layer bisection (CAMELID_GEMMA4_DUMP_LAYERS=1).
            if std::env::var("CAMELID_GEMMA4_DUMP_LAYERS").is_ok_and(|v| v == "1") {
                let l2 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
                eprintln!(
                    "[h] pos {pos} layer {l} l2 {l2:.6} first4 [{:.6}, {:.6}, {:.6}, {:.6}]",
                    h[0], h[1], h[2], h[3]
                );
            }
        }

        if !is_tail {
            return Ok(Gemma4StepOutput::Hidden(h));
        }

        let t_out = std::time::Instant::now();
        let last = rms_norm(&h, Some(&self.output_norm), eps);
        let vocab = self.config.vocab_size.unwrap() as usize;
        // token_embd is vocab-major (row v = the v-th embedding), so the tied
        // logits are a single block-wise Q8 matvec — far faster than per-row
        // dequantize_elements over the whole 262k vocab.
        let mut logits = self.token_embd.matvec(hidden, vocab, &last);
        if let Some(cap) = self.g.final_logit_softcapping {
            soft_cap_in_place(&mut logits, cap);
        }
        if timing {
            use std::sync::atomic::Ordering::Relaxed;
            // The PLE prep ran inside the embed window; attention/ffn windows
            // bracket the per-layer work; everything after the last layer is
            // the output projection (norm + 262K-vocab GEMV + soft-cap).
            embed_us = embed_us.min(t_start.elapsed().as_micros() as u64);
            CPU_EMBED_US.fetch_add(embed_us, Relaxed);
            CPU_ATTN_US.fetch_add(attn_us, Relaxed);
            CPU_FFN_US.fetch_add(ffn_us, Relaxed);
            CPU_OUTPROJ_US.fetch_add(t_out.elapsed().as_micros() as u64, Relaxed);
            CPU_STEP_N.fetch_add(1, Relaxed);
        }
        Ok(Gemma4StepOutput::Logits(logits))
    }

    /// Greedily generate up to `max_new` tokens from `prompt`, with an incremental
    /// KV cache (one forward step per token). Returns (decoded continuation, the
    /// generated token ids).
    #[allow(clippy::explicit_counter_loop)] // `pos` is an absolute sequence index, not a count
    pub fn generate_greedy(&self, prompt: &str, max_new: usize) -> Result<(String, Vec<u32>)> {
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        if std::env::var("CAMELID_GEMMA4_DUMP_PROMPT_TOKENS").is_ok() {
            eprintln!("[prompt tokens] {prompt_tokens:?}");
        }
        let eot = gemma4_stop_token_ids(&self.tokenizer);

        let mut logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.step(tok, pos, &mut kc, &mut vc)?;
        }
        // Lossless n-gram speculative decode (opt-in, single-node non-MoE rows): verify
        // a batch of drafted tokens in ONE weight pass via `step_chunk`. Output is
        // token-for-token identical to the greedy loop below — every committed token is
        // the target's own argmax — so it makes no support/parity claim, only speed.
        if std::env::var("CAMELID_GEMMA4_SPEC_DECODE").is_ok() && self.supports_chunk_forward() {
            let generated =
                self.spec_decode_generate(&mut kc, &mut vc, logits, &prompt_tokens, &eot, max_new)?;
            if cpu_timing_enabled() {
                report_cpu_timing();
            }
            let text = self.tokenizer.decode(&generated, true)?;
            return Ok((text, generated));
        }
        let mut generated = Vec::new();
        let mut pos = prompt_tokens.len();
        for _ in 0..max_new {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            if eot.contains(&next) {
                break;
            }
            generated.push(next);
            logits = self.step(next, pos, &mut kc, &mut vc)?;
            pos += 1;
        }
        if cpu_timing_enabled() {
            report_cpu_timing();
        }
        let text = self.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }

    /// Lossless n-gram speculative decode, forced on (no env var). Returns the SAME
    /// `(text, ids)` as [`Self::generate_greedy`] token-for-token — speculation only
    /// changes how many tokens fall out of one weight read. Requires a single-node
    /// non-MoE row ([`Self::supports_chunk_forward`]); falls back to the plain greedy
    /// loop otherwise. Exposed for the spec-vs-greedy parity test and the CLI flag.
    pub fn generate_greedy_speculative(
        &self,
        prompt: &str,
        max_new: usize,
    ) -> Result<(String, Vec<u32>)> {
        if !self.supports_chunk_forward() {
            return self.generate_greedy(prompt, max_new);
        }
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let mut logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.step(tok, pos, &mut kc, &mut vc)?;
        }
        let generated =
            self.spec_decode_generate(&mut kc, &mut vc, logits, &prompt_tokens, &eot, max_new)?;
        let text = self.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }

    /// Lossless greedy n-gram speculative decode for single-node non-MoE gemma4 rows.
    /// Given the prefilled caches and the prefill `logits` (predicting the first new
    /// position), repeatedly: commit `t0 = argmax(logits)`, draft its continuation from
    /// history (prompt-lookup), verify `[t0, drafts..]` in ONE batched `step_chunk`,
    /// accept the longest prefix of drafts that equals the target's own argmax, roll the
    /// KV cache back to the accepted length, and carry the divergence position's logits
    /// into the next round. Emits exactly the greedy token stream; drafts only change how
    /// many tokens fall out of a single weight read.
    #[allow(clippy::needless_range_loop)]
    fn spec_decode_generate(
        &self,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
        mut logits: Vec<f32>,
        prompt_tokens: &[u32],
        eot: &[u32],
        max_new: usize,
    ) -> Result<Vec<u32>> {
        use crate::inference::speculative::{
            accepted_draft_prefix, NGramDrafter, DEFAULT_NGRAM_DRAFT_TOKENS,
        };
        let max_draft = std::env::var("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NGRAM_DRAFT_TOKENS)
            .max(1);
        let drafter = NGramDrafter::default();
        let argmax = |l: &[f32]| -> u32 {
            l.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap()
        };
        let (mut accepted_rounds, mut accepted_drafts) = (0u64, 0u64);
        let spec_timing = std::env::var("CAMELID_GEMMA4_SPEC_TIMING").is_ok();
        let mut history = prompt_tokens.to_vec();
        let mut generated: Vec<u32> = Vec::new();
        let mut pos = prompt_tokens.len();
        while generated.len() < max_new {
            // t0 is the target's own next-token argmax — always greedy-correct.
            let t0 = argmax(&logits);
            if eot.contains(&t0) {
                break;
            }
            generated.push(t0);
            history.push(t0);
            if generated.len() >= max_new {
                break;
            }
            let budget = max_new - generated.len();
            let drafts = drafter.draft(&history, max_draft.min(budget));
            // Verify [t0, d1..dm] at positions pos..pos+m in one weight pass: rows[i]
            // predicts position pos+i+1.
            let mut chunk = Vec::with_capacity(1 + drafts.len());
            chunk.push(t0);
            chunk.extend_from_slice(&drafts);
            let rows = self.step_chunk(&chunk, pos, kc, vc)?;
            let preds: Vec<u32> = (0..drafts.len()).map(|i| argmax(&rows[i])).collect();
            let j = accepted_draft_prefix(&drafts, &preds);
            accepted_rounds += 1;
            accepted_drafts += j as u64;
            let mut stopped = false;
            for &d in &drafts[..j] {
                if generated.len() >= max_new {
                    break;
                }
                if eot.contains(&d) {
                    stopped = true;
                    break;
                }
                generated.push(d);
                history.push(d);
            }
            if stopped {
                break;
            }
            // Keep KV through the last accepted position (pos+j); discard the rejected
            // draft tail. rows[j] predicts pos+j+1 → it's next round's t0 source.
            let keep = pos + j + 1;
            for li in 0..kc.len() {
                kc[li].truncate(keep);
                vc[li].truncate(keep);
            }
            pos = keep;
            logits = rows.into_iter().nth(j).expect("rows[j] exists");
        }
        if spec_timing {
            let toks = generated.len().max(1) as f64;
            eprintln!(
                "[spec] {} tokens in {accepted_rounds} verify passes ({:.2} tokens/pass; {accepted_drafts} drafts accepted)",
                generated.len(),
                toks / accepted_rounds.max(1) as f64,
            );
        }
        Ok(generated)
    }

    /// Greedy decode that emits the incremental decoded-text delta after each new
    /// token via `on_delta`. The delta is computed by decoding the cumulative
    /// generated sequence and yielding the newly-appended suffix, which keeps
    /// SentencePiece spacing/multi-byte pieces correct (token-at-a-time decode
    /// would mangle them). Returns the same `(text, ids)` as `generate_greedy`.
    #[allow(clippy::explicit_counter_loop)] // `pos` is an absolute sequence index
    pub fn generate_greedy_streaming<F: FnMut(&str)>(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_delta: F,
    ) -> Result<(String, Vec<u32>)> {
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        if std::env::var("CAMELID_GEMMA4_DUMP_PROMPT_TOKENS").is_ok() {
            eprintln!("[prompt tokens] {prompt_tokens:?}");
        }
        let eot = gemma4_stop_token_ids(&self.tokenizer);

        let mut logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.step(tok, pos, &mut kc, &mut vc)?;
        }
        let mut generated = Vec::new();
        let mut emitted = String::new();
        let mut pos = prompt_tokens.len();
        for _ in 0..max_new {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            if eot.contains(&next) {
                break;
            }
            generated.push(next);
            // Decode cumulatively and emit only the newly-appended suffix.
            let full = self.tokenizer.decode(&generated, true)?;
            if let Some(delta) = full.strip_prefix(&emitted) {
                if !delta.is_empty() {
                    on_delta(delta);
                }
            }
            emitted = full;
            logits = self.step(next, pos, &mut kc, &mut vc)?;
            pos += 1;
        }
        Ok((emitted, generated))
    }
}

/// GPU-resident gemma4 decode runtime: the Q8 layer weights live on the GPU (nocopy
/// `WirePages`), the per-layer KV caches persist on the GPU, and each token's forward
/// runs in one Metal command buffer ([`crate::metal::Gemma4ResidentModel`]). The
/// per-token embedding, PLE `pli`, and dual-θ RoPE tables are computed on the CPU and
/// uploaded. Gated by `crate::metal::gemma4_gpu_enabled()` at the call site. Numerics
/// follow the CPU [`Gemma4Runtime`] (attention score scale = 1.0 — gemma folds it in).
#[cfg(target_os = "macos")]
pub struct Gemma4GpuRuntime {
    model: crate::metal::Gemma4ResidentModel,
    tokenizer: Tokenizer,
    g: Gemma4Metadata,
    /// token_embd + per_layer_token_embd stay in the FILE-BACKED mmap (not owned RAM).
    /// The 8GB layer weights are anonymous GPU WirePages; if these embeddings were also
    /// owned/anonymous, the OS would swap the WirePages under 16GB pressure (no file
    /// cache to evict) and the GPU forward would thrash. File-backed pages are evicted
    /// (and cheaply re-read) instead — robust, at the cost of a cold-fault on the
    /// per-token row gather.
    token_embd: WireQuant,
    per_layer_token_embd: Option<WireQuant>,
    /// GGUF `rope_freqs.weight` factors — applied on FULL attention layers'
    /// cos/sin tables only (the reference's proportional rope).
    rope_factors: Option<Vec<f32>>,
    _mmap: Arc<GgufWireMmap>,
    hidden: usize,
    ple_dim: usize,
    n_layers: usize,
    /// QAT hybrid lane: the tied head is Q6_K (no GPU kernel), so the GPU runs the
    /// decoder layers (Q4_0) and the CPU runs the head. False for the all-Q8 path,
    /// where the head is encoded on the GPU inside `forward_token`.
    head_on_cpu: bool,
    /// Held for the CPU head (`head_on_cpu`): output RMS-norm weights + vocab.
    output_norm: Vec<f32>,
    vocab: usize,
    eps: f32,
}

#[cfg(target_os = "macos")]
impl Gemma4GpuRuntime {
    /// Load the model with the Q8 layer weights resident on the GPU. `max_positions`
    /// is the KV-cache capacity (must cover prompt + generated tokens).
    pub fn load(path: &Path, max_positions: usize) -> Result<Self> {
        let gguf = read_metadata(path)?;
        let config = LlamaModelConfig::from_gguf(&gguf)?;
        let g = config.gemma4.clone().ok_or_else(|| {
            BackendError::UnsupportedModelArchitecture("not a gemma4 model".into())
        })?;
        let binding = Gemma4Binding::bind(&gguf, &config)?;
        let store = TensorStore::open(path, &gguf);
        // The GPU-resident decode kernels run the layer projections as either Q8_0
        // (34-byte wire blocks) or Q4_0 (18-byte QAT wire blocks) — both are
        // parity-gated GPU GEMVs. The tied head is read separately: Q8_0 runs on the
        // GPU (inside forward_token); Q6_K (the QAT tied head, no GPU kernel) runs on
        // the CPU via the held WireQuant. Layer 0's attn_q is representative of the
        // projection format (the export quantizes every layer's projections alike).
        let layer_fmt = match store
            .descriptor(&binding.layers[0].attn_q.name)?
            .tensor_type
        {
            GgufTensorType::Q8_0 => crate::metal::GemmaWireFmt::Q8_0,
            GgufTensorType::Q4_0 => crate::metal::GemmaWireFmt::Q4_0,
            other => {
                return Err(BackendError::UnsupportedTensorType(format!(
                    "gemma4 GPU runtime supports Q8_0 or Q4_0 layer projections; \
                     layer 0 attn_q is {other:?}"
                )));
            }
        };
        let head_on_cpu = match store.descriptor(&binding.token_embedding.name)?.tensor_type {
            GgufTensorType::Q8_0 => false, // GPU Q8 head
            GgufTensorType::Q6K => true,   // CPU Q6_K head (QAT tied head)
            other => {
                return Err(BackendError::UnsupportedTensorType(format!(
                    "gemma4 GPU runtime supports a Q8_0 or Q6_K tied head; \
                     token embedding is {other:?}"
                )));
            }
        };
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        // The mmap backs token_embd + per_layer_token_embd (file-backed = evictable, so
        // it never forces the anonymous GPU WirePages to swap). GPU layer weights load
        // separately as page-aligned WirePages.
        let mmap = GgufWireMmap::map(path)?;
        // Warm the embedding mmap off the loading thread (matching the CPU lane): the
        // QAT hybrid head reads the whole Q6_K tied table every token on the CPU, and
        // every row gather hits this mapping, so the first token would otherwise pay the
        // cold page-fault cost serially. madvise(WILLNEED) on a USB-backed volume blocks
        // until the range is paged in, so it MUST NOT run on the loading thread.
        {
            let mmap = mmap.clone();
            std::thread::spawn(move || mmap.advise_willneed());
        }
        let q8 = |name: &str| WireQuant::new(&store, &mmap, name);
        let f32t = |name: &str| -> Result<Vec<f32>> { Ok(store.load_cpu_f32(name)?.data) };

        let hidden = config.embedding_length as usize;
        let heads = config.attention_head_count as usize;
        let n_layers = config.block_count as usize;
        let vocab = config.vocab_size.unwrap() as usize;
        let eps = config.rms_norm_epsilon;
        let ple_dim = g.per_layer_input_dim as usize;
        let softcap = g.final_logit_softcapping.unwrap_or(0.0);

        let file = std::fs::File::open(path).map_err(|e| BackendError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let pages = |name: &str| -> Result<Arc<crate::wire_mmap::WirePages>> {
            let desc = store.descriptor(name)?;
            crate::wire_mmap::WirePages::read_from_file(
                &file,
                desc.absolute_offset,
                desc.n_bytes as usize,
            )
        };

        let plan = g.layer_plan(n_layers, heads);
        let mut layers = Vec::with_capacity(n_layers);
        let mut ple = Vec::with_capacity(n_layers);
        let mut layer_scales = Vec::with_capacity(n_layers);
        let mut owns_kv = Vec::with_capacity(n_layers);
        let mut kv_source = Vec::with_capacity(n_layers);
        for (l, lb) in binding.layers.iter().enumerate() {
            let hd = g.head_dim_at(l) as usize;
            // Per-layer geometry (12B varies kv heads, E2B varies FFN width).
            let kv_heads = plan[l].kv_heads;
            let ffn_dim = g.ffn_length_at(l) as usize;
            let owns = plan[l].owns_kv;
            // Trimmed shared-KV exports (e.g. E4B QAT) omit attn_k / attn_k_norm /
            // attn_v on non-owning layers: those layers project no K/V and run
            // attention against the source layer's cache, so the resident attention
            // never reads these tensors. Pass never-read placeholders to keep the
            // layer shape uniform. A KV-owning layer that omits them is a real error.
            let q_pages_arc = pages(&lb.attn_q.name)?;
            let k_pages_arc = match &lb.attn_k {
                Some(d) => pages(&d.name)?,
                None if !owns => Arc::clone(&q_pages_arc),
                None => {
                    return Err(BackendError::UnsupportedTensorType(format!(
                        "gemma4 GPU runtime requires attn_k on KV-owning layers; \
                         layer {l} omits it"
                    )));
                }
            };
            let k_norm_v = match &lb.attn_k_norm {
                Some(d) => f32t(&d.name)?,
                None if !owns => vec![0.0f32; hd],
                None => {
                    return Err(BackendError::UnsupportedTensorType(format!(
                        "gemma4 GPU runtime requires attn_k_norm on KV-owning layers; \
                         layer {l} omits it"
                    )));
                }
            };
            let layer = crate::metal::Gemma4ResidentLayer::from_wire_pages(
                layer_fmt,
                f32t(&lb.attn_norm.name)?,
                f32t(&lb.attn_q_norm.name)?,
                k_norm_v,
                f32t(&lb.post_attention_norm.name)?,
                f32t(&lb.ffn_norm.name)?,
                f32t(&lb.post_ffw_norm.name)?,
                &q_pages_arc,
                &k_pages_arc,
                lb.attn_v
                    .as_ref()
                    .map(|d| pages(&d.name))
                    .transpose()?
                    .as_ref(),
                &pages(&lb.attn_output.name)?,
                &pages(&lb.ffn_gate.name)?,
                &pages(&lb.ffn_up.name)?,
                &pages(&lb.ffn_down.name)?,
                heads,
                kv_heads,
                hd,
                ffn_dim,
                eps,
            )
            .ok_or_else(|| {
                BackendError::UnsupportedModelArchitecture("Metal unavailable".into())
            })?;
            layers.push(layer);
            // layer_output_scale is unconditional in the reference. E-series
            // layers apply it inside the PLE encode; dense layers (no PLE) get
            // it standalone via `layer_scales`.
            let output_scale = lb
                .ple_output_scale
                .as_ref()
                .map(|d| f32t(&d.name))
                .transpose()?
                .and_then(|v| v.first().copied())
                .unwrap_or(1.0);
            layer_scales.push(output_scale);
            ple.push(match (&lb.ple_inp_gate, &lb.ple_proj, &lb.post_norm) {
                (Some(ig), Some(pj), Some(pn)) => Some(crate::metal::Gemma4ResidentPle {
                    inp_gate: f32t(&ig.name)?,
                    proj: f32t(&pj.name)?,
                    post_norm: f32t(&pn.name)?,
                    output_scale,
                }),
                _ => None,
            });
            owns_kv.push(plan[l].owns_kv);
            kv_source.push(plan[l].kv_source_layer);
        }

        let token_embd = q8(&binding.token_embedding.name)?;
        let output_norm = f32t(&binding.output_norm.name)?;
        // QAT hybrid (Q6_K head on CPU): don't hand the tied table to the GPU head — pass
        // an empty slice so no ~0.5 GB head buffer is uploaded. The all-Q8 lane passes the
        // wire bytes for the GPU head as before.
        let head_wire: &[u8] = if head_on_cpu { &[] } else { token_embd.bytes() };
        let model = crate::metal::Gemma4ResidentModel::new(
            layers,
            ple,
            layer_scales,
            owns_kv,
            kv_source,
            head_wire,
            output_norm.clone(),
            hidden,
            vocab,
            softcap,
            eps,
            max_positions,
            1.0, // gemma folds the attention scale into the (QK-normed) query
        )
        .ok_or_else(|| BackendError::UnsupportedModelArchitecture("Metal unavailable".into()))?;

        let mut model = model;
        let per_layer_model_proj = binding
            .per_layer_model_proj
            .as_ref()
            .map(|d| f32t(&d.name))
            .transpose()?;
        let per_layer_proj_norm = binding
            .per_layer_proj_norm
            .as_ref()
            .map(|d| f32t(&d.name))
            .transpose()?;
        // Move the per-token pli computation onto the GPU (folded-constant matvec +
        // per-head norm + residual-add), eliminating the ~12ms/token CPU prep.
        if let (Some(proj), Some(pn)) = (&per_layer_model_proj, &per_layer_proj_norm) {
            model.set_pli(proj, pn, ple_dim);
        }

        Ok(Self {
            model,
            tokenizer,
            per_layer_token_embd: binding
                .per_layer_token_embd
                .as_ref()
                .map(|d| q8(&d.name))
                .transpose()?,
            rope_factors: binding
                .rope_freqs
                .as_ref()
                .map(|d| f32t(&d.name))
                .transpose()?,
            token_embd,
            g,
            _mmap: mmap,
            hidden,
            ple_dim,
            n_layers,
            head_on_cpu,
            output_norm,
            vocab,
            eps,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Run one token's forward on the GPU and return the next-token logits.
    fn forward(&self, token: u32, position: usize) -> Result<Vec<f32>> {
        let t_prep = std::time::Instant::now();
        let hidden = self.hidden;
        let ple_dim = self.ple_dim;
        let ple_total = self.n_layers * ple_dim;
        let filled = position + 1;
        // Scaled input embedding (CPU gather).
        let h0: Vec<f32> = self
            .token_embd
            .dequantize_elements(token as usize * hidden, hidden)?
            .iter()
            .map(|v| v * (hidden as f32).sqrt())
            .collect();
        // PLE `pli` is computed ON the GPU (Gemma4ResidentModel::set_pli) — the CPU
        // only gathers this token's per_layer_token_embd row, with the gemma constants
        // (ple_dim^0.5 * FRAC_1_SQRT_2) folded in so the GPU just residual-adds it.
        let ti: Vec<f32> = if let Some(te) = self.per_layer_token_embd.as_ref() {
            let scale = (ple_dim as f32).sqrt() * std::f32::consts::FRAC_1_SQRT_2;
            te.dequantize_elements(token as usize * ple_total, ple_total)?
                .iter()
                .map(|v| v * scale)
                .collect()
        } else {
            Vec::new()
        };
        // Per-layer RoPE tables (dual θ, per-type head_dim) + sliding window start.
        let win = self.g.sliding_window as usize;
        let inputs: Vec<crate::metal::Gemma4TokenLayerInput> = (0..self.n_layers)
            .map(|l| {
                let hd = self.g.head_dim_at(l) as usize;
                let theta = self.g.rope_freq_base_at(l);
                let half = hd / 2;
                // Frequency factors (proportional rope) on FULL layers only.
                let factors = if self.g.is_sliding_layer(l) {
                    None
                } else {
                    self.rope_factors.as_deref()
                };
                let (mut cos_t, mut sin_t) = (vec![0f32; half], vec![0f32; half]);
                for i in 0..half {
                    let mut freq = theta.powf(-(2.0 * i as f32) / hd as f32);
                    if let Some(factors) = factors {
                        freq /= factors[i];
                    }
                    let (s, c) = (position as f32 * freq).sin_cos();
                    cos_t[i] = c;
                    sin_t[i] = s;
                }
                let window_start = if self.g.is_sliding_layer(l) {
                    filled.saturating_sub(win)
                } else {
                    0
                };
                crate::metal::Gemma4TokenLayerInput {
                    cos_t,
                    sin_t,
                    pli: Vec::new(), // pli now computed on the GPU; not passed per-layer
                    window_start,
                }
            })
            .collect();
        let prep_us = t_prep.elapsed().as_micros();
        let t_gpu = std::time::Instant::now();
        // All-Q8 path: the GPU encodes the head and returns logits directly. QAT hybrid
        // path: the GPU returns the final hidden state and the CPU runs the Q6_K tied
        // head (rms_norm -> Q6_K logits matvec -> final_logit_softcap), matching the CPU
        // runtime's head exactly.
        let logits = if self.head_on_cpu {
            let last_hidden = self
                .model
                .forward_token_hidden(&h0, &inputs, &ti, position)
                .ok_or_else(|| {
                    BackendError::UnsupportedModelArchitecture("gpu forward failed".into())
                })?;
            let last = rms_norm(&last_hidden, Some(&self.output_norm), self.eps);
            let mut logits = self.token_embd.matvec(self.hidden, self.vocab, &last);
            if let Some(cap) = self.g.final_logit_softcapping {
                soft_cap_in_place(&mut logits, cap);
            }
            logits
        } else {
            self.model
                .forward_token(&h0, &inputs, &ti, position)
                .ok_or_else(|| {
                    BackendError::UnsupportedModelArchitecture("gpu forward failed".into())
                })?
        };
        if std::env::var("CAMELID_GEMMA4_GPU_TIMING").is_ok() {
            PREP_US.fetch_add(prep_us as u64, std::sync::atomic::Ordering::Relaxed);
            GPU_US.fetch_add(
                t_gpu.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            FWD_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(logits)
    }

    /// Greedy generate up to `max_new` tokens from `prompt` on the GPU.
    #[allow(clippy::explicit_counter_loop)] // `pos` is an absolute sequence index
    pub fn generate_greedy(&self, prompt: &str, max_new: usize) -> Result<(String, Vec<u32>)> {
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let mut logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.forward(tok, pos)?;
        }
        let mut generated = Vec::new();
        let mut pos = prompt_tokens.len();
        for _ in 0..max_new {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap();
            if eot.contains(&next) {
                break;
            }
            generated.push(next);
            logits = self.forward(next, pos)?;
            pos += 1;
        }
        if std::env::var("CAMELID_GEMMA4_GPU_TIMING").is_ok() {
            use std::sync::atomic::Ordering::Relaxed;
            let (n, prep, gpu) = (
                FWD_N.load(Relaxed).max(1),
                PREP_US.load(Relaxed),
                GPU_US.load(Relaxed),
            );
            eprintln!(
                "[gpu-timing] {n} forwards: cpu prep avg {}us, gpu avg {}us (total {}us/fwd)",
                prep / n,
                gpu / n,
                (prep + gpu) / n
            );
        }
        let text = self.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }
}

#[cfg(target_os = "macos")]
static PREP_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(target_os = "macos")]
static GPU_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(target_os = "macos")]
static FWD_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
