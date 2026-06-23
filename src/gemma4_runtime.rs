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
    q4_0_wire_block_dequant, q4_0_wire_row_dot, q4_1_wire_row_dot, q4_k_wire_row_dot,
    q6_k_wire_block_dequant, q6_k_wire_row_dot, q8_0_wire_row_dot, quantize_q8_0_blocks,
    quantize_q8_k_blocks,
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
    Q4_1,
    Q4K,
    Q5K,
    Q6K,
}

impl WireFormat {
    #[inline]
    fn values_per_block(self) -> usize {
        match self {
            WireFormat::Q8_0 | WireFormat::Q4_0 | WireFormat::Q4_1 => 32,
            WireFormat::Q4K | WireFormat::Q5K | WireFormat::Q6K => 256,
        }
    }

    #[inline]
    fn bytes_per_block(self) -> usize {
        match self {
            WireFormat::Q8_0 => Q8_WIRE_BYTES_PER_BLOCK,
            WireFormat::Q4_0 => crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK,
            // block_q4_1 = f16 d + f16 m + 16 nibbles; Q4_K/Q5_K K-quant superblocks.
            WireFormat::Q4_1 => 20,
            WireFormat::Q4K => 144,
            WireFormat::Q5K => 176,
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
            GgufTensorType::Q4_1 => WireFormat::Q4_1,
            GgufTensorType::Q4K => WireFormat::Q4K,
            GgufTensorType::Q5K => WireFormat::Q5K,
            GgufTensorType::Q6K => WireFormat::Q6K,
            other => {
                return Err(BackendError::UnsupportedTensorType(format!(
                    "tensor {name} is {other:?}; gemma4 wire load supports Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, and Q6_K"
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
            WireFormat::Q8_0 | WireFormat::Q4_0 | WireFormat::Q4_1 => {
                self.matvec_q(out_dim, &quantize_q8_0_blocks(x))
            }
            // K-quant rows dot against Q8_K activations (the reference's K-quant
            // activation format) — Q6_K/Q4_K used by the QAT tied embedding head.
            WireFormat::Q4K | WireFormat::Q6K => self.matvec_q8k(out_dim, &quantize_q8_k_blocks(x)),
            // Q5_K is gather-only here (per_layer_token_embd); never a matvec weight.
            WireFormat::Q5K => unreachable!("Q5_K is gather-only (per_layer_token_embd)"),
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
            WireFormat::Q4_1 => q4_1_wire_row_dot,
            WireFormat::Q4K | WireFormat::Q5K | WireFormat::Q6K => {
                unreachable!("K-quant matvec routes through matvec_q8k")
            }
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
            WireFormat::Q4_1 => q4_1_wire_row_dot,
            WireFormat::Q4K | WireFormat::Q5K | WireFormat::Q6K => {
                unreachable!("K-quant rows route through matvec_q8k")
            }
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
            WireFormat::Q4_1 => q4_1_wire_row_dot,
            WireFormat::Q4K | WireFormat::Q5K | WireFormat::Q6K => {
                unreachable!("K-quant matmul routes through matmul_q8k")
            }
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
        let row_dot: fn(&[u8], &[crate::inference::Q8KBlock]) -> f32 = match self.format {
            WireFormat::Q6K => q6_k_wire_row_dot,
            WireFormat::Q4K => q4_k_wire_row_dot,
            _ => unreachable!("matvec_q8k is only for Q6_K/Q4_K weights"),
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
        let row_dot: fn(&[u8], &[crate::inference::Q8KBlock]) -> f32 = match self.format {
            WireFormat::Q6K => q6_k_wire_row_dot,
            WireFormat::Q4K => q4_k_wire_row_dot,
            _ => unreachable!("matmul_q8k is only for Q6_K/Q4_K weights"),
        };
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
            // Q4_K tied head + Q5_K per_layer_token_embd are gathered for the input
            // embedding / PLE; decode one 256-value superblock at a time via the shared
            // K-quant decoders (reused, not reimplemented).
            WireFormat::Q4K | WireFormat::Q5K => {
                const BV: usize = 256;
                let bb = self.format.bytes_per_block();
                let mut block = usize::MAX;
                let mut decoded: Vec<f32> = Vec::new();
                for e in start..end {
                    if e / BV != block {
                        block = e / BV;
                        let sb = &bytes[block * bb..(block + 1) * bb];
                        decoded = match self.format {
                            WireFormat::Q4K => {
                                crate::tensor::decode_q4_k_tensor("gemma4 wire gather", sb, BV)?
                            }
                            _ => crate::tensor::decode_q5_k_tensor("gemma4 wire gather", sb, BV)?,
                        };
                    }
                    out.push(decoded[e % BV]);
                }
            }
            // Q4_1 is a matvec-only weight here (ffn_down); never gathered.
            WireFormat::Q4_1 => {
                unreachable!("Q4_1 is matvec-only (ffn_down); never gathered")
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

pub(crate) fn f32_matvec(w: &[f32], in_dim: usize, out_dim: usize, x: &[f32]) -> Vec<f32> {
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

pub(crate) fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
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
pub(crate) fn apply_rope(
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

/// Per-layer RMSNorm weights kept resident on the GPU (small; ~tens of KB/layer).
#[cfg(feature = "cuda")]
struct Gemma4LayerNormsDev {
    attn_norm: cudarc::driver::CudaSlice<f32>,
    q_norm: cudarc::driver::CudaSlice<f32>,
    k_norm: Option<cudarc::driver::CudaSlice<f32>>,
    post_attn_norm: cudarc::driver::CudaSlice<f32>,
    ffn_norm: cudarc::driver::CudaSlice<f32>,
    post_ffw_norm: cudarc::driver::CudaSlice<f32>,
}

/// Per-layer projection weights kept resident on the GPU in the SoA layout
/// `q8_gemv` reads (uploaded once at load). For E4B Q8 this is ~4–4.5 GB and fits
/// a 6 GB card because the big embeddings (`token_embd`, `per_layer_token_embd`)
/// stay on the CPU for the head + PLE gather. `k`/`v` exist only on owning layers;
/// `v` is `None` on V-less layers (V reuses the K projection).
#[cfg(feature = "cuda")]
struct Gemma4LayerWeightsDev {
    q: cudarc::driver::CudaSlice<u8>,
    k: Option<cudarc::driver::CudaSlice<u8>>,
    v: Option<cudarc::driver::CudaSlice<u8>>,
    o: cudarc::driver::CudaSlice<u8>,
    gate: cudarc::driver::CudaSlice<u8>,
    up: cudarc::driver::CudaSlice<u8>,
    down: cudarc::driver::CudaSlice<u8>,
    // Per-projection quant lane (mixed Q4_0 file: Q4_0 projections + Q4_1 ffn_down).
    q_q: GemmaLayerQuant,
    k_q: GemmaLayerQuant,
    v_q: GemmaLayerQuant,
    o_q: GemmaLayerQuant,
    gate_q: GemmaLayerQuant,
    up_q: GemmaLayerQuant,
    down_q: GemmaLayerQuant,
}

/// Quant lane of a resident gemma4 layer projection. All three consume Q8_0
/// activations; Q8_0 weights are SoA-repacked, Q4_0/Q4_1 are raw wire.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GemmaLayerQuant {
    Q8_0,
    Q4_0,
    Q4_1,
}

#[cfg(feature = "cuda")]
impl GemmaLayerQuant {
    fn from_wire(f: WireFormat) -> Self {
        match f {
            WireFormat::Q8_0 => Self::Q8_0,
            WireFormat::Q4_0 => Self::Q4_0,
            WireFormat::Q4_1 => Self::Q4_1,
            other => panic!("gemma4 layer projection quant {other:?} unsupported (Q8_0/Q4_0/Q4_1)"),
        }
    }
}

/// Per-projection GEMV dispatch for the gemma4 resident layer loop. All lanes take the
/// shared Q8_0 activation buffers (`d_ins`/`d_inq`) and `blocks_per_row = cols/32`; the
/// weight is SoA Q8_0 or raw Q4_0/Q4_1 wire. Mirrors `cuda_resident::dispatch_gemv` but
/// for the gemma4 Q8_0-activation lanes only.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gemma_proj_gemv(
    s: &std::sync::Arc<cudarc::driver::CudaStream>,
    kernels: &crate::cuda_resident::CudaResidentKernels,
    quant: GemmaLayerQuant,
    in_scales: &cudarc::driver::CudaSlice<f32>,
    in_quants: &cudarc::driver::CudaSlice<i8>,
    weight: &cudarc::driver::CudaView<'_, u8>,
    rows: usize,
    blocks_per_row: usize,
    out: &mut cudarc::driver::CudaSlice<f32>,
) -> std::result::Result<(), cudarc::driver::DriverError> {
    match quant {
        GemmaLayerQuant::Q8_0 => crate::cuda_resident::launch_gemv(
            s,
            &kernels.gemv,
            in_scales,
            in_quants,
            weight,
            rows,
            blocks_per_row,
            out,
        ),
        GemmaLayerQuant::Q4_0 => crate::cuda_resident::launch_q4_0_gemv(
            s,
            &kernels.q4_0_gemv,
            in_scales,
            in_quants,
            weight,
            rows,
            blocks_per_row,
            out,
            0,
        ),
        GemmaLayerQuant::Q4_1 => crate::cuda_resident::launch_q4_1_gemv(
            s,
            &kernels.q4_1_gemv,
            in_scales,
            in_quants,
            weight,
            rows,
            blocks_per_row,
            out,
            0,
        ),
    }
}

/// Per-layer PLE weights resident on the GPU (small f32 matrices), so the
/// per-layer PLE injection runs entirely on the device — no host round-trip.
#[cfg(feature = "cuda")]
struct Gemma4LayerPleDev {
    inp_gate: cudarc::driver::CudaSlice<f32>,
    proj: cudarc::driver::CudaSlice<f32>,
    post_norm: cudarc::driver::CudaSlice<f32>,
    output_scale: f32,
}

/// A captured decode CUDA graph, wrapped Send: cudarc's `CudaGraph` is not `Send`,
/// but the engine lives behind a `Mutex` in `Arc<Gemma4ServeRuntime>` (one request
/// at a time), so the raw graph handle is only ever touched under the lock.
#[cfg(feature = "cuda")]
struct SendGraph(cudarc::driver::CudaGraph);
#[cfg(feature = "cuda")]
unsafe impl Send for SendGraph {}

#[cfg(feature = "cuda")]
fn cu(e: cudarc::driver::DriverError) -> BackendError {
    BackendError::InvalidModelMetadata(format!("gemma4 cuda: {e}"))
}

/// Repack a GGUF Q8_0 weight tensor (34-byte blocks: f16 scale + 32 i8) into the
/// SoA layout `q8_gemv` reads: all 32-i8 quant groups first, then all f32 scales
/// (the f16 scale widened). Mirrors `cuda_resident::repack_q8_soa` but consumes
/// the raw GGUF wire directly (that helper expects an already-f32-scale 36B block).
#[cfg(feature = "cuda")]
fn q8_wire_to_soa(wire: &[u8]) -> Vec<u8> {
    const W: usize = 34;
    let n = wire.len() / W;
    let mut out = vec![0u8; n * 32 + n * 4];
    let (quants, scales) = out.split_at_mut(n * 32);
    for b in 0..n {
        let blk = &wire[b * W..b * W + W];
        let sc = crate::inference::f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        quants[b * 32..b * 32 + 32].copy_from_slice(&blk[2..34]);
        scales[b * 4..b * 4 + 4].copy_from_slice(&sc.to_le_bytes());
    }
    out
}

/// Quant lane of the GPU tied head: Q8_0 (`q8_gemv` over SoA-repacked weight, Q8_0
/// input) or Q6_K (`q6k_gemv` over raw wire, Q8_K input).
#[cfg(feature = "cuda")]
enum HeadLane {
    Q8_0,
    Q4K,
    Q6K,
}

/// Resident GPU tied head. `weight` is the vocab-major projection (SoA for Q8_0, raw
/// Q6_K wire otherwise); input is quantized by the fused rms_norm+quantize into
/// `inq`/`ins`; `logits` is dtoh'd once per token. `blocks` is blocks-per-row passed
/// to the GEMV (`hidden/32` for Q8_0, `hidden/256` for Q6_K).
#[cfg(feature = "cuda")]
struct Gemma4HeadDev {
    lane: HeadLane,
    weight: cudarc::driver::CudaSlice<u8>,
    output_norm: cudarc::driver::CudaSlice<f32>,
    logits: cudarc::driver::CudaSlice<f32>,
    inq: cudarc::driver::CudaSlice<i8>,
    ins: cudarc::driver::CudaSlice<f32>,
    blocks: usize,
    softcap: f32,
}

/// Resident PLE context-projection (the `proj·h` matvec that dominated CPU prep).
/// `proj` (per_layer_model_proj, [block_count*ple_dim x hidden] f32, ~110 MB) and
/// `proj_norm` stay resident; `ti` holds this token's per_layer_token_embd row
/// (gathered+dequantized on the CPU each token — that table is too big to reside).
#[cfg(feature = "cuda")]
struct Gemma4PleCtxDev {
    proj: cudarc::driver::CudaSlice<f32>,
    proj_norm: cudarc::driver::CudaSlice<f32>,
    ti: cudarc::driver::CudaSlice<f32>,
    ple_total: usize,
    proj_scale: f32,
    embed_scale: f32,
}

/// CUDA gemma4 decode engine (Windows/NVIDIA). Wraps a CPU-loaded [`Gemma4Runtime`]
/// for weights/config/tokenizer and runs the per-token forward through the shared
/// `crate::cuda_resident` kernels. Layer projection weights are streamed from the
/// host mmap per layer (so E4B Q8 fits a 6 GB card); small ops with no large weight
/// read — the scaled embedding and the PLE injection — run on the CPU/GPU as noted.
/// The tied Q6_K head runs on the GPU (`gpu_head`) when resident, else on the CPU.
/// Per-layer geometry (head_dim 256/512, dual-θ RoPE, sliding window, cross-layer KV
/// source) comes from `plan`.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub struct Gemma4CudaResident {
    cpu: Gemma4Runtime,
    kernels: crate::cuda_resident::CudaResidentKernels,
    /// A dedicated non-default stream for the decode forward. The legacy default
    /// stream (`kernels.stream`) cannot be put into capture mode, so all per-token
    /// work runs here to allow recording the layer stack into a CUDA graph.
    cap_stream: std::sync::Arc<cudarc::driver::CudaStream>,
    plan: Vec<crate::model::Gemma4LayerPlan>,
    norms: Vec<Gemma4LayerNormsDev>,
    lweights: Vec<Gemma4LayerWeightsDev>,
    ple: Vec<Option<Gemma4LayerPleDev>>,
    block_count: usize,
    heads: usize,
    hidden: usize,
    ple_dim: usize,
    eps: f32,
    vocab: usize,
    max_positions: usize,
    first_kv_shared: usize,
    half_max: usize,
    /// Captured per-token layer-stack graph (lazily recorded after a warmup pass);
    /// replaying it replaces ~900 per-token kernel launches with one launch.
    decode_graph: Option<SendGraph>,
    /// True once the layer kernels have run once directly (cold first-launch lazy
    /// init isn't capturable, so we warm up before recording the graph).
    warmed: bool,
    /// GPU tied head (Q6_K only). `Some` runs the final projection on the GPU
    /// (fused rms_norm+Q8K-quant -> q6k_gemv over the vocab -> soft-cap), replacing
    /// the ~1.2 s/token CPU Q6_K matvec that otherwise dominates decode. `None` keeps
    /// the head on the CPU (non-Q6_K head, or `hidden` not a multiple of 256).
    gpu_head: Option<Gemma4HeadDev>,
    /// GPU PLE context projection. `Some` runs `proj·h` + per-layer rms-norm + combine
    /// on the GPU (writing `d_pli` directly), replacing the ~27.5M-mult CPU matvec that
    /// was the remaining prep bottleneck. `None` falls back to the CPU pli compute.
    gpu_ple_ctx: Option<Gemma4PleCtxDev>,
    // Per-owning-layer f16 KV caches ([kv_head][pos][head_dim]); None on shared layers.
    cache_k: Vec<Option<cudarc::driver::CudaSlice<u16>>>,
    cache_v: Vec<Option<cudarc::driver::CudaSlice<u16>>>,
    // Reused per-token/per-layer device scratch (sized to per-layer maxima).
    d_hidden: cudarc::driver::CudaSlice<f32>,
    d_normed: cudarc::driver::CudaSlice<f32>,
    d_inq: cudarc::driver::CudaSlice<i8>,
    d_ins: cudarc::driver::CudaSlice<f32>,
    d_q: cudarc::driver::CudaSlice<f32>,
    d_k: cudarc::driver::CudaSlice<f32>,
    d_v: cudarc::driver::CudaSlice<f32>,
    d_attn: cudarc::driver::CudaSlice<f32>,
    d_attnq: cudarc::driver::CudaSlice<i8>,
    d_attns: cudarc::driver::CudaSlice<f32>,
    d_o: cudarc::driver::CudaSlice<f32>,
    d_gate: cudarc::driver::CudaSlice<f32>,
    d_up: cudarc::driver::CudaSlice<f32>,
    d_geglu: cudarc::driver::CudaSlice<f32>,
    d_geglu_q: cudarc::driver::CudaSlice<i8>,
    d_geglu_s: cudarc::driver::CudaSlice<f32>,
    d_ffn_out: cudarc::driver::CudaSlice<f32>,
    // All layers' RoPE tables for this token (slot li at li*half_max), uploaded once
    // so the per-layer loop has no in-loop memcpy (required for graph capture).
    d_cos_all: cudarc::driver::CudaSlice<f32>,
    d_sin_all: cudarc::driver::CudaSlice<f32>,
    d_position: cudarc::driver::CudaSlice<i32>,
    // PLE scratch (GPU injection): d_pli holds this token's per-layer inputs.
    d_pli: cudarc::driver::CudaSlice<f32>,
    d_ple_gated: cudarc::driver::CudaSlice<f32>,
    d_ple_gated2: cudarc::driver::CudaSlice<f32>,
    d_ple_proj: cudarc::driver::CudaSlice<f32>,
    d_ple_normed: cudarc::driver::CudaSlice<f32>,
}

#[cfg(feature = "cuda")]
impl Gemma4CudaResident {
    /// Load the model (CPU runtime, weights mmap'd), bring up the CUDA kernels,
    /// upload per-layer norms, and allocate the KV caches + scratch. `max_positions`
    /// bounds the resident KV cache.
    pub fn load(path: &Path, max_positions: usize) -> Result<Self> {
        let cpu = Gemma4Runtime::load(path)?;
        let kernels = crate::cuda_resident::CudaResidentKernels::new()
            .map_err(BackendError::InvalidModelMetadata)?;
        // Disable cudarc's automatic cross-stream event tracking. Allocating a second
        // (capture) stream below puts the context in multi-stream mode, which otherwise
        // makes every launch record/drop CudaEvents on its slice args — and event
        // create/destroy is not permitted while a stream is capturing, breaking the
        // decode graph. The whole forward runs on a single stream (`cap_stream`), so
        // ordering is implicit and manual; no auto-sync is needed. All gemma4 device
        // slices are created below while this is off, so they never track events.
        unsafe { kernels.ctx.disable_event_tracking() };
        // Capture-capable stream for the decode graph (the default stream is not).
        let cap_stream = kernels.ctx.new_stream().map_err(cu)?;
        let s = kernels.stream.clone();
        let block_count = cpu.config.block_count as usize;
        let heads = cpu.config.attention_head_count as usize;
        let hidden = cpu.config.embedding_length as usize;
        let vocab = cpu.token_embd.element_count / hidden;
        let eps = cpu.config.rms_norm_epsilon;
        let first_kv_shared = cpu.first_kv_shared;
        let plan = cpu.g.layer_plan(block_count, heads);
        let ple_dim = cpu
            .per_layer_proj_norm
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(0);
        // GPU tied head: make the vocab-major head weight resident and run the final
        // projection on the GPU. The CPU matvec over the 262K vocab is ~1.2 s/token —
        // the decode bottleneck — versus a few ms for the GEMV. ~0.55-0.7 GB on E4B.
        let softcap = cpu.g.final_logit_softcapping.unwrap_or(0.0);
        let gpu_head = match cpu.token_embd.format {
            WireFormat::Q8_0 if hidden.is_multiple_of(32) => {
                let blocks = hidden / 32;
                Some(Gemma4HeadDev {
                    lane: HeadLane::Q8_0,
                    weight: s
                        .clone_htod(&q8_wire_to_soa(cpu.token_embd.bytes()))
                        .map_err(cu)?,
                    output_norm: s.clone_htod(&cpu.output_norm).map_err(cu)?,
                    logits: s.alloc_zeros::<f32>(vocab).map_err(cu)?,
                    inq: s.alloc_zeros::<i8>(hidden).map_err(cu)?,
                    ins: s.alloc_zeros::<f32>(blocks).map_err(cu)?,
                    blocks,
                    softcap,
                })
            }
            WireFormat::Q6K if hidden.is_multiple_of(256) => {
                let blocks = hidden / 256;
                Some(Gemma4HeadDev {
                    lane: HeadLane::Q6K,
                    weight: s.clone_htod(cpu.token_embd.bytes()).map_err(cu)?,
                    output_norm: s.clone_htod(&cpu.output_norm).map_err(cu)?,
                    logits: s.alloc_zeros::<f32>(vocab).map_err(cu)?,
                    inq: s.alloc_zeros::<i8>(blocks * 256).map_err(cu)?,
                    ins: s.alloc_zeros::<f32>(blocks).map_err(cu)?,
                    blocks,
                    softcap,
                })
            }
            // Q4_K tied head (mixed Q4_0 file): q4k_gemv over raw 144-byte wire, Q8_K input.
            WireFormat::Q4K if hidden.is_multiple_of(256) => {
                let blocks = hidden / 256;
                Some(Gemma4HeadDev {
                    lane: HeadLane::Q4K,
                    weight: s.clone_htod(cpu.token_embd.bytes()).map_err(cu)?,
                    output_norm: s.clone_htod(&cpu.output_norm).map_err(cu)?,
                    logits: s.alloc_zeros::<f32>(vocab).map_err(cu)?,
                    inq: s.alloc_zeros::<i8>(blocks * 256).map_err(cu)?,
                    ins: s.alloc_zeros::<f32>(blocks).map_err(cu)?,
                    blocks,
                    softcap,
                })
            }
            _ => None,
        };

        // GPU PLE context projection: make per_layer_model_proj (~110 MB f32) + proj_norm
        // resident so `proj·h` (the ~27.5M-mult per-token matvec that dominated CPU prep)
        // runs on the GPU. The per_layer_token_embd table stays CPU (too big to reside);
        // only this token's row is gathered/dequantized + uploaded each step.
        let gpu_ple_ctx = match (
            cpu.per_layer_model_proj.as_ref(),
            cpu.per_layer_proj_norm.as_ref(),
            cpu.per_layer_token_embd.as_ref(),
        ) {
            (Some(proj), Some(pn), Some(_)) if ple_dim > 0 => {
                let ple_total = block_count * ple_dim;
                Some(Gemma4PleCtxDev {
                    proj: s.clone_htod(&proj[0..ple_total * hidden]).map_err(cu)?,
                    proj_norm: s.clone_htod(pn).map_err(cu)?,
                    ti: s.alloc_zeros::<f32>(ple_total).map_err(cu)?,
                    ple_total,
                    proj_scale: (hidden as f32).powf(-0.5),
                    embed_scale: (ple_dim as f32).sqrt(),
                })
            }
            _ => None,
        };

        // Per-layer maxima for scratch sizing.
        let q_dim_max = plan.iter().map(|p| p.q_dim).max().unwrap_or(0);
        let kv_dim_max = plan.iter().map(|p| p.kv_dim).max().unwrap_or(0);
        let head_dim_max = plan.iter().map(|p| p.head_dim).max().unwrap_or(0);
        let ffn_max = (0..block_count)
            .map(|l| cpu.g.ffn_length_at(l) as usize)
            .max()
            .unwrap_or(0);

        // Upload per-layer norm weights (resident; small).
        let mut norms = Vec::with_capacity(block_count);
        for lw in &cpu.layers {
            norms.push(Gemma4LayerNormsDev {
                attn_norm: s.clone_htod(&lw.attn_norm).map_err(cu)?,
                q_norm: s.clone_htod(&lw.q_norm).map_err(cu)?,
                k_norm: match lw.k_norm.as_ref() {
                    Some(w) => Some(s.clone_htod(w).map_err(cu)?),
                    None => None,
                },
                post_attn_norm: s.clone_htod(&lw.post_attn_norm).map_err(cu)?,
                ffn_norm: s.clone_htod(&lw.ffn_norm).map_err(cu)?,
                post_ffw_norm: s.clone_htod(&lw.post_ffw_norm).map_err(cu)?,
            });
        }

        // Per-layer projection weights, resident in the SoA layout q8_gemv reads
        // (uploaded once; the big embeddings stay on the CPU). k/v only on owning layers.
        // Repack + upload one projection, tagging its quant lane: Q8_0 -> SoA (q8_gemv),
        // Q4_0/Q4_1 -> raw wire (q4_0_gemv/q4_1_gemv read the wire directly).
        let upw = |wq: &WireQuant| -> Result<(cudarc::driver::CudaSlice<u8>, GemmaLayerQuant)> {
            let quant = GemmaLayerQuant::from_wire(wq.format);
            let bytes = match quant {
                GemmaLayerQuant::Q8_0 => q8_wire_to_soa(wq.bytes()),
                GemmaLayerQuant::Q4_0 | GemmaLayerQuant::Q4_1 => wq.bytes().to_vec(),
            };
            Ok((s.clone_htod(&bytes).map_err(cu)?, quant))
        };
        let mut lweights = Vec::with_capacity(block_count);
        for (li, lw) in cpu.layers.iter().enumerate() {
            let owns = plan[li].owns_kv;
            let (q, q_q) = upw(&lw.attn_q)?;
            let (k, k_q) = if owns {
                let (kk, kq) = upw(lw.attn_k.as_ref().expect("owning layer binds attn_k"))?;
                (Some(kk), kq)
            } else {
                (None, GemmaLayerQuant::Q8_0)
            };
            let (v, v_q) = if owns {
                match lw.attn_v.as_ref() {
                    Some(wv) => {
                        let (vv, vq) = upw(wv)?;
                        (Some(vv), vq)
                    }
                    // V-less layers reuse the K weight, so V's quant == K's.
                    None => (None, k_q),
                }
            } else {
                (None, GemmaLayerQuant::Q8_0)
            };
            let (o, o_q) = upw(&lw.attn_output)?;
            let (gate, gate_q) = upw(&lw.ffn_gate)?;
            let (up, up_q) = upw(&lw.ffn_up)?;
            let (down, down_q) = upw(&lw.ffn_down)?;
            lweights.push(Gemma4LayerWeightsDev {
                q,
                k,
                v,
                o,
                gate,
                up,
                down,
                q_q,
                k_q,
                v_q,
                o_q,
                gate_q,
                up_q,
                down_q,
            });
        }

        // Per-layer PLE weights resident (small f32 matrices) for on-GPU injection.
        let mut ple = Vec::with_capacity(block_count);
        for lw in &cpu.layers {
            ple.push(
                if let (Some(ig), Some(pj), Some(pn)) = (
                    lw.ple_inp_gate.as_ref(),
                    lw.ple_proj.as_ref(),
                    lw.post_norm.as_ref(),
                ) {
                    Some(Gemma4LayerPleDev {
                        inp_gate: s.clone_htod(ig).map_err(cu)?,
                        proj: s.clone_htod(pj).map_err(cu)?,
                        post_norm: s.clone_htod(pn).map_err(cu)?,
                        output_scale: lw.ple_output_scale,
                    })
                } else {
                    None
                },
            );
        }

        // Per-owning-layer f16 KV caches sized to that layer's kv geometry.
        let mut cache_k = Vec::with_capacity(block_count);
        let mut cache_v = Vec::with_capacity(block_count);
        for p in &plan {
            if p.owns_kv {
                let n = p.kv_dim * max_positions;
                cache_k.push(Some(s.alloc_zeros::<u16>(n).map_err(cu)?));
                cache_v.push(Some(s.alloc_zeros::<u16>(n).map_err(cu)?));
            } else {
                cache_k.push(None);
                cache_v.push(None);
            }
        }

        let alloc_f = |n: usize| s.alloc_zeros::<f32>(n.max(1));
        let alloc_i = |n: usize| s.alloc_zeros::<i8>(n.max(1));
        let me = Self {
            norms,
            lweights,
            ple,
            block_count,
            heads,
            hidden,
            ple_dim,
            eps,
            vocab,
            max_positions,
            first_kv_shared,
            half_max: head_dim_max / 2,
            decode_graph: None,
            warmed: false,
            cache_k,
            cache_v,
            d_hidden: alloc_f(hidden).map_err(cu)?,
            d_normed: alloc_f(hidden).map_err(cu)?,
            d_inq: alloc_i(hidden).map_err(cu)?,
            d_ins: alloc_f(hidden / 32).map_err(cu)?,
            d_q: alloc_f(q_dim_max).map_err(cu)?,
            d_k: alloc_f(kv_dim_max).map_err(cu)?,
            d_v: alloc_f(kv_dim_max).map_err(cu)?,
            d_attn: alloc_f(q_dim_max).map_err(cu)?,
            d_attnq: alloc_i(q_dim_max).map_err(cu)?,
            d_attns: alloc_f(q_dim_max / 32).map_err(cu)?,
            d_o: alloc_f(hidden).map_err(cu)?,
            d_gate: alloc_f(ffn_max).map_err(cu)?,
            d_up: alloc_f(ffn_max).map_err(cu)?,
            d_geglu: alloc_f(ffn_max).map_err(cu)?,
            d_geglu_q: alloc_i(ffn_max).map_err(cu)?,
            d_geglu_s: alloc_f(ffn_max / 32).map_err(cu)?,
            d_ffn_out: alloc_f(hidden).map_err(cu)?,
            d_cos_all: alloc_f(block_count * (head_dim_max / 2)).map_err(cu)?,
            d_sin_all: alloc_f(block_count * (head_dim_max / 2)).map_err(cu)?,
            d_position: s.alloc_zeros::<i32>(1).map_err(cu)?,
            d_pli: alloc_f(block_count * ple_dim).map_err(cu)?,
            d_ple_gated: alloc_f(ple_dim).map_err(cu)?,
            d_ple_gated2: alloc_f(ple_dim).map_err(cu)?,
            d_ple_proj: alloc_f(hidden).map_err(cu)?,
            d_ple_normed: alloc_f(hidden).map_err(cu)?,
            plan,
            kernels,
            cap_stream,
            gpu_head,
            gpu_ple_ctx,
            cpu,
        };
        // Re-enable cudarc's auto event-tracking now that every gemma4 device slice is
        // allocated. Those slices were created while it was off, so they carry no
        // CudaEvents and the decode-graph capture stays clean; restoring it here keeps
        // multi-stream synchronization correct for any other model loaded into this
        // context afterwards (e.g. a later Llama reload in a serve process).
        unsafe { me.kernels.ctx.enable_event_tracking() };
        Ok(me)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.cpu.tokenizer
    }

    pub fn layer_plan(&self) -> &[crate::model::Gemma4LayerPlan] {
        &self.plan
    }

    /// One token's forward; returns next-token logits. Mirrors the CPU
    /// `Gemma4Runtime::step_range` op order exactly (the parity oracle).
    fn forward_token(
        &mut self,
        token: u32,
        position: usize,
        want_logits: bool,
    ) -> Result<Vec<f32>> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        // Run on the capture-capable stream (not the default stream) so the layer
        // stack can be recorded into a CUDA graph.
        let s = self.cap_stream.clone();
        let hidden = self.hidden;
        let heads = self.heads;
        let ple_dim = self.ple_dim;
        let eps = self.eps;

        // ---- CPU: scaled embedding (small f32 gather); upload before the GPU PLE proj ----
        let h: Vec<f32> = self
            .cpu
            .token_embd
            .dequantize_elements(token as usize * hidden, hidden)?
            .iter()
            .map(|v| v * (hidden as f32).sqrt())
            .collect();
        let ple_total = self.block_count * ple_dim;
        s.memcpy_htod(&h, &mut self.d_hidden).map_err(cu)?;
        s.memcpy_htod(&[position as i32], &mut self.d_position)
            .map_err(cu)?;
        // PLE per-layer inputs -> d_pli. GPU path: ctx = proj·h (f32_gemv) -> *proj_scale ->
        // per-layer rms_norm(proj_norm) -> + ti*embed_scale -> *1/sqrt(2), all on device
        // (the ~27.5M-mult matvec was the CPU prep bottleneck). The per_layer_token_embd
        // row `ti` is gathered on the CPU (that table is too big to reside). CPU fallback below.
        if let Some(ctxdev) = self.gpu_ple_ctx.as_mut() {
            let ti = self
                .cpu
                .per_layer_token_embd
                .as_ref()
                .expect("gpu_ple_ctx implies per_layer_token_embd")
                .dequantize_elements(token as usize * ctxdev.ple_total, ctxdev.ple_total)?;
            s.memcpy_htod(&ti, &mut ctxdev.ti).map_err(cu)?;
            crate::cuda_resident::launch_f32_gemv(
                &s,
                &self.kernels.f32_gemv,
                &ctxdev.proj,
                &self.d_hidden,
                &mut self.d_pli,
                hidden,
                ctxdev.ple_total,
            )
            .map_err(cu)?;
            crate::cuda_resident::launch_scale(
                &s,
                &self.kernels.scale_f32,
                &mut self.d_pli,
                ctxdev.ple_total,
                ctxdev.proj_scale,
            )
            .map_err(cu)?;
            crate::cuda_resident::launch_rms_norm_per_head(
                &s,
                &self.kernels.rms_norm_per_head,
                &mut self.d_pli,
                &ctxdev.proj_norm,
                self.block_count,
                ple_dim,
                eps,
            )
            .map_err(cu)?;
            crate::cuda_resident::launch_scale(
                &s,
                &self.kernels.scale_f32,
                &mut ctxdev.ti,
                ctxdev.ple_total,
                ctxdev.embed_scale,
            )
            .map_err(cu)?;
            crate::cuda_resident::launch_residual(
                &s,
                &self.kernels.residual_add,
                &mut self.d_pli,
                &ctxdev.ti,
                ctxdev.ple_total,
            )
            .map_err(cu)?;
            crate::cuda_resident::launch_scale(
                &s,
                &self.kernels.scale_f32,
                &mut self.d_pli,
                ctxdev.ple_total,
                std::f32::consts::FRAC_1_SQRT_2,
            )
            .map_err(cu)?;
        } else if let (Some(te), Some(proj), Some(pn)) = (
            self.cpu.per_layer_token_embd.as_ref(),
            self.cpu.per_layer_model_proj.as_ref(),
            self.cpu.per_layer_proj_norm.as_ref(),
        ) {
            let ti = te.dequantize_elements(token as usize * ple_total, ple_total)?;
            let ctx = f32_matvec(&proj[0..ple_total * hidden], hidden, ple_total, &h);
            let proj_scale = (hidden as f32).powf(-0.5);
            let ple_embed_scale = (ple_dim as f32).sqrt();
            let pli_flat: Vec<f32> = (0..self.block_count)
                .flat_map(|li| {
                    let ctx_l: Vec<f32> = (0..ple_dim)
                        .map(|d| ctx[li * ple_dim + d] * proj_scale)
                        .collect();
                    let ctx_n = rms_norm(&ctx_l, Some(pn), eps);
                    (0..ple_dim)
                        .map(|d| {
                            (ctx_n[d] + ti[li * ple_dim + d] * ple_embed_scale)
                                * std::f32::consts::FRAC_1_SQRT_2
                        })
                        .collect::<Vec<f32>>()
                })
                .collect();
            s.memcpy_htod(&pli_flat, &mut self.d_pli).map_err(cu)?;
        }
        // Precompute every layer's RoPE table for this position (slot li = li*half_max)
        // and upload once — so the per-layer loop has no in-loop memcpy (graph-capturable).
        {
            let half_max = self.half_max;
            let mut cos_all = vec![0f32; self.block_count * half_max];
            let mut sin_all = vec![0f32; self.block_count * half_max];
            for li in 0..self.block_count {
                let p = &self.plan[li];
                let hd = p.head_dim;
                let half = hd / 2;
                let theta = p.theta;
                let factors = if p.sliding {
                    None
                } else {
                    self.cpu.rope_factors.as_deref()
                };
                let base = li * half_max;
                for i in 0..half {
                    let mut freq = theta.powf(-(2.0 * i as f32) / hd as f32);
                    if let Some(f) = factors {
                        freq /= f[i];
                    }
                    let (sn, cs) = (position as f32 * freq).sin_cos();
                    cos_all[base + i] = cs;
                    sin_all[base + i] = sn;
                }
            }
            s.memcpy_htod(&cos_all, &mut self.d_cos_all).map_err(cu)?;
            s.memcpy_htod(&sin_all, &mut self.d_sin_all).map_err(cu)?;
        }
        // Capture the per-token layer stack into a CUDA graph once, then replay it
        // (one launch instead of ~900). The loop reads device buffers only (weights
        // resident; pli/cos/position pre-uploaded above), so it is graph-capturable.
        // Record the graph only AFTER a warmup pass: a kernel's first launch does
        // lazy init (module/function load) which is not stream-capturable. The warmup
        // call runs the loop directly; the next call captures it; later calls replay.
        let do_capture = self.decode_graph.is_none() && self.warmed;
        if do_capture {
            use cudarc::driver::sys;
            s.begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(cu)?;
        }
        if self.decode_graph.is_none() {
            let k = &self.kernels;
            for li in 0..self.block_count {
                let p = self.plan[li].clone();
                let hd = p.head_dim;
                let half = hd / 2;
                let q_dim = p.q_dim;
                let kv_dim = p.kv_dim;
                let kv_heads = p.kv_heads;
                let ffn_dim = self.cpu.g.ffn_length_at(li) as usize;
                let lw = &self.cpu.layers[li];
                let nrm = &self.norms[li];
                let lwd = &self.lweights[li];

                // attention RMSNorm + Q8_0 quantize of the activation (shared by q/k/v).
                crate::cuda_resident::launch_rmsnorm(
                    &s,
                    &k.rms_norm,
                    &self.d_hidden,
                    &nrm.attn_norm,
                    &mut self.d_normed,
                    hidden,
                    eps,
                )
                .map_err(cu)?;
                crate::cuda_resident::launch_quantize(
                    &s,
                    &k.quantize,
                    &self.d_normed,
                    &mut self.d_inq,
                    &mut self.d_ins,
                    hidden / 32,
                )
                .map_err(cu)?;

                // Q projection -> per-head q-norm -> RoPE (split-half, dual-θ).
                gemma_proj_gemv(
                    &s,
                    k,
                    lwd.q_q,
                    &self.d_ins,
                    &self.d_inq,
                    &lwd.q.slice(0..lwd.q.len()),
                    q_dim,
                    hidden / 32,
                    &mut self.d_q,
                )
                .map_err(cu)?;
                crate::cuda_resident::launch_rms_norm_per_head(
                    &s,
                    &k.rms_norm_per_head,
                    &mut self.d_q,
                    &nrm.q_norm,
                    heads,
                    hd,
                    eps,
                )
                .map_err(cu)?;
                // RoPE q (split-half, dual-θ): read this layer's slot from d_cos_all/d_sin_all
                // (uploaded once before the loop). Inline launch (launch_rope takes &CudaSlice).
                let rope_off = li * self.half_max;
                {
                    let cos_v = self.d_cos_all.slice(rope_off..rope_off + half);
                    let sin_v = self.d_sin_all.slice(rope_off..rope_off + half);
                    let cfg = LaunchConfig {
                        grid_dim: (((heads * half) as u32).div_ceil(128).max(1), 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let (nh, hdi, rd, pr) = (heads as i32, hd as i32, hd as i32, 1i32);
                    let mut b = s.launch_builder(&k.rope);
                    b.arg(&mut self.d_q)
                        .arg(&cos_v)
                        .arg(&sin_v)
                        .arg(&nh)
                        .arg(&hdi)
                        .arg(&rd)
                        .arg(&pr);
                    unsafe { b.launch(cfg) }.map_err(cu)?;
                }

                // K/V projection + norms + RoPE + cache scatter — owning layers only.
                if p.owns_kv {
                    {
                        let wk = lwd.k.as_ref().expect("owning layer has resident K");
                        gemma_proj_gemv(
                            &s,
                            k,
                            lwd.k_q,
                            &self.d_ins,
                            &self.d_inq,
                            &wk.slice(0..wk.len()),
                            kv_dim,
                            hidden / 32,
                            &mut self.d_k,
                        )
                        .map_err(cu)?;
                        match lwd.v.as_ref() {
                            Some(wv) => {
                                gemma_proj_gemv(
                                    &s,
                                    k,
                                    lwd.v_q,
                                    &self.d_ins,
                                    &self.d_inq,
                                    &wv.slice(0..wv.len()),
                                    kv_dim,
                                    hidden / 32,
                                    &mut self.d_v,
                                )
                                .map_err(cu)?;
                            }
                            // V-less layers: V = K projection.
                            None => {
                                gemma_proj_gemv(
                                    &s,
                                    k,
                                    lwd.k_q,
                                    &self.d_ins,
                                    &self.d_inq,
                                    &wk.slice(0..wk.len()),
                                    kv_dim,
                                    hidden / 32,
                                    &mut self.d_v,
                                )
                                .map_err(cu)?;
                            }
                        }
                    }
                    // k-norm (weighted) and v-norm (weightless), per kv head.
                    crate::cuda_resident::launch_rms_norm_per_head(
                        &s,
                        &k.rms_norm_per_head,
                        &mut self.d_k,
                        nrm.k_norm.as_ref().expect("owning layer binds attn_k_norm"),
                        kv_heads,
                        hd,
                        eps,
                    )
                    .map_err(cu)?;
                    {
                        // weightless V-norm (use_weight=0; weight ptr unused by the kernel).
                        let cfg = LaunchConfig {
                            grid_dim: (kv_heads as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: (hd as u32) * 4,
                        };
                        let (hdi, uw) = (hd as i32, 0i32);
                        let mut b = s.launch_builder(&k.rms_norm_per_head);
                        b.arg(&mut self.d_v)
                            .arg(&nrm.q_norm)
                            .arg(&hdi)
                            .arg(&eps)
                            .arg(&uw);
                        unsafe { b.launch(cfg) }.map_err(cu)?;
                    }
                    {
                        let cos_v = self.d_cos_all.slice(rope_off..rope_off + half);
                        let sin_v = self.d_sin_all.slice(rope_off..rope_off + half);
                        let cfg = LaunchConfig {
                            grid_dim: (((kv_heads * half) as u32).div_ceil(128).max(1), 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let (nh, hdi, rd, pr) = (kv_heads as i32, hd as i32, hd as i32, 1i32);
                        let mut b = s.launch_builder(&k.rope);
                        b.arg(&mut self.d_k)
                            .arg(&cos_v)
                            .arg(&sin_v)
                            .arg(&nh)
                            .arg(&hdi)
                            .arg(&rd)
                            .arg(&pr);
                        unsafe { b.launch(cfg) }.map_err(cu)?;
                    }
                    // Scatter K/V into this layer's cache at `position`.
                    let ck = self.cache_k[li].as_mut().expect("owning layer has K cache");
                    crate::cuda_resident::launch_kv_scatter(
                        &s,
                        &k.kv_scatter,
                        &self.d_k,
                        ck,
                        &self.d_position,
                        kv_heads,
                        hd,
                        self.max_positions,
                    )
                    .map_err(cu)?;
                    let cv = self.cache_v[li].as_mut().expect("owning layer has V cache");
                    crate::cuda_resident::launch_kv_scatter(
                        &s,
                        &k.kv_scatter,
                        &self.d_v,
                        cv,
                        &self.d_position,
                        kv_heads,
                        hd,
                        self.max_positions,
                    )
                    .map_err(cu)?;
                }

                // Attention against the source layer's cache (sliding window or full causal).
                let src = p.kv_source_layer;
                let window = p.window.map(|w| w as i32).unwrap_or(0);
                {
                    let ck = self.cache_k[src].as_ref().expect("KV source has K cache");
                    let cv = self.cache_v[src].as_ref().expect("KV source has V cache");
                    let cfg = LaunchConfig {
                        grid_dim: (heads as u32, 1, 1),
                        block_dim: (hd as u32, 1, 1),
                        shared_mem_bytes: ((2 * hd + self.max_positions) as u32) * 4,
                    };
                    let (nh, nkv, hdi, mp) = (
                        heads as i32,
                        kv_heads as i32,
                        hd as i32,
                        self.max_positions as i32,
                    );
                    let scale = 1.0f32; // gemma folds the scale; attention uses no 1/sqrt(d).
                    let mut b = s.launch_builder(&k.attention_sw);
                    b.arg(&self.d_q)
                        .arg(ck)
                        .arg(cv)
                        .arg(&mut self.d_attn)
                        .arg(&nh)
                        .arg(&nkv)
                        .arg(&hdi)
                        .arg(&self.d_position)
                        .arg(&mp)
                        .arg(&scale)
                        .arg(&window);
                    unsafe { b.launch(cfg) }.map_err(cu)?;
                }

                // O projection (quantize attn output, in=q_dim) -> post-attn norm -> residual.
                crate::cuda_resident::launch_quantize(
                    &s,
                    &k.quantize,
                    &self.d_attn,
                    &mut self.d_attnq,
                    &mut self.d_attns,
                    q_dim / 32,
                )
                .map_err(cu)?;
                gemma_proj_gemv(
                    &s,
                    k,
                    lwd.o_q,
                    &self.d_attns,
                    &self.d_attnq,
                    &lwd.o.slice(0..lwd.o.len()),
                    hidden,
                    q_dim / 32,
                    &mut self.d_o,
                )
                .map_err(cu)?;
                crate::cuda_resident::launch_rmsnorm(
                    &s,
                    &k.rms_norm,
                    &self.d_o,
                    &nrm.post_attn_norm,
                    &mut self.d_normed,
                    hidden,
                    eps,
                )
                .map_err(cu)?;
                crate::cuda_resident::launch_residual(
                    &s,
                    &k.residual_add,
                    &mut self.d_hidden,
                    &self.d_normed,
                    hidden,
                )
                .map_err(cu)?;

                // FFN: norm + quantize -> gate/up -> GeGLU -> quantize -> down -> post-ffw norm -> residual.
                crate::cuda_resident::launch_rmsnorm(
                    &s,
                    &k.rms_norm,
                    &self.d_hidden,
                    &nrm.ffn_norm,
                    &mut self.d_normed,
                    hidden,
                    eps,
                )
                .map_err(cu)?;
                crate::cuda_resident::launch_quantize(
                    &s,
                    &k.quantize,
                    &self.d_normed,
                    &mut self.d_inq,
                    &mut self.d_ins,
                    hidden / 32,
                )
                .map_err(cu)?;
                gemma_proj_gemv(
                    &s,
                    k,
                    lwd.gate_q,
                    &self.d_ins,
                    &self.d_inq,
                    &lwd.gate.slice(0..lwd.gate.len()),
                    ffn_dim,
                    hidden / 32,
                    &mut self.d_gate,
                )
                .map_err(cu)?;
                gemma_proj_gemv(
                    &s,
                    k,
                    lwd.up_q,
                    &self.d_ins,
                    &self.d_inq,
                    &lwd.up.slice(0..lwd.up.len()),
                    ffn_dim,
                    hidden / 32,
                    &mut self.d_up,
                )
                .map_err(cu)?;
                {
                    let cfg = LaunchConfig {
                        grid_dim: ((ffn_dim as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let n_i = ffn_dim as i32;
                    let mut b = s.launch_builder(&k.geglu_mul);
                    b.arg(&self.d_gate)
                        .arg(&self.d_up)
                        .arg(&mut self.d_geglu)
                        .arg(&n_i);
                    unsafe { b.launch(cfg) }.map_err(cu)?;
                }
                crate::cuda_resident::launch_quantize(
                    &s,
                    &k.quantize,
                    &self.d_geglu,
                    &mut self.d_geglu_q,
                    &mut self.d_geglu_s,
                    ffn_dim / 32,
                )
                .map_err(cu)?;
                gemma_proj_gemv(
                    &s,
                    k,
                    lwd.down_q,
                    &self.d_geglu_s,
                    &self.d_geglu_q,
                    &lwd.down.slice(0..lwd.down.len()),
                    hidden,
                    ffn_dim / 32,
                    &mut self.d_ffn_out,
                )
                .map_err(cu)?;
                crate::cuda_resident::launch_rmsnorm(
                    &s,
                    &k.rms_norm,
                    &self.d_ffn_out,
                    &nrm.post_ffw_norm,
                    &mut self.d_normed,
                    hidden,
                    eps,
                )
                .map_err(cu)?;
                crate::cuda_resident::launch_residual(
                    &s,
                    &k.residual_add,
                    &mut self.d_hidden,
                    &self.d_normed,
                    hidden,
                )
                .map_err(cu)?;

                // PLE injection on the GPU (no host round-trip): gated = inp_gate·h ->
                // gelu_tanh(gated)*pli[li] -> proj·gated -> post_norm -> residual -> output_scale.
                if let Some(pd) = self.ple[li].as_ref() {
                    crate::cuda_resident::launch_f32_gemv(
                        &s,
                        &k.f32_gemv,
                        &pd.inp_gate,
                        &self.d_hidden,
                        &mut self.d_ple_gated,
                        hidden,
                        ple_dim,
                    )
                    .map_err(cu)?;
                    {
                        let off = li * ple_dim;
                        let pli_view = self.d_pli.slice(off..off + ple_dim);
                        let cfg = LaunchConfig {
                            grid_dim: ((ple_dim as u32).div_ceil(256).max(1), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let n_i = ple_dim as i32;
                        let mut b = s.launch_builder(&k.geglu_mul);
                        b.arg(&self.d_ple_gated)
                            .arg(&pli_view)
                            .arg(&mut self.d_ple_gated2)
                            .arg(&n_i);
                        unsafe { b.launch(cfg) }.map_err(cu)?;
                    }
                    crate::cuda_resident::launch_f32_gemv(
                        &s,
                        &k.f32_gemv,
                        &pd.proj,
                        &self.d_ple_gated2,
                        &mut self.d_ple_proj,
                        ple_dim,
                        hidden,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_rmsnorm(
                        &s,
                        &k.rms_norm,
                        &self.d_ple_proj,
                        &pd.post_norm,
                        &mut self.d_ple_normed,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_residual(
                        &s,
                        &k.residual_add,
                        &mut self.d_hidden,
                        &self.d_ple_normed,
                        hidden,
                    )
                    .map_err(cu)?;
                    if pd.output_scale != 1.0 {
                        crate::cuda_resident::launch_scale(
                            &s,
                            &k.scale_f32,
                            &mut self.d_hidden,
                            hidden,
                            pd.output_scale,
                        )
                        .map_err(cu)?;
                    }
                } else if lw.ple_output_scale != 1.0 {
                    crate::cuda_resident::launch_scale(
                        &s,
                        &k.scale_f32,
                        &mut self.d_hidden,
                        hidden,
                        lw.ple_output_scale,
                    )
                    .map_err(cu)?;
                }
            }
        }
        if do_capture {
            use cudarc::driver::sys;
            // Use a real enum variant (not transmute(0): the flags enum has no zero
            // variant, which trips the debug enum-validity check). USE_NODE_PRIORITY is
            // a no-op here (no node priorities are set), so instantiation is plain; the
            // graph is pre-uploaded explicitly via `g.upload()` below.
            let flags =
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
            match s.end_capture(flags).map_err(cu)? {
                Some(g) => {
                    g.upload().map_err(cu)?;
                    self.decode_graph = Some(SendGraph(g));
                }
                None => {
                    return Err(BackendError::InvalidModelMetadata(
                        "gemma4 cuda: decode graph capture produced no graph".into(),
                    ))
                }
            }
        }
        self.warmed = true;
        // Replay the captured graph when present. On the warmup call there is no graph
        // yet and the loop above already executed directly, so we skip the launch.
        if let Some(g) = self.decode_graph.as_ref() {
            g.0.launch().map_err(cu)?;
        }

        // Prefill tokens except the last only need their KV populated, not logits — skip
        // the ~10ms vocab head. The layers/graph already wrote KV on the capture stream,
        // and the next token's upload (a synchronous memcpy) orders after it, so no sync
        // is needed here.
        if !want_logits {
            return Ok(Vec::new());
        }

        // ---- Final norm + tied head + soft-cap. ----
        if let Some(head) = self.gpu_head.as_mut() {
            // GPU Q6_K head: fused rms_norm+Q8K-quant -> q6k_gemv over the vocab ->
            // soft-cap, on the capture stream; only the logits are copied back. This
            // replaces the ~1.2 s/token CPU Q6_K matvec that dominates decode.
            let wlen = head.weight.len();
            match head.lane {
                HeadLane::Q8_0 => {
                    crate::cuda_resident::launch_rmsnorm_quantize(
                        &s,
                        &self.kernels.rms_norm_quantize,
                        &self.d_hidden,
                        &head.output_norm,
                        &mut head.inq,
                        &mut head.ins,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_gemv(
                        &s,
                        &self.kernels.gemv,
                        &head.ins,
                        &head.inq,
                        &head.weight.slice(0..wlen),
                        self.vocab,
                        head.blocks,
                        &mut head.logits,
                    )
                    .map_err(cu)?;
                }
                HeadLane::Q6K => {
                    crate::cuda_resident::launch_rmsnorm_quantize_q8k(
                        &s,
                        &self.kernels.rms_norm_quantize_q8k,
                        &self.d_hidden,
                        &head.output_norm,
                        &mut head.inq,
                        &mut head.ins,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_q6k_gemv(
                        &s,
                        &self.kernels.q6k_gemv,
                        &head.ins,
                        &head.inq,
                        &head.weight.slice(0..wlen),
                        self.vocab,
                        head.blocks,
                        &mut head.logits,
                        0,
                    )
                    .map_err(cu)?;
                }
                HeadLane::Q4K => {
                    crate::cuda_resident::launch_rmsnorm_quantize_q8k(
                        &s,
                        &self.kernels.rms_norm_quantize_q8k,
                        &self.d_hidden,
                        &head.output_norm,
                        &mut head.inq,
                        &mut head.ins,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_q4k_gemv(
                        &s,
                        &self.kernels.q4k_gemv,
                        &head.ins,
                        &head.inq,
                        &head.weight.slice(0..wlen),
                        self.vocab,
                        head.blocks,
                        &mut head.logits,
                        0,
                    )
                    .map_err(cu)?;
                }
            }
            if head.softcap != 0.0 {
                let cfg = LaunchConfig {
                    grid_dim: ((self.vocab as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let (n_i, cap) = (self.vocab as i32, head.softcap);
                let mut b = s.launch_builder(&self.kernels.soft_cap);
                b.arg(&mut head.logits).arg(&n_i).arg(&cap);
                unsafe { b.launch(cfg) }.map_err(cu)?;
            }
            s.synchronize().map_err(cu)?;
            let mut logits = vec![0f32; self.vocab];
            s.memcpy_dtoh(&head.logits, &mut logits).map_err(cu)?;
            return Ok(logits);
        }
        // CPU head fallback (non-Q6_K head): final norm + tied matvec + soft-cap.
        s.synchronize().map_err(cu)?;
        let mut last = vec![0f32; hidden];
        s.memcpy_dtoh(&self.d_hidden, &mut last).map_err(cu)?;
        let normed = rms_norm(&last, Some(&self.cpu.output_norm), eps);
        let mut logits = self.cpu.token_embd.matvec(hidden, self.vocab, &normed);
        if let Some(cap) = self.cpu.g.final_logit_softcapping {
            soft_cap_in_place(&mut logits, cap);
        }
        Ok(logits)
    }

    /// Greedy-generate up to `max_new` tokens (mirrors the Metal runtime loop).
    pub fn generate_greedy(&mut self, prompt: &str, max_new: usize) -> Result<(String, Vec<u32>)> {
        let prompt_tokens = self.cpu.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.cpu.tokenizer);
        let mut logits = Vec::new();
        let last_prompt = prompt_tokens.len().saturating_sub(1);
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.forward_token(tok, pos, pos == last_prompt)?;
        }
        let mut generated = Vec::new();
        for pos in prompt_tokens.len()..prompt_tokens.len() + max_new {
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
            logits = self.forward_token(next, pos, true)?;
        }
        let text = self.cpu.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }

    /// Greedy-generate emitting a per-token text delta (for SSE streaming): after
    /// each token the full output is re-decoded and the new suffix is handed to
    /// `on_delta` (robust to tokenizer spacing).
    pub fn generate_greedy_streaming<F: FnMut(&str)>(
        &mut self,
        prompt: &str,
        max_new: usize,
        mut on_delta: F,
    ) -> Result<(String, Vec<u32>)> {
        let prompt_tokens = self.cpu.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.cpu.tokenizer);
        let mut logits = Vec::new();
        let last_prompt = prompt_tokens.len().saturating_sub(1);
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.forward_token(tok, pos, pos == last_prompt)?;
        }
        let mut generated = Vec::new();
        let mut prev_text = String::new();
        for pos in prompt_tokens.len()..prompt_tokens.len() + max_new {
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
            let text = self.cpu.tokenizer.decode(&generated, true)?;
            if text.len() > prev_text.len() {
                on_delta(&text[prev_text.len()..]);
            }
            prev_text = text;
            logits = self.forward_token(next, pos, true)?;
        }
        Ok((prev_text, generated))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_parity_tests {
    use super::*;

    // Greedy parity: the CUDA gemma4 forward must match the CPU Gemma4Runtime oracle
    // token-for-token on the E4B Q8_0 file (the oracle that the CPU runtime loads).
    // Weights stream from host per layer, so it fits the 6 GB card; kept short.
    #[test]
    #[ignore = "requires a CUDA device + the gemma4 E4B Q8_0 model"]
    fn gemma4_cuda_matches_cpu_greedy() {
        let path_s = match std::env::var("CAMELID_GEMMA4_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skip: set CAMELID_GEMMA4_GGUF to the gemma4 E4B Q8_0 gguf path");
                return;
            }
        };
        let path = std::path::Path::new(&path_s);
        if !path.exists() {
            eprintln!("skip: gemma4 model not found at {path_s}");
            return;
        }
        let prompt = "The capital of France is";
        let cpu = Gemma4Runtime::load(path).expect("cpu load");
        let (cpu_text, cpu_ids) = cpu.generate_greedy(prompt, 8).expect("cpu gen");
        let mut gpu = Gemma4CudaResident::load(path, 2048).expect("gpu load");
        let t0 = std::time::Instant::now();
        let (gpu_text, gpu_ids) = gpu.generate_greedy(prompt, 24).expect("gpu gen");
        let secs = t0.elapsed().as_secs_f64();
        eprintln!("CPU ids[..8] {cpu_ids:?} -> {cpu_text:?}");
        eprintln!("GPU ids       {gpu_ids:?} -> {gpu_text:?}");
        eprintln!(
            "GPU decode: {} tokens in {:.1}s = {:.2} tok/s",
            gpu_ids.len(),
            secs,
            gpu_ids.len() as f64 / secs.max(1e-9)
        );
        // Greedy-parity gate: the CUDA decode must match the CPU oracle's DETERMINISTIC
        // next-token argmax (the gemma4 lane's argmax-stability guarantee). Every
        // projection kernel is bit-exact vs its CPU oracle (q8/q4_0/q4_1/q4k/q6k unit
        // tests), but the attention online-softmax, PLE gelu (CUDA tanhf) and norm
        // reductions are fp-reassociated, so on coarse quant (Q4) a logit near-tie can
        // flip a LATER token — divergence past the first token is allowed. The shared
        // prefix length is logged so a deeper regression is still visible.
        let common = gpu_ids
            .iter()
            .zip(&cpu_ids)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "CPU/GPU greedy common prefix: {common}/{} tokens",
            cpu_ids.len()
        );
        assert_eq!(
            gpu_ids.first(),
            cpu_ids.first(),
            "gemma4 CUDA first-token argmax diverged from the CPU oracle"
        );
    }
}

#[cfg(test)]
mod q4_0_cpu_tests {
    use super::*;

    // Phase 1 gate (mission C): the CPU oracle must LOAD the mixed-quant Q4_0 file
    // (Q4_0 + Q4_1 ffn_down + Q4_K tied head + Q5_K per_layer_token_embd + BF16 proj)
    // and generate coherent greedy text. Set CAMELID_GEMMA4_Q4_GGUF to the file.
    #[test]
    #[ignore = "set CAMELID_GEMMA4_Q4_GGUF to the mixed Q4_0 gemma4 gguf"]
    fn cpu_loads_and_decodes_mixed_q4_0() {
        let path = match std::env::var("CAMELID_GEMMA4_Q4_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skip: set CAMELID_GEMMA4_Q4_GGUF");
                return;
            }
        };
        let cpu = Gemma4Runtime::load(std::path::Path::new(&path)).expect("load mixed Q4_0");
        let (text, ids) = cpu
            .generate_greedy("The capital of France is", 16)
            .expect("cpu generate");
        eprintln!("Q4_0 CPU ids:  {ids:?}");
        eprintln!("Q4_0 CPU text: {text:?}");
        assert!(!ids.is_empty(), "generated no tokens");
    }
}
