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

use crate::gguf::{read_metadata, GgufFile, GgufTensorType};
use crate::ghost::{GhostFile, GhostMoeExpert, GhostMoeTensorView};
use crate::inference::gemma4::{gelu_tanh, soft_cap_in_place};
use crate::inference::{
    nvfp4_wire_block_dequant, nvfp4_wire_row_dot, q4_0_wire_block_dequant, q4_0_wire_row_dot,
    q4_1_wire_row_dot, q4_k_wire_row_dot, q6_k_wire_block_dequant, q6_k_wire_row_dot,
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

/// Result of a cooperatively-cancellable Gemma 4 generation.
///
/// Cancellation is not an inference failure: the HTTP owner went away, so the
/// runtime returns the number of tokens it had already committed and releases
/// its KV/expert state at the next forward boundary.  Keeping this distinct
/// from [`BackendError`] lets serving drop a disconnected request quietly while
/// still surfacing genuine model failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gemma4GenerationOutcome {
    Complete { text: String, token_ids: Vec<u32> },
    Cancelled { generated_tokens: usize },
}

/// The wire quant formats the gemma4 CPU runtime reads in place. Q8_0 is the
/// proven baseline lane; Q4_0 and Q6_K are the QAT-row formats (all the QAT
/// linear weights are Q4_0; the tied token/per-layer embeddings are Q6_K).
/// NVFP4 is the BASALT pilot matmul-weight format (D17/D-B1: pin `block_nvfp4`
/// byte-for-byte, 64-element/36-byte superblocks; the pilot's embeddings/norms
/// stay in the Q8_0 baseline formats per `basalt_eval_protocol.md` §1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WireFormat {
    Q8_0,
    Q4_0,
    Q4_1,
    Q4K,
    Q5K,
    Q6K,
    Nvfp4,
}

impl WireFormat {
    #[inline]
    fn values_per_block(self) -> usize {
        match self {
            WireFormat::Q8_0 | WireFormat::Q4_0 | WireFormat::Q4_1 => 32,
            WireFormat::Q4K | WireFormat::Q5K | WireFormat::Q6K => 256,
            WireFormat::Nvfp4 => crate::tensor::NVFP4_VALUES_PER_BLOCK, // 64
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
            WireFormat::Nvfp4 => crate::tensor::NVFP4_WIRE_BYTES_PER_BLOCK, // 36
        }
    }

    /// Wire bytes of one weight row spanning `q8_blocks` Q8_0 activation blocks
    /// (32 values each) — for the Q8_0-activation matvec family only. For the
    /// 32-value formats this is `q8_blocks * bytes_per_block`; one 64-value
    /// NVFP4 superblock spans TWO activation blocks, so its row is
    /// `q8_blocks / 2` superblocks wide.
    #[inline]
    fn row_bytes_for_q8_blocks(self, q8_blocks: usize) -> usize {
        debug_assert!(
            (q8_blocks * Q8_VALUES_PER_BLOCK).is_multiple_of(self.values_per_block()),
            "row of {q8_blocks} Q8_0 blocks is not whole {self:?} blocks"
        );
        q8_blocks * Q8_VALUES_PER_BLOCK / self.values_per_block() * self.bytes_per_block()
    }
}

/// A quantized weight read straight from the memory-mapped GGUF — no eager
/// decode and no second resident copy. The mmap pages fault in on first touch
/// (during the first generation) and stay in the OS page cache after, so
/// `load()` is ~instant instead of spending ~240s materializing 8GB of decoded
/// blocks up front. Dequant happens inline in the matmul — only the block
/// scale is decoded per block per pass (negligible next to the mul-adds it
/// scales). Any tensor type outside [`WireFormat`] fails closed at load.
#[derive(Clone)]
enum WireBacking {
    Mmap {
        mmap: Arc<GgufWireMmap>,
        offset: u64,
    },
    /// Bounded routed-expert bytes read from a v2 `.cghost` group. `range`
    /// selects one of the two tensors while both share a single cache allocation.
    Owned {
        bytes: Arc<[u8]>,
        range: std::ops::Range<usize>,
    },
}

#[derive(Clone)]
struct WireQuant {
    backing: WireBacking,
    element_count: usize,
    format: WireFormat,
}

impl WireQuant {
    fn format_for_type(tensor_type: GgufTensorType, name: &str) -> Result<WireFormat> {
        match tensor_type {
            GgufTensorType::Q8_0 => Ok(WireFormat::Q8_0),
            GgufTensorType::Q4_0 => Ok(WireFormat::Q4_0),
            GgufTensorType::Q4_1 => Ok(WireFormat::Q4_1),
            GgufTensorType::Q4K => Ok(WireFormat::Q4K),
            GgufTensorType::Q5K => Ok(WireFormat::Q5K),
            GgufTensorType::Q6K => Ok(WireFormat::Q6K),
            GgufTensorType::NVFP4 => Ok(WireFormat::Nvfp4),
            other => Err(BackendError::UnsupportedTensorType(format!(
                "tensor {name} is {other:?}; gemma4 wire load supports Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, and NVFP4"
            ))),
        }
    }

    fn new(store: &TensorStore, mmap: &Arc<GgufWireMmap>, name: &str) -> Result<Self> {
        let desc = store.descriptor(name)?;
        let format = Self::format_for_type(desc.tensor_type, name)?;
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
        // BASALT D17/T5 fail-closed: NVFP4 files carrying NaN-sentinel UE4M3
        // scale bytes (0x7F/0xFF) are refused at load — the pin's own CPU and
        // CUDA backends disagree on 0xFF, so such a file has no well-defined
        // cross-backend oracle. The runnable lane refuses inside
        // `decode_nvfp4_tensor`; this wire lane never runs that decoder (the
        // matvec reads wire bytes in place), so the scan lives here. One
        // sequential pass over the tensor's mapped bytes (which the first
        // generation would fault in anyway); zero scales admit.
        if format == WireFormat::Nvfp4 {
            let wire = mmap.bytes(desc.absolute_offset, byte_len)?;
            if let Some(block_idx) = crate::tensor::nvfp4_find_nan_scale(wire) {
                return Err(BackendError::InvalidTensorData(format!(
                    "tensor {name}: NVFP4 block {block_idx} carries a NaN-sentinel UE4M3 \
                     scale byte (0x7F/0xFF) — refusing per D17/T5 (fail closed at load)"
                )));
            }
        }
        Ok(Self {
            backing: WireBacking::Mmap {
                mmap: mmap.clone(),
                offset: desc.absolute_offset,
            },
            element_count,
            format,
        })
    }

    fn from_ghost_tensor(
        expert: &GhostMoeExpert,
        view: &GhostMoeTensorView,
        name: &str,
    ) -> Result<Self> {
        let (bytes, range) = expert.tensor_backing(view);
        Self::from_owned_wire(bytes, range, view.dtype, &view.dims, name)
    }

    fn from_owned_wire(
        bytes: Arc<[u8]>,
        range: std::ops::Range<usize>,
        tensor_type: GgufTensorType,
        dims: &[u64],
        name: &str,
    ) -> Result<Self> {
        let format = Self::format_for_type(tensor_type, name)?;
        let element_count = dims
            .iter()
            .try_fold(1u64, |count, dim| count.checked_mul(*dim))
            .ok_or_else(|| {
                BackendError::InvalidTensorData(format!(
                    "ghost expert tensor {name} element count overflows"
                ))
            })? as usize;
        if !element_count.is_multiple_of(format.values_per_block()) {
            return Err(BackendError::InvalidTensorData(format!(
                "ghost expert tensor {name} element count {element_count} is not block-aligned"
            )));
        }
        let expected = element_count / format.values_per_block() * format.bytes_per_block();
        if range.start > range.end || range.end > bytes.len() || range.len() != expected {
            return Err(BackendError::InvalidTensorData(format!(
                "ghost expert tensor {name} has {} wire bytes; expected {expected}",
                range.len()
            )));
        }
        Ok(Self {
            backing: WireBacking::Owned { bytes, range },
            element_count,
            format,
        })
    }

    /// Typed load-time guard for weights bound to a matvec/matmul role
    /// (projection, expert band, or tied head). Q5_K is GATHER-ONLY in this
    /// lane (`per_layer_token_embd`; no Q5_K row-dot kernel is wired here), so
    /// admitting it into a matvec role would surface as a forward-time panic —
    /// refuse it at load instead (invariant I-unknown-type: typed refusal,
    /// never a reachable panic). Every other [`WireFormat`] has a matvec route.
    fn require_matvec_capable(self, name: &str) -> Result<Self> {
        if self.format == WireFormat::Q5K {
            return Err(BackendError::UnsupportedTensorType(format!(
                "tensor {name} is Q5_K; the gemma4 wire lane serves Q5_K gather-only \
                 (per_layer_token_embd) — it cannot be a projection/head weight"
            )));
        }
        Ok(self)
    }

    /// The tensor's full wire-byte slice. Bounds were validated in `new`.
    #[inline]
    fn bytes(&self) -> &[u8] {
        let byte_len =
            self.element_count / self.format.values_per_block() * self.format.bytes_per_block();
        match &self.backing {
            WireBacking::Mmap { mmap, offset } => mmap
                .bytes(*offset, byte_len)
                .expect("wire quant range validated at load"),
            WireBacking::Owned { bytes, range } => &bytes[range.clone()],
        }
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
            // NVFP4 rides the Q8_0-activation family: the pin's
            // `ggml_vec_dot_nvfp4_q8_0_generic` dots NVFP4 superblocks against
            // Q8_0 activation blocks, exactly like Q8_0/Q4_0/Q4_1.
            WireFormat::Q8_0 | WireFormat::Q4_0 | WireFormat::Q4_1 | WireFormat::Nvfp4 => {
                self.matvec_q(out_dim, &quantize_q8_0_blocks(x))
            }
            // K-quant rows dot against Q8_K activations (the reference's K-quant
            // activation format) — Q6_K/Q4_K used by the QAT tied embedding head.
            WireFormat::Q4K | WireFormat::Q6K => self.matvec_q8k(out_dim, &quantize_q8_k_blocks(x)),
            // Q5_K is gather-only here (per_layer_token_embd); never a matvec
            // weight — `require_matvec_capable` refuses it typed at load.
            WireFormat::Q5K => unreachable!("Q5_K is gather-only (per_layer_token_embd)"),
        }
    }

    /// One projection off a [`SharedActivation`], routed by the SAME family
    /// split as the top-level [`Self::matvec`]: K-quant weights (Q4_K/Q6_K)
    /// dot Q8_K activations via [`Self::matvec_q8k`], everything else keeps
    /// the Q8_0-activation fast path via [`Self::matvec_q`] byte-for-byte.
    ///
    /// This is the SHA_E3 crash fix: the per-layer projection call sites used
    /// to pre-quantize the shared activation to Q8_0 once and call `matvec_q`
    /// directly, which has no K-quant arms — a latent pre-BASALT gap (no
    /// gemma4 K-quant matmul row existed) that panicked `unreachable!` at
    /// forward time on the campaign's Q4K-mm/Q4_K_M rows. The shared
    /// activation is still quantized at most once PER FAMILY per call site
    /// (lazily), so single-family files pay exactly the old quantize count.
    fn matvec_proj(&self, out_dim: usize, x: &SharedActivation) -> Vec<f32> {
        match self.format {
            WireFormat::Q8_0 | WireFormat::Q4_0 | WireFormat::Q4_1 | WireFormat::Nvfp4 => {
                self.matvec_q(out_dim, x.q8_0())
            }
            WireFormat::Q4K | WireFormat::Q6K => self.matvec_q8k(out_dim, x.q8_k()),
            // Structurally unreachable: `require_matvec_capable` refuses Q5_K
            // in every matvec-role binding at load (typed, I-unknown-type).
            WireFormat::Q5K => unreachable!("Q5_K matvec roles are refused at load"),
        }
    }

    /// Batched sibling of [`Self::matvec_proj`] for the spec-verify chunk
    /// path: routes to [`Self::matmul_q`] / [`Self::matmul_q8k`] by the same
    /// family split, off a [`SharedActivationBatch`].
    fn matmul_proj(&self, out_dim: usize, xs: &SharedActivationBatch) -> Vec<Vec<f32>> {
        match self.format {
            WireFormat::Q8_0 | WireFormat::Q4_0 | WireFormat::Q4_1 | WireFormat::Nvfp4 => {
                self.matmul_q(out_dim, xs.q8_0())
            }
            WireFormat::Q4K | WireFormat::Q6K => self.matmul_q8k(out_dim, xs.q8_k()),
            WireFormat::Q5K => unreachable!("Q5_K matvec roles are refused at load"),
        }
    }

    /// Row-band sibling of [`Self::matvec_proj`] for the MoE expert matrices:
    /// routes to [`Self::matvec_q_rows`] / [`Self::matvec_q8k_rows`] by the
    /// same family split.
    fn matvec_rows_proj(
        &self,
        row_start: usize,
        out_count: usize,
        x: &SharedActivation,
    ) -> Vec<f32> {
        match self.format {
            WireFormat::Q8_0 | WireFormat::Q4_0 | WireFormat::Q4_1 | WireFormat::Nvfp4 => {
                self.matvec_q_rows(row_start, out_count, x.q8_0())
            }
            WireFormat::Q4K | WireFormat::Q6K => {
                self.matvec_q8k_rows(row_start, out_count, x.q8_k())
            }
            WireFormat::Q5K => unreachable!("Q5_K matvec roles are refused at load"),
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
        let row_bytes = self.format.row_bytes_for_q8_blocks(xq.len());
        let bytes = self.bytes();
        let row_dot: fn(&[u8], &[Q8_0Block]) -> f32 = match self.format {
            WireFormat::Q8_0 => q8_0_wire_row_dot,
            WireFormat::Q4_0 => q4_0_wire_row_dot,
            WireFormat::Q4_1 => q4_1_wire_row_dot,
            WireFormat::Nvfp4 => nvfp4_wire_row_dot,
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
        let row_bytes = self.format.row_bytes_for_q8_blocks(xq.len());
        let bytes = self.bytes();
        let row_dot: fn(&[u8], &[Q8_0Block]) -> f32 = match self.format {
            WireFormat::Q8_0 => q8_0_wire_row_dot,
            WireFormat::Q4_0 => q4_0_wire_row_dot,
            WireFormat::Q4_1 => q4_1_wire_row_dot,
            WireFormat::Nvfp4 => nvfp4_wire_row_dot,
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

    /// Repack a contiguous band of `row_count` Q4_0 output rows starting at
    /// `row_start` into the interleaved 8-row [`Q4_0PackedRows8`] layout the AVX2
    /// GEMV consumes. `blocks_per_row` = in_dim/32. `row_start`, `row_count`, and
    /// `blocks_per_row` are all block/8-aligned for the expert bands. Called once
    /// per expert per session (cached), not per token.
    fn pack_rows(
        &self,
        row_start: usize,
        row_count: usize,
        blocks_per_row: usize,
    ) -> crate::tensor::Q4_0PackedRows8 {
        debug_assert_eq!(self.format, WireFormat::Q4_0);
        let row_bytes = blocks_per_row * self.format.bytes_per_block();
        let bytes = self.bytes();
        let start = row_start * row_bytes;
        let band = &bytes[start..start + row_count * row_bytes];
        crate::tensor::Q4_0PackedRows8::from_q4_0_bytes(row_count, blocks_per_row, band)
            .expect("Q4_0 expert band repack (rows multiple of 8, block-aligned)")
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
        let row_bytes = self.format.row_bytes_for_q8_blocks(xqs[0].len());
        let bytes = self.bytes();
        // The batched NVFP4 variant is the same shared-read pattern as its
        // siblings: one weight-row read, `row_dot` looped over the K
        // activations below (correctness-first; no perf claim).
        let row_dot: fn(&[u8], &[Q8_0Block]) -> f32 = match self.format {
            WireFormat::Q8_0 => q8_0_wire_row_dot,
            WireFormat::Q4_0 => q4_0_wire_row_dot,
            WireFormat::Q4_1 => q4_1_wire_row_dot,
            WireFormat::Nvfp4 => nvfp4_wire_row_dot,
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

    /// [`Self::matvec_q_rows`] for K-quant (Q4_K/Q6_K) weights against a Q8_K
    /// activation: dot a contiguous band of `out_count` output rows starting at
    /// `row_start` — the MoE expert-band path when the expert matrices are
    /// K-quants. Same fixed row chunking as [`Self::matvec_q8k`], and rows land
    /// at fixed indices, so `out[i]` is bit-identical to row `row_start + i` of
    /// the full [`Self::matvec_q8k`] (greedy parity safe).
    fn matvec_q8k_rows(
        &self,
        row_start: usize,
        out_count: usize,
        xq: &[crate::inference::Q8KBlock],
    ) -> Vec<f32> {
        const ROW_CHUNK: usize = 64;
        let row_bytes = xq.len() * self.format.bytes_per_block();
        let bytes = self.bytes();
        let row_dot: fn(&[u8], &[crate::inference::Q8KBlock]) -> f32 = match self.format {
            WireFormat::Q6K => q6_k_wire_row_dot,
            WireFormat::Q4K => q4_k_wire_row_dot,
            _ => unreachable!("matvec_q8k_rows is only for Q6_K/Q4_K weights"),
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
            // Q4_1 is a matvec-only weight here (ffn_down); no gather decoder is
            // wired. A Q4_1 embedding table would land here, so refuse typed
            // (I-unknown-type: never a reachable panic) — this arm was an
            // `unreachable!` until the SHA_E3 K-quant routing fix swept the
            // lane's reachable-panic arms.
            WireFormat::Q4_1 => {
                return Err(BackendError::UnsupportedTensorType(
                    "gemma4 wire lane cannot gather Q4_1 elements (Q4_1 is a \
                     matvec-only weight format here)"
                        .into(),
                ))
            }
            // NVFP4 gather: decode one 64-value superblock at a time via the
            // pin-bitwise hot-path twin (same pattern as the Q4_0 arm). The
            // BASALT pilot rows keep embeddings Q8_0 (matmul weights only are
            // NVFP4), so this arm only runs on non-pilot shapes; sentinel scale
            // bytes were already refused at load (D17/T5).
            WireFormat::Nvfp4 => {
                const BV: usize = crate::tensor::NVFP4_VALUES_PER_BLOCK;
                const BB: usize = crate::tensor::NVFP4_WIRE_BYTES_PER_BLOCK;
                let mut block = usize::MAX;
                let mut decoded = [0f32; BV];
                for e in start..end {
                    if e / BV != block {
                        block = e / BV;
                        decoded = nvfp4_wire_block_dequant(&bytes[block * BB..(block + 1) * BB]);
                    }
                    out.push(decoded[e % BV]);
                }
            }
        }
        Ok(out)
    }
}

/// A shared per-layer activation with each matvec activation family quantized
/// LAZILY, at most once, however many projections consume it (q/k/v share the
/// pre-attention norm; gate/up share the pre-FFN norm). The Q8_0-family
/// projections (Q8_0/Q4_0/Q4_1/NVFP4) dot Q8_0 blocks; K-quant projections
/// (Q4_K/Q6_K) dot Q8_K blocks — a mixed-format layer quantizes once per
/// family, a single-family layer pays exactly the old single quantize.
/// Single-threaded by construction (a per-step local; rayon parallelism lives
/// INSIDE the matvecs, over output rows), hence the plain `OnceCell`.
struct SharedActivation<'a> {
    x: &'a [f32],
    q8_0: std::cell::OnceCell<Vec<Q8_0Block>>,
    q8_k: std::cell::OnceCell<Vec<crate::inference::Q8KBlock>>,
}

impl<'a> SharedActivation<'a> {
    fn new(x: &'a [f32]) -> Self {
        Self {
            x,
            q8_0: std::cell::OnceCell::new(),
            q8_k: std::cell::OnceCell::new(),
        }
    }

    fn q8_0(&self) -> &[Q8_0Block] {
        self.q8_0.get_or_init(|| quantize_q8_0_blocks(self.x))
    }

    fn q8_k(&self) -> &[crate::inference::Q8KBlock] {
        self.q8_k.get_or_init(|| quantize_q8_k_blocks(self.x))
    }
}

/// The batched (spec-verify [`Gemma4Runtime::step_chunk`]) sibling of
/// [`SharedActivation`]: K activation rows, each quantized family computed
/// lazily once for the whole chunk. Quantization is a pure per-row function,
/// so laziness cannot change any value.
struct SharedActivationBatch<'a> {
    xs: &'a [Vec<f32>],
    q8_0: std::cell::OnceCell<Vec<Vec<Q8_0Block>>>,
    q8_k: std::cell::OnceCell<Vec<Vec<crate::inference::Q8KBlock>>>,
}

impl<'a> SharedActivationBatch<'a> {
    fn new(xs: &'a [Vec<f32>]) -> Self {
        Self {
            xs,
            q8_0: std::cell::OnceCell::new(),
            q8_k: std::cell::OnceCell::new(),
        }
    }

    fn q8_0(&self) -> &[Vec<Q8_0Block>] {
        self.q8_0
            .get_or_init(|| self.xs.iter().map(|x| quantize_q8_0_blocks(x)).collect())
    }

    fn q8_k(&self) -> &[Vec<crate::inference::Q8KBlock>] {
        self.q8_k
            .get_or_init(|| self.xs.iter().map(|x| quantize_q8_k_blocks(x)).collect())
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

/// One expert's two projection matrices, pre-repacked into the interleaved
/// 8-row layout [`crate::tensor::Q4_0PackedRows8`] the AVX2 GEMV consumes. Built
/// lazily on first use and cached (see [`ExpertPackCache`]) so the repack is paid
/// once per expert per session instead of once per token — the packed GEMV then
/// runs with no per-call repack/alloc, which is what makes it beat the (already
/// autovectorized) scalar wire dot.
struct PackedExpert {
    /// Fused gate‖up, `2*n_ff_exp` rows × (n_embd/32) blocks/row.
    gate_up: crate::tensor::Q4_0PackedRows8,
    /// Down, `n_embd` rows × (n_ff_exp/32) blocks/row.
    down: crate::tensor::Q4_0PackedRows8,
}

impl PackedExpert {
    fn byte_len(&self) -> usize {
        self.gate_up.byte_len() + self.down.byte_len()
    }
}

/// Bounded host-RAM cache of [`PackedExpert`]s for ONE MoE layer, keyed by expert
/// index. A greedy decode fires a small, stable subset of the 128 experts, so a
/// modest cap keeps the hot experts pre-packed (steady-state SIMD GEMV with no
/// repack) while bounding the extra RAM — the packed form is a second copy of the
/// expert weights (~11% larger than the mmap wire bytes), so caching ALL experts
/// of ALL layers would blow this box's RAM. Eviction is FIFO on the insertion
/// order (the working set is stable, so FIFO ≈ LRU here). Budget in MiB via
/// `CAMELID_GEMMA4_EXPERT_PACK_MIB` (default 1024; 0 disables the SIMD pack path,
/// falling back to the scalar wire dot). Correctness is independent of the cache:
/// a miss that cannot be cached just repacks on the fly, still bit-exact.
struct ExpertPackCache {
    entries: std::collections::HashMap<u16, Arc<PackedExpert>>,
    order: std::collections::VecDeque<u16>,
    bytes: usize,
    budget_bytes: usize,
}

impl ExpertPackCache {
    fn new(budget_bytes: usize) -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
            bytes: 0,
            budget_bytes,
        }
    }

    fn get(&self, e: u16) -> Option<Arc<PackedExpert>> {
        self.entries.get(&e).cloned()
    }

    /// Insert `packed` for expert `e`, evicting FIFO until it fits the budget.
    /// If a single expert exceeds the budget it is not cached (returned Arc is
    /// still usable by the caller for this one token).
    fn insert(&mut self, e: u16, packed: Arc<PackedExpert>) {
        let sz = packed.byte_len();
        if sz > self.budget_bytes {
            return;
        }
        while self.bytes + sz > self.budget_bytes {
            let Some(old) = self.order.pop_front() else {
                break;
            };
            if let Some(p) = self.entries.remove(&old) {
                self.bytes -= p.byte_len();
            }
        }
        if self.entries.insert(e, packed).is_none() {
            self.order.push_back(e);
            self.bytes += sz;
        }
    }
}

/// Per-layer expert-pack budget (bytes) from `CAMELID_GEMMA4_EXPERT_PACK_MIB`
/// (default 1024 MiB). `0` disables pre-packing (scalar wire-dot fallback).
fn expert_pack_budget_bytes() -> usize {
    std::env::var("CAMELID_GEMMA4_EXPERT_PACK_MIB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1024)
        .saturating_mul(1024 * 1024)
}

/// Observable state of the bounded Ghost-MoE expert cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GhostMoeCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub bytes_read: u64,
    pub resident_experts: usize,
    pub resident_bytes: usize,
    pub budget_bytes: usize,
}

struct GhostMoeCacheState {
    entries: std::collections::HashMap<(usize, usize), GhostMoeCacheEntry>,
    /// Retained wire bytes per transformer layer. Each layer owns a hard slice
    /// of the global budget, so the layer-major forward pass cannot bulldoze
    /// another layer's hot experts before the next token reaches it.
    layer_bytes: Vec<usize>,
    /// Monotonic access stamp used as the LRU tie-break between equal-frequency
    /// entries. Frequency is periodically aged per layer (see `touch_layer`).
    clock: u64,
    layer_accesses: Vec<u64>,
    bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
    bytes_read: u64,
}

struct GhostMoeCacheEntry {
    expert: Arc<GhostMoeExpert>,
    frequency: u16,
    last_used: u64,
}

impl GhostMoeCacheState {
    /// Advance one layer's LFU epoch. Aging prevents experts that were hot near
    /// the start of a long conversation from becoming permanently unevictable;
    /// LRU remains the deterministic tie-break after the inexpensive decay.
    fn touch_layer(&mut self, layer: usize) {
        self.clock = self.clock.saturating_add(1);
        let Some(accesses) = self.layer_accesses.get_mut(layer) else {
            return;
        };
        *accesses = accesses.saturating_add(1);
        if !(*accesses).is_multiple_of(256) {
            return;
        }
        for (&(entry_layer, _), entry) in &mut self.entries {
            if entry_layer == layer {
                entry.frequency = (entry.frequency / 2).max(1);
            }
        }
    }

    /// Remove the least-frequently used entry in `layer`, breaking frequency
    /// ties by oldest access. The forward accumulation never depends on this
    /// order; it only chooses which immutable wire record stays resident.
    fn evict_one_from_layer(&mut self, layer: usize) -> bool {
        let victim = self
            .entries
            .iter()
            .filter(|(&(entry_layer, _), _)| entry_layer == layer)
            .min_by_key(|(_, entry)| (entry.frequency, entry.last_used))
            .map(|(&key, _)| key);
        let Some(victim) = victim else {
            return false;
        };
        let evicted = self
            .entries
            .remove(&victim)
            .expect("selected ghost MoE victim disappeared");
        let size = evicted.expert.byte_len();
        self.bytes = self.bytes.saturating_sub(size);
        self.layer_bytes[layer] = self.layer_bytes[layer].saturating_sub(size);
        self.evictions = self.evictions.saturating_add(1);
        true
    }
}

/// One cache for the whole model, rather than one nominal budget per layer.
/// This is the memory-ceiling invariant: regardless of how many of Gemma 4's
/// 30×128 experts a session routes to, retained wire bytes never exceed the
/// configured global budget. A too-large entry remains usable for the current
/// layer but is not retained.
struct GhostMoeExpertCache {
    file: Arc<GhostFile>,
    budget_bytes: usize,
    /// One non-overlapping budget segment per model layer. Remainder bytes are
    /// assigned to the first layers; the sum is exactly `budget_bytes`.
    layer_budgets: Vec<usize>,
    /// Positioned reads for one routed top-k can be issued concurrently on
    /// SSD/NVMe without tying up Rayon compute workers. Set
    /// `CAMELID_GEMMA4_GHOST_READ_THREADS=1` for rotational or strictly serial
    /// storage. Windows' unbuffered reader is serialized internally, so it
    /// defaults to one thread there.
    read_pool: Option<rayon::ThreadPool>,
    state: std::sync::Mutex<GhostMoeCacheState>,
}

impl GhostMoeExpertCache {
    fn new(file: Arc<GhostFile>, budget_bytes: usize) -> Self {
        let layer_count = file.index.block_count.max(1);
        let base_layer_budget = budget_bytes / layer_count;
        let remainder = budget_bytes % layer_count;
        let layer_budgets = (0..layer_count)
            .map(|layer| base_layer_budget + usize::from(layer < remainder))
            .collect::<Vec<_>>();
        let default_read_threads = if cfg!(windows) { 1 } else { 4 };
        let read_threads = std::env::var("CAMELID_GEMMA4_GHOST_READ_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(default_read_threads)
            .clamp(1, 8);
        let read_pool = (read_threads > 1)
            .then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(read_threads)
                    .thread_name(|index| format!("ghost-moe-read-{index}"))
                    .build()
                    .ok()
            })
            .flatten();
        Self {
            file,
            budget_bytes,
            layer_budgets,
            read_pool,
            state: std::sync::Mutex::new(GhostMoeCacheState {
                entries: std::collections::HashMap::new(),
                layer_bytes: vec![0; layer_count],
                clock: 0,
                layer_accesses: vec![0; layer_count],
                bytes: 0,
                hits: 0,
                misses: 0,
                evictions: 0,
                bytes_read: 0,
            }),
        }
    }

    #[cfg(test)]
    fn get(&self, layer: usize, expert: usize) -> Result<Arc<GhostMoeExpert>> {
        self.get_many(layer, &[expert]).map(|mut values| {
            values
                .pop()
                .expect("one requested ghost MoE expert must produce one result")
        })
    }

    /// Borrow an already-resident immutable record without changing cache
    /// frequency, recency, or observable hit/miss counters. The persistent
    /// Metal lane uses this only as a slot-fill source: routing remains owned by
    /// the normal caller, and a miss falls through to direct positioned I/O.
    fn peek_resident(&self, layer: usize, expert: usize) -> Option<Arc<GhostMoeExpert>> {
        self.state
            .lock()
            .ok()?
            .entries
            .get(&(layer, expert))
            .map(|entry| Arc::clone(&entry.expert))
    }

    /// Resolve one layer's routed top-k as a batch. Cache hits are cloned under
    /// one short lock; misses are sorted by expert index (the v2 `.cghost`
    /// physical order) and read concurrently when a read pool is available.
    /// The returned vector is restored to `experts` order, so callers retain the
    /// router's exact floating-point accumulation order.
    fn get_many(&self, layer: usize, experts: &[usize]) -> Result<Vec<Arc<GhostMoeExpert>>> {
        if layer >= self.layer_budgets.len() {
            return Err(BackendError::InvalidModelMetadata(format!(
                "ghost MoE cache layer {layer} is outside its {}-layer layout",
                self.layer_budgets.len()
            )));
        }
        let mut resolved: Vec<Option<Arc<GhostMoeExpert>>> = vec![None; experts.len()];
        // Chunked prefill deliberately passes repeated route selections. Keep
        // their count and recency so an over-budget layer retains the experts
        // the prompt actually favored, rather than whichever numeric IDs were
        // inserted last after the physical reads were sorted.
        let mut missing_requests = std::collections::HashMap::<usize, (u16, u64)>::new();
        {
            let mut state = self.state.lock().expect("ghost MoE cache poisoned");
            for (slot, &expert) in experts.iter().enumerate() {
                let key = (layer, expert);
                state.touch_layer(layer);
                let now = state.clock;
                if let Some(entry) = state.entries.get_mut(&key) {
                    entry.frequency = entry.frequency.saturating_add(1);
                    entry.last_used = now;
                    resolved[slot] = Some(Arc::clone(&entry.expert));
                    state.hits = state.hits.saturating_add(1);
                } else {
                    state.misses = state.misses.saturating_add(1);
                    let request = missing_requests.entry(expert).or_insert((0, now));
                    request.0 = request.0.saturating_add(1);
                    request.1 = now;
                }
            }
        }

        // Expert groups are emitted in ascending expert order within a layer.
        // Sorting therefore gives the serial fallback monotonic file offsets;
        // par_iter preserves this indexed result order while allowing NVMe to
        // service a shallow queue of independent positioned reads.
        let mut missing: Vec<usize> = missing_requests.keys().copied().collect();
        missing.sort_unstable();
        let read_one = |&expert: &usize| -> Result<(usize, Arc<GhostMoeExpert>)> {
            Ok((expert, Arc::new(self.file.read_moe_expert(layer, expert)?)))
        };
        let mut loaded: Vec<(usize, Arc<GhostMoeExpert>)> = match &self.read_pool {
            Some(pool) if missing.len() > 1 => {
                pool.install(|| missing.par_iter().map(read_one).collect::<Result<Vec<_>>>())?
            }
            _ => missing.iter().map(read_one).collect::<Result<Vec<_>>>()?,
        };
        // I/O stays in physical order, but cache admission runs from the least
        // useful cold route to the most useful. Thus the final bounded segment
        // contains the highest-frequency, most-recent prompt experts even when
        // their numeric IDs were read first.
        loaded.sort_unstable_by_key(|(expert, _)| {
            missing_requests
                .get(expert)
                .copied()
                .expect("every loaded expert came from the missing request set")
        });

        let mut loaded_by_expert = std::collections::HashMap::with_capacity(loaded.len());
        {
            let mut state = self.state.lock().expect("ghost MoE cache poisoned");
            for (expert, loaded) in loaded {
                let key = (layer, expert);
                let size = loaded.byte_len();
                state.bytes_read = state.bytes_read.saturating_add(size as u64);

                // A second request can win the race while I/O is in flight. Use
                // its immutable entry rather than replacing it, but still report
                // the physical bytes this request actually read.
                let (request_frequency, request_last_used) = missing_requests
                    .get(&expert)
                    .copied()
                    .expect("every loaded expert came from the missing request set");
                if let Some(existing) = state.entries.get_mut(&key) {
                    existing.frequency = existing.frequency.saturating_add(request_frequency);
                    existing.last_used = existing.last_used.max(request_last_used);
                    loaded_by_expert.insert(expert, Arc::clone(&existing.expert));
                    continue;
                }

                let layer_budget = self.layer_budgets[layer];
                if size <= layer_budget {
                    while state.layer_bytes[layer].saturating_add(size) > layer_budget {
                        if !state.evict_one_from_layer(layer) {
                            break;
                        }
                    }
                    if state.layer_bytes[layer].saturating_add(size) <= layer_budget {
                        state.entries.insert(
                            key,
                            GhostMoeCacheEntry {
                                expert: Arc::clone(&loaded),
                                frequency: request_frequency,
                                last_used: request_last_used,
                            },
                        );
                        state.layer_bytes[layer] = state.layer_bytes[layer].saturating_add(size);
                        state.bytes = state.bytes.saturating_add(size);
                    }
                }
                loaded_by_expert.insert(expert, loaded);
            }
        }

        for (slot, &expert) in experts.iter().enumerate() {
            if resolved[slot].is_none() {
                resolved[slot] = loaded_by_expert.get(&expert).cloned();
            }
        }
        resolved
            .into_iter()
            .map(|value| {
                value.ok_or_else(|| {
                    BackendError::InvalidModelMetadata(
                        "ghost MoE batch read lost a requested expert".into(),
                    )
                })
            })
            .collect()
    }

    fn stats(&self) -> GhostMoeCacheStats {
        let state = self.state.lock().expect("ghost MoE cache poisoned");
        GhostMoeCacheStats {
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
            bytes_read: state.bytes_read,
            resident_experts: state.entries.len(),
            resident_bytes: state.bytes,
            budget_bytes: self.budget_bytes,
        }
    }
}

#[derive(Clone)]
struct GhostMoeLayer {
    layer_idx: usize,
    cache: Arc<GhostMoeExpertCache>,
}

/// Persistent routed-expert residency bounds. Gemma 4 routes eight experts per
/// token, so eight is the correctness floor. Sixteen preserves the established
/// default; larger opt-in slabs trade unified memory for fewer multi-megabyte
/// `.cghost` reads when routes churn across tokens.
#[cfg(any(target_os = "macos", test))]
const GHOST_METAL_EXPERT_SLOTS_MIN: usize = 8;
#[cfg(any(target_os = "macos", test))]
const GHOST_METAL_EXPERT_SLOTS_DEFAULT: usize = 16;
#[cfg(any(target_os = "macos", test))]
const GHOST_METAL_EXPERT_SLOTS_MAX: usize = 128;

#[cfg(any(target_os = "macos", test))]
fn parse_ghost_metal_slots_per_layer(value: Option<&str>) -> usize {
    value
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(GHOST_METAL_EXPERT_SLOTS_DEFAULT)
        .clamp(GHOST_METAL_EXPERT_SLOTS_MIN, GHOST_METAL_EXPERT_SLOTS_MAX)
}

#[cfg(target_os = "macos")]
fn ghost_metal_slots_per_layer_from_env() -> usize {
    let raw = std::env::var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER").ok();
    let slots = parse_ghost_metal_slots_per_layer(raw.as_deref());
    if let Some(raw) = raw {
        match raw.trim().parse::<usize>() {
            Ok(requested) if requested != slots => eprintln!(
                "[gemma4-ghost-metal] requested {requested} slots/layer; clamped to supported range {GHOST_METAL_EXPERT_SLOTS_MIN}..={GHOST_METAL_EXPERT_SLOTS_MAX}: using {slots}"
            ),
            Err(_) => eprintln!(
                "[gemma4-ghost-metal] invalid CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER={raw:?}; using default {GHOST_METAL_EXPERT_SLOTS_DEFAULT}"
            ),
            _ => {}
        }
    }
    slots
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GhostMetalSlotEntry {
    expert: usize,
    frequency: u16,
    last_used: u64,
}

/// Deterministic per-layer LFU/LRU directory for the persistent Metal expert
/// slots. The directory deliberately knows nothing about Metal: the caller
/// supplies a loader that writes a missing expert directly into the selected
/// slot's shared storage. A mapping is committed only after that loader
/// succeeds, so a short read can never make partially initialized GPU bytes
/// addressable.
///
/// Slots selected by the current route are pinned until the whole route has
/// been resolved. Consequently a route with at most `entries.len()` distinct
/// experts cannot evict one of its own earlier selections while filling later
/// misses. Eviction chooses the least frequently used unpinned slot and uses
/// oldest access as its stable tie-break.
#[derive(Debug)]
struct GhostMetalSlotDirectory {
    entries: Vec<Option<GhostMetalSlotEntry>>,
    clock: u64,
    accesses: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GhostMetalSlotLoad {
    slot: usize,
    expert: usize,
    frequency: u16,
    last_used: u64,
}

#[derive(Debug, PartialEq, Eq)]
struct GhostMetalSlotPlan {
    /// Slot IDs in the router's original top-k order.
    route_slots: Vec<usize>,
    /// Distinct cache misses. Their slots have already been invalidated in the
    /// directory and become visible again only through `commit_load`.
    loads: Vec<GhostMetalSlotLoad>,
    /// Route entries served without another slot fill. Repeated experts within
    /// the same plan count as hits because they do not cause additional I/O.
    hits: usize,
    /// Resident entries invalidated to make room for this plan's loads.
    evictions: usize,
}

impl GhostMetalSlotDirectory {
    fn new(slot_count: usize) -> Self {
        Self {
            entries: vec![None; slot_count],
            clock: 0,
            accesses: 0,
        }
    }

    fn plan(&mut self, experts: &[usize]) -> Result<GhostMetalSlotPlan> {
        let distinct = experts
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len();
        if distinct > self.entries.len() {
            return Err(BackendError::InvalidModelMetadata(format!(
                "Ghost Metal route has {distinct} distinct experts but only {} slots",
                self.entries.len()
            )));
        }

        let mut pinned = vec![false; self.entries.len()];
        let mut route_slots = Vec::with_capacity(experts.len());
        let mut loads = Vec::<GhostMetalSlotLoad>::new();
        let mut planned = std::collections::HashMap::<usize, usize>::new();
        let mut evictions = 0usize;

        for &expert in experts {
            self.clock = self.clock.saturating_add(1);
            self.accesses = self.accesses.saturating_add(1);
            // Prevent ancient prompt routes from becoming permanently
            // unevictable during a long decode. This mirrors the host cache's
            // inexpensive per-layer frequency aging.
            if self.accesses.is_multiple_of(256) {
                for entry in self.entries.iter_mut().flatten() {
                    entry.frequency = (entry.frequency / 2).max(1);
                }
            }

            if let Some((slot, entry)) =
                self.entries
                    .iter_mut()
                    .enumerate()
                    .find_map(|(slot, entry)| {
                        entry
                            .as_mut()
                            .filter(|entry| entry.expert == expert)
                            .map(|entry| (slot, entry))
                    })
            {
                entry.frequency = entry.frequency.saturating_add(1);
                entry.last_used = self.clock;
                pinned[slot] = true;
                route_slots.push(slot);
                continue;
            }

            if let Some(&load_idx) = planned.get(&expert) {
                let load = &mut loads[load_idx];
                load.frequency = load.frequency.saturating_add(1);
                load.last_used = self.clock;
                route_slots.push(load.slot);
                continue;
            }

            let slot = self
                .entries
                .iter()
                .enumerate()
                .filter(|(slot, _)| !pinned[*slot])
                .min_by_key(|(slot, entry)| match entry {
                    // Always consume a free slot before evicting a resident
                    // expert. `slot` makes the choice deterministic.
                    None => (0u8, 0u16, 0u64, *slot),
                    Some(entry) => (1, entry.frequency, entry.last_used, *slot),
                })
                .map(|(slot, _)| slot)
                .expect("distinct expert count was checked against slot count");

            // Invalidate an evicted record before I/O starts. A failed or short
            // positioned read can therefore never leave the old expert ID
            // pointing at partially overwritten bytes.
            if self.entries[slot].is_some() {
                evictions += 1;
            }
            self.entries[slot] = None;
            let load_idx = loads.len();
            loads.push(GhostMetalSlotLoad {
                slot,
                expert,
                frequency: 1,
                last_used: self.clock,
            });
            planned.insert(expert, load_idx);
            pinned[slot] = true;
            route_slots.push(slot);
        }
        let hits = experts.len().saturating_sub(loads.len());
        Ok(GhostMetalSlotPlan {
            route_slots,
            loads,
            hits,
            evictions,
        })
    }

    fn commit_load(&mut self, load: GhostMetalSlotLoad) {
        debug_assert!(self.entries.get(load.slot).is_some_and(Option::is_none));
        self.entries[load.slot] = Some(GhostMetalSlotEntry {
            expert: load.expert,
            frequency: load.frequency,
            last_used: load.last_used,
        });
    }
}

/// Retain the hottest `limit` prompt experts while preserving the original
/// repeated route sequence for LFU/recency evidence. Frequency wins; the most
/// recent occurrence breaks ties; expert ID is the final deterministic key.
fn ghost_metal_prewarm_sequence(
    routed_experts: &[usize],
    expert_count: usize,
    limit: usize,
) -> Vec<usize> {
    if expert_count == 0 || limit == 0 {
        return Vec::new();
    }
    let mut frequency = vec![0usize; expert_count];
    let mut last_used = vec![0usize; expert_count];
    for (position, &expert) in routed_experts.iter().enumerate() {
        if expert < expert_count {
            frequency[expert] += 1;
            last_used[expert] = position;
        }
    }
    let mut ranked = (0..expert_count)
        .filter(|&expert| frequency[expert] > 0)
        .collect::<Vec<_>>();
    ranked.sort_unstable_by_key(|&expert| {
        (
            std::cmp::Reverse(frequency[expert]),
            std::cmp::Reverse(last_used[expert]),
            expert,
        )
    });
    ranked.truncate(limit);
    let selected = ranked.into_iter().collect::<std::collections::HashSet<_>>();
    routed_experts
        .iter()
        .copied()
        .filter(|expert| selected.contains(expert))
        .collect()
}

/// Cumulative slot-directory and I/O telemetry. This lives under the existing
/// expert-runtime mutex, so the hot path needs no atomics or extra locking.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct GhostMetalSlotStats {
    route_lookups: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    host_fills: u64,
    prewarm_copies: u64,
    direct_reads: u64,
    direct_read_bytes: u64,
    direct_read_failures: u64,
}

#[cfg(target_os = "macos")]
impl GhostMetalSlotStats {
    fn saturating_delta(self, earlier: Self) -> Self {
        Self {
            route_lookups: self.route_lookups.saturating_sub(earlier.route_lookups),
            hits: self.hits.saturating_sub(earlier.hits),
            misses: self.misses.saturating_sub(earlier.misses),
            evictions: self.evictions.saturating_sub(earlier.evictions),
            host_fills: self.host_fills.saturating_sub(earlier.host_fills),
            prewarm_copies: self.prewarm_copies.saturating_sub(earlier.prewarm_copies),
            direct_reads: self.direct_reads.saturating_sub(earlier.direct_reads),
            direct_read_bytes: self
                .direct_read_bytes
                .saturating_sub(earlier.direct_read_bytes),
            direct_read_failures: self
                .direct_read_failures
                .saturating_sub(earlier.direct_read_failures),
        }
    }

    fn add_assign(&mut self, other: Self) {
        self.route_lookups = self.route_lookups.saturating_add(other.route_lookups);
        self.hits = self.hits.saturating_add(other.hits);
        self.misses = self.misses.saturating_add(other.misses);
        self.evictions = self.evictions.saturating_add(other.evictions);
        self.host_fills = self.host_fills.saturating_add(other.host_fills);
        self.prewarm_copies = self.prewarm_copies.saturating_add(other.prewarm_copies);
        self.direct_reads = self.direct_reads.saturating_add(other.direct_reads);
        self.direct_read_bytes = self
            .direct_read_bytes
            .saturating_add(other.direct_read_bytes);
        self.direct_read_failures = self
            .direct_read_failures
            .saturating_add(other.direct_read_failures);
    }
}

#[cfg(target_os = "macos")]
struct GhostMetalExpertLayer {
    directory: GhostMetalSlotDirectory,
    slots: crate::metal::Gemma4Q4ExpertSlots,
    stats: GhostMetalSlotStats,
}

#[cfg(target_os = "macos")]
struct GhostMetalExpertRuntime {
    engine: crate::metal::Gemma4Q4ExpertMetal,
    layers: Vec<GhostMetalExpertLayer>,
    fused_fast: bool,
    common: Option<crate::metal::Gemma4GhostCommonMetal>,
    sequence_mode: GhostMetalSequenceMode,
}

#[cfg(target_os = "macos")]
fn ghost_metal_timing_enabled() -> bool {
    std::env::var("CAMELID_GEMMA4_GHOST_METAL_TIMING")
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

#[cfg(target_os = "macos")]
fn ghost_metal_stats_enabled() -> bool {
    ghost_metal_timing_enabled()
        || std::env::var("CAMELID_GEMMA4_GHOST_METAL_STATS")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GhostMetalSequenceMode {
    Idle,
    Cpu,
    /// The prompt is advancing the authoritative host KV cache in layer-major
    /// chunks. Decode may switch to Metal only after an atomic cache import.
    HybridPrefill,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GhostPrefillPlan {
    ScalarCpu,
    CpuChunk,
    ScalarMetal,
    HybridChunk,
}

fn select_ghost_prefill_plan(
    chunk_eligible: bool,
    hybrid_enabled: bool,
    prompt_len: usize,
    required_positions: usize,
    common_capacity: Option<usize>,
) -> GhostPrefillPlan {
    match common_capacity {
        Some(capacity) if required_positions <= capacity => {
            if chunk_eligible && hybrid_enabled && prompt_len > 1 {
                GhostPrefillPlan::HybridChunk
            } else {
                GhostPrefillPlan::ScalarMetal
            }
        }
        _ if chunk_eligible && prompt_len > 1 => GhostPrefillPlan::CpuChunk,
        _ => GhostPrefillPlan::ScalarCpu,
    }
}

/// A generation request owns the persistent common-core KV state. Resetting it
/// on every exit (success, error, or cancellation) prevents a later request from
/// inheriting a hybrid/import decision if it returns before another position-zero
/// scalar step can reselect the lane.
#[cfg(target_os = "macos")]
struct GhostMetalSequenceCleanup<'a> {
    lane: &'a std::sync::Mutex<Option<GhostMetalExpertRuntime>>,
}

#[cfg(target_os = "macos")]
impl<'a> GhostMetalSequenceCleanup<'a> {
    fn new(lane: &'a std::sync::Mutex<Option<GhostMetalExpertRuntime>>) -> Self {
        Self { lane }
    }
}

#[cfg(target_os = "macos")]
impl Drop for GhostMetalSequenceCleanup<'_> {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.lane.lock() {
            if let Some(runtime) = guard.as_mut() {
                runtime.sequence_mode = GhostMetalSequenceMode::Idle;
                if let Some(common) = runtime.common.as_mut() {
                    common.reset_sequence();
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
enum GhostMetalExpertAttempt {
    Output(Vec<f32>),
    /// A positioned read or directory preparation failed. The immutable CPU
    /// Ghost cache remains authoritative and should retry the route normally.
    CpuFallback,
    /// Metal dispatch failed after successful slot preparation. Drop the lane
    /// so subsequent layers do not repeatedly pay a known-bad GPU attempt.
    DisableMetal,
}

#[cfg(target_os = "macos")]
enum GhostMetalCommonAttempt {
    Complete,
    Pending(crate::metal::Gemma4Q4ExpertPending),
    CpuFallback,
    DisableMetal,
}

/// Result of one token on the persistent Ghost common-core lane. Prompt
/// prefill advances every non-final token without materializing the hidden
/// state or running the tied vocabulary head; decode (and the final prompt
/// token) requests logits normally.
#[cfg(target_os = "macos")]
enum GhostCommonStepOutput {
    Advanced,
    Logits(Vec<f32>),
}

/// Ensures an independently queued shared branch reaches a terminal command
/// state on every error/cancellation edge before its persistent scratch can be
/// reused by another request.
#[cfg(target_os = "macos")]
struct GhostCommonPendingGuard(Option<crate::metal::Gemma4GhostCommonPending>);

#[cfg(target_os = "macos")]
impl GhostCommonPendingGuard {
    fn new(pending: crate::metal::Gemma4GhostCommonPending) -> Self {
        Self(Some(pending))
    }

    fn finish(&mut self) -> bool {
        self.0
            .take()
            .and_then(crate::metal::Gemma4GhostCommonPending::wait)
            .is_some()
    }

    fn take(&mut self) -> Option<crate::metal::Gemma4GhostCommonPending> {
        self.0.take()
    }
}

#[cfg(target_os = "macos")]
impl Drop for GhostCommonPendingGuard {
    fn drop(&mut self) {
        if let Some(pending) = self.0.take() {
            let _ = pending.wait();
        }
    }
}

/// Owns both commands that finish a fused-fast layer. The expert+tail command
/// is later in the singleton Metal queue, so draining it first also proves the
/// shared branch has reached a terminal GPU state. Drop preserves that ordering
/// on every error and cancellation edge before persistent scratch is reused.
#[cfg(target_os = "macos")]
struct GhostLayerPendingGuard {
    shared: Option<crate::metal::Gemma4GhostCommonPending>,
    tail: Option<crate::metal::Gemma4Q4ExpertPending>,
}

#[cfg(target_os = "macos")]
impl GhostLayerPendingGuard {
    fn new(
        shared: crate::metal::Gemma4GhostCommonPending,
        tail: crate::metal::Gemma4Q4ExpertPending,
    ) -> Self {
        Self {
            shared: Some(shared),
            tail: Some(tail),
        }
    }

    fn finish(&mut self) -> bool {
        let tail_ok = self
            .tail
            .take()
            .and_then(crate::metal::Gemma4Q4ExpertPending::wait)
            .is_some();
        let shared_ok = self
            .shared
            .take()
            .and_then(crate::metal::Gemma4GhostCommonPending::wait)
            .is_some();
        tail_ok && shared_ok
    }
}

#[cfg(target_os = "macos")]
impl Drop for GhostLayerPendingGuard {
    fn drop(&mut self) {
        if let Some(tail) = self.tail.take() {
            let _ = tail.wait();
        }
        if let Some(shared) = self.shared.take() {
            let _ = shared.wait();
        }
    }
}

#[cfg(target_os = "macos")]
impl GhostMetalExpertRuntime {
    fn new(layer_count: usize, fused_fast: bool, slots_per_layer: usize) -> Option<Self> {
        if !(GHOST_METAL_EXPERT_SLOTS_MIN..=GHOST_METAL_EXPERT_SLOTS_MAX).contains(&slots_per_layer)
        {
            return None;
        }
        let engine = crate::metal::Gemma4Q4ExpertMetal::new()?;
        let mut layers = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            let slots = crate::metal::Gemma4Q4ExpertSlots::new(slots_per_layer)?;
            debug_assert_eq!(slots.slot_count(), slots_per_layer);
            layers.push(GhostMetalExpertLayer {
                directory: GhostMetalSlotDirectory::new(slots_per_layer),
                slots,
                stats: GhostMetalSlotStats::default(),
            });
        }
        Some(Self {
            engine,
            layers,
            fused_fast,
            common: None,
            sequence_mode: GhostMetalSequenceMode::Idle,
        })
    }

    fn resident_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.slots.slot_count() * layer.slots.slot_stride_bytes())
            .sum()
    }

    fn slots_per_layer(&self) -> usize {
        self.layers
            .first()
            .map_or(0, |layer| layer.slots.slot_count())
    }

    fn slot_stats(&self) -> GhostMetalSlotStats {
        self.layers
            .iter()
            .fold(GhostMetalSlotStats::default(), |mut total, layer| {
                total.add_assign(layer.stats);
                total
            })
    }

    /// Seed a layer's persistent slots from immutable expert records already
    /// fetched for chunked prompt prefill. `request_sequence` contains only the
    /// selected bounded working set but retains every occurrence in prompt route
    /// order, so the directory learns real frequency/recency rather than an
    /// arbitrary expert-ID order. This is a host-memory copy, never disk I/O.
    fn prewarm_layer_from_records(
        &mut self,
        layer_idx: usize,
        request_sequence: &[usize],
        records: &std::collections::HashMap<usize, Arc<GhostMoeExpert>>,
    ) -> bool {
        if request_sequence.is_empty() {
            return true;
        }
        let Some(layer) = self.layers.get_mut(layer_idx) else {
            return false;
        };
        let expected_bytes = layer.slots.slot_record_bytes();
        let sources_valid = request_sequence
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .all(|expert| {
                records
                    .get(&expert)
                    .is_some_and(|record| record.byte_len() == expected_bytes)
            });
        if !sources_valid {
            // Preflight before `plan`: bad/missing prompt metadata must not
            // invalidate an otherwise usable resident slot.
            return false;
        }
        let plan = match layer.directory.plan(request_sequence) {
            Ok(plan) => plan,
            Err(err) => {
                eprintln!("[gemma4-ghost-metal] prompt slot plan failed: {err}");
                return false;
            }
        };
        let started = std::time::Instant::now();
        let mut copied = 0usize;
        for load in plan.loads {
            let Some(expert) = records.get(&load.expert) else {
                return false;
            };
            let (bytes, _) = expert.tensor_backing(&expert.gate_up);
            if bytes.len() != layer.slots.slot_record_bytes() {
                return false;
            }
            let Some(destination) = layer.slots.slot_bytes_mut(load.slot) else {
                return false;
            };
            destination.copy_from_slice(&bytes);
            layer.directory.commit_load(load);
            copied += 1;
        }
        layer.stats.prewarm_copies = layer.stats.prewarm_copies.saturating_add(copied as u64);
        if copied > 0 && ghost_metal_timing_enabled() {
            eprintln!(
                "[gemma4-ghost-metal-fill] layer={layer_idx} prompt={} disk=0 bytes={:.2}MiB wall={}us",
                copied,
                copied * layer.slots.slot_record_bytes() / (1024 * 1024),
                started.elapsed().as_micros(),
            );
        }
        true
    }

    /// Fill this layer's missing fixed slots directly from `.cghost`. The read
    /// pool sees disjoint mutable chunks of one shared Metal slab, so up to eight
    /// cache misses become concurrent positioned reads with no intermediate copy.
    fn prepare_layer_routes(
        &mut self,
        ghost: &GhostMoeLayer,
        experts: &[usize],
        route_scales: &[f32],
        resident_sources: &std::collections::HashMap<usize, Arc<GhostMoeExpert>>,
    ) -> Option<[crate::metal::Gemma4Q4ExpertRoute; 8]> {
        if experts.len() != 8 || route_scales.len() != 8 {
            return None;
        }
        let layer = self.layers.get_mut(ghost.layer_idx)?;
        let plan = match layer.directory.plan(experts) {
            Ok(plan) => plan,
            Err(err) => {
                eprintln!("[gemma4-ghost-metal] slot plan failed: {err}");
                return None;
            }
        };
        let GhostMetalSlotPlan {
            route_slots,
            loads,
            hits,
            evictions,
        } = plan;
        layer.stats.route_lookups = layer
            .stats
            .route_lookups
            .saturating_add(experts.len() as u64);
        layer.stats.hits = layer.stats.hits.saturating_add(hits as u64);
        layer.stats.misses = layer.stats.misses.saturating_add(loads.len() as u64);
        layer.stats.evictions = layer.stats.evictions.saturating_add(evictions as u64);

        if !loads.is_empty() {
            let fill_started = std::time::Instant::now();
            let stride = layer.slots.slot_stride_bytes();
            let record_bytes = layer.slots.slot_record_bytes();
            debug_assert_eq!(record_bytes, crate::metal::GEMMA4_Q4_EXPERT_RECORD_BYTES);
            let file = &ghost.cache.file;
            let mut host_fills = 0usize;
            let mut disk_loads = Vec::with_capacity(loads.len());
            for load in loads.iter().copied() {
                let Some((bytes, _)) = resident_sources
                    .get(&load.expert)
                    .map(|expert| expert.tensor_backing(&expert.gate_up))
                else {
                    disk_loads.push(load);
                    continue;
                };
                if bytes.len() != record_bytes {
                    disk_loads.push(load);
                    continue;
                }
                let Some(destination) = layer.slots.slot_bytes_mut(load.slot) else {
                    eprintln!(
                        "[gemma4-ghost-metal] host-cache fill selected invalid slot {}",
                        load.slot
                    );
                    return None;
                };
                destination.copy_from_slice(&bytes);
                layer.directory.commit_load(load);
                host_fills += 1;
            }
            layer.stats.host_fills = layer.stats.host_fills.saturating_add(host_fills as u64);

            let jobs = disk_loads
                .iter()
                .map(|load| (load.slot, *load))
                .collect::<std::collections::HashMap<_, _>>();
            let results: Vec<(GhostMetalSlotLoad, Result<()>)> = if disk_loads.len() == 1 {
                let load = disk_loads[0];
                let result = layer
                    .slots
                    .slot_bytes_mut(load.slot)
                    .ok_or_else(|| {
                        BackendError::InvalidModelMetadata(format!(
                            "Ghost Metal slot {} is outside the layer slab",
                            load.slot
                        ))
                    })
                    .and_then(|destination| {
                        file.read_moe_expert_into(ghost.layer_idx, load.expert, destination)
                    });
                vec![(load, result)]
            } else if disk_loads.is_empty() {
                Vec::new()
            } else if let Some(pool) = &ghost.cache.read_pool {
                let slab = layer.slots.slab_bytes_mut();
                pool.install(|| {
                    slab.par_chunks_mut(stride)
                        .enumerate()
                        .filter_map(|(slot, chunk)| {
                            jobs.get(&slot).copied().map(|load| {
                                let result = file.read_moe_expert_into(
                                    ghost.layer_idx,
                                    load.expert,
                                    &mut chunk[..record_bytes],
                                );
                                (load, result)
                            })
                        })
                        .collect()
                })
            } else {
                layer
                    .slots
                    .slab_bytes_mut()
                    .chunks_mut(stride)
                    .enumerate()
                    .filter_map(|(slot, chunk)| {
                        jobs.get(&slot).copied().map(|load| {
                            let result = file.read_moe_expert_into(
                                ghost.layer_idx,
                                load.expert,
                                &mut chunk[..record_bytes],
                            );
                            (load, result)
                        })
                    })
                    .collect()
            };

            let mut all_loaded = results.len() == disk_loads.len();
            let mut direct_reads = 0usize;
            let mut direct_read_failures = disk_loads.len().saturating_sub(results.len());
            for (load, result) in results {
                match result {
                    Ok(()) => {
                        layer.directory.commit_load(load);
                        direct_reads += 1;
                    }
                    Err(err) => {
                        all_loaded = false;
                        direct_read_failures += 1;
                        eprintln!(
                            "[gemma4-ghost-metal] layer {} expert {} direct slot read failed: {err}",
                            ghost.layer_idx, load.expert
                        );
                    }
                }
            }
            layer.stats.direct_reads = layer.stats.direct_reads.saturating_add(direct_reads as u64);
            layer.stats.direct_read_bytes = layer
                .stats
                .direct_read_bytes
                .saturating_add((direct_reads as u64).saturating_mul(record_bytes as u64));
            layer.stats.direct_read_failures = layer
                .stats
                .direct_read_failures
                .saturating_add(direct_read_failures as u64);
            if !all_loaded {
                return None;
            }
            if ghost_metal_timing_enabled() {
                eprintln!(
                    "[gemma4-ghost-metal-fill] layer={} host={} disk={} bytes={:.2}MiB wall={}us",
                    ghost.layer_idx,
                    host_fills,
                    disk_loads.len(),
                    loads.len() * record_bytes / (1024 * 1024),
                    fill_started.elapsed().as_micros(),
                );
            }
        }

        Some(std::array::from_fn(|rank| {
            crate::metal::Gemma4Q4ExpertRoute {
                slot: route_slots[rank],
                scale: route_scales[rank],
            }
        }))
    }

    /// Host-activation compatibility wrapper used by the established CPU
    /// common-core lane.
    fn run_layer(
        &mut self,
        ghost: &GhostMoeLayer,
        experts: &[usize],
        route_scales: &[f32],
        input: &[Q8_0Block],
        hidden: usize,
        resident_sources: &std::collections::HashMap<usize, Arc<GhostMoeExpert>>,
    ) -> GhostMetalExpertAttempt {
        let Some(routes) =
            self.prepare_layer_routes(ghost, experts, route_scales, resident_sources)
        else {
            return GhostMetalExpertAttempt::CpuFallback;
        };
        let Some(layer) = self.layers.get(ghost.layer_idx) else {
            return GhostMetalExpertAttempt::CpuFallback;
        };
        let mut output = vec![0.0f32; hidden];
        let diagnostics = if self.fused_fast {
            self.engine
                .run_q8_into(input, &layer.slots, &routes, &mut output)
        } else {
            self.engine
                .run_q8_into_parity(input, &layer.slots, &routes, &mut output)
        };
        match diagnostics {
            Some(_) => GhostMetalExpertAttempt::Output(output),
            None => GhostMetalExpertAttempt::DisableMetal,
        }
    }

    /// Pure device-chain wrapper used by the persistent common core. Slot I/O
    /// runs while the already-enqueued shared branch consumes Metal bandwidth;
    /// expert reduce and the MoE tail then execute in queue order.
    fn run_layer_common(
        &mut self,
        ghost: &GhostMoeLayer,
        experts: &[usize],
        route_scales: &[f32],
        resident_sources: &std::collections::HashMap<usize, Arc<GhostMoeExpert>>,
    ) -> GhostMetalCommonAttempt {
        let Some(routes) =
            self.prepare_layer_routes(ghost, experts, route_scales, resident_sources)
        else {
            return GhostMetalCommonAttempt::CpuFallback;
        };
        let Some(layer) = self.layers.get(ghost.layer_idx) else {
            return GhostMetalCommonAttempt::CpuFallback;
        };
        let Some(common) = self.common.as_mut() else {
            return GhostMetalCommonAttempt::CpuFallback;
        };
        if self.fused_fast {
            match self.engine.enqueue_common_with_tail(
                common,
                ghost.layer_idx,
                &layer.slots,
                &routes,
            ) {
                Some(pending) => GhostMetalCommonAttempt::Pending(pending),
                None => GhostMetalCommonAttempt::DisableMetal,
            }
        } else {
            match self.engine.run_common_with_tail(
                common,
                ghost.layer_idx,
                &layer.slots,
                &routes,
                false,
            ) {
                Some(_) => GhostMetalCommonAttempt::Complete,
                None => GhostMetalCommonAttempt::DisableMetal,
            }
        }
    }
}

/// Timing-gated request delta reporter. Generation already serializes the
/// persistent common-core lane, so a start/end snapshot is sufficient and
/// costs only two short mutex acquisitions outside the layer hot path.
#[cfg(target_os = "macos")]
struct GhostMetalGenerationStatsGuard<'a> {
    lane: &'a std::sync::Mutex<Option<GhostMetalExpertRuntime>>,
    start: Option<(GhostMetalSlotStats, usize, usize)>,
    started: std::time::Instant,
}

#[cfg(target_os = "macos")]
impl<'a> GhostMetalGenerationStatsGuard<'a> {
    fn new(lane: &'a std::sync::Mutex<Option<GhostMetalExpertRuntime>>) -> Self {
        let start = ghost_metal_stats_enabled()
            .then(|| {
                lane.lock().ok().and_then(|runtime| {
                    runtime.as_ref().map(|runtime| {
                        (
                            runtime.slot_stats(),
                            runtime.layers.len(),
                            runtime.slots_per_layer(),
                        )
                    })
                })
            })
            .flatten();
        Self {
            lane,
            start,
            started: std::time::Instant::now(),
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for GhostMetalGenerationStatsGuard<'_> {
    fn drop(&mut self) {
        let Some((start, layer_count, slots_per_layer)) = self.start else {
            return;
        };
        let Some(end) = self
            .lane
            .lock()
            .ok()
            .and_then(|runtime| runtime.as_ref().map(GhostMetalExpertRuntime::slot_stats))
        else {
            eprintln!(
                "[gemma4-ghost-metal-summary] slots lane became unavailable during generation"
            );
            return;
        };
        let delta = end.saturating_delta(start);
        let requests = delta.hits.saturating_add(delta.misses);
        let hit_rate = if requests == 0 {
            0.0
        } else {
            100.0 * delta.hits as f64 / requests as f64
        };
        let routed_positions = if layer_count == 0 {
            0
        } else {
            delta.route_lookups / (layer_count as u64 * 8)
        };
        let direct_mib = delta.direct_read_bytes as f64 / (1024.0 * 1024.0);
        let direct_mib_per_position = if routed_positions == 0 {
            0.0
        } else {
            direct_mib / routed_positions as f64
        };
        eprintln!(
            "[gemma4-ghost-metal-summary] layers={layer_count} slots/layer={slots_per_layer} routed_positions={routed_positions} lookups={} hits={} misses={} hit_rate={hit_rate:.1}% evictions={} host_fills={} prewarm_copies={} direct_reads={} direct_read_bytes={direct_mib:.1}MiB direct_read_per_position={direct_mib_per_position:.1}MiB read_failures={} wall={:.1}ms",
            delta.route_lookups,
            delta.hits,
            delta.misses,
            delta.evictions,
            delta.host_fills,
            delta.prewarm_copies,
            delta.direct_reads,
            delta.direct_read_failures,
            self.started.elapsed().as_secs_f64() * 1_000.0,
        );
    }
}

/// GEMV a whole pre-packed [`crate::tensor::Q4_0PackedRows8`] band against a Q8
/// activation, returning one f32 per row. One rayon task per group of 8 rows runs
/// the AVX2 [`crate::inference::q4_0_packed_gemv8`] into eight fixed output slots
/// — no repack, no per-call allocation. Bit-identical to `matvec_q_rows` on the
/// same rows (the kernel is proven bit-exact vs the scalar wire dot and the row
/// order is preserved).
fn packed_band_matvec(packed: &crate::tensor::Q4_0PackedRows8, xq: &[Q8_0Block]) -> Vec<f32> {
    debug_assert_eq!(packed.blocks_per_row, xq.len());
    let mut out = vec![0f32; packed.rows];
    out.par_chunks_mut(8).enumerate().for_each(|(g, dst)| {
        let group_block_start = g * packed.blocks_per_row;
        let mut acc = [0f32; 8];
        crate::inference::q4_0_packed_gemv8(packed, group_block_start, xq, &mut acc);
        dst.copy_from_slice(&acc);
    });
    out
}

/// One policy gate for every Ghost-MoE Metal dispatch. The CLI/UI GPU switch is
/// live, so this must be evaluated at each use rather than latched when the model
/// loads. Deterministic mode remains authoritative even if the runtime switch is
/// subsequently turned back on.
#[inline]
fn ghost_metal_acceleration_allowed(deterministic: bool, runtime_gpu_enabled: bool) -> bool {
    !deterministic && runtime_gpu_enabled
}

#[inline]
fn ghost_metal_acceleration_enabled() -> bool {
    ghost_metal_acceleration_allowed(
        crate::inference::deterministic_mode_enabled(),
        crate::cuda::gpu_accel_enabled(),
    )
}

/// Run one disk-paged Q4_0 expert projection on Metal while preserving the
/// CPU Ghost-MoE Q4_0 x Q8_0 row-dot contract. The expert remains bounded by
/// the host cache: its wire bytes are copied into one transient shared Metal
/// buffer for this projection and are not retained in an unbounded GPU cache.
///
/// Opt in with `CAMELID_GEMMA4_GHOST_METAL=1`. It remains off by default until
/// a real 26B sweep proves that transient expert uploads and command-buffer
/// count beat the CPU lane; the longer-lived fixed-slot runtime is the target.
fn ghost_metal_q4_matmul(
    weight: &WireQuant,
    rows: usize,
    inputs: &[&[Q8_0Block]],
) -> Option<Vec<Vec<f32>>> {
    #[cfg(target_os = "macos")]
    {
        let enabled = std::env::var("CAMELID_GEMMA4_GHOST_METAL").is_ok_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        });
        if !enabled || !ghost_metal_acceleration_enabled() || weight.format != WireFormat::Q4_0 {
            return None;
        }
        let output = crate::metal::try_gemma4_q4_0_matmul_q8_batch(inputs, weight.bytes(), rows)?;
        static ANNOUNCED: std::sync::Once = std::sync::Once::new();
        ANNOUNCED.call_once(|| {
            eprintln!(
                "[ghost-moe-metal] ordered Q4_0 expert GEMMs active (Metal; CPU fallback retained)"
            );
        });
        Some(output)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (weight, rows, inputs);
        None
    }
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
    /// Lazy per-expert pre-packed (interleaved 8-row) form of the two Q4_0 expert
    /// matrices, for the AVX2 GEMV expert path. Populated on first use, bounded by
    /// [`expert_pack_budget_bytes`]. `None` when the experts are not Q4_0 (the
    /// pack path only supports Q4_0) or the budget is 0.
    pack_cache: Option<std::sync::Mutex<ExpertPackCache>>,
    /// Present only on the v2 `.cghost` lane. The mmap-backed expert tensors
    /// remain untouched; selected experts come from this bounded global cache.
    ghost: Option<GhostMoeLayer>,
}

impl MoeWeights {
    /// Return expert `e`'s pre-packed (interleaved 8-row) projections, packing
    /// and caching them on first use. `None` when the pack path is disabled
    /// (non-Q4_0 experts or a 0 budget) — the caller then uses the scalar wire
    /// dot. `hidden` = n_embd (gate_up in_dim), `two_nff` = 2*n_ff_exp (gate_up
    /// row count / down in_dim). Packing happens under the cache lock but the
    /// returned `Arc` is cloned out, so the GEMV runs lock-free.
    fn packed_expert(&self, e: usize, hidden: usize, two_nff: usize) -> Option<Arc<PackedExpert>> {
        let cache = self.pack_cache.as_ref()?;
        let key = e as u16;
        {
            let guard = cache.lock().expect("expert pack cache poisoned");
            if let Some(p) = guard.get(key) {
                return Some(p);
            }
        }
        // Miss: pack this expert's two bands (outside the lock is not required —
        // the pack is the same work regardless — but we build then insert under
        // the lock so concurrent callers converge; decode is single-threaded here
        // so there is no real contention).
        let gu_blocks = hidden / 32; // gate_up in_dim = n_embd
        let down_blocks = two_nff / 2 / 32; // down in_dim = n_ff_exp
        let gate_up = self.gate_up_exps.pack_rows(e * two_nff, two_nff, gu_blocks);
        let down = self.down_exps.pack_rows(e * hidden, hidden, down_blocks);
        let packed = Arc::new(PackedExpert { gate_up, down });
        let mut guard = cache.lock().expect("expert pack cache poisoned");
        guard.insert(key, packed.clone());
        Some(packed)
    }
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

/// A v3 Ghost index binds the GGUF by cryptographic source identity, so a
/// human-facing filename is only diagnostic and may legitimately change after
/// the splice was created. Legacy indexes have no such binding and retain the
/// old basename guard as their only source-model check.
fn ghost_source_filename_admitted(
    has_source_identity: bool,
    declared_source_model: &str,
    actual_filename: Option<&str>,
) -> bool {
    has_source_identity
        || declared_source_model.is_empty()
        || actual_filename.is_none_or(|actual| declared_source_model == actual)
}

#[cfg(target_os = "macos")]
fn build_ghost_common_metal(
    path: &Path,
    store: &TensorStore,
    binding: &Gemma4Binding,
    config: &LlamaModelConfig,
    g: &Gemma4Metadata,
    layers: &[LayerWeights],
    max_positions: usize,
) -> Result<Option<crate::metal::Gemma4GhostCommonMetal>> {
    let mut refusals = Vec::new();
    let expect = |refusals: &mut Vec<String>, admitted: bool, detail: String| {
        if !admitted {
            refusals.push(detail);
        }
    };
    expect(
        &mut refusals,
        config.block_count as usize == 30,
        format!("block_count={} (expected 30)", config.block_count),
    );
    expect(
        &mut refusals,
        config.embedding_length as usize == 2_816,
        format!(
            "embedding_length={} (expected 2816)",
            config.embedding_length
        ),
    );
    expect(
        &mut refusals,
        config.attention_head_count as usize == 16,
        format!(
            "attention_head_count={} (expected 16)",
            config.attention_head_count
        ),
    );
    expect(
        &mut refusals,
        g.sliding_window as usize == 1_024,
        format!("sliding_window={} (expected 1024)", g.sliding_window),
    );
    expect(
        &mut refusals,
        g.num_kv_shared_layers == 0,
        format!(
            "num_kv_shared_layers={} (expected 0)",
            g.num_kv_shared_layers
        ),
    );
    expect(&mut refusals, max_positions > 0, "max_positions=0".into());
    expect(
        &mut refusals,
        layers.len() == 30,
        format!("loaded layer count={} (expected 30)", layers.len()),
    );
    expect(
        &mut refusals,
        binding.layers.len() == 30,
        format!("bound layer count={} (expected 30)", binding.layers.len()),
    );
    match config.moe.as_ref() {
        Some(moe) => {
            expect(
                &mut refusals,
                moe.expert_count as usize == 128,
                format!("expert_count={} (expected 128)", moe.expert_count),
            );
            expect(
                &mut refusals,
                moe.expert_used_count as usize == 8,
                format!("expert_used_count={} (expected 8)", moe.expert_used_count),
            );
        }
        None => refusals.push("MoE metadata is absent".into()),
    }
    for (layer_idx, layer) in layers.iter().enumerate() {
        let mut layer_refusals = Vec::new();
        if layer.ple_inp_gate.is_some() {
            layer_refusals.push("PLE input gate is present");
        }
        if layer.ple_proj.is_some() {
            layer_refusals.push("PLE projection is present");
        }
        if layer.post_norm.is_some() {
            layer_refusals.push("PLE post norm is present");
        }
        // Gemma 4 26B carries a learned scalar on every layer. It is not PLE:
        // the reference applies it unconditionally after the layer, and the
        // Metal tail uploads/applies this exact value in `configure_moe`.
        if !layer.ple_output_scale.is_finite() {
            layer_refusals.push("layer output scale is non-finite");
        }
        let require_q4 =
            |refusals: &mut Vec<&'static str>, name: &'static str, format: WireFormat| {
                if format != WireFormat::Q4_0 {
                    refusals.push(name);
                }
            };
        require_q4(
            &mut layer_refusals,
            "attn_q is not Q4_0",
            layer.attn_q.format,
        );
        match layer.attn_k.as_ref() {
            Some(weight) => require_q4(&mut layer_refusals, "attn_k is not Q4_0", weight.format),
            None => layer_refusals.push("attn_k is absent"),
        }
        if let Some(weight) = layer.attn_v.as_ref() {
            require_q4(&mut layer_refusals, "attn_v is not Q4_0", weight.format);
        }
        require_q4(
            &mut layer_refusals,
            "attn_output is not Q4_0",
            layer.attn_output.format,
        );
        require_q4(
            &mut layer_refusals,
            "ffn_gate is not Q4_0",
            layer.ffn_gate.format,
        );
        require_q4(
            &mut layer_refusals,
            "ffn_up is not Q4_0",
            layer.ffn_up.format,
        );
        require_q4(
            &mut layer_refusals,
            "ffn_down is not Q4_0",
            layer.ffn_down.format,
        );
        match layer.moe.as_ref() {
            Some(moe) => {
                if moe.n_expert != 128 {
                    layer_refusals.push("MoE expert count is not 128");
                }
                if moe.n_expert_used != 8 {
                    layer_refusals.push("MoE top-k is not 8");
                }
                if moe.n_ff_exp != 704 {
                    layer_refusals.push("MoE expert FF width is not 704");
                }
                if moe.ghost.is_none() {
                    layer_refusals.push("MoE weights are not Ghost-backed");
                }
            }
            None => layer_refusals.push("MoE weights are absent"),
        }
        if !layer_refusals.is_empty() {
            refusals.push(format!("layer {layer_idx}: {}", layer_refusals.join(", ")));
        }
    }
    if !refusals.is_empty() {
        for refusal in refusals {
            eprintln!("[gemma4-ghost-common] admission refused: {refusal}");
        }
        return Ok(None);
    }
    let file = std::fs::File::open(path).map_err(|source| BackendError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let pages = |name: &str| -> Result<Arc<crate::wire_mmap::WirePages>> {
        let descriptor = store.descriptor(name)?;
        crate::wire_mmap::WirePages::read_from_file(
            &file,
            descriptor.absolute_offset,
            descriptor.n_bytes as usize,
        )
    };

    let mut resident_layers = Vec::with_capacity(layers.len());
    let mut post_norm_1 = Vec::with_capacity(layers.len());
    let mut moe_configs = Vec::with_capacity(layers.len());
    for (layer_idx, (layer, bound)) in layers.iter().zip(&binding.layers).enumerate() {
        let k_descriptor = bound.attn_k.as_ref().ok_or_else(|| {
            BackendError::InvalidModelMetadata(format!(
                "Ghost common Metal layer {layer_idx} omits attn_k"
            ))
        })?;
        let q_pages = pages(&bound.attn_q.name)?;
        let k_pages = pages(&k_descriptor.name)?;
        let v_pages = bound
            .attn_v
            .as_ref()
            .map(|descriptor| pages(&descriptor.name))
            .transpose()?;
        let resident = crate::metal::Gemma4ResidentLayer::from_wire_pages_owned(
            crate::metal::GemmaWireFmt::Q4_0,
            layer.attn_norm.clone(),
            layer.q_norm.clone(),
            layer.k_norm.clone().ok_or_else(|| {
                BackendError::InvalidModelMetadata(format!(
                    "Ghost common Metal layer {layer_idx} omits attn_k_norm"
                ))
            })?,
            layer.post_attn_norm.clone(),
            layer.ffn_norm.clone(),
            layer.post_ffw_norm.clone(),
            &q_pages,
            &k_pages,
            v_pages.as_ref(),
            &pages(&bound.attn_output.name)?,
            &pages(&bound.ffn_gate.name)?,
            &pages(&bound.ffn_up.name)?,
            &pages(&bound.ffn_down.name)?,
            config.attention_head_count as usize,
            g.kv_heads_at(layer_idx) as usize,
            g.head_dim_at(layer_idx) as usize,
            g.ffn_length_at(layer_idx) as usize,
            config.rms_norm_epsilon,
        )
        .ok_or_else(|| {
            BackendError::UnsupportedModelArchitecture(
                "Metal unavailable while constructing Ghost common core".into(),
            )
        })?;
        let moe = layer
            .moe
            .as_ref()
            .expect("exact Ghost common preflight requires MoE on every layer");
        resident_layers.push(resident);
        post_norm_1.push(moe.post_norm_1.clone());
        moe_configs.push(crate::metal::Gemma4GhostMoeLayerConfig {
            router: moe.gate_inp.clone(),
            gate_input_scale: moe.gate_inp_scale.clone(),
            pre_norm_2: moe.pre_norm_2.clone(),
            post_norm_2: moe.post_norm_2.clone(),
            layer_output_scale: layer.ple_output_scale,
        });
    }
    let Some(mut common) =
        crate::metal::Gemma4GhostCommonMetal::new_26b(resident_layers, post_norm_1, max_positions)
    else {
        return Ok(None);
    };
    if !common.configure_moe(moe_configs) {
        return Ok(None);
    }
    Ok(Some(common))
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
    ghost_moe_cache: Option<Arc<GhostMoeExpertCache>>,
    /// Opt-in disk-paged Q4_0 expert engine. One reusable Metal executor serves
    /// a bounded, load-time-configured set of 16-KiB-aligned slots per layer.
    /// The inner `Option` is
    /// cleared after a Metal command failure so the established CPU Ghost lane
    /// remains the permanent fallback for the rest of the session.
    #[cfg(target_os = "macos")]
    metal_q4_experts: std::sync::Mutex<Option<GhostMetalExpertRuntime>>,
    /// The common-core KV cache is model-owned. Hold this for a complete public
    /// generation request so two callers cannot interleave position-zero resets
    /// and token steps on the same persistent Metal buffers.
    #[cfg(target_os = "macos")]
    ghost_common_generation: std::sync::Mutex<()>,
    /// Ghost-MoE keeps the decoder math on the correctness-first CPU lane for
    /// now, but the 605 MB Q6_K tied output table is already covered by Camelid's
    /// parity-tested Metal K-quant kernel. On macOS this optional no-copy head
    /// removes one full CPU sweep of that table per generated token; any Metal
    /// load/dispatch failure falls back to `token_embd.matvec` below.
    #[cfg(target_os = "macos")]
    metal_q6k_head: Option<crate::metal::Gemma4Q6KHead>,
}

/// Metal components constructed for a single-node Ghost-MoE runtime.
///
/// These are load/runtime-ownership facts only. The process-wide GPU switch
/// and deterministic-mode gate remain live policy and are applied by the API
/// health snapshot, so toggling acceleration updates the UI without reloading
/// the model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gemma4GhostMetalComponents {
    pub common: bool,
    pub experts: bool,
    pub head: bool,
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

    /// Load Gemma 4 with routed experts supplied by a v2 expert-spliced
    /// `.cghost` file. Shared weights, router, embeddings/head, and norms stay
    /// on the existing GGUF wire path; only top-k expert blobs enter the bounded
    /// global cache.
    pub fn load_ghost_moe(
        path: &Path,
        cghost: &Path,
        cache_mib: usize,
        evict_page_cache: bool,
    ) -> Result<Self> {
        let budget_bytes = cache_mib.checked_mul(1024 * 1024).ok_or_else(|| {
            BackendError::InvalidModelMetadata(format!(
                "ghost MoE cache size {cache_mib} MiB overflows usize"
            ))
        })?;
        let ghost = Arc::new(GhostFile::open_with_options(cghost, evict_page_cache)?);
        Self::load_layer_range_impl(path, None, Some((ghost, budget_bytes)))
    }
}

/// BASALT D-B2 fail-closed (DECISIONS.md D17): a ModelOpt-converted NVFP4 GGUF
/// carries per-tensor sidecar scales as separate `.scale` / `.input_scale`
/// tensors that MUST be multiplied post-matmul. The gemma4 wire lane does not
/// implement that multiply, and silently ignoring the sidecars would compute
/// wrong logits — so an NVFP4 file that carries any refuses at load, mirroring
/// the runnable-lane admission check. Pin-quantized rows (the BASALT pilot
/// artifacts, receipted at G2) carry none; the pilot's real
/// `blk.N.layer_output_scale.weight` tensors do NOT match these suffixes.
pub(crate) fn nvfp4_sidecar_check(tensors: &[crate::gguf::GgufTensorDescriptor]) -> Result<()> {
    if tensors
        .iter()
        .any(|t| t.tensor_type == GgufTensorType::NVFP4)
    {
        if let Some(sidecar) = tensors
            .iter()
            .find(|t| t.name.ends_with(".scale") || t.name.ends_with(".input_scale"))
        {
            return Err(BackendError::UnsupportedGguf(format!(
                "NVFP4 GGUF carries per-tensor sidecar scale tensor {}; the gemma4 \
                 wire lane does not apply sidecar scales and refuses rather than \
                 compute wrong logits (BASALT D-B2)",
                sidecar.name
            )));
        }
    }
    Ok(())
}

/// BASALT Amendment 3 §9 platform gate (DECISIONS.md D17 micro-decisions),
/// GABBRO M2 narrowing: NVFP4 admits on Windows AND macOS in this release, and
/// refuses on every other target (Linux et al.). macOS joined the admit set once
/// its CPU wire-lane decode was proven bit-exact on Apple Silicon (GABBRO Gate
/// G-M1, `qa/evidence-bundles/gabbro/phase1/`). This is a RUNTIME check (`cfg!`
/// inside ordinary code), deliberately NOT a `#[cfg]` wall: the decode code
/// compiles on every target, and refused callers get this named refusal instead
/// of a missing symbol. Enforced in BOTH lanes — runnable admission
/// (`runnable::admit`) and this gemma4 wire-lane load path — because either lane
/// alone could otherwise reach NVFP4 weights on an unvalidated platform. Fires
/// AFTER [`nvfp4_sidecar_check`] so the D-B2 posture stays platform-independent.
///
/// NOTE (GABBRO M2): the refusal message reads "Windows/macOS-only" and the
/// support matrices are truthed-up to Windows+macOS in this same ratchet PR
/// (Tim folded the surface truth-up into M2). macOS runs NVFP4 on both the CPU wire
/// lane and the Metal resident GPU lane (GABBRO M3 + followup), the Metal lane
/// guarded by `gemma4_metal_layer_fmt` (covered set) + `nvfp4_metal_sentinel_check`
/// (D17/T5). The fn name `nvfp4_windows_only_check` is retained as an optional
/// internal rename follow-up (pub(crate); not a user surface).
pub(crate) fn nvfp4_windows_only_check(
    tensors: &[crate::gguf::GgufTensorDescriptor],
) -> Result<()> {
    if !cfg!(target_os = "windows")
        && !cfg!(target_os = "macos")
        && tensors
            .iter()
            .any(|t| t.tensor_type == GgufTensorType::NVFP4)
    {
        return Err(BackendError::UnsupportedGguf(
            "NVFP4 is Windows/macOS-only in this release; see SUPPORT_MATRIX".into(),
        ));
    }
    Ok(())
}

/// GABBRO M3-followup (D17/T5 fail-closed): the macOS GPU lane
/// ([`Gemma4GpuRuntime::load`]) now RUNS NVFP4 layer projections (kernel
/// `nvfp4_block_linear_row_ksplit_f32y_wire`), reading their wire bytes RAW via
/// WirePages — which bypasses `WireQuant::new`'s NaN-sentinel scan. So the T5 guard
/// lives here: scan every NVFP4 tensor's UE4M3 scale bytes and refuse `0x7F`/`0xFF`
/// (the pin's CPU and CUDA backends disagree on `0xFF`, so such a file has no
/// well-defined cross-backend oracle), matching the CPU wire lane. Clean NVFP4 — and
/// files without NVFP4 — admit. The shared [`crate::tensor::nvfp4_find_nan_scale`]
/// does the byte scan; called once the mmap is available.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn nvfp4_metal_sentinel_check(
    tensors: &[crate::gguf::GgufTensorDescriptor],
    mmap: &GgufWireMmap,
) -> Result<()> {
    for t in tensors {
        if t.tensor_type == GgufTensorType::NVFP4 {
            let wire = mmap.bytes(t.absolute_offset, t.n_bytes as usize)?;
            if let Some(block_idx) = crate::tensor::nvfp4_find_nan_scale(wire) {
                return Err(BackendError::InvalidTensorData(format!(
                    "tensor {}: NVFP4 block {block_idx} carries a NaN-sentinel UE4M3 \
                     scale byte (0x7F/0xFF) — refusing on the Metal resident lane per D17/T5",
                    t.name
                )));
            }
        }
    }
    Ok(())
}

/// GABBRO M3-followup: the Metal resident lane's covered layer-projection formats —
/// Q8_0 / Q4_0 / NVFP4, each a parity-gated GPU GEMV. Any other format refuses TYPED
/// and NAMED (invariant I-unknown-type, L4 cell) rather than mis-binding. Extracted so
/// it unit-tests without a real model; the load site probes layer-0 `attn_q` (the
/// export quantizes every layer's projections alike).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn gemma4_metal_layer_fmt(tensor_type: GgufTensorType) -> Result<crate::metal::GemmaWireFmt> {
    // The only non-test caller is `Gemma4GpuRuntime::load`, which is `cfg(macos)`-gated, so
    // the non-macOS lib build sees this as dead; allow it there (the `#[cfg(test)]` covered-set
    // test still exercises it on every platform). Mirrors `nvfp4_metal_sentinel_check`.
    match tensor_type {
        GgufTensorType::Q8_0 => Ok(crate::metal::GemmaWireFmt::Q8_0),
        GgufTensorType::Q4_0 => Ok(crate::metal::GemmaWireFmt::Q4_0),
        GgufTensorType::NVFP4 => Ok(crate::metal::GemmaWireFmt::Nvfp4),
        other => Err(BackendError::UnsupportedTensorType(format!(
            "gemma4 GPU runtime supports Q8_0/Q4_0/NVFP4 layer projections; \
             layer 0 attn_q is {other:?}"
        ))),
    }
}

/// BASALT Amendment 3 review fix (CUDA lane typed refusal), extended at SHA_E
/// review and lifted at Phase 4: the CUDA-resident gemma4 lane repacks layer
/// projections via `GemmaLayerQuant::from_wire`, whose catch-all PANICS on any
/// format outside its covered set. Phase 4 (BASALT G4) added the NVFP4 raw-wire
/// GEMV (`nvfp4_gemv`), so the covered set is now Q8_0/Q4_0/Q4_1/NVFP4. Every
/// remaining lane-uncovered format — the K-quants the CPU wire lane serves (the
/// campaign's own Q4K-mm / Q4_K_M rows) — must still refuse with a typed, named
/// error before that panic seam is reachable (invariant I-unknown-type, L3 cell).
/// cfg-independent over [`WireFormat`]s so it unit-tests without CUDA hardware;
/// the `cfg(feature = "cuda")` load site ([`Gemma4CudaResident::load`]) wires it.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn nvfp4_cuda_lane_check<I: IntoIterator<Item = WireFormat>>(formats: I) -> Result<()> {
    for f in formats {
        match f {
            WireFormat::Q8_0 | WireFormat::Q4_0 | WireFormat::Q4_1 | WireFormat::Nvfp4 => {}
            other => {
                return Err(BackendError::UnsupportedGguf(format!(
                    "gemma4 CUDA-resident lane covers Q8_0/Q4_0/Q4_1/NVFP4 layer projections; \
                     {other:?} is not wired (the CPU lane serves it) — refusing instead \
                     of reaching the repack panic (BASALT I-unknown-type, L3)"
                )));
            }
        }
    }
    Ok(())
}

/// BASALT Amendment 3 review fix (step-boundary proof): the forced-decode loop's
/// boundary bookkeeping, extracted so a unit test can prove the off-by-one
/// contract with a scripted step closure and no model. Drives `step` (feed one
/// token, return the next prediction state) over the forced list starting from
/// the prompt-end prediction, guaranteeing:
///
/// 1. `observe(i, state)` sees the prediction computed BEFORE `forced[i]` is fed
///    as the next input (the teacher-forcing boundary);
/// 2. exactly `forced.len()` observations fire;
/// 3. the FINAL forced token is never fed (its prediction is already observed;
///    feeding it would only compute an unrecorded extra step).
///
/// [`Gemma4Runtime::forced_decode`] rewires through this; the real forward step
/// is untouched.
pub(crate) fn drive_forced_steps<P, E>(
    forced: &[u32],
    prompt_end_prediction: P,
    mut step: impl FnMut(u32) -> std::result::Result<P, E>,
    mut observe: impl FnMut(usize, &P),
) -> std::result::Result<(), E> {
    let mut prediction = prompt_end_prediction;
    for (i, &tok) in forced.iter().enumerate() {
        observe(i, &prediction);
        if i + 1 < forced.len() {
            prediction = step(tok)?;
        }
    }
    Ok(())
}

/// Drive scalar prompt prefill while projecting the tied output head exactly
/// once, at the final prompt position. This tiny model-independent seam makes
/// the call-count contract testable without constructing a multi-gigabyte
/// Gemma runtime.
fn drive_scalar_prefill<T, E>(
    tokens: &[u32],
    mut step: impl FnMut(u32, usize, bool) -> std::result::Result<Option<T>, E>,
) -> std::result::Result<T, E> {
    let (&last_token, prefix) = tokens
        .split_last()
        .expect("prefill validates that the prompt is non-empty");
    for (pos, &token) in prefix.iter().enumerate() {
        let output = step(token, pos, false)?;
        debug_assert!(output.is_none());
    }
    Ok(step(last_token, prefix.len(), true)?
        .expect("the final scalar prefill step projects the output head"))
}

impl Gemma4Runtime {
    /// Merged byte spans of the wire tensors a `range` shard actually streams,
    /// for scoping the background `MADV_WILLNEED` warm-up.
    ///
    /// Readahead is bounded by device bandwidth, so advising the whole mapping
    /// spends it on bytes this shard never streams. A gemma4 GGUF's data section
    /// opens with `per_layer_token_embd` (2.5GB on E2B) — a *gather-only* table,
    /// one row per layer per token — so warming all of it front-loads the wrong
    /// bytes and, under memory pressure, evicts the layer weights the first step
    /// is actually blocked on. Every other non-layer tensor is either small or
    /// streamed whole each step (the tied head), so only the gather table is
    /// excluded.
    fn shard_warm_spans(
        gguf: &GgufFile,
        range: &std::ops::Range<usize>,
        exclude_routed_experts: bool,
    ) -> Vec<(usize, usize)> {
        let wanted = |name: &str| -> bool {
            if exclude_routed_experts
                && (name.ends_with(".ffn_gate_up_exps.weight")
                    || name.ends_with(".ffn_down_exps.weight"))
            {
                return false;
            }
            match name.strip_prefix("blk.") {
                Some(rest) => rest
                    .split_once('.')
                    .and_then(|(idx, _)| idx.parse::<usize>().ok())
                    .is_some_and(|layer| range.contains(&layer)),
                None => name != "per_layer_token_embd.weight",
            }
        };
        let mut spans: Vec<(usize, usize)> = gguf
            .tensors
            .iter()
            .filter(|t| t.n_bytes > 0 && wanted(&t.name))
            .map(|t| (t.absolute_offset as usize, t.n_bytes as usize))
            .collect();
        spans.sort_unstable();
        // Coalesce touching/overlapping spans so the kernel sees a few long
        // sequential runs rather than hundreds of small ones.
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
        for (offset, len) in spans {
            match merged.last_mut() {
                Some((m_off, m_len)) if offset <= m_off.saturating_add(*m_len) => {
                    *m_len = (offset + len).saturating_sub(*m_off).max(*m_len);
                }
                _ => merged.push((offset, len)),
            }
        }
        merged
    }

    /// Load only the given contiguous global layer range (None = all layers).
    /// Fails closed if the range would separate a KV-sharing layer from the
    /// cache it reads (the split must keep every shared layer on the same shard
    /// as its source layer).
    pub fn load_layer_range(path: &Path, range: Option<std::ops::Range<usize>>) -> Result<Self> {
        Self::load_layer_range_impl(path, range, None)
    }

    fn load_layer_range_impl(
        path: &Path,
        range: Option<std::ops::Range<usize>>,
        ghost_moe: Option<(Arc<GhostFile>, usize)>,
    ) -> Result<Self> {
        let gguf = read_metadata(path)?;
        // BASALT D-B2 fail-closed (DECISIONS.md D17): a ModelOpt-converted NVFP4
        // GGUF carries per-tensor sidecar scales as separate `.scale` /
        // `.input_scale` tensors that MUST be multiplied post-matmul. This wire
        // lane does not implement that multiply, and silently ignoring the
        // sidecars would compute wrong logits — so an NVFP4 file that carries
        // any refuses here, mirroring the runnable-lane admission check.
        // Pin-quantized rows (the BASALT pilot artifacts) carry none.
        nvfp4_sidecar_check(&gguf.tensors)?;
        // BASALT Amendment 3 §9 + GABBRO M2: NVFP4 admits on Windows and macOS
        // in this release (other targets refuse) — a runtime platform gate
        // (after the sidecar check so D-B2 stays platform-independent), mirrored
        // in runnable admission.
        nvfp4_windows_only_check(&gguf.tensors)?;
        let config = LlamaModelConfig::from_gguf(&gguf)?;
        let g = config.gemma4.clone().ok_or_else(|| {
            BackendError::UnsupportedModelArchitecture("not a gemma4 model".into())
        })?;
        let binding = Gemma4Binding::bind(&gguf, &config)?;
        let ghost_moe_cache = match ghost_moe {
            Some((ghost, budget_bytes)) => {
                let moe = config.moe.as_ref().ok_or_else(|| {
                    BackendError::UnsupportedModelArchitecture(
                        "ghost MoE mode requires a Gemma 4 mixture-of-experts model".into(),
                    )
                })?;
                ghost.validate_moe_layout(
                    config.block_count as usize,
                    moe.expert_count as usize,
                    moe.expert_used_count as usize,
                )?;
                ghost.validate_moe_binding(&binding, moe.expert_count as usize)?;
                let filename = path.file_name().and_then(|name| name.to_str());
                if !ghost_source_filename_admitted(
                    ghost.index.source_identity.is_some(),
                    &ghost.index.source_model,
                    filename,
                ) {
                    return Err(BackendError::InvalidModelMetadata(format!(
                        "legacy .cghost source model {:?} does not match GGUF filename {:?}",
                        ghost.index.source_model,
                        filename.unwrap_or("<non-UTF-8>")
                    )));
                }
                ghost.validate_moe_source_identity(path, &binding, moe.expert_count as usize)?;
                Some(Arc::new(GhostMoeExpertCache::new(ghost, budget_bytes)))
            }
            None => None,
        };
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
            // Warm only the spans this shard streams, not all 5GB — see
            // `shard_warm_spans`. Still off the loading thread: the advisory
            // blocks on macOS over USB until the range is resident.
            let spans = Self::shard_warm_spans(&gguf, &range, ghost_moe_cache.is_some());
            std::thread::spawn(move || {
                for (offset, len) in spans {
                    mmap.advise_willneed_range(offset, len);
                }
            });
        }
        let q8 = |name: &str| WireQuant::new(&store, &mmap, name);
        // Matvec-role loads (projections, expert bands, the tied head) refuse
        // Q5_K typed at load — it is gather-only in this lane and would
        // otherwise panic at forward time (I-unknown-type, SHA_E3).
        let q8m = |name: &str| -> Result<WireQuant> { q8(name)?.require_matvec_capable(name) };
        let f32t = |name: &str| -> Result<Vec<f32>> { Ok(store.load_cpu_f32(name)?.data) };

        let mut layers = Vec::with_capacity(range.len());
        for (local_idx, l) in binding.layers[range.clone()].iter().enumerate() {
            let layer_idx = range.start + local_idx;
            layers.push(LayerWeights {
                attn_norm: f32t(&l.attn_norm.name)?,
                attn_q: q8m(&l.attn_q.name)?,
                attn_k: l.attn_k.as_ref().map(|d| q8m(&d.name)).transpose()?,
                attn_v: l.attn_v.as_ref().map(|d| q8m(&d.name)).transpose()?,
                attn_output: q8m(&l.attn_output.name)?,
                q_norm: f32t(&l.attn_q_norm.name)?,
                k_norm: l.attn_k_norm.as_ref().map(|d| f32t(&d.name)).transpose()?,
                post_attn_norm: f32t(&l.post_attention_norm.name)?,
                ffn_norm: f32t(&l.ffn_norm.name)?,
                ffn_gate: q8m(&l.ffn_gate.name)?,
                ffn_up: q8m(&l.ffn_up.name)?,
                ffn_down: q8m(&l.ffn_down.name)?,
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
                        let gate_up = q8m(&m.gate_up_exps.name)?;
                        let down = q8m(&m.down_exps.name)?;
                        let two_nff =
                            gate_up.element_count / (n_expert * config.embedding_length as usize);
                        // Enable the AVX2 pre-pack expert path only when BOTH expert
                        // matrices are Q4_0 (the pack format) and a budget is set.
                        let budget = expert_pack_budget_bytes();
                        let pack_cache = if ghost_moe_cache.is_none()
                            && budget > 0
                            && gate_up.format == WireFormat::Q4_0
                            && down.format == WireFormat::Q4_0
                        {
                            Some(std::sync::Mutex::new(ExpertPackCache::new(budget)))
                        } else {
                            None
                        };
                        Ok(MoeWeights {
                            gate_inp: f32t(&m.gate_inp.name)?,
                            gate_inp_scale: f32t(&m.gate_inp_scale.name)?,
                            gate_up_exps: gate_up,
                            down_exps: down,
                            down_exps_scale: f32t(&m.down_exps_scale.name)?,
                            pre_norm_2: f32t(&m.pre_norm_2.name)?,
                            post_norm_1: f32t(&m.post_norm_1.name)?,
                            post_norm_2: f32t(&m.post_norm_2.name)?,
                            n_expert,
                            n_expert_used: moe_meta.expert_used_count as usize,
                            n_ff_exp: two_nff / 2,
                            pack_cache,
                            ghost: ghost_moe_cache.as_ref().map(|cache| GhostMoeLayer {
                                layer_idx,
                                cache: Arc::clone(cache),
                            }),
                        })
                    })
                    .transpose()?,
            });
        }

        #[cfg(target_os = "macos")]
        let metal_q4_experts = {
            let flag = |name: &str| {
                std::env::var(name).is_ok_and(|value| {
                    matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "on" | "yes"
                    )
                })
            };
            // Deterministic mode is process-pinned, so avoid compiling kernels or
            // allocating the persistent slot slab there. The live GPU toggle is instead
            // checked at every dispatch, allowing the UI to re-enable an already
            // loaded non-deterministic model without a reload.
            let enabled = flag("CAMELID_GEMMA4_GHOST_METAL_SLOTS")
                && !crate::inference::deterministic_mode_enabled();
            let fused_fast = flag("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST");
            let common_enabled = flag("CAMELID_GEMMA4_GHOST_METAL_COMMON");
            let slots_per_layer = if enabled {
                ghost_metal_slots_per_layer_from_env()
            } else {
                GHOST_METAL_EXPERT_SLOTS_DEFAULT
            };
            let moe_meta = config.moe.as_ref();
            let exact_geometry = ghost_moe_cache.is_some()
                && range.start == 0
                && range.end == block_count
                && config.embedding_length as usize == 2_816
                && moe_meta.is_some_and(|moe| {
                    moe.expert_count as usize == 128 && moe.expert_used_count as usize == 8
                })
                && layers.iter().all(|layer| {
                    layer.moe.as_ref().is_some_and(|moe| {
                        moe.n_ff_exp == 704
                            && moe.n_expert_used == 8
                            && moe.gate_up_exps.format == WireFormat::Q4_0
                            && moe.down_exps.format == WireFormat::Q4_0
                    })
                });
            let exact_records = if enabled && exact_geometry {
                let cache = ghost_moe_cache
                    .as_ref()
                    .expect("exact Ghost Metal geometry requires a Ghost cache");
                let expert_count = moe_meta
                    .expect("exact Ghost Metal geometry requires MoE metadata")
                    .expert_count as usize;
                match cache.file.validate_moe_expert_record_layouts(
                    block_count,
                    expert_count,
                    crate::metal::GEMMA4_Q4_EXPERT_RECORD_BYTES,
                ) {
                    Ok(()) => true,
                    Err(err) => {
                        eprintln!(
                            "[gemma4-ghost-metal] persistent slot record layout refused: {err}"
                        );
                        false
                    }
                }
            } else {
                false
            };
            let mut lane = if enabled && exact_geometry && exact_records {
                GhostMetalExpertRuntime::new(block_count, fused_fast, slots_per_layer)
            } else {
                None
            };
            if common_enabled {
                let max_positions = std::env::var("CAMELID_GEMMA4_GHOST_METAL_CONTEXT")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .filter(|&value| value > 0)
                    .unwrap_or(4_096)
                    .min(config.context_length as usize);
                if let Some(runtime) = lane.as_mut() {
                    match build_ghost_common_metal(
                        path,
                        &store,
                        &binding,
                        &config,
                        &g,
                        &layers,
                        max_positions,
                    ) {
                        Ok(Some(mut common)) => {
                            let q4_simd_fast = common.enable_fused_fast_q4(fused_fast);
                            let geometry = common.geometry();
                            eprintln!(
                                "[gemma4-ghost-common] ACTIVE: full Metal common core, context={} positions, f32 KV={:.2}GiB, router/shared/expert/tail device-chained, mode={}, q4-row={}",
                                geometry.max_positions,
                                geometry.kv_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                                if fused_fast { "fused-fast" } else { "CPU-GeGLU parity" },
                                if q4_simd_fast { "simdgroup-ordered" } else { "scalar-ordered" },
                            );
                            runtime.common = Some(common);
                        }
                        Ok(None) => eprintln!(
                            "[gemma4-ghost-common] requested but exact Gemma 4 26B Q4_0/no-PLE geometry was not admitted; CPU common core remains active"
                        ),
                        Err(error) => eprintln!(
                            "[gemma4-ghost-common] construction failed: {error}; CPU common core remains active"
                        ),
                    }
                } else {
                    eprintln!(
                        "[gemma4-ghost-common] requested but persistent expert slots are unavailable; CPU common core remains active"
                    );
                }
            }
            if let Some(lane) = lane.as_ref() {
                eprintln!(
                    "[gemma4-ghost-metal] persistent Q4_0 slots enabled: layers={} slots/layer={} resident={:.2}GiB mode={}",
                    block_count,
                    lane.slots_per_layer(),
                    lane.resident_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                    if fused_fast { "fused-fast" } else { "CPU-GeGLU parity" },
                );
            } else if enabled {
                eprintln!(
                    "[gemma4-ghost-metal] persistent slots unavailable or model geometry is not exact Gemma 4 26B Q4_0; using CPU Ghost experts"
                );
            }
            std::sync::Mutex::new(lane)
        };

        let first_kv_shared = config.block_count as usize - g.num_kv_shared_layers as usize;
        // Bind the common tied table once so the CPU fallback and the optional
        // no-copy Metal head share the exact same validated GGUF descriptor.
        let token_embd = q8m(&binding.token_embedding.name)?;
        let output_norm = f32t(&binding.output_norm.name)?;
        #[cfg(target_os = "macos")]
        let metal_q6k_head = {
            let explicitly_disabled = std::env::var("CAMELID_GEMMA4_GHOST_METAL_HEAD")
                .is_ok_and(|value| value == "0" || value.eq_ignore_ascii_case("false"));
            let eligible = ghost_moe_cache.is_some()
                && range.end == block_count
                && token_embd.format == WireFormat::Q6K
                && !explicitly_disabled
                && !crate::inference::deterministic_mode_enabled();
            let head = if eligible {
                match &token_embd.backing {
                    WireBacking::Mmap { mmap, offset } => crate::metal::Gemma4Q6KHead::new(
                        Arc::clone(mmap),
                        *offset,
                        token_embd.bytes().len(),
                        &output_norm,
                        token_embd.element_count / config.embedding_length as usize,
                        g.final_logit_softcapping.unwrap_or(0.0),
                        config.rms_norm_epsilon,
                    ),
                    WireBacking::Owned { .. } => None,
                }
            } else {
                None
            };
            if head.is_some() {
                eprintln!("[gemma4-ghost] Metal Q6_K tied head enabled (no-copy, file-backed)");
            } else if eligible {
                eprintln!("[gemma4-ghost] Metal Q6_K tied head unavailable; using CPU fallback");
            }
            head
        };
        Ok(Self {
            tokenizer,
            first_layer: range.start,
            // The tied head matvecs token_embd on the tail shard, so it takes
            // the matvec-role guard; per_layer_token_embd stays gather-only
            // (plain q8) — Q5_K is legitimate there.
            token_embd,
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
            output_norm,
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
            ghost_moe_cache,
            #[cfg(target_os = "macos")]
            metal_q4_experts,
            #[cfg(target_os = "macos")]
            ghost_common_generation: std::sync::Mutex::new(()),
            #[cfg(target_os = "macos")]
            metal_q6k_head,
            layers,
            config,
            g,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// `None` on the normal resident/mmap lane; live bounded-cache counters on
    /// the v2 Ghost-MoE lane.
    pub fn ghost_moe_cache_stats(&self) -> Option<GhostMoeCacheStats> {
        self.ghost_moe_cache.as_ref().map(|cache| cache.stats())
    }

    /// Metal components still owned by this Ghost runtime. The persistent
    /// expert lane can disable itself after a command failure, so this is read
    /// live rather than latched at model load.
    pub fn ghost_metal_components(&self) -> Gemma4GhostMetalComponents {
        #[cfg(target_os = "macos")]
        {
            let (experts, common) = self
                .metal_q4_experts
                .lock()
                .map(|guard| {
                    let experts = guard.is_some();
                    let common = guard
                        .as_ref()
                        .and_then(|runtime| runtime.common.as_ref())
                        .is_some_and(crate::metal::Gemma4GhostCommonMetal::moe_configured);
                    (experts, common)
                })
                .unwrap_or((false, false));
            Gemma4GhostMetalComponents {
                common,
                experts,
                head: self.metal_q6k_head.is_some(),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            Gemma4GhostMetalComponents::default()
        }
    }

    /// Backwards-compatible common-core construction probe used by the real
    /// fixture gate. Live GPU/deterministic policy is deliberately not folded
    /// into this model-owned state.
    pub fn ghost_common_metal_active(&self) -> bool {
        self.ghost_metal_components().common
    }

    /// Select one authoritative KV lane before prompt position zero. The budget
    /// covers every forward the request may need, so a configured 4K Metal cache
    /// never fails halfway through a longer request: that request stays on CPU
    /// from the start instead.
    fn prepare_ghost_prefill(
        &self,
        prompt_len: usize,
        future_forwards: usize,
    ) -> Result<GhostPrefillPlan> {
        let required_positions = prompt_len.checked_add(future_forwards).ok_or_else(|| {
            BackendError::InvalidModelMetadata(
                "Gemma 4 prompt plus decode position count overflows usize".into(),
            )
        })?;
        if required_positions > self.config.context_length as usize {
            return Err(BackendError::InvalidModelMetadata(format!(
                "Gemma 4 request needs {required_positions} positions, exceeding the model context length {}",
                self.config.context_length
            )));
        }
        let chunk_eligible = self.ghost_moe_cache.is_some() && self.supports_chunk_forward();
        let hybrid_enabled = !std::env::var("CAMELID_GEMMA4_GHOST_HYBRID_PREFILL")
            .is_ok_and(|value| value == "0" || value.eq_ignore_ascii_case("false"));

        #[cfg(target_os = "macos")]
        {
            let gpu_allowed = ghost_metal_acceleration_enabled();
            let mut guard = self.metal_q4_experts.lock().map_err(|_| {
                BackendError::InvalidModelMetadata("Ghost Metal runtime mutex is poisoned".into())
            })?;
            let Some(runtime) = guard.as_mut() else {
                return Ok(select_ghost_prefill_plan(
                    chunk_eligible,
                    hybrid_enabled,
                    prompt_len,
                    required_positions,
                    None,
                ));
            };
            let configured_capacity = runtime
                .common
                .as_ref()
                .filter(|common| common.moe_configured())
                .map(crate::metal::Gemma4GhostCommonMetal::max_positions);
            let common_capacity = gpu_allowed.then_some(configured_capacity).flatten();
            let plan = select_ghost_prefill_plan(
                chunk_eligible,
                hybrid_enabled,
                prompt_len,
                required_positions,
                common_capacity,
            );
            if let Some(common) = runtime.common.as_mut() {
                common.reset_sequence();
            }
            runtime.sequence_mode = match plan {
                GhostPrefillPlan::ScalarCpu | GhostPrefillPlan::CpuChunk => {
                    GhostMetalSequenceMode::Cpu
                }
                GhostPrefillPlan::ScalarMetal => GhostMetalSequenceMode::Metal,
                GhostPrefillPlan::HybridChunk => GhostMetalSequenceMode::HybridPrefill,
            };
            if let Some(capacity) = configured_capacity {
                if required_positions > capacity {
                    eprintln!(
                        "[gemma4-ghost-common] request needs {required_positions} positions but Metal capacity is {capacity}; using the CPU KV lane from position zero"
                    );
                }
            }
            Ok(plan)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(select_ghost_prefill_plan(
                chunk_eligible,
                hybrid_enabled,
                prompt_len,
                required_positions,
                None,
            ))
        }
    }

    /// Commit a completed CPU chunk prefill to the persistent common-core cache.
    /// Any refusal leaves the host cache authoritative and pins the rest of this
    /// request to CPU; host rows are released only after all Metal layers import.
    fn finish_ghost_hybrid_prefill(
        &self,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
        positions: usize,
    ) -> Result<bool> {
        #[cfg(target_os = "macos")]
        {
            let started = std::time::Instant::now();
            let imported = {
                let mut guard = self.metal_q4_experts.lock().map_err(|_| {
                    BackendError::InvalidModelMetadata(
                        "Ghost Metal runtime mutex is poisoned".into(),
                    )
                })?;
                let Some(runtime) = guard.as_mut() else {
                    return Ok(false);
                };
                if runtime.sequence_mode != GhostMetalSequenceMode::HybridPrefill
                    || !ghost_metal_acceleration_enabled()
                {
                    runtime.sequence_mode = GhostMetalSequenceMode::Cpu;
                    if let Some(common) = runtime.common.as_mut() {
                        common.reset_sequence();
                    }
                    return Ok(false);
                }
                let Some(common) = runtime.common.as_mut() else {
                    runtime.sequence_mode = GhostMetalSequenceMode::Cpu;
                    return Ok(false);
                };
                match common.import_position_major_kv(kc, vc, positions) {
                    Ok(()) => {
                        runtime.sequence_mode = GhostMetalSequenceMode::Metal;
                        true
                    }
                    Err(error) => {
                        eprintln!(
                            "[gemma4-ghost-common] CPU prefill KV import refused: {error}; continuing this request on CPU"
                        );
                        common.reset_sequence();
                        runtime.sequence_mode = GhostMetalSequenceMode::Cpu;
                        false
                    }
                }
            };
            if imported {
                // Drop the per-position allocations, not merely their f32 contents.
                // At a 4K context this returns about 1.72 GiB before decode begins.
                for layer in kc.iter_mut().chain(vc.iter_mut()) {
                    layer.clear();
                    layer.shrink_to_fit();
                }
                if ghost_metal_timing_enabled() {
                    eprintln!(
                        "[gemma4-ghost-common] imported {positions} CPU-prefilled positions into Metal KV in {:.1}ms",
                        started.elapsed().as_secs_f64() * 1_000.0
                    );
                }
            }
            Ok(imported)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (kc, vc, positions);
            Ok(false)
        }
    }

    #[cfg(target_os = "macos")]
    fn lock_ghost_common_generation(&self) -> Result<std::sync::MutexGuard<'_, ()>> {
        self.ghost_common_generation.lock().map_err(|_| {
            BackendError::InvalidModelMetadata(
                "Ghost common Metal generation mutex is poisoned".into(),
            )
        })
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

    /// Logit-vector length of this model's tied head (`token_embd` rows) — the
    /// exact length `step` returns, and therefore the bound the BASALT harness
    /// uses to validate teacher-forced token ids before decoding.
    pub fn vocab_size(&self) -> usize {
        self.token_embd.element_count / self.hidden_size()
    }

    /// Final RMSNorm + tied vocabulary projection + logit soft-cap. Ghost-MoE
    /// on macOS first tries the persistent no-copy Q6_K Metal head; every other
    /// model/platform, and any soft Metal failure, executes the established CPU
    /// wire-dot path unchanged.
    fn project_logits(&self, hidden: &[f32]) -> Vec<f32> {
        #[cfg(target_os = "macos")]
        if ghost_metal_acceleration_enabled() {
            if let Some(head) = self.metal_q6k_head.as_ref() {
                if let Some(logits) = head.forward(hidden) {
                    return logits;
                }
            }
        }
        self.project_logits_cpu(hidden)
    }

    fn project_logits_cpu(&self, hidden: &[f32]) -> Vec<f32> {
        let last = rms_norm(
            hidden,
            Some(&self.output_norm),
            self.config.rms_norm_epsilon,
        );
        let mut logits = self
            .token_embd
            .matvec(self.hidden_size(), self.vocab_size(), &last);
        if let Some(cap) = self.g.final_logit_softcapping {
            soft_cap_in_place(&mut logits, cap);
        }
        logits
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

    #[cfg(target_os = "macos")]
    fn try_ghost_common_step(
        &self,
        token: u32,
        pos: usize,
        project_head: bool,
    ) -> Result<Option<GhostCommonStepOutput>> {
        let gpu_allowed = ghost_metal_acceleration_enabled();
        let mut guard = self.metal_q4_experts.lock().map_err(|_| {
            BackendError::InvalidModelMetadata("Ghost Metal runtime mutex is poisoned".into())
        })?;
        let Some(runtime) = guard.as_mut() else {
            return Ok(None);
        };

        // Public generation can pin CPU/Hybrid before position zero. Preserve
        // those decisions; Idle direct callers and a stale/completed Metal
        // sequence still get the legacy position-zero reset.
        if pos == 0
            && matches!(
                runtime.sequence_mode,
                GhostMetalSequenceMode::Idle | GhostMetalSequenceMode::Metal
            )
        {
            runtime.sequence_mode = if gpu_allowed
                && runtime
                    .common
                    .as_ref()
                    .is_some_and(crate::metal::Gemma4GhostCommonMetal::moe_configured)
            {
                if let Some(common) = runtime.common.as_mut() {
                    common.reset_sequence();
                }
                GhostMetalSequenceMode::Metal
            } else {
                GhostMetalSequenceMode::Cpu
            };
        }
        match runtime.sequence_mode {
            GhostMetalSequenceMode::Cpu
            | GhostMetalSequenceMode::HybridPrefill
            | GhostMetalSequenceMode::Idle => return Ok(None),
            GhostMetalSequenceMode::Metal if !gpu_allowed => {
                return Err(BackendError::UnsupportedModelArchitecture(
                    "Ghost common Metal was disabled during an active request; retry the request so Camelid can select one KV lane from position zero".into(),
                ));
            }
            GhostMetalSequenceMode::Metal => {}
        }

        let token_started = std::time::Instant::now();
        let forward = (|| -> Result<GhostCommonStepOutput> {
            let hidden = self.config.embedding_length as usize;
            let h0: Vec<f32> = self
                .token_embd
                .dequantize_elements(token as usize * hidden, hidden)?
                .iter()
                .map(|value| value * (hidden as f32).sqrt())
                .collect();
            let common = runtime.common.as_mut().ok_or_else(|| {
                BackendError::UnsupportedModelArchitecture(
                    "Ghost common Metal state disappeared during an active request".into(),
                )
            })?;
            if pos >= common.max_positions() {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "Ghost common Metal context capacity {} is smaller than requested position {pos}; increase CAMELID_GEMMA4_GHOST_METAL_CONTEXT and reload",
                    common.max_positions()
                )));
            }
            if !common.write_hidden(&h0) {
                return Err(BackendError::InvalidTensorData(
                    "Ghost common Metal rejected the token embedding".into(),
                ));
            }

            let mut previous_layer_pending: Option<(usize, GhostLayerPendingGuard)> = None;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if !ghost_metal_acceleration_enabled() {
                    return Err(BackendError::UnsupportedModelArchitecture(
                        "GPU acceleration was disabled during a Ghost common token; retry the request from position zero".into(),
                    ));
                }
                let head_dim = self.g.head_dim_at(layer_idx) as usize;
                let theta = self.g.rope_freq_base_at(layer_idx);
                let factors = if self.g.is_sliding_layer(layer_idx) {
                    None
                } else {
                    self.rope_factors.as_deref()
                };
                let mut cos_t = vec![0.0f32; head_dim / 2];
                let mut sin_t = vec![0.0f32; head_dim / 2];
                for i in 0..head_dim / 2 {
                    let mut frequency = theta.powf(-(2.0 * i as f32) / head_dim as f32);
                    if let Some(factors) = factors {
                        frequency /= factors[i];
                    }
                    let (sin, cos) = (pos as f32 * frequency).sin_cos();
                    cos_t[i] = cos;
                    sin_t[i] = sin;
                }

                let attention_router_pending = runtime
                    .common
                    .as_mut()
                    .and_then(|common| {
                        common.enqueue_attention_router(layer_idx, &cos_t, &sin_t, pos)
                    })
                    .ok_or_else(|| {
                        BackendError::UnsupportedModelArchitecture(format!(
                            "Ghost common Metal attention/router failed to enqueue at layer {layer_idx} position {pos}"
                        ))
                    })?;
                // Queue shared immediately behind the fused attention/router
                // command. Waiting exposes only 128 logits while Metal has already
                // advanced into shared Q4 work; route selection and direct slot
                // reads overlap that work with no idle queue bubble.
                let mut shared_pending = GhostCommonPendingGuard::new(
                    runtime
                    .common
                    .as_mut()
                    .and_then(|common| common.enqueue_shared_branch(layer_idx))
                    .ok_or_else(|| {
                        BackendError::UnsupportedModelArchitecture(format!(
                            "Ghost common Metal shared branch failed to enqueue at layer {layer_idx}"
                        ))
                    })?,
                );
                if attention_router_pending.wait().is_none() {
                    return Err(BackendError::UnsupportedModelArchitecture(format!(
                        "Ghost common Metal attention/router failed at layer {layer_idx} position {pos}"
                    )));
                }
                // The singleton Metal queue completed this layer's
                // attention/router only after the preceding fused expert/tail.
                // Drain that older handle now: it is an immediate status check,
                // not another GPU synchronization point on the steady-state path.
                if let Some((pending_layer, mut pending)) = previous_layer_pending.take() {
                    if !pending.finish() {
                        return Err(BackendError::UnsupportedModelArchitecture(format!(
                            "Ghost common Metal asynchronous expert/tail failed at layer {pending_layer}"
                        )));
                    }
                }
                let logits = runtime
                    .common
                    .as_ref()
                    .and_then(crate::metal::Gemma4GhostCommonMetal::read_router_logits)
                    .ok_or_else(|| {
                        BackendError::UnsupportedModelArchitecture(
                            "Ghost common Metal router logits were unavailable".into(),
                        )
                    })?;
                if logits.iter().any(|value| !value.is_finite()) {
                    return Err(BackendError::InvalidTensorData(format!(
                        "Ghost common Metal router produced non-finite logits at layer {layer_idx}"
                    )));
                }
                let max_logit = logits.iter().copied().fold(f32::MIN, f32::max);
                let mut probabilities: Vec<f32> = logits
                    .iter()
                    .map(|value| (*value - max_logit).exp())
                    .collect();
                let probability_sum: f32 = probabilities.iter().sum();
                if !probability_sum.is_finite() || probability_sum <= 0.0 {
                    return Err(BackendError::InvalidTensorData(format!(
                        "Ghost common Metal router softmax failed at layer {layer_idx}"
                    )));
                }
                for probability in &mut probabilities {
                    *probability /= probability_sum;
                }
                let moe = layer.moe.as_ref().ok_or_else(|| {
                    BackendError::InvalidModelMetadata(format!(
                        "Ghost common Metal layer {layer_idx} has no MoE weights"
                    ))
                })?;
                let mut experts: Vec<usize> = (0..moe.n_expert).collect();
                experts.sort_unstable_by(|&a, &b| {
                    probabilities[b]
                        .partial_cmp(&probabilities[a])
                        .expect("finite router probabilities")
                        .then(a.cmp(&b))
                });
                experts.truncate(moe.n_expert_used);
                if std::env::var_os("CAMELID_GEMMA4_ROUTE_TRACE").is_some() {
                    eprintln!("[route-metal] l={layer_idx} e={experts:?}");
                }
                let selected_sum = experts
                    .iter()
                    .map(|&expert| probabilities[expert])
                    .sum::<f32>()
                    .max(6.103_515e-5);
                let route_scales: Vec<f32> = experts
                    .iter()
                    .map(|&expert| {
                        moe.down_exps_scale[expert] * (probabilities[expert] / selected_sum)
                    })
                    .collect();
                let ghost = moe.ghost.as_ref().ok_or_else(|| {
                    BackendError::InvalidModelMetadata(format!(
                        "Ghost common Metal layer {layer_idx} is not Ghost-backed"
                    ))
                })?;

                let resident_sources = std::collections::HashMap::new();
                let expert_attempt =
                    runtime.run_layer_common(ghost, &experts, &route_scales, &resident_sources);
                match expert_attempt {
                    GhostMetalCommonAttempt::Pending(tail) => {
                        let shared = shared_pending.take().ok_or_else(|| {
                            BackendError::UnsupportedModelArchitecture(format!(
                                "Ghost common Metal lost the shared command at layer {layer_idx}"
                            ))
                        })?;
                        previous_layer_pending =
                            Some((layer_idx, GhostLayerPendingGuard::new(shared, tail)));
                    }
                    GhostMetalCommonAttempt::Complete => {
                        if !shared_pending.finish() {
                            return Err(BackendError::UnsupportedModelArchitecture(format!(
                                "Ghost common Metal shared branch failed at layer {layer_idx}"
                            )));
                        }
                    }
                    GhostMetalCommonAttempt::CpuFallback => {
                        return Err(BackendError::InvalidTensorData(format!(
                            "Ghost common Metal slot preparation failed at layer {layer_idx}"
                        )));
                    }
                    GhostMetalCommonAttempt::DisableMetal => {
                        return Err(BackendError::UnsupportedModelArchitecture(format!(
                            "Ghost common Metal expert/tail dispatch failed at layer {layer_idx}"
                        )));
                    }
                }
            }
            // Layer 29 has no following attention/router fence to imply its
            // completion. Drain it before the head reads hidden, and also before
            // a headless prefill step returns and permits the next token.
            if let Some((pending_layer, mut pending)) = previous_layer_pending.take() {
                if !pending.finish() {
                    return Err(BackendError::UnsupportedModelArchitecture(format!(
                        "Ghost common Metal asynchronous expert/tail failed at final layer {pending_layer}"
                    )));
                }
            }
            if !project_head {
                return Ok(GhostCommonStepOutput::Advanced);
            }
            let final_hidden = runtime
                .common
                .as_ref()
                .map(crate::metal::Gemma4GhostCommonMetal::read_hidden)
                .ok_or_else(|| {
                    BackendError::UnsupportedModelArchitecture(
                        "Ghost common Metal final hidden was unavailable".into(),
                    )
                })?;
            Ok(GhostCommonStepOutput::Logits(
                self.project_logits(&final_hidden),
            ))
        })();

        match forward {
            Ok(output) => {
                if std::env::var("CAMELID_GEMMA4_GHOST_COMMON_TIMING")
                    .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
                {
                    eprintln!(
                        "[gemma4-ghost-common-token] pos={pos} layers={} head={} wall={}us effective={:.2}tok/s",
                        self.layers.len(),
                        if project_head { "on" } else { "off" },
                        token_started.elapsed().as_micros(),
                        1.0 / token_started.elapsed().as_secs_f64().max(f64::EPSILON),
                    );
                }
                Ok(Some(output))
            }
            Err(error) if pos == 0 => {
                eprintln!(
                    "[gemma4-ghost-common] first-position Metal attempt failed: {error}; restarting this request on the CPU lane"
                );
                runtime.sequence_mode = GhostMetalSequenceMode::Cpu;
                if let Some(common) = runtime.common.as_mut() {
                    common.reset_sequence();
                }
                Ok(None)
            }
            Err(error) => {
                runtime.sequence_mode = GhostMetalSequenceMode::Idle;
                Err(error)
            }
        }
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
        #[cfg(target_os = "macos")]
        if let Some(output) = self.try_ghost_common_step(token, pos, true)? {
            return match output {
                GhostCommonStepOutput::Logits(logits) => Ok(logits),
                GhostCommonStepOutput::Advanced => unreachable!("head was requested"),
            };
        }
        match self.step_range(token, pos, None, kc, vc)? {
            Gemma4StepOutput::Logits(logits) => Ok(logits),
            Gemma4StepOutput::Hidden(_) => Err(BackendError::InvalidModelMetadata(
                "step() requires a runtime that owns the final layer; use step_range \
                 on interior shards"
                    .into(),
            )),
        }
    }

    /// Advance one scalar token's transformer/KV state without running the
    /// output vocabulary projection. Used for every non-final prompt token;
    /// decode still calls [`Self::step`] and therefore returns logits.
    fn step_without_head(
        &self,
        token: u32,
        pos: usize,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
    ) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(output) = self.try_ghost_common_step(token, pos, false)? {
            return match output {
                GhostCommonStepOutput::Advanced => Ok(()),
                GhostCommonStepOutput::Logits(_) => unreachable!("head was suppressed"),
            };
        }
        match self.step_range_with_head(token, pos, None, kc, vc, false)? {
            Gemma4StepOutput::Hidden(_) => Ok(()),
            Gemma4StepOutput::Logits(_) => unreachable!("head was suppressed"),
        }
    }

    /// True when the batched [`Self::step_chunk`] forward is usable: single-node
    /// (this runtime owns every layer including the head), with either dense rows
    /// or Ghost-backed MoE rows. Mmap-backed MoE still uses the scalar lane: its
    /// packed-expert cache has different batching/lifetime tradeoffs, while the
    /// Ghost lane needs chunking to keep prompt prefill from rereading the same
    /// routed expert once per token.
    fn supports_chunk_forward(&self) -> bool {
        self.first_layer == 0
            && self.first_layer + self.layers.len() == self.config.block_count as usize
            && self
                .layers
                .iter()
                .all(|lw| lw.moe.as_ref().is_none_or(|moe| moe.ghost.is_some()))
    }

    /// Speculative decode remains on its previously-proven dense-only surface.
    /// Ghost chunking is enabled for prompt prefill independently; widening the
    /// draft/rollback lane belongs behind its own parity and performance gate.
    fn supports_speculative_chunk_forward(&self) -> bool {
        self.supports_chunk_forward() && self.layers.iter().all(|lw| lw.moe.is_none())
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
        self.step_chunk_with_head(tokens, start_pos, kc, vc, true)
    }

    /// Shared chunk body. Prompt prefill requests only the final row's tied
    /// head, while speculative verification and parity tests need every row.
    #[allow(clippy::needless_range_loop)]
    fn step_chunk_with_head(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
        all_logits: bool,
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
            let xn_rows: Vec<Vec<f32>> = hs
                .iter()
                .map(|h| rms_norm(h, Some(&lw.attn_norm), eps))
                .collect();
            let xnq = SharedActivationBatch::new(&xn_rows);
            let mut q_rows = lw.attn_q.matmul_proj(q_dim, &xnq);
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
                    .matmul_proj(kv_dim, &xnq);
                let mut v_rows = match lw.attn_v.as_ref() {
                    Some(wv) => wv.matmul_proj(kv_dim, &xnq),
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
            let mut attn_rows: Vec<Vec<f32>> = Vec::with_capacity(kk);
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
                attn_rows.push(attn);
            }
            // o-projection batched, then residual + post-attn norm per token.
            let attn_b = SharedActivationBatch::new(&attn_rows);
            let o_rows = lw.attn_output.matmul_proj(hidden, &attn_b);
            for i in 0..kk {
                let on = rms_norm(&o_rows[i], Some(&lw.post_attn_norm), eps);
                for (a, b) in hs[i].iter_mut().zip(&on) {
                    *a += b;
                }
            }

            // --- FFN, batched ---
            // Dense rows share each weight pass across the chunk. Ghost-MoE rows
            // additionally route the whole chunk first, load the union of selected
            // experts once, and reuse each immutable expert record for every row
            // that selected it (see `moe_layer_ffn_chunk`).
            let ffn_out_rows = if lw.moe.is_some() {
                self.moe_layer_ffn_chunk(li, &hs)?
            } else {
                let ffn_rows: Vec<Vec<f32>> = hs
                    .iter()
                    .map(|h| rms_norm(h, Some(&lw.ffn_norm), eps))
                    .collect();
                let ffnq = SharedActivationBatch::new(&ffn_rows);
                let gate_rows = lw.ffn_gate.matmul_proj(ffn_dim, &ffnq);
                let up_rows = lw.ffn_up.matmul_proj(ffn_dim, &ffnq);
                let act_rows: Vec<Vec<f32>> = (0..kk)
                    .map(|i| {
                        gate_rows[i]
                            .iter()
                            .zip(&up_rows[i])
                            .map(|(g, u)| gelu_tanh(*g) * u)
                            .collect()
                    })
                    .collect();
                let actq = SharedActivationBatch::new(&act_rows);
                lw.ffn_down
                    .matmul_proj(hidden, &actq)
                    .into_iter()
                    .map(|mlp| rms_norm(&mlp, Some(&lw.post_ffw_norm), eps))
                    .collect()
            };
            for i in 0..kk {
                for (a, b) in hs[i].iter_mut().zip(&ffn_out_rows[i]) {
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

        // --- tied head ---
        let vocab = self.config.vocab_size.unwrap() as usize;
        if !all_logits {
            // Prompt prefill consumes only the final position. Avoid allocating
            // and computing (K-1) enormous 262K-vocabulary rows that are thrown
            // away immediately by the generation loop.
            let logits =
                self.project_logits(hs.last().expect("non-empty chunk has a final hidden row"));
            return Ok(vec![logits]);
        }

        // The all-logits form is normally the dense speculative-verify lane and
        // retains its shared-weight batched CPU projection below. Ghost's parity
        // harness also requests every row; keep that harness on the same Metal
        // head as scalar `step` so it compares the chunk scheduler, not two head
        // implementations with different floating-point reduction orders.
        #[cfg(target_os = "macos")]
        if ghost_metal_acceleration_enabled() && self.metal_q6k_head.is_some() {
            return Ok(hs.iter().map(|h| self.project_logits(h)).collect());
        }

        // Speculative verification needs a vocabulary row at every position.
        let lastq: Vec<Vec<f32>> = hs
            .iter()
            .map(|h| rms_norm(h, Some(&self.output_norm), eps))
            .collect();
        // Family-routed like every projection (SHA_E3): the old open-coded
        // match sent only Q6_K through the Q8_K family, so a Q4_K tied head
        // hit `matmul_q`'s K-quant unreachable! on this batched path.
        let lastb = SharedActivationBatch::new(&lastq);
        let mut logits_rows: Vec<Vec<f32>> = self.token_embd.matmul_proj(vocab, &lastb);
        if let Some(cap) = self.g.final_logit_softcapping {
            for logits in logits_rows.iter_mut() {
                soft_cap_in_place(logits, cap);
            }
        }
        Ok(logits_rows)
    }

    fn ghost_metal_q4_is_enabled(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            ghost_metal_acceleration_enabled()
                && self
                    .metal_q4_experts
                    .lock()
                    .is_ok_and(|lane| lane.is_some())
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    fn ghost_metal_q4_slots_per_layer(&self) -> Option<usize> {
        #[cfg(target_os = "macos")]
        {
            if !ghost_metal_acceleration_enabled() {
                return None;
            }
            self.metal_q4_experts
                .lock()
                .ok()
                .and_then(|lane| lane.as_ref().map(GhostMetalExpertRuntime::slots_per_layer))
                .filter(|&slots| slots > 0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }

    fn try_ghost_metal_q4_experts(
        &self,
        ghost: &GhostMoeLayer,
        experts: &[usize],
        route_scales: &[f32],
        input: &[Q8_0Block],
        hidden: usize,
    ) -> Option<Vec<f32>> {
        #[cfg(target_os = "macos")]
        {
            if !ghost_metal_acceleration_enabled() {
                return None;
            }
            // Clone host-cache sources before taking the Metal mutex. Arc bytes
            // remain valid across eviction and the strict lock ordering avoids
            // coupling cache admission to synchronous GPU use.
            let resident_sources = experts
                .iter()
                .copied()
                .filter_map(|expert| {
                    ghost
                        .cache
                        .peek_resident(ghost.layer_idx, expert)
                        .map(|record| (expert, record))
                })
                .collect::<std::collections::HashMap<_, _>>();
            let mut guard = self.metal_q4_experts.lock().ok()?;
            let lane = guard.as_mut()?;
            match lane.run_layer(
                ghost,
                experts,
                route_scales,
                input,
                hidden,
                &resident_sources,
            ) {
                GhostMetalExpertAttempt::Output(output) => Some(output),
                GhostMetalExpertAttempt::CpuFallback => None,
                GhostMetalExpertAttempt::DisableMetal => {
                    eprintln!(
                        "[gemma4-ghost-metal] Metal expert dispatch failed; disabling persistent slots and using CPU Ghost experts"
                    );
                    *guard = None;
                    None
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (ghost, experts, route_scales, input, hidden);
            None
        }
    }

    fn prewarm_ghost_metal_q4(
        &self,
        layer_idx: usize,
        request_sequence: &[usize],
        records: &std::collections::HashMap<usize, Arc<GhostMoeExpert>>,
    ) {
        #[cfg(target_os = "macos")]
        if ghost_metal_acceleration_enabled() {
            if let Ok(mut guard) = self.metal_q4_experts.lock() {
                if let Some(lane) = guard.as_mut() {
                    let _ = lane.prewarm_layer_from_records(layer_idx, request_sequence, records);
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = (layer_idx, request_sequence, records);
    }

    /// Layer-major sibling of [`Self::moe_layer_ffn`] for Ghost prompt chunks.
    ///
    /// Routing is still computed independently for every token row. After all
    /// routes are known, the union of selected experts is fetched in one cache
    /// operation and each expert's immutable record is reused by every routed
    /// row. Expert projections are batched per expert, but each token's final
    /// mixture is accumulated in its original top-k route order. That last
    /// detail keeps the floating-point addition order identical to the scalar
    /// forward rather than making output depend on the union's expert order.
    fn moe_layer_ffn_chunk(&self, li: usize, attn_rows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let hidden = self.config.embedding_length as usize;
        let eps = self.config.rms_norm_epsilon;
        let l = self.first_layer + li;
        let ffn_dim = self.g.ffn_length_at(l) as usize;
        let lw = &self.layers[li];
        let moe = lw
            .moe
            .as_ref()
            .expect("moe_layer_ffn_chunk called on a non-MoE layer");
        let ghost = moe
            .ghost
            .as_ref()
            .expect("chunk forward admits only Ghost-backed MoE layers");

        // Preserve the scalar router operation order row-for-row: norm, scale,
        // F32 matvec, all-expert softmax, probability sort, selected-weight sum.
        let inv = 1.0f32 / (hidden as f32).sqrt();
        let mut route_indices = Vec::with_capacity(attn_rows.len());
        let mut route_probs = Vec::with_capacity(attn_rows.len());
        let mut route_wsums = Vec::with_capacity(attn_rows.len());
        let mut selected = vec![false; moe.n_expert];
        for attn_out in attn_rows {
            let mut r = rms_norm(attn_out, None, eps);
            for (rv, sv) in r.iter_mut().zip(&moe.gate_inp_scale) {
                *rv = *rv * inv * sv;
            }
            let logits = f32_matvec(&moe.gate_inp, hidden, moe.n_expert, &r);
            let maxl = logits.iter().cloned().fold(f32::MIN, f32::max);
            let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
            let sum: f32 = probs.iter().sum();
            for p in probs.iter_mut() {
                *p /= sum;
            }
            let mut idx: Vec<usize> = (0..moe.n_expert).collect();
            idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
            idx.truncate(moe.n_expert_used);
            if std::env::var_os("CAMELID_GEMMA4_ROUTE_TRACE").is_some() {
                eprintln!("[route] l={l} e={idx:?}");
            }
            let mut wsum: f32 = idx.iter().map(|&e| probs[e]).sum();
            wsum = wsum.max(6.103_515e-5);
            for &e in &idx {
                selected[e] = true;
            }
            route_indices.push(idx);
            route_probs.push(probs);
            route_wsums.push(wsum);
        }
        let unique_experts: Vec<usize> = selected
            .iter()
            .enumerate()
            .filter_map(|(e, &is_selected)| is_selected.then_some(e))
            .collect();
        let routed_experts: Vec<usize> = route_indices
            .iter()
            .flat_map(|indices| indices.iter().copied())
            .collect();

        // Dense shared-expert branch. The batched projections use the exact same
        // row-dot kernels as the scalar lane, only reusing each weight row across
        // all prompt activations.
        let dense_mlp = || {
            let xn_rows: Vec<Vec<f32>> = attn_rows
                .iter()
                .map(|attn_out| rms_norm(attn_out, Some(&lw.ffn_norm), eps))
                .collect();
            let xnq = SharedActivationBatch::new(&xn_rows);
            let gate_rows = lw.ffn_gate.matmul_proj(ffn_dim, &xnq);
            let up_rows = lw.ffn_up.matmul_proj(ffn_dim, &xnq);
            let act_rows: Vec<Vec<f32>> = gate_rows
                .iter()
                .zip(&up_rows)
                .map(|(gate, up)| {
                    gate.iter()
                        .zip(up)
                        .map(|(g, u)| gelu_tanh(*g) * u)
                        .collect()
                })
                .collect();
            let actq = SharedActivationBatch::new(&act_rows);
            lw.ffn_down
                .matmul_proj(hidden, &actq)
                .into_iter()
                .map(|mlp| rms_norm(&mlp, Some(&moe.post_norm_1), eps))
                .collect::<Vec<_>>()
        };

        // Disk reads and the independent dense branch overlap. `get_many`
        // returns in the caller's route order even if its deduplicated,
        // physical-order reads complete out of order. Repeated route IDs are
        // intentional cache-frequency evidence.
        let (paged_experts, mut mlp_rows) = rayon::join(
            || ghost.cache.get_many(ghost.layer_idx, &routed_experts),
            dense_mlp,
        );
        let paged_experts = paged_experts?;
        let mut expert_records: Vec<Option<Arc<GhostMoeExpert>>> = vec![None; moe.n_expert];
        for (&e, expert) in routed_experts.iter().zip(paged_experts) {
            expert_records[e] = Some(expert);
        }

        // The chunk already owns every immutable prompt expert Arc, including
        // records that a small host-cache segment could not retain. Rank the
        // layer's working set by frequency then recency and fill its configured
        // persistent capacity
        // into persistent Metal slots while CPU expert math consumes the same
        // read-only records. This converts the first decode's likely routes from
        // cold positioned reads into slot hits without adding another disk pass.
        let prewarm = if let Some(slots_per_layer) = self.ghost_metal_q4_slots_per_layer() {
            let request_sequence =
                ghost_metal_prewarm_sequence(&routed_experts, moe.n_expert, slots_per_layer);
            let records = request_sequence
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .map(|expert| {
                    (
                        expert,
                        Arc::clone(
                            expert_records[expert]
                                .as_ref()
                                .expect("ranked prompt expert record was not resolved"),
                        ),
                    )
                })
                .collect::<std::collections::HashMap<_, _>>();
            Some((request_sequence, records))
        } else {
            None
        };

        let cur_moe_rows: Vec<Vec<f32>> = attn_rows
            .iter()
            .map(|attn_out| rms_norm(attn_out, Some(&moe.pre_norm_2), eps))
            .collect();
        let two_nff = 2 * moe.n_ff_exp;
        let compute_expert_outputs = || -> Result<Vec<Vec<Option<Vec<f32>>>>> {
            // expert_outputs[row][route_slot] is filled in expert-major compute
            // order, then consumed below in route-major accumulation order.
            let mut expert_outputs: Vec<Vec<Option<Vec<f32>>>> = route_indices
                .iter()
                .map(|idx| (0..idx.len()).map(|_| None).collect())
                .collect();
            for &e in &unique_experts {
                let routed: Vec<(usize, usize)> = route_indices
                    .iter()
                    .enumerate()
                    .flat_map(|(row, idx)| {
                        idx.iter()
                            .enumerate()
                            .filter_map(move |(slot, &selected_e)| {
                                (selected_e == e).then_some((row, slot))
                            })
                    })
                    .collect();
                debug_assert!(!routed.is_empty());
                let x_rows: Vec<Vec<f32>> = routed
                    .iter()
                    .map(|&(row, _)| cur_moe_rows[row].clone())
                    .collect();
                let xq = SharedActivationBatch::new(&x_rows);
                let expert = expert_records[e]
                    .as_ref()
                    .expect("unique Ghost expert record was not resolved");
                let gate_up =
                    WireQuant::from_ghost_tensor(expert, &expert.gate_up, "ghost gate_up_exps")?;
                let xq_refs: Vec<&[Q8_0Block]> = xq.q8_0().iter().map(Vec::as_slice).collect();
                let gate_up_rows = ghost_metal_q4_matmul(&gate_up, two_nff, &xq_refs)
                    .unwrap_or_else(|| gate_up.matmul_proj(two_nff, &xq));
                let act_rows: Vec<Vec<f32>> = gate_up_rows
                    .iter()
                    .map(|gate_up| {
                        (0..moe.n_ff_exp)
                            .map(|o| gelu_tanh(gate_up[o]) * gate_up[o + moe.n_ff_exp])
                            .collect()
                    })
                    .collect();
                let actq = SharedActivationBatch::new(&act_rows);
                let down = WireQuant::from_ghost_tensor(expert, &expert.down, "ghost down_exps")?;
                let actq_refs: Vec<&[Q8_0Block]> = actq.q8_0().iter().map(Vec::as_slice).collect();
                let y_rows = ghost_metal_q4_matmul(&down, hidden, &actq_refs)
                    .unwrap_or_else(|| down.matmul_proj(hidden, &actq));
                for ((row, slot), y) in routed.into_iter().zip(y_rows) {
                    expert_outputs[row][slot] = Some(y);
                }
            }
            Ok(expert_outputs)
        };
        let (_, expert_outputs) = rayon::join(
            || {
                if let Some((request_sequence, records)) = &prewarm {
                    self.prewarm_ghost_metal_q4(ghost.layer_idx, request_sequence, records);
                }
            },
            compute_expert_outputs,
        );
        let mut expert_outputs = expert_outputs?;

        let mut out = Vec::with_capacity(attn_rows.len());
        for row in 0..attn_rows.len() {
            let mut moe_acc = vec![0f32; hidden];
            for (slot, &e) in route_indices[row].iter().enumerate() {
                let w = route_probs[row][e] / route_wsums[row];
                let scale = moe.down_exps_scale[e] * w;
                let y = expert_outputs[row][slot]
                    .take()
                    .expect("routed Ghost expert output was not computed");
                for (a, yv) in moe_acc.iter_mut().zip(&y) {
                    *a += yv * scale;
                }
            }
            let cur_moe = rms_norm(&moe_acc, Some(&moe.post_norm_2), eps);
            let mut combined = std::mem::take(&mut mlp_rows[row]);
            for (c, m) in combined.iter_mut().zip(&cur_moe) {
                *c += m;
            }
            out.push(rms_norm(&combined, Some(&lw.post_ffw_norm), eps));
        }
        Ok(out)
    }

    /// Compute the full two-branch FFN output for a MoE (A4B/26B) layer.
    ///
    /// `li` is the LOCAL layer index (must have `self.layers[li].moe.is_some()`);
    /// `attn_out` is the post-attention residual (the current hidden state before
    /// the FFN). Returns `ffn_out`, the composed dense+expert result that the
    /// caller ADDS to the residual (`h += ffn_out`).
    ///
    /// This is the single source of truth for the MoE FFN math: the CPU forward
    /// loop calls it for MoE layers, and the CUDA-resident lane reuses it to run
    /// the (bit-exact) FFN on the CPU while attention stays on the GPU. Keeping the
    /// math in one place means the two runtimes cannot diverge on the FFN.
    pub(crate) fn moe_layer_ffn(&self, li: usize, attn_out: &[f32]) -> Result<Vec<f32>> {
        let hidden = self.config.embedding_length as usize;
        let eps = self.config.rms_norm_epsilon;
        let l = self.first_layer + li;
        let ffn_dim = self.g.ffn_length_at(l) as usize;
        let lw = &self.layers[li];
        let moe = lw
            .moe
            .as_ref()
            .expect("moe_layer_ffn called on a non-MoE layer");

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
        idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(moe.n_expert_used);
        if std::env::var_os("CAMELID_GEMMA4_ROUTE_TRACE").is_some() {
            eprintln!("[route] l={l} e={idx:?}");
        }
        // sum-normalize the selected weights (clamped), w_scale=1.
        let mut wsum: f32 = idx.iter().map(|&e| probs[e]).sum();
        wsum = wsum.max(6.103_515e-5);

        // Dense "shared expert" MLP branch: ffn_norm -> parallel GeGLU -> down.
        // On the Ghost-MoE lane its math is independent of the already-computed
        // router, so overlap it with the selected experts' positioned reads.
        // The two branches are still combined in the exact original order below.
        let dense_mlp = || {
            let xn = rms_norm(attn_out, Some(&lw.ffn_norm), eps);
            let xnq = SharedActivation::new(&xn);
            let gate = lw.ffn_gate.matvec_proj(ffn_dim, &xnq);
            let up = lw.ffn_up.matvec_proj(ffn_dim, &xnq);
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| gelu_tanh(*g) * u)
                .collect();
            let mlp = lw.ffn_down.matvec(ffn_dim, hidden, &act);
            // Dense branch keeps its own post-norm (post_norm_1).
            rms_norm(&mlp, Some(&moe.post_norm_1), eps)
        };
        let cur_moe = rms_norm(attn_out, Some(&moe.pre_norm_2), eps);
        let cur_moe_q = SharedActivation::new(&cur_moe);
        // Materialize the tiny Q8_0 activation before `rayon::join`: the lazy
        // `SharedActivation` uses single-threaded OnceCell and is intentionally
        // not Sync, while its completed Vec of plain Q8 blocks is safe to share
        // with the independent Metal/read worker.
        let cur_moe_q8 = cur_moe_q.q8_0().to_vec();
        let route_scales: Vec<f32> = idx
            .iter()
            .map(|&e| moe.down_exps_scale[e] * (probs[e] / wsum))
            .collect();
        let (paged_experts, mlp, metal_moe_acc) = match &moe.ghost {
            Some(ghost) if self.ghost_metal_q4_is_enabled() => {
                // The Metal lane owns direct slot reads and both dominant expert
                // projections. Keep the independent shared-expert MLP on CPU at
                // the same time. If the opt-in lane soft-fails, retry through the
                // immutable host cache without changing the established result.
                let (metal, mlp) = rayon::join(
                    || {
                        self.try_ghost_metal_q4_experts(
                            ghost,
                            &idx,
                            &route_scales,
                            &cur_moe_q8,
                            hidden,
                        )
                    },
                    dense_mlp,
                );
                match metal {
                    Some(acc) => (None, mlp, Some(acc)),
                    None => (
                        Some(ghost.cache.get_many(ghost.layer_idx, &idx)?),
                        mlp,
                        None,
                    ),
                }
            }
            Some(ghost) => {
                let (experts, mlp) =
                    rayon::join(|| ghost.cache.get_many(ghost.layer_idx, &idx), dense_mlp);
                (Some(experts?), mlp, None)
            }
            None => (None, dense_mlp(), None),
        };

        let two_nff = 2 * moe.n_ff_exp;
        let moe_acc = if let Some(metal_moe_acc) = metal_moe_acc {
            metal_moe_acc
        } else {
            let mut moe_acc = vec![0f32; hidden];
            // Pre-packed (interleaved 8-row) expert matrices for the AVX2 GEMV,
            // packed once per expert per session and cached; `None` disables the
            // fast path. Paged Ghost experts deliberately use their wire record.
            for (route_slot, &e) in idx.iter().enumerate() {
                let paged = paged_experts
                    .as_ref()
                    .map(|experts| -> Result<(WireQuant, WireQuant)> {
                        let expert = &experts[route_slot];
                        Ok((
                            WireQuant::from_ghost_tensor(
                                expert,
                                &expert.gate_up,
                                "ghost gate_up_exps",
                            )?,
                            WireQuant::from_ghost_tensor(expert, &expert.down, "ghost down_exps")?,
                        ))
                    })
                    .transpose()?;
                let packed = if paged.is_none() {
                    moe.packed_expert(e, hidden, two_nff)
                } else {
                    None
                };
                // fused gate‖up for expert e: rows e*2nff .. +2nff,
                // in_dim=n_embd.
                let metal_gate_up = paged.as_ref().and_then(|(gate_up, _)| {
                    ghost_metal_q4_matmul(gate_up, two_nff, &[cur_moe_q.q8_0()])
                        .and_then(|mut rows| rows.pop())
                });
                let gate_up = metal_gate_up.unwrap_or_else(|| match (&paged, &packed) {
                    (Some((gate_up, _)), _) => gate_up.matvec_rows_proj(0, two_nff, &cur_moe_q),
                    (None, Some(p)) => packed_band_matvec(&p.gate_up, cur_moe_q.q8_0()),
                    (None, None) => {
                        moe.gate_up_exps
                            .matvec_rows_proj(e * two_nff, two_nff, &cur_moe_q)
                    }
                });
                let hexp: Vec<f32> = (0..moe.n_ff_exp)
                    .map(|o| gelu_tanh(gate_up[o]) * gate_up[o + moe.n_ff_exp])
                    .collect();
                let hexp_q = SharedActivation::new(&hexp);
                // down for expert e: rows e*n_embd .. +n_embd, in_dim=n_ff_exp.
                let metal_y = paged.as_ref().and_then(|(_, down)| {
                    ghost_metal_q4_matmul(down, hidden, &[hexp_q.q8_0()])
                        .and_then(|mut rows| rows.pop())
                });
                let y = metal_y.unwrap_or_else(|| match (&paged, &packed) {
                    (Some((_, down)), _) => down.matvec_rows_proj(0, hidden, &hexp_q),
                    (None, Some(p)) => packed_band_matvec(&p.down, hexp_q.q8_0()),
                    (None, None) => moe.down_exps.matvec_rows_proj(e * hidden, hidden, &hexp_q),
                });
                let scale = route_scales[route_slot];
                for (a, yv) in moe_acc.iter_mut().zip(&y) {
                    *a += yv * scale;
                }
            }
            moe_acc
        };
        let cur_moe = rms_norm(&moe_acc, Some(&moe.post_norm_2), eps);

        // combine the two branches, then the shared post_ffw_norm.
        let mut combined = mlp;
        for (c, m) in combined.iter_mut().zip(&cur_moe) {
            *c += m;
        }
        Ok(rms_norm(&combined, Some(&lw.post_ffw_norm), eps))
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
        self.step_range_with_head(token, pos, h_in, kc, vc, true)
    }

    /// Internal scalar forward with an optional tied output head. Public shard
    /// callers retain the historical `step_range` behavior; prompt prefill can
    /// suppress the otherwise-unused 605 MB Q6_K projection on prefix tokens.
    fn step_range_with_head(
        &self,
        token: u32,
        pos: usize,
        h_in: Option<Vec<f32>>,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
        project_head: bool,
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
            // q/k/v all project the same normed input — quantize it once per
            // activation family (lazily; K-quant projections dot Q8_K).
            let xnq = SharedActivation::new(&xn);
            let mut q = lw.attn_q.matvec_proj(q_dim, &xnq);
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
                    .matvec_proj(kv_dim, &xnq);
                // V-less layers (12B full attention) reuse the raw K projection
                // as V — reference: `if v_proj is not present, use Kcur as Vcur`.
                // V then takes the usual weightless norm and never RoPE.
                let mut v = match lw.attn_v.as_ref() {
                    Some(wv) => wv.matvec_proj(kv_dim, &xnq),
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
            // FFN. MoE (A4B/26B) rows run the two-branch dense+expert block via
            // the shared `moe_layer_ffn` helper (single source of truth, also used
            // by the CUDA-resident lane); dense rows run just the shared-expert MLP:
            // ffn_norm -> parallel GeGLU -> down -> post_ffw_norm.
            let ffn_out = if lw.moe.is_some() {
                self.moe_layer_ffn(li, &h)?
            } else {
                let xn = rms_norm(&h, Some(&lw.ffn_norm), eps);
                let xnq = SharedActivation::new(&xn);
                let gate = lw.ffn_gate.matvec_proj(ffn_dim, &xnq);
                let up = lw.ffn_up.matvec_proj(ffn_dim, &xnq);
                let act: Vec<f32> = gate
                    .iter()
                    .zip(&up)
                    .map(|(g, u)| gelu_tanh(*g) * u)
                    .collect();
                let mlp = lw.ffn_down.matvec(ffn_dim, hidden, &act);
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

        if !is_tail || !project_head {
            return Ok(Gemma4StepOutput::Hidden(h));
        }

        let t_out = std::time::Instant::now();
        // token_embd is vocab-major (row v = the v-th embedding). The helper
        // selects the no-copy Q6_K Metal head for Ghost-MoE on macOS and keeps
        // this exact CPU wire-dot implementation as its fallback.
        let logits = self.project_logits(&h);
        if timing {
            use std::sync::atomic::Ordering::Relaxed;
            // The PLE prep ran inside the embed window; attention/ffn windows
            // bracket the per-layer work; everything after the last layer is
            // the output projection (norm + 262K-vocab GEMV + soft-cap).
            embed_us = embed_us.min(t_start.elapsed().as_micros() as u64);
            CPU_EMBED_US.fetch_add(embed_us, Relaxed);
            CPU_ATTN_US.fetch_add(attn_us, Relaxed);
            CPU_FFN_US.fetch_add(ffn_us, Relaxed);
            CPU_OUTPROJ_US.fetch_add(
                t_out.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            CPU_STEP_N.fetch_add(1, Relaxed);
        }
        Ok(Gemma4StepOutput::Logits(logits))
    }

    /// Prefill a freshly-created KV cache and return the logits at the final
    /// prompt position. Ghost-MoE uses bounded layer-major chunks so routes for
    /// several prompt tokens are known together and each unique expert record is
    /// read at most once per layer/chunk. Other runtimes retain the scalar path.
    fn prefill_tokens(
        &self,
        prompt_tokens: &[u32],
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
        future_forwards: usize,
    ) -> Result<Vec<f32>> {
        if prompt_tokens.is_empty() {
            return Err(BackendError::InvalidModelMetadata(
                "Gemma 4 tokenizer produced an empty prompt".into(),
            ));
        }
        let plan = self.prepare_ghost_prefill(prompt_tokens.len(), future_forwards)?;
        if matches!(
            plan,
            GhostPrefillPlan::CpuChunk | GhostPrefillPlan::HybridChunk
        ) {
            // Bound transient routed-expert/head output memory. Sixteen covers
            // the complete short chat template in the common case; longer
            // prompts retain the same win independently in each chunk.
            let chunk_size = std::env::var("CAMELID_GEMMA4_GHOST_PREFILL_CHUNK")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|&value| value > 0)
                .unwrap_or(16)
                .min(64);
            let mut logits = Vec::new();
            for (chunk_idx, tokens) in prompt_tokens.chunks(chunk_size).enumerate() {
                let start_pos = chunk_idx * chunk_size;
                let mut rows = self.step_chunk_with_head(tokens, start_pos, kc, vc, false)?;
                logits = rows.pop().expect("non-empty prefill chunk has logits");
            }
            if plan == GhostPrefillPlan::HybridChunk {
                let _ = self.finish_ghost_hybrid_prefill(kc, vc, prompt_tokens.len())?;
            }
            Ok(logits)
        } else {
            drive_scalar_prefill(prompt_tokens, |token, pos, project_head| {
                if project_head {
                    self.step(token, pos, kc, vc).map(Some)
                } else {
                    self.step_without_head(token, pos, kc, vc).map(|()| None)
                }
            })
        }
    }

    /// Cancellation-aware form of [`Self::prefill_tokens`] used by serving.
    /// A forward already submitted to CPU/Metal is allowed to finish, then the
    /// next chunk/token boundary observes the stop signal before touching any
    /// more model or Ghost-MoE state.
    fn prefill_tokens_cancellable<C: FnMut() -> bool>(
        &self,
        prompt_tokens: &[u32],
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
        future_forwards: usize,
        should_cancel: &mut C,
    ) -> Result<Option<Vec<f32>>> {
        if prompt_tokens.is_empty() {
            return Err(BackendError::InvalidModelMetadata(
                "Gemma 4 tokenizer produced an empty prompt".into(),
            ));
        }
        if should_cancel() {
            return Ok(None);
        }
        let plan = self.prepare_ghost_prefill(prompt_tokens.len(), future_forwards)?;
        if matches!(
            plan,
            GhostPrefillPlan::CpuChunk | GhostPrefillPlan::HybridChunk
        ) {
            let chunk_size = std::env::var("CAMELID_GEMMA4_GHOST_PREFILL_CHUNK")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|&value| value > 0)
                .unwrap_or(16)
                .min(64);
            let mut logits = Vec::new();
            for (chunk_idx, tokens) in prompt_tokens.chunks(chunk_size).enumerate() {
                if should_cancel() {
                    return Ok(None);
                }
                let start_pos = chunk_idx * chunk_size;
                let mut rows = self.step_chunk_with_head(tokens, start_pos, kc, vc, false)?;
                logits = rows.pop().expect("non-empty prefill chunk has logits");
            }
            if should_cancel() {
                return Ok(None);
            }
            if plan == GhostPrefillPlan::HybridChunk {
                let _ = self.finish_ghost_hybrid_prefill(kc, vc, prompt_tokens.len())?;
                // Import is synchronous but can copy a large context. Observe a
                // disconnect that arrived during it before starting decode; the
                // request cleanup guard resets the just-seeded Metal sequence.
                if should_cancel() {
                    return Ok(None);
                }
            }
            Ok(Some(logits))
        } else {
            let (&last_token, prefix) = prompt_tokens
                .split_last()
                .expect("prefill validates that the prompt is non-empty");
            for (pos, &token) in prefix.iter().enumerate() {
                if should_cancel() {
                    return Ok(None);
                }
                self.step_without_head(token, pos, kc, vc)?;
            }
            if should_cancel() {
                return Ok(None);
            }
            let logits = self.step(last_token, prefix.len(), kc, vc)?;
            if should_cancel() {
                Ok(None)
            } else {
                Ok(Some(logits))
            }
        }
    }

    /// Greedily generate up to `max_new` tokens from `prompt`, with an incremental
    /// KV cache (one forward step per token). Returns (decoded continuation, the
    /// generated token ids).
    #[allow(clippy::explicit_counter_loop)] // `pos` is an absolute sequence index, not a count
    pub fn generate_greedy(&self, prompt: &str, max_new: usize) -> Result<(String, Vec<u32>)> {
        #[cfg(target_os = "macos")]
        let _ghost_common_request = self.lock_ghost_common_generation()?;
        #[cfg(target_os = "macos")]
        let _ghost_metal_stats = GhostMetalGenerationStatsGuard::new(&self.metal_q4_experts);
        #[cfg(target_os = "macos")]
        let _ghost_sequence_cleanup = GhostMetalSequenceCleanup::new(&self.metal_q4_experts);
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        if std::env::var("CAMELID_GEMMA4_DUMP_PROMPT_TOKENS").is_ok() {
            eprintln!("[prompt tokens] {prompt_tokens:?}");
        }
        let eot = gemma4_stop_token_ids(&self.tokenizer);

        let mut logits =
            self.prefill_tokens(&prompt_tokens, &mut kc, &mut vc, max_new.saturating_sub(1))?;
        // Lossless n-gram speculative decode (opt-in, single-node non-MoE rows): verify
        // a batch of drafted tokens in ONE weight pass via `step_chunk`. Output is
        // token-for-token identical to the greedy loop below — every committed token is
        // the target's own argmax — so it makes no support/parity claim, only speed.
        if std::env::var("CAMELID_GEMMA4_SPEC_DECODE").is_ok()
            && self.supports_speculative_chunk_forward()
        {
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
        for generated_index in 0..max_new {
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
            // The final allowed token is already known from `logits`. Feeding it
            // back through the whole model would only compute the prediction for
            // a token the caller did not request. That extra step is especially
            // expensive for Ghost-MoE (30 layers x 8 paged experts), so stop at
            // the exact generation boundary without changing any returned id.
            if generated_index + 1 < max_new {
                logits = self.step(next, pos, &mut kc, &mut vc)?;
                pos += 1;
            }
        }
        if cpu_timing_enabled() {
            report_cpu_timing();
        }
        let text = self.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }

    /// Serve-safe greedy generation.  The caller owns serialization; this
    /// method supplies the other half of that contract by relinquishing model
    /// state at the next prompt/decode forward boundary after cancellation.
    pub fn generate_greedy_cancellable<C: FnMut() -> bool>(
        &self,
        prompt: &str,
        max_new: usize,
        should_cancel: C,
    ) -> Result<Gemma4GenerationOutcome> {
        self.generate_greedy_controlled(prompt, max_new, None::<fn(&str)>, should_cancel)
    }

    /// Streaming counterpart to [`Self::generate_greedy_cancellable`].
    pub fn generate_greedy_streaming_cancellable<F: FnMut(&str), C: FnMut() -> bool>(
        &self,
        prompt: &str,
        max_new: usize,
        on_delta: F,
        should_cancel: C,
    ) -> Result<Gemma4GenerationOutcome> {
        self.generate_greedy_controlled(prompt, max_new, Some(on_delta), should_cancel)
    }

    fn generate_greedy_controlled<F: FnMut(&str), C: FnMut() -> bool>(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_delta: Option<F>,
        mut should_cancel: C,
    ) -> Result<Gemma4GenerationOutcome> {
        #[cfg(target_os = "macos")]
        let _ghost_common_request = self.lock_ghost_common_generation()?;
        #[cfg(target_os = "macos")]
        let _ghost_metal_stats = GhostMetalGenerationStatsGuard::new(&self.metal_q4_experts);
        #[cfg(target_os = "macos")]
        let _ghost_sequence_cleanup = GhostMetalSequenceCleanup::new(&self.metal_q4_experts);
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        if std::env::var("CAMELID_GEMMA4_DUMP_PROMPT_TOKENS").is_ok() {
            eprintln!("[prompt tokens] {prompt_tokens:?}");
        }
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let Some(mut logits) = self.prefill_tokens_cancellable(
            &prompt_tokens,
            &mut kc,
            &mut vc,
            max_new.saturating_sub(1),
            &mut should_cancel,
        )?
        else {
            return Ok(Gemma4GenerationOutcome::Cancelled {
                generated_tokens: 0,
            });
        };
        let mut generated = Vec::new();
        let mut emitted = String::new();
        let mut pos = prompt_tokens.len();
        for generated_index in 0..max_new {
            if should_cancel() {
                return Ok(Gemma4GenerationOutcome::Cancelled {
                    generated_tokens: generated.len(),
                });
            }
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
            if let Some(on_delta) = on_delta.as_mut() {
                let full = self.tokenizer.decode(&generated, true)?;
                if let Some(delta) = full.strip_prefix(&emitted) {
                    if !delta.is_empty() {
                        on_delta(delta);
                    }
                }
                emitted = full;
            }
            // A dropped SSE receiver is discovered by `on_delta`; check again
            // before starting the expensive forward for the next token.
            if generated_index + 1 < max_new {
                if should_cancel() {
                    return Ok(Gemma4GenerationOutcome::Cancelled {
                        generated_tokens: generated.len(),
                    });
                }
                logits = self.step(next, pos, &mut kc, &mut vc)?;
                pos += 1;
            }
        }
        if cpu_timing_enabled() {
            report_cpu_timing();
        }
        let text = if on_delta.is_some() {
            emitted
        } else {
            self.tokenizer.decode(&generated, true)?
        };
        Ok(Gemma4GenerationOutcome::Complete {
            text,
            token_ids: generated,
        })
    }

    /// BASALT Phase 3 forced-decode harness surface (`basalt_eval_protocol.md`
    /// §5.1): teacher-force `forced` through the model. Prefills `prompt` exactly
    /// like [`Self::generate_greedy`], then at each continuation step `i` observes
    /// the FULL next-token logit vector (the distribution predicting continuation
    /// position `i`) via `on_step(i, &logits)` BEFORE feeding `forced[i]` as the
    /// next input — regardless of the model's argmax, ignoring stop tokens (the
    /// forced list defines the step count) and never taking the speculative path.
    /// NO engine math changes: the forward pass is the same [`Self::step`] loop
    /// the greedy decoder drives; only the next-token choice differs. The final
    /// forced token is not fed (its prediction is already observed; feeding it
    /// would only compute an unrecorded extra step). Returns the prompt token ids.
    pub fn forced_decode<F: FnMut(usize, &[f32])>(
        &self,
        prompt: &str,
        forced: &[u32],
        mut on_step: F,
    ) -> Result<Vec<u32>> {
        #[cfg(target_os = "macos")]
        let _ghost_common_request = self.lock_ghost_common_generation()?;
        #[cfg(target_os = "macos")]
        let _ghost_metal_stats = GhostMetalGenerationStatsGuard::new(&self.metal_q4_experts);
        #[cfg(target_os = "macos")]
        let _ghost_sequence_cleanup = GhostMetalSequenceCleanup::new(&self.metal_q4_experts);
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let logits = self.prefill_tokens(
            &prompt_tokens,
            &mut kc,
            &mut vc,
            forced.len().saturating_sub(1),
        )?;
        // Boundary bookkeeping lives in `drive_forced_steps` (unit-tested with a
        // scripted step fn): observe step i's logits BEFORE feeding forced[i];
        // exactly forced.len() observations; the final forced token never fed.
        let mut pos = prompt_tokens.len();
        drive_forced_steps(
            forced,
            logits,
            |tok| -> Result<Vec<f32>> {
                let next = self.step(tok, pos, &mut kc, &mut vc)?;
                pos += 1;
                Ok(next)
            },
            |i, logits: &Vec<f32>| on_step(i, logits),
        )?;
        Ok(prompt_tokens)
    }

    /// [`Self::generate_greedy`] with a per-step FULL-logit observer, for the
    /// BASALT Phase 3 harness (`--dump-step-logits` without `--force-tokens`):
    /// `on_step(i, &logits)` fires for every continuation logit vector BEFORE its
    /// argmax is taken — including the final vector whose argmax is a stop token
    /// (that step is observed, then the loop breaks without emitting the token).
    /// Always drives the plain one-token [`Self::step`] loop (the speculative
    /// path does not surface per-step logits); the token-choice math is identical
    /// to [`Self::generate_greedy`], so the emitted ids match the unobserved
    /// greedy decode of the same prompt.
    pub fn generate_greedy_observed<F: FnMut(usize, &[f32])>(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_step: F,
    ) -> Result<(String, Vec<u32>)> {
        #[cfg(target_os = "macos")]
        let _ghost_common_request = self.lock_ghost_common_generation()?;
        #[cfg(target_os = "macos")]
        let _ghost_metal_stats = GhostMetalGenerationStatsGuard::new(&self.metal_q4_experts);
        #[cfg(target_os = "macos")]
        let _ghost_sequence_cleanup = GhostMetalSequenceCleanup::new(&self.metal_q4_experts);
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let mut logits = self.prefill_tokens(&prompt_tokens, &mut kc, &mut vc, max_new)?;
        let mut generated = Vec::new();
        for i in 0..max_new {
            on_step(i, &logits);
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
            // `next` is generated token #(generated.len()-1), sitting at absolute
            // position prompt_len + that index — identical to generate_greedy's
            // running `pos` counter.
            let pos = prompt_tokens.len() + generated.len() - 1;
            logits = self.step(next, pos, &mut kc, &mut vc)?;
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
        if !self.supports_speculative_chunk_forward() {
            return self.generate_greedy(prompt, max_new);
        }
        #[cfg(target_os = "macos")]
        let _ghost_common_request = self.lock_ghost_common_generation()?;
        #[cfg(target_os = "macos")]
        let _ghost_metal_stats = GhostMetalGenerationStatsGuard::new(&self.metal_q4_experts);
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let logits = self.prefill_tokens(&prompt_tokens, &mut kc, &mut vc, 0)?;
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
        #[cfg(target_os = "macos")]
        let _ghost_common_request = self.lock_ghost_common_generation()?;
        #[cfg(target_os = "macos")]
        let _ghost_metal_stats = GhostMetalGenerationStatsGuard::new(&self.metal_q4_experts);
        #[cfg(target_os = "macos")]
        let _ghost_sequence_cleanup = GhostMetalSequenceCleanup::new(&self.metal_q4_experts);
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        if std::env::var("CAMELID_GEMMA4_DUMP_PROMPT_TOKENS").is_ok() {
            eprintln!("[prompt tokens] {prompt_tokens:?}");
        }
        let eot = gemma4_stop_token_ids(&self.tokenizer);

        let mut logits =
            self.prefill_tokens(&prompt_tokens, &mut kc, &mut vc, max_new.saturating_sub(1))?;
        let mut generated = Vec::new();
        let mut emitted = String::new();
        let mut pos = prompt_tokens.len();
        for generated_index in 0..max_new {
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
            // Do not run a discarded full forward after the last requested
            // token. The sampled token/delta is already complete at this point.
            if generated_index + 1 < max_new {
                logits = self.step(next, pos, &mut kc, &mut vc)?;
                pos += 1;
            }
        }
        if cpu_timing_enabled() {
            report_cpu_timing();
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
        // BASALT Amendment 3 (D-B2 sidecar guard): this lane never ran the sidecar
        // check, so a sidecar-bearing NVFP4 file could compute wrong logits — refuse
        // it here, before any binding (cfg-independent, unit-tested on every host).
        // GABBRO M3-followup lifted the blanket NVFP4 refusal (the Metal resident lane
        // now runs NVFP4 layer projections via nvfp4_block_linear_row_ksplit_f32y_wire);
        // the D17/T5 NaN-sentinel guard moved to nvfp4_metal_sentinel_check below,
        // where the mmap is available to scan the wire bytes the raw upload reads.
        nvfp4_sidecar_check(&gguf.tensors)?;
        let config = LlamaModelConfig::from_gguf(&gguf)?;
        let g = config.gemma4.clone().ok_or_else(|| {
            BackendError::UnsupportedModelArchitecture("not a gemma4 model".into())
        })?;
        let binding = Gemma4Binding::bind(&gguf, &config)?;
        let store = TensorStore::open(path, &gguf);
        // The GPU-resident decode kernels run the layer projections as Q8_0 (34-byte
        // wire blocks), Q4_0 (18-byte QAT wire blocks), or NVFP4 (36-byte 64-value
        // superblocks; GABBRO M3) — all parity-gated GPU GEMVs. The tied head is read
        // separately: Q8_0 runs on the GPU (inside forward_token); Q6_K (the QAT tied
        // head, no GPU kernel) runs on the CPU via the held WireQuant. Layer 0's attn_q
        // is representative of the projection format (the export quantizes every
        // layer's projections alike).
        let layer_fmt = gemma4_metal_layer_fmt(
            store
                .descriptor(&binding.layers[0].attn_q.name)?
                .tensor_type,
        )?;
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
        // GABBRO M3-followup (D17/T5 fail-closed): the resident lane reads NVFP4 layer
        // wire RAW via WirePages (bypassing WireQuant::new's sentinel scan), so the
        // NaN-sentinel guard fires here — one pass over each NVFP4 tensor's UE4M3 scale
        // bytes before any GPU upload; 0x7F/0xFF refuses fail-closed, matching the CPU
        // wire lane. (nvfp4_sidecar_check for D-B2 already ran up top.)
        nvfp4_metal_sentinel_check(&gguf.tensors, &mmap)?;
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
    // MoE-only: the dense shared-expert branch's own post-norm (`post_norm_1`) and
    // the sparse expert-sum post-norm (`post_norm_2`). Resident so the whole MoE
    // dense + compose runs on the GPU (M4). `None` on dense rows.
    moe_post_norm_1: Option<cudarc::driver::CudaSlice<f32>>,
    moe_post_norm_2: Option<cudarc::driver::CudaSlice<f32>>,
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

/// Quant lane of a resident gemma4 layer projection. All consume Q8_0
/// activations; Q8_0 weights are SoA-repacked, Q4_0/Q4_1/NVFP4 are raw wire.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GemmaLayerQuant {
    Q8_0,
    Q4_0,
    Q4_1,
    Nvfp4,
}

#[cfg(feature = "cuda")]
impl GemmaLayerQuant {
    fn from_wire(f: WireFormat) -> Self {
        match f {
            WireFormat::Q8_0 => Self::Q8_0,
            WireFormat::Q4_0 => Self::Q4_0,
            WireFormat::Q4_1 => Self::Q4_1,
            // BASALT Phase 4: NVFP4 layer projections now reside on the CUDA lane
            // (nvfp4_gemv, raw 36-byte wire). `nvfp4_cuda_lane_check` still refuses
            // every other uncovered format before this catch-all can panic.
            WireFormat::Nvfp4 => Self::Nvfp4,
            other => {
                panic!("gemma4 layer projection quant {other:?} unsupported (Q8_0/Q4_0/Q4_1/NVFP4)")
            }
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
        // BASALT Phase 4: NVFP4 raw-wire GEMV. `launch_nvfp4_gemv` returns a typed
        // Nvfp4LaunchError; the odd-block variant is the I-k-div lane guard and is
        // structurally unreachable here (the parse boundary refuses non-%64 NVFP4
        // first-dims at load — k_div_fixture_trips_parse_refusal — so every gemma4
        // projection reaching the CUDA GEMV has an even Q8_0-block count), matching
        // the codebase's guard-then-unreachable idiom (matvec's Q5_K arm).
        GemmaLayerQuant::Nvfp4 => match crate::cuda_resident::launch_nvfp4_gemv(
            s,
            &kernels.nvfp4_gemv,
            in_scales,
            in_quants,
            weight,
            rows,
            blocks_per_row,
            out,
            0,
        ) {
            Ok(()) => Ok(()),
            Err(crate::cuda_resident::Nvfp4LaunchError::Driver(e)) => Err(e),
            Err(crate::cuda_resident::Nvfp4LaunchError::OddBlocksPerRow(bpr)) => unreachable!(
                "gemma4 NVFP4 projection reached the CUDA GEMV with an odd Q8_0-block count \
                 {bpr} (in_dim % 64 != 0); the parse boundary refuses non-%64 NVFP4 tensors \
                 before load"
            ),
        },
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
/// compact SoA layout `q8_gemv` reads: all 32-i8 quant groups first, then the
/// original f16 scale bits. Mirrors `cuda_resident::repack_q8_soa` but consumes
/// the raw GGUF wire directly (that helper expects an already-f32-scale 36B block).
#[cfg(feature = "cuda")]
fn q8_wire_to_soa(wire: &[u8]) -> Vec<u8> {
    const W: usize = 34;
    let n = wire.len() / W;
    let mut out = vec![0u8; n * 32 + n * 2];
    let (quants, scales) = out.split_at_mut(n * 32);
    for b in 0..n {
        let blk = &wire[b * W..b * W + W];
        quants[b * 32..b * 32 + 32].copy_from_slice(&blk[2..34]);
        scales[b * 2..b * 2 + 2].copy_from_slice(&blk[0..2]);
    }
    out
}

/// Quant lane of the GPU tied head: Q8_0 (`q8_gemv`, Q8_0 input), Q4_K (`q4k_gemv`,
/// Q8_K input) or Q6_K (`q6k_gemv`, Q8_K input). Each lane's GEMV reads a specific
/// GPU-side byte layout — see [`gemma4_head_upload`], which is the ONLY way the head
/// weight may reach VRAM.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HeadLane {
    Q8_0,
    Q4K,
    Q6K,
}

/// Convert the tied head's GGUF wire bytes into the GPU layout ITS GEMV reads.
///
/// None of these kernels reads the stock wire: `q8_gemv` wants the SoA split (quants
/// then f16 scales), `q4k_gemv` wants the quant-byte swizzle that makes each aux
/// lane's four stride-8 bytes one aligned i32, and `q6k_gemv` indexes super-blocks at
/// a 224-byte PADDED stride, not the 210-byte wire stride. This mirrors
/// `cuda_resident::repack_for_lane`, which is what every OTHER resident lane in the
/// tree already routes through.
///
/// Root cause of the gemma4 Q4_0 mis-decode: the Q4_K and Q6_K arms used to
/// `clone_htod` the raw wire while only the Q8_0 arm repacked. Since a Q4_0-quantized
/// gemma4 export carries a **Q4_K** `token_embd` (the E2B Q4_0 row does), that row ran
/// `q4k_gemv` over unswizzled bytes: every logit was formed from correctly-addressed
/// but wrongly-PAIRED nibbles, so the lane emitted fluent-looking nonsense instead of
/// refusing. Measured before the fix, "Name the capital of France in one word." →
/// "passe dép oficialmenteynam shalthapp lenghtynam" on CUDA vs "Paris" on the CPU
/// runtime. The Q6_K arm had the same defect one step worse — a 210-vs-224 stride
/// mismatch that also reads past the end of the allocation.
///
/// Head lane is chosen from `token_embd`'s format, which no admission check inspects,
/// so the only gemma4 row ever validated on this lane (E4B Q8_0) was the one whose
/// head happened to be Q8_0. Routing all three lanes through one function is what
/// keeps that class of bug from coming back.
#[cfg(feature = "cuda")]
fn gemma4_head_upload(lane: HeadLane, wire: &[u8]) -> Vec<u8> {
    match lane {
        HeadLane::Q8_0 => q8_wire_to_soa(wire),
        HeadLane::Q4K => crate::cuda_resident::swz_q4k_blocks(wire),
        HeadLane::Q6K => crate::cuda_resident::pad_q6k_blocks(wire),
    }
}

/// Resident GPU tied head. `weight` is the vocab-major projection in its lane's GPU
/// layout (always via [`gemma4_head_upload`]); input is quantized by the fused
/// rms_norm+quantize into `inq`/`ins`; `logits` is dtoh'd once per token. `blocks` is
/// blocks-per-row passed to the GEMV (`hidden/32` for Q8_0, `hidden/256` for K-quants).
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

/// One cached MoE expert's two Q4_0 weight slices, resident on the GPU. `gate_up`
/// is the fused gate‖up rows (`2*n_ff_exp × hidden`) and `down` is the down rows
/// (`hidden × n_ff_exp`) — the exact byte ranges `moe_layer_ffn`'s CPU path reads
/// from the mmap for this expert. `last_used` is the LRU recency stamp.
#[cfg(feature = "cuda")]
struct SserExpertDev {
    gate_up: cudarc::driver::CudaSlice<u8>,
    down: cudarc::driver::CudaSlice<u8>,
    last_used: u64,
}

/// SSER (self-specializing expert residency) VRAM cache: a per-(layer,expert) LRU
/// of Q4_0 expert weight slices. A single user's session fires a skewed, stable
/// subset of the experts; keeping the hot ones resident lets their two GEMVs run on
/// the GPU (336 GB/s) instead of the CPU (the ~187 ms/token MoE wall). Gated behind
/// `CAMELID_SSER_CACHE` (off = M1 all-CPU MoE); capacity `CAMELID_SSER_CACHE_EXPERTS`
/// (#experts). Eviction is LRU on miss-when-full. Bit-exact: the GPU `q4_0_gemv` is
/// proven bit-identical to the CPU `q4_0_wire_row_dot` the cache-miss path uses.
// Throwaway M3 profiling counters (env CAMELID_SSER_PROFILE): total ns spent in
// the dense-MLP branch, router, and expert loop across all MoE layers/tokens.
#[cfg(feature = "cuda")]
static SSER_PROF_DENSE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "cuda")]
static SSER_PROF_ROUTER_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "cuda")]
static SSER_PROF_EXPERT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "cuda")]
static SSER_PROF_HIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "cuda")]
static SSER_PROF_MISS_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "cuda")]
struct SserCache {
    entries: std::collections::HashMap<(u16, u16), SserExpertDev>,
    capacity: usize,
    clock: u64,
    // Diagnostics (per-generate; reset by the harness before each run).
    hits: u64,
    misses: u64,
}

#[cfg(feature = "cuda")]
impl SserCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            capacity: capacity.max(1),
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Mark `key` most-recently-used and return true if resident.
    fn touch(&mut self, key: (u16, u16)) -> bool {
        self.clock += 1;
        let clock = self.clock;
        if let Some(e) = self.entries.get_mut(&key) {
            e.last_used = clock;
            true
        } else {
            false
        }
    }

    /// Evict the least-recently-used entry if at capacity (called before an insert).
    fn evict_if_full(&mut self) {
        if self.entries.len() < self.capacity {
            return;
        }
        if let Some((&victim, _)) = self.entries.iter().min_by_key(|(_, e)| e.last_used) {
            self.entries.remove(&victim);
        }
    }
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
    /// Token sequence currently represented in the persistent KV cache (the last request's
    /// prompt + its generated tokens). On the next request the longest matching prefix is
    /// reused, so only the genuinely new tokens are prefilled — this keeps multi-turn TTFT
    /// roughly constant instead of growing with conversation length.
    cached_tokens: Vec<u32>,
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
    // M4: holds the MoE dense shared-expert branch (branch A) result on-device
    // (`rms_norm(down_out, post_norm_1)`) while the sparse expert branch runs, so the
    // two branches can be composed on the GPU without a host round-trip.
    d_mlp: cudarc::driver::CudaSlice<f32>,
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
    // SSER (M2): per-(layer,expert) VRAM LRU of Q4_0 expert slices. `None` when
    // `CAMELID_SSER_CACHE` is unset (M1 all-CPU MoE). Wrapped in a `RefCell` so the
    // cached FFN can mutate the LRU/counters through a shared `&self` — the per-token
    // forward loop holds long-lived immutable borrows of `self.kernels`/`self.cpu`
    // that a `&mut self` MoE call would conflict with. Device scratch for the expert
    // GEMVs is allocated locally per call (batch-1 tiny GEMVs are launch-bound, so the
    // alloc is negligible and it keeps the hot path `&self`).
    sser: Option<std::cell::RefCell<SserCache>>,
}

#[cfg(feature = "cuda")]
impl Gemma4CudaResident {
    /// Load the model (CPU runtime, weights mmap'd), bring up the CUDA kernels,
    /// upload per-layer norms, and allocate the KV caches + scratch. `max_positions`
    /// bounds the resident KV cache.
    pub fn load(path: &Path, max_positions: usize) -> Result<Self> {
        let cpu = Gemma4Runtime::load(path)?;
        // BASALT Amendment 3 review fix: refuse NVFP4 layer projections with a
        // typed error BEFORE the `GemmaLayerQuant::from_wire` catch-all (`upw`
        // below) can panic. The CPU wire lane serves NVFP4 in this release;
        // CUDA-resident NVFP4 is Phase 4 (BASALT).
        nvfp4_cuda_lane_check(cpu.layers.iter().flat_map(|lw| {
            [
                Some(lw.attn_q.format),
                lw.attn_k.as_ref().map(|w| w.format),
                lw.attn_v.as_ref().map(|w| w.format),
                Some(lw.attn_output.format),
                Some(lw.ffn_gate.format),
                Some(lw.ffn_up.format),
                Some(lw.ffn_down.format),
                lw.moe.as_ref().map(|m| m.gate_up_exps.format),
                lw.moe.as_ref().map(|m| m.down_exps.format),
            ]
            .into_iter()
            .flatten()
        }))?;
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
                        .clone_htod(&gemma4_head_upload(HeadLane::Q8_0, cpu.token_embd.bytes()))
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
                    weight: s
                        .clone_htod(&gemma4_head_upload(HeadLane::Q6K, cpu.token_embd.bytes()))
                        .map_err(cu)?,
                    output_norm: s.clone_htod(&cpu.output_norm).map_err(cu)?,
                    logits: s.alloc_zeros::<f32>(vocab).map_err(cu)?,
                    inq: s.alloc_zeros::<i8>(blocks * 256).map_err(cu)?,
                    ins: s.alloc_zeros::<f32>(blocks).map_err(cu)?,
                    blocks,
                    softcap,
                })
            }
            // Q4_K tied head (the format a Q4_0-quantized gemma4 export carries):
            // q4k_gemv over the SWIZZLED 144-byte super-blocks, Q8_K input.
            WireFormat::Q4K if hidden.is_multiple_of(256) => {
                let blocks = hidden / 256;
                Some(Gemma4HeadDev {
                    lane: HeadLane::Q4K,
                    weight: s
                        .clone_htod(&gemma4_head_upload(HeadLane::Q4K, cpu.token_embd.bytes()))
                        .map_err(cu)?,
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
                moe_post_norm_1: match lw.moe.as_ref() {
                    Some(m) => Some(s.clone_htod(&m.post_norm_1).map_err(cu)?),
                    None => None,
                },
                moe_post_norm_2: match lw.moe.as_ref() {
                    Some(m) => Some(s.clone_htod(&m.post_norm_2).map_err(cu)?),
                    None => None,
                },
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
                // Q4_0/Q4_1/NVFP4 residency is raw wire passthrough: the GEMV reads
                // the packed nibbles + in-block scales directly, so the 4.x-bpw
                // footprint is preserved in VRAM (no host-side dequant/expansion).
                GemmaLayerQuant::Q4_0 | GemmaLayerQuant::Q4_1 | GemmaLayerQuant::Nvfp4 => {
                    wq.bytes().to_vec()
                }
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
        // SSER (M2): enable the per-(layer,expert) VRAM cache only when the model has
        // MoE layers AND the flag is set. Capacity defaults to ~1000 experts (the
        // measured hot set); each cached expert is ~2*n_ff_exp*(hidden/32)*18 +
        // hidden*(n_ff_exp/32)*18 bytes of Q4_0 wire (~3.3 MB on the 26B), so ~1000
        // experts ≈ ~3.3 GB — under the ~3.6 GB free after the resident set. Tunable
        // via CAMELID_SSER_CACHE_EXPERTS.
        let first_moe = cpu.layers.iter().find_map(|lw| lw.moe.as_ref());
        let sser = if let (Some(moe), true) =
            (first_moe, std::env::var_os("CAMELID_SSER_CACHE").is_some())
        {
            // Per-expert VRAM cost: the two Q4_0 slices this expert's GEMVs read.
            // gate_up = 2*n_ff_exp rows of hidden values; down = hidden rows of
            // n_ff_exp values; Q4_0 packs 32 values per 18-byte block.
            const WB: usize = crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK;
            let two_nff = 2 * moe.n_ff_exp;
            let per_expert_bytes = two_nff * (hidden / 32) * WB + hidden * (moe.n_ff_exp / 32) * WB;
            // Budget: keep the cache under ~80% of the free VRAM after the resident set
            // (leaving headroom for the per-token scratch + the KV cache growth).
            let (free, _total) = cudarc::driver::result::mem_get_info().unwrap_or((0, 0));
            // Cache budget = free VRAM at load MINUS a fixed reserve for the transient
            // per-miss weight uploads (a few pooled ~6 MiB `clone_htod` buffers) and
            // driver slack. The KV caches and per-token scratch are already allocated
            // ABOVE (so `free` excludes them) — the only dynamic post-cache consumer is
            // those small transient buffers, whose need is ~constant, not proportional
            // to free VRAM. A fixed reserve therefore lets the cache claim far more of
            // the card than the old flat 0.80 factor did: on the 6 GB box this lifts
            // the cap ~690 -> ~820 experts, cutting the miss count and measuring
            // +~50% steady decode (miss-bound, capacity-limited). Reserve tunable via
            // CAMELID_SSER_CACHE_RESERVE_MIB; a hard 0.98 cap on the free fraction is a
            // final belt-and-suspenders against a pathologically small `free`.
            let reserve_mib = std::env::var("CAMELID_SSER_CACHE_RESERVE_MIB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(160);
            let reserve = reserve_mib * 1024 * 1024;
            let hard_cap = (free as f64 * 0.98) as usize;
            let budget = free.saturating_sub(reserve).min(hard_cap);
            let fit_cap = budget.checked_div(per_expert_bytes).unwrap_or(0);
            let req_cap = std::env::var("CAMELID_SSER_CACHE_EXPERTS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1000);
            // Honor the smaller of the requested capacity and what free VRAM allows.
            let cap = req_cap.min(fit_cap).max(1);
            eprintln!(
                "[sser] expert-residency cache ON: capacity {cap} experts ({} MiB each; requested {req_cap}, VRAM-fit {fit_cap}; {} MiB free)",
                per_expert_bytes / (1024 * 1024),
                free / (1024 * 1024),
            );
            Some(SserCache::new(cap))
        } else {
            None
        };
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
            cached_tokens: Vec::new(),
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
            d_mlp: alloc_f(hidden).map_err(cu)?,
            d_cos_all: alloc_f(block_count * (head_dim_max / 2)).map_err(cu)?,
            d_sin_all: alloc_f(block_count * (head_dim_max / 2)).map_err(cu)?,
            d_position: s.alloc_zeros::<i32>(1).map_err(cu)?,
            d_pli: alloc_f(block_count * ple_dim).map_err(cu)?,
            d_ple_gated: alloc_f(ple_dim).map_err(cu)?,
            d_ple_gated2: alloc_f(ple_dim).map_err(cu)?,
            d_ple_proj: alloc_f(hidden).map_err(cu)?,
            d_ple_normed: alloc_f(hidden).map_err(cu)?,
            sser: sser.map(std::cell::RefCell::new),
            plan,
            kernels,
            cap_stream,
            gpu_head,
            gpu_ple_ctx,
            cpu,
        };
        // Every device slice above was allocated + zeroed (`alloc_zeros`) on the DEFAULT
        // stream (`kernels.stream`), but the per-token forward runs on `cap_stream`. With
        // event-tracking disabled during load there is no automatic cross-stream ordering,
        // so the first forward's uploads on `cap_stream` (e.g. the RoPE cos/sin table) can
        // race the still-in-flight load-time memsets on the default stream — which then
        // clobber the just-uploaded values with zeros (observed: cos=0 at position 0 →
        // K zeroed → wrong tokens). Drain the default stream here so all load-time zeroing
        // is complete before any cap_stream work begins.
        me.kernels.stream.synchronize().map_err(cu)?;
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

    /// SSER cache diagnostics: `(hits, misses, resident_experts, capacity)`.
    /// `None` when the cache is disabled (`CAMELID_SSER_CACHE` unset).
    pub fn sser_stats(&self) -> Option<(u64, u64, usize, usize)> {
        if std::env::var_os("CAMELID_SSER_PROFILE").is_some() {
            use std::sync::atomic::Ordering::Relaxed;
            let d = SSER_PROF_DENSE_NS.load(Relaxed) as f64 / 1e6;
            let r = SSER_PROF_ROUTER_NS.load(Relaxed) as f64 / 1e6;
            let e = SSER_PROF_EXPERT_NS.load(Relaxed) as f64 / 1e6;
            let hit = SSER_PROF_HIT_NS.load(Relaxed) as f64 / 1e6;
            let miss = SSER_PROF_MISS_NS.load(Relaxed) as f64 / 1e6;
            eprintln!(
                "[sser-profile] MoE CPU-side totals: dense-MLP {d:.0} ms, router {r:.0} ms, expert-loop {e:.0} ms (sum {:.0} ms)",
                d + r + e
            );
            eprintln!(
                "[sser-profile]   expert-loop split: hit-path {hit:.0} ms, miss-path {miss:.0} ms, rest(prep+dtoh+sync) {:.0} ms",
                e - hit - miss
            );
        }
        self.sser.as_ref().map(|c| {
            let c = c.borrow();
            (c.hits, c.misses, c.entries.len(), c.capacity)
        })
    }

    /// Reset the SSER hit/miss counters (keeps resident weights). Lets the harness
    /// separate warm-up misses from steady-state hit-rate. No-op when disabled.
    pub fn sser_reset_counters(&self) {
        if let Some(c) = self.sser.as_ref() {
            let mut c = c.borrow_mut();
            c.hits = 0;
            c.misses = 0;
        }
    }

    /// SSER (M2/M3/M4) sparse-expert branch of the MoE FFN. Runs the router on the
    /// CPU (tiny), then every selected expert's two GEMVs on the GPU — cached in VRAM
    /// (hit) or uploaded+promoted (miss) — accumulating each expert's weighted
    /// down-GEMV into an on-device buffer (`scaled_axpy`). Returns that device
    /// accumulator (the sparse expert sum, BEFORE `post_norm_2`); the caller composes
    /// it on-device with the GPU dense branch (M4). `attn_out` is the post-attention
    /// residual (already copied device->host by the caller for the router).
    ///
    /// The dense "shared expert" branch and the final compose+norms now run on the
    /// GPU in the layer loop (M4); this method owns ONLY the router + 8 expert GEMVs.
    ///
    /// Parity: the GPU `q4_0_gemv` is bit-identical to the CPU `q4_0_wire_row_dot`
    /// the miss path uses (proven in `q4_0_gemv_matches_oracle`), and the GPU GeGLU
    /// (`geglu_mul`) matches `gelu_tanh` within the accepted f16-KV/tanhf floor — so
    /// cached and uncached experts produce the same content tokens as M1.
    #[allow(clippy::too_many_lines)]
    fn moe_layer_ffn_cached(
        &self,
        li: usize,
        attn_out: &[f32],
    ) -> Result<cudarc::driver::CudaSlice<f32>> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let s = self.cap_stream.clone();
        let k = &self.kernels;
        let hidden = self.hidden;
        let eps = self.eps;
        let cpu = &self.cpu;
        let lw = &cpu.layers[li];
        let l = cpu.first_layer + li;
        let moe = lw
            .moe
            .as_ref()
            .expect("moe_layer_ffn_cached called on a non-MoE layer");
        let sser = self
            .sser
            .as_ref()
            .expect("moe_layer_ffn_cached requires the SSER cache");

        let prof = std::env::var_os("CAMELID_SSER_PROFILE").is_some();
        let tp1 = std::time::Instant::now();

        // --- Router (CPU, identical). ---
        let mut r = rms_norm(attn_out, None, eps);
        let inv = 1.0f32 / (hidden as f32).sqrt();
        for (rv, sv) in r.iter_mut().zip(&moe.gate_inp_scale) {
            *rv = *rv * inv * sv;
        }
        let logits = f32_matvec(&moe.gate_inp, hidden, moe.n_expert, &r);
        let maxl = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        let mut idx: Vec<usize> = (0..moe.n_expert).collect();
        idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(moe.n_expert_used);
        if std::env::var_os("CAMELID_GEMMA4_ROUTE_TRACE").is_some() {
            eprintln!("[route] l={l} e={idx:?}");
        }
        let mut wsum: f32 = idx.iter().map(|&e| probs[e]).sum();
        wsum = wsum.max(6.103_515e-5);
        if prof {
            SSER_PROF_ROUTER_NS.fetch_add(
                tp1.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        let tp2 = std::time::Instant::now();

        // --- Expert branch: quantize the shared input once (CPU), upload once. ---
        let cur_moe = rms_norm(attn_out, Some(&moe.pre_norm_2), eps);
        let cur_moe_q = quantize_q8_0_blocks(&cur_moe);
        let two_nff = 2 * moe.n_ff_exp;
        let nff = moe.n_ff_exp;
        let gu_blocks = hidden / 32; // gate_up in_dim = hidden
        let down_blocks = nff / 32; // down in_dim = n_ff_exp
        let gu_row_bytes = gu_blocks * crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK;
        let down_row_bytes = down_blocks * crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK;

        // Upload the shared Q8_0 expert input (scales + concatenated i8 quants) once —
        // every selected expert dots against the same activation. Device scratch is
        // allocated locally (keeps the hot path `&self`; batch-1 GEMVs are launch-bound
        // so the alloc cost is negligible next to the per-expert launch overhead).
        let in_scales: Vec<f32> = cur_moe_q.iter().map(|b| b.scale).collect();
        let mut in_quants = vec![0i8; gu_blocks * 32];
        for (b, blk) in cur_moe_q.iter().enumerate() {
            in_quants[b * 32..(b + 1) * 32].copy_from_slice(&blk.quants);
        }
        let d_in_s = s.clone_htod(&in_scales).map_err(cu)?;
        let d_in_q = s.clone_htod(&in_quants).map_err(cu)?;
        let mut d_gate_up = s.alloc_zeros::<f32>(two_nff).map_err(cu)?;
        let mut d_geglu = s.alloc_zeros::<f32>(nff).map_err(cu)?;
        let mut d_geglu_q = s.alloc_zeros::<i8>(nff).map_err(cu)?;
        let mut d_geglu_s = s.alloc_zeros::<f32>(down_blocks).map_err(cu)?;
        let mut d_y = s.alloc_zeros::<f32>(hidden).map_err(cu)?;
        // M3/M4 on-device MoE accumulator: every selected expert (hit OR uploaded-miss)
        // folds its weighted down-GEMV output straight into this device buffer (one
        // scaled_axpy launch each). In M4 the buffer is RETURNED to the caller and
        // composed with the dense branch on-device — no per-layer dtoh at all.
        let mut d_moe_acc = s.alloc_zeros::<f32>(hidden).map_err(cu)?;

        let gate_up_bytes = moe.gate_up_exps.bytes();
        let down_bytes = moe.down_exps.bytes();

        for &e in &idx {
            let w = probs[e] / wsum;
            let scale = moe.down_exps_scale[e] * w;
            let key = (l as u16, e as u16);

            let cached = sser.borrow_mut().touch(key);
            let te = std::time::Instant::now();
            // On a MISS, upload the expert's two Q4_0 slices and insert them into the
            // VRAM cache (promotion) BEFORE running — then the GPU pipeline below reads
            // the freshly-resident slices exactly as it does for a hit. This moves the
            // expensive part of a miss (the ~1.8 ms CPU expert matvec, which profiling
            // showed was ~72% of all MoE time) onto the GPU: a miss now costs only the
            // ~6 MiB weight htod + the same tiny GEMV launches a hit already pays.
            if !cached {
                sser.borrow_mut().misses += 1;
                let gu_off = e * two_nff * gu_row_bytes;
                let down_off = e * hidden * down_row_bytes;
                let gu_slice = &gate_up_bytes[gu_off..gu_off + two_nff * gu_row_bytes];
                let down_slice = &down_bytes[down_off..down_off + hidden * down_row_bytes];
                let gu_dev = s.clone_htod(gu_slice).map_err(cu)?;
                let down_dev = s.clone_htod(down_slice).map_err(cu)?;
                let mut c = sser.borrow_mut();
                c.evict_if_full();
                c.clock += 1;
                let stamp = c.clock;
                c.entries.insert(
                    key,
                    SserExpertDev {
                        gate_up: gu_dev,
                        down: down_dev,
                        last_used: stamp,
                    },
                );
            } else {
                sser.borrow_mut().hits += 1;
            }

            // --- GPU pipeline (hit OR promoted-miss): fused gate‖up GEMV -> GeGLU ->
            // quantize -> down GEMV -> weighted on-device accumulate. Every expert now
            // takes the identical bit-exact GPU path, so the token stream no longer
            // depends on cache warmth (removes host/device path-divergence). Hold the
            // shared cache borrow for the whole launch sequence so the resident weight
            // views stay valid; `touch`/`insert` above made `key` the newest entry, so
            // it cannot be evicted while this borrow is live. ---
            {
                let c = sser.borrow();
                let ent = c.entries.get(&key).expect("hit or just-promoted miss");
                let gu_dev = ent.gate_up.slice(0..ent.gate_up.len());
                let down_dev = ent.down.slice(0..ent.down.len());
                // gate‖up: two_nff rows, gu_blocks blocks/row.
                crate::cuda_resident::launch_q4_0_gemv(
                    &s,
                    &k.q4_0_gemv,
                    &d_in_s,
                    &d_in_q,
                    &gu_dev,
                    two_nff,
                    gu_blocks,
                    &mut d_gate_up,
                    0,
                )
                .map_err(cu)?;
                // GeGLU: gelu_tanh(gate[o]) * up[o] where gate = out[0..nff], up = out[nff..2nff].
                {
                    let gate_v = d_gate_up.slice(0..nff);
                    let up_v = d_gate_up.slice(nff..two_nff);
                    let cfg = LaunchConfig {
                        grid_dim: ((nff as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let n_i = nff as i32;
                    let mut b = s.launch_builder(&k.geglu_mul);
                    b.arg(&gate_v).arg(&up_v).arg(&mut d_geglu).arg(&n_i);
                    unsafe { b.launch(cfg) }.map_err(cu)?;
                }
                crate::cuda_resident::launch_quantize(
                    &s,
                    &k.quantize,
                    &d_geglu,
                    &mut d_geglu_q,
                    &mut d_geglu_s,
                    down_blocks,
                )
                .map_err(cu)?;
                // down: hidden rows, down_blocks blocks/row.
                crate::cuda_resident::launch_q4_0_gemv(
                    &s,
                    &k.q4_0_gemv,
                    &d_geglu_s,
                    &d_geglu_q,
                    &down_dev,
                    hidden,
                    down_blocks,
                    &mut d_y,
                    0,
                )
                .map_err(cu)?;
            }
            // On-device weighted accumulate: d_moe_acc[i] += d_y[i] * scale (deferred to
            // one dtoh after the loop). scaled_axpy(acc, y, scale, n): acc += y*scale.
            {
                let cfg = LaunchConfig {
                    grid_dim: ((hidden as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n_i = hidden as i32;
                let mut b = s.launch_builder(&k.scaled_axpy);
                b.arg(&mut d_moe_acc).arg(&d_y).arg(&scale).arg(&n_i);
                unsafe { b.launch(cfg) }.map_err(cu)?;
            }
            if prof {
                let ns = te.elapsed().as_nanos() as u64;
                use std::sync::atomic::Ordering::Relaxed;
                if cached {
                    SSER_PROF_HIT_NS.fetch_add(ns, Relaxed);
                } else {
                    SSER_PROF_MISS_NS.fetch_add(ns, Relaxed);
                }
            }
        }
        // Every selected expert (hit OR uploaded-miss) accumulated into `d_moe_acc` in
        // strict idx order, so the layer's expert sum is one left-to-right f32
        // accumulation identical to M1's single-buffer host loop. In M4 we return the
        // device buffer directly (no dtoh) — the caller applies post_norm_2, composes
        // with the dense branch, applies post_ffw_norm, and adds to the residual, all
        // on the GPU.
        if prof {
            SSER_PROF_EXPERT_NS.fetch_add(
                tp2.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        Ok(d_moe_acc)
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
        //
        // MoE (A4B/26B) rows compute their FFN on the CPU via a per-layer device<->host
        // round-trip (synchronize + memcpy). That CANNOT live inside a captured/replayed
        // graph, so disable capture entirely for any model with a MoE layer and always
        // run the explicit per-launch path. (Dense/E-series models keep the graph.)
        let has_moe = self.cpu.layers.iter().any(|lw| lw.moe.is_some());
        let do_capture = !has_moe && self.decode_graph.is_none() && self.warmed;
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

                // FFN. MoE (A4B/26B) rows have TWO branches off the post-attention
                // residual `attn_out` (in `d_hidden`): (A) a dense "shared expert" MLP
                // and (B) the sparse 8-expert branch. With the SSER cache ON (M4) BOTH
                // branches run on the GPU and are composed on-device — the CPU only
                // runs the tiny router. With the cache OFF (M1) the whole two-branch
                // block runs on the CPU via the shared bit-exact `moe_layer_ffn`
                // helper. Either way `d_hidden` must be settled (attention done) before
                // reading it, so synchronize + dtoh first.
                if lw.moe.is_some() && self.sser.is_some() {
                    // --- M4: both MoE branches on the GPU. ---
                    // The router still runs on the CPU, so copy the post-attention
                    // residual to the host once (branch A + the experts read `d_hidden`
                    // on-device; only the top-8 pick needs the host copy).
                    s.synchronize().map_err(cu)?;
                    let mut attn_out_host = vec![0f32; hidden];
                    s.memcpy_dtoh(&self.d_hidden, &mut attn_out_host)
                        .map_err(cu)?;

                    // Branch A — dense shared-expert MLP, GPU (reuses the dense-row
                    // FFN kernels): rms_norm(attn_out, ffn_norm) -> quantize -> gate/up
                    // GEMV -> GeGLU -> quantize -> down GEMV -> rms_norm(_, post_norm_1)
                    // -> d_mlp. Differs from the dense-row path ONLY by post_norm_1 (vs
                    // post_ffw_norm) and by parking the result in d_mlp instead of
                    // folding straight into the residual.
                    let prof = std::env::var_os("CAMELID_SSER_PROFILE").is_some();
                    let ta = std::time::Instant::now();
                    let post_norm_1 = nrm
                        .moe_post_norm_1
                        .as_ref()
                        .expect("MoE layer binds post_norm_1");
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
                        post_norm_1,
                        &mut self.d_mlp,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    if prof {
                        SSER_PROF_DENSE_NS.fetch_add(
                            ta.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }

                    // Branch B — sparse expert sum on-device (returns d_moe_acc, the
                    // weighted expert accumulation BEFORE post_norm_2). Takes `&self`
                    // (LRU behind a RefCell), so it coexists with the loop-level
                    // `&self.kernels` borrow.
                    let d_moe_acc = self.moe_layer_ffn_cached(li, &attn_out_host)?;

                    // Compose on-device: rms_norm(moe_acc, post_norm_2) -> + d_mlp ->
                    // rms_norm(_, post_ffw_norm) -> add to the residual. Bit-identical
                    // op order to the CPU `moe_layer_ffn` tail.
                    let post_norm_2 = nrm
                        .moe_post_norm_2
                        .as_ref()
                        .expect("MoE layer binds post_norm_2");
                    crate::cuda_resident::launch_rmsnorm(
                        &s,
                        &k.rms_norm,
                        &d_moe_acc,
                        post_norm_2,
                        &mut self.d_ffn_out,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    // d_ffn_out (cur_moe) += d_mlp (dense branch).
                    crate::cuda_resident::launch_residual(
                        &s,
                        &k.residual_add,
                        &mut self.d_ffn_out,
                        &self.d_mlp,
                        hidden,
                    )
                    .map_err(cu)?;
                    // rms_norm(combined, post_ffw_norm) -> d_normed -> + residual.
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
                } else if lw.moe.is_some() {
                    // --- M1: whole two-branch MoE FFN on the CPU (cache OFF). ---
                    s.synchronize().map_err(cu)?;
                    let mut attn_out_host = vec![0f32; hidden];
                    s.memcpy_dtoh(&self.d_hidden, &mut attn_out_host)
                        .map_err(cu)?;
                    let ffn_out = self.cpu.moe_layer_ffn(li, &attn_out_host)?;
                    s.memcpy_htod(&ffn_out, &mut self.d_ffn_out).map_err(cu)?;
                    crate::cuda_resident::launch_residual(
                        &s,
                        &k.residual_add,
                        &mut self.d_hidden,
                        &self.d_ffn_out,
                        hidden,
                    )
                    .map_err(cu)?;
                } else {
                    // Dense row: norm + quantize -> gate/up -> GeGLU -> quantize ->
                    // down -> post-ffw norm -> residual, all on the GPU.
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
                }

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
    /// Prefill `prompt_tokens`, reusing the longest prefix already present in the KV cache
    /// from the previous request (cross-request prefix cache) and only running
    /// `forward_token` for the new suffix. Returns the logits predicting the first new
    /// token. Output-equivalent to a full re-prefill: the KV for shared-prefix positions is
    /// identical (same tokens, same positions), so only redundant compute is skipped. The
    /// caller extends `cached_tokens` with any tokens it then generates. Disable with
    /// `CAMELID_GEMMA4_NO_PREFIX_CACHE=1`.
    fn prefill_reusing_cache(&mut self, prompt_tokens: &[u32]) -> Result<Vec<f32>> {
        let n = prompt_tokens.len();
        debug_assert!(n >= 1);
        // Hard cap: the prompt must leave at least one slot for a generated token, and the
        // KV cache is bounded by `max_positions`. Without this, prefilling past the cache
        // overflowed it and the generation silently produced nothing.
        if n >= self.max_positions {
            return Err(BackendError::InvalidModelMetadata(format!(
                "conversation is {n} tokens, which exceeds the gemma4 {}-token context \
                 window — please start a new chat",
                self.max_positions
            )));
        }
        let disabled = std::env::var("CAMELID_GEMMA4_NO_PREFIX_CACHE").is_ok_and(|v| v == "1");
        let mut p = 0usize;
        if !disabled {
            let cap = self.max_positions.min(n);
            while p < cap
                && p < self.cached_tokens.len()
                && prompt_tokens[p] == self.cached_tokens[p]
            {
                p += 1;
            }
        }
        // Always run at least the final prompt token to produce its logits.
        let start = p.min(n - 1);
        let last = n - 1;
        let mut logits = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for pos in start..n {
            logits = self.forward_token(prompt_tokens[pos], pos, pos == last)?;
        }
        self.cached_tokens.clear();
        self.cached_tokens.extend_from_slice(prompt_tokens);
        Ok(logits)
    }

    pub fn generate_greedy(&mut self, prompt: &str, max_new: usize) -> Result<(String, Vec<u32>)> {
        let prompt_tokens = self.cpu.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.cpu.tokenizer);
        let mut logits = self.prefill_reusing_cache(&prompt_tokens)?;
        let mut generated = Vec::new();
        let decode_end = (prompt_tokens.len() + max_new).min(self.max_positions);
        for pos in prompt_tokens.len()..decode_end {
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
        // The cache now also holds the generated tokens — record them so the next request
        // can reuse this turn's full sequence as a prefix.
        self.cached_tokens.extend_from_slice(&generated);
        let text = self.cpu.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }

    pub fn generate_greedy_cancellable<C: FnMut() -> bool>(
        &mut self,
        prompt: &str,
        max_new: usize,
        should_cancel: C,
    ) -> Result<Gemma4GenerationOutcome> {
        self.generate_greedy_streaming_cancellable(prompt, max_new, |_| {}, should_cancel)
    }

    /// Greedy generate returning per-decode-token wall-clock times (seconds), for the
    /// SSER warm-up-curve measurement. `per_token[i]` is the time to produce
    /// `generated[i]` (the forward that emitted the NEXT logits), excluding prefill.
    pub fn generate_greedy_timed(
        &mut self,
        prompt: &str,
        max_new: usize,
    ) -> Result<(String, Vec<u32>, Vec<f64>)> {
        let prompt_tokens = self.cpu.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.cpu.tokenizer);
        let mut logits = self.prefill_reusing_cache(&prompt_tokens)?;
        let mut generated = Vec::new();
        let mut per_token = Vec::new();
        let decode_end = (prompt_tokens.len() + max_new).min(self.max_positions);
        for pos in prompt_tokens.len()..decode_end {
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
            let t = std::time::Instant::now();
            logits = self.forward_token(next, pos, true)?;
            per_token.push(t.elapsed().as_secs_f64());
        }
        self.cached_tokens.extend_from_slice(&generated);
        let text = self.cpu.tokenizer.decode(&generated, true)?;
        Ok((text, generated, per_token))
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
        let mut logits = self.prefill_reusing_cache(&prompt_tokens)?;
        let mut generated = Vec::new();
        let mut prev_text = String::new();
        let decode_end = (prompt_tokens.len() + max_new).min(self.max_positions);
        for pos in prompt_tokens.len()..decode_end {
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
        self.cached_tokens.extend_from_slice(&generated);
        Ok((prev_text, generated))
    }

    pub fn generate_greedy_streaming_cancellable<F: FnMut(&str), C: FnMut() -> bool>(
        &mut self,
        prompt: &str,
        max_new: usize,
        mut on_delta: F,
        mut should_cancel: C,
    ) -> Result<Gemma4GenerationOutcome> {
        if should_cancel() {
            self.cached_tokens.clear();
            return Ok(Gemma4GenerationOutcome::Cancelled {
                generated_tokens: 0,
            });
        }
        let prompt_tokens = self.cpu.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.cpu.tokenizer);
        let mut logits = self.prefill_reusing_cache(&prompt_tokens)?;
        if should_cancel() {
            self.cached_tokens.clear();
            return Ok(Gemma4GenerationOutcome::Cancelled {
                generated_tokens: 0,
            });
        }
        let mut generated = Vec::new();
        let mut prev_text = String::new();
        let decode_end = (prompt_tokens.len() + max_new).min(self.max_positions);
        for pos in prompt_tokens.len()..decode_end {
            if should_cancel() {
                self.cached_tokens.clear();
                return Ok(Gemma4GenerationOutcome::Cancelled {
                    generated_tokens: generated.len(),
                });
            }
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
            if should_cancel() {
                self.cached_tokens.clear();
                return Ok(Gemma4GenerationOutcome::Cancelled {
                    generated_tokens: generated.len(),
                });
            }
            logits = self.forward_token(next, pos, true)?;
        }
        self.cached_tokens.extend_from_slice(&generated);
        Ok(Gemma4GenerationOutcome::Complete {
            text: prev_text,
            token_ids: generated,
        })
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_parity_tests {
    use super::*;

    /// Deterministic filler for the head-upload layout tests (no rand dep).
    fn lcg_bytes(n: usize, mut seed: u32) -> Vec<u8> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 24) as u8
            })
            .collect()
    }

    // Root-cause regression for the gemma4 Q4_0 mis-decode. The tied-head GEMVs do
    // NOT read the stock GGUF wire: `q4k_gemv` indexes a SWIZZLED quant region and
    // `q6k_gemv` indexes super-blocks at a 224-byte PADDED stride. The Q4_K and Q6_K
    // head arms used to `clone_htod` the raw wire, so a Q4_0-quantized gemma4 export
    // (whose token_embd is Q4_K) computed every logit from wrongly-paired nibbles —
    // fluent-looking nonsense, no error. `gemma4_head_upload` is now the single upload
    // path; this pins each lane's layout so raw passthrough cannot come back.
    //
    // Asserted with explicit index arithmetic rather than by calling the repack
    // helpers back, so the test still fails if a helper is changed to agree with a
    // broken caller.
    #[test]
    fn gemma4_head_upload_matches_each_lane_gemv_layout() {
        // --- Q4_K: pure byte permutation of the quant region, header untouched. ---
        const Q4K_WIRE: usize = 144;
        let blocks = 5usize;
        let wire = lcg_bytes(blocks * Q4K_WIRE, 0x4b_4b_01);
        let up = gemma4_head_upload(HeadLane::Q4K, &wire);
        assert_eq!(up.len(), wire.len(), "the swizzle must not change size");
        assert!(
            up != wire,
            "raw passthrough is the defect: q4k_gemv reads swizzled quant bytes"
        );
        for b in 0..blocks {
            let (s, d) = (&wire[b * Q4K_WIRE..], &up[b * Q4K_WIRE..]);
            assert_eq!(
                &d[..16],
                &s[..16],
                "d/dmin/packed-scale header is untouched"
            );
            // The four stride-8 bytes an aux lane consumes must land contiguous.
            for g in 0..4 {
                for l in 0..8 {
                    for k in 0..4 {
                        assert_eq!(
                            d[16 + g * 32 + l * 4 + k],
                            s[16 + g * 32 + l + k * 8],
                            "q4k swizzle mismatch at block {b} group {g} lane {l} k {k}"
                        );
                    }
                }
            }
        }

        // --- Q6_K: 210-byte wire blocks padded to the 224-byte stride the kernel
        // indexes. Uploading raw here under-sizes the buffer AND mis-addresses every
        // block past the first, so this length check is the load-bearing assertion.
        const Q6K_WIRE: usize = 210;
        const Q6K_PADDED: usize = 224;
        let wire = lcg_bytes(blocks * Q6K_WIRE, 0x6b_6b_02);
        let up = gemma4_head_upload(HeadLane::Q6K, &wire);
        assert_eq!(
            up.len(),
            blocks * Q6K_PADDED,
            "q6k_gemv strides super-blocks by 224 B, not the 210 B wire"
        );
        for b in 0..blocks {
            assert_eq!(
                &up[b * Q6K_PADDED..b * Q6K_PADDED + Q6K_WIRE],
                &wire[b * Q6K_WIRE..(b + 1) * Q6K_WIRE],
                "q6k payload block {b} must survive the pad verbatim"
            );
        }

        // --- Q8_0: the SoA split q8_gemv reads (all quants, then all f16 scales). ---
        const Q8_WIRE: usize = 34;
        let wire = lcg_bytes(blocks * Q8_WIRE, 0x08_08_03);
        let up = gemma4_head_upload(HeadLane::Q8_0, &wire);
        assert_eq!(up.len(), wire.len());
        for b in 0..blocks {
            let src = &wire[b * Q8_WIRE..(b + 1) * Q8_WIRE];
            assert_eq!(
                &up[b * 32..b * 32 + 32],
                &src[2..34],
                "q8 quants are SoA-first"
            );
            assert_eq!(
                &up[blocks * 32 + b * 2..blocks * 32 + b * 2 + 2],
                &src[0..2],
                "q8 f16 scales trail the quant plane"
            );
        }
    }

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

/// BASALT Phase 3: the gemma4 wire lane's NVFP4 seam — WireFormat constants,
/// WireQuant admission (incl. the D17/T5 sentinel refusal), and matvec/matmul
/// consistency with `nvfp4_wire_row_dot`. Fixture-anchored + deterministic; no
/// model loads (a bare temp file of wire bytes plus a hand-built descriptor is
/// all `WireQuant::new` consumes).
#[cfg(test)]
mod nvfp4_wire_tests {
    use super::*;
    use crate::gguf::{GgufFile, GgufTensorDescriptor};

    /// Deterministic non-sentinel NVFP4 wire blocks: UE4M3 scale bytes drawn
    /// from a fixed safe set (0x00 zero through 0x7E max-normal; never
    /// 0x7F/0xFF), qs bytes from a small LCG-ish pattern.
    pub(super) fn synth_wire(superblocks: usize) -> Vec<u8> {
        const SAFE_SCALES: [u8; 8] = [0x00, 0x10, 0x2C, 0x38, 0x40, 0x51, 0x66, 0x7E];
        let mut wire = Vec::with_capacity(superblocks * 36);
        for b in 0..superblocks {
            for s in 0..4 {
                wire.push(SAFE_SCALES[(b + s) % SAFE_SCALES.len()]);
            }
            for j in 0..32 {
                wire.push(((b * 37 + j * 11 + 5) % 256) as u8);
            }
        }
        wire
    }

    pub(super) fn desc(
        name: &str,
        tensor_type: GgufTensorType,
        dims: &[u64],
        n_bytes: u64,
    ) -> GgufTensorDescriptor {
        GgufTensorDescriptor {
            name: name.into(),
            dimensions: dims.to_vec(),
            tensor_type,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes,
        }
    }

    #[test]
    fn sidecar_check_refuses_nvfp4_with_scale_tensors() {
        // ModelOpt-converted shape: NVFP4 weight + its sidecar `.scale` /
        // `.input_scale` F32 tensors — the wire lane must refuse (D-B2).
        for sidecar_name in ["blk.0.attn_q.scale", "blk.0.attn_q.input_scale"] {
            let tensors = vec![
                desc("blk.0.attn_q.weight", GgufTensorType::NVFP4, &[64, 4], 144),
                desc(sidecar_name, GgufTensorType::F32, &[1], 4),
            ];
            let err = nvfp4_sidecar_check(&tensors).expect_err("sidecar must refuse");
            let msg = err.to_string();
            assert!(msg.contains(sidecar_name), "{msg}");
            assert!(msg.contains("D-B2"), "{msg}");
        }
    }

    #[test]
    fn sidecar_check_admits_pilot_shapes() {
        // The pilot's real `layer_output_scale.weight` name must NOT false-positive,
        // and sidecar-suffixed names without any NVFP4 tensor are out of scope.
        let pilot = vec![
            desc("blk.0.attn_q.weight", GgufTensorType::NVFP4, &[64, 4], 144),
            desc(
                "blk.0.layer_output_scale.weight",
                GgufTensorType::F32,
                &[1, 4],
                16,
            ),
        ];
        nvfp4_sidecar_check(&pilot).expect("pilot shape admits");

        let no_nvfp4 = vec![
            desc("blk.0.attn_q.weight", GgufTensorType::Q8_0, &[64, 4], 136),
            desc("blk.0.attn_q.scale", GgufTensorType::F32, &[1], 4),
        ];
        nvfp4_sidecar_check(&no_nvfp4).expect("no NVFP4 -> check is out of scope");
    }

    /// Write `wire` to a temp file and wrap it in the two inputs WireQuant::new
    /// takes. The returned NamedTempFile keeps the mapping's backing file alive.
    pub(super) fn fixture(
        wire: &[u8],
        descs: Vec<GgufTensorDescriptor>,
    ) -> (tempfile::NamedTempFile, TensorStore, Arc<GgufWireMmap>) {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().expect("temp wire file");
        f.write_all(wire).expect("write wire bytes");
        f.flush().expect("flush wire bytes");
        let gguf = GgufFile {
            path: f.path().to_path_buf(),
            version: 3,
            tensor_count: descs.len() as i64,
            metadata_count: 0,
            alignment: 32,
            data_start_offset: 0,
            metadata: std::collections::BTreeMap::new(),
            tensors: descs,
        };
        let store = TensorStore::open(f.path(), &gguf);
        let mmap = GgufWireMmap::map(f.path()).expect("map wire file");
        (f, store, mmap)
    }

    #[test]
    fn wire_format_nvfp4_constants() {
        assert_eq!(WireFormat::Nvfp4.values_per_block(), 64);
        assert_eq!(WireFormat::Nvfp4.bytes_per_block(), 36);
        // 4 Q8_0 activation blocks = 128 values = 2 NVFP4 superblocks = 72 B...
        assert_eq!(WireFormat::Nvfp4.row_bytes_for_q8_blocks(4), 72);
        // ...while the 32-value formats keep their 1:1 block mapping.
        assert_eq!(WireFormat::Q8_0.row_bytes_for_q8_blocks(4), 4 * 34);
        assert_eq!(WireFormat::Q4_0.row_bytes_for_q8_blocks(4), 4 * 18);
    }

    #[test]
    fn wire_quant_new_admits_nvfp4_and_still_refuses_uncovered() {
        // 2 superblocks = 128 elements = 72 wire bytes, dims [64, 2].
        let wire = synth_wire(2);
        let (_f, store, mmap) = fixture(
            &wire,
            vec![
                desc("blk.0.attn_q.weight", GgufTensorType::NVFP4, &[64, 2], 72),
                desc("blk.0.attn_k.weight", GgufTensorType::BF16, &[64, 2], 256),
            ],
        );

        let wq = WireQuant::new(&store, &mmap, "blk.0.attn_q.weight").expect("NVFP4 admits");
        assert_eq!(wq.format, WireFormat::Nvfp4);
        assert_eq!(wq.element_count, 128);
        assert_eq!(wq.bytes(), &wire[..]);

        // An uncovered type keeps the fail-closed refusal, now naming NVFP4 as covered.
        // (WireQuant holds an Arc<GgufWireMmap> and derives no Debug, so match
        // instead of expect_err.)
        match WireQuant::new(&store, &mmap, "blk.0.attn_k.weight") {
            Err(BackendError::UnsupportedTensorType(msg)) => {
                assert!(msg.contains("gemma4 wire load supports"), "msg: {msg}");
                assert!(msg.contains("NVFP4"), "covered list names NVFP4: {msg}");
            }
            Err(other) => panic!("expected UnsupportedTensorType, got {other:?}"),
            Ok(_) => panic!("BF16 must stay refused"),
        }
    }

    #[test]
    fn wire_quant_new_refuses_nan_sentinel_scale_bytes() {
        for sentinel in [0x7Fu8, 0xFFu8] {
            let mut wire = synth_wire(2);
            wire[36 + 2] = sentinel; // block 1, d[2]
            let (_f, store, mmap) = fixture(
                &wire,
                vec![desc(
                    "blk.0.ffn_up.weight",
                    GgufTensorType::NVFP4,
                    &[64, 2],
                    72,
                )],
            );
            match WireQuant::new(&store, &mmap, "blk.0.ffn_up.weight") {
                Err(BackendError::InvalidTensorData(msg)) => {
                    assert!(msg.contains("NaN-sentinel"), "msg: {msg}");
                    assert!(msg.contains("block 1"), "first offending block: {msg}");
                }
                Err(other) => panic!("expected InvalidTensorData, got {other:?}"),
                Ok(_) => panic!("sentinel scale byte must refuse at load (D17/T5)"),
            }
        }
        // Zero scales admit (D17/T5: only the sentinel bytes refuse).
        let mut wire = synth_wire(2);
        for b in 0..2 {
            for s in 0..4 {
                wire[b * 36 + s] = 0x00;
            }
        }
        let (_f, store, mmap) = fixture(
            &wire,
            vec![desc(
                "blk.0.ffn_up.weight",
                GgufTensorType::NVFP4,
                &[64, 2],
                72,
            )],
        );
        WireQuant::new(&store, &mmap, "blk.0.ffn_up.weight").expect("zero scales admit");
    }

    #[test]
    fn metal_sentinel_check_refuses_nan_sentinel_nvfp4() {
        // GABBRO M3-followup: the Metal resident lane now RUNS NVFP4, reading wire raw
        // (bypassing WireQuant's scan), so nvfp4_metal_sentinel_check is the T5 guard —
        // a NaN-sentinel UE4M3 scale byte refuses at load, naming the tensor.
        for sentinel in [0x7Fu8, 0xFFu8] {
            let mut wire = synth_wire(2);
            wire[36 + 2] = sentinel; // block 1, d[2]
            let descs = vec![desc(
                "blk.7.ffn_down.weight",
                GgufTensorType::NVFP4,
                &[64, 2],
                72,
            )];
            let (_f, _store, mmap) = fixture(&wire, descs.clone());
            match nvfp4_metal_sentinel_check(&descs, &mmap) {
                Err(BackendError::InvalidTensorData(msg)) => {
                    assert!(msg.contains("NaN-sentinel"), "msg: {msg}");
                    assert!(
                        msg.contains("blk.7.ffn_down.weight"),
                        "names the tensor: {msg}"
                    );
                }
                Err(other) => panic!("expected InvalidTensorData, got {other:?}"),
                Ok(()) => panic!("sentinel-bearing NVFP4 must refuse on the Metal lane (D17/T5)"),
            }
        }
    }

    #[test]
    fn metal_sentinel_check_admits_clean_nvfp4_and_non_nvfp4() {
        // Clean NVFP4 admits (the lane runs it now); files without NVFP4 are out of scope.
        let wire = synth_wire(2);
        let descs = vec![desc(
            "blk.0.attn_q.weight",
            GgufTensorType::NVFP4,
            &[64, 2],
            72,
        )];
        let (_f, _store, mmap) = fixture(&wire, descs.clone());
        nvfp4_metal_sentinel_check(&descs, &mmap).expect("clean NVFP4 admits on the Metal lane");

        let descs2 = vec![desc(
            "blk.0.attn_q.weight",
            GgufTensorType::Q8_0,
            &[64, 2],
            136,
        )];
        let (_f2, _store2, mmap2) = fixture(&synth_wire(2), descs2.clone());
        nvfp4_metal_sentinel_check(&descs2, &mmap2).expect("non-NVFP4 files keep loading");
    }

    #[test]
    fn nvfp4_matvec_and_matmul_match_row_dot() {
        // 4 output rows x 128 inputs = 8 superblocks of wire.
        let (in_dim, out_dim) = (128usize, 4usize);
        let wire = synth_wire(8);
        let (_f, store, mmap) = fixture(
            &wire,
            vec![desc(
                "blk.0.attn_q.weight",
                GgufTensorType::NVFP4,
                &[in_dim as u64, out_dim as u64],
                wire.len() as u64,
            )],
        );
        let wq = WireQuant::new(&store, &mmap, "blk.0.attn_q.weight").expect("load");

        let x: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.37).sin() * 3.0)
            .collect();
        let xq = quantize_q8_0_blocks(&x);
        let row_bytes = WireFormat::Nvfp4.row_bytes_for_q8_blocks(xq.len());
        assert_eq!(row_bytes, 2 * 36, "two superblocks per 128-value row");

        // matvec (public dispatch) must equal the row dot on each wire row, bitwise.
        let out = wq.matvec(in_dim, out_dim, &x);
        for o in 0..out_dim {
            let want = nvfp4_wire_row_dot(&wire[o * row_bytes..(o + 1) * row_bytes], &xq);
            assert_eq!(
                out[o].to_bits(),
                want.to_bits(),
                "matvec row {o}: got {} want {want}",
                out[o]
            );
        }

        // matvec_q_rows: a row band must land on the same dots.
        let rows = wq.matvec_q_rows(1, 2, &xq);
        for (i, o) in (1..3).enumerate() {
            assert_eq!(rows[i].to_bits(), out[o].to_bits(), "row band offset {o}");
        }

        // Batched matmul_q over K activations == matvec_q per activation, bitwise
        // (the spec-verify shared-weight-read contract).
        let xs: Vec<Vec<f32>> = (0..3)
            .map(|k| {
                (0..in_dim)
                    .map(|i| ((i as f32) * 0.11 + k as f32 * 0.7).cos() * 2.0)
                    .collect()
            })
            .collect();
        let xqs: Vec<Vec<Q8_0Block>> = xs.iter().map(|x| quantize_q8_0_blocks(x)).collect();
        let batched = wq.matmul_q(out_dim, &xqs);
        for (k, xq) in xqs.iter().enumerate() {
            let single = wq.matvec_q(out_dim, xq);
            for o in 0..out_dim {
                assert_eq!(
                    batched[k][o].to_bits(),
                    single[o].to_bits(),
                    "matmul_q[{k}][{o}] != matvec_q"
                );
            }
        }
    }
}

/// BASALT Phase 3 SHA_E3 (§3 freeze-move crash fix) — K-quant LAYER-PROJECTION
/// routing. The per-layer projection call sites used to pre-quantize the shared
/// activation to Q8_0 and call `matvec_q` directly, which has no K-quant arms:
/// any gemma4 file with Q4_K/Q5_K/Q6_K projection matmuls panicked
/// `unreachable!` at forward time (latent pre-BASALT; probe-proven on the
/// campaign's Q4K-mm row). These tests pin the fixed dispatch three ways:
/// (1) K-quant projections route through the Q8_K family and land bit-equal to
/// the top-level [`WireQuant::matvec`] — the pre-existing, correct route — and
/// to the raw wire row dots; (2) the Q8_0-family dispatch stays byte-identical
/// to the direct Q8_0-activation path it replaced (NVFP4 non-disturbance at the
/// unit seam); (3) Q5_K matvec roles and Q4_1 gathers refuse TYPED, never panic
/// (invariant I-unknown-type, L2).
#[cfg(test)]
mod kquant_projection_tests {
    use super::nvfp4_wire_tests::{desc, fixture, synth_wire};
    use super::*;
    use crate::inference::{
        Q4_K_WIRE_BYTES_PER_BLOCK, Q5_K_WIRE_BYTES_PER_BLOCK, Q6_K_WIRE_BYTES_PER_BLOCK,
    };

    /// Deterministic K-quant wire: LCG byte fill, then tame f16 scale fields
    /// (per-block byte offsets in `f16_offs`) so no block scale is inf/NaN —
    /// the same recipe as the inference-layer K-quant dot tests.
    fn synth_kquant_wire(blocks: usize, bytes_per_block: usize, f16_offs: &[usize]) -> Vec<u8> {
        let mut wire = vec![0u8; blocks * bytes_per_block];
        for (i, b) in wire.iter_mut().enumerate() {
            *b = ((i * 131 + 17) % 256) as u8;
        }
        for blk in wire.chunks_exact_mut(bytes_per_block) {
            for (j, &off) in f16_offs.iter().enumerate() {
                let v = if j == 0 { 0.0173f32 } else { 0.0049 };
                blk[off..off + 2].copy_from_slice(&crate::tensor::f32_to_f16_bits(v).to_le_bytes());
            }
        }
        wire
    }

    fn activation(in_dim: usize, seed: f32) -> Vec<f32> {
        (0..in_dim)
            .map(|i| ((i as f32) * 0.37 + seed).sin() * 3.0)
            .collect()
    }

    #[test]
    fn kquant_projection_dispatch_matches_top_level_matvec_bitwise() {
        // 5 output rows x 512 inputs = 2 superblocks per row. Oracle #1 is the
        // top-level `matvec` (the route that was always correct for K-quants);
        // oracle #2 is the raw wire row dot on the same bytes.
        let (in_dim, out_dim) = (512usize, 5usize);
        let blocks_per_row = in_dim / 256;
        for (tt, bb, f16_offs) in [
            (
                GgufTensorType::Q4K,
                Q4_K_WIRE_BYTES_PER_BLOCK,
                vec![0usize, 2],
            ),
            (GgufTensorType::Q6K, Q6_K_WIRE_BYTES_PER_BLOCK, vec![208]),
        ] {
            let wire = synth_kquant_wire(blocks_per_row * out_dim, bb, &f16_offs);
            let (_f, store, mmap) = fixture(
                &wire,
                vec![desc(
                    "blk.0.attn_q.weight",
                    tt,
                    &[in_dim as u64, out_dim as u64],
                    wire.len() as u64,
                )],
            );
            let wq = WireQuant::new(&store, &mmap, "blk.0.attn_q.weight").expect("K-quant admits");

            let x = activation(in_dim, 0.0);
            let oracle = wq.matvec(in_dim, out_dim, &x);
            let sa = SharedActivation::new(&x);
            let got = wq.matvec_proj(out_dim, &sa);

            let xq = quantize_q8_k_blocks(&x);
            let row_bytes = blocks_per_row * bb;
            for o in 0..out_dim {
                assert_eq!(
                    got[o].to_bits(),
                    oracle[o].to_bits(),
                    "{tt:?} matvec_proj row {o} != top-level matvec"
                );
                let w_row = &wire[o * row_bytes..(o + 1) * row_bytes];
                let dot = match tt {
                    GgufTensorType::Q4K => q4_k_wire_row_dot(w_row, &xq),
                    _ => q6_k_wire_row_dot(w_row, &xq),
                };
                assert_eq!(
                    got[o].to_bits(),
                    dot.to_bits(),
                    "{tt:?} matvec_proj row {o} != wire row dot"
                );
            }
        }
    }

    #[test]
    fn kquant_batched_and_row_band_projections_match_single_dispatch() {
        // matmul_proj (the spec-verify chunk path) must equal matvec_proj per
        // activation, and matvec_rows_proj (the MoE expert-band path) must
        // land on the corresponding rows of the full matvec — all bitwise.
        let (in_dim, out_dim) = (256usize, 6usize);
        for (tt, bb, f16_offs) in [
            (
                GgufTensorType::Q4K,
                Q4_K_WIRE_BYTES_PER_BLOCK,
                vec![0usize, 2],
            ),
            (GgufTensorType::Q6K, Q6_K_WIRE_BYTES_PER_BLOCK, vec![208]),
        ] {
            let wire = synth_kquant_wire(out_dim, bb, &f16_offs);
            let (_f, store, mmap) = fixture(
                &wire,
                vec![desc(
                    "blk.0.ffn_up.weight",
                    tt,
                    &[in_dim as u64, out_dim as u64],
                    wire.len() as u64,
                )],
            );
            let wq = WireQuant::new(&store, &mmap, "blk.0.ffn_up.weight").expect("K-quant admits");

            let xs: Vec<Vec<f32>> = (0..3).map(|k| activation(in_dim, k as f32 * 0.7)).collect();
            let xb = SharedActivationBatch::new(&xs);
            let batched = wq.matmul_proj(out_dim, &xb);
            for (k, x) in xs.iter().enumerate() {
                let sa = SharedActivation::new(x);
                let single = wq.matvec_proj(out_dim, &sa);
                for o in 0..out_dim {
                    assert_eq!(
                        batched[k][o].to_bits(),
                        single[o].to_bits(),
                        "{tt:?} matmul_proj[{k}][{o}] != matvec_proj"
                    );
                }
            }

            let sa = SharedActivation::new(&xs[0]);
            let full = wq.matvec_proj(out_dim, &sa);
            let band = wq.matvec_rows_proj(2, 3, &sa);
            for (i, o) in (2..5).enumerate() {
                assert_eq!(
                    band[i].to_bits(),
                    full[o].to_bits(),
                    "{tt:?} matvec_rows_proj row band offset {o}"
                );
            }
        }
    }

    #[test]
    fn q8_0_family_dispatch_is_byte_identical_to_the_direct_q8_0_path() {
        // NO behavior change for the matvec_q family (NVFP4/Q8_0 shown here;
        // they share the dispatch arm with Q4_0/Q4_1): the routed calls must
        // equal the pre-fix direct calls on the eagerly-quantized activation.
        // Q8_0: 2 blocks/row of 34 bytes (f16 scale at +0), 4 rows x 64 inputs.
        let (in_dim, out_dim) = (64usize, 4usize);
        let q8_wire =
            synth_kquant_wire((in_dim / 32) * out_dim, Q8_WIRE_BYTES_PER_BLOCK, &[0usize]);
        // NVFP4: the pilot format — 128 inputs = 2 superblocks/row, 4 rows.
        let nv_wire = synth_wire(8);
        let cases: [(GgufTensorType, usize, &[u8]); 2] = [
            (GgufTensorType::Q8_0, in_dim, &q8_wire),
            (GgufTensorType::NVFP4, 128, &nv_wire),
        ];
        for (tt, in_dim, wire) in cases {
            let (_f, store, mmap) = fixture(
                wire,
                vec![desc(
                    "blk.0.attn_q.weight",
                    tt,
                    &[in_dim as u64, out_dim as u64],
                    wire.len() as u64,
                )],
            );
            let wq = WireQuant::new(&store, &mmap, "blk.0.attn_q.weight").expect("admits");

            let xs: Vec<Vec<f32>> = (0..3).map(|k| activation(in_dim, k as f32 * 0.7)).collect();
            let xqs: Vec<Vec<Q8_0Block>> = xs.iter().map(|x| quantize_q8_0_blocks(x)).collect();

            let sa = SharedActivation::new(&xs[0]);
            let via_dispatch = wq.matvec_proj(out_dim, &sa);
            let direct = wq.matvec_q(out_dim, &xqs[0]);
            for o in 0..out_dim {
                assert_eq!(
                    via_dispatch[o].to_bits(),
                    direct[o].to_bits(),
                    "{tt:?} matvec_proj row {o} != direct matvec_q"
                );
            }
            let band_dispatch = wq.matvec_rows_proj(1, 2, &sa);
            let band_direct = wq.matvec_q_rows(1, 2, &xqs[0]);
            for i in 0..2 {
                assert_eq!(
                    band_dispatch[i].to_bits(),
                    band_direct[i].to_bits(),
                    "{tt:?} matvec_rows_proj row {i} != direct matvec_q_rows"
                );
            }
            let xb = SharedActivationBatch::new(&xs);
            let batch_dispatch = wq.matmul_proj(out_dim, &xb);
            let batch_direct = wq.matmul_q(out_dim, &xqs);
            for k in 0..xs.len() {
                for o in 0..out_dim {
                    assert_eq!(
                        batch_dispatch[k][o].to_bits(),
                        batch_direct[k][o].to_bits(),
                        "{tt:?} matmul_proj[{k}][{o}] != direct matmul_q"
                    );
                }
            }
        }
    }

    #[test]
    fn q5k_matvec_roles_refuse_typed_at_load() {
        // Q5_K stays admitted for gather (per_layer_token_embd) but must
        // refuse TYPED in any matvec role — pre-fix it loaded fine and
        // panicked `unreachable!` in the forward pass.
        let wire = synth_kquant_wire(2, Q5_K_WIRE_BYTES_PER_BLOCK, &[0, 2]);
        let (_f, store, mmap) = fixture(
            &wire,
            vec![desc(
                "blk.0.attn_q.weight",
                GgufTensorType::Q5K,
                &[256, 2],
                wire.len() as u64,
            )],
        );
        let wq = WireQuant::new(&store, &mmap, "blk.0.attn_q.weight")
            .expect("Q5_K admits for gather roles");
        assert_eq!(wq.format, WireFormat::Q5K);
        wq.dequantize_elements(0, 4)
            .expect("Q5_K gather stays served");
        match wq.require_matvec_capable("blk.0.attn_q.weight") {
            Err(BackendError::UnsupportedTensorType(msg)) => {
                assert!(msg.contains("Q5_K"), "{msg}");
                assert!(msg.contains("gather-only"), "{msg}");
            }
            Err(other) => panic!("expected UnsupportedTensorType, got {other:?}"),
            Ok(_) => panic!("Q5_K must refuse matvec roles at load"),
        }
    }

    #[test]
    fn q4_1_gather_refuses_typed_instead_of_panicking() {
        // The sibling reachable-panic arm swept with the SHA_E3 fix: a Q4_1
        // embedding gather is not wired, so it must be a typed refusal.
        let wire = synth_kquant_wire(2, 20, &[0, 2]);
        let (_f, store, mmap) = fixture(
            &wire,
            vec![desc(
                "blk.0.ffn_down.weight",
                GgufTensorType::Q4_1,
                &[32, 2],
                wire.len() as u64,
            )],
        );
        let wq = WireQuant::new(&store, &mmap, "blk.0.ffn_down.weight").expect("Q4_1 admits");
        match wq.dequantize_elements(0, 4) {
            Err(BackendError::UnsupportedTensorType(msg)) => {
                assert!(msg.contains("Q4_1"), "{msg}");
            }
            Err(other) => panic!("expected UnsupportedTensorType, got {other:?}"),
            Ok(_) => panic!("Q4_1 gather must be a typed refusal"),
        }
    }
}

/// BASALT Amendment 3: the GPU-lane typed refusals and the §9 platform gate.
/// All helpers are cfg-independent, so these run on every host — no CUDA/Metal
/// hardware and no model loads (descriptor lists and raw [`WireFormat`]s only).
#[cfg(test)]
mod gpu_lane_refusal_tests {
    use super::*;
    use crate::gguf::GgufTensorDescriptor;

    fn desc(name: &str, tensor_type: GgufTensorType) -> GgufTensorDescriptor {
        GgufTensorDescriptor {
            name: name.into(),
            dimensions: vec![64, 1],
            tensor_type,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: 36,
        }
    }

    #[test]
    fn cuda_lane_check_admits_nvfp4_after_the_phase4_lift() {
        // BASALT Phase 4 (G4) inverted the pre-Phase-4 refusal: NVFP4 layer
        // projections now RESIDE on the CUDA lane (nvfp4_gemv), so an NVFP4
        // format in the projection set must ADMIT — a positive control that the
        // Phase-4 lift landed and the old "NVFP4 is Phase 4" refusal is gone.
        // (Regression guard: ratchet R3 requires this flip in the same PR that
        // closes the six L3 open:P4 cells.)
        nvfp4_cuda_lane_check([WireFormat::Q8_0, WireFormat::Nvfp4, WireFormat::Q4_0])
            .expect("NVFP4 projections now admit on the CUDA lane (Phase 4)");
    }

    #[test]
    fn cuda_lane_check_admits_the_supported_projection_formats() {
        // Every format from_wire actually supports must keep loading — Q8_0/Q4_0/
        // Q4_1 (pre-BASALT) plus NVFP4 (Phase 4). I-carveout boundary-preservation:
        // the K-quant refusal must not bleed onto the formats this lane serves.
        nvfp4_cuda_lane_check([
            WireFormat::Q8_0,
            WireFormat::Q4_0,
            WireFormat::Q4_1,
            WireFormat::Nvfp4,
        ])
        .expect("Q8_0/Q4_0/Q4_1/NVFP4 projections stay admitted");
        nvfp4_cuda_lane_check(std::iter::empty()).expect("no projections is vacuously fine");
    }

    #[test]
    fn cuda_lane_check_refuses_every_lane_uncovered_format_typed() {
        // SHA_E review finding #1: the campaign's own K-quant rows (Q4K-mm,
        // Q4_K_M-df/-im) load clean on the CPU wire lane but would hit the
        // from_wire repack panic on the CUDA lane. Every format outside the
        // lane's covered set must refuse TYPED and NAMED — never a panic
        // (invariant I-unknown-type, L3 cell).
        for uncovered in [WireFormat::Q4K, WireFormat::Q5K, WireFormat::Q6K] {
            match nvfp4_cuda_lane_check([WireFormat::Q8_0, uncovered]) {
                Err(BackendError::UnsupportedGguf(msg)) => {
                    assert!(
                        msg.contains(&format!("{uncovered:?}")),
                        "refusal must name the format: {msg}"
                    );
                    assert!(
                        msg.contains("covers Q8_0/Q4_0/Q4_1/NVFP4"),
                        "refusal must name the covered set: {msg}"
                    );
                }
                Err(other) => panic!("expected UnsupportedGguf, got {other:?}"),
                Ok(()) => panic!("{uncovered:?} projection must refuse in the CUDA lane"),
            }
        }
    }

    // GABBRO M3-followup: the blanket `nvfp4_metal_lane_check` refusal was lifted (the
    // Metal resident lane now RUNS NVFP4), replaced by `nvfp4_metal_sentinel_check` (the
    // T5 guard). Its tests need real wire bytes, so they live in the fixture-bearing mod:
    // `metal_sentinel_check_refuses_nan_sentinel_nvfp4` and
    // `metal_sentinel_check_admits_clean_nvfp4_and_non_nvfp4`.

    #[test]
    fn metal_layer_fmt_covers_q8_q4_nvfp4_refuses_others() {
        // I-unknown-type (L4): the Metal resident lane covers Q8_0/Q4_0/NVFP4 layer
        // projections; every other format refuses TYPED and NAMED, never a mis-bind.
        use crate::metal::GemmaWireFmt;
        assert_eq!(
            gemma4_metal_layer_fmt(GgufTensorType::Q8_0).unwrap(),
            GemmaWireFmt::Q8_0
        );
        assert_eq!(
            gemma4_metal_layer_fmt(GgufTensorType::Q4_0).unwrap(),
            GemmaWireFmt::Q4_0
        );
        assert_eq!(
            gemma4_metal_layer_fmt(GgufTensorType::NVFP4).unwrap(),
            GemmaWireFmt::Nvfp4
        );
        for uncovered in [
            GgufTensorType::Q6K,
            GgufTensorType::BF16,
            GgufTensorType::Q4K,
        ] {
            match gemma4_metal_layer_fmt(uncovered) {
                Err(BackendError::UnsupportedTensorType(msg)) => {
                    assert!(
                        msg.contains(&format!("{uncovered:?}")),
                        "names the format: {msg}"
                    );
                    assert!(
                        msg.contains("Q8_0/Q4_0/NVFP4"),
                        "names the covered set: {msg}"
                    );
                }
                other => panic!("uncovered format must refuse typed: {other:?}"),
            }
        }
    }

    #[test]
    fn windows_only_check_ignores_files_without_nvfp4() {
        // Platform-independent: the §9 gate only ever looks at NVFP4-bearing
        // files, so every other row is untouched on every OS.
        let tensors = vec![desc("blk.0.attn_q.weight", GgufTensorType::Q8_0)];
        nvfp4_windows_only_check(&tensors).expect("non-NVFP4 files admit everywhere");
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn windows_only_check_admits_nvfp4_on_windows() {
        // §9 twin (runs on the Windows leg): admission still works where the
        // release actually supports NVFP4.
        let tensors = vec![desc("blk.0.ffn_down.weight", GgufTensorType::NVFP4)];
        nvfp4_windows_only_check(&tensors).expect("NVFP4 admits on Windows");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn windows_only_check_admits_nvfp4_on_macos() {
        // GABBRO M2 twin (runs on the macOS leg): NVFP4 now admits on macOS too,
        // once the Apple-Silicon CPU decode was proven bit-exact (Gate G-M1).
        let tensors = vec![desc("blk.0.ffn_down.weight", GgufTensorType::NVFP4)];
        nvfp4_windows_only_check(&tensors).expect("NVFP4 admits on macOS (GABBRO M2)");
    }

    #[test]
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    fn windows_only_check_refuses_nvfp4_off_windows() {
        // §9 twin (runs on the linux leg — macOS now admits, GABBRO M2): the
        // named TK2 refusal on the still-unvalidated platforms.
        let tensors = vec![desc("blk.0.ffn_down.weight", GgufTensorType::NVFP4)];
        match nvfp4_windows_only_check(&tensors) {
            Err(BackendError::UnsupportedGguf(msg)) => {
                assert_eq!(
                    msg,
                    "NVFP4 is Windows/macOS-only in this release; see SUPPORT_MATRIX"
                );
            }
            Err(other) => panic!("expected UnsupportedGguf, got {other:?}"),
            Ok(()) => panic!("NVFP4 must refuse on unvalidated platforms (Amendment 3 §9)"),
        }
    }

    /// BASALT Phase 4 — L3 I-plat (cfg-twinned per §9.1): the shared §9 platform
    /// gate `nvfp4_windows_only_check` fires inside `Gemma4Runtime::load` (via
    /// `load_layer_range`), which is the FIRST act of `Gemma4CudaResident::load`,
    /// so it fronts the CUDA lane's entry before any CUDA initialization. This is
    /// the L3-native twin: the off-Windows legs assert the CUDA lane's shared
    /// entry gate yields the named TK2 refusal (no GPU needed to observe it —
    /// the gate is upstream of every CUDA call); the Windows leg asserts the pilot
    /// shape admits through the gate so the CUDA lane can bind (D-B3 carve-out).
    #[test]
    fn cuda_resident_platform_gate_fronts_the_cuda_lane_entry() {
        let pilot = vec![desc("blk.0.ffn_down.weight", GgufTensorType::NVFP4)];
        #[cfg(any(target_os = "windows", target_os = "macos"))]
        {
            nvfp4_windows_only_check(&pilot).expect(
                "NVFP4 admits through the §9 gate on Windows/macOS so the resident lane binds",
            );
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        {
            match nvfp4_windows_only_check(&pilot) {
                Err(BackendError::UnsupportedGguf(msg)) => assert_eq!(
                    msg, "NVFP4 is Windows/macOS-only in this release; see SUPPORT_MATRIX",
                    "the CUDA lane's shared entry gate must yield the named TK2 refusal"
                ),
                Err(other) => panic!("expected UnsupportedGguf, got {other:?}"),
                Ok(()) => panic!(
                    "the §9 gate fronting Gemma4CudaResident::load must refuse NVFP4 on unvalidated platforms"
                ),
            }
        }
    }

    /// BASALT Phase 4 — L3 I-sidecar: `Gemma4CudaResident::load`'s first act is
    /// `Gemma4Runtime::load`, whose `nvfp4_sidecar_check` (D-B2) fires before any
    /// CUDA work, so the CUDA lane cannot bind a sidecar-bearing NVFP4 file — it
    /// inherits the shared refusal. Driven here on the shared helper (cfg-
    /// independent, no GPU/model); the end-to-end file trip on the same seam is
    /// `sidecar_fixture_trips_d_b2_end_to_end`. Also asserts the pilot shape (no
    /// sidecar) admits, so the guard the CUDA lane relies on is exact, not blanket.
    #[test]
    fn cuda_resident_load_inherits_shared_sidecar_refusal() {
        let sidecar = vec![
            desc("blk.0.attn_q.weight", GgufTensorType::NVFP4),
            desc("blk.0.attn_q.scale", GgufTensorType::F32),
        ];
        match nvfp4_sidecar_check(&sidecar) {
            Err(BackendError::UnsupportedGguf(msg)) => {
                assert!(
                    msg.contains("blk.0.attn_q.scale"),
                    "names the sidecar: {msg}"
                );
                assert!(msg.contains("D-B2"), "cites D-B2: {msg}");
            }
            Err(other) => panic!("expected UnsupportedGguf, got {other:?}"),
            Ok(()) => panic!("sidecar-bearing NVFP4 must refuse before the CUDA lane binds (D-B2)"),
        }
        let pilot = vec![desc("blk.0.attn_q.weight", GgufTensorType::NVFP4)];
        nvfp4_sidecar_check(&pilot).expect("pilot NVFP4 has no sidecar; the CUDA lane may bind");
    }
}

/// BASALT Amendment 3 review fix #4: the forced-decode step-boundary proof.
/// A scripted fake step fn stands in for the model: `predicted = 1000 + fed`,
/// prompt-end prediction 999 — so every observation uniquely identifies WHICH
/// token had been fed before it, and any off-by-one is unmissable.
#[cfg(test)]
mod forced_step_boundary_tests {
    use super::drive_forced_steps;

    #[test]
    fn observes_before_feeding_and_never_feeds_the_final_token() {
        let forced = [10u32, 20, 30];
        // One interleaved event log proves strict ordering, not just counts.
        // (RefCell: both closures append to the same log.)
        let events = std::cell::RefCell::new(Vec::<String>::new());
        let mut fed: Vec<u32> = Vec::new();
        drive_forced_steps::<u32, std::convert::Infallible>(
            &forced,
            999,
            |tok| {
                fed.push(tok);
                events.borrow_mut().push(format!("fed={tok}"));
                Ok(1000 + tok)
            },
            |i, &pred| events.borrow_mut().push(format!("obs{i}={pred}")),
        )
        .unwrap();
        let events = events.into_inner();

        // Step i's recorded prediction is the state from BEFORE forced[i] was
        // fed: obs0 sees the prompt-end prediction (999), obs1 sees 1000+forced[0],
        // obs2 sees 1000+forced[1]. If the loop fed first and observed second,
        // obs_i would read 1000+forced[i] instead.
        assert_eq!(
            events,
            vec!["obs0=999", "fed=10", "obs1=1010", "fed=20", "obs2=1020"]
        );
        // count == forced.len(): exactly 3 observations fired (asserted above by
        // the full event log), and the FINAL forced token (30) was never fed.
        assert_eq!(fed, vec![10, 20]);
    }

    #[test]
    fn single_forced_token_observes_once_and_feeds_nothing() {
        let mut observed = Vec::new();
        drive_forced_steps::<u32, std::convert::Infallible>(
            &[42],
            7,
            |_| panic!("a single forced token must never be fed"),
            |i, &pred| observed.push((i, pred)),
        )
        .unwrap();
        assert_eq!(observed, vec![(0, 7)]);
    }

    #[test]
    fn empty_forced_list_neither_observes_nor_feeds() {
        // The CLI refuses empty lists upstream; the construct itself is total.
        drive_forced_steps::<u32, std::convert::Infallible>(
            &[],
            0,
            |_| panic!("nothing to feed"),
            |_, _| panic!("nothing to observe"),
        )
        .unwrap();
    }

    #[test]
    fn step_errors_propagate_after_the_boundary_observation() {
        let mut observed = 0usize;
        let err = drive_forced_steps::<u32, &'static str>(
            &[1, 2],
            0,
            |_| Err("step failed"),
            |_, _| observed += 1,
        )
        .unwrap_err();
        assert_eq!(err, "step failed");
        // The step-0 observation (pre-feed) had already fired.
        assert_eq!(observed, 1);
    }
}

#[cfg(test)]
mod scalar_prefill_head_tests {
    use super::drive_scalar_prefill;

    #[test]
    fn projects_only_the_final_prompt_position() {
        let calls = std::cell::RefCell::new(Vec::new());
        let logits = drive_scalar_prefill(&[10, 20, 30, 40], |token, pos, project_head| {
            calls.borrow_mut().push((token, pos, project_head));
            Ok::<Option<u32>, std::convert::Infallible>(project_head.then_some(token + 1))
        })
        .unwrap();

        assert_eq!(logits, 41);
        assert_eq!(
            calls.into_inner(),
            vec![
                (10, 0, false),
                (20, 1, false),
                (30, 2, false),
                (40, 3, true),
            ]
        );
    }
}

#[cfg(test)]
mod ghost_hybrid_prefill_plan_tests {
    use super::{select_ghost_prefill_plan, GhostPrefillPlan};

    #[test]
    fn multi_token_common_prefill_defaults_to_hybrid_but_has_a_scalar_kill_switch() {
        assert_eq!(
            select_ghost_prefill_plan(true, true, 16, 31, Some(4096)),
            GhostPrefillPlan::HybridChunk
        );
        assert_eq!(
            select_ghost_prefill_plan(true, false, 16, 31, Some(4096)),
            GhostPrefillPlan::ScalarMetal
        );
    }

    #[test]
    fn one_token_prompt_stays_scalar_metal_and_over_capacity_stays_cpu_from_zero() {
        assert_eq!(
            select_ghost_prefill_plan(true, true, 1, 8, Some(4096)),
            GhostPrefillPlan::ScalarMetal
        );
        assert_eq!(
            select_ghost_prefill_plan(true, true, 4000, 4127, Some(4096)),
            GhostPrefillPlan::CpuChunk
        );
        assert_eq!(
            select_ghost_prefill_plan(false, true, 1, 4127, Some(4096)),
            GhostPrefillPlan::ScalarCpu
        );
    }
}

#[cfg(test)]
mod ghost_moe_wire_tests {
    use super::*;
    use crate::ghost::{
        CghostGroup, CghostIndex, CghostLayout, CghostTensor, CGHOST_ALIGN, CGHOST_MAGIC,
    };
    use std::io::{Seek, SeekFrom, Write};

    #[test]
    fn ghost_metal_dispatch_requires_runtime_gpu_and_non_deterministic_mode() {
        assert!(ghost_metal_acceleration_allowed(false, true));
        assert!(!ghost_metal_acceleration_allowed(false, false));
        assert!(!ghost_metal_acceleration_allowed(true, true));
        assert!(!ghost_metal_acceleration_allowed(true, false));
    }

    #[test]
    fn cryptographic_ghost_identity_survives_a_gguf_rename() {
        assert!(ghost_source_filename_admitted(
            true,
            "original.gguf",
            Some("renamed.gguf")
        ));
        assert!(ghost_source_filename_admitted(
            false,
            "original.gguf",
            Some("original.gguf")
        ));
        assert!(!ghost_source_filename_admitted(
            false,
            "original.gguf",
            Some("renamed.gguf")
        ));
        assert!(ghost_source_filename_admitted(
            false,
            "",
            Some("anything.gguf")
        ));
    }

    fn cache_fixture(
        block_count: usize,
        expert_count: usize,
    ) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.cghost");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(CGHOST_MAGIC).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        let mut cursor = (CGHOST_MAGIC.len() + 8) as u64;
        let mut groups = Vec::new();
        for layer in 0..block_count {
            for expert in 0..expert_count {
                let aligned = cursor.next_multiple_of(CGHOST_ALIGN);
                file.write_all(&vec![0; (aligned - cursor) as usize])
                    .unwrap();
                cursor = aligned;
                let marker = (layer * expert_count + expert) as u8;
                file.write_all(&[marker; 4]).unwrap();
                groups.push(CghostGroup {
                    id: format!("blk.{layer}.exp.{expert}"),
                    tensors: vec![
                        CghostTensor {
                            name: "gate".into(),
                            role: "gate_up_exps".into(),
                            dtype: GgufTensorType::Q4_0,
                            dims: vec![32, 2],
                            offset: cursor,
                            len: 2,
                        },
                        CghostTensor {
                            name: "down".into(),
                            role: "down_exps".into(),
                            dtype: GgufTensorType::Q4_0,
                            dims: vec![32, 2],
                            offset: cursor + 2,
                            len: 2,
                        },
                    ],
                    source_sample_sha256: None,
                });
                cursor += 4;
            }
        }
        let index = CghostIndex {
            version: 2,
            layout: CghostLayout::MoeExperts,
            source_model: "cache.gguf".into(),
            block_count,
            tied_output: true,
            expert_count: Some(expert_count),
            expert_used_count: Some(1),
            source_identity: None,
            groups,
        };
        let index_bytes = serde_json::to_vec(&index).unwrap();
        file.write_all(&index_bytes).unwrap();
        file.seek(SeekFrom::Start(CGHOST_MAGIC.len() as u64))
            .unwrap();
        file.write_all(&cursor.to_le_bytes()).unwrap();
        drop(file);
        (dir, path)
    }

    #[test]
    fn owned_expert_wire_matches_the_existing_q4_row_kernel_bitwise() {
        let rows = 2usize;
        let row_bytes = crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK;
        let mut wire = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            let base = row * row_bytes;
            let scale = crate::tensor::f32_to_f16_bits(0.125 + row as f32 * 0.0625);
            wire[base..base + 2].copy_from_slice(&scale.to_le_bytes());
            for (i, byte) in wire[base + 2..base + row_bytes].iter_mut().enumerate() {
                *byte = ((i * 17 + row * 29) & 0xff) as u8;
            }
        }

        // Prefix/suffix sentinels prove the owned view honors its range instead
        // of assuming the expert tensor begins at allocation offset zero.
        let prefix = 7usize;
        let mut allocation = vec![0xa5; prefix];
        allocation.extend_from_slice(&wire);
        allocation.extend_from_slice(&[0x5a; 11]);
        let allocation: Arc<[u8]> = allocation.into();
        let weight = WireQuant::from_owned_wire(
            allocation,
            prefix..prefix + wire.len(),
            GgufTensorType::Q4_0,
            &[32, rows as u64],
            "test ghost expert",
        )
        .unwrap();
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.31).sin()).collect();
        let activation = SharedActivation::new(&x);
        let got = weight.matvec_rows_proj(0, rows, &activation);
        let xq = activation.q8_0();
        for row in 0..rows {
            let expected = q4_0_wire_row_dot(&wire[row * row_bytes..(row + 1) * row_bytes], xq);
            assert_eq!(got[row].to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn owned_expert_wire_rejects_a_truncated_view() {
        let bytes: Arc<[u8]> = vec![0u8; 17].into();
        let err = match WireQuant::from_owned_wire(
            bytes,
            0..17,
            GgufTensorType::Q4_0,
            &[32, 1],
            "truncated ghost expert",
        ) {
            Ok(_) => panic!("one Q4_0 row requires 18 bytes"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("expected 18"));
    }

    #[test]
    fn global_expert_cache_never_exceeds_its_byte_budget() {
        let (_dir, path) = cache_fixture(1, 2);

        let cache = GhostMoeExpertCache::new(Arc::new(GhostFile::open(&path).unwrap()), 4);
        cache.get(0, 0).unwrap();
        cache.get(0, 0).unwrap(); // hit
        cache.get(0, 1).unwrap(); // must evict expert 0
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.resident_experts, 1);
        assert!(stats.resident_bytes <= stats.budget_bytes);
    }

    #[test]
    fn resident_peek_supplies_slot_bytes_without_changing_cache_stats() {
        let (_dir, path) = cache_fixture(1, 2);
        let cache = GhostMoeExpertCache::new(Arc::new(GhostFile::open(&path).unwrap()), 4);
        cache.get(0, 1).unwrap();
        let before = cache.stats();
        let resident = cache
            .peek_resident(0, 1)
            .expect("resident expert should be available as a slot-fill source");
        let (bytes, _) = resident.tensor_backing(&resident.gate_up);
        assert_eq!(bytes[0], 1);
        assert_eq!(cache.stats(), before);
        assert!(cache.peek_resident(0, 0).is_none());
        assert_eq!(cache.stats(), before);
    }

    #[test]
    fn batch_read_restores_router_order_after_sorted_parallel_io() {
        let (_dir, path) = cache_fixture(1, 3);
        let cache = GhostMoeExpertCache::new(Arc::new(GhostFile::open(&path).unwrap()), 12);
        let routed = cache.get_many(0, &[2, 0, 1]).unwrap();
        let markers = routed
            .iter()
            .map(|expert| {
                let (bytes, range) = expert.tensor_backing(&expert.gate_up);
                bytes[range.start]
            })
            .collect::<Vec<_>>();
        assert_eq!(markers, vec![2, 0, 1]);
        assert_eq!(cache.stats().misses, 3);
    }

    #[test]
    fn batch_read_warms_an_over_budget_segment_by_route_frequency() {
        let (_dir, path) = cache_fixture(1, 3);
        // The segment holds two experts. Expert 0 is physically read first but
        // requested most often, so admission must move it behind colder rows.
        let cache = GhostMoeExpertCache::new(Arc::new(GhostFile::open(&path).unwrap()), 8);
        cache.get_many(0, &[0, 2, 0, 1, 0]).unwrap();
        {
            let state = cache.state.lock().unwrap();
            assert!(state.entries.contains_key(&(0, 0)));
            assert_eq!(state.entries.get(&(0, 0)).unwrap().frequency, 3);
        }
        cache.get(0, 0).unwrap();
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 5);
        assert_eq!(stats.resident_experts, 2);
    }

    #[test]
    fn layer_segments_prevent_cross_layer_scan_pollution() {
        let (_dir, path) = cache_fixture(2, 2);
        // Two four-byte segments: inserting a second layer-0 expert may evict
        // layer 0's old entry, but it must not bulldoze layer 1's resident hit.
        let cache = GhostMoeExpertCache::new(Arc::new(GhostFile::open(&path).unwrap()), 8);
        cache.get(0, 0).unwrap();
        cache.get(1, 0).unwrap();
        cache.get(0, 1).unwrap();
        cache.get(1, 0).unwrap();
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 3);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.resident_experts, 2);
        assert!(stats.resident_bytes <= stats.budget_bytes);
    }

    #[test]
    fn segment_evicts_lfu_then_uses_lru_as_tie_break() {
        let (_dir, path) = cache_fixture(1, 3);
        let cache = GhostMoeExpertCache::new(Arc::new(GhostFile::open(&path).unwrap()), 8);
        cache.get(0, 0).unwrap();
        cache.get(0, 1).unwrap();
        cache.get(0, 0).unwrap(); // expert 0 frequency = 2
        cache.get(0, 2).unwrap(); // expert 1 is the LFU victim
        cache.get(0, 0).unwrap();

        let state = cache.state.lock().unwrap();
        assert!(state.entries.contains_key(&(0, 0)));
        assert!(!state.entries.contains_key(&(0, 1)));
        assert!(state.entries.contains_key(&(0, 2)));
        assert_eq!(state.hits, 2);
        assert_eq!(state.misses, 3);
        assert_eq!(state.evictions, 1);
    }

    #[test]
    fn metal_slot_plan_preserves_route_order_and_deduplicates_loads() {
        let mut directory = GhostMetalSlotDirectory::new(4);
        let plan = directory.plan(&[9, 2, 9, 4]).unwrap();
        assert_eq!(plan.route_slots, vec![0, 1, 0, 2]);
        assert_eq!(plan.hits, 1);
        assert_eq!(plan.evictions, 0);
        assert_eq!(
            plan.loads
                .iter()
                .map(|load| (load.slot, load.expert, load.frequency))
                .collect::<Vec<_>>(),
            vec![(0, 9, 2), (1, 2, 1), (2, 4, 1)]
        );
        for load in plan.loads {
            directory.commit_load(load);
        }

        let hits = directory.plan(&[4, 9]).unwrap();
        assert_eq!(hits.route_slots, vec![2, 0]);
        assert_eq!(hits.hits, 2);
        assert_eq!(hits.evictions, 0);
        assert!(hits.loads.is_empty());
    }

    #[test]
    fn metal_slot_count_config_defaults_and_clamps_to_safe_bounds() {
        assert_eq!(parse_ghost_metal_slots_per_layer(None), 16);
        assert_eq!(parse_ghost_metal_slots_per_layer(Some("invalid")), 16);
        assert_eq!(parse_ghost_metal_slots_per_layer(Some("0")), 8);
        assert_eq!(parse_ghost_metal_slots_per_layer(Some("8")), 8);
        assert_eq!(parse_ghost_metal_slots_per_layer(Some("24")), 24);
        assert_eq!(parse_ghost_metal_slots_per_layer(Some("32")), 32);
        assert_eq!(parse_ghost_metal_slots_per_layer(Some("96")), 96);
        assert_eq!(parse_ghost_metal_slots_per_layer(Some("4096")), 128);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_slot_stats_delta_tracks_churn_without_wraparound() {
        let before = GhostMetalSlotStats {
            route_lookups: 80,
            hits: 40,
            misses: 40,
            evictions: 24,
            direct_reads: 36,
            direct_read_bytes: 120_000,
            ..GhostMetalSlotStats::default()
        };
        let after = GhostMetalSlotStats {
            route_lookups: 96,
            hits: 50,
            misses: 46,
            evictions: 29,
            host_fills: 2,
            prewarm_copies: 3,
            direct_reads: 40,
            direct_read_bytes: 133_380,
            direct_read_failures: 1,
        };
        assert_eq!(
            after.saturating_delta(before),
            GhostMetalSlotStats {
                route_lookups: 16,
                hits: 10,
                misses: 6,
                evictions: 5,
                host_fills: 2,
                prewarm_copies: 3,
                direct_reads: 4,
                direct_read_bytes: 13_380,
                direct_read_failures: 1,
            }
        );
    }

    #[test]
    fn metal_slot_failed_read_never_publishes_partial_bytes() {
        let mut directory = GhostMetalSlotDirectory::new(2);
        let warm = directory.plan(&[1, 2]).unwrap();
        for load in warm.loads {
            directory.commit_load(load);
        }
        // Make expert 1 hotter so expert 2 is the deterministic victim.
        assert!(directory.plan(&[1]).unwrap().loads.is_empty());
        let failed = directory.plan(&[3]).unwrap();
        assert_eq!(failed.loads.len(), 1);
        assert_eq!(failed.loads[0].slot, 1);
        assert_eq!(failed.loads[0].expert, 3);
        // Deliberately do not commit: this models a failed positioned read.
        assert!(directory.entries[1].is_none());

        // Neither the evicted expert nor the failed replacement may hit. The
        // empty slot is safely reused and published only after commit.
        let retry = directory.plan(&[2]).unwrap();
        assert_eq!(retry.loads.len(), 1);
        assert_eq!(retry.loads[0].slot, 1);
        assert_eq!(retry.loads[0].expert, 2);
    }

    #[test]
    fn metal_slot_route_is_preflighted_before_any_eviction() {
        let mut directory = GhostMetalSlotDirectory::new(2);
        let warm = directory.plan(&[5, 6]).unwrap();
        for load in warm.loads {
            directory.commit_load(load);
        }
        let before = directory.entries.clone();
        let err = directory.plan(&[7, 8, 9]).unwrap_err();
        assert!(err.to_string().contains("3 distinct experts"));
        assert_eq!(directory.entries, before);
    }

    #[test]
    fn metal_prompt_prewarm_honors_configured_limit_and_preserves_route_evidence() {
        let broad_routes: Vec<usize> = (0..40).collect();
        assert_eq!(
            ghost_metal_prewarm_sequence(&broad_routes, 40, 8),
            (32..40).collect::<Vec<_>>()
        );
        assert_eq!(
            ghost_metal_prewarm_sequence(&broad_routes, 40, 32),
            (8..40).collect::<Vec<_>>()
        );
        let routed: Vec<usize> = (0..18).collect();
        assert_eq!(
            ghost_metal_prewarm_sequence(&routed, 18, 16),
            (2..18).collect::<Vec<_>>(),
            "equal-frequency ties should retain the most recent experts"
        );

        let routed = vec![0, 1, 2, 0, 3, 1, 4];
        assert_eq!(
            ghost_metal_prewarm_sequence(&routed, 5, 2),
            vec![0, 1, 0, 1],
            "filtered route must retain occurrence count and original order"
        );
        assert_eq!(
            ghost_metal_prewarm_sequence(&routed, 5, 1),
            vec![1, 1],
            "recency breaks equal-frequency ties"
        );
    }

    /// Opt-in production-row admission gate. Unlike the synthetic kernel
    /// fixtures, this proves the complete GGUF + expert-spliced `.cghost` load
    /// reaches an actually configured persistent common core. It intentionally
    /// stops before generation so admission regressions can be diagnosed
    /// without paying prompt/decode time.
    #[cfg(target_os = "macos")]
    #[test]
    fn ghost_common_real_model_admits_when_fixture_is_configured() {
        if !crate::metal::detect_metal_device().available {
            eprintln!("SKIP Ghost common admission: no Metal device");
            return;
        }
        let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost common admission: set CAMELID_GEMMA4_GGUF");
            return;
        };
        let Some(cghost) =
            std::env::var_os("CAMELID_GEMMA4_GHOST_CGHOST").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost common admission: set CAMELID_GEMMA4_GHOST_CGHOST");
            return;
        };
        let flag = |name: &str| {
            std::env::var(name).is_ok_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
        };
        if !flag("CAMELID_GEMMA4_GHOST_METAL_SLOTS") || !flag("CAMELID_GEMMA4_GHOST_METAL_COMMON") {
            eprintln!(
                "SKIP Ghost common admission: enable CAMELID_GEMMA4_GHOST_METAL_SLOTS=1 and CAMELID_GEMMA4_GHOST_METAL_COMMON=1"
            );
            return;
        }
        let slot_env = std::env::var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER").ok();
        let expected_slots = parse_ghost_metal_slots_per_layer(slot_env.as_deref());

        let runtime = Gemma4Runtime::load_ghost_moe(&model, &cghost, 64, false)
            .expect("load production Ghost-MoE fixture");
        assert_eq!(runtime.layers.len(), 30);
        assert!(
            runtime
                .layers
                .iter()
                .all(|layer| layer.ple_output_scale.is_finite()),
            "all learned layer output scales must be finite"
        );
        assert!(
            runtime
                .layers
                .iter()
                .any(|layer| layer.ple_output_scale.to_bits() != 1.0f32.to_bits()),
            "production fixture must exercise learned non-unit layer scales"
        );
        assert!(
            runtime.ghost_common_metal_active(),
            "production Ghost-MoE fixture did not configure the persistent Metal common core"
        );
        let slot_guard = runtime
            .metal_q4_experts
            .lock()
            .expect("Ghost Metal runtime mutex poisoned");
        assert_eq!(
            slot_guard
                .as_ref()
                .expect("persistent slot lane is absent")
                .slots_per_layer(),
            expected_slots,
            "production slot slab did not honor the configured capacity"
        );
        drop(slot_guard);
        eprintln!(
            "[gemma4-ghost-common-test] ACTIVE: production GGUF/cghost pair admitted with 30 learned layer-output scales and {expected_slots} slots/layer"
        );
    }

    /// Opt-in real 26B tied-head gate. It compares the established CPU Q6_K
    /// projection with the no-copy Metal head on the exact local GGUF and prints
    /// cold/warm timings for performance diagnosis. No fixture is required in CI.
    #[cfg(target_os = "macos")]
    #[test]
    fn ghost_metal_q6k_head_matches_cpu_argmax_when_fixture_is_configured() {
        if !crate::metal::detect_metal_device().available {
            eprintln!("SKIP Ghost Metal head parity: no Metal device");
            return;
        }
        let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost Metal head parity: set CAMELID_GEMMA4_GGUF");
            return;
        };
        let Some(cghost) =
            std::env::var_os("CAMELID_GEMMA4_GHOST_CGHOST").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost Metal head parity: set CAMELID_GEMMA4_GHOST_CGHOST");
            return;
        };
        if std::env::var("CAMELID_GEMMA4_GHOST_METAL_HEAD")
            .is_ok_and(|value| value == "0" || value.eq_ignore_ascii_case("false"))
        {
            eprintln!("SKIP Ghost Metal head parity: Metal head explicitly disabled");
            return;
        }
        let mut runtime = Gemma4Runtime::load_ghost_moe(&model, &cghost, 64, false)
            .expect("load Ghost-MoE fixture");
        let head = runtime
            .metal_q6k_head
            .as_ref()
            .expect("real 26B Q6_K Ghost fixture should bind the Metal head");
        let hidden_size = runtime.hidden_size();
        // A real tied embedding row supplies a representative, deterministic
        // activation without paying for a full 30-layer Ghost forward.
        let hidden = runtime
            .token_embd
            .dequantize_elements(100 * hidden_size, hidden_size)
            .expect("gather representative hidden row");

        let cpu_started = std::time::Instant::now();
        let cpu = runtime.project_logits_cpu(&hidden);
        let cpu_elapsed = cpu_started.elapsed();
        let cold_started = std::time::Instant::now();
        let metal = head.forward(&hidden).expect("cold Metal head forward");
        let cold_elapsed = cold_started.elapsed();
        let warm_started = std::time::Instant::now();
        let metal_warm = head.forward(&hidden).expect("warm Metal head forward");
        let warm_elapsed = warm_started.elapsed();
        assert_eq!(metal, metal_warm, "reused Metal head must be deterministic");

        let argmax = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
                .unwrap()
        };
        let max_abs = cpu
            .iter()
            .zip(&metal)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "Ghost Q6_K head: CPU={:.3}s Metal-cold={:.3}s Metal-warm={:.3}s max_abs={max_abs:.6}",
            cpu_elapsed.as_secs_f64(),
            cold_elapsed.as_secs_f64(),
            warm_elapsed.as_secs_f64(),
        );
        assert_eq!(
            argmax(&metal),
            argmax(&cpu),
            "Metal Q6_K head changed the real model's next-token argmax"
        );
        assert_eq!(
            metal, cpu,
            "strict ordered Metal Q6_K + CPU soft-cap must match the CPU head bit-for-bit"
        );

        // Natural hidden/token gate: run one real Ghost step twice with fresh KV,
        // first through Metal and then with only the head removed. Both paths use
        // the exact same decoder/runtime and must choose the same greedy token.
        let natural_started = std::time::Instant::now();
        let (metal_text, metal_ids) = runtime
            .generate_greedy("Hello", 1)
            .expect("natural Metal-head generation");
        let natural_metal_elapsed = natural_started.elapsed();
        let saved_head = runtime
            .metal_q6k_head
            .take()
            .expect("fixture bound a Metal head above");
        let natural_started = std::time::Instant::now();
        let (cpu_text, cpu_ids) = runtime
            .generate_greedy("Hello", 1)
            .expect("natural CPU-head generation");
        let natural_cpu_elapsed = natural_started.elapsed();
        runtime.metal_q6k_head = Some(saved_head);
        eprintln!(
            "Ghost natural one-token: Metal-head={:.3}s CPU-head={:.3}s ids={metal_ids:?}",
            natural_metal_elapsed.as_secs_f64(),
            natural_cpu_elapsed.as_secs_f64(),
        );
        assert_eq!(metal_ids, cpu_ids, "natural greedy token changed");
        assert_eq!(metal_text, cpu_text, "natural decoded token changed");
    }

    /// Opt-in real 26B parity and timing gate for the persistent Q4_0 expert
    /// slots. Run with the two fixture paths plus
    /// `CAMELID_GEMMA4_GHOST_METAL_SLOTS=1`. The first half isolates one natural
    /// MoE layer and requires every final FFN bit to match the CPU Ghost oracle;
    /// the second emits two tokens so token #2 is predicted by a full 30-layer
    /// decode through the persistent slot lane.
    #[cfg(target_os = "macos")]
    #[test]
    fn ghost_metal_q4_slots_match_real_layer_and_natural_decode() {
        if !crate::metal::detect_metal_device().available {
            eprintln!("SKIP Ghost Metal Q4 slots parity: no Metal device");
            return;
        }
        let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost Metal Q4 slots parity: set CAMELID_GEMMA4_GGUF");
            return;
        };
        let Some(cghost) =
            std::env::var_os("CAMELID_GEMMA4_GHOST_CGHOST").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost Metal Q4 slots parity: set CAMELID_GEMMA4_GHOST_CGHOST");
            return;
        };
        let runtime = Gemma4Runtime::load_ghost_moe(&model, &cghost, 256, false)
            .expect("load real Ghost-MoE fixture");
        if !runtime.ghost_metal_q4_is_enabled() {
            eprintln!("SKIP Ghost Metal Q4 slots parity: set CAMELID_GEMMA4_GHOST_METAL_SLOTS=1");
            return;
        }

        let hidden_size = runtime.hidden_size();
        let hidden = runtime
            .token_embd
            .dequantize_elements(100 * hidden_size, hidden_size)
            .expect("gather representative real hidden row");
        let metal_started = std::time::Instant::now();
        let metal_layer = runtime
            .moe_layer_ffn(0, &hidden)
            .expect("real layer-0 persistent-slot FFN");
        let metal_layer_elapsed = metal_started.elapsed();
        let saved_lane = runtime
            .metal_q4_experts
            .lock()
            .expect("Metal expert mutex poisoned")
            .take()
            .expect("fixture bound persistent Metal expert slots");
        let cpu_started = std::time::Instant::now();
        let cpu_layer = runtime
            .moe_layer_ffn(0, &hidden)
            .expect("real layer-0 CPU Ghost FFN");
        let cpu_layer_elapsed = cpu_started.elapsed();
        *runtime
            .metal_q4_experts
            .lock()
            .expect("Metal expert mutex poisoned") = Some(saved_lane);
        assert_eq!(
            metal_layer, cpu_layer,
            "persistent Q4_0 parity lane changed a real layer FFN bit"
        );

        let metal_started = std::time::Instant::now();
        let (metal_text, metal_ids) = runtime
            .generate_greedy("Hello", 2)
            .expect("natural persistent-slot generation");
        let metal_cold_decode_elapsed = metal_started.elapsed();
        // The first pass must allocate/fault and fill up to eight slots in every
        // layer. Repeat the identical decode before removing the lane so the
        // receipt separates that one-time cost from steady-state slot hits.
        let metal_started = std::time::Instant::now();
        let (metal_warm_text, metal_warm_ids) = runtime
            .generate_greedy("Hello", 2)
            .expect("warm persistent-slot generation");
        let metal_warm_decode_elapsed = metal_started.elapsed();
        assert_eq!(metal_warm_ids, metal_ids, "warm Metal ids changed");
        assert_eq!(metal_warm_text, metal_text, "warm Metal text changed");
        let saved_lane = runtime
            .metal_q4_experts
            .lock()
            .expect("Metal expert mutex poisoned")
            .take()
            .expect("persistent Metal expert slots remained active");
        let cpu_started = std::time::Instant::now();
        let (cpu_text, cpu_ids) = runtime
            .generate_greedy("Hello", 2)
            .expect("natural CPU Ghost generation");
        let cpu_decode_elapsed = cpu_started.elapsed();
        *runtime
            .metal_q4_experts
            .lock()
            .expect("Metal expert mutex poisoned") = Some(saved_lane);

        eprintln!(
            "Ghost Q4 slots real parity: layer Metal={:.3}s CPU={:.3}s; two-token Metal-cold={:.3}s Metal-warm={:.3}s CPU={:.3}s ids={metal_ids:?}",
            metal_layer_elapsed.as_secs_f64(),
            cpu_layer_elapsed.as_secs_f64(),
            metal_cold_decode_elapsed.as_secs_f64(),
            metal_warm_decode_elapsed.as_secs_f64(),
            cpu_decode_elapsed.as_secs_f64(),
        );
        assert_eq!(metal_ids, cpu_ids, "persistent slots changed greedy ids");
        assert_eq!(
            metal_text, cpu_text,
            "persistent slots changed decoded text"
        );
    }

    /// Opt-in real-model parity gate for the layer-major Ghost prefill. The
    /// normal test suite has no 26B fixture, so this skips unless both paths are
    /// supplied. It compares every prompt-position logit bit, not just argmax,
    /// which also proves the expert-major compute schedule restores each row's
    /// original route-rank accumulation order.
    #[test]
    fn ghost_chunk_prefill_matches_scalar_step_bitwise_when_fixture_is_configured() {
        let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost chunk parity: set CAMELID_GEMMA4_GGUF");
            return;
        };
        let Some(cghost) =
            std::env::var_os("CAMELID_GEMMA4_GHOST_CGHOST").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost chunk parity: set CAMELID_GEMMA4_GHOST_CGHOST");
            return;
        };
        let runtime = Gemma4Runtime::load_ghost_moe(&model, &cghost, 1024, false)
            .expect("load Ghost-MoE fixture");
        assert!(runtime.supports_chunk_forward());
        let tokens = runtime
            .tokenizer
            .encode("Hello from chunked Ghost MoE.", true, true)
            .expect("tokenize parity prompt");

        let (mut scalar_k, mut scalar_v) = runtime.empty_kv_caches();
        let mut scalar_rows = Vec::with_capacity(tokens.len());
        for (pos, &token) in tokens.iter().enumerate() {
            scalar_rows.push(
                runtime
                    .step(token, pos, &mut scalar_k, &mut scalar_v)
                    .expect("scalar prompt step"),
            );
        }

        let (mut chunk_k, mut chunk_v) = runtime.empty_kv_caches();
        let chunk_rows = runtime
            .step_chunk(&tokens, 0, &mut chunk_k, &mut chunk_v)
            .expect("layer-major prompt chunk");
        assert_eq!(scalar_rows.len(), chunk_rows.len());
        for (position, (scalar, chunk)) in scalar_rows.iter().zip(&chunk_rows).enumerate() {
            assert_eq!(scalar.len(), chunk.len());
            for (token_id, (&a, &b)) in scalar.iter().zip(chunk).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "Ghost chunk diverged at position {position}, logit {token_id}"
                );
            }
        }

        let (mut final_k, mut final_v) = runtime.empty_kv_caches();
        let final_only = runtime
            .step_chunk_with_head(&tokens, 0, &mut final_k, &mut final_v, false)
            .expect("final-head-only prompt chunk");
        assert_eq!(final_only.len(), 1);
        for (token_id, (&a, &b)) in scalar_rows
            .last()
            .unwrap()
            .iter()
            .zip(&final_only[0])
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "final-head-only chunk diverged at logit {token_id}"
            );
        }

        let assert_kv_bits = |label: &str,
                              expected_k: &Gemma4KvCache,
                              expected_v: &Gemma4KvCache,
                              actual_k: &Gemma4KvCache,
                              actual_v: &Gemma4KvCache| {
            assert_eq!(expected_k.len(), actual_k.len(), "{label} K layers");
            assert_eq!(expected_v.len(), actual_v.len(), "{label} V layers");
            for layer in 0..expected_k.len() {
                assert_eq!(
                    expected_k[layer].len(),
                    actual_k[layer].len(),
                    "{label} K positions at layer {layer}"
                );
                assert_eq!(
                    expected_v[layer].len(),
                    actual_v[layer].len(),
                    "{label} V positions at layer {layer}"
                );
                for position in 0..expected_k[layer].len() {
                    for (index, (&expected, &actual)) in expected_k[layer][position]
                        .iter()
                        .zip(&actual_k[layer][position])
                        .enumerate()
                    {
                        assert_eq!(
                            expected.to_bits(),
                            actual.to_bits(),
                            "{label} K layer={layer} position={position} index={index}"
                        );
                    }
                    for (index, (&expected, &actual)) in expected_v[layer][position]
                        .iter()
                        .zip(&actual_v[layer][position])
                        .enumerate()
                    {
                        assert_eq!(
                            expected.to_bits(),
                            actual.to_bits(),
                            "{label} V layer={layer} position={position} index={index}"
                        );
                    }
                }
            }
        };
        assert_kv_bits("chunk-vs-scalar", &scalar_k, &scalar_v, &chunk_k, &chunk_v);
        assert_kv_bits(
            "final-only-vs-scalar",
            &scalar_k,
            &scalar_v,
            &final_k,
            &final_v,
        );
    }
}
