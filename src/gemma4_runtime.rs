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
use crate::inference::{q8_0_wire_row_dot, quantize_q8_0_blocks};
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

/// A Q8_0 weight read straight from the memory-mapped GGUF — no eager decode and
/// no second resident copy. The mmap pages fault in on first touch (during the
/// first generation) and stay in the OS page cache after, so `load()` is ~instant
/// instead of spending ~240s materializing 8GB of `Q8_0Block` structs up front.
/// Dequant happens inline in the matmul, exactly where it happened before — only
/// the f16 scale is now decoded per block per pass (negligible next to the 32
/// mul-adds it scales).
struct WireQ8 {
    mmap: Arc<GgufWireMmap>,
    offset: u64,
    element_count: usize,
}

impl WireQ8 {
    fn new(store: &TensorStore, mmap: &Arc<GgufWireMmap>, name: &str) -> Result<Self> {
        let desc = store.descriptor(name)?;
        if desc.tensor_type != GgufTensorType::Q8_0 {
            return Err(BackendError::UnsupportedTensorType(format!(
                "tensor {name} is {:?}; gemma4 wire load requires Q8_0",
                desc.tensor_type
            )));
        }
        let element_count = desc.dimensions.iter().product::<u64>() as usize;
        if !element_count.is_multiple_of(Q8_VALUES_PER_BLOCK) {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {name} element count {element_count} is not block-aligned"
            )));
        }
        let byte_len = element_count / Q8_VALUES_PER_BLOCK * Q8_WIRE_BYTES_PER_BLOCK;
        if desc.n_bytes as usize != byte_len {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {name} q8_0 byte size {} != expected {byte_len}",
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
        })
    }

    /// The tensor's full wire-byte slice. Bounds were validated in `new`.
    #[inline]
    fn bytes(&self) -> &[u8] {
        let byte_len = self.element_count / Q8_VALUES_PER_BLOCK * Q8_WIRE_BYTES_PER_BLOCK;
        self.mmap
            .bytes(self.offset, byte_len)
            .expect("wire q8 range validated at load")
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
            in_dim % Q8_VALUES_PER_BLOCK,
            0,
            "matvec assumes block-aligned rows"
        );
        self.matvec_q(out_dim, &quantize_q8_0_blocks(x))
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
        const BB: usize = Q8_WIRE_BYTES_PER_BLOCK;
        const ROW_CHUNK: usize = 64;
        let row_bytes = xq.len() * BB;
        let bytes = self.bytes();
        let mut out = vec![0f32; out_dim];
        out.par_chunks_mut(ROW_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let base = chunk_idx * ROW_CHUNK;
                for (i, d) in dst.iter_mut().enumerate() {
                    let o = base + i;
                    *d = q8_0_wire_row_dot(&bytes[o * row_bytes..(o + 1) * row_bytes], xq);
                }
            });
        out
    }

    /// Dequantize a contiguous element range [start, start+len) — used for
    /// row-major embedding lookups into vocab-major Q8 tables.
    fn dequantize_elements(&self, start: usize, len: usize) -> Result<Vec<f32>> {
        const BV: usize = Q8_VALUES_PER_BLOCK;
        const BB: usize = Q8_WIRE_BYTES_PER_BLOCK;
        let end = start.checked_add(len).ok_or_else(|| {
            BackendError::InvalidTensorData("q8_0 dequant range overflows usize".into())
        })?;
        if end > self.element_count {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "q8_0 dequant range {start}..{end} exceeds element count {}",
                self.element_count
            )));
        }
        let bytes = self.bytes();
        let mut out = Vec::with_capacity(len);
        for e in start..end {
            let block = e / BV;
            let within = e % BV;
            let scale = Self::block_scale(bytes, block);
            let q = bytes[block * BB + 2 + within] as i8;
            out.push(scale * q as f32);
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
    attn_q: WireQ8,
    attn_k: WireQ8,
    attn_v: Option<WireQ8>, // None on V-less layers (V = K projection)
    attn_output: WireQ8,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: WireQ8,
    ffn_up: WireQ8,
    ffn_down: WireQ8,
    post_ffw_norm: Vec<f32>,
    // PLE (E-series); inp_gate/proj are small F32 matrices in the GGUF.
    post_norm: Option<Vec<f32>>,
    ple_inp_gate: Option<Vec<f32>>,
    ple_proj: Option<Vec<f32>>,
    ple_output_scale: f32,
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
    token_embd: WireQ8,
    per_layer_token_embd: Option<WireQ8>,
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
        // pay the whole cold-fault cost serially.
        let mmap = GgufWireMmap::map(path)?;
        mmap.advise_willneed();
        let q8 = |name: &str| WireQ8::new(&store, &mmap, name);
        let f32t = |name: &str| -> Result<Vec<f32>> { Ok(store.load_cpu_f32(name)?.data) };

        let mut layers = Vec::with_capacity(range.len());
        for l in &binding.layers[range.clone()] {
            layers.push(LayerWeights {
                attn_norm: f32t(&l.attn_norm.name)?,
                attn_q: q8(&l.attn_q.name)?,
                attn_k: q8(&l.attn_k.name)?,
                attn_v: l.attn_v.as_ref().map(|d| q8(&d.name)).transpose()?,
                attn_output: q8(&l.attn_output.name)?,
                q_norm: f32t(&l.attn_q_norm.name)?,
                k_norm: f32t(&l.attn_k_norm.name)?,
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
                let mut k = lw.attn_k.matvec_q(kv_dim, &xnq);
                // V-less layers (12B full attention) reuse the raw K projection
                // as V — reference: `if v_proj is not present, use Kcur as Vcur`.
                // V then takes the usual weightless norm and never RoPE.
                let mut v = match lw.attn_v.as_ref() {
                    Some(wv) => wv.matvec_q(kv_dim, &xnq),
                    None => k.clone(),
                };
                for hh in 0..kv_heads {
                    let s = &mut k[hh * head_dim..(hh + 1) * head_dim];
                    s.copy_from_slice(&rms_norm(s, Some(&lw.k_norm), eps));
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
            let xn = rms_norm(&h, Some(&lw.ffn_norm), eps);
            // gate and up both project the same normed input — quantize it once.
            let xnq = quantize_q8_0_blocks(&xn);
            let gate = lw.ffn_gate.matvec_q(ffn_dim, &xnq);
            let up = lw.ffn_up.matvec_q(ffn_dim, &xnq);
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| gelu_tanh(*g) * u)
                .collect();
            let down = lw.ffn_down.matvec(ffn_dim, hidden, &act);
            let dn = rms_norm(&down, Some(&lw.post_ffw_norm), eps);
            for (a, b) in h.iter_mut().zip(&dn) {
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
        let eot = gemma4_stop_token_ids(&self.tokenizer);

        let mut logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.step(tok, pos, &mut kc, &mut vc)?;
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
    token_embd: WireQ8,
    per_layer_token_embd: Option<WireQ8>,
    /// GGUF `rope_freqs.weight` factors — applied on FULL attention layers'
    /// cos/sin tables only (the reference's proportional rope).
    rope_factors: Option<Vec<f32>>,
    _mmap: Arc<GgufWireMmap>,
    hidden: usize,
    ple_dim: usize,
    n_layers: usize,
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
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        // The mmap backs token_embd + per_layer_token_embd (file-backed = evictable, so
        // it never forces the anonymous GPU WirePages to swap). GPU layer weights load
        // separately as page-aligned WirePages.
        let mmap = GgufWireMmap::map(path)?;
        let q8 = |name: &str| WireQ8::new(&store, &mmap, name);
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
        let mut owns_kv = Vec::with_capacity(n_layers);
        let mut kv_source = Vec::with_capacity(n_layers);
        for (l, lb) in binding.layers.iter().enumerate() {
            let hd = g.head_dim_at(l) as usize;
            // Per-layer geometry (12B varies kv heads, E2B varies FFN width).
            let kv_heads = plan[l].kv_heads;
            let ffn_dim = g.ffn_length_at(l) as usize;
            let layer = crate::metal::Gemma4ResidentLayer::from_wire_pages(
                f32t(&lb.attn_norm.name)?,
                f32t(&lb.attn_q_norm.name)?,
                f32t(&lb.attn_k_norm.name)?,
                f32t(&lb.post_attention_norm.name)?,
                f32t(&lb.ffn_norm.name)?,
                f32t(&lb.post_ffw_norm.name)?,
                &pages(&lb.attn_q.name)?,
                &pages(&lb.attn_k.name)?,
                &pages(
                    &lb.attn_v
                        .as_ref()
                        .ok_or_else(|| {
                            BackendError::UnsupportedModelArchitecture(format!(
                                "gemma4 layer {l} has no attn_v (V-less full attention, \
                                 12B-class row): the GPU-resident path has no K-as-V \
                                 kernels yet; use the CPU or distributed runtime"
                            ))
                        })?
                        .name,
                )?,
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
            ple.push(match (&lb.ple_inp_gate, &lb.ple_proj, &lb.post_norm) {
                (Some(ig), Some(pj), Some(pn)) => Some(crate::metal::Gemma4ResidentPle {
                    inp_gate: f32t(&ig.name)?,
                    proj: f32t(&pj.name)?,
                    post_norm: f32t(&pn.name)?,
                    output_scale: lb
                        .ple_output_scale
                        .as_ref()
                        .map(|d| f32t(&d.name))
                        .transpose()?
                        .and_then(|v| v.first().copied())
                        .unwrap_or(1.0),
                }),
                _ => None,
            });
            owns_kv.push(plan[l].owns_kv);
            kv_source.push(plan[l].kv_source_layer);
        }

        let token_embd = q8(&binding.token_embedding.name)?;
        let output_norm = f32t(&binding.output_norm.name)?;
        let model = crate::metal::Gemma4ResidentModel::new(
            layers,
            ple,
            owns_kv,
            kv_source,
            token_embd.bytes(),
            output_norm,
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
        let logits = self
            .model
            .forward_token(&h0, &inputs, &ti, position)
            .ok_or_else(|| {
                BackendError::UnsupportedModelArchitecture("gpu forward failed".into())
            })?;
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
