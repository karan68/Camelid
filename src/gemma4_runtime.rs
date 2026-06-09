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
        if element_count % Q8_VALUES_PER_BLOCK != 0 {
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
    fn matvec_q(&self, out_dim: usize, xq: &[Q8_0Block]) -> Vec<f32> {
        const BB: usize = Q8_WIRE_BYTES_PER_BLOCK;
        let row_bytes = xq.len() * BB;
        let bytes = self.bytes();
        (0..out_dim)
            .into_par_iter()
            .map(|o| q8_0_wire_row_dot(&bytes[o * row_bytes..(o + 1) * row_bytes], xq))
            .collect()
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

fn apply_rope(vec: &mut [f32], heads: usize, head_dim: usize, position: usize, theta: f32) {
    let half = head_dim / 2;
    for h in 0..heads {
        let base = h * head_dim;
        for i in 0..half {
            let freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
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
    attn_v: WireQ8,
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

/// A loaded Gemma 4 model ready to generate.
pub struct Gemma4Runtime {
    config: LlamaModelConfig,
    g: Gemma4Metadata,
    tokenizer: Tokenizer,
    layers: Vec<LayerWeights>,
    token_embd: WireQ8,
    per_layer_token_embd: Option<WireQ8>,
    per_layer_model_proj: Option<Vec<f32>>, // BF16 -> f32
    per_layer_proj_norm: Option<Vec<f32>>,
    output_norm: Vec<f32>,
    first_kv_shared: usize,
    last_sliding_layer: usize,
    last_full_layer: usize,
}

impl Gemma4Runtime {
    pub fn load(path: &Path) -> Result<Self> {
        let gguf = read_metadata(path)?;
        let config = LlamaModelConfig::from_gguf(&gguf)?;
        let g = config.gemma4.clone().ok_or_else(|| {
            BackendError::UnsupportedModelArchitecture("not a gemma4 model".into())
        })?;
        let binding = Gemma4Binding::bind(&gguf, &config)?;
        let store = TensorStore::open(path, &gguf);
        let tokenizer = Tokenizer::from_gguf(&gguf)?;

        // Memory-map the GGUF once. Q8 weights are referenced in place (no eager
        // decode); kick off background readahead so the first generation does not
        // pay the whole cold-fault cost serially.
        let mmap = GgufWireMmap::map(path)?;
        mmap.advise_willneed();
        let q8 = |name: &str| WireQ8::new(&store, &mmap, name);
        let f32t = |name: &str| -> Result<Vec<f32>> { Ok(store.load_cpu_f32(name)?.data) };

        let mut layers = Vec::with_capacity(binding.layers.len());
        for l in &binding.layers {
            layers.push(LayerWeights {
                attn_norm: f32t(&l.attn_norm.name)?,
                attn_q: q8(&l.attn_q.name)?,
                attn_k: q8(&l.attn_k.name)?,
                attn_v: q8(&l.attn_v.name)?,
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
        let hidden = self.config.embedding_length as usize;
        let heads = self.config.attention_head_count as usize;
        let kv_heads = self.config.attention_head_count_kv as usize;
        let ffn_dim = self.config.feed_forward_length as usize;
        let ple_dim = self.g.per_layer_input_dim as usize;
        let eps = self.config.rms_norm_epsilon;
        let n_layers = self.layers.len();
        let ple_total = n_layers * ple_dim;
        let group = heads / kv_heads;
        let win = self.g.sliding_window as usize;

        let mut h: Vec<f32> = self
            .token_embd
            .dequantize_elements(token as usize * hidden, hidden)?
            .iter()
            .map(|v| v * (hidden as f32).sqrt())
            .collect();

        // per-layer input (token-identity + context) for this token: [n_layers][ple_dim]
        let pli: Vec<Vec<f32>> = if let (Some(te), Some(proj), Some(pn)) = (
            self.per_layer_token_embd.as_ref(),
            self.per_layer_model_proj.as_ref(),
            self.per_layer_proj_norm.as_ref(),
        ) {
            let ti = te.dequantize_elements(token as usize * ple_total, ple_total)?;
            let ctx = f32_matvec(proj, hidden, ple_total, &h);
            let proj_scale = (hidden as f32).powf(-0.5);
            let ple_embed_scale = (ple_dim as f32).sqrt();
            (0..n_layers)
                .map(|l| {
                    let ctx_l: Vec<f32> = (0..ple_dim)
                        .map(|d| ctx[l * ple_dim + d] * proj_scale)
                        .collect();
                    let ctx_n = rms_norm(&ctx_l, Some(pn), eps);
                    (0..ple_dim)
                        .map(|d| {
                            (ctx_n[d] + ti[l * ple_dim + d] * ple_embed_scale)
                                * std::f32::consts::FRAC_1_SQRT_2
                        })
                        .collect()
                })
                .collect()
        } else {
            Vec::new()
        };

        for l in 0..n_layers {
            let lw = &self.layers[l];
            let sliding = self.g.is_sliding_layer(l);
            let head_dim = self.g.head_dim_at(l) as usize;
            let theta = self.g.rope_freq_base_at(l);
            let q_dim = heads * head_dim;
            let kv_dim = kv_heads * head_dim;

            let xn = rms_norm(&h, Some(&lw.attn_norm), eps);
            // q/k/v all project the same normed input — quantize it once.
            let xnq = quantize_q8_0_blocks(&xn);
            let mut q = lw.attn_q.matvec_q(q_dim, &xnq);
            for hh in 0..heads {
                let s = &mut q[hh * head_dim..(hh + 1) * head_dim];
                s.copy_from_slice(&rms_norm(s, Some(&lw.q_norm), eps));
            }
            apply_rope(&mut q, heads, head_dim, pos, theta);

            if l < self.first_kv_shared {
                let mut k = lw.attn_k.matvec_q(kv_dim, &xnq);
                let mut v = lw.attn_v.matvec_q(kv_dim, &xnq);
                for hh in 0..kv_heads {
                    let s = &mut k[hh * head_dim..(hh + 1) * head_dim];
                    s.copy_from_slice(&rms_norm(s, Some(&lw.k_norm), eps));
                    let sv = &mut v[hh * head_dim..(hh + 1) * head_dim];
                    sv.copy_from_slice(&rms_norm(sv, None, eps));
                }
                apply_rope(&mut k, kv_heads, head_dim, pos, theta);
                kc[l].push(k);
                vc[l].push(v);
            }
            let src = if l < self.first_kv_shared {
                l
            } else if sliding {
                self.last_sliding_layer
            } else {
                self.last_full_layer
            };
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
                for (gv, pv) in gated.iter_mut().zip(&pli[l]) {
                    *gv = gelu_tanh(*gv) * pv;
                }
                let proj = f32_matvec(pj, ple_dim, hidden, &gated);
                let pnv = rms_norm(&proj, Some(pnn), eps);
                for (a, b) in h.iter_mut().zip(&pnv) {
                    *a += b;
                }
                for v in h.iter_mut() {
                    *v *= lw.ple_output_scale;
                }
            }
        }

        let last = rms_norm(&h, Some(&self.output_norm), eps);
        let vocab = self.config.vocab_size.unwrap() as usize;
        // token_embd is vocab-major (row v = the v-th embedding), so the tied
        // logits are a single block-wise Q8 matvec — far faster than per-row
        // dequantize_elements over the whole 262k vocab.
        let mut logits = self.token_embd.matvec(hidden, vocab, &last);
        if let Some(cap) = self.g.final_logit_softcapping {
            soft_cap_in_place(&mut logits, cap);
        }
        Ok(logits)
    }

    /// Greedily generate up to `max_new` tokens from `prompt`, with an incremental
    /// KV cache (one forward step per token). Returns (decoded continuation, the
    /// generated token ids).
    pub fn generate_greedy(&self, prompt: &str, max_new: usize) -> Result<(String, Vec<u32>)> {
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot: Vec<u32> = self
            .tokenizer
            .encode("<end_of_turn>", false, true)
            .ok()
            .into_iter()
            .flatten()
            .collect();

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
        let text = self.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }

    /// Greedy decode that emits the incremental decoded-text delta after each new
    /// token via `on_delta`. The delta is computed by decoding the cumulative
    /// generated sequence and yielding the newly-appended suffix, which keeps
    /// SentencePiece spacing/multi-byte pieces correct (token-at-a-time decode
    /// would mangle them). Returns the same `(text, ids)` as `generate_greedy`.
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
        let eot: Vec<u32> = self
            .tokenizer
            .encode("<end_of_turn>", false, true)
            .ok()
            .into_iter()
            .flatten()
            .collect();

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
    token_embd: WireQ8,
    per_layer_token_embd: Option<WireQ8>,
    per_layer_model_proj: Option<Vec<f32>>,
    per_layer_proj_norm: Option<Vec<f32>>,
    _mmap: Arc<GgufWireMmap>,
    hidden: usize,
    ple_dim: usize,
    n_layers: usize,
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
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        // The mmap backs token_embd + per_layer_token_embd (CPU gathers); the GPU layer
        // weights are loaded separately as page-aligned WirePages for nocopy residency.
        let mmap = GgufWireMmap::map(path)?;
        let q8 = |name: &str| WireQ8::new(&store, &mmap, name);
        let f32t = |name: &str| -> Result<Vec<f32>> { Ok(store.load_cpu_f32(name)?.data) };

        let hidden = config.embedding_length as usize;
        let ffn_dim = config.feed_forward_length as usize;
        let heads = config.attention_head_count as usize;
        let kv_heads = config.attention_head_count_kv as usize;
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

        let plan = g.layer_plan(n_layers, heads, kv_heads);
        let mut layers = Vec::with_capacity(n_layers);
        let mut ple = Vec::with_capacity(n_layers);
        let mut owns_kv = Vec::with_capacity(n_layers);
        let mut kv_source = Vec::with_capacity(n_layers);
        for (l, lb) in binding.layers.iter().enumerate() {
            let hd = g.head_dim_at(l) as usize;
            let layer = crate::metal::Gemma4ResidentLayer::from_wire_pages(
                f32t(&lb.attn_norm.name)?,
                f32t(&lb.attn_q_norm.name)?,
                f32t(&lb.attn_k_norm.name)?,
                f32t(&lb.post_attention_norm.name)?,
                f32t(&lb.ffn_norm.name)?,
                f32t(&lb.post_ffw_norm.name)?,
                &pages(&lb.attn_q.name)?,
                &pages(&lb.attn_k.name)?,
                &pages(&lb.attn_v.name)?,
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

        Ok(Self {
            model,
            tokenizer,
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
            token_embd,
            g,
            _mmap: mmap,
            hidden,
            ple_dim,
            n_layers,
            eps,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Run one token's forward on the GPU and return the next-token logits.
    fn forward(&self, token: u32, position: usize) -> Result<Vec<f32>> {
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
        // Per-layer PLE input `pli` (same math as Gemma4Runtime::step).
        let pli: Vec<Vec<f32>> = if let (Some(te), Some(proj), Some(pn)) = (
            self.per_layer_token_embd.as_ref(),
            self.per_layer_model_proj.as_ref(),
            self.per_layer_proj_norm.as_ref(),
        ) {
            let ti = te.dequantize_elements(token as usize * ple_total, ple_total)?;
            let ctx = f32_matvec(proj, hidden, ple_total, &h0);
            let proj_scale = (hidden as f32).powf(-0.5);
            let ple_embed_scale = (ple_dim as f32).sqrt();
            (0..self.n_layers)
                .map(|l| {
                    let ctx_l: Vec<f32> = (0..ple_dim)
                        .map(|d| ctx[l * ple_dim + d] * proj_scale)
                        .collect();
                    let ctx_n = rms_norm(&ctx_l, Some(pn), self.eps);
                    (0..ple_dim)
                        .map(|d| {
                            (ctx_n[d] + ti[l * ple_dim + d] * ple_embed_scale)
                                * std::f32::consts::FRAC_1_SQRT_2
                        })
                        .collect()
                })
                .collect()
        } else {
            vec![Vec::new(); self.n_layers]
        };
        // Per-layer RoPE tables (dual θ, per-type head_dim) + sliding window start.
        let win = self.g.sliding_window as usize;
        let inputs: Vec<crate::metal::Gemma4TokenLayerInput> = (0..self.n_layers)
            .map(|l| {
                let hd = self.g.head_dim_at(l) as usize;
                let theta = self.g.rope_freq_base_at(l);
                let half = hd / 2;
                let (mut cos_t, mut sin_t) = (vec![0f32; half], vec![0f32; half]);
                for i in 0..half {
                    let freq = theta.powf(-(2.0 * i as f32) / hd as f32);
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
                    pli: pli[l].clone(),
                    window_start,
                }
            })
            .collect();
        self.model
            .forward_token(&h0, &inputs, position)
            .ok_or_else(|| BackendError::UnsupportedModelArchitecture("gpu forward failed".into()))
    }

    /// Greedy generate up to `max_new` tokens from `prompt` on the GPU.
    pub fn generate_greedy(&self, prompt: &str, max_new: usize) -> Result<(String, Vec<u32>)> {
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot: Vec<u32> = self
            .tokenizer
            .encode("<end_of_turn>", false, true)
            .ok()
            .into_iter()
            .flatten()
            .collect();
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
        let text = self.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }
}
