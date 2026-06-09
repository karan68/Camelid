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

use crate::gguf::read_metadata;
use crate::inference::gemma4::{gelu_tanh, soft_cap_in_place};
use crate::model::{Gemma4Binding, Gemma4Metadata, LlamaModelConfig};
use crate::tensor::{Q8_0TensorBlocks, TensorStore};
use crate::tokenizer::Tokenizer;
use crate::{BackendError, Result};
use rayon::prelude::*;
use std::path::Path;

/// y[o] = sum_i dequant(W[o*in + i]) * x[i]. Q8 blocks store the tensor in raw
/// GGUF order (out-major, `in` contiguous); rows are block-aligned (in % 32 == 0),
/// so we walk blocks: one scale multiply per 32 quants, contiguous + vectorizable.
fn q8_matvec(w: &Q8_0TensorBlocks, in_dim: usize, out_dim: usize, x: &[f32]) -> Vec<f32> {
    const BV: usize = 32;
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(in_dim % BV, 0, "q8_matvec assumes block-aligned rows");
    let blocks_per_row = in_dim / BV;
    (0..out_dim)
        .into_par_iter()
        .map(|o| {
            let row_blocks = &w.blocks[o * blocks_per_row..(o + 1) * blocks_per_row];
            let mut sum = 0.0f32;
            for (b, block) in row_blocks.iter().enumerate() {
                let xb = &x[b * BV..b * BV + BV];
                let mut bsum = 0.0f32;
                for j in 0..BV {
                    bsum += f32::from(block.quants[j]) * xb[j];
                }
                sum += block.scale * bsum;
            }
            sum
        })
        .collect()
}

fn f32_matvec(w: &[f32], in_dim: usize, out_dim: usize, x: &[f32]) -> Vec<f32> {
    (0..out_dim)
        .into_par_iter()
        .map(|o| w[o * in_dim..(o + 1) * in_dim].iter().zip(x).map(|(a, b)| a * b).sum())
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
    attn_q: Q8_0TensorBlocks,
    attn_k: Q8_0TensorBlocks,
    attn_v: Q8_0TensorBlocks,
    attn_output: Q8_0TensorBlocks,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: Q8_0TensorBlocks,
    ffn_up: Q8_0TensorBlocks,
    ffn_down: Q8_0TensorBlocks,
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
    token_embd: Q8_0TensorBlocks,
    per_layer_token_embd: Option<Q8_0TensorBlocks>,
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
        let g = config
            .gemma4
            .clone()
            .ok_or_else(|| BackendError::UnsupportedModelArchitecture("not a gemma4 model".into()))?;
        let binding = Gemma4Binding::bind(&gguf, &config)?;
        let store = TensorStore::open(path, &gguf);
        let tokenizer = Tokenizer::from_gguf(&gguf)?;

        let q8 = |name: &str| store.load_q8_0_blocks(name);
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
            last_sliding_layer: (0..first_kv_shared).rev().find(|&l| g.is_sliding_layer(l)).unwrap_or(0),
            last_full_layer: (0..first_kv_shared).rev().find(|&l| !g.is_sliding_layer(l)).unwrap_or(0),
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
    fn step(&self, token: u32, pos: usize, kc: &mut [Vec<Vec<f32>>], vc: &mut [Vec<Vec<f32>>]) -> Result<Vec<f32>> {
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
                    let ctx_l: Vec<f32> = (0..ple_dim).map(|d| ctx[l * ple_dim + d] * proj_scale).collect();
                    let ctx_n = rms_norm(&ctx_l, Some(pn), eps);
                    (0..ple_dim)
                        .map(|d| (ctx_n[d] + ti[l * ple_dim + d] * ple_embed_scale) * std::f32::consts::FRAC_1_SQRT_2)
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
            let mut q = q8_matvec(&lw.attn_q, hidden, q_dim, &xn);
            for hh in 0..heads {
                let s = &mut q[hh * head_dim..(hh + 1) * head_dim];
                s.copy_from_slice(&rms_norm(s, Some(&lw.q_norm), eps));
            }
            apply_rope(&mut q, heads, head_dim, pos, theta);

            if l < self.first_kv_shared {
                let mut k = q8_matvec(&lw.attn_k, hidden, kv_dim, &xn);
                let mut v = q8_matvec(&lw.attn_v, hidden, kv_dim, &xn);
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
            let lo = if sliding { (pos + 1).saturating_sub(win) } else { 0 };
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
            let o = q8_matvec(&lw.attn_output, q_dim, hidden, &attn);
            let on = rms_norm(&o, Some(&lw.post_attn_norm), eps);
            for (a, b) in h.iter_mut().zip(&on) {
                *a += b;
            }
            let xn = rms_norm(&h, Some(&lw.ffn_norm), eps);
            let gate = q8_matvec(&lw.ffn_gate, hidden, ffn_dim, &xn);
            let up = q8_matvec(&lw.ffn_up, hidden, ffn_dim, &xn);
            let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| gelu_tanh(*g) * u).collect();
            let down = q8_matvec(&lw.ffn_down, ffn_dim, hidden, &act);
            let dn = rms_norm(&down, Some(&lw.post_ffw_norm), eps);
            for (a, b) in h.iter_mut().zip(&dn) {
                *a += b;
            }
            if let (Some(ig), Some(pj), Some(pnn)) =
                (lw.ple_inp_gate.as_ref(), lw.ple_proj.as_ref(), lw.post_norm.as_ref())
            {
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
        let mut logits = q8_matvec(&self.token_embd, hidden, vocab, &last);
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
