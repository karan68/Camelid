//! Gemma 4 inference runtime — loads a gemma4 GGUF and generates text.
//!
//! The forward pass math is the one validated bit-for-bit against llama.cpp in
//! `tests/gemma4_forward.rs` (prompt "The capital of France is" → " Paris...").
//! Weights stay Q8_0 in memory (the model fits in ~8GB; a full f32 materialization
//! would not fit a 16GB box), and matmuls dequantize on the fly via [`q8_matvec`].
//!
//! This is a correctness-first runtime (plain scalar f32 accumulation, re-prefill
//! greedy decode). It exists to make Gemma 4 actually generate end-to-end; the
//! incremental KV cache and the Q8 GEMM fast paths are follow-up optimizations.

use crate::gguf::read_metadata;
use crate::inference::gemma4::{gelu_tanh, soft_cap_in_place};
use crate::model::{Gemma4Binding, Gemma4Metadata, LlamaModelConfig};
use crate::tensor::{Q8_0TensorBlocks, TensorStore};
use crate::tokenizer::Tokenizer;
use crate::{BackendError, Result};
use std::path::Path;

/// y[o] = sum_i dequant(W[o*in + i]) * x[i], where the Q8 blocks store the tensor
/// in raw GGUF order (out-major, `in` contiguous). This is the transpose of the
/// `dot_row_f32` row convention, so it is implemented directly here.
fn q8_matvec(w: &Q8_0TensorBlocks, in_dim: usize, out_dim: usize, x: &[f32]) -> Vec<f32> {
    const BV: usize = 32;
    debug_assert_eq!(x.len(), in_dim);
    (0..out_dim)
        .map(|o| {
            let base = o * in_dim;
            let mut sum = 0.0f32;
            for (i, &xi) in x.iter().enumerate() {
                let idx = base + i;
                let block = &w.blocks[idx / BV];
                sum += block.scale * f32::from(block.quants[idx % BV]) * xi;
            }
            sum
        })
        .collect()
}

fn f32_matvec(w: &[f32], in_dim: usize, out_dim: usize, x: &[f32]) -> Vec<f32> {
    (0..out_dim)
        .map(|o| {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            row.iter().zip(x).map(|(a, b)| a * b).sum()
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
            first_kv_shared: config.block_count as usize - g.num_kv_shared_layers as usize,
            layers,
            config,
            g,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Run the full forward over `tokens` and return the last position's logits.
    fn forward_logits(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.config.embedding_length as usize;
        let heads = self.config.attention_head_count as usize;
        let kv_heads = self.config.attention_head_count_kv as usize;
        let ffn_dim = self.config.feed_forward_length as usize;
        let ple_dim = self.g.per_layer_input_dim as usize;
        let eps = self.config.rms_norm_epsilon;
        let seq = tokens.len();
        let embed_scale = (hidden as f32).sqrt();
        let ple_embed_scale = (ple_dim as f32).sqrt();
        let n_layers = self.layers.len();
        let ple_total = n_layers * ple_dim;

        // embeddings (scaled)
        let mut hs: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&t| {
                let row = self.token_embd.dequantize_elements(t as usize * hidden, hidden)?;
                Ok(row.iter().map(|v| v * embed_scale).collect())
            })
            .collect::<Result<_>>()?;

        // per-layer input (PLE), computed once from the scaled embeddings
        let per_layer_input: Vec<Vec<Vec<f32>>> = if let (Some(ple_te), Some(proj), Some(pnorm)) = (
            self.per_layer_token_embd.as_ref(),
            self.per_layer_model_proj.as_ref(),
            self.per_layer_proj_norm.as_ref(),
        ) {
            let proj_scale = (hidden as f32).powf(-0.5);
            (0..seq)
                .map(|t| {
                    let ti = ple_te.dequantize_elements(tokens[t] as usize * ple_total, ple_total)?;
                    let ctx = f32_matvec(proj, hidden, ple_total, &hs[t]);
                    Ok((0..n_layers)
                        .map(|l| {
                            let ctx_l: Vec<f32> =
                                (0..ple_dim).map(|d| ctx[l * ple_dim + d] * proj_scale).collect();
                            let ctx_n = rms_norm(&ctx_l, Some(pnorm), eps);
                            (0..ple_dim)
                                .map(|d| {
                                    (ctx_n[d] + ti[l * ple_dim + d] * ple_embed_scale)
                                        * std::f32::consts::FRAC_1_SQRT_2
                                })
                                .collect()
                        })
                        .collect())
                })
                .collect::<Result<_>>()?
        } else {
            Vec::new()
        };

        let mut shared_kv: [Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>; 2] = [None, None];
        for l in 0..n_layers {
            let lw = &self.layers[l];
            let sliding = self.g.is_sliding_layer(l);
            let head_dim = self.g.head_dim_at(l) as usize;
            let theta = self.g.rope_freq_base_at(l);
            let q_dim = heads * head_dim;
            let kv_dim = kv_heads * head_dim;
            let compute_kv = l < self.first_kv_shared;

            let mut qs = vec![Vec::new(); seq];
            let mut ks = vec![Vec::new(); seq];
            let mut vs = vec![Vec::new(); seq];
            for t in 0..seq {
                let xn = rms_norm(&hs[t], Some(&lw.attn_norm), eps);
                let mut q = q8_matvec(&lw.attn_q, hidden, q_dim, &xn);
                for h in 0..heads {
                    let s = &mut q[h * head_dim..(h + 1) * head_dim];
                    s.copy_from_slice(&rms_norm(s, Some(&lw.q_norm), eps));
                }
                apply_rope(&mut q, heads, head_dim, t, theta);
                qs[t] = q;
                if compute_kv {
                    let mut k = q8_matvec(&lw.attn_k, hidden, kv_dim, &xn);
                    let mut v = q8_matvec(&lw.attn_v, hidden, kv_dim, &xn);
                    for h in 0..kv_heads {
                        let s = &mut k[h * head_dim..(h + 1) * head_dim];
                        s.copy_from_slice(&rms_norm(s, Some(&lw.k_norm), eps));
                        let sv = &mut v[h * head_dim..(h + 1) * head_dim];
                        sv.copy_from_slice(&rms_norm(sv, None, eps));
                    }
                    apply_rope(&mut k, kv_heads, head_dim, t, theta);
                    ks[t] = k;
                    vs[t] = v;
                }
            }
            // cross-layer KV sharing
            let slot = usize::from(!sliding);
            if compute_kv {
                shared_kv[slot] = Some((ks.clone(), vs.clone()));
            } else {
                let (sk, sv) = shared_kv[slot].as_ref().expect("shared kv present");
                ks = sk.clone();
                vs = sv.clone();
            }

            // attention (causal + sliding window), then o_proj + post-norm + residual
            let group = heads / kv_heads;
            let win = self.g.sliding_window as usize;
            for t in 0..seq {
                let mut attn = vec![0f32; q_dim];
                for h in 0..heads {
                    let kvh = h / group;
                    let qh = &qs[t][h * head_dim..(h + 1) * head_dim];
                    let lo = if sliding { (t + 1).saturating_sub(win) } else { 0 };
                    let mut scores: Vec<f32> = (lo..=t)
                        .map(|p| {
                            let kp = &ks[p][kvh * head_dim..(kvh + 1) * head_dim];
                            qh.iter().zip(kp).map(|(a, b)| a * b).sum()
                        })
                        .collect();
                    let m = scores.iter().cloned().fold(f32::MIN, f32::max);
                    let mut den = 0f32;
                    for s in &mut scores {
                        *s = (*s - m).exp();
                        den += *s;
                    }
                    let out = &mut attn[h * head_dim..(h + 1) * head_dim];
                    for (idx, p) in (lo..=t).enumerate() {
                        let w = scores[idx] / den;
                        let vp = &vs[p][kvh * head_dim..(kvh + 1) * head_dim];
                        for d in 0..head_dim {
                            out[d] += w * vp[d];
                        }
                    }
                }
                let o = q8_matvec(&lw.attn_output, q_dim, hidden, &attn);
                let on = rms_norm(&o, Some(&lw.post_attn_norm), eps);
                for (a, b) in hs[t].iter_mut().zip(&on) {
                    *a += b;
                }
            }
            // FFN (GeGLU) + post-norm + residual
            for t in 0..seq {
                let xn = rms_norm(&hs[t], Some(&lw.ffn_norm), eps);
                let gate = q8_matvec(&lw.ffn_gate, hidden, ffn_dim, &xn);
                let up = q8_matvec(&lw.ffn_up, hidden, ffn_dim, &xn);
                let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| gelu_tanh(*g) * u).collect();
                let down = q8_matvec(&lw.ffn_down, ffn_dim, hidden, &act);
                let dn = rms_norm(&down, Some(&lw.post_ffw_norm), eps);
                for (a, b) in hs[t].iter_mut().zip(&dn) {
                    *a += b;
                }
            }
            // PLE injection + layer_output_scale
            if let (Some(ig), Some(pj), Some(pn)) =
                (lw.ple_inp_gate.as_ref(), lw.ple_proj.as_ref(), lw.post_norm.as_ref())
            {
                for t in 0..seq {
                    let mut gated = f32_matvec(ig, hidden, ple_dim, &hs[t]);
                    for (gv, pv) in gated.iter_mut().zip(&per_layer_input[t][l]) {
                        *gv = gelu_tanh(*gv) * pv;
                    }
                    let proj = f32_matvec(pj, ple_dim, hidden, &gated);
                    let pnv = rms_norm(&proj, Some(pn), eps);
                    for (a, b) in hs[t].iter_mut().zip(&pnv) {
                        *a += b;
                    }
                    for v in hs[t].iter_mut() {
                        *v *= lw.ple_output_scale;
                    }
                }
            }
        }

        // final norm + tied logits + softcap (last position only)
        let last = rms_norm(&hs[seq - 1], Some(&self.output_norm), eps);
        let vocab = self.config.vocab_size.unwrap() as usize;
        let mut logits = Vec::with_capacity(vocab);
        for v in 0..vocab {
            let row = self.token_embd.dequantize_elements(v * hidden, hidden)?;
            logits.push(row.iter().zip(&last).map(|(a, b)| a * b).sum::<f32>());
        }
        if let Some(cap) = self.g.final_logit_softcapping {
            soft_cap_in_place(&mut logits, cap);
        }
        Ok(logits)
    }

    /// Greedily generate up to `max_new` tokens from `prompt` (text), stopping at
    /// an end-of-turn token. Returns the decoded text. Correctness-first
    /// (re-prefill per step); slow but matches llama.cpp.
    pub fn generate_greedy(&self, prompt: &str, max_new: usize) -> Result<String> {
        let mut tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot = self.tokenizer.encode("<end_of_turn>", false, true).ok();
        let eos_ids: Vec<u32> = eot.into_iter().flatten().collect();
        let mut generated = Vec::new();
        for _ in 0..max_new {
            let logits = self.forward_logits(&tokens)?;
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            if eos_ids.contains(&next) {
                break;
            }
            generated.push(next);
            tokens.push(next);
        }
        self.tokenizer.decode(&generated, true)
    }
}
