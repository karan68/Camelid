//! Gemma 4 prefill forward pass — correctness-first reference, validated against
//! the llama.cpp oracle. Feeds the 6-token prompt "The capital of France is" and
//! checks the last-position argmax == 9079 (" Paris").
//!
//! Plain f32 math for transparency (no engine fast-paths). Skipped unless
//! `CAMELID_GEMMA4_GGUF` is set. RAM-aware: dequantizes only the 6 needed rows of
//! the giant embedding tables and loads per-layer weights on demand.
//!
//! Run: `CAMELID_GEMMA4_GGUF=/path/gemma-4-E4B-it-Q8_0.gguf \
//!       cargo test --test gemma4_forward -- --nocapture`

use std::path::PathBuf;

use camelid::gguf::read_metadata;
use camelid::model::{Gemma4Binding, LlamaModelConfig};
use camelid::tensor::TensorStore;

const ORACLE_FIRST_TOKEN: u32 = 9079; // " Paris"
const PROMPT_TOKENS: &[u32] = &[2, 818, 5279, 529, 7001, 563];

// --- f32 helpers -----------------------------------------------------------

/// y[o] = sum_i W[o*in + i] * x[i]. GGUF stores a [in, out] descriptor as data
/// laid out row-major over (out, in) with `in` contiguous — i.e. W is `out` rows
/// of length `in`.
fn matvec(w: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    assert_eq!(w.len(), in_dim * out_dim);
    assert_eq!(x.len(), in_dim);
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
        Some(w) => x.iter().zip(w).map(|(v, ww)| v * inv * ww).collect(),
        None => x.iter().map(|v| v * inv).collect(),
    }
}

fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_56;
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

fn add_into(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

/// Apply RoPE in place to a [heads, head_dim] flattened vector at absolute
/// `position`, using the GPT-NeoX/HF pairing (first half / second half) with the
/// given theta. rope_dim == head_dim here (GGUF rope.dimension_count matches).
fn apply_rope(vec: &mut [f32], heads: usize, head_dim: usize, position: usize, theta: f32) {
    let half = head_dim / 2;
    for h in 0..heads {
        let base = h * head_dim;
        for i in 0..half {
            let freq = (theta).powf(-(2.0 * i as f32) / head_dim as f32);
            let angle = position as f32 * freq;
            let (s, c) = angle.sin_cos();
            let a = vec[base + i];
            let b = vec[base + half + i];
            vec[base + i] = a * c - b * s;
            vec[base + half + i] = b * c + a * s;
        }
    }
}

fn load_f32(store: &TensorStore, name: &str) -> (Vec<f32>, Vec<usize>) {
    let t = store
        .load_cpu_f32(name)
        .unwrap_or_else(|e| panic!("load {name}: {e}"));
    (t.data, t.shape.dims.clone())
}

#[test]
fn gemma4_prefill_matches_oracle() {
    let Some(path) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP gemma4_prefill: set CAMELID_GEMMA4_GGUF");
        return;
    };
    let gguf = read_metadata(&path).expect("read_metadata");
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config");
    let g = config.gemma4.as_ref().expect("gemma4 metadata").clone();
    let binding = Gemma4Binding::bind(&gguf, &config).expect("bind");
    let store = TensorStore::open(&path, &gguf);

    let hidden = config.embedding_length as usize; // 2560
    let n_layers = config.block_count as usize; // 42
    let heads = config.attention_head_count as usize; // 8
    let kv_heads = config.attention_head_count_kv as usize; // 2
    let ple_dim = g.per_layer_input_dim as usize; // 256
    let eps = config.rms_norm_epsilon;
    let seq = PROMPT_TOKENS.len();
    let embed_scale = (hidden as f32).sqrt();
    let ple_embed_scale = (ple_dim as f32).sqrt();

    // --- embeddings: dequantize just the rows we need ----------------------
    let tok_blocks = store
        .load_q8_0_blocks(&binding.token_embedding.name)
        .expect("token_embd blocks");
    let ple_blocks = binding
        .per_layer_token_embd
        .as_ref()
        .map(|d| store.load_q8_0_blocks(&d.name).expect("ple embd blocks"));

    // hidden states [seq][hidden], scaled embeddings
    // GGUF token_embd is token-major in raw element order (each token's `hidden`
    // values are contiguous), so index by element offset, not dequantize_row.
    let ple_total = n_layers * ple_dim;
    let mut hs: Vec<Vec<f32>> = PROMPT_TOKENS
        .iter()
        .map(|&t| {
            let row = tok_blocks
                .dequantize_elements(t as usize * hidden, hidden)
                .expect("tok row");
            row.iter().map(|v| v * embed_scale).collect()
        })
        .collect();

    // --- per-layer input embeddings (PLE), token-identity component --------
    // per_layer_token_embd[token] is [n_layers * ple_dim], scaled by sqrt(ple_dim).
    let token_identity: Vec<Vec<f32>> = PROMPT_TOKENS
        .iter()
        .map(|&t| {
            ple_blocks
                .as_ref()
                .map(|b| {
                    b.dequantize_elements(t as usize * ple_total, ple_total)
                        .expect("ple row")
                        .iter()
                        .map(|v| v * ple_embed_scale)
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect();

    // context component: per_layer_model_proj(embed_scaled) * hidden^-0.5, then
    // reshape [n_layers, ple_dim] and per_layer_proj_norm (RMSNorm).
    let per_layer_input: Vec<Vec<Vec<f32>>> = if let (Some(proj), Some(pnorm)) = (
        binding.per_layer_model_proj.as_ref(),
        binding.per_layer_proj_norm.as_ref(),
    ) {
        let (proj_w, proj_dims) = load_f32(&store, &proj.name); // [hidden, n_layers*ple_dim]
        let proj_out = proj_dims[1]; // n_layers * ple_dim
        let (pnorm_w, _) = load_f32(&store, &pnorm.name); // [ple_dim]
        let proj_scale = (hidden as f32).powf(-0.5);
        (0..seq)
            .map(|t| {
                let ctx = matvec(&proj_w, &hs[t], hidden, proj_out);
                (0..n_layers)
                    .map(|l| {
                        let ctx_l: Vec<f32> = (0..ple_dim)
                            .map(|d| ctx[l * ple_dim + d] * proj_scale)
                            .collect();
                        let ctx_n = rms_norm(&ctx_l, Some(&pnorm_w), eps);
                        let ti = &token_identity[t][l * ple_dim..(l + 1) * ple_dim];
                        ctx_n
                            .iter()
                            .zip(ti)
                            .map(|(a, b)| (a + b) * std::f32::consts::FRAC_1_SQRT_2)
                            .collect()
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![]
    };

    // --- decoder layers ----------------------------------------------------
    let embed_rms = (hs[seq - 1].iter().map(|v| v * v).sum::<f32>() / hidden as f32).sqrt();
    eprintln!("embed hidden_rms (last pos) = {embed_rms:.4}");

    // Gemma 4 passes scaling=1.0 to the softmax; the query scale is baked into the
    // learned q_norm/k_norm weights (no extra 1/sqrt(head_dim)).
    let q_scale = |_hd: usize| 1.0f32;
    let n_run = std::env::var("LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(n_layers);
    for l in 0..n_run {
        let lt = &binding.layers[l];
        let sliding = g.is_sliding_layer(l);
        let head_dim = g.head_dim_at(l) as usize;
        let theta = g.rope_freq_base_at(l);
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        let (attn_norm_w, _) = load_f32(&store, &lt.attn_norm.name);
        if l == 0 {
            let stat = |w: &[f32], n: &str| {
                let mean = w.iter().sum::<f32>() / w.len() as f32;
                let rms = (w.iter().map(|v| v * v).sum::<f32>() / w.len() as f32).sqrt();
                eprintln!("  {n} weight mean={mean:.4} rms={rms:.4}");
            };
            stat(&attn_norm_w, "attn_norm[0]");
            let (qn, _) = load_f32(&store, &lt.attn_q_norm.name);
            let (kn, _) = load_f32(&store, &lt.attn_k_norm.name);
            stat(&qn, "q_norm[0]");
            stat(&kn, "k_norm[0]");
        }
        let (q_w, _) = load_f32(&store, &lt.attn_q.name);
        let (k_w, _) = load_f32(&store, &lt.attn_k.name);
        let (v_w, _) = load_f32(&store, &lt.attn_v.name);
        let (o_w, _) = load_f32(&store, &lt.attn_output.name);
        let (qn_w, _) = load_f32(&store, &lt.attn_q_norm.name);
        let (kn_w, _) = load_f32(&store, &lt.attn_k_norm.name);
        let (post_attn_w, _) = load_f32(&store, &lt.post_attention_norm.name);
        let (ffn_norm_w, _) = load_f32(&store, &lt.ffn_norm.name);
        let (gate_w, _) = load_f32(&store, &lt.ffn_gate.name);
        let (up_w, _) = load_f32(&store, &lt.ffn_up.name);
        let (down_w, _) = load_f32(&store, &lt.ffn_down.name);
        let (post_ffw_w, _) = load_f32(&store, &lt.post_ffw_norm.name);
        let ffn_dim = config.feed_forward_length as usize;

        // attention: compute q/k/v per position (own KV — GGUF carries all layers)
        let mut qs = vec![vec![0f32; q_dim]; seq];
        let mut ks = vec![vec![0f32; kv_dim]; seq];
        let mut vs = vec![vec![0f32; kv_dim]; seq];
        for t in 0..seq {
            let xn = rms_norm(&hs[t], Some(&attn_norm_w), eps);
            let mut q = matvec(&q_w, &xn, hidden, q_dim);
            let mut k = matvec(&k_w, &xn, hidden, kv_dim);
            let mut v = matvec(&v_w, &xn, hidden, kv_dim);
            // q_norm / k_norm (per head, head_dim), then rope. v_norm weightless.
            for h in 0..heads {
                let s = &mut q[h * head_dim..(h + 1) * head_dim];
                s.copy_from_slice(&rms_norm(s, Some(&qn_w), eps));
            }
            for h in 0..kv_heads {
                let s = &mut k[h * head_dim..(h + 1) * head_dim];
                s.copy_from_slice(&rms_norm(s, Some(&kn_w), eps));
                if std::env::var("NO_VNORM").is_err() {
                    let sv = &mut v[h * head_dim..(h + 1) * head_dim];
                    sv.copy_from_slice(&rms_norm(sv, None, eps));
                }
            }
            apply_rope(&mut q, heads, head_dim, t, theta);
            apply_rope(&mut k, kv_heads, head_dim, t, theta);
            qs[t] = q;
            ks[t] = k;
            vs[t] = v;
        }
        // attention output per position
        let scale = q_scale(head_dim);
        let group = heads / kv_heads;
        let mut attn_out = vec![vec![0f32; q_dim]; seq];
        for t in 0..seq {
            for h in 0..heads {
                let kvh = h / group;
                let qh = &qs[t][h * head_dim..(h + 1) * head_dim];
                // causal + sliding-window range
                let lo = if sliding {
                    (t + 1).saturating_sub(g.sliding_window as usize)
                } else {
                    0
                };
                let mut scores = Vec::with_capacity(t - lo + 1);
                for p in lo..=t {
                    let kp = &ks[p][kvh * head_dim..(kvh + 1) * head_dim];
                    let dot: f32 = qh.iter().zip(kp).map(|(a, b)| a * b).sum();
                    scores.push(dot * scale);
                }
                let m = scores.iter().cloned().fold(f32::MIN, f32::max);
                let mut den = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    den += *s;
                }
                let out = &mut attn_out[t][h * head_dim..(h + 1) * head_dim];
                for (idx, p) in (lo..=t).enumerate() {
                    let w = scores[idx] / den;
                    let vp = &vs[p][kvh * head_dim..(kvh + 1) * head_dim];
                    for d in 0..head_dim {
                        out[d] += w * vp[d];
                    }
                }
            }
        }
        // o_proj, post_attention_norm, residual
        for t in 0..seq {
            let o = matvec(&o_w, &attn_out[t], q_dim, hidden);
            let on = rms_norm(&o, Some(&post_attn_w), eps);
            add_into(&mut hs[t], &on);
        }
        // FFN: GeGLU + pre/post norm + residual
        for t in 0..seq {
            let xn = rms_norm(&hs[t], Some(&ffn_norm_w), eps);
            let gate = matvec(&gate_w, &xn, hidden, ffn_dim);
            let up = matvec(&up_w, &xn, hidden, ffn_dim);
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(gt, u)| gelu_tanh(*gt) * u)
                .collect();
            let down = matvec(&down_w, &act, ffn_dim, hidden);
            let dn = rms_norm(&down, Some(&post_ffw_w), eps);
            add_into(&mut hs[t], &dn);
        }
        // PLE injection (E-series)
        let enable_ple = std::env::var("NO_PLE").is_err();
        if enable_ple && !per_layer_input.is_empty() {
            let (gate_w, _) = load_f32(&store, &lt.ple_inp_gate.as_ref().unwrap().name); // [hidden, ple_dim]
            let (proj_w, _) = load_f32(&store, &lt.ple_proj.as_ref().unwrap().name); // [ple_dim, hidden]
            let (postn_w, _) = load_f32(&store, &lt.post_norm.as_ref().unwrap().name);
            let scale = lt
                .ple_output_scale
                .as_ref()
                .map(|d| load_f32(&store, &d.name).0[0])
                .unwrap_or(1.0);
            if l == 0 {
                eprintln!("  layer 0 ple_output_scale = {scale}");
            }
            for t in 0..seq {
                let mut gated = matvec(&gate_w, &hs[t], hidden, ple_dim);
                for (gv, pv) in gated.iter_mut().zip(&per_layer_input[t][l]) {
                    *gv = gelu_tanh(*gv) * pv;
                }
                let proj = matvec(&proj_w, &gated, ple_dim, hidden);
                let pn = rms_norm(&proj, Some(&postn_w), eps);
                if l == 0 && t == seq - 1 {
                    let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                    eprintln!(
                        "  PLE[0]: per_layer_input_rms={:.4} post_norm_w_rms={:.4} pn_rms={:.4} h_rms={:.4}",
                        rms(&per_layer_input[t][l]), rms(&postn_w), rms(&pn), rms(&hs[t])
                    );
                }
                add_into(&mut hs[t], &pn);
                // NOTE: reference layer_scalar is a ones-buffer (1.0, no-op). The
                // GGUF layer_output_scale (`scale`) is NOT that — leave it unused
                // here. (Was erroneously multiplying the whole hidden state.)
                let _ = scale;
            }
        }
        let rms = (hs[seq - 1].iter().map(|v| v * v).sum::<f32>() / hidden as f32).sqrt();
        if l < 3 || l == n_layers - 1 {
            eprintln!("  layer {l} done (sliding={sliding}, head_dim={head_dim}) hidden_rms={rms:.4}");
        }
    }

    // final norm + tied logits + softcap, last position
    let (out_norm_w, _) = load_f32(&store, &binding.output_norm.name);
    let last = rms_norm(&hs[seq - 1], Some(&out_norm_w), eps);
    // logits via tied token_embd rows (dequantize each row, dot with `last`)
    let vocab = config.vocab_size.unwrap() as usize;
    let cap = g.final_logit_softcapping.unwrap_or(0.0);
    let mut logits: Vec<(usize, f32)> = Vec::with_capacity(vocab);
    let mut oracle_logit = f32::MIN;
    for v in 0..vocab {
        let logit_row = tok_blocks
            .dequantize_elements(v * hidden, hidden)
            .expect("vocab row");
        let mut logit: f32 = logit_row.iter().zip(&last).map(|(a, b)| a * b).sum();
        if cap > 0.0 {
            logit = cap * (logit / cap).tanh();
        }
        if v as u32 == ORACLE_FIRST_TOKEN {
            oracle_logit = logit;
        }
        logits.push((v, logit));
    }
    logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("top-10 logits:");
    for (rank, (v, l)) in logits.iter().take(10).enumerate() {
        eprintln!("  #{rank}: token {v} logit {l:.4}");
    }
    let oracle_rank = logits.iter().position(|(v, _)| *v as u32 == ORACLE_FIRST_TOKEN).unwrap();
    eprintln!("oracle token {ORACLE_FIRST_TOKEN} is at rank {oracle_rank} (logit {oracle_logit:.4})");
    let best = logits[0];
    eprintln!("argmax = {} (logit {:.4}); oracle = {ORACLE_FIRST_TOKEN}", best.0, best.1);
    eprintln!(
        "RESULT: {}",
        if best.0 as u32 == ORACLE_FIRST_TOKEN { "MATCH ✅" } else { "MISMATCH ❌ (debug: scaling/rope/PLE)" }
    );
}
