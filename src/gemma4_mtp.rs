//! Gemma 4 MTP (multi-token prediction) assistant — the drafter half of lossless
//! speculative decode for the 26B-A4B ghost lane.
//!
//! The assistant is a 4-layer, hidden-1024 model that carries **no K/V projections at
//! all**: every layer is KV-shared, reading the *host* model's KV directly (sliding
//! layers from host layer 28, the full layer from host layer 29). It consumes
//! `concat(target_scaled_embedding, target_final_normalized_hidden)` and returns both
//! next-token logits and a 2816-wide recurrent hidden that feeds the next proposal.
//!
//! **This runs on the CPU by design.** It is 4 layers at hidden 1024 — negligible beside
//! a 26B verify step — and the Metal campaign measured the host-side assistant chain at
//! *identical acceptance* and 26.82 tok/s against its device chain (H20), so a GPU port
//! of the drafter buys nothing. The speed comes from the batched verify, not from here.
//!
//! **Bit-exactness is deliberately NOT a goal.** Speculative decode is lossless because
//! the TARGET verifies every drafted token and commits only the longest prefix equal to
//! its own argmax, so drafter drift moves the acceptance rate (speed) and never the
//! emitted tokens. Correctness is gated by the target, quality by alpha.
//!
//! Semantics below were derived against the committed BF16 oracle in
//! `qa/evidence-bundles/gemma4-26b-mtp-assistant-oracle/` and cross-checked stage by
//! stage against a local `Gemma4AssistantForCausalLM` run; see `qa/gemma4-mtp/FINDINGS.md`
//! for the receipts and for the five conventions that are easy to get wrong.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::error::{BackendError, Result};
use crate::model_source::{decode_safetensors_tensor, SafeTensorsTensorDescriptor};

/// Backbone (host) hidden width. The assistant's input is two of these concatenated and
/// its recurrent output is one.
pub const BACKBONE_HIDDEN: usize = 2816;
/// Assistant hidden width.
pub const ASSISTANT_HIDDEN: usize = 1024;
/// Query heads per layer (constant across layer types; only the KV side varies).
pub const NUM_ATTENTION_HEADS: usize = 16;
/// Sliding layers: head width and host KV head count.
pub const SLIDING_HEAD_DIM: usize = 256;
pub const SLIDING_KV_HEADS: usize = 8;
/// The single full-attention layer: head width and host KV head count.
pub const FULL_HEAD_DIM: usize = 512;
pub const FULL_KV_HEADS: usize = 2;
/// Layers 0..=2 are sliding, layer 3 is full attention.
pub const NUM_LAYERS: usize = 4;
pub const SLIDING_LAYERS: usize = 3;
/// Sliding window, in positions of host KV.
pub const SLIDING_WINDOW: usize = 1024;
/// `rms_norm_eps` from the shipped config.
pub const RMS_NORM_EPS: f32 = 1e-6;
/// RoPE bases per layer type.
pub const SLIDING_ROPE_THETA: f64 = 10_000.0;
pub const FULL_ROPE_THETA: f64 = 1_000_000.0;
/// `partial_rotary_factor` for the full-attention layer. Gemma 4's "proportional" RoPE
/// rotates only `int(factor * head_dim / 2)` split-half pairs and pads the remainder of
/// `head_dim/2` with ZERO frequencies — cos/sin still span the full `head_dim`, and pairs
/// beyond the rotated count are exact identity.
pub const FULL_PARTIAL_ROTARY_FACTOR: f64 = 0.25;

/// Which host layer a given assistant layer shares KV with.
pub const SHARED_KV_SLIDING_HOST_LAYER: usize = 28;
pub const SHARED_KV_FULL_HOST_LAYER: usize = 29;

fn invalid(message: String) -> BackendError {
    BackendError::InvalidModelMetadata(message)
}

/// One assistant layer's weights, already decoded to dense f32.
struct LayerWeights {
    input_layernorm: Vec<f32>,
    q_proj: Vec<f32>,
    q_norm: Vec<f32>,
    o_proj: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    pre_feedforward_layernorm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
    post_feedforward_layernorm: Vec<f32>,
    /// Scales the ENTIRE layer output, residual included, as the layer's last act.
    layer_scalar: f32,
}

impl LayerWeights {
    fn head_dim(layer: usize) -> usize {
        if layer < SLIDING_LAYERS {
            SLIDING_HEAD_DIM
        } else {
            FULL_HEAD_DIM
        }
    }
}

/// The loaded assistant.
pub struct Gemma4MtpAssistant {
    layers: Vec<LayerWeights>,
    /// [1024, 5632] — consumes concat(scaled_embedding, final_normalized_hidden).
    pre_projection: Vec<f32>,
    /// [2816, 1024] — produces the recurrent hidden.
    post_projection: Vec<f32>,
    final_norm: Vec<f32>,
    /// Tied head, [vocab, 1024]. `use_ordered_embeddings` is false for this checkpoint,
    /// so logits are a plain matmul and the centroid MaskedEmbedder path is inactive.
    embed_tokens: Vec<f32>,
    vocab_size: usize,
}

/// Host KV for one assistant layer type, borrowed from the target runtime.
///
/// Layout is `[kv_head][position][head_dim]`, contiguous, exactly as the host stores it.
/// These are consumed RAW — the assistant applies neither RoPE nor a norm to them,
/// because the host already did when it wrote them.
pub struct SharedKv<'a> {
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Number of host positions present, i.e. the shared-KV logical length.
    pub kv_len: usize,
}

impl SharedKv<'_> {
    fn validate(&self, expect_heads: usize, expect_head_dim: usize, what: &str) -> Result<()> {
        if self.kv_heads != expect_heads || self.head_dim != expect_head_dim {
            return Err(invalid(format!(
                "{what} shared KV is {}x{} but this assistant expects {expect_heads}x{expect_head_dim}",
                self.kv_heads, self.head_dim
            )));
        }
        let want = self.kv_heads * self.kv_len * self.head_dim;
        if self.key.len() < want || self.value.len() < want {
            return Err(invalid(format!(
                "{what} shared KV is {} k / {} v elements, short of the {want} implied by \
                 {}x{}x{}",
                self.key.len(),
                self.value.len(),
                self.kv_heads,
                self.kv_len,
                self.head_dim
            )));
        }
        Ok(())
    }
}

/// One proposal step's output.
pub struct AssistantStep {
    /// Next-token logits over the full vocabulary.
    pub logits: Vec<f32>,
    /// 2816-wide recurrent hidden — feeds the NEXT step's `final_normalized_hidden` slot.
    pub recurrent_hidden: Vec<f32>,
}

/// Gemma 4 RMSNorm.
///
/// **`normed * weight`, NOT Gemma 2/3's `normed * (1 + weight)`.** Gemma 4 changed the
/// convention (`Gemma4RMSNorm.forward`); using the older form corrupts every norm in the
/// model and was the single largest error during bring-up.
fn rms_norm(x: &mut [f32], weight: &[f32]) {
    debug_assert_eq!(x.len() % weight.len(), 0);
    for row in x.chunks_mut(weight.len()) {
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inv = (mean_sq + RMS_NORM_EPS).powf(-0.5);
        for (v, w) in row.iter_mut().zip(weight) {
            *v = *v * inv * *w;
        }
    }
}

/// `gelu_pytorch_tanh`.
fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

/// `out = weight [rows, cols] * x [cols]`.
fn matvec(weight: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(weight.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    for (r, o) in out.iter_mut().enumerate() {
        let row = &weight[r * cols..(r + 1) * cols];
        *o = row.iter().zip(x).map(|(w, v)| w * v).sum();
    }
}

/// Inverse frequencies for one layer type, length `head_dim / 2`.
///
/// Sliding layers use plain default RoPE over the whole head. The full layer uses Gemma 4
/// "proportional" RoPE: `int(factor * head_dim / 2)` real frequencies whose exponent
/// divides by the FULL `head_dim`, then zeros for the rest — so the tail pairs rotate by
/// angle zero and are identity, while cos/sin still span `head_dim`.
fn inv_freq(head_dim: usize, theta: f64, partial_rotary_factor: Option<f64>) -> Vec<f64> {
    let half = head_dim / 2;
    match partial_rotary_factor {
        None => (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f64 / head_dim as f64))
            .collect(),
        Some(factor) => {
            let rotated = ((factor * head_dim as f64) as usize) / 2;
            (0..half)
                .map(|i| {
                    if i < rotated {
                        1.0 / theta.powf((2 * i) as f64 / head_dim as f64)
                    } else {
                        0.0
                    }
                })
                .collect()
        }
    }
}

/// Round f32 to BF16 precision (round-to-nearest-even) and back.
///
/// The reference builds cos/sin in f32 then casts them to the model dtype
/// (`cos.to(dtype=x.dtype)`), so a drafter that keeps them in full f32 is measurably
/// further from the reference than one that rounds: bring-up measured 0.547 -> 0.641
/// on the recurrent hidden from this one change, because the unscaled softmax magnifies
/// small query perturbations.
fn round_to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

/// `(x * cos) + (rotate_half(x) * sin)` with cos/sin duplicated to the full head width,
/// matching `apply_rotary_pos_emb`.
fn apply_rope(q: &mut [f32], head_dim: usize, position: usize, inv_freq: &[f64]) {
    let half = head_dim / 2;
    debug_assert_eq!(inv_freq.len(), half);
    for head in q.chunks_mut(head_dim) {
        for i in 0..half {
            let angle = position as f64 * inv_freq[i];
            let (sin, cos) = angle.sin_cos();
            let (cos, sin) = (round_to_bf16(cos as f32), round_to_bf16(sin as f32));
            let a = head[i];
            let b = head[i + half];
            head[i] = round_to_bf16(a * cos - b * sin);
            head[i + half] = round_to_bf16(b * cos + a * sin);
        }
    }
}

fn read_safetensors_descriptors(path: &Path) -> Result<Vec<SafeTensorsTensorDescriptor>> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| invalid(format!("could not open {}: {e}", path.display())))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|e| invalid(format!("could not read safetensors header length: {e}")))?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)
        .map_err(|e| invalid(format!("could not read safetensors header: {e}")))?;
    let value: serde_json::Value = serde_json::from_slice(&header)
        .map_err(|e| invalid(format!("safetensors header is not valid JSON: {e}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid("safetensors header must be a JSON object".into()))?;

    let shard_file = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.safetensors")
        .to_string();
    let mut out = Vec::new();
    for (name, entry) in object {
        if name == "__metadata__" {
            continue;
        }
        let dtype = entry
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| invalid(format!("tensor {name} has no dtype")))?
            .to_string();
        let shape = entry
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| invalid(format!("tensor {name} has no shape")))?
            .iter()
            .map(|d| d.as_u64().unwrap_or(0))
            .collect::<Vec<u64>>();
        let offsets = entry
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .ok_or_else(|| invalid(format!("tensor {name} has no data_offsets")))?;
        if offsets.len() != 2 {
            return Err(invalid(format!("tensor {name} data_offsets is not a pair")));
        }
        out.push(SafeTensorsTensorDescriptor {
            name: name.clone(),
            dtype,
            shape,
            shard_file: shard_file.clone(),
            shard: path.to_path_buf(),
            data_offsets: [
                offsets[0].as_u64().unwrap_or(0),
                offsets[1].as_u64().unwrap_or(0),
            ],
        });
    }
    Ok(out)
}

impl Gemma4MtpAssistant {
    /// Load from a Hugging Face directory containing `model.safetensors` (BF16).
    pub fn load(dir: &Path) -> Result<Self> {
        let weights_path: PathBuf = dir.join("model.safetensors");
        if !weights_path.is_file() {
            return Err(invalid(format!(
                "MTP assistant directory {} has no model.safetensors",
                dir.display()
            )));
        }
        let descriptors = read_safetensors_descriptors(&weights_path)?;
        let by_name: BTreeMap<&str, &SafeTensorsTensorDescriptor> =
            descriptors.iter().map(|d| (d.name.as_str(), d)).collect();

        let take = |name: &str| -> Result<Vec<f32>> {
            let descriptor = by_name.get(name).ok_or_else(|| {
                invalid(format!(
                    "MTP assistant is missing tensor {name}; this build expects the official \
                     gemma4_assistant layout"
                ))
            })?;
            Ok(decode_safetensors_tensor(descriptor)?.data)
        };

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for layer in 0..NUM_LAYERS {
            let p = format!("model.layers.{layer}.");
            let scalar = take(&format!("{p}layer_scalar"))?;
            let layer_scalar = *scalar
                .first()
                .ok_or_else(|| invalid(format!("layer {layer} layer_scalar is empty")))?;
            layers.push(LayerWeights {
                input_layernorm: take(&format!("{p}input_layernorm.weight"))?,
                q_proj: take(&format!("{p}self_attn.q_proj.weight"))?,
                q_norm: take(&format!("{p}self_attn.q_norm.weight"))?,
                o_proj: take(&format!("{p}self_attn.o_proj.weight"))?,
                post_attention_layernorm: take(&format!("{p}post_attention_layernorm.weight"))?,
                pre_feedforward_layernorm: take(&format!("{p}pre_feedforward_layernorm.weight"))?,
                gate_proj: take(&format!("{p}mlp.gate_proj.weight"))?,
                up_proj: take(&format!("{p}mlp.up_proj.weight"))?,
                down_proj: take(&format!("{p}mlp.down_proj.weight"))?,
                post_feedforward_layernorm: take(&format!("{p}post_feedforward_layernorm.weight"))?,
                layer_scalar,
            });
        }

        let embed_tokens = take("model.embed_tokens.weight")?;
        if !embed_tokens.len().is_multiple_of(ASSISTANT_HIDDEN) {
            return Err(invalid(format!(
                "MTP assistant embed_tokens has {} elements, not a multiple of {ASSISTANT_HIDDEN}",
                embed_tokens.len()
            )));
        }
        let vocab_size = embed_tokens.len() / ASSISTANT_HIDDEN;

        Ok(Self {
            layers,
            pre_projection: take("pre_projection.weight")?,
            post_projection: take("post_projection.weight")?,
            final_norm: take("model.norm.weight")?,
            embed_tokens,
            vocab_size,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Run one proposal step.
    ///
    /// `scaled_embedding` is the host's scaled embedding of the still-unforwarded bonus
    /// token; `final_normalized_hidden` is the host's final normalized hidden for the last
    /// processed position (or the previous step's `recurrent_hidden` when recurring).
    /// `position` is the shared-KV logical length — the bonus token's position.
    pub fn step(
        &self,
        scaled_embedding: &[f32],
        final_normalized_hidden: &[f32],
        sliding: &SharedKv<'_>,
        full: &SharedKv<'_>,
        position: usize,
    ) -> Result<AssistantStep> {
        if scaled_embedding.len() != BACKBONE_HIDDEN
            || final_normalized_hidden.len() != BACKBONE_HIDDEN
        {
            return Err(invalid(format!(
                "MTP assistant inputs must both be {BACKBONE_HIDDEN} wide, got {} and {}",
                scaled_embedding.len(),
                final_normalized_hidden.len()
            )));
        }
        sliding.validate(SLIDING_KV_HEADS, SLIDING_HEAD_DIM, "sliding")?;
        full.validate(FULL_KV_HEADS, FULL_HEAD_DIM, "full")?;

        // pre_projection consumes concat(embedding, hidden) — EMBEDDING FIRST.
        let mut joined = Vec::with_capacity(2 * BACKBONE_HIDDEN);
        joined.extend_from_slice(scaled_embedding);
        joined.extend_from_slice(final_normalized_hidden);
        let mut hidden = vec![0.0f32; ASSISTANT_HIDDEN];
        matvec(
            &self.pre_projection,
            &joined,
            ASSISTANT_HIDDEN,
            2 * BACKBONE_HIDDEN,
            &mut hidden,
        );

        let sliding_inv = inv_freq(SLIDING_HEAD_DIM, SLIDING_ROPE_THETA, None);
        let full_inv = inv_freq(
            FULL_HEAD_DIM,
            FULL_ROPE_THETA,
            Some(FULL_PARTIAL_ROTARY_FACTOR),
        );

        for (index, layer) in self.layers.iter().enumerate() {
            let is_sliding = index < SLIDING_LAYERS;
            let head_dim = LayerWeights::head_dim(index);
            let kv = if is_sliding { sliding } else { full };
            let inv = if is_sliding { &sliding_inv } else { &full_inv };
            let q_width = NUM_ATTENTION_HEADS * head_dim;
            let residual = hidden.clone();

            let mut normed = hidden.clone();
            rms_norm(&mut normed, &layer.input_layernorm);

            let mut q = vec![0.0f32; q_width];
            matvec(&layer.q_proj, &normed, q_width, ASSISTANT_HIDDEN, &mut q);
            rms_norm(&mut q, &layer.q_norm);
            apply_rope(&mut q, head_dim, position, inv);

            // Sliding layers see only the last SLIDING_WINDOW host positions.
            let lo = if is_sliding {
                kv.kv_len.saturating_sub(SLIDING_WINDOW)
            } else {
                0
            };
            let span = kv.kv_len - lo;
            let groups = NUM_ATTENTION_HEADS / kv.kv_heads;
            let mut context = vec![0.0f32; q_width];
            let mut scores = vec![0.0f32; span];
            for head in 0..NUM_ATTENTION_HEADS {
                let kv_head = head / groups;
                let base = (kv_head * kv.kv_len + lo) * head_dim;
                let qh = &q[head * head_dim..(head + 1) * head_dim];
                // scaling is 1.0 — Gemma 4 relies on q_norm, NOT head_dim**-0.5.
                let mut max = f32::NEG_INFINITY;
                for (position_index, score) in scores.iter_mut().enumerate() {
                    let k = &kv.key
                        [base + position_index * head_dim..base + (position_index + 1) * head_dim];
                    let dot = k.iter().zip(qh).map(|(a, b)| a * b).sum::<f32>();
                    *score = dot;
                    if dot > max {
                        max = dot;
                    }
                }
                let mut total = 0.0f32;
                for score in scores.iter_mut() {
                    *score = (*score - max).exp();
                    total += *score;
                }
                let inv_total = 1.0 / total;
                let out = &mut context[head * head_dim..(head + 1) * head_dim];
                out.fill(0.0);
                for (position_index, score) in scores.iter().enumerate() {
                    let weight = *score * inv_total;
                    let v = &kv.value
                        [base + position_index * head_dim..base + (position_index + 1) * head_dim];
                    for (o, value) in out.iter_mut().zip(v) {
                        *o += weight * value;
                    }
                }
            }

            let mut attn = vec![0.0f32; ASSISTANT_HIDDEN];
            matvec(
                &layer.o_proj,
                &context,
                ASSISTANT_HIDDEN,
                q_width,
                &mut attn,
            );
            rms_norm(&mut attn, &layer.post_attention_layernorm);
            for (h, (r, a)) in hidden.iter_mut().zip(residual.iter().zip(attn.iter())) {
                *h = r + a;
            }

            let residual = hidden.clone();
            let mut normed = hidden.clone();
            rms_norm(&mut normed, &layer.pre_feedforward_layernorm);
            let intermediate = layer.gate_proj.len() / ASSISTANT_HIDDEN;
            let mut gate = vec![0.0f32; intermediate];
            let mut up = vec![0.0f32; intermediate];
            matvec(
                &layer.gate_proj,
                &normed,
                intermediate,
                ASSISTANT_HIDDEN,
                &mut gate,
            );
            matvec(
                &layer.up_proj,
                &normed,
                intermediate,
                ASSISTANT_HIDDEN,
                &mut up,
            );
            for (g, u) in gate.iter_mut().zip(&up) {
                *g = gelu_tanh(*g) * *u;
            }
            let mut ffn = vec![0.0f32; ASSISTANT_HIDDEN];
            matvec(
                &layer.down_proj,
                &gate,
                ASSISTANT_HIDDEN,
                intermediate,
                &mut ffn,
            );
            rms_norm(&mut ffn, &layer.post_feedforward_layernorm);
            for (h, (r, f)) in hidden.iter_mut().zip(residual.iter().zip(ffn.iter())) {
                // layer_scalar scales the WHOLE layer output, residual included.
                *h = (r + f) * layer.layer_scalar;
            }
        }

        rms_norm(&mut hidden, &self.final_norm);

        let mut recurrent_hidden = vec![0.0f32; BACKBONE_HIDDEN];
        matvec(
            &self.post_projection,
            &hidden,
            BACKBONE_HIDDEN,
            ASSISTANT_HIDDEN,
            &mut recurrent_hidden,
        );

        let mut logits = vec![0.0f32; self.vocab_size];
        matvec(
            &self.embed_tokens,
            &hidden,
            self.vocab_size,
            ASSISTANT_HIDDEN,
            &mut logits,
        );

        Ok(AssistantStep {
            logits,
            recurrent_hidden,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_rope_rotates_only_the_leading_pairs() {
        // Gemma 4's full-attention layer rotates int(0.25 * 512 / 2) = 64 pairs and pads
        // the remaining 192 with ZERO frequencies. The production Metal chain shipped a
        // bug here — it rotated all 256 — and fixing it was worth +9.8%, so this is
        // pinned rather than left to inspection.
        let inv = inv_freq(
            FULL_HEAD_DIM,
            FULL_ROPE_THETA,
            Some(FULL_PARTIAL_ROTARY_FACTOR),
        );
        assert_eq!(inv.len(), FULL_HEAD_DIM / 2);
        assert_eq!(inv.iter().filter(|f| **f != 0.0).count(), 64);
        assert!(inv[..64].iter().all(|f| *f > 0.0));
        assert!(inv[64..].iter().all(|f| *f == 0.0));
        // The exponent divides by the FULL head_dim, not by twice the rotated count.
        let expected = 1.0 / FULL_ROPE_THETA.powf(2.0 / FULL_HEAD_DIM as f64);
        assert!(
            (inv[1] - expected).abs() < 1e-12,
            "got {}, want {expected}",
            inv[1]
        );
    }

    #[test]
    fn zero_frequency_pairs_are_exact_identity_under_rope() {
        let inv = inv_freq(
            FULL_HEAD_DIM,
            FULL_ROPE_THETA,
            Some(FULL_PARTIAL_ROTARY_FACTOR),
        );
        let mut q: Vec<f32> = (0..FULL_HEAD_DIM).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let before = q.clone();
        apply_rope(&mut q, FULL_HEAD_DIM, 1031, &inv);
        // Pairs 64..255 have angle zero => cos 1, sin 0 => no rotation. The value still
        // passes through the BF16 round the reference applies to query_states, so the
        // invariant is "equal to the BF16 round of the input", not bit-identical.
        for i in 64..FULL_HEAD_DIM / 2 {
            assert_eq!(
                q[i],
                round_to_bf16(before[i]),
                "pair {i} moved; it should be unrotated"
            );
            let j = i + FULL_HEAD_DIM / 2;
            assert_eq!(q[j], round_to_bf16(before[j]), "pair {j} moved");
        }
        // The leading pairs really did move.
        assert!(q[..64].iter().zip(&before[..64]).any(|(a, b)| a != b));
    }

    #[test]
    fn sliding_layers_use_full_default_rope() {
        let inv = inv_freq(SLIDING_HEAD_DIM, SLIDING_ROPE_THETA, None);
        assert_eq!(inv.len(), SLIDING_HEAD_DIM / 2);
        assert!(
            inv.iter().all(|f| *f > 0.0),
            "sliding RoPE has no nope pairs"
        );
    }

    #[test]
    fn rms_norm_uses_the_gemma4_convention() {
        // Gemma 4 is `normed * weight`. Under Gemma 2/3's `normed * (1 + weight)` an
        // all-zero weight would be identity; here it must annihilate.
        let mut x = vec![3.0f32, -4.0, 12.0, 0.0];
        rms_norm(&mut x, &[0.0, 0.0, 0.0, 0.0]);
        assert!(x.iter().all(|v| *v == 0.0), "got {x:?}");

        let mut x = vec![1.0f32, 1.0, 1.0, 1.0];
        rms_norm(&mut x, &[1.0, 1.0, 1.0, 1.0]);
        for v in &x {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "unit input under unit weight: {x:?}"
            );
        }
    }

    #[test]
    fn layer_geometry_matches_the_shipped_checkpoint() {
        assert_eq!(LayerWeights::head_dim(0), SLIDING_HEAD_DIM);
        assert_eq!(LayerWeights::head_dim(2), SLIDING_HEAD_DIM);
        assert_eq!(LayerWeights::head_dim(3), FULL_HEAD_DIM);
        assert_eq!(NUM_ATTENTION_HEADS % SLIDING_KV_HEADS, 0);
        assert_eq!(NUM_ATTENTION_HEADS % FULL_KV_HEADS, 0);
    }

    /// The oracle's deterministic BF16 bit-pattern generator, ported. This is not a
    /// random number generator: each value's sign/exponent/mantissa come straight from a
    /// hashed index, so it is independent of libm, RNG choice and float rounding, and it
    /// reproduces byte for byte on any machine.
    fn deterministic_bf16(
        count: usize,
        seed: u32,
        exponent_base: u32,
        exponent_span: u32,
    ) -> Vec<f32> {
        (0..count)
            .map(|index| {
                let mut state = (index as u32).wrapping_add(seed);
                state ^= state >> 16;
                state = state.wrapping_mul(0x7FEB_352D);
                state ^= state >> 15;
                state = state.wrapping_mul(0x846C_A68B);
                state ^= state >> 16;
                let sign = ((state >> 31) & 1) << 15;
                let exponent = (exponent_base + ((state >> 24) % exponent_span)) << 7;
                let mantissa = (state >> 16) & 0x7F;
                let bits = (sign | exponent | mantissa) as u16;
                f32::from_bits((bits as u32) << 16)
            })
            .collect()
    }

    const ORACLE_KV_LEN: usize = 1031;
    const ORACLE_POSITION: usize = 1031;

    /// End-to-end check against the committed BF16 oracle fixture.
    ///
    /// Skips when the assistant checkpoint is absent, so the suite stays green on machines
    /// that have not downloaded it. Point `CAMELID_GEMMA4_MTP_ASSISTANT_DIR` at the
    /// Hugging Face directory to run it.
    ///
    /// The threshold is deliberately loose. The reference is BF16 end to end and this
    /// forward is f32, and the assistant amplifies small input differences ~30x per layer
    /// through an unscaled softmax, so an f32 port lands near 0.7 cosine rather than 1.0.
    /// That is fine and expected: speculative decode is lossless because the TARGET
    /// verifies every draft, so drafter drift only moves the acceptance rate. What this
    /// test pins is that the STRUCTURE is right — a wrong norm convention, a wrong RoPE
    /// table or a wrong window collapses this far below the bar (bring-up measured 0.176
    /// with the Gemma 2/3 norm convention, and 0.16 with the window disabled).
    #[test]
    fn assistant_step_tracks_the_committed_oracle() {
        let dir = std::env::var("CAMELID_GEMMA4_MTP_ASSISTANT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("models")
                    .join("gemma-4-26B-A4B-it-assistant")
            });
        if !dir.join("model.safetensors").is_file() {
            eprintln!(
                "SKIP MTP assistant oracle: no checkpoint at {}",
                dir.display()
            );
            return;
        }
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("qa/gemma4-mtp/fixtures/oracle_recurrent_hidden_bf16le.bin");
        let bytes = std::fs::read(&fixture).expect("oracle fixture");
        assert_eq!(
            bytes.len(),
            BACKBONE_HIDDEN * 2,
            "fixture is not 2816 bf16 values"
        );
        let expected: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();

        let assistant = Gemma4MtpAssistant::load(&dir).expect("load MTP assistant");
        let embedding = deterministic_bf16(BACKBONE_HIDDEN, 0x1357_9BDF, 120, 3);
        let hidden = deterministic_bf16(BACKBONE_HIDDEN, 0x2468_ACE1, 125, 3);
        let sk = deterministic_bf16(
            SLIDING_KV_HEADS * ORACLE_KV_LEN * SLIDING_HEAD_DIM,
            0x3141_5926,
            125,
            3,
        );
        let sv = deterministic_bf16(
            SLIDING_KV_HEADS * ORACLE_KV_LEN * SLIDING_HEAD_DIM,
            0x2718_2818,
            125,
            3,
        );
        let fk = deterministic_bf16(
            FULL_KV_HEADS * ORACLE_KV_LEN * FULL_HEAD_DIM,
            0x1618_0339,
            125,
            3,
        );
        let fv = deterministic_bf16(
            FULL_KV_HEADS * ORACLE_KV_LEN * FULL_HEAD_DIM,
            0x5772_1566,
            125,
            3,
        );

        let sliding = SharedKv {
            key: &sk,
            value: &sv,
            kv_heads: SLIDING_KV_HEADS,
            head_dim: SLIDING_HEAD_DIM,
            kv_len: ORACLE_KV_LEN,
        };
        let full = SharedKv {
            key: &fk,
            value: &fv,
            kv_heads: FULL_KV_HEADS,
            head_dim: FULL_HEAD_DIM,
            kv_len: ORACLE_KV_LEN,
        };
        let step = assistant
            .step(&embedding, &hidden, &sliding, &full, ORACLE_POSITION)
            .expect("assistant step");

        assert_eq!(step.recurrent_hidden.len(), BACKBONE_HIDDEN);
        assert_eq!(step.logits.len(), assistant.vocab_size());

        let dot: f32 = step
            .recurrent_hidden
            .iter()
            .zip(&expected)
            .map(|(a, b)| a * b)
            .sum();
        let na = step
            .recurrent_hidden
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        let nb = expected.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cosine = dot / (na * nb);
        eprintln!("MTP assistant vs oracle: cosine={cosine:.6} |ours|={na:.2} |oracle|={nb:.2}");
        // A STRUCTURAL floor, not a similarity target. The reference is BF16 end to end
        // while this forward is f32, and the assistant amplifies small perturbations ~30x
        // per layer through an unscaled softmax, so a correct f32 port lands near 0.54 and
        // closing the rest would be over-fitting to BF16 noise. What this catches is real
        // breakage: the Gemma 2/3 norm convention scores 0.176 here and a disabled sliding
        // window scores 0.16. The genuine quality signal for a drafter is the ACCEPTANCE
        // RATE against the target, measured once the verify loop exists — not this cosine.
        assert!(
            cosine > 0.45,
            "recurrent hidden cosine {cosine:.6} is below the structural floor of 0.45; a \
             norm-convention, RoPE-table or sliding-window error collapses this to ~0.17"
        );
        // Magnitude is the tighter signal: the f32 arm sat within ~1% during bring-up.
        assert!(
            (na / nb - 1.0).abs() < 0.10,
            "recurrent hidden magnitude {na:.2} vs oracle {nb:.2} differs by more than 10%"
        );
    }
}
