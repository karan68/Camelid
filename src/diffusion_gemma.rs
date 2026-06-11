//! Experimental DiffusionGemma encoder runtime (recon/evidence lane only).
//!
//! Implements the ENCODER side of the block-diffusion model: a causal
//! prompt-prefill forward over the shared Gemma-4 backbone with the
//! encoder-mode per-layer output scalars, producing the per-layer K/V the
//! decoder will later cross-attend into. See
//! `docs/recon/DIFFUSIONGEMMA_RECON.md` for the verified architecture facts
//! and the lane charter rules. Nothing here is a support claim; the public
//! posture stays "Active development — recon/evidence-only. Not supported."
//!
//! Correctness-first by construction:
//! - Weights are read lazily from the GGUF mmap and dotted with quantized
//!   activations mirroring the reference CPU kernels (Q8_0 activations for
//!   Q8_0/Q5_0 rows, Q8_K activations for Q4_K/Q6_K rows) — the same
//!   discipline that earned the gemma4 rows their parity.
//! - The forward emits full per-layer checkpoints (K/V, post-attention
//!   residual, MoE router logits, selected experts, scaled layer output) so
//!   the Phase 2 gate can compare layer-by-layer against the pinned
//!   llama.cpp reference (`scripts/dg-encoder-dump.cpp`).
//! - Anything the recon has not proven fails closed with a typed error.

use std::path::Path;

mod refmath;

use crate::gguf::{read_metadata, GgufTensorDescriptor, GgufTensorType};

use crate::inference::{
    q4_k_wire_row_dot, q5_0_wire_row_dot, q6_k_wire_block_dequant, q6_k_wire_row_dot,
    q8_0_wire_row_dot, quantize_q8_k_blocks, Q8KBlock, Q4_K_WIRE_BYTES_PER_BLOCK,
    Q5_0_WIRE_BYTES_PER_BLOCK, Q6_K_VALUES_PER_BLOCK, Q6_K_WIRE_BYTES_PER_BLOCK,
};
use crate::model::Gemma4Metadata;
use crate::tensor::{Q8_0Block, TensorStore};
use crate::wire_mmap::GgufWireMmap;
use crate::{BackendError, Result};

const Q8_0_WIRE_BYTES_PER_BLOCK: usize = 34;

/// The reference CPU GELU (`GGML_GELU_FP16`, on unconditionally at the pin):
/// clamp at ±10, otherwise evaluate the tanh approximation AT the
/// f16-rounded input and round the result to f16 — the exact semantics of
/// the `ggml_table_gelu_f16` lookup. The polynomial factoring mirrors
/// `ggml_gelu_f32` exactly (`x*(1 + A*x²)`, not `x + A*x³` — the two round
/// differently in f32). Camelid's exact-tanh `gelu_tanh` is a systematic
/// ~1e-3 divergence per FFN activation against this table; the DG lane's
/// checkpoint gate requires the table semantics.
// constants written to full reference precision on purpose (don't let clippy
// round them and shift the output)
#[allow(clippy::excessive_precision)]
fn dg_gelu(x: f32) -> f32 {
    if x <= -10.0 {
        return 0.0;
    }
    if x >= 10.0 {
        return x;
    }
    const SQRT_2_OVER_PI: f32 = 0.797_884_560_802_865_4;
    const GELU_COEF_A: f32 = 0.044_715;
    let v = crate::tensor::f16_round(x);
    let g = 0.5 * v * (1.0 + (SQRT_2_OVER_PI * v * (1.0 + GELU_COEF_A * v * v)).tanh());
    crate::tensor::f16_round(g)
}

/// Quantized wire formats this lane has dequant-parity evidence for
/// (Phase 0.5) and a reference-mirroring row dot for (Phase 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DgFormat {
    Q8_0,
    Q5_0,
    Q4K,
    Q6K,
}

impl DgFormat {
    fn from_tensor_type(t: GgufTensorType, name: &str) -> Result<Self> {
        match t {
            GgufTensorType::Q8_0 => Ok(Self::Q8_0),
            GgufTensorType::Q5_0 => Ok(Self::Q5_0),
            GgufTensorType::Q4K => Ok(Self::Q4K),
            GgufTensorType::Q6K => Ok(Self::Q6K),
            other => Err(BackendError::UnsupportedTensorType(format!(
                "tensor {name} is {other:?}; the DiffusionGemma encoder supports Q8_0, Q5_0, \
                 Q4_K, and Q6_K weight rows (the formats with committed parity evidence)"
            ))),
        }
    }

    fn values_per_block(self) -> usize {
        match self {
            Self::Q8_0 | Self::Q5_0 => 32,
            Self::Q4K | Self::Q6K => Q6_K_VALUES_PER_BLOCK,
        }
    }

    fn bytes_per_block(self) -> usize {
        match self {
            Self::Q8_0 => Q8_0_WIRE_BYTES_PER_BLOCK,
            Self::Q5_0 => Q5_0_WIRE_BYTES_PER_BLOCK,
            Self::Q4K => Q4_K_WIRE_BYTES_PER_BLOCK,
            Self::Q6K => Q6_K_WIRE_BYTES_PER_BLOCK,
        }
    }
}

/// One input activation, quantized once per projection group in both shapes
/// the row dots consume.
struct DgActivation {
    q8_0: Vec<Q8_0Block>,
    q8_k: Option<Vec<Q8KBlock>>,
}

impl DgActivation {
    /// `len` must be 32-aligned; the Q8_K form is built only when it is also
    /// 256-aligned (rows of other widths never bind K-quant formats — the
    /// loader's block-alignment validation guarantees it).
    fn new(x: &[f32]) -> Self {
        let q8_k = if x.len().is_multiple_of(Q6_K_VALUES_PER_BLOCK) {
            Some(quantize_q8_k_blocks(x))
        } else {
            None
        };
        // ARM reference semantics (refmath::quantize_q8_0_arm): nearest-even
        // rounding (vcvtnq) and the scale stored at f16 precision. DG-lane
        // local: the proven gemma4 rows keep their committed behavior.
        Self {
            q8_0: refmath::quantize_q8_0_arm(x),
            q8_k,
        }
    }
}

/// A quantized weight matrix read lazily from the GGUF mmap; rows are dotted
/// in place against pre-quantized activations with the reference-mirroring
/// kernels.
struct DgWire {
    mmap: std::sync::Arc<GgufWireMmap>,
    offset: u64,
    in_dim: usize,
    rows: usize,
    format: DgFormat,
}

impl DgWire {
    fn bind(
        mmap: &std::sync::Arc<GgufWireMmap>,
        desc: &GgufTensorDescriptor,
        expect_in_dim: usize,
    ) -> Result<Self> {
        let format = DgFormat::from_tensor_type(desc.tensor_type, &desc.name)?;
        let element_count = desc.dimensions.iter().product::<u64>() as usize;
        let in_dim = desc.dimensions.first().copied().unwrap_or(0) as usize;
        if in_dim != expect_in_dim {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {} in_dim {in_dim} != expected {expect_in_dim}",
                desc.name
            )));
        }
        if !in_dim.is_multiple_of(format.values_per_block()) {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {} rows of {in_dim} are not aligned to {:?} blocks",
                desc.name, format
            )));
        }
        let rows = element_count / in_dim;
        let byte_len = element_count / format.values_per_block() * format.bytes_per_block();
        if desc.n_bytes as usize != byte_len {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {} {:?} byte size {} != expected {byte_len}",
                desc.name, format, desc.n_bytes
            )));
        }
        mmap.bytes(desc.absolute_offset, byte_len)?;
        Ok(Self {
            mmap: mmap.clone(),
            offset: desc.absolute_offset,
            in_dim,
            rows,
            format,
        })
    }

    fn row_bytes(&self) -> usize {
        self.in_dim / self.format.values_per_block() * self.format.bytes_per_block()
    }

    /// y[r] = dequant(W[first_row + r]) · x for `n_rows` rows.
    fn matvec_rows(&self, first_row: usize, n_rows: usize, x: &DgActivation) -> Result<Vec<f32>> {
        if first_row + n_rows > self.rows {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "row range [{first_row}, {}) exceeds {} rows",
                first_row + n_rows,
                self.rows
            )));
        }
        let rb = self.row_bytes();
        let bytes = self
            .mmap
            .bytes(self.offset + (first_row * rb) as u64, n_rows * rb)?;
        let mut out = vec![0f32; n_rows];
        match self.format {
            DgFormat::Q8_0 => {
                for (r, y) in out.iter_mut().enumerate() {
                    *y = q8_0_wire_row_dot(&bytes[r * rb..(r + 1) * rb], &x.q8_0);
                }
            }
            DgFormat::Q5_0 => {
                for (r, y) in out.iter_mut().enumerate() {
                    *y = q5_0_wire_row_dot(&bytes[r * rb..(r + 1) * rb], &x.q8_0);
                }
            }
            DgFormat::Q4K => {
                let xk = x
                    .q8_k
                    .as_ref()
                    .expect("K-quant rows imply 256-aligned input");
                for (r, y) in out.iter_mut().enumerate() {
                    *y = q4_k_wire_row_dot(&bytes[r * rb..(r + 1) * rb], xk);
                }
            }
            DgFormat::Q6K => {
                let xk = x
                    .q8_k
                    .as_ref()
                    .expect("K-quant rows imply 256-aligned input");
                for (r, y) in out.iter_mut().enumerate() {
                    *y = q6_k_wire_row_dot(&bytes[r * rb..(r + 1) * rb], xk);
                }
            }
        }
        Ok(out)
    }

    /// Dense `mul_mat` semantics: Q8_0 weights route through the
    /// tinyBLAS_Q0_ARM element order (llamafile engages for dense Q8_0 GEMMs
    /// at the pin); every other format uses the vec_dot path. The MoE expert
    /// path (`mul_mat_id`) never uses tinyBLAS — experts call `matvec_rows`.
    fn matvec_dense(&self, x: &DgActivation) -> Result<Vec<f32>> {
        if self.format == DgFormat::Q8_0 {
            let rb = self.row_bytes();
            let bytes = self.mmap.bytes(self.offset, self.rows * rb)?;
            let mut out = vec![0f32; self.rows];
            for (r, y) in out.iter_mut().enumerate() {
                *y = refmath::tinyblas_q8_0_dot(&bytes[r * rb..(r + 1) * rb], &x.q8_0);
            }
            return Ok(out);
        }
        self.matvec_rows(0, self.rows, x)
    }
}

struct DgLayer {
    attn_norm: Vec<f32>,
    attn_q: DgWire,
    attn_k: DgWire,
    /// `None` on V-less layers (V = raw K projection).
    attn_v: Option<DgWire>,
    attn_output: DgWire,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: DgWire,
    ffn_up: DgWire,
    ffn_down: DgWire,
    post_ffw_norm: Vec<f32>,
    // MoE branch (every DiffusionGemma layer carries one)
    post_norm_1: Vec<f32>,
    pre_norm_2: Vec<f32>,
    post_norm_2: Vec<f32>,
    gate_inp: Vec<f32>,
    gate_inp_scale: Vec<f32>,
    gate_up_exps: DgWire,
    down_exps: DgWire,
    down_exps_scale: Vec<f32>,
    /// Decoder (canvas) per-layer output scalar — unused by the encoder but
    /// validated at load so Phase 3 starts from a bound weight.
    #[allow(dead_code)]
    out_scale: f32,
    /// Encoder (prompt) per-layer output scalar.
    enc_out_scale: f32,
}

/// Per-layer encoder checkpoints, shapes matching the reference dump rows
/// (positions are the slow axis; vectors are row-major `[P * width]`).
pub struct DgLayerTrace {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub moe_logits: Vec<f32>,
    pub moe_topk: Vec<i32>,
    pub out_scaled: Vec<f32>,
}

pub struct DgEncoderTrace {
    pub n_pos: usize,
    pub inp_scaled: Vec<f32>,
    pub layers: Vec<DgLayerTrace>,
    /// `output_norm` of the LAST position only (the reference PREFILL
    /// requests logits for the final row, so `result_norm` has one row).
    pub result_norm_last: Vec<f32>,
}

pub struct DgEncoderRuntime {
    g: Gemma4Metadata,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    n_expert: usize,
    n_expert_used: usize,
    n_ff_exp: usize,
    eps: f32,
    layers: Vec<DgLayer>,
    token_embd: DgWire,
    output_norm: Vec<f32>,
    rope_factors: Option<Vec<f32>>,
}

impl DgEncoderRuntime {
    pub fn load(path: &Path) -> Result<Self> {
        let gguf = read_metadata(path)?;
        let arch = gguf.architecture().unwrap_or_default().to_string();
        if arch != "diffusion-gemma" {
            return Err(BackendError::UnsupportedModelArchitecture(format!(
                "DgEncoderRuntime requires general.architecture diffusion-gemma, got {arch:?}"
            )));
        }
        let g = Gemma4Metadata::from_gguf(&gguf, &arch).ok_or_else(|| {
            BackendError::InvalidModelMetadata("diffusion-gemma metadata parse failed".into())
        })?;
        let meta_u32 = |suffix: &str| -> Result<u32> {
            gguf.metadata_u32(&format!("{arch}.{suffix}"))
                .ok_or_else(|| {
                    BackendError::InvalidModelMetadata(format!("missing {arch}.{suffix}"))
                })
        };
        let n_layer = meta_u32("block_count")? as usize;
        let n_embd = meta_u32("embedding_length")? as usize;
        let n_head = meta_u32("attention.head_count")? as usize;
        let n_expert = meta_u32("expert_count")? as usize;
        let n_expert_used = meta_u32("expert_used_count")? as usize;
        let n_ff_exp = meta_u32("expert_feed_forward_length")? as usize;
        let eps = gguf
            .metadata_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);
        if gguf
            .metadata_u32("diffusion.canvas_length")
            .or_else(|| {
                gguf.metadata_string("diffusion.canvas_length")
                    .and_then(|s| s.parse().ok())
            })
            .is_none()
        {
            return Err(BackendError::InvalidModelMetadata(
                "missing diffusion.canvas_length (not a DiffusionGemma file?)".into(),
            ));
        }

        let store = TensorStore::open(path, &gguf);
        let mmap = GgufWireMmap::map(path)?;
        let desc = |name: &str| -> Result<&GgufTensorDescriptor> {
            gguf.tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| BackendError::InvalidModelMetadata(format!("missing tensor {name}")))
        };
        let f32t = |name: &str| -> Result<Vec<f32>> { Ok(store.load_cpu_f32(name)?.data) };
        let scalar = |name: &str| -> Result<f32> {
            let v = f32t(name)?;
            v.first()
                .copied()
                .ok_or_else(|| BackendError::InvalidTensorData(format!("{name} is empty")))
        };

        let mut layers = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let head_dim = g.head_dim_at(l) as usize;
            let q_dim = n_head * head_dim;
            let ffn_dim = g.ffn_length_at(l) as usize;
            let t = |suffix: &str| format!("blk.{l}.{suffix}");
            let wire = |name: &str, in_dim: usize| -> Result<DgWire> {
                DgWire::bind(&mmap, desc(name)?, in_dim)
            };
            let attn_v = if gguf.tensors.iter().any(|d| d.name == t("attn_v.weight")) {
                Some(wire(&t("attn_v.weight"), n_embd)?)
            } else {
                None
            };
            let layer = DgLayer {
                attn_norm: f32t(&t("attn_norm.weight"))?,
                attn_q: wire(&t("attn_q.weight"), n_embd)?,
                attn_k: wire(&t("attn_k.weight"), n_embd)?,
                attn_v,
                attn_output: wire(&t("attn_output.weight"), q_dim)?,
                q_norm: f32t(&t("attn_q_norm.weight"))?,
                k_norm: f32t(&t("attn_k_norm.weight"))?,
                post_attn_norm: f32t(&t("post_attention_norm.weight"))?,
                ffn_norm: f32t(&t("ffn_norm.weight"))?,
                ffn_gate: wire(&t("ffn_gate.weight"), n_embd)?,
                ffn_up: wire(&t("ffn_up.weight"), n_embd)?,
                ffn_down: wire(&t("ffn_down.weight"), ffn_dim)?,
                post_ffw_norm: f32t(&t("post_ffw_norm.weight"))?,
                post_norm_1: f32t(&t("post_ffw_norm_1.weight"))?,
                pre_norm_2: f32t(&t("pre_ffw_norm_2.weight"))?,
                post_norm_2: f32t(&t("post_ffw_norm_2.weight"))?,
                gate_inp: f32t(&t("ffn_gate_inp.weight"))?,
                gate_inp_scale: f32t(&t("ffn_gate_inp.scale"))?,
                gate_up_exps: wire(&t("ffn_gate_up_exps.weight"), n_embd)?,
                down_exps: wire(&t("ffn_down_exps.weight"), n_ff_exp)?,
                down_exps_scale: f32t(&t("ffn_down_exps.scale"))?,
                out_scale: scalar(&t("layer_output_scale.weight"))?,
                enc_out_scale: scalar(&t("enc_layer_output_scale.weight"))?,
            };
            if layer.gate_up_exps.rows != 2 * n_ff_exp * n_expert {
                return Err(BackendError::InvalidTensorData(format!(
                    "layer {l} gate_up_exps rows {} != 2*n_ff_exp*n_expert {}",
                    layer.gate_up_exps.rows,
                    2 * n_ff_exp * n_expert
                )));
            }
            if layer.down_exps.rows != n_embd * n_expert {
                return Err(BackendError::InvalidTensorData(format!(
                    "layer {l} down_exps rows {} != n_embd*n_expert {}",
                    layer.down_exps.rows,
                    n_embd * n_expert
                )));
            }
            layers.push(layer);
        }

        let token_embd = DgWire::bind(&mmap, desc("token_embd.weight")?, n_embd)?;
        if token_embd.format != DgFormat::Q6K {
            return Err(BackendError::UnsupportedTensorType(format!(
                "token_embd is {:?}; the tracked row's embedding is Q6_K and only that \
                 row gather has parity evidence",
                token_embd.format
            )));
        }
        let rope_factors = if gguf.tensors.iter().any(|d| d.name == "rope_freqs.weight") {
            Some(f32t("rope_freqs.weight")?)
        } else {
            None
        };

        Ok(Self {
            g,
            n_layer,
            n_embd,
            n_head,
            n_expert,
            n_expert_used,
            n_ff_exp,
            eps,
            layers,
            token_embd,
            output_norm: f32t("output_norm.weight")?,
            rope_factors,
        })
    }

    /// Dequantize one embedding row (Q6_K) — mirrors the reference get_rows.
    fn embed_row(&self, token: u32) -> Result<Vec<f32>> {
        let blocks_per_row = self.n_embd / Q6_K_VALUES_PER_BLOCK;
        let rb = blocks_per_row * Q6_K_WIRE_BYTES_PER_BLOCK;
        let bytes = self
            .token_embd
            .mmap
            .bytes(self.token_embd.offset + (token as usize * rb) as u64, rb)?;
        let mut row = Vec::with_capacity(self.n_embd);
        for b in 0..blocks_per_row {
            row.extend_from_slice(&q6_k_wire_block_dequant(
                &bytes[b * Q6_K_WIRE_BYTES_PER_BLOCK..(b + 1) * Q6_K_WIRE_BYTES_PER_BLOCK],
            ));
        }
        Ok(row)
    }

    /// ENCODER prefill: one causal forward over the prompt with encoder-mode
    /// per-layer scalars, layer-major (each weight region is streamed once),
    /// emitting full per-layer checkpoints for the Phase 2 parity gate.
    pub fn encoder_prefill(&self, prompt: &[u32]) -> Result<DgEncoderTrace> {
        self.encoder_prefill_impl(prompt, None)
    }

    /// Diagnostic variant: pin the MoE routing to externally supplied expert
    /// indices (`routing[layer][pos*k + slot]`, the reference's selection)
    /// instead of camelid's own top-k. Used by the Phase 2 divergence report
    /// to isolate knife-edge router ties from every continuous checkpoint:
    /// camelid still computes and reports its own router logits/top-k; only
    /// the experts actually EXECUTED follow the pinned set (with camelid's
    /// own probabilities renormalized over that set, exactly as the
    /// reference renormalizes over its own set).
    pub fn encoder_prefill_with_pinned_routing(
        &self,
        prompt: &[u32],
        routing: &[Vec<i32>],
    ) -> Result<DgEncoderTrace> {
        if routing.len() != self.n_layer {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "pinned routing has {} layers, model has {}",
                routing.len(),
                self.n_layer
            )));
        }
        self.encoder_prefill_impl(prompt, Some(routing))
    }

    fn encoder_prefill_impl(
        &self,
        prompt: &[u32],
        pinned_routing: Option<&[Vec<i32>]>,
    ) -> Result<DgEncoderTrace> {
        let n = prompt.len();
        if n == 0 {
            return Err(BackendError::RuntimeShapeMismatch("empty prompt".into()));
        }
        let eps = self.eps;
        let hidden = self.n_embd;
        let heads = self.n_head;
        let win = self.g.sliding_window as usize;
        let embed_scale = (hidden as f32).sqrt();

        // embeddings (scaled)
        let mut h: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut inp_scaled = Vec::with_capacity(n * hidden);
        for &tok in prompt {
            let mut e = self.embed_row(tok)?;
            for v in e.iter_mut() {
                *v *= embed_scale;
            }
            inp_scaled.extend_from_slice(&e);
            h.push(e);
        }

        let mut traces = Vec::with_capacity(self.n_layer);
        for (l, lw) in self.layers.iter().enumerate() {
            let sliding = self.g.is_sliding_layer(l);
            let head_dim = self.g.head_dim_at(l) as usize;
            let kv_heads = self.g.kv_heads_at(l) as usize;
            let theta = self.g.rope_freq_base_at(l);
            let q_dim = heads * head_dim;
            let kv_dim = kv_heads * head_dim;
            let group = heads / kv_heads;
            let rope_factors = if sliding {
                None
            } else {
                self.rope_factors.as_deref()
            };

            // ---- projections for every position (layer-major) ----
            let mut qs: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut ks: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut vs: Vec<Vec<f32>> = Vec::with_capacity(n);
            for (pos, hp) in h.iter().enumerate() {
                let xn = refmath::rms_norm(hp, Some(&lw.attn_norm), eps);
                let xq = DgActivation::new(&xn);
                let mut q = lw.attn_q.matvec_dense(&xq)?;
                for hh in 0..heads {
                    let s = &mut q[hh * head_dim..(hh + 1) * head_dim];
                    s.copy_from_slice(&refmath::rms_norm(s, Some(&lw.q_norm), eps));
                }
                refmath::rope_neox(&mut q, heads, head_dim, pos, theta, rope_factors);

                let mut k = lw.attn_k.matvec_dense(&xq)?;
                // V-less layers reuse the RAW K projection as V; V then takes
                // the weightless norm and never RoPE.
                let mut v = match lw.attn_v.as_ref() {
                    Some(wv) => wv.matvec_dense(&xq)?,
                    None => k.clone(),
                };
                for hh in 0..kv_heads {
                    let s = &mut k[hh * head_dim..(hh + 1) * head_dim];
                    s.copy_from_slice(&refmath::rms_norm(s, Some(&lw.k_norm), eps));
                    let sv = &mut v[hh * head_dim..(hh + 1) * head_dim];
                    sv.copy_from_slice(&refmath::rms_norm(sv, None, eps));
                }
                refmath::rope_neox(&mut k, kv_heads, head_dim, pos, theta, rope_factors);
                qs.push(q);
                ks.push(k);
                vs.push(v);
            }

            // ---- causal attention (SWA-clipped on sliding layers) ----
            // V columns, contiguous over positions — the reference makes V
            // contiguous (cont∘transpose) so each KQV output element is one
            // vec_dot_f32 over n_kv; replicate that memory shape.
            let mut v_cols: Vec<Vec<f32>> = vec![vec![0f32; n]; kv_heads * head_dim];
            for (p, vp) in vs.iter().enumerate() {
                for (di, &val) in vp.iter().enumerate() {
                    v_cols[di][p] = val;
                }
            }

            let mut attn_out = Vec::with_capacity(n * hidden);
            for pos in 0..n {
                let lo = if sliding {
                    (pos + 1).saturating_sub(win)
                } else {
                    0
                };
                let mut attn = vec![0f32; q_dim];
                for hh in 0..heads {
                    let kvh = hh / group;
                    let qh = &qs[pos][hh * head_dim..(hh + 1) * head_dim];
                    // reference shape: KQ over the FULL row, additive -inf
                    // mask, then one softmax over the whole row (the masked
                    // slots' exp is exactly 0 in the reference's v_expf)
                    let mut row: Vec<f32> = (0..n)
                        .map(|p| {
                            if p < lo || p > pos {
                                f32::NEG_INFINITY
                            } else {
                                let kp = &ks[p][kvh * head_dim..(kvh + 1) * head_dim];
                                refmath::vec_dot_f32(qh, kp)
                            }
                        })
                        .collect();
                    refmath::softmax_row(&mut row);
                    let out = &mut attn[hh * head_dim..(hh + 1) * head_dim];
                    for (d, o) in out.iter_mut().enumerate() {
                        *o = refmath::vec_dot_f32(&v_cols[kvh * head_dim + d], &row);
                    }
                }
                let aq = DgActivation::new(&attn);
                let o = lw.attn_output.matvec_dense(&aq)?;
                let on = refmath::rms_norm(&o, Some(&lw.post_attn_norm), eps);
                for (a, b) in h[pos].iter_mut().zip(&on) {
                    *a += b;
                }
                attn_out.extend_from_slice(&h[pos]);
            }

            // ---- dense shared-expert MLP + 128-expert MoE ----
            let mut moe_logits_all = Vec::with_capacity(n * self.n_expert);
            let mut moe_topk_all = Vec::with_capacity(n * self.n_expert_used);
            let mut out_scaled = Vec::with_capacity(n * hidden);
            for (pos, hp) in h.iter_mut().enumerate() {
                let attn_resid = hp.clone();
                let xn = refmath::rms_norm(&attn_resid, Some(&lw.ffn_norm), eps);
                let xq = DgActivation::new(&xn);
                let gate = lw.ffn_gate.matvec_dense(&xq)?;
                let up = lw.ffn_up.matvec_dense(&xq)?;
                let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| dg_gelu(*g) * u).collect();
                let actq = DgActivation::new(&act);
                let mlp = lw.ffn_down.matvec_dense(&actq)?;
                let mlp = refmath::rms_norm(&mlp, Some(&lw.post_norm_1), eps);

                // Router: weightless RMS of the post-attention residual,
                // scaled by 1/sqrt(n_embd), then the elementwise input scale.
                let mut r = refmath::rms_norm(&attn_resid, None, eps);
                let inv = 1.0f32 / (hidden as f32).sqrt();
                for (rv, sv) in r.iter_mut().zip(&lw.gate_inp_scale) {
                    *rv = *rv * inv * sv;
                }
                // tinyBLAS f32 engages for this GEMM (m=128 %4==0, n=17>=4, k 4-aligned)
                let logits: Vec<f32> = (0..self.n_expert)
                    .map(|e| {
                        refmath::tinyblas_f32_dot(&lw.gate_inp[e * hidden..(e + 1) * hidden], &r)
                    })
                    .collect();
                moe_logits_all.extend_from_slice(&logits);

                // softmax over all experts, then top-k by probability
                let maxl = logits.iter().cloned().fold(f32::MIN, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let sum: f32 = probs.iter().sum();
                for p in probs.iter_mut() {
                    *p /= sum;
                }
                let mut idx: Vec<usize> = (0..self.n_expert).collect();
                idx.sort_unstable_by(|&a, &b| {
                    probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b))
                });
                idx.truncate(self.n_expert_used);
                for &e in &idx {
                    moe_topk_all.push(e as i32);
                }
                // Diagnostic pinned routing: execute the supplied expert set
                // instead of our own (probabilities renormalize over the
                // executed set either way). The reported moe_topk above stays
                // camelid's own selection.
                if let Some(routing) = pinned_routing {
                    let k = self.n_expert_used;
                    idx = routing[l][pos * k..(pos + 1) * k]
                        .iter()
                        .map(|&e| e as usize)
                        .collect();
                }
                // ggml_sum_rows accumulates in double; the clamp constant is
                // exactly 2^-14 (6.103515625e-5)
                let selected: Vec<f32> = idx.iter().map(|&e| probs[e]).collect();
                let mut wsum = refmath::vec_sum_f32(&selected);
                wsum = wsum.max(f32::from_bits(0x3880_0000));

                let cur_moe = refmath::rms_norm(&attn_resid, Some(&lw.pre_norm_2), eps);
                let cur_moe_q = DgActivation::new(&cur_moe);
                let two_nff = 2 * self.n_ff_exp;
                let mut moe_acc = vec![0f32; hidden];
                for &e in &idx {
                    let w = probs[e] / wsum;
                    let gate_up = lw
                        .gate_up_exps
                        .matvec_rows(e * two_nff, two_nff, &cur_moe_q)?;
                    let hexp: Vec<f32> = (0..self.n_ff_exp)
                        .map(|o| dg_gelu(gate_up[o]) * gate_up[o + self.n_ff_exp])
                        .collect();
                    let hexp_q = DgActivation::new(&hexp);
                    let y = lw.down_exps.matvec_rows(e * hidden, hidden, &hexp_q)?;
                    // reference order: down → ×per-expert scale → ×weight,
                    // separate multiplies, slots summed in selection order
                    let s_e = lw.down_exps_scale[e];
                    for (a, yv) in moe_acc.iter_mut().zip(&y) {
                        *a += yv * s_e * w;
                    }
                }
                let moe_out = refmath::rms_norm(&moe_acc, Some(&lw.post_norm_2), eps);

                let mut combined = mlp;
                for (c, m) in combined.iter_mut().zip(&moe_out) {
                    *c += m;
                }
                let ffn_out = refmath::rms_norm(&combined, Some(&lw.post_ffw_norm), eps);
                for (a, b) in hp.iter_mut().zip(&ffn_out) {
                    *a += b;
                }
                // ENCODER mode: prompt rows scale by the encoder scalar.
                for v in hp.iter_mut() {
                    *v *= lw.enc_out_scale;
                }
                out_scaled.extend_from_slice(hp);
            }

            let mut k_flat = Vec::with_capacity(n * kv_dim);
            let mut v_flat = Vec::with_capacity(n * kv_dim);
            for pos in 0..n {
                k_flat.extend_from_slice(&ks[pos]);
                v_flat.extend_from_slice(&vs[pos]);
            }
            traces.push(DgLayerTrace {
                k: k_flat,
                v: v_flat,
                attn_out,
                moe_logits: moe_logits_all,
                moe_topk: moe_topk_all,
                out_scaled,
            });
        }

        let result_norm_last = refmath::rms_norm(&h[n - 1], Some(&self.output_norm), eps);

        Ok(DgEncoderTrace {
            n_pos: n,
            inp_scaled,
            layers: traces,
            result_norm_last,
        })
    }
}

/// Re-export the metadata so the parity test can sanity-check shapes.
impl DgEncoderRuntime {
    pub fn n_layer(&self) -> usize {
        self.n_layer
    }
    pub fn kv_dim_at(&self, l: usize) -> usize {
        self.g.kv_heads_at(l) as usize * self.g.head_dim_at(l) as usize
    }
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }
    pub fn n_expert(&self) -> usize {
        self.n_expert
    }
    pub fn n_expert_used(&self) -> usize {
        self.n_expert_used
    }
}

/// Loading any non-diffusion file through this runtime must fail closed.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)]
    fn dg_gelu_mirrors_reference_table_semantics() {
        // clamps
        assert_eq!(dg_gelu(-10.0), 0.0);
        assert_eq!(dg_gelu(-12.5), 0.0);
        assert_eq!(dg_gelu(10.0), 10.0);
        assert_eq!(dg_gelu(42.5), 42.5);
        // interior: result is the tanh approximation evaluated at the
        // f16-rounded input, rounded to f16 (both roundings observable)
        let x = 0.123_456_7f32;
        let v = crate::tensor::f16_round(x);
        assert_ne!(v, x, "test value must not be f16-representable");
        let expect = crate::tensor::f16_round(
            0.5 * v * (1.0 + (0.797_884_56f32 * v * (1.0 + 0.044_715 * v * v)).tanh()),
        );
        assert_eq!(dg_gelu(x).to_bits(), expect.to_bits());
        // f16 quantization is observable: two nearby inputs that round to
        // the same f16 produce the identical output
        let y = x + 1e-6;
        assert_eq!(dg_gelu(x).to_bits(), dg_gelu(y).to_bits());
    }

    #[test]
    fn dg_format_fails_closed_on_unproven_types() {
        let err = DgFormat::from_tensor_type(GgufTensorType::Q4_0, "blk.0.test").unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedTensorType(_)));
        let err = DgFormat::from_tensor_type(GgufTensorType::F16, "blk.0.test").unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedTensorType(_)));
    }
}
