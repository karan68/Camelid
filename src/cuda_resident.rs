//! GPU-resident decode kernels for the CUDA backend (`--features cuda`).
//!
//! This module holds the CUDA kernels that, together, run a full Llama decode
//! step on the GPU with weights resident and one sync per token — the analog of
//! `metal.rs`'s resident decode path. Each kernel mirrors the exact math of the
//! CPU reference (RMSNorm, Q8_0 quantize + dot, RoPE adjacent-even-odd,
//! GQA attention with f16-rounded KV, SwiGLU, residual, greedy argmax) so the
//! produced tokens are identical. The kernels are validated against small CPU
//! references in this file before being assembled into the per-token forward.
//!
//! The whole module is behind `#[cfg(feature = "cuda")]`; nothing here compiles
//! into the default build.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaStream};
use cudarc::nvrtc::{CompileOptions, Ptx};

/// CUDA C source for every resident-decode kernel. Compiled once via NVRTC with
/// `--fmad=false` and `arch=compute_61` (for `__dp4a`).
const KERNELS: &str = r#"
// ---- RMSNorm: out[i] = x[i] * rsqrt(mean(x^2)+eps) * weight[i] -------------
// One block, blockDim threads, shared-memory sum of squares.
extern "C" __global__ void rms_norm_f32(
    const float* __restrict__ x, const float* __restrict__ weight,
    float* __restrict__ out, int n, float eps
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) local += x[i] * x[i];
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean_sq = sdata[0] / (float)n;
    float scale = 1.0f / sqrtf(mean_sq + eps);
    for (int i = tid; i < n; i += blockDim.x) out[i] = x[i] * scale * weight[i];
}

// ---- Quantize f32 row to Q8_0 blocks (matches quantize_q8_0_block) ---------
// One thread per 32-value block. scale is f16-rounded; quants use the unrounded
// inverse and round-half-to-even, clamped to [-128, 127].
extern "C" __global__ void quantize_q8_0(
    const float* __restrict__ x, signed char* __restrict__ quants,
    float* __restrict__ scales, int n_blocks
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float* xb = x + (long)b * 32;
    float max_abs = 0.0f;
    for (int j = 0; j < 32; j++) { float a = fabsf(xb[j]); if (a > max_abs) max_abs = a; }
    float unrounded = max_abs / 127.0f;
    scales[b] = __half2float(__float2half(unrounded)); // f16-rounded block scale
    float inv = (unrounded == 0.0f) ? 0.0f : 1.0f / unrounded;
    signed char* qb = quants + (long)b * 32;
    for (int j = 0; j < 32; j++) {
        float v = rintf(xb[j] * inv);
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        qb[j] = (signed char)v;
    }
}

// ---- Q8_0 GEMV: one warp per output row, __dp4a dot, warp reduction --------
// weight_bytes: rows * blocks_per_row * 36 (f32 scale + 32 i8). input given as
// separate Q8 scales+quants. Token-identical (reassociated f32 sum).
extern "C" __global__ void q8_gemv(
    const float* __restrict__ input_scales, const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes, int rows, int blocks_per_row,
    float* __restrict__ output
) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = gtid >> 5;
    int lane = gtid & 31;
    if (row >= rows) return;
    const unsigned char* wrow = weight_bytes + (long)row * blocks_per_row * 36;
    float partial = 0.0f;
    for (int b = lane; b < blocks_per_row; b += 32) {
        const unsigned char* blk = wrow + (long)b * 36;
        float w_scale = __ldg(reinterpret_cast<const float*>(blk));
        const int* wq = reinterpret_cast<const int*>(blk + 4);
        const int* iq = reinterpret_cast<const int*>(input_quants + (long)b * 32);
        int int_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) int_sum = __dp4a(wq[k], iq[k], int_sum);
        partial += (float)int_sum * w_scale * input_scales[b];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, off);
    if (lane == 0) output[row] = partial;
}

// ---- RoPE: adjacent-even-odd, forward. cos/sin are per-pair (rope_dim/2). ---
extern "C" __global__ void rope_rotate(
    float* __restrict__ vec, const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    int n_heads, int head_dim, int rope_dim
) {
    int pairs = rope_dim >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heads * pairs) return;
    int head = idx / pairs;
    int pair = idx % pairs;
    float c = cos_t[pair], s = sin_t[pair];
    float* h = vec + (long)head * head_dim;
    int d0 = 2 * pair, d1 = d0 + 1;
    float x0 = h[d0], x1 = h[d1];
    h[d0] = x0 * c - x1 * s;
    h[d1] = x0 * s + x1 * c;
}

// ---- KV scatter: write current position's K (or V) with f16 round-trip -----
// cache layout [kv_head][position][head_dim].
extern "C" __global__ void kv_scatter(
    const float* __restrict__ src, float* __restrict__ cache,
    int position, int n_kv_heads, int head_dim, int max_pos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_kv_heads * head_dim) return;
    int kv_head = idx / head_dim;
    int d = idx % head_dim;
    float v = __half2float(__float2half(src[(long)kv_head * head_dim + d]));
    cache[((long)kv_head * max_pos + position) * head_dim + d] = v;
}

// ---- Attention decode: per query head, GQA, scale, softmax, weighted V -----
// One block per query head. cache_k/v layout [kv_head][position][head_dim].
extern "C" __global__ void attention_decode(
    const float* __restrict__ q, const float* __restrict__ cache_k,
    const float* __restrict__ cache_v, float* __restrict__ out,
    int n_heads, int n_kv_heads, int head_dim, int position_count, int max_pos, float scale
) {
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int repeats = n_heads / n_kv_heads;
    int kv_head = head / repeats;
    const float* qh = q + (long)head * head_dim;
    const float* kbase = cache_k + (long)kv_head * max_pos * head_dim;
    const float* vbase = cache_v + (long)kv_head * max_pos * head_dim;

    extern __shared__ float shared[];
    float* qsh = shared;                 // head_dim
    float* scores = shared + head_dim;   // position_count
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();

    // scores
    for (int p = tid; p < position_count; p += blockDim.x) {
        const float* kp = kbase + (long)p * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += qsh[d] * kp[d];
        scores[p] = dot * scale;
    }
    __syncthreads();

    // max (single-thread reduce — position_count is modest; keep it simple/correct)
    __shared__ float s_max, s_sum;
    if (tid == 0) {
        float m = scores[0];
        for (int p = 1; p < position_count; p++) if (scores[p] > m) m = scores[p];
        s_max = m;
    }
    __syncthreads();
    // exp + sum
    for (int p = tid; p < position_count; p += blockDim.x) scores[p] = expf(scores[p] - s_max);
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int p = 0; p < position_count; p++) sum += scores[p];
        s_sum = sum;
    }
    __syncthreads();
    float inv = 1.0f / s_sum;
    // weighted V: each thread handles a subset of output dims
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p < position_count; p++) acc += (scores[p] * inv) * vbase[(long)p * head_dim + d];
        out[(long)head * head_dim + d] = acc;
    }
}

// ---- SwiGLU: out[i] = silu(gate[i]) * up[i], silu(x)=x/(1+exp(-x)) ---------
extern "C" __global__ void silu_mul(
    const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

// ---- Residual add: acc[i] += add[i] ---------------------------------------
extern "C" __global__ void residual_add(float* __restrict__ acc, const float* __restrict__ add, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += add[i];
}

// ---- Greedy argmax (strict >, first index wins ties) ----------------------
// Single block. Each thread scans a stride; reduce in shared keeping lower idx
// on ties to match the CPU `>` scan.
extern "C" __global__ void argmax_f32(
    const float* __restrict__ logits, int n, unsigned int* __restrict__ out_idx
) {
    extern __shared__ float sh[];
    float* sval = sh;                                  // blockDim
    int* sidx = (int*)(sh + blockDim.x);               // blockDim
    int tid = threadIdx.x;
    float best = -3.4e38f; int besti = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        if (logits[i] > best) { best = logits[i]; besti = i; }
    }
    sval[tid] = best; sidx[tid] = besti;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s]; int oi = sidx[tid + s];
            // strict >: take the other only if strictly greater, else keep lower index
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov; sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) out_idx[0] = (unsigned int)sidx[0];
}
"#;

/// Compiled kernel set + a CUDA context/stream, used to run resident-decode
/// kernels. (The full per-token `forward_token` orchestration is assembled on
/// top of these once every kernel passes its parity check.)
// Some kernel handles are only exercised by the per-kernel parity tests until
// `forward_token` (next step) drives the whole sequence.
#[allow(dead_code)]
pub struct CudaResidentKernels {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) rms_norm: CudaFunction,
    pub(crate) quantize: CudaFunction,
    pub(crate) gemv: CudaFunction,
    pub(crate) rope: CudaFunction,
    pub(crate) kv_scatter: CudaFunction,
    pub(crate) attention: CudaFunction,
    pub(crate) silu_mul: CudaFunction,
    pub(crate) residual_add: CudaFunction,
    pub(crate) argmax: CudaFunction,
}

impl CudaResidentKernels {
    pub fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("CudaContext::new: {e}"))?;
        let stream = ctx.default_stream();
        let opts = CompileOptions {
            fmad: Some(false),
            arch: Some("compute_61"),
            ..Default::default()
        };
        let ptx: Ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNELS, opts)
            .map_err(|e| format!("nvrtc: {e}"))?;
        let m = ctx
            .load_module(ptx)
            .map_err(|e| format!("load_module: {e}"))?;
        let f = |name: &str| {
            m.load_function(name)
                .map_err(|e| format!("load {name}: {e}"))
        };
        Ok(Self {
            rms_norm: f("rms_norm_f32")?,
            quantize: f("quantize_q8_0")?,
            gemv: f("q8_gemv")?,
            rope: f("rope_rotate")?,
            kv_scatter: f("kv_scatter")?,
            attention: f("attention_decode")?,
            silu_mul: f("silu_mul")?,
            residual_add: f("residual_add")?,
            argmax: f("argmax_f32")?,
            ctx,
            stream,
        })
    }
}

#[cfg(test)]
mod tests;
