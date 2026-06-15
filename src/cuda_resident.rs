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
// ---- f16 round-trip (header-free) ------------------------------------------
// Bit-exact port of inference.rs f32_to_f16_bits / f16_bits_to_f32 (IEEE-754
// round-half-to-even). Used wherever the CPU reference rounds a value through
// f16 (Q8_0 block scales, KV cache writes). Pure integer/float bit ops via the
// always-available __float_as_uint / __uint_as_float builtins, so NVRTC needs
// no cuda_fp16.h (whose __float2half/__half2float are not always defined).
__device__ __forceinline__ unsigned short f32_to_f16_bits(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned short sign = (unsigned short)((bits >> 16) & 0x8000u);
    int exp = (int)((bits >> 23) & 0xffu);
    unsigned int mant = bits & 0x007fffffu;
    if (exp == 0xff) {
        return (unsigned short)(sign | (mant == 0u ? 0x7c00u : 0x7e00u));
    }
    int half_exp = exp - 127 + 15;
    if (half_exp >= 0x1f) {
        return (unsigned short)(sign | 0x7c00u);
    }
    if (half_exp <= 0) {
        if (half_exp < -10) return sign;
        unsigned int mantissa = mant | 0x00800000u;
        int shift = 14 - half_exp;
        unsigned short half_mant = (unsigned short)(mantissa >> shift);
        unsigned int round_bit = 1u << (shift - 1);
        if ((mantissa & round_bit) != 0u &&
            ((mantissa & (round_bit - 1u)) != 0u || (half_mant & 1u) != 0u)) {
            half_mant = (unsigned short)(half_mant + 1);
        }
        return (unsigned short)(sign | half_mant);
    }
    unsigned short half = (unsigned short)(sign
        | ((unsigned short)half_exp << 10) | (unsigned short)(mant >> 13));
    if ((mant & 0x00001000u) != 0u && ((mant & 0x00000fffu) != 0u || (half & 1u) != 0u)) {
        half = (unsigned short)(half + 1);
    }
    return half;
}
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    unsigned int sign = ((unsigned int)(bits & 0x8000u)) << 16;
    unsigned int exp = (bits & 0x7c00u) >> 10;
    unsigned int frac = (unsigned int)(bits & 0x03ffu);
    unsigned int out;
    if (exp == 0u) {
        if (frac == 0u) {
            out = sign;
        } else {
            unsigned int mant = frac;
            int e = -14;
            while ((mant & 0x0400u) == 0u) { mant <<= 1; e -= 1; }
            mant &= 0x03ffu;
            unsigned int exp32 = (unsigned int)(e + 127);
            out = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (frac << 13);
    } else {
        unsigned int exp32 = exp + (127u - 15u);
        out = sign | (exp32 << 23) | (frac << 13);
    }
    return __uint_as_float(out);
}
__device__ __forceinline__ float f16_round(float x) {
    return f16_bits_to_f32(f32_to_f16_bits(x));
}

// ---- RMSNorm: out[i] = x[i] * rsqrt(mean(x^2)+eps) * weight[i] -------------
// One block, blockDim threads, shared-memory sum of squares.
extern "C" __global__ void rms_norm_f32(
    const float* __restrict__ x, const float* __restrict__ weight,
    float* __restrict__ out, int n, float eps
) {
    // Thread 0 sums the squares sequentially (i = 0,1,2,...) to match the CPU
    // reference's reduction order exactly; a tree reduction reassociates the sum
    // and shifts mean_square in the last bits, which over 22 layers can flip a
    // near-tie token. The per-element apply below is order-independent.
    __shared__ float s_scale;
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += x[i] * x[i];
        float mean_sq = sum / (float)n;
        s_scale = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    float scale = s_scale;
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
    scales[b] = f16_round(unrounded); // f16-rounded block scale
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
    // Warp per output row: the 32 lanes load the row's blocks coalesced (lane L
    // reads block L, L+32, ...) and the integer block dot (__dp4a) is exact, but
    // the per-block float terms are summed sequentially by lane 0 in block order
    // (b = 0,1,2,...). That reproduces the CPU reference's exact float summation
    // order (acc += int_sum * w_scale * x_scale, left-associated) so the decode
    // is token-identical to the CPU path rather than merely close — a warp
    // tree-reduction reassociates the block sum and, compounded over the layers,
    // can flip a near-tie argmax. Coalesced loads keep it fast; only the final
    // accumulation is serial (blocks_per_row adds, cheap).
    extern __shared__ float terms[]; // (blockDim/32) * blocks_per_row
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    float* myterms = terms + (long)warp * blocks_per_row;
    if (row < rows) {
        const unsigned char* wrow = weight_bytes + (long)row * blocks_per_row * 36;
        for (int b = lane; b < blocks_per_row; b += 32) {
            const unsigned char* blk = wrow + (long)b * 36;
            float w_scale = __ldg(reinterpret_cast<const float*>(blk));
            const int* wq = reinterpret_cast<const int*>(blk + 4);
            const int* iq = reinterpret_cast<const int*>(input_quants + (long)b * 32);
            int int_sum = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) int_sum = __dp4a(wq[k], iq[k], int_sum);
            myterms[b] = (float)int_sum * w_scale * input_scales[b];
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) {
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) acc += myterms[b];
        output[row] = acc;
    }
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
    float v = f16_round(src[(long)kv_head * head_dim + d]);
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
        let ordinal = crate::cuda::selected_device_ordinal();
        let ctx =
            CudaContext::new(ordinal).map_err(|e| format!("CudaContext::new({ordinal}): {e}"))?;
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

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

// ---- Free launch helpers (take explicit refs so callers can pass disjoint
// fields of the resident state without the `&self` whole-struct borrow). ----

#[allow(clippy::too_many_arguments)]
fn launch_rmsnorm(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n: usize,
    eps: f32,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 4,
    };
    let n_i = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(x).arg(w).arg(out).arg(&n_i).arg(&eps);
    unsafe { b.launch(cfg) }.map(|_| ())
}

fn launch_quantize(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    quants: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    n_blocks: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 64u32;
    let cfg = LaunchConfig {
        grid_dim: ((n_blocks as u32).div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let nb = n_blocks as i32;
    let mut b = s.launch_builder(f);
    b.arg(x).arg(quants).arg(scales).arg(&nb);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn launch_gemv(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    in_scales: &CudaSlice<f32>,
    in_quants: &CudaSlice<i8>,
    weight: &CudaSlice<u8>,
    rows: usize,
    blocks_per_row: usize,
    out: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    // 8 warps/block, one warp per output row; shared holds each warp's per-block
    // float terms for the in-order lane-0 reduction.
    let block = 256u32;
    let warps_per_block = block / 32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: warps_per_block * (blocks_per_row as u32) * 4,
    };
    let (r, bpr) = (rows as i32, blocks_per_row as i32);
    let mut b = s.launch_builder(f);
    b.arg(in_scales)
        .arg(in_quants)
        .arg(weight)
        .arg(&r)
        .arg(&bpr)
        .arg(out);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn launch_rope(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    vec: &mut CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let total = (n_heads * (rope_dim / 2)) as u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(128).max(1), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (nh, hd, rd) = (n_heads as i32, head_dim as i32, rope_dim as i32);
    let mut b = s.launch_builder(f);
    b.arg(vec).arg(cos).arg(sin).arg(&nh).arg(&hd).arg(&rd);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn launch_kv_scatter(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    src: &CudaSlice<f32>,
    cache: &mut CudaSlice<f32>,
    position: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_pos: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let total = (n_kv_heads * head_dim) as u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(128).max(1), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (p, nkv, hd, mp) = (
        position as i32,
        n_kv_heads as i32,
        head_dim as i32,
        max_pos as i32,
    );
    let mut b = s.launch_builder(f);
    b.arg(src).arg(cache).arg(&p).arg(&nkv).arg(&hd).arg(&mp);
    unsafe { b.launch(cfg) }.map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn launch_attention(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    q: &CudaSlice<f32>,
    cache_k: &CudaSlice<f32>,
    cache_v: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position_count: usize,
    max_pos: usize,
    scale: f32,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: ((head_dim + position_count) * 4) as u32,
    };
    let (nh, nkv, hd, pc, mp) = (
        n_heads as i32,
        n_kv_heads as i32,
        head_dim as i32,
        position_count as i32,
        max_pos as i32,
    );
    let mut b = s.launch_builder(f);
    b.arg(q)
        .arg(cache_k)
        .arg(cache_v)
        .arg(out)
        .arg(&nh)
        .arg(&nkv)
        .arg(&hd)
        .arg(&pc)
        .arg(&mp)
        .arg(&scale);
    unsafe { b.launch(cfg) }.map(|_| ())
}

fn launch_silu_mul(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(gate).arg(up).arg(out).arg(&n_i);
    unsafe { b.launch(cfg) }.map(|_| ())
}

fn launch_residual(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    acc: &mut CudaSlice<f32>,
    add: &CudaSlice<f32>,
    n: usize,
) -> Result<(), cudarc::driver::DriverError> {
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(acc).arg(add).arg(&n_i);
    unsafe { b.launch(cfg) }.map(|_| ())
}

fn launch_argmax(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    logits: &CudaSlice<f32>,
    n: usize,
    out_idx: &mut CudaSlice<u32>,
) -> Result<(), cudarc::driver::DriverError> {
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 8,
    };
    let n_i = n as i32;
    let mut b = s.launch_builder(f);
    b.arg(logits).arg(&n_i).arg(out_idx);
    unsafe { b.launch(cfg) }.map(|_| ())
}

/// One layer's GPU-resident Q8_0 weights + norm vectors.
struct ResidentLayer {
    q: CudaSlice<u8>,
    k: CudaSlice<u8>,
    v: CudaSlice<u8>,
    o: CudaSlice<u8>,
    gate: CudaSlice<u8>,
    up: CudaSlice<u8>,
    down: CudaSlice<u8>,
    attn_norm: CudaSlice<f32>,
    ffn_norm: CudaSlice<f32>,
}

/// GPU-resident Llama decode engine. Weights and KV cache live on the GPU; one
/// `forward_token` call runs the whole per-token forward with a single sync.
pub struct CudaResidentDecode {
    k: CudaResidentKernels,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    ffn_dim: usize,
    rope_dim: usize,
    max_pos: usize,
    vocab: usize,
    eps: f32,
    q_width: usize,
    kv_width: usize,
    layers: Vec<ResidentLayer>,
    final_norm: CudaSlice<f32>,
    output_weight: CudaSlice<u8>,
    cache_k: Vec<CudaSlice<f32>>,
    cache_v: Vec<CudaSlice<f32>>,
    /// Number of KV positions materialized on the GPU (so the driver knows
    /// whether the session needs (re)seeding from the CPU history).
    filled: usize,
    // per-token scratch (reused)
    d_hidden: CudaSlice<f32>,
    d_normed: CudaSlice<f32>,
    d_q: CudaSlice<f32>,
    d_k: CudaSlice<f32>,
    d_v: CudaSlice<f32>,
    d_attn: CudaSlice<f32>,
    d_proj: CudaSlice<f32>,
    d_gate: CudaSlice<f32>,
    d_up: CudaSlice<f32>,
    d_ffn_act: CudaSlice<f32>,
    d_in_scales: CudaSlice<f32>,
    d_in_quants: CudaSlice<i8>,
    d_logits: CudaSlice<f32>,
    d_sampled: CudaSlice<u32>,
    d_cos: CudaSlice<f32>,
    d_sin: CudaSlice<f32>,
}

impl CudaResidentDecode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        hidden: usize,
        ffn_dim: usize,
        rope_dim: usize,
        max_pos: usize,
        vocab: usize,
        eps: f32,
    ) -> Result<Self, String> {
        let k = CudaResidentKernels::new()?;
        let s = &k.stream;
        let q_width = n_heads * head_dim;
        let kv_width = n_kv_heads * head_dim;
        let max_in = hidden.max(ffn_dim).max(q_width); // widest quantize input
        let alloc_f = |n: usize| s.alloc_zeros::<f32>(n).map_err(|e| format!("alloc: {e}"));
        let cache_k = (0..n_layers)
            .map(|_| s.alloc_zeros::<f32>(kv_width * max_pos))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("kv alloc: {e}"))?;
        let cache_v = (0..n_layers)
            .map(|_| s.alloc_zeros::<f32>(kv_width * max_pos))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("kv alloc: {e}"))?;
        Ok(Self {
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden,
            ffn_dim,
            rope_dim,
            max_pos,
            vocab,
            eps,
            q_width,
            kv_width,
            layers: Vec::with_capacity(n_layers),
            final_norm: alloc_f(hidden)?,
            output_weight: s.alloc_zeros::<u8>(1).map_err(|e| format!("alloc: {e}"))?,
            cache_k,
            cache_v,
            filled: 0,
            d_hidden: alloc_f(hidden)?,
            d_normed: alloc_f(max_in)?,
            d_q: alloc_f(q_width)?,
            d_k: alloc_f(kv_width)?,
            d_v: alloc_f(kv_width)?,
            d_attn: alloc_f(q_width)?,
            d_proj: alloc_f(hidden)?,
            d_gate: alloc_f(ffn_dim)?,
            d_up: alloc_f(ffn_dim)?,
            d_ffn_act: alloc_f(ffn_dim)?,
            d_in_scales: alloc_f(max_in / 32)?,
            d_in_quants: s
                .alloc_zeros::<i8>(max_in)
                .map_err(|e| format!("alloc: {e}"))?,
            d_logits: alloc_f(vocab)?,
            d_sampled: s.alloc_zeros::<u32>(1).map_err(|e| format!("alloc: {e}"))?,
            d_cos: alloc_f(rope_dim / 2)?,
            d_sin: alloc_f(rope_dim / 2)?,
            k,
        })
    }

    /// Upload one layer's resident weights (Q8_0 36-byte block bytes) + norms.
    #[allow(clippy::too_many_arguments)]
    pub fn set_layer(
        &mut self,
        q: &[u8],
        kk: &[u8],
        v: &[u8],
        o: &[u8],
        gate: &[u8],
        up: &[u8],
        down: &[u8],
        attn_norm: &[f32],
        ffn_norm: &[f32],
    ) -> Result<(), String> {
        let s = &self.k.stream;
        let up_u8 = |b: &[u8]| s.clone_htod(b).map_err(|e| format!("htod: {e}"));
        let up_f = |b: &[f32]| s.clone_htod(b).map_err(|e| format!("htod: {e}"));
        self.layers.push(ResidentLayer {
            q: up_u8(q)?,
            k: up_u8(kk)?,
            v: up_u8(v)?,
            o: up_u8(o)?,
            gate: up_u8(gate)?,
            up: up_u8(up)?,
            down: up_u8(down)?,
            attn_norm: up_f(attn_norm)?,
            ffn_norm: up_f(ffn_norm)?,
        });
        Ok(())
    }

    pub fn set_output(&mut self, final_norm: &[f32], output_weight: &[u8]) -> Result<(), String> {
        let s = &self.k.stream;
        self.final_norm = s.clone_htod(final_norm).map_err(|e| format!("htod: {e}"))?;
        self.output_weight = s
            .clone_htod(output_weight)
            .map_err(|e| format!("htod: {e}"))?;
        Ok(())
    }

    /// Whether `set_layer` has been called for every layer + the output stage.
    pub fn weights_ready(&self) -> bool {
        self.layers.len() == self.n_layers && self.output_weight.len() > 1
    }

    pub fn filled(&self) -> usize {
        self.filled
    }

    pub fn set_filled(&mut self, filled: usize) {
        self.filled = filled;
    }

    /// Seed one layer's KV cache from CPU history. `ck`/`cv` hold positions
    /// `[0, position)` laid out `[kv_head][position'][head_dim]` (stride
    /// `position`); they are written into the existing GPU cache buffers (stride
    /// `max_pos`) in place. For each KV head, positions `[0, position)` are
    /// contiguous in both layouts, so this is one host->device copy of
    /// `position * head_dim` floats per head — `position * kv_width` total, not
    /// the whole `max_pos`-sized buffer. (Re-uploading the full buffer made
    /// seeding a 14-token prompt cost ~160 ms of pointless PCIe traffic.) The CPU
    /// history is already f16-rounded, so it is copied as-is.
    pub fn seed_layer(
        &mut self,
        layer: usize,
        ck: &[f32],
        cv: &[f32],
        position: usize,
    ) -> Result<(), String> {
        if layer >= self.n_layers {
            return Err("seed_layer: layer out of range".into());
        }
        if position == 0 {
            return Ok(());
        }
        let (hd, max_pos, n_kv) = (self.head_dim, self.max_pos, self.n_kv_heads);
        let span = position * hd;
        let s = self.k.stream.clone();
        for h in 0..n_kv {
            let hsrc = h * span; // host: head h's [0,position) block
            let gdst = h * max_pos * hd; // gpu: head h's base (positions 0..)
            let mut vk = self.cache_k[layer].slice_mut(gdst..gdst + span);
            s.memcpy_htod(&ck[hsrc..hsrc + span], &mut vk)
                .map_err(|e| format!("seed htod k: {e}"))?;
            let mut vv = self.cache_v[layer].slice_mut(gdst..gdst + span);
            s.memcpy_htod(&cv[hsrc..hsrc + span], &mut vv)
                .map_err(|e| format!("seed htod v: {e}"))?;
        }
        Ok(())
    }

    /// Run one decode step on the GPU. `embedding` is the current token's f32
    /// embedding; `cos`/`sin` are the per-pair RoPE tables for `position`;
    /// `scale` = 1/sqrt(head_dim). With `compute_logits`, also runs the final
    /// norm + output projection + greedy argmax and returns the sampled token.
    /// One device sync at the end.
    /// Run the full per-token forward on the GPU, leaving the final logits in
    /// `d_logits` when `compute_logits`. Does NOT sample or sync — the public
    /// wrappers (`forward_token` greedy, `forward_token_logits` sampling) add the
    /// tail they need so the forward body is shared.
    #[allow(clippy::too_many_arguments)]
    fn forward_pass(
        &mut self,
        embedding: &[f32],
        cos: &[f32],
        sin: &[f32],
        position: usize,
        scale: f32,
        compute_logits: bool,
    ) -> Result<(), String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda forward: {e}");
        let s = self.k.stream.clone();
        let hb = self.hidden / 32; // hidden blocks
        let fb = self.ffn_dim / 32; // ffn blocks
        let qb = self.q_width / 32; // q_width blocks
        let pos_count = position + 1;

        s.memcpy_htod(embedding, &mut self.d_hidden).map_err(map)?;
        s.memcpy_htod(cos, &mut self.d_cos).map_err(map)?;
        s.memcpy_htod(sin, &mut self.d_sin).map_err(map)?;

        for li in 0..self.n_layers {
            // attention norm + quantize
            launch_rmsnorm(
                &s,
                &self.k.rms_norm,
                &self.d_hidden,
                &self.layers[li].attn_norm,
                &mut self.d_normed,
                self.hidden,
                self.eps,
            )
            .map_err(map)?;
            launch_quantize(
                &s,
                &self.k.quantize,
                &self.d_normed,
                &mut self.d_in_quants,
                &mut self.d_in_scales,
                hb,
            )
            .map_err(map)?;
            // Q,K,V
            launch_gemv(
                &s,
                &self.k.gemv,
                &self.d_in_scales,
                &self.d_in_quants,
                &self.layers[li].q,
                self.q_width,
                hb,
                &mut self.d_q,
            )
            .map_err(map)?;
            launch_gemv(
                &s,
                &self.k.gemv,
                &self.d_in_scales,
                &self.d_in_quants,
                &self.layers[li].k,
                self.kv_width,
                hb,
                &mut self.d_k,
            )
            .map_err(map)?;
            launch_gemv(
                &s,
                &self.k.gemv,
                &self.d_in_scales,
                &self.d_in_quants,
                &self.layers[li].v,
                self.kv_width,
                hb,
                &mut self.d_v,
            )
            .map_err(map)?;
            // RoPE on Q and K
            launch_rope(
                &s,
                &self.k.rope,
                &mut self.d_q,
                &self.d_cos,
                &self.d_sin,
                self.n_heads,
                self.head_dim,
                self.rope_dim,
            )
            .map_err(map)?;
            launch_rope(
                &s,
                &self.k.rope,
                &mut self.d_k,
                &self.d_cos,
                &self.d_sin,
                self.n_kv_heads,
                self.head_dim,
                self.rope_dim,
            )
            .map_err(map)?;
            // KV write
            launch_kv_scatter(
                &s,
                &self.k.kv_scatter,
                &self.d_k,
                &mut self.cache_k[li],
                position,
                self.n_kv_heads,
                self.head_dim,
                self.max_pos,
            )
            .map_err(map)?;
            launch_kv_scatter(
                &s,
                &self.k.kv_scatter,
                &self.d_v,
                &mut self.cache_v[li],
                position,
                self.n_kv_heads,
                self.head_dim,
                self.max_pos,
            )
            .map_err(map)?;
            // attention
            launch_attention(
                &s,
                &self.k.attention,
                &self.d_q,
                &self.cache_k[li],
                &self.cache_v[li],
                &mut self.d_attn,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                pos_count,
                self.max_pos,
                scale,
            )
            .map_err(map)?;
            // O projection + residual
            launch_quantize(
                &s,
                &self.k.quantize,
                &self.d_attn,
                &mut self.d_in_quants,
                &mut self.d_in_scales,
                qb,
            )
            .map_err(map)?;
            launch_gemv(
                &s,
                &self.k.gemv,
                &self.d_in_scales,
                &self.d_in_quants,
                &self.layers[li].o,
                self.hidden,
                qb,
                &mut self.d_proj,
            )
            .map_err(map)?;
            launch_residual(
                &s,
                &self.k.residual_add,
                &mut self.d_hidden,
                &self.d_proj,
                self.hidden,
            )
            .map_err(map)?;
            // ffn norm + gate/up + silu + down + residual
            launch_rmsnorm(
                &s,
                &self.k.rms_norm,
                &self.d_hidden,
                &self.layers[li].ffn_norm,
                &mut self.d_normed,
                self.hidden,
                self.eps,
            )
            .map_err(map)?;
            launch_quantize(
                &s,
                &self.k.quantize,
                &self.d_normed,
                &mut self.d_in_quants,
                &mut self.d_in_scales,
                hb,
            )
            .map_err(map)?;
            launch_gemv(
                &s,
                &self.k.gemv,
                &self.d_in_scales,
                &self.d_in_quants,
                &self.layers[li].gate,
                self.ffn_dim,
                hb,
                &mut self.d_gate,
            )
            .map_err(map)?;
            launch_gemv(
                &s,
                &self.k.gemv,
                &self.d_in_scales,
                &self.d_in_quants,
                &self.layers[li].up,
                self.ffn_dim,
                hb,
                &mut self.d_up,
            )
            .map_err(map)?;
            launch_silu_mul(
                &s,
                &self.k.silu_mul,
                &self.d_gate,
                &self.d_up,
                &mut self.d_ffn_act,
                self.ffn_dim,
            )
            .map_err(map)?;
            launch_quantize(
                &s,
                &self.k.quantize,
                &self.d_ffn_act,
                &mut self.d_in_quants,
                &mut self.d_in_scales,
                fb,
            )
            .map_err(map)?;
            launch_gemv(
                &s,
                &self.k.gemv,
                &self.d_in_scales,
                &self.d_in_quants,
                &self.layers[li].down,
                self.hidden,
                fb,
                &mut self.d_proj,
            )
            .map_err(map)?;
            launch_residual(
                &s,
                &self.k.residual_add,
                &mut self.d_hidden,
                &self.d_proj,
                self.hidden,
            )
            .map_err(map)?;
        }

        if !compute_logits {
            return Ok(());
        }
        // final norm + output projection -> d_logits (no argmax / no sync here).
        launch_rmsnorm(
            &s,
            &self.k.rms_norm,
            &self.d_hidden,
            &self.final_norm,
            &mut self.d_normed,
            self.hidden,
            self.eps,
        )
        .map_err(map)?;
        launch_quantize(
            &s,
            &self.k.quantize,
            &self.d_normed,
            &mut self.d_in_quants,
            &mut self.d_in_scales,
            hb,
        )
        .map_err(map)?;
        launch_gemv(
            &s,
            &self.k.gemv,
            &self.d_in_scales,
            &self.d_in_quants,
            &self.output_weight,
            self.vocab,
            hb,
            &mut self.d_logits,
        )
        .map_err(map)?;
        Ok(())
    }

    /// Greedy decode: full forward + GPU argmax. Returns the sampled token id, or
    /// `None` when logits were not requested. One device sync at the end.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token(
        &mut self,
        embedding: &[f32],
        cos: &[f32],
        sin: &[f32],
        position: usize,
        scale: f32,
        compute_logits: bool,
    ) -> Result<Option<u32>, String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda forward: {e}");
        let s = self.k.stream.clone();
        self.forward_pass(embedding, cos, sin, position, scale, compute_logits)?;
        if !compute_logits {
            self.k.ctx.synchronize().map_err(map)?;
            return Ok(None);
        }
        launch_argmax(
            &s,
            &self.k.argmax,
            &self.d_logits,
            self.vocab,
            &mut self.d_sampled,
        )
        .map_err(map)?;
        let mut out = [0u32; 1];
        s.memcpy_dtoh(&self.d_sampled, &mut out).map_err(map)?;
        self.k.ctx.synchronize().map_err(map)?;
        Ok(Some(out[0]))
    }

    /// Sampling decode: full forward on the GPU, returns the full f32 logits row
    /// so the CPU sampler can apply temperature / top-p / top-k. This keeps the
    /// whole layer stack on the GPU for non-greedy generation (the chat UI's
    /// default), instead of falling back to the CPU layer loop. One device sync.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token_logits(
        &mut self,
        embedding: &[f32],
        cos: &[f32],
        sin: &[f32],
        position: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        let map = |e: cudarc::driver::DriverError| format!("cuda forward: {e}");
        let s = self.k.stream.clone();
        self.forward_pass(embedding, cos, sin, position, scale, true)?;
        let mut logits = vec![0f32; self.vocab];
        s.memcpy_dtoh(&self.d_logits, &mut logits).map_err(map)?;
        self.k.ctx.synchronize().map_err(map)?;
        Ok(logits)
    }
}

#[cfg(test)]
mod tests;
