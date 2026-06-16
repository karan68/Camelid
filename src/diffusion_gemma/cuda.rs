//! Experimental CUDA acceleration for the DiffusionGemma forward.
//!
//! Two stages run on the GPU behind `--features cuda` (opt-out
//! `CAMELID_DG_CUDA=0`), each falling back to the CPU path on any failure:
//!
//! * `sc_soft_embedding_gpu` — the self-conditioning soft-embedding matmul
//!   (`emb_t @ probs`), the multi-step bottleneck. `emb_t` (~1.47 GB f16) is
//!   held resident; f32 accumulation (non-bit-exact vs the CPU f16 emulation).
//! * `lm_head_q6k_gpu` — the tied Q6_K lm_head GEMV over the canvas rows. The
//!   integer dot is exact (i64) and the per-block f32 term is fused in block
//!   order, mirroring the CPU `q6_k_dot` reduction, so this stage is
//!   bit-close / bit-identical to the CPU oracle.
//!
//! A single CUDA context/module is shared; each large weight tensor is uploaded
//! once and cached resident (keyed by host pointer/len).
#![cfg(feature = "cuda")]

use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

// NVRTC has no cuda_fp16.h here, so f16->f32 is hand-rolled (mirrors the
// resident engine's `f16_bits_to_f32`).
const KERNEL: &str = r#"
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    unsigned int sign = ((unsigned int)(bits & 0x8000u)) << 16;
    unsigned int exp = (bits & 0x7c00u) >> 10;
    unsigned int frac = (unsigned int)(bits & 0x03ffu);
    unsigned int out;
    if (exp == 0u) {
        if (frac == 0u) {
            out = sign;
        } else {
            unsigned int mant = frac; int e = -14;
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

// Self-conditioning soft-embedding: one block per output (pos, e); the block
// strides over the vocab, accumulates in f32, reduces in shared.
extern "C" __global__ void sc_soft_embedding(
    const unsigned short* __restrict__ emb_t,  // [hidden * n_vocab]
    const unsigned short* __restrict__ probs,  // [c * n_vocab]
    float* __restrict__ soft,                  // [c * hidden]
    int hidden, int n_vocab, int c, float embed_scale)
{
    long out_idx = (long)blockIdx.x;
    int pos = (int)(out_idx / hidden);
    int e = (int)(out_idx % hidden);
    if (pos >= c) return;
    const unsigned short* erow = emb_t + (long)e * (long)n_vocab;
    const unsigned short* prow = probs + (long)pos * (long)n_vocab;
    float acc = 0.0f;
    for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
        acc += f16_bits_to_f32(erow[v]) * f16_bits_to_f32(prow[v]);
    }
    __shared__ float sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) soft[(long)pos * (long)hidden + e] = sh[0] * embed_scale;
}

// Q6_K x Q8_K GEMV (lm_head). One thread per output (pos, row): decode the
// row's Q6_K blocks and dot with the position's Q8_K activation, mirroring the
// CPU q6_k_dot reduction (exact i64 integer math; fused per-block f32 term in
// block order). wire = [rows * bpr * 210] u8; act_scales = [c * bpr] f32;
// act_quants = [c * bpr * 256] i8; out = [c * rows] f32 (row-major per pos).
extern "C" __global__ void q6k_gemv_q8k(
    const unsigned char* __restrict__ wire,
    const float* __restrict__ act_scales,
    const signed char* __restrict__ act_quants,
    int rows, int bpr, int c,
    float* __restrict__ out)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)c * rows;
    if (idx >= total) return;
    int pos = (int)(idx / rows);
    int row = (int)(idx % rows);
    const unsigned char* rowp = wire + (long)row * bpr * 210;
    const signed char* act_q = act_quants + (long)pos * bpr * 256;
    const float* act_s = act_scales + (long)pos * bpr;
    float sumf = 0.0f;
    for (int b = 0; b < bpr; b++) {
        const unsigned char* block = rowp + (long)b * 210;
        const unsigned char* ql = block;
        const unsigned char* qh = block + 128;
        const signed char* scales = (const signed char*)(block + 192);
        unsigned short d_bits = (unsigned short)block[208]
            | ((unsigned short)block[209] << 8);
        float d_all = f16_bits_to_f32(d_bits);
        const signed char* q8 = act_q + (long)b * 256;
        float y_d = act_s[b];
        long isum = 0;
        for (int half = 0; half < 2; half++) {
            const unsigned char* qlh = ql + half * 64;
            const unsigned char* qhh = qh + half * 32;
            const signed char* q8h = q8 + half * 128;
            const signed char* sc = scales + half * 8;
            long gs0 = 0, gs1 = 0, gs2 = 0, gs3 = 0, gs4 = 0, gs5 = 0, gs6 = 0, gs7 = 0;
            for (int l = 0; l < 32; l++) {
                int v0 = (qlh[l] & 0xF) | ((qhh[l] & 3) << 4);
                int v1 = (qlh[32 + l] & 0xF) | (((qhh[l] >> 2) & 3) << 4);
                int v2 = (qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4);
                int v3 = (qlh[32 + l] >> 4) | (((qhh[l] >> 6) & 3) << 4);
                if (l < 16) {
                    gs0 += (long)v0 * q8h[l];
                    gs2 += (long)v1 * q8h[32 + l];
                    gs4 += (long)v2 * q8h[64 + l];
                    gs6 += (long)v3 * q8h[96 + l];
                } else {
                    gs1 += (long)v0 * q8h[l];
                    gs3 += (long)v1 * q8h[32 + l];
                    gs5 += (long)v2 * q8h[64 + l];
                    gs7 += (long)v3 * q8h[96 + l];
                }
            }
            isum += gs0 * (long)sc[0] + gs1 * (long)sc[1] + gs2 * (long)sc[2]
                + gs3 * (long)sc[3] + gs4 * (long)sc[4] + gs5 * (long)sc[5]
                + gs6 * (long)sc[6] + gs7 * (long)sc[7];
        }
        long isum_mins = 0;
        for (int t = 0; t < 16; t++) {
            long bs = 0;
            for (int l = 0; l < 16; l++) bs += q8[t * 16 + l];
            isum_mins += bs * (long)scales[t];
        }
        float dd = d_all * y_d;
        sumf = fmaf(dd, (float)(isum - 32 * isum_mins), sumf);
    }
    out[(long)pos * rows + row] = sumf;
}
"#;

struct Engine {
    stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
    sc_func: CudaFunction,
    lm_func: CudaFunction,
    /// Resident transposed embedding (f16) for the SC matmul.
    sc_emb: Option<(CudaSlice<u16>, (usize, usize))>,
    /// Resident Q6_K lm_head weight (wire bytes).
    lm_wire: Option<(CudaSlice<u8>, (usize, usize))>,
}

// SAFETY: the engine is only touched while holding ENGINE's mutex (the same
// single-owner discipline the resident decode cache uses for cudarc handles).
unsafe impl Send for Engine {}

static ENGINE: OnceLock<Mutex<Option<Engine>>> = OnceLock::new();

fn gate_off() -> bool {
    std::env::var("CAMELID_DG_CUDA").as_deref() == Ok("0") || !crate::cuda::is_available()
}

fn build_engine() -> Result<Engine, String> {
    let ordinal = crate::cuda::selected_device_ordinal();
    let ctx = CudaContext::new(ordinal).map_err(|e| format!("CudaContext::new({ordinal}): {e}"))?;
    let stream = ctx.default_stream();
    let opts = CompileOptions {
        fmad: Some(false),
        arch: Some("compute_61"),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(KERNEL, opts).map_err(|e| format!("nvrtc: {e}"))?;
    let m = ctx.load_module(ptx).map_err(|e| format!("load_module: {e}"))?;
    let sc_func = m
        .load_function("sc_soft_embedding")
        .map_err(|e| format!("load sc_soft_embedding: {e}"))?;
    let lm_func = m
        .load_function("q6k_gemv_q8k")
        .map_err(|e| format!("load q6k_gemv_q8k: {e}"))?;
    Ok(Engine {
        stream,
        ctx,
        sc_func,
        lm_func,
        sc_emb: None,
        lm_wire: None,
    })
}

/// `soft[pos*hidden + e] = (Σ_v emb_t[e][v] * probs[pos][v]) * embed_scale` on
/// the GPU (f32 accumulate). `emb_t` is cached resident across calls. Returns
/// `None` (→ CPU fallback) on any failure.
pub(crate) fn sc_soft_embedding_gpu(
    emb_t: &[u16],
    probs_f16: &[u16],
    c: usize,
    hidden: usize,
    n_vocab: usize,
    embed_scale: f32,
) -> Option<Vec<f32>> {
    if gate_off() {
        return None;
    }
    let cell = ENGINE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    if guard.is_none() {
        match build_engine() {
            Ok(e) => *guard = Some(e),
            Err(err) => {
                eprintln!("[dg-cuda] engine build failed ({err}); CPU fallback");
                return None;
            }
        }
    }
    let eng = guard.as_mut()?;
    let mut run = || -> Result<Vec<f32>, String> {
        let s = eng.stream.clone();
        let key = (emb_t.as_ptr() as usize, emb_t.len());
        if eng.sc_emb.as_ref().map(|(_, k)| *k != key).unwrap_or(true) {
            let mut dev = s
                .alloc_zeros::<u16>(emb_t.len())
                .map_err(|e| format!("alloc emb_t: {e}"))?;
            s.memcpy_htod(emb_t, &mut dev)
                .map_err(|e| format!("upload emb_t: {e}"))?;
            eng.sc_emb = Some((dev, key));
        }
        let emb_dev = &eng.sc_emb.as_ref().unwrap().0;
        let mut probs_dev = s
            .alloc_zeros::<u16>(probs_f16.len())
            .map_err(|e| format!("alloc probs: {e}"))?;
        s.memcpy_htod(probs_f16, &mut probs_dev)
            .map_err(|e| format!("upload probs: {e}"))?;
        let mut soft_dev = s
            .alloc_zeros::<f32>(c * hidden)
            .map_err(|e| format!("alloc soft: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: ((c * hidden) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (h, nv, cc) = (hidden as i32, n_vocab as i32, c as i32);
        let mut b = s.launch_builder(&eng.sc_func);
        b.arg(emb_dev)
            .arg(&probs_dev)
            .arg(&mut soft_dev)
            .arg(&h)
            .arg(&nv)
            .arg(&cc)
            .arg(&embed_scale);
        unsafe { b.launch(cfg) }.map_err(|e| format!("launch: {e}"))?;
        let mut out = vec![0f32; c * hidden];
        s.memcpy_dtoh(&soft_dev, &mut out)
            .map_err(|e| format!("download: {e}"))?;
        eng.ctx.synchronize().map_err(|e| format!("sync: {e}"))?;
        Ok(out)
    };
    match run() {
        Ok(v) => Some(v),
        Err(err) => {
            eprintln!("[dg-cuda] sc matmul failed ({err}); CPU fallback");
            None
        }
    }
}

/// Q6_K lm_head GEMV over the `c` canvas positions: `out[pos*rows + row]` =
/// dot(Q6_K row, Q8_K activation[pos]). `wire` (the Q6_K weight bytes) is cached
/// resident. `act_scales`/`act_quants` are the per-position Q8_K blocks packed
/// SoA. Returns the `[c*rows]` logits, or `None` on any failure.
pub(crate) fn lm_head_q6k_gpu(
    wire: &[u8],
    rows: usize,
    bpr: usize,
    act_scales: &[f32],
    act_quants: &[i8],
    c: usize,
) -> Option<Vec<f32>> {
    if gate_off() {
        return None;
    }
    debug_assert_eq!(wire.len(), rows * bpr * 210);
    debug_assert_eq!(act_scales.len(), c * bpr);
    debug_assert_eq!(act_quants.len(), c * bpr * 256);
    let cell = ENGINE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    if guard.is_none() {
        match build_engine() {
            Ok(e) => *guard = Some(e),
            Err(err) => {
                eprintln!("[dg-cuda] engine build failed ({err}); CPU fallback");
                return None;
            }
        }
    }
    let eng = guard.as_mut()?;
    let mut run = || -> Result<Vec<f32>, String> {
        let s = eng.stream.clone();
        let key = (wire.as_ptr() as usize, wire.len());
        if eng.lm_wire.as_ref().map(|(_, k)| *k != key).unwrap_or(true) {
            let mut dev = s
                .alloc_zeros::<u8>(wire.len())
                .map_err(|e| format!("alloc lm wire: {e}"))?;
            s.memcpy_htod(wire, &mut dev)
                .map_err(|e| format!("upload lm wire: {e}"))?;
            eng.lm_wire = Some((dev, key));
        }
        let wire_dev = &eng.lm_wire.as_ref().unwrap().0;
        let mut sc_dev = s
            .alloc_zeros::<f32>(act_scales.len())
            .map_err(|e| format!("alloc act scales: {e}"))?;
        s.memcpy_htod(act_scales, &mut sc_dev)
            .map_err(|e| format!("upload act scales: {e}"))?;
        let mut q_dev = s
            .alloc_zeros::<i8>(act_quants.len())
            .map_err(|e| format!("alloc act quants: {e}"))?;
        s.memcpy_htod(act_quants, &mut q_dev)
            .map_err(|e| format!("upload act quants: {e}"))?;
        let mut out_dev = s
            .alloc_zeros::<f32>(c * rows)
            .map_err(|e| format!("alloc logits: {e}"))?;
        let total = (c * rows) as u32;
        let block = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (r, bp, cc) = (rows as i32, bpr as i32, c as i32);
        let mut b = s.launch_builder(&eng.lm_func);
        b.arg(wire_dev)
            .arg(&sc_dev)
            .arg(&q_dev)
            .arg(&r)
            .arg(&bp)
            .arg(&cc)
            .arg(&mut out_dev);
        unsafe { b.launch(cfg) }.map_err(|e| format!("launch: {e}"))?;
        let mut out = vec![0f32; c * rows];
        s.memcpy_dtoh(&out_dev, &mut out)
            .map_err(|e| format!("download logits: {e}"))?;
        eng.ctx.synchronize().map_err(|e| format!("sync: {e}"))?;
        Ok(out)
    };
    match run() {
        Ok(v) => Some(v),
        Err(err) => {
            eprintln!("[dg-cuda] lm_head failed ({err}); CPU fallback");
            None
        }
    }
}
