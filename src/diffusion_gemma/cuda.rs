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
//! * `expert_rows_gemv_gpu` — MoE expert row-range GEMVs on a VRAM-resident
//!   expert pool (budget-gated; `CAMELID_DG_EXPERT_VRAM_MB`). The Q4_K/Q8_0
//!   kernels mirror `q4_k_dot_scalar` / `q0_pair_dot` exactly → BIT-IDENTICAL
//!   to the CPU path (unit-gated). Besides speed this is a *capacity* lever:
//!   resident expert bytes are never faulted by the CPU-side mmap again.
//!
//! `CAMELID_DG_CUDA_SC=0` keeps just the (non-bit-exact) SC stage on CPU so
//! bit-exact-stage runs can be compared byte-for-byte against the CPU oracle.
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

// Q4_K scale/min helper: word pair unpacked with the kmask scheme; g in 0..8.
__device__ __forceinline__ long long kq4_byte(unsigned int w0, unsigned int w1, int g) {
    unsigned int w = (g < 4) ? w0 : w1;
    return (long long)((w >> (8 * (g & 3))) & 0xffu);
}

// Q4_K x Q8_K row-range GEMV (MoE expert gate_up rows). One thread per output
// row; mirrors refmath::q4_k_dot_scalar EXACTLY: per superblock, exact 64-bit
// integer mins/main dots, then the two fused f32 terms IN ORDER
// (sumf = fmaf(-dmin, prod, sumf); sumf = fmaf(d, sumi1+sumi2, sumf)).
// wire = the WHOLE resident tensor; rows [first_row, first_row+n_rows) are
// dotted with ONE activation (act_scales [bpr] f32, act_quants [bpr*256] i8).
extern "C" __global__ void q4k_rows_gemv(
    const unsigned char* __restrict__ wire,
    const float* __restrict__ act_scales,
    const signed char* __restrict__ act_quants,
    long long first_row, int n_rows, int bpr,
    float* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const unsigned char* rowp = wire + (first_row + (long long)r) * (long long)bpr * 144;
    float sumf = 0.0f;
    for (int i = 0; i < bpr; i++) {
        const unsigned char* block = rowp + (long long)i * 144;
        float yd = act_scales[i];
        const signed char* q8 = act_quants + (long long)i * 256;
        float d = yd * f16_bits_to_f32((unsigned short)block[0]
            | ((unsigned short)block[1] << 8));
        float dmin = yd * f16_bits_to_f32((unsigned short)block[2]
            | ((unsigned short)block[3] << 8));
        const unsigned char* sc = block + 4;
        const unsigned char* qs = block + 16;
        unsigned int utmp0 = (unsigned int)sc[0] | ((unsigned int)sc[1] << 8)
            | ((unsigned int)sc[2] << 16) | ((unsigned int)sc[3] << 24);
        unsigned int utmp1 = (unsigned int)sc[4] | ((unsigned int)sc[5] << 8)
            | ((unsigned int)sc[6] << 16) | ((unsigned int)sc[7] << 24);
        unsigned int utmp2 = (unsigned int)sc[8] | ((unsigned int)sc[9] << 8)
            | ((unsigned int)sc[10] << 16) | ((unsigned int)sc[11] << 24);
        unsigned int mins0 = utmp1 & 0x3f3f3f3fu;
        unsigned int mins1 = ((utmp2 >> 4) & 0x0f0f0f0fu)
            | (((utmp1 >> 6) & 0x03030303u) << 4);
        unsigned int scw0 = utmp0 & 0x3f3f3f3fu;
        unsigned int scw1 = (utmp2 & 0x0f0f0f0fu)
            | (((utmp0 >> 6) & 0x03030303u) << 4);
        // mins side: per-32 activation sums x mins (bsum pairs), exact ints.
        long long prod = 0;
        for (int g = 0; g < 8; g++) {
            int bs = 0;
            for (int t = 0; t < 32; t++) bs += q8[g * 32 + t];
            prod += (long long)bs * kq4_byte(mins0, mins1, g);
        }
        sumf = fmaf(-dmin, (float)prod, sumf);
        // main side: 4 chunks; low nibbles dot q8[64j..+32] with scale(2j),
        // high nibbles dot q8[64j+32..+32] with scale(2j+1).
        long long sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < 4; j++) {
            const unsigned char* q4 = qs + j * 32;
            const signed char* q8j = q8 + j * 64;
            long long lo = 0, hi = 0;
            for (int t = 0; t < 32; t++) {
                lo += (long long)(q4[t] & 0xf) * q8j[t];
                hi += (long long)(q4[t] >> 4) * q8j[32 + t];
            }
            sumi1 += lo * kq4_byte(scw0, scw1, 2 * j);
            sumi2 += hi * kq4_byte(scw0, scw1, 2 * j + 1);
        }
        sumf = fmaf(d, (float)(sumi1 + sumi2), sumf);
    }
    out[r] = sumf;
}

// Q8_0 x Q8_0 row-range GEMV (MoE expert down rows). One thread per output
// row; mirrors refmath::q0_pair_dot EXACTLY: 8 accumulators acc[parity][lane],
// per block acc[i%2][l] = fmaf(lane_dot, d_w*y_scale, acc), and the fixed
// pairwise reduction tree at the end. wire = whole resident tensor; the
// activation is act_scales [nb] f32 + act_quants [nb*32] i8.
extern "C" __global__ void q8_0_rows_gemv(
    const unsigned char* __restrict__ wire,
    const float* __restrict__ act_scales,
    const signed char* __restrict__ act_quants,
    long long first_row, int n_rows, int nb,
    float* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const unsigned char* rowp = wire + (first_row + (long long)r) * (long long)nb * 34;
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < nb; i++) {
        const unsigned char* block = rowp + (long long)i * 34;
        float dw = f16_bits_to_f32((unsigned short)block[0]
            | ((unsigned short)block[1] << 8));
        float s = dw * act_scales[i];
        const signed char* wq = (const signed char*)(block + 2);
        const signed char* yq = act_quants + (long long)i * 32;
        float* acc = (i & 1) ? acc1 : acc0;
        for (int l = 0; l < 4; l++) {
            int lane = 0;
            for (int t = 0; t < 4; t++) {
                lane += (int)wq[4 * l + t] * (int)yq[4 * l + t];
                lane += (int)wq[16 + 4 * l + t] * (int)yq[16 + 4 * l + t];
            }
            acc[l] = fmaf((float)lane, s, acc[l]);
        }
    }
    out[r] = ((acc0[0] + acc0[1]) + (acc0[2] + acc0[3]))
        + ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
}
"#;

struct Engine {
    stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
    sc_func: CudaFunction,
    lm_func: CudaFunction,
    q4k_rows_func: CudaFunction,
    q80_rows_func: CudaFunction,
    /// Resident transposed embedding (f16) for the SC matmul.
    sc_emb: Option<(CudaSlice<u16>, (usize, usize))>,
    /// Resident Q6_K lm_head weight (wire bytes).
    lm_wire: Option<(CudaSlice<u8>, (usize, usize))>,
    /// MoE expert pool: whole expert tensors resident in VRAM, keyed by
    /// (host base ptr, len). Uploaded greedily until the byte budget runs out;
    /// tensors that miss the budget are remembered so they fail fast to CPU.
    expert_pool: std::collections::HashMap<(usize, usize), CudaSlice<u8>>,
    expert_rejected: std::collections::HashSet<(usize, usize)>,
    /// Remaining expert-pool byte budget; `None` until first use (computed
    /// from free VRAM minus a reserve, or `CAMELID_DG_EXPERT_VRAM_MB`).
    expert_budget: Option<u64>,
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
    let m = ctx
        .load_module(ptx)
        .map_err(|e| format!("load_module: {e}"))?;
    let sc_func = m
        .load_function("sc_soft_embedding")
        .map_err(|e| format!("load sc_soft_embedding: {e}"))?;
    let lm_func = m
        .load_function("q6k_gemv_q8k")
        .map_err(|e| format!("load q6k_gemv_q8k: {e}"))?;
    let q4k_rows_func = m
        .load_function("q4k_rows_gemv")
        .map_err(|e| format!("load q4k_rows_gemv: {e}"))?;
    let q80_rows_func = m
        .load_function("q8_0_rows_gemv")
        .map_err(|e| format!("load q8_0_rows_gemv: {e}"))?;
    Ok(Engine {
        stream,
        ctx,
        sc_func,
        lm_func,
        q4k_rows_func,
        q80_rows_func,
        sc_emb: None,
        lm_wire: None,
        expert_pool: std::collections::HashMap::new(),
        expert_rejected: std::collections::HashSet::new(),
        expert_budget: None,
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
    // Per-stage opt-out: the SC matmul is the one non-bit-exact GPU stage
    // (f32 accumulation). CAMELID_DG_CUDA_SC=0 keeps it on CPU so a run with
    // the bit-exact stages (expert pool, lm_head) can be compared byte-for-
    // byte against the CPU-pure oracle.
    if std::env::var("CAMELID_DG_CUDA_SC").as_deref() == Ok("0") {
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

/// Which expert-tensor GEMV kernel to run (matches the DG expert formats).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DgExpertKind {
    /// 144-byte superblocks; activation is Q8_K (scales [bpr], quants [bpr*256]).
    Q4K,
    /// 34-byte blocks; activation is Q8_0 (scales [nb], quants [nb*32]).
    Q80,
}

/// MoE expert row-range GEMV on the VRAM-resident expert pool.
///
/// `tensor` is the WHOLE tensor's wire bytes (an mmap slice — creating it does
/// not fault pages; only an upload reads it). On first sight of a tensor it is
/// uploaded resident if the pool budget allows, otherwise remembered as
/// rejected (fast CPU fallback thereafter). The kernels mirror the CPU
/// `q4_k_dot_scalar` / `q0_pair_dot` reductions exactly, so a resident expert
/// computes bit-identically to the CPU path — where a weight lives never
/// changes the math. Budget: `CAMELID_DG_EXPERT_VRAM_MB` or free VRAM minus a
/// reserve for the SC embedding + lm_head + scratch.
pub(crate) fn expert_rows_gemv_gpu(
    tensor: &[u8],
    kind: DgExpertKind,
    first_row: usize,
    n_rows: usize,
    blocks_per_row: usize,
    act_scales: &[f32],
    act_quants: &[i8],
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
    let key = (tensor.as_ptr() as usize, tensor.len());
    if eng.expert_rejected.contains(&key) {
        return None;
    }
    if !eng.expert_pool.contains_key(&key) {
        // Initialize the pool budget on first use.
        if eng.expert_budget.is_none() {
            let budget = match std::env::var("CAMELID_DG_EXPERT_VRAM_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
            {
                Some(mb) => mb * 1024 * 1024,
                None => {
                    const RESERVE: u64 = 2200 * 1024 * 1024;
                    let free = cudarc::driver::result::mem_get_info()
                        .map(|(f, _)| f as u64)
                        .unwrap_or(0);
                    free.saturating_sub(RESERVE)
                }
            };
            eprintln!(
                "[dg-cuda] expert pool budget {:.2} GiB",
                budget as f64 / (1u64 << 30) as f64
            );
            eng.expert_budget = Some(budget);
        }
        let budget = eng.expert_budget.unwrap();
        if (tensor.len() as u64) > budget {
            eng.expert_rejected.insert(key);
            eprintln!(
                "[dg-cuda] expert tensor {:.0} MiB over remaining budget {:.0} MiB; CPU keeps it",
                tensor.len() as f64 / (1 << 20) as f64,
                budget as f64 / (1 << 20) as f64
            );
            return None;
        }
        let s = eng.stream.clone();
        let mut dev = match s.alloc_zeros::<u8>(tensor.len()) {
            Ok(d) => d,
            Err(e) => {
                eng.expert_rejected.insert(key);
                eprintln!("[dg-cuda] expert alloc failed ({e}); CPU keeps it");
                return None;
            }
        };
        if let Err(e) = s.memcpy_htod(tensor, &mut dev) {
            eng.expert_rejected.insert(key);
            eprintln!("[dg-cuda] expert upload failed ({e}); CPU keeps it");
            return None;
        }
        eng.expert_budget = Some(budget - tensor.len() as u64);
        eng.expert_pool.insert(key, dev);
        eprintln!(
            "[dg-cuda] expert tensor resident: {:.0} MiB ({} tensors, {:.2} GiB budget left)",
            tensor.len() as f64 / (1 << 20) as f64,
            eng.expert_pool.len(),
            eng.expert_budget.unwrap() as f64 / (1u64 << 30) as f64
        );
    }
    let func = match kind {
        DgExpertKind::Q4K => &eng.q4k_rows_func,
        DgExpertKind::Q80 => &eng.q80_rows_func,
    };
    let run = || -> Result<Vec<f32>, String> {
        let s = eng.stream.clone();
        let wire_dev = eng.expert_pool.get(&key).unwrap();
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
            .alloc_zeros::<f32>(n_rows)
            .map_err(|e| format!("alloc out: {e}"))?;
        let block = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((n_rows as u32).div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (fr, nr, bp) = (first_row as i64, n_rows as i32, blocks_per_row as i32);
        let mut b = s.launch_builder(func);
        b.arg(wire_dev)
            .arg(&sc_dev)
            .arg(&q_dev)
            .arg(&fr)
            .arg(&nr)
            .arg(&bp)
            .arg(&mut out_dev);
        unsafe { b.launch(cfg) }.map_err(|e| format!("launch: {e}"))?;
        let mut out = vec![0f32; n_rows];
        s.memcpy_dtoh(&out_dev, &mut out)
            .map_err(|e| format!("download: {e}"))?;
        eng.ctx.synchronize().map_err(|e| format!("sync: {e}"))?;
        Ok(out)
    };
    match run() {
        Ok(v) => Some(v),
        Err(err) => {
            eprintln!("[dg-cuda] expert gemv failed ({err}); CPU fallback");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// GPU expert kernels vs the CPU reference dots on random data — the GPU
    /// output must be BIT-IDENTICAL per row (same integer dots, same fused f32
    /// order). Skips when no CUDA device is usable.
    #[test]
    fn expert_gemv_gpu_bit_identical_to_cpu() {
        if gate_off() {
            eprintln!("skipping: CUDA unavailable or gated off");
            return;
        }
        let mut state: u64 = 0x0dd0_57a7_e5ee_d001;

        // ---- Q4_K x Q8_K ----
        let (rows, bpr) = (13usize, 3usize);
        let mut wire = vec![0u8; rows * bpr * 144];
        for b in wire.iter_mut() {
            *b = (xorshift(&mut state) & 0xff) as u8;
        }
        // Keep the f16 d/dmin scales finite: real quantized weights never
        // carry Inf/NaN scales, and NaN payloads canonicalize differently on
        // GPU vs x86 (payload bits, not math, would fail the comparison).
        for sb in 0..rows * bpr {
            wire[sb * 144 + 1] &= 0x3f;
            wire[sb * 144 + 3] &= 0x3f;
        }
        let mut blocks = Vec::with_capacity(bpr);
        let mut scales = vec![0f32; bpr];
        let mut quants = vec![0i8; bpr * 256];
        for i in 0..bpr {
            let mut qs = [0i8; 256];
            for q in qs.iter_mut() {
                *q = (xorshift(&mut state) & 0xff) as u8 as i8;
            }
            let d = (xorshift(&mut state) % 1000) as f32 / 333.0 + 0.001;
            scales[i] = d;
            quants[i * 256..(i + 1) * 256].copy_from_slice(&qs);
            blocks.push(crate::inference::Q8KBlock { d, qs });
        }
        // full range and a sub-range (first_row exercised)
        for (first, n) in [(0usize, rows), (5usize, 4usize)] {
            let gpu =
                expert_rows_gemv_gpu(&wire, DgExpertKind::Q4K, first, n, bpr, &scales, &quants)
                    .expect("gpu q4k gemv");
            for r in 0..n {
                let row = &wire[(first + r) * bpr * 144..(first + r + 1) * bpr * 144];
                let cpu = super::super::refmath::q4_k_dot_arm(row, &blocks);
                assert_eq!(
                    cpu.to_bits(),
                    gpu[r].to_bits(),
                    "q4k row {} (first={first}): cpu {cpu} gpu {}",
                    first + r,
                    gpu[r]
                );
            }
        }

        // ---- Q8_0 x Q8_0 ----
        let (rows, nb) = (11usize, 4usize);
        let mut wire = vec![0u8; rows * nb * 34];
        for b in wire.iter_mut() {
            *b = (xorshift(&mut state) & 0xff) as u8;
        }
        // Finite f16 d scales, as above.
        for blk in 0..rows * nb {
            wire[blk * 34 + 1] &= 0x3f;
        }
        let mut q80 = Vec::with_capacity(nb);
        let mut scales = vec![0f32; nb];
        let mut quants = vec![0i8; nb * 32];
        for i in 0..nb {
            let mut qv = [0i8; 32];
            for q in qv.iter_mut() {
                *q = (xorshift(&mut state) & 0xff) as u8 as i8;
            }
            let s = (xorshift(&mut state) % 1000) as f32 / 777.0 + 0.002;
            scales[i] = s;
            quants[i * 32..(i + 1) * 32].copy_from_slice(&qv);
            q80.push(crate::tensor::Q8_0Block {
                scale: s,
                quants: qv,
            });
        }
        for (first, n) in [(0usize, rows), (3usize, 6usize)] {
            let gpu =
                expert_rows_gemv_gpu(&wire, DgExpertKind::Q80, first, n, nb, &scales, &quants)
                    .expect("gpu q8_0 gemv");
            for r in 0..n {
                let row = &wire[(first + r) * nb * 34..(first + r + 1) * nb * 34];
                let cpu = super::super::refmath::q8_0_dot_arm(row, &q80);
                assert_eq!(
                    cpu.to_bits(),
                    gpu[r].to_bits(),
                    "q8_0 row {} (first={first}): cpu {cpu} gpu {}",
                    first + r,
                    gpu[r]
                );
            }
        }
    }
}
