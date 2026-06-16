//! Experimental CUDA acceleration for DiffusionGemma's self-conditioning
//! soft-embedding matmul (`emb_t @ probs`), the multi-step CPU bottleneck
//! (~499s/step on the reference scalar f16 emulation).
//!
//! `emb_t` (the transposed token embedding, `[hidden * n_vocab]` f16 ≈ 1.47 GB)
//! is uploaded ONCE and held resident in VRAM (it fits a 6 GB card with room to
//! spare); per step only the per-position softmax probabilities (f16) are
//! uploaded. The dot accumulates in f32 on the GPU — this is NOT bit-identical
//! to the CPU path's f16-per-op emulation (GPU reduction is non-bit-exact by
//! design, per the lane charter), so the gate is COHERENT output + token
//! closeness to the CPU oracle, not bit-identity.
//!
//! Any failure (no device, OOM, launch error) returns `None` so the caller
//! falls back to the CPU path. Opt-out with `CAMELID_DG_CUDA=0`.
#![cfg(feature = "cuda")]

use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

// NVRTC has no cuda_fp16.h here, so f16->f32 is hand-rolled (mirrors the
// resident engine's `f16_bits_to_f32`). One block per output element (pos, e);
// the block strides over the vocab, accumulates in f32, and reduces in shared.
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
"#;

struct ScEngine {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    emb_t: CudaSlice<u16>,
    hidden: usize,
    n_vocab: usize,
    /// (ptr, len) of the host emb_t this resident copy was built from.
    key: (usize, usize),
    ctx: Arc<CudaContext>,
}

// SAFETY: the engine is only ever touched while holding SC_ENGINE's mutex (the
// same single-owner discipline the resident decode cache uses for cudarc
// handles, which are otherwise not Sync).
unsafe impl Send for ScEngine {}

static SC_ENGINE: OnceLock<Mutex<Option<ScEngine>>> = OnceLock::new();

fn build_engine(emb_t: &[u16], hidden: usize, n_vocab: usize) -> Result<ScEngine, String> {
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
    let func = m
        .load_function("sc_soft_embedding")
        .map_err(|e| format!("load sc_soft_embedding: {e}"))?;
    let mut dev = stream
        .alloc_zeros::<u16>(emb_t.len())
        .map_err(|e| format!("alloc emb_t ({} u16): {e}", emb_t.len()))?;
    stream
        .memcpy_htod(emb_t, &mut dev)
        .map_err(|e| format!("upload emb_t: {e}"))?;
    ctx.synchronize().map_err(|e| format!("sync: {e}"))?;
    Ok(ScEngine {
        stream,
        func,
        emb_t: dev,
        hidden,
        n_vocab,
        key: (emb_t.as_ptr() as usize, emb_t.len()),
        ctx,
    })
}

/// `soft[pos*hidden + e] = (Σ_v emb_t[e][v] * probs[pos][v]) * embed_scale` for
/// every canvas position, computed on the GPU. `emb_t` is cached resident
/// across calls (keyed by pointer/len — `sc_emb_t()` returns a stable buffer).
/// Returns `None` (→ CPU fallback) if CUDA is unavailable or any step fails.
pub(crate) fn sc_soft_embedding_gpu(
    emb_t: &[u16],
    probs_f16: &[u16],
    c: usize,
    hidden: usize,
    n_vocab: usize,
    embed_scale: f32,
) -> Option<Vec<f32>> {
    if std::env::var("CAMELID_DG_CUDA").as_deref() == Ok("0") {
        return None;
    }
    if !crate::cuda::is_available() {
        return None;
    }
    debug_assert_eq!(emb_t.len(), hidden * n_vocab);
    debug_assert_eq!(probs_f16.len(), c * n_vocab);

    let cell = SC_ENGINE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    let key = (emb_t.as_ptr() as usize, emb_t.len());
    let stale = guard.as_ref().map(|e| e.key != key).unwrap_or(true);
    if stale {
        match build_engine(emb_t, hidden, n_vocab) {
            Ok(e) => *guard = Some(e),
            Err(err) => {
                eprintln!("[dg-cuda] sc engine build failed ({err}); falling back to CPU");
                return None;
            }
        }
    }
    let eng = guard.as_mut()?;

    let run = || -> Result<Vec<f32>, String> {
        let s = eng.stream.clone();
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
        let (h, nv, cc) = (eng.hidden as i32, eng.n_vocab as i32, c as i32);
        let mut b = s.launch_builder(&eng.func);
        b.arg(&eng.emb_t)
            .arg(&probs_dev)
            .arg(&mut soft_dev)
            .arg(&h)
            .arg(&nv)
            .arg(&cc)
            .arg(&embed_scale);
        unsafe { b.launch(cfg) }.map_err(|e| format!("launch: {e}"))?;
        let mut out = vec![0f32; c * hidden];
        s.memcpy_dtoh(&soft_dev, &mut out)
            .map_err(|e| format!("download soft: {e}"))?;
        eng.ctx.synchronize().map_err(|e| format!("sync: {e}"))?;
        Ok(out)
    };

    match run() {
        Ok(v) => Some(v),
        Err(err) => {
            eprintln!("[dg-cuda] sc matmul failed ({err}); falling back to CPU");
            None
        }
    }
}
