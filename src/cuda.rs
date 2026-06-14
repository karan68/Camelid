//! Optional CUDA GPU backend (additive, gated behind `--features cuda`).
//!
//! The CPU path remains the default build and the correctness reference. This
//! backend must reproduce the exact CPU/llama.cpp parity evidence — a CUDA lane
//! that is fast but diverges from the parity audit is a regression, not a
//! feature. To that end the Q8_0 dot kernel mirrors the CPU reference
//! (`dot_q8_0_encoded_row_with_scales`) operation-for-operation: one thread per
//! output row, an exact integer block dot, then a sequential f32 accumulation
//! of `(int_sum as f32) * weight_scale * input_scale` in block order, compiled
//! with `--fmad=false` so the GPU does not fuse the multiply/add the CPU keeps
//! separate. That yields bit-identical f32 logits and therefore identical
//! greedy argmax / token IDs.
//!
//! Mirrors `src/metal.rs`'s shape: the module is always present; the real
//! implementation is `#[cfg(feature = "cuda")]` and a stub returns "unavailable"
//! otherwise, so callers never need their own cfg gates.

/// Result of probing for a usable CUDA device at startup.
#[derive(Debug, Clone, Default)]
pub struct CudaDeviceInfo {
    pub available: bool,
    pub device_name: Option<String>,
    /// Why CUDA is unavailable (feature off, no device, init error), for logs.
    pub reason: Option<String>,
}

#[cfg(feature = "cuda")]
pub use backend::{
    detect_cuda_device, try_q8_0_block_linear_row, try_q8_0_encoded_linear_row,
    try_q8_0_encoded_linear_rows,
};

#[cfg(not(feature = "cuda"))]
pub use stub::{
    detect_cuda_device, try_q8_0_block_linear_row, try_q8_0_encoded_linear_row,
    try_q8_0_encoded_linear_rows,
};

#[cfg(not(feature = "cuda"))]
mod stub {
    use super::CudaDeviceInfo;

    pub fn detect_cuda_device() -> CudaDeviceInfo {
        CudaDeviceInfo {
            available: false,
            device_name: None,
            reason: Some("built without the `cuda` feature".to_string()),
        }
    }

    /// Decode-shaped Q8_0 linear (one input row × `rows` weight rows). Returns
    /// `false` so the caller falls back to the CPU reference path.
    #[allow(clippy::too_many_arguments)]
    pub fn try_q8_0_encoded_linear_row(
        _input_scales: &[f32],
        _input_quants: &[i8],
        _weight_bytes: &[u8],
        _weight_scales: &[f32],
        _rows: usize,
        _blocks_per_row: usize,
        _output: &mut [f32],
    ) -> bool {
        false
    }

    /// Prefill-shaped Q8_0 linear (`input_rows` × `weight_rows`). Returns `false`
    /// so the caller falls back to the CPU reference path.
    #[allow(clippy::too_many_arguments)]
    pub fn try_q8_0_encoded_linear_rows(
        _input_scales: &[f32],
        _input_quants: &[i8],
        _weight_bytes: &[u8],
        _weight_scales: &[f32],
        _input_rows: usize,
        _weight_rows: usize,
        _blocks_per_row: usize,
        _output: &mut [f32],
    ) -> bool {
        false
    }

    /// Decode-shaped Q8_0 linear over the in-memory `Q8_0Block` byte layout
    /// (36 bytes/block: f32 scale + 32 i8 quants), matching the engine's
    /// retained block-dot path. Returns `false` (CPU fallback).
    pub fn try_q8_0_block_linear_row(
        _input_scales: &[f32],
        _input_quants: &[i8],
        _weight_block_bytes: &[u8],
        _rows: usize,
        _blocks_per_row: usize,
        _output: &mut [f32],
    ) -> bool {
        false
    }
}

/// Number of Q8_0 matmuls the CUDA backend has completed on the GPU this
/// process. Zero means the GPU path never ran (e.g. CPU-only fallback). Used by
/// tests/diagnostics to prove the GPU lane was actually exercised.
#[cfg(feature = "cuda")]
pub fn cuda_q8_run_count() -> u64 {
    backend::run_count()
}

#[cfg(not(feature = "cuda"))]
pub fn cuda_q8_run_count() -> u64 {
    0
}

#[cfg(feature = "cuda")]
mod backend {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};

    use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::{CompileOptions, Ptx};
    use std::sync::Arc;

    use super::CudaDeviceInfo;

    static RUN_COUNT: AtomicU64 = AtomicU64::new(0);
    static LOGGED: AtomicBool = AtomicBool::new(false);
    static ENTRY_LOGGED: AtomicBool = AtomicBool::new(false);

    pub(super) fn run_count() -> u64 {
        RUN_COUNT.load(Ordering::Relaxed)
    }

    /// CUDA C source for the Q8_0 encoded linear kernel. One thread computes one
    /// output row. The integer block dot is exact (i32); the f32 accumulation is
    /// sequential in block order and matches the CPU reference exactly. Built
    /// with `--fmad=false` (see [`compile_options`]) so the `a*b*c + sum` is not
    /// fused into an FMA — the CPU keeps those operations separate.
    const Q8_KERNEL_SRC: &str = r#"
extern "C" __global__ void q8_0_encoded_linear_rows(
    const float* __restrict__ input_scales,   // [input_rows * blocks_per_row]
    const signed char* __restrict__ input_quants, // [input_rows * blocks_per_row * 32]
    const unsigned char* __restrict__ weight_bytes, // [weight_rows * blocks_per_row * 34]
    const float* __restrict__ weight_scales,   // [weight_rows * blocks_per_row]
    const int input_rows,
    const int weight_rows,
    const int blocks_per_row,
    float* __restrict__ output                 // [input_rows * weight_rows]
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)input_rows * weight_rows;
    if (idx >= total) return;
    int in_row = (int)(idx / weight_rows);
    int w_row = (int)(idx % weight_rows);

    const float* in_scales = input_scales + (long)in_row * blocks_per_row;
    const signed char* in_quants = input_quants + (long)in_row * blocks_per_row * 32;
    const unsigned char* w_bytes = weight_bytes + (long)w_row * blocks_per_row * 34;
    const float* w_scales = weight_scales + (long)w_row * blocks_per_row;

    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const unsigned char* wblk = w_bytes + (long)b * 34 + 2; // skip 2-byte f16 scale
        const signed char* iblk = in_quants + (long)b * 32;
        int int_sum = 0;
        for (int j = 0; j < 32; j++) {
            int_sum += (int)((signed char)wblk[j]) * (int)iblk[j];
        }
        sum += (float)int_sum * w_scales[b] * in_scales[b];
    }
    output[(long)in_row * weight_rows + w_row] = sum;
}

// Decode matvec over the in-memory Q8_0Block byte layout (36 bytes/block:
// f32 scale at offset 0, then 32 i8 quants). One input row (given as separate
// scales + quants) against `rows` weight rows. Mirrors the CPU `q8_0_dot_rows`
// term order: per row, sequential over blocks, `(int_sum as f32) * w_scale *
// i_scale`, summed left-to-right. fmad=false keeps the multiply/add unfused.
extern "C" __global__ void q8_0_block_linear_row(
    const float* __restrict__ input_scales,   // [blocks_per_row]
    const signed char* __restrict__ input_quants, // [blocks_per_row * 32]
    const unsigned char* __restrict__ weight_bytes, // [rows * blocks_per_row * 36]
    const int rows,
    const int blocks_per_row,
    float* __restrict__ output                 // [rows]
) {
    long r = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const unsigned char* wrow = weight_bytes + (long)r * blocks_per_row * 36;
    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const unsigned char* blk = wrow + (long)b * 36;
        float w_scale = *reinterpret_cast<const float*>(blk);
        const signed char* wq = reinterpret_cast<const signed char*>(blk + 4);
        const signed char* iq = input_quants + (long)b * 32;
        int int_sum = 0;
        for (int j = 0; j < 32; j++) {
            int_sum += (int)wq[j] * (int)iq[j];
        }
        sum += (float)int_sum * w_scale * input_scales[b];
    }
    output[r] = sum;
}
"#;

    struct CudaBackend {
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        kernel: CudaFunction,
        kernel_block: CudaFunction,
        device_name: String,
    }

    // SAFETY: cudarc's context/stream/function handles are Send + Sync; we
    // additionally serialize all access behind a Mutex.
    fn backend() -> Option<&'static Mutex<CudaBackend>> {
        static BACKEND: OnceLock<Option<Mutex<CudaBackend>>> = OnceLock::new();
        BACKEND
            .get_or_init(|| match init_backend() {
                Ok(b) => Some(Mutex::new(b)),
                Err(_) => None,
            })
            .as_ref()
    }

    fn compile_options() -> CompileOptions {
        CompileOptions {
            // Match the CPU reference's separate multiply/add: do not let the
            // compiler contract `a*b*c + sum` into a fused multiply-add, which
            // would round differently and could flip a near-tie token.
            fmad: Some(false),
            ..Default::default()
        }
    }

    fn init_backend() -> Result<CudaBackend, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("CudaContext::new failed: {e}"))?;
        let stream = ctx.default_stream();
        let device_name = ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".to_string());
        let ptx: Ptx = cudarc::nvrtc::compile_ptx_with_opts(Q8_KERNEL_SRC, compile_options())
            .map_err(|e| format!("nvrtc compile failed: {e}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("load_module failed: {e}"))?;
        let kernel = module
            .load_function("q8_0_encoded_linear_rows")
            .map_err(|e| format!("load_function failed: {e}"))?;
        let kernel_block = module
            .load_function("q8_0_block_linear_row")
            .map_err(|e| format!("load_function (block) failed: {e}"))?;
        Ok(CudaBackend {
            ctx,
            stream,
            kernel,
            kernel_block,
            device_name,
        })
    }

    pub fn detect_cuda_device() -> CudaDeviceInfo {
        match backend() {
            Some(b) => {
                let guard = b.lock().expect("cuda backend mutex poisoned");
                CudaDeviceInfo {
                    available: true,
                    device_name: Some(guard.device_name.clone()),
                    reason: None,
                }
            }
            None => CudaDeviceInfo {
                available: false,
                device_name: None,
                reason: Some("no usable CUDA device or initialization failed".to_string()),
            },
        }
    }

    /// Run the Q8_0 encoded linear on the GPU. `input_rows` input rows (each
    /// `blocks_per_row` blocks of 32 i8 quants + per-block f32 scale) are dotted
    /// against `weight_rows` weight rows (each `blocks_per_row` 34-byte blocks +
    /// per-block decoded f32 scale). Output is `input_rows * weight_rows` f32,
    /// laid out row-major by input row. Returns `false` (caller falls back to
    /// CPU) on any error or shape mismatch.
    #[allow(clippy::too_many_arguments)]
    fn run(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        weight_scales: &[f32],
        input_rows: usize,
        weight_rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        if std::env::var("CAMELID_CUDA_TRACE").as_deref() == Ok("1")
            && !ENTRY_LOGGED.swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "[cuda-trace] run() first call: input_rows={input_rows} weight_rows={weight_rows} blocks_per_row={blocks_per_row} in_scales={} in_quants={} w_bytes={} w_scales={} out={} backend={}",
                input_scales.len(),
                input_quants.len(),
                weight_bytes.len(),
                weight_scales.len(),
                output.len(),
                backend().is_some(),
            );
        }
        if input_rows == 0 || weight_rows == 0 || blocks_per_row == 0 {
            return false;
        }
        // Shape guards: bail to CPU rather than risk an out-of-bounds GPU read.
        if input_scales.len() != input_rows * blocks_per_row
            || input_quants.len() != input_rows * blocks_per_row * 32
            || weight_bytes.len() != weight_rows * blocks_per_row * 34
            || weight_scales.len() != weight_rows * blocks_per_row
            || output.len() != input_rows * weight_rows
        {
            return false;
        }
        let Some(b) = backend() else {
            return false;
        };
        let guard = b.lock().expect("cuda backend mutex poisoned");
        match run_inner(
            &guard,
            input_scales,
            input_quants,
            weight_bytes,
            weight_scales,
            input_rows,
            weight_rows,
            blocks_per_row,
            output,
        ) {
            Ok(()) => {
                RUN_COUNT.fetch_add(1, Ordering::Relaxed);
                if !LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "[cuda] Q8_0 hybrid decode active on {} — first GPU matmul completed",
                        guard.device_name
                    );
                }
                true
            }
            Err(_) => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_inner(
        b: &CudaBackend,
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        weight_scales: &[f32],
        input_rows: usize,
        weight_rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> Result<(), cudarc::driver::DriverError> {
        let stream = &b.stream;
        let d_in_scales = stream.clone_htod(input_scales)?;
        let d_in_quants = stream.clone_htod(input_quants)?;
        let d_w_bytes = stream.clone_htod(weight_bytes)?;
        let d_w_scales = stream.clone_htod(weight_scales)?;
        let mut d_out = stream.alloc_zeros::<f32>(output.len())?;

        let total = (input_rows * weight_rows) as u32;
        let block_dim = 256u32;
        let grid_dim = total.div_ceil(block_dim);
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let input_rows_i = input_rows as i32;
        let weight_rows_i = weight_rows as i32;
        let blocks_per_row_i = blocks_per_row as i32;
        let mut builder = stream.launch_builder(&b.kernel);
        builder
            .arg(&d_in_scales)
            .arg(&d_in_quants)
            .arg(&d_w_bytes)
            .arg(&d_w_scales)
            .arg(&input_rows_i)
            .arg(&weight_rows_i)
            .arg(&blocks_per_row_i)
            .arg(&mut d_out);
        // SAFETY: the kernel reads the four input buffers and writes d_out, all
        // sized to match the launch dimensions per the shape guards in `run`.
        unsafe { builder.launch(cfg)? };
        stream.memcpy_dtoh(&d_out, output)?;
        b.ctx.synchronize()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_q8_0_encoded_linear_row(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        weight_scales: &[f32],
        rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        // Decode: a single input row against `rows` weight rows.
        run(
            input_scales,
            input_quants,
            weight_bytes,
            weight_scales,
            1,
            rows,
            blocks_per_row,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_q8_0_encoded_linear_rows(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        weight_scales: &[f32],
        input_rows: usize,
        weight_rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        run(
            input_scales,
            input_quants,
            weight_bytes,
            weight_scales,
            input_rows,
            weight_rows,
            blocks_per_row,
            output,
        )
    }

    /// Decode matvec over the in-memory `Q8_0Block` byte layout. `weight_bytes`
    /// is `rows * blocks_per_row * 36` bytes (f32 scale + 32 i8 quants/block);
    /// the input row is given as separate `blocks_per_row` scales + quants.
    /// Returns `false` (CPU fallback) on any error or shape mismatch.
    fn run_block(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        if rows == 0 || blocks_per_row == 0 {
            return false;
        }
        if std::env::var("CAMELID_CUDA_TRACE").as_deref() == Ok("1")
            && !ENTRY_LOGGED.swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "[cuda-trace] run_block() first call: rows={rows} blocks_per_row={blocks_per_row} in_scales={} in_quants={} w_bytes={} out={}",
                input_scales.len(),
                input_quants.len(),
                weight_bytes.len(),
                output.len(),
            );
        }
        if input_scales.len() != blocks_per_row
            || input_quants.len() != blocks_per_row * 32
            || weight_bytes.len() != rows * blocks_per_row * 36
            || output.len() != rows
        {
            return false;
        }
        let Some(b) = backend() else {
            return false;
        };
        let guard = b.lock().expect("cuda backend mutex poisoned");
        match run_block_inner(
            &guard,
            input_scales,
            input_quants,
            weight_bytes,
            rows,
            blocks_per_row,
            output,
        ) {
            Ok(()) => {
                RUN_COUNT.fetch_add(1, Ordering::Relaxed);
                if !LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "[cuda] Q8_0 hybrid decode active on {} — first GPU matmul completed",
                        guard.device_name
                    );
                }
                true
            }
            Err(_) => false,
        }
    }

    fn run_block_inner(
        b: &CudaBackend,
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> Result<(), cudarc::driver::DriverError> {
        let stream = &b.stream;
        let d_in_scales = stream.clone_htod(input_scales)?;
        let d_in_quants = stream.clone_htod(input_quants)?;
        let d_w_bytes = stream.clone_htod(weight_bytes)?;
        let mut d_out = stream.alloc_zeros::<f32>(output.len())?;

        let block_dim = 256u32;
        let grid_dim = (rows as u32).div_ceil(block_dim);
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let rows_i = rows as i32;
        let blocks_per_row_i = blocks_per_row as i32;
        let mut builder = stream.launch_builder(&b.kernel_block);
        builder
            .arg(&d_in_scales)
            .arg(&d_in_quants)
            .arg(&d_w_bytes)
            .arg(&rows_i)
            .arg(&blocks_per_row_i)
            .arg(&mut d_out);
        // SAFETY: buffers are sized to the launch dimensions per the shape
        // guards in `run_block`.
        unsafe { builder.launch(cfg)? };
        stream.memcpy_dtoh(&d_out, output)?;
        b.ctx.synchronize()?;
        Ok(())
    }

    /// Decode-shaped Q8_0 linear over the in-memory `Q8_0Block` byte layout.
    pub fn try_q8_0_block_linear_row(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_block_bytes: &[u8],
        rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        run_block(
            input_scales,
            input_quants,
            weight_block_bytes,
            rows,
            blocks_per_row,
            output,
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // Reference dot, identical in operation order to the CPU engine's
        // `dot_q8_0_encoded_row_with_scales`: exact i32 block dot, then a
        // sequential f32 accumulation of `(int_sum as f32) * w_scale * i_scale`
        // in block order, no FMA. The GPU kernel must match this bit-for-bit.
        fn reference_row(
            input_scales: &[f32],
            input_quants: &[i8],
            weight_block_quants: &[&[i8]],
            weight_scales: &[f32],
            blocks_per_row: usize,
        ) -> f32 {
            let mut sum = 0.0f32;
            for b in 0..blocks_per_row {
                let mut int_sum = 0i32;
                for j in 0..32 {
                    int_sum +=
                        i32::from(weight_block_quants[b][j]) * i32::from(input_quants[b * 32 + j]);
                }
                sum += int_sum as f32 * weight_scales[b] * input_scales[b];
            }
            sum
        }

        // Tiny deterministic LCG so the test needs no rand dependency and is
        // reproducible across runs/platforms.
        struct Lcg(u64);
        impl Lcg {
            fn next_u32(&mut self) -> u32 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (self.0 >> 32) as u32
            }
            fn next_i8(&mut self) -> i8 {
                (self.next_u32() & 0xff) as u8 as i8
            }
            fn next_scale(&mut self) -> f32 {
                // Small positive f16-ish scales, like real Q8_0 block scales.
                ((self.next_u32() % 1000) as f32 + 1.0) / 4096.0
            }
        }

        // Requires a CUDA device; ignored by default so GPU-less CI (which has
        // no NVIDIA driver) compiles but does not run it. Run on a CUDA host
        // with `cargo test --features cuda -- --ignored`.
        #[test]
        #[ignore = "requires a CUDA device"]
        fn cuda_q8_kernel_is_bit_identical_to_cpu_reference() {
            if !detect_cuda_device().available {
                eprintln!("skipping: no CUDA device available");
                return;
            }
            let blocks_per_row = 64usize; // TinyLlama hidden 2048 = 64 blocks of 32
            let weight_rows = 300usize; // not a multiple of the 256 block size
            let mut rng = Lcg(0x1234_5678_9abc_def0);

            // Input row.
            let mut input_quants = vec![0i8; blocks_per_row * 32];
            for q in input_quants.iter_mut() {
                *q = rng.next_i8();
            }
            let input_scales: Vec<f32> = (0..blocks_per_row).map(|_| rng.next_scale()).collect();

            // Weight rows: 34-byte blocks (2-byte scale header + 32 quants) plus
            // a separately decoded f32 scale per block (as the engine passes).
            let mut weight_bytes = vec![0u8; weight_rows * blocks_per_row * 34];
            let mut weight_scales = vec![0f32; weight_rows * blocks_per_row];
            for r in 0..weight_rows {
                for b in 0..blocks_per_row {
                    let blk = (r * blocks_per_row + b) * 34;
                    // header bytes are ignored by the kernel; fill with noise.
                    weight_bytes[blk] = rng.next_i8() as u8;
                    weight_bytes[blk + 1] = rng.next_i8() as u8;
                    for j in 0..32 {
                        weight_bytes[blk + 2 + j] = rng.next_i8() as u8;
                    }
                    weight_scales[r * blocks_per_row + b] = rng.next_scale();
                }
            }

            // CPU reference.
            let mut expected = vec![0f32; weight_rows];
            for (r, slot) in expected.iter_mut().enumerate() {
                let block_quants: Vec<&[i8]> = (0..blocks_per_row)
                    .map(|b| {
                        let start = (r * blocks_per_row + b) * 34 + 2;
                        // reinterpret the u8 quants as i8
                        let bytes = &weight_bytes[start..start + 32];
                        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), 32) }
                    })
                    .collect();
                *slot = reference_row(
                    &input_scales,
                    &input_quants,
                    &block_quants,
                    &weight_scales[r * blocks_per_row..(r + 1) * blocks_per_row],
                    blocks_per_row,
                );
            }

            // GPU.
            let mut got = vec![0f32; weight_rows];
            let ok = try_q8_0_encoded_linear_row(
                &input_scales,
                &input_quants,
                &weight_bytes,
                &weight_scales,
                weight_rows,
                blocks_per_row,
                &mut got,
            );
            assert!(ok, "GPU kernel did not run");

            let mut mismatches = 0;
            for (r, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                if g.to_bits() != e.to_bits() {
                    if mismatches < 5 {
                        eprintln!(
                            "row {r}: gpu={g} ({:#010x}) cpu={e} ({:#010x})",
                            g.to_bits(),
                            e.to_bits()
                        );
                    }
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "{mismatches}/{weight_rows} rows differ bit-for-bit from the CPU reference"
            );
        }

        #[test]
        #[ignore = "requires a CUDA device"]
        fn cuda_block_kernel_is_bit_identical_to_cpu_reference() {
            if !detect_cuda_device().available {
                eprintln!("skipping: no CUDA device available");
                return;
            }
            let blocks_per_row = 64usize;
            let weight_rows = 257usize;
            let mut rng = Lcg(0x0fed_cba9_8765_4321);

            // Single input row, as separate scales + quants (what the engine
            // passes via with_q8_0_block_scales_and_quants).
            let mut input_quants = vec![0i8; blocks_per_row * 32];
            for q in input_quants.iter_mut() {
                *q = rng.next_i8();
            }
            let input_scales: Vec<f32> = (0..blocks_per_row).map(|_| rng.next_scale()).collect();

            // Weight rows in the Q8_0Block byte layout: 36 bytes/block =
            // f32 scale (LE) + 32 i8 quants.
            let mut weight_bytes = vec![0u8; weight_rows * blocks_per_row * 36];
            let mut weight_scales = vec![0f32; weight_rows * blocks_per_row];
            for r in 0..weight_rows {
                for b in 0..blocks_per_row {
                    let blk = (r * blocks_per_row + b) * 36;
                    let scale = rng.next_scale();
                    weight_scales[r * blocks_per_row + b] = scale;
                    weight_bytes[blk..blk + 4].copy_from_slice(&scale.to_le_bytes());
                    for j in 0..32 {
                        weight_bytes[blk + 4 + j] = rng.next_i8() as u8;
                    }
                }
            }

            // CPU reference (same op order as q8_0_dot_rows scalar path).
            let mut expected = vec![0f32; weight_rows];
            for (r, slot) in expected.iter_mut().enumerate() {
                let block_quants: Vec<&[i8]> = (0..blocks_per_row)
                    .map(|b| {
                        let start = (r * blocks_per_row + b) * 36 + 4;
                        let bytes = &weight_bytes[start..start + 32];
                        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), 32) }
                    })
                    .collect();
                *slot = reference_row(
                    &input_scales,
                    &input_quants,
                    &block_quants,
                    &weight_scales[r * blocks_per_row..(r + 1) * blocks_per_row],
                    blocks_per_row,
                );
            }

            let mut got = vec![0f32; weight_rows];
            let ok = try_q8_0_block_linear_row(
                &input_scales,
                &input_quants,
                &weight_bytes,
                weight_rows,
                blocks_per_row,
                &mut got,
            );
            assert!(ok, "GPU block kernel did not run");

            let mut mismatches = 0;
            for (g, e) in got.iter().zip(expected.iter()) {
                if g.to_bits() != e.to_bits() {
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "{mismatches}/{weight_rows} block-kernel rows differ bit-for-bit from CPU reference"
            );
        }
    }
}
