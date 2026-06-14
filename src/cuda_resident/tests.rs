//! Per-kernel parity tests for the resident-decode kernels. Each test runs one
//! kernel on the GPU and compares to a small CPU reference, so a divergence is
//! isolated to a single kernel. All require a CUDA device (`#[ignore]`d in
//! GPU-less CI); run with `cargo test --features cuda -- --ignored`.

use super::CudaResidentKernels;
use cudarc::driver::{LaunchConfig, PushKernelArg};

// Deterministic LCG so the tests need no rand dependency.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 // [-1, 1)
    }
}

fn kernels() -> Option<CudaResidentKernels> {
    CudaResidentKernels::new().ok()
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.iter().zip(b).all(|(x, y)| {
        let d = (x - y).abs() / y.abs().max(1.0);
        d < tol
    })
}

#[test]
#[ignore = "requires a CUDA device"]
fn rms_norm_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n = 2048usize;
    let eps = 1e-5f32;
    let mut rng = Lcg(1);
    let x: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let w: Vec<f32> = (0..n).map(|_| rng.next_f32() * 0.5 + 1.0).collect();
    // CPU reference
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    let expected: Vec<f32> = x.iter().zip(&w).map(|(v, wv)| v * scale * wv).collect();
    // GPU
    let dx = k.stream.clone_htod(&x).unwrap();
    let dw = k.stream.clone_htod(&w).unwrap();
    let mut dout = k.stream.alloc_zeros::<f32>(n).unwrap();
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 4,
    };
    let n_i = n as i32;
    let mut b = k.stream.launch_builder(&k.rms_norm);
    b.arg(&dx).arg(&dw).arg(&mut dout).arg(&n_i).arg(&eps);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0f32; n];
    k.stream.memcpy_dtoh(&dout, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got, &expected, 1e-4), "rms_norm diverged");
}

#[test]
#[ignore = "requires a CUDA device"]
fn quantize_q8_0_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n_blocks = 64usize;
    let n = n_blocks * 32;
    let mut rng = Lcg(7);
    let x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 3.0).collect();
    // CPU reference (quantize_q8_0_block): f16-rounded scale, unrounded inverse,
    // round-half-to-even, clamp [-128,127].
    let mut exp_scales = vec![0f32; n_blocks];
    let mut exp_quants = vec![0i8; n];
    for bidx in 0..n_blocks {
        let blk = &x[bidx * 32..bidx * 32 + 32];
        let max_abs = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let unrounded = max_abs / 127.0;
        exp_scales[bidx] =
            crate::inference::f16_bits_to_f32(crate::inference::f32_to_f16_bits(unrounded));
        let inv = if unrounded == 0.0 {
            0.0
        } else {
            1.0 / unrounded
        };
        for j in 0..32 {
            let v = (blk[j] * inv).round_ties_even().clamp(-128.0, 127.0);
            exp_quants[bidx * 32 + j] = v as i8;
        }
    }
    // GPU
    let dx = k.stream.clone_htod(&x).unwrap();
    let mut dq = k.stream.alloc_zeros::<i8>(n).unwrap();
    let mut ds = k.stream.alloc_zeros::<f32>(n_blocks).unwrap();
    let block = 64u32;
    let cfg = LaunchConfig {
        grid_dim: ((n_blocks as u32).div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let nb_i = n_blocks as i32;
    let mut b = k.stream.launch_builder(&k.quantize);
    b.arg(&dx).arg(&mut dq).arg(&mut ds).arg(&nb_i);
    unsafe { b.launch(cfg).unwrap() };
    let mut gq = vec![0i8; n];
    let mut gs = vec![0f32; n_blocks];
    k.stream.memcpy_dtoh(&dq, &mut gq).unwrap();
    k.stream.memcpy_dtoh(&ds, &mut gs).unwrap();
    k.ctx.synchronize().unwrap();
    assert_eq!(gq, exp_quants, "quantize quants diverged");
    assert!(close(&gs, &exp_scales, 1e-6), "quantize scales diverged");
}

#[test]
#[ignore = "requires a CUDA device"]
fn rope_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n_heads = 4usize;
    let head_dim = 64usize;
    let rope_dim = 64usize;
    let base = 10000f32;
    let position = 13usize;
    let mut rng = Lcg(3);
    let vec: Vec<f32> = (0..n_heads * head_dim).map(|_| rng.next_f32()).collect();
    // cos/sin tables per pair
    let pairs = rope_dim / 2;
    let cos_t: Vec<f32> = (0..pairs)
        .map(|p| {
            let theta = base.powf(-(2.0 * p as f32) / rope_dim as f32);
            (position as f32 * theta).cos()
        })
        .collect();
    let sin_t: Vec<f32> = (0..pairs)
        .map(|p| {
            let theta = base.powf(-(2.0 * p as f32) / rope_dim as f32);
            (position as f32 * theta).sin()
        })
        .collect();
    // CPU reference: adjacent-even-odd forward
    let mut expected = vec.clone();
    for head in 0..n_heads {
        for p in 0..pairs {
            let (c, s) = (cos_t[p], sin_t[p]);
            let d0 = head * head_dim + 2 * p;
            let d1 = d0 + 1;
            let x0 = vec[d0];
            let x1 = vec[d1];
            expected[d0] = x0 * c - x1 * s;
            expected[d1] = x0 * s + x1 * c;
        }
    }
    // GPU
    let mut dvec = k.stream.clone_htod(&vec).unwrap();
    let dcos = k.stream.clone_htod(&cos_t).unwrap();
    let dsin = k.stream.clone_htod(&sin_t).unwrap();
    let (nh, hd, rd) = (n_heads as i32, head_dim as i32, rope_dim as i32);
    let total = (n_heads * pairs) as u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = k.stream.launch_builder(&k.rope);
    b.arg(&mut dvec)
        .arg(&dcos)
        .arg(&dsin)
        .arg(&nh)
        .arg(&hd)
        .arg(&rd);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0f32; n_heads * head_dim];
    k.stream.memcpy_dtoh(&dvec, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got, &expected, 1e-5), "rope diverged");
}

#[test]
#[ignore = "requires a CUDA device"]
fn silu_mul_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n = 5632usize;
    let mut rng = Lcg(5);
    let gate: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0).collect();
    let up: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let expected: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    let dg = k.stream.clone_htod(&gate).unwrap();
    let du = k.stream.clone_htod(&up).unwrap();
    let mut dout = k.stream.alloc_zeros::<f32>(n).unwrap();
    let n_i = n as i32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = k.stream.launch_builder(&k.silu_mul);
    b.arg(&dg).arg(&du).arg(&mut dout).arg(&n_i);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0f32; n];
    k.stream.memcpy_dtoh(&dout, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got, &expected, 1e-5), "silu_mul diverged");
}

#[test]
#[ignore = "requires a CUDA device"]
fn argmax_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n = 32000usize;
    let mut rng = Lcg(9);
    let mut logits: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    logits[12345] = 5.0; // clear winner
                         // CPU strict-> scan
    let mut best = logits[0];
    let mut besti = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if *v > best {
            best = *v;
            besti = i;
        }
    }
    let dl = k.stream.clone_htod(&logits).unwrap();
    let mut didx = k.stream.alloc_zeros::<u32>(1).unwrap();
    let block = 256u32;
    let n_i = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 8, // f32 val + i32 idx per thread
    };
    let mut b = k.stream.launch_builder(&k.argmax);
    b.arg(&dl).arg(&n_i).arg(&mut didx);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0u32; 1];
    k.stream.memcpy_dtoh(&didx, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert_eq!(got[0] as usize, besti, "argmax diverged");
}

#[test]
#[ignore = "requires a CUDA device"]
fn attention_decode_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n_heads = 32usize;
    let n_kv = 4usize;
    let head_dim = 64usize;
    let max_pos = 128usize;
    let position_count = 40usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let repeats = n_heads / n_kv;
    let mut rng = Lcg(11);
    let q: Vec<f32> = (0..n_heads * head_dim).map(|_| rng.next_f32()).collect();
    let mut cache_k = vec![0f32; n_kv * max_pos * head_dim];
    let mut cache_v = vec![0f32; n_kv * max_pos * head_dim];
    for kv in 0..n_kv {
        for p in 0..position_count {
            for d in 0..head_dim {
                cache_k[(kv * max_pos + p) * head_dim + d] = rng.next_f32();
                cache_v[(kv * max_pos + p) * head_dim + d] = rng.next_f32();
            }
        }
    }
    // CPU reference
    let mut expected = vec![0f32; n_heads * head_dim];
    for head in 0..n_heads {
        let kv_head = head / repeats;
        let qh = &q[head * head_dim..head * head_dim + head_dim];
        let mut scores = vec![0f32; position_count];
        for p in 0..position_count {
            let kbase = (kv_head * max_pos + p) * head_dim;
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += qh[d] * cache_k[kbase + d];
            }
            scores[p] = dot * scale;
        }
        let m = scores.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            sum += *s;
        }
        let inv = 1.0 / sum;
        for d in 0..head_dim {
            let mut acc = 0f32;
            for p in 0..position_count {
                acc += (scores[p] * inv) * cache_v[(kv_head * max_pos + p) * head_dim + d];
            }
            expected[head * head_dim + d] = acc;
        }
    }
    // GPU
    let dq = k.stream.clone_htod(&q).unwrap();
    let dk = k.stream.clone_htod(&cache_k).unwrap();
    let dv = k.stream.clone_htod(&cache_v).unwrap();
    let mut dout = k.stream.alloc_zeros::<f32>(n_heads * head_dim).unwrap();
    let (nh, nkv, hd, pc, mp) = (
        n_heads as i32,
        n_kv as i32,
        head_dim as i32,
        position_count as i32,
        max_pos as i32,
    );
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: ((head_dim + position_count) * 4) as u32,
    };
    let mut b = k.stream.launch_builder(&k.attention);
    b.arg(&dq)
        .arg(&dk)
        .arg(&dv)
        .arg(&mut dout)
        .arg(&nh)
        .arg(&nkv)
        .arg(&hd)
        .arg(&pc)
        .arg(&mp)
        .arg(&scale);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0f32; n_heads * head_dim];
    k.stream.memcpy_dtoh(&dout, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got, &expected, 1e-4), "attention diverged");
}
