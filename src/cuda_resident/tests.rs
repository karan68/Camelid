//! Per-kernel parity tests for the resident-decode kernels. Each test runs one
//! kernel on the GPU and compares to a small CPU reference, so a divergence is
//! isolated to a single kernel. All require a CUDA device (`#[ignore]`d in
//! GPU-less CI); run with `cargo test --features cuda -- --ignored`.

use super::{CudaResidentDecode, CudaResidentKernels};
use cudarc::driver::{LaunchConfig, PushKernelArg};

// f16 round-trip matching the engine.
fn f16rt(x: f32) -> f32 {
    crate::inference::f16_bits_to_f32(crate::inference::f32_to_f16_bits(x))
}

// Quantize an f32 weight tensor [rows*k] to rows*(k/32) Q8_0 36-byte blocks
// (f32 scale LE + 32 i8 quants), the layout the GPU GEMV reads.
fn quantize_blocks(w: &[f32], k: usize) -> Vec<u8> {
    let n_blocks = w.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * 36);
    let _ = k;
    for b in 0..n_blocks {
        let blk = &w[b * 32..b * 32 + 32];
        let max_abs = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let unrounded = max_abs / 127.0;
        let scale = f16rt(unrounded);
        let inv = if unrounded == 0.0 {
            0.0
        } else {
            1.0 / unrounded
        };
        out.extend_from_slice(&scale.to_le_bytes());
        for &x in blk {
            let q = (x * inv).round_ties_even().clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}

// Quantize an activation row to per-block (scale, quants).
fn quantize_row(x: &[f32]) -> (Vec<f32>, Vec<i8>) {
    let nb = x.len() / 32;
    let mut scales = vec![0f32; nb];
    let mut quants = vec![0i8; x.len()];
    for b in 0..nb {
        let blk = &x[b * 32..b * 32 + 32];
        let max_abs = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let unrounded = max_abs / 127.0;
        scales[b] = f16rt(unrounded);
        let inv = if unrounded == 0.0 {
            0.0
        } else {
            1.0 / unrounded
        };
        for (j, &xv) in blk.iter().enumerate() {
            quants[b * 32 + j] = (xv * inv).round_ties_even().clamp(-128.0, 127.0) as i8;
        }
    }
    (scales, quants)
}

// CPU reference Q8 matmul: quantized input row dotted against Q8 weight blocks,
// rows outputs. Sequential block accumulation (the CPU engine's order).
fn cpu_q8_dot(in_s: &[f32], in_q: &[i8], wblocks: &[u8], rows: usize, bpr: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows];
    for (r, slot) in out.iter_mut().enumerate() {
        let mut sum = 0f32;
        for b in 0..bpr {
            let blk = (r * bpr + b) * 36;
            let ws = f32::from_le_bytes(wblocks[blk..blk + 4].try_into().unwrap());
            let mut int_sum = 0i32;
            for j in 0..32 {
                let wq = wblocks[blk + 4 + j] as i8;
                int_sum += i32::from(wq) * i32::from(in_q[b * 32 + j]);
            }
            sum += int_sum as f32 * ws * in_s[b];
        }
        *slot = sum;
    }
    out
}

fn cpu_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w).map(|(v, wv)| v * scale * wv).collect()
}

fn cpu_rope(vec: &mut [f32], cos: &[f32], sin: &[f32], n_heads: usize, head_dim: usize) {
    let pairs = cos.len();
    for head in 0..n_heads {
        for p in 0..pairs {
            let (c, s) = (cos[p], sin[p]);
            let d0 = head * head_dim + 2 * p;
            let (x0, x1) = (vec[d0], vec[d0 + 1]);
            vec[d0] = x0 * c - x1 * s;
            vec[d0 + 1] = x0 * s + x1 * c;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_attention(
    q: &[f32],
    ck: &[f32],
    cv: &[f32],
    pos_count: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    max_pos: usize,
    scale: f32,
) -> Vec<f32> {
    let repeats = n_heads / n_kv;
    let mut out = vec![0f32; n_heads * head_dim];
    for head in 0..n_heads {
        let kv = head / repeats;
        let qh = &q[head * head_dim..head * head_dim + head_dim];
        let mut scores = vec![0f32; pos_count];
        for (p, sc) in scores.iter_mut().enumerate() {
            let base = (kv * max_pos + p) * head_dim;
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += qh[d] * ck[base + d];
            }
            *sc = dot * scale;
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
            for (p, sc) in scores.iter().enumerate() {
                acc += (sc * inv) * cv[(kv * max_pos + p) * head_dim + d];
            }
            out[head * head_dim + d] = acc;
        }
    }
    out
}

#[test]
#[ignore = "requires a CUDA device"]
fn full_forward_token_matches_cpu() {
    let Some(_k) = kernels() else {
        return;
    };
    // Tiny Llama-shaped model.
    let n_layers = 2usize;
    let hidden = 64usize;
    let n_heads = 2usize;
    let n_kv = 1usize;
    let head_dim = 32usize;
    let rope_dim = 32usize;
    let ffn = 128usize;
    let vocab = 96usize;
    let max_pos = 16usize;
    let eps = 1e-5f32;
    let base = 10000f32;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut rng = Lcg(0xabcdef);
    let rand = |rng: &mut Lcg, n: usize| (0..n).map(|_| rng.next_f32()).collect::<Vec<f32>>();

    // Per-layer f32 weights, quantized to blocks (the same blocks feed CPU + GPU).
    struct LayerF {
        q: Vec<u8>,
        k: Vec<u8>,
        v: Vec<u8>,
        o: Vec<u8>,
        gate: Vec<u8>,
        up: Vec<u8>,
        down: Vec<u8>,
        an: Vec<f32>,
        fnv: Vec<f32>,
    }
    let mut layers = Vec::new();
    for _ in 0..n_layers {
        layers.push(LayerF {
            q: quantize_blocks(&rand(&mut rng, q_width * hidden), hidden),
            k: quantize_blocks(&rand(&mut rng, kv_width * hidden), hidden),
            v: quantize_blocks(&rand(&mut rng, kv_width * hidden), hidden),
            o: quantize_blocks(&rand(&mut rng, hidden * q_width), q_width),
            gate: quantize_blocks(&rand(&mut rng, ffn * hidden), hidden),
            up: quantize_blocks(&rand(&mut rng, ffn * hidden), hidden),
            down: quantize_blocks(&rand(&mut rng, hidden * ffn), ffn),
            an: rand(&mut rng, hidden)
                .iter()
                .map(|v| v * 0.2 + 1.0)
                .collect(),
            fnv: rand(&mut rng, hidden)
                .iter()
                .map(|v| v * 0.2 + 1.0)
                .collect(),
        });
    }
    let final_norm: Vec<f32> = rand(&mut rng, hidden)
        .iter()
        .map(|v| v * 0.2 + 1.0)
        .collect();
    let output_w = quantize_blocks(&rand(&mut rng, vocab * hidden), hidden);

    // Build the GPU engine.
    let mut engine = CudaResidentDecode::new(
        n_layers, n_heads, n_kv, head_dim, hidden, ffn, rope_dim, max_pos, vocab, eps,
    )
    .unwrap();
    for l in &layers {
        engine
            .set_layer(
                &l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down, &l.an, &l.fnv,
            )
            .unwrap();
    }
    engine.set_output(&final_norm, &output_w).unwrap();

    // CPU reference KV cache, layout [kv_head][position][head_dim] per layer.
    let mut cpu_k = vec![vec![0f32; kv_width * max_pos]; n_layers];
    let mut cpu_v = vec![vec![0f32; kv_width * max_pos]; n_layers];

    let pairs = rope_dim / 2;
    for position in 0..4usize {
        let emb = rand(&mut rng, hidden);
        let cos: Vec<f32> = (0..pairs)
            .map(|p| {
                let theta = base.powf(-(2.0 * p as f32) / rope_dim as f32);
                (position as f32 * theta).cos()
            })
            .collect();
        let sin: Vec<f32> = (0..pairs)
            .map(|p| {
                let theta = base.powf(-(2.0 * p as f32) / rope_dim as f32);
                (position as f32 * theta).sin()
            })
            .collect();

        // CPU reference forward
        let mut hidden_v = emb.clone();
        for (li, l) in layers.iter().enumerate() {
            let normed = cpu_rmsnorm(&hidden_v, &l.an, eps);
            let (is, iq) = quantize_row(&normed);
            let mut q = cpu_q8_dot(&is, &iq, &l.q, q_width, hidden / 32);
            let mut kv_k = cpu_q8_dot(&is, &iq, &l.k, kv_width, hidden / 32);
            let kv_v = cpu_q8_dot(&is, &iq, &l.v, kv_width, hidden / 32);
            cpu_rope(&mut q, &cos, &sin, n_heads, head_dim);
            cpu_rope(&mut kv_k, &cos, &sin, n_kv, head_dim);
            for kv in 0..n_kv {
                for d in 0..head_dim {
                    cpu_k[li][(kv * max_pos + position) * head_dim + d] =
                        f16rt(kv_k[kv * head_dim + d]);
                    cpu_v[li][(kv * max_pos + position) * head_dim + d] =
                        f16rt(kv_v[kv * head_dim + d]);
                }
            }
            let ctx = cpu_attention(
                &q,
                &cpu_k[li],
                &cpu_v[li],
                position + 1,
                n_heads,
                n_kv,
                head_dim,
                max_pos,
                scale,
            );
            let (cs, cq) = quantize_row(&ctx);
            let o = cpu_q8_dot(&cs, &cq, &l.o, hidden, q_width / 32);
            for i in 0..hidden {
                hidden_v[i] += o[i];
            }
            let fnormed = cpu_rmsnorm(&hidden_v, &l.fnv, eps);
            let (fs, fq) = quantize_row(&fnormed);
            let gate = cpu_q8_dot(&fs, &fq, &l.gate, ffn, hidden / 32);
            let up = cpu_q8_dot(&fs, &fq, &l.up, ffn, hidden / 32);
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
                .collect();
            let (as_, aq) = quantize_row(&act);
            let down = cpu_q8_dot(&as_, &aq, &l.down, hidden, ffn / 32);
            for i in 0..hidden {
                hidden_v[i] += down[i];
            }
        }
        let fnormed = cpu_rmsnorm(&hidden_v, &final_norm, eps);
        let (s, qq) = quantize_row(&fnormed);
        let logits = cpu_q8_dot(&s, &qq, &output_w, vocab, hidden / 32);
        let cpu_tok = logits
            .iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                },
            )
            .0 as u32;

        let gpu_tok = engine
            .forward_token(&emb, &cos, &sin, position, scale, true)
            .unwrap()
            .unwrap();
        assert_eq!(
            gpu_tok, cpu_tok,
            "token mismatch at position {position}: gpu={gpu_tok} cpu={cpu_tok}"
        );
    }
}

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
        // Stages the full row in shared for the in-order sum (matches launch_rmsnorm).
        shared_mem_bytes: (n as u32) * 4,
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
fn gemm_batched_matches_per_token() {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 96usize;
    let bpr = 4usize; // K_dim = 128
    let kdim = bpr * 32;
    let ktok = 4usize;
    let mut rng = Lcg(7);
    // Weight [rows*kdim] -> 36-byte Q8 blocks -> SoA layout the kernel reads.
    let w: Vec<f32> = (0..rows * kdim).map(|_| rng.next_f32()).collect();
    let wblocks = quantize_blocks(&w, kdim);
    let wsoa = super::repack_q8_soa(&wblocks);
    // K inputs laid out [token][block]; CPU reference per token.
    let mut in_s = vec![0f32; ktok * bpr];
    let mut in_q = vec![0i8; ktok * kdim];
    let mut cpu_out = vec![0f32; ktok * rows];
    for t in 0..ktok {
        let x: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
        let (s, q) = quantize_row(&x);
        in_s[t * bpr..(t + 1) * bpr].copy_from_slice(&s);
        in_q[t * kdim..(t + 1) * kdim].copy_from_slice(&q);
        let r = cpu_q8_dot(&s, &q, &wblocks, rows, bpr);
        cpu_out[t * rows..(t + 1) * rows].copy_from_slice(&r);
    }
    let d_is = k.stream.clone_htod(&in_s).unwrap();
    let d_iq = k.stream.clone_htod(&in_q).unwrap();
    let d_w = k.stream.clone_htod(&wsoa).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(ktok * rows).unwrap();
    super::launch_gemm_batched(
        &k.stream,
        &k.gemm_batched,
        &d_is,
        &d_iq,
        &d_w,
        rows,
        bpr,
        ktok,
        &mut d_out,
    )
    .unwrap();
    let mut got = vec![0f32; ktok * rows];
    k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    // Tree reduction vs the CPU's sequential block sum -> close, not bit-exact.
    assert!(close(&got, &cpu_out, 1e-3), "batched gemm diverged");
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
