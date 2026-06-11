//! Reference-order float math for the DiffusionGemma lane (Phase 2, option A).
//!
//! Every function here reproduces, operation-for-operation, the float
//! semantics of the kernel the pinned llama.cpp build ACTUALLY EXECUTES on
//! this machine's CPU path for the encoder graph (Apple M-series: NEON +
//! dotprod + i8mm, no SVE; GGML_LLAMAFILE=ON; `use_extra_bufts=false`).
//! Sources are cited per function against the pinned commit. The pin proved
//! bit-deterministic across thread counts, so matching these orders exactly
//! is what makes bit-exact checkpoint parity achievable.
//!
//! Scalar Rust ports of SIMD code keep the lane structure explicit (arrays of
//! 4 lanes) because lane grouping and reduction trees determine the rounding.
//! `f32::mul_add` lowers to the same fused multiply-add the NEON `vfmaq`
//! family executes on aarch64.

// index-based lane loops intentionally mirror the reference SIMD structure
#![allow(clippy::needless_range_loop)]

use crate::tensor::f16_round;

/// `ggml_compute_forward_rms_norm_f32` (ops.cpp): the sum of squares
/// accumulates SEQUENTIALLY IN DOUBLE, `mean = sum/ne` collapses to f32, and
/// the output is `x[i] * scale * w[i]` (the fused-mul order; the unfused
/// graph computes the same values).
pub(crate) fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut sum = 0f64;
    for &v in x {
        sum += (v * v) as f64;
    }
    let mean = (sum / x.len() as f64) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();
    match weight {
        Some(w) => x.iter().zip(w).map(|(&v, &wv)| v * scale * wv).collect(),
        None => x.iter().map(|&v| v * scale).collect(),
    }
}

/// `ggml_vec_dot_f32` (vec.cpp, the non-SVE GGML_SIMD branch on aarch64):
/// 4 four-lane FMA accumulators striding 16 elements, reduced pairwise
/// (acc0+=acc2, acc1+=acc3, acc0+=acc1, then vaddvq's (l0+l1)+(l2+l3)),
/// with plain-multiply scalar leftovers added afterwards.
pub(crate) fn vec_dot_f32(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let np = n & !15;
    let mut acc = [[0f32; 4]; 4];
    let mut i = 0;
    while i < np {
        for (j, accj) in acc.iter_mut().enumerate() {
            let o = i + j * 4;
            for l in 0..4 {
                accj[l] = x[o + l].mul_add(y[o + l], accj[l]);
            }
        }
        i += 16;
    }
    for l in 0..4 {
        acc[0][l] += acc[2][l];
        acc[1][l] += acc[3][l];
        acc[0][l] += acc[1][l];
    }
    let mut sumf = (acc[0][0] + acc[0][1]) + (acc[0][2] + acc[0][3]);
    for k in np..n {
        sumf += x[k] * y[k];
    }
    sumf
}

/// `ggml_v_expf` (vec.h, the ARM NEON variant — an Arm-limited-routine
/// polynomial). Per-lane scalar port: the reference's group-wide fast path
/// (`!vpaddd(c)` → `k + j*k`) computes the same product as the per-lane
/// fallback (`k + k*j`), so lane grouping does not change values; only the
/// per-lane special cases (`|n| > 126`, `|n| > 192`) matter. All FMAs are
/// fused, mirroring vfmaq/vfmsq; constants are exact bit patterns of the
/// reference's hex-float literals.
pub(crate) fn v_expf_lanes(x: [f32; 4]) -> [f32; 4] {
    let r = f32::from_bits(0x4b40_0000); // 0x1.8p23
    let inv_ln2 = f32::from_bits(0x3fb8_aa3b); // 0x1.715476p+0
    let ln2_hi = f32::from_bits(0x3f31_7200); // 0x1.62e4p-1
    let ln2_lo = f32::from_bits(0x35bf_be8e); // 0x1.7f7d1cp-20
    let c_fffdb6 = f32::from_bits(0x3eff_fedb); // 0x1.fffdb6p-2
    let c_555e66 = f32::from_bits(0x3e2a_af33); // 0x1.555e66p-3
    let c_573e2e = f32::from_bits(0x3d2b_9f17); // 0x1.573e2ep-5
    let c_0e4020 = f32::from_bits(0x3c07_2010); // 0x1.0e4020p-7
    let c_ffffec = f32::from_bits(0x3f7f_fff6); // 0x1.ffffecp-1

    let mut out = [0f32; 4];
    for (l, &xl) in x.iter().enumerate() {
        let z = xl.mul_add(inv_ln2, r);
        let n = z - r;
        // vfmsq_f32(a, b, c) = a - b*c, fused
        let b1 = (-n).mul_add(ln2_hi, xl);
        let b = (-n).mul_add(ln2_lo, b1);
        let e = z.to_bits() << 23;
        let k = f32::from_bits(e.wrapping_add(1.0f32.to_bits()));
        let c = n.abs() > 126.0;

        let u = b * b;
        let inner1 = c_555e66.mul_add(b, c_fffdb6);
        let inner2 = c_0e4020.mul_add(b, c_573e2e);
        let inner3 = inner2.mul_add(u, inner1);
        let j = inner3.mul_add(u, c_ffffec * b);

        out[l] = if !c {
            k.mul_add(j, k)
        } else {
            let d = if n <= 0.0 { 0x8200_0000u32 } else { 0 };
            let s1 = f32::from_bits(d.wrapping_add(0x7f00_0000));
            let s2 = f32::from_bits(e.wrapping_sub(d));
            if n.abs() > 192.0 {
                s1 * s1
            } else {
                s2.mul_add(j, s2) * s1
            }
        };
    }
    out
}

/// `ggml_compute_forward_soft_max_f32` + `ggml_vec_soft_max_f32` (no sinks,
/// scale 1.0): operates over the FULL row (masked entries already -inf from
/// the additive mask), max is order-free, exp runs in 4-lane `v_expf` groups
/// with each group's lane sum (`vaddvq`: (l0+l1)+(l2+l3)) added to a DOUBLE
/// accumulator, the tail uses libm `expf`, and the final normalization is
/// `(1.0/sum)` IN DOUBLE collapsed to f32 then multiplied per element.
pub(crate) fn softmax_row(row: &mut [f32]) {
    let n = row.len();
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    let mut i = 0;
    while i + 3 < n {
        let vals = v_expf_lanes([
            row[i] - max,
            row[i + 1] - max,
            row[i + 2] - max,
            row[i + 3] - max,
        ]);
        row[i..i + 4].copy_from_slice(&vals);
        sum += ((vals[0] + vals[1]) + (vals[2] + vals[3])) as f64;
        i += 4;
    }
    while i < n {
        let val = (row[i] - max).exp();
        sum += val as f64;
        row[i] = val;
        i += 1;
    }
    let s = (1.0f64 / sum) as f32;
    for v in row.iter_mut() {
        *v *= s;
    }
}

/// tinyBLAS f32 GEMM per-element semantics (sgemm.cpp `tinyBLAS<4,
/// float32x4_t,...>` on NEON): each output element is a 4-lane FMA
/// accumulator striding k by 4, reduced by vaddvq's pairwise tree. The
/// mnpack/tile machinery only groups elements; it never changes a single
/// element's accumulation order. Engages for f32 matmuls with m%4==0 and
/// n>=4 (the MoE router here); k must be 4-aligned.
pub(crate) fn tinyblas_f32_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.len().is_multiple_of(4));
    let mut acc = [0f32; 4];
    let mut i = 0;
    while i < a.len() {
        for l in 0..4 {
            acc[l] = a[i + l].mul_add(b[i + l], acc[l]);
        }
        i += 4;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// tinyBLAS_Q0_ARM per-element semantics (sgemm.cpp, dotprod): per Q8_0
/// block, lane L of the integer dot covers weight/activation bytes
/// {4L..4L+3} and {16+4L..16+4L+3} (the chained lo/hi vdotq), the lane
/// converts to f32 and FMA-accumulates scaled by `d_a*d_b`, and the final
/// reduction is vaddvq's pairwise tree. Engages for DENSE Q8_0 matmuls
/// (mul_mat; the MoE expert path uses vec_dot instead).
pub(crate) fn tinyblas_q8_0_dot(weight_wire: &[u8], input: &[crate::tensor::Q8_0Block]) -> f32 {
    const WIRE: usize = 34;
    let mut acc = [0f32; 4];
    for (i, y) in input.iter().enumerate() {
        let block = &weight_wire[i * WIRE..(i + 1) * WIRE];
        let d = crate::tensor::f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..34];
        let s = d * y.scale;
        for l in 0..4 {
            let mut lane = 0i32;
            for t in 0..4 {
                lane += (qs[4 * l + t] as i8 as i32) * (y.quants[4 * l + t] as i32);
                lane += (qs[16 + 4 * l + t] as i8 as i32) * (y.quants[16 + 4 * l + t] as i32);
            }
            acc[l] = (lane as f32).mul_add(s, acc[l]);
        }
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// ARM `quantize_row_q8_0` (arch/arm/quants.c): NEON amax (order-free),
/// `d = amax/127` with the int8s from `vcvtnq` — round to nearest EVEN (the
/// generic reference uses roundf/away; the ARM build does not) — and the
/// scale STORED as f16. The stored-scale rounding matters because the dot
/// kernels read it back.
pub(crate) fn quantize_q8_0_arm(x: &[f32]) -> Vec<crate::tensor::Q8_0Block> {
    debug_assert!(x.len().is_multiple_of(32));
    x.chunks_exact(32)
        .map(|chunk| {
            let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let mut quants = [0i8; 32];
            for (q, &v) in quants.iter_mut().zip(chunk) {
                *q = (v * id).round_ties_even() as i32 as i8;
            }
            crate::tensor::Q8_0Block {
                scale: f16_round(d),
                quants,
            }
        })
        .collect()
}

/// `ggml_rope_cache_init` + the NEOX `rotate_pairs` application
/// (ops.cpp): theta starts at the position and is multiplied CUMULATIVELY by
/// `theta_scale = freq_base^(-2/n_dims)` per pair (one powf then repeated
/// f32 multiplies — not a powf per pair), each pair's angle is
/// `freq_scale * (theta / freq_factor)` (freq_scale = 1 here), and the
/// rotation pairs (i, i+half) with libm sinf/cosf.
pub(crate) fn rope_neox(
    vec: &mut [f32],
    heads: usize,
    head_dim: usize,
    position: usize,
    freq_base: f32,
    factors: Option<&[f32]>,
) {
    let half = head_dim / 2;
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    // cache is per position, shared by all heads
    let mut cache = vec![0f32; head_dim];
    let mut theta = position as f32;
    for i in 0..half {
        let ff = factors.map_or(1.0, |f| f[i]);
        let angle = theta / ff;
        cache[2 * i] = angle.cos();
        cache[2 * i + 1] = angle.sin();
        theta *= theta_scale;
    }
    for h in 0..heads {
        let base = h * head_dim;
        for i in 0..half {
            let (c, s) = (cache[2 * i], cache[2 * i + 1]);
            let x0 = vec[base + i];
            let x1 = vec[base + half + i];
            vec[base + i] = x0 * c - x1 * s;
            vec[base + half + i] = x0 * s + x1 * c;
        }
    }
}

/// `ggml_vec_sum_f32`: sequential DOUBLE accumulation collapsed to f32
/// (used for the MoE selected-weight normalization sum).
pub(crate) fn vec_sum_f32(x: &[f32]) -> f32 {
    let mut sum = 0f64;
    for &v in x {
        sum += v as f64;
    }
    sum as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_dot_f32_tree_matches_naive_within_fp() {
        let x: Vec<f32> = (0..37).map(|i| ((i as f32) * 0.7).sin()).collect();
        let y: Vec<f32> = (0..37).map(|i| ((i as f32) * 1.3).cos()).collect();
        let tree = vec_dot_f32(&x, &y);
        let naive: f64 = x
            .iter()
            .zip(&y)
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();
        assert!((tree as f64 - naive).abs() < 1e-5);
    }

    #[test]
    fn v_expf_close_to_libm_in_normal_range() {
        for &v in &[-5.0f32, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 20.0] {
            let got = v_expf_lanes([v; 4])[0];
            let want = v.exp();
            let rel = ((got - want) / want).abs();
            assert!(rel < 1e-5, "v={v}: got {got} want {want}");
        }
        // -inf (a masked attention slot) must produce exactly 0
        assert_eq!(v_expf_lanes([f32::NEG_INFINITY; 4])[0], 0.0);
    }

    #[test]
    fn softmax_row_normalizes() {
        let mut row = vec![0.5f32, -1.0, 2.0, f32::NEG_INFINITY, 0.25, 1.5, -0.5];
        softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert_eq!(row[3], 0.0);
    }

    #[test]
    fn quantize_q8_0_arm_uses_nearest_even_and_f16_scale() {
        // values land exactly on .5 boundaries after scaling: 127 * (x/amax)
        let mut x = [0f32; 32];
        x[0] = 2.0; // amax → q = 127
        x[1] = 2.0 * (10.5 / 127.0); // scaled exactly 10.5 → nearest even = 10
        x[2] = 2.0 * (11.5 / 127.0); // scaled exactly 11.5 → nearest even = 12
        let blocks = quantize_q8_0_arm(&x);
        assert_eq!(blocks[0].quants[0], 127);
        assert_eq!(blocks[0].quants[1], 10, "ties round to even, not away");
        assert_eq!(blocks[0].quants[2], 12);
        let d = 2.0f32 / 127.0;
        assert_eq!(blocks[0].scale.to_bits(), f16_round(d).to_bits());
    }

    #[test]
    fn rope_cumulative_theta_differs_from_powf_only_marginally() {
        let mut v: Vec<f32> = (0..8).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let orig = v.clone();
        rope_neox(&mut v, 1, 8, 3, 10000.0, None);
        // rotation preserves pair norms
        for i in 0..4 {
            let n_before = orig[i] * orig[i] + orig[i + 4] * orig[i + 4];
            let n_after = v[i] * v[i] + v[i + 4] * v[i + 4];
            assert!((n_before - n_after).abs() < 1e-4);
        }
    }
}
