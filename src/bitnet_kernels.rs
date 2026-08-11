//! Cleanroom BitNet matrix kernels over the public I2_S wire contract.
//!
//! The linked Microsoft artifacts all store one canonical representation:
//! four ternary values per byte, interleaved in 128-value tiles, followed by a
//! tensor-wide f32 scale. Camelid keeps those bytes unchanged and offers three
//! execution strategies over them:
//!
//! - `i2_s`: direct ternary-by-A8 integer accumulation;
//! - `tl1`: two-weight lookup-table integer accumulation;
//! - `tl2`: three-weight lookup-table integer accumulation.
//!
//! These are independently implemented behavioral kernels. They do not ingest
//! BitNet.cpp's model-specific, pre-permuted TL1/TL2 GGUF layouts and do not use
//! its generated kernel headers.

use rayon::prelude::*;

use crate::{BackendError, Result};

pub(crate) const KERNEL_ENV: &str = "CAMELID_BITNET_KERNEL";
pub(crate) const GPU_ENV: &str = "CAMELID_BITNET_GPU";

pub(crate) fn gpu_allowed() -> bool {
    std::env::var(GPU_ENV)
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "disabled" | "no"
            )
        })
        .unwrap_or(true)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum BitNetKernelMode {
    #[default]
    Auto = 0,
    I2S = 1,
    Tl1 = 2,
    Tl2 = 3,
}

impl BitNetKernelMode {
    pub(crate) fn from_env() -> Self {
        let Ok(value) = std::env::var(KERNEL_ENV) else {
            return Self::Auto;
        };
        if let Some(mode) = Self::parse(&value) {
            return mode;
        }
        static INVALID_LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        INVALID_LOGGED.get_or_init(|| {
            eprintln!(
                "[bitnet] ignoring invalid {KERNEL_ENV}={value:?}; expected auto, i2_s, tl1, or tl2"
            );
        });
        Self::Auto
    }

    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Some(Self::Auto),
            "i2_s" | "i2s" | "direct" => Some(Self::I2S),
            "tl1" => Some(Self::Tl1),
            "tl2" => Some(Self::Tl2),
            _ => None,
        }
    }

    pub(crate) fn effective_cpu(self) -> Self {
        match self {
            Self::Auto => Self::I2S,
            explicit => explicit,
        }
    }

    pub(crate) fn gpu_code(self) -> u32 {
        match self.effective_cpu() {
            Self::I2S => 0,
            Self::Tl1 => 1,
            Self::Tl2 => 2,
            Self::Auto => unreachable!("auto is resolved above"),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self.effective_cpu() {
            Self::I2S => "i2_s",
            Self::Tl1 => "tl1",
            Self::Tl2 => "tl2",
            Self::Auto => unreachable!("auto is resolved above"),
        }
    }
}

struct I2SMatrix<'a> {
    packed: &'a [u8],
    scale: f32,
    input_width: usize,
    output_rows: usize,
}

impl<'a> I2SMatrix<'a> {
    fn new(bytes: &'a [u8], input_width: usize, output_rows: usize) -> Result<Self> {
        if input_width == 0 || output_rows == 0 {
            return Err(BackendError::InvalidTensorData(
                "I2_S kernel requires non-zero matrix dimensions".into(),
            ));
        }
        let elements = input_width.checked_mul(output_rows).ok_or_else(|| {
            BackendError::InvalidTensorData("I2_S matrix element count overflow".into())
        })?;
        if !elements.is_multiple_of(128) {
            return Err(BackendError::InvalidTensorData(format!(
                "I2_S matrix element count {elements} is not aligned to 128-value tiles"
            )));
        }
        let packed_len = elements / 4;
        let expected = packed_len.checked_add(32).ok_or_else(|| {
            BackendError::InvalidTensorData("I2_S matrix byte count overflow".into())
        })?;
        if bytes.len() != expected {
            return Err(BackendError::InvalidTensorData(format!(
                "I2_S matrix has {} bytes, expected {packed_len} packed bytes plus a 32-byte trailer",
                bytes.len()
            )));
        }
        let scale = f32::from_le_bytes(
            bytes[packed_len..packed_len + 4]
                .try_into()
                .expect("four-byte I2_S scale"),
        );
        if !scale.is_finite() {
            return Err(BackendError::InvalidTensorData(format!(
                "I2_S matrix scale must be finite, got {scale}"
            )));
        }
        Ok(Self {
            packed: &bytes[..packed_len],
            scale,
            input_width,
            output_rows,
        })
    }

    #[inline]
    fn ternary(&self, logical_index: usize) -> i8 {
        let tile = logical_index / 128;
        let within = logical_index % 128;
        let byte = self.packed[tile * 32 + within % 32];
        let shift = 6 - 2 * (within / 32);
        match (byte >> shift) & 3 {
            0 => -1,
            2 => 1,
            1 | 3 => 0,
            _ => unreachable!(),
        }
    }

    fn row_dot(
        &self,
        row: usize,
        input: &[i8],
        activation_scale: f32,
        mode: BitNetKernelMode,
    ) -> f32 {
        debug_assert!(row < self.output_rows);
        debug_assert_eq!(input.len(), self.input_width);
        let base = row * self.input_width;
        let unscaled = match mode.effective_cpu() {
            BitNetKernelMode::I2S => {
                let mut sum = 0_i32;
                for (column, value) in input.iter().copied().enumerate() {
                    sum += i32::from(self.ternary(base + column)) * i32::from(value);
                }
                sum
            }
            BitNetKernelMode::Tl1 => {
                let mut sum = 0_i32;
                let mut pairs = input.chunks_exact(2);
                for (pair_index, pair) in pairs.by_ref().enumerate() {
                    let column = pair_index * 2;
                    let left = usize::try_from(self.ternary(base + column) + 1)
                        .expect("ternary TL1 digit");
                    let right = usize::try_from(self.ternary(base + column + 1) + 1)
                        .expect("ternary TL1 digit");
                    let a = i32::from(pair[0]);
                    let b = i32::from(pair[1]);
                    let table = [-a - b, -a, -a + b, -b, 0, b, a - b, a, a + b];
                    sum += table[left * 3 + right];
                }
                let consumed = input.len() - pairs.remainder().len();
                for (tail, value) in pairs.remainder().iter().copied().enumerate() {
                    sum += i32::from(self.ternary(base + consumed + tail)) * i32::from(value);
                }
                sum
            }
            BitNetKernelMode::Tl2 => {
                let mut sum = 0_i32;
                let mut triples = input.chunks_exact(3);
                for (triple_index, triple) in triples.by_ref().enumerate() {
                    let column = triple_index * 3;
                    let d0 = usize::try_from(self.ternary(base + column) + 1)
                        .expect("ternary TL2 digit");
                    let d1 = usize::try_from(self.ternary(base + column + 1) + 1)
                        .expect("ternary TL2 digit");
                    let d2 = usize::try_from(self.ternary(base + column + 2) + 1)
                        .expect("ternary TL2 digit");
                    let mut table = [0_i32; 27];
                    for a in 0..3 {
                        for b in 0..3 {
                            for c in 0..3 {
                                table[a * 9 + b * 3 + c] = (a as i32 - 1) * i32::from(triple[0])
                                    + (b as i32 - 1) * i32::from(triple[1])
                                    + (c as i32 - 1) * i32::from(triple[2]);
                            }
                        }
                    }
                    sum += table[d0 * 9 + d1 * 3 + d2];
                }
                let consumed = input.len() - triples.remainder().len();
                for (tail, value) in triples.remainder().iter().copied().enumerate() {
                    sum += i32::from(self.ternary(base + consumed + tail)) * i32::from(value);
                }
                sum
            }
            BitNetKernelMode::Auto => unreachable!("auto is resolved above"),
        };
        unscaled as f32 * activation_scale * self.scale
    }
}

/// Quantize contiguous f32 activation rows to BitNet's symmetric A8 contract.
///
/// The returned scale for each row is its dequantization scale (`max_abs / 127`).
/// An all-zero row has a zero scale and zero quantized values. Keeping this
/// conversion on the host gives the CPU, Metal, and CUDA kernels identical
/// rounding and clamping semantics.
pub(crate) fn quantize_activation_rows(input: &[f32], width: usize) -> Result<(Vec<i8>, Vec<f32>)> {
    if width == 0 {
        return Err(BackendError::RuntimeShapeMismatch(
            "BitNet A8 quantization requires a non-zero row width".into(),
        ));
    }
    if input.is_empty() || !input.len().is_multiple_of(width) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "BitNet A8 input length {} is not a non-zero multiple of row width {width}",
            input.len()
        )));
    }

    let mut quantized = Vec::with_capacity(input.len());
    let mut scales = Vec::with_capacity(input.len() / width);
    for (row_index, row) in input.chunks_exact(width).enumerate() {
        let mut max_abs = 0.0_f32;
        for (column, value) in row.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(BackendError::InvalidTensorData(format!(
                    "BitNet A8 activation at row {row_index}, column {column} is non-finite"
                )));
            }
            max_abs = max_abs.max(value.abs());
        }

        if max_abs == 0.0 {
            quantized.resize(quantized.len() + width, 0);
            scales.push(0.0);
            continue;
        }

        let dequant_scale = max_abs / 127.0;
        let quant_multiplier = 127.0 / max_abs;
        quantized.extend(row.iter().map(|value| {
            let rounded = (*value * quant_multiplier).round();
            rounded.clamp(-128.0, 127.0) as i8
        }));
        scales.push(dequant_scale);
    }
    Ok((quantized, scales))
}

pub(crate) fn i2_s_matvec(
    bytes: &[u8],
    input: &[f32],
    output_rows: usize,
    mode: BitNetKernelMode,
) -> Result<Vec<f32>> {
    log_cpu_once(mode);
    let matrix = I2SMatrix::new(bytes, input.len(), output_rows)?;
    let (quantized, activation_scales) = quantize_activation_rows(input, input.len())?;
    let activation_scale = activation_scales[0];
    Ok((0..output_rows)
        .into_par_iter()
        .map(|row| matrix.row_dot(row, &quantized, activation_scale, mode))
        .collect())
}

pub(crate) fn i2_s_matmul(
    bytes: &[u8],
    inputs: &[Vec<f32>],
    output_rows: usize,
    mode: BitNetKernelMode,
) -> Result<Vec<Vec<f32>>> {
    log_cpu_once(mode);
    let input_width = inputs.first().map(Vec::len).ok_or_else(|| {
        BackendError::InvalidTensorData("I2_S matmul requires at least one input row".into())
    })?;
    if inputs.iter().any(|input| input.len() != input_width) {
        return Err(BackendError::RuntimeShapeMismatch(
            "I2_S matmul input rows have inconsistent widths".into(),
        ));
    }
    let matrix = I2SMatrix::new(bytes, input_width, output_rows)?;
    inputs
        .par_iter()
        .map(|input| {
            let (quantized, activation_scales) = quantize_activation_rows(input, input_width)?;
            let activation_scale = activation_scales[0];
            Ok((0..output_rows)
                .map(|row| matrix.row_dot(row, &quantized, activation_scale, mode))
                .collect())
        })
        .collect()
}

fn log_cpu_once(mode: BitNetKernelMode) {
    static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    LOGGED.get_or_init(|| {
        eprintln!(
            "[bitnet] CPU I2_S cleanroom kernel active (mode={})",
            mode.as_str()
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_i2_s(values: &[i8], scale: f32) -> Vec<u8> {
        assert!(values.len().is_multiple_of(128));
        let mut packed = vec![0_u8; values.len() / 4];
        for (index, value) in values.iter().copied().enumerate() {
            let tile = index / 128;
            let within = index % 128;
            let code = match value {
                -1 => 0,
                0 => 1,
                1 => 2,
                _ => panic!("not ternary"),
            };
            let shift = 6 - 2 * (within / 32);
            packed[tile * 32 + within % 32] |= code << shift;
        }
        packed.extend_from_slice(&scale.to_le_bytes());
        packed.extend_from_slice(&[0; 28]);
        packed
    }

    fn w158a8_scalar_oracle(weights: &[i8], input: &[f32], weight_scale: f32) -> f32 {
        assert_eq!(weights.len(), input.len());
        let max_abs = input.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
        if max_abs == 0.0 {
            return 0.0;
        }
        let multiplier = 127.0 / max_abs;
        let integer_dot = weights
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(weight, value)| {
                let activation = (value * multiplier).round().clamp(-128.0, 127.0) as i64;
                i64::from(weight) * activation
            })
            .sum::<i64>();
        integer_dot as f32 * (max_abs / 127.0) * weight_scale
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 2.0e-6 * expected.abs().max(1.0),
            "actual={actual} expected={expected}"
        );
    }

    #[test]
    fn a8_quantization_is_rowwise_symmetric_and_handles_zero_rows() {
        let input = [
            -2.0, -1.0, 0.5, 2.0, -0.25, 0.25, 0.125, -0.125, 0.0, 0.0, 0.0, 0.0,
        ];
        let (quantized, scales) = quantize_activation_rows(&input, 4).unwrap();
        assert_eq!(
            quantized,
            vec![-127, -64, 32, 127, -127, 127, 64, -64, 0, 0, 0, 0]
        );
        assert_eq!(scales.len(), 3);
        assert_close(scales[0], 2.0 / 127.0);
        assert_close(scales[1], 0.25 / 127.0);
        assert_eq!(scales[2], 0.0);
    }

    #[test]
    fn a8_quantization_rejects_empty_ragged_and_non_finite_inputs() {
        assert!(quantize_activation_rows(&[], 4).is_err());
        assert!(quantize_activation_rows(&[1.0], 0).is_err());
        assert!(quantize_activation_rows(&[1.0, 2.0, 3.0], 2).is_err());
        assert!(quantize_activation_rows(&[1.0, f32::NAN], 2).is_err());
        assert!(quantize_activation_rows(&[f32::INFINITY, 1.0], 2).is_err());
    }

    #[test]
    fn cleanroom_i2_s_tl1_and_tl2_modes_match_w158a8_scalar_oracle() {
        let rows = 3;
        let width = 128;
        let values = (0..rows * width)
            .map(|index| match index % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect::<Vec<_>>();
        let scale = 0.25_f32;
        let bytes = encode_i2_s(&values, scale);
        let input = (0..width)
            .map(|index| ((index as f32 + 0.25) * 0.173).sin() * 2.75)
            .collect::<Vec<_>>();
        let second_input = input
            .iter()
            .enumerate()
            .map(|(index, value)| value * 0.37 + (index as f32 * 0.11).cos() * 0.2)
            .collect::<Vec<_>>();
        let expected = values
            .chunks_exact(width)
            .map(|row| w158a8_scalar_oracle(row, &input, scale))
            .collect::<Vec<_>>();
        let second_expected = values
            .chunks_exact(width)
            .map(|row| w158a8_scalar_oracle(row, &second_input, scale))
            .collect::<Vec<_>>();
        for mode in [
            BitNetKernelMode::I2S,
            BitNetKernelMode::Tl1,
            BitNetKernelMode::Tl2,
        ] {
            let actual = i2_s_matvec(&bytes, &input, rows, mode).unwrap();
            for (actual, expected) in actual.into_iter().zip(expected.iter().copied()) {
                assert_close(actual, expected);
            }
            let batch =
                i2_s_matmul(&bytes, &[input.clone(), second_input.clone()], rows, mode).unwrap();
            for (actual_row, expected_row) in batch
                .iter()
                .zip([expected.as_slice(), second_expected.as_slice()])
            {
                for (actual, expected) in actual_row.iter().zip(expected_row) {
                    assert_close(*actual, *expected);
                }
            }
        }
    }

    #[test]
    fn lookup_modes_are_integer_equivalent_to_direct_after_a8_quantization() {
        let rows = 5;
        let width = 128;
        let values = (0..rows * width)
            .map(|index| match (index * 17 + 5) % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect::<Vec<_>>();
        let bytes = encode_i2_s(&values, 0.03125);
        let input = (0..width)
            .map(|index| ((index as f32 + 0.25) * 0.173).sin())
            .collect::<Vec<_>>();
        let direct = i2_s_matvec(&bytes, &input, rows, BitNetKernelMode::I2S).unwrap();
        for mode in [BitNetKernelMode::Tl1, BitNetKernelMode::Tl2] {
            let lookup = i2_s_matvec(&bytes, &input, rows, mode).unwrap();
            assert_eq!(lookup, direct, "{mode:?}");
        }
    }

    #[test]
    fn kernel_mode_parser_is_explicit_and_invalid_values_fall_back_at_the_env_boundary() {
        assert_eq!(
            BitNetKernelMode::parse("auto"),
            Some(BitNetKernelMode::Auto)
        );
        assert_eq!(BitNetKernelMode::parse("I2_S"), Some(BitNetKernelMode::I2S));
        assert_eq!(BitNetKernelMode::parse("tl1"), Some(BitNetKernelMode::Tl1));
        assert_eq!(
            BitNetKernelMode::parse(" TL2 "),
            Some(BitNetKernelMode::Tl2)
        );
        assert_eq!(BitNetKernelMode::parse("upstream-generated"), None);
    }

    #[test]
    fn malformed_i2_s_kernel_inputs_fail_closed() {
        assert!(i2_s_matvec(&[0; 64], &[0.0; 127], 1, BitNetKernelMode::I2S).is_err());
        let mut non_finite = encode_i2_s(&[0; 128], 1.0);
        non_finite[32..36].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(i2_s_matvec(&non_finite, &[0.0; 128], 1, BitNetKernelMode::I2S).is_err());
        let valid = encode_i2_s(&[0; 128], 1.0);
        let mut non_finite_activation = [0.0; 128];
        non_finite_activation[17] = f32::NEG_INFINITY;
        assert!(i2_s_matvec(&valid, &non_finite_activation, 1, BitNetKernelMode::I2S).is_err());
    }
}
