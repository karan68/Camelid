//! Cleanroom BitNet matrix kernels over the public I2_S wire contract.
//!
//! The linked Microsoft artifacts all store one canonical representation:
//! four ternary values per byte, interleaved in 128-value tiles, followed by a
//! tensor-wide f32 scale. Camelid keeps those bytes unchanged and offers three
//! execution strategies over them:
//!
//! - `i2_s`: direct ternary accumulation;
//! - `tl1`: two-weight lookup-table accumulation;
//! - `tl2`: three-weight lookup-table accumulation.
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

    fn row_dot(&self, row: usize, input: &[f32], mode: BitNetKernelMode) -> f32 {
        debug_assert!(row < self.output_rows);
        debug_assert_eq!(input.len(), self.input_width);
        let base = row * self.input_width;
        let unscaled = match mode.effective_cpu() {
            BitNetKernelMode::I2S => {
                let mut sum = 0.0_f32;
                for (column, value) in input.iter().copied().enumerate() {
                    sum += f32::from(self.ternary(base + column)) * value;
                }
                sum
            }
            BitNetKernelMode::Tl1 => {
                let mut sum = 0.0_f32;
                let mut pairs = input.chunks_exact(2);
                for (pair_index, pair) in pairs.by_ref().enumerate() {
                    let column = pair_index * 2;
                    let left = usize::try_from(self.ternary(base + column) + 1)
                        .expect("ternary TL1 digit");
                    let right = usize::try_from(self.ternary(base + column + 1) + 1)
                        .expect("ternary TL1 digit");
                    let a = pair[0];
                    let b = pair[1];
                    let table = [-a - b, -a, -a + b, -b, 0.0, b, a - b, a, a + b];
                    sum += table[left * 3 + right];
                }
                let consumed = input.len() - pairs.remainder().len();
                for (tail, value) in pairs.remainder().iter().copied().enumerate() {
                    sum += f32::from(self.ternary(base + consumed + tail)) * value;
                }
                sum
            }
            BitNetKernelMode::Tl2 => {
                let mut sum = 0.0_f32;
                let mut triples = input.chunks_exact(3);
                for (triple_index, triple) in triples.by_ref().enumerate() {
                    let column = triple_index * 3;
                    let d0 = usize::try_from(self.ternary(base + column) + 1)
                        .expect("ternary TL2 digit");
                    let d1 = usize::try_from(self.ternary(base + column + 1) + 1)
                        .expect("ternary TL2 digit");
                    let d2 = usize::try_from(self.ternary(base + column + 2) + 1)
                        .expect("ternary TL2 digit");
                    let mut table = [0.0_f32; 27];
                    for a in 0..3 {
                        for b in 0..3 {
                            for c in 0..3 {
                                table[a * 9 + b * 3 + c] = (a as f32 - 1.0) * triple[0]
                                    + (b as f32 - 1.0) * triple[1]
                                    + (c as f32 - 1.0) * triple[2];
                            }
                        }
                    }
                    sum += table[d0 * 9 + d1 * 3 + d2];
                }
                let consumed = input.len() - triples.remainder().len();
                for (tail, value) in triples.remainder().iter().copied().enumerate() {
                    sum += f32::from(self.ternary(base + consumed + tail)) * value;
                }
                sum
            }
            BitNetKernelMode::Auto => unreachable!("auto is resolved above"),
        };
        unscaled * self.scale
    }
}

pub(crate) fn i2_s_matvec(
    bytes: &[u8],
    input: &[f32],
    output_rows: usize,
    mode: BitNetKernelMode,
) -> Result<Vec<f32>> {
    log_cpu_once(mode);
    let matrix = I2SMatrix::new(bytes, input.len(), output_rows)?;
    Ok((0..output_rows)
        .into_par_iter()
        .map(|row| matrix.row_dot(row, input, mode))
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
            Ok((0..output_rows)
                .map(|row| matrix.row_dot(row, input, mode))
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

    #[test]
    fn cleanroom_i2_s_tl1_and_tl2_modes_match_dense_ternary_math() {
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
            .map(|index| (index as i32 % 7 - 3) as f32)
            .collect::<Vec<_>>();
        let expected = values
            .chunks_exact(width)
            .map(|row| {
                row.iter()
                    .zip(&input)
                    .map(|(weight, value)| f32::from(*weight) * *value)
                    .sum::<f32>()
                    * scale
            })
            .collect::<Vec<_>>();
        for mode in [
            BitNetKernelMode::I2S,
            BitNetKernelMode::Tl1,
            BitNetKernelMode::Tl2,
        ] {
            assert_eq!(i2_s_matvec(&bytes, &input, rows, mode).unwrap(), expected);
            assert_eq!(
                i2_s_matmul(&bytes, &[input.clone(), input.clone()], rows, mode).unwrap(),
                vec![expected.clone(), expected.clone()]
            );
        }
    }

    #[test]
    fn lookup_modes_stay_within_float_rounding_of_direct_on_fractional_inputs() {
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
            for (actual, expected) in lookup.iter().zip(&direct) {
                assert!(
                    (actual - expected).abs() <= 2.0e-6 * expected.abs().max(1.0),
                    "{mode:?}: actual={actual} direct={expected}"
                );
            }
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
    }
}
