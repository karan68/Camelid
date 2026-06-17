//! Dequant-to-f32 for the runnable lane's v1 quant set.
//!
//! Breadth comes from one small dispatch over per-format routines, not a per-format
//! kernel matrix (`RUNNABLE_LANE_SPEC.md`, principle #3). The runnable lane is f32
//! only — no Metal/CUDA fast paths. Each covered quant routes to the crate's already
//! existing, validated block decoder; F16/F32 are handled inline. Correctness is
//! anchored externally against ggml reference fixtures (Gate 2), not trusted from the
//! internal paths it reuses.
//!
//! Covered v1 set: `F32, F16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K`. Anything else is refused
//! — admission (`super::admit`) should already have rejected it, but dequant fails
//! closed regardless.

use crate::error::{BackendError, Result};
use crate::gguf::GgufTensorType;
use crate::tensor::{
    decode_q4_0_tensor, decode_q4_k_tensor, decode_q5_k_tensor, decode_q6_k_tensor,
    decode_q8_0_tensor, f16_bits_to_f32,
};

/// Dequantize one tensor's wire bytes to a flat row-major `Vec<f32>` of
/// `n_elements` values. `tensor_name` is threaded through only for error messages.
pub fn dequantize(
    tensor_type: GgufTensorType,
    bytes: &[u8],
    n_elements: usize,
    tensor_name: &str,
) -> Result<Vec<f32>> {
    match tensor_type {
        GgufTensorType::F32 => dequantize_f32(bytes, n_elements, tensor_name),
        GgufTensorType::F16 => dequantize_f16(bytes, n_elements, tensor_name),
        GgufTensorType::Q8_0 => decode_q8_0_tensor(tensor_name, bytes, n_elements),
        GgufTensorType::Q4_0 => decode_q4_0_tensor(tensor_name, bytes, n_elements),
        GgufTensorType::Q4K => decode_q4_k_tensor(tensor_name, bytes, n_elements),
        GgufTensorType::Q5K => decode_q5_k_tensor(tensor_name, bytes, n_elements),
        GgufTensorType::Q6K => decode_q6_k_tensor(tensor_name, bytes, n_elements),
        other => Err(BackendError::UnsupportedTensorType(format!(
            "tensor {tensor_name} is {other:?}; runnable dequant covers \
             F32, F16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K"
        ))),
    }
}

fn dequantize_f32(bytes: &[u8], n_elements: usize, name: &str) -> Result<Vec<f32>> {
    expect_len(bytes.len(), n_elements * 4, name, "F32")?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn dequantize_f16(bytes: &[u8], n_elements: usize, name: &str) -> Result<Vec<f32>> {
    expect_len(bytes.len(), n_elements * 2, name, "F16")?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

fn expect_len(actual: usize, expected: usize, name: &str, ty: &str) -> Result<()> {
    if actual != expected {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} {ty} byte length {actual} does not match expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_roundtrips_bit_exact() {
        let vals = [-1.5f32, 0.0, 3.25, f32::MIN_POSITIVE, -0.0];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = dequantize(GgufTensorType::F32, &bytes, vals.len(), "t").unwrap();
        for (a, b) in out.iter().zip(vals.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn f16_known_values() {
        // 0x3C00 = 1.0, 0xC000 = -2.0, 0x0000 = 0.0 in IEEE half.
        let bytes = [0x00, 0x3C, 0x00, 0xC0, 0x00, 0x00];
        let out = dequantize(GgufTensorType::F16, &bytes, 3, "t").unwrap();
        assert_eq!(out, vec![1.0, -2.0, 0.0]);
    }

    #[test]
    fn wrong_length_fails_closed() {
        let err = dequantize(GgufTensorType::F32, &[0u8; 6], 2, "t").unwrap_err();
        assert!(matches!(err, BackendError::InvalidTensorData(_)));
    }

    #[test]
    fn uncovered_quant_refused() {
        let err = dequantize(GgufTensorType::Q2K, &[0u8; 84], 256, "blk.0").unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedTensorType(_)));
    }
}
