//! CUDA-resident packing primitives for the tracked four-layer Gemma 4 MTP assistant.
//!
//! This module deliberately has no runtime wiring.  It validates the exact BF16
//! safetensors tensor set, converts matrices directly from BF16 to ggml Q4_0, and
//! emits the quants-first SoA byte layout consumed by the resident CUDA Q4_0
//! kernels.  A caller supplies a per-matrix visitor, so the source never expands
//! into the roughly 1.56 GiB f32 representation and only one packed matrix is live
//! at a time.  With CUDA enabled, [`Gemma4MtpCudaWeights`] uploads each completed
//! matrix into its own device allocation before the host pack is released.

use crate::{BackendError, Result};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaView};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::sync::Arc;

pub const GEMMA4_MTP_CUDA_MATRIX_COUNT: usize = 23;
pub const GEMMA4_MTP_CUDA_F32_TENSOR_COUNT: usize = 25;
pub const GEMMA4_MTP_CUDA_MATRIX_ELEMENTS: u64 = 419_692_544;
pub const GEMMA4_MTP_CUDA_SOURCE_MATRIX_BYTES: u64 = 839_385_088;
pub const GEMMA4_MTP_CUDA_SOURCE_PAYLOAD_BYTES: u64 = 839_422_472;
pub const GEMMA4_MTP_CUDA_MATRIX_BYTES: u64 = 236_077_056;
pub const GEMMA4_MTP_CUDA_F32_VALUES: u64 = 18_692;
pub const GEMMA4_MTP_CUDA_F32_BYTES: u64 = 74_768;
pub const GEMMA4_MTP_CUDA_RESIDENT_BYTES: u64 = 236_151_824;

/// Stable identity and device-slot index for every resident Q4_0 matrix.
///
/// The explicit identity keeps future compute wiring from depending on string
/// lookups while preserving the checkpoint/ledger order used during streaming.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Gemma4MtpCudaMatrixId {
    Embedding = 0,
    Layer0QProj,
    Layer0OProj,
    Layer0GateProj,
    Layer0UpProj,
    Layer0DownProj,
    Layer1QProj,
    Layer1OProj,
    Layer1GateProj,
    Layer1UpProj,
    Layer1DownProj,
    Layer2QProj,
    Layer2OProj,
    Layer2GateProj,
    Layer2UpProj,
    Layer2DownProj,
    Layer3QProj,
    Layer3OProj,
    Layer3GateProj,
    Layer3UpProj,
    Layer3DownProj,
    PreProjection,
    PostProjection,
}

impl Gemma4MtpCudaMatrixId {
    pub const ALL: [Self; GEMMA4_MTP_CUDA_MATRIX_COUNT] = [
        Self::Embedding,
        Self::Layer0QProj,
        Self::Layer0OProj,
        Self::Layer0GateProj,
        Self::Layer0UpProj,
        Self::Layer0DownProj,
        Self::Layer1QProj,
        Self::Layer1OProj,
        Self::Layer1GateProj,
        Self::Layer1UpProj,
        Self::Layer1DownProj,
        Self::Layer2QProj,
        Self::Layer2OProj,
        Self::Layer2GateProj,
        Self::Layer2UpProj,
        Self::Layer2DownProj,
        Self::Layer3QProj,
        Self::Layer3OProj,
        Self::Layer3GateProj,
        Self::Layer3UpProj,
        Self::Layer3DownProj,
        Self::PreProjection,
        Self::PostProjection,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        MATRIX_SPECS[self.index()].name
    }
}

/// Stable identity and compact-buffer slot for every f32 norm/scalar tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Gemma4MtpCudaF32Id {
    Layer0InputNorm = 0,
    Layer0QNorm,
    Layer0PostAttentionNorm,
    Layer0PreFeedforwardNorm,
    Layer0PostFeedforwardNorm,
    Layer0Scalar,
    Layer1InputNorm,
    Layer1QNorm,
    Layer1PostAttentionNorm,
    Layer1PreFeedforwardNorm,
    Layer1PostFeedforwardNorm,
    Layer1Scalar,
    Layer2InputNorm,
    Layer2QNorm,
    Layer2PostAttentionNorm,
    Layer2PreFeedforwardNorm,
    Layer2PostFeedforwardNorm,
    Layer2Scalar,
    Layer3InputNorm,
    Layer3QNorm,
    Layer3PostAttentionNorm,
    Layer3PreFeedforwardNorm,
    Layer3PostFeedforwardNorm,
    Layer3Scalar,
    FinalNorm,
}

impl Gemma4MtpCudaF32Id {
    pub const ALL: [Self; GEMMA4_MTP_CUDA_F32_TENSOR_COUNT] = [
        Self::Layer0InputNorm,
        Self::Layer0QNorm,
        Self::Layer0PostAttentionNorm,
        Self::Layer0PreFeedforwardNorm,
        Self::Layer0PostFeedforwardNorm,
        Self::Layer0Scalar,
        Self::Layer1InputNorm,
        Self::Layer1QNorm,
        Self::Layer1PostAttentionNorm,
        Self::Layer1PreFeedforwardNorm,
        Self::Layer1PostFeedforwardNorm,
        Self::Layer1Scalar,
        Self::Layer2InputNorm,
        Self::Layer2QNorm,
        Self::Layer2PostAttentionNorm,
        Self::Layer2PreFeedforwardNorm,
        Self::Layer2PostFeedforwardNorm,
        Self::Layer2Scalar,
        Self::Layer3InputNorm,
        Self::Layer3QNorm,
        Self::Layer3PostAttentionNorm,
        Self::Layer3PreFeedforwardNorm,
        Self::Layer3PostFeedforwardNorm,
        Self::Layer3Scalar,
        Self::FinalNorm,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        F32_SPECS[self.index()].name
    }
}

const Q4_0_BLOCK_VALUES: usize = 32;
const Q4_0_QUANT_BYTES: usize = 16;
const Q4_0_SCALE_BYTES: usize = 2;
const Q4_0_BLOCK_BYTES: usize = Q4_0_QUANT_BYTES + Q4_0_SCALE_BYTES;
const STREAM_CHUNK_TARGET_BYTES: usize = 1 << 20;

#[derive(Clone, Copy)]
struct MatrixSpec {
    name: &'static str,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Copy)]
struct F32Spec {
    name: &'static str,
    elements: usize,
}

// Order is the future resident-buffer order, not safetensors file order.  It
// matches the Metal full-Q4 layout: tied embedding, five matrices per layer,
// then the recurrent pre/post projections.
const MATRIX_SPECS: [MatrixSpec; GEMMA4_MTP_CUDA_MATRIX_COUNT] = [
    MatrixSpec {
        name: "model.embed_tokens.weight",
        rows: 262_144,
        cols: 1_024,
    },
    MatrixSpec {
        name: "model.layers.0.self_attn.q_proj.weight",
        rows: 4_096,
        cols: 1_024,
    },
    MatrixSpec {
        name: "model.layers.0.self_attn.o_proj.weight",
        rows: 1_024,
        cols: 4_096,
    },
    MatrixSpec {
        name: "model.layers.0.mlp.gate_proj.weight",
        rows: 8_192,
        cols: 1_024,
    },
    MatrixSpec {
        name: "model.layers.0.mlp.up_proj.weight",
        rows: 8_192,
        cols: 1_024,
    },
    MatrixSpec {
        name: "model.layers.0.mlp.down_proj.weight",
        rows: 1_024,
        cols: 8_192,
    },
    MatrixSpec {
        name: "model.layers.1.self_attn.q_proj.weight",
        rows: 4_096,
        cols: 1_024,
    },
    MatrixSpec {
        name: "model.layers.1.self_attn.o_proj.weight",
        rows: 1_024,
        cols: 4_096,
    },
    MatrixSpec {
        name: "model.layers.1.mlp.gate_proj.weight",
        rows: 8_192,
        cols: 1_024,
    },
    MatrixSpec {
        name: "model.layers.1.mlp.up_proj.weight",
        rows: 8_192,
        cols: 1_024,
    },
    MatrixSpec {
        name: "model.layers.1.mlp.down_proj.weight",
        rows: 1_024,
        cols: 8_192,
    },
    MatrixSpec {
        name: "model.layers.2.self_attn.q_proj.weight",
        rows: 4_096,
        cols: 1_024,
    },
    MatrixSpec {
        name: "model.layers.2.self_attn.o_proj.weight",
        rows: 1_024,
        cols: 4_096,
    },
    MatrixSpec {
        name: "model.layers.2.mlp.gate_proj.weight",
        rows: 8_192,
        cols: 1_024,
    },
    MatrixSpec {
        name: "model.layers.2.mlp.up_proj.weight",
        rows: 8_192,
        cols: 1_024,
    },
    MatrixSpec {
        name: "model.layers.2.mlp.down_proj.weight",
        rows: 1_024,
        cols: 8_192,
    },
    MatrixSpec {
        name: "model.layers.3.self_attn.q_proj.weight",
        rows: 8_192,
        cols: 1_024,
    },
    MatrixSpec {
        name: "model.layers.3.self_attn.o_proj.weight",
        rows: 1_024,
        cols: 8_192,
    },
    MatrixSpec {
        name: "model.layers.3.mlp.gate_proj.weight",
        rows: 8_192,
        cols: 1_024,
    },
    MatrixSpec {
        name: "model.layers.3.mlp.up_proj.weight",
        rows: 8_192,
        cols: 1_024,
    },
    MatrixSpec {
        name: "model.layers.3.mlp.down_proj.weight",
        rows: 1_024,
        cols: 8_192,
    },
    MatrixSpec {
        name: "pre_projection.weight",
        rows: 1_024,
        cols: 5_632,
    },
    MatrixSpec {
        name: "post_projection.weight",
        rows: 2_816,
        cols: 1_024,
    },
];

// Stable f32-device-buffer order.  The source values remain BF16 on disk and
// occupy only 37,384 bytes; conversion produces this ledger's 74,768 bytes.
const F32_SPECS: [F32Spec; GEMMA4_MTP_CUDA_F32_TENSOR_COUNT] = [
    F32Spec {
        name: "model.layers.0.input_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.0.self_attn.q_norm.weight",
        elements: 256,
    },
    F32Spec {
        name: "model.layers.0.post_attention_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.0.pre_feedforward_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.0.post_feedforward_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.0.layer_scalar",
        elements: 1,
    },
    F32Spec {
        name: "model.layers.1.input_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.1.self_attn.q_norm.weight",
        elements: 256,
    },
    F32Spec {
        name: "model.layers.1.post_attention_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.1.pre_feedforward_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.1.post_feedforward_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.1.layer_scalar",
        elements: 1,
    },
    F32Spec {
        name: "model.layers.2.input_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.2.self_attn.q_norm.weight",
        elements: 256,
    },
    F32Spec {
        name: "model.layers.2.post_attention_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.2.pre_feedforward_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.2.post_feedforward_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.2.layer_scalar",
        elements: 1,
    },
    F32Spec {
        name: "model.layers.3.input_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.3.self_attn.q_norm.weight",
        elements: 512,
    },
    F32Spec {
        name: "model.layers.3.post_attention_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.3.pre_feedforward_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.3.post_feedforward_layernorm.weight",
        elements: 1_024,
    },
    F32Spec {
        name: "model.layers.3.layer_scalar",
        elements: 1,
    },
    F32Spec {
        name: "model.norm.weight",
        elements: 1_024,
    },
];

/// One matrix in the contiguous CUDA Q4_0 pack.
///
/// Each matrix is independently SoA: its visitor slice is
/// `[quants_bytes][scale_bytes]`. `scales_byte_offset` is relative to that slice;
/// `pack_byte_offset` is the matrix's location if all visitor slices are joined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemma4MtpCudaMatrixLayout {
    pub id: Gemma4MtpCudaMatrixId,
    pub name: &'static str,
    pub rows: usize,
    pub cols: usize,
    pub pack_byte_offset: u64,
    pub quants_bytes: u64,
    pub scales_byte_offset: u64,
    pub scale_bytes: u64,
    pub packed_bytes: u64,
    pub source_bf16_bytes: u64,
}

impl Gemma4MtpCudaMatrixLayout {
    pub fn blocks_per_row(&self) -> usize {
        self.cols / Q4_0_BLOCK_VALUES
    }

    pub fn absolute_scales_byte_offset(&self) -> u64 {
        self.pack_byte_offset + self.scales_byte_offset
    }
}

/// One decoded f32 norm/scalar tensor in the compact auxiliary buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemma4MtpCudaF32Layout {
    pub id: Gemma4MtpCudaF32Id,
    pub name: &'static str,
    pub elements: usize,
    pub element_offset: u64,
    pub byte_offset: u64,
    pub byte_len: u64,
}

/// Exact device-resident weight ledger for the tracked assistant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemma4MtpCudaLedger {
    pub matrices: Vec<Gemma4MtpCudaMatrixLayout>,
    pub f32_tensors: Vec<Gemma4MtpCudaF32Layout>,
    pub matrix_elements: u64,
    pub source_matrix_bytes: u64,
    pub matrix_bytes: u64,
    pub f32_values: u64,
    pub f32_bytes: u64,
    pub total_bytes: u64,
}

/// Builds and validates the official 23-matrix / 18,692-f32-value CUDA ledger.
pub fn gemma4_mtp_cuda_ledger() -> Result<Gemma4MtpCudaLedger> {
    let mut matrices = Vec::with_capacity(MATRIX_SPECS.len());
    let mut pack_cursor = 0u64;
    let mut matrix_elements = 0u64;

    for (id, spec) in Gemma4MtpCudaMatrixId::ALL.into_iter().zip(MATRIX_SPECS) {
        if spec.rows == 0 || spec.cols == 0 || !spec.cols.is_multiple_of(Q4_0_BLOCK_VALUES) {
            return Err(invalid(format!(
                "matrix {} has invalid Q4_0 geometry {}x{}",
                spec.name, spec.rows, spec.cols
            )));
        }
        let rows = u64::try_from(spec.rows).map_err(|_| invalid("matrix rows exceed u64"))?;
        let cols = u64::try_from(spec.cols).map_err(|_| invalid("matrix cols exceed u64"))?;
        let elements = rows
            .checked_mul(cols)
            .ok_or_else(|| invalid("matrix element count overflow"))?;
        let blocks = elements / Q4_0_BLOCK_VALUES as u64;
        let quants_bytes = blocks
            .checked_mul(Q4_0_QUANT_BYTES as u64)
            .ok_or_else(|| invalid("Q4_0 quant byte count overflow"))?;
        let scale_bytes = blocks
            .checked_mul(Q4_0_SCALE_BYTES as u64)
            .ok_or_else(|| invalid("Q4_0 scale byte count overflow"))?;
        let packed_bytes = quants_bytes
            .checked_add(scale_bytes)
            .ok_or_else(|| invalid("Q4_0 matrix byte count overflow"))?;
        let source_bf16_bytes = elements
            .checked_mul(2)
            .ok_or_else(|| invalid("BF16 matrix byte count overflow"))?;
        matrices.push(Gemma4MtpCudaMatrixLayout {
            id,
            name: spec.name,
            rows: spec.rows,
            cols: spec.cols,
            pack_byte_offset: pack_cursor,
            quants_bytes,
            scales_byte_offset: quants_bytes,
            scale_bytes,
            packed_bytes,
            source_bf16_bytes,
        });
        pack_cursor = pack_cursor
            .checked_add(packed_bytes)
            .ok_or_else(|| invalid("Q4_0 pack offset overflow"))?;
        matrix_elements = matrix_elements
            .checked_add(elements)
            .ok_or_else(|| invalid("matrix element ledger overflow"))?;
    }

    let mut f32_tensors = Vec::with_capacity(F32_SPECS.len());
    let mut f32_values = 0u64;
    for (id, spec) in Gemma4MtpCudaF32Id::ALL.into_iter().zip(F32_SPECS) {
        let elements = u64::try_from(spec.elements)
            .map_err(|_| invalid("f32 tensor element count exceeds u64"))?;
        let byte_offset = f32_values
            .checked_mul(4)
            .ok_or_else(|| invalid("f32 tensor byte offset overflow"))?;
        let byte_len = elements
            .checked_mul(4)
            .ok_or_else(|| invalid("f32 tensor byte count overflow"))?;
        f32_tensors.push(Gemma4MtpCudaF32Layout {
            id,
            name: spec.name,
            elements: spec.elements,
            element_offset: f32_values,
            byte_offset,
            byte_len,
        });
        f32_values = f32_values
            .checked_add(elements)
            .ok_or_else(|| invalid("f32 value ledger overflow"))?;
    }
    let f32_bytes = f32_values
        .checked_mul(4)
        .ok_or_else(|| invalid("f32 ledger byte count overflow"))?;
    let source_matrix_bytes = matrix_elements
        .checked_mul(2)
        .ok_or_else(|| invalid("source matrix byte ledger overflow"))?;
    let total_bytes = pack_cursor
        .checked_add(f32_bytes)
        .ok_or_else(|| invalid("resident ledger byte count overflow"))?;

    let ledger = Gemma4MtpCudaLedger {
        matrices,
        f32_tensors,
        matrix_elements,
        source_matrix_bytes,
        matrix_bytes: pack_cursor,
        f32_values,
        f32_bytes,
        total_bytes,
    };
    validate_official_ledger(&ledger)?;
    Ok(ledger)
}

fn validate_official_ledger(ledger: &Gemma4MtpCudaLedger) -> Result<()> {
    let actual = (
        ledger.matrices.len(),
        ledger.f32_tensors.len(),
        ledger.matrix_elements,
        ledger.source_matrix_bytes,
        ledger.matrix_bytes,
        ledger.f32_values,
        ledger.f32_bytes,
        ledger.total_bytes,
    );
    let expected = (
        GEMMA4_MTP_CUDA_MATRIX_COUNT,
        GEMMA4_MTP_CUDA_F32_TENSOR_COUNT,
        GEMMA4_MTP_CUDA_MATRIX_ELEMENTS,
        GEMMA4_MTP_CUDA_SOURCE_MATRIX_BYTES,
        GEMMA4_MTP_CUDA_MATRIX_BYTES,
        GEMMA4_MTP_CUDA_F32_VALUES,
        GEMMA4_MTP_CUDA_F32_BYTES,
        GEMMA4_MTP_CUDA_RESIDENT_BYTES,
    );
    if actual != expected {
        return Err(invalid(format!(
            "internal Gemma 4 MTP CUDA ledger mismatch: {actual:?}, expected {expected:?}"
        )));
    }

    let mut cursor = 0u64;
    for (index, matrix) in ledger.matrices.iter().enumerate() {
        if matrix.pack_byte_offset != cursor
            || matrix.id.index() != index
            || matrix.id.name() != matrix.name
            || matrix.scales_byte_offset != matrix.quants_bytes
            || matrix.quants_bytes + matrix.scale_bytes != matrix.packed_bytes
        {
            return Err(invalid(format!(
                "matrix {} is not a contiguous quants-first SoA slice",
                matrix.name
            )));
        }
        cursor = cursor
            .checked_add(matrix.packed_bytes)
            .ok_or_else(|| invalid("matrix layout cursor overflow"))?;
    }
    if cursor != ledger.matrix_bytes {
        return Err(invalid("matrix layouts do not cover the matrix ledger"));
    }
    for (index, tensor) in ledger.f32_tensors.iter().enumerate() {
        if tensor.id.index() != index || tensor.id.name() != tensor.name {
            return Err(invalid(format!(
                "f32 tensor {} does not match compact-buffer slot {index}",
                tensor.name
            )));
        }
    }
    Ok(())
}

/// Measured CUDA allocations owned by [`Gemma4MtpCudaWeights`].
///
/// This is deliberately distinct from [`Gemma4MtpCudaLedger`]: the latter is
/// the expected file/layout contract, while this value is rebuilt from the
/// actual device slices after upload and checked against that contract.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemma4MtpCudaResidentAccounting {
    pub device_ordinal: usize,
    pub matrix_allocations: usize,
    pub matrix_bytes: u64,
    pub f32_allocations: usize,
    pub f32_values: u64,
    pub f32_bytes: u64,
    pub total_bytes: u64,
}

/// Fully CUDA-resident weight pack for the tracked Gemma 4 MTP assistant.
///
/// Every Q4_0 matrix has a separate device allocation.  This preserves its
/// independent quants-first SoA layout: matrix kernels receive one allocation
/// and use that matrix's relative `scales_byte_offset`.  The 25 small
/// norms/scalars share one compact f32 allocation and are addressed by
/// [`Gemma4MtpCudaF32Id`].
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct Gemma4MtpCudaWeights {
    matrices: [CudaSlice<u8>; GEMMA4_MTP_CUDA_MATRIX_COUNT],
    f32_parameters: CudaSlice<f32>,
    ledger: Gemma4MtpCudaLedger,
    accounting: Gemma4MtpCudaResidentAccounting,
}

#[cfg(feature = "cuda")]
impl Gemma4MtpCudaWeights {
    /// Streams, quantizes, and uploads an exact assistant checkpoint.
    ///
    /// Only one host Q4_0 matrix pack exists at a time.  Its transfer is
    /// synchronized before the streaming visitor returns, so the host pack can
    /// be released immediately and no 236 MiB aggregate host checkpoint is
    /// retained.  A context/stream mismatch is rejected before any allocation.
    pub fn load(
        source: &Gemma4MtpCudaSource,
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        validate_official_ledger(source.ledger())?;
        if context.as_ref() != stream.context().as_ref() {
            return Err(invalid(
                "Gemma 4 MTP CUDA upload stream belongs to a different context",
            ));
        }

        let mut device_matrices = Vec::with_capacity(GEMMA4_MTP_CUDA_MATRIX_COUNT);
        let mut next_matrix = 0usize;
        source.stream_q4_0_soa(|layout, packed| {
            let expected = source
                .ledger()
                .matrices
                .get(next_matrix)
                .ok_or_else(|| invalid("CUDA upload produced too many assistant matrices"))?;
            if layout != expected || layout.id.index() != next_matrix {
                return Err(invalid(format!(
                    "CUDA upload matrix {} arrived in slot {next_matrix}, expected {}",
                    layout.name, expected.name
                )));
            }
            let expected_bytes = usize::try_from(layout.packed_bytes)
                .map_err(|_| invalid(format!("matrix {} exceeds usize", layout.name)))?;
            if packed.len() != expected_bytes {
                return Err(invalid(format!(
                    "packed matrix {} has {} bytes, expected {expected_bytes}",
                    layout.name,
                    packed.len()
                )));
            }

            let device = stream.clone_htod(packed).map_err(|error| {
                cuda_upload_error(format!("upload matrix {}", layout.name), error)
            })?;
            if device.len() != expected_bytes || device.context().as_ref() != context.as_ref() {
                return Err(invalid(format!(
                    "CUDA allocation for {} does not match its upload ledger",
                    layout.name
                )));
            }
            // The stream visitor's packed slice is dropped on return.  Complete
            // the transfer here rather than relying on pageable-memory staging.
            stream.synchronize().map_err(|error| {
                cuda_upload_error(format!("synchronize matrix {}", layout.name), error)
            })?;
            device_matrices.push(device);
            next_matrix += 1;
            Ok(())
        })?;
        if next_matrix != GEMMA4_MTP_CUDA_MATRIX_COUNT {
            return Err(invalid(format!(
                "CUDA upload produced {next_matrix} matrices, expected {GEMMA4_MTP_CUDA_MATRIX_COUNT}"
            )));
        }
        let matrices: [CudaSlice<u8>; GEMMA4_MTP_CUDA_MATRIX_COUNT] = device_matrices
            .try_into()
            .map_err(|matrices: Vec<CudaSlice<u8>>| {
                invalid(format!(
                    "CUDA upload retained {} matrices, expected {GEMMA4_MTP_CUDA_MATRIX_COUNT}",
                    matrices.len()
                ))
            })?;

        let f32_host = source.load_f32_parameters()?;
        let f32_parameters = stream
            .clone_htod(&f32_host)
            .map_err(|error| cuda_upload_error("upload f32 norms/scalars", error))?;
        stream
            .synchronize()
            .map_err(|error| cuda_upload_error("synchronize f32 norms/scalars", error))?;
        drop(f32_host);

        let accounting = resident_accounting(&matrices, &f32_parameters)?;
        if accounting.device_ordinal != context.ordinal() {
            return Err(invalid(format!(
                "Gemma 4 MTP CUDA upload landed on device {}, expected {}",
                accounting.device_ordinal,
                context.ordinal()
            )));
        }
        let weights = Self {
            matrices,
            f32_parameters,
            ledger: source.ledger().clone(),
            accounting,
        };
        weights.validate_resident_accounting()?;
        Ok(weights)
    }

    pub fn ledger(&self) -> &Gemma4MtpCudaLedger {
        &self.ledger
    }

    pub fn accounting(&self) -> &Gemma4MtpCudaResidentAccounting {
        &self.accounting
    }

    pub fn matrix(&self, id: Gemma4MtpCudaMatrixId) -> &CudaSlice<u8> {
        &self.matrices[id.index()]
    }

    pub fn matrix_layout(&self, id: Gemma4MtpCudaMatrixId) -> &Gemma4MtpCudaMatrixLayout {
        &self.ledger.matrices[id.index()]
    }

    pub fn f32_parameters(&self) -> &CudaSlice<f32> {
        &self.f32_parameters
    }

    pub fn f32_layout(&self, id: Gemma4MtpCudaF32Id) -> &Gemma4MtpCudaF32Layout {
        &self.ledger.f32_tensors[id.index()]
    }

    /// Returns the exact device view for one norm or scalar tensor.
    pub fn f32_tensor(&self, id: Gemma4MtpCudaF32Id) -> CudaView<'_, f32> {
        let layout = self.f32_layout(id);
        let start = usize::try_from(layout.element_offset)
            .expect("official Gemma 4 MTP f32 offset always fits usize");
        self.f32_parameters.slice(start..start + layout.elements)
    }

    /// Re-measures every owned allocation and verifies it against the official
    /// ledger.  Runtime wiring can use this as a fail-closed bring-up gate.
    pub fn validate_resident_accounting(&self) -> Result<()> {
        validate_official_ledger(&self.ledger)?;
        let measured = resident_accounting(&self.matrices, &self.f32_parameters)?;
        if measured != self.accounting {
            return Err(invalid(format!(
                "Gemma 4 MTP CUDA device accounting changed: {measured:?}, loaded {:?}",
                self.accounting
            )));
        }
        let expected = Gemma4MtpCudaResidentAccounting {
            device_ordinal: self.accounting.device_ordinal,
            matrix_allocations: GEMMA4_MTP_CUDA_MATRIX_COUNT,
            matrix_bytes: self.ledger.matrix_bytes,
            f32_allocations: 1,
            f32_values: self.ledger.f32_values,
            f32_bytes: self.ledger.f32_bytes,
            total_bytes: self.ledger.total_bytes,
        };
        if measured != expected {
            return Err(invalid(format!(
                "Gemma 4 MTP CUDA resident accounting mismatch: {measured:?}, expected {expected:?}"
            )));
        }
        for (layout, matrix) in self.ledger.matrices.iter().zip(&self.matrices) {
            let actual = u64::try_from(matrix.num_bytes())
                .map_err(|_| invalid("CUDA matrix byte count exceeds u64"))?;
            if actual != layout.packed_bytes || matrix.ordinal() != self.accounting.device_ordinal {
                return Err(invalid(format!(
                    "CUDA matrix {} owns {actual} bytes on device {}, expected {} bytes on device {}",
                    layout.name,
                    matrix.ordinal(),
                    layout.packed_bytes,
                    self.accounting.device_ordinal
                )));
            }
        }
        if self.f32_parameters.ordinal() != self.accounting.device_ordinal {
            return Err(invalid("CUDA f32 parameters are on the wrong device"));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn resident_accounting(
    matrices: &[CudaSlice<u8>; GEMMA4_MTP_CUDA_MATRIX_COUNT],
    f32_parameters: &CudaSlice<f32>,
) -> Result<Gemma4MtpCudaResidentAccounting> {
    let device_ordinal = matrices[0].ordinal();
    let matrix_bytes = matrices.iter().try_fold(0u64, |total, matrix| {
        if matrix.ordinal() != device_ordinal {
            return Err(invalid("CUDA matrix allocations span multiple devices"));
        }
        let bytes = u64::try_from(matrix.num_bytes())
            .map_err(|_| invalid("CUDA matrix byte count exceeds u64"))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| invalid("CUDA matrix accounting overflow"))
    })?;
    let f32_values = u64::try_from(f32_parameters.len())
        .map_err(|_| invalid("CUDA f32 value count exceeds u64"))?;
    let f32_bytes = u64::try_from(f32_parameters.num_bytes())
        .map_err(|_| invalid("CUDA f32 byte count exceeds u64"))?;
    if f32_parameters.ordinal() != device_ordinal {
        return Err(invalid(
            "CUDA matrix and f32 allocations are on different devices",
        ));
    }
    let total_bytes = matrix_bytes
        .checked_add(f32_bytes)
        .ok_or_else(|| invalid("CUDA resident accounting overflow"))?;
    Ok(Gemma4MtpCudaResidentAccounting {
        device_ordinal,
        matrix_allocations: matrices.len(),
        matrix_bytes,
        f32_allocations: 1,
        f32_values,
        f32_bytes,
        total_bytes,
    })
}

#[cfg(feature = "cuda")]
fn cuda_upload_error(
    operation: impl std::fmt::Display,
    error: impl std::fmt::Display,
) -> BackendError {
    BackendError::InvalidTensorData(format!("Gemma 4 MTP CUDA {operation}: {error}"))
}

#[derive(Clone, Debug)]
struct SourceTensor {
    data_start: u64,
    data_end: u64,
}

impl SourceTensor {
    fn byte_len(&self) -> u64 {
        self.data_end - self.data_start
    }
}

#[derive(Deserialize)]
struct SafetensorsDescriptor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// Validated, file-backed source for a future CUDA-resident assistant upload.
#[derive(Clone, Debug)]
pub struct Gemma4MtpCudaSource {
    path: PathBuf,
    payload_file_offset: u64,
    file_bytes: u64,
    tensors: BTreeMap<String, SourceTensor>,
    ledger: Gemma4MtpCudaLedger,
}

impl Gemma4MtpCudaSource {
    /// Opens and validates the exact 48-tensor BF16 assistant safetensors file.
    /// Tensor payload order may differ, but names, shapes, dtypes, byte counts,
    /// contiguity, and the complete 839,422,472-byte payload are pinned.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|source| io_error(path, source))?;
        let file_bytes = file
            .metadata()
            .map_err(|source| io_error(path, source))?
            .len();
        let mut length_bytes = [0u8; 8];
        file.read_exact(&mut length_bytes)
            .map_err(|source| io_error(path, source))?;
        let header_bytes = u64::from_le_bytes(length_bytes);
        let payload_file_offset = 8u64
            .checked_add(header_bytes)
            .ok_or_else(|| invalid("safetensors header offset overflow"))?;
        if payload_file_offset > file_bytes {
            return Err(invalid(format!(
                "safetensors header ends at {payload_file_offset}, past {file_bytes}-byte file"
            )));
        }
        let header_len = usize::try_from(header_bytes)
            .map_err(|_| invalid("safetensors header is too large for this host"))?;
        let mut header = Vec::new();
        header
            .try_reserve_exact(header_len)
            .map_err(|error| invalid(format!("cannot allocate safetensors header: {error}")))?;
        header.resize(header_len, 0);
        file.read_exact(&mut header)
            .map_err(|source| io_error(path, source))?;

        let payload_bytes = file_bytes - payload_file_offset;
        let tensors = parse_and_validate_header(&header, payload_bytes)?;
        Ok(Self {
            path: path.to_path_buf(),
            payload_file_offset,
            file_bytes,
            tensors,
            ledger: gemma4_mtp_cuda_ledger()?,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    pub fn payload_file_offset(&self) -> u64 {
        self.payload_file_offset
    }

    pub fn ledger(&self) -> &Gemma4MtpCudaLedger {
        &self.ledger
    }

    /// Convenience wrapper for [`Gemma4MtpCudaWeights::load`].
    #[cfg(feature = "cuda")]
    pub fn load_cuda_weights(
        &self,
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> Result<Gemma4MtpCudaWeights> {
        Gemma4MtpCudaWeights::load(self, context, stream)
    }

    /// Converts and visits one independently-SoA Q4_0 matrix at a time.
    ///
    /// Source reads use a roughly 1 MiB scratch chunk.  The largest live output
    /// is the 150,994,944-byte embedding pack; no full BF16 matrix and no f32
    /// matrix is materialized.  The visitor must consume or copy the slice before
    /// returning because it is dropped before the next matrix is packed.
    pub fn stream_q4_0_soa<F>(&self, mut visitor: F) -> Result<()>
    where
        F: FnMut(&Gemma4MtpCudaMatrixLayout, &[u8]) -> Result<()>,
    {
        let mut file = File::open(&self.path).map_err(|source| io_error(&self.path, source))?;
        for layout in &self.ledger.matrices {
            let source = self.tensor(layout.name)?;
            if source.byte_len() != layout.source_bf16_bytes {
                return Err(invalid(format!(
                    "tensor {} has {} source bytes, expected {}",
                    layout.name,
                    source.byte_len(),
                    layout.source_bf16_bytes
                )));
            }
            let absolute = self
                .payload_file_offset
                .checked_add(source.data_start)
                .ok_or_else(|| invalid("matrix source offset overflow"))?;
            file.seek(SeekFrom::Start(absolute))
                .map_err(|error| io_error(&self.path, error))?;
            let packed = pack_matrix_from_reader(&mut file, &self.path, layout)?;
            visitor(layout, &packed)?;
        }
        Ok(())
    }

    /// Decodes the 25 norms/scalars into one 18,692-value f32 buffer in ledger
    /// order.  Matrix values are never decoded by this method.
    pub fn load_f32_parameters(&self) -> Result<Vec<f32>> {
        let capacity = usize::try_from(self.ledger.f32_values)
            .map_err(|_| invalid("f32 parameter count exceeds usize"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(capacity)
            .map_err(|error| invalid(format!("cannot allocate f32 parameter buffer: {error}")))?;
        let mut file = File::open(&self.path).map_err(|source| io_error(&self.path, source))?;
        let mut bf16_bytes = Vec::new();

        for layout in &self.ledger.f32_tensors {
            let source = self.tensor(layout.name)?;
            let expected_bytes = u64::try_from(layout.elements)
                .ok()
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(|| invalid("BF16 parameter byte count overflow"))?;
            if source.byte_len() != expected_bytes {
                return Err(invalid(format!(
                    "tensor {} has {} bytes, expected {expected_bytes}",
                    layout.name,
                    source.byte_len()
                )));
            }
            let byte_len = usize::try_from(expected_bytes)
                .map_err(|_| invalid("BF16 parameter byte count exceeds usize"))?;
            bf16_bytes.resize(byte_len, 0);
            let absolute = self
                .payload_file_offset
                .checked_add(source.data_start)
                .ok_or_else(|| invalid("parameter source offset overflow"))?;
            file.seek(SeekFrom::Start(absolute))
                .map_err(|error| io_error(&self.path, error))?;
            file.read_exact(&mut bf16_bytes)
                .map_err(|error| io_error(&self.path, error))?;
            values.extend(
                bf16_bytes
                    .chunks_exact(2)
                    .map(|pair| bf16_bits_to_f32(u16::from_le_bytes([pair[0], pair[1]]))),
            );
        }
        if values.len() != capacity {
            return Err(invalid(format!(
                "decoded {} f32 parameters, expected {capacity}",
                values.len()
            )));
        }
        Ok(values)
    }

    fn tensor(&self, name: &str) -> Result<&SourceTensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| BackendError::TensorNotFound(format!("Gemma 4 MTP tensor {name}")))
    }
}

fn parse_and_validate_header(
    header: &[u8],
    payload_bytes: u64,
) -> Result<BTreeMap<String, SourceTensor>> {
    if payload_bytes != GEMMA4_MTP_CUDA_SOURCE_PAYLOAD_BYTES {
        return Err(invalid(format!(
            "Gemma 4 MTP BF16 payload has {payload_bytes} bytes, expected {GEMMA4_MTP_CUDA_SOURCE_PAYLOAD_BYTES}"
        )));
    }
    let root: serde_json::Value = serde_json::from_slice(header)
        .map_err(|error| invalid(format!("invalid safetensors header JSON: {error}")))?;
    let object = root
        .as_object()
        .ok_or_else(|| invalid("safetensors header root is not an object"))?;
    if let Some(metadata) = object.get("__metadata__") {
        if !metadata.is_object() {
            return Err(invalid("safetensors __metadata__ is not an object"));
        }
    }
    let tensor_count = object.len() - usize::from(object.contains_key("__metadata__"));
    let expected_count = MATRIX_SPECS.len() + F32_SPECS.len();
    if tensor_count != expected_count {
        return Err(invalid(format!(
            "Gemma 4 MTP safetensors has {tensor_count} tensors, expected {expected_count}"
        )));
    }

    let mut tensors = BTreeMap::new();
    let mut ranges = Vec::with_capacity(expected_count);
    for (name, shape) in required_tensor_specs() {
        let value = object
            .get(name)
            .ok_or_else(|| invalid(format!("missing Gemma 4 MTP tensor {name}")))?;
        let descriptor: SafetensorsDescriptor = serde_json::from_value(value.clone())
            .map_err(|error| invalid(format!("invalid descriptor for {name}: {error}")))?;
        if descriptor.dtype != "BF16" {
            return Err(invalid(format!(
                "tensor {name} has dtype {}, expected BF16",
                descriptor.dtype
            )));
        }
        if descriptor.shape != shape {
            return Err(invalid(format!(
                "tensor {name} has shape {:?}, expected {shape:?}",
                descriptor.shape
            )));
        }
        let [data_start, data_end] = descriptor.data_offsets;
        if data_end < data_start || data_end > payload_bytes {
            return Err(invalid(format!(
                "tensor {name} has invalid payload range {data_start}..{data_end}"
            )));
        }
        let elements = shape
            .iter()
            .try_fold(1u64, |product, dimension| product.checked_mul(*dimension));
        let expected_bytes = elements
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| invalid(format!("tensor {name} byte count overflow")))?;
        if data_end - data_start != expected_bytes {
            return Err(invalid(format!(
                "tensor {name} covers {} bytes, expected {expected_bytes}",
                data_end - data_start
            )));
        }
        ranges.push((data_start, data_end, name));
        tensors.insert(
            name.to_owned(),
            SourceTensor {
                data_start,
                data_end,
            },
        );
    }

    ranges.sort_unstable_by_key(|range| range.0);
    let mut cursor = 0u64;
    for (start, end, name) in ranges {
        if start != cursor {
            return Err(invalid(format!(
                "tensor {name} begins at payload byte {start}, expected contiguous byte {cursor}"
            )));
        }
        cursor = end;
    }
    if cursor != payload_bytes {
        return Err(invalid(format!(
            "tensor payload covers {cursor} bytes, expected {payload_bytes}"
        )));
    }
    Ok(tensors)
}

fn required_tensor_specs() -> impl Iterator<Item = (&'static str, Vec<u64>)> {
    MATRIX_SPECS
        .iter()
        .map(|spec| (spec.name, vec![spec.rows as u64, spec.cols as u64]))
        .chain(
            F32_SPECS
                .iter()
                .map(|spec| (spec.name, vec![spec.elements as u64])),
        )
}

fn pack_matrix_from_reader(
    reader: &mut File,
    path: &Path,
    layout: &Gemma4MtpCudaMatrixLayout,
) -> Result<Vec<u8>> {
    let packed_len = usize::try_from(layout.packed_bytes)
        .map_err(|_| invalid(format!("packed tensor {} exceeds usize", layout.name)))?;
    let quants_len = usize::try_from(layout.quants_bytes)
        .map_err(|_| invalid(format!("quant plane {} exceeds usize", layout.name)))?;
    let mut packed = Vec::new();
    packed.try_reserve_exact(packed_len).map_err(|error| {
        invalid(format!(
            "cannot allocate packed tensor {}: {error}",
            layout.name
        ))
    })?;
    packed.resize(packed_len, 0);
    let (quants, scales) = packed.split_at_mut(quants_len);

    let source_row_bytes = layout.cols.checked_mul(2).ok_or_else(|| {
        invalid(format!(
            "source row byte count overflow for {}",
            layout.name
        ))
    })?;
    let rows_per_chunk = (STREAM_CHUNK_TARGET_BYTES / source_row_bytes).max(1);
    let scratch_len = rows_per_chunk
        .checked_mul(source_row_bytes)
        .ok_or_else(|| {
            invalid(format!(
                "stream scratch byte count overflow for {}",
                layout.name
            ))
        })?;
    let mut scratch = vec![0u8; scratch_len];
    let blocks_per_row = layout.blocks_per_row();
    let mut row_base = 0usize;

    while row_base < layout.rows {
        let row_count = rows_per_chunk.min(layout.rows - row_base);
        let bytes_to_read = row_count * source_row_bytes;
        reader
            .read_exact(&mut scratch[..bytes_to_read])
            .map_err(|error| io_error(path, error))?;
        for local_row in 0..row_count {
            let row = row_base + local_row;
            let source_row =
                &scratch[local_row * source_row_bytes..(local_row + 1) * source_row_bytes];
            for block_in_row in 0..blocks_per_row {
                let block_index = row * blocks_per_row + block_in_row;
                let source_start = block_in_row * Q4_0_BLOCK_VALUES * 2;
                let source_block = &source_row[source_start..source_start + Q4_0_BLOCK_VALUES * 2];
                let quant_start = block_index * Q4_0_QUANT_BYTES;
                let scale_start = block_index * Q4_0_SCALE_BYTES;
                quantize_q4_0_bf16_le_block(
                    source_block,
                    &mut quants[quant_start..quant_start + Q4_0_QUANT_BYTES],
                    &mut scales[scale_start..scale_start + Q4_0_SCALE_BYTES],
                );
            }
        }
        row_base += row_count;
    }
    Ok(packed)
}

/// Directly packs row-major BF16 bit patterns into per-matrix quants-first Q4_0 SoA.
/// This is the allocation-based primitive used by tests and small future callers;
/// [`Gemma4MtpCudaSource::stream_q4_0_soa`] is the bounded-source-memory file path.
pub fn pack_bf16_q4_0_soa(input: &[u16], rows: usize, cols: usize) -> Result<Vec<u8>> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(Q4_0_BLOCK_VALUES) {
        return Err(invalid(format!(
            "Q4_0 input geometry {rows}x{cols} is empty or not divisible by {Q4_0_BLOCK_VALUES}"
        )));
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("Q4_0 input element count overflow"))?;
    if input.len() != elements {
        return Err(invalid(format!(
            "Q4_0 input has {} values, expected {elements} for {rows}x{cols}",
            input.len()
        )));
    }
    let blocks = elements / Q4_0_BLOCK_VALUES;
    let quant_bytes = blocks
        .checked_mul(Q4_0_QUANT_BYTES)
        .ok_or_else(|| invalid("Q4_0 quant plane byte count overflow"))?;
    let packed_bytes = blocks
        .checked_mul(Q4_0_BLOCK_BYTES)
        .ok_or_else(|| invalid("Q4_0 packed byte count overflow"))?;
    let mut packed = vec![0u8; packed_bytes];
    let (quants, scales) = packed.split_at_mut(quant_bytes);
    for (block_index, block) in input.chunks_exact(Q4_0_BLOCK_VALUES).enumerate() {
        let quant_start = block_index * Q4_0_QUANT_BYTES;
        let scale_start = block_index * Q4_0_SCALE_BYTES;
        quantize_q4_0_block(
            block,
            &mut quants[quant_start..quant_start + Q4_0_QUANT_BYTES],
            &mut scales[scale_start..scale_start + Q4_0_SCALE_BYTES],
        );
    }
    Ok(packed)
}

fn quantize_q4_0_bf16_le_block(input: &[u8], quants: &mut [u8], scale: &mut [u8]) {
    debug_assert_eq!(input.len(), Q4_0_BLOCK_VALUES * 2);
    let mut values = [0u16; Q4_0_BLOCK_VALUES];
    for (destination, pair) in values.iter_mut().zip(input.chunks_exact(2)) {
        *destination = u16::from_le_bytes([pair[0], pair[1]]);
    }
    quantize_q4_0_block(&values, quants, scale);
}

fn quantize_q4_0_block(input: &[u16], quants: &mut [u8], scale_out: &mut [u8]) {
    debug_assert_eq!(input.len(), Q4_0_BLOCK_VALUES);
    debug_assert_eq!(quants.len(), Q4_0_QUANT_BYTES);
    debug_assert_eq!(scale_out.len(), Q4_0_SCALE_BYTES);

    let mut values = [0.0f32; Q4_0_BLOCK_VALUES];
    let mut max_abs = 0.0f32;
    let mut signed_max = 0.0f32;
    for (destination, bits) in values.iter_mut().zip(input) {
        let value = bf16_bits_to_f32(*bits);
        *destination = value;
        let absolute = value.abs();
        if absolute > max_abs {
            max_abs = absolute;
            signed_max = value;
        }
    }

    // ggml Q4_0: strict `>` makes the first max-magnitude value choose the
    // signed scale, including an exact positive/negative tie.
    let scale = signed_max / -8.0;
    let inverse_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };
    scale_out.copy_from_slice(&crate::tensor::f32_to_f16_bits(scale).to_le_bytes());
    for index in 0..Q4_0_QUANT_BYTES {
        let low = (values[index] * inverse_scale + 8.5)
            .floor()
            .clamp(0.0, 15.0) as u8;
        let high = (values[index + Q4_0_QUANT_BYTES] * inverse_scale + 8.5)
            .floor()
            .clamp(0.0, 15.0) as u8;
        quants[index] = (low & 0x0f) | ((high & 0x0f) << 4);
    }
}

#[inline]
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn invalid(message: impl Into<String>) -> BackendError {
    BackendError::InvalidModelMetadata(message.into())
}

fn io_error(path: &Path, source: std::io::Error) -> BackendError {
    BackendError::Io {
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(value: f32) -> u16 {
        // All test inputs are exactly representable in BF16 except where the
        // bit pattern itself is supplied directly.
        (value.to_bits() >> 16) as u16
    }

    fn synthetic_official_header() -> Vec<u8> {
        let mut root = serde_json::Map::new();
        root.insert("__metadata__".into(), serde_json::json!({ "format": "pt" }));
        let mut cursor = 0u64;
        for (name, shape) in required_tensor_specs() {
            let elements = shape.iter().product::<u64>();
            let end = cursor + elements * 2;
            root.insert(
                name.into(),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [cursor, end],
                }),
            );
            cursor = end;
        }
        assert_eq!(cursor, GEMMA4_MTP_CUDA_SOURCE_PAYLOAD_BYTES);
        serde_json::to_vec(&root).unwrap()
    }

    #[test]
    fn official_ledger_pins_every_matrix_and_resident_byte() {
        let ledger = gemma4_mtp_cuda_ledger().unwrap();
        assert_eq!(ledger.matrices.len(), 23);
        assert_eq!(ledger.f32_tensors.len(), 25);
        assert_eq!(ledger.matrix_elements, 419_692_544);
        assert_eq!(ledger.source_matrix_bytes, 839_385_088);
        assert_eq!(ledger.matrix_bytes, 236_077_056);
        assert_eq!(ledger.f32_values, 18_692);
        assert_eq!(ledger.f32_bytes, 74_768);
        assert_eq!(ledger.total_bytes, 236_151_824);

        let expected = [
            ("model.embed_tokens.weight", 262_144, 1_024, 150_994_944),
            (
                "model.layers.0.self_attn.q_proj.weight",
                4_096,
                1_024,
                2_359_296,
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                1_024,
                4_096,
                2_359_296,
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                8_192,
                1_024,
                4_718_592,
            ),
            ("model.layers.0.mlp.up_proj.weight", 8_192, 1_024, 4_718_592),
            (
                "model.layers.0.mlp.down_proj.weight",
                1_024,
                8_192,
                4_718_592,
            ),
            (
                "model.layers.1.self_attn.q_proj.weight",
                4_096,
                1_024,
                2_359_296,
            ),
            (
                "model.layers.1.self_attn.o_proj.weight",
                1_024,
                4_096,
                2_359_296,
            ),
            (
                "model.layers.1.mlp.gate_proj.weight",
                8_192,
                1_024,
                4_718_592,
            ),
            ("model.layers.1.mlp.up_proj.weight", 8_192, 1_024, 4_718_592),
            (
                "model.layers.1.mlp.down_proj.weight",
                1_024,
                8_192,
                4_718_592,
            ),
            (
                "model.layers.2.self_attn.q_proj.weight",
                4_096,
                1_024,
                2_359_296,
            ),
            (
                "model.layers.2.self_attn.o_proj.weight",
                1_024,
                4_096,
                2_359_296,
            ),
            (
                "model.layers.2.mlp.gate_proj.weight",
                8_192,
                1_024,
                4_718_592,
            ),
            ("model.layers.2.mlp.up_proj.weight", 8_192, 1_024, 4_718_592),
            (
                "model.layers.2.mlp.down_proj.weight",
                1_024,
                8_192,
                4_718_592,
            ),
            (
                "model.layers.3.self_attn.q_proj.weight",
                8_192,
                1_024,
                4_718_592,
            ),
            (
                "model.layers.3.self_attn.o_proj.weight",
                1_024,
                8_192,
                4_718_592,
            ),
            (
                "model.layers.3.mlp.gate_proj.weight",
                8_192,
                1_024,
                4_718_592,
            ),
            ("model.layers.3.mlp.up_proj.weight", 8_192, 1_024, 4_718_592),
            (
                "model.layers.3.mlp.down_proj.weight",
                1_024,
                8_192,
                4_718_592,
            ),
            ("pre_projection.weight", 1_024, 5_632, 3_244_032),
            ("post_projection.weight", 2_816, 1_024, 1_622_016),
        ];
        let mut cursor = 0u64;
        for (layout, (name, rows, cols, bytes)) in ledger.matrices.iter().zip(expected) {
            assert_eq!((layout.name, layout.rows, layout.cols), (name, rows, cols));
            assert_eq!(layout.pack_byte_offset, cursor);
            assert_eq!(layout.packed_bytes, bytes);
            assert_eq!(layout.scales_byte_offset, layout.quants_bytes);
            assert_eq!(layout.quants_bytes + layout.scale_bytes, bytes);
            cursor += bytes;
        }
        assert_eq!(cursor, 236_077_056);
    }

    #[test]
    fn typed_device_slots_cover_the_manifest_in_ledger_order() {
        let ledger = gemma4_mtp_cuda_ledger().unwrap();
        assert_eq!(Gemma4MtpCudaMatrixId::ALL.len(), ledger.matrices.len());
        for (slot, (id, layout)) in Gemma4MtpCudaMatrixId::ALL
            .into_iter()
            .zip(&ledger.matrices)
            .enumerate()
        {
            assert_eq!(id.index(), slot);
            assert_eq!(layout.id, id);
            assert_eq!(layout.name, id.name());
        }

        assert_eq!(Gemma4MtpCudaF32Id::ALL.len(), ledger.f32_tensors.len());
        for (slot, (id, layout)) in Gemma4MtpCudaF32Id::ALL
            .into_iter()
            .zip(&ledger.f32_tensors)
            .enumerate()
        {
            assert_eq!(id.index(), slot);
            assert_eq!(layout.id, id);
            assert_eq!(layout.name, id.name());
        }
        assert_eq!(
            Gemma4MtpCudaMatrixId::PreProjection.name(),
            "pre_projection.weight"
        );
        assert_eq!(Gemma4MtpCudaF32Id::FinalNorm.name(), "model.norm.weight");
    }

    #[test]
    fn official_f32_ledger_pins_layer_widths_and_offsets() {
        let ledger = gemma4_mtp_cuda_ledger().unwrap();
        assert_eq!(ledger.f32_tensors[0].element_offset, 0);
        assert_eq!(ledger.f32_tensors[1].elements, 256);
        assert_eq!(ledger.f32_tensors[19].elements, 512);
        assert_eq!(ledger.f32_tensors.last().unwrap().name, "model.norm.weight");
        assert_eq!(
            ledger.f32_tensors.last().unwrap().element_offset + 1_024,
            18_692
        );
        for pair in ledger.f32_tensors.windows(2) {
            assert_eq!(
                pair[0].element_offset + pair[0].elements as u64,
                pair[1].element_offset
            );
        }
    }

    #[test]
    fn strict_manifest_accepts_only_the_complete_bf16_tensor_set() {
        let header = synthetic_official_header();
        let tensors =
            parse_and_validate_header(&header, GEMMA4_MTP_CUDA_SOURCE_PAYLOAD_BYTES).unwrap();
        assert_eq!(tensors.len(), 48);

        let mut wrong_shape: serde_json::Value = serde_json::from_slice(&header).unwrap();
        wrong_shape["model.layers.3.self_attn.q_proj.weight"]["shape"] =
            serde_json::json!([4_096, 1_024]);
        let wrong_shape = serde_json::to_vec(&wrong_shape).unwrap();
        assert!(
            parse_and_validate_header(&wrong_shape, GEMMA4_MTP_CUDA_SOURCE_PAYLOAD_BYTES).is_err()
        );

        let mut wrong_dtype: serde_json::Value = serde_json::from_slice(&header).unwrap();
        wrong_dtype["model.norm.weight"]["dtype"] = serde_json::json!("F32");
        let wrong_dtype = serde_json::to_vec(&wrong_dtype).unwrap();
        assert!(
            parse_and_validate_header(&wrong_dtype, GEMMA4_MTP_CUDA_SOURCE_PAYLOAD_BYTES).is_err()
        );
    }

    #[test]
    fn q4_zero_block_preserves_negative_zero_and_center_codes() {
        let packed = pack_bf16_q4_0_soa(&[0u16; 32], 1, 32).unwrap();
        assert_eq!(&packed[..16], &[0x88; 16]);
        assert_eq!(&packed[16..], &[0x00, 0x80]);
    }

    #[test]
    fn q4_first_max_abs_wins_positive_negative_ties() {
        let mut positive_first = [0u16; 32];
        positive_first[0] = bf16(8.0);
        positive_first[1] = bf16(-8.0);
        let positive_first = pack_bf16_q4_0_soa(&positive_first, 1, 32).unwrap();
        assert_eq!(
            u16::from_le_bytes([positive_first[16], positive_first[17]]),
            0xbc00
        );
        assert_eq!(&positive_first[..2], &[0x80, 0x8f]);

        let mut negative_first = [0u16; 32];
        negative_first[0] = bf16(-8.0);
        negative_first[1] = bf16(8.0);
        let negative_first = pack_bf16_q4_0_soa(&negative_first, 1, 32).unwrap();
        assert_eq!(
            u16::from_le_bytes([negative_first[16], negative_first[17]]),
            0x3c00
        );
        assert_eq!(&negative_first[..2], &[0x80, 0x8f]);
    }

    #[test]
    fn q4_nibble_half_steps_and_saturation_match_ggml() {
        let mut input = [0u16; 32];
        input[0] = bf16(-8.0); // selects +1.0 scale
        input[1] = bf16(0.5); // floor(0.5 + 8.5) = 9
        input[2] = bf16(-0.5); // floor(-0.5 + 8.5) = 8
        input[3] = bf16(8.0); // clamps 16 to 15
        let packed = pack_bf16_q4_0_soa(&input, 1, 32).unwrap();
        assert_eq!(&packed[..4], &[0x80, 0x89, 0x88, 0x8f]);
        assert_eq!(&packed[16..], &[0x00, 0x3c]);
    }

    #[test]
    fn q4_scale_conversion_pins_rne_ties_and_subnormals() {
        // Halfway between f16 1.0 (even) and its successor rounds to 1.0.
        assert_eq!(
            crate::tensor::f32_to_f16_bits(1.0 + 2.0f32.powi(-11)),
            0x3c00
        );
        // Halfway between odd 0x3c01 and even 0x3c02 rounds upward to even.
        assert_eq!(
            crate::tensor::f32_to_f16_bits(1.0 + 3.0 * 2.0f32.powi(-11)),
            0x3c02
        );
        assert_eq!(crate::tensor::f32_to_f16_bits(2.0f32.powi(-25)), 0x0000);
        assert_eq!(
            crate::tensor::f32_to_f16_bits(f32::from_bits(0x3300_0001)),
            0x0001
        );

        // BF16 0x3481 divided by -8 lands just beyond the f16 subnormal tie.
        let mut positive = [0u16; 32];
        positive[0] = 0x3481;
        let packed = pack_bf16_q4_0_soa(&positive, 1, 32).unwrap();
        assert_eq!(u16::from_le_bytes([packed[16], packed[17]]), 0x8001);
        let mut negative = [0u16; 32];
        negative[0] = 0xb481;
        let packed = pack_bf16_q4_0_soa(&negative, 1, 32).unwrap();
        assert_eq!(u16::from_le_bytes([packed[16], packed[17]]), 0x0001);
    }

    #[test]
    fn q4_direct_packer_matches_the_canonical_ggml_block_encoder() {
        let mut state = 0x6d74_7051u32;
        for _ in 0..128 {
            let mut input = [0u16; 32];
            for bits in &mut input {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let sign = ((state >> 31) as u16) << 15;
                let exponent = (120 + ((state >> 23) % 11)) as u16;
                let mantissa = ((state >> 16) & 0x7f) as u16;
                *bits = sign | (exponent << 7) | mantissa;
            }
            let f32_values: [f32; 32] = input.map(bf16_bits_to_f32);
            let expected = crate::tensor::kv_quant::quantize_block_q4_0(&f32_values);
            let packed = pack_bf16_q4_0_soa(&input, 1, 32).unwrap();
            assert_eq!(&packed[..16], &expected.qs);
            assert_eq!(u16::from_le_bytes([packed[16], packed[17]]), expected.scale);
        }
    }

    #[test]
    fn q4_two_block_output_is_quants_first_soa() {
        let mut input = [0u16; 64];
        input[32] = bf16(-8.0);
        let packed = pack_bf16_q4_0_soa(&input, 1, 64).unwrap();
        assert_eq!(packed.len(), 36);
        assert_eq!(&packed[..16], &[0x88; 16]);
        assert_eq!(packed[16], 0x80);
        assert!(packed[17..32].iter().all(|byte| *byte == 0x88));
        assert_eq!(&packed[32..], &[0x00, 0x80, 0x00, 0x3c]);
    }

    #[test]
    fn q4_packer_rejects_invalid_geometry_and_lengths() {
        assert!(pack_bf16_q4_0_soa(&[], 0, 32).is_err());
        assert!(pack_bf16_q4_0_soa(&[0; 31], 1, 31).is_err());
        assert!(pack_bf16_q4_0_soa(&[0; 31], 1, 32).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device and CAMELID_GEMMA4_MTP_SAFETENSORS; uploads 236 MiB"]
    fn official_device_upload_matches_resident_ledger() {
        let path = std::env::var_os("CAMELID_GEMMA4_MTP_SAFETENSORS")
            .expect("set CAMELID_GEMMA4_MTP_SAFETENSORS to the official BF16 checkpoint");
        let source = Gemma4MtpCudaSource::open(path).unwrap();
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let weights = source
            .load_cuda_weights(&context, &stream)
            .expect("stream and upload the official assistant");

        weights.validate_resident_accounting().unwrap();
        assert_eq!(weights.accounting().matrix_allocations, 23);
        assert_eq!(weights.accounting().matrix_bytes, 236_077_056);
        assert_eq!(weights.accounting().f32_allocations, 1);
        assert_eq!(weights.accounting().f32_values, 18_692);
        assert_eq!(weights.accounting().f32_bytes, 74_768);
        assert_eq!(weights.accounting().total_bytes, 236_151_824);
        for id in Gemma4MtpCudaMatrixId::ALL {
            assert_eq!(
                weights.matrix(id).num_bytes() as u64,
                weights.matrix_layout(id).packed_bytes
            );
        }
    }
}
