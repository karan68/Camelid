use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use rayon::prelude::*;

use crate::{
    gguf::{GgufFile, GgufTensorDescriptor, GgufTensorType},
    BackendError, Result,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    pub dims: Vec<usize>,
}

impl TensorShape {
    pub fn from_gguf_dims(dims: &[u64]) -> Result<Self> {
        let dims = dims
            .iter()
            .map(|dim| {
                usize::try_from(*dim).map_err(|_| {
                    BackendError::InvalidTensorData(format!("dimension {dim} does not fit usize"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { dims })
    }

    pub fn element_count(&self) -> Result<usize> {
        self.dims.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim).ok_or_else(|| {
                BackendError::InvalidTensorData("tensor element count overflow".to_string())
            })
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeDType {
    F32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q8_0Block {
    pub scale: f32,
    pub quants: [i8; 32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuTensor {
    pub name: String,
    pub shape: TensorShape,
    pub dtype: RuntimeDType,
    pub source_type: Option<GgufTensorType>,
    pub q8_0_blocks: Option<Vec<Q8_0Block>>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q8_0TensorBlocks {
    pub name: String,
    pub shape: TensorShape,
    pub blocks: Vec<Q8_0Block>,
}

impl Q8_0TensorBlocks {
    pub fn element_count(&self) -> Result<usize> {
        self.shape.element_count()
    }

    pub fn byte_size_if_f32_materialized(&self) -> Result<usize> {
        self.element_count()?.checked_mul(4).ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "tensor {} f32 materialization byte size overflow",
                self.name
            ))
        })
    }

    pub fn dequantize_elements(&self, start: usize, len: usize) -> Result<Vec<f32>> {
        const BLOCK_VALUES: usize = 32;
        let end = start.checked_add(len).ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "tensor {} q8_0 dequant range overflows usize",
                self.name
            ))
        })?;
        let element_count = self.element_count()?;
        if end > element_count {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "tensor {} q8_0 dequant range {start}..{end} exceeds element count {element_count}",
                self.name
            )));
        }

        let mut out = Vec::with_capacity(len);
        for element_idx in start..end {
            let block_idx = element_idx / BLOCK_VALUES;
            let quant_idx = element_idx % BLOCK_VALUES;
            let block = self.blocks.get(block_idx).ok_or_else(|| {
                BackendError::InvalidTensorData(format!(
                    "tensor {} q8_0 block index {block_idx} missing for element {element_idx}",
                    self.name
                ))
            })?;
            out.push(block.scale * f32::from(block.quants[quant_idx]));
        }
        Ok(out)
    }

    pub fn dequantize_row(&self, row: usize) -> Result<Vec<f32>> {
        let (_rows, cols) = self.rank2_row_shape(row, "row dequant")?;
        self.dequantize_elements(row * cols, cols)
    }

    pub fn dot_row_f32(&self, row: usize, input: &[f32]) -> Result<f32> {
        const BLOCK_VALUES: usize = 32;
        let (_rows, cols) = self.rank2_row_shape(row, "row dot")?;
        if input.len() != cols {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "tensor {} q8_0 row dot expected input width {cols}, got {}",
                self.name,
                input.len()
            )));
        }

        let row_start = row.checked_mul(cols).ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "tensor {} q8_0 row dot offset overflows usize",
                self.name
            ))
        })?;
        let mut sum = 0.0f32;
        for (col, input_value) in input.iter().enumerate() {
            let element_idx = row_start + col;
            let block_idx = element_idx / BLOCK_VALUES;
            let quant_idx = element_idx % BLOCK_VALUES;
            let block = self.blocks.get(block_idx).ok_or_else(|| {
                BackendError::InvalidTensorData(format!(
                    "tensor {} q8_0 block index {block_idx} missing for row {row} col {col}",
                    self.name
                ))
            })?;
            sum += (block.scale * f32::from(block.quants[quant_idx])) * input_value;
        }
        Ok(sum)
    }

    pub fn dot_all_rows_f32(&self, input: &[f32], name: impl Into<String>) -> Result<CpuTensor> {
        const BLOCK_VALUES: usize = 32;
        let (rows, cols) = self.rank2_shape("all-row dot")?;
        if input.len() != cols {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "tensor {} q8_0 all-row dot expected input width {cols}, got {}",
                self.name,
                input.len()
            )));
        }

        let mut data = Vec::with_capacity(rows);
        if cols % BLOCK_VALUES == 0 {
            let blocks_per_row = cols / BLOCK_VALUES;
            let expected_blocks = rows.checked_mul(blocks_per_row).ok_or_else(|| {
                BackendError::InvalidTensorData(format!(
                    "tensor {} q8_0 all-row dot block count overflows usize",
                    self.name
                ))
            })?;
            if self.blocks.len() != expected_blocks {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "tensor {} q8_0 all-row dot expected {expected_blocks} blocks for shape {:?}, got {}",
                    self.name,
                    self.shape.dims,
                    self.blocks.len()
                )));
            }

            for row_blocks in self.blocks.chunks_exact(blocks_per_row) {
                let mut row_sum = 0.0_f32;
                for (block, input_block) in row_blocks.iter().zip(input.chunks_exact(BLOCK_VALUES))
                {
                    for (quant, input_value) in block.quants.iter().zip(input_block) {
                        row_sum += (block.scale * f32::from(*quant)) * input_value;
                    }
                }
                data.push(row_sum);
            }
        } else {
            for row in 0..rows {
                data.push(self.dot_row_f32(row, input)?);
            }
        }

        Ok(CpuTensor {
            name: name.into(),
            shape: TensorShape { dims: vec![rows] },
            dtype: RuntimeDType::F32,
            source_type: None,
            q8_0_blocks: None,
            data,
        })
    }

    pub fn dot_single_input_row_f32(
        &self,
        input: &CpuTensor,
        name: impl Into<String>,
    ) -> Result<CpuTensor> {
        if input.shape.dims.len() != 2 || input.shape.dims[0] != 1 {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "tensor {} q8_0 lazy linear expected single input row, got {:?}",
                self.name, input.shape.dims
            )));
        }
        let mut output = self.dot_all_rows_f32(&input.data, name)?;
        output.shape.dims.insert(0, 1);
        Ok(output)
    }

    fn rank2_shape(&self, op: &str) -> Result<(usize, usize)> {
        if self.shape.dims.len() != 2 {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "tensor {} q8_0 {op} requires rank-2 shape, got {:?}",
                self.name, self.shape.dims
            )));
        }
        let rows = self.shape.dims[0];
        let cols = self.shape.dims[1];
        Ok((rows, cols))
    }

    fn rank2_row_shape(&self, row: usize, op: &str) -> Result<(usize, usize)> {
        let (rows, cols) = self.rank2_shape(op)?;
        if row >= rows {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "tensor {} q8_0 row {row} out of range for {rows} rows",
                self.name
            )));
        }
        Ok((rows, cols))
    }
}

impl CpuTensor {
    pub fn from_f32(name: impl Into<String>, dims: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let shape = TensorShape { dims };
        let expected = shape.element_count()?;
        if expected != data.len() {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "tensor data length {} does not match shape element count {expected}",
                data.len()
            )));
        }
        Ok(Self {
            name: name.into(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: None,
            q8_0_blocks: None,
            data,
        })
    }

    pub fn from_f32_with_source_type(
        name: impl Into<String>,
        dims: Vec<usize>,
        data: Vec<f32>,
        source_type: Option<GgufTensorType>,
    ) -> Result<Self> {
        let mut tensor = Self::from_f32(name, dims, data)?;
        tensor.source_type = source_type;
        Ok(tensor)
    }

    pub fn from_f32_with_q8_0_blocks(
        name: impl Into<String>,
        dims: Vec<usize>,
        data: Vec<f32>,
        q8_0_blocks: Vec<Q8_0Block>,
    ) -> Result<Self> {
        let mut tensor = Self::from_f32(name, dims, data)?;
        tensor.source_type = Some(GgufTensorType::Q8_0);
        tensor.q8_0_blocks = Some(q8_0_blocks);
        Ok(tensor)
    }

    pub fn rank(&self) -> usize {
        self.shape.dims.len()
    }

    pub fn dim(&self, idx: usize) -> Result<usize> {
        self.shape.dims.get(idx).copied().ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(format!(
                "tensor {} rank {} has no dimension {idx}",
                self.name,
                self.rank()
            ))
        })
    }

    pub fn matmul(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        require_rank(self, 2, "matmul lhs")?;
        require_rank(rhs, 2, "matmul rhs")?;
        let m = self.dim(0)?;
        let k = self.dim(1)?;
        let rhs_k = rhs.dim(0)?;
        let n = rhs.dim(1)?;
        if k != rhs_k {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "matmul shape mismatch: lhs {:?}, rhs {:?}",
                self.shape.dims, rhs.shape.dims
            )));
        }
        let mut out = vec![0.0; m * n];
        if should_parallelize_linear_output(n) {
            for row in 0..m {
                let lhs_start = row * k;
                let out_start = row * n;
                let out_row = &mut out[out_start..out_start + n];
                out_row
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(col, out_value)| {
                        let mut sum = 0.0;
                        for inner in 0..k {
                            let lhs_value = self.data[lhs_start + inner];
                            if lhs_value == 0.0 {
                                continue;
                            }
                            sum += lhs_value * rhs.data[inner * n + col];
                        }
                        *out_value = sum;
                    });
            }
        } else if should_parallelize_linear_output(m * n) {
            out.par_chunks_mut(n)
                .enumerate()
                .for_each(|(row, out_row)| {
                    let lhs_start = row * k;
                    for inner in 0..k {
                        let lhs_value = self.data[lhs_start + inner];
                        if lhs_value == 0.0 {
                            continue;
                        }
                        let rhs_start = inner * n;
                        let rhs_row = &rhs.data[rhs_start..rhs_start + n];
                        for col in 0..n {
                            out_row[col] += lhs_value * rhs_row[col];
                        }
                    }
                });
        } else {
            for row in 0..m {
                let lhs_start = row * k;
                let out_start = row * n;
                let out_row = &mut out[out_start..out_start + n];
                for inner in 0..k {
                    let lhs_value = self.data[lhs_start + inner];
                    if lhs_value == 0.0 {
                        continue;
                    }
                    let rhs_start = inner * n;
                    let rhs_row = &rhs.data[rhs_start..rhs_start + n];
                    for col in 0..n {
                        out_row[col] += lhs_value * rhs_row[col];
                    }
                }
            }
        }
        Self::from_f32(name, vec![m, n], out)
    }

    pub fn matmul_rhs_transposed(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        require_rank(self, 2, "matmul rhs-transposed lhs")?;
        require_rank(rhs, 2, "matmul rhs-transposed rhs")?;
        let m = self.dim(0)?;
        let k = self.dim(1)?;
        let n = rhs.dim(0)?;
        let rhs_k = rhs.dim(1)?;
        if k != rhs_k {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "matmul rhs-transposed shape mismatch: lhs {:?}, rhs {:?}",
                self.shape.dims, rhs.shape.dims
            )));
        }
        let mut out = vec![0.0; m * n];
        if should_parallelize_linear_output(n) {
            for row in 0..m {
                let lhs_start = row * k;
                let lhs_row = &self.data[lhs_start..lhs_start + k];
                let out_start = row * n;
                let out_row = &mut out[out_start..out_start + n];
                out_row
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(col, out_value)| {
                        let rhs_start = col * k;
                        let rhs_row = &rhs.data[rhs_start..rhs_start + k];
                        *out_value = dot_product(lhs_row, rhs_row);
                    });
            }
        } else if should_parallelize_linear_output(m * n) {
            out.par_chunks_mut(n)
                .enumerate()
                .for_each(|(row, out_row)| {
                    let lhs_start = row * k;
                    let lhs_row = &self.data[lhs_start..lhs_start + k];
                    for (col, out_value) in out_row.iter_mut().enumerate() {
                        let rhs_start = col * k;
                        let rhs_row = &rhs.data[rhs_start..rhs_start + k];
                        *out_value = dot_product(lhs_row, rhs_row);
                    }
                });
        } else {
            for row in 0..m {
                let lhs_start = row * k;
                let lhs_row = &self.data[lhs_start..lhs_start + k];
                let out_start = row * n;
                let out_row = &mut out[out_start..out_start + n];
                for (col, out_value) in out_row.iter_mut().enumerate() {
                    let rhs_start = col * k;
                    let rhs_row = &rhs.data[rhs_start..rhs_start + k];
                    *out_value = dot_product(lhs_row, rhs_row);
                }
            }
        }
        Self::from_f32(name, vec![m, n], out)
    }

    pub fn add(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        self.zip_same_shape(rhs, name, |a, b| a + b)
    }

    pub fn mul(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        self.zip_same_shape(rhs, name, |a, b| a * b)
    }

    pub fn silu_mul(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        self.zip_same_shape(rhs, name, |a, b| (a / (1.0 + (-a).exp())) * b)
    }

    pub fn silu(&self, name: impl Into<String>) -> Result<Self> {
        Self::from_f32(
            name,
            self.shape.dims.clone(),
            self.data.iter().map(|x| x / (1.0 + (-x).exp())).collect(),
        )
    }

    pub fn rms_norm(&self, weight: &Self, eps: f32, name: impl Into<String>) -> Result<Self> {
        require_rank(self, 2, "rms_norm input")?;
        require_rank(weight, 1, "rms_norm weight")?;
        let rows = self.dim(0)?;
        let cols = self.dim(1)?;
        if weight.dim(0)? != cols {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "rms_norm weight shape {:?} does not match input shape {:?}",
                weight.shape.dims, self.shape.dims
            )));
        }
        let mut out = vec![0.0; self.data.len()];
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let mean_square =
                self.data[start..end].iter().map(|v| v * v).sum::<f32>() / cols as f32;
            let scale = 1.0 / (mean_square + eps).sqrt();
            for col in 0..cols {
                out[start + col] = self.data[start + col] * scale * weight.data[col];
            }
        }
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

    pub fn softmax_last_dim(&self, name: impl Into<String>) -> Result<Self> {
        if self.shape.dims.is_empty() {
            return Err(BackendError::RuntimeShapeMismatch(
                "softmax requires at least one dimension".to_string(),
            ));
        }
        let cols = *self.shape.dims.last().expect("non-empty dims");
        if cols == 0 || !self.data.len().is_multiple_of(cols) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "softmax invalid shape {:?} for data length {}",
                self.shape.dims,
                self.data.len()
            )));
        }
        let mut out = self.data.clone();
        for row in out.chunks_exact_mut(cols) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            if sum == 0.0 || !sum.is_finite() {
                return Err(BackendError::RuntimeShapeMismatch(
                    "softmax produced invalid normalization sum".to_string(),
                ));
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

    pub fn embedding_lookup(&self, token_ids: &[u32], name: impl Into<String>) -> Result<Self> {
        require_rank(self, 2, "embedding weight")?;
        let vocab = self.dim(0)?;
        let width = self.dim(1)?;
        let mut out = Vec::with_capacity(token_ids.len() * width);
        for token_id in token_ids {
            let token_idx = usize::try_from(*token_id).map_err(|_| {
                BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} does not fit usize"
                ))
            })?;
            if token_idx >= vocab {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} out of range for vocab size {vocab}"
                )));
            }
            let start = token_idx * width;
            out.extend_from_slice(&self.data[start..start + width]);
        }
        Self::from_f32(name, vec![token_ids.len(), width], out)
    }

    pub fn transpose_2d(&self, name: impl Into<String>) -> Result<Self> {
        require_rank(self, 2, "transpose")?;
        let rows = self.dim(0)?;
        let cols = self.dim(1)?;
        let mut out = vec![0.0; self.data.len()];
        for row in 0..rows {
            for col in 0..cols {
                out[col * rows + row] = self.data[row * cols + col];
            }
        }
        Self::from_f32(name, vec![cols, rows], out)
    }

    fn zip_same_shape(
        &self,
        rhs: &Self,
        name: impl Into<String>,
        f: impl Fn(f32, f32) -> f32,
    ) -> Result<Self> {
        if self.shape != rhs.shape {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "shape mismatch: lhs {:?}, rhs {:?}",
                self.shape.dims, rhs.shape.dims
            )));
        }
        Self::from_f32(
            name,
            self.shape.dims.clone(),
            self.data
                .iter()
                .zip(rhs.data.iter())
                .map(|(a, b)| f(*a, *b))
                .collect(),
        )
    }
}

fn require_rank(tensor: &CpuTensor, rank: usize, op: &str) -> Result<()> {
    if tensor.rank() != rank {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "{op} expected rank {rank}, got shape {:?}",
            tensor.shape.dims
        )));
    }
    Ok(())
}

fn dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut sum = 0.0;
    let mut idx = 0;
    while idx + 4 <= lhs.len() {
        sum += lhs[idx] * rhs[idx];
        sum += lhs[idx + 1] * rhs[idx + 1];
        sum += lhs[idx + 2] * rhs[idx + 2];
        sum += lhs[idx + 3] * rhs[idx + 3];
        idx += 4;
    }
    while idx < lhs.len() {
        sum += lhs[idx] * rhs[idx];
        idx += 1;
    }
    sum
}
const DEFAULT_PARALLEL_LINEAR_MIN_OUTPUTS: usize = 1024;

pub(crate) fn should_parallelize_linear_output(output_width: usize) -> bool {
    parallel_linear_enabled()
        && output_width >= parallel_linear_min_outputs()
        && rayon::current_num_threads() > 1
}

fn parallel_linear_enabled() -> bool {
    match env::var("BACKENDINFERENCE_PARALLEL_LINEAR") {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes" | "enabled"
        ),
        Err(_) => false,
    }
}

fn parallel_linear_min_outputs() -> usize {
    env::var("BACKENDINFERENCE_PARALLEL_LINEAR_MIN_OUTPUTS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_PARALLEL_LINEAR_MIN_OUTPUTS)
}

pub struct TensorStore {
    path: PathBuf,
    descriptors: HashMap<String, GgufTensorDescriptor>,
}

impl TensorStore {
    pub fn open(path: impl AsRef<Path>, gguf: &GgufFile) -> Self {
        let descriptors = gguf
            .tensors
            .iter()
            .cloned()
            .map(|desc| (desc.name.clone(), desc))
            .collect();
        Self {
            path: path.as_ref().to_path_buf(),
            descriptors,
        }
    }

    pub fn descriptor(&self, name: &str) -> Result<&GgufTensorDescriptor> {
        self.descriptors
            .get(name)
            .ok_or_else(|| BackendError::TensorNotFound(name.to_string()))
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let desc = self.descriptor(name)?;
        let len = usize::try_from(desc.n_bytes).map_err(|_| {
            BackendError::InvalidTensorData(format!("tensor {name} byte length does not fit usize"))
        })?;
        let mut file = File::open(&self.path).map_err(|source| BackendError::Io {
            path: self.path.clone(),
            source,
        })?;
        file.seek(SeekFrom::Start(desc.absolute_offset))
            .map_err(|source| BackendError::Io {
                path: self.path.clone(),
                source,
            })?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes)
            .map_err(|source| BackendError::Io {
                path: self.path.clone(),
                source,
            })?;
        Ok(bytes)
    }

    pub fn load_q8_0_blocks(&self, name: &str) -> Result<Q8_0TensorBlocks> {
        let desc = self.descriptor(name)?.clone();
        if desc.tensor_type != GgufTensorType::Q8_0 {
            return Err(BackendError::UnsupportedTensorType(format!(
                "tensor {name} has storage type {:?}; q8_0 block-only load requires Q8_0",
                desc.tensor_type
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        let expected_elements = shape.element_count()?;
        let blocks = decode_q8_0_blocks(name, &bytes, expected_elements)?;
        Ok(Q8_0TensorBlocks {
            name: name.to_string(),
            shape,
            blocks,
        })
    }

    pub fn load_cpu_f32(&self, name: &str) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        let bytes = self.tensor_bytes(name)?;
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        let expected_elements = shape.element_count()?;
        let mut q8_0_blocks = None;
        let data = match desc.tensor_type {
            GgufTensorType::F32 => decode_f32_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::F16 => decode_f16_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::BF16 => decode_bf16_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q8_0 => {
                let decoded = decode_q8_0_tensor(name, &bytes, expected_elements)?;
                q8_0_blocks = Some(decode_q8_0_blocks(name, &bytes, expected_elements)?);
                decoded
            }
            other => {
                return Err(BackendError::UnsupportedTensorType(format!(
                    "tensor {name} has unsupported storage type {other:?}; supported for CPU f32 load: F32, F16, BF16, Q8_0"
                )))
            }
        };
        Ok(CpuTensor {
            name: name.to_string(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(desc.tensor_type),
            q8_0_blocks,
            data,
        })
    }
}

fn decode_f32_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    if bytes.len() != expected_elements * 4 {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} f32 byte length {} does not match expected {}",
            bytes.len(),
            expected_elements * 4
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("exact chunk length")))
        .collect())
}

fn decode_f16_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    if bytes.len() != expected_elements * 2 {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} f16 byte length {} does not match expected {}",
            bytes.len(),
            expected_elements * 2
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| {
            f16_bits_to_f32(u16::from_le_bytes(
                chunk.try_into().expect("exact chunk length"),
            ))
        })
        .collect())
}

fn decode_bf16_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    if bytes.len() != expected_elements * 2 {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} bf16 byte length {} does not match expected {}",
            bytes.len(),
            expected_elements * 2
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| {
            f32::from_bits(
                u32::from(u16::from_le_bytes(
                    chunk.try_into().expect("exact chunk length"),
                )) << 16,
            )
        })
        .collect())
}

fn decode_q8_0_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q8_0_blocks(name, bytes, expected_elements)?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        for q in block.quants {
            out.push(block.scale * f32::from(q));
        }
    }
    Ok(out)
}

fn decode_q8_0_blocks(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<Q8_0Block>> {
    const BLOCK_VALUES: usize = 32;
    const BLOCK_BYTES: usize = 34;
    if !expected_elements.is_multiple_of(BLOCK_VALUES) {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} q8_0 element count {expected_elements} is not divisible by {BLOCK_VALUES}"
        )));
    }
    let expected_bytes = expected_elements / BLOCK_VALUES * BLOCK_BYTES;
    if bytes.len() != expected_bytes {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} q8_0 byte length {} does not match expected {expected_bytes}",
            bytes.len()
        )));
    }
    let mut blocks = Vec::with_capacity(expected_elements / BLOCK_VALUES);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let mut quants = [0_i8; BLOCK_VALUES];
        for (idx, q) in block[2..].iter().enumerate() {
            quants[idx] = *q as i8;
        }
        blocks.push(Q8_0Block { scale, quants });
    }
    Ok(blocks)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits & 0x7c00) >> 10;
    let frac = u32::from(bits & 0x03ff);

    let out = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut mant = frac;
                let mut e = -14i32;
                while (mant & 0x0400) == 0 {
                    mant <<= 1;
                    e -= 1;
                }
                mant &= 0x03ff;
                let exp32 = u32::try_from(e + 127).expect("subnormal f16 exponent in range");
                sign | (exp32 << 23) | (mant << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp32 = u32::from(exp) + (127 - 15);
            sign | (exp32 << 23) | (frac << 13)
        }
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::{f16_bits_to_f32, CpuTensor};

    #[test]
    fn matmul_rhs_transposed_handles_single_row_vectors() {
        let lhs = CpuTensor::from_f32("lhs", vec![1, 5], vec![1.0, -2.0, 3.0, 0.5, 4.0]).unwrap();
        let rhs = CpuTensor::from_f32(
            "rhs_t",
            vec![3, 5],
            vec![
                2.0, 0.0, -1.0, 4.0, 0.5, // first output row
                -3.0, 1.0, 0.0, 2.0, -0.5, // second output row
                1.0, 1.0, 1.0, 1.0, 1.0, // third output row
            ],
        )
        .unwrap();

        let actual = lhs.matmul_rhs_transposed(&rhs, "out").unwrap();

        assert_eq!(actual.shape.dims, vec![1, 3]);
        assert_eq!(actual.data, vec![3.0, -6.0, 6.5]);
    }

    #[test]
    fn matmul_rhs_transposed_handles_rectangular_batches() {
        let lhs = CpuTensor::from_f32(
            "lhs",
            vec![2, 3],
            vec![
                1.0, 2.0, 3.0, // row 0
                4.0, 5.0, 6.0, // row 1
            ],
        )
        .unwrap();
        let rhs = CpuTensor::from_f32(
            "rhs_t",
            vec![2, 3],
            vec![
                7.0, 8.0, 9.0, // output 0
                1.0, 0.0, -1.0, // output 1
            ],
        )
        .unwrap();

        let actual = lhs.matmul_rhs_transposed(&rhs, "out").unwrap();

        assert_eq!(actual.shape.dims, vec![2, 2]);
        assert_eq!(actual.data, vec![50.0, -2.0, 122.0, -2.0]);
    }

    #[test]
    fn matmul_wide_output_matches_reference() {
        let lhs_values = vec![1.0, -2.0, 0.5, 3.0, -0.25];
        let output_width = 1031;
        let rhs_values = (0..lhs_values.len() * output_width)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.01)
            .collect::<Vec<_>>();
        let lhs =
            CpuTensor::from_f32("lhs", vec![1, lhs_values.len()], lhs_values.clone()).unwrap();
        let rhs = CpuTensor::from_f32(
            "rhs",
            vec![lhs_values.len(), output_width],
            rhs_values.clone(),
        )
        .unwrap();

        let actual = lhs.matmul(&rhs, "out").unwrap();

        let expected = (0..output_width)
            .map(|col| {
                lhs_values
                    .iter()
                    .enumerate()
                    .map(|(inner, lhs_value)| lhs_value * rhs_values[inner * output_width + col])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.shape.dims, vec![1, output_width]);
        assert_eq!(actual.data, expected);
    }

    #[test]
    fn matmul_rhs_transposed_wide_output_matches_reference() {
        let lhs_values = vec![1.0, -2.0, 0.5, 3.0, -0.25];
        let output_width = 1031;
        let rhs_values = (0..output_width * lhs_values.len())
            .map(|idx| ((idx % 41) as f32 - 20.0) * 0.01)
            .collect::<Vec<_>>();
        let lhs =
            CpuTensor::from_f32("lhs", vec![1, lhs_values.len()], lhs_values.clone()).unwrap();
        let rhs = CpuTensor::from_f32(
            "rhs_t",
            vec![output_width, lhs_values.len()],
            rhs_values.clone(),
        )
        .unwrap();

        let actual = lhs.matmul_rhs_transposed(&rhs, "out").unwrap();

        let expected = (0..output_width)
            .map(|row| {
                let row_start = row * lhs_values.len();
                lhs_values
                    .iter()
                    .zip(&rhs_values[row_start..row_start + lhs_values.len()])
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.shape.dims, vec![1, output_width]);
        assert_eq!(actual.data, expected);
    }

    #[test]
    fn converts_f16_bits_to_f32() {
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0xc000), -2.0);
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
    }
}
