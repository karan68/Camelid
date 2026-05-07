use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{Read, Seek, SeekFrom},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};

const RETAIN_Q8_BLOCKS_ENV: &str = "BACKENDINFERENCE_RETAIN_Q8_0_BLOCKS";
const Q8_FILE_CACHE_BYTES_ENV: &str = "BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES";
// Keep lazy Q8_0 file reads memory-safe by default. The bounded chunk cache is an
// explicit diagnostic/performance probe until long-context prefill has row-specific evidence.
const DEFAULT_Q8_FILE_CACHE_BYTES: usize = 0;

use rayon::prelude::*;
use serde::Serialize;

use crate::{
    gguf::{GgufFile, GgufTensorDescriptor, GgufTensorType},
    BackendError, Result,
};

#[cfg(target_os = "macos")]
pub(crate) fn disable_file_cache_best_effort(file: &File) {
    use std::{os::fd::AsRawFd, os::raw::c_int};

    const F_RDAHEAD: c_int = 45;
    const F_NOCACHE: c_int = 48;
    unsafe extern "C" {
        fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
    }

    // Best-effort only: the lazy Q8 path streams model bytes repeatedly, and on macOS the
    // default file cache/readahead can consume free pages even when Camelid RSS stays low.
    // Keep both calls non-fatal: older kernels/filesystems may reject one knob but honor the other.
    let _ = unsafe { fcntl(file.as_raw_fd(), F_RDAHEAD, 0) };
    let _ = unsafe { fcntl(file.as_raw_fd(), F_NOCACHE, 1) };
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn disable_file_cache_best_effort(_file: &File) {}

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

#[derive(Debug, Clone)]
pub struct Q8_0FileBacking {
    pub path: PathBuf,
    pub absolute_offset: u64,
    pub num_blocks: usize,
    file_handle: Arc<OnceLock<Arc<File>>>,
}

impl Q8_0FileBacking {
    pub fn new(path: PathBuf, absolute_offset: u64, num_blocks: usize) -> Self {
        Self {
            path,
            absolute_offset,
            num_blocks,
            file_handle: Arc::new(OnceLock::new()),
        }
    }

    pub fn file(&self) -> Result<Arc<File>> {
        if let Some(file) = self.file_handle.get() {
            return Ok(file.clone());
        }
        let file = File::open(&self.path).map_err(|source| BackendError::Io {
            path: self.path.clone(),
            source,
        })?;
        disable_file_cache_best_effort(&file);
        let file = Arc::new(file);
        if self.file_handle.set(file.clone()).is_err() {
            return Ok(self
                .file_handle
                .get()
                .expect("q8_0 file handle must exist after OnceLock set race")
                .clone());
        }
        Ok(file)
    }

    pub fn file_handle_cached(&self) -> bool {
        self.file_handle.get().is_some()
    }

    pub fn storage_bytes(&self) -> u64 {
        const Q8_0_BLOCK_BYTES: u64 = 34;
        (self.num_blocks as u64).saturating_mul(Q8_0_BLOCK_BYTES)
    }

    pub fn f32_materialization_bytes(&self) -> u64 {
        const Q8_0_BLOCK_VALUES: u64 = 32;
        (self.num_blocks as u64)
            .saturating_mul(Q8_0_BLOCK_VALUES)
            .saturating_mul(std::mem::size_of::<f32>() as u64)
    }

    pub fn retained_block_bytes(&self) -> u64 {
        (self.num_blocks as u64).saturating_mul(std::mem::size_of::<Q8_0Block>() as u64)
    }

    pub(crate) fn read_exact_at_cached(&self, out: &mut [u8], offset: u64) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        if q8_file_cache_get(&self.path, offset, out) {
            return Ok(());
        }
        let file = self.file()?;
        file.read_exact_at(out, offset)
            .map_err(|source| BackendError::Io {
                path: self.path.clone(),
                source,
            })?;
        record_q8_0_file_read(out.len());
        q8_file_cache_insert(self.path.clone(), offset, out);
        Ok(())
    }
}

impl PartialEq for Q8_0FileBacking {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.absolute_offset == other.absolute_offset
            && self.num_blocks == other.num_blocks
    }
}

impl Eq for Q8_0FileBacking {}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuTensor {
    pub name: String,
    pub shape: TensorShape,
    pub dtype: RuntimeDType,
    pub source_type: Option<GgufTensorType>,
    pub q8_0_blocks: Option<Vec<Q8_0Block>>,
    pub q8_0_file_backing: Option<Q8_0FileBacking>,
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
            q8_0_file_backing: None,
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
            q8_0_file_backing: None,
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

    pub fn with_q8_0_file_backing(mut self, backing: Q8_0FileBacking) -> Self {
        self.q8_0_file_backing = Some(backing);
        self
    }

    pub fn q8_0_file_backed_linear(
        name: impl Into<String>,
        shape: TensorShape,
        backing: Q8_0FileBacking,
    ) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(GgufTensorType::Q8_0),
            q8_0_blocks: None,
            q8_0_file_backing: Some(backing),
            data: Vec::new(),
        }
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
        if let Some(backing) = self.q8_0_file_backing.as_ref() {
            return self.embedding_lookup_q8_0_file_backed(token_ids, name, vocab, width, backing);
        }
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

    fn embedding_lookup_q8_0_file_backed(
        &self,
        token_ids: &[u32],
        name: impl Into<String>,
        vocab: usize,
        width: usize,
        backing: &Q8_0FileBacking,
    ) -> Result<Self> {
        const Q8_0_BLOCK_VALUES: usize = 32;
        const Q8_0_BLOCK_BYTES: usize = 34;
        if self.source_type != Some(GgufTensorType::Q8_0) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "file-backed embedding {} must come from Q8_0 storage",
                self.name
            )));
        }
        if !width.is_multiple_of(Q8_0_BLOCK_VALUES) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "file-backed q8_0 embedding width {width} is not divisible by {Q8_0_BLOCK_VALUES}"
            )));
        }
        let blocks_per_row = width / Q8_0_BLOCK_VALUES;
        let expected_blocks = vocab.checked_mul(blocks_per_row).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "file-backed q8_0 embedding block count overflow".to_string(),
            )
        })?;
        if backing.num_blocks != expected_blocks {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "file-backed q8_0 embedding block count {} does not match expected {expected_blocks}",
                backing.num_blocks
            )));
        }
        let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;
        let mut row = vec![0_u8; row_bytes];
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
            let offset = backing.absolute_offset + (token_idx * row_bytes) as u64;
            backing.read_exact_at_cached(&mut row, offset)?;
            for block in row.chunks_exact(Q8_0_BLOCK_BYTES) {
                let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
                out.extend(block[2..].iter().map(|q| scale * f32::from(*q as i8)));
            }
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

pub(crate) fn dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
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

static Q8_0_FILE_READ_CALLS: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_READ_BYTES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_HIT_BYTES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_MISS_BYTES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_INSERTS: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_INSERT_BYTES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_EVICTIONS: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_EVICTED_BYTES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_MERGES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_MERGED_BYTES: AtomicU64 = AtomicU64::new(0);
static Q8_FILE_CACHE: OnceLock<Mutex<Q8FileCache>> = OnceLock::new();

#[derive(Debug, Default, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct Q8_0FileReadStats {
    pub read_calls: u64,
    pub read_bytes: u64,
    pub cache_hits: u64,
    pub cache_hit_bytes: u64,
    pub cache_misses: u64,
    pub cache_miss_bytes: u64,
    pub cache_inserts: u64,
    pub cache_insert_bytes: u64,
    pub cache_evictions: u64,
    pub cache_evicted_bytes: u64,
    pub cache_merges: u64,
    pub cache_merged_bytes: u64,
    pub cache_entries: u64,
    pub cache_bytes: u64,
    pub cache_capacity_bytes: u64,
}

impl Q8_0FileReadStats {
    pub fn saturating_delta_since(self, start: Self) -> Self {
        Self {
            read_calls: self.read_calls.saturating_sub(start.read_calls),
            read_bytes: self.read_bytes.saturating_sub(start.read_bytes),
            cache_hits: self.cache_hits.saturating_sub(start.cache_hits),
            cache_hit_bytes: self.cache_hit_bytes.saturating_sub(start.cache_hit_bytes),
            cache_misses: self.cache_misses.saturating_sub(start.cache_misses),
            cache_miss_bytes: self.cache_miss_bytes.saturating_sub(start.cache_miss_bytes),
            cache_inserts: self.cache_inserts.saturating_sub(start.cache_inserts),
            cache_insert_bytes: self
                .cache_insert_bytes
                .saturating_sub(start.cache_insert_bytes),
            cache_evictions: self.cache_evictions.saturating_sub(start.cache_evictions),
            cache_evicted_bytes: self
                .cache_evicted_bytes
                .saturating_sub(start.cache_evicted_bytes),
            cache_merges: self.cache_merges.saturating_sub(start.cache_merges),
            cache_merged_bytes: self
                .cache_merged_bytes
                .saturating_sub(start.cache_merged_bytes),
            cache_entries: self.cache_entries,
            cache_bytes: self.cache_bytes,
            cache_capacity_bytes: self.cache_capacity_bytes,
        }
    }
}

pub(crate) fn record_q8_0_file_read(bytes: usize) {
    Q8_0_FILE_READ_CALLS.fetch_add(1, Ordering::Relaxed);
    Q8_0_FILE_READ_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

pub fn q8_0_file_read_stats() -> Q8_0FileReadStats {
    let cache_capacity_bytes = q8_file_cache_capacity_bytes();
    let (cache_entries, cache_bytes) = q8_file_cache_snapshot(cache_capacity_bytes);
    Q8_0FileReadStats {
        read_calls: Q8_0_FILE_READ_CALLS.load(Ordering::Relaxed),
        read_bytes: Q8_0_FILE_READ_BYTES.load(Ordering::Relaxed),
        cache_hits: Q8_0_FILE_CACHE_HITS.load(Ordering::Relaxed),
        cache_hit_bytes: Q8_0_FILE_CACHE_HIT_BYTES.load(Ordering::Relaxed),
        cache_misses: Q8_0_FILE_CACHE_MISSES.load(Ordering::Relaxed),
        cache_miss_bytes: Q8_0_FILE_CACHE_MISS_BYTES.load(Ordering::Relaxed),
        cache_inserts: Q8_0_FILE_CACHE_INSERTS.load(Ordering::Relaxed),
        cache_insert_bytes: Q8_0_FILE_CACHE_INSERT_BYTES.load(Ordering::Relaxed),
        cache_evictions: Q8_0_FILE_CACHE_EVICTIONS.load(Ordering::Relaxed),
        cache_evicted_bytes: Q8_0_FILE_CACHE_EVICTED_BYTES.load(Ordering::Relaxed),
        cache_merges: Q8_0_FILE_CACHE_MERGES.load(Ordering::Relaxed),
        cache_merged_bytes: Q8_0_FILE_CACHE_MERGED_BYTES.load(Ordering::Relaxed),
        cache_entries,
        cache_bytes,
        cache_capacity_bytes: cache_capacity_bytes as u64,
    }
}

#[derive(Debug, Default)]
struct Q8FileCache {
    entries: Vec<Q8FileCacheEntry>,
    bytes: usize,
}

#[derive(Debug)]
struct Q8FileCacheEntry {
    path: PathBuf,
    offset: u64,
    bytes: Vec<u8>,
}

fn q8_file_cache_get(path: &Path, offset: u64, out: &mut [u8]) -> bool {
    let capacity = q8_file_cache_capacity_bytes();
    if capacity == 0 {
        return false;
    }
    let Some(cache) = Q8_FILE_CACHE.get() else {
        record_q8_file_cache_miss(out.len());
        return false;
    };
    let mut cache = cache.lock().expect("q8 file cache mutex poisoned");
    cache.apply_capacity(capacity);
    let Some(pos) = cache
        .entries
        .iter()
        .position(|entry| q8_file_cache_entry_covers(entry, path, offset, out.len()))
    else {
        record_q8_file_cache_miss(out.len());
        return false;
    };
    let entry = cache.entries.remove(pos);
    let start = (offset - entry.offset) as usize;
    out.copy_from_slice(&entry.bytes[start..start + out.len()]);
    cache.entries.push(entry);
    Q8_0_FILE_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
    Q8_0_FILE_CACHE_HIT_BYTES.fetch_add(out.len() as u64, Ordering::Relaxed);
    true
}

fn record_q8_file_cache_miss(bytes: usize) {
    Q8_0_FILE_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
    Q8_0_FILE_CACHE_MISS_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

fn q8_file_cache_entry_covers(
    entry: &Q8FileCacheEntry,
    path: &Path,
    offset: u64,
    len: usize,
) -> bool {
    let Some(request_end) = offset.checked_add(len as u64) else {
        return false;
    };
    let Some(entry_end) = entry.offset.checked_add(entry.bytes.len() as u64) else {
        return false;
    };
    entry.path == path && entry.offset <= offset && request_end <= entry_end
}

fn q8_file_cache_insert(path: PathBuf, offset: u64, bytes: &[u8]) {
    let capacity = q8_file_cache_capacity_bytes();
    if capacity == 0 || bytes.len() > capacity {
        return;
    }
    let cache = Q8_FILE_CACHE.get_or_init(|| Mutex::new(Q8FileCache::default()));
    let mut cache = cache.lock().expect("q8 file cache mutex poisoned");
    cache.apply_capacity(capacity);
    cache.insert(path, offset, bytes.to_vec(), capacity);
}

fn q8_file_cache_capacity_bytes() -> usize {
    env::var(Q8_FILE_CACHE_BYTES_ENV)
        .ok()
        .and_then(|value| parse_byte_count(&value))
        .unwrap_or(DEFAULT_Q8_FILE_CACHE_BYTES)
}

pub(crate) fn parse_byte_count_env(key: &str) -> Option<usize> {
    env::var(key)
        .ok()
        .and_then(|value| parse_byte_count(&value))
}

fn parse_byte_count(value: &str) -> Option<usize> {
    let normalized = value
        .trim()
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace() && *ch != '_')
        .collect::<String>()
        .to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }

    let digits_len = normalized
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()
        .unwrap_or(0);
    if digits_len == 0 {
        return None;
    }

    let base = normalized[..digits_len].parse::<usize>().ok()?;
    let multiplier = match &normalized[digits_len..] {
        "" | "b" => 1usize,
        "k" | "kb" | "kib" => 1024usize,
        "m" | "mb" | "mib" => 1024usize.checked_mul(1024)?,
        "g" | "gb" | "gib" => 1024usize.checked_mul(1024)?.checked_mul(1024)?,
        _ => return None,
    };
    base.checked_mul(multiplier)
}

fn q8_file_cache_snapshot(capacity: usize) -> (u64, u64) {
    let Some(cache) = Q8_FILE_CACHE.get() else {
        return (0, 0);
    };
    let mut cache = cache.lock().expect("q8 file cache mutex poisoned");
    cache.apply_capacity(capacity);
    (cache.entries.len() as u64, cache.bytes as u64)
}

fn q8_file_cache_try_merge_entries(
    left: &Q8FileCacheEntry,
    right: &Q8FileCacheEntry,
    capacity: usize,
) -> Option<Q8FileCacheEntry> {
    if left.path != right.path {
        return None;
    }
    let left_end = left.offset.checked_add(left.bytes.len() as u64)?;
    let right_end = right.offset.checked_add(right.bytes.len() as u64)?;
    if left_end < right.offset || right_end < left.offset {
        return None;
    }
    let merged_offset = left.offset.min(right.offset);
    let merged_end = left_end.max(right_end);
    let merged_len = usize::try_from(merged_end.checked_sub(merged_offset)?).ok()?;

    let mut merged_bytes = vec![0u8; merged_len];
    let left_start = usize::try_from(left.offset.checked_sub(merged_offset)?).ok()?;
    merged_bytes[left_start..left_start + left.bytes.len()].copy_from_slice(&left.bytes);
    let right_start = usize::try_from(right.offset.checked_sub(merged_offset)?).ok()?;
    // Let the newest read win for overlapping bytes. The cache is only populated
    // from immutable GGUF payload reads, so equal bytes are expected; this keeps
    // the behavior deterministic for tests and any future synthetic cache probes.
    merged_bytes[right_start..right_start + right.bytes.len()].copy_from_slice(&right.bytes);

    let merged = Q8FileCacheEntry {
        path: left.path.clone(),
        offset: merged_offset,
        bytes: merged_bytes,
    };
    Some(q8_file_cache_trim_merged_entry_to_capacity(
        merged,
        right.offset,
        right.bytes.len(),
        capacity,
    ))
}

fn q8_file_cache_trim_merged_entry_to_capacity(
    mut entry: Q8FileCacheEntry,
    newest_offset: u64,
    newest_len: usize,
    capacity: usize,
) -> Q8FileCacheEntry {
    if entry.bytes.len() <= capacity {
        return entry;
    }

    debug_assert!(newest_len <= capacity);
    let entry_end = entry.offset + entry.bytes.len() as u64;
    let newest_end = newest_offset + newest_len as u64;
    debug_assert!(entry.offset <= newest_offset);
    debug_assert!(newest_end <= entry_end);

    // Keep a contiguous cache window that retains the newest read. This matters for
    // sequential Q8 tensor streams where adjacent 32 MiB chunks can coalesce up to
    // the cache cap: when the next chunk arrives, dropping the whole old coalesced
    // entry would collapse a 320 MiB tail cache down to one chunk. Trimming preserves
    // the most recent contiguous window instead, which is the part most likely to be
    // reused by the next long-prefill chunk.
    let capacity_u64 = capacity as u64;
    let max_window_start = entry_end - capacity_u64;
    let lower_start = entry.offset.max(newest_end.saturating_sub(capacity_u64));
    let upper_start = newest_offset.min(max_window_start);
    let window_start = if lower_start <= upper_start {
        upper_start
    } else {
        lower_start.clamp(entry.offset, max_window_start)
    };
    let trim_start = (window_start - entry.offset) as usize;
    let trim_end = trim_start + capacity;
    entry.bytes = entry.bytes[trim_start..trim_end].to_vec();
    entry.offset = window_start;
    entry
}

impl Q8FileCache {
    fn apply_capacity(&mut self, capacity: usize) {
        if capacity == 0 {
            self.entries.clear();
            self.bytes = 0;
            return;
        }
        while self.bytes > capacity {
            self.evict_oldest();
        }
    }

    fn insert(&mut self, path: PathBuf, offset: u64, bytes: Vec<u8>, capacity: usize) {
        if let Some(pos) = self
            .entries
            .iter()
            .position(|entry| q8_file_cache_entry_covers(entry, &path, offset, bytes.len()))
        {
            let start = (offset - self.entries[pos].offset) as usize;
            if self.entries[pos].bytes[start..start + bytes.len()] == bytes {
                let entry = self.entries.remove(pos);
                self.entries.push(entry);
                return;
            }
        }

        let mut entry = Q8FileCacheEntry {
            path,
            offset,
            bytes,
        };
        let mut pos = 0usize;
        while pos < self.entries.len() {
            if let Some(merged) =
                q8_file_cache_try_merge_entries(&self.entries[pos], &entry, capacity)
            {
                let old = self.entries.remove(pos);
                self.bytes = self.bytes.saturating_sub(old.bytes.len());
                Q8_0_FILE_CACHE_MERGES.fetch_add(1, Ordering::Relaxed);
                Q8_0_FILE_CACHE_MERGED_BYTES
                    .fetch_add(merged.bytes.len() as u64, Ordering::Relaxed);
                entry = merged;
                pos = 0;
            } else {
                pos += 1;
            }
        }
        self.bytes = self.bytes.saturating_add(entry.bytes.len());
        Q8_0_FILE_CACHE_INSERTS.fetch_add(1, Ordering::Relaxed);
        Q8_0_FILE_CACHE_INSERT_BYTES.fetch_add(entry.bytes.len() as u64, Ordering::Relaxed);
        self.entries.push(entry);
        while self.bytes > capacity {
            self.evict_oldest();
        }
    }

    fn evict_oldest(&mut self) {
        if self.entries.is_empty() {
            self.bytes = 0;
            return;
        }
        let entry = self.entries.remove(0);
        self.bytes = self.bytes.saturating_sub(entry.bytes.len());
        Q8_0_FILE_CACHE_EVICTIONS.fetch_add(1, Ordering::Relaxed);
        Q8_0_FILE_CACHE_EVICTED_BYTES.fetch_add(entry.bytes.len() as u64, Ordering::Relaxed);
    }
}

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

    pub fn load_q8_0_file_backed_linear(&self, name: &str) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        if desc.tensor_type != GgufTensorType::Q8_0 {
            return self.load_cpu_f32(name);
        }
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        if shape.dims.len() != 2 {
            return self.load_cpu_f32(name);
        }
        let expected_elements = shape.element_count()?;
        if expected_elements % 32 != 0 {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {name} Q8_0 element count {expected_elements} is not block aligned"
            )));
        }
        Ok(CpuTensor::q8_0_file_backed_linear(
            name,
            shape,
            Q8_0FileBacking::new(
                self.path.clone(),
                desc.absolute_offset,
                expected_elements / 32,
            ),
        ))
    }

    pub fn load_cpu_f32(&self, name: &str) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        let bytes = self.tensor_bytes(name)?;
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        let expected_elements = shape.element_count()?;
        let retain_q8_0_blocks = matches!(
            env::var(RETAIN_Q8_BLOCKS_ENV).as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        );
        let mut q8_0_blocks = None;
        let mut q8_0_file_backing = None;
        let data = match desc.tensor_type {
            GgufTensorType::F32 => decode_f32_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::F16 => decode_f16_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::BF16 => decode_bf16_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q8_0 => {
                let decoded = decode_q8_0_tensor(name, &bytes, expected_elements)?;
                if retain_q8_0_blocks {
                    q8_0_blocks = Some(decode_q8_0_blocks(name, &bytes, expected_elements)?);
                } else {
                    q8_0_file_backing = Some(Q8_0FileBacking::new(
                        self.path.clone(),
                        desc.absolute_offset,
                        expected_elements / 32,
                    ));
                }
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
            q8_0_file_backing,
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
    use super::{
        f16_bits_to_f32, parse_byte_count, q8_0_file_read_stats, q8_file_cache_get,
        q8_file_cache_insert, CpuTensor,
    };
    use crate::test_support::env_lock;

    #[test]
    fn q8_file_cache_disabled_path_does_not_store_or_hit() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES", "0");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-disabled-{}",
            std::process::id()
        ));

        let start = q8_0_file_read_stats();
        q8_file_cache_insert(path.clone(), 10, b"abcdefgh");
        let mut out = [0_u8; 8];
        assert!(!q8_file_cache_get(&path, 10, &mut out));
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_hit_bytes, 0);
        assert_eq!(stats.cache_entries, 0);
        assert_eq!(stats.cache_bytes, 0);
        assert_eq!(stats.cache_capacity_bytes, 0);
        std::env::remove_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_byte_count_env_parser_accepts_binary_suffixes() {
        assert_eq!(parse_byte_count("1024"), Some(1024));
        assert_eq!(parse_byte_count("1 KiB"), Some(1024));
        assert_eq!(parse_byte_count("2_mib"), Some(2 * 1024 * 1024));
        assert_eq!(parse_byte_count("3GB"), Some(3 * 1024 * 1024 * 1024));
        assert_eq!(parse_byte_count(""), None);
        assert_eq!(parse_byte_count("1.5MiB"), None);
        assert_eq!(parse_byte_count("many"), None);
    }

    #[test]
    fn q8_file_cache_serves_matching_chunks_and_evicts_to_capacity() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES", "8");
        let first_path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-first-{}",
            std::process::id()
        ));
        let second_path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-second-{}",
            std::process::id()
        ));
        q8_file_cache_insert(first_path.clone(), 10, b"abcdefgh");
        let mut out = [0_u8; 8];
        let start = q8_0_file_read_stats();
        assert!(q8_file_cache_get(&first_path, 10, &mut out));
        assert_eq!(&out, b"abcdefgh");
        let after_first = q8_0_file_read_stats().saturating_delta_since(start);
        assert_eq!(after_first.cache_hits, 1);
        assert_eq!(after_first.cache_hit_bytes, 8);
        assert_eq!(after_first.cache_entries, 1);
        assert_eq!(after_first.cache_bytes, 8);
        assert_eq!(after_first.cache_capacity_bytes, 8);

        q8_file_cache_insert(second_path.clone(), 20, b"ijklmnop");
        let mut evicted = [0_u8; 8];
        assert!(!q8_file_cache_get(&first_path, 10, &mut evicted));
        assert!(q8_file_cache_get(&second_path, 20, &mut evicted));
        assert_eq!(&evicted, b"ijklmnop");
        let after_second = q8_0_file_read_stats().saturating_delta_since(start);
        assert_eq!(after_second.cache_hits, 2);
        assert_eq!(after_second.cache_hit_bytes, 16);
        assert_eq!(after_second.cache_entries, 1);
        assert_eq!(after_second.cache_bytes, 8);
        std::env::remove_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_serves_subranges_from_retained_chunks() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-subrange-{}",
            std::process::id()
        ));
        q8_file_cache_insert(path.clone(), 100, b"abcdefghijklmnop");

        let start = q8_0_file_read_stats();
        let mut out = [0_u8; 4];
        assert!(q8_file_cache_get(&path, 104, &mut out));
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(&out, b"efgh");
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_hit_bytes, 4);
        std::env::remove_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_coalesces_adjacent_chunks_for_cross_boundary_reuse() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-adjacent-{}",
            std::process::id()
        ));
        q8_file_cache_insert(path.clone(), 100, b"abcdefgh");
        q8_file_cache_insert(path.clone(), 108, b"ijklmnop");

        let start = q8_0_file_read_stats();
        let mut out = [0_u8; 8];
        assert!(q8_file_cache_get(&path, 104, &mut out));
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(&out, b"efghijkl");
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_hit_bytes, 8);
        assert_eq!(stats.cache_entries, 1);
        assert_eq!(stats.cache_bytes, 16);
        std::env::remove_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_reports_miss_insert_merge_and_eviction_stats() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES", "0");
        let _ = q8_0_file_read_stats();
        std::env::set_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-stats-{}",
            std::process::id()
        ));
        let other_path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-stats-other-{}",
            std::process::id()
        ));

        let start = q8_0_file_read_stats();
        let mut miss = [0_u8; 4];
        assert!(!q8_file_cache_get(&path, 100, &mut miss));
        q8_file_cache_insert(path.clone(), 100, b"abcdefgh");
        q8_file_cache_insert(path.clone(), 108, b"ijklmnop");
        let mut hit = [0_u8; 8];
        assert!(q8_file_cache_get(&path, 104, &mut hit));
        q8_file_cache_insert(other_path, 200, b"qrstuvwx");
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(&hit, b"efghijkl");
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_miss_bytes, 4);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_hit_bytes, 8);
        assert_eq!(stats.cache_inserts, 3);
        assert_eq!(stats.cache_insert_bytes, 32);
        assert_eq!(stats.cache_merges, 1);
        assert_eq!(stats.cache_merged_bytes, 16);
        assert_eq!(stats.cache_evictions, 1);
        assert_eq!(stats.cache_evicted_bytes, 16);
        assert_eq!(stats.cache_entries, 1);
        assert_eq!(stats.cache_bytes, 8);
        assert_eq!(stats.cache_capacity_bytes, 16);
        std::env::remove_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_trims_coalesced_stream_to_newest_capacity_window() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-trim-window-{}",
            std::process::id()
        ));
        q8_file_cache_insert(path.clone(), 100, b"abcdefgh");
        q8_file_cache_insert(path.clone(), 108, b"ijklmnop");
        q8_file_cache_insert(path.clone(), 116, b"qrstuvwx");

        let start = q8_0_file_read_stats();
        let mut evicted = [0_u8; 8];
        let mut retained = [0_u8; 16];
        assert!(!q8_file_cache_get(&path, 100, &mut evicted));
        assert!(q8_file_cache_get(&path, 108, &mut retained));
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(&retained, b"ijklmnopqrstuvwx");
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_hit_bytes, 16);
        assert_eq!(stats.cache_entries, 1);
        assert_eq!(stats.cache_bytes, 16);
        std::env::remove_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_coalesces_overlapping_chunks_with_newest_bytes() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES", "12");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-overlap-{}",
            std::process::id()
        ));
        q8_file_cache_insert(path.clone(), 100, b"abcdefgh");
        q8_file_cache_insert(path.clone(), 104, b"WXYZmnop");

        let start = q8_0_file_read_stats();
        let mut out = [0_u8; 10];
        assert!(q8_file_cache_get(&path, 102, &mut out));
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(&out, b"cdWXYZmnop");
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_hit_bytes, 10);
        assert_eq!(stats.cache_entries, 1);
        assert_eq!(stats.cache_bytes, 12);
        std::env::remove_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_skips_reinserting_identical_fully_covered_subranges() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-covered-{}",
            std::process::id()
        ));
        q8_file_cache_insert(path.clone(), 100, b"abcdefghijklmnop");
        q8_file_cache_insert(path.clone(), 104, b"efgh");

        let start = q8_0_file_read_stats();
        let mut out = [0_u8; 16];
        assert!(q8_file_cache_get(&path, 100, &mut out));
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(&out, b"abcdefghijklmnop");
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_hit_bytes, 16);
        assert_eq!(stats.cache_entries, 1);
        assert_eq!(stats.cache_bytes, 16);
        std::env::remove_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_keeps_newest_bytes_for_conflicting_covered_subranges() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-covered-conflict-{}",
            std::process::id()
        ));
        q8_file_cache_insert(path.clone(), 100, b"abcdefghijklmnop");
        q8_file_cache_insert(path.clone(), 104, b"WXYZ");

        let start = q8_0_file_read_stats();
        let mut out = [0_u8; 16];
        assert!(q8_file_cache_get(&path, 100, &mut out));
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(&out, b"abcdWXYZijklmnop");
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_hit_bytes, 16);
        assert_eq!(stats.cache_entries, 1);
        assert_eq!(stats.cache_bytes, 16);
        std::env::remove_var("BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES");
    }

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
