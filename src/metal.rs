#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
};

#[cfg(target_os = "macos")]
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDeviceInfo {
    pub available: bool,
    pub device_name: Option<String>,
    pub low_power: Option<bool>,
    pub headless: Option<bool>,
    pub removable: Option<bool>,
    pub has_unified_memory: Option<bool>,
    pub registry_id: Option<u64>,
    pub max_threads_per_threadgroup: Option<(u64, u64, u64)>,
    pub note: Option<String>,
}

#[cfg(target_os = "macos")]
struct MetalLinearKernel {
    device: Device,
    queue: CommandQueue,
    descriptor_pipeline: ComputePipelineState,
    transposed_pipeline: ComputePipelineState,
    q8_0_encoded_pipeline: ComputePipelineState,
    q8_0_block_pipeline: ComputePipelineState,
}

#[cfg(target_os = "macos")]
struct MetalLinearCache {
    input_buffer: Option<Buffer>,
    input_capacity_bytes: usize,
    output_buffer: Option<Buffer>,
    output_capacity_bytes: usize,
    scalar_buffer: Option<Buffer>,
    scalar_capacity_bytes: usize,
    q8_input_scales_buffer: Option<Buffer>,
    q8_input_scales_capacity_bytes: usize,
    q8_input_quants_buffer: Option<Buffer>,
    q8_input_quants_capacity_bytes: usize,
    q8_encoded_rows_buffer: Option<Buffer>,
    q8_encoded_rows_capacity_bytes: usize,
    q8_weight_scales_buffer: Option<Buffer>,
    q8_weight_scales_capacity_bytes: usize,
    weight_buffers: HashMap<(usize, usize), Buffer>,
}

#[cfg(target_os = "macos")]
impl MetalLinearCache {
    fn new() -> Self {
        Self {
            input_buffer: None,
            input_capacity_bytes: 0,
            output_buffer: None,
            output_capacity_bytes: 0,
            scalar_buffer: None,
            scalar_capacity_bytes: 0,
            q8_input_scales_buffer: None,
            q8_input_scales_capacity_bytes: 0,
            q8_input_quants_buffer: None,
            q8_input_quants_capacity_bytes: 0,
            q8_encoded_rows_buffer: None,
            q8_encoded_rows_capacity_bytes: 0,
            q8_weight_scales_buffer: None,
            q8_weight_scales_capacity_bytes: 0,
            weight_buffers: HashMap::new(),
        }
    }

    fn shared_buffer(
        device: &Device,
        slot: &mut Option<Buffer>,
        capacity: &mut usize,
        needed: usize,
    ) -> Buffer {
        if slot.is_none() || *capacity < needed {
            *slot = Some(device.new_buffer(needed as u64, MTLResourceOptions::StorageModeShared));
            *capacity = needed;
        }
        slot.as_ref().expect("buffer just initialized").to_owned()
    }

    fn input_buffer(&mut self, device: &Device, needed: usize) -> Buffer {
        Self::shared_buffer(
            device,
            &mut self.input_buffer,
            &mut self.input_capacity_bytes,
            needed,
        )
    }

    fn output_buffer(&mut self, device: &Device, needed: usize) -> Buffer {
        Self::shared_buffer(
            device,
            &mut self.output_buffer,
            &mut self.output_capacity_bytes,
            needed,
        )
    }

    fn scalar_buffer(&mut self, device: &Device, needed: usize) -> Buffer {
        Self::shared_buffer(
            device,
            &mut self.scalar_buffer,
            &mut self.scalar_capacity_bytes,
            needed,
        )
    }

    fn q8_input_scales_buffer(&mut self, device: &Device, needed: usize) -> Buffer {
        Self::shared_buffer(
            device,
            &mut self.q8_input_scales_buffer,
            &mut self.q8_input_scales_capacity_bytes,
            needed,
        )
    }

    fn q8_input_quants_buffer(&mut self, device: &Device, needed: usize) -> Buffer {
        Self::shared_buffer(
            device,
            &mut self.q8_input_quants_buffer,
            &mut self.q8_input_quants_capacity_bytes,
            needed,
        )
    }

    fn q8_encoded_rows_buffer(&mut self, device: &Device, needed: usize) -> Buffer {
        Self::shared_buffer(
            device,
            &mut self.q8_encoded_rows_buffer,
            &mut self.q8_encoded_rows_capacity_bytes,
            needed,
        )
    }

    fn q8_weight_scales_buffer(&mut self, device: &Device, needed: usize) -> Buffer {
        Self::shared_buffer(
            device,
            &mut self.q8_weight_scales_buffer,
            &mut self.q8_weight_scales_capacity_bytes,
            needed,
        )
    }

    fn weight_buffer(&mut self, device: &Device, weights: &[f32]) -> Buffer {
        let key = (weights.as_ptr() as usize, weights.len());
        if let Some(buffer) = self.weight_buffers.get(&key) {
            return buffer.to_owned();
        }
        let buffer = device.new_buffer(
            std::mem::size_of_val(weights) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        write_buffer_f32(&buffer, weights);
        self.weight_buffers.insert(key, buffer.to_owned());
        buffer
    }
}

#[cfg(target_os = "macos")]
static METAL_LINEAR_KERNEL: OnceLock<Option<MetalLinearKernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_LINEAR_CACHE: OnceLock<Mutex<MetalLinearCache>> = OnceLock::new();

#[cfg(target_os = "macos")]
const LINEAR_ROW_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void linear_row_f32(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= cols) return;
    float sum = 0.0;
    for (uint inner = 0; inner < rows; ++inner) {
        sum += input[inner] * weights[inner * cols + gid];
    }
    output[gid] += sum;
}

kernel void linear_row_transposed_f32(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= cols) return;
    float sum = 0.0;
    uint base = gid * rows;
    for (uint inner = 0; inner < rows; ++inner) {
        sum += input[inner] * weights[base + inner];
    }
    output[gid] = sum;
}

kernel void q8_0_encoded_linear_row(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device const char* encoded_rows [[buffer(2)]],
    device const float* weight_scales [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& blocks_per_row [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    constexpr uint block_values = 32;
    constexpr uint encoded_block_bytes = 34;
    float sum = 0.0;
    uint row_base = gid * blocks_per_row * encoded_block_bytes;
    uint scale_base = gid * blocks_per_row;
    for (uint block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        int int_sum = 0;
        uint encoded_base = row_base + block_idx * encoded_block_bytes + 2;
        uint input_base = block_idx * block_values;
        for (uint lane = 0; lane < block_values; ++lane) {
            int_sum += int(encoded_rows[encoded_base + lane]) * int(input_quants[input_base + lane]);
        }
        sum += float(int_sum) * weight_scales[scale_base + block_idx] * input_scales[block_idx];
    }
    output[gid] = sum;
}

kernel void q8_0_block_linear_row(
    device const float* input_scales [[buffer(0)]],
    device const char* input_quants [[buffer(1)]],
    device const char* weight_blocks [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    constexpr uint block_values = 32;
    constexpr uint q8_block_bytes = 36;
    float sum = 0.0;
    uint row_base = gid * blocks_per_row * q8_block_bytes;
    for (uint block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        int int_sum = 0;
        uint block_base = row_base + block_idx * q8_block_bytes;
        device const float* weight_scale = reinterpret_cast<device const float*>(weight_blocks + block_base);
        uint weight_quant_base = block_base + 4;
        uint input_base = block_idx * block_values;
        for (uint lane = 0; lane < block_values; ++lane) {
            int_sum += int(weight_blocks[weight_quant_base + lane]) * int(input_quants[input_base + lane]);
        }
        sum += float(int_sum) * (*weight_scale) * input_scales[block_idx];
    }
    output[gid] = sum;
}
"#;

#[cfg(target_os = "macos")]
fn metal_linear_kernel() -> Option<&'static MetalLinearKernel> {
    METAL_LINEAR_KERNEL
        .get_or_init(|| {
            let device = Device::system_default()?;
            let options = CompileOptions::new();
            let library = device
                .new_library_with_source(LINEAR_ROW_SHADER, &options)
                .ok()?;
            let descriptor_function = library.get_function("linear_row_f32", None).ok()?;
            let descriptor_pipeline = device
                .new_compute_pipeline_state_with_function(&descriptor_function)
                .ok()?;
            let transposed_function = library
                .get_function("linear_row_transposed_f32", None)
                .ok()?;
            let transposed_pipeline = device
                .new_compute_pipeline_state_with_function(&transposed_function)
                .ok()?;
            let q8_0_encoded_function =
                library.get_function("q8_0_encoded_linear_row", None).ok()?;
            let q8_0_encoded_pipeline = device
                .new_compute_pipeline_state_with_function(&q8_0_encoded_function)
                .ok()?;
            let q8_0_block_function = library.get_function("q8_0_block_linear_row", None).ok()?;
            let q8_0_block_pipeline = device
                .new_compute_pipeline_state_with_function(&q8_0_block_function)
                .ok()?;
            let queue = device.new_command_queue();
            Some(MetalLinearKernel {
                device,
                queue,
                descriptor_pipeline,
                transposed_pipeline,
                q8_0_encoded_pipeline,
                q8_0_block_pipeline,
            })
        })
        .as_ref()
}

#[cfg(target_os = "macos")]
fn metal_linear_cache() -> &'static Mutex<MetalLinearCache> {
    METAL_LINEAR_CACHE.get_or_init(|| Mutex::new(MetalLinearCache::new()))
}

#[cfg(target_os = "macos")]
fn write_buffer_f32(buffer: &Buffer, values: &[f32]) {
    write_buffer_bytes(buffer, values);
}

#[cfg(target_os = "macos")]
fn write_buffer_u8(buffer: &Buffer, values: &[u8]) {
    write_buffer_bytes(buffer, values);
}

#[cfg(target_os = "macos")]
fn write_buffer_i8(buffer: &Buffer, values: &[i8]) {
    write_buffer_bytes(buffer, values);
}

#[cfg(target_os = "macos")]
fn write_buffer_bytes<T>(buffer: &Buffer, values: &[T]) {
    let len = std::mem::size_of_val(values);
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().cast::<u8>(),
            len,
        );
    }
}

#[cfg(target_os = "macos")]
fn read_buffer_f32(buffer: &Buffer, out: &mut [f32]) {
    let len = std::mem::size_of_val(out);
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().cast::<u8>(),
            out.as_mut_ptr().cast::<u8>(),
            len,
        );
    }
}

#[cfg(target_os = "macos")]
pub fn try_linear_row_f32(
    input_row: &[f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    output: &mut [f32],
) -> bool {
    try_linear_row_impl(input_row, weights, rows, cols, output, false)
}

#[cfg(target_os = "macos")]
pub fn try_linear_row_transposed_f32(
    input_row: &[f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    output: &mut [f32],
) -> bool {
    try_linear_row_impl(input_row, weights, rows, cols, output, true)
}

#[cfg(target_os = "macos")]
fn try_linear_row_impl(
    input_row: &[f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    output: &mut [f32],
    transposed: bool,
) -> bool {
    if rows != input_row.len() || cols != output.len() || weights.len() != rows.saturating_mul(cols)
    {
        return false;
    }
    let Some(kernel) = metal_linear_kernel() else {
        return false;
    };
    let mut cache = metal_linear_cache()
        .lock()
        .expect("metal linear cache poisoned");
    let input_buffer = cache.input_buffer(&kernel.device, std::mem::size_of_val(input_row));
    let weight_buffer = cache.weight_buffer(&kernel.device, weights);
    let output_buffer = cache.output_buffer(&kernel.device, std::mem::size_of_val(output));
    let scalar_buffer = cache.scalar_buffer(&kernel.device, 2 * std::mem::size_of::<u32>());
    write_buffer_f32(&input_buffer, input_row);
    write_buffer_f32(&output_buffer, output);
    unsafe {
        let scalars = scalar_buffer.contents() as *mut u32;
        *scalars = rows as u32;
        *scalars.add(1) = cols as u32;
    }

    let command_buffer = kernel.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(if transposed {
        &kernel.transposed_pipeline
    } else {
        &kernel.descriptor_pipeline
    });
    encoder.set_buffer(0, Some(&input_buffer), 0);
    encoder.set_buffer(1, Some(&weight_buffer), 0);
    encoder.set_buffer(2, Some(&output_buffer), 0);
    encoder.set_buffer(3, Some(&scalar_buffer), 0);
    encoder.set_buffer(4, Some(&scalar_buffer), std::mem::size_of::<u32>() as u64);

    let pipeline = if transposed {
        &kernel.transposed_pipeline
    } else {
        &kernel.descriptor_pipeline
    };
    let width = pipeline.thread_execution_width().max(1);
    let threads_per_group = metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroups = metal::MTLSize {
        width: (cols as u64).div_ceil(width),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    drop(cache);
    read_buffer_f32(&output_buffer, output);
    true
}

#[cfg(target_os = "macos")]
pub fn try_q8_0_encoded_linear_row(
    input_scales: &[f32],
    input_quants: &[i8],
    encoded_rows: &[u8],
    weight_scales: &[f32],
    rows: usize,
    blocks_per_row: usize,
    output: &mut [f32],
) -> bool {
    const Q8_0_BLOCK_VALUES: usize = 32;
    const Q8_0_ENCODED_BLOCK_BYTES: usize = 34;
    if rows == 0 || blocks_per_row == 0 || output.len() != rows {
        return false;
    }
    if input_scales.len() != blocks_per_row
        || input_quants.len() != blocks_per_row.saturating_mul(Q8_0_BLOCK_VALUES)
        || encoded_rows.len()
            != rows
                .saturating_mul(blocks_per_row)
                .saturating_mul(Q8_0_ENCODED_BLOCK_BYTES)
        || weight_scales.len() != rows.saturating_mul(blocks_per_row)
    {
        return false;
    }
    let Some(kernel) = metal_linear_kernel() else {
        return false;
    };
    let mut cache = metal_linear_cache()
        .lock()
        .expect("metal linear cache poisoned");
    let input_scales_buffer =
        cache.q8_input_scales_buffer(&kernel.device, std::mem::size_of_val(input_scales));
    let input_quants_buffer =
        cache.q8_input_quants_buffer(&kernel.device, std::mem::size_of_val(input_quants));
    let encoded_rows_buffer =
        cache.q8_encoded_rows_buffer(&kernel.device, std::mem::size_of_val(encoded_rows));
    let weight_scales_buffer =
        cache.q8_weight_scales_buffer(&kernel.device, std::mem::size_of_val(weight_scales));
    let output_buffer = cache.output_buffer(&kernel.device, std::mem::size_of_val(output));
    let scalar_buffer = cache.scalar_buffer(&kernel.device, 2 * std::mem::size_of::<u32>());
    write_buffer_f32(&input_scales_buffer, input_scales);
    write_buffer_i8(&input_quants_buffer, input_quants);
    write_buffer_u8(&encoded_rows_buffer, encoded_rows);
    write_buffer_f32(&weight_scales_buffer, weight_scales);
    unsafe {
        let scalars = scalar_buffer.contents() as *mut u32;
        *scalars = blocks_per_row as u32;
        *scalars.add(1) = rows as u32;
    }

    let command_buffer = kernel.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&kernel.q8_0_encoded_pipeline);
    encoder.set_buffer(0, Some(&input_scales_buffer), 0);
    encoder.set_buffer(1, Some(&input_quants_buffer), 0);
    encoder.set_buffer(2, Some(&encoded_rows_buffer), 0);
    encoder.set_buffer(3, Some(&weight_scales_buffer), 0);
    encoder.set_buffer(4, Some(&output_buffer), 0);
    encoder.set_buffer(5, Some(&scalar_buffer), 0);
    encoder.set_buffer(6, Some(&scalar_buffer), std::mem::size_of::<u32>() as u64);

    let width = kernel.q8_0_encoded_pipeline.thread_execution_width().max(1);
    let threads_per_group = metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroups = metal::MTLSize {
        width: (rows as u64).div_ceil(width),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    drop(cache);
    read_buffer_f32(&output_buffer, output);
    true
}

#[cfg(target_os = "macos")]
pub fn try_q8_0_block_linear_row(
    input_scales: &[f32],
    input_quants: &[i8],
    weight_blocks: &[u8],
    rows: usize,
    blocks_per_row: usize,
    output: &mut [f32],
) -> bool {
    const Q8_0_BLOCK_VALUES: usize = 32;
    const Q8_0_BLOCK_BYTES: usize = 36;
    if rows == 0 || blocks_per_row == 0 || output.len() != rows {
        return false;
    }
    if input_scales.len() != blocks_per_row
        || input_quants.len() != blocks_per_row.saturating_mul(Q8_0_BLOCK_VALUES)
        || weight_blocks.len()
            != rows
                .saturating_mul(blocks_per_row)
                .saturating_mul(Q8_0_BLOCK_BYTES)
    {
        return false;
    }
    let Some(kernel) = metal_linear_kernel() else {
        return false;
    };
    let mut cache = metal_linear_cache()
        .lock()
        .expect("metal linear cache poisoned");
    let input_scales_buffer =
        cache.q8_input_scales_buffer(&kernel.device, std::mem::size_of_val(input_scales));
    let input_quants_buffer =
        cache.q8_input_quants_buffer(&kernel.device, std::mem::size_of_val(input_quants));
    let weight_blocks_buffer = cache.q8_encoded_rows_buffer(&kernel.device, weight_blocks.len());
    let output_buffer = cache.output_buffer(&kernel.device, std::mem::size_of_val(output));
    let scalar_buffer = cache.scalar_buffer(&kernel.device, 2 * std::mem::size_of::<u32>());
    write_buffer_f32(&input_scales_buffer, input_scales);
    write_buffer_i8(&input_quants_buffer, input_quants);
    write_buffer_u8(&weight_blocks_buffer, weight_blocks);
    unsafe {
        let scalars = scalar_buffer.contents() as *mut u32;
        *scalars = blocks_per_row as u32;
        *scalars.add(1) = rows as u32;
    }

    let command_buffer = kernel.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&kernel.q8_0_block_pipeline);
    encoder.set_buffer(0, Some(&input_scales_buffer), 0);
    encoder.set_buffer(1, Some(&input_quants_buffer), 0);
    encoder.set_buffer(2, Some(&weight_blocks_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);
    encoder.set_buffer(4, Some(&scalar_buffer), 0);
    encoder.set_buffer(5, Some(&scalar_buffer), std::mem::size_of::<u32>() as u64);

    let width = kernel.q8_0_block_pipeline.thread_execution_width().max(1);
    let threads_per_group = metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroups = metal::MTLSize {
        width: (rows as u64).div_ceil(width),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    drop(cache);
    read_buffer_f32(&output_buffer, output);
    true
}

#[cfg(not(target_os = "macos"))]
pub fn try_linear_row_f32(
    _input_row: &[f32],
    _weights: &[f32],
    _rows: usize,
    _cols: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_os = "macos"))]
pub fn try_linear_row_transposed_f32(
    _input_row: &[f32],
    _weights: &[f32],
    _rows: usize,
    _cols: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_os = "macos"))]
pub fn try_q8_0_encoded_linear_row(
    _input_scales: &[f32],
    _input_quants: &[i8],
    _encoded_rows: &[u8],
    _weight_scales: &[f32],
    _rows: usize,
    _blocks_per_row: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_os = "macos"))]
pub fn try_q8_0_block_linear_row(
    _input_scales: &[f32],
    _input_quants: &[i8],
    _weight_blocks: &[u8],
    _rows: usize,
    _blocks_per_row: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(target_os = "macos")]
pub fn detect_metal_device() -> MetalDeviceInfo {
    match Device::system_default() {
        Some(device) => {
            let threadgroup = device.max_threads_per_threadgroup();
            MetalDeviceInfo {
                available: true,
                device_name: Some(device.name().to_string()),
                low_power: Some(device.is_low_power()),
                headless: Some(device.is_headless()),
                removable: Some(device.is_removable()),
                has_unified_memory: Some(device.has_unified_memory()),
                registry_id: Some(device.registry_id()),
                max_threads_per_threadgroup: Some((
                    threadgroup.width,
                    threadgroup.height,
                    threadgroup.depth,
                )),
                note: Some(
                    "Metal device detected. Camelid has an opt-in experimental dense linear-row kernel path on macOS; broader inference offload is still in progress.".to_string(),
                ),
            }
        }
        None => MetalDeviceInfo {
            available: false,
            device_name: None,
            low_power: None,
            headless: None,
            removable: None,
            has_unified_memory: None,
            registry_id: None,
            max_threads_per_threadgroup: None,
            note: Some("No Metal system device was reported by macOS.".to_string()),
        },
    }
}

#[cfg(not(target_os = "macos"))]
pub fn detect_metal_device() -> MetalDeviceInfo {
    MetalDeviceInfo {
        available: false,
        device_name: None,
        low_power: None,
        headless: None,
        removable: None,
        has_unified_memory: None,
        registry_id: None,
        max_threads_per_threadgroup: None,
        note: Some("Metal is only available on macOS builds.".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_linear_row_matches_cpu_for_small_dense_accumulation() {
        if !detect_metal_device().available {
            return;
        }

        let input = [2.0_f32, -1.0, 0.5];
        let weights = [
            1.0_f32, 2.0, -3.0, 4.0, // row 0
            -2.0, 0.5, 1.5, -1.0, // row 1
            0.25, -4.0, 2.0, 0.0, // row 2
        ];
        let mut output = [1.0_f32, -2.0, 0.5, 3.0];
        let mut expected = output;
        for col in 0..expected.len() {
            for row in 0..input.len() {
                expected[col] += input[row] * weights[row * expected.len() + col];
            }
        }

        assert!(try_linear_row_f32(
            &input,
            &weights,
            input.len(),
            output.len(),
            &mut output
        ));
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_linear_row_transposed_matches_cpu_for_small_dense_dot_rows() {
        if !detect_metal_device().available {
            return;
        }

        let input = [2.0_f32, -1.0, 0.5];
        let weights = [
            1.0_f32, 2.0, -3.0, 4.0, -2.0, 0.5, 1.5, -1.0, 0.25, -4.0, 2.0, 0.0,
        ];
        let mut output = [0.0_f32; 4];
        let mut expected = [0.0_f32; 4];
        for col in 0..expected.len() {
            for row in 0..input.len() {
                expected[col] += input[row] * weights[col * input.len() + row];
            }
        }

        assert!(try_linear_row_transposed_f32(
            &input,
            &weights,
            input.len(),
            expected.len(),
            &mut output
        ));
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_q8_0_encoded_linear_row_matches_cpu_for_small_rows() {
        if !detect_metal_device().available {
            return;
        }

        let input_scales = [0.25_f32];
        let input_quants = [
            1_i8, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ];
        let row0 = [
            -1_i8, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20,
            -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31, 32,
        ];
        let row1 = [
            2_i8, 1, -2, -1, 3, 2, -3, -2, 4, 3, -4, -3, 5, 4, -5, -4, 6, 5, -6, -5, 7, 6, -7, -6,
            8, 7, -8, -7, 9, 8, -9, -8,
        ];
        let mut encoded_rows = Vec::new();
        for row in [&row0, &row1] {
            encoded_rows.extend_from_slice(&[0, 0]);
            encoded_rows.extend(row.iter().map(|value| *value as u8));
        }
        let weight_scales = [0.5_f32, 0.125];
        let mut output = [0.0_f32; 2];
        let expected = [
            input_quants
                .iter()
                .zip(row0)
                .map(|(a, b)| i32::from(*a) * i32::from(b))
                .sum::<i32>() as f32
                * input_scales[0]
                * weight_scales[0],
            input_quants
                .iter()
                .zip(row1)
                .map(|(a, b)| i32::from(*a) * i32::from(b))
                .sum::<i32>() as f32
                * input_scales[0]
                * weight_scales[1],
        ];

        assert!(try_q8_0_encoded_linear_row(
            &input_scales,
            &input_quants,
            &encoded_rows,
            &weight_scales,
            2,
            1,
            &mut output,
        ));
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_q8_0_block_linear_row_matches_cpu_for_small_rows() {
        if !detect_metal_device().available {
            return;
        }

        let input_scales = [0.25_f32];
        let input_quants = [
            1_i8, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ];
        let row0 = [
            -1_i8, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20,
            -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31, 32,
        ];
        let row1 = [
            2_i8, 1, -2, -1, 3, 2, -3, -2, 4, 3, -4, -3, 5, 4, -5, -4, 6, 5, -6, -5, 7, 6, -7, -6,
            8, 7, -8, -7, 9, 8, -9, -8,
        ];
        let weight_scales = [0.5_f32, 0.125];
        let mut weight_blocks = Vec::new();
        for (scale, row) in weight_scales.iter().zip([&row0, &row1]) {
            weight_blocks.extend_from_slice(&scale.to_le_bytes());
            weight_blocks.extend(row.iter().map(|value| *value as u8));
        }
        let mut output = [0.0_f32; 2];
        let expected = [
            input_quants
                .iter()
                .zip(row0)
                .map(|(a, b)| i32::from(*a) * i32::from(b))
                .sum::<i32>() as f32
                * input_scales[0]
                * weight_scales[0],
            input_quants
                .iter()
                .zip(row1)
                .map(|(a, b)| i32::from(*a) * i32::from(b))
                .sum::<i32>() as f32
                * input_scales[0]
                * weight_scales[1],
        ];

        assert!(try_q8_0_block_linear_row(
            &input_scales,
            &input_quants,
            &weight_blocks,
            2,
            1,
            &mut output,
        ));
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn metal_linear_row_stub_returns_false() {
        let input = [1.0_f32, 2.0];
        let weights = [3.0_f32, 4.0, 5.0, 6.0];
        let mut output = [0.0_f32, 0.0];
        assert!(!try_linear_row_f32(&input, &weights, 2, 2, &mut output));
        assert_eq!(output, [0.0, 0.0]);
    }
}
