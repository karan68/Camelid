//! Prism/Bonsai Qwen3-VL vision-projector loader and image preprocessing.
//!
//! The published 27B rows pair the Qwen3.5 language model with a separate
//! `qwen3vl_merger` GGUF. Large matrices remain page-backed so Metal can wrap
//! them with `newBufferWithBytesNoCopy`; only biases, norms and the learned
//! position table are decoded to f32 on the host.

use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

#[cfg(target_os = "macos")]
use std::sync::Mutex;

use image::{imageops::FilterType, RgbImage};

use crate::error::{BackendError, Result};
use crate::gguf::{self, GgufFile, GgufTensorDescriptor, GgufTensorType};
use crate::wire_mmap::WirePages;

use super::dequant;

#[derive(Clone)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct VisionMat {
    pub pages: Arc<WirePages>,
    pub tensor_type: GgufTensorType,
    pub input: usize,
    pub output: usize,
}

impl VisionMat {
    #[cfg(target_os = "macos")]
    pub(crate) fn metal_weight(&self) -> Result<crate::metal::ResidentWeightBytes<'_>> {
        let format = match self.tensor_type {
            GgufTensorType::F32 => crate::metal::ResidentWeightFormat::DenseF32,
            GgufTensorType::F16 => crate::metal::ResidentWeightFormat::DenseF16,
            GgufTensorType::Q8_0 => crate::metal::ResidentWeightFormat::Q8_0,
            other => {
                return Err(BackendError::UnsupportedTensorType(format!(
                    "Qwen3-VL Metal weight uses unsupported type {other:?}"
                )))
            }
        };
        Ok(crate::metal::ResidentWeightBytes::WirePages {
            format,
            pages: &self.pages,
        })
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct PrismVisionLayer {
    pub ln1_weight: Vec<f32>,
    pub ln1_bias: Vec<f32>,
    pub qkv: VisionMat,
    pub qkv_bias: Vec<f32>,
    pub attn_output: VisionMat,
    pub attn_output_bias: Vec<f32>,
    pub ln2_weight: Vec<f32>,
    pub ln2_bias: Vec<f32>,
    pub ffn_up: VisionMat,
    pub ffn_up_bias: Vec<f32>,
    pub ffn_down: VisionMat,
    pub ffn_down_bias: Vec<f32>,
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct PrismVisionModel {
    pub image_size: usize,
    pub patch_size: usize,
    pub merge: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub heads: usize,
    pub eps: f32,
    pub projection: usize,
    pub patch_0: VisionMat,
    pub patch_1: VisionMat,
    pub patch_bias: Vec<f32>,
    /// Learned 48x48 table, position-major `[2304, hidden]`.
    pub position: Vec<f32>,
    pub layers: Vec<PrismVisionLayer>,
    pub post_weight: Vec<f32>,
    pub post_bias: Vec<f32>,
    pub merger_0: VisionMat,
    pub merger_0_bias: Vec<f32>,
    pub merger_2: VisionMat,
    pub merger_2_bias: Vec<f32>,
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct PrismVisionInput {
    /// Tile-major patches `[patch_count, 3 * patch_size * patch_size]`.
    pub patches: Vec<f32>,
    /// Learned position rows in the same tile-major order `[patch_count, hidden]`.
    pub position: Vec<f32>,
    pub patch_width: usize,
    pub patch_height: usize,
}

/// Image embeddings plus their merged 2-D token grid. The language decoder
/// needs both: embeddings feed the model while the grid drives Qwen3.5's
/// interleaved multimodal RoPE positions.
pub struct PrismVisionEmbedding {
    pub embeddings: Vec<Vec<f32>>,
    pub grid_width: usize,
    pub grid_height: usize,
}

/// Loaded Prism Qwen3-VL projector with its native Metal graph retained across
/// requests. The backing GGUF pages stay owned by `model` while the Metal
/// buffers refer to them through NoCopy allocations.
pub struct PrismVisionProjector {
    model: PrismVisionModel,
    #[cfg(target_os = "macos")]
    encoder: Mutex<crate::metal::PrismVisionMetalEncoder>,
}

impl PrismVisionProjector {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let model = PrismVisionModel::load(path)?;
        #[cfg(target_os = "macos")]
        {
            let encoder = model.metal_encoder()?;
            Ok(Self {
                model,
                encoder: Mutex::new(encoder),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = model;
            Err(BackendError::UnsupportedGguf(
                "Prism Qwen3-VL projector execution currently requires macOS Metal".into(),
            ))
        }
    }

    pub fn projection_dim(&self) -> usize {
        self.model.projection
    }

    pub fn encode_image(
        &self,
        path: impl AsRef<Path>,
        min_tokens: usize,
        max_tokens: usize,
    ) -> Result<PrismVisionEmbedding> {
        let input = self.model.preprocess(path, min_tokens, max_tokens)?;
        self.encode_preprocessed(input)
    }

    /// Decode an uploaded PNG/JPEG directly from memory and project it without
    /// writing user content to a temporary file.
    pub fn encode_image_bytes(
        &self,
        bytes: &[u8],
        min_tokens: usize,
        max_tokens: usize,
    ) -> Result<PrismVisionEmbedding> {
        let mut reader = image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|error| BackendError::InvalidTensorData(format!("detect image: {error}")))?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(16_384);
        limits.max_image_height = Some(16_384);
        limits.max_alloc = Some(256 * 1024 * 1024);
        reader.limits(limits);
        let image = reader
            .decode()
            .map_err(|error| BackendError::InvalidTensorData(format!("decode image: {error}")))?
            .to_rgb8();
        let input = self.model.preprocess_rgb(image, min_tokens, max_tokens)?;
        self.encode_preprocessed(input)
    }

    fn encode_preprocessed(&self, input: PrismVisionInput) -> Result<PrismVisionEmbedding> {
        #[cfg(target_os = "macos")]
        {
            let mut encoder = self.encoder.lock().map_err(|_| {
                BackendError::InvalidTensorData("Prism vision Metal mutex poisoned".into())
            })?;
            let embeddings = encoder
                .encode(
                    &input.patches,
                    &input.position,
                    input.patch_width,
                    input.patch_height,
                )
                .ok_or_else(|| {
                    BackendError::InvalidTensorData(
                        "Prism vision Metal graph refused the preprocessed image".into(),
                    )
                })?;
            let grid_width = input.patch_width / self.model.merge;
            let grid_height = input.patch_height / self.model.merge;
            if embeddings.len() != grid_width * grid_height {
                return Err(BackendError::InvalidTensorData(
                    "Prism vision encoder returned the wrong image-token count".into(),
                ));
            }
            Ok(PrismVisionEmbedding {
                embeddings,
                grid_width,
                grid_height,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = input;
            Err(BackendError::UnsupportedGguf(
                "Prism Qwen3-VL projector execution currently requires macOS Metal".into(),
            ))
        }
    }
}

#[cfg(test)]
impl PrismVisionInput {
    pub(crate) fn patch_count(&self) -> usize {
        self.patch_width * self.patch_height
    }

    pub(crate) fn output_tokens(&self, merge: usize) -> usize {
        self.patch_count() / (merge * merge)
    }
}

impl PrismVisionModel {
    pub(crate) fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let gguf = gguf::read_metadata(path)?;
        if gguf.architecture() != Some("clip")
            || gguf.metadata_string("clip.projector_type") != Some("qwen3vl_merger")
            || gguf.metadata_bool("clip.has_vision_encoder") != Some(true)
        {
            return Err(BackendError::InvalidModelMetadata(
                "vision GGUF must be clip/qwen3vl_merger with a vision encoder".into(),
            ));
        }
        let get_u32 = |key: &str| {
            gguf.metadata_u32(key).ok_or_else(|| {
                BackendError::InvalidModelMetadata(format!("vision GGUF missing {key}"))
            })
        };
        let image_size = get_u32("clip.vision.image_size")? as usize;
        let patch_size = get_u32("clip.vision.patch_size")? as usize;
        let merge = get_u32("clip.vision.spatial_merge_size")? as usize;
        let hidden = get_u32("clip.vision.embedding_length")? as usize;
        let ffn = get_u32("clip.vision.feed_forward_length")? as usize;
        let heads = get_u32("clip.vision.attention.head_count")? as usize;
        let blocks = get_u32("clip.vision.block_count")? as usize;
        let projection = get_u32("clip.vision.projection_dim")? as usize;
        let eps = gguf
            .metadata_f32("clip.vision.attention.layer_norm_epsilon")
            .ok_or_else(|| {
                BackendError::InvalidModelMetadata(
                    "vision GGUF missing attention LayerNorm epsilon".into(),
                )
            })?;
        if image_size == 0
            || patch_size == 0
            || merge != 2
            || hidden == 0
            || ffn == 0
            || heads == 0
            || !hidden.is_multiple_of(heads)
            || blocks == 0
            || projection == 0
        {
            return Err(BackendError::InvalidModelMetadata(
                "unsupported or degenerate qwen3vl_merger geometry".into(),
            ));
        }

        let mut file = File::open(path).map_err(|source| BackendError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let patch_0 = load_mat(&mut file, &gguf, "v.patch_embd.weight")?;
        let patch_1 = load_mat(&mut file, &gguf, "v.patch_embd.weight.1")?;
        let patch_bias = load_vec(&mut file, &gguf, "v.patch_embd.bias")?;
        let position = load_vec(&mut file, &gguf, "v.position_embd.weight")?;
        if patch_0.input != 3 * patch_size * patch_size
            || patch_0.output != hidden
            || patch_1.input != patch_0.input
            || patch_1.output != hidden
            || patch_bias.len() != hidden
            || position.len() != hidden * (image_size / patch_size).pow(2)
        {
            return Err(BackendError::InvalidTensorData(
                "qwen3vl patch/position tensor geometry mismatch".into(),
            ));
        }

        let mut layers = Vec::with_capacity(blocks);
        for layer in 0..blocks {
            let n = |suffix: &str| format!("v.blk.{layer}.{suffix}");
            layers.push(PrismVisionLayer {
                ln1_weight: load_vec(&mut file, &gguf, &n("ln1.weight"))?,
                ln1_bias: load_vec(&mut file, &gguf, &n("ln1.bias"))?,
                qkv: load_mat(&mut file, &gguf, &n("attn_qkv.weight"))?,
                qkv_bias: load_vec(&mut file, &gguf, &n("attn_qkv.bias"))?,
                attn_output: load_mat(&mut file, &gguf, &n("attn_out.weight"))?,
                attn_output_bias: load_vec(&mut file, &gguf, &n("attn_out.bias"))?,
                ln2_weight: load_vec(&mut file, &gguf, &n("ln2.weight"))?,
                ln2_bias: load_vec(&mut file, &gguf, &n("ln2.bias"))?,
                ffn_up: load_mat(&mut file, &gguf, &n("ffn_up.weight"))?,
                ffn_up_bias: load_vec(&mut file, &gguf, &n("ffn_up.bias"))?,
                ffn_down: load_mat(&mut file, &gguf, &n("ffn_down.weight"))?,
                ffn_down_bias: load_vec(&mut file, &gguf, &n("ffn_down.bias"))?,
            });
        }
        for (index, layer) in layers.iter().enumerate() {
            let vector_lengths_ok = layer.ln1_weight.len() == hidden
                && layer.ln1_bias.len() == hidden
                && layer.qkv_bias.len() == 3 * hidden
                && layer.attn_output_bias.len() == hidden
                && layer.ln2_weight.len() == hidden
                && layer.ln2_bias.len() == hidden
                && layer.ffn_up_bias.len() == ffn
                && layer.ffn_down_bias.len() == hidden;
            let matrix_shapes_ok = layer.qkv.input == hidden
                && layer.qkv.output == 3 * hidden
                && layer.attn_output.input == hidden
                && layer.attn_output.output == hidden
                && layer.ffn_up.input == hidden
                && layer.ffn_up.output == ffn
                && layer.ffn_down.input == ffn
                && layer.ffn_down.output == hidden;
            if !vector_lengths_ok || !matrix_shapes_ok {
                return Err(BackendError::InvalidTensorData(format!(
                    "qwen3vl layer {index} tensor geometry mismatch"
                )));
            }
        }

        let post_weight = load_vec(&mut file, &gguf, "v.post_ln.weight")?;
        let post_bias = load_vec(&mut file, &gguf, "v.post_ln.bias")?;
        let merger_0 = load_mat(&mut file, &gguf, "mm.0.weight")?;
        let merger_0_bias = load_vec(&mut file, &gguf, "mm.0.bias")?;
        let merger_2 = load_mat(&mut file, &gguf, "mm.2.weight")?;
        let merger_2_bias = load_vec(&mut file, &gguf, "mm.2.bias")?;
        let merged = hidden * merge * merge;
        if post_weight.len() != hidden
            || post_bias.len() != hidden
            || merger_0.input != merged
            || merger_0.output != merged
            || merger_0_bias.len() != merged
            || merger_2.input != merged
            || merger_2.output != projection
            || merger_2_bias.len() != projection
        {
            return Err(BackendError::InvalidTensorData(
                "qwen3vl post-norm/merger tensor geometry mismatch".into(),
            ));
        }

        Ok(Self {
            image_size,
            patch_size,
            merge,
            hidden,
            ffn,
            heads,
            eps,
            projection,
            patch_0,
            patch_1,
            patch_bias,
            position,
            layers,
            post_weight,
            post_bias,
            merger_0,
            merger_0_bias,
            merger_2,
            merger_2_bias,
        })
    }

    /// Decode, smart-resize, normalize and patchify one image. `max_tokens`
    /// controls the Qwen3-VL 32x32-pixel output-token cap used by Bonsai-demo.
    pub(crate) fn preprocess(
        &self,
        path: impl AsRef<Path>,
        min_tokens: usize,
        max_tokens: usize,
    ) -> Result<PrismVisionInput> {
        let image = image::ImageReader::open(path.as_ref())
            .map_err(|source| BackendError::Io {
                path: path.as_ref().to_path_buf(),
                source,
            })?
            .decode()
            .map_err(|error| BackendError::InvalidTensorData(format!("decode image: {error}")))?
            .to_rgb8();
        self.preprocess_rgb(image, min_tokens, max_tokens)
    }

    pub(crate) fn preprocess_rgb(
        &self,
        image: RgbImage,
        min_tokens: usize,
        max_tokens: usize,
    ) -> Result<PrismVisionInput> {
        let align = self.patch_size * self.merge;
        let (width, height) = smart_resize(
            image.width() as usize,
            image.height() as usize,
            align,
            min_tokens.max(1) * align * align,
            max_tokens.max(min_tokens.max(1)) * align * align,
        );
        let resized =
            image::imageops::resize(&image, width as u32, height as u32, FilterType::Triangle);
        let patch_width = width / self.patch_size;
        let patch_height = height / self.patch_size;
        if !patch_width.is_multiple_of(self.merge) || !patch_height.is_multiple_of(self.merge) {
            return Err(BackendError::InvalidTensorData(
                "smart-resized image is not aligned to the spatial merge".into(),
            ));
        }
        let patch_elements = 3 * self.patch_size * self.patch_size;
        let mut patches = Vec::with_capacity(patch_width * patch_height * patch_elements);
        for tile_y in (0..patch_height).step_by(self.merge) {
            for tile_x in (0..patch_width).step_by(self.merge) {
                for dy in 0..self.merge {
                    for dx in 0..self.merge {
                        let patch_y = tile_y + dy;
                        let patch_x = tile_x + dx;
                        for channel in 0..3 {
                            for y in 0..self.patch_size {
                                for x in 0..self.patch_size {
                                    let pixel = resized.get_pixel(
                                        (patch_x * self.patch_size + x) as u32,
                                        (patch_y * self.patch_size + y) as u32,
                                    );
                                    patches.push(pixel[channel] as f32 * (2.0 / 255.0) - 1.0);
                                }
                            }
                        }
                    }
                }
            }
        }
        let position = interpolate_positions_tile_major(
            &self.position,
            self.hidden,
            self.image_size / self.patch_size,
            patch_width,
            patch_height,
            self.merge,
        );
        Ok(PrismVisionInput {
            patches,
            position,
            patch_width,
            patch_height,
        })
    }

    /// Resolve the page-backed projector into a resident Metal encoder. The
    /// returned object holds all small bias/norm buffers and reuses the GGUF
    /// pages for every large matrix without copying them into a second host
    /// allocation.
    #[cfg(target_os = "macos")]
    pub(crate) fn metal_encoder(&self) -> Result<crate::metal::PrismVisionMetalEncoder> {
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                Ok(crate::metal::PrismVisionMetalLayerInput {
                    ln1_weight: &layer.ln1_weight,
                    ln1_bias: &layer.ln1_bias,
                    qkv: layer.qkv.metal_weight()?,
                    qkv_bias: &layer.qkv_bias,
                    attn_output: layer.attn_output.metal_weight()?,
                    attn_output_bias: &layer.attn_output_bias,
                    ln2_weight: &layer.ln2_weight,
                    ln2_bias: &layer.ln2_bias,
                    ffn_up: layer.ffn_up.metal_weight()?,
                    ffn_up_bias: &layer.ffn_up_bias,
                    ffn_down: layer.ffn_down.metal_weight()?,
                    ffn_down_bias: &layer.ffn_down_bias,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        crate::metal::PrismVisionMetalEncoder::new(
            crate::metal::PrismVisionMetalConfig {
                patch_input: 3 * self.patch_size * self.patch_size,
                hidden: self.hidden,
                ffn: self.ffn,
                heads: self.heads,
                merge: self.merge,
                projection: self.projection,
                eps: self.eps,
            },
            self.patch_0.metal_weight()?,
            self.patch_1.metal_weight()?,
            &self.patch_bias,
            &layers,
            &self.post_weight,
            &self.post_bias,
            self.merger_0.metal_weight()?,
            &self.merger_0_bias,
            self.merger_2.metal_weight()?,
            &self.merger_2_bias,
        )
        .ok_or_else(|| {
            BackendError::InvalidTensorData(
                "Qwen3-VL projector could not initialize its resident Metal graph".into(),
            )
        })
    }
}

fn find<'a>(gguf: &'a GgufFile, name: &str) -> Result<&'a GgufTensorDescriptor> {
    gguf.tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| BackendError::TensorNotFound(name.into()))
}

fn load_mat(file: &mut File, gguf: &GgufFile, name: &str) -> Result<VisionMat> {
    let descriptor = find(gguf, name)?;
    if descriptor.dimensions.len() < 2 {
        return Err(BackendError::InvalidTensorData(format!(
            "vision matrix {name} is not rank >= 2"
        )));
    }
    if !matches!(
        descriptor.tensor_type,
        GgufTensorType::F32 | GgufTensorType::F16 | GgufTensorType::Q8_0
    ) {
        return Err(BackendError::UnsupportedTensorType(format!(
            "vision matrix {name} uses {:?}",
            descriptor.tensor_type
        )));
    }
    let output = *descriptor.dimensions.last().unwrap() as usize;
    let input = descriptor.dimensions[..descriptor.dimensions.len() - 1]
        .iter()
        .product::<u64>() as usize;
    let byte_len = usize::try_from(descriptor.n_bytes).map_err(|_| {
        BackendError::InvalidTensorData(format!("vision matrix {name} is too large"))
    })?;
    Ok(VisionMat {
        pages: WirePages::read_from_file(file, descriptor.absolute_offset, byte_len)?,
        tensor_type: descriptor.tensor_type,
        input,
        output,
    })
}

fn load_vec(file: &mut File, gguf: &GgufFile, name: &str) -> Result<Vec<f32>> {
    let descriptor = find(gguf, name)?;
    let mut bytes = vec![0u8; descriptor.n_bytes as usize];
    file.seek(SeekFrom::Start(descriptor.absolute_offset))
        .map_err(|source| BackendError::Io {
            path: gguf.path.clone(),
            source,
        })?;
    file.read_exact(&mut bytes)
        .map_err(|source| BackendError::Io {
            path: gguf.path.clone(),
            source,
        })?;
    let elements = descriptor.dimensions.iter().product::<u64>() as usize;
    dequant::dequantize(descriptor.tensor_type, &bytes, elements, name)
}

fn round_to(value: f64, factor: usize) -> usize {
    ((value / factor as f64).round() as usize).max(1) * factor
}

fn smart_resize(
    width: usize,
    height: usize,
    align: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let mut out_w = round_to(width as f64, align);
    let mut out_h = round_to(height as f64, align);
    if out_w * out_h > max_pixels {
        let beta = ((width * height) as f64 / max_pixels as f64).sqrt();
        out_w = (((width as f64 / beta) / align as f64).floor() as usize).max(1) * align;
        out_h = (((height as f64 / beta) / align as f64).floor() as usize).max(1) * align;
    } else if out_w * out_h < min_pixels {
        let beta = (min_pixels as f64 / (width * height).max(1) as f64).sqrt();
        out_w = (((width as f64 * beta) / align as f64).ceil() as usize).max(1) * align;
        out_h = (((height as f64 * beta) / align as f64).ceil() as usize).max(1) * align;
    }
    (out_w, out_h)
}

fn interpolate_positions_tile_major(
    source: &[f32],
    hidden: usize,
    source_side: usize,
    width: usize,
    height: usize,
    merge: usize,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(width * height * hidden);
    for tile_y in (0..height).step_by(merge) {
        for tile_x in (0..width).step_by(merge) {
            for dy in 0..merge {
                for dx in 0..merge {
                    let y = tile_y + dy;
                    let x = tile_x + dx;
                    let source_y = ((y as f32 + 0.5) * source_side as f32 / height as f32 - 0.5)
                        .clamp(0.0, (source_side - 1) as f32);
                    let source_x = ((x as f32 + 0.5) * source_side as f32 / width as f32 - 0.5)
                        .clamp(0.0, (source_side - 1) as f32);
                    let y0 = source_y.floor() as usize;
                    let x0 = source_x.floor() as usize;
                    let y1 = (y0 + 1).min(source_side - 1);
                    let x1 = (x0 + 1).min(source_side - 1);
                    let fy = source_y - y0 as f32;
                    let fx = source_x - x0 as f32;
                    for channel in 0..hidden {
                        let at = |py: usize, px: usize| {
                            source[(py * source_side + px) * hidden + channel]
                        };
                        let top = at(y0, x0) + (at(y0, x1) - at(y0, x0)) * fx;
                        let bottom = at(y1, x0) + (at(y1, x1) - at(y1, x0)) * fx;
                        output.push(top + (bottom - top) * fy);
                    }
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3vl_smart_resize_is_merge_aligned_and_bounded() {
        assert_eq!(
            smart_resize(1442, 720, 32, 8 * 1024, 1024 * 1024),
            (1440, 704)
        );
        assert_eq!(smart_resize(32, 32, 32, 1024, 1024), (32, 32));
        let (width, height) = smart_resize(12_000, 9_000, 32, 8 * 1024, 1024 * 1024);
        assert_eq!(width % 32, 0);
        assert_eq!(height % 32, 0);
        assert!(width * height <= 1024 * 1024);
    }

    #[test]
    fn real_prism_mmproj_loads_as_page_backed_qwen3vl() {
        let Ok(path) = std::env::var("CAMELID_PRISM_MMPROJ") else {
            eprintln!("SKIP: set CAMELID_PRISM_MMPROJ");
            return;
        };
        let model = PrismVisionModel::load(path).expect("load Prism mmproj");
        assert_eq!(model.hidden, 1152);
        assert_eq!(model.layers.len(), 27);
        assert_eq!(model.projection, 5120);
        assert_eq!(model.patch_0.pages.byte_len(), 16 * 16 * 3 * 1152 * 4);
        assert_eq!(model.layers[0].qkv.tensor_type, GgufTensorType::Q8_0);
        assert_eq!(model.layers[0].ffn_down.tensor_type, GgufTensorType::F16);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn real_prism_mmproj_encodes_a_tiny_image_on_metal() {
        let (Ok(model_path), Ok(image_path)) = (
            std::env::var("CAMELID_PRISM_MMPROJ"),
            std::env::var("CAMELID_PRISM_IMAGE"),
        ) else {
            eprintln!("SKIP: set CAMELID_PRISM_MMPROJ and CAMELID_PRISM_IMAGE");
            return;
        };
        let model = PrismVisionModel::load(model_path).expect("load Prism mmproj");
        let input = model
            .preprocess(image_path, 1, 1)
            .expect("preprocess tiny image");
        assert_eq!(input.patch_count(), 4);
        assert_eq!(input.output_tokens(model.merge), 1);
        let mut encoder = model.metal_encoder().expect("build Metal encoder");
        let output = encoder
            .encode(
                &input.patches,
                &input.position,
                input.patch_width,
                input.patch_height,
            )
            .expect("encode tiny image");
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), model.projection);
        assert!(output[0].iter().all(|value| value.is_finite()));
        let l2 = output[0]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        eprintln!("Qwen3-VL tiny image embedding: l2={l2:.6}");
        assert!(l2 > 1.0);
    }
}
