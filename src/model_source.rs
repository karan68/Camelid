use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{BackendError, Result};

/// Source-level model formats that Camelid can reason about before runtime loading.
///
/// This layer is intentionally narrower than generation support: a Hugging Face
/// SafeTensors directory can be detected and reported without becoming runnable.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSourceKind {
    Gguf,
    HuggingFaceSafeTensors,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelSourceManifest {
    pub id: String,
    pub kind: ModelSourceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_config: Option<HfLlamaConfigSummary>,
    #[serde(skip_serializing)]
    pub root: PathBuf,
    #[serde(skip_serializing)]
    pub weight_files: Vec<PathBuf>,
    pub tensor_descriptors: Vec<SafeTensorsTensorDescriptor>,
    #[serde(skip_serializing)]
    pub config_path: Option<PathBuf>,
    #[serde(skip_serializing)]
    pub tokenizer_path: Option<PathBuf>,
    #[serde(skip_serializing)]
    pub tokenizer_config_path: Option<PathBuf>,
    #[serde(skip_serializing)]
    pub special_tokens_map_path: Option<PathBuf>,
    #[serde(skip_serializing)]
    pub generation_config_path: Option<PathBuf>,
    #[serde(skip_serializing)]
    pub shard_index_path: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HfLlamaConfigSummary {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub max_position_embeddings: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: u32,
    pub tie_word_embeddings: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SafeTensorsTensorDescriptor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub shard_file: String,
    #[serde(skip_serializing)]
    pub shard: PathBuf,
    pub data_offsets: [u64; 2],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelSourceReadiness {
    pub metadata_ready: bool,
    pub tokenizer_ready: bool,
    pub weights_ready: bool,
    pub generation_ready: bool,
    pub blockers: Vec<ModelSourceBlocker>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelSourceBlocker {
    pub code: &'static str,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelSourceInspection {
    pub manifest: ModelSourceManifest,
    pub readiness: ModelSourceReadiness,
}

#[derive(Debug, Deserialize)]
struct HfConfigProbe {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
    hidden_size: Option<u32>,
    num_hidden_layers: Option<u32>,
    intermediate_size: Option<u32>,
    num_attention_heads: Option<u32>,
    num_key_value_heads: Option<u32>,
    max_position_embeddings: Option<u32>,
    rms_norm_eps: Option<f32>,
    rope_theta: Option<f32>,
    vocab_size: Option<u32>,
    tie_word_embeddings: Option<bool>,
    rope_scaling: Option<Value>,
    sliding_window: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct SafeTensorsHeaderTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// Inspect a local model source without constructing runtime weights.
///
/// Existing GGUF callers should continue to use the current loader path. This
/// helper gives the SafeTensors lane a descriptor/readiness seam and deliberately
/// leaves Hugging Face SafeTensors generation disabled.
pub fn inspect_model_source(path: impl AsRef<Path>) -> Result<ModelSourceInspection> {
    let path = path.as_ref();
    let metadata = fs::metadata(path).map_err(|source| BackendError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    if metadata.is_file() && has_extension(path, "gguf") {
        return Ok(inspect_gguf_file(path));
    }

    if metadata.is_dir() {
        return inspect_hugging_face_safetensors_dir(path);
    }

    Err(BackendError::InvalidModelMetadata(format!(
        "unsupported model source path {}; expected a .gguf file or local Hugging Face directory",
        public_path_label(path)
    )))
}

fn inspect_gguf_file(path: &Path) -> ModelSourceInspection {
    ModelSourceInspection {
        manifest: ModelSourceManifest {
            id: source_id(path),
            kind: ModelSourceKind::Gguf,
            hf_config: None,
            root: path.to_path_buf(),
            weight_files: vec![path.to_path_buf()],
            tensor_descriptors: Vec::new(),
            config_path: None,
            tokenizer_path: None,
            tokenizer_config_path: None,
            special_tokens_map_path: None,
            generation_config_path: None,
            shard_index_path: None,
        },
        readiness: ModelSourceReadiness {
            metadata_ready: true,
            tokenizer_ready: false,
            weights_ready: true,
            generation_ready: false,
            blockers: vec![blocker(
                "gguf_runtime_loader_unchanged",
                "GGUF source detection is present, but generation readiness is still owned by the existing GGUF runtime loader",
            )],
        },
    }
}

fn inspect_hugging_face_safetensors_dir(path: &Path) -> Result<ModelSourceInspection> {
    let config_path = existing_child(path, "config.json");
    let tokenizer_path = existing_child(path, "tokenizer.json");
    let tokenizer_config_path = existing_child(path, "tokenizer_config.json");
    let special_tokens_map_path = existing_child(path, "special_tokens_map.json");
    let generation_config_path = existing_child(path, "generation_config.json");
    let shard_index_path = existing_child(path, "model.safetensors.index.json");
    let weight_files = safetensors_files(path)?;

    let mut blockers = Vec::new();
    let (metadata_ready, hf_config) = config_path
        .as_ref()
        .map(|config_path| hf_config_summary(config_path, &mut blockers))
        .unwrap_or_else(|| {
            blockers.push(blocker(
                "missing_config_json",
                "Hugging Face SafeTensors directories must include config.json",
            ));
            (false, None)
        });

    let tokenizer_ready = tokenizer_path.is_some();
    if !tokenizer_ready {
        blockers.push(blocker(
            "missing_tokenizer_json",
            "Hugging Face SafeTensors directories must include tokenizer.json before tokenizer readiness can be reported",
        ));
    }

    let (weights_ready, tensor_descriptors) =
        hf_weights_ready(&weight_files, shard_index_path.as_deref(), &mut blockers)?;
    blockers.push(blocker(
        "generation_disabled",
        "SafeTensors generation remains disabled until tokenizer parity, tensor orientation, dtype decode, and one-token dense execution fixtures pass",
    ));

    Ok(ModelSourceInspection {
        manifest: ModelSourceManifest {
            id: source_id(path),
            kind: ModelSourceKind::HuggingFaceSafeTensors,
            hf_config,
            root: path.to_path_buf(),
            weight_files,
            tensor_descriptors,
            config_path,
            tokenizer_path,
            tokenizer_config_path,
            special_tokens_map_path,
            generation_config_path,
            shard_index_path,
        },
        readiness: ModelSourceReadiness {
            metadata_ready,
            tokenizer_ready,
            weights_ready,
            generation_ready: false,
            blockers,
        },
    })
}

fn hf_config_summary(
    config_path: &Path,
    blockers: &mut Vec<ModelSourceBlocker>,
) -> (bool, Option<HfLlamaConfigSummary>) {
    let Ok(bytes) = fs::read(config_path) else {
        blockers.push(blocker(
            "config_json_unreadable",
            format!(
                "could not read required Hugging Face config file {}",
                public_path_label(config_path)
            ),
        ));
        return (false, None);
    };
    let Ok(config) = serde_json::from_slice::<HfConfigProbe>(&bytes) else {
        blockers.push(blocker(
            "invalid_config_json",
            format!(
                "required Hugging Face config file {} is not valid JSON",
                public_path_label(config_path)
            ),
        ));
        return (false, None);
    };

    if config.model_type.as_deref() != Some("llama") {
        blockers.push(blocker(
            "unsupported_model_type",
            format!(
                "only dense LLaMA-family SafeTensors config metadata is in scope for the first readiness slice; got {:?}",
                config.model_type
            ),
        ));
        return (false, None);
    }

    let Some(architectures) = config.architectures.clone() else {
        blockers.push(blocker(
            "missing_config_field",
            "Hugging Face config.json is missing required field architectures",
        ));
        return (false, None);
    };
    if !architectures
        .iter()
        .any(|architecture| architecture == "LlamaForCausalLM")
    {
        blockers.push(blocker(
            "unsupported_architecture",
            format!(
                "only LlamaForCausalLM SafeTensors configs are in scope for this readiness slice; got {:?}",
                architectures
            ),
        ));
        return (false, None);
    }

    if config.sliding_window.is_some() {
        blockers.push(blocker(
            "unsupported_sliding_window_attention",
            "sliding-window attention needs an explicit Camelid runtime mapping before SafeTensors metadata can be ready",
        ));
        return (false, None);
    }

    if config.rope_scaling.is_some() {
        blockers.push(blocker(
            "unsupported_rope_scaling",
            "rope_scaling needs an explicit Camelid HF config mapping before SafeTensors metadata can be ready",
        ));
        return (false, None);
    }

    let Some(hidden_size) = required_hf_u32(&config, "hidden_size", config.hidden_size, blockers)
    else {
        return (false, None);
    };
    let Some(num_hidden_layers) = required_hf_u32(
        &config,
        "num_hidden_layers",
        config.num_hidden_layers,
        blockers,
    ) else {
        return (false, None);
    };
    let Some(intermediate_size) = required_hf_u32(
        &config,
        "intermediate_size",
        config.intermediate_size,
        blockers,
    ) else {
        return (false, None);
    };
    let Some(num_attention_heads) = required_hf_u32(
        &config,
        "num_attention_heads",
        config.num_attention_heads,
        blockers,
    ) else {
        return (false, None);
    };
    let Some(num_key_value_heads) = required_hf_u32(
        &config,
        "num_key_value_heads",
        config.num_key_value_heads,
        blockers,
    ) else {
        return (false, None);
    };
    let Some(max_position_embeddings) = required_hf_u32(
        &config,
        "max_position_embeddings",
        config.max_position_embeddings,
        blockers,
    ) else {
        return (false, None);
    };
    let Some(vocab_size) = required_hf_u32(&config, "vocab_size", config.vocab_size, blockers)
    else {
        return (false, None);
    };
    let Some(rms_norm_eps) =
        required_hf_f32(&config, "rms_norm_eps", config.rms_norm_eps, blockers)
    else {
        return (false, None);
    };
    let Some(rope_theta) = required_hf_f32(&config, "rope_theta", config.rope_theta, blockers)
    else {
        return (false, None);
    };
    let Some(tie_word_embeddings) = config.tie_word_embeddings else {
        blockers.push(blocker(
            "missing_config_field",
            "Hugging Face config.json is missing required field tie_word_embeddings",
        ));
        return (false, None);
    };

    if hidden_size % num_attention_heads != 0 {
        blockers.push(blocker(
            "invalid_attention_geometry",
            format!(
                "hidden_size {hidden_size} must be divisible by num_attention_heads {num_attention_heads} before SafeTensors metadata can be ready"
            ),
        ));
        return (false, None);
    }
    if num_key_value_heads > num_attention_heads {
        blockers.push(blocker(
            "invalid_attention_geometry",
            format!(
                "num_key_value_heads {num_key_value_heads} must be <= num_attention_heads {num_attention_heads} before SafeTensors metadata can be ready"
            ),
        ));
        return (false, None);
    }

    (
        true,
        Some(HfLlamaConfigSummary {
            model_type: "llama".to_string(),
            architectures,
            hidden_size,
            num_hidden_layers,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            vocab_size,
            tie_word_embeddings,
        }),
    )
}

fn required_hf_u32(
    _config: &HfConfigProbe,
    field: &'static str,
    value: Option<u32>,
    blockers: &mut Vec<ModelSourceBlocker>,
) -> Option<u32> {
    match value {
        Some(value) if value > 0 => Some(value),
        Some(_) => {
            blockers.push(blocker(
                "invalid_config_field",
                format!("Hugging Face config.json field {field} must be greater than zero"),
            ));
            None
        }
        None => {
            blockers.push(blocker(
                "missing_config_field",
                format!("Hugging Face config.json is missing required field {field}"),
            ));
            None
        }
    }
}

fn required_hf_f32(
    _config: &HfConfigProbe,
    field: &'static str,
    value: Option<f32>,
    blockers: &mut Vec<ModelSourceBlocker>,
) -> Option<f32> {
    match value {
        Some(value) if value.is_finite() && value > 0.0 => Some(value),
        Some(_) => {
            blockers.push(blocker(
                "invalid_config_field",
                format!(
                    "Hugging Face config.json field {field} must be finite and greater than zero"
                ),
            ));
            None
        }
        None => {
            blockers.push(blocker(
                "missing_config_field",
                format!("Hugging Face config.json is missing required field {field}"),
            ));
            None
        }
    }
}

fn hf_weights_ready(
    weight_files: &[PathBuf],
    shard_index_path: Option<&Path>,
    blockers: &mut Vec<ModelSourceBlocker>,
) -> Result<(bool, Vec<SafeTensorsTensorDescriptor>)> {
    if weight_files.is_empty() {
        blockers.push(blocker(
            "missing_safetensors_weights",
            "Hugging Face SafeTensors directories must include at least one .safetensors weight file",
        ));
        return Ok((false, Vec::new()));
    }

    let mut ready = true;
    let tensor_descriptors = inspect_safetensors_headers(weight_files, blockers)?;
    if tensor_descriptors.is_empty() {
        ready = false;
    }
    if !safetensors_dtypes_ready(&tensor_descriptors, blockers) {
        ready = false;
    }

    if weight_files.len() > 1 && shard_index_path.is_none() {
        blockers.push(blocker(
            "missing_shard_index",
            "sharded SafeTensors directories must include model.safetensors.index.json before weights readiness can be reported",
        ));
        ready = false;
    }

    if let Some(shard_index_path) = shard_index_path {
        let bytes = fs::read(shard_index_path).map_err(|source| BackendError::Io {
            path: shard_index_path.to_path_buf(),
            source,
        })?;
        let Ok(index) = serde_json::from_slice::<Value>(&bytes) else {
            blockers.push(blocker(
                "invalid_shard_index_json",
                format!(
                    "SafeTensors shard index file {} is not valid JSON",
                    public_path_label(shard_index_path)
                ),
            ));
            return Ok((false, tensor_descriptors));
        };
        if !hf_shard_index_weight_map_ready(&index, weight_files, &tensor_descriptors, blockers) {
            ready = false;
        }
    }

    Ok((ready, tensor_descriptors))
}

fn inspect_safetensors_headers(
    weight_files: &[PathBuf],
    blockers: &mut Vec<ModelSourceBlocker>,
) -> Result<Vec<SafeTensorsTensorDescriptor>> {
    let mut tensor_descriptors = Vec::new();
    let mut seen = BTreeMap::<String, String>::new();
    for weight_file in weight_files {
        match parse_safetensors_header(weight_file) {
            Ok(file_descriptors) => {
                if file_descriptors.is_empty() {
                    blockers.push(blocker(
                        "empty_safetensors_header",
                        format!(
                            "SafeTensors shard {} does not list any tensor descriptors",
                            public_path_label(weight_file)
                        ),
                    ));
                }
                for descriptor in file_descriptors {
                    if let Some(first_shard) = seen.insert(
                        descriptor.name.clone(),
                        public_path_label(&descriptor.shard),
                    ) {
                        blockers.push(blocker(
                            "duplicate_safetensors_tensor",
                            format!(
                                "SafeTensors tensor {} appears in more than one shard: {}, {}",
                                descriptor.name,
                                first_shard,
                                public_path_label(&descriptor.shard)
                            ),
                        ));
                    }
                    tensor_descriptors.push(descriptor);
                }
            }
            Err(message) => blockers.push(blocker(
                "invalid_safetensors_header",
                format!(
                    "SafeTensors shard {} has an invalid header: {}",
                    public_path_label(weight_file),
                    message
                ),
            )),
        }
    }
    tensor_descriptors.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(tensor_descriptors)
}

fn parse_safetensors_header(
    path: &Path,
) -> std::result::Result<Vec<SafeTensorsTensorDescriptor>, String> {
    let mut file = fs::File::open(path).map_err(|err| format!("could not open shard: {err}"))?;
    let file_len = file
        .metadata()
        .map_err(|err| format!("could not stat shard: {err}"))?
        .len();
    if file_len < 8 {
        return Err("file is shorter than the 8-byte SafeTensors header length".to_string());
    }

    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|err| format!("could not read header length: {err}"))?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    let header_len = usize::try_from(header_len)
        .map_err(|_| "header length does not fit this platform".to_string())?;
    let header_end = 8u64
        .checked_add(header_len as u64)
        .ok_or_else(|| "header length overflows file offset arithmetic".to_string())?;
    if header_end > file_len {
        return Err("header length extends past end of file".to_string());
    }

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|err| format!("could not read header JSON: {err}"))?;

    let header = serde_json::from_slice::<Value>(&header_bytes)
        .map_err(|err| format!("header JSON is invalid: {err}"))?;
    let object = header
        .as_object()
        .ok_or_else(|| "header JSON must be an object".to_string())?;
    let payload_len = file_len - header_end;
    let mut descriptors = Vec::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let tensor = serde_json::from_value::<SafeTensorsHeaderTensor>(value.clone())
            .map_err(|err| format!("tensor descriptor {name} is invalid: {err}"))?;
        if tensor.data_offsets[0] > tensor.data_offsets[1] {
            return Err(format!(
                "tensor descriptor {name} has descending data_offsets"
            ));
        }
        if tensor.data_offsets[1] > payload_len {
            return Err(format!(
                "tensor descriptor {name} data_offsets extend past shard payload"
            ));
        }
        if let Some(dtype_size) = safetensors_dtype_size(&tensor.dtype) {
            let element_count = tensor.shape.iter().try_fold(1u64, |acc, dim| {
                acc.checked_mul(*dim).ok_or_else(|| {
                    format!("tensor descriptor {name} shape element count overflows")
                })
            })?;
            let expected_len = element_count
                .checked_mul(dtype_size)
                .ok_or_else(|| format!("tensor descriptor {name} byte length overflows"))?;
            let actual_len = tensor.data_offsets[1] - tensor.data_offsets[0];
            if actual_len != expected_len {
                return Err(format!(
                    "tensor descriptor {name} data_offsets length {actual_len} does not match shape/dtype byte length {expected_len}"
                ));
            }
        }
        descriptors.push(SafeTensorsTensorDescriptor {
            name: name.clone(),
            dtype: tensor.dtype,
            shape: tensor.shape,
            shard_file: public_path_label(path),
            shard: path.to_path_buf(),
            data_offsets: tensor.data_offsets,
        });
    }
    Ok(descriptors)
}

fn safetensors_dtype_size(dtype: &str) -> Option<u64> {
    match dtype {
        "F32" => Some(4),
        "F16" | "BF16" => Some(2),
        _ => None,
    }
}

fn safetensors_dtypes_ready(
    tensor_descriptors: &[SafeTensorsTensorDescriptor],
    blockers: &mut Vec<ModelSourceBlocker>,
) -> bool {
    let unsupported = tensor_descriptors
        .iter()
        .filter(|descriptor| !matches!(descriptor.dtype.as_str(), "F32" | "F16" | "BF16"))
        .map(|descriptor| format!("{}:{}", descriptor.name, descriptor.dtype))
        .collect::<Vec<_>>();
    if unsupported.is_empty() {
        return true;
    }
    blockers.push(blocker(
        "unsupported_safetensors_dtype",
        format!(
            "only F32, F16, and BF16 SafeTensors descriptors are in scope for this readiness slice; unsupported tensors: {}",
            unsupported.join(", ")
        ),
    ));
    false
}

fn hf_shard_index_weight_map_ready(
    index: &Value,
    weight_files: &[PathBuf],
    tensor_descriptors: &[SafeTensorsTensorDescriptor],
    blockers: &mut Vec<ModelSourceBlocker>,
) -> bool {
    let Some(weight_map) = index.get("weight_map").and_then(Value::as_object) else {
        blockers.push(blocker(
            "missing_shard_index_weight_map",
            "model.safetensors.index.json must include a weight_map object before sharded weights readiness can be reported",
        ));
        return false;
    };
    if weight_map.is_empty() {
        blockers.push(blocker(
            "empty_shard_index_weight_map",
            "model.safetensors.index.json weight_map must list at least one tensor shard before sharded weights readiness can be reported",
        ));
        return false;
    }

    let available = weight_files
        .iter()
        .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
        .collect::<BTreeSet<_>>();
    let mut missing = BTreeSet::new();
    let mut invalid = BTreeSet::new();
    let mut invalid_filenames = BTreeSet::new();
    let mut missing_tensors = BTreeSet::new();
    let tensors_by_shard = safetensors_tensors_by_shard(tensor_descriptors);
    for (tensor_name, shard_name) in weight_map {
        let Some(shard_name) = shard_name.as_str() else {
            invalid.insert(tensor_name.as_str());
            continue;
        };
        if !is_plain_safetensors_shard_filename(shard_name) {
            invalid_filenames.insert(tensor_name.as_str());
            continue;
        }
        if !available.contains(shard_name) {
            missing.insert(shard_name);
            continue;
        }
        if !tensors_by_shard
            .get(shard_name)
            .is_some_and(|tensors| tensors.contains(tensor_name.as_str()))
        {
            missing_tensors.insert(format!("{tensor_name} in {shard_name}"));
        }
    }

    if !invalid.is_empty() {
        blockers.push(blocker(
            "invalid_shard_index_weight_map",
            format!(
                "model.safetensors.index.json weight_map entries must map tensor names to shard filenames; invalid entries: {}",
                invalid.into_iter().collect::<Vec<_>>().join(", ")
            ),
        ));
        return false;
    }
    if !invalid_filenames.is_empty() {
        blockers.push(blocker(
            "invalid_shard_index_shard_filename",
            format!(
                "model.safetensors.index.json weight_map shard values must be local shard filenames, not paths; invalid tensor entries: {}",
                invalid_filenames.into_iter().collect::<Vec<_>>().join(", ")
            ),
        ));
        return false;
    }
    if !missing.is_empty() {
        blockers.push(blocker(
            "missing_sharded_weight_file",
            format!(
                "model.safetensors.index.json references shard files that are not present locally: {}",
                missing.into_iter().collect::<Vec<_>>().join(", ")
            ),
        ));
        return false;
    }
    if !missing_tensors.is_empty() {
        blockers.push(blocker(
            "shard_index_tensor_not_found",
            format!(
                "model.safetensors.index.json references tensors not found in the listed shard headers: {}",
                missing_tensors.into_iter().collect::<Vec<_>>().join(", ")
            ),
        ));
        return false;
    }

    true
}

fn safetensors_tensors_by_shard(
    tensor_descriptors: &[SafeTensorsTensorDescriptor],
) -> BTreeMap<&str, BTreeSet<&str>> {
    let mut tensors_by_shard = BTreeMap::<&str, BTreeSet<&str>>::new();
    for descriptor in tensor_descriptors {
        if let Some(shard_name) = descriptor.shard.file_name().and_then(|name| name.to_str()) {
            tensors_by_shard
                .entry(shard_name)
                .or_default()
                .insert(descriptor.name.as_str());
        }
    }
    tensors_by_shard
}

fn is_plain_safetensors_shard_filename(value: &str) -> bool {
    !value.is_empty()
        && !value.contains('/')
        && !value.contains('\\')
        && value != "."
        && value != ".."
        && has_extension(Path::new(value), "safetensors")
}

fn safetensors_files(path: &Path) -> Result<Vec<PathBuf>> {
    let entries = fs::read_dir(path).map_err(|source| BackendError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| BackendError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let candidate = entry.path();
        if candidate.is_file() && has_extension(&candidate, "safetensors") {
            files.push(candidate);
        }
    }
    files.sort();
    Ok(files)
}

fn existing_child(root: &Path, name: &str) -> Option<PathBuf> {
    let candidate = root.join(name);
    candidate.is_file().then_some(candidate)
}

fn has_extension(path: &Path, extension: &str) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case(extension))
}

fn source_id(path: &Path) -> String {
    let name = if has_extension(path, "gguf") {
        path.file_stem().or_else(|| path.file_name())
    } else {
        path.file_name()
    };
    name.and_then(|value| value.to_str())
        .unwrap_or("model")
        .to_string()
}

fn public_path_label(path: &Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model source file")
        .to_string()
}

fn blocker(code: &'static str, message: impl Into<String>) -> ModelSourceBlocker {
    ModelSourceBlocker {
        code,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::*;

    #[test]
    fn complete_hf_safetensors_directory_reports_readiness_but_not_generation() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        fs::write(dir.path().join("tokenizer_config.json"), "{}").unwrap();
        fs::write(dir.path().join("special_tokens_map.json"), "{}").unwrap();
        fs::write(dir.path().join("generation_config.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        write_safetensors_file(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("lm_head.weight", "BF16", &[1, 1])],
        );
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"model.embed_tokens.weight":"model-00001-of-00002.safetensors","lm_head.weight":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert_eq!(
            inspection.manifest.kind,
            ModelSourceKind::HuggingFaceSafeTensors
        );
        assert_eq!(inspection.manifest.weight_files.len(), 2);
        assert!(inspection.manifest.config_path.is_some());
        let hf_config = inspection.manifest.hf_config.as_ref().unwrap();
        assert_eq!(hf_config.hidden_size, 16);
        assert_eq!(hf_config.num_hidden_layers, 2);
        assert_eq!(hf_config.intermediate_size, 64);
        assert_eq!(hf_config.num_attention_heads, 4);
        assert_eq!(hf_config.num_key_value_heads, 2);
        assert_eq!(hf_config.max_position_embeddings, 128);
        assert_eq!(hf_config.vocab_size, 32000);
        assert!(!hf_config.tie_word_embeddings);
        assert!(inspection.manifest.tokenizer_path.is_some());
        assert!(inspection.manifest.shard_index_path.is_some());
        assert_eq!(inspection.manifest.tensor_descriptors.len(), 2);
        assert_eq!(
            inspection.manifest.tensor_descriptors[0].name,
            "lm_head.weight"
        );
        assert_eq!(inspection.manifest.tensor_descriptors[0].dtype, "BF16");
        assert_eq!(
            inspection.manifest.tensor_descriptors[0].shard_file,
            "model-00002-of-00002.safetensors"
        );
        assert_eq!(
            inspection.manifest.tensor_descriptors[1].name,
            "model.embed_tokens.weight"
        );
        assert_eq!(inspection.manifest.tensor_descriptors[1].shape, vec![1, 1]);
        assert!(inspection.readiness.metadata_ready);
        assert!(inspection.readiness.tokenizer_ready);
        assert!(inspection.readiness.weights_ready);
        assert!(!inspection.readiness.generation_ready);
        assert_blocker_codes(&inspection, &["generation_disabled"]);
    }

    #[test]
    fn missing_tokenizer_fixture_has_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "F16", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(inspection.readiness.metadata_ready);
        assert!(!inspection.readiness.tokenizer_ready);
        assert!(inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["missing_tokenizer_json", "generation_disabled"],
        );
    }

    #[test]
    fn sharded_weights_without_index_have_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        write_safetensors_file(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("lm_head.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(inspection.readiness.metadata_ready);
        assert!(inspection.readiness.tokenizer_ready);
        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(&inspection, &["missing_shard_index", "generation_disabled"]);
    }

    #[test]
    fn invalid_shard_index_has_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        write_safetensors_file(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("lm_head.weight", "F32", &[1, 1])],
        );
        fs::write(dir.path().join("model.safetensors.index.json"), "not json").unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["invalid_shard_index_json", "generation_disabled"],
        );
        assert_public_blocker_message_without_local_path(
            &inspection.readiness.blockers[0].message,
            dir.path(),
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("model.safetensors.index.json"));
    }

    #[test]
    fn shard_index_without_weight_map_has_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        write_safetensors_file(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("lm_head.weight", "F32", &[1, 1])],
        );
        fs::write(dir.path().join("model.safetensors.index.json"), "{}").unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["missing_shard_index_weight_map", "generation_disabled"],
        );
    }

    #[test]
    fn shard_index_referencing_missing_weight_file_has_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        write_safetensors_file(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("lm_head.weight", "F32", &[1, 1])],
        );
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"model.embed_tokens.weight":"model-00003-of-00003.safetensors"}}"#,
        )
        .unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["missing_sharded_weight_file", "generation_disabled"],
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("model-00003-of-00003.safetensors"));
    }

    #[test]
    fn invalid_config_json_has_sanitized_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), "not json").unwrap();
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.metadata_ready);
        assert!(inspection.readiness.tokenizer_ready);
        assert!(inspection.readiness.weights_ready);
        assert_blocker_codes(&inspection, &["invalid_config_json", "generation_disabled"]);
        assert_public_blocker_message_without_local_path(
            &inspection.readiness.blockers[0].message,
            dir.path(),
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("config.json"));
    }

    #[test]
    fn unsupported_config_fixture_has_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"model_type":"mistral","architectures":["MistralForCausalLM"]}"#,
        )
        .unwrap();
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.metadata_ready);
        assert!(inspection.readiness.tokenizer_ready);
        assert!(inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["unsupported_model_type", "generation_disabled"],
        );
    }

    #[test]
    fn missing_required_hf_config_field_blocks_metadata_readiness() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"model_type":"llama","architectures":["LlamaForCausalLM"],"hidden_size":16}"#,
        )
        .unwrap();
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.metadata_ready);
        assert!(inspection.manifest.hf_config.is_none());
        assert_blocker_codes(
            &inspection,
            &["missing_config_field", "generation_disabled"],
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("num_hidden_layers"));
    }

    #[test]
    fn invalid_hf_attention_geometry_blocks_metadata_readiness() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config_with_overrides(
            dir.path(),
            &[
                ("hidden_size", serde_json::json!(18)),
                ("num_attention_heads", serde_json::json!(4)),
            ],
        );
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.metadata_ready);
        assert_blocker_codes(
            &inspection,
            &["invalid_attention_geometry", "generation_disabled"],
        );
    }

    #[test]
    fn rope_scaling_config_blocks_metadata_readiness_until_mapped() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config_with_overrides(
            dir.path(),
            &[(
                "rope_scaling",
                serde_json::json!({"type":"linear","factor":2.0}),
            )],
        );
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.metadata_ready);
        assert_blocker_codes(
            &inspection,
            &["unsupported_rope_scaling", "generation_disabled"],
        );
    }

    #[test]
    fn shard_index_path_values_have_sanitized_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        write_safetensors_file(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("lm_head.weight", "F32", &[1, 1])],
        );
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"model.embed_tokens.weight":"../private/model-00001-of-00002.safetensors","lm_head.weight":"C:\\private\\model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["invalid_shard_index_shard_filename", "generation_disabled"],
        );
        let message = &inspection.readiness.blockers[0].message;
        assert!(message.contains("model.embed_tokens.weight"));
        assert!(message.contains("lm_head.weight"));
        assert!(!message.contains("../private"));
        assert!(!message.contains("C:"));
        assert!(!message.contains("model-00001-of-00002.safetensors"));
        assert!(!message.contains("model-00002-of-00002.safetensors"));
    }

    #[test]
    fn shard_index_invalid_entries_are_reported_in_stable_tensor_order() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        write_safetensors_file(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("lm_head.weight", "F32", &[1, 1])],
        );
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"z.weight":42,"a.weight":false}}"#,
        )
        .unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["invalid_shard_index_weight_map", "generation_disabled"],
        );
        let message = &inspection.readiness.blockers[0].message;
        assert!(message.ends_with("invalid entries: a.weight, z.weight"));
    }

    #[test]
    fn hf_directory_source_id_preserves_dotted_model_name() {
        let root = tempfile::tempdir().unwrap();
        let model_dir = root.path().join("Meta-Llama-3.1-8B-Instruct");
        fs::create_dir(&model_dir).unwrap();
        write_llama_config(&model_dir);
        fs::write(model_dir.join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            &model_dir,
            "model.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(&model_dir).unwrap();

        assert_eq!(inspection.manifest.id, "Meta-Llama-3.1-8B-Instruct");
    }

    #[test]
    fn invalid_safetensors_header_blocks_weight_readiness() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        fs::write(dir.path().join("model.safetensors"), b"too short").unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert!(inspection.manifest.tensor_descriptors.is_empty());
        assert_blocker_codes(
            &inspection,
            &["invalid_safetensors_header", "generation_disabled"],
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("model.safetensors"));
    }

    #[test]
    fn unsupported_safetensors_dtype_blocks_weight_readiness() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "I64", &[1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_eq!(inspection.manifest.tensor_descriptors.len(), 1);
        assert_blocker_codes(
            &inspection,
            &["unsupported_safetensors_dtype", "generation_disabled"],
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("model.embed_tokens.weight:I64"));
    }

    #[test]
    fn shard_index_tensor_missing_from_header_blocks_weight_readiness() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00001.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"lm_head.weight":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["shard_index_tensor_not_found", "generation_disabled"],
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("lm_head.weight in model-00001-of-00001.safetensors"));
    }

    #[test]
    fn serialized_hf_inspection_does_not_expose_local_paths() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();
        let json = serde_json::to_string(&inspection).unwrap();

        assert_public_blocker_message_without_local_path(&json, dir.path());
        assert!(!json.contains("root"));
        assert!(!json.contains("weight_files"));
        assert!(!json.contains("config_path"));
        assert!(!json.contains("tokenizer_path"));
        assert!(!json.contains("\"shard\""));
        assert!(json.contains("shard_file"));
        assert!(json.contains("model.safetensors"));
    }

    #[test]
    fn safetensors_shape_dtype_offset_mismatch_blocks_weight_readiness() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        let header = serde_json::json!({
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 4]
            }
        });
        write_safetensors_bytes(dir.path(), "model.safetensors", &header, &[0, 0, 0, 0]);

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["invalid_safetensors_header", "generation_disabled"],
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("does not match shape/dtype byte length 8"));
    }

    #[test]
    fn gguf_file_source_id_strips_only_gguf_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf");
        fs::write(&path, b"").unwrap();

        let inspection = inspect_model_source(&path).unwrap();

        assert_eq!(inspection.manifest.id, "TinyLlama-1.1B-Chat-v1.0.Q8_0");
    }

    #[test]
    fn unsupported_source_file_error_uses_public_label_without_parent_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-model.bin");
        fs::write(&path, b"").unwrap();

        let err = inspect_model_source(&path).unwrap_err().to_string();

        assert!(err.contains("not-a-model.bin"));
        assert_public_blocker_message_without_local_path(&err, dir.path());
    }

    fn write_llama_config(root: &Path) {
        write_llama_config_with_overrides(root, &[]);
    }

    fn write_llama_config_with_overrides(root: &Path, overrides: &[(&str, Value)]) {
        let mut config = serde_json::Map::from_iter([
            ("model_type".to_string(), serde_json::json!("llama")),
            (
                "architectures".to_string(),
                serde_json::json!(["LlamaForCausalLM"]),
            ),
            ("hidden_size".to_string(), serde_json::json!(16)),
            ("num_hidden_layers".to_string(), serde_json::json!(2)),
            ("intermediate_size".to_string(), serde_json::json!(64)),
            ("num_attention_heads".to_string(), serde_json::json!(4)),
            ("num_key_value_heads".to_string(), serde_json::json!(2)),
            (
                "max_position_embeddings".to_string(),
                serde_json::json!(128),
            ),
            ("rms_norm_eps".to_string(), serde_json::json!(0.00001)),
            ("rope_theta".to_string(), serde_json::json!(10000.0)),
            ("vocab_size".to_string(), serde_json::json!(32000)),
            ("tie_word_embeddings".to_string(), serde_json::json!(false)),
        ]);
        for (key, value) in overrides {
            config.insert((*key).to_string(), value.clone());
        }
        fs::write(
            root.join("config.json"),
            serde_json::to_vec(&Value::Object(config)).unwrap(),
        )
        .unwrap();
    }

    fn write_safetensors_file(root: &Path, name: &str, tensors: &[(&str, &str, &[u64])]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        let mut payload = Vec::new();
        for (tensor_name, dtype, shape) in tensors {
            let byte_len = tensor_byte_len(dtype, shape);
            header.insert(
                (*tensor_name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + byte_len],
                }),
            );
            payload.resize(payload.len() + usize::try_from(byte_len).unwrap(), 0);
            offset += byte_len;
        }
        write_safetensors_bytes(root, name, &Value::Object(header), &payload);
    }

    fn write_safetensors_bytes(root: &Path, name: &str, header: &Value, payload: &[u8]) {
        let header = serde_json::to_vec(header).unwrap();
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(payload);
        fs::write(root.join(name), file).unwrap();
    }

    fn tensor_byte_len(dtype: &str, shape: &[u64]) -> u64 {
        let element_size = match dtype {
            "F16" | "BF16" => 2,
            "F32" | "I32" => 4,
            "I64" => 8,
            other => panic!("test fixture does not know dtype {other}"),
        };
        shape.iter().product::<u64>() * element_size
    }

    fn assert_blocker_codes(inspection: &ModelSourceInspection, expected: &[&str]) {
        let actual = inspection
            .readiness
            .blockers
            .iter()
            .map(|blocker| blocker.code)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    fn assert_public_blocker_message_without_local_path(message: &str, root: &Path) {
        let root = root.display().to_string();
        assert!(
            !message.contains(&root),
            "blocker message leaked local path {root:?}: {message}"
        );
        assert!(
            !message.contains("/var/") && !message.contains("/private/"),
            "blocker message leaked a private temp path: {message}"
        );
    }
}
