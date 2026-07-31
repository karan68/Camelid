use std::{fs, path::Path};

use camelid::{
    error::BackendError,
    gguf::read_metadata,
    inference::LlamaKvCachePlan,
    model::{
        is_runnable_only_arch, LlamaAttentionTensors, LlamaFfnTensors, LlamaModelConfig,
        LlamaMoeExpertTensors, LlamaTensorBinding,
    },
};

#[test]
fn extracts_dense_llama_model_config() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("llama.gguf");
    write_llama_gguf(&path, true);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();

    assert_eq!(config.context_length, 128);
    assert_eq!(config.embedding_length, 8);
    assert_eq!(config.block_count, 1);
    assert_eq!(config.feed_forward_length, 16);
    assert_eq!(config.attention_head_count, 2);
    assert_eq!(config.attention_head_count_kv, 1);
    assert_eq!(config.rope_dimension_count, Some(4));
    assert_eq!(config.rope_freq_base, Some(10_000.0));
    assert_eq!(config.rms_norm_epsilon, 1e-6);
    assert_eq!(config.vocab_size, Some(4));
    assert_eq!(config.file_type, Some(0));
}

#[test]
fn defaults_missing_kv_heads_to_attention_heads_for_tinyllama_style_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tinyllama-no-kv-heads.gguf");
    write_llama_gguf_without_kv_head_metadata(&path);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();
    let cache_plan = LlamaKvCachePlan::from_config(&config).unwrap();

    assert_eq!(config.attention_head_count, 2);
    assert_eq!(config.attention_head_count_kv, config.attention_head_count);
    assert_eq!(config.rope_dimension_count, Some(4));
    assert_eq!(config.rope_freq_base, Some(10_000.0));
    assert_eq!(
        binding.layers[0].attention_k().unwrap().dimensions,
        vec![8, 8]
    );
    assert_eq!(
        binding.layers[0].attention_v().unwrap().dimensions,
        vec![8, 8]
    );
    assert_eq!(cache_plan.kv_head_count, 2);
    assert_eq!(cache_plan.head_dim, 4);
    assert_eq!(cache_plan.key_shape, vec![1, 128, 2, 4]);
}

#[test]
fn accepts_mistral_metadata_on_llama_dense_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mistral.gguf");
    write_mistral_gguf(&path);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();
    let cache_plan = LlamaKvCachePlan::from_config(&config).unwrap();

    assert_eq!(gguf.architecture(), Some("mistral"));
    assert_eq!(config.context_length, 128);
    assert_eq!(config.embedding_length, 8);
    assert_eq!(config.attention_head_count, 2);
    assert_eq!(config.attention_head_count_kv, 1);
    assert_eq!(config.rope_dimension_count, Some(4));
    assert_eq!(config.rope_freq_base, Some(10_000.0));
    assert_eq!(config.rms_norm_epsilon, 1e-6);
    assert_eq!(
        binding.layers[0].attention_q().unwrap().name,
        "blk.0.attn_q.weight"
    );
    assert_eq!(
        binding.layers[0].attention_k().unwrap().dimensions,
        vec![8, 4]
    );
    assert_eq!(cache_plan.kv_head_count, 1);
    assert_eq!(cache_plan.head_dim, 4);
}

#[test]
fn binds_mixtral_moe_metadata_and_expert_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixtral-moe.gguf");
    write_mixtral_moe_gguf(&path);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();
    let moe = config.moe.as_ref().unwrap();

    assert_eq!(moe.family_label, "Mixtral");
    assert_eq!(moe.expert_count, 8);
    assert_eq!(moe.expert_used_count, 2);
    match &binding.layers[0].ffn {
        LlamaFfnTensors::MoE {
            router,
            gate_experts,
            up_experts,
            down_experts,
        } => {
            assert_eq!(router.name, "blk.0.ffn_gate_inp.weight");
            assert!(
                matches!(gate_experts, LlamaMoeExpertTensors::Merged(desc) if desc.name == "blk.0.ffn_gate_exps.weight")
            );
            assert!(
                matches!(up_experts, LlamaMoeExpertTensors::Merged(desc) if desc.name == "blk.0.ffn_up_exps.weight")
            );
            assert!(
                matches!(down_experts, LlamaMoeExpertTensors::Merged(desc) if desc.name == "blk.0.ffn_down_exps.weight")
            );
        }
        LlamaFfnTensors::Dense { .. } | LlamaFfnTensors::DeepSeekMoE { .. } => {
            panic!("expected Mixtral MoE FFN tensors")
        }
    }
}

#[test]
fn accepts_llama3_style_gqa_metadata_and_rope_theta() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("llama3-gqa.gguf");
    write_scaled_llama3_style_gqa_gguf(&path);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();
    let cache_plan = LlamaKvCachePlan::from_config(&config).unwrap();

    assert_eq!(config.context_length, 8192);
    assert_eq!(config.embedding_length, 32);
    assert_eq!(config.attention_head_count, 8);
    assert_eq!(config.attention_head_count_kv, 2);
    assert_eq!(config.rope_dimension_count, Some(4));
    assert_eq!(config.rope_freq_base, Some(500_000.0));
    assert_eq!(config.rope_scaling_type.as_deref(), Some("llama3"));
    assert_eq!(config.rope_scaling_factor, Some(32.0));
    assert_eq!(config.rope_scaling_original_context_length, Some(8192));
    assert_eq!(config.rope_scaling_low_freq_factor, Some(1.0));
    assert_eq!(config.rope_scaling_high_freq_factor, Some(4.0));
    assert_eq!(
        binding.rope_freqs.as_ref().unwrap().name,
        "rope_freqs.weight"
    );
    assert_eq!(binding.rope_freqs.as_ref().unwrap().dimensions, vec![2]);
    assert_eq!(
        binding.layers[0].attention_q().unwrap().dimensions,
        vec![32, 32]
    );
    assert_eq!(
        binding.layers[0].attention_k().unwrap().dimensions,
        vec![32, 8]
    );
    assert_eq!(
        binding.layers[0].attention_v().unwrap().dimensions,
        vec![32, 8]
    );
    assert_eq!(cache_plan.kv_head_count, 2);
    assert_eq!(cache_plan.head_dim, 4);
    assert_eq!(cache_plan.key_shape, vec![1, 8192, 2, 4]);
}

#[test]
fn infers_vocab_size_from_real_gguf_token_embedding_shape_when_metadata_is_absent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("llama.gguf");
    write_llama_gguf_without_vocab_metadata(&path);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();

    assert_eq!(config.vocab_size, Some(4));
    assert_eq!(binding.token_embedding.name, "token_embd.weight");
}

#[test]
fn binds_required_llama_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("llama.gguf");
    write_llama_gguf(&path, true);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();

    assert_eq!(binding.token_embedding.name, "token_embd.weight");
    assert_eq!(binding.output_norm.name, "output_norm.weight");
    assert_eq!(binding.output.name, "output.weight");
    assert!(!binding.output_is_tied_embedding);
    assert_eq!(binding.layers.len(), 1);
    assert_eq!(
        binding.layers[0].attention_q().unwrap().name,
        "blk.0.attn_q.weight"
    );
    match &binding.layers[0].ffn {
        LlamaFfnTensors::Dense { down, .. } => assert_eq!(down.name, "blk.0.ffn_down.weight"),
        LlamaFfnTensors::MoE { .. } | LlamaFfnTensors::DeepSeekMoE { .. } => {
            panic!("expected dense FFN tensors")
        }
    }
}

#[test]
fn falls_back_to_tied_output_embedding_when_output_weight_is_absent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("llama.gguf");
    write_llama_gguf(&path, false);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();

    assert!(binding.output_is_tied_embedding);
    assert_eq!(binding.output.name, "token_embd.weight");
}

#[test]
fn reports_missing_required_tensor_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("llama.gguf");
    write_llama_gguf_missing_attention_q(&path);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let err = LlamaTensorBinding::bind(&gguf, &config)
        .unwrap_err()
        .to_string();

    assert!(err.contains("blk.0.attn_q.weight"));
}

#[test]
fn rejects_descriptor_shape_that_cannot_feed_dense_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("llama.gguf");
    write_llama_gguf_with_bad_attention_k_shape(&path);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let err = LlamaTensorBinding::bind(&gguf, &config)
        .unwrap_err()
        .to_string();

    assert!(err.contains("attention k"));
    assert!(err.contains("blk.0.attn_k.weight"));
}

#[test]
fn rejects_dense_config_when_attention_heads_do_not_divide_embedding() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("llama.gguf");
    write_llama_gguf(&path, true);

    let gguf = read_metadata(&path).unwrap();
    let mut config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    config.attention_head_count = 3;
    let err = LlamaTensorBinding::bind(&gguf, &config)
        .unwrap_err()
        .to_string();

    assert!(err.contains("embedding length 8"));
    assert!(err.contains("attention head count 3"));
}

#[test]
fn rejects_dense_config_when_vocab_size_is_missing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("llama.gguf");
    write_llama_gguf(&path, true);

    let gguf = read_metadata(&path).unwrap();
    let mut config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    config.vocab_size = None;
    let err = LlamaTensorBinding::bind(&gguf, &config)
        .unwrap_err()
        .to_string();

    assert!(err.contains("llama.vocab_size"));
    assert!(err.contains("dense tensor validation"));
}

fn write_llama_gguf(path: &Path, include_output: bool) {
    write_llama_gguf_with_skip(path, include_output, None);
}

fn write_llama_gguf_without_vocab_metadata(path: &Path) {
    write_llama_gguf_with_options(path, true, None, None, false, true, true);
}

fn write_llama_gguf_without_kv_head_metadata(path: &Path) {
    write_llama_gguf_with_options(path, true, None, None, true, false, false);
}

fn write_llama_gguf_missing_attention_q(path: &Path) {
    write_llama_gguf_with_skip(path, true, Some("blk.0.attn_q.weight"));
}

fn write_llama_gguf_with_bad_attention_k_shape(path: &Path) {
    write_llama_gguf_with_shape_override(
        path,
        true,
        None,
        Some(("blk.0.attn_k.weight", vec![8, 3])),
    );
}

fn write_mistral_gguf(path: &Path) {
    write_architecture_prefixed_gguf(path, "mistral", 128, 8, 16, 2, Some(1), 4, 10_000.0, 1e-6);
}

fn write_mixtral_moe_gguf(path: &Path) {
    let tensors: Vec<(&str, Vec<i64>)> = vec![
        ("token_embd.weight", vec![8, 4]),
        ("output_norm.weight", vec![8]),
        ("blk.0.attn_norm.weight", vec![8]),
        ("blk.0.attn_q.weight", vec![8, 8]),
        ("blk.0.attn_k.weight", vec![8, 4]),
        ("blk.0.attn_v.weight", vec![8, 4]),
        ("blk.0.attn_output.weight", vec![8, 8]),
        ("blk.0.ffn_norm.weight", vec![8]),
        ("blk.0.ffn_gate_inp.weight", vec![8, 8]),
        ("blk.0.ffn_gate_exps.weight", vec![8, 16, 8]),
        ("blk.0.ffn_up_exps.weight", vec![8, 16, 8]),
        ("blk.0.ffn_down_exps.weight", vec![16, 8, 8]),
        ("output.weight", vec![8, 4]),
    ];

    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_i64(&mut b, tensors.len() as i64);
    push_i64(&mut b, 15);

    push_kv_string(&mut b, "general.architecture", "llama");
    push_kv_string(&mut b, "general.name", "Mixtral 8x7B Instruct v0.1");
    push_kv_string(&mut b, "general.basename", "Mixtral");
    push_kv_u32(&mut b, "general.file_type", 7);
    push_kv_u32(&mut b, "llama.context_length", 128);
    push_kv_u32(&mut b, "llama.embedding_length", 8);
    push_kv_u32(&mut b, "llama.block_count", 1);
    push_kv_u32(&mut b, "llama.feed_forward_length", 16);
    push_kv_u32(&mut b, "llama.attention.head_count", 2);
    push_kv_u32(&mut b, "llama.attention.head_count_kv", 1);
    push_kv_u32(&mut b, "llama.rope.dimension_count", 4);
    push_kv_f32(&mut b, "llama.rope.freq_base", 10_000.0);
    push_kv_f32(&mut b, "llama.attention.layer_norm_rms_epsilon", 1e-6);
    push_kv_u32(&mut b, "llama.expert_count", 8);
    push_kv_u32(&mut b, "llama.expert_used_count", 2);

    let mut relative_offset = 0u64;
    for (name, dims) in &tensors {
        push_string(&mut b, name);
        push_u32(&mut b, dims.len() as u32);
        for dim in dims {
            push_i64(&mut b, *dim);
        }
        push_i32(&mut b, 0); // f32
        push_u64(&mut b, relative_offset);
        relative_offset += dims.iter().product::<i64>() as u64 * 4;
    }

    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    b.extend(vec![0u8; relative_offset as usize]);
    fs::write(path, b).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn write_architecture_prefixed_gguf(
    path: &Path,
    architecture: &str,
    context_length: u32,
    embedding_length: u32,
    feed_forward_length: u32,
    attention_head_count: u32,
    attention_head_count_kv: Option<u32>,
    rope_dimension_count: u32,
    rope_freq_base: f32,
    rms_norm_epsilon: f32,
) {
    let kv_width = (embedding_length as usize
        * attention_head_count_kv.unwrap_or(attention_head_count) as usize)
        / attention_head_count as usize;
    let tensors: Vec<(&str, Vec<i64>)> = vec![
        ("token_embd.weight", vec![4, embedding_length as i64]),
        ("output_norm.weight", vec![embedding_length as i64]),
        ("blk.0.attn_norm.weight", vec![embedding_length as i64]),
        (
            "blk.0.attn_q.weight",
            vec![embedding_length as i64, embedding_length as i64],
        ),
        (
            "blk.0.attn_k.weight",
            vec![embedding_length as i64, kv_width as i64],
        ),
        (
            "blk.0.attn_v.weight",
            vec![embedding_length as i64, kv_width as i64],
        ),
        (
            "blk.0.attn_output.weight",
            vec![embedding_length as i64, embedding_length as i64],
        ),
        ("blk.0.ffn_norm.weight", vec![embedding_length as i64]),
        (
            "blk.0.ffn_gate.weight",
            vec![embedding_length as i64, feed_forward_length as i64],
        ),
        (
            "blk.0.ffn_up.weight",
            vec![embedding_length as i64, feed_forward_length as i64],
        ),
        (
            "blk.0.ffn_down.weight",
            vec![feed_forward_length as i64, embedding_length as i64],
        ),
        ("output.weight", vec![embedding_length as i64, 4]),
    ];

    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_i64(&mut b, tensors.len() as i64);
    push_i64(&mut b, 11 + i64::from(attention_head_count_kv.is_some()));

    push_kv_string(&mut b, "general.architecture", architecture);
    push_kv_u32(&mut b, "general.file_type", 0);
    push_kv_u32(
        &mut b,
        &format!("{architecture}.context_length"),
        context_length,
    );
    push_kv_u32(
        &mut b,
        &format!("{architecture}.embedding_length"),
        embedding_length,
    );
    push_kv_u32(&mut b, &format!("{architecture}.block_count"), 1);
    push_kv_u32(
        &mut b,
        &format!("{architecture}.feed_forward_length"),
        feed_forward_length,
    );
    push_kv_u32(
        &mut b,
        &format!("{architecture}.attention.head_count"),
        attention_head_count,
    );
    if let Some(kv_heads) = attention_head_count_kv {
        push_kv_u32(
            &mut b,
            &format!("{architecture}.attention.head_count_kv"),
            kv_heads,
        );
    }
    push_kv_u32(
        &mut b,
        &format!("{architecture}.rope.dimension_count"),
        rope_dimension_count,
    );
    push_kv_f32(
        &mut b,
        &format!("{architecture}.rope.freq_base"),
        rope_freq_base,
    );
    push_kv_f32(
        &mut b,
        &format!("{architecture}.attention.layer_norm_rms_epsilon"),
        rms_norm_epsilon,
    );
    push_kv_u32(&mut b, &format!("{architecture}.vocab_size"), 4);

    let mut relative_offset = 0u64;
    for (name, dims) in &tensors {
        push_string(&mut b, name);
        push_u32(&mut b, dims.len() as u32);
        for dim in dims {
            push_i64(&mut b, *dim);
        }
        push_i32(&mut b, 0);
        push_u64(&mut b, relative_offset);
        relative_offset += dims.iter().product::<i64>() as u64 * 4;
    }

    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    b.extend(vec![0u8; relative_offset as usize]);
    fs::write(path, b).unwrap();
}

fn write_scaled_llama3_style_gqa_gguf(path: &Path) {
    let tensors: Vec<(&str, Vec<i64>)> = vec![
        ("token_embd.weight", vec![4, 32]),
        ("rope_freqs.weight", vec![2]),
        ("output_norm.weight", vec![32]),
        ("blk.0.attn_norm.weight", vec![32]),
        ("blk.0.attn_q.weight", vec![32, 32]),
        ("blk.0.attn_k.weight", vec![32, 8]),
        ("blk.0.attn_v.weight", vec![32, 8]),
        ("blk.0.attn_output.weight", vec![32, 32]),
        ("blk.0.ffn_norm.weight", vec![32]),
        ("blk.0.ffn_gate.weight", vec![32, 64]),
        ("blk.0.ffn_up.weight", vec![32, 64]),
        ("blk.0.ffn_down.weight", vec![64, 32]),
        ("output.weight", vec![32, 4]),
    ];

    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_i64(&mut b, tensors.len() as i64);
    push_i64(&mut b, 17);

    push_kv_string(&mut b, "general.architecture", "llama");
    push_kv_u32(&mut b, "general.file_type", 0);
    push_kv_u32(&mut b, "llama.context_length", 8192);
    push_kv_u32(&mut b, "llama.embedding_length", 32);
    push_kv_u32(&mut b, "llama.block_count", 1);
    push_kv_u32(&mut b, "llama.feed_forward_length", 64);
    push_kv_u32(&mut b, "llama.attention.head_count", 8);
    push_kv_u32(&mut b, "llama.attention.head_count_kv", 2);
    push_kv_u32(&mut b, "llama.rope.dimension_count", 4);
    push_kv_f32(&mut b, "llama.rope.freq_base", 500_000.0);
    push_kv_string(&mut b, "llama.rope.scaling.type", "llama3");
    push_kv_f32(&mut b, "llama.rope.scaling.factor", 32.0);
    push_kv_u32(&mut b, "llama.rope.scaling.original_context_length", 8192);
    push_kv_f32(&mut b, "llama.rope.scaling.low_freq_factor", 1.0);
    push_kv_f32(&mut b, "llama.rope.scaling.high_freq_factor", 4.0);
    push_kv_f32(&mut b, "llama.attention.layer_norm_rms_epsilon", 1e-5);
    push_kv_u32(&mut b, "llama.vocab_size", 4);

    let mut relative_offset = 0u64;
    for (name, dims) in &tensors {
        push_string(&mut b, name);
        push_u32(&mut b, dims.len() as u32);
        for dim in dims {
            push_i64(&mut b, *dim);
        }
        push_i32(&mut b, 0); // f32
        push_u64(&mut b, relative_offset);
        relative_offset += dims.iter().product::<i64>() as u64 * 4;
        while !relative_offset.is_multiple_of(32) {
            relative_offset += 1;
        }
    }

    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    b.extend(vec![0u8; relative_offset as usize]);
    fs::write(path, b).unwrap();
}

fn write_llama_gguf_with_skip(path: &Path, include_output: bool, skip: Option<&str>) {
    write_llama_gguf_with_shape_override(path, include_output, skip, None);
}

fn write_llama_gguf_with_shape_override(
    path: &Path,
    include_output: bool,
    skip: Option<&str>,
    shape_override: Option<(&str, Vec<i64>)>,
) {
    write_llama_gguf_with_options(
        path,
        include_output,
        skip,
        shape_override,
        true,
        false,
        true,
    );
}

fn write_llama_gguf_with_options(
    path: &Path,
    include_output: bool,
    skip: Option<&str>,
    shape_override: Option<(&str, Vec<i64>)>,
    include_vocab_metadata: bool,
    real_gguf_embedding_order: bool,
    include_kv_head_metadata: bool,
) {
    let token_embedding_shape = if real_gguf_embedding_order {
        vec![8, 4]
    } else {
        vec![4, 8]
    };
    let mut tensors: Vec<(&str, Vec<i64>)> = vec![
        ("token_embd.weight", token_embedding_shape),
        ("output_norm.weight", vec![8]),
        ("blk.0.attn_norm.weight", vec![8]),
        ("blk.0.attn_q.weight", vec![8, 8]),
        (
            "blk.0.attn_k.weight",
            if include_kv_head_metadata {
                vec![8, 4]
            } else {
                vec![8, 8]
            },
        ),
        (
            "blk.0.attn_v.weight",
            if include_kv_head_metadata {
                vec![8, 4]
            } else {
                vec![8, 8]
            },
        ),
        ("blk.0.attn_output.weight", vec![8, 8]),
        ("blk.0.ffn_norm.weight", vec![8]),
        ("blk.0.ffn_gate.weight", vec![8, 16]),
        ("blk.0.ffn_up.weight", vec![8, 16]),
        ("blk.0.ffn_down.weight", vec![16, 8]),
    ];
    if include_output {
        tensors.push(("output.weight", vec![8, 4]));
    }
    if let Some((override_name, override_dims)) = shape_override {
        for (name, dims) in &mut tensors {
            if *name == override_name {
                *dims = override_dims.clone();
            }
        }
    }
    tensors.retain(|(name, _)| Some(*name) != skip);

    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_i64(&mut b, tensors.len() as i64);
    let metadata_count =
        10 + u64::from(include_kv_head_metadata) + u64::from(include_vocab_metadata);
    push_i64(&mut b, metadata_count as i64);

    push_kv_string(&mut b, "general.architecture", "llama");
    push_kv_u32(&mut b, "general.file_type", 0);
    push_kv_u32(&mut b, "llama.context_length", 128);
    push_kv_u32(&mut b, "llama.embedding_length", 8);
    push_kv_u32(&mut b, "llama.block_count", 1);
    push_kv_u32(&mut b, "llama.feed_forward_length", 16);
    push_kv_u32(&mut b, "llama.attention.head_count", 2);
    if include_kv_head_metadata {
        push_kv_u32(&mut b, "llama.attention.head_count_kv", 1);
    }
    push_kv_u32(&mut b, "llama.rope.dimension_count", 4);
    push_kv_f32(&mut b, "llama.rope.freq_base", 10_000.0);
    push_kv_f32(&mut b, "llama.attention.layer_norm_rms_epsilon", 1e-6);
    if include_vocab_metadata {
        push_kv_u32(&mut b, "llama.vocab_size", 4);
    }

    let mut relative_offset = 0u64;
    for (name, dims) in &tensors {
        push_string(&mut b, name);
        push_u32(&mut b, dims.len() as u32);
        for dim in dims {
            push_i64(&mut b, *dim);
        }
        push_i32(&mut b, 0); // f32
        push_u64(&mut b, relative_offset);
        relative_offset += dims.iter().product::<i64>() as u64 * 4;
    }

    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    b.extend(vec![0u8; relative_offset as usize]);
    fs::write(path, b).unwrap();
}

fn push_kv_string(b: &mut Vec<u8>, key: &str, value: &str) {
    push_string(b, key);
    push_i32(b, 8);
    push_string(b, value);
}

fn push_kv_u32(b: &mut Vec<u8>, key: &str, value: u32) {
    push_string(b, key);
    push_i32(b, 4);
    push_u32(b, value);
}

fn push_kv_f32(b: &mut Vec<u8>, key: &str, value: f32) {
    push_string(b, key);
    push_i32(b, 6);
    b.extend_from_slice(&value.to_le_bytes());
}

fn push_string(b: &mut Vec<u8>, value: &str) {
    push_u64(b, value.len() as u64);
    b.extend_from_slice(value.as_bytes());
}

fn push_u32(b: &mut Vec<u8>, value: u32) {
    b.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(b: &mut Vec<u8>, value: i32) {
    b.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(b: &mut Vec<u8>, value: u64) {
    b.extend_from_slice(&value.to_le_bytes());
}

fn push_i64(b: &mut Vec<u8>, value: i64) {
    b.extend_from_slice(&value.to_le_bytes());
}

// ---------------------------------------------------------------------------
// Qwen3 QK-norm binder coverage (feat/qwen3-support, Gate 1).
//
// Qwen3 applies a per-head RMSNorm to Q and K after the projections and before
// RoPE (`attn_q_norm`/`attn_k_norm`, shape `[head_dim]`). The dense binder must
// carry them for qwen3 and fail closed both directions: qwen3 missing them, and
// a plain Llama-family row unexpectedly carrying them.
// ---------------------------------------------------------------------------

/// Build a tiny qwen3 GGUF (1 block, embedding 16, 2 heads, 1 KV head,
/// head_dim 8, tied embeddings). `include_qk_norm` toggles the
/// `attn_q_norm`/`attn_k_norm` tensors. head_dim is 8 so the F32 norm tensors
/// (32 bytes) honor the GGUF 32-byte data alignment.
fn write_qwen3_gguf(path: &Path, include_qk_norm: bool) {
    write_attention_fixture_gguf(path, "qwen3", include_qk_norm, false);
}

fn write_attention_fixture_gguf(
    path: &Path,
    architecture: &str,
    include_qk_norm: bool,
    include_qkv_biases: bool,
) {
    let mut tensors: Vec<(&str, Vec<i64>)> = vec![
        ("token_embd.weight", vec![4, 16]),
        ("output_norm.weight", vec![16]),
        ("blk.0.attn_norm.weight", vec![16]),
        ("blk.0.attn_q.weight", vec![16, 16]),
        ("blk.0.attn_k.weight", vec![16, 8]),
        ("blk.0.attn_v.weight", vec![16, 8]),
        ("blk.0.attn_output.weight", vec![16, 16]),
        ("blk.0.ffn_norm.weight", vec![16]),
        ("blk.0.ffn_gate.weight", vec![16, 32]),
        ("blk.0.ffn_up.weight", vec![16, 32]),
        ("blk.0.ffn_down.weight", vec![32, 16]),
    ];
    if include_qk_norm {
        // head_dim = embedding_length / head_count = 16 / 2 = 8.
        tensors.push(("blk.0.attn_q_norm.weight", vec![8]));
        tensors.push(("blk.0.attn_k_norm.weight", vec![8]));
    }
    if include_qkv_biases {
        tensors.push(("blk.0.attn_q.bias", vec![16]));
        tensors.push(("blk.0.attn_k.bias", vec![8]));
        tensors.push(("blk.0.attn_v.bias", vec![8]));
    }

    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_i64(&mut b, tensors.len() as i64);
    push_i64(&mut b, 14);

    push_kv_string(&mut b, "general.architecture", architecture);
    push_kv_string(&mut b, "general.name", "QK Norm Fixture");
    push_kv_u32(&mut b, "general.file_type", 0);
    push_kv_u32(&mut b, &format!("{architecture}.context_length"), 128);
    push_kv_u32(&mut b, &format!("{architecture}.embedding_length"), 16);
    push_kv_u32(&mut b, &format!("{architecture}.block_count"), 1);
    push_kv_u32(&mut b, &format!("{architecture}.feed_forward_length"), 32);
    push_kv_u32(&mut b, &format!("{architecture}.attention.head_count"), 2);
    push_kv_u32(
        &mut b,
        &format!("{architecture}.attention.head_count_kv"),
        1,
    );
    push_kv_u32(&mut b, &format!("{architecture}.attention.key_length"), 8);
    push_kv_u32(&mut b, &format!("{architecture}.attention.value_length"), 8);
    push_kv_f32(
        &mut b,
        &format!("{architecture}.rope.freq_base"),
        1_000_000.0,
    );
    push_kv_f32(
        &mut b,
        &format!("{architecture}.attention.layer_norm_rms_epsilon"),
        1e-6,
    );
    push_kv_u32(&mut b, &format!("{architecture}.vocab_size"), 4);

    let mut relative_offset = 0u64;
    for (name, dims) in &tensors {
        push_string(&mut b, name);
        push_u32(&mut b, dims.len() as u32);
        for dim in dims {
            push_i64(&mut b, *dim);
        }
        push_i32(&mut b, 0);
        push_u64(&mut b, relative_offset);
        relative_offset += dims.iter().product::<i64>() as u64 * 4;
    }

    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    b.extend(vec![0u8; relative_offset as usize]);
    fs::write(path, b).unwrap();
}

#[test]
fn qwen3_binds_per_head_qk_norm_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("qwen3.gguf");
    write_qwen3_gguf(&path, true);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();

    let layer = &binding.layers[0];
    let q_norm = layer
        .attention_q_norm()
        .expect("qwen3 must bind attn_q_norm");
    let k_norm = layer
        .attention_k_norm()
        .expect("qwen3 must bind attn_k_norm");
    // Shape is [head_dim] = [8].
    assert_eq!(q_norm.dimensions, vec![8]);
    assert_eq!(k_norm.dimensions, vec![8]);
}

#[test]
fn qwen3_without_qk_norm_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("qwen3-no-qknorm.gguf");
    write_qwen3_gguf(&path, false);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let err = LlamaTensorBinding::bind(&gguf, &config)
        .expect_err("qwen3 missing attn_q_norm/attn_k_norm must fail closed");
    let msg = format!("{err}");
    assert!(
        msg.contains("QK-norm") && msg.to_lowercase().contains("qwen3"),
        "unexpected error: {msg}"
    );
}

#[test]
fn llama_family_with_unexpected_qk_norm_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("llama-with-qknorm.gguf");
    // A plain llama row that unexpectedly carries QK-norm tensors must not be
    // silently accepted (the Llama forward path would drop them).
    write_attention_fixture_gguf(&path, "llama", true, false);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let err = LlamaTensorBinding::bind(&gguf, &config)
        .expect_err("llama row carrying QK-norm tensors must fail closed");
    assert!(
        format!("{err}").contains("QK-norm"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen2_binds_required_qkv_projection_biases() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("qwen2-with-biases.gguf");
    write_attention_fixture_gguf(&path, "qwen2", false, true);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();

    let LlamaAttentionTensors::Standard {
        biases: Some(biases),
        ..
    } = &binding.layers[0].attention
    else {
        panic!("qwen2 must bind its Q/K/V projection biases");
    };
    assert_eq!(biases.q.dimensions, vec![16]);
    assert_eq!(biases.k.dimensions, vec![8]);
    assert_eq!(biases.v.dimensions, vec![8]);
}

#[test]
fn qwen2_without_qkv_projection_biases_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("qwen2-without-biases.gguf");
    write_attention_fixture_gguf(&path, "qwen2", false, false);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let err = LlamaTensorBinding::bind(&gguf, &config)
        .expect_err("qwen2 missing Q/K/V projection biases must fail closed");
    let message = format!("{err}");
    assert!(
        message.contains("qwen2") && message.contains("bias"),
        "unexpected error: {message}"
    );
}

// ---------------------------------------------------------------------------
// Qwen3 explicit head_dim (feat/qwen3-family-headdim): sizes where
// head_dim != embedding_length/head_count (0.6B/4B/32B). q_width = heads*head_dim
// is then WIDER than embedding_length, so the binder/dims must source head_dim
// from attention.key_length, not embedding/heads.
// ---------------------------------------------------------------------------

/// embed 16, 2 q heads, 1 kv head, key_length 16 → head_dim 16 (NOT 16/2=8),
/// q_width = 2*16 = 32 (> embedding), kv_width = 1*16 = 16. Tied embeddings.
fn write_qwen3_explicit_head_dim_gguf(path: &Path) {
    let tensors: Vec<(&str, Vec<i64>)> = vec![
        ("token_embd.weight", vec![4, 16]),
        ("output_norm.weight", vec![16]),
        ("blk.0.attn_norm.weight", vec![16]),
        ("blk.0.attn_q.weight", vec![16, 32]),
        ("blk.0.attn_q_norm.weight", vec![16]),
        ("blk.0.attn_k.weight", vec![16, 16]),
        ("blk.0.attn_k_norm.weight", vec![16]),
        ("blk.0.attn_v.weight", vec![16, 16]),
        ("blk.0.attn_output.weight", vec![32, 16]),
        ("blk.0.ffn_norm.weight", vec![16]),
        ("blk.0.ffn_gate.weight", vec![16, 32]),
        ("blk.0.ffn_up.weight", vec![16, 32]),
        ("blk.0.ffn_down.weight", vec![32, 16]),
    ];
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_i64(&mut b, tensors.len() as i64);
    push_i64(&mut b, 14);
    push_kv_string(&mut b, "general.architecture", "qwen3");
    push_kv_string(&mut b, "general.name", "Qwen3 Explicit HeadDim Fixture");
    push_kv_u32(&mut b, "general.file_type", 0);
    push_kv_u32(&mut b, "qwen3.context_length", 128);
    push_kv_u32(&mut b, "qwen3.embedding_length", 16);
    push_kv_u32(&mut b, "qwen3.block_count", 1);
    push_kv_u32(&mut b, "qwen3.feed_forward_length", 32);
    push_kv_u32(&mut b, "qwen3.attention.head_count", 2);
    push_kv_u32(&mut b, "qwen3.attention.head_count_kv", 1);
    push_kv_u32(&mut b, "qwen3.attention.key_length", 16);
    push_kv_u32(&mut b, "qwen3.attention.value_length", 16);
    push_kv_f32(&mut b, "qwen3.rope.freq_base", 1_000_000.0);
    push_kv_f32(&mut b, "qwen3.attention.layer_norm_rms_epsilon", 1e-6);
    push_kv_u32(&mut b, "qwen3.vocab_size", 4);
    let mut relative_offset = 0u64;
    for (name, dims) in &tensors {
        push_string(&mut b, name);
        push_u32(&mut b, dims.len() as u32);
        for dim in dims {
            push_i64(&mut b, *dim);
        }
        push_i32(&mut b, 0);
        push_u64(&mut b, relative_offset);
        relative_offset += dims.iter().product::<i64>() as u64 * 4;
    }
    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    b.extend(vec![0u8; relative_offset as usize]);
    fs::write(path, b).unwrap();
}

#[test]
fn qwen3_explicit_head_dim_binds_wide_q_projection() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("qwen3-explicit-headdim.gguf");
    write_qwen3_explicit_head_dim_gguf(&path);

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    // head_dim comes from key_length (16), NOT embedding/heads (16/2=8).
    assert_eq!(config.attention_key_length, Some(16));
    // KV cache plan reflects the explicit head_dim.
    let cache_plan = LlamaKvCachePlan::from_config(&config).unwrap();
    assert_eq!(cache_plan.head_dim, 16);

    // Binder accepts the wide q projection [16, 32] (q_width 32 > embedding 16)
    // and the matching output [32, 16] — fails closed without explicit-head_dim plumbing.
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();
    assert_eq!(
        binding.layers[0].attention_q().unwrap().dimensions,
        vec![16, 32]
    );
    assert_eq!(binding.layers[0].attention_output.dimensions, vec![32, 16]);
    assert!(binding.layers[0].attention_q_norm().is_some());
}

// ---------------------------------------------------------------------------
// gemma3 full-norm-set binder coverage (feat/gemma3-metal-resident, Phase 1).
//
// gemma3 carries FOUR norm tensors per layer beyond the Llama pair: per-head
// `attn_q_norm`/`attn_k_norm` (shape `[head_dim]`, applied before RoPE) and the
// full-width sandwich norms `post_attention_norm`/`post_ffw_norm` (shape
// `[embedding_length]`, applied before each residual add). Before Phase 1 the
// dense binder classified gemma3 neither expects- nor forbids-QK-norm, so all
// of them silently bound `(None, None)` — the disclosed mis-binding behind the
// serve router's fail-closed divert. The binder must now carry all four and
// fail closed when any is missing, and `LlamaModelConfig::from_gguf` must
// parse the window/rope schedule metadata fail-closed (no silent defaults for
// required keys). Since the campaign's Phase 3b flip, the Metal-resident lane
// consumes these tensors on resident-capable hosts; everywhere else gemma3
// still routes to the runnable bridge (capability-aware predicate) and the
// CPU dense forward fails closed at forward dispatch (hazard H4).
// ---------------------------------------------------------------------------

struct Gemma3FixtureOptions {
    include_qk_norm: bool,
    include_post_norms: bool,
    /// Decoder depth: `gemma3.block_count` AND the per-block tensor set both
    /// follow this, so schedule-derivation tests can run `from_gguf` at the
    /// real row's 26-layer depth (not just the CI-default single block).
    block_count: u32,
    /// `Some(w)` writes `gemma3.attention.sliding_window = w` (including a
    /// fail-closed `0`); `None` omits the key entirely.
    sliding_window: Option<u32>,
    /// `Some(b)` writes `gemma3.rope.freq_base = b` (including fail-closed
    /// non-positive values); `None` omits the key entirely.
    global_base: Option<f32>,
    /// `Some` writes an explicit scalar `gemma3.attention.sliding_window_pattern`.
    pattern_key: Option<u32>,
    /// Writes the pattern key as an f32 (malformed: the key must be an integer).
    pattern_key_malformed: bool,
    /// `Some` writes an explicit `gemma3.rope.freq_base_swa`.
    local_base_key: Option<f32>,
    /// Writes `gemma3.rope.freq_base_swa` as a u32 (malformed: must be a float).
    local_base_key_malformed: bool,
}

impl Default for Gemma3FixtureOptions {
    fn default() -> Self {
        Self {
            include_qk_norm: true,
            include_post_norms: true,
            block_count: 1,
            sliding_window: Some(512),
            global_base: Some(1_000_000.0),
            pattern_key: None,
            pattern_key_malformed: false,
            local_base_key: None,
            local_base_key_malformed: false,
        }
    }
}

/// Build a tiny gemma3 GGUF (`block_count` blocks, embedding 16, 2 heads, 1 KV
/// head, head_dim 8, tied embeddings) mirroring the real row's key/tensor
/// layout: `attention.sliding_window` + `rope.freq_base` are the only
/// window/rope keys the real gemma-3-1b-it-Q8_0 file carries.
fn write_gemma3_gguf(path: &Path, options: &Gemma3FixtureOptions) {
    let mut tensors: Vec<(String, Vec<i64>)> = vec![
        ("token_embd.weight".to_string(), vec![4, 16]),
        ("output_norm.weight".to_string(), vec![16]),
    ];
    for blk in 0..options.block_count {
        let name = |t: &str| format!("blk.{blk}.{t}.weight");
        tensors.push((name("attn_norm"), vec![16]));
        tensors.push((name("attn_q"), vec![16, 16]));
        tensors.push((name("attn_k"), vec![16, 8]));
        tensors.push((name("attn_v"), vec![16, 8]));
        tensors.push((name("attn_output"), vec![16, 16]));
        tensors.push((name("ffn_norm"), vec![16]));
        tensors.push((name("ffn_gate"), vec![16, 32]));
        tensors.push((name("ffn_up"), vec![16, 32]));
        tensors.push((name("ffn_down"), vec![32, 16]));
        if options.include_qk_norm {
            // head_dim = attention.key_length = 8.
            tensors.push((name("attn_q_norm"), vec![8]));
            tensors.push((name("attn_k_norm"), vec![8]));
        }
        if options.include_post_norms {
            // Sandwich norms are full-width [embedding_length].
            tensors.push((name("post_attention_norm"), vec![16]));
            tensors.push((name("post_ffw_norm"), vec![16]));
        }
    }

    let mut metadata_count = 12i64;
    if options.sliding_window.is_some() {
        metadata_count += 1;
    }
    if options.global_base.is_some() {
        metadata_count += 1;
    }
    if options.pattern_key.is_some() || options.pattern_key_malformed {
        metadata_count += 1;
    }
    if options.local_base_key.is_some() || options.local_base_key_malformed {
        metadata_count += 1;
    }

    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_i64(&mut b, tensors.len() as i64);
    push_i64(&mut b, metadata_count);

    push_kv_string(&mut b, "general.architecture", "gemma3");
    push_kv_string(&mut b, "general.name", "Gemma3 Norm Fixture");
    push_kv_u32(&mut b, "general.file_type", 0);
    push_kv_u32(&mut b, "gemma3.context_length", 128);
    push_kv_u32(&mut b, "gemma3.embedding_length", 16);
    push_kv_u32(&mut b, "gemma3.block_count", options.block_count);
    push_kv_u32(&mut b, "gemma3.feed_forward_length", 32);
    push_kv_u32(&mut b, "gemma3.attention.head_count", 2);
    push_kv_u32(&mut b, "gemma3.attention.head_count_kv", 1);
    push_kv_u32(&mut b, "gemma3.attention.key_length", 8);
    push_kv_u32(&mut b, "gemma3.attention.value_length", 8);
    if let Some(base) = options.global_base {
        push_kv_f32(&mut b, "gemma3.rope.freq_base", base);
    }
    push_kv_f32(&mut b, "gemma3.attention.layer_norm_rms_epsilon", 1e-6);
    if let Some(window) = options.sliding_window {
        push_kv_u32(&mut b, "gemma3.attention.sliding_window", window);
    }
    if let Some(period) = options.pattern_key {
        push_kv_u32(&mut b, "gemma3.attention.sliding_window_pattern", period);
    } else if options.pattern_key_malformed {
        push_kv_f32(&mut b, "gemma3.attention.sliding_window_pattern", 6.0);
    }
    if let Some(base) = options.local_base_key {
        push_kv_f32(&mut b, "gemma3.rope.freq_base_swa", base);
    } else if options.local_base_key_malformed {
        push_kv_u32(&mut b, "gemma3.rope.freq_base_swa", 10_000);
    }

    let mut relative_offset = 0u64;
    for (name, dims) in &tensors {
        push_string(&mut b, name);
        push_u32(&mut b, dims.len() as u32);
        for dim in dims {
            push_i64(&mut b, *dim);
        }
        push_i32(&mut b, 0);
        push_u64(&mut b, relative_offset);
        relative_offset += dims.iter().product::<i64>() as u64 * 4;
    }
    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    b.extend(vec![0u8; relative_offset as usize]);
    fs::write(path, b).unwrap();

    // NOTE: no vocab_size key — from_gguf infers it from token_embd (4), like
    // the real row would if trimmed; keeps the fixture at 13 base keys.
}

#[test]
fn gemma3_binds_qk_and_sandwich_norms_with_window_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gemma3.gguf");
    write_gemma3_gguf(&path, &Gemma3FixtureOptions::default());

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();

    let layer = &binding.layers[0];
    // NAME-pinned bindings: the two QK norms share shape [8] and the two
    // sandwich norms share shape [16], so `.dimensions` alone cannot tell a
    // transposed `find_tensor` lookup from a correct one — asserting the bound
    // descriptor's `.name` on the specific field makes a swap fail loudly.
    let q_norm = layer
        .attention_q_norm()
        .expect("gemma3 must bind attn_q_norm");
    assert_eq!(q_norm.name, "blk.0.attn_q_norm.weight");
    assert_eq!(q_norm.dimensions, vec![8]);
    let k_norm = layer
        .attention_k_norm()
        .expect("gemma3 must bind attn_k_norm");
    assert_eq!(k_norm.name, "blk.0.attn_k_norm.weight");
    assert_eq!(k_norm.dimensions, vec![8]);
    let post_attn = layer
        .post_attention_norm
        .as_ref()
        .expect("gemma3 must bind post_attention_norm");
    assert_eq!(post_attn.name, "blk.0.post_attention_norm.weight");
    assert_eq!(post_attn.dimensions, vec![16]);
    let post_ffw = layer
        .post_ffw_norm
        .as_ref()
        .expect("gemma3 must bind post_ffw_norm");
    assert_eq!(post_ffw.name, "blk.0.post_ffw_norm.weight");
    assert_eq!(post_ffw.dimensions, vec![16]);

    // Window/rope metadata parses from the same keys the real row carries,
    // with the reference-pinned constants for the keys no conversion writes.
    let meta = config.gemma3.as_ref().expect("gemma3 metadata must parse");
    assert_eq!(meta.sliding_window, 512);
    assert_eq!(meta.sliding_window_pattern, 6);
    assert_eq!(meta.rope_freq_base_global, 1_000_000.0);
    assert_eq!(meta.rope_freq_base_local, 10_000.0);
    assert_eq!(meta.layer_is_sliding, vec![true]);
    assert_eq!(meta.embed_scale, 4.0); // sqrt(embedding_length 16)
    assert!(meta.ffn_geglu);
    assert!(meta.rope_neox_pairing);
    // The dense-path pairing flag stays false (the CPU dense forward is
    // fail-closed for gemma3; the resident lane forces pairing host-side from
    // the metadata flag above).
    assert!(!config.rope_neox_pairing);

    // CRITICAL invariant, updated for the Phase 3b flip: gemma3 is no longer
    // UNCONDITIONALLY runnable-only (on a resident-capable host the Metal
    // lane serves it), but on every host where the resident lane cannot serve
    // it must still classify for the runnable bridge — never the CPU dense
    // forward, whose forward dispatch fails closed for windowed archs (H4).
    assert!(!is_runnable_only_arch(&config.architecture));
    // Both capability legs false => bridge; Phase 3c added the quantization
    // leg (a non-Q8_0 gemma3 has no resident lane on ANY host).
    assert!(camelid::model::arch_requires_runnable_bridge_given(
        &config.architecture,
        false,
        true
    ));
    assert!(camelid::model::arch_requires_runnable_bridge_given(
        &config.architecture,
        true,
        false
    ));
    assert!(camelid::model::arch_has_windowed_attention(&config));
}

#[test]
fn gemma3_without_qk_norm_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gemma3-no-qknorm.gguf");
    write_gemma3_gguf(
        &path,
        &Gemma3FixtureOptions {
            include_qk_norm: false,
            ..Default::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let err = LlamaTensorBinding::bind(&gguf, &config)
        .expect_err("gemma3 missing attn_q_norm/attn_k_norm must fail closed");
    let msg = format!("{err}");
    assert!(
        msg.contains("QK-norm") && msg.contains("gemma3"),
        "unexpected error: {msg}"
    );
}

#[test]
fn gemma3_without_sandwich_norms_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gemma3-no-postnorms.gguf");
    write_gemma3_gguf(
        &path,
        &Gemma3FixtureOptions {
            include_post_norms: false,
            ..Default::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let err = LlamaTensorBinding::bind(&gguf, &config)
        .expect_err("gemma3 missing post_attention_norm/post_ffw_norm must fail closed");
    let msg = format!("{err}");
    assert!(
        msg.contains("sandwich") && msg.contains("gemma3"),
        "unexpected error: {msg}"
    );
}

#[test]
fn gemma3_without_sliding_window_key_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gemma3-no-window.gguf");
    write_gemma3_gguf(
        &path,
        &Gemma3FixtureOptions {
            sliding_window: None,
            ..Default::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let err = LlamaModelConfig::from_gguf(&gguf)
        .expect_err("gemma3 missing attention.sliding_window must fail closed at config parse");
    let msg = format!("{err}");
    assert!(
        msg.contains("gemma3.attention.sliding_window") && msg.contains("fails closed"),
        "unexpected error: {msg}"
    );
}

#[test]
fn gemma3_explicit_pattern_and_local_base_keys_override_reference_constants() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gemma3-explicit-keys.gguf");
    write_gemma3_gguf(
        &path,
        &Gemma3FixtureOptions {
            pattern_key: Some(4),
            local_base_key: Some(50_000.0),
            ..Default::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let meta = config.gemma3.as_ref().unwrap();
    assert_eq!(meta.sliding_window_pattern, 4);
    assert_eq!(meta.rope_freq_base_local, 50_000.0);
}

#[test]
fn gemma3_malformed_pattern_key_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gemma3-bad-pattern.gguf");
    write_gemma3_gguf(
        &path,
        &Gemma3FixtureOptions {
            pattern_key_malformed: true,
            ..Default::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let err = LlamaModelConfig::from_gguf(&gguf).expect_err(
        "an explicit but non-integer sliding_window_pattern must fail closed, not fall back",
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("sliding_window_pattern"),
        "unexpected error: {msg}"
    );
}

/// Schedule derivation exercised THROUGH `from_gguf` at the real row's depth
/// (the unit test hand-builds the struct; this one proves the parse itself
/// derives the schedule from the resolved pattern). Expectations are literal
/// lists — NOT the production `(i + 1) % pattern` expression — so a regression
/// in the derivation (e.g. deriving from `REFERENCE_SLIDING_WINDOW_PATTERN`
/// instead of the resolved pattern) cannot be mirrored into the expectation.
#[test]
fn gemma3_from_gguf_derives_the_26_layer_reference_schedule() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gemma3-26-layers.gguf");
    write_gemma3_gguf(
        &path,
        &Gemma3FixtureOptions {
            block_count: 26,
            ..Default::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let meta = config.gemma3.as_ref().expect("gemma3 metadata must parse");

    assert_eq!(meta.layer_is_sliding.len(), 26);
    let globals: Vec<usize> = meta
        .layer_is_sliding
        .iter()
        .enumerate()
        .filter(|(_, sliding)| !**sliding)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(globals, vec![5, 11, 17, 23]);
    assert!(
        meta.is_sliding_layer(25),
        "layer 25 must be local (no forced-global final layer)"
    );
    for idx in 0..26 {
        let expect_global = matches!(idx, 5 | 11 | 17 | 23);
        assert_eq!(!meta.is_sliding_layer(idx), expect_global, "layer {idx}");
        assert_eq!(
            meta.rope_freq_base_at(idx),
            if expect_global { 1_000_000.0 } else { 10_000.0 },
            "layer {idx} rope base"
        );
        assert_eq!(
            meta.layer_window(idx),
            if expect_global { None } else { Some(512) },
            "layer {idx} window"
        );
    }
}

/// The explicit `gemma3.attention.sliding_window_pattern` override must reach
/// the schedule DERIVATION through `from_gguf` (not merely the stored field):
/// 12 layers at pattern 4 puts the globals at 3/7/11 (literal list), and the
/// `gemma3.rope.freq_base_swa` override must reach the per-layer accessor.
#[test]
fn gemma3_from_gguf_pattern_override_drives_the_schedule_derivation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gemma3-12-layers-pattern-4.gguf");
    write_gemma3_gguf(
        &path,
        &Gemma3FixtureOptions {
            block_count: 12,
            pattern_key: Some(4),
            local_base_key: Some(50_000.0),
            ..Default::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let meta = config.gemma3.as_ref().expect("gemma3 metadata must parse");

    assert_eq!(meta.sliding_window_pattern, 4);
    assert_eq!(meta.layer_is_sliding.len(), 12);
    let globals: Vec<usize> = meta
        .layer_is_sliding
        .iter()
        .enumerate()
        .filter(|(_, sliding)| !**sliding)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(globals, vec![3, 7, 11]);
    for idx in 0..12 {
        let expect_global = matches!(idx, 3 | 7 | 11);
        assert_eq!(!meta.is_sliding_layer(idx), expect_global, "layer {idx}");
        assert_eq!(
            meta.rope_freq_base_at(idx),
            if expect_global { 1_000_000.0 } else { 50_000.0 },
            "layer {idx} rope base"
        );
    }
}

#[test]
fn gemma3_zero_sliding_window_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gemma3-zero-window.gguf");
    write_gemma3_gguf(
        &path,
        &Gemma3FixtureOptions {
            sliding_window: Some(0),
            ..Default::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let err = LlamaModelConfig::from_gguf(&gguf)
        .expect_err("a zero-width sliding window must fail closed at config parse");
    assert!(
        matches!(err, BackendError::InvalidModelMetadata(_)),
        "expected InvalidModelMetadata, got: {err:?}"
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("gemma3.attention.sliding_window") && msg.contains("greater than zero"),
        "unexpected error: {msg}"
    );
}

#[test]
fn gemma3_missing_global_rope_base_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gemma3-no-global-base.gguf");
    write_gemma3_gguf(
        &path,
        &Gemma3FixtureOptions {
            global_base: None,
            ..Default::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let err = LlamaModelConfig::from_gguf(&gguf)
        .expect_err("gemma3 missing rope.freq_base must fail closed at config parse");
    assert!(
        matches!(err, BackendError::InvalidModelMetadata(_)),
        "expected InvalidModelMetadata, got: {err:?}"
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("gemma3.rope.freq_base") && msg.contains("fails closed"),
        "unexpected error: {msg}"
    );
}

#[test]
fn gemma3_non_positive_global_rope_base_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gemma3-zero-global-base.gguf");
    write_gemma3_gguf(
        &path,
        &Gemma3FixtureOptions {
            global_base: Some(0.0),
            ..Default::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let err = LlamaModelConfig::from_gguf(&gguf)
        .expect_err("gemma3 non-positive rope.freq_base must fail closed at config parse");
    assert!(
        matches!(err, BackendError::InvalidModelMetadata(_)),
        "expected InvalidModelMetadata, got: {err:?}"
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("gemma3.rope.freq_base") && msg.contains("greater than zero"),
        "unexpected error: {msg}"
    );
}

#[test]
fn gemma3_malformed_local_base_key_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gemma3-bad-local-base.gguf");
    write_gemma3_gguf(
        &path,
        &Gemma3FixtureOptions {
            local_base_key_malformed: true,
            ..Default::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let err = LlamaModelConfig::from_gguf(&gguf).expect_err(
        "an explicit but non-float rope.freq_base_swa must fail closed, not fall back to \
         the reference local base",
    );
    assert!(
        matches!(err, BackendError::InvalidModelMetadata(_)),
        "expected InvalidModelMetadata, got: {err:?}"
    );
    let msg = format!("{err}");
    assert!(msg.contains("freq_base_swa"), "unexpected error: {msg}");
}

#[test]
fn gemma3_non_positive_local_base_key_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gemma3-zero-local-base.gguf");
    write_gemma3_gguf(
        &path,
        &Gemma3FixtureOptions {
            local_base_key: Some(0.0),
            ..Default::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let err = LlamaModelConfig::from_gguf(&gguf).expect_err(
        "an explicit but non-positive rope.freq_base_swa must fail closed, not fall back to \
         the reference local base",
    );
    assert!(
        matches!(err, BackendError::InvalidModelMetadata(_)),
        "expected InvalidModelMetadata, got: {err:?}"
    );
    let msg = format!("{err}");
    assert!(msg.contains("freq_base_swa"), "unexpected error: {msg}");
}

/// Real-row gate (campaign Phase 1 parity gate): every one of the 1B row's
/// 26x4 = 104 norm tensors binds non-None from the actual GGUF, the window/
/// pattern/rope-base metadata round-trips exactly, the structural flags are
/// set, and the runnable-only guard still fires after binding succeeds.
/// Skipped unless `CAMELID_GEMMA3_GGUF` points at the real
/// gemma-3-1b-it-Q8_0 file (same convention as `CAMELID_GEMMA4_GGUF` in
/// tests/gemma4_metadata.rs).
#[test]
fn gemma3_real_row_binds_all_104_norm_tensors_and_window_schedule() {
    let Some(path) = std::env::var_os("CAMELID_GEMMA3_GGUF") else {
        eprintln!(
            "SKIP gemma3_real_row_binds_all_104_norm_tensors_and_window_schedule: \
             set CAMELID_GEMMA3_GGUF to the gemma-3-1b-it-Q8_0 GGUF"
        );
        return;
    };
    let gguf = read_metadata(Path::new(&path)).unwrap();
    assert_eq!(gguf.architecture(), Some("gemma3"));
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();

    // (a) All 104 norm tensors bind non-None with the real shapes.
    assert_eq!(binding.layers.len(), 26);
    let mut bound_norms = 0usize;
    for (idx, layer) in binding.layers.iter().enumerate() {
        let q_norm = layer
            .attention_q_norm()
            .unwrap_or_else(|| panic!("layer {idx} attn_q_norm must bind"));
        let k_norm = layer
            .attention_k_norm()
            .unwrap_or_else(|| panic!("layer {idx} attn_k_norm must bind"));
        let post_attn = layer
            .post_attention_norm
            .as_ref()
            .unwrap_or_else(|| panic!("layer {idx} post_attention_norm must bind"));
        let post_ffw = layer
            .post_ffw_norm
            .as_ref()
            .unwrap_or_else(|| panic!("layer {idx} post_ffw_norm must bind"));
        // NAME-pinned per layer: both QK norms are [256] and both sandwich
        // norms are [1152], so shape assertions alone would pass with the
        // lookups transposed — pin each field to its exact tensor name.
        assert_eq!(
            q_norm.name,
            format!("blk.{idx}.attn_q_norm.weight"),
            "layer {idx} q_norm name"
        );
        assert_eq!(
            k_norm.name,
            format!("blk.{idx}.attn_k_norm.weight"),
            "layer {idx} k_norm name"
        );
        assert_eq!(
            post_attn.name,
            format!("blk.{idx}.post_attention_norm.weight"),
            "layer {idx} post_attention_norm name"
        );
        assert_eq!(
            post_ffw.name,
            format!("blk.{idx}.post_ffw_norm.weight"),
            "layer {idx} post_ffw_norm name"
        );
        assert_eq!(q_norm.dimensions, vec![256], "layer {idx} q_norm shape");
        assert_eq!(k_norm.dimensions, vec![256], "layer {idx} k_norm shape");
        assert_eq!(
            post_attn.dimensions,
            vec![1152],
            "layer {idx} post_attention_norm shape"
        );
        assert_eq!(
            post_ffw.dimensions,
            vec![1152],
            "layer {idx} post_ffw_norm shape"
        );
        bound_norms += 4;
    }
    assert_eq!(bound_norms, 104);

    // (b) Window/pattern/rope-base metadata round-trips exactly: window 512
    // (includes the current position), pattern 6 with globals at 5/11/17/23
    // (and NO forced-global final layer), local base 10000, global base 1e6.
    let meta = config.gemma3.as_ref().expect("gemma3 metadata must parse");
    assert_eq!(meta.sliding_window, 512);
    assert_eq!(meta.sliding_window_pattern, 6);
    assert_eq!(meta.rope_freq_base_local, 10_000.0);
    assert_eq!(meta.rope_freq_base_global, 1_000_000.0);
    let globals: Vec<usize> = meta
        .layer_is_sliding
        .iter()
        .enumerate()
        .filter(|(_, sliding)| !**sliding)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(globals, vec![5, 11, 17, 23]);
    assert!(meta.is_sliding_layer(25), "layer 25 must stay local");
    for idx in 0..26 {
        if globals.contains(&idx) {
            assert_eq!(meta.rope_freq_base_at(idx), 1_000_000.0, "layer {idx}");
            assert_eq!(meta.layer_window(idx), None, "layer {idx}");
        } else {
            assert_eq!(meta.rope_freq_base_at(idx), 10_000.0, "layer {idx}");
            assert_eq!(meta.layer_window(idx), Some(512), "layer {idx}");
        }
    }

    // (c) Structural flags: GeGLU, sqrt(1152) embed scale, forced split-half
    // pairing on the metadata (dense-path flag deliberately unchanged).
    assert!(meta.ffn_geglu);
    assert_eq!(meta.embed_scale, (1152.0f32).sqrt());
    assert!(meta.rope_neox_pairing);
    assert!(!config.rope_neox_pairing);

    // CRITICAL invariant, updated for the Phase 3b flip: binding succeeded
    // and the resident encodes are in hand, so gemma3 left the unconditional
    // runnable-only set — but the routing stays capability-aware: on a host
    // where the Metal-resident lane cannot serve, the runnable bridge is
    // still the only correct serve path, and the CPU dense forward fails
    // closed at forward dispatch (H4) via `arch_has_windowed_attention`.
    assert!(!is_runnable_only_arch(&config.architecture));
    // Both capability legs false => bridge; Phase 3c added the quantization
    // leg (a non-Q8_0 gemma3 has no resident lane on ANY host).
    assert!(camelid::model::arch_requires_runnable_bridge_given(
        &config.architecture,
        false,
        true
    ));
    assert!(camelid::model::arch_requires_runnable_bridge_given(
        &config.architecture,
        true,
        false
    ));
    assert!(camelid::model::arch_has_windowed_attention(&config));
}
