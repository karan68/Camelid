use std::{fs, net::TcpListener, path::Path};
use tempfile::tempdir;

use camelid::{
    distributed::{
        run_network_benchmark_worker_on_listener, run_worker_loop_on_listener, DistributedClient,
        PipelineRole, DISTRIBUTED_CLIENT, DISTRIBUTED_RANGE,
    },
    gguf::read_metadata,
    inference::{LlamaInferenceSession, LlamaLoadedWeights},
    model::{LlamaModelConfig, LlamaTensorBinding},
    tensor::{CpuTensor, TensorStore},
};

/// Bind loopback on an OS-assigned free port.
///
/// These tests used to hard-code 127.0.0.1:8099 and 127.0.0.1:8098. On a
/// developer box that also runs `camelid serve` the bind lost the race, the
/// worker thread died silently, and the coordinator connected to whatever else
/// held the port — producing an opaque `ConnectionReset`/`UnexpectedEof` on the
/// *client* read with no hint that the worker never started. Binding here (not
/// inside the worker thread) also means the socket is listening before the
/// coordinator connects, so no readiness sleep is needed.
fn bind_ephemeral() -> TcpListener {
    TcpListener::bind("127.0.0.1:0").expect("bind loopback ephemeral port")
}

#[test]
fn test_distributed_pipeline_parallel_inference() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("tiny_model.gguf");

    // Write a tiny 2-layer model GGUF file
    write_tiny_llama_gguf(&model_path);

    let gguf = read_metadata(&model_path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();

    // 1. Worker Setup (Loads layer 1..2)
    let store_worker = TensorStore::open(&model_path, &gguf);
    let worker_weights = LlamaLoadedWeights::load_distributed(
        &store_worker,
        &binding,
        1,     // layer_start
        2,     // layer_end (2-layer model)
        false, // load_embedding
        false, // load_output
    )
    .unwrap();

    let worker_config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let worker_session = LlamaInferenceSession::new(worker_config, worker_weights).unwrap();

    // Spawn Worker in background thread. The listener is already bound, so the
    // coordinator below can connect immediately.
    let listener = bind_ephemeral();
    let worker_addr = listener.local_addr().unwrap().to_string();
    let _worker_handle = std::thread::spawn(move || {
        if let Err(e) = run_worker_loop_on_listener(listener, worker_session) {
            // Surface the worker's own failure. Without this the only symptom is
            // an unexplained EOF on the coordinator's read.
            eprintln!("distributed worker loop exited with error: {e:#}");
        }
    });

    // 2. Coordinator Setup (Loads layer 0..1)
    let client = DistributedClient::connect(&worker_addr).unwrap();
    let _ = DISTRIBUTED_CLIENT.set(client);
    let _ = DISTRIBUTED_RANGE.set((0, 1));

    let store_coord = TensorStore::open(&model_path, &gguf);
    let coord_weights = LlamaLoadedWeights::load_distributed(
        &store_coord,
        &binding,
        0,    // layer_start
        1,    // layer_end
        true, // load_embedding
        true, // load_output
    )
    .unwrap();

    let coord_config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let mut coord_session = LlamaInferenceSession::new(coord_config, coord_weights).unwrap();

    // Run forward pass on coordinator (which will delegate layer 1 to worker!)
    let output = coord_session
        .forward_single_token(0)
        .expect("coordinator forward pass (layer 1 is delegated to the worker over TCP)");

    // Assert logits are computed and the dimensions are correct
    assert_eq!(output.logits.shape.dims, vec![1, 16]); // [1, vocab_size]
    assert_eq!(output.logits.data.len(), 16);

    // Verify that all logits are valid finite floats
    for &val in &output.logits.data {
        assert!(val.is_finite());
    }
}

#[test]
fn test_network_benchmark() {
    // Spawn benchmark worker in background thread on an already-bound listener.
    let listener = bind_ephemeral();
    let worker_addr = listener.local_addr().unwrap().to_string();
    let _worker_handle = std::thread::spawn(move || {
        if let Err(e) = run_network_benchmark_worker_on_listener(listener) {
            eprintln!("network benchmark worker exited with error: {e:#}");
        }
    });

    // Run coordinator benchmark locally
    let result = camelid::distributed::run_network_benchmark_coordinator(
        &worker_addr,
        10,   // ping_count
        1024, // payload_size
        10,   // bandwidth_mb
    );

    assert!(
        result.is_ok(),
        "Network benchmark coordinator failed: {:?}",
        result.err()
    );
}

// Helpers to write a tiny mock GGUF Llama model
fn header(tensor_count: i64, metadata_count: i64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_i64(&mut b, tensor_count);
    push_i64(&mut b, metadata_count);
    b
}

/// Regression: in a pipeline-parallel layer split, the first node owns the token
/// embedding and the LAST node owns output_norm + the output projection (it computes
/// the final norm and logits). A 2-node CLI pipeline was previously broken because the
/// output weights were loaded on the first node, leaving the last node unable to
/// produce logits (`rms_norm weight shape [0]`).
#[test]
fn pipeline_load_puts_output_weights_on_last_node() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("tiny_model.gguf");
    write_tiny_llama_gguf(&model_path);
    let gguf = read_metadata(&model_path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();
    let store = TensorStore::open(&model_path, &gguf);

    // First node (layers 0..1): owns the embedding, not the output weights.
    let first = LlamaLoadedWeights::load(&store, &binding, Some(0..1)).unwrap();
    assert_ne!(
        first.token_embedding.shape.dims,
        vec![0],
        "first node loads embedding"
    );
    assert_eq!(
        first.output_norm.shape.dims,
        vec![0],
        "first node has no output_norm"
    );
    first.validate_dense_shapes(&config).unwrap();

    // Last node (layers 1..2): owns output_norm + output, not the embedding.
    let last = LlamaLoadedWeights::load(&store, &binding, Some(1..2)).unwrap();
    assert_eq!(
        last.token_embedding.shape.dims,
        vec![0],
        "last node has no embedding"
    );
    assert_ne!(
        last.output_norm.shape.dims,
        vec![0],
        "last node loads output_norm"
    );
    last.validate_dense_shapes(&config).unwrap();
}

/// Regression for defect D1. `serve-distributed` computes logits on the **coordinator** —
/// `run_worker_loop`'s `forward_worker_layers` returns a bare hidden state, applying
/// neither `output_norm` nor the output projection — so a coordinator holding a head shard
/// must still own the output head. That is what `PipelineRole::Coordinator` encodes, and
/// what `test_distributed_pipeline_parallel_inference` above has always assumed.
///
/// The shipped API loader instead derived ownership positionally through
/// `LlamaLoadedWeights::load`, which reserves the output head for the LAST shard (see
/// `pipeline_load_puts_output_weights_on_last_node`, correct for that other topology).
/// The second half here is the negative control: it reproduces in-process the exact
/// failure that mismatch produced over HTTP.
#[test]
fn distributed_coordinator_head_shard_owns_the_output_head() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("tiny_model.gguf");
    write_tiny_llama_gguf(&model_path);
    let gguf = read_metadata(&model_path).unwrap();
    let binding =
        LlamaTensorBinding::bind(&gguf, &LlamaModelConfig::from_gguf(&gguf).unwrap()).unwrap();
    let store = TensorStore::open(&model_path, &gguf);

    // Shaped like a hidden state the worker hands back: [seq, hidden].
    let from_worker = CpuTensor::from_f32("wire_activation", vec![1, 8], vec![0.0; 8]).unwrap();

    let (load_embedding, load_output) = PipelineRole::Coordinator.tensor_ownership();
    let coordinator =
        LlamaLoadedWeights::load_distributed(&store, &binding, 0, 1, load_embedding, load_output)
            .unwrap();
    let coordinator =
        LlamaInferenceSession::new(LlamaModelConfig::from_gguf(&gguf).unwrap(), coordinator)
            .unwrap();
    let logits = coordinator
        .forward_final_norm_and_logits(&from_worker)
        .expect("a coordinator owning the output head can finalize the forward pass");
    assert_eq!(logits.shape.dims, vec![1, 16], "[seq, vocab]");

    let positional = LlamaLoadedWeights::load(&store, &binding, Some(0..1)).unwrap();
    assert_eq!(
        positional.output_norm.shape.dims,
        vec![0],
        "positional ownership leaves a head shard without output_norm"
    );
    assert_eq!(
        positional.output.as_ref().map(|o| o.shape.dims.clone()),
        Some(vec![0]),
        "positional ownership leaves a head shard without the output projection"
    );
    let positional =
        LlamaInferenceSession::new(LlamaModelConfig::from_gguf(&gguf).unwrap(), positional)
            .unwrap();
    let err = positional
        .forward_final_norm_and_logits(&from_worker)
        .expect_err("positional ownership cannot finalize; this was the shipped defect");
    // Which check trips first depends on the finalize path and on whether the model ties
    // its output to the embedding; the served 1B model reported the projection
    // (`token_embd.weight, got [0]`), this untied fixture reports the norm. Both are the
    // same missing output head, so assert the empty shape rather than one message.
    assert!(
        err.to_string().contains("[0]"),
        "expected a missing-output-head shape error, got: {err}"
    );
}

fn write_tiny_llama_gguf(path: &Path) {
    // 2 layers, tiny dimensions: hidden=8, vocab=16, ffn=16, heads=2, kv_heads=2
    // We have exactly 21 tensors: token_embedding, output_norm, output + 9 tensors per layer (18 total) = 21 tensors.
    let mut b = header(21, 13);

    // Model hparams metadata
    push_kv_string(&mut b, "general.architecture", "llama");
    push_kv_string(&mut b, "general.name", "tiny-llama-distributed");
    push_kv_u32(&mut b, "llama.context_length", 32);
    push_kv_u32(&mut b, "llama.embedding_length", 8);
    push_kv_u32(&mut b, "llama.block_count", 2);
    push_kv_u32(&mut b, "llama.feed_forward_length", 16);
    push_kv_u32(&mut b, "llama.attention.head_count", 2);
    push_kv_u32(&mut b, "llama.attention.head_count_kv", 2);
    push_kv_f32(&mut b, "llama.attention.layer_norm_rms_epsilon", 1e-5);
    push_kv_f32(&mut b, "llama.rope.freq_base", 10000.0);
    push_kv_u32(&mut b, "general.alignment", 32);

    // Tokenizer metadata (minimal)
    push_kv_string(&mut b, "tokenizer.ggml.model", "llama");

    // Add empty array of tokens to keep tokenizer parser happy
    push_string(&mut b, "tokenizer.ggml.tokens");
    push_u32(&mut b, 9); // array type
    push_u32(&mut b, 8); // string type
    push_u64(&mut b, 16); // array length
    for i in 0..16 {
        push_string(&mut b, &format!("T{}", i));
    }

    // Tensors layout:
    // Name, dims, type, offset
    let tensors = vec![
        ("token_embd.weight", vec![16, 8]), // vocab x hidden
        ("output_norm.weight", vec![8]),
        ("output.weight", vec![8, 16]), // hidden x vocab
        ("blk.0.attn_norm.weight", vec![8]),
        ("blk.0.attn_q.weight", vec![8, 8]),
        ("blk.0.attn_k.weight", vec![8, 8]),
        ("blk.0.attn_v.weight", vec![8, 8]),
        ("blk.0.attn_output.weight", vec![8, 8]),
        ("blk.0.ffn_norm.weight", vec![8]),
        ("blk.0.ffn_gate.weight", vec![8, 16]),
        ("blk.0.ffn_up.weight", vec![8, 16]),
        ("blk.0.ffn_down.weight", vec![16, 8]),
        ("blk.1.attn_norm.weight", vec![8]),
        ("blk.1.attn_q.weight", vec![8, 8]),
        ("blk.1.attn_k.weight", vec![8, 8]),
        ("blk.1.attn_v.weight", vec![8, 8]),
        ("blk.1.attn_output.weight", vec![8, 8]),
        ("blk.1.ffn_norm.weight", vec![8]),
        ("blk.1.ffn_gate.weight", vec![8, 16]),
        ("blk.1.ffn_up.weight", vec![8, 16]),
        ("blk.1.ffn_down.weight", vec![16, 8]),
    ];

    let mut current_offset = 0u64;
    for (name, dims) in &tensors {
        push_string(&mut b, name);
        push_u32(&mut b, dims.len() as u32);
        for &dim in dims {
            push_i64(&mut b, dim as i64);
        }
        push_i32(&mut b, 0); // f32
        push_u64(&mut b, current_offset);

        let elements: usize = dims.iter().product();
        current_offset += (elements * 4) as u64;
    }

    while !b.len().is_multiple_of(32) {
        b.push(0);
    }

    // Write actual float data (all 0.1f32 to be valid non-NaN)
    let total_elements: usize = tensors
        .iter()
        .map(|(_, dims)| dims.iter().product::<usize>())
        .sum();
    for _ in 0..total_elements {
        b.extend_from_slice(&0.1f32.to_le_bytes());
    }

    fs::write(path, b).unwrap();
}

fn push_kv_string(b: &mut Vec<u8>, key: &str, value: &str) {
    push_string(b, key);
    push_u32(b, 8); // string type
    push_string(b, value);
}

fn push_kv_u32(b: &mut Vec<u8>, key: &str, value: u32) {
    push_string(b, key);
    push_u32(b, 4); // u32 type
    push_u32(b, value);
}

fn push_kv_f32(b: &mut Vec<u8>, key: &str, value: f32) {
    push_string(b, key);
    push_u32(b, 5); // f32 type
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
