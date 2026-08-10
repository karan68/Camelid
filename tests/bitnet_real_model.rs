//! Real-artifact gates for Microsoft's three official BitNet GGUFs.
//!
//! Download the files to `target/bitnet-fixtures/`, then run:
//! `cargo test --test bitnet_real_model -- --ignored --nocapture`.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use camelid::embedding::EmbeddingRuntime;
use camelid::gguf::{read_metadata, GgufTensorType};
use camelid::runnable::RunnableModel;
use sha2::{Digest, Sha256};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("bitnet-fixtures")
        .join(name)
}

fn assert_sha256(path: &Path, expected: &str) {
    let mut reader = BufReader::new(File::open(path).expect("open BitNet fixture"));
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer).expect("hash BitNet fixture");
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    assert_eq!(format!("{:x}", hasher.finalize()), expected);
}

fn assert_i2_s_projections(path: &Path, expected_name: &str, expected_count: usize) {
    let gguf = read_metadata(path).expect("read BitNet metadata");
    assert_eq!(gguf.model_name(), Some(expected_name));
    assert_eq!(gguf.metadata_u32("general.file_type"), Some(40));
    let i2_s = gguf
        .tensors
        .iter()
        .filter(|tensor| tensor.tensor_type == GgufTensorType::I2S)
        .count();
    assert_eq!(i2_s, expected_count);
}

fn bitnet_gpu_allowed() -> bool {
    std::env::var("CAMELID_BITNET_GPU")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "disabled" | "no"
            )
        })
        .unwrap_or(true)
}

fn assert_gpu_dispatch(metal_before: u64, cuda_before: u64) {
    if bitnet_gpu_allowed() && camelid::cuda::gpu_accel_enabled() {
        match camelid::cuda::gpu_acceleration_info().backend {
            "metal" => assert!(
                camelid::metal::metal_bitnet_run_count() > metal_before,
                "BitNet execution silently fell back instead of executing Metal"
            ),
            "cuda" => assert!(
                camelid::cuda::cuda_bitnet_run_count() > cuda_before,
                "BitNet execution silently fell back instead of executing CUDA"
            ),
            _ => {}
        }
    }
}

#[test]
#[ignore = "requires the SHA-pinned 1.19 GB Microsoft BitNet causal GGUF"]
fn bitnet_b1_58_2b_4t_loads_the_exact_i2_s_graph() {
    let path = fixture("ggml-model-i2_s.gguf");
    assert_sha256(
        &path,
        "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
    );
    assert_i2_s_projections(&path, "bitnet2b", 210);
    let model = RunnableModel::load(path.to_str().expect("UTF-8 fixture path"))
        .expect("load causal BitNet runtime");
    assert_eq!(model.architecture, "bitnet-b1.58");
    assert_eq!(model.d_model, 2_560);
    assert_eq!(model.n_layers, 30);
    let metal_before = camelid::metal::metal_bitnet_run_count();
    let cuda_before = camelid::cuda::cuda_bitnet_run_count();
    let logits = model
        .forward_logits(&[1])
        .expect("execute one causal BitNet position");
    assert_gpu_dispatch(metal_before, cuda_before);
    assert_eq!(logits.len(), model.vocab);
    assert!(logits.iter().all(|value| value.is_finite()));
}

fn assert_embedding(
    filename: &str,
    expected_sha256: &str,
    expected_name: &str,
    expected_i2_s: usize,
    expected_dimensions: usize,
) {
    let path = fixture(filename);
    assert_sha256(&path, expected_sha256);
    assert_i2_s_projections(&path, expected_name, expected_i2_s);
    let runtime = EmbeddingRuntime::load(&path).expect("load BitNet embedding runtime");
    let text = "A camel stores fat in its hump.";
    let metal_before = camelid::metal::metal_bitnet_run_count();
    let cuda_before = camelid::cuda::cuda_bitnet_run_count();
    let embedding = runtime.embed(text, None).expect("embed probe");
    assert_gpu_dispatch(metal_before, cuda_before);
    let repeated = runtime.embed(text, None).expect("repeat embed probe");
    assert_eq!(embedding, repeated, "embedding must be deterministic");
    assert_eq!(embedding.len(), expected_dimensions);
    assert!(embedding.iter().all(|value| value.is_finite()));
    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "L2 norm was {norm}");
}

#[test]
#[ignore = "requires the SHA-pinned 428 MB Microsoft BitNet embedding GGUF"]
fn bitnet_embedding_0_6b_executes_and_normalizes() {
    assert_embedding(
        "bitnet-embeddings-0.6b-bf16-i2_s.gguf",
        "c89c64f05a2d3f83565250a6762640197fc624df866d4a1bd5853f811219af17",
        "bitnet-embeddings-0.6b",
        196,
        1_024,
    );
}

#[test]
#[ignore = "requires the SHA-pinned 367 MB Microsoft BitNet embedding GGUF"]
fn bitnet_embedding_270m_executes_and_normalizes() {
    assert_embedding(
        "bitnet-embeddings-270m-bf16-i2_s.gguf",
        "8ee5ae971b103cd55758934be54e5c9f7cc2b58b15890615acce8e649988c751",
        "bitnet-embeddings-270m",
        126,
        640,
    );
}
