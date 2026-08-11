# Experimental BitNet runtime

Camelid recognizes three exact Microsoft model rows:

- `BitNet-b1.58-2B-4T` for causal generation;
- `BitNet-embedding-0.6B` for embeddings and reranking;
- `BitNet-embedding-270M` for embeddings and reranking.

These rows remain experimental until reference parity and bounded runtime
evidence are complete. Embedding-only files fail closed on generation APIs.

## Cleanroom kernel contract

Camelid consumes the canonical `I2_S` tensors published in the official GGUFs:
four ternary values (`-1`, `0`, `+1`) per byte, interleaved in 128-value tiles,
followed by the tensor-wide scale trailer. The bytes remain unchanged and are
page-backed for zero-copy Metal access and cached CUDA upload.

`CAMELID_BITNET_KERNEL` selects an independently implemented execution strategy:

| Value | Strategy |
|---|---|
| `auto` or unset | direct `i2_s` accumulation |
| `i2_s` | decode and accumulate each ternary weight |
| `tl1` | group two weights and select from a 9-entry activation table |
| `tl2` | group three weights and select from a 27-entry activation table |

All three strategies evaluate the same canonical weights. They are behavioral
cleanroom kernels, not readers for BitNet.cpp's separately permuted,
model-specific TL1/TL2 files or generated headers.

CPU is the correctness oracle. Metal and CUDA are attempted when Camelid's
platform-wide GPU switch is enabled; a device, compilation, allocation, or
launch failure falls back to CPU. Set `CAMELID_BITNET_GPU=0` to force the CPU
kernel without disabling acceleration for other model families.

## Hardware verification

The synthetic CPU and Metal tests cover `i2_s`, `tl1`, and `tl2`. Real-artifact
tests SHA-pin the three Microsoft GGUFs under `target/bitnet-fixtures/`.

On macOS:

```sh
cargo test --lib bitnet_i2_s_cleanroom_modes_execute_on_metal -- --nocapture
cargo test --test bitnet_real_model bitnet_embedding_270m_executes_and_normalizes -- --ignored --nocapture
CAMELID_BITNET_GPU=0 cargo test --test bitnet_real_model bitnet_embedding_270m_executes_and_normalizes -- --ignored --nocapture
```

On the Windows CUDA validation machine (CUDA is part of Camelid's default
Windows build):

```powershell
$env:CAMELID_REQUIRE_CUDA_TESTS = "1"
cargo test --lib cuda_bitnet_i2_s_cleanroom_modes_match_cpu_oracle -- --ignored --nocapture
cargo test --test bitnet_real_model bitnet_embedding_270m_executes_and_normalizes -- --ignored --nocapture
$env:CAMELID_BITNET_KERNEL = "tl1" # repeat with i2_s and tl2
cargo test --test bitnet_real_model bitnet_embedding_270m_executes_and_normalizes -- --ignored --nocapture
```

The hardware tests compare against the CPU oracle and require the backend run
counter to advance. A silent GPU-to-CPU fallback therefore fails the CUDA/Metal
assertion instead of producing a misleading green test.

Expected fixture names and SHA-256 digests are recorded in
`tests/bitnet_real_model.rs`.
