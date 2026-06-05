# Camelid

[![CI][ci-badge]][ci-workflow]

Camelid is a Rust-native local LLM inference engine focused on GGUF inference, correctness, and local performance.

It loads GGUF models directly, exposes an OpenAI-style local API, and is built around reproducible benchmark evidence instead of hype.

Camelid is not a wrapper around Ollama or llama.cpp. It is its own inference/runtime project written in Rust.

![Camelid WebUI chat surface](docs/assets/camelid-readme-chat-surface-dark.png)

The local web frontend: a dark, collapsed-rail chat surface that enables chat only for model rows the compatibility contract recognizes.

## Why Camelid?

- **Rust-native inference** — the tokenizer, GGUF loader, CPU kernels, and Metal GPU path are all implemented in this repository; one static binary, no Python.
- **Direct GGUF loading** — point it at a `.gguf` file; no conversion or import step.
- **OpenAI-style local API** — `/v1/chat/completions` and `/v1/completions` with streaming, served locally.
- **Correctness-focused development** — optimized paths are gated on token-for-token parity with a reference implementation before they ship; unsupported configurations fail closed with typed errors instead of guessing.
- **Reproducible benchmark evidence** — every published number comes from a committed evidence bundle with raw logs, commands, and versions. No raw log, no claim.
- **Apple Silicon performance work** — a Metal-resident path (GPU prefill, GPU decode with on-GPU greedy sampling) that is measured against llama.cpp and MLX-LM on the same host, with wins, ties, and losses all stated.

## Status

| Feature | Status | Notes |
|---|---|---|
| GGUF loading | Working | Direct load with metadata/tensor inspection (`camelid inspect`). |
| Q8_0 inference | Working | The validated quantization. Support is per exact model row — see [`SUPPORT_MATRIX_v0.1.md`](SUPPORT_MATRIX_v0.1.md) (TinyLlama 1.1B, Llama 3.2 1B/3B, Llama 3 8B; Mistral/Mixtral are validation-in-progress and fail closed). |
| OpenAI-style API | Working | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, plus local capability/health routes. |
| Streaming chat | Working | SSE streaming on the chat endpoint. |
| Apple Silicon Metal path | Working | GPU-resident prefill and decode, selected automatically when a Metal device is present; falls back to validated CPU paths otherwise. |
| Web frontend | Working | Local React/Vite chat surface; enables chat only for model rows the compatibility contract recognizes. |
| Other quantizations | Not supported | Fail closed in v0.1. |
| Distributed worker | Experimental | `serve-distributed` / pipeline worker-master commands exist; not part of the v0.1 support claim. |
| Ghost mode (layer streaming) | Experimental | `ghost-run` executes one transformer block at a time from a repacked file for a strict memory ceiling; trades throughput for memory, no prefetch yet. |

## Quickstart

Build:

```bash
cargo build --release
```

Serve a local GGUF model (Q8_0):

```bash
./target/release/camelid serve \
  --model /path/to/Llama-3.2-3B-Instruct-Q8_0.gguf \
  --threads 4
```

The server listens on `127.0.0.1:8181` by default. List the loaded model (its `id` comes from the GGUF metadata):

```bash
curl -s http://127.0.0.1:8181/v1/models
```

Chat (replace the model `id` with the one returned above):

```bash
curl -s http://127.0.0.1:8181/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama 3.2 3B Instruct",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Add `"stream": true` for SSE streaming. To run the local web frontend:

```bash
cd frontend && npm ci && npm run dev
```

## Evidence

Camelid benchmark claims are only listed when raw logs or reproducible commands are available in the repo. If there is no raw log, there is no benchmark claim.

Same-host snapshot on one Apple M4 (10-core GPU, 16 GB), Llama 3.2 3B Instruct Q8_0, greedy sampling, three same-session rounds with alternating runtime order (medians):

| Lane | Camelid | llama.cpp (Metal) | MLX-LM (8-bit) |
| --- | ---: | ---: | ---: |
| Prefill, 601-token prompt (tok/s) | **587.3** | 543.7 | 577.9 |
| Decode, short context (tok/s) | **29.7** | 29.1 | 29.1 |

Reading boundary: this is a same-session result on one exact model row and one machine, with narrow margins — not a durable or general claim. Some lanes read below the comparators (decode at long context trails MLX-LM); deeper prompt depths are covered by single warm probes rather than protocol-grade rounds. All of it — methods, raw logs, per-round detail, and the lanes where Camelid loses — is in [`BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md) and the committed bundles under [`qa/evidence-bundles/`](qa/evidence-bundles/).

Correctness evidence (token-parity gates, per-row validation artifacts) is indexed in [`COMPATIBILITY.md`](COMPATIBILITY.md) and [`CORRECTNESS_v0.1.md`](docs/release/CORRECTNESS_v0.1.md).

## Documentation

- [`SUPPORT_MATRIX_v0.1.md`](SUPPORT_MATRIX_v0.1.md) — which exact model rows are supported, and with what evidence
- [`COMPATIBILITY.md`](COMPATIBILITY.md) — the durable support contract
- [`BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md) — benchmark snapshots and claim rules
- [`STATUS.md`](STATUS.md) — current evidence snapshot and blockers
- [`ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md) — implementation architecture
- [`RELEASE_NOTES_v0.1.md`](docs/release/RELEASE_NOTES_v0.1.md) — v0.1 release notes
- [`ROADMAP.md`](ROADMAP.md) — planned engineering sequence

Validation for code changes:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

## License

Camelid is licensed under the [MIT License](LICENSE).

Camelid's tokenizer, reference compatibility layouts, and validation benchmarks are inspired by and checked against [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (Copyright (c) 2023-2026 The ggml authors, MIT License). Camelid maintains its own Rust-native codebase while crediting the reference work of the `ggml` ecosystem.

[ci-badge]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml
