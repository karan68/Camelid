<div align="center">

# 🐪 Camelid

**A Rust-native local LLM inference engine — GGUF in, OpenAI-style API out, every claim backed by reproducible evidence.**

[![CI][ci-badge]][ci-workflow]
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Language: Rust](https://img.shields.io/badge/language-Rust-orange.svg)
![Platform: Apple Silicon · CPU](https://img.shields.io/badge/platform-Apple%20Silicon%20·%20CPU-lightgrey.svg)

</div>

Camelid loads GGUF models directly, serves them over a local OpenAI-style API, and gates every optimized path on token-for-token parity with a reference implementation. It is **not** a wrapper around Ollama or llama.cpp — the tokenizer, GGUF loader, CPU kernels, and Metal GPU path are all implemented in this repository, shipping as a single static Rust binary with no Python.

![Camelid WebUI chat surface](docs/assets/camelid-readme-chat-surface-dark.png)

<div align="center"><sub>The local web frontend — a dark, collapsed-rail chat surface that unlocks chat only for model rows the compatibility contract recognizes.</sub></div>

---

## Why Camelid

| | |
|---|---|
| 🦀 **Rust-native** | Tokenizer, GGUF loader, CPU kernels, and Metal GPU path live in this repo. One static binary, no Python. |
| 📦 **Direct GGUF** | Point it at a `.gguf` file — no conversion or import step. |
| 🔌 **OpenAI-style API** | `/v1/chat/completions` and `/v1/completions` with SSE streaming, served locally. |
| ✅ **Correctness-first** | Optimized paths ship only after token-for-token parity with a reference; unsupported configs fail closed with typed errors. |
| 🧾 **Proof-carrying** | Any request can emit a sealed *parity receipt* — exact GGUF (SHA-256), exact input, exact tokens — independently re-verifiable against llama.cpp on your own machine, including 7B receipts on a 16 GB Mac. |
| 📊 **Evidence-gated** | Every published number comes from a committed bundle with raw logs, commands, and versions. No raw log, no claim. |
| ⚡ **Apple Silicon path** | A Metal-resident pipeline (GPU prefill, GPU decode with on-GPU greedy sampling) measured head-to-head against llama.cpp and MLX-LM — wins, ties, and losses all stated. |
| 🚀 **Fast model loading** | On Apple Silicon the server maps Q8_0 weights for the GPU to read in place instead of reading and copying them, so reloads are quick and peak memory stays lower. |

---

## Supported models

Support is **per exact model row** (a specific GGUF at a specific quantization), each backed by committed evidence. Anything not listed fails closed.

| Model row | Quant | Serve lane | Evidence |
|---|---|---|---|
| TinyLlama 1.1B Chat | Q8_0 | single-node | Current verified gate |
| Llama 3.2 1B Instruct | Q8_0 | single-node | Exact-row + bounded context 512→8192 |
| Llama 3.2 3B Instruct | Q8_0 | single-node | Exact-row smoke + API/WebUI + bounded context |
| Llama 3 8B Instruct | Q8_0 | single-node | Exact-row + bounded context 512→2048 |
| Mistral 7B Instruct v0.3 | Q8_0 | single-node | Exact-row smoke + bounded context 512→8192 + GPU/CPU parity |
| **Gemma 4 E2B-It** | Q8_0 | single-node (CPU + Metal) | Greedy parity + bounded context **512→8192** |
| **Gemma 4 E4B-It** | Q8_0 | single-node (CPU + Metal) | Greedy parity + bounded context **512→8192** |
| **Gemma 4 12B-It** | Q8_0 | two-Mac distributed | Distributed parity + serve/WebUI smoke |
| **Gemma 4 26B-A4B-It QAT** | Q4_0 (128-expert MoE) | two-Mac distributed | Distributed parity + serve/WebUI smoke |

> **Fails closed (by design):** Mixtral-8x7B v0.1 (validation-in-progress, one-token runtime only); Gemma 4 26B-A4B **Q8_0** (26.9 GB) and 31B (over the 2×16 GB envelope); Gemma 4 MTP/drafter rows; **DiffusionGemma 26B-A4B** (recognized, but a discrete block-diffusion encoder-decoder — not runnable on an autoregressive engine; see [recon](docs/recon/DIFFUSIONGEMMA_26B_A4B_RECON.md)); multimodal input; and all other quantizations in v0.1.

Per-row detail and the exact evidence artifacts live in [`SUPPORT_MATRIX_v0.1.md`](SUPPORT_MATRIX_v0.1.md) and [`COMPATIBILITY.md`](COMPATIBILITY.md).

---

## Engine status

| Capability | Status | Notes |
|---|---|---|
| GGUF loading | ✅ Working | Direct load with metadata/tensor inspection (`camelid inspect`). |
| Q8_0 inference | ✅ Working | The validated quantization; support is per exact row (see above). |
| Gemma 4 engine | ✅ Working | From-scratch `gemma4` engine — see [Gemma 4](#gemma-4) below. |
| OpenAI-style API | ✅ Working | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, plus capability/health routes. |
| Streaming chat | ✅ Working | SSE streaming on the chat endpoint. |
| Apple Silicon Metal path | ✅ Working | GPU-resident prefill and decode, auto-selected when a Metal device is present; CPU fallback otherwise. |
| Web frontend | ✅ Working | Local React/Vite chat surface; unlocks chat only for recognized model rows. |
| Parity receipts | ✅ Working | Opt-in sealed record of one request; `camelid verify-receipt` re-checks it against llama.cpp (incl. 7B on a 16 GB host). |
| Two-Mac distributed serve | ✅ Working | Layer sharding over TCP for rows too large for one 16 GB host (Gemma 4 12B, 26B-A4B). |
| Other quantizations | ⛔ Not supported | Fail closed in v0.1. |
| Ghost mode (layer streaming) | 🧪 Experimental | `ghost-run` executes one block at a time for a strict memory ceiling; trades throughput for memory. |

---

## Gemma 4

Camelid implements Gemma 4 from scratch in the `gemma4` engine: per-layer-type sliding/global attention (the GGUF `sliding_window_pattern` is authoritative; E2B is 4:1), per-layer FFN widths and KV-head counts, QK-norm, dual-θ RoPE, GeGLU, Per-Layer-Embeddings, cross-layer KV sharing, and the `<|turn>`/`<turn|>` chat markers with thinking-channel suppression. Multimodal input fails closed with a typed error.

**E2B-It & E4B-It (Q8_0, single-node).** Five-prompt greedy parity against the pinned llama.cpp oracle on **both** the CPU and the Metal GPU-resident runtime, plus **checked bounded context packs at 512 / 1024 / 2048 / 4096 / 8192** (recall-style, oracle recall asserted at capture — full-budget CPU+GPU passes at every bucket, no recorded frontiers). The chat template is locked byte- and token-exact (`qa/gemma4/template_shapes_v1.json`, both thinking modes). A Metal GPU-resident decode path (`camelid gemma4-generate-gpu`) runs the full E4B forward on the GPU at the memory-bandwidth wall. A QAT Q4_0/Q6_K wire lane is committed parity (E4B QAT basic_v1: 3/5 full-budget + 2 probe-verified frontiers).

**12B-It (Q8_0) & 26B-A4B-It QAT (Q4_0, MoE) — two-Mac distributed.** These rows are too large for a single 16 GB host, so the supported lane is distributed layer sharding over TCP: `gemma4-master`/`gemma4-worker` split one row across two machines with a versioned handshake and per-packet checksums, and distributed greedy output is asserted token-identical to single-node (`tests/gemma4_distributed_parity.rs`). The 26B row is a 128-expert MoE (Q4_0 experts + Q6_K tied head) with the dense shared-expert + sparse top-8 branch implemented end to end.

Proven on two 16 GB M4 Mac minis, full `basic_v1` pack vs the pinned reference:

| Row | Distributed = single-node | vs. reference |
|---|---|---|
| 12B-It Q8_0 | 5/5 token-identical | 3/5 full-budget + recorded comparator frontiers |
| 26B-A4B-It QAT Q4_0 | identical (f32 wire) | 2/5 full-budget token-identical + 3/5 probe-verified knife-edge frontiers |

Both rows serve over HTTP through the same lane — set `CAMELID_GEMMA4_SERVE=1` plus `CAMELID_GEMMA4_WORKER`/`CAMELID_GEMMA4_SPLIT`, and `/v1/chat/completions` (incl. SSE) and `/v1/completions` route through a persistent master shard with per-request worker sessions (wire protocol v1). The distributed serve/WebUI promotion smoke is green for both. Evidence bundles are under [`qa/evidence-bundles/`](qa/evidence-bundles/); setup is in [`docs/gemma4-two-mac-cluster.md`](docs/gemma4-two-mac-cluster.md).

> Scope guardrails: these are exact-row claims only — no Gemma-family-wide support, and no model-native/larger context beyond the checked packs.

---

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

Chat (replace the model `id` with the one returned above; add `"stream": true` for SSE):

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

Run the local web frontend:

```bash
cd frontend && npm ci && npm run dev
```

---

## Evidence

Benchmark claims are listed only when raw logs or reproducible commands are committed. If there is no raw log, there is no benchmark claim.

Same-host snapshot on one Apple M4 (10-core GPU, 16 GB), Llama 3.2 3B Instruct Q8_0, greedy sampling, three same-session rounds with alternating runtime order (medians):

| Lane | Camelid | llama.cpp (Metal) | MLX-LM (8-bit) |
| --- | ---: | ---: | ---: |
| Prefill, 601-token prompt (tok/s) | **587.3** | 543.7 | 577.9 |
| Decode, short context (tok/s) | **29.7** | 29.1 | 29.1 |

> **Reading boundary:** a same-session result on one exact model row and one machine, with narrow margins — not a durable or general claim. Some lanes read below the comparators (decode at long context trails MLX-LM), and deeper prompt depths use single warm probes rather than protocol-grade rounds. Full methods, raw logs, per-round detail, and the lanes where Camelid loses are in [`BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md) and the bundles under [`qa/evidence-bundles/`](qa/evidence-bundles/).

Correctness evidence (token-parity gates, per-row validation artifacts) is indexed in [`COMPATIBILITY.md`](COMPATIBILITY.md) and [`CORRECTNESS_v0.1.md`](docs/release/CORRECTNESS_v0.1.md).

### Parity receipts

A parity receipt is a verifiable record of one request: the exact GGUF (by SHA-256), the exact input, and the exact tokens produced. Opt in with `"camelid_receipt": true` on `/v1/chat/completions` or `/v1/completions`, then check it on any machine:

```bash
camelid verify-receipt receipt.json --gguf path/to/exact-model.Q8_0.gguf
```

The verifier recomputes the receipt's digest, confirms your GGUF is the named file, replays the request through Camelid, and re-runs it through llama.cpp — in two isolated passes so each model loads within one model's memory footprint, which lets a 7B receipt verify on a 16 GB Mac. Receipts exist only for deterministic (greedy) runs; sampled runs are stamped `reproducible: false`. **A receipt verifies a single request; it does not change the release ledger or promote any lane.** Details in [`RECEIPTS.md`](RECEIPTS.md).

To measure *any* local runtime — not only Camelid — by determinism, cross-runtime agreement, tokenizer parity, and provability on the same model bytes, see the [conformance suite](docs/CONFORMANCE.md).

---

## Documentation

| Doc | What's in it |
|---|---|
| [`SUPPORT_MATRIX_v0.1.md`](SUPPORT_MATRIX_v0.1.md) | Which exact model rows are supported, and with what evidence |
| [`COMPATIBILITY.md`](COMPATIBILITY.md) | The durable support contract |
| [`BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md) | Benchmark snapshots and claim rules |
| [`RECEIPTS.md`](RECEIPTS.md) | Verifiable single-request parity receipts |
| [`docs/CONFORMANCE.md`](docs/CONFORMANCE.md) | Measure any runtime by one ruler |
| [`STATUS.md`](STATUS.md) | Current evidence snapshot and blockers |
| [`ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md) | Implementation architecture |
| [`docs/gemma4-two-mac-cluster.md`](docs/gemma4-two-mac-cluster.md) | Two-Mac distributed serve setup |
| [`RELEASE_NOTES_v0.1.md`](docs/release/RELEASE_NOTES_v0.1.md) | v0.1 release notes |
| [`ROADMAP.md`](ROADMAP.md) | Planned engineering sequence |

Validation for code changes:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

---

## License

Camelid is licensed under the [MIT License](LICENSE).

Camelid's tokenizer, reference compatibility layouts, and validation benchmarks are inspired by and checked against [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (Copyright © 2023–2026 The ggml authors, MIT License). Camelid maintains its own Rust-native codebase while crediting the reference work of the `ggml` ecosystem.

[ci-badge]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml
