# Camelid

[![CI][ci-badge]][ci-workflow]

![Camelid banner](assets/camelid-banner.png)

Camelid is a Rust-native local inference backend for GGUF language models. It is being built for teams that want a local-model stack they can audit, explain, and operate with confidence: exact support rows, reproducible validation artifacts, and release language that stays inside the evidence.

Many local-inference projects optimize for breadth of apparent compatibility. Camelid optimizes for trust. A lane is not "supported" because metadata parses, a tokenizer round-trips, or a prompt partially works once. Camelid promotes a lane only when runtime behavior, API capability reporting, frontend readiness, documentation, and artifact-backed validation all agree for the exact model, tokenizer path, and quantization being claimed.

**Naming note.** Camelid is the product name. The repository, crate, binary, some API diagnostics, and several scripts still use `backendinference` during the transition. Keep current commands, package identifiers, and tests on those names until a separate rename plan is validated.

**Reference-credit note.** Camelid is original Rust code, and it keeps visible credit for the reference work behind tokenizer checks, compatibility baselines, and parity evidence. In particular, llama.cpp / ggml remains explicitly acknowledged here and in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) because Camelid still relies on it as a MIT-licensed inspiration, tokenizer reference, compatibility baseline, and parity benchmark.

## What Camelid is today

Camelid is a local GGUF inference server with a deliberately small public support line and a growing validation bench behind it. The goal is simple: make local models feel boringly dependable — load the exact row, expose honest readiness, and generate through an API and WebUI without pretending nearby models are supported.

The current product slice includes:

- a Rust server with OpenAI-compatible completions and chat endpoints
- GGUF metadata/tensor parsing, tokenizer binding, and typed unsupported-state errors
- exact-row capability reporting through `/api/capabilities`
- a React/Vite WebUI that only unlocks chat when runtime readiness and support-contract readiness both agree
- parity harnesses against llama.cpp so support moves by evidence, not vibes

## Current support line

| Exact lane | Public posture | Evidence today | Still needed |
| --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | Supported current gate | Five 50-token prompt audits match known-good llama-server prompt token IDs, generated token arrays, and generated text. | No implied support for adjacent TinyLlama quantizations or other model families. |
| Llama 3.2 1B Instruct Q8_0 | Supported exact-row smoke | Exact GGUF load, `/v1/completions`, `/v1/chat/completions`, frontend smoke, compact parity, and broader prompt-pack evidence support short local chat for this 1B Instruct Q8_0 row only. | Longer context, broader chat-template behavior, performance/portability evidence, and no implied support for neighboring rows. |
| Llama 3.2 3B Instruct Q8_0 | Supported exact-row smoke | Exact GGUF load, `/v1/completions`, `/v1/chat/completions`, frontend smoke, compact prompt-token/1-token/5-token/50-token parity, and the broader three-prompt 50-token pack now match llama.cpp for this 3B Instruct Q8_0 row only. | Longer context, stronger perf/portability evidence, and no implied support for neighboring rows. |
| Llama 3 8B Instruct Q8_0 | Groundwork only with compact parity validation | Compact-header `hello` matches llama.cpp for prompt-token, 1-token, 5-token, and bounded 50-token parity, with basic API smoke and bounded-memory evidence. | Broader prompt/chat-template parity, WebUI readiness, performance envelope, and portable packaging evidence. |

TinyLlama and the exact Llama 3.2 1B/3B Instruct Q8_0 rows are the supported generation lanes today, with the Llama rows intentionally limited to a short local-chat smoke envelope. Broader Llama 3-family support has not been promoted, and nothing adjacent inherits support.

## Start here

For most readers, the fastest path through the repo is:

1. [`COMPATIBILITY.md`](COMPATIBILITY.md) — the authoritative support ledger.
2. [`STATUS.md`](STATUS.md) — the current evidence boundary, blocker state, and artifact references.
3. [`ROADMAP.md`](ROADMAP.md) — the ordered path to the next support expansion.
4. [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — the reference-tool and MIT-license notices behind Camelid's public evidence trail.

## How support moves

Camelid promotes a row only when runtime behavior, API capability reporting, frontend readiness, docs, and artifact-backed validation all agree for the exact model, tokenizer path, and quantization being claimed.

That keeps the product honest without making the front door miserable: run what is supported, inspect what is being validated, and treat every promotion as an exact-row release decision.

## What Camelid can prove today

### TinyLlama 1.1B Chat Q8_0

Phase 7 real-model hardening is complete for the current TinyLlama Q8_0 gate. The key correctness guardrail is the token-major `output.weight` interpretation required by the GGUF storage layout.

Current validated facts:

- TinyLlama-style untied `output.weight` descriptors report dimensions `[hidden, vocab]`, but the stored Q8_0 payload is consumed as contiguous token rows for final logits.
- The `hello` chat first token now selects token `29907` / `"C"` at rank 1.
- The 50-token `hello` stream starts `[29907, 13946, 368, 29991, ...]` and decodes to text beginning `"Certainly! Here are some examples..."`.
- The current five-prompt gate matches known-good llama-server prompt token IDs, generated token arrays, and generated text for 50 completion tokens.

Current gate artifacts:

- `target/edge-prompt-audit-fixed-20260428T1530/short.json`
- `target/edge-prompt-audit-fixed-20260428T1530/trailing-spaces.json`
- `target/edge-prompt-audit-fixed-20260428T1530/special-chars.json`
- `target/edge-prompt-audit-fixed-20260428T1530/longer.json`
- `target/edge-prompt-dequant-default-20260428T1604/multiline-long-default-50.json`

The separate fixed-audit `multiline` row also matches, but it stops at EOS after 41 tokens, so it is not counted as one of the five 50-token gate artifacts.

### Adjacent Llama 3-family rows

Camelid now has two exact Llama 3.2 smoke-supported rows. These are deliberately narrow release claims: they require the exact Instruct Q8_0 row, a loaded runtime with `generation_ready=true`, and the short local-chat envelope validated by the API and frontend smoke harnesses.

- **Llama 3.2 1B Instruct Q8_0:** exact-row load, completions, chat-completions, frontend smoke, compact parity, and the broader prompt pack are validated for this 1B Instruct Q8_0 row. This does not promote neighboring Llama sizes, base variants, other quantizations, longer contexts, or broad chat-template behavior.
- **Llama 3.2 3B Instruct Q8_0:** exact-row load, completions, chat-completions, frontend smoke, compact prompt-token/1-token/5-token/50-token parity, and the broader three-prompt 50-token pack are validated for this 3B Instruct Q8_0 row. The previous JSON-shaped prompt divergence was fixed by the file-backed Q8_0 dot-parity correction; remaining follow-up is longer-context, performance, portability, and broader chat-template coverage. See [`QA_LLAMA32_3B_Q8_ACCEPTANCE.md`](QA_LLAMA32_3B_Q8_ACCEPTANCE.md) and [`STATUS.md`](STATUS.md) for the exact evidence boundary.
- **Llama 3 8B Instruct Q8_0:** the exact tracked Q8_0 GGUF now matches llama.cpp on Ubuntu for the compact-header `hello` harness at prompt-token, deterministic 1-token, deterministic 5-token, and bounded 50-token parity, with basic API smoke and bounded-memory evidence on top of the earlier metadata/config/tokenizer/template and lazy/file-backed Q8 groundwork. That is still groundwork, not a support promotion: Camelid does not claim supported generation, broader 8B prompt/chat-template parity, WebUI readiness, a performance envelope, or portable packaging for this row yet.

Fresh tokenizer revalidations and standalone Q8 block benchmarks are seam evidence only. They do not, by themselves, expand a generation-support claim.

## Current product surface

Within that deliberately narrow support contract, Camelid already exposes a coherent local backend product slice today:

- Rust CLI/server with `/health` and `/v1/health`
- GGUF metadata parsing, tensor descriptor parsing, alignment and bounds validation, and malformed-fixture coverage
- local model load/current/metadata/tokenizer/capability endpoints under `/api/*`
- OpenAI-compatible `/v1/models`, `GET /v1/models/:model`, `/v1/completions`, and `/v1/chat/completions` for supported loaded dense GGUF models
- non-streaming JSON responses and OpenAI-style SSE streaming chunks
- LLaMA/SPM tokenizer metadata loading plus encode/decode endpoints
- CPU `f32` tensor loading/conversion for F32, F16, BF16, and Q8_0, with Q8_0 retained-block groundwork for future lazy execution
- dense decoder binding, KV-cache planning, RoPE, causal KV attention, one-token-at-a-time CPU generation, greedy/sampled controls, stop sequences, and exact-prompt prefix reuse for repeated requests
- typed unsupported or invalid-state errors for unsupported tokenizer/model families, unsupported quantizations, unsafe materialization, multi-choice generation, and unimplemented logprob fields
- a React/Vite frontend in [`frontend/`](frontend/) that enables local chat only when the loaded model is both runtime-ready and covered by an exact supported compatibility row

## Run the supported path today

Build the current binary, start the server, and load a local TinyLlama Q8_0 GGUF:

```bash
git checkout main
git pull --ff-only
cargo build --release --bin backendinference
target/release/backendinference serve --addr 127.0.0.1:8181
```

In another shell from the repository root:

```bash
curl -s http://127.0.0.1:8181/api/models/load \
  -H 'content-type: application/json' \
  -d '{"id":"tinyllama-q8","path":"models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"}'

curl -s http://127.0.0.1:8181/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"tinyllama-q8","messages":[{"role":"user","content":"hello"}],"max_tokens":50,"temperature":0}'
```

Expected current behavior: the first generated token for `hello` is `29907` / `"C"`, and the 50-token smoke completes with ordinary text.

## Reproduce the supported-lane parity audit

Start from a clean release build. Ensure a `llama-server` binary is in `PATH`, or pass `--llama-server <path>`:

```bash
node scripts/chat-parity-tinyllama.mjs \
  --backend http://127.0.0.1:8181 \
  --llama-url http://127.0.0.1:8183 \
  --model models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --model-id tinyllama-q8 \
  --start-llama-server \
  --max-tokens 50 \
  --diagnostics-out target/chat-parity-postfix-50-token-audit.json
```

Current post-fix result: `prompt_tokens_match=true`, `generated_text_match=true`, `completion_tokens=50` on both sides, and `first_divergent_generated_token_index=-1`.

## Output projection layout guardrail

The token-major `output.weight` behavior is a GGUF file-layout requirement, not a macOS/ARM-specific workaround. Windows, Linux, Intel, and ARM builds must preserve the same interpretation.

```bash
target/release/backendinference tensor-dump \
  models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --tensor output.weight --token 29907 --token 29903 --token 8241 --token 28651 \
  > target/output-projection-layout-check.json

node scripts/check-output-projection-layout.mjs \
  --tensor-dump target/output-projection-layout-check.json
```

Expected output includes `output_projection_layout_ok=true`, `gguf_dimensions=[2048,32000]`, `storage_row_stride_bytes=2176`, logical token rows with `stride=1`, and descriptor columns with `stride=32000` only as comparison evidence.

## Inventory and readiness gates

Start with a local inventory pass so new GGUFs are discovered without loading them or inferring support from filenames:

```bash
node scripts/small-model-inventory.mjs \
  --out target/small-model-inventory.json \
  --markdown-out target/small-model-inventory.md
```

Then run the manifest-driven readiness gate before starting reference servers or parity runs for new local GGUFs:

```bash
node scripts/small-model-readiness.mjs \
  --out target/small-model-readiness.json \
  --markdown-out target/small-model-readiness.md
```

The readiness gate inspects each present manifest GGUF with `backendinference inspect`, binds LLaMA metadata/tensor shapes, chooses the current tokenizer/template lane, estimates eager `f32` plus retained-source CPU materialization against `BACKENDINFERENCE_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES`, and reports whether the existing TinyLlama or Llama 3 parity harness is safe to run. A `load_and_generation_candidate` row is only an inventory/readiness result; support still needs target-specific deterministic parity, API, and frontend evidence. The exact 3B WebUI row is tracked separately in [`QA_LLAMA32_3B_Q8_ACCEPTANCE.md`](QA_LLAMA32_3B_Q8_ACCEPTANCE.md); its short local-chat smoke and broader three-prompt parity pack are supported, while longer contexts, performance, portability, and broader chat-template coverage remain follow-up gates.

## Frontend

A React/Vite frontend lives in [`frontend/`](frontend/). It targets Camelid's current local API surface (`/v1/health`, `/v1/models`, `/api/models/current`, `/api/models/load`, `/api/capabilities`, and `/v1/chat/completions`) and keeps unsupported hosted, catalog, and provider features visibly planned instead of presenting them as runnable.

```bash
cd frontend
npm install
npm run dev
```

By default the UI talks to `http://127.0.0.1:8181`. Use `cd frontend && npm run smoke:tiny` while the backend and Vite dev server are running to load a generated tiny GGUF fixture and test the local chat path end to end. See [`frontend/README.md`](frontend/README.md) for details.

## Validation commands

Run these before pushing meaningful code changes:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo doc --no-deps --all-features
```

For docs-only changes, at minimum run:

```bash
git diff --check
```

## Documentation map

- [`DOCS.md`](DOCS.md) — documentation index and reading order.
- [`COMPATIBILITY.md`](COMPATIBILITY.md) — evidence-based support matrix and promotion checklist.
- [`ROADMAP.md`](ROADMAP.md) — phase gates, active priorities, and next support-changing milestones.
- [`STATUS.md`](STATUS.md) — current evidence snapshot and promotion blockers.
- [`FULL_SUPPORT_BLOCKER_MATRIX.md`](FULL_SUPPORT_BLOCKER_MATRIX.md) — four-row full-support owner matrix with exact missing evidence by row.
- [`QA_LLAMA32_3B_Q8_ACCEPTANCE.md`](QA_LLAMA32_3B_Q8_ACCEPTANCE.md) — exact Llama 3.2 3B Q8_0 artifact path, parity/WebUI acceptance checklist, and current blocker.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — Rust architecture proposal and module boundaries.
- [`SAFETENSORS_PLAN.md`](SAFETENSORS_PLAN.md) — SafeTensors/Hugging Face model-source planning lane.
- [`TOKENIZER_RECON.md`](TOKENIZER_RECON.md) — tokenizer implementation notes.
- [`TENSOR_RECON.md`](TENSOR_RECON.md) — tensor/runtime implementation notes.
- [`INFERENCE_RECON.md`](INFERENCE_RECON.md) — inference engine implementation notes.
- [`SAMPLING_API_RECON.md`](SAMPLING_API_RECON.md) — sampling/API planning notes.
- [`ATTENTION_CHECKPOINTS.md`](ATTENTION_CHECKPOINTS.md) — attention checkpoint bundle schema and validator notes.
- [`FORGELOCAL_INTEGRATION.md`](FORGELOCAL_INTEGRATION.md) — integration planning notes.
- [`DECISIONS.md`](DECISIONS.md) — design decision log.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — third-party acknowledgements and current license notices, including llama.cpp / ggml.
- [`STATUS_ARCHIVE_2026-04.md`](STATUS_ARCHIVE_2026-04.md) — detailed historical status log.

## License and acknowledgements

Camelid is licensed under the [MIT License](LICENSE).

Camelid is inspired by and validated against [`llama.cpp`](https://github.com/ggml-org/llama.cpp), which is licensed under the MIT License:

> Copyright (c) 2023-2026 The ggml authors

The llama.cpp project and the broader GGUF ecosystem made the modern local-model path practical. Camelid keeps its runtime implementation Rust-native, but reference comparisons, tokenizer fixtures, and parity gates intentionally credit llama.cpp wherever it is used as the compatibility baseline. If Camelid distributes any copied or derived llama.cpp source, binaries, scripts, or substantial portions, the applicable llama.cpp MIT copyright and permission notice must remain with that distribution.

[ci-badge]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml
