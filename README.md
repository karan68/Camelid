# Camelid

[![CI][ci-badge]][ci-workflow]

![Camelid banner](assets/camelid-banner.png)

**Camelid is a Rust-native local inference backend for GGUF language models, built for teams that want local performance, clear support boundaries, and evidence they can inspect.**

Many local-model stacks are easy to demo and hard to trust. Camelid closes that gap with disciplined support claims, clear readiness signals, and a product surface that stays aligned across runtime, API, UI, and documentation.

Camelid does not treat “probably works” as “supported.” Support moves on evidence.

> **Current public posture:** Camelid achieves 1:1 parity with llama.cpp for five exact GGUF rows within bounded published validation envelopes: TinyLlama at the current validated gate; Llama 3.2 1B/3B Q8_0 through checked bounded 512/1024/2048-context packs; Llama 3 8B Q8_0 through exact-row smoke plus checked bounded 512/1024/2048-context packs on current `main`; and `Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf` through the checked short-prompt MoE/API/WebUI/RSS envelope. The Mixtral row is exact-row supported only after refreshed six-prompt 5-token parity, OpenAI-compatible API smoke, WebUI readiness, RSS/timing, and scrubbed manifest/checksum evidence passed. `Mistral-7B-Instruct-v0.3.Q8_0.gguf` remains an active exact-row bring-up lane, not a supported row. These are exact-row bounded claims only; wider model-native context beyond the checked packs, production throughput, portability, arbitrary templates, neighboring rows, and broad-family behavior remain outside the claim.

## Milestone at a glance

Camelid's current milestone is a synchronized product surface: backend runtime, OpenAI-compatible API, WebUI readiness, docs, and durable evidence all point at the same support contract.

- **Five exact Q8_0 rows are public and evidence-backed.** TinyLlama remains the full current gate; Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B are supported within validated bounds; `Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf` is exact-row supported for the checked short-prompt MoE/API/WebUI/RSS envelope. Mistral 7B Instruct v0.3 remains fail-closed as an active bring-up lane until its current-head promotion checklist is complete.
- **The UI and API fail closed instead of guessing.** Chat unlocks only when the loaded local GGUF is `loaded_now=true`, `generation_ready=true`, and matched to an exact supported `/api/capabilities` row.
- **The context ladder is explicit.** The four supported Llama/TinyLlama rows have checked bounded 512-context evidence; Llama 3.2 1B/3B and Llama 3 8B also have checked 1024 and 2048 packs. The Mixtral exact row is checked only in the short-prompt MoE/API/WebUI/RSS envelope; no longer-context bucket is promoted for Mixtral.

![Camelid WebUI chat surface](docs/assets/camelid-readme-chat-surface-dark.png)

The WebUI screenshot above is intentionally simple and product-forward while still reflecting the local-first runtime contract. It is not a broad model-family or arbitrary-GGUF support claim.

## Current work tracks

Camelid is advancing on two tracks:

- **Supported-row hardening:** preserve TinyLlama as the full current gate, move the verified Llama rows and the Mixtral exact row toward the same normalized bar, and keep support wording tied to green evidence. The remaining gaps are broader context, arbitrary/Jinja template coverage, production-throughput evidence, portability, and repeated current-head bundles.
  - Mixtral continuation hardening is active: the checked exact-row short-prompt envelope is still green, and the continuation path now accepts exact prompt-token reuse for repro/debug work, but long-generation and wider-prompt claims remain blocked until separate evidence closes that lane.
- **Active next-model bring-up:** Mistral 7B Instruct is the lead exact-row bring-up lane; Qwen 2.5 7B Instruct and Gemma 2 9B Instruct remain planned exact-row candidates.
  - `Mistral-7B-Instruct-v0.3.Q8_0.gguf` — active validation only: source/SHA, exact tokenizer/template references, 1-token generation parity, broader five-prompt/50-token parity, bounded 512/1024/2048, and checked 4096/8192 context evidence now exist. Support remains fail-closed pending row-specific API/WebUI/RSS evidence and synchronized capability/frontend/docs wording. No Mistral-family, neighboring-row, arbitrary-template, production-throughput, portability, or full-support claim is implied.
  - `Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf` — exact-row supported for the checked short-prompt MoE/API/WebUI/RSS envelope: refreshed six-prompt 5-token parity against llama.cpp, OpenAI-compatible API smoke, WebUI readiness, RSS/timing, and scrubbed manifest/checksum evidence are documented. Neighboring rows, broad-family behavior, long context, arbitrary templates, production throughput, portability, and broader/full support remain unclaimed.
  - `Qwen2.5-7B-Instruct-Q8_0.gguf` — planned exact-row candidate; tokenizer/template and architecture mapping still need row-specific proof.
  - `gemma-2-9b-it-Q8_0.gguf` — planned exact-row candidate; tokenizer/template and Gemma2 runtime behavior still need row-specific proof.
  - Public wording for the Mixtral row is limited to “exact-row supported” / “validated exact row” inside the checked short-prompt MoE/API/WebUI/RSS envelope. Qwen and Gemma remain “not supported yet” until each row has its own source/SHA/license, tokenizer/template, parity, bounded load/readiness, API/WebUI, RSS/timing, scrubbed manifest, and checksum evidence. See [`COMPATIBILITY.md`](COMPATIBILITY.md#locked-next-family-readiness-language) for the row-by-row promotion checklist.

Roadmap items such as multi-model concurrency remain roadmap items, not current support.

## Why Camelid matters

Most local-model stacks emphasize broad compatibility before they can explain what is truly production-ready. Camelid takes the opposite approach.

It is for teams and builders who need clear answers to a few practical questions:

- **What exactly works?**
- **What evidence backs that claim?**
- **Will the API and UI tell the same truth?**
- **Can we widen support without hand-waving?**

That discipline is not just a docs style. It is the product.

## What makes it different

Camelid gives you:

- a Rust server with OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
- GGUF metadata and tensor parsing, tokenizer binding, and typed unsupported-state errors
- exact-row capability reporting through `/api/capabilities`
- a React/Vite WebUI that unlocks chat only when runtime readiness and support-contract readiness agree
- parity and validation harnesses used to compare behavior against llama.cpp before support language moves

**Naming note.** Camelid is the product name. The repository, crate, binary, and diagnostics use `camelid`.

**Reference credit.** Camelid is original Rust code and keeps visible credit for the reference work behind tokenizer checks, compatibility baselines, and parity evidence. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for current acknowledgements and MIT-license notices, including llama.cpp / ggml.

## Support matrix

Camelid’s public support boundary is intentionally narrow and exact-row. Read each row literally; nothing adjacent inherits support.

| Exact lane | Public status | Green evidence today | Crisp caveat |
| --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | **Verified support** | End-to-end generation, broader five-prompt/50-token parity, bounded template-shape checks, bounded 512-context coverage, and backend RSS/perf sampling. | This is the trusted current gate, not a promise about other TinyLlama variants or quants. |
| Llama 3.2 1B Instruct Q8_0 | **Verified support (bounded)** | Load, completions, chat completions, WebUI validation, compact/broader parity, bounded template-shape checks, bounded unique-chat perf/RSS, and checked 512/1024/2048-context packs. | The 2048 pass is exact-row only after the RoPE frequency-factor fix; it is not model-native/larger-context, arbitrary-template, production-throughput, or portability support. |
| Llama 3.2 3B Instruct Q8_0 | **Verified support (bounded)** | Load, completions, chat completions, WebUI validation, compact/broader 50-token parity, bounded template-shape checks, bounded unique-chat perf/RSS, checked 512/1024/2048-context packs, and an opt-in parallel Q8 first-token direction probe. | The parallel Q8 result is a direction probe, not production throughput; broader/full support still needs model-native/larger context, arbitrary-template, and portability evidence. |
| Llama 3 8B Instruct Q8_0 | **Verified support (bounded)** | Load, completions, chat completions, WebUI validation, compact parity, three-prompt 50-token parity, checked 512/1024/2048-context packs, compact chat-template-shapes pack, bounded memory evidence, structured RSS/Q8 file-read counters, and lazy-Q8 hot-path measurements. | The 1024/2048 pass is exact-row/current-head only via `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json`; no model-native/larger context beyond checked packs, production throughput, arbitrary templates, portability, neighboring-row, or broad 8B/Llama support is implied. |
| Mistral-7B-Instruct-v0.3.Q8_0.gguf | **In active validation; not supported yet** | Source/SHA, exact tokenizer/template references, 1-token generation parity, broader five-prompt/50-token parity, bounded 512/1024/2048 bring-up, and checked 4096/8192 context validation are green; latest context bundle: `qa/evidence-bundles/mistral-7b-v0.3-q8-context-4096-8192-ubuntu-20260509T005229Z-head-9e3c64f2cfab/manifest.json`. | No Mistral-family support, neighboring variants, arbitrary templates, model-native/larger context, API/WebUI readiness, production throughput, portability, or full support until row-specific readiness evidence and surfaces land. |
| Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf | **Verified support (bounded exact row)** | Refreshed six-prompt 5-token parity, OpenAI-compatible API smoke, WebUI readiness, RSS/timing, and scrubbed manifest/checksum evidence. Key bundles: `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-backend-parity-refresh-20260511/`, `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-api-smoke-20260511/`, `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-webui-readiness-20260511/`, `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-rss-timing-runtime-20260511/`, and `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-manifest-checksum-20260511/`. | Exact row and checked short-prompt/API/WebUI/RSS envelope only; no neighboring-row, long-context, arbitrary-template, production-throughput, portability, or broader/full claim. |
| Qwen2.5-7B-Instruct-Q8_0.gguf | **Planned exact-row candidate** | Candidate row selected for acquisition/tokenizer planning only. | No Qwen support until tokenizer/template references, architecture mapping, bounded load, parity, API/WebUI, RSS, and bundle evidence exist. |
| gemma-2-9b-it-Q8_0.gguf | **Planned exact-row candidate** | Candidate row selected for acquisition/tokenizer planning only. | No Gemma support until tokenizer/template references, architecture mapping, bounded load, parity, API/WebUI, RSS, and bundle evidence exist. |

### Latest bounded model checks

The maintainer matrix now includes five exact Q8_0 supported rows with checked row-specific evidence. Mistral remains an active validation row, not a supported row. These are bounded checks, not universal model claims.

| Exact row | Latest checked bucket | Result | Output checked |
| --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | direct chat validation | PASS | `Certainly! Here` |
| Llama 3.2 1B Instruct Q8_0 | 2048-context bounded recall pack | PASS | `CMLD-204` |
| Llama 3.2 3B Instruct Q8_0 | 2048-context bounded recall pack | PASS | `CMLD-204` |
| Llama 3 8B Instruct Q8_0 | 2048-context bounded recall pack | PASS on current `main` | `CMLD-204` |
| Mistral-7B-Instruct-v0.3.Q8_0.gguf | Active validation: Ubuntu bounded bring-up plus exact tokenizer/template, 1-token, broader five-prompt/50-token parity, and checked 4096/8192 context evidence | PASS for validation lane; still unsupported | API/WebUI/RSS readiness and synchronized support surfaces still required before any support claim |
| Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf | Exact-row promotion gates: six short prompts at max_tokens=5, API smoke, WebUI readiness, RSS/timing, and manifest/checksum | PASS for the checked exact-row envelope | No neighboring-row, long-context, arbitrary-template, production-throughput, portability, or broader/full claim |

### Read this boundary carefully

- Support does **not** inherit across model sizes, variants, quantizations, tokenizer lanes, or nearby GGUFs.
- Support language currently means only the exact supported rows above; Mistral has no support claim yet and may only be discussed as the exact active validation row above. The Mixtral row may only be described as exact-row supported inside its checked short-prompt MoE/API/WebUI/RSS envelope.
- Checked context packs do **not** imply model-native or broader context support.
- Bounded template-shape or perf evidence does **not** imply arbitrary template execution or production portability.
- The next major 8B, Mistral, and Mixtral-row gaps are broader context, arbitrary templates, production throughput, portability, repeated durability evidence, and normalized support bundles; Mistral also still needs row-specific API/WebUI/RSS readiness, and the Mixtral exact row still needs separate longer-context, broader-prompt, and stable long-generation continuation evidence before widening the checked envelope.

Authoritative details live in [`COMPATIBILITY.md`](COMPATIBILITY.md). The current evidence snapshot lives in [`STATUS.md`](STATUS.md).

## Start here

- [`COMPATIBILITY.md`](COMPATIBILITY.md) — authoritative support matrix and promotion rules
- [`STATUS.md`](STATUS.md) — current evidence boundary, recent moves, and blockers
- [`docs/CONTRIBUTOR_QUICKSTART.md`](docs/CONTRIBUTOR_QUICKSTART.md) — shortest safe local path
- [`docs/VALIDATION_MATRIX.md`](docs/VALIDATION_MATRIX.md) — expected checks by change type
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution expectations and PR guidance
- [`DOCS.md`](DOCS.md) — full documentation map

## Quickstart

This quickstart verifies that Camelid builds and the backend starts on your machine. It is **not** a one-command chat demo: the repository does not bundle supported GGUF model files, and end-to-end local chat requires additional setup.

### 1) Build and run the server

```bash
git checkout main
git pull --ff-only
cargo build --release --bin camelid
target/release/camelid serve --addr 127.0.0.1:8181
```

Toolchain note: Camelid currently requires Rust/Cargo 1.87+. If your host exposes an older system `cargo`, use the rustup-managed toolchain described in [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

### 2) Verify the server is responding

From another shell:

```bash
curl -s http://127.0.0.1:8181/api/capabilities
```

That confirms the backend is responding.

### 3) Before you expect local chat to work

You will need:

- a supported GGUF model file already present on your machine
- the model path wired into a load request you control
- any extra contributor setup described in [`docs/CONTRIBUTOR_QUICKSTART.md`](docs/CONTRIBUTOR_QUICKSTART.md) and [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)

For a first supported local run, TinyLlama is the clearest path — but it is **not bundled** in this repository, and this README is not a copy-paste chat demo.

## Frontend

Camelid includes a built-in React/Vite frontend in [`frontend/`](frontend/) so the local runtime ships with a real product surface, not just a backend API.


The UI is designed to feel straightforward and approachable while still staying honest about model readiness. It keeps the main path simple — pick a local model, see whether Camelid reports it as ready, and start chatting when the runtime and support contract agree.

A few principles define the WebUI:

- **Chat-first and intuitive:** the interface emphasizes the primary action instead of burying it in operator-only controls.
- **Honest readiness signals:** chat only unlocks when the loaded model is runtime-ready and matched to an exact supported support-contract row.
- **One product story across surfaces:** the backend, API, UI, and docs are intended to agree instead of sending mixed signals about what is actually supported.

```bash
cd frontend
npm ci
npm run dev
```

By default, the UI talks to `http://127.0.0.1:8181` and only unlocks local chat when the loaded model is both runtime-ready and covered by the current support contract.

See [`frontend/README.md`](frontend/README.md) for frontend-specific details.

## How support moves

A row is promoted only when all of these agree for the exact lane being claimed:

1. runtime behavior
2. artifact-backed validation
3. documentation
4. API capability reporting
5. frontend readiness behavior

That is Camelid’s core discipline: **evidence first, broader claims later**.

## Contributing

If you want to contribute, start with the docs written for safe local iteration and contributor onboarding:

- [`docs/CONTRIBUTOR_QUICKSTART.md`](docs/CONTRIBUTOR_QUICKSTART.md)
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)
- [`docs/VALIDATION_MATRIX.md`](docs/VALIDATION_MATRIX.md)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)

Camelid intentionally separates the public support contract, the local contributor path, and maintainer-only evidence workflows. That keeps the project welcoming without leaking operator-only details or pretending every internal validation lane is part of normal onboarding.

## Validation

Use [`docs/VALIDATION_MATRIX.md`](docs/VALIDATION_MATRIX.md) to pick the smallest correct validation lane for your change.

Common baseline checks:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo doc --no-deps --all-features
bash scripts/check-public-scrub.sh
```

For docs-only changes, the minimum expected checks are:

```bash
git diff --check
bash scripts/check-public-scrub.sh
```

If your change affects the frontend, also run:

```bash
cd frontend
npm ci
npm run build
```

## Documentation map

- [`DOCS.md`](DOCS.md) — full documentation index
- [`COMPATIBILITY.md`](COMPATIBILITY.md) — support ledger
- [`STATUS.md`](STATUS.md) — current evidence snapshot
- [`ROADMAP.md`](ROADMAP.md) — delivery plan of record
- [`FULL_SUPPORT_BLOCKER_MATRIX.md`](FULL_SUPPORT_BLOCKER_MATRIX.md) — row-by-row missing evidence for broader promotion
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — architecture and module planning
- [`DECISIONS.md`](DECISIONS.md) — design decision log

## License and acknowledgements

Camelid is licensed under the [MIT License](LICENSE).

Camelid is inspired by and validated against [`llama.cpp`](https://github.com/ggml-org/llama.cpp), which is licensed under the MIT License:

> Copyright (c) 2023-2026 The ggml authors

The llama.cpp project and the broader GGUF ecosystem made the modern local-model path practical. Camelid keeps its runtime implementation Rust-native while intentionally crediting llama.cpp wherever reference comparisons, tokenizer fixtures, and parity gates rely on it.

[ci-badge]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml
