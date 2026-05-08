# Camelid

[![CI][ci-badge]][ci-workflow]

![Camelid banner](assets/camelid-banner.png)

**Camelid is a Rust-native local inference backend for GGUF language models. It is built for teams that care about local performance, clear support boundaries, and evidence they can inspect.**

Many local-model stacks are easy to demo and hard to trust. Camelid is designed to close that gap with disciplined support claims, straightforward readiness signals, and a product surface that stays aligned across runtime, API, UI, and documentation.

Camelid does not treat “probably works” as “supported.” Support moves only when the evidence is real.

> **Current public posture:** Camelid achieves 1:1 parity with llama.cpp for four exact GGUF rows within a bounded published validation envelope: TinyLlama at the current validated gate, plus Llama 3.2 1B/3B Q8_0 and Llama 3 8B Q8_0 through checked bounded 512/1024/2048-context packs where cited. `Mistral-7B-Instruct-v0.3.Q8_0.gguf` remains an active exact-row bring-up lane, not a supported row: source/SHA and early tokenizer/template/1-token artifacts exist, but current-head promotion still stays fail-closed pending support-grade parity closure, broader prompt evidence, API/WebUI/RSS evidence, and a scrubbed bundle that includes verified GGUF size/SHA. The 8B bundle at `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260508T202751Z-head-86ad5390d265/manifest.json` remains exact-row bounded-pack evidence for its cited runtime/API/frontend head. These are exact-row bounded-pack claims only; wider model-native context, production throughput, portability, arbitrary templates, and broad-family support remain outside the claim.

## Milestone at a glance

Camelid's current milestone is not a loose compatibility demo. It is a synchronized product surface: backend runtime, OpenAI-compatible API, WebUI readiness, docs, and durable evidence all point at the same exact support contract.

- **Four exact Q8_0 rows are public and evidence-backed.** TinyLlama remains the full current gate; Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B have checked bounded support for their exact Q8_0 rows where row-specific PASS bundles are cited. Mistral 7B Instruct v0.3 remains fail-closed as an active bring-up lane until its current-head promotion checklist is complete.
- **The UI and API fail closed instead of guessing.** Chat unlocks only when the loaded local GGUF is `loaded_now=true`, `generation_ready=true`, and matched to an exact supported `/api/capabilities` row.
- **The context ladder is explicit.** TinyLlama stays at the current gate with bounded 512-context coverage; Llama 3.2 1B/3B/8B have checked bounded 512/1024/2048-context evidence where row-specific PASS bundles are cited. Mistral context evidence remains bring-up evidence only until the exact row is promoted through the support checklist. These checked packs do not imply model-native/larger context, arbitrary-template, production-throughput, portability, or broader-family support.

![Camelid WebUI chat surface](docs/assets/camelid-readme-chat-surface-dark.png)

The WebUI screenshot above is intentionally the dark, collapsed-rail chat surface: the chat entrypoint is simple and product-forward, while still reflecting the local-first runtime contract. It is not a broad model-family or arbitrary-GGUF support claim.

## Current work tracks

Camelid is advancing on two tracks, and both stay gated by CI plus artifact-backed support language:

- **Supported-row hardening:** preserve TinyLlama as the current full gate, move the Llama exact-row verified lanes toward the same normalized bar, and keep the Mistral exact-row bring-up lane fail-closed until its promotion checklist is green. The next promotable evidence remains model-native/larger-context behavior beyond checked packs, arbitrary/Jinja template coverage, production-throughput evidence, portability, and repeated current-head bundles. CI reliability is non-negotiable: no public support wording should move if the gate is red.
- **Active next-model bring-up set:** Mistral 7B Instruct v0.3 is the first next-family exact row under active validation, but it is not supported yet. Camelid is also publicly working on **Mixtral 8x7B Instruct**, **Qwen 2.5 7B Instruct**, and **Gemma 2 9B Instruct** as planned exact-row candidates.
  - `Mistral-7B-Instruct-v0.3.Q8_0.gguf` — active validation only: source/SHA, tokenizer/template, 1-token, and bounded Ubuntu artifacts exist, but current-head support remains fail-closed pending support-grade parity closure, broader prompt coverage, API/WebUI/RSS evidence, scrubbed bundle checksums, and verified GGUF size/SHA. No Mistral-family, neighboring-row, 50-token, arbitrary-template, production-throughput, portability, or full-support claim is implied.
  - `Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf` — planned first MoE exact-row candidate; expert routing and bounded load/parity work still need to be proven.
  - `Qwen2.5-7B-Instruct-Q8_0.gguf` — planned exact-row candidate; tokenizer/template and architecture mapping still need row-specific proof.
  - `gemma-2-9b-it-Q8_0.gguf` — planned exact-row candidate; tokenizer/template and Gemma2 runtime behavior still need row-specific proof.
  - Public wording for Mixtral, Qwen, and Gemma remains “not supported yet” until each row has its own source/SHA/license, tokenizer/template, parity, bounded load/readiness, API/WebUI, RSS/timing, scrubbed manifest, and checksum evidence. See [`COMPATIBILITY.md`](COMPATIBILITY.md#locked-next-family-readiness-language) for the family-by-family promotion checklist.
- **README frontend surface:** keep the built-in UI polished, honest, and product-ready so the runtime, API, docs, and WebUI all tell the same support story without implying family-wide, arbitrary-GGUF, portability, or production-throughput support.
- **Future multi-model concurrency:** add first-class support for running multiple local models at once so agents/OpenClaw workloads can use different models simultaneously for different tasks. This is a roadmap/TODO item, not current support.

## Why Camelid matters

Most local-model stacks emphasize broad compatibility before they can explain what is truly production-ready. Camelid takes the opposite approach.

It is for teams and builders who need to answer practical questions with confidence:

- **What exactly works?**
- **What evidence backs that claim?**
- **Will the API and UI tell the same truth?**
- **Can we widen support without hand-waving?**

That discipline is not just a docs style. It is the product.

## What makes it different

Camelid currently gives you:

- a Rust server with OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
- GGUF metadata and tensor parsing, tokenizer binding, and typed unsupported-state errors
- exact-row capability reporting through `/api/capabilities`
- a React/Vite WebUI that unlocks chat only when runtime readiness and support-contract readiness both agree
- parity and validation harnesses used to compare behavior against llama.cpp before support language moves

**Naming note.** Camelid is the product name. The repository, crate, binary, and some diagnostics still use `camelid` during the transition. Keep current commands and package identifiers on those names until a separate rename plan lands.

**Reference credit.** Camelid is original Rust code, and it keeps visible credit for the reference work behind tokenizer checks, compatibility baselines, and parity evidence. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for current acknowledgements and MIT-license notices, including llama.cpp / ggml.

## Support matrix

Camelid’s public support boundary is intentionally narrow and exact-row. Read each row literally; nothing adjacent inherits support.

| Exact lane | Public status | Green evidence today | Crisp caveat |
| --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | **Verified support** | End-to-end generation, broader five-prompt/50-token parity, bounded template-shape checks, bounded 512-context coverage, and backend RSS/perf sampling. | This is the trusted current gate, not a promise about other TinyLlama variants or quants. |
| Llama 3.2 1B Instruct Q8_0 | **Verified support (bounded)** | Load, completions, chat completions, WebUI validation, compact/broader parity, bounded template-shape checks, bounded unique-chat perf/RSS, and checked 512/1024/2048-context packs. | The 2048 pass is exact-row only after the RoPE frequency-factor fix; it is not model-native/larger-context, arbitrary-template, production-throughput, or portability support. |
| Llama 3.2 3B Instruct Q8_0 | **Verified support (bounded)** | Load, completions, chat completions, WebUI validation, compact/broader 50-token parity, bounded template-shape checks, bounded unique-chat perf/RSS, checked 512/1024/2048-context packs, and an opt-in parallel Q8 first-token direction probe. | The parallel Q8 result is a direction probe, not production throughput; broader/full support still needs model-native/larger context, arbitrary-template, and portability evidence. |
| Llama 3 8B Instruct Q8_0 | **Verified support (bounded)** | Load, completions, chat completions, WebUI validation, compact parity, three-prompt 50-token parity, checked 512/1024/2048-context packs, compact chat-template-shapes pack, bounded memory evidence, structured RSS/Q8 file-read counters, and lazy-Q8 hot-path measurements. | The 1024/2048 pass is exact-row bounded-pack evidence for runtime/API/frontend head `86ad5390d265`; no model-native/larger context, production throughput, arbitrary templates, portability, neighboring-row, or broad 8B/Llama support is implied. |
| Mistral-7B-Instruct-v0.3.Q8_0.gguf | **In active validation; not supported yet** | Source/SHA, tokenizer/template reference, first-token parity, and bounded Ubuntu artifacts exist as bring-up evidence only. A fresh 50-token broader parity run is not yet green, so support stays fail-closed. | No Mistral-family support, neighboring variants, 50-token parity, arbitrary templates, model-native/larger context, API/WebUI readiness, production throughput, portability, or full support. |
| Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf | **Planned exact-row candidate** | Candidate row selected for acquisition/metadata planning only. | No MoE/Mixtral support until expert routing, tokenizer/template, bounded load, parity, API/WebUI, RSS, and bundle evidence exist. |
| Qwen2.5-7B-Instruct-Q8_0.gguf | **Planned exact-row candidate** | Candidate row selected for acquisition/tokenizer planning only. | No Qwen support until tokenizer/template references, architecture mapping, bounded load, parity, API/WebUI, RSS, and bundle evidence exist. |
| gemma-2-9b-it-Q8_0.gguf | **Planned exact-row candidate** | Candidate row selected for acquisition/tokenizer planning only. | No Gemma support until tokenizer/template references, architecture mapping, bounded load, parity, API/WebUI, RSS, and bundle evidence exist. |

### Latest bounded model checks

The maintainer matrix now includes four exact Q8_0 supported rows with checked row-specific evidence. Mistral remains an active validation row, not a supported row. These are bounded checks, not universal model claims.

| Exact row | Latest checked bucket | Result | Output checked |
| --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | direct chat validation | PASS | `Certainly! Here` |
| Llama 3.2 1B Instruct Q8_0 | 2048-context bounded recall pack | PASS | `CMLD-204` |
| Llama 3.2 3B Instruct Q8_0 | 2048-context bounded recall pack | PASS | `CMLD-204` |
| Llama 3 8B Instruct Q8_0 | 2048-context bounded recall pack | PASS | `CMLD-204` |
| Mistral-7B-Instruct-v0.3.Q8_0.gguf | Active validation: tokenizer/template + first-token artifacts, Ubuntu bounded bring-up, and broader 50-token parity follow-up | BLOCKED | Fresh broader 50-token run is not green; no support claim |

### Read this boundary carefully

- Support does **not** inherit across model sizes, variants, quantizations, tokenizer lanes, or nearby GGUFs.
- “Llama support” and “Mistral support” currently mean only the exact rows above; no broad family claim is implied.
- Checked context packs do **not** imply model-native or broader context support.
- Bounded template-shape or perf evidence does **not** imply arbitrary template execution or production portability.
- The next major 8B and Mistral support gaps are model-native/larger context beyond checked packs, arbitrary templates, production throughput, portability, broader prompt/token coverage, repeated durability evidence, and normalized full-support bundles beyond the fresh bounded PASS bundles cited above.

Authoritative details live in [`COMPATIBILITY.md`](COMPATIBILITY.md). The current evidence snapshot lives in [`STATUS.md`](STATUS.md).

## Start here

### If you want the truth first

1. [`COMPATIBILITY.md`](COMPATIBILITY.md) — authoritative support matrix and promotion rules
2. [`STATUS.md`](STATUS.md) — current evidence boundary, recent moves, and blockers
3. [`ROADMAP.md`](ROADMAP.md) — what must happen next to widen support honestly
4. [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — acknowledgements and license notices

### If you want to run Camelid or contribute

1. [`docs/CONTRIBUTOR_QUICKSTART.md`](docs/CONTRIBUTOR_QUICKSTART.md) — shortest safe local path
2. [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) — toolchain, environment, and path guidance
3. [`docs/VALIDATION_MATRIX.md`](docs/VALIDATION_MATRIX.md) — expected checks by change type
4. [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution expectations and PR guidance
5. [`DOCS.md`](DOCS.md) — full documentation map

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

That confirms the backend is up.

### 3) Before you expect local chat to work

You will need all of the following:

- a supported GGUF model file already present on your machine
- the model path wired into a load request you control
- any extra contributor setup described in [`docs/CONTRIBUTOR_QUICKSTART.md`](docs/CONTRIBUTOR_QUICKSTART.md) and [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)

For a first supported local run, TinyLlama is the clearest path — but it is **not bundled** in this repository, and the README should not be read as a copy-paste chat demo.

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

Camelid intentionally separates the public support contract, the local contributor path, and maintainer-only evidence/promotion workflows. That keeps the project welcoming without leaking operator-only details or pretending every internal validation lane is part of normal onboarding.

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
