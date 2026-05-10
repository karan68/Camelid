# Camelid

[![CI][ci-badge]][ci-workflow]

![Camelid banner](assets/camelid-banner.png)

**Camelid is a Rust-native local inference backend for GGUF language models. It is built for teams that care about local performance, clear support boundaries, and evidence they can inspect.**

Many local-model stacks are easy to demo and hard to trust. Camelid is designed to close that gap with disciplined support claims, straightforward readiness signals, and a product surface that stays aligned across runtime, API, UI, and documentation.

Camelid does not treat “probably works” as “supported.” Support moves only when the evidence is real.

> **Current public posture:** Camelid achieves 1:1 parity with llama.cpp for four exact GGUF rows within a bounded published validation envelope: TinyLlama at the current validated gate, Llama 3.2 1B/3B Q8_0 through checked bounded 512/1024/2048-context packs, and Llama 3 8B Q8_0 through exact-row smoke plus checked bounded 512/1024/2048-context packs on current `main`. The fresh current-head 8B 1024/2048 PASS bundle (`qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json`) closes those exact bounded buckets after docs/API/frontend alignment. `Mistral-7B-Instruct-v0.3.Q8_0.gguf` remains an active exact-row bring-up lane, not a supported row: source/SHA, tokenizer/template, 1-token generation, broader five-prompt/50-token parity, bounded 512/1024/2048, and checked 4096/8192 context evidence now exist, but promotion stays fail-closed until API/WebUI/RSS readiness and the full support surfaces are synchronized. `Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf` now has bounded backend MoE runtime evidence: lazy/file-backed rank-3 Q8 expert routing, selected top-k weight renormalization, one-token smoke, and 5-of-6 short-prompt 5-token parity against llama.cpp. It is not broad Mixtral support; a known `Count to three.` step-2 near-tie divergence and missing API/WebUI/long-context promotion evidence keep it fail-closed. The current blocker is localized to llama.cpp's default f16/f16 KV-cache attention/KQV numeric path; coarse RoPE, prefill/decode, and scalar f16-rounding variants have been rejected as fixes. These are exact-row bounded-pack claims only; wider model-native context beyond the checked packs, production throughput, portability, arbitrary templates, frontend readiness for Mixtral, and broad-family support remain outside the claim.

## Milestone at a glance

Camelid's current milestone is not a loose compatibility demo. It is a synchronized product surface: backend runtime, OpenAI-compatible API, WebUI readiness, docs, and durable evidence all point at the same exact support contract.

- **Four exact Q8_0 rows are public and evidence-backed, with Mixtral/MoE now in bounded backend runtime validation.** TinyLlama remains the full current gate; Llama 3.2 1B and Llama 3.2 3B have checked bounded 512/1024/2048 support where row-specific PASS artifacts exist; Llama 3 8B has exact-row smoke plus checked 512/1024/2048-context support on current `main`, backed by the fresh current-head canonical bundle. Mistral 7B Instruct v0.3 remains fail-closed as an active bring-up lane until its current-head promotion checklist is complete. Mixtral 8x7B Instruct v0.1 has bounded backend MoE runtime evidence only, with one known short-prompt divergence still blocking broader support; latest evidence narrows that divergence to the exact f16 KV-cache/KQV attention path rather than tokenizer, prompt construction, router top-k, or a global RoPE setting.
- **The UI and API fail closed instead of guessing.** Chat unlocks only when the loaded local GGUF is `loaded_now=true`, `generation_ready=true`, and matched to an exact supported `/api/capabilities` row.
- **The context ladder is explicit.** TinyLlama stays at the current gate with bounded 512-context coverage; Llama 3.2 1B/3B have checked bounded 512/1024/2048-context evidence, while Llama 3 8B is checked through the bounded 512/1024/2048-context packs on current `main`, with the 1024/2048 buckets tied to the fresh current-head PASS bundle. Mistral context evidence now includes checked 512/1024/2048 and 4096/8192 bring-up passes, but remains validation evidence only until the exact row is promoted through the support checklist. These checked packs do not imply model-native/larger context beyond the checked packs, arbitrary-template, production-throughput, portability, or broader-family support.

![Camelid WebUI chat surface](docs/assets/camelid-readme-chat-surface-dark.png)

The WebUI screenshot above is intentionally the dark, collapsed-rail chat surface: the chat entrypoint is simple and product-forward, while still reflecting the local-first runtime contract. It is not a broad model-family or arbitrary-GGUF support claim.

## Current work tracks

Camelid is advancing on two tracks, and both stay gated by CI plus artifact-backed support language:

- **Supported-row hardening:** preserve TinyLlama as the current full gate, move the Llama exact-row verified lanes toward the same normalized bar, and keep the Mistral exact-row bring-up lane fail-closed until its promotion checklist is green. The next promotable evidence remains model-native/larger-context behavior beyond checked packs, arbitrary/Jinja template coverage, production-throughput evidence, portability, and repeated current-head bundles. CI reliability is non-negotiable: no public support wording should move if the gate is red.
- **Active next-model bring-up set:** Mistral 7B Instruct v0.3 is the first next-family exact row under active validation, but it is not supported yet. Camelid is also publicly working on **Mixtral 8x7B Instruct** as a bounded backend MoE runtime validation lane, plus **Qwen 2.5 7B Instruct** and **Gemma 2 9B Instruct** as planned exact-row candidates.
  - `Mistral-7B-Instruct-v0.3.Q8_0.gguf` — active validation only: source/SHA, exact tokenizer/template references, 1-token generation parity, broader five-prompt/50-token parity, bounded 512/1024/2048, and checked 4096/8192 context evidence now exist. Support remains fail-closed pending row-specific API/WebUI/RSS evidence and synchronized capability/frontend/docs wording. No Mistral-family, neighboring-row, arbitrary-template, production-throughput, portability, or full-support claim is implied.
  - `Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf` — bounded backend MoE runtime evidence: source/SHA/license, tokenizer/template references, lazy/file-backed rank-3 Q8 expert routing, selected top-k weight renormalization, one-token runtime smoke, and 5-of-6 short-prompt 5-token parity against llama.cpp are documented. Broader support remains blocked by the known `Count to three.` step-2 near-tie divergence plus API/WebUI, RSS/timing, long-context, and promotion-bundle evidence. Recent negative probes reject a global RoPE-pairing change and coarse f16 attention-rounding variants; the next technical target is exact ggml f16 KV-cache/KQV numeric behavior.
  - `Qwen2.5-7B-Instruct-Q8_0.gguf` — planned exact-row candidate; tokenizer/template and architecture mapping still need row-specific proof.
  - `gemma-2-9b-it-Q8_0.gguf` — planned exact-row candidate; tokenizer/template and Gemma2 runtime behavior still need row-specific proof.
  - Public wording for Mixtral is limited to bounded exact-row backend MoE runtime evidence until the known divergence and readiness gates close. Qwen and Gemma remain “not supported yet” until each row has its own source/SHA/license, tokenizer/template, parity, bounded load/readiness, API/WebUI, RSS/timing, scrubbed manifest, and checksum evidence. See [`COMPATIBILITY.md`](COMPATIBILITY.md#locked-next-family-readiness-language) for the family-by-family promotion checklist.
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
| Llama 3 8B Instruct Q8_0 | **Verified support (bounded)** | Load, completions, chat completions, WebUI validation, compact parity, three-prompt 50-token parity, checked 512/1024/2048-context packs, compact chat-template-shapes pack, bounded memory evidence, structured RSS/Q8 file-read counters, and lazy-Q8 hot-path measurements. | The 1024/2048 pass is exact-row/current-head only via `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json`; no model-native/larger context beyond checked packs, production throughput, arbitrary templates, portability, neighboring-row, or broad 8B/Llama support is implied. |
| Mistral-7B-Instruct-v0.3.Q8_0.gguf | **In active validation; not supported yet** | Source/SHA, exact tokenizer/template references, 1-token generation parity, broader five-prompt/50-token parity, bounded 512/1024/2048 bring-up, and checked 4096/8192 context validation are green; latest context bundle: `qa/evidence-bundles/mistral-7b-v0.3-q8-context-4096-8192-ubuntu-20260509T005229Z-head-9e3c64f2cfab/manifest.json`. | No Mistral-family support, neighboring variants, arbitrary templates, model-native/larger context, API/WebUI readiness, production throughput, portability, or full support until row-specific readiness evidence and surfaces land. |
| Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf | **Bounded backend MoE runtime validation** | Source/SHA/license, tokenizer/template references, lazy/file-backed rank-3 Q8 expert routing, selected top-k weight renormalization, one-token smoke, 5-of-6 short-prompt 5-token parity, and negative evidence rejecting global RoPE, cache-type, prefill/decode, and coarse f16 attention-rounding variants. Key bundles: `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-support-probe-20260509/manifest.json`, `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-kv-cache-type-probe-20260509/manifest.json`, and `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-prefill-decode-kqv-rejection-20260509/manifest.json`. | Not broad Mixtral support: `Count to three.` still has a step-2 near-tie divergence localized to the f16 KV-cache/KQV attention path; no frontend green state, long-context support, production throughput, neighboring-row, or full-support claim. |
| Qwen2.5-7B-Instruct-Q8_0.gguf | **Planned exact-row candidate** | Candidate row selected for acquisition/tokenizer planning only. | No Qwen support until tokenizer/template references, architecture mapping, bounded load, parity, API/WebUI, RSS, and bundle evidence exist. |
| gemma-2-9b-it-Q8_0.gguf | **Planned exact-row candidate** | Candidate row selected for acquisition/tokenizer planning only. | No Gemma support until tokenizer/template references, architecture mapping, bounded load, parity, API/WebUI, RSS, and bundle evidence exist. |

### Latest bounded model checks

The maintainer matrix now includes four exact Q8_0 supported rows with checked row-specific evidence. Mistral remains an active validation row, not a supported row. Mixtral now has bounded backend MoE runtime evidence, but not frontend/broad support. These are bounded checks, not universal model claims.

| Exact row | Latest checked bucket | Result | Output checked |
| --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | direct chat validation | PASS | `Certainly! Here` |
| Llama 3.2 1B Instruct Q8_0 | 2048-context bounded recall pack | PASS | `CMLD-204` |
| Llama 3.2 3B Instruct Q8_0 | 2048-context bounded recall pack | PASS | `CMLD-204` |
| Llama 3 8B Instruct Q8_0 | 2048-context bounded recall pack | PASS on current `main` | `CMLD-204` |
| Mistral-7B-Instruct-v0.3.Q8_0.gguf | Active validation: Ubuntu bounded bring-up plus exact tokenizer/template, 1-token, broader five-prompt/50-token parity, and checked 4096/8192 context evidence | PASS for validation lane; still unsupported | API/WebUI/RSS readiness and synchronized support surfaces still required before any support claim |
| Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf | Bounded backend MoE runtime probe: one-token smoke, selected-weight renormalized top-k routing, lazy/file-backed rank-3 Q8 experts, and six short prompts at max_tokens=5 | 5/6 prompts match; `Count to three.` remains a known step-2 near-tie divergence, now narrowed to f16 KV-cache/KQV attention numerics | Backend runtime validation only; no frontend/long-context/broad support claim |

### Read this boundary carefully

- Support does **not** inherit across model sizes, variants, quantizations, tokenizer lanes, or nearby GGUFs.
- “Llama support” currently means only the exact supported rows above; Mistral has no support claim yet and may only be discussed as the exact active validation row above. Mixtral may only be discussed as bounded exact-row backend MoE runtime evidence until its known divergence and readiness gates close.
- Checked context packs do **not** imply model-native or broader context support.
- Bounded template-shape or perf evidence does **not** imply arbitrary template execution or production portability.
- The next major 8B, Mistral, and Mixtral support gaps are model-native/larger context beyond checked packs, arbitrary templates, production throughput, portability, repeated durability evidence, and normalized full-support bundles; for Mistral, row-specific API/WebUI/RSS readiness and synchronized capability/frontend/docs surfaces remain required before support; for Mixtral, the known f16 KV-cache/KQV step-2 divergence plus API/WebUI/RSS/long-context evidence remain required before broader support.

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
