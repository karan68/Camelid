# Camelid

[![CI][ci-badge]][ci-workflow]

![Camelid banner](assets/camelid-banner.png)

**Camelid is a Rust-native local inference backend for GGUF language models built for people who want local AI they can actually trust.**

Most local-model stacks are easy to demo and hard to trust. Camelid is built to close that gap: fast local inference, honest readiness signals, and support claims narrow enough to verify.

It does not blur “probably works” into “supported.” Camelid publishes exact support rows, keeps the API and WebUI aligned with reality, and moves only when the evidence is real.

The goal is simple: local chat that feels sharp, legible, and dependable — without pretending unsupported paths are ready.

> **Current public posture:** one fully validated TinyLlama lane and three intentionally narrow exact-row Llama smoke lanes. Nearby models do not inherit support.

## Why Camelid matters

Most local-model stacks sell compatibility first and precision later. Camelid flips that.

Camelid is for teams and builders who need to answer practical questions with confidence:

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

**Naming note.** Camelid is the product name. The repository, crate, binary, and some diagnostics still use `backendinference` during the transition. Keep current commands and package identifiers on those names until a separate rename plan lands.

**Reference credit.** Camelid is original Rust code, and it keeps visible credit for the reference work behind tokenizer checks, compatibility baselines, and parity evidence. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for current acknowledgements and MIT-license notices, including llama.cpp / ggml.

## Support matrix

Camelid’s public support boundary is intentionally narrow and exact-row.

| Exact lane | Public status | What Camelid can honestly claim today |
| --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | **Supported current gate** | End-to-end parity-backed support for the current validated TinyLlama row, including broader parity, bounded template-shape checks, bounded 512-context coverage, and bounded backend RSS/perf sampling. |
| Llama 3.2 1B Instruct Q8_0 | **Supported exact-row smoke** | Exact-row load, completions, chat completions, WebUI smoke, bounded prompt-pack parity, bounded template-shape checks, bounded unique-chat perf/RSS envelope, and bounded 512/1024-context packs; the attempted 2048-context pack is blocked by first-token divergence and is not support. |
| Llama 3.2 3B Instruct Q8_0 | **Supported exact-row smoke** | Exact-row load, completions, chat completions, WebUI smoke, bounded prompt-pack parity, bounded template-shape checks, bounded unique-chat perf/RSS envelope, and bounded 512/1024/2048-context packs. |
| Llama 3 8B Instruct Q8_0 | **Supported exact-row smoke** | Exact-row load, completions, chat completions, WebUI smoke, bounded three-prompt parity, one bounded 512-context pack, one bounded compact chat-template-shapes pack, and bounded memory evidence. |

### Read this boundary carefully

- Support does **not** inherit across model sizes, variants, quantizations, tokenizer lanes, or nearby GGUFs.
- “Llama support” currently means only the exact rows above.
- Checked context packs do **not** imply model-native or broader context support.
- Bounded template-shape or perf evidence does **not** imply arbitrary template execution or production portability.

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

This quickstart verifies that Camelid builds and the backend starts on your machine. It is **not** a one-command end-to-end chat demo: the repository does not bundle supported GGUF model files, and supported local chat requires additional setup.

### 1) Build and run the server

```bash
git checkout main
git pull --ff-only
cargo build --release --bin backendinference
target/release/backendinference serve --addr 127.0.0.1:8181
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

Camelid includes a React/Vite frontend in [`frontend/`](frontend/).

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

If you want to contribute, start with the docs written for safe local iteration:

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
