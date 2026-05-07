# Contributor Quickstart

Last updated: 2026-05-06

This guide is the shortest safe path to getting productive in Camelid locally.

## Start here

1. Read [`README.md`](../README.md) for the current public support contract.
2. Read [`CONTRIBUTING.md`](../CONTRIBUTING.md) for contribution expectations.
3. Use this guide to get a local backend/frontend loop running.
4. Use [`docs/VALIDATION_MATRIX.md`](VALIDATION_MATRIX.md) to choose the smallest meaningful validation lane for your change.

Current public support is exact-row: TinyLlama Q8_0 is the supported gate, and Llama 3.2 1B/3B plus Llama 3 8B Q8_0 are checked through bounded 512/1024/2048-context packs. Do not broaden those claims to model-native/larger context beyond checked packs, production throughput, portability, local experiments, arbitrary templates, or adjacent GGUFs without new evidence bundles and synchronized docs/API/frontend updates.

## Prerequisites

Required local tools:

- Rust/Cargo 1.87+
- Node.js + npm for `frontend/`
- Git

Helpful but not required for every contribution:

- `llama-server` in `PATH` for parity comparisons
- local GGUF model files for supported-row smoke or parity work

If your host exposes an older distro `cargo`, source `$HOME/.cargo/env` first or use `scripts/with-rustup-cargo.sh ...` so the rustup-managed toolchain is used.

## Backend quickstart

This gets the backend running locally. By itself, it does **not** guarantee a working chat demo, because the repository does not bundle supported GGUF model files.

From the repo root:

```bash
git checkout main
git pull --ff-only
cargo build --release --bin backendinference
target/release/backendinference serve --addr 127.0.0.1:8181
```

If you prefer the rustup wrapper:

```bash
scripts/with-rustup-cargo.sh build --release --bin backendinference
target/release/backendinference serve --addr 127.0.0.1:8181
```

## Frontend quickstart

The frontend can run against your local backend, but chat readiness still depends on loading a supported local model file and meeting the current support contract.

In another shell:

```bash
cd frontend
npm ci
npm run dev
```

Default frontend URL:

```text
http://127.0.0.1:4175
```

Default backend API base:

```text
http://127.0.0.1:8181
```

## Basic local checks by change type

### Docs-only change

```bash
git diff --check
bash scripts/check-public-scrub.sh
```

### Backend or shared code change

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo doc --no-deps --all-features
bash scripts/check-public-scrub.sh
```

### Frontend change

```bash
cd frontend
npm ci
npm run build
```

If the frontend change affects live chat, model loading, or readiness gating, also run the frontend smoke flow described in [`frontend/README.md`](../frontend/README.md).

## What is intentionally not turnkey yet

These areas still require real manual setup and should not be described as one-command contributor onboarding:

- downloading or hosting large real model files
- setting up `llama-server` and reference-model parity hosts
- reproducing every remote Ubuntu validation-lane rerun locally
- maintainer-only SSH or remote validation workflows

## Public vs maintainer-only workflows

Public contributor docs should cover local development, public evidence, and exact support boundaries.

Maintainer-only workflows should stay out of public onboarding docs when they depend on private infrastructure, SSH access, local machine paths, or unpublished operational habits.

## Next docs to use

- [`docs/CONFIGURATION.md`](CONFIGURATION.md) — local config and env-var guidance
- [`docs/VALIDATION_MATRIX.md`](VALIDATION_MATRIX.md) — expected checks by change class
- [`frontend/README.md`](../frontend/README.md) — frontend-specific smoke and contract notes
