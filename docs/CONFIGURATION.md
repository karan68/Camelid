# Configuration Guide

Last updated: 2026-05-05

This guide documents Camelid's current local configuration reality without pretending every workflow is fully automated.

## Toolchain expectations

### Rust / Cargo

Camelid currently requires Rust/Cargo 1.87+.

On hosts where `/usr/bin/cargo` is older than the required toolchain, prefer one of these paths:

```bash
source "$HOME/.cargo/env"
```

or:

```bash
scripts/with-rustup-cargo.sh build --release --bin backendinference
scripts/with-rustup-cargo.sh test --all-targets --all-features
```

### Node / npm

The frontend expects a working Node.js + npm install. Use `npm ci` when you want a reproducible install from the committed lockfile.

## Backend runtime defaults

The common local backend bind address in repo docs is:

```text
127.0.0.1:8181
```

Typical start command:

```bash
target/release/backendinference serve --addr 127.0.0.1:8181
```

## Frontend API base override

The frontend defaults to:

```text
http://127.0.0.1:8181
```

Override it for local dev/build with:

```bash
VITE_BACKENDINFERENCE_API_BASE=http://127.0.0.1:8181 npm run dev
```

You can also edit the API base in the UI while testing.

## Model-path guidance

Repo examples often use paths such as:

```text
models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
$CAMELID_MODEL_DIR/Llama-3.2-1B-Instruct-Q8_0.gguf
$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf
$CAMELID_MODEL_DIR/Meta-Llama-3-8B-Instruct-Q8_0.gguf
```

These are example local paths, not a guarantee that the repo fetches or manages model files for you.

Recommended practice:

- keep local GGUFs outside version control
- use stable local paths during validation so commands and artifacts stay reproducible
- avoid documenting private absolute paths in public artifacts or docs

## Environment and local-shell assumptions

Current public docs assume:

- `cargo` resolves to a Rust 1.87+ toolchain
- `node` and `npm` are available for frontend work
- `llama-server` is in `PATH` only when you are running parity comparisons

Backend runtime knobs used during performance work:

- `BACKENDINFERENCE_PREFILL_CHUNK_TOKENS` controls how many non-final prompt tokens the backend processes per chunk in the chunked prefill path. Default: `32`. Set it to `0` or `1` to force the older sequential prefill path while debugging. This is a runtime/performance knob only; it is not support evidence for any model row by itself.

If a command depends on more than that, document the requirement in the same PR.

## Maintainer-only/private workflows

The following are intentionally not public contributor requirements:

- SSH-based validation-lane access
- private host aliases or machine-specific setup
- unpublished remote worktree conventions
- local absolute paths from a maintainer workstation

Public docs may mention that some promotion-grade reruns happen on an approved Ubuntu validation lane, but they should not expose private operator details.

## Documentation rule of thumb

When adding a new variable, path convention, or host assumption:

1. document the public/local requirement here if contributors need it
2. keep private operator details out of public docs
3. avoid claiming a workflow is turnkey unless the repo actually makes it turnkey
