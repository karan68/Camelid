# Configuration Guide

Last updated: 2026-05-06

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
scripts/with-rustup-cargo.sh build --release --bin camelid
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
target/release/camelid serve --addr 127.0.0.1:8181
```

## Frontend API base override

The frontend defaults to:

```text
http://127.0.0.1:8181
```

Override it for local dev/build with:

```bash
VITE_CAMELID_API_BASE=http://127.0.0.1:8181 npm run dev
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

- `CAMELID_PREFILL_CHUNK_TOKENS` controls how many non-final prompt tokens the backend processes per chunk in the chunked prefill path. Default: `256`, matching the current long-prefill performance lane while keeping the global lazy Q8 file cache disabled outside explicit/scoped reuse. Set it to `1` to force the older sequential prefill path while debugging; invalid/zero values fall back to the default. This is a runtime/performance knob only; it is not support evidence for any model row by itself; the separate published source/runtime-head PASS bundle and synchronized docs/API/frontend updates are what close exact Llama 3 8B checked 1024/2048 packs; the knob itself is not evidence for today's checkout.
- `CAMELID_PREFILL_LAYER_MAJOR` controls the long-context prefill schedule that processes all prefill chunks one layer at a time, reusing file-backed Q8_0 weights across chunks before moving to the next layer. By default it is enabled only when lazy Q8_0 file-backed weights are present. Set it to `0`, `false`, `off`, or `disabled` to force the older chunk-major schedule while debugging.
- `CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS` controls the per-layer prompt chunk size only for the layer-major schedule. Default: `512`, unless `CAMELID_PREFILL_CHUNK_TOKENS` is explicitly set, in which case the shared chunk setting is reused for comparability. It also accepts `all`, `full`, `prompt`, or `unbounded` for one diagnostic full-prompt prefill chunk. This is a runtime/performance knob only and does not promote any 8B 1024/2048 support bucket by itself.
- `CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES` controls the layer-major-only scoped Q8_0 raw-byte reuse window when lazy file-backed Q8_0 weights are present and `CAMELID_Q8_0_FILE_CACHE_BYTES` is unset. Default: `268435456` (256 MiB) only for multi-chunk layer-major prefill, where file-backed Q8_0 weights can be reused across chunks; single-chunk prefill skips the default scoped cache unless this scoped knob is set explicitly. Set it to `0` to disable the scoped layer-major cache, or set the global cache knob explicitly to take over all Q8 file-reader cache sizing. This is a bounded RSS/read-reuse tuning knob only and does not promote any 8B support bucket by itself.
- `CAMELID_PREFILL_LAYER_MAJOR_ATTRIBUTION` enables optional structured per-layer/per-prefill-chunk attribution for the layer-major schedule inside forward-memory timings. This is diagnostic instrumentation for memory/Q8 read attribution, not support evidence or a promotion signal by itself.
- Q8 byte-count knobs accept plain bytes or binary suffixes (`KiB`/`MiB`/`GiB`, also `K`/`M`/`G`; underscores and spaces are ignored). This covers `CAMELID_Q8_0_FILE_CACHE_BYTES`, `CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES`, `CAMELID_Q8_0_FILE_READER_CHUNK_BYTES`, `CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES`, and `CAMELID_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES` without changing their numeric defaults.
- `CAMELID_Q8_0_FILE_READER_CHUNK_BYTES` controls the target Q8_0 row-read chunk size for borrowed/file-backed row readers. Default: `33554432` (32 MiB). This is a read-pattern/performance knob only.
- `CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES` caps reusable f32 output scratch for multi-row lazy-Q8 file-backed matmuls. Default: `67108864` (64 MiB). This is an RSS/read-reuse tuning knob only.
- `CAMELID_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES` caps how much per-thread Q8 file-reader scratch capacity is retained after oversized row, scale, quantized-input, and output chunks. Default: `67108864` (64 MiB). This is an RSS headroom knob only; it does not promote 8B 1024/2048 support by itself.

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
