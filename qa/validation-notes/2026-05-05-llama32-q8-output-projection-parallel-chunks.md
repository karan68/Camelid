# Llama 3.2 Q8 output-projection chunking — 2026-05-05

Scope: runtime-only hardening for the exact Llama 3.2 1B/3B Q8_0 full-support memory/perf lane. This does not widen the public support claim and does not replace row-specific Ubuntu evidence.

Change:

- `matmul_rhs_transposed_q8_0_block_reader` now uses the same guarded `BACKENDINFERENCE_PARALLEL_LINEAR=on` / `BACKENDINFERENCE_PARALLEL_LINEAR_MIN_OUTPUTS` chunk-parallel row-dot path already used by file-backed per-row linear accumulation.
- The default remains unchanged because `BACKENDINFERENCE_PARALLEL_LINEAR` is opt-in.
- Each output row still uses the deterministic scalar Q8_0 encoded-row dot; parallelism only partitions independent output rows inside the already bounded file-reader chunk.

Why it closes a row+box sub-check:

- Llama 3.2 1B/3B already have bounded unique-chat RSS/perf evidence, but full support still needs stronger production-throughput work.
- This closes the code/runtime sub-box for optional chunk-parallel file-backed Q8_0 output projection, the hot final projection path that 1B/3B share with the larger Llama rows.
- Evidence is local unit coverage only; promotion-grade latency/RSS numbers still require the approved Ubuntu lane.

Validation:

```bash
cargo fmt --all -- --check
cargo test -q q8_0_block_reader_linear_matches_q8_path_with_parallel_chunks --lib
cargo test -q q8_0_block_reader --lib
cargo test -q q8_0_file_backed_accumulate_matches_q8_block_dot_across_chunks --lib
```

Result: all passed locally.
