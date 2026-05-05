# 2026-05-05 — chunked prompt-prefill runtime slice

Scope: backend performance/memory architecture only. This does not promote any Llama row, and it does not clear the frozen-red Llama 3 8B 1024/2048-context blockers.

Context:

- Tim's 8B rule was satisfied first: the current 8B 1024-context diagnostic was inspected before any further 8B long-context work.
- On the approved Ubuntu lane, clean public head `96b452723993c33c6c6140824673f62834839993` ran the exact 8B 1024 pack and failed red: llama.cpp accepted the 881-token prompt, but Camelid `/v1/chat/completions` timed out after the 900-second parity harness timeout before producing a parity report.
- Therefore the work pivoted to engine performance/memory architecture instead of another 8B 2048 promotion attempt.

Change:

- Prompt prefill can now process non-final prompt tokens in bounded chunks instead of replaying each prefill token through the full single-token path.
- The chunk path batches token embedding lookup, per-layer Q/K/V projections, RoPE application, KV-cache writes, causal attention context, attention output, and gated FFN activation for the prefill portion.
- The final prompt token still runs through the existing single-token path so logits, output normalization, diagnostics, and sampling behavior remain aligned with the established generation path.
- `matmul_rhs_transposed_q8_0_block_reader` now quantizes all input rows once and reuses each file-backed Q8_0 weight chunk across those rows before advancing the reader chunk. This reduces repeated Q8 file reads for batched prefill/projected rows.
- `BACKENDINFERENCE_PREFILL_CHUNK_TOKENS` controls chunk size; the default is `32`. Values `0` or `1` fall back to sequential prefill.

Follow-up instrumentation:

- Chunked prompt prefill now participates in structured forward-memory diagnostics when `BACKENDINFERENCE_FORWARD_RSS_TIMINGS=1` or `BACKENDINFERENCE_FORWARD_MEMORY_TRACE=1` is enabled.
- The prefill chunk records start/embedding/layers/end samples, Q8 file-read deltas, and per-layer phase samples for attention norm/Q/K/RoPE/V, KV-cache write, attention context/output/residual, FFN norm/activation/down/residual.
- This is instrumentation for the performance/memory architecture lane only; it is not a row-support promotion or an 8B long-context PASS artifact.

Validation:

```bash
./scripts/with-rustup-cargo.sh test
./scripts/with-rustup-cargo.sh fmt --check
./scripts/with-rustup-cargo.sh clippy --all-targets -- -D warnings
./scripts/with-rustup-cargo.sh build --release --bin backendinference
```

Result: all passed locally.

Ubuntu follow-up validation:

- Approved validation target: `ubuntu@54.69.75.77` using `/Users/timtoole/Documents/cert/ubuntu.pem`.
- Isolated remote checkout base: `c4b51de7e7f7cc1bf16bd77ceded4589246b003e`.
- Applied patch SHA256: `cc30a2cdcae54ba02c2e82b86089ff7dc99dc834592719ab3ecb1a0cab111c42`.
- Remote work dir: `/tmp/camelid-prefill-memory-lx1w0o/repo`.
- Remote gates passed: `fmt --check`, focused `chunked_prefill_matches_sequential_prefill_outputs_and_cache`, full `test -q --lib` (`106 passed` on Linux), `clippy -q --lib -- -D warnings`, and `build -q --release --bin backendinference`.

Focused coverage:

- `chunked_prefill_matches_sequential_prefill_outputs_and_cache` compares chunked vs sequential prompt prefill for next-token output, logits, hidden state, KV-cache position, keys, and values.
- The same test now asserts chunked prefill emits structured memory timings and all expected per-layer phase samples under `BACKENDINFERENCE_FORWARD_RSS_TIMINGS=1`.
- `q8_0_file_backed_batch_matmul_reuses_chunk_reads_across_input_rows` confirms a 3-row batched Q8_0 file-backed matmul reuses two weight chunk reads instead of rereading per input row while matching the existing Q8 block-dot output.

Claim boundary:

- This is code/runtime evidence for the backend architecture lane only.
- It is not a PASS artifact for Llama 3 8B 1024/2048 context.
- It is not broad Llama-family support, production throughput evidence, portability evidence, or a frontend/API support-status change.
- Any future 8B long-context status change still requires a fresh row-specific Ubuntu PASS artifact after the backend completes within the parity harness timeout.
