# 2026-05-06 — Q8 scale-buffer reuse and 8B 2048 diagnostic chain

Scope: diagnostic/performance structural headroom only. This does not promote 8B 1024/2048 support and does not widen docs/API claims.

Change:

- Reused the file-backed Q8 row-chunk scale decode buffer with a thread-local `Vec<f32>`, mirroring the existing row-byte chunk reuse.
- This removes a per-matmul allocation in both the batch block-reader path and the borrowed single-row file-reader path while keeping the same Q8 bytes, scales, dot products, cache policy, and output layout.

Remote diagnostics:

- Canonical Ubuntu host `/home/ubuntu/work/Camelid-diag-4c30c53` produced a diagnostic 8B 1024 compact-context parity PASS at source head `4c30c53`:
  - artifact copied locally: `target/remote-llama3-8b-context-1024-4c30c53-20260506T051812Z/`
  - prompt/generated parity: `prompt_tokens_all_match=true`, `generated_tokens_all_match=true`, `generated_text_all_match=true`
  - reference prompt tokens: 881
  - lazy Q8, prefill chunk 128, Q8 file cache disabled
  - backend trace final: `q8_file_read_bytes=91781055744` (`87529.24 MiB`), cache hit bytes 0, backend RSS trace ~890068 KiB at logits
- A source-`f8c2d66` 8B 2048 no-cache diagnostic was launched on the canonical Ubuntu host from archived worktree `/home/ubuntu/work/Camelid-longctx-f8c2d66-20260506T061712Z`:
  - run script: `target/run-8b-2048-diag-f8c2d66.sh`
  - active artifact root: `target/llama3-8b-context-2048-diag-20260506T061807Z-source-f8c2d66/`
  - env: lazy Q8 on, retained Q8 blocks off, chunk tokens 128, Q8 file cache bytes 0
- A follow-on source-`f8c2d66` 8B 2048 cache hypothesis run is queued to start after the no-cache run exits:
  - run script: `target/run-8b-2048-cache-diag-f8c2d66.sh`
  - env delta: `BACKENDINFERENCE_Q8_0_FILE_CACHE_BYTES=335544320` (320 MiB)

Local gates:

- `./scripts/with-rustup-cargo.sh fmt --all -- --check`
- `./scripts/with-rustup-cargo.sh test -q q8_0_file_backed_batch_matmul_reuses_chunk_reads_across_input_rows --lib`
- `./scripts/with-rustup-cargo.sh test -q q8_0_file_backed_accumulate_matches_q8_block_dot_across_chunks --lib`
- `./scripts/with-rustup-cargo.sh test -q q8_file_cache --lib`
- `./scripts/with-rustup-cargo.sh test -q prefill --lib`
- `./scripts/with-rustup-cargo.sh clippy -q --all-targets -- -D warnings`

Claim boundary: performance-only. 8B 1024/2048 remain blocked from support-promotion/docs/API widening until fresh PASS artifacts are reviewed and explicitly aligned.
