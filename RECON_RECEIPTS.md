# RECON_RECEIPTS.md — Phase 1 recon for parity receipts (working note, not for commit)

Date: 2026-06-05. Repo: /Volumes/Untitled/Camelid-push @ main (65a1c35), clean tree.

## Confirmed binary name

**`camelid`** — the rename is complete in code:
- `Cargo.toml`: `[package] name = "camelid"`, `[lib] name = "camelid"`, `[[bin]] name = "camelid"` (plus a second bin `repack-ghost`).
- `src/main.rs:37` — `#[command(name = "camelid", ...)]`.
- `grep -rl backendinference src/ scripts/ frontend/src` → only `scripts/check-public-scrub.sh` (the scrub scanner itself, by design) and one comment in `frontend/src/hooks/useDashboardData.js`. All env vars in src are `CAMELID_*`.
- Remaining `backendinference` references live only in historical docs (`STATUS_ARCHIVE_2026-04.md`, CONTEXT.md glossary "avoid" note).

→ Every `backendinference …` placeholder in the spec becomes `camelid …`. The new subcommand goes on the existing `camelid` bin. **No rename discrepancy to fix.**

## Module map (exact paths)

| Concern | Location |
|---|---|
| Router setup | `src/api/mod.rs` — `router_with_state()` (~1024), `serve()` (~1084) |
| `POST /v1/completions` handler | `src/api/mod.rs` — `completions()` (~3115) |
| `POST /v1/chat/completions` handler | `src/api/mod.rs` — `chat_completions()` (~3212) |
| Token-by-token decode loop | `src/api/mod.rs` — `generate_token_ids()` (~4693); steps via `inference.rs` `generate_next_token_with_history_diagnostics()` / fast greedy `generate_next_token_greedy_resident()` |
| GGUF load + metadata | `src/gguf/reader.rs` — `read_metadata(path) -> GgufFile`; `GgufFile { path, metadata: BTreeMap, tensors, ... }`; `gguf.architecture()` reads `general.architecture` |
| Tokenizer | `src/tokenizer/mod.rs` — `Tokenizer::from_gguf()` (kind from `tokenizer.ggml.model`: "llama"→SPM, "gpt2"→BPE), `encode()`, `decode()`; kind string via `TokenizerModel::as_summary_model()` → `"llama_spm"`/`"gpt2_bpe"` |
| CLI | `src/main.rs` — clap derive; `Cli` (~37), `enum Command` (~44): Serve, ServeDistributed, BenchNetwork, Inspect, TensorDump, BenchDenseHotloops, BenchQ8Blocks, DistributeWorker, DistributeMaster, BenchGenerate, GhostRun |
| Top-level modules (`src/lib.rs`) | api, cluster, distributed, error, execution_plan, gguf, ghost, inference, metal, model, model_source, tensor, tokenizer — **no receipt/evidence module exists; new module: `src/receipt/mod.rs`** |

## Parity script (`scripts/chat-parity-tinyllama.mjs`)

- Talks to Camelid over HTTP (`CAMELID_API_BASE`, default `http://127.0.0.1:8181`) and llama-server (`TINYLLAMA_LLAMA_SERVER[_URL]`, default port 8183; args `-ngl 0 -c 512 --no-warmup`).
- Request: `{ model, messages:[{role:'user',content}], max_tokens, stream:false, temperature:0 }` (+ Camelid diag fields `camelid_logit_token_ids`, `camelid_dense_diagnostics`; llama side adds `logprobs/top_logprobs`).
- Output written to `--diagnostics-out` / `TINYLLAMA_CHAT_DIAGNOSTICS_OUT`, pretty JSON.
- Match semantics: `prompt_tokens_match` = exact token-array equality; `generated_text_match` = exact string equality; `firstDifference()` returns first divergent index or **-1**.
- Output fields (verbatim, top level): backend, llama_server, model, message, expected_prompt, backend_prompt_tokens, llama_prompt_tokens, prompt_tokens_match, backend_generated_tokens, llama_generated_tokens, llama_generated_tokens_from_text, llama_top_logprobs, backend_diagnostic_token_ids, backend_dense_metadata, backend_top_logits, backend_output_projection, backend_dense, backend_text, llama_text, generated_text_match, first_generated_text_diff_index, backend_usage, llama_usage, camelid.
- Camelid response carries `camelid: { prompt_token_ids, generated_token_ids, dense_metadata, top_logits, step_top_logits, output_projection, timings_ms }` — the receipt `result` block maps directly onto `prompt_token_ids` / `generated_token_ids`.
- Sibling scripts: chat-parity-llama3/mistral/mixtral.mjs, test-chat-parity-harness.mjs.

## Existing artifact (verbatim fields)

Newest equivalent: `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260522T1828Z-origin-main-ffn-chain-complete-gate/parity-one-token.json`. Top-level keys: backend, llama_server, model, model_id, message, messages, render_mode, expected_prompt, expected_prompt_char_count, reference_prompt_token_count, reference_context, llama_flash_attn, llama_completion_prompt_kind, llama_server_args, prompt_tokens_match, generated_tokens_match, generated_text_match, first_generated_token_diff_index, first_generated_token_logit_comparison, first_generated_text_diff_index, backend_prompt_tokens, reference_prompt_tokens, backend_generated_tokens, llama_generated_tokens, llama_top_logprobs, backend_diagnostic_token_ids, backend_text, llama_text, backend_usage, llama_usage, camelid.

## GGUF hashing / provenance

- **No SHA-256 (or any digest) of the GGUF exists in src/** — Phase 3 must add it. No `sha2` in deps; adding `sha2 = "0.10"` (pure Rust, streaming `Digest` API). The repo's dependency-light directive was tokenizer-specific (no regex); mainstream crates (axum/tokio/uuid) are accepted.
- `build.rs` exists but embeds nothing about version/commit (only Accelerate linking + optional x86 AMX shim) — Phase 3 adds the `git rev-parse HEAD` / `git describe` build-time embed there.

## Build environment

Per workspace convention, build with the dedicated T7 target dir to avoid the global cargo-lock stall:
`CARGO_TARGET_DIR=/Volumes/Untitled/cargo-targets/Camelid-push cargo …`
