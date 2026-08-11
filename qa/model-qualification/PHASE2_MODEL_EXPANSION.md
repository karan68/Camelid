# Phase 2 model expansion

This campaign evaluates 20 exact GGUF rows. It does not make family-wide claims: repository, revision, filename, byte size, SHA-256, tokenizer, template, and quantization are row identity.

The real artifacts and heavyweight build/oracle working sets live on external storage under `/Volumes/Untitled`. Small redacted receipts, fixtures, and policy records remain in the repository so the result is reproducible and reviewable.

## Current result

| Exact row | Family | Load/forward | Short greedy parity vs llama.cpp b9632 | Promotion state |
|---|---|---:|---:|---|
| LFM2.5 1.2B Instruct Q8_0 | LFM2 | pass | 6/6 pass | API/WebUI pass; 512-context pending |
| LFM2.5 1.2B Thinking Q8_0 | LFM2 | pass | fail at token 3 | hold |
| Gemma 3 270M-It Q8_0 | Gemma 3 | pass | 6/6 pass | API/WebUI pass; 512-context pending |
| Gemma 3 4B-It Q8_0 | Gemma 3 | pass | fail at token 3 | hold |
| Llama 3.1 8B Instruct Q8_0 | Llama 3.1 | pass | fail at token 3 | hold |
| Mistral 7B Instruct v0.2 Q8_0 | Mistral | pass | 6/6 pass | API/WebUI pass; 512-context pending |
| Qwen2.5 0.5B Instruct Q8_0 | Qwen2.5 | pass | 3/4; final-token flip | hold |
| Qwen2.5 1.5B Instruct Q8_0 | Qwen2.5 | pass | fail at token 4 | hold |
| Qwen2.5 Coder 1.5B Instruct Q8_0 | Qwen2.5 Coder | pass | 6/6 pass | API/WebUI pass; 512-context pending |
| Qwen3.5 0.8B Q8_0 | Qwen3.5 | pass | 6/6 pass | API/WebUI pass; 512-context pending |
| Qwen3.5 2B Q8_0 | Qwen3.5 | pass | 6/6 pass | API/WebUI pass; 512-context pending |
| Qwen3.5 4B Q8_0 | Qwen3.5 | pass | 3/6 | hold |
| Qwen3.5 9B Q8_0 | Qwen3.5 | pass | fail at token 3 | hold |
| DeepSeek R1 Distill Qwen 1.5B Q8_0 | DeepSeek/Qwen2.5 | pass | fail at token 3 | hold |
| DeepSeek R1 Distill Llama 8B Q8_0 | DeepSeek/Llama | pass | 6/6 pass | API/WebUI pass; 512-context pending |
| Phi-3 Mini 4K Instruct Q8_0 | Phi-3 | pass | incremental-decode failure | hold |
| Phi-4 Mini Instruct Q8_0 | Phi-4/Phi-3 GGUF | fail | blocked | hold |
| Gemma 2 9B-It Q8_0 | Gemma 2 | pass | 6/6 pass | API/WebUI pass; 512-context pending |
| SmolLM3 3B Q8_0 | SmolLM3 | pass | 6/6 raw pass | API/WebUI pass; template/context hold |
| Aya Expanse 8B Q4_K_M | Command-R | pass | 5/6; token-3 math flip | hold |

Nine rows pass the complete short raw-greedy oracle pack and all nine pass the guarded API/WebUI surface bundle. Eleven remain fail-closed: ten on deterministic parity or template scope and one on tensor binding. A passing short and surface pack is a support candidate, not yet a `supported_exact_row_smoke` promotion; the mandatory bounded 512-context gate still applies.

Machine-readable authorities:

- `qa/model-qualification/phase2-roster.json`
- `qa/model-qualification/phase2-runtime-matrix.json`
- `qa/model-qualification/phase2-runtime/*.json`
- `qa/model-qualification/fixtures/phase2/*.json`
