# Llama 3.2 3B Instruct Q8_0 Parity Acceptance

Last updated: 2026-05-02

> [!NOTE]
> This QA checklist is an acceptance document for one exact model row. It does not change the
> public support contract by itself. For current support truth, use [`COMPATIBILITY.md`](COMPATIBILITY.md)
> and [`STATUS.md`](STATUS.md).

QA checklist for the exact Llama 3.2 3B WebUI real-chat acceptance gate.

## Exact target artifact

- **Source repo:** `bartowski/Llama-3.2-3B-Instruct-GGUF`
- **Required filename:** `Llama-3.2-3B-Instruct-Q8_0.gguf`
- **Expected local path:** `$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf`
- **Resolve URL:** `https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf`
- **Expected size from earlier HEAD check:** `3,421,899,296 bytes` (`3.187 GiB`)
- **Earlier HEAD ETag/Xet hash:** `291ce1d4ca0fcef86407b7c6531bf85a1c348c65d5d3c69c57c98fec6483bb1f`

Current state: the exact GGUF is now present at the expected model-dir path, Camelid metadata/API
load evidence exists, and the Ubuntu compact-header `hello` harness now has prompt-token parity
plus deterministic 1-token, 5-token, and bounded 50-token generation parity. The blocker has
moved from artifact presence to broader prompt/chat-template, API, and WebUI acceptance.

## Current blocker summary

- `/api/models/load` succeeds for the exact 3B target.
- The latest file-backed lazy-Q8 recovery materially reduced the earlier eager dense-load spike.
- The Ubuntu compact-header `hello` harness now matches llama.cpp for prompt tokens plus deterministic 1-token, 5-token, and bounded 50-token generation.
- Broader Ubuntu prompt-pack runs continue to uncover a real exact-row divergence on the JSON-shaped prompt `answer with valid JSON for {"ok":true,"value":2}`. The latest downloaded-model matrix (`target/downloaded-llama-matrix-20260502T231000Z/summary.json`) again passed `hello` and the alpacas prompt for 3B, then failed the JSON-shaped prompt at the first generated token despite matching prompt tokens (`\`` vs ``\``, a close logit tie).
- Therefore the row remains blocked before broader prompt/chat-template coverage, API chat acceptance,
  WebUI acceptance, and stronger performance follow-up evidence.

## Disk and memory expectations

- Keep the artifact in the configured `$CAMELID_MODEL_DIR` location.
- Use bounded runs with process-memory sampling before any WebUI promotion.
- Do not infer safety from the 1B or 8B rows.

## Acceptance checklist

Do not mark the 3B row green until all applicable items have artifact paths.

1. **Model presence** — exact filename exists at the expected model-dir path; record size and hash.
2. **Readiness/inspect** — `scripts/small-model-readiness.mjs` or equivalent reports the row and
   records the exact blocker or safe candidate state.
3. **Rendered prompt** — capture the compact Llama 3 prompt Camelid currently renders.
4. **Reference token IDs** — use llama.cpp `llama-tokenize --ids` against the exact 3B GGUF.
5. **Camelid prompt-token parity** — run `scripts/chat-parity-llama3.mjs --require-prompt-match`.
6. **First generated token parity** — run deterministic greedy `--max-tokens 1 --require-generated-match`.
7. **Short greedy output parity** — run deterministic greedy `--max-tokens 5 --require-generated-match`.
8. **API load/chat smoke** — capture `/v1/health`, `/api/models/current`, `/api/models/tokenizer`,
   `/v1/chat/completions`, and process-memory samples.
9. **WebUI smoke** — only after API parity is green, capture real chat evidence plus memory samples.
10. **Regression preservation** — keep TinyLlama Q8_0 and Llama 3.2 1B evidence green.

## Current status

Status: **acceptance target with compact parity evidence**

The exact 3B artifact now exists, and the Ubuntu compact-header `hello` harness now matches
llama.cpp for prompt tokens plus deterministic 1-token, 5-token, and bounded 50-token generation.
Follow-on broader prompt-pack runs (`target/parity-broad-20260502T033606Z` and the downloaded-model
matrix at `target/downloaded-llama-matrix-20260502T231000Z/summary.json`) cleared `hello` and the
three-bullet alpaca prompt, then failed on the JSON-shaped prompt because Camelid and llama.cpp
selected different first generated backtick tokens even though prompt tokens still matched. The
current work is to fix that exact divergence, rerun the broader pack cleanly, and only then widen
the API/WebUI support contract.
