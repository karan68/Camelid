# Llama 3.2 3B Instruct Q8_0 Parity Acceptance

Last updated: 2026-05-01

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
load evidence exists, and one healthy Ubuntu backend-only first-token artifact now exists. The
blocker has moved from artifact presence to repeat bounded parity/API/WebUI acceptance.

## Current blocker summary

- `/api/models/load` succeeds for the exact 3B target.
- The latest file-backed lazy-Q8 recovery materially reduced the earlier eager dense-load spike.
- One healthy Ubuntu backend-only `/v1/completions` probe returned a first token for `hello`.
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
llama.cpp for prompt tokens plus deterministic 1-token and 5-token generation. The current work
is to widen that exact-row evidence into broader prompt coverage plus API and WebUI acceptance
without changing the support contract early.
