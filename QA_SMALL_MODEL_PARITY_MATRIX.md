# QA Small-Model Parity Matrix

Last updated: 2026-05-04

> [!NOTE]
> This matrix is a QA evidence summary, not the public support ledger. For current support truth,
> use [`COMPATIBILITY.md`](COMPATIBILITY.md), [`STATUS.md`](STATUS.md), and the owner matrix in
> [`FULL_SUPPORT_BLOCKER_MATRIX.md`](FULL_SUPPORT_BLOCKER_MATRIX.md).

## Scope

This matrix summarizes the four currently relevant Q8_0 rows without turning partial evidence into
full-support language:

- TinyLlama 1.1B Chat Q8_0
- Llama 3.2 1B Instruct Q8_0
- Llama 3.2 3B Instruct Q8_0
- Llama 3 8B Instruct Q8_0

## Matrix

| Target | Quant | Current QA position | Prompt-token parity | First-token parity | Short generation parity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| TinyLlama 1.1B Chat | Q8_0 | Supported gate evidence is green | PASS | PASS | PASS | Matches known-good llama-server on the active TinyLlama gate. Keep this as the release anchor and refresh artifacts when packaging the four-row evidence set. |
| Llama 3.2 1B Instruct | Q8_0 | Supported exact-row smoke | PASS for compact-header prompt and broader prompt pack | PASS | PASS for compact and broader short-generation packs | Exact 1B Instruct Q8_0 short local chat is smoke-supported only; longer context, stronger memory/perf, portability, and broader template coverage remain expansion gates. |
| Llama 3.2 3B Instruct | Q8_0 | Supported exact-row smoke | PASS for compact-header prompt and post-Q8-dot broader prompt pack | PASS | PASS for compact 1/5/50-token and broader 3-prompt/50-token packs | The previous JSON-shaped broader prompt blocker is fixed for the current pack. The row remains limited to exact-row short-chat smoke until longer-context, memory/perf, portability, and broader template evidence land. |
| Llama 3 8B Instruct | Q8_0 | Supported exact-row smoke | PASS for compact `hello`, broader 5-token pack, and first 512-context pack | PASS for compact `hello` and covered packs | PASS for compact `hello` 5-token, bounded 50-token, broader 5-token, and first 512-context pack | Exact 8B Instruct Q8_0 smoke is supported only for the documented envelopes; larger contexts, stronger memory/perf, portability, and broader template coverage remain expansion gates. |

## Current evidence summary

### TinyLlama 1.1B Chat Q8_0

- Prompt IDs match known-good reference.
- First generated token matches `29907` / `"C"`.
- Short deterministic generation matches.
- This is the live supported generation gate.

Representative artifacts cited by the public docs:

- `target/autonomous-small-model-parity-20260429T134615Z-head-9049492/tinyllama-q8-chat-parity-5tok.json`
- `target/chat-parity-postfix-50-token-audit.json`

### Llama 3.2 1B Instruct Q8_0

- Compact-header prompt IDs match known-good reference.
- First generated token matches `9906` / `"Hello"`.
- Compact deterministic generation matches `[9906,0,2650,649,358]` / `"Hello! How can I"`.
- The broader downloaded prompt pack also passed for prompt tokens, generated token IDs, and generated text.
- `/api/models/load`, `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence are documented for the exact row.
- This is a supported exact-row smoke lane, not broad Llama-family support.

Representative artifacts cited by the public docs:

- `target/autonomous-small-model-parity-20260429T134615Z-head-9049492/llama32-1b-q8-chat-parity-5tok.json`
- `target/qa-small-model-parity-20260429T1338Z-head-35bfd58/`
- `target/parity-50tok-20260502T031820Z/llama32-1b-50tok/report.json`
- `target/downloaded-llama-matrix-20260502T231000Z/summary.json`

### Llama 3.2 3B Instruct Q8_0

- The exact GGUF exists in the tracked model-dir lane used by the validation runs.
- Metadata and `/api/models/load` work for the exact row.
- Compact prompt-token, deterministic 1-token, deterministic 5-token, and bounded 50-token parity passed.
- The post-Q8-dot broader three-prompt 50-token pack passes for prompt tokens, generated token IDs, and generated text.
- `/v1/completions`, `/v1/chat/completions`, frontend smoke, and a five-prompt API smoke pack are documented for the exact row.
- This is a supported exact-row smoke lane, not broad Llama-family support.

Representative artifacts cited by the public docs:

- `target/parity-20260502T030911Z/llama32-3b-1tok/report.json`
- `target/parity-20260502T030911Z/llama32-3b-5tok/report.json`
- `target/parity-50tok-20260502T031820Z/llama32-3b-50tok/report.json`
- `target/camelid-regression-q8dot-20260502T232633Z/llama32-3b-compact/summary.json`
- `target/camelid-llama32-3b-broad-50-after-q8dot-clean-20260502T233427Z/pack/summary.json`

### Llama 3 8B Instruct Q8_0

- Tokenizer, metadata, config/template, retained-Q8, and lazy/file-backed Q8 groundwork exist.
- Compact-header `hello` now has prompt-token parity plus deterministic 1-token, 5-token, and bounded 50-token generation parity.
- Basic API smoke, frontend smoke, and bounded memory evidence are documented for the exact tracked Q8_0 GGUF.
- The later long-timeout broader 5-token pack passed for `hello`, alpacas, and JSON.
- The first bounded 512-context pack passed on the reopened Ubuntu validation lane and is summarized at `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json`.
- Do not treat this as broad/full 8B support: larger context buckets, broader chat-template coverage, support-grade memory/perf, portability, and synchronized docs/API/frontend promotion evidence remain required before widening the claim.

Representative artifacts cited by the public docs:

- `target/ubuntu-llama3-8b-q8-current-head-20260502T000207Z/`
- `target/parity-20260502T030911Z/llama3-8b-1tok/report.json`
- `target/parity-20260502T030911Z/llama3-8b-5tok/report.json`
- `target/parity-50tok-20260502T031820Z/llama3-8b-50tok/report.json`
- `target/downloaded-llama-matrix-20260502T231000Z/summary.json`
- `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`
- `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json`

## Artifact caveat

Most representative artifacts live under gitignored `target/` paths and are not present in a fresh
public checkout. That is fine for local validation, but a four-row full-support release should also
publish a durable artifact manifest with exact commands, model SHA256 values, current commit, and
checksums for every cited report.

## Usage rule

Treat this file as QA context only. Support changes must be reflected in `COMPATIBILITY.md`,
`STATUS.md`, `/api/capabilities`, and frontend readiness copy together.
