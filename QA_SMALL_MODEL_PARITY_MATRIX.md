# QA Small-Model Parity Matrix

Last updated: 2026-04-30

> [!NOTE]
> This matrix is a QA evidence summary, not the public support ledger. For current support truth,
> use [`COMPATIBILITY.md`](COMPATIBILITY.md) and [`STATUS.md`](STATUS.md).

## Scope

This matrix summarizes the currently relevant small-model evidence lanes without inventing results
for absent or still-blocked rows.

Current local rows of interest:

- TinyLlama 1.1B Chat Q8_0
- Llama 3.2 1B Instruct Q8_0
- Llama 3.2 3B Instruct Q8_0
- Llama 3 8B Instruct Q8_0

## Matrix

| Target | Quant | Current QA position | Prompt-token parity | First-token parity | Short generation parity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| TinyLlama 1.1B Chat | Q8_0 | Supported gate evidence is green | PASS | PASS | PASS | Matches known-good llama-server on the active TinyLlama gate. |
| Llama 3.2 1B Instruct | Q8_0 | Narrow evidence row only | PASS for compact-header prompt | PASS | PASS for one 5-token prompt | Useful evidence only; no broader support promotion. |
| Llama 3.2 3B Instruct | Q8_0 | Acceptance target, still blocked | NOT YET | NOT YET | NOT YET | Exact GGUF now exists and metadata/API load works, but the guarded first chat still stops before any generated token. |
| Llama 3 8B Instruct | Q8_0 | Groundwork only | Tokenizer/reference evidence only | NOT RUN | NOT RUN | Keep generation blocked until lazy or mmap or on-demand Q8 execution plus bounded evidence exist. |

## Current evidence summary

### TinyLlama 1.1B Chat Q8_0

- Prompt IDs match known-good reference.
- First generated token matches `29907` / `"C"`.
- Short deterministic generation matches.
- This is the live supported generation gate.

Representative artifacts:

- `target/autonomous-small-model-parity-20260429T134615Z-head-9049492/tinyllama-q8-chat-parity-5tok.json`
- `target/chat-parity-postfix-50-token-audit.json`

### Llama 3.2 1B Instruct Q8_0

- Compact-header prompt IDs match known-good reference.
- First generated token matches `9906` / `"Hello"`.
- One 5-token deterministic generation matches `[9906,0,2650,649,358]` / `"Hello! How can I"`.
- This remains a narrow evidence row only.

Representative artifacts:

- `target/autonomous-small-model-parity-20260429T134615Z-head-9049492/llama32-1b-q8-chat-parity-5tok.json`

### Llama 3.2 3B Instruct Q8_0

- Older missing-artifact notes are now stale.
- The exact GGUF exists at the tracked model-dir path.
- Metadata and `/api/models/load` work.
- The latest guarded first-chat retry still stopped before any generated token.

Representative artifacts:

- `target/llama32-3b-backend-only-smoke-20260430T193126Z/`
- `target/llama32-3b-lazy-q8-smoke-20260430T213910Z/`

### Llama 3 8B Instruct Q8_0

- Tokenizer, metadata, and retained-Q8 groundwork exist.
- No generation attempt should be treated as a support claim on the current memory budget.

## Usage rule

Treat this file as QA context only. Support changes must be reflected in `COMPATIBILITY.md`,
`STATUS.md`, `/api/capabilities`, and frontend readiness copy together.
