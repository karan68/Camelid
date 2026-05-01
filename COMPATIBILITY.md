# Camelid Compatibility Matrix

Last updated: 2026-05-01

`COMPATIBILITY.md` is Camelid's release contract. It defines what Camelid may describe as supported in the README, frontend readiness copy, release notes, and `/api/capabilities` without overstating the validated envelope. If another document sounds broader, this file wins.

Use this document to answer one release question: **may Camelid honestly say this exact lane is supported yet?** If a claim cannot be mapped to a specific row here, it should not appear in product copy, UI language, API readiness text, or release messaging. In practice, this is the source of truth product, docs, QA, backend, and frontend should reduce to before changing any public support language.

## Status labels

Treat the labels below as release language, not implementation optimism:

- **Supported** means the exact model family, tokenizer path, quantization, API surface, and evidence bundle are in place.
- **Evidence only** means the row has useful artifacts, but those artifacts do not promote neighboring rows.
- **Acceptance target** means Camelid has chosen the next exact lane to prove, not that runtime support already exists.
- **Groundwork only** means implementation or validation pieces exist, but the product must still say `not supported` until the blocking runtime and evidence work are complete.

## Release posture today

Camelid's public support language is intentionally narrow, evidence-bound, and easy to audit. The four rows below are the entire release posture Camelid may claim today, and every public surface should reduce back to this same ledger.

- **Supported generation gate:** TinyLlama 1.1B Chat Q8_0 is the only supported generation lane today. Camelid matches known-good llama-server behavior across the five-prompt, 50-token TinyLlama audit, including prompt token IDs, generated token arrays, and generated text.
- **Evidence-only lane:** Llama 3.2 1B Instruct Q8_0 has one compact-header `hello` prompt that matches llama.cpp for five deterministic generated tokens. That is narrow evidence, not broader Llama 3 support.
- **Acceptance target:** Llama 3.2 3B Instruct Q8_0 is the exact next WebUI real-chat target. The exact tracked GGUF is present locally, `/api/models/load` succeeds with low backend RSS after streaming metadata parsing, and the latest file-backed lazy-Q8 retry shows the earlier eager dense-load spike is materially reduced. The first guarded chat still stops before any token under host free-page pressure, so no 3B prompt-token, first-token, short-generation, parity, or WebUI evidence should be inferred from the 1B or 8B rows.
- **Groundwork-only lane:** Llama 3 8B Instruct Q8_0 has metadata/config/tokenizer/template evidence, independent tokenizer reference fixtures, a materialization-budget guard, Q8_0 block-only retained-weight groundwork, and serial row/all-row dot primitives in place, but generation remains blocked until lazy or on-demand Q8_0 linear execution is wired through attention, FFN, and output projection and QA captures bounded first-token parity and memory evidence.

## Operating rules

Nothing adjacent inherits support across model size, quantization, tokenizer lane, API surface, or frontend state. README, `STATUS.md`, `/api/capabilities`, and frontend readiness copy should continue to mirror this exact ledger.

`/api/capabilities` exposes the same compatibility rows as `model_compatibility`. Read each row literally: metadata parsing does not imply tokenizer parity, tokenizer parity does not imply generation, tensor loading does not imply safe execution, and one supported row must never lend support to adjacent model sizes or quantizations.

Executive summary: TinyLlama Q8_0 is the live supported gate; Llama 3.2 1B is a useful but narrow evidence row; Llama 3.2 3B is the chosen next acceptance target but remains blocked before first-token evidence; and Llama 3 8B is still groundwork only until its own bounded runtime artifacts exist.

## Current release ledger

The table below is the authoritative row-by-row support ledger reflected in `/api/capabilities`.

| Target | Family | Quant | Status | Metadata | Tokenizer | Tensors | Generation | Parity | Performance | Frontend | Exact evidence boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | LLaMA/SPM decoder | Q8_0 | Supported current gate | Validated | Validated | Validated | Validated | Validated | Measured | Validated | Five-prompt, 50-token parity gate against known-good llama-server. Artifact paths and timing notes live in `STATUS.md`. |
| Llama 3.2 1B Instruct Q8_0 | LLaMA decoder + Llama 3 BPE | Q8_0 | Evidence only / not a supported gate | Validated | Validated for compact prompt | Validated | One-prompt smoke only | 5-token evidence for one prompt | Early memory runs only | Not promoted | `bartowski/Llama-3.2-1B-Instruct-GGUF` loads locally on the 16 GiB Mac mini and matches llama.cpp for one compact-header `hello` prompt through five deterministic tokens `[9906,0,2650,649,358]` / `"Hello! How can I"`. This is narrow evidence only. |
| Llama 3.2 3B Instruct Q8_0 | LLaMA decoder + Llama 3 BPE | Q8_0 | Acceptance target / blocked before first token | Validated metadata/API load | Available but not parity-audited | Metadata validated; file-backed lazy-Q8 seam now partially wired | Still blocked before first generated token | Not started | Guarded failure artifact only | Frontend target card only | The exact tracked GGUF is present locally and `/api/models/load` succeeds with low backend RSS after streaming metadata parsing. The latest guarded backend-only retry with `BACKENDINFERENCE_LAZY_Q8_0_LINEAR=1` reduced the old dense-load spike substantially, but the safety guard still stopped the run before any token under host free-page pressure; see `STATUS.md`. |
| Llama 3 8B Instruct Q8_0 | LLaMA decoder + Llama 3 BPE | Q8_0 | Groundwork only / generation blocked | Real artifact inspected + config guarded | Reference parity guarded | Q8_0 block-only groundwork; eager f32 blocked by budget | Blocked until lazy/on-demand execution | Not started | Memory guarded | Not started | Local metadata confirms GQA heads 32/KV 8, RoPE base 500000, tokenizer `gpt2` with `llama-bpe`, 128256 tokens, 280147 merges, BOS 128000, EOS 128001, inferred EOT 128009, and the Llama 3 instruct template. `TensorStore::load_q8_0_blocks`, `Q8_0TensorBlocks`, serial `dot_row_f32`/`dot_all_rows_f32`, and `dot_single_input_row_f32` are groundwork only. |
| LLaMA/SPM Q4_0/Q5_0 | LLaMA/SPM decoder | Q4_0/Q5_0 | Planned Phase 10 | Descriptor guarded | Planned per model | Typed unsupported | Blocked until dequant | Not started | Not started | Not started | Fixture coverage proves descriptors parse, but CPU f32 loading rejects until real dequant/matmul support exists. |
| LLaMA/SPM Q4_K_M/Q5_K_M | LLaMA/SPM decoder | Q4_K_M/Q5_K_M | Planned Phase 10 | Descriptor guarded | Planned per model | Typed unsupported | Blocked until dequant | Not started | Not started | Not started | Start after simpler Q4_0/Q5_0 support has loader, matmul, and parity evidence. |
| Mistral GGUF | Mistral | Not selected | Planned model family | Not started | Not started | Not started | Not started | Not started | Not started | Not started | Choose a concrete target and tokenizer/chat-template fixtures before generation work. |

## Status-promotion checklist

Before any Phase 9-15 row moves from planned or blocked to supported, require all of the following for that exact target, quantization, API lane, and context bucket:

- A typed capability or unsupported-state change in `/api/capabilities` and matching documentation here.
- A reproducible command or test plus artifact path in `STATUS.md`.
- Independent reference or parity evidence whenever the claim is about tokenizer IDs, generated tokens/text, sampling, or context behavior.
- Memory/performance evidence that clearly distinguishes retained quantized weights, avoided `f32` materialization, bounded activation/output buffers, and any optimized-kernel determinism guardrail.

For **Llama 3.2 3B** specifically, tracked local model presence is now satisfied, and the latest file-backed lazy-Q8 retry is useful seam evidence only. The next promotable evidence is memory-safe bounded prompt-token and first-token parity with process/VM samples before any WebUI promotion.

For **Llama 3 8B** specifically, the next promotable evidence is not another tokenizer freshness pass or standalone `bench-q8-blocks` report. It is lazy or on-demand Q8_0 linear execution wired into attention, FFN, and output projection, followed by a bounded first-token generation retry with process-memory samples and parity or failure artifacts.

## Quantization formats

| Format | Status | Evidence / next action |
| --- | --- | --- |
| F32 | Supported reference path | CPU tensor path and fixture tests. |
| F16 | Supported reference path | Decoded into CPU tensor path with tests. |
| BF16 | Supported reference path | Decoded into CPU tensor path with tests. |
| Q8_0 | Supported current gate | TinyLlama Q8_0 parity gate; Q8 optimized block-dot remains guarded/opt-in unless parity evidence says otherwise. |
| Q4_0 / Q5_0 | Planned | Phase 10 legacy smaller-quant lane. |
| Q4_K_M / Q5_K_M | Planned | Phase 10 K-quant lane after simpler quant validation. |
| IQ / other GGUF quants | Future | Not implied support. |

## Model families

| Family | Status | Evidence / next action |
| --- | --- | --- |
| LLaMA/SPM decoder | Supported current gate | TinyLlama Q8_0 path; broader LLaMA-family validation planned. |
| Larger LLaMA-family instruct models | Planned | Phase 11 active expansion target after the TinyLlama gate; Llama 3-style GQA/RoPE-theta config and Llama 3 `gpt2`/`llama-bpe` tokenizer/template fixtures are guarded, but real tensor-load/generation/parity/performance evidence is still required. |
| LLaMA decoder + Llama 3 BPE | Planned / narrow small-model parity evidence | The concrete Llama 3 8B Instruct Q8_0 artifact's tokenizer metadata, merges, special IDs, and instruct template are fixture guarded, with independent llama.cpp `llama-tokenize --ids` reference IDs for the current prompts; the separate Llama 3.2 1B Instruct Q8_0 artifact has a one-prompt/5-token compact-header parity pass against llama.cpp; and the exact Llama 3.2 3B Instruct Q8_0 WebUI target now has a local exact artifact plus metadata/API-load evidence, but first generation is memory-blocked. Broader Llama 3 support, full chat-template behavior, 3B/8B generation, other BPE pre-tokenizers, frontend readiness, and performance remain unsupported until separately scoped. |
| Mistral-family GGUF | Planned | Evaluate after LLaMA-family evidence is stable. |
| Qwen / Gemma / Phi / Falcon / Mamba / others | Future | Track explicitly; do not claim until scoped, implemented, and audited. |

## Tokenizer and chat templates

| Surface | Status | Evidence / next action |
| --- | --- | --- |
| LLaMA/SPM tokenizer | Supported current gate | Includes whitespace, multiline, special/control-token, and EOS behavior from the current TinyLlama gate. |
| LLaMA marker chat template | Supported current gate | Current TinyLlama chat template path. |
| Llama 3 GPT-2/BPE `llama-bpe` tokenizer | Planned / reference parity guarded | Parses GGUF tokens, token types, merges, BOS/EOS, inferred EOT, and byte-unicode BPE encode/decode for the Llama 3 path; local metadata/tokenizer smokes validated Camelid artifact IDs for `hello`, ` hello`, `\n\n`, the rendered header prompt, `The quick brown fox jumps over the lazy dog.`, and `<|begin_of_text|>hello how's it going?`. Checked-in llama.cpp `llama-tokenize --ids` reference fixtures now assert the current prompt IDs, so this is tokenizer parity evidence only and not generation support. |
| Llama 3 instruct chat template | Planned / fixture guarded | Renders `<|start_header_id|>{role}<|end_header_id|>\n\n{trimmed content}<|eot_id|>` and appends the assistant header generation prompt with parse-special tokenization. |
| Other tokenizer families | Planned/future | Add detection, fixtures, known-good token-ID audits, and honest unsupported errors. |

## Context length

| Context bucket | Status | Evidence / next action |
| --- | --- | --- |
| Short prompt / 50-token audit | Supported current gate | Current TinyLlama Q8_0 gate. |
| 512 tokens | Planned | Phase 13 audit bucket. |
| 1k / 2k tokens | Planned | Phase 13 progressive audit buckets. |
| Model-native context | Future | Validate only where memory/performance permits. |

## API and provider surface

| Feature | Status | Evidence / next action |
| --- | --- | --- |
| `/v1/chat/completions` | Supported current gate | Non-streaming local generation for loaded supported dense GGUF models. |
| SSE streaming | Supported current gate | OpenAI-compatible token stream path exists for supported dense models. |
| `/v1/models`, `/api/models/load`, `/api/models/current` | Supported current gate | Local GGUF load/list/readiness path used by the frontend. |
| `/api/capabilities` | Supported current gate | Exposes explicit support contract, supported/planned quants, model families, and API features. |
| Multi-choice generation | Unsupported | Keep typed unsupported until implemented/tested. |
| Rich OpenAI-compatible logprobs | Partial/planned | Diagnostic logit surfaces exist; complete API parity remains Phase 14 work. |
| Local OpenAI-compatible provider registration | Open integration verification | Verify registration/use by the target local client surface before calling integration complete. |

## Phase 9-15 next actions and owners

- **Phase 9 — Support contract / Docs + Backend + QA + Frontend:** keep this matrix and `/api/capabilities` in lockstep; add typed unsupported coverage whenever a planned lane is visible to users; keep UI compatibility hints exact-row and quant-aware so saved paths, catalog entries, or same-family model names cannot inherit support without a matching row and `generation_ready=true`.
- **Phase 10 — Quantization / Backend + QA:** select one real Q4_0 or Q5_0 LLaMA/SPM GGUF target first; add loader/dequant tests, matmul parity evidence, and a real-model smoke before changing status from planned.
- **Phase 11 — Llama 3 / Backend + QA:** keep Llama 3 below support until each concrete target has the right artifact-backed evidence. For Llama 3.2 3B Q8_0, first requirement is the exact GGUF at the tracked model-dir path, then bounded prompt-token/first-token/5-token/API/WebUI acceptance evidence from `QA_LLAMA32_3B_Q8_ACCEPTANCE.md` without borrowing from the 1B or 8B rows. For Llama 3 8B, current eager CPU `f32` materialization is guarded off before host memory pressure and the exact next Backend/Performance action remains to route retained Q8_0 blocks plus `dot_row_f32`/`dot_all_rows_f32` and the `dot_single_input_row_f32` adapter through the LLaMA attention, FFN, and output-projection linear execution paths behind a lazy/on-demand seam, preserving the serial/block-aligned determinism guardrails until optimized kernels have separate zero-delta evidence.
- **Phase 12 — Tokenizers/templates / Backend + Docs:** Llama 3 `gpt2`/`llama-bpe` now has fixture-guarded Camelid token-ID/chat-template coverage plus independent llama.cpp reference IDs for the current prompts; require the same dual evidence before calling future tokenizer/template lanes parity-backed. Tokenizer parity alone is not generation readiness, and repeated green-light revalidations should be recorded as freshness evidence rather than status expansion.
- **Phase 13 — Context/KV / QA + Backend:** audit 512, 1k, and 2k context buckets after lazy 8B execution and a bounded first-token retry are artifact-backed; publish per-target tested context limits here and in readiness/API copy.
- **Phase 14 — API/sampling / Backend + QA:** leave multi-choice, `best_of`, and rich logprobs typed-unsupported until implemented with deterministic greedy and then seeded sampling coverage; frontend/API copy should keep those controls guarded or disabled.
- **Phase 15 — Performance/packaging / Performance + Docs:** keep the 8B-class materialization budget guard documented as the safe default and the Q8_0 block-only/serial row-dot/all-row-dot path framed as groundwork; carry deterministic-parallelism metadata (`serial_only_q8_0_block_rows`, no default parallel Q8 kernel, future serial-vs-parallel fail threshold `1e-7`) with memory evidence so optimized kernels require their own parity guardrails; re-baseline after lazy/on-demand execution and correctness milestones; document portable commands only after they are validated outside Tim-specific local paths.

Current evidence handoff: Llama 3 8B remains blocked at generation because eager `f32` materialization is intentionally stopped by the 24 GiB guard. The next status-changing evidence must be a lazy/on-demand Q8_0 execution slice that consumes the row-dot/all-row-dot primitives and rank-2 adapter across attention, FFN, and output-projection linear calls, plus a bounded first-token retry with process-memory samples. Current `bench-q8-blocks` memory fields and representative attention Q/K/V/output, FFN, and output-projection shape evidence should travel with that handoff: retained Q8 payload, avoided `f32` materialization, bounded dot input, and optional all-row output vector distinguish safe lazy-execution scratch/output buffers from unsafe eager `f32` weight decoding. The deterministic-parallelism metadata should travel too: current Q8 block rows are serial-only, no parallel Q8 kernel is enabled by default, and any future serial-vs-parallel comparison must target zero delta with a `1e-7` fail threshold before it can affect support claims. The independent reference token dumps for the existing Llama 3 fixture prompts are complete for tokenizer parity evidence, but they are not a generation-support prerequisite by themselves and do not unlock 8B generation.

Docs/frontend/API wording rule: Llama 3 rows may say metadata/config/tokenizer/template/Q8-block groundwork is present, but they must remain planned/blocked for generation, parity, performance, frontend readiness, and portable packaging until the lazy execution retry produces artifacts. TinyLlama Q8_0 is still the only supported current generation gate. Frontend cards should match compatibility rows by exact family + quant where possible, call out quant mismatches instead of falling back to same-family support, and reserve green/readiness styling for runtime `loaded_now=true` plus `generation_ready=true`.

## How to keep this matrix honest

- Update this file whenever a support claim changes.
- Keep `/api/capabilities` aligned with this file.
- Add artifacts and commands to `STATUS.md` when a new row moves from planned/future to supported.
- Prefer narrower truthful support over broad implied compatibility.
