# Camelid Compatibility Matrix

Last updated: 2026-05-04

`COMPATIBILITY.md` is Camelid's release contract. It defines what Camelid may describe as supported in the README, frontend readiness copy, release notes, and `/api/capabilities` without overstating the validated envelope. If another document or UI sounds broader, this file wins.

Use this document to answer one release question: **may Camelid honestly say this exact lane is supported yet?** If a claim cannot be mapped to a specific row here, it should not appear in product copy, UI language, API readiness text, or release messaging.

Practical reading rule: if a statement cannot be reduced to an exact row in this file, Camelid should not publish that statement as product truth.

## Release-language definitions

Treat the labels below as release language, not implementation optimism:

- **Supported** means the exact model family, tokenizer path, quantization, API surface, and evidence bundle are in place.
- **Evidence only** means the row has useful artifacts, but those artifacts do not promote neighboring rows.
- **Acceptance target** means Camelid has chosen the next exact lane to prove, not that runtime support already exists.
- **Groundwork only** means implementation or validation pieces exist, but the product must still say `not supported` until the blocking runtime and evidence work are complete.

## Executive release posture

Camelid's public support language is intentionally narrow, evidence-bound, and easy to audit. The current answer is deliberately stricter than "it ran once":

- **Supported current gate:** TinyLlama 1.1B Chat Q8_0 remains the live supported gate.
- **Normalization rule:** Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B Instruct Q8_0 stay in exact-row validation until they meet the same normalized standard as TinyLlama: backend parity, API evidence, WebUI evidence, repeated runs, 50-token output, memory bounds, and current-head durable evidence bundles.
- **8B checkmark, not full support:** Llama 3 8B Instruct Q8_0 has valuable parity evidence, including a fresh 512-context pass, but it is not fully supported until the normalized evidence set is complete and repeatable.
- **Explicit non-claim:** no broad Llama 3-family support exists today; neighboring variants remain unsupported unless they have their own exact row and evidence.

Nothing adjacent inherits support. Support does not spread across nearby sizes, neighboring quantizations, matching tokenizers, or partial runtime seams.

## Governing rules

Two rules keep this matrix honest across docs, API signals, and UI copy:

- **Support rule:** nothing adjacent inherits support across model size, quantization, tokenizer lane, API surface, or frontend state.
- **Credit rule:** visible llama.cpp / ggml acknowledgement and the MIT notice remain part of parity-backed release claims.

README, `STATUS.md`, `/api/capabilities`, and frontend readiness copy should continue to mirror this exact ledger. `/api/capabilities` exposes the same compatibility rows as `model_compatibility`; read each row literally. Metadata parsing does not imply tokenizer parity, tokenizer parity does not imply generation, tensor loading does not imply safe execution, and one supported row must never lend support to adjacent model sizes or quantizations.

In plain terms: TinyLlama Q8_0 is still the only full supported gate; exact Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B Instruct Q8_0 rows are evidence-backed validation lanes until the same current-head full-support checklist is complete for each row.

## Durable evidence anchors

- `qa/evidence-bundles/four-row-public-20260503T024327Z/manifest.json` plus `qa/evidence-bundles/four-row-public-20260503T024327Z/SHA256SUMS` are the committed carry-forward row bundles/checksums for the public smoke boundary.
- `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json` is the committed four-row Ubuntu perf/portability summary.
- `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/manifest.json` plus its per-row manifests/checksums are the durable current-head citation layer for exact rerun tracks, blocker notes, and command files.
- Raw `target/` artifacts may appear as drill-down references, but they should not stand alone as the release-facing evidence anchor.

## Current release ledger

The compact table below is the authoritative release ledger reflected in `/api/capabilities`. It is intentionally short: what is the row, how far along is it, what is already true, and what must happen next.

Full-support requires the same checklist on every row: backend parity, API evidence, WebUI evidence, repeated runs, 50-token output, bounded memory/RSS evidence, and a current-head durable evidence bundle.

| Lane | Status | What Camelid can honestly say now | Missing before full support |
| --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | Supported current gate | End-to-end generation, parity, performance envelope, repeated 50-token runs, durable evidence, and frontend readiness are validated for the current five-prompt gate. | Preserve without regression; rerun when the current-head bundle is regenerated. |
| Llama 3.2 1B Instruct Q8_0 | Evidence-backed validation lane | Exact-row load, compact/broader parity, API smoke, and frontend smoke exist for this exact row. | Normalize repeated current-head 50-token parity, API/WebUI evidence, memory bounds, and durable bundle evidence to the TinyLlama bar. |
| Llama 3.2 3B Instruct Q8_0 | Evidence-backed validation lane | Exact-row load, compact prompt-token/1-token/5-token/50-token parity, broader three-prompt 50-token parity, API smoke, and frontend smoke exist for this exact row. | Normalize repeated current-head 50-token parity, API/WebUI evidence, memory bounds, and durable bundle evidence to the TinyLlama bar. |
| Llama 3 8B Instruct Q8_0 | Evidence-backed validation lane | Compact parity, three-prompt 5-token parity, 512-context parity, API/frontend smoke, and bounded-memory evidence exist for this exact row. | Repeat 50-token parity and 512-context evidence on current head, refresh API/WebUI evidence, close performance/RSS bounds, and publish the durable bundle before full support. |

### Row details

#### TinyLlama 1.1B Chat Q8_0
- **Family / quant:** LLaMA/SPM decoder, Q8_0
- **Validated now:** metadata, tokenizer, tensors, generation, parity, performance envelope, and frontend readiness
- **Promotion blocker:** none for the current supported claim

#### Llama 3.2 1B Instruct Q8_0
- **Family / quant:** LLaMA decoder + Llama 3 BPE, Q8_0
- **Validated now:** metadata, tokenizer path, tensor load, compact parity, broader prompt-pack parity, `/api/models/load`, `/v1/completions`, `/v1/chat/completions`, and frontend smoke are validated for the exact 1B Instruct Q8_0 row
- **Missing gates:** longer context, broader chat-template behavior, stronger memory/performance evidence, and portable packaging
- **Support boundary:** evidence-backed validation only for this exact 1B Instruct Q8_0 row; no full-support or neighboring-row claim yet

#### Llama 3.2 3B Instruct Q8_0
- **Family / quant:** LLaMA decoder + Llama 3 BPE, Q8_0
- **Validated now:** the exact tracked GGUF has validated metadata/load, compact prompt-token plus deterministic 1-token/5-token/50-token parity, broader three-prompt 50-token parity, `/v1/completions`, `/v1/chat/completions`, frontend smoke, and a five-prompt API smoke pack
- **Missing gates:** longer contexts, stronger memory/performance follow-up evidence, portable packaging, and broader chat-template coverage
- **Support boundary:** evidence-backed validation only for this exact 3B Instruct Q8_0 row; no full-support or neighboring-row claim yet

#### Llama 3 8B Instruct Q8_0
- **Family / quant:** LLaMA decoder + Llama 3 BPE, Q8_0
- **Validated now:** metadata/config/template handling, tokenizer reference parity, compact `hello` prompt-token/1-token/5-token/50-token parity, the three-prompt 5-token Ubuntu parity run, `/v1/completions`, `/v1/chat/completions`, frontend smoke, and bounded-memory evidence are validated for the exact 8B Instruct Q8_0 row
- **Missing gates:** longer-context evidence, larger prompt packs, stronger performance evidence, portable packaging evidence, and broader chat-template coverage
- **Support boundary:** evidence-backed validation only for this exact Llama 3 8B Instruct Q8_0 row; no full-support, longer-context, or neighboring-row claim yet

### Planned lanes

| Lane | Current state | Main blocker |
| --- | --- | --- |
| LLaMA/SPM Q4_0 / Q5_0 | Descriptor parsing is guarded and unsupported behavior is typed. | Real dequant/matmul support and full runtime evidence do not exist yet. |
| LLaMA/SPM Q4_K_M / Q5_K_M | Initial planning boundary only. | Start after simpler Q4_0/Q5_0 rows have concrete artifact-backed support work. |
| Mistral GGUF | No validated release evidence yet. | A concrete target row, tokenizer/template fixtures, and runtime evidence still need to be selected and built. |

## Status-promotion checklist

Before any Phase 9-15 row moves from planned or blocked to supported, require all of the following for that exact target, quantization, API lane, and context bucket:

- A typed capability or unsupported-state change in `/api/capabilities` and matching documentation here.
- A reproducible command or test plus artifact path in `STATUS.md`.
- Independent reference or parity evidence whenever the claim is about tokenizer IDs, generated tokens/text, sampling, or context behavior.
- Memory/performance evidence that clearly distinguishes retained quantized weights, avoided `f32` materialization, bounded activation/output buffers, and any optimized-kernel determinism guardrail.

For **Llama 3.2 1B** specifically, the committed carry-forward row bundle at `qa/evidence-bundles/four-row-public-20260503T024327Z/llama32_1b_instruct_q8_0.bundle.json` plus the current-head row manifest at `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/llama32_1b_instruct_q8_0/manifest.json` are the durable citation anchors for the exact-row smoke boundary. The public row bundle preserves the current compact/broader smoke evidence and checksums, while the current-head manifest pins the next longer-context, chat-template, and perf reruns to this exact row only.

For **Llama 3.2 3B** specifically, the committed carry-forward row bundle at `qa/evidence-bundles/four-row-public-20260503T024327Z/llama32_3b_instruct_q8_0.bundle.json` preserves the exact-row API/WebUI/compact-parity smoke boundary, and the current-head row manifest at `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/llama32_3b_instruct_q8_0/manifest.json` is the durable citation target for the post-Q8-dot broader three-prompt handoff plus the next context/template/perf tracks. The next promotable evidence is still longer context, stronger performance/portability, and broader chat-template coverage before expanding the claim.

For **Llama 3 8B** specifically, the durable citation anchors are the current-head row manifest at `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/llama3_8b_instruct_q8_0/manifest.json` plus the committed perf/portability envelope at `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json`; the public carry-forward bundle at `qa/evidence-bundles/four-row-public-20260503T024327Z/llama3_8b_instruct_q8_0.bundle.json` is intentionally a pre-promotion guarded-WebUI smoke slice from source smoke commit `c5e6d7e`. That keeps the exact-row short-smoke promotion boundary and the known 512-context blocker visible without widening beyond this row.

## Quantization formats

| Format | Status | Evidence / next action |
| --- | --- | --- |
| F32 | Supported reference path | CPU tensor path and fixture tests. |
| F16 | Supported reference path | Decoded into CPU tensor path with tests. |
| BF16 | Supported reference path | Decoded into CPU tensor path with tests. |
| Q8_0 | Supported for TinyLlama current gate; validation lanes for exact Llama rows | TinyLlama Q8_0 parity gate remains supported. Exact Llama 3.2 1B/3B and Llama 3 8B Instruct Q8_0 rows remain evidence-backed validation lanes until normalized full-support evidence is complete. |
| Q4_0 / Q5_0 | Planned | Phase 10 legacy smaller-quant lane. |
| Q4_K_M / Q5_K_M | Planned | Phase 10 K-quant lane after simpler quant validation. |
| IQ / other GGUF quants | Future | Not implied support. |

## Model families

| Family | Status | Evidence / next action |
| --- | --- | --- |
| LLaMA/SPM decoder | Supported current gate | TinyLlama Q8_0 path; broader LLaMA-family validation planned. |
| Larger LLaMA-family instruct models | Evidence-backed validation lanes | Exact Llama 3.2 1B/3B and Llama 3 8B Instruct Q8_0 have row-specific artifacts, but broad Llama-family behavior and full support remain blocked until each row meets the normalized TinyLlama-standard checklist. |
| LLaMA decoder + Llama 3 BPE | Exact-row validation pending normalization | The Llama 3.2 1B row has compact/broader parity plus API/WebUI smoke; the 3B row has compact and broader 50-token parity plus API/WebUI smoke; the 8B row has compact parity, 5-token broader parity, 512-context parity, API/WebUI smoke, and bounded-memory evidence. None of those rows is full support until repeated current-head 50-token, API, WebUI, memory-bound, and durable-bundle evidence is normalized across the row. |
| Mistral-family GGUF | Planned | Evaluate after LLaMA-family evidence is stable. |
| Qwen / Gemma / Phi / Falcon / Mamba / others | Future | Track explicitly; do not claim until scoped, implemented, and audited. For Qwen specifically, the first promotable prerequisite is one exact GGUF row with tokenizer/chat-template fixtures, llama.cpp token-reference checks, and bounded load plus prompt-token parity evidence before any runtime-support wording. |

## Tokenizer and chat templates

| Surface | Status | Evidence / next action |
| --- | --- | --- |
| LLaMA/SPM tokenizer | Supported current gate | Includes whitespace, multiline, special/control-token, and EOS behavior from the current TinyLlama gate. |
| LLaMA marker chat template | Supported current gate | Current TinyLlama chat template path. |
| Llama 3 GPT-2/BPE `llama-bpe` tokenizer | Exact-row smoke supported where paired with 1B/3B/8B evidence | Parses GGUF tokens, token types, merges, BOS/EOS, inferred EOT, and byte-unicode BPE encode/decode for the Llama 3 path; local metadata/tokenizer smokes validated Camelid artifact IDs for `hello`, ` hello`, `\n\n`, the rendered header prompt, `The quick brown fox jumps over the lazy dog.`, and `<|begin_of_text|>hello how's it going?`. Tokenizer parity contributes to the exact 1B/3B/8B smoke rows, but does not by itself unlock neighboring rows or broader generation support. |
| Llama 3 instruct chat template | Exact-row smoke supported where paired with 1B/3B/8B evidence | Renders `<|start_header_id|>{role}<|end_header_id|>\n\n{trimmed content}<|eot_id|>` and appends the assistant header generation prompt with parse-special tokenization; broader template behavior remains gated by row-specific prompt-pack evidence. |
| Other tokenizer families | Planned/future | Add detection, fixtures, known-good token-ID audits, and honest unsupported errors. |

## Context length

| Context bucket | Status | Evidence / next action |
| --- | --- | --- |
| Short prompt / 50-token audit | Supported for TinyLlama; normalization pending for exact Llama rows | TinyLlama has the current 50-token gate. Exact Llama rows need repeated current-head 50-token parity plus API/WebUI/memory evidence before full support. |
| 512 tokens | Planned | Phase 13 audit bucket. |
| 1k / 2k tokens | Planned | Phase 13 progressive audit buckets. |
| Model-native context | Future | Validate only where memory/performance permits. |

## API and provider surface

| Feature | Status | Evidence / next action |
| --- | --- | --- |
| `/v1/chat/completions` | Supported current gate; validation evidence for exact Llama rows | Non-streaming local generation is supported for loaded supported GGUF rows. Exact Llama 1B/3B/8B rows have smoke evidence but remain full-support normalization pending. |
| SSE streaming | Supported current gate | OpenAI-compatible token stream path exists for supported dense models. |
| `/v1/models`, `/api/models/load`, `/api/models/current` | Supported current gate; validation evidence for exact Llama rows | Local GGUF load/list/readiness path used by the frontend; exact Llama rows need normalized current-head reruns before full-support language. |
| `/api/capabilities` | Supported contract surface | Exposes explicit support contract, supported/planned quants, model families, and API features; row statuses must distinguish TinyLlama support from Llama validation lanes. |
| Multi-choice generation | Unsupported | Keep typed unsupported until implemented/tested. |
| Rich OpenAI-compatible logprobs | Partial/planned | Diagnostic logit surfaces exist; complete API parity remains Phase 14 work. |
| Local OpenAI-compatible provider registration | Open integration verification | Verify registration/use by the target local client surface before calling integration complete. |

## Phase 9-15 next actions and owners

- **Phase 9 — Support contract / Docs + Backend + QA + Frontend:** keep this matrix and `/api/capabilities` in lockstep; add typed unsupported coverage whenever a planned lane is visible to users; keep UI compatibility hints exact-row and quant-aware so saved paths, catalog entries, or same-family model names cannot inherit support without a matching row and `generation_ready=true`.
- **Phase 10 — Quantization / Backend + QA:** select one real Q4_0 or Q5_0 LLaMA/SPM GGUF target first; add loader/dequant tests, matmul parity evidence, and a real-model smoke before changing status from planned.
- **Phase 11 — Llama 3 / Backend + QA:** keep broad Llama 3 below support until each concrete target has the right artifact-backed evidence. For Llama 3.2 1B Q8_0, exact-row short-chat smoke support is now strengthened by a passing three-prompt llama.cpp pack in the downloaded-model matrix. For Llama 3.2 3B Q8_0, exact-row smoke support is now strengthened by compact parity, API, frontend, smoke-pack evidence, and the post-Q8-dot clean broader three-prompt 50-token parity rerun; longer-context, performance/portability, and broader chat-template evidence remain the next requirements before widening the row. For Llama 3 8B, compact prompt-token, 1-token, 5-token, and bounded 50-token parity, basic API/frontend smoke, bounded memory evidence, and the long-timeout three-prompt 5-token parity run now support the exact row; the next Backend/Performance action is to broaden prompt-pack length, longer context, chat-template coverage, performance, and portability evidence, preserving the serial/block-aligned determinism guardrails until optimized kernels have separate zero-delta evidence.
- **Phase 12 — Tokenizers/templates / Backend + Docs:** Llama 3 `gpt2`/`llama-bpe` now has fixture-guarded Camelid token-ID/chat-template coverage plus independent llama.cpp reference IDs for the current prompts; require the same dual evidence before calling future tokenizer/template lanes parity-backed. Tokenizer parity alone is not generation readiness, and repeated green-light revalidations should be recorded as freshness evidence rather than status expansion.
- **Phase 13 — Context/KV / QA + Backend:** audit 512, 1k, and 2k context buckets after lazy 8B execution and a bounded first-token retry are artifact-backed; publish per-target tested context limits here and in readiness/API copy.
- **Phase 14 — API/sampling / Backend + QA:** leave multi-choice, `best_of`, and rich logprobs typed-unsupported until implemented with deterministic greedy and then seeded sampling coverage; frontend/API copy should keep those controls guarded or disabled.
- **Phase 15 — Performance/packaging / Performance + Docs:** keep the 8B-class materialization budget guard documented as the safe default and the Q8_0 block-only/serial row-dot/all-row-dot path framed as groundwork; carry deterministic-parallelism metadata (`serial_only_q8_0_block_rows`, no default parallel Q8 kernel, future serial-vs-parallel fail threshold `1e-7`) with memory evidence so optimized kernels require their own parity guardrails; re-baseline after lazy/on-demand execution and correctness milestones; document portable commands only after they are validated outside Tim-specific local paths.

Current evidence handoff: Llama 3 8B now has compact-header `hello` parity against llama.cpp for prompt tokens plus deterministic 1-token, 5-token, and bounded 50-token generation, basic API/frontend smoke, bounded memory evidence, and a passing three-prompt 5-token Ubuntu parity run for the exact same row. The next status-changing evidence must broaden prompt-pack length, chat-template coverage, longer context, performance, and portability artifacts for that exact row. Current `bench-q8-blocks` memory fields and representative attention Q/K/V/output, FFN, and output-projection shape evidence should travel with that handoff: retained Q8 payload, avoided `f32` materialization, bounded dot input, and optional all-row output vector distinguish safe lazy-execution scratch/output buffers from unsafe eager `f32` weight decoding. The deterministic-parallelism metadata should travel too: current Q8 block rows are serial-only, no parallel Q8 kernel is enabled by default, and any future serial-vs-parallel comparison must target zero delta with a `1e-7` fail threshold before it can affect support claims. The independent reference token dumps for the existing Llama 3 fixture prompts are complete for tokenizer parity evidence, but they do not unlock broader 8B generation support by themselves.

Docs/frontend/API wording rule: Llama 3 rows may say metadata/config/tokenizer/template/Q8-block groundwork is present, and where true they may cite compact parity, API-smoke, frontend-smoke, and memory evidence. TinyLlama Q8_0 is the current supported generation gate; the exact Llama 3.2 1B/3B plus Llama 3 8B Instruct Q8_0 rows remain evidence-backed validation lanes until repeated current-head parity, API, WebUI, memory-bound, 50-token, and durable-bundle evidence is complete. Neighboring rows remain blocked. Frontend cards should match compatibility rows by exact family + quant where possible, call out quant mismatches instead of falling back to same-family support, and reserve green/readiness styling for runtime `loaded_now=true` plus `generation_ready=true`.

## How to keep this matrix honest

- Update this file whenever a support claim changes.
- Keep `/api/capabilities` aligned with this file.
- Add artifacts and commands to `STATUS.md` when a new row moves from planned/future to supported.
- Prefer narrower truthful support over broad implied compatibility.
