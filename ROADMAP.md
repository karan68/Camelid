# Camelid Roadmap

Last updated: 2026-05-09

`ROADMAP.md` is Camelid's delivery plan of record. It is not a backlog and it is not a feature wish list. It answers one product question: **what must happen next for Camelid to widen its support boundary without weakening credibility?** The sequencing is intentional: protect the supported lane, remove the next exact blocker, and widen claims only when the resulting evidence can survive scrutiny.

[`COMPATIBILITY.md`](COMPATIBILITY.md) defines what Camelid can honestly support today. [`STATUS.md`](STATUS.md) records the artifacts, evidence boundaries, and blocker state behind that posture. Detailed completed-phase history lives in `ROADMAP_ARCHIVE.md` and `STATUS.md`. Read this file as operating sequence, not aspiration.

Executive summary: Camelid now has the TinyLlama supported gate plus exact Llama 3.2 1B/3B/8B Q8_0 rows through bounded 2048-context packs on current `main` where row-specific PASS artifacts exist. The fresh 8B 1024/2048 PASS bundle at `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json` closes only those exact bounded buckets. Broader/full 8B support remains gated on model-native/larger context, arbitrary templates, production throughput, portability, and repeated durability evidence.

Practical reading rule: if a task does not protect the current gate, remove the next exact blocker, or prepare aligned support-language updates, it is secondary to this roadmap.

## Program objective

Camelid is not pursuing breadth for its own sake. The roadmap exists to expand capability only when the product can expand claims just as responsibly and defend them with row-specific evidence.

Current program posture:

- **Supported generation gates:** TinyLlama 1.1B Chat Q8_0 remains supported; exact Llama 3.2 1B/3B/8B rows are supported through checked bounded 512/1024/2048-context packs where row-specific PASS artifacts exist.
- **Scope boundary:** the Llama support claim is exact-row only: model version/size, Instruct variant, Q8_0 quantization, loaded runtime readiness, and the checked smoke/parity/context envelope all matter.
- **8B promoted lane:** Llama 3 8B Instruct Q8_0 now has compact parity, a three-prompt 50-token Ubuntu parity run, API/frontend smoke, bounded memory evidence, checked bounded 512/1024/2048-context packs, and one bounded compact chat-template-shapes pack for the exact tracked Q8_0 GGUF; broader/full 8B support remains gated.
- **Explicit non-claim:** no broad Llama-family support exists today; neighboring variants remain unsupported unless they have their own exact row and evidence.

Nothing inherits support from a nearby size, quantization, family, tokenizer lane, API surface, or UI state.

Near-term thesis: protect the trusted TinyLlama gate plus the exact Llama 3.2 1B/3B/8B bounded-2048 rows; broaden only with stronger row-specific evidence while every public surface stays synchronized with the exact support boundary.

## Roadmap operating rules

Four rules drive prioritization and sequencing:

- **Protect the current gate first.** TinyLlama Q8_0 remains the release anchor.
- **Remove the next honest blocker.** The highest-leverage work is the exact runtime seam that can create the next promotable artifact.
- **Move public surfaces together.** Documentation, API signals, and frontend readiness should change in the same change window.
- **Cite committed evidence anchors first.** The public bundle manifest/checksums, perf/portability envelope, reopened-lane API + frontend smoke manifest, 1B/3B bounded 1024/2048-context bundles, the current-head 8B 1024/2048 bundle, 8B broader 50-token bundle, 8B 512-context bundle, 8B compact chat-template-shapes bundle, and current-head per-row manifests are the roadmap-facing evidence layer; raw `target/` artifacts are drill-down only.

## What changed in the support line

Recent work moved the release ledger only where the evidence, API, frontend, and docs now agree.

- TinyLlama Q8_0 remains the trusted release gate.
- Llama 3.2 1B Q8_0 is now a supported exact-row smoke lane after compact parity, broader prompt-pack parity, API smoke, frontend smoke, and bounded 512/1024/2048 context-pack evidence aligned; the 2048 pack turned green only after the RoPE frequency-factor fix.
- Llama 3.2 3B Q8_0 is now a supported exact-row smoke lane after exact-GGUF load, compact prompt-token/1-token/5-token/50-token parity, API smoke, frontend smoke, and bounded 512/1024/2048 context-pack evidence aligned.
- Llama 3.2 3B no longer has the JSON-shaped broader prompt-pack blocker; the post-Q8-dot clean three-prompt 50-token rerun now passes against llama.cpp.
- Llama 3 8B Q8_0 moved from groundwork-only to supported exact-row smoke after Ubuntu three-prompt parity, API/frontend smoke, bounded memory evidence, checked bounded 512/1024/2048-context packs, and compact chat-template-shapes packs aligned for that exact row only.

Near-term objective: preserve the supported TinyLlama gate and exact Llama 3.2 1B/3B/8B bounded-2048 lanes; do not widen to model-native/larger context, arbitrary templates, production throughput, portability, or adjacent rows without new evidence.

## Delivery sequence: now, next, later

This is the highest-level execution order. **Now** protects the current gate and clears the next blocker. **Next** is what Camelid may promote once bounded evidence exists. **Later** stays intentionally downstream of correctness and support-discipline work.

### Now

Protect the supported lanes and clear the next blocker before widening claims.

- Protect the validated TinyLlama Q8_0 gate.
- Protect the exact Llama 3.2 1B/3B/8B bounded-2048 rows.
- Preserve the Llama 3.2 1B/3B broader prompt-pack plus bounded 512/1024/2048 context-pack wins while expanding only after model-native/larger-context, stronger performance/portability, and broader chat-template evidence land.
- Preserve the Llama 3 8B exact-row promotion through the checked 512/1024/2048-context packs on current `main`; older 1024/2048 bundles remain historical source-head evidence only.
- Keep README, `COMPATIBILITY.md`, `ROADMAP.md`, `STATUS.md`, `/api/capabilities`, and frontend readiness copy aligned.

### Next

Promote only what can be defended row by row.

- Close the active next-model bring-up set as exact-row evidence lanes first, never as family-wide support claims. **Mixtral 8x7B Instruct** is now exact-row supported for the checked short-prompt MoE/API/WebUI/RSS envelope; current work is continuation/long-generation hardening, where the exact prompt-token continuation path is being tightened but the public claim does not widen until separate long-context and broader-prompt evidence is green.
- Widen Llama 3.2 3B Q8_0 beyond short-chat smoke only if broader prompt/chat-template, memory/performance, API, and WebUI evidence all land.
- Broaden quantization support beyond Q8_0 with tests, docs, and exact-row evidence.
- Expand tokenizer and chat-template coverage for additional supported rows.
- Extend correctness checks into longer prompt and context buckets.

### Later

Broaden the product surface only after correctness and release discipline are stable.

- Richer OpenAI API completeness beyond the current supported subset.
- Measured performance optimization after correctness gates are stable.
- Packaging and portability work across non-primary platforms.
- Broader model-family expansion beyond current LLaMA-family priorities.
- First-class multi-model concurrency so Camelid can keep multiple local models loaded at once and serve agent/OpenClaw workloads that need different models simultaneously.
- For Qwen specifically, start with one exact GGUF target and do not schedule runtime-promotion work until tokenizer/chat-template fixtures, llama.cpp token-reference checks, and bounded load plus prompt-token parity are in place for that row.

## Milestone table

| Milestone | Status | What must be true |
| --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 supported gate | Complete | End-to-end generation parity artifacts exist and docs/API/frontend agree. |
| Llama 3.2 1B Instruct Q8_0 exact-row smoke | Complete / narrow support | Compact parity, broader prompt-pack parity, API smoke, frontend smoke, and bounded 512/1024/2048 context packs agree for this exact 1B Q8_0 row; the 2048 pack is exact-row only after the RoPE frequency-factor fix. |
| Llama 3.2 3B Instruct Q8_0 exact-row smoke | Complete / narrow support | Exact GGUF load, compact prompt-token/1-token/5-token/50-token parity, broader three-prompt parity, API smoke, frontend smoke, and bounded 512/1024/2048 context packs agree for this exact 3B Q8_0 row. |
| Llama 3 8B Instruct Q8_0 exact-row smoke | Complete / narrow support through checked 512/1024/2048 bounded packs | Compact prompt-token/1-token/5-token/50-token parity, the three-prompt 50-token pack, API smoke, frontend smoke, bounded memory evidence, checked bounded 512/1024/2048-context packs, and the compact chat-template-shapes pack support this exact 8B Q8_0 row only. |
| Quantization breadth beyond Q8_0 | Planned | Each quant format has loader/runtime tests, docs, and at least one row-specific real-model artifact. |
| Longer-context correctness | Planned | Context-length claims are backed by model-specific audits and documented limits. |
| API and sampling completeness | Planned | Newly supported fields have tests, honest docs, and typed unsupported errors removed only after implementation. |
| Performance and portability | Planned | Optimizations and platform claims are backed by reproducible measurements and stable behavior. |

## Active roadmap lanes

### Compatibility matrix and support contract

`COMPATIBILITY.md` is the support ledger. This roadmap governs when rows are allowed to move.

Current required discipline:

- TinyLlama 1.1B Chat Q8_0 remains a supported generation gate.
- Llama 3.2 1B Q8_0 is supported as an exact-row smoke lane with compact/broader parity plus bounded 512/1024/2048 context-pack evidence after the RoPE frequency-factor fix; model-native/larger-context and broader chat-template expansion remain gated.
- Llama 3.2 3B Q8_0 is supported as an exact-row smoke lane with compact and broader three-prompt parity plus bounded 512/1024/2048 context-pack evidence; model-native/larger-context and broader chat-template expansion remain gated.
- Llama 3 8B Q8_0 is supported as an exact-row smoke/parity lane with compact parity, the three-prompt 50-token pass, API/frontend smoke, bounded memory evidence, checked bounded 512/1024/2048-context packs, and one compact chat-template-shapes pack; model-native/larger context beyond checked packs, broader chat-template, production performance, and portability expansion remain gated.
- Frontend readiness must remain exact-row and exact-quant aware.
- Support-language updates should point first to the committed `qa/evidence-bundles/...` manifests/checksums and only then to raw `target/` drill-down artifacts.

Promotion evidence must update docs, API capability reporting, and frontend readiness language in the same change window.

### Lazy Q8 execution for larger LLaMA-family models

This is the highest-leverage active engineering lane.

What exists now:

- retained Q8_0 block loading
- serial `dot_row_f32`, `dot_all_rows_f32`, and single-input-row adapters
- CPU materialization-budget guardrails
- Llama 3 tokenizer, config, GQA, and RoPE groundwork
- a code-only chunked prefill slice (`CAMELID_PREFILL_CHUNK_TOKENS`, default `128`) that batches non-final prompt tokens through embedding, Q/K/V, RoPE, KV writes, causal attention context, attention output, and FFN while leaving the final logits token on the established single-token path
- Q8_0 file-backed batched matmul read reuse across input rows for bounded prefill chunks, plus a layer-major lazy-Q8 prefill schedule that reuses each layer's file-backed weights across all prefill chunks before moving to the next layer

What still needs to happen:

- measure chunked prefill on approved row-specific runtime lanes before using it in support claims
- keep retained-Q8 linear execution wired through attention, FFN, and final output projection without unsafe eager dense materialization
- keep bounded scratch/output behavior explicit and measured
- verify first-token and longer-prompt generation with row-specific parity/RSS evidence before promoting any larger context box

What does **not** count as promotion evidence by itself:

- tokenizer freshness
- metadata load success
- standalone block benchmarks
- artifact presence on disk

### Quantization breadth

Camelid should broaden quant support only after the larger-model Q8 execution seam is trustworthy.

Priority shape:

- keep Q8_0 as the correctness baseline
- add the next real-world quant formats with the highest practical value
- require loader tests, runtime math checks, and at least one row-specific real-model artifact per supported quantization

No quant format is supported just because its metadata parses.

### Tokenizer and chat-template expansion

Tokenizer support remains part of the release contract, not a side detail.

Near-term expectations:

- preserve the current LLaMA/SPM and Llama 3 template behavior
- add fixtures for whitespace, multiline prompts, control tokens, EOS behavior, and prompt-shape edge cases
- keep unsupported tokenizer families as typed unsupported states until a full support lane exists

Tokenizer parity alone does not promote generation support.

### Longer-context correctness

Short-prompt success is not enough for broader support claims.

This lane should expand in bounded steps:

- validated short prompts
- 512-token bucket
- 1k-token bucket
- 2k-token bucket
- larger model-specific buckets only when memory/runtime evidence supports them

For each promoted context bucket, Camelid should have:

- prompt-token evidence
- generation evidence where applicable
- clear model-specific documented limits
- no hidden inference from nearby rows

### OpenAI API and sampling completeness

Camelid already exposes a narrow but real OpenAI-compatible local surface. The roadmap here is to expand completeness without faking compatibility.

Active rule set:

- implement deterministic correctness first
- keep unsupported combinations as typed errors until behavior is real
- add richer fields only with tests and documentation

Near-term candidates include:

- richer logprob support
- broader streaming metadata completeness
- multi-choice generation
- stronger seeded sampling validation

### Performance, packaging, and portability

Performance work matters, but it should follow correctness and support honesty.

Execution order:

- preserve the validated baseline
- measure bottlenecks after each correctness milestone
- optimize only where evidence says it matters
- keep optimized kernels behind parity guardrails until proven

Portability and packaging should remain explicit:

- no implied non-macOS support without validation
- no implied portable model-path assumptions without documentation
- no release packaging claim before reproducible setup instructions exist

## Promotion rules

A row may move forward only when all of the following are true:

1. Runtime behavior works for the exact row being claimed.
2. Evidence is captured for the exact scope being promoted.
3. Documentation says exactly what the evidence supports and nothing broader.
4. API capability reporting reflects the same boundary.
5. Frontend readiness and UI language reflect the same boundary.
6. Unsupported adjacent rows remain visibly unsupported.

Practical examples:

- A 1B row does not promote a 3B or 8B row.
- Metadata load does not promote generation support.
- Tokenizer parity does not promote runtime readiness.
- A first-token artifact does not automatically promote longer-context correctness.
- A benchmark does not promote portable packaging or production-readiness claims.

## Non-goals

For the current roadmap window, Camelid is **not** trying to:

- match every feature of mature inference runtimes
- claim broad LLaMA-family support from a narrow artifact set
- treat local artifact presence as runtime support
- infer readiness across neighboring sizes or quantizations
- advertise hosted/provider/catalog features that are not wired and tested
- prioritize GPU acceleration ahead of stable CPU correctness and evidence-backed model breadth

## Archived and completed phases

Early repo setup, backend skeleton, GGUF metadata parsing, tokenizer bring-up, tensor loading, and first-generation-lane work are complete enough that they no longer need full tactical detail here.

See:

- `ROADMAP_ARCHIVE.md` for concise completed-phase history
- `STATUS.md` for tactical runs, artifact paths, benchmark outputs, and diagnostic notes

The important completed milestone for current planning is simple: Camelid has one validated TinyLlama Q8_0 end-to-end generation gate, and every future milestone must preserve that trust.
