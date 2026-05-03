# Camelid Roadmap

Last updated: 2026-05-02

`ROADMAP.md` is Camelid's delivery plan of record. It is not a backlog and it is not a feature wish list. It answers one product question: **what must happen next for Camelid to widen its support boundary without weakening credibility?** The sequencing is intentional: protect the supported lane, remove the next exact blocker, and widen claims only when the resulting evidence can survive scrutiny.

[`COMPATIBILITY.md`](COMPATIBILITY.md) defines what Camelid can honestly support today. [`STATUS.md`](STATUS.md) records the artifacts, evidence boundaries, and blocker state behind that posture. Detailed completed-phase history lives in `ROADMAP_ARCHIVE.md` and `STATUS.md`. Read this file as operating sequence, not aspiration.

Executive summary: Camelid now has the TinyLlama supported gate plus exact Llama 3.2 1B/3B and Llama 3 8B short-chat/parity smoke gates. This roadmap exists to preserve those gates, widen them only with stronger row-specific evidence, and keep broader Llama-family work honest.

Practical reading rule: if a task does not protect the current gate, remove the next exact blocker, or prepare aligned support-language updates, it is secondary to this roadmap.

## Program objective

Camelid is not pursuing breadth for its own sake. The roadmap exists to expand capability only when the product can expand claims just as responsibly and defend them with row-specific evidence.

Current program posture:

- **Supported generation gates:** TinyLlama 1.1B Chat Q8_0 remains supported, and the exact Llama 3.2 1B/3B plus Llama 3 8B Instruct Q8_0 rows are smoke-supported for short local chat/parity.
- **Scope boundary:** the Llama support claim is exact-row only: model version/size, Instruct variant, Q8_0 quantization, loaded runtime readiness, and the short smoke/parity envelope all matter.
- **8B promoted lane:** Llama 3 8B Instruct Q8_0 now has compact parity, a long-timeout three-prompt 5-token Ubuntu parity run, API/frontend smoke, and bounded memory evidence for the exact tracked Q8_0 GGUF.
- **Explicit non-claim:** no broad Llama-family support exists today; neighboring variants remain unsupported unless they have their own exact row and evidence.

Nothing inherits support from a nearby size, quantization, family, tokenizer lane, API surface, or UI state.

Near-term thesis: protect the trusted TinyLlama and exact Llama 3.2 1B/3B plus Llama 3 8B smoke/parity gates, then broaden only with stronger row-specific evidence while every public surface stays synchronized with the exact support boundary.

## Roadmap operating rules

Three rules drive prioritization and sequencing:

- **Protect the current gate first.** TinyLlama Q8_0 remains the release anchor.
- **Remove the next honest blocker.** The highest-leverage work is the exact runtime seam that can create the next promotable artifact.
- **Move public surfaces together.** Documentation, API signals, and frontend readiness should change in the same change window.

## What changed in the support line

Recent work moved the release ledger only where the evidence, API, frontend, and docs now agree.

- TinyLlama Q8_0 remains the trusted release gate.
- Llama 3.2 1B Q8_0 is now a supported exact-row smoke lane after compact parity, broader prompt-pack parity, API smoke, and frontend smoke aligned.
- Llama 3.2 3B Q8_0 is now a supported exact-row smoke lane after exact-GGUF load, compact prompt-token/1-token/5-token/50-token parity, API smoke, and frontend smoke aligned.
- Llama 3.2 3B no longer has the JSON-shaped broader prompt-pack blocker; the post-Q8-dot clean three-prompt 50-token rerun now passes against llama.cpp.
- Llama 3 8B Q8_0 moved from groundwork-only to supported exact-row smoke after the long-timeout Ubuntu three-prompt 5-token parity run, API/frontend smoke, and bounded memory evidence aligned.

Near-term objective: preserve the supported TinyLlama and exact Llama 3.2 1B/3B plus Llama 3 8B short-chat/parity lanes, then widen only after row-specific evidence is in hand.

## Delivery sequence: now, next, later

This is the highest-level execution order. **Now** protects the current gate and clears the next blocker. **Next** is what Camelid may promote once bounded evidence exists. **Later** stays intentionally downstream of correctness and support-discipline work.

### Now

Protect the supported lanes and clear the next blocker before widening claims.

- Protect the validated TinyLlama Q8_0 gate.
- Protect the exact Llama 3.2 1B/3B and Llama 3 8B Instruct Q8_0 short-chat/parity smoke gates.
- Preserve the Llama 3.2 3B broader prompt-pack win while expanding only after longer-context, stronger performance/portability, and broader chat-template evidence land.
- Preserve the Llama 3 8B exact-row promotion while broadening prompt-pack length, chat-template coverage, longer context, WebUI/performance, and portability evidence without widening the support claim.
- Keep README, `COMPATIBILITY.md`, `ROADMAP.md`, `STATUS.md`, `/api/capabilities`, and frontend readiness copy aligned.

### Next

Promote only what can be defended row by row.

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
- For Qwen specifically, start with one exact GGUF target and do not schedule runtime-promotion work until tokenizer/chat-template fixtures, llama.cpp token-reference checks, and bounded load plus prompt-token parity are in place for that row.

## Milestone table

| Milestone | Status | What must be true |
| --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 supported gate | Complete | End-to-end generation parity artifacts exist and docs/API/frontend agree. |
| Llama 3.2 1B Instruct Q8_0 exact-row smoke | Complete / narrow support | Compact parity, broader prompt-pack parity, API smoke, and frontend smoke agree for this exact 1B Q8_0 row. |
| Llama 3.2 3B Instruct Q8_0 exact-row smoke | Complete / narrow support | Exact GGUF load, compact prompt-token/1-token/5-token/50-token parity, API smoke, and frontend smoke agree for this exact 3B Q8_0 row. |
| Llama 3 8B Instruct Q8_0 exact-row smoke | Complete / narrow support | Compact prompt-token/1-token/5-token/50-token parity, long-timeout three-prompt 5-token parity, API smoke, frontend smoke, and bounded memory evidence agree for this exact 8B Q8_0 row. |
| Quantization breadth beyond Q8_0 | Planned | Each quant format has loader/runtime tests, docs, and at least one row-specific real-model artifact. |
| Longer-context correctness | Planned | Context-length claims are backed by model-specific audits and documented limits. |
| API and sampling completeness | Planned | Newly supported fields have tests, honest docs, and typed unsupported errors removed only after implementation. |
| Performance and portability | Planned | Optimizations and platform claims are backed by reproducible measurements and stable behavior. |

## Active roadmap lanes

### Compatibility matrix and support contract

`COMPATIBILITY.md` is the support ledger. This roadmap governs when rows are allowed to move.

Current required discipline:

- TinyLlama 1.1B Chat Q8_0 remains a supported generation gate.
- Llama 3.2 1B Q8_0 is supported only as an exact-row short-chat smoke lane.
- Llama 3.2 3B Q8_0 is supported as an exact-row smoke lane with compact and broader three-prompt parity evidence; longer-context and broader chat-template expansion remain gated.
- Llama 3 8B Q8_0 is supported as an exact-row smoke/parity lane with compact parity, long-timeout three-prompt 5-token parity, API/frontend smoke, and bounded memory evidence; longer-context, broader chat-template, performance, and portability expansion remain gated.
- Frontend readiness must remain exact-row and exact-quant aware.

Promotion evidence must update docs, API capability reporting, and frontend readiness language in the same change window.

### Lazy Q8 execution for larger LLaMA-family models

This is the highest-leverage active engineering lane.

What exists now:

- retained Q8_0 block loading
- serial `dot_row_f32`, `dot_all_rows_f32`, and single-input-row adapters
- CPU materialization-budget guardrails
- Llama 3 tokenizer, config, GQA, and RoPE groundwork

What still needs to happen:

- wire retained-Q8 linear execution through attention projections
- wire it through FFN projections
- wire it through final output projection
- keep bounded scratch/output behavior explicit and measured
- verify first-token generation without unsafe eager dense materialization

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
