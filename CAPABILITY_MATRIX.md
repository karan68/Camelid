# Camelid Capability Matrix

Generated: 2026-06-25 · Platform: **Windows x86_64 (MSVC)**, backends CPU + CUDA · Oracle: llama.cpp `acd79d6` (MSVC)

This file is the **front door for capability coverage** under `MODEL_CAPABILITY_COVERAGE_CONDUCTOR.md`, the way `COMPATIBILITY.md` is the front door for *support*. The two axes are distinct (conductor §7): this matrix grows what each model can *do* in Camelid; it never edits support-ledger posture.

Per-cell vocabulary (conductor §5):

- `n/a` — the model genuinely **cannot** do this (its template/metadata cannot express it). Correct, not a gap. Never to be promoted.
- `open` — the model **can**, Camelid **cannot yet** (or actively rejects it). This is the work.
- `wip` — implemented in Camelid, but **no `camelid.capability-receipt/v1`** exists yet, so not claimable as done.
- `done → <receipt>` — implemented **and** validated by the matching oracle class, with a receipt.

**Phase 0 status:** 3 / 4 rows discovered + hash-anchored (TinyLlama, Llama 3.2 1B, Llama 3.2 3B). **Llama 3 8B Q8_0 is not downloaded** — its column (`*`) is PROVISIONAL (spec-derived, not GGUF-anchored, Phase 0 INCOMPLETE).

**Cell tally** (4 model columns × 13 capabilities = 52 cells): `n/a` 4 · `open` 17 · `wip` 13 · `done` 18. The `done` cells are each backed by a `camelid.capability-receipt/v1` under `qa/capability/receipts/`, validated this pass on the **3 on-disk rows** (TinyLlama, Llama 3.2 1B, Llama 3.2 3B) — no cross-row inheritance (conductor §6). The 8B column stays provisional until its GGUF is downloaded.

## Matrix

| Capability | Oracle | TinyLlama 1.1B | Llama 3.2 1B | Llama 3.2 3B | Llama 3 8B * | Camelid state |
| --- | :---: | :---: | :---: | :---: | :---: | --- |
| `gen.n_choices` | C | done | done | done | open | typed_error_stub |
| `gen.stream_usage` | C | done | done | done | open | drives |
| `gen.length_stop` | D/C | done | done | done | open | drives |
| `sampling.full_set` | I | done | done | done | open | partial |
| `sampling.seed_determinism` | I | done | done | done | open | drives |
| `logprobs.top_logprobs` | D | open | open | open | open | typed_error_stub |
| `chat.system_multiturn` | D | wip | wip | wip | open | drives |
| `tools.function_calling` | B | n/a | wip | wip | n/a | partial |
| `structured.json_grammar` | B | open | open | open | open | typed_error_stub |
| `context.full_length` | D | wip | wip | wip | open | drives |
| `context.rope_scaling` | D | n/a | wip | wip | n/a | drives |
| `observ.usage_timing` | C | done | done | done | open | drives |
| `load.quant_breadth` | D | wip | wip | wip | open | partial |

`*` Llama 3 8B column is provisional (GGUF not on disk). Oracle classes: **D**=deterministic parity (vs llama.cpp), **I**=invariant, **C**=contract/OpenAI-shape, **B**=behavioral battery.

## Per-capability notes (Camelid-side state & the work)

- **`gen.n_choices`** — Multi-choice n>1 independent generations (decoder/API)
  - DONE (class C, 3 on-disk rows): n>1 now runs independent generations with per-choice seed base+i (distinct yet reproducible), prompt counted once + completion summed; n>1+stream and n>1+receipt fail closed (HTTP 400). Was a 400 stub. Receipt minted.
- **`gen.stream_usage`** — OpenAI stream_options.include_usage terminal usage chunk (API)
  - DONE (class C, 3 on-disk rows): stream_options.include_usage emits a terminal usage chunk (empty choices + usage ints). Merged PR #321; receipt minted this pass.
- **`gen.length_stop`** — max_tokens, stop sequences, EOS stop behavior
  - DONE (class C, 3 on-disk rows): max_tokens honored + EOS stop observed e2e (finish_reason length/stop). Receipt minted. (Bit-exact stop-parity vs llama.cpp not separately run.)
- **`sampling.full_set`** — top_p,top_k,min_p,temperature,repetition/frequency/presence_penalty,seed,logit_bias
  - DONE (class I, 3 on-disk rows): min_p + repeat_penalty added to SamplingConfig + sampler + all request structs (were 400 stubs); 7 invariant tests + API-contract test + e2e accept/reject. Receipt minted.
- **`sampling.seed_determinism`** — fixed seed reproduces token-for-token across runs
  - DONE (class I, 3 on-disk rows): the degenerate fixed-per-seed RNG was replaced with a per-position SplitMix64 stream (seeded_unit_interval_at) — a fresh draw each decode step, still reproducible; e2e identical text across two seeded runs. Receipt minted.
- **`logprobs.top_logprobs`** — logprobs/top_logprobs at temp=0 (any causal LM)
  - OPEN — API rejects (HTTP 400 stub, api/mod.rs:6732). Deferred this pass: needs per-step full-vocab log_softmax capture in the shared decode loop (CPU + GPU-resident + spec lanes) + two OpenAI shapes + class-D oracle parity.
- **`chat.system_multiturn`** — system role + multi-turn template fidelity (per THIS template)
  - wip — per-arch renderers drive system + multi-turn (api/mod.rs:8789+); not exercised by this pass's smoke (single user turn). Needs a system+multi-turn e2e to earn its receipt.
- **`tools.function_calling`** — native tool/function calling — REQUIRES tool-call branch or tool/ipython control tokens in THIS model template
  - Input renders native tool protocol (Qwen3/Mistral, api/mod.rs:8904/9295) but NO structured tool_calls output; tool_choice stubbed (6750). Output-side is the work (1B/3B only).
- **`structured.json_grammar`** — JSON mode + GBNF grammar-constrained decode (decoder-side, model-agnostic)
  - OPEN — rejected (HTTP 400 stub, api/mod.rs:6753); no GBNF/grammar mask in the sampler.
- **`context.full_length`** — full TRAINED context length, exact value from metadata
  - wip — honored from GGUF metadata as KV cap (model.rs:141, inference.rs:2086). Trained length differs per row: TL 2048 / 1B 131072 / 3B 131072 / 8B 8192*. Needs near-limit validation (memory predict-and-abort).
- **`context.rope_scaling`** — RoPE scaling/extension — REQUIRES a rope scaling type/factor declared in metadata
  - Engine implements None/Linear/Llama3/YaRN (rope.rs:296-518). TinyLlama & 8B: no scaling (n/a). Llama 3.2 1B/3B: llama3 scaling baked into rope_freqs.weight (no metadata key) reaching 131072 → capable/wip; validate parity at extended positions vs llama.cpp.
- **`observ.usage_timing`** — prompt/completion/total token accounting + optional timing fields
  - DONE (class C, 3 on-disk rows): prompt/completion/total usage present + arithmetically correct e2e. Receipt minted. (ttft timing still not populated — not claimed.)
- **`load.quant_breadth`** — (SECONDARY) loadability across additional quants beyond Q8_0
  - Q8_0 supported; broader K-quant decode (Q4_K/Q5_0/Q6_K) gated/experimental (wire_dequant.rs). SECONDARY axis.

## Notable Phase 0 findings

- **`tools.function_calling` splits by row, exactly as the conductor demands.** TinyLlama → `n/a` (template has only user/system/assistant branches, no tool/ipython tokens). Llama 3.2 1B & 3B → **capable** (their embedded templates carry the Llama-3.1-style tool-call branch + `Environment: ipython`). Original Llama 3 8B → `n/a`/provisional (pre-3.1 template, no tool tokens — must be re-confirmed against the real GGUF, since a mislabeled 3.1 file would flip it).
- **`context.rope_scaling` splits 2 / 2 after a synthesis correction.** TinyLlama (native 2048) and original Llama 3 8B (native 8192) declare no scaling → `n/a`. The Llama 3.2 1B & 3B reach 131072 via **llama3 rope scaling baked into a `rope_freqs.weight` tensor** (no `rope.scaling.*` *metadata* key, but the transform is real and Camelid applies the llama3 path, `rope.rs:507`) → reclassified **capable / class D / `wip`**. The over-strict "metadata-key-required" Phase 0 prompt had marked these `n/a`; that was an underclaim (conductor §0/§4.6) and is corrected here. Their validation overlaps `context.full_length` at the 131072 frontier — parity at extended positions vs llama.cpp.
- **`context.full_length` differs sharply per row** — 2048 / 131072 / 131072 / 8192 — and must never be cross-claimed. Memory/abort projection (conductor §9) governs the 131072 rows before any near-limit validation.
- **6 capabilities are now `done` on the 3 on-disk rows**, each with a `camelid.capability-receipt/v1`: the **sampling lane** (`sampling.full_set` — `min_p`+`repeat_penalty` added; `sampling.seed_determinism` — degenerate per-seed RNG fixed to a per-step SplitMix64 stream, **a real correctness bug**), **`gen.n_choices`** (n>1 independent reproducibly-seeded choices, converted from a 400 stub), and the contract caps `gen.stream_usage`, `gen.length_stop`, `observ.usage_timing`.
- **`tools.function_calling` splits by row, exactly as the conductor demands.** TinyLlama → `n/a` (template has only user/system/assistant branches, no tool/ipython tokens). Llama 3.2 1B & 3B → **`wip`** (templates carry the Llama-3.1-style tool-call branch + `Environment: ipython`; Camelid renders the input protocol but emits no structured `tool_calls` yet). Original Llama 3 8B → `n/a`/provisional.
- **Two greenfield `open` lanes remain** (HTTP-400 stubs, model-agnostic): `logprobs.top_logprobs` and `structured.json_grammar`.
- **`context.full_length` differs sharply per row** — 2048 / 131072 / 131072 / 8192 — and must never be cross-claimed; near-limit validation (with memory predict-and-abort) keeps it `wip`. `context.rope_scaling` is `n/a` on TinyLlama/8B and `wip` on the 1B/3B (tensor-baked llama3 scaling — see below).

## Recommended sequencing (conductor §6) — progress

1. ✅ **Sampling lane** (`sampling.full_set` + `sampling.seed_determinism`) — `min_p`/`repeat_penalty` added, per-step RNG fixed, class-I invariants + e2e. **DONE.**
2. ◐ **`gen.n_choices` + `logprobs.top_logprobs`** — `gen.n_choices` **DONE** (class C); `logprobs.top_logprobs` **deferred** (decode-loop surgery + class-D oracle parity) — the clear next lane.
3. ✅ **Receipts for the already-driving caps** — `gen.stream_usage`, `gen.length_stop`, `observ.usage_timing` minted **DONE**; `chat.system_multiturn` + `context.full_length` still `wip` (need a system/multi-turn and a near-limit e2e respectively).
4. **`tools.function_calling`** (1B/3B only) — build the structured `tool_calls` output side; validate over a behavioral battery (class B). Gate strictly on the manifest (TinyLlama/8B stay `n/a`).
5. **`structured.json_grammar`** — GBNF/JSON-mode constrained decode (class B), all rows.
6. **Download Llama 3 8B Q8_0** to complete its Phase 0 manifest, re-resolve `tools`/`full_length`/`rope_scaling`, and extend the done receipts to that row.
7. *(secondary)* `load.quant_breadth` — only if widening the load matrix is in scope.

## Artifacts

- Per-row manifests: `qa/capability/capability-manifest.<row>.json` (schema `camelid.capability-manifest/v1`).
- Capability receipts: `qa/capability/receipts/capability-receipt.<cap>.<row>.json` (schema `camelid.capability-receipt/v1`) — **18 minted** this pass (6 caps × 3 on-disk rows).
- E2E harness: `qa/capability/smoke.sh` (boots each model on Windows CPU, exercises the validated caps).
- Checksums: `qa/capability/SHA256SUMS` (manifests + receipts).
- Provenance: produced on branch `feat/capability-conductor` (base `12f202d0`) with the sampling + n_choices diff **uncommitted** at receipt time — re-seal against the commit once landed.
