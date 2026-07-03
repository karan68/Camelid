# Camelid Capability Matrix

Generated: 2026-06-25 · Platform: **Windows x86_64 (MSVC)**, backends CPU + CUDA · Oracle: llama.cpp `acd79d6` (MSVC)

This file is the **front door for capability coverage** under `MODEL_CAPABILITY_COVERAGE_CONDUCTOR.md`, the way `COMPATIBILITY.md` is the front door for *support*. The two axes are distinct (conductor §7): this matrix grows what each model can *do* in Camelid; it never edits support-ledger posture.

Per-cell vocabulary (conductor §5):

- `n/a` — the model genuinely **cannot** do this (its template/metadata cannot express it). Correct, not a gap. Never to be promoted.
- `open` — the model **can**, Camelid **cannot yet** (or actively rejects it). This is the work.
- `wip` — implemented in Camelid, but **no `camelid.capability-receipt/v1`** exists yet, so not claimable as done.
- `done → <receipt>` — implemented **and** validated by the matching oracle class, with a receipt.

**Phase 0 status:** 3 / 4 rows discovered + hash-anchored (TinyLlama, Llama 3.2 1B, Llama 3.2 3B). **Llama 3 8B Q8_0 is not downloaded** — its column (`*`) is PROVISIONAL (spec-derived, not GGUF-anchored, Phase 0 INCOMPLETE).

**Cell tally** (4 model columns × 13 capabilities = 52 cells): `n/a` 4 · `open` 11 · `wip` 5 · `done` 32. The `done` cells are each backed by a `camelid.capability-receipt/v1` under `qa/capability/receipts/`, validated on the **3 on-disk rows** (TinyLlama, Llama 3.2 1B, Llama 3.2 3B) — no cross-row inheritance (conductor §6). The 8B column stays provisional until its GGUF is downloaded.

Two `wip` cells are **host-limited, not model-limited**: `context.full_length` on Llama 3.2 1B and 3B is validated bit-exact across a long context (feasible frontier 8511 / 8304 tokens), but their trained **131072** context materializes an **8.0 GiB / 28 GiB** f32 CPU KV cache that this 15.74 GiB box cannot safely hold — so those cells stay `wip` (the model supports it; this host cannot reach it). See `qa/validation-notes/2026-06-25-capability-context-host-limits.md`, which also flags that Camelid has no pre-flight KV predict-and-abort. The remaining `wip` is `load.quant_breadth` ×3 (secondary axis).

## Matrix

| Capability | Oracle | TinyLlama 1.1B | Llama 3.2 1B | Llama 3.2 3B | Llama 3 8B * | Camelid state |
| --- | :---: | :---: | :---: | :---: | :---: | --- |
| `gen.n_choices` | C | done | done | done | open | typed_error_stub |
| `gen.stream_usage` | C | done | done | done | open | drives |
| `gen.length_stop` | D/C | done | done | done | open | drives |
| `sampling.full_set` | I | done | done | done | open | partial |
| `sampling.seed_determinism` | I | done | done | done | open | drives |
| `logprobs.top_logprobs` | D | done | done | done | open | drives |
| `chat.system_multiturn` | D | done | done | done | open | drives |
| `tools.function_calling` | B | n/a | done | done | n/a | drives |
| `structured.json_grammar` | B | done | done | done | open | drives |
| `context.full_length` | D | done | wip | wip | open | drives |
| `context.rope_scaling` | D | n/a | done | done | n/a | drives |
| `observ.usage_timing` | C | done | done | done | open | drives |
| `load.quant_breadth` | D | wip | wip | wip | open | partial |

`*` Llama 3 8B column is provisional (GGUF not on disk). Oracle classes: **D**=deterministic parity (vs llama.cpp), **I**=invariant, **C**=contract/OpenAI-shape, **B**=behavioral battery.

## macOS / Metal Matrix (WIN2METAL Phase 2 · Bucket A)

Validated: 2026-06-26 · Platform: **macOS (Apple M4, 16 GB)**, backend **Metal (GPU-resident decode + prefill)** · Oracle: llama.cpp `9632` (`acd79d603`, **Metal**) · Binary: **frozen base `28f224b`** (`camelid v0.1.7-92-g28f224b`, `/Volumes/Untitled/camelid-base-28f224b`) — **validation only, no `src/` changes**.

Per-cell vocabulary for this platform (conductor §5 + the WIN2METAL honesty rule): `done → <receipt>` = re-validated on Metal with a `camelid.capability-receipt/v1` under `macos/receipts/`; `host-limited` = the row cannot be exercised on this 16 GB Metal box (a clean limit, **not** a pass and **not** a model gap); `gap` = a real Metal-specific failure (**none found**); `n/a` = the model genuinely cannot (template/metadata-intrinsic, platform-independent).

| Capability | Oracle | Llama 3.2 1B (Metal) | Llama 3.2 3B (Metal) | TinyLlama | Llama 3 8B |
| --- | :---: | :---: | :---: | :---: | :---: |
| `gen.n_choices` | C | done | done | host-limited | host-limited |
| `gen.stream_usage` | C | done | done | host-limited | host-limited |
| `gen.length_stop` | D/C | done | done | host-limited | host-limited |
| `sampling.full_set` | I | done | done | host-limited | host-limited |
| `sampling.seed_determinism` | I | done | done | host-limited | host-limited |
| `logprobs.top_logprobs` | D | done | done | host-limited | host-limited |
| `chat.system_multiturn` | D | done | done | host-limited | host-limited |
| `tools.function_calling` | B | done | done | n/a | n/a |
| `structured.json_grammar` | B | done | done | host-limited | host-limited |
| `context.full_length` | D | host-limited | host-limited | host-limited | host-limited |
| `context.rope_scaling` | D | done | done | n/a | n/a |
| `observ.usage_timing` | C | done | done | host-limited | host-limited |
| `load.quant_breadth` | D | — | — | — | — |

**Mac cell tally (the 2 on-disk rows):** **22 `done`** (11 capabilities × {1B, 3B}), each backed by a Metal `camelid.capability-receipt/v1` under `qa/capability/macos/receipts/`. No cross-row inheritance. **0 `gap`** — nothing regressed on the Metal backend.

**Why TinyLlama / 8B are `host-limited` here:** the TinyLlama GGUF is not on this Mac's disk and Llama 3 8B Q8_0 OOMs at 16 GB — neither is a model limit, just this host. Their `tools.function_calling` and `context.rope_scaling` cells stay `n/a` because that is **model-intrinsic** (no tool branch / no rope scaling), true regardless of platform. `context.full_length` is `host-limited` on **every** row: the trained 131072 context materializes a KV cache beyond safe RAM on 16 GB (not a model limit). `load.quant_breadth` (—) is the secondary axis, not exercised in Bucket A.

**Key Metal findings (honest):**
- **logprobs watch item → PASS (no gap).** Under the GPU-resident Metal decode (`mac_logprobs_out/<row>/serve.log`: `[resident-dispatch] metal_enabled=true`, per-step `[resident] gpu_busy` lines), requesting logprobs **forces the full-vocab LogitsStage** (`ResidentTokenOut::Data`) rather than the GPU-sample shortcut (`ResidentTokenOut::Sampled`, token-id only): the response carries real ranked `top_logprobs` (5 alternatives) and the temp=0 greedy invariant holds on Metal (chosen == `top_logprobs[0]`, all ≤ 0, sorted descending) on **both** rows; chat `logprobs.content[]` + completions `logprobs.{tokens,token_logprobs,top_logprobs,text_offset}` both populated; `logprobs+stream` → typed 400, `logprobs+n>1` → fail-closed. (Values not asserted bit-exact — the cross-impl f32 reduction-order frontier.)
- **`context.rope_scaling` Metal caveat (recorded, not papered over).** Both Mac runs mirror the salvaged 1B invocation (`--target-tokens 8400 --verify-mode reference-only`), which tokenizes to **5813 tokens — WITHIN the original 8192 window**, so the **`> 8192` extension regime was not reached on Metal** (it remains as-validated on the Windows rows). The byte-exact match vs llama.cpp Metal (`first_divergent_token_index=-1`, 8 generated tokens) still confirms Camelid applies the GGUF's baked `rope_freqs.weight` (llama3 NTK-by-parts) scaling identically to the oracle across all 5813 positions, plus whole-context KV correctness — hence `done`, with this caveat carried in each receipt.
- **`chat.system_multiturn`** is byte-exact vs llama.cpp Metal on the pinned prompt (`first_divergent_token_index=-1`, answer "Tokyo."); the only template diff is the oracle's live-date system preamble (camelid 53 vs llama 73 tokens) — an intended, non-deterministic cross-engine difference, same posture as Windows.
- **`tools.function_calling`** drives on Metal for both rows (structured `tool_calls`, `finish_reason=tool_calls`, `tool_choice:"none"` suppresses). 3B arguments are semantically correct; 1B is structurally valid but semantically imperfect (a model limitation, not the parser) — identical split to Windows.

Mac artifacts: per-row manifests `qa/capability/macos/capability-manifest.<row>.json`; receipts `qa/capability/macos/receipts/capability-receipt.<cap>.<row>.json` (22); raw evidence `qa/capability/mac_{smoke,tools,json,logprobs,sm,ctx}_out/<row>/`; checksums folded into `qa/capability/SHA256SUMS`.

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
  - DONE (class D, 3 on-disk rows): per-step log_softmax capture in the decode loop (greedy-fast bypassed) -> chat logprobs.content[] + completions logprobs.{tokens,token_logprobs,top_logprobs,text_offset}. Greedy invariant + shapes validated e2e; token IDs bit-exact vs llama.cpp acd79d6 (values within the ~5e-2 f32 envelope). Non-streaming single-choice. Receipt minted.
- **`chat.system_multiturn`** — system role + multi-turn template fidelity (per THIS template)
  - DONE (class D, 3 on-disk rows): a system + 2-user-turn + 1-assistant-turn conversation, greedy. GENERATION PARITY bit-exact vs llama.cpp acd79d6 on all 3 rows (prompt-pinned `verify-receipt`, first_divergent_token_index=-1). TEMPLATE FIDELITY: TinyLlama byte+token-identical to llama.cpp; Llama 3.2 1B/3B differ ONLY by the live-date "Cutting Knowledge Date / Today Date" system preamble llama.cpp injects — a non-deterministic, intended cross-engine difference (src/receipt/verify.rs:662), so parity is asserted on the pinned prompt. Harness qa/capability/system_multiturn_parity.mjs. Receipts minted.
- **`tools.function_calling`** — native tool/function calling — REQUIRES tool-call branch or tool/ipython control tokens in THIS model template
  - DONE (class B, Llama 3.2 1B/3B): input renders tools via the model template; output now parses the Llama 3.x {name,parameters} tool-call into OpenAI tool_calls (parse_tool_calls), finish_reason=tool_calls, content emptied. tool_choice:none suppresses. Battery 6/6 structurally valid. TinyLlama/8B n/a (no tool branch).
- **`structured.json_grammar`** — JSON mode + GBNF grammar-constrained decode (decoder-side, model-agnostic)
  - DONE (class B, 3 on-disk rows): response_format json_object -> JSON-grammar-constrained decoding (src/grammar.rs PDA + per-step logit mask in the decode loop). Battery 12/12 valid JSON. Non-streaming; json_schema/GBNF + force-close-at-max_tokens are follow-ups.
- **`context.full_length`** — full TRAINED context length, exact value from metadata
  - DONE for TinyLlama (class D): validated bit-exact at 1953 tok = 95% of the full trained 2048; KV ~88 MiB, fully reachable; receipt minted. 1B/3B: **HOST-LIMITED, stays `wip`** — validated bit-exact at the feasible frontier (8511 / 8304 tok) but the trained 131072 needs 8.0 GiB / 28 GiB f32 CPU KV, unreachable with safe headroom on this 15.74 GiB box (NOT a model limit). Cap honored from GGUF metadata (model.rs:141, kv_cache.rs:23). Camelid has **no pre-flight KV predict-and-abort** (would OOM mid-gen at kv_cache.rs:135-136) — flagged as a follow-up. Harness qa/capability/context_parity.mjs (projects KV + aborts before an unsafe ctx). See qa/validation-notes/2026-06-25-capability-context-host-limits.md.
- **`context.rope_scaling`** — RoPE scaling/extension (llama3 baked rope_freqs) at positions beyond the original context
  - DONE (class D, Llama 3.2 1B/3B): bit-exact parity vs llama.cpp acd79d6 at positions BEYOND the original 8192 context (1B 8511 tok, full self+reference verify; 3B 8304 tok, reference-only) — the llama3-scaled regime. CORRECTION to the manifest synthesis_correction: scaling is baked into the rope_freqs.weight tensor (no rope.scaling.* metadata key); both engines read it identically — Camelid via rope.rs:499-500 under the `RopeScalingKind::None` arm, NOT the dormant metadata-llama3 branch at rope.rs:507. TinyLlama & 8B: no scaling (n/a). Receipts minted.
- **`observ.usage_timing`** — prompt/completion/total token accounting + optional timing fields
  - DONE (class C, 3 on-disk rows): prompt/completion/total usage present + arithmetically correct e2e. Receipt minted. (ttft timing still not populated — not claimed.)
- **`load.quant_breadth`** — (SECONDARY) loadability across additional quants beyond Q8_0
  - Q8_0 supported; broader K-quant decode (Q4_K/Q5_0/Q6_K) gated/experimental (wire_dequant.rs). SECONDARY axis.
    - **UPDATE (K-quant conductor Phase 1):** **Q4_K_M (mixed Q4_K + Q6_K) is now GPU-resident parity-certified** for `Qwen3-4B-Q4_K_M` — token+text-identical to llama.cpp `acd79d6` at 1/5/50 tokens (`qa/evidence-bundles/qwen3-4b-q4_k_m-windows-cuda-resident-parity-20260628T003317Z-head-0dccbf74/`, `all_pass=true`), running on the in-tree `q4k_gemv`/`q6k_gemv` resident kernels. Exact-row only (this Qwen3-4B GGUF). Still **NOT** done: CPU K-quant decode (none exists — the CPU path errors `data_len=0` on wire-only K-quant tensors; that is Phase 2), the static execution-plan's K-quant disclosure (mislabels the lane `cpu_reference`/`dense_or_other` — follow-up), and Q5_0 / other K-quant files.
    - **Llama-3.2-3B-Q4_K_M (conductor primary) ALSO certified** — GPU-resident decode token+text-identical to llama.cpp `acd79d6` on 5/8 confident probes at 1/5/50 (incl. long-context + code completion to depth 50); 3 open-ended probes diverge at a benign measured near-tie (`qa/evidence-bundles/llama-3.2-3b-q4_k_m-windows-cuda-resident-parity-20260628T004547Z-head-bb3c3528/`). Both Phase-1 rows promoted in `SUPPORT_MATRIX_v0.1.md` + `COMPATIBILITY.md`.
    - **Llama-3.2-3B-Q5_K_M also certified (exact row only)** — GPU-resident decode token+text-identical to pinned llama.cpp `acd79d603` at 1/5/50 on the raw completion harness (`qa/evidence-bundles/llama-3.2-3b-q5_k_m-windows-cuda-resident-parity-20260703T201602Z-head-ab88962/`, `all_pass=true`), running the new `q5k_gemv` lane plus `q6k_gemv` for mixed Q5_K/Q6_K tensors. No broader Q5_K_M family, neighboring-row, long-context, or throughput claim is implied.
    - **Phase 2 — CPU K-quant decode now DEFAULT-ON + certified.** `CAMELID_X86_Q4K_DECODE` flipped from default-off (which CRASHED CPU K-quant decode: wire-only load, `data_len=0`) to default-on (opt out with `=0`); the GPU lane never reaches this CPU chokepoint. Q4_K uses AVX2 (`q4_k_dot_avx2`), Q6_K uses the 8-lane scalar oracle. Certified token+text-identical to llama.cpp CPU on confident probes incl. code to depth 50 (`qa/evidence-bundles/qwen3-4b-q4_k_m-windows-cpu-kquant-decode-parity-20260628T015051Z-head-a86fb46b/`). NOT done: Q6_K AVX2 in the 8-lane order (perf follow-up, bandwidth-bound/likely-null), the serve f32-materialization-guard false-positive on K-quant (bypass needed), and Linux/macOS parity confirmation.

## Notable Phase 0 findings

- **`tools.function_calling` splits by row, exactly as the conductor demands.** TinyLlama → `n/a` (template has only user/system/assistant branches, no tool/ipython tokens). Llama 3.2 1B & 3B → **capable** (their embedded templates carry the Llama-3.1-style tool-call branch + `Environment: ipython`). Original Llama 3 8B → `n/a`/provisional (pre-3.1 template, no tool tokens — must be re-confirmed against the real GGUF, since a mislabeled 3.1 file would flip it).
- **`context.rope_scaling` splits 2 / 2 after a synthesis correction.** TinyLlama (native 2048) and original Llama 3 8B (native 8192) declare no scaling → `n/a`. The Llama 3.2 1B & 3B reach 131072 via **llama3 rope scaling baked into a `rope_freqs.weight` tensor** (no `rope.scaling.*` *metadata* key, but the transform is real). Camelid reads that tensor at `rope.rs:499-500` under the `RopeScalingKind::None` arm — NOT the metadata-llama3 branch at `rope.rs:507`, which is dormant for these GGUFs (an earlier note misnamed this path). Reclassified **capable / class D**, and **now `done`** — validated bit-exact at positions > 8192 (1B 8511 tok, 3B 8304 tok) vs llama.cpp. The over-strict "metadata-key-required" Phase 0 prompt had marked these `n/a`; that was an underclaim (conductor §0/§4.6), corrected here.
- **`context.full_length` differs sharply per row** — 2048 / 131072 / 131072 / 8192 — and must never be cross-claimed. Memory/abort projection (conductor §9) governs the 131072 rows before any near-limit validation.
- **6 capabilities are now `done` on the 3 on-disk rows**, each with a `camelid.capability-receipt/v1`: the **sampling lane** (`sampling.full_set` — `min_p`+`repeat_penalty` added; `sampling.seed_determinism` — degenerate per-seed RNG fixed to a per-step SplitMix64 stream, **a real correctness bug**), **`gen.n_choices`** (n>1 independent reproducibly-seeded choices, converted from a 400 stub), and the contract caps `gen.stream_usage`, `gen.length_stop`, `observ.usage_timing`.
- **`tools.function_calling` splits by row, exactly as the conductor demands.** TinyLlama → `n/a` (no tool branch). Llama 3.2 1B & 3B → **`done`** (class B): Camelid renders tools via the model template on input and parses the Llama 3.x tool-call output back into structured `tool_calls` (battery 6/6 valid; `tool_choice:"none"` suppresses). Original Llama 3 8B → `n/a`/provisional.
- **Every greenfield `open` lane is now `done`.** The last one, `structured.json_grammar` (JSON-mode constrained decode), is `done` on the 3 on-disk rows (class B — battery 12/12 valid JSON via a byte-level JSON PDA + per-step logit mask). `logprobs.top_logprobs` is `done` (class D — token IDs bit-exact vs llama.cpp; values within the f32 envelope). The only non-`done` cells left are `wip` (implemented, receipt pending a row-specific e2e) or the provisional 8B column.
- **`context.full_length` differs sharply per row** — 2048 / 131072 / 131072 / 8192 — and must never be cross-claimed. TinyLlama is now `done` (validated to 1953/2048); 1B & 3B stay `wip` = **host-limited** (the trained 131072 needs 8/28 GiB f32 CPU KV, beyond safe RAM on this 15.74 GiB box — not a model limit; conductor §9 memory/abort projection). `context.rope_scaling` is `n/a` on TinyLlama/8B and now **`done`** on the 1B/3B (tensor-baked llama3 scaling, bit-exact > 8192).

## Recommended sequencing (conductor §6) — progress

1. ✅ **Sampling lane** (`sampling.full_set` + `sampling.seed_determinism`) — `min_p`/`repeat_penalty` added, per-step RNG fixed, class-I invariants + e2e. **DONE.**
2. ✅ **`gen.n_choices` + `logprobs.top_logprobs`** — both **DONE**: n_choices (class C) and logprobs (class D — chat + completions; token IDs bit-exact vs llama.cpp, values within the f32 envelope; non-streaming single-choice).
3. ✅ **Receipts for the already-driving caps** — `gen.stream_usage`, `gen.length_stop`, `observ.usage_timing` minted **DONE**.
4. ✅ **`tools.function_calling`** (1B/3B) — **DONE** (class B): input rendered via the model template, output parsed into structured `tool_calls` (battery 6/6). Gated on the manifest (TinyLlama/8B `n/a`).
5. ✅ **`structured.json_grammar`** — **DONE** (class B): response_format json_object -> JSON-grammar-constrained decode (byte-level PDA + per-step logit mask); battery 12/12 valid JSON. (json_schema/GBNF + force-close-at-max_tokens are follow-ups.)
6. ✅ **`chat.system_multiturn`** (class D, 3 rows) — **DONE**: system + multi-turn parity bit-exact vs llama.cpp (prompt-pinned); TinyLlama byte+token-identical templates, 1B/3B differ only by the live-date preamble. Harness `system_multiturn_parity.mjs`.
7. ✅ **`context.full_length` + `context.rope_scaling`** (class D) — TinyLlama full_length **DONE** (1953/2048); 1B & 3B rope_scaling **DONE** (bit-exact at 8511 / 8304 tok, positions > 8192). 1B/3B `context.full_length` stay **`wip` = host-limited** (trained 131072 KV 8/28 GiB > safe RAM here, not a model limit). Harness `context_parity.mjs`; host-limit note in `qa/validation-notes/`.
8. **Download Llama 3 8B Q8_0** to complete its Phase 0 manifest, re-resolve `tools`/`full_length`/`rope_scaling`, and extend the done receipts to that row.
9. *(secondary)* `load.quant_breadth` — only if widening the load matrix is in scope.

## Artifacts

- Per-row manifests: `qa/capability/capability-manifest.<row>.json` (schema `camelid.capability-manifest/v1`).
- Capability receipts: `qa/capability/receipts/capability-receipt.<cap>.<row>.json` (schema `camelid.capability-receipt/v1`) — **32 minted** (the original 26 + `chat.system_multiturn` ×3 + `context.full_length` ×1 [TinyLlama] + `context.rope_scaling` ×2 [1B/3B]).
- E2E harnesses: `qa/capability/smoke.sh` (sampling/n_choices/stream_usage), `tools_smoke.sh`, `json_smoke.sh`, **`system_multiturn_parity.mjs`** (system+multi-turn parity + template-fidelity diff), **`context_parity.mjs`** (long-context / rope parity with a built-in KV predict-and-abort). All Windows CPU, `CUDA_VISIBLE_DEVICES=-1`.
- Host-limit / safety note: `qa/validation-notes/2026-06-25-capability-context-host-limits.md` (1B/3B `context.full_length` host limits + the missing in-engine KV predict-and-abort).
- Checksums: `qa/capability/SHA256SUMS` (manifests + receipts).
- Provenance: the original 26 receipts were minted on branch `feat/capability-conductor` (base `12f202d0`); the 6 new receipts on `feat/capability-context-chat` (base `bde66bc7`, main) with the lane diff **uncommitted** at receipt time — re-seal against the commit once landed.
