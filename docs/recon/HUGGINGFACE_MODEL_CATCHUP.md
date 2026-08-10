# Hugging Face Model Catch-up

Status: active, Phase 1 started 2026-08-09. The first exact row, LFM2.5 2.6B
Q8_0, completed all eight gates and was promoted on 2026-08-10; Phase 1
continues for the remaining families while the Phase 2 qualification factory is
being built from the same fail-closed gates.

## Goal

Expand Camelid by graph family, not by adding an unverified filename list. A
single qualified graph can unlock many Hugging Face GGUF rows, but every public
support claim remains exact-row and evidence-backed until adjacent sizes,
templates, quants, and context buckets have their own receipts.

The Models page groups downloaded, curated, and live Hugging Face results by
family. Those groups are collapsible so expanding the catalog does not make the
page progressively harder to scan.

## Four phases

| Phase | Scope | Main deliverable | Exit condition |
|---|---|---|---|
| **1. Qualify the graphs already in Camelid** | LFM2, Qwen2/Qwen2.5, Phi-3, Gemma 2, SmolLM3, and Qwen3 MoE | A pinned representative row, a fail-closed gate report, and either an experimental exact-row receipt or a named HOLD for every family | Every row in `qa/model-qualification/phase1-roster.json` has a source disposition and a concrete result for metadata, tokenizer, template, load/smoke, greedy parity, API/WebUI, and the first 512-token context bucket. A HOLD with a reproducible blocker counts as a disposition; a metadata parse does not count as support. |
| **2. Turn qualification into a factory** | Adjacent sizes and quants of qualified graphs; remote GGUF discovery | Automated source pinning, range/header inspection, tokenizer/template fixture capture, two-phase reference comparison, and report generation | A candidate can move from an HF repository/file selector to a scrubbed qualification report without hand-editing support surfaces. Missing artifacts or fixtures report `blocked`, never pass by test skip. |
| **3. Add mainstream missing decoder graphs** | High-leverage text families selected from current HF demand, such as Command R/Cohere, GPT-NeoX/StarCoder, Falcon, and DeepSeek MLA variants | New graph/config/binder implementations behind the same Phase 2 qualification gates | Each selected graph has at least one exact-row CPU reference receipt and typed refusal for unsupported variants. Prioritization is refreshed from live demand before this phase starts. |
| **4. Frontier formats and modalities** | Native SafeTensors ingestion, newer hybrid/recurrent and MoE graphs, multimodal/projector rows, and non-text modalities | Format-neutral model acquisition plus modality-specific parity and UI contracts | Each format/modality has explicit admission, resource limits, parity semantics, API behavior, and UI readiness. No text-only receipt is allowed to imply vision/audio/native-HF support. |

## Phase 1 work order

1. **LFM2** is complete for the exact `LFM2.5-2.6B-Q8_0.gguf` row. Its
   immutable source identity, tokenizer/template, runnable CPU smoke/parity,
   API/SSE/Models-page contract, and exact 512-token receipt all pass. This is
   an exact-row promotion, not family-wide LFM2 support.
2. **Qwen2.5 0.5B** is the cheapest new-family proof. The bootstrap row is the
   official Q8_0 GGUF so graph parity is measured with the least quantization
   ambiguity. Start on the CPU reference lane because Qwen2 projection biases
   are intentionally refused by the resident GPU path.
3. **Phi-3 mini** already has tokenizer/template evidence. Isolate the known
   incremental-decode/KV-cache divergence from fresh-prefill behavior.
4. **Gemma 2** has a runnable graph but needs a Gemma-2-specific chat-template
   fixture and real-row proof; it must not borrow Gemma 3's renderer by name.
5. **SmolLM3** actually declares `tokenizer.ggml.pre = smaug-bpe`, an exact
   llama3-regex alias in the pinned reference. Legacy `smollm` remains a distinct
   two-pass dialect and must not be used as a shortcut for the real row.
6. **Qwen3 MoE** needs an authoritative artifact anchor and a routing-semantics
   audit before a large download. Synthetic browse fixtures are not candidates.

The work order reflects implementation closeness. The full LFM2 artifact has
now been identity-checked and exercised at its pinned SHA-256; its promotion no
longer depends on historical receipts. Qwen2.5 0.5B remains the smaller
bootstrap artifact for the next unfinished family.

## Phase 1 current dispositions

The first overnight pass closed every unsafe ambiguity it reached without
turning preparatory evidence into support. LFM2 subsequently completed its
promotion; the other five families remain active dispositions:

| Family | Current result | Next honest gate |
|---|---|---|
| LFM2 | **Complete / promoted exact-row smoke.** All eight Phase 1 gates pass for `LFM2.5-2.6B-Q8_0.gguf` on the Windows CPU runnable lane. | Apple-Silicon resident-Metal parity, context above 512 tokens, and separately evidenced neighboring sizes/quants or broader sampling/tool boundaries. |
| Qwen2.5 | Source, metadata, tokenizer, template, and the exact 512-token context bucket pass; strict greedy parity remains failed on one stable cross-engine numeric frontier. | Layer trace under one fully pinned oracle launch recipe and complete API/WebUI/SSE smoke; do not waive the token mismatch. |
| Phi-3 | Generic head-dim-96 CPU cache reads are bit-identical; the unproven partial Metal PV tile now falls back. | Repeat the exact-row Metal prefill/decode receipt before changing the HOLD. |
| Gemma 2 | Source identity, seven-case tokenizer parity, and the exact IT template route pass; the bounded header capture awaits regeneration under the clean-head provenance contract. | Clean-head metadata receipt, then full-artifact identity/load and greedy parity. |
| SmolLM3 | Source is pinned and the real `smaug-bpe` construction gap is closed; its bounded header capture awaits clean-head regeneration and its dynamic chat template has an executable HOLD. | Clean-head metadata receipt and exact-row token IDs, then a deterministic renderer for its date/reasoning/system/tool contract. |
| Qwen3 MoE | Official 32.48 GB row is pinned; a bounded 32 MiB prefix confirms all 579 descriptors. | Full-artifact identity/load and MoE parity; remote metadata alone is not a promotion. |

Phase 1 remains active for Qwen2.5, Phi-3, Gemma 2, SmolLM3, and Qwen3 MoE.
LFM2's completed exact row does not promote those families or any adjacent
LFM2 artifact. A `blocked` or `fail` gate is intentional information.

## Qualification gates

Each representative row crosses these gates in order:

1. `source` — repository, immutable revision, exact file, license/access note,
   byte size, and SHA-256.
2. `metadata` — `camelid inspect` agrees on architecture, tokenizer metadata,
   quantization, tensor inventory, and required config.
3. `tokenizer` — Camelid token IDs match an independent pinned oracle pack.
4. `template` — rendered chat bytes and prompt IDs match the model's exact
   template shapes; raw-completion parity cannot substitute for this gate.
5. `load_smoke` — the intended serve lane loads and produces deterministic,
   finite, non-degenerate output. `runnable-smoke` is used only after the
   architecture/tokenizer/quant tuple is oracle-qualified.
6. `parity` — engine-emitted greedy token IDs match a pinned reference in a
   two-phase run where the engines do not coexist.
7. `api_webui` — load/current/models/capabilities/completion/chat/streaming and
   the collapsed Models-page presentation agree on the exact row's status.
8. `context` — start at a preflighted 512-token bucket; larger buckets are
   explicit follow-up evidence.

Promotion remains a separate change to the ledger, API contract, docs, and UI.
The Phase 1 roster is intentionally much thinner than that promotion contract.
LFM2.5 2.6B Q8_0 is the first row to complete both.

### LFM2 exact-row promotion

The committed promotion bundle at
`qa/evidence-bundles/lfm2-2.6b-q8-phase1-promotion-20260810/` records all eight
Phase 1 gates as passing for one immutable artifact:

- repo/revision: `LiquidAI/LFM2.5-2.6B-GGUF` /
  `b421ad1d549afeda6a0fb2ad3a697cb5a7879adc`
- file: `LFM2.5-2.6B-Q8_0.gguf`, `2874779456` bytes, SHA-256
  `36587fdf27bdfc69caf2637273679a0870ec155162161bde6fd16e8c70bdb757`
- result: `source`, `metadata`, `tokenizer`, `template`, `load_smoke`, `parity`,
  `api_webui`, and `context` all pass on the Windows x86_64 CPU runnable lane;
  the short gate matches all 96 generated IDs and the native chat receipt
  matches all eight generated IDs/text at exactly 512 rendered prompt tokens
- support decision: `supported_exact_row_smoke`; non-streaming and streaming
  chat plus the Models-page identity/readiness contract pass, while legacy raw
  completions and tools remain intentionally typed fail-closed

Next evidence is the bundle's Apple-Silicon Metal handoff, then bounded context
above 512 tokens and separately qualified broader sizes, quants, sampling, and
tool behavior. None of those follow-ups is implied by the Windows CPU promotion.

## Phase 1 bootstrap receipt

The first artifact is pinned to official repository revision
`9217f5db79a29953eb74d5343926648285ec7e67`:

- repo/file: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` /
  `qwen2.5-0.5b-instruct-q8_0.gguf`
- size: `675710816` bytes
- SHA-256: `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`
- license: Apache-2.0
- independent reference: llama.cpp b9632 (`acd79d603`), Windows x64 CPU

The committed oracle fixture records raw prompt IDs, 16-token greedy
continuations, and the exact one-turn ChatML rendering. Passing that fixture is
still only raw graph/tokenizer evidence; chat-template/API/context gates remain
separate.

Current bootstrap result: source, metadata, tokenizer, template, and the exact
512-token context gate pass. The context receipt used a clean current-head
binary, replayed Camelid deterministically, and matched all eight generated
tokens and text against pinned llama.cpp. This does not waive the independent
short-prompt result below.
Raw greedy parity matches 3 of 4 prompts; the remaining prompt reverses two
near-tied candidates, so it is recorded as a hard parity failure rather than
waived. Load/smoke and complete API/WebUI gates remain blocked, and the row
stays experimental/unverified.

## Phase 2 factory foundation

Phase 2 has started without weakening the Phase 1 evidence boundary:

- `scripts/hf-qualification-source.mjs` resolves a Hugging Face repo/file
  selector to an immutable revision, LFS byte size, SHA-256, license, and access
  state using the Hub's file-metadata response. It can independently hash an
  existing local artifact against that lock.
- `scripts/model-qualification-factory.mjs` walks the roster in priority order,
  resolves filenames under `CAMELID_MODELS_DIR`, and writes one scrubbed report
  per row plus an index. Missing directories, artifacts, fixtures, binaries, or
  source identities become `blocked`; they never disappear as skipped tests.
  `--resolve-source` performs the immutable Hub preflight, while
  `--inspect-header` additionally runs the bounded remote-header lane and keeps
  per-row failures from aborting the rest of the batch.
- `scripts/check-model-qualification-report.mjs` validates the committed report
  contract, fail-closed overall status, and privacy boundary before evidence is
  accepted.
- `scripts/hf-qualification-header.mjs` range-fetches at most 64 MiB and invokes
  `camelid inspect-prefix` with the pinned full byte length. The command validates
  metadata and every tensor descriptor against the real artifact bounds while
  redacting the temporary prefix path. Receipts pin the inspector version and
  binary SHA-256, immutable source identity, exact `Content-Range`, prefix hash,
  and compact tensor-inventory hash. It aborts if a server ignores `Range`, so
  inspecting a 30+ GiB candidate cannot silently become a full download.
- `scripts/hf-qualification-tokenizer.mjs` provides the first bounded
  exact-row tokenizer lane for Gemma 2. It compares Camelid against pinned
  llama.cpp using unchanged tokenizer metadata from the immutable prefix, while
  explicitly declining any weight-load or generation claim.

The factory does not download multi-gigabyte artifacts implicitly and does not
run smoke or generation unless those probes are explicitly requested. This
keeps routine inventory runs cheap while making evidence-producing runs
deliberate and reproducible.

## Phase 3 first decoder slice

Aya Expanse 8B Q4_K_M replaces the earlier 37.2 GB Command R planning row as a
manageable representative. Its public 5.06 GB artifact and license are pinned,
and a 16 MiB header prefix establishes the exact 32-layer Command-R shape. The
runnable implementation now models ordinary LayerNorm, adjacent/interleaved
RoPE, tied output, and the parallel attention/FFN residual from one normalized
input. Admission and loading are pinned to that exact metadata/tensor mix;
64-layer Q/K-normalized Command-R rows and `cohere2` remain typed refusals.

This is attemptability, not support. The full artifact was not downloaded,
smoke remains outside the oracle-qualified set, and Command-R chat returns a
typed 422 until the Aya template and prompt-token fixtures exist. The next
status-changing evidence is a full-file hash plus tokenizer/template and greedy
parity against the pinned reference.

## Phase 4 format boundary started

`camelid inspect-source` exposes the existing format-neutral source manifest for
either a GGUF file or a local Hugging Face directory containing `config.json`
and SafeTensors shards. For the SafeTensors lane it validates the supported
dense-LLaMA config subset, shard index, tensor headers, role names, shapes, and
currently decoded dtypes. The isolated `tokenizer.json` adapter uses the pinned
Hugging Face `tokenizers` crate with the pure-Rust regex backend, validates the
serialized pipeline and vocabulary, and records three no-special-token
encode/decode probes with typed blockers. This is readiness inspection only:
special-token and chat-template parity, weight orientation, one-token execution,
and exact-row receipts remain required, and native SafeTensors generation is
intentionally disabled.
