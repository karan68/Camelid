# Hugging Face Model Catch-up

Status: active, Phase 1 started 2026-08-09.

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

1. **LFM2** has the most complete graph evidence. Close exact source provenance,
   streaming/API/WebUI, and a bounded context receipt for
   `LFM2.5-2.6B-Q8_0.gguf`.
2. **Qwen2.5 0.5B** is the cheapest new-family proof. The bootstrap row is the
   official Q8_0 GGUF so graph parity is measured with the least quantization
   ambiguity. Start on the CPU reference lane because Qwen2 projection biases
   are intentionally refused by the resident GPU path.
3. **Phi-3 mini** already has tokenizer/template evidence. Isolate the known
   incremental-decode/KV-cache divergence from fresh-prefill behavior.
4. **Gemma 2** has a runnable graph but needs a Gemma-2-specific chat-template
   fixture and real-row proof; it must not borrow Gemma 3's renderer by name.
5. **SmolLM3** needs the `tokenizer.ggml.pre = smollm` dialect before its NoPE
   schedule can be checked on a real row.
6. **Qwen3 MoE** needs an authoritative artifact anchor and a routing-semantics
   audit before a large download. Synthetic browse fixtures are not candidates.

The work order reflects implementation closeness. The first newly downloaded
bootstrap artifact is Qwen2.5 0.5B because it is much smaller than the LFM2 row;
the existing LFM2 receipts can be audited without reacquiring its weights.

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

Current bootstrap result: source, metadata, tokenizer, and template gates pass.
Raw greedy parity matches 3 of 4 prompts; the remaining prompt reverses two
near-tied candidates, so it is recorded as a hard parity failure rather than
waived. Load/smoke, complete API/WebUI, and context gates remain blocked, and
the row stays experimental/unverified.
