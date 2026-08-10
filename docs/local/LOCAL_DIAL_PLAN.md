# LOCAL DIAL — End-to-End Build Plan (SOURCE OF TRUTH)

> **Status:** Phase 0 not started. Nothing implemented yet.
> **Checkout:** `C:\camelid-dial` — git worktree, branch `feat/local-dial`, base `342b0f058` (tag **v0.6.1**, == `upstream/main` == `origin/main` on 2026-08-08).
> **This file is LOCAL ONLY.** It lives under `docs/local/`, which is in `.git/info/exclude` (shared git dir `C:/camelid-fork/.git`). It must NEVER be committed, and **no shipped code may cite it** — a module that references a doc we never publish becomes a dangling reference the moment the doc moves. Contracts go in module doc-comments; this file is scaffolding.

## How to use this document

1. Phases are executed **in order**. A phase is DONE only when its **Exit Gate** passes verbatim.
2. Every phase records its result in §10 Changelog. Update it as you go — this file is the memory.
3. **Do not implement ahead of the current phase.** Each phase is designed to be independently revertable and to leave `main` behaviour byte-identical until the phase that deliberately changes it (Phase 6).
4. Claims in §1 were verified by reading the tree at `342b0f058`. If a claim turns out to be wrong, **fix this document first**, then proceed. Never "fix" code against a false premise from an older note.

---

## 1. VERIFIED GROUND TRUTH (read at `342b0f058`, 2026-08-08)

Everything in this section was confirmed by reading the actual tree. Line numbers are from `C:\camelid-dial`.

### 1.1 Fit advisor — `src/fit.rs`
| Fact | Location |
| --- | --- |
| `pub enum FitVerdict` = `FitsResident, FitsWithOffload, CpuOnlyOk, InsufficientFreeMemory, WontFit, Unknown` | fit.rs:45 |
| `pub struct FitInputs { weight_bytes, kv_bytes_at_ctx }` | fit.rs:135 |
| `pub fn assess(hw: &HardwareProfile, m: &FitInputs) -> FitVerdict` | fit.rs:276 |
| `pub fn assess_gpu_resident(hw, m) -> FitVerdict` | fit.rs:285 |
| `pub fn advisory_footprint(weight_bytes) -> FitInputs` (flat 25 % pad) | fit.rs:323 |
| `pub fn exact_footprint(...)`, `pub fn kv_bytes(dims, ctx, kv)` | fit.rs:482, 469 |
| `ADVISORY_OVERHEAD_PERCENT=25`, `ADVISORY_CONTEXT_TOKENS=4096`, `ACTIVATION_SCRATCH_BYTES=512 MiB` | fit.rs:317, 447, 464 |
| **NO function assesses two models simultaneously.** | — (confirmed absent) |

### 1.2 Hardware probe — `src/capability.rs`
- `HardwareProfile` carries `cuda_vram_total_bytes` / `cuda_vram_free_bytes`, `host_ram_total_bytes` / `host_ram_free_bytes`, `cpu_logical_cores`, `simd`, Metal + CUDA flags.
- `detect()` = full live probe (opens a CUDA context). `cached()` = `OnceLock` snapshot.
- **macOS RAM now works** — it delegates to `crate::gait::host_ram_status()`. The old "(0,0) on macOS" note in my memory is **STALE**. Unknown-RAM handling is still required for genuinely unsupported platforms.

### 1.3 Multi-model residency — `src/api/mod.rs`
- `loaded_models: Arc<RwLock<HashMap<String, LoadedModel>>>` — **N models may be resident**.
- `active_model_id: Arc<RwLock<Option<String>>>` — **exactly one is "active"** at a time.
- `cached_weights: Arc<RwLock<HashMap<String, Arc<LlamaLoadedWeights>>>>` (mod.rs:147), `model_last_used` (mod.rs:149).
- `model_transition: Arc<tokio::sync::Mutex<()>>` serialises load/unload.
- **`load_weights_lru()` (mod.rs:12297) is the critical function.** It sums the materialised CPU weight bytes of every *other* cached model and, if `current_sum + estimated > limit`, **evicts the LRU model** (mod.rs:12367) in a loop until it fits.
- **THE BUDGET:** `DEFAULT_CPU_WEIGHT_MATERIALIZATION_LIMIT_BYTES = 6 GiB` (mod.rs:82), overridable by `CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES` (mod.rs:83), read at `cpu_weight_materialization_limit_bytes()` (mod.rs:14107).
- `POST /api/models/unload` → `unload_model` (mod.rs:2415 / 11868).

> **PHASE 0 CORRECTION — this budget is NOT the binding constraint by default.** See §1.12: the estimator counts quantized 2-D/3-D linears as **zero**, so a typical Q8_0 / K-quant model contributes only its small non-linear tensors. The eviction path is real but is only reachable when a user disables the lazy-Q8 default. The real constraint is host RAM (§1.13). R1 has been re-ranked accordingly in §4.

### 1.12 CPU-weight materialisation estimator — exact semantics (verified)
`estimate_cpu_weight_materialization_bytes` (mod.rs:14191) computes, per tensor:

| Tensor shape/type | Counted bytes |
| --- | --- |
| Q8_0, rank 2 or 3, **when `lazy_q8_linear`** | **0** (file-backed) |
| Q4_K / Q5_K / Q6_K / Q2_K / Q3_K, rank 2 or 3 | **0** (wire-resident) |
| Q4_0 / Q4_1, rank 3 (MoE expert packs) | **0** (streamed) |
| Q1_0 / Q2_0* / Pq2_0, rank 2 (Prism) | `desc.n_bytes` |
| **anything else** (incl. all 1-D norms, F16/F32 rank-2) | `element_count * 4` |

**Defaults, both read and confirmed:**
- `lazy_q8_linear_materialization_enabled()` (mod.rs:14129) — **defaults TRUE** (`NotPresent => true`); only `0/false/off/disabled` disables it.
- `cpu_weight_materialization_retains_q8_blocks()` (mod.rs:14122) — **defaults FALSE** (only `1/true/yes` enable).

⇒ Under stock env, every quantized linear in our models counts **0**. The 6 GiB budget is effectively non-binding here. It becomes binding only under `CAMELID_LAZY_Q8_0_LINEAR=0` or `CAMELID_RETAIN_Q8_0_BLOCKS=1`.

### 1.13 `assess()` decision rule — the actual gate (verified, and it surprised me)
`assess_with_headroom` (fit.rs:224):
1. `has_usable_gpu` = `cuda_available && cuda_vram_free_bytes > 0` (fit.rs:172).
2. GPU present → `cuda_vram::evaluate(free, footprint, headroom)`; `DEFAULT_MIN_HEADROOM_MIB = 512` (cuda_vram.rs:31).
   - **Ok → then a second gate:** `match usable_ram { Some(ram) if footprint > ram => FitVerdict::Unknown, _ => FitsResident }`.
     A VRAM-resident load still **stages through host RAM**, so a starved host yields `Unknown`, *not* `FitsResident`.
   - Err → offload / `negative_verdict` split into `InsufficientFreeMemory` vs `WontFit`.
3. No GPU → `CpuOnlyOk` / negative / `Unknown`.

- `usable_host_ram_bytes` (fit.rs:159) = **80 % of FREE RAM, with NO total floor** — deliberate (fit.rs:31-38: flooring "overcommits a starved host").
- `FitInputs::footprint_bytes()` = `weight_bytes + kv_bytes_at_ctx` (fit.rs:143).
- `advisory_footprint(w)` = `{ weight_bytes: w, kv_bytes_at_ctx: w*25/100 }` ⇒ footprint = **w × 1.25** (fit.rs:323).
- fit.rs:56-63 explicitly warns that collapsing `InsufficientFreeMemory` into `WontFit` is "a factual error with a real cost… reads as 'this product does not work here'". **The Dial must honour that same distinction.**

> **Known repo docs defect (found in passing, not ours to fix here):** the `usable_host_ram_bytes` doc-comment's first line still claims "the same `max(80% of available, 25% of total)` formula", which the code below it deliberately does **not** implement. Candidate for a separate one-line docs PR.

### 1.4 Generation entry points — `src/api/mod.rs`
- Routes: `/v1/chat/completions` → `chat_completions` (mod.rs:13349), `/v1/completions`, `/v1/models`.
- `chat_completions_multi_choice` (mod.rs:13265) already implements **n>1**: `MAX_N_CHOICES = 8` (mod.rs:15444); `prepare_multi_choice` derives a per-choice seed from a `base_seed` (mod.rs:12901-12919), so **n>1 is already reproducible and deterministic**.
- **n>1 rejects streaming** (mod.rs:13076), rejects `camelid_receipt` (13084), rejects logprobs (13122).
- `prepare_generation` (mod.rs:14596) → `PreparedGeneration` (mod.rs:1992); `run_decode_job_serialized` (mod.rs:15754).
- `CAMELID_GENERATION_TIMEOUT_MS`, default **15 min** (mod.rs:87, 104); typed `503 generation_timeout` (mod.rs:15900-15936).
- **Precedent for a non-OpenAI request field: `camelid_receipt` already exists** on the chat request. Our dial field follows it exactly.
- Engine: `engine::EngineHandle`; `DEFAULT_CONTINUOUS_BATCH_SLOTS = 2` (runtime_config.rs) — one compute thread, one token step at a time.

### 1.5 Workspace + agent loop
- `src/chat/workspace_bridge.rs` (**not** `src/api/`): `WorkspaceEvent` enum at :32, serde-tagged (`session.started`, `turn.started`, `memory.updated`, `memory.compacted`, `model.delta`, `model.timing`, `model.answer`, `tool.call`, `approval.required`, `tool.result`, …). **Adding a variant is the established extension pattern.**
- `WORKSPACE_MODEL_STEP_TIMEOUT = 90 s` (workspace_bridge.rs:28); `DEFAULT_APPROVAL_TIMEOUT = 5 min` (:26); `driver.set_stream_control(cancel, WORKSPACE_MODEL_STEP_TIMEOUT)` (:461).
- `src/api/workspace.rs`: `EVENT_CLAIM_TIMEOUT = 30 s` (:34); `blocks_model_transition()` (:124, :153); `cancel_session` (:1693).
- Agent loop: `src/chat/agent.rs` — `run_loop` (:316), `LoopEnd` (:164), `RunConfig.max_steps` (:30), `tool_profile: ToolProfile` (:50), `trait Approver` (:134).

### 1.6 Catalog
- `curated_catalog()` at mod.rs:26341. `CatalogItem` at :26028 with **`task_tags: &'static [&'static str]`** at :26044.
- Tags actually in use: `general`, `tools`, `reasoning`, `coding`, `embeddings`, `retrieval`.
- `CatalogItemView` (:26115) adds `oracle_qualified` (:26126), `task_tags` (:26138), `fit_confidence` (:26142).

### 1.7 Runtime config — `src/runtime_config.rs`
- Env vars: `CAMELID_QUEUE_DEPTH`, `CAMELID_NGRAM_INDEX_MAX_ENTRIES`, `CAMELID_KV_POOL_BUDGET_BYTES`, `CAMELID_CONTINUOUS_BATCH_SLOTS`, `CAMELID_X86_KQUANT_MATMUL_OWNER`, `CAMELID_MOE_EXPERT_STORAGE`, `CAMELID_MIXTRAL_LONG_GENERATION`.
- Helpers: **`env_flag_default_off`** (:143), **`env_flag_default_on`** (:158), `bounded_usize`, `bounded_u64`. Invalid values **fail closed to the documented default**.

### 1.8 Frontend
- Views: `ChatWorkspace.jsx`, `WorkspaceView.jsx`, `SettingsView.jsx`, `ModelsView.jsx`, + 10 more.
- `frontend/src/lib/`: `modelActivation.js` (exports `modelFilenameFromPath`, `loadLocalModelForChat`, `unloadLocalModel`, `warmGenerationPath`), `chatGate.js` (`getChatGateState`), `chatCompletionStream.js` (`extractSseEvents`, `readStreamingChatCompletion`, …), `capabilities.js`, `workspaceAgent.js`, `catalogActivation.js`, `firstRunActivation.js`.
- localStorage keys already in use (our new key must match this convention): `camelid.selectedModelId`, `camelid.systemPrompt`, `camelid.maxTokens`, `camelid.samplingParams.*`, `camelid.activeTab`, `camelid.conversations`, `camelid.workspaceSetupPercent`, `camelid.observatory.renderer`, `camelid.clusterTopology`, …
- **No existing user-facing tier/mode/effort/preset selector exists.** The Dial is a NEW surface, not an extension.

### 1.9 Capability contract & ledger
- `GET /api/capabilities` → `capabilities` (mod.rs:4904) → `capabilities_response()` (mod.rs:4937).
- CI enforces: `scripts/check-ledger-schema.mjs` (validates `status` / `full_support_status` literals in `src/api/mod.rs` against `ledger/camelid-ledger.schema.json` enums — **a struct-init literal with a non-vocabulary value fails CI**), `scripts/check-ledger-drift.mjs`, `scripts/check-fit-verdict-vocabulary.mjs` (**WebUI fit vocabulary must equal `src/fit.rs`** — directly relevant to us), `scripts/check-public-scrub.sh`, `scripts/check-public-evidence-claims.mjs`.

### 1.10 CI gates (`.github/workflows/ci.yml`)
| Job | Commands |
| --- | --- |
| `public-scrub` | `check-public-scrub.sh`, `audit-evidence-bundle-privacy.mjs --strict`, `check-evidence-bundle-checksums.sh`, `check-public-evidence-claims.mjs`, `check-ledger-schema.mjs`, `check-ledger-drift.mjs`, `check-fit-verdict-vocabulary.mjs`, `check-cuda-prefill-parity-gate.mjs`, `test-readme-screenshot.mjs` |
| `rust` (linux/macos/windows) | `cargo fmt --all -- --check` → `cargo clippy --all-targets <features> -- -D warnings` → `cargo test --all-targets <features>` → release bit-exactness suites → `cargo doc --no-deps` |
| `desktop` | `cargo clippy -p camelid-desktop --all-targets -- -D warnings`, `cargo test -p camelid-desktop --all-targets` |
| `validation-scripts` | glob `scripts/test-*.mjs` |
| `frontend` | `npm ci` → `npm run build` → 14 smokes (`smoke:model-state`, `smoke:model-lanes`, `smoke:catalog-activation`, `smoke:first-run`, `smoke:first-run-card`, `smoke:offline-banner`, `smoke:catalog-browse`, `smoke:catalog-companion`, `smoke:model-deletion`, `smoke:3b-closure`, `smoke:integration`, `smoke:workspace`, `smoke:workspace-visual`, `smoke:ui`) |
| `ci-gate` | requires no failed job |

> **fmt runs FIRST in the rust job** — a formatting slip fails all three OS legs in ~40 s.

### 1.11 A precedent worth obeying
`src/quality/mod.rs:14-20` documents that a `routing` submodule was once *declared* (`pub mod routing;`) but never written, and the doc claimed a routing policy that did not exist. Both were removed with the note: *"if a routing policy is wanted, it should be added with its own evidence rather than re-declared here."*
**Rule for us: never declare, document, or advertise a capability before the code and its evidence exist.**

### 1.14 The authoritative load guard — and the invariant it hands us (verified)
`fit_preload_guard` (mod.rs:7224) → `fit_preload_message` (mod.rs:7091) refuses **iff `verdict.refuses_load()`**:

```rust
// src/fit.rs:104
pub fn refuses_load(self) -> bool {
    match self {
        FitVerdict::WontFit | FitVerdict::InsufficientFreeMemory => true,
        FitVerdict::FitsResident | FitVerdict::FitsWithOffload
        | FitVerdict::CpuOnlyOk | FitVerdict::Unknown => false,
    }
}
```

**`Unknown` does NOT refuse a load** — it is explicitly "advisory-blind: never treated as a failure" (fit.rs:70). Three distinct refusal codes exist, all 422:

| Code | When |
| --- | --- |
| `model_requires_unload` | another of our models is resident and releasing it would free enough — checked first, most actionable |
| `host_memory_unavailable` | `InsufficientFreeMemory` — "close some applications and retry" |
| `model_too_large_for_host` | `WontFit` |

Footprint basis: `exact_preload_footprint(size, dims, hw, metal_resident, metal_kv_dtype)` (mod.rs:7221 / 7256), falling back to `advisory_footprint(size)` (mod.rs:7264) only when dims are unreadable. Override: `CAMELID_SKIP_FIT_CHECK=1`.

> **THE GOVERNING INVARIANT FOR THIS FEATURE:**
> **A Dial tier is non-`Ready` if and only if `FitVerdict::refuses_load()` is true for its resolved model.**
> Anything stricter tells the user to free memory for a load that would have worked; anything looser offers a tier that 422s on click. This is a single, mechanically testable property — Phase 1 and Phase 3 both assert it.

> **Tier switching is already solved — do not invent a mechanism.** `model_requires_unload` names the remedy itself: `POST /api/models/load` with `"replace": true`, "the app's Load button does this", which swaps in one step. Phase 6 uses exactly that.

---

## 2. PRODUCT DEFINITION

### 2.1 One sentence
**The Dial lets a user trade time for care using only the models already on their machine, and tells the truth about what it can actually do on that machine.**

### 2.2 Tiers
| Tier | Model selection | Extra passes | Extra RAM |
| --- | --- | --- | --- |
| `low` | smallest fit-qualified installed model | none | none |
| `medium` **(default)** | best mid-size model that `FitsResident` | none | none |
| `high` | same model as `medium` | **+1 self-critique pass, same weights** | none |
| `ultra` | largest model that fits | +1 review pass — by a **second model** only if dual-residency is *proven*, else falls back to `high` behaviour on the larger model | ≤ 0 (fallback) or second model (proven) |

### 2.3 Non-goals (explicit, do not drift)
- **Not** a new model architecture. MoE already exists in this engine and is unrelated.
- **Not** automatic per-message tier switching. Silent model swaps are the #1 way to make this feel "slow as hell" (a swap costs seconds and gigabytes; see §1.3).
- **Not** a cloud router. No network model calls, ever.
- **Not** a quality *claim*. Until Phase 8 produces measurements, the feature ships describing what it *does*, never asserting it makes answers better.
- **Not** a replacement for the existing model picker. The Dial sits above it; manual model choice remains.

---

## 3. ARCHITECTURE DECISIONS

| # | Decision | Rationale |
| --- | --- | --- |
| **AD-1** | New module `src/dial.rs`, registered in `lib.rs` after `fit`. Name `dial` — `quality` is taken (§1.11). | Pure, testable, no engine coupling. |
| **AD-2** | Tier resolution is a **pure function** of (installed models, `HardwareProfile`, budget, surface). No I/O, no async, no globals. | Makes the entire decision layer unit-testable with zero model files — the only way to test the edge cases in §5 honestly. |
| **AD-3** | Request field `camelid_dial: Option<String>` on the chat request, mirroring the existing `camelid_receipt`. Absent ⇒ **exactly today's behaviour**. | Keeps OpenAI compatibility and gives us a byte-identical ablation control. |
| **AD-4** | The critique pass **reuses the existing generation path** (`prepare_generation` → `run_decode_job_serialized`). It is a second ordinary generation, not a new decode mode. | Zero risk to the decode loop, KV cache, or sampling contract. |
| **AD-5** | Dual-residency requires a new **`fit::assess_pair()`** that models the *real* constraint from §1.3 (the 6 GiB CPU-weight materialisation budget), not just total RAM. | Assessing against RAM alone would green-light a pair that `load_weights_lru` will immediately thrash. |
| **AD-6** | The dual-model oracle is **default-OFF** behind `env_flag_default_off` **and** a Settings toggle, and the fit check is a **hard veto** even when both are on. | Matches repo convention for resource-costly features (`CAMELID_MOE_EXPERT_STORAGE`, etc.) and fails closed on tight machines. |
| **AD-7** | Streaming: the draft streams as today; a critique pass emits a **new, additive phase** rather than retroactively rewriting streamed text. In Workspace this is a new `WorkspaceEvent` variant (§1.5 pattern). | Silently replacing text a user already read is a worse experience than showing the revision. |
| **AD-8** | Tier switching is an **explicit, visible** action with a real loading state, single-flighted, and refused while a session `blocks_model_transition()`. | Reuses the existing, already-correct exclusion machinery instead of inventing a second one. |
| **AD-9** | Critique passes are **non-recursive by construction**: the critique request is built with the dial field cleared. | Makes infinite regress impossible rather than bounded. |
| **AD-10** | Kill switch `CAMELID_DIAL=0` disables the whole feature; the resolver then reports every tier unavailable and all request handling takes the untouched path. | Single revert lever for a shipped release. |
| **AD-11** | **Never silently substitute a smaller model for a tier whose intended model is `NeedsFreeMemory`.** The tier reports the model it wants, the shortfall, and an explicit user action; substitution only ever happens on an explicit click. *(Added by Phase 0 / P5.)* | A user on `ultra` being quietly served a 1B is exactly the kind of unverifiable claim this repo refuses everywhere else. Honest refusal beats a silent downgrade. |
| **AD-12** | **The review pass is BOUNDED: it emits a verdict plus a patch, never a full regeneration.** *(Added by Phase 0 / Result 5.)* | Measured CPU decode is 3–8 tok/s, so a 64-token answer costs 8–28 s. A review that rewrites the whole answer doubles that to 16–56 s and makes `high` unusable. Bounding the review is what keeps the upper tiers shippable. |

---

## 4. RISK REGISTER (derived from §1 — these are facts, not guesses)

| ID | Risk | Evidence | Mitigation phase |
| --- | --- | --- | --- |
| **R0** | **(TOP RISK, found in Phase 0)** Availability was being **re-derived** from `FitVerdict` instead of bound to the engine's own predicate. With 1.25 GiB free, 6 of 8 models return `Unknown` — which the loader **accepts**. Mapping `Unknown ⇒ Unavailable` renders `medium`/`high`/`ultra` broken on a machine that runs them; mapping it ⇒ `NeedsFreeMemory` nags for memory the load did not need. Both are wrong. | §1.13, §1.14, `refuses_load` fit.rs:104, measured §10.2/§10.4 | Phase 1 — bind to `FitVerdict::refuses_load()` (§1.14 invariant) |
| **R1** | Loading an oracle evicts the primary → reload ping-pong every turn. **Re-ranked in Phase 0: non-binding under stock env** (§1.12) — only reachable when the lazy-Q8 default is disabled. Keep the guard, drop the alarm. | `load_weights_lru` mod.rs:12297-12374; estimator mod.rs:14191 | Phase 2 (`assess_pair`), Phase 5 |
| **R2** | Two-pass turn blows the 90 s Workspace step timeout. | workspace_bridge.rs:28 | Phase 4 |
| **R3** | Two-pass turn exceeds the context budget (`prompt + draft + critique instructions`). | Workspace budget events, `MemoryUpdated` fields | Phase 4 |
| **R4** | n>1 already rejects streaming; dial + n>1 interaction undefined. | mod.rs:13076 | Phase 3 (reject explicitly) |
| **R5** | A tier can be offerable in Chat but not Workspace. **Measured in Phase 0: only 5 rows in the whole contract are `tool_capable`, and exactly 2 are on this disk** (Qwen3-4B-Q4_K_M, Llama-3.2-3B-Q8_0) — so Chat has 6 candidates and Workspace has 2. The asymmetry is large, not theoretical. | mod.rs:737, rows 5155/5245/5436/6477/6528; §10.4 | Phase 1 (surface-aware resolution) |
| **R9** | Tier availability flickers as free RAM fluctuates (no total-RAM floor, fit.rs:31-38 by design), so a tier can change state when a browser tab closes. | §1.13 | Phase 6 (present transient states as actionable, never as capability) |
| **R6** | Fit vocabulary drift breaks CI. | `check-fit-verdict-vocabulary.mjs` in `public-scrub` | Phase 2 |
| **R7** | Ledger literal scanner fails CI on a stray `status: "…"`. | `check-ledger-schema.mjs` | Phase 8 |
| **R8** | Critique makes the answer *worse* and we would not know. | — | Phase 7 (measure before claiming) |

---

## 5. PHASES

Each phase: **Goal → Files → Contract → Tests → Exit Gate → Rollback.**
Every phase must leave `cargo fmt`, `clippy -D warnings`, and the full test suite green.

---

### PHASE 0 — Ground truth lock & pre-registration
**Goal:** Freeze the facts and *pre-register predictions* before writing code, so we cannot rationalise a bad result later.

**Files:** this document only.

**Work**
1. Re-verify every §1 claim against the tree (they were verified 2026-08-08; re-check if the base moves).
2. Inventory the models actually on this machine (`C:\models`, `C:\camelid-dial\models`, `C:\camelid-fork\models`): filename, bytes, sha256, arch, quant. Record in §10.
3. Compute, by hand, what each tier *should* resolve to on this machine, and whether `ultra` dual-residency is even possible under the 6 GiB budget. **Write the prediction down before Phase 1.**
4. Record baseline timings for a fixed prompt on each installed model (single pass) — the denominator for every later cost claim.

**Tests:** none (no code).

**Exit Gate**
- [ ] §1 re-verified; corrections applied.
- [ ] Model inventory + hand-computed tier predictions recorded in §10.
- [ ] Baseline single-pass timings recorded.
- [ ] **Prediction registered:** does this machine support a real dual-model `ultra`? Yes/No, with the arithmetic.

**Rollback:** n/a.

---

### PHASE 1 — Pure tier resolution core (`src/dial.rs`)
**Goal:** Decide which model each tier maps to, and why, as a pure function. **No wiring. Zero behaviour change.**

**Files:** `src/dial.rs` (new), `src/lib.rs` (one `pub mod dial;` line).

**Contract**
```
DialTier      = Low | Medium | High | Ultra
DialSurface   = Chat | Workspace
CandidateModel { id, filename, weight_bytes, fit: FitVerdict, tool_capable, supported_row, task_tags }
TierPlan      { tier, primary_model_id, review: ReviewMode, availability, reason }
ReviewMode    = None | SelfCritique | SecondModel(id)
Availability  = Ready { capacity_verified: bool } | NeedsFreeMemory { shortfall_bytes } | Unavailable(reason)

resolve_tier(tier, surface, &[CandidateModel], &HardwareProfile, budget_bytes) -> TierPlan
resolve_all(surface, ...) -> [TierPlan; 4]
```
Rules:
- Ordering is **total and deterministic**: sort by `weight_bytes`, tie-break by `id` (stable across restarts).
- **Availability is tri-state, and is defined by the authoritative guard (§1.14), not re-derived:**
  | `FitVerdict` | `refuses_load()` | → `Availability` |
  | --- | --- | --- |
  | `FitsResident`, `FitsWithOffload`, `CpuOnlyOk` | false | `Ready { capacity_verified: true }` |
  | `Unknown` | **false** | `Ready { capacity_verified: false }` — offerable, labelled "capacity could not be confirmed". **Not** a refusal: the loader would accept it. |
  | `InsufficientFreeMemory` | true | `NeedsFreeMemory { shortfall_bytes }` — transient and actionable, never "unsupported" |
  | `WontFit` | true | `Unavailable("too large for this machine")` |
- **Invariant (asserted by test):** `!matches!(availability, Ready { .. }) == verdict.refuses_load()`.
- `Workspace` surface additionally requires `tool_capable`.
- Every `TierPlan` carries a **human-readable `reason`** — the UI never invents an explanation.
- The footprint basis must match the load guard's (`exact_preload_footprint`, §1.14), not the coarser catalog pad, so the Dial can never offer a tier the loader refuses.

**Tests (all pure, no model files, no network)**

*Happy path*
1. Four distinct models → four distinct sensible plans.
2. `medium` prefers the largest `FitsResident` over a larger `FitsWithOffload`.

*Edge cases (each is a named test)*
3. **Zero models** → all four tiers `Unavailable`, distinct reason, no panic.
4. **Exactly one model** → all four tiers resolve to it; `high`/`ultra` still get `SelfCritique`; reason explicitly says all tiers share one model.
5. **Two models of identical bytes** → deterministic tie-break; resolving twice yields identical output.
6. **All models `WontFit`** → all tiers `Unavailable`, reason points at capacity.
7. **`FitVerdict::Unknown`** → `Ready { capacity_verified: false }` — **not** a refusal (the R0 regression test; the loader accepts `Unknown`, so the Dial must too).
7b. **`InsufficientFreeMemory`** → `NeedsFreeMemory` with a non-zero `shortfall_bytes`.
7c. **Invariant test (property-style, all six verdicts × a size grid):** `!matches!(availability, Ready{..}) == verdict.refuses_load()`. This is the §1.14 governing invariant and is the single most important test in Phase 1.
7d. **Real Phase-0 host vector**: the exact measured profile from §10.2 (15.71 GiB total / 1.25 GiB free / 7.77 GiB free VRAM) with the eight real model sizes → reproduces **P5**; the same function with the idle profile reproduces **P6**. Differing only in `HardwareProfile` proves availability is a pure function of the host, not of hidden state.
8. **Workspace + no tool-capable model** → all tiers `Unavailable` on Workspace **while Chat stays Available** (asymmetry pinned by test — R5).
9. **Unsupported/non-oracle-qualified model** → selectable but flagged in `reason`.
10. **Ordering stability**: shuffle the input vector 100× → identical plans every time.
11. `budget_bytes = 0` → no `SecondModel` is ever proposed.
12. Duplicate model ids in input → rejected/deduped deterministically, no panic.

*Negative control*
13. **Ablation:** deleting the `tool_capable` check makes test 8 fail. (Prove the test is load-bearing.)

**Exit Gate**
- [x] ≥13 unit tests green (**19**); ablation 13 verified to fail when the rule is removed.
- [x] `cargo fmt --all -- --check` clean.
- [x] `cargo clippy --all-targets -- -D warnings` clean.
- [x] `cargo test --lib` — 1697 passed / 0 failed / 84 ignored.
- [x] `git diff --stat` = exactly 2 files.

**PHASE 1 COMPLETE.** See §10.8.

**Rollback:** delete `src/dial.rs` + the `lib.rs` line.

---

### PHASE 2 — `fit::assess_pair()` (dual-residency truth)
**Goal:** Answer "can these two models be resident together on THIS machine without eviction?" — modelling the **real** constraint (§1.3), not just RAM.

**Files:** `src/fit.rs` (additive only — no existing signature changes).

**Contract**
```
PairVerdict = BothResident | BothCpuOk | OnlyPrimary(reason) | Unknown
assess_pair(hw, primary: &FitInputs, oracle: &FitInputs, cpu_weight_budget_bytes: u64) -> PairVerdict
```
Must account for: combined weights vs the **CPU-weight materialisation budget** (default 6 GiB), combined KV, activation scratch counted **once per concurrently-executing model** (they run sequentially, not simultaneously — document this assumption in the doc-comment), and VRAM when GPU-resident.

> **Do not touch `FitVerdict`.** `check-fit-verdict-vocabulary.mjs` compares the WebUI vocabulary to `src/fit.rs` (R6). `PairVerdict` is a *separate* type; verify the script does not scan it, and if it does, update the WebUI side in the same commit.

**Tests**
1. Two small models under budget → `BothResident`.
2. Combined weights **1 byte over** budget → `OnlyPrimary` (boundary test).
3. Combined weights **exactly at** budget → defined, documented, tested (inclusive).
4. Primary alone already over budget → `OnlyPrimary` with a distinct reason.
5. `hw` RAM unknown → `Unknown` (abstain).
6. Budget = `u64::MAX` (env unset path) → falls back to RAM/VRAM limits, not "always yes".
7. Overflow safety: `weight_bytes = u64::MAX` → no panic, no wraparound.
8. GPU present but VRAM only fits one → `BothCpuOk` or `OnlyPrimary`, never `BothResident`.
9. **Consistency invariant:** if `assess_pair` says `BothResident`, then `assess` on each model individually must not say `WontFit`. Property-style test over a grid of sizes.
10. **Ablation:** removing the budget term makes test 2 pass when it must fail.

**Exit Gate**
- [x] 10 tests green, ablation verified.
- [x] `node scripts/check-fit-verdict-vocabulary.mjs` exits 0.
- [x] fmt + clippy + `cargo test --lib` green, no pre-existing test perturbed.

**PHASE 2 COMPLETE.** See §10.9.

**Rollback:** revert `src/fit.rs` (additive-only, so a clean revert).

---

### PHASE 3 — Read-only HTTP surface (`GET /api/dial`)
**Goal:** Expose what the Dial *would* do, with reasons. **Still zero generation change.**

**Files:** `src/api/mod.rs` (route + handler + response types).

**Contract**
- `GET /api/dial` → `{ enabled, surface, tiers: [{ tier, available, primary_model_id, primary_model_name, review_mode, reason, estimated_relative_cost }], dual_model_oracle: { setting_enabled, hardware_permits, effective } }`
- Query `?surface=chat|workspace` (default `chat`).
- Uses `HardwareProfile::cached()` (**not** `detect()`) — a live probe re-inits CUDA and this endpoint may be polled.
- **Side-effect free.** No model loads, no downloads, no background warms. (The repo has been burned by a GET that scheduled fetches; do not repeat it.)
- Honours `CAMELID_DIAL=0` → `enabled: false`, all tiers unavailable.

**Tests (Rust, `#[cfg(test)]`, no model files)**
1. Model-less server → 200 with all tiers unavailable (not 404, not 500).
2. Kill switch set → `enabled: false`.
3. `?surface=workspace` differs from `?surface=chat` when no tool-capable model exists.
4. Invalid `surface` value → typed **400**, not a silent default.
5. Response shape is stable/serialisable; snapshot-pinned field names.
6. **Side-effect proof:** call the endpoint 50× → `loaded_models` unchanged, no download registered, no fit-dims fetch scheduled.
7. Concurrency: 32 simultaneous calls → no deadlock, no lock inversion with `model_transition`.

**Exit Gate**
- [x] 7 tests green (**8** — a regression test was added for a defect the live run found).
- [x] fmt + clippy + full `cargo test --all-targets` green (**65 targets, 2136 passed, 0 failed, 90 ignored**).
- [x] Manual: boot the real binary model-less, `curl /api/dial` → sane JSON.
- [x] **Ablation:** `/v1/chat/completions` byte-identical output before/after this phase for a fixed seed+prompt.

**PHASE 3 COMPLETE.** See §10.10.

**Rollback:** remove route + handler.

---

### PHASE 4 — Self-critique pass (Workspace first)
**Goal:** The first real behaviour change: `high` runs a second pass with the **same** model. Workspace only.

**Files:** `src/chat/workspace_bridge.rs` (new `WorkspaceEvent` variants), `src/api/workspace.rs` (plumb tier), `src/dial.rs` (critique prompt construction — pure).

**Contract**
- Critique prompt = `(original task, draft answer, review instruction)`, built by a **pure function** in `dial.rs` so it is unit-testable without a model.
- Critique request has the dial field **cleared** (AD-9) — recursion impossible.
- New events: `dial.draft_ready`, `dial.review_started`, `dial.review_finished { changed: bool }`. Additive; existing consumers unaffected.
- **Guards, all fail-open to the draft:**
  - if `prompt + draft + instruction` exceeds the context budget → **skip review**, emit `review_skipped { reason: "context_budget" }`, return the draft.
  - if the remaining step-time budget < a configured floor → skip review (R2).
  - if the review output is empty / unparseable → return the draft unchanged.
  - if the review says "no change needed" → return the draft, `changed: false`.
- **Cancellation:** cancel during draft ⇒ no review; cancel during review ⇒ return the draft, clearly labelled — never lose completed work, never return nothing.

**Tests**
*Rust unit (no model)*
1. Critique prompt builder: deterministic, includes task + draft, no template injection from draft content.
2. Draft containing the review-instruction text verbatim → still built unambiguously (prompt-injection style edge case).
3. Empty draft → review skipped.
4. Context-budget guard triggers at the boundary (exact token arithmetic, ±1 test).
5. Time-budget guard triggers below the floor.
6. Recursion guard: the constructed critique request has no dial field. **Ablation:** removing the clear causes an infinite-regress test to trip a depth counter.

*Integration (bridge-level, canned model)*
7. Two-pass turn emits events in exactly: `draft_ready` → `review_started` → `review_finished`.
8. Review changes the answer → final answer is the revised one and `changed: true`.
9. Review declines → final answer is byte-identical to the draft.
10. Cancel during draft → no review events at all.
11. Cancel during review → draft returned, terminal state correct, no orphaned turn.
12. Review exceeds 90 s step timeout → turn does **not** fail; draft returned with a skip/timeout reason (R2).
13. `low`/`medium` → **zero** dial events emitted (single pass unchanged).
14. Tool calls in the draft are **not** re-executed by the review pass.

*Negative control*
15. With `CAMELID_DIAL=0`, a `high` request behaves exactly like `medium`.

**Exit Gate**
- [ ] 15 tests green; ablations 6 and 15 verified.
- [ ] `npm run smoke:workspace` + `smoke:workspace-visual` green.
- [ ] fmt + clippy + `cargo test --all-targets` green.
- [ ] **Byte-identical control:** `low`/`medium` output unchanged vs Phase 3 binary, fixed seed, sha256 compared.

**Rollback:** revert the three files; events are additive so no consumer breaks.

---

### PHASE 5 — Dual-model oracle (`ultra`), hardware-vetoed
**Goal:** A genuinely different second model reviews — **only when provably safe**.

**Files:** `src/dial.rs`, `src/api/mod.rs` (oracle load/keep-warm), `src/runtime_config.rs` (flag).

**Contract**
- Flag `CAMELID_DIAL_DUAL_MODEL` via **`env_flag_default_off`** (§1.7 helper).
- Engage **only if** `assess_pair(...) == BothResident` **and** the flag is on **and** the Settings toggle is on. Any one false ⇒ silently degrade to Phase-4 self-critique on the larger model, with the reason surfaced.
- **Never trigger eviction of the primary.** Before loading the oracle, compute the projected `current_sum + oracle_bytes` exactly as `load_weights_lru` does (§1.3) and refuse if it would evict (R1).
- Oracle load failure (corrupt file, disk error, OOM) ⇒ degrade to self-critique, **not** an error to the user.
- Oracle is loaded but **not made `active_model_id`** — the primary stays active.

**Tests**
1. `assess_pair` says `OnlyPrimary` → oracle never loads (assert `loaded_models` count unchanged).
2. Flag off + hardware fine → no oracle (flag is authoritative).
3. Flag on + hardware insufficient → no oracle, degradation reason surfaced.
4. Oracle load fails mid-way → degrades, turn still completes.
5. **Eviction guard (the R1 test):** construct a budget where loading the oracle *would* evict the primary → assert we refuse and the primary's weights remain cached.
6. **Ping-pong regression:** 5 consecutive `ultra` turns → assert the primary is loaded **once**, not 5×.
7. `active_model_id` never changes to the oracle.
8. Oracle unload/cleanup on tier change; no leak across 20 tier switches.
9. Concurrency: tier switch during an in-flight oracle load → single-flighted, no double load.
10. **Ablation:** removing the eviction guard makes test 5 fail.

**Exit Gate**
- [ ] 10 tests green; ablation 10 verified.
- [ ] Live receipt on the real binary **if and only if** Phase 0 predicted this machine can host a pair; otherwise record an honest "cannot be validated on this hardware" and mark the tier experimental.
- [ ] fmt + clippy + full suite green.

**Rollback:** flag defaults off — a revert is a one-line default flip plus removing the load path.

---

### PHASE 6 — Chat surface + UI (the visible feature)
**Goal:** Ship the dial in the UI. This is the phase that changes default UX.

**Files:** `frontend/src/components/dial/` (new), `ChatWorkspace.jsx`, `WorkspaceView.jsx`, `SettingsView.jsx`, `frontend/src/lib/dial.js` (new), `src/api/mod.rs` (accept `camelid_dial` on chat requests).

**Contract**
- Dial control in Chat + Workspace; default **`medium`**; persisted at **`camelid.dialTier`** (matches §1.8 key convention).
- Unavailable tiers are **visibly disabled with the reason on hover** — never silently missing.
- Settings gets one toggle: *"Allow a second model to review (only when your hardware allows)"*, default **off**, showing the live `hardware_permits` verdict from `/api/dial`.
- `camelid_dial` absent from a request ⇒ **exactly today's path** (AD-3).
- `camelid_dial` + `n>1` ⇒ typed **400** (R4), explicit not silent.
- `camelid_dial` + `stream` at `high`/`ultra` ⇒ draft streams, then an additive review phase (AD-7).
- Persisted tier whose model no longer exists ⇒ re-resolve on boot, fall back to `medium`, no crash.

**Tests**
*Rust*
1. Request without `camelid_dial` → byte-identical to base (fixed seed, sha256).
2. Invalid tier string → 400.
3. `camelid_dial` + `n=2` → 400 with a clear message.
4. Unknown/unavailable tier requested → typed refusal naming the reason.

*Frontend smoke (`frontend/scripts/dial-smoke.mjs`, registered as `smoke:dial` in package.json + ci.yml — follow the existing smoke pattern; CI's frontend job has Chrome)*
5. Dial renders in Chat and Workspace; default `medium`.
6. Unavailable tier is disabled and shows its reason.
7. Selection persists across reload (`camelid.dialTier`).
8. Persisted tier for a deleted model → falls back to `medium`, no console error.
9. Settings toggle reflects `hardware_permits`; disabled with explanation when hardware says no.
10. **Fresh profile / no localStorage** → no crash on first load (the repo has been burned by exactly this; test in a clean puppeteer profile).
11. 390 px mobile viewport → no overflow, control reachable.
12. Model-less backend → dial visible but all tiers disabled, honest empty state.

**Exit Gate**
- [ ] 12 tests green.
- [ ] `npm run build` + **all 14 existing smokes** + the new `smoke:dial` green.
- [ ] fmt + clippy + full Rust suite green.
- [ ] `bash scripts/check-public-scrub.sh` exits 0 (no local paths / RFC1918 in new files).
- [ ] Live manual pass on the real binary: switch all four tiers, observe honest states.

**Rollback:** revert frontend files; backend field stays optional and unused.

---

### PHASE 7 — Measurement (earn the right to claim anything)
**Goal:** Find out whether the critique pass actually helps, and what it costs. **No product claim may be made before this phase.**

**Files:** harness under `docs/local/dial/` (local, uncommitted) + an evidence bundle if results justify it.

**Method (locked in advance so we cannot p-hack)**
- Fixed task set: ≥30 prompts across `coding`, `reasoning`, `general`, with **objectively checkable** answers where possible (compiles / passes a test / exact match).
- Arms: `medium` (control), `high` (self-critique), `ultra` (dual, only if Phase 0/5 allow).
- **Interleaved A/B/A/B, never blocked** — this box has a documented 1.9–3× run-to-run swing; blocked designs here have produced retracted results before.
- Report **internal ratios** (review-cost ÷ draft-cost within the same run) alongside wall-clock, because ratios reproduced within 3 % historically while absolute tok/s did not.
- Record: correctness delta, **regression rate (how often review made it worse)**, added latency, peak RSS.
- Pre-register the prediction before running.

**Exit Gate**
- [ ] Pre-registered prediction recorded in §10 **before** any run.
- [ ] ≥30 tasks × ≥3 interleaved reps per arm.
- [ ] Correctness delta **and regression rate** both reported.
- [ ] Verdict recorded honestly, including "no measurable benefit" if that is the result — in which case `high`/`ultra` ship described as "spends more time reviewing", with no quality claim, or are cut.

**Rollback:** n/a (measurement only).

---

### PHASE 8 — Contract, docs, closure
**Goal:** Declare the feature honestly in the capability contract; make CI enforce the truth.

**Files:** `src/api/mod.rs` (capabilities entry), `ledger/`, `README.md`, `COMPATIBILITY.md`.

**Contract**
- Capability entry describes **exactly** what was measured in Phase 7 — scope-limited, no generalisation beyond the tested models/hardware.
- **CI trap (R7):** never write a non-vocabulary `status: "…"` as a struct-init literal; `check-ledger-schema.mjs` greps every such literal *including test code*. Build with an allowed placeholder then assign.

**Tests**
1. `node scripts/check-ledger-schema.mjs` exits 0.
2. `node scripts/check-ledger-drift.mjs` exits 0.
3. `node scripts/check-public-evidence-claims.mjs` exits 0.
4. `bash scripts/check-public-scrub.sh` exits 0.
5. Capabilities response contains the dial entry with the exact measured scope.

**Exit Gate**
- [ ] All 5 green.
- [ ] Full CI-equivalent local run (§7) green.
- [ ] Feature described in README with **zero** unmeasured claims.
- [ ] **Ask before pushing.**

---

## 6. GLOBAL EDGE-CASE CATALOGUE

Every row must be covered by a named test by the phase listed. This is the checklist for "rigorous".

| # | Edge case | Phase |
| --- | --- | --- |
| 1 | Zero models installed | 1, 3, 6 |
| 2 | Exactly one model (all tiers collapse) | 1, 6 |
| 3 | Identical-size models (tie-break determinism) | 1 |
| 4 | All models `WontFit` | 1, 3 |
| 5 | `FitVerdict::Unknown` → abstain | 1, 2 |
| 6 | Workspace needs tool-capable; Chat does not | 1, 3 |
| 7 | Model file deleted between resolve and use | 5, 6 |
| 8 | GGUF header unparseable → advisory pad fallback | 1 |
| 9 | Combined weights over the 6 GiB budget (R1) | 2, 5 |
| 10 | Budget boundary ±1 byte | 2 |
| 11 | Budget env changed at runtime | 2 |
| 12 | Oracle load fails mid-way | 5 |
| 13 | Oracle load would evict primary | 5 |
| 14 | Ping-pong across consecutive turns | 5 |
| 15 | Cancel during draft | 4 |
| 16 | Cancel during review | 4 |
| 17 | Review exceeds 90 s step timeout (R2) | 4 |
| 18 | Review exceeds generation timeout | 4 |
| 19 | Context budget exceeded by two-pass (R3) | 4 |
| 20 | Empty / garbage review output | 4 |
| 21 | Review declines to change | 4 |
| 22 | Recursion guard | 4 |
| 23 | Draft contains injection-like text | 4 |
| 24 | Tool calls not re-executed by review | 4 |
| 25 | `camelid_dial` + `n>1` (R4) | 6 |
| 26 | Streaming + review phase ordering | 4, 6 |
| 27 | temp=0 determinism across both passes | 4, 7 |
| 28 | Tier switch mid-generation | 6 |
| 29 | Tier switch while `blocks_model_transition()` | 6 |
| 30 | Rapid tier toggling (single-flight) | 5, 6 |
| 31 | Persisted tier for a deleted model | 6 |
| 32 | Fresh profile / empty localStorage | 6 |
| 33 | Kill switch `CAMELID_DIAL=0` | 3, 4 |
| 34 | Invalid tier string | 3, 6 |
| 35 | No-dial request byte-identical to base | 3, 4, 6 |
| 36 | 390 px mobile viewport | 6 |
| 37 | 32 concurrent `/api/dial` calls | 3 |
| 38 | Review makes the answer worse (regression rate) | 7 |

---

## 7. GATES (run these; do not improvise)

```powershell
$cargo = "$env:USERPROFILE\.cargo\bin\cargo.exe"
cd C:\camelid-dial

& $cargo fmt --all -- --check
& $cargo clippy --all-targets -- -D warnings
& $cargo test --lib
& $cargo test --all-targets

cd frontend
npm.cmd run build
npm.cmd run smoke:dial          # from Phase 6
cd ..

& "C:\Program Files\Git\bin\bash.exe" scripts/check-public-scrub.sh
node scripts/check-ledger-schema.mjs
node scripts/check-ledger-drift.mjs
node scripts/check-fit-verdict-vocabulary.mjs
```

**Environment notes for this box (learned the hard way):**
- `cargo` is not on PATH — use the full path above.
- PowerShell terminals wedge often: write a `.ps1` and run it with `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -File <path>`; redirect long output to a file and read the file.
- The first token of a command is sometimes eaten — prefix `$null=1;`.
- `npm.ps1` is blocked by execution policy — use `npm.cmd`.
- `cargo test --all-targets` at full parallelism has previously died with bogus ICEs from **memory exhaustion**, not source defects. If that happens: `-j 2` + `CARGO_INCREMENTAL=0`. Never read it as a regression.
- LNK1104 (link.exe) is transient on this box — retry before believing a failure.
- The VS Code workspace root is a **stale zip** without `src/`. Always search `C:\camelid-dial` via terminal, never trust workspace-wide grep for engine code.

---

## 8. NON-GOALS / DEFERRED

- Automatic tier selection from task difficulty (needs Phase 7 data first).
- Best-of-N / majority voting as a tier (`MAX_N_CHOICES=8` and deterministic per-choice seeding already exist at mod.rs:12901 — a natural Phase 9, deliberately out of scope now).
- Cross-machine tiers (the cluster lane is separate and measured slower for this shape).
- Any claim that the Dial improves answer quality, until Phase 7 says so with numbers.
- Chat-surface critique before Workspace has proven it (Workspace has checkable actions; Chat does not).

---

## 9. OPEN QUESTIONS (resolve as we hit them; record the answer here)

1. Should `medium` prefer the largest `FitsResident` model, or the best `reasoning`/`coding`-tagged one? → Decide in Phase 1 with the real inventory.
2. Exact review instruction wording — must be model-agnostic across llama/qwen/gemma templates. → Phase 4.
3. Does the review pass reuse the conversation's KV, or start clean? Clean is simpler and safer; measure the cost in Phase 7.
4. If Phase 7 shows no benefit, do `high`/`ultra` ship as honest "more time spent" tiers, or get cut to a two-position dial? → Decide with the data.
5. **OQ-5 (opened by Phase 0 Result 3) — ANSWERED 2026-08-10, see §10.12.** The premise is **not reproducible**: on today's binary all four models select **CUDA resident** lanes. Phase 0's `vram_used_mib = 0` is a **measurement artifact** (VRAM is allocated at first *generation*, not at load; Phase 0 sampled before any completion). Phase 0's `backend=cpu_*` strings were real but I could not reproduce or explain them — recorded as unexplained rather than guessed. **Consequence: C7's "host-RAM-first" is retained but is no longer sufficient — the binding constraint for dual residency is VRAM, and one model already consumes 61-82% of it on this box.**

---

## 10. CHANGELOG / PHASE RESULTS

| Date | Phase | Result |
| --- | --- | --- |
| 2026-08-08 | — | Worktree `C:\camelid-dial` created on `feat/local-dial` off `342b0f058` (v0.6.1). §1 ground truth verified against the real tree. Plan written. |
| 2026-08-08 | **0** | **COMPLETE — see §10.1-§10.7.** 35/35 claims re-verified, 4/4 absence checks clean. Release binary built (EXIT=0) and 4 models baselined on it. Predictions P1–P8 registered before implementation; **P2 confirmed** (`Unknown` models load), **P4 "falsified"** (everything ran on CPU, `vram_used=0`) — **that falsification is itself PARTLY RETRACTED on 2026-08-10, see §10.12**. 10 corrections applied (C1–C10), 2 new architecture decisions (AD-11, AD-12), 1 new risk (R9, observed live), 1 new open question (OQ-5). |
| 2026-08-08 | **1** | **COMPLETE — see §10.8.** `src/dial.rs` + one line in `src/lib.rs`. 19 unit tests pass, ablation verified, fmt/clippy/full-suite green, diff scope exactly 2 files. |
| 2026-08-09 | **2** | **COMPLETE — see §10.9.** `fit::assess_pair` added to `src/fit.rs`, **additive-only proven** (296 insertions, 0 deletions). 10 tests pass, ablation verified, vocabulary check exit 0, full suite 1707/0/84. |
| 2026-08-09 | **3** | **COMPLETE — see §10.10.** `GET /api/dial` in `src/api/mod.rs`. 8 tests pass, generation **byte-identical** vs Phase 0 on two models, live model-less boot verified. A live run caught a real defect (dial blind to `--model` outside the models dir) — fixed and pinned. |
| 2026-08-10 | **4** | **COMPLETE — see §10.11.** First real behaviour change: `high`/`ultra` run a second self-critique pass in the Workspace. 23 tests pass, 2 ablations verified, both workspace smokes green, chat generation still **byte-identical**. AD-12 implemented as *verdict-first* rather than diff/patch — recorded as a deviation, not coded around silently. |
| 2026-08-10 | **OQ-5** | **ANSWERED — see §10.12.** Phase 0's "everything runs on CPU" is **not reproducible**; all four models select CUDA resident lanes today. One measurement was outright invalid (`vram_used_mib` sampled before generation). **Phase 5's premise changes: dual residency is VRAM-bound, and one model already takes 61-82% of VRAM on this box.** |
| 2026-08-10 | **5 (pre-design)** | **MEASURED BEFORE CODING — see §10.13.** Dual residency *works* (both models load and generate, primary unharmed), but the GPU resident engine is **exclusive**: it is torn down and rebuilt on **every** model switch, costing **~1.7 s per `ultra` turn** against a warm generation of 0.2-0.4 s. Also found: `POST /api/models/load` **moves `active_model_id` to the oracle**, which the Phase 5 contract forbids. Both facts change the design; neither was in the plan. |
| 2026-08-10 | **5** | **DEFERRED by decision.** Phase 5 is not built. Rationale: it needs a refactor of `load_model` (the lane's first non-additive change) to buy a mechanism costing ~1.7 s/turn whose *benefit is still unmeasured*. Phase 7 answers whether review helps at all; building `ultra` first risks paying for a path Phase 7 then deletes. Revisit only if Phase 7 shows review is worth it. |
| 2026-08-10 | **7 (pre-registration)** | **PREDICTION REGISTERED BEFORE ANY RUN — see §10.14.** Feasibility verified live: `dial=high` session creation works, an invalid tier is refused with 400, and the Workspace surface gives `medium` and `high` the **same** model (`Llama-3.2-3B-Instruct-Q8_0.gguf`), so the arms differ only by the review pass. |
| 2026-08-10 | **DEFECT (mine) — FIXED** | **Phase 7's harness caught a Phase 3 defect before it could produce a bogus measurement — see §10.15.** The dial advertised Workspace `medium`/`high`/`ultra` on `Llama-3.2-3B-Instruct-Q8_0.gguf`, which the session gate refuses with **422 `model_not_tool_capable`**: the local copy is a different publisher's build and `tool_capable` is earned against **specific bytes**. The dial checked only the filename, bypassing the hash pin. Fixed, ablated, and re-verified live. |
| 2026-08-10 | **DEFECT (not mine) — FOUND** | **The Workspace agent loop drops every structured tool call on the streaming path — see §10.16.** `client.rs`'s own doc says reading only text "loses every tool call", but that rule is implemented only non-streaming. `run_live` sets a delta sink, so Workspace streams, so no Workspace turn can ever call a tool. Blocks the Workspace surface for tool-using models entirely. **Not caused by the dial** — reproduced with no `dial` field. |
| 2026-08-10 | **7** | **COMPLETE — see §10.17. VERDICT: the review pass costs ~5x latency and delivers ZERO measurable correctness benefit.** 90 trials, `medium` 75/90 vs `high` 75/90, **delta 0.00 pp**, 0 regressions, 0 rescues, 93.3% of reviews decline, median latency **4.99x**. Q1/Q2/Q5 held; **Q3 and Q4 falsified**. Per the pre-registered commitment, `high` gets **no quality claim**. |
| 2026-08-10 | **7b (pre-registration)** | **§10.17 had a CEILING EFFECT — registering a second experiment before running it, see §10.18.** An 83% baseline on single-fact questions is the one regime where self-critique *cannot* help: correct answers need no fix, and unknown facts cannot be fixed by a reviewer with no new information. E2 tests **multi-constraint instruction-following**, where violations are mechanically visible on re-reading. Both results will be reported. |
| 2026-08-10 | **7b** | **COMPLETE — see §10.19. Delta 0 pp AGAIN, and a far more damning signal: a 100% decline rate.** 72 trials, `medium` 57/72 vs `high` 57/72. The reviewer replied `NO CHANGES` to **every single answer**, including 15 that violated explicit, trivially checkable constraints (9 words when 7 were demanded; a sentence full of `e` when `e` was banned). **The reviewer is not reviewing.** Also caught: **6 of my 21 validators were broken** and silently failed *correct* answers — fixed and self-tested before the run. |
| 2026-08-11 | **7d** | **RUNNING (full n=72) — see §10.21.** 7c's instruction told the reviewer to check *silently*, which for an autoregressive model forbids the only place it can compute. 7d removes that restriction: the reviewer emits its check, then states a verdict after a delimiter. **Smoke n=12: delta 0.00 pp, 9.14x latency.** The dumps make the mechanism visible for the first time, and it is worse than "the prompt was wrong". |
| 2026-08-10 | **7c** | **COMPLETE — see §10.20. THIRD zero. `high`/`ultra` are CUT.** Same tasks, same drafts, one variable changed — the review instruction. `medium` 57/72, **shipped reviewer 57/72 (0 changes), directive reviewer 57/72 (3 changes)**. Both deltas **0.00 pp**; 0 rescues, 0 regressions. A reviewer explicitly told to enumerate and check each requirement became *more active* but not more useful: all 3 changes were one task reordering an already-correct list. |

---

### 10.21 Phase 7d — the mechanism, observed directly, and a defect in the shipped guard

**Why there was a fourth experiment at all.** §10.18 committed to no *third task set*, and 7d does not add one — the 24 tasks, the model, the temperature, the drafting step and the interpretation rules are byte-for-byte those of 7b/7c. What changed is that I found a defect **in my own 7c design**. The directive instruction said:

> "Work through these steps **silently** … do not show your checking."

A transformer has no hidden scratchpad; serial computation happens only in emitted tokens. I had instructed the reviewer to count in the one place where counting is impossible. 7c therefore did not test *"can a reviewer check constraints"* — it tested *"can a reviewer check constraints without computing"*, whose answer was never in doubt. 7d removes exactly that restriction: the reviewer writes its check, then a `----- VERDICT -----` line, then its decision. `verdict_found` is recorded per row so a reviewer that ignores the format is **counted**, never silently scored as a decline.

**Smoke result (n=12, one rep):** `medium` 9/12, worked reviewer 9/12, **delta 0.00 pp**, decline 91.7%, malformed 0%, **median latency 9.14x**. W1/W4/W5 held; **W2 and W3 falsified**.

**But the aggregate is not the finding.** For tasks 3, 9, 12, 21 and 23 the harness dumps the reviewer's verbatim text, and those dumps show *why* every experiment in this phase returned zero. There are **two** distinct failure modes.

**Mode 1 — detection fails.** Task 3 asks for exactly seven words. The draft *"The sea whispers secrets to the shore at dawn."* is nine.

```
- The answer contains exactly seven words: "The sea whispers secrets to the shore at dawn."
```

It quoted the nine-word sentence **as the evidence for its own false claim**, and never enumerated. It also invented requirements the request never contained ("does not contain any forbidden words", "contains the required word 'sea'") — lifted from the *examples in my instruction*, which is prompt text being treated as fact.

**Mode 1, decisive form.** Task 9 asks for exactly four words; the draft is `Golden Flame.` Here the reviewer *did* obey the enumeration order:

```
- The answer uses exactly four words: "Golden Flame." (1. "Golden", 2. "Flame")
```

**It listed two items and concluded four.** This kills the hypothesis that 7d was testing. The model was allowed to compute, it computed, it wrote the result on the page — and the verdict contradicted it. The enumeration has no causal path to the conclusion; it is decoration produced because decoration was requested.

**Mode 2 — repair fails even when detection succeeds.** Task 12 forbids the letter `e`. The reviewer got the check **right**:

> "contains the letter e, so this requirement is violated."

and then could not act on it. It emitted the same failing sentence nineteen times — *"a valid sentence without the letter 'e' would be: 'Brew a cup of coffee and savor its rich aroma.' — this version still contains 'e', so a valid alternative is needed."* — until it exhausted its 900-token budget. **187 seconds on a single row (~25x).**

**What this means for the concept, stated precisely.** A review pass is only useful if it is an *independent* check. It is not one. The same weights that could not satisfy the constraint while writing cannot satisfy it while repairing, so the best a perfect reviewer prompt can achieve is Mode 2 — correct detection followed by a degenerate repair, which is *worse* than declining. This explains all four experiments with a single cause and is why no fifth is warranted.

**A defect in shipped Phase 4 code, found by this run.** `dial::interpret_review` fails open on an **empty** review, but not on a **garbage** one: any non-empty reply that does not begin with `NO CHANGES` is adopted as the answer. On task 12 the user would have received the nineteen-fold repetition **instead of** the perfectly reasonable draft. There is no guard for degenerate or budget-truncated reviewer output, and the ~5-9x latency is spent *earning* that outcome. This is an independent argument for removing the review pass rather than leaving it dormant behind a flag: a dormant feature that can replace a good answer with loop output is a liability, not an option.

**Final aggregate (run stopped at n=47 of 72; the lane was cut by owner decision, not by the data).**

| metric | value |
| --- | --- |
| `medium` correct | 35/47 (74.47%) |
| `high` (worked reviewer) correct | 35/47 (74.47%) |
| **delta** | **0.00 pp** |
| repairs (wrong -> right) | **0** |
| breaks (right -> wrong) | **0** |
| reviewer declined | 45/47 (95.7%) |
| latency | **10.81x** (118 s of drafting -> 1157 s of reviewing) |
| worst single review | 93 s, task 12 (the Mode 2 loop) |

The partial sample does not weaken the conclusion: 0 repairs in 47 trials, and the direction of every prior experiment is identical. **Across all four experiments the review pass corrected 281 answers zero times.** No fifth experiment is warranted, and the lane is closed here.

**Closure decision (OQ-4, resolved).** `high` and `ultra` are cut. Phases 1-3 (tier resolution, `/api/dial`, honest availability) are kept and are genuinely useful on the Chat surface, where the tiers select different models (1B -> 3B -> 3B Q8). On the Workspace surface only one artifact is tool-capable, so all four tiers collapse onto Qwen3-4B and the review pass was the *only* thing distinguishing them — which is precisely why it measured zero. Untested alternative worth a future look: map tiers to **thinking mode** (`camelid_enable_thinking`) rather than to a second pass.

---

### 10.20 Phase 7c result — the prompt was not the problem either, and the verdict is final

**The question.** 7b's 100% decline rate looked like a *prompt* defect: the shipped instruction lets a reviewer jump straight to `NO CHANGES`. 7c tests that directly. Tasks, drafts, model, temperature, request format and interpretation are **identical**; the single variable is the review instruction. Both reviewers score the **same draft**, so this is a clean A/B of the instruction alone.

The candidate instruction forces the work the shipped one merely implies: *"Step 1: list every explicit requirement… Step 2: verify the answer against each one at a time, counting words or characters where a count is demanded. Step 3: if and only if EVERY requirement is satisfied, reply NO CHANGES."*

| Arm | Correct | Changes made |
| --- | --- | --- |
| `medium` (no review) | **57/72 = 79.17%** | — |
| **shipped** reviewer | **57/72 = 79.17%** | **0** |
| **directive** reviewer | **57/72 = 79.17%** | **3** |

**Both deltas 0.00 pp. Zero rescues. Zero regressions.**

The directive prompt *did* change the reviewer's behaviour — 0 changes became 3, so the instruction was read and acted on. But all 3 are **one task across 3 reps**, and the change was cosmetic reordering of an answer that was already correct:

```
DRAFT: France; Germany; Italy; Spain;
FINAL: Spain; France; Germany; Italy;
```

Meanwhile the same 15 constraint violations from §10.19 went past both reviewers untouched, three times each. **Making the reviewer more active did not make it more correct.** It found nothing, because it is not actually counting — it is producing text that looks like the output of counting.

**VERDICT: `high` and `ultra` are CUT.** §10.18 committed to this in advance: *"if R2 is falsified in this regime too… `high` is cut, not relabelled."* The mechanism has now failed three pre-registered experiments:

| | Regime | Delta |
| --- | --- | --- |
| §10.17 | short factual answers — where review *cannot* help (ceiling) | **0.00 pp** |
| §10.19 | multi-constraint instructions — where review *should* help | **0.00 pp** |
| §10.20 | same, with a reviewer explicitly told how to check | **0.00 pp** |

Across **234 measured trials** the review pass corrected **zero** facts and **zero** constraint violations, at a **~5x** median latency cost. There is no configuration left that this evidence supports trying.

**What the dial becomes.** `low` / `medium` — pick a smaller or larger local model, with honest per-tier availability. That is a real and defensible feature: it is the Fit Advisor made usable, and everything in Phases 1-3 (tier resolution, `/api/dial`, the availability tri-state bound to `refuses_load()`) still stands and is still correct. What does **not** ship is the claim that spending more time improves the answer, because it does not.

**What this costs, stated plainly.** Phase 4's review pass (the `dial.*` events, `run_review_pass`, the guards) becomes dead code the moment `high` is cut. It is well-tested, safe and fail-open — and it should still be removed rather than left dormant, per §1.11: *never advertise or retain a capability whose evidence does not exist.* Removing it is a clean revert of one file plus the tier plumbing.

**What I am NOT claiming.** Not "self-critique never works". The honest scope is: *a 4B Q4_K_M model, reviewing its own answer, on short-answer and constraint-following tasks, on this host, with two different reviewer prompts, produced no measurable improvement.* A larger reviewer, a genuinely different model (Phase 5's oracle, never built), or long-form generation could behave differently — none of which this lane measured, and none of which may be claimed.

---

### 10.19 Phase 7b result — the reviewer is not reviewing

**Result.** 24 constraint tasks × 3 reps = 72 trials.

| | |
| --- | --- |
| `medium` (draft only) | **57/72 = 79.17%** |
| `high` (draft + review) | **57/72 = 79.17%** |
| **Delta** | **0.00 pp** |
| Worse / better | 0 / 0 |
| **Reviews that declined** | **72/72 = 100%** |
| Median latency multiplier | **5.04x** |

**Scoring §10.18:** R1 **held on its discard guard** (79.17% < 85%, so this is not a ceiling artefact) though the point band of 30-75% was missed on the high side — stated plainly rather than rounded in my favour. R5 **held** (5.04x). **R2, R3 and R4 all FALSIFIED**: no positive delta, decline rate went *up* not down, and not one regression.

**The finding that matters.** A 100% decline rate is not "the answers were fine" — 15 of them were wrong. The reviewer approved, verbatim:

| Task | Required | Draft the reviewer approved |
| --- | --- | --- |
| t3 | exactly seven words | *"The sea whispers secrets to the shore at dawn."* — **nine** |
| t9 | exactly four words | *"Golden Flame."* — **two** |
| t12 | no letter `e` | *"Brew a cup of coffee and savor its rich aroma."* |
| t21 | exactly `!!!` and nothing else | *"Wow! Wow! Wow!"* |
| t23 | exactly six words | *"Rays scatter, blue dominates."* — **four** |

Every one is checkable by counting. The reviewer caught **none**, three times each, deterministically. Five tasks fail 3/3 in **both** arms — the model cannot satisfy them and the reviewer cannot see that it hasn't.

**A methodology defect I caught in my own harness before it could lie to me.** The first version of the validators called a helper defined in the task file, but `& script.ps1` runs in a child scope, so the helper was gone when the scriptblocks executed and my `try/catch` swallowed the error into `false`. **6 of 21 validators silently failed CORRECT answers.** That would have manufactured a fake low baseline and made the review look better than it is — the mirror image of §10.17's ceiling effect. Fixed by inlining, then **self-tested against hand-written compliant and non-compliant answers for every task: 0 of 21 broken.** The rule this earns: *a validator is a measuring instrument and must be calibrated against known inputs before it is trusted.*

---

### 10.18 Phase 7b pre-registration — testing the regime where the mechanism could actually work

**Why a second experiment is not moving the goalposts.** §10.17's result stands and is reported unchanged. But its task set has a defect I should have seen when designing it: the baseline was **83.3% correct on short single-fact questions**. In that regime a reviewer has nothing to do — a correct answer needs no change, and the 15 failures were facts the model does not know, which a reviewer holding no extra information cannot supply either. **§10.17 measured a regime where the mechanism is structurally unable to help.** That is a flaw in my experiment design, not a property of self-critique, and saying otherwise would over-claim from the data.

**The regime where it should work, if it works at all.** Multi-constraint instruction-following: *"list exactly three, one per line, no numbering"*. Small models violate such constraints often, and — crucially — **a reviewer re-reading the original request can verify every constraint mechanically, with no information it lacked on the first pass.** That is precisely the case §10.17 could not create.

**Method.** Identical machinery to §10.17 (same verbatim `dial.rs` protocol, same shared-draft design so only the review differs, `temperature=0`, 3 reps). Only the task set changes: 24 constraint tasks scored by **programmatic validators** (count lines, check casing, check forbidden/required tokens) written before any output is seen. Baseline is expected to be far below ceiling, which is the point — there must be room to improve.

**PREDICTIONS (registered before the run):**

| # | Prediction | Falsified by |
| --- | --- | --- |
| **R1** | **Baseline is well below ceiling: `medium` scores 30-75%.** If it lands ≥85% the experiment has the same ceiling defect and its result must be discarded, not reported as a null. | medium ≥85% or ≤25% |
| **R2** | **Correctness delta is positive and larger than §10.17's: +5 to +25 pp.** Constraint violations are exactly what a re-read catches. | delta ≤0 pp |
| **R3** | **The decline rate falls well below §10.17's 93.3%** — under 70% — because there is genuinely something to fix. | decline ≥70% |
| **R4** | **Regressions appear: ≥1 case where `medium` was right and `high` wrong.** A reviewer empowered to rewrite will sometimes break a compliant answer. | 0 regressions |
| **R5** | Latency multiplier stays high (>3x), because the §10.17 prefill finding is structural, not task-dependent. | median <3x |

**Committed in advance:** if **R2 is falsified** (delta ≤0) in *this* regime too, then self-critique has failed in both the regime where it could not help and the regime where it should have. At that point `high` is cut, not relabelled — two pre-registered failures is enough, and I will not go looking for a third task set.

---

### 10.17 Phase 7 result — measured, and the answer is no

**Scope, stated first.** This measures the **review protocol** (`dial::review_instruction` / `review_request` / `interpret_review`, copied verbatim into the harness) driven directly against the real model. It does **not** exercise the full Workspace surface, because §10.16's streaming defect makes any Workspace turn impossible. Since `run_review_pass` is exactly one model step with no tools, the protocol is faithful — but this is not end-to-end validation, and must not be described as such.

**Setup.** `Qwen3-4B-Q4_K_M.gguf` (bytes match the certified pin), 30 objectively checkable tasks across coding/reasoning/general, 3 reps, `temperature=0`, one draft per task/rep **shared by both arms** so the arms differ only by the review. Grading is word-boundary regex against answers fixed before any output was seen. No LLM judge.

| Metric | Result |
| --- | --- |
| `medium` (draft only) | **75/90 = 83.33%** |
| `high` (draft + review) | **75/90 = 83.33%** |
| **Correctness delta** | **0.00 percentage points** |
| Review made it worse | **0** |
| Review made it better | **0** |
| Reviews that declined | **84/90 = 93.3%** |
| Mean draft / review latency | 1181 ms / 3694 ms |
| **Median latency multiplier** | **4.99x** |

Per category, identical in both arms: coding 27/30, reasoning 21/30, general 27/30.

**Scoring the pre-registration (§10.14):**

| | Prediction | Outcome |
| --- | --- | --- |
| Q1 | delta within -5..+10 pp | **HELD** (0 pp) |
| Q2 | ≥60% of reviews decline | **HELD** (93.3%) |
| Q3 | at least one regression | **FALSIFIED** — 0 regressions |
| Q4 | median multiplier < 2.0x | **FALSIFIED** — 4.99x |
| Q5 | no lost answers | **HELD** (0 empty finals) |

**Q3 falsified in the good direction.** I predicted the review would sometimes make things worse. It never did, in 90 trials. The fail-open design (§10.11) works: the draft survives every ambiguous reply. `high` is **safe**.

**Q4 falsified badly, and my Phase 4 reasoning was wrong.** I argued a declined review is cheap because it emits "a handful of tokens". That is true of *output* and irrelevant to *cost*: the review prompt carries the instruction **plus the task plus the whole draft**, so prefill dominates. A review that replies `NO CHANGES` still costs **3.1x the draft it declined**. `estimated_relative_cost = 2.0` (a structural pass count, §10.10) understates real latency by ~2.5x. Anything user-facing must not imply 2x.

**What the 6 changes actually did — the finding that settles it.** All 6 land on just **2 tasks (14, 20), reproduced identically across all 3 reps** (deterministic at temp=0), so there are really **2 distinct behaviours, not 6 events**. In every case the review took an **already-correct** answer and *stripped its reasoning*:

```
DRAFT: To find the smallest number among 12, 4, and 64 ... The smallest is 4. **Answer: 4**.
FINAL: **Answer: 4**.
```

**The review never corrected a fact. Not once in 90 trials.** It only shortened correct answers. That is cosmetic compression sold at 5x the price.

**VERDICT — and the pre-registered commitment is honoured.** §10.14 committed, before the data: *"the honest outcome is that `high` ships described as 'spends more time reviewing' with no quality claim — or is cut."*

- **`high` gets NO quality claim.** There is no measured benefit to claim.
- On this evidence I recommend **cutting `high`/`ultra` from the dial**, or keeping them only as an explicitly-labelled, default-off "spends more time reviewing" option. A 5x latency cost for a 0 pp delta is not a feature.
- **Phase 5 (`ultra`) stays deferred, and this strengthens that call.** Building a dual-model oracle costing a further ~1.7 s/turn (§10.13) to extend a mechanism with zero measured benefit would have been exactly the wasted refactor deferring it avoided.

**Honest limits of this result.** One model (4B, Q4_K_M); one host; tasks with short factual answers, which is the regime where a second pass has least to add — a longer-form or multi-step task might behave differently. The grader checks a canonical token, so a change in prose that leaves the token intact scores as no change; the 6-change inspection above is the mitigation, and it found nothing a stricter grader would have caught. **Do not generalise this to "self-critique never helps"** — the claim earned here is narrower: *on this model, on these tasks, it cost 5x and changed nothing.*

---

### 10.16 Defect (not mine): the Workspace loop drops every structured tool call when streaming

**Symptom.** Every Workspace turn ends `step_capped` with **zero `tool.call` events**, zero content deltas, and a grounding-guard notice repeated until the step cap (139 s at the default 12 steps). Two guards fire depending on phrasing: *"Workspace inspection is required before answering this file request"* and *"Workspace must read each named file before answering"*.

**Isolation ladder, each rung measured:**

| Test | Result |
| --- | --- |
| model, plain generation | `READY` — fine |
| model + tools, `stream:false` | `finish_reason=tool_calls`, `list_dir({"path":"src"})` — fine |
| model + tools, `stream:true` | `delta.tool_calls` present, content empty — fine |
| **Workspace agent loop** | **0 tool calls** |

**Root cause, in the code.** `src/chat/client.rs`'s own doc comment states the rule: *"The server emits structured `tool_calls` here — reading only the text loses every tool call."* That rule is implemented **only** on the non-streaming path (`body.pointer("/choices/0/message/tool_calls")`). The streaming path accumulates content and recovers calls by **text parsing** (`tool_parse::parse(&turn.content, &self.family)`). But when the model tool-calls, the server sends the call in `delta.tool_calls` with **empty content** — measured above. So streaming accumulates nothing, the text parse finds nothing, and the call is lost.

`run_live` calls `set_delta_sink(...)` ⇒ **the Workspace always streams** ⇒ a tool-using Workspace turn can never succeed.

**Not caused by this lane.** Reproduced with requests carrying **no `dial` field at all**. Independent of Phase 4 and of the dial fix in §10.15.

**Hypotheses tested and killed** (so nobody re-treads them): my `max_steps=6/max_tokens=256` (defaults 12/512 fail identically); the grounding guard being the cause (a goal naming no file arms a *different* guard and still gets 0 tool calls); Qwen3 thinking mode (opt-in via `camelid_enable_thinking`, default-off, repo is parity-locked thinking-disabled).

**Not fixed here.** It is a different subsystem from the dial lane and deserves its own change, tests and ablation. Recorded so it is not rediscovered from scratch.

---

### 10.15 Defect: the dial advertised a Workspace tier the session gate refuses

**Found by** the Phase 7 pilot, whose every turn came back `session_refused`. The harness reporting an honest failure is exactly what a pilot is for.

**Symptom (live):**

```
POST /api/agent/workspace/sessions   ->  422
{"code":"model_not_tool_capable",
 "message":"the active exact model row has not earned tool-capable status"}
```

…while `/api/dial?surface=workspace` advertised that same model for three of four tiers.

**Root cause — and my first diagnosis was WRONG.** I initially blamed a missing `status.starts_with("supported")` clause (the gate's `tool_capable_compatibility_rows` has it; my `dial_tool_capable_row_ids` did not). That is a real divergence but **does not explain this failure**: `llama32_3b_instruct_q8_0` is `tool_capable: true` **and** `supported_exact_row_smoke`, so it passes both filters. Had I stopped at the plausible answer I would have shipped a fix that fixed nothing.

The real cause is the **hash pin**:

| | sha256 |
| --- | --- |
| pinned for `Llama-3.2-3B-Instruct-Q8_0.gguf` | `f34112a11b7dad74ab517dedf6dcf00d624c9adac2dc0c72c719ca0478554ef2` |
| **the local copy** | `b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921` |

Different bytes — a different publisher's build (the catalog row is 3421898816 bytes; the local file is ~3263 MiB). `tool_capable` is earned per exact row by a committed agent-eval receipt **against specific bytes**, and `tool_capable_row_for_loaded_artifact` enforces that pin for exactly the reason its own doc gives: *"a same-named replacement inherits an agent battery it never passed."* **The gate was right; the dial was wrong**, because it matched on filename alone and never looked at the bytes.

**Fix.** `dial_tool_capable_artifact(path, filename)` now answers as the gate does — the row must be tool-capable **and**, when the artifact is hash-pinned, the digest must match — using `receipt::sha256_file_hex_cached`, the loader's own cache, so the warm path is a stat. Only hash-pinned artifacts pay for a digest. My parallel `dial_tool_capable_row_ids` is **deleted**, so there is no second resolution path left to drift; `tool_capable_row_for_filename` became `pub(super)` and is now the single source.

**Gates:** 7 tool-capability tests pass; `cargo test --lib` **1740 pass / 0 fail / 84 ignored** (1738 + the 2 new); clippy `-D warnings` exit 0; fmt exit 0. Whole-lane diff is now 2 deletions (the Phase 4 import line + this visibility change).

**Ablation — and a second mistake worth recording.** My first ablation **passed**, which I nearly accepted. It passed because `cargo test --lib dial` does not match test names lacking the word "dial", so the new tests never ran. Re-run under a matching filter, the ablation behaves: removing the digest check fails **exactly** `a_hash_pinned_artifact_with_the_wrong_bytes_is_not_tool_capable` (6 pass / 1 fail) while its complement still passes; restored, 7 pass / 0 fail. **A green ablation is a broken ablation — check the filter actually selected the test.**

**Live re-verified** on a rebuilt binary (exit 0, 6m07s, mtime `17:34:21` vs the previous `12:55:23`, checked before use):
- `surface=workspace` → all four tiers now resolve to `Qwen3-4B-Q4_K_M.gguf`; the uncertified Llama build is **gone**.
- The session gate **accepts** what the dial now advertises (session created).
- `surface=chat` still offers the Llama builds, which is correct — Chat does not require tool capability.

**Consequence for Phase 7:** `Qwen3-4B-Q4_K_M.gguf`'s local bytes **match** its pin (`7485fe6f…fdf5`), so it stays tool-capable and every Workspace tier resolves to it. The `medium` vs `high` comparison is still clean — the arms differ only by the review pass — so §10.14's pre-registration stands unchanged, with the model corrected from Llama-3.2-3B-Q8_0 to Qwen3-4B-Q4_K_M.

---

### 10.14 Phase 7 pre-registration (written BEFORE any measurement run)

Recorded first so the result cannot be rationalised afterwards. The plan requires this; §10.5's P2/P4 pre-registration is the precedent.

**What is being compared.** `medium` (one pass) vs `high` (draft + self-critique). Verified live on the Workspace surface: both tiers resolve to **`Llama-3.2-3B-Instruct-Q8_0.gguf`**, so the *only* difference between arms is Phase 4's review pass. `ultra` is excluded — Phase 5 is deferred and `ultra` currently resolves to self-critique anyway, i.e. it would be a duplicate of `high`.

**Feasibility facts established before designing the run:**
- `POST /api/agent/workspace/sessions` with `{"dial":"high"}` returns a session (`state=waiting_for_events`).
- `{"dial":"turbo"}` is refused **HTTP 400** — `resolve_dial_tier` works on the live binary.
- The management guard needs `Origin` matching the host **or** `sec-fetch-site: same-origin`; a request with neither gets **403**. The harness must send them.
- Two tool-capable models are installed: `Llama-3.2-3B-Instruct-Q8_0.gguf`, `Qwen3-4B-Q4_K_M.gguf`.

**PREDICTIONS (falsifiable, registered now):**

| # | Prediction | How it is falsified |
| --- | --- | --- |
| **Q1** | **Correctness delta is small: between -5 and +10 percentage points.** A 3B model reviewing its own answer has no information it lacked on the first pass. | A delta outside that band either way. |
| **Q2** | **The decline path dominates: ≥60% of `high` turns emit `dial.review_finished{changed:false}`.** `interpret_review` fails open, and `NO CHANGES` is the cheap reply. | <60% unchanged. |
| **Q3** | **Regression rate is non-zero: ≥1 task where `medium` was right and `high` was wrong.** This is the number that decides whether the tier can ever be default-on. | Zero regressions across all reps. |
| **Q4** | **Latency multiplier is below the advertised 2.0×**, because a declined review emits a handful of tokens, not a full answer. `estimated_relative_cost` is a structural pass count, not a latency claim (§10.10) — this checks it does not accidentally read as one. | Median ratio ≥2.0×. |
| **Q5** | **No crashes, no lost answers.** Every `high` turn produces an answer, since every guard and failure path keeps the draft. | Any turn ending without an answer. |

**Method, locked now.** Fixed task set with objectively checkable answers over a purpose-built fixture workspace whose ground truth I control; grader written before any output is seen; **no LLM judge**. Arms interleaved `medium`/`high` per task, never blocked, because this box has a documented 1.9-3x run-to-run swing. Recorded per turn: correctness, `changed` flag, wall time, and whether a review was skipped and why.

**Committed in advance:** if Q1 lands inside the band and Q3 shows regressions, the honest outcome is that `high` ships described as *"spends more time reviewing"* with **no quality claim** — or is cut. That sentence is written here, before the data, on purpose.

---

### 10.13 Phase 5 pre-design measurements (run before writing any code)

The Phase 5 contract gates the oracle on `assess_pair(...) == BothResident`, a **host-RAM** verdict. §10.12 showed the GPU lane is live, so I measured what actually happens rather than trusting that gate.

**Finding 1 — dual residency works, and the primary survives it.** With the 1B primary GPU-resident and generating, `POST /api/models/load {replace:false}` for the 3B succeeded in 3.9 s. `/v1/models` then listed both, generation on the *oracle* worked, and generation on the *primary* worked again afterwards. No OOM, no eviction line in the log, no error. So the naive fear ("a second model breaks the first") is **disproven**.

**Finding 2 — but the GPU resident engine is EXCLUSIVE, and it is rebuilt on every switch.** VRAM never holds both models; it alternates:

```
16 layers resident -> 28 layers resident -> 16 layers resident -> ...   (8 rebuilds for 8 switches)
vram: 5019 -> 6747 -> 5019 -> 6747 ...  MiB
```

| Call | Wall time (8 output tokens) |
| --- | --- |
| A warm (no switch) | 201 / 206 / 224 ms |
| B warm (no switch) | 348 / 355 / 396 ms |
| **switch to B** | 3814 (first) / 1527 / **1511 ms** |
| **switch to A** | 762 / 757 / **794 ms** |

Subtracting the warm baseline, a switch costs **~1156 ms into B and ~552 ms back into A**. An `ultra` turn is primary → oracle → primary, so it pays **~1.7 s of pure resident-engine rebuild per turn**, on top of the second pass — against a warm generation of 0.2-0.4 s. **The switching, not the thinking, would dominate `ultra` on a GPU host.**

**Finding 3 — the existing load path violates the Phase 5 contract.** The contract says *"Oracle is loaded but **not made `active_model_id`** — the primary stays active."* Measured: after loading the oracle, `active_model_id` **moved to the oracle** (`Llama 3.2 1B Instruct` → `Llama 3.2 3B Instruct`). `load_model` writes `*state.active_model_id.write().await = Some(id)` unconditionally (api/mod.rs:11562). **Phase 5 cannot reuse `load_model` as-is**; it needs the load core factored out with activation left to the handler. Planned test 7 would have caught this — it is good that the contract named it.

**Design consequences, recorded before implementation:**
- **I am NOT adding a "refuse whenever the GPU lane is live" veto**, even though it is tempting. It would make `ultra` dead code on every GPU machine — including the only one I can test on — and, more importantly, *whether ~1.7 s buys a better answer is exactly the question Phase 7 exists to answer*. Vetoing now would be me pre-empting the measurement phase with an aesthetic judgement.
- **The switch cost must be surfaced, not hidden.** `estimated_relative_cost` already refuses to be a fake latency multiplier (§10.10); the oracle path should likewise report that it pays a model-swap, so Phase 7 measures a cost the user was told about.
- **C7 stands amended**: `assess_pair == BothResident` remains necessary but is not sufficient, and it is not the *interesting* constraint. The interesting one is that the GPU resident engine is single-occupancy.

---

### 10.12 OQ-5 result — the GPU lane, and what it does to Phase 5

**Everything below is measured on the Phase 4 release binary (mtime `08-10 12:55:23`), same host, no relevant env vars set (`CAMELID_DETERMINISTIC`, `CAMELID_GPU`, `CAMELID_CUDA_RESIDENT_DECODE` all empty).**

**1. The premise is not reproducible.** Every model Phase 0 recorded as CPU now selects a CUDA resident lane:

| Model | Phase 0 backend (08-08) | Today's backend (08-10) |
| --- | --- | --- |
| Llama-3.2-1B-Q4_K_M | `cpu_kquant_block_dot` | **`cuda_resident_kquant_runtime`** |
| Llama-3.2-3B-Q4_K_M | `cpu_kquant_block_dot` | **`cuda_resident_kquant_runtime`** |
| Qwen3-4B-Q4_K_M | `cpu_kquant_block_dot` | **`cuda_resident_kquant_runtime`** |
| Llama-3.2-3B-Q8_0 | `cpu_q8_runtime_repack` | **`cuda_resident_q8_runtime`** |

The engine's own log agrees: `cuda_available=true`, `gpu_mode=Auto`, `gpu_backend="cuda"`, `gpu_enabled=true`, and the plan reason reads *"CUDA resident decode active"*.

**2. One Phase 0 number was simply invalid.** VRAM is allocated at **first generation**, not at load:

| Model | at load | after 1st completion | steady |
| --- | --- | --- | --- |
| Llama-3.2-1B-Q4_K_M | 91 MiB | **5019 MiB** | 5019 MiB |
| Llama-3.2-3B-Q4_K_M | 91 MiB | **6747 MiB** | 6747 MiB |
| Qwen3-4B-Q4_K_M | 91 MiB | **6747 MiB** | 6747 MiB |

`phase0_baseline.ps1` sampled `nvidia-smi` immediately after `generation_ready` and **before any completion**, so it could only ever have read ~0/91. **`vram_used_mib = 0` never meant "ran on CPU"; it meant "has not generated yet".** VRAM returns to 0 MiB on shutdown — no leak.

**3. What I could not explain, stated as such.** The `backend=cpu_*` strings Phase 0 recorded are *not* a sampling artifact, and I could not reproduce or explain them. Hypotheses **tested and disproven**, so nobody re-treads them:
- *"CUDA is not in the default build"* — **false.** `Cargo.toml` says `build.rs` turns on the `cuda` cfg for Windows and x86_64 Linux, with non-optional `cudarc`; no `--features` flag is needed.
- *"Missing CUDA DLLs beside the exe"* — **false.** No DLLs beside the exe, and it works anyway; the driver's `nvcuda.dll` in System32 is what cudarc loads.
- *"The laptop was on battery and the dGPU was parked"* — **false.** Today's entire run was on **battery** (`PowerOnline=False`, `Discharging=True`) with the GPU at P0/1890 MHz.
- *"`serve` re-plans per load and fails closed to the safe CPU plan"* (a real hazard the planner's own doc comment warns about) — **not applicable.** `AppState` captures `planner_env: PlannerEnv::capture()` once and every load goes through `plan_for_model_with_env`. Verified good; do not re-raise.

**4. The consequence that actually matters — Phase 5's premise changes.** A single resident model consumes **5019-6747 MiB of 8188 MiB (61-82%)**. Two GPU-resident models **cannot** coexist on this box, at any of the sizes we have. Note also that VRAM use is *not* proportional to weights: 1B (770 MiB on disk) took 5019 MiB, while 3B (1925 MiB) and 4B (2386 MiB) both took 6747 MiB — the engine sizes an arena against available VRAM, so "weights fit" is the wrong question entirely.

Therefore, for Phase 5:
- **C7 (host-RAM-first) is retained but demoted from sufficient to necessary.** Judging a pair on host RAM alone would happily admit two 4B models into 15.7 GiB of host RAM while the runtime has nowhere to put the second one.
- **`ultra` must be VRAM-vetoed, and on this hardware the veto will essentially always fire.** That is the tier working as designed (AD: degrade, never fail) — `ultra` degrades to self-critique. Phase 5 must therefore be built as *a veto that usually says no*, and must not grow a keep-warm second-model path that can never run here.
- **The Dial must never assume a lane.** It should read the runtime's actual selected backend rather than infer it, which is exactly the mistake Phase 0's method made.

**5. Method correction, applied.** Any future VRAM measurement in this lane must sample **after** at least one completion. `phase0_baseline.ps1` is retained as the latency/parity harness but its `vram_used_mib` column is not evidence of anything and should be read as "pre-generation".

---

### 10.11 Phase 4 result

**Shipped:** the review pass end to end — pure protocol core in `src/dial.rs`, the pass itself in `src/chat/workspace_bridge.rs`, tier plumbing in `src/api/workspace.rs`. Whole-lane diff is now **additive except one line**: the `super::agent` import extended with `ModelDriver, ModelStep`.

| Gate | Required | Result |
| --- | --- | --- |
| Tests | 15 | **23 passed / 0 failed** (7 pure + 13 bridge + 3 API) |
| `cargo fmt --all -- --check` | exit 0 | exit 0 |
| `cargo clippy --all-targets -- -D warnings` | exit 0 | exit 0, **0 diagnostics** |
| `cargo test --all-targets` | green | **2159 passed / 0 failed / 90 ignored** = Phase 3's 2136 + exactly the 23 added |
| `npm run smoke:workspace` | PASS | PASS |
| `npm run smoke:workspace-visual` | PASS | PASS — desktop, mobile, resume-preview, cancel-failure, cancel-settled |
| **Byte-identical control** | low/medium unchanged | **PASS, 4/4 reps** — 1B-Q4_K_M `e6cd169da6f22f75`, 3B-Q4_K_M `e52f19cab4a3f904`, prompt sha256 `69d481e4…f15c33`, `temperature=0`, `max_tokens=64` |

The control ran on a freshly built release binary (11m36s, mtime `08-10 12:55:23` vs the Phase 3 binary's `08-09 01:51:03`, checked **before** use so a stale binary could not pass it by accident).

**Shape of the change.** The review runs *after* `run_loop` returns `LoopEnd::Answered`, reading the draft back out of `history` — `run_loop` already pushes the answer there (`agent.rs`), so no reporter wrapping or loop surgery was needed. The revision is emitted as a **second `model.answer`**, which is exactly what the persistence layer keeps (`assistant_answer = Some(content.clone())` on every `ModelAnswer`, last one wins), so the durable turn records the final answer with no change to the store.

**Every path keeps the draft.** Guards, transport failures, and replies the module cannot read all fall back to the draft, so the worst case of a review is the answer the caller would have had anyway. The event contract is: exactly one of `dial.review_finished` or `dial.review_skipped` follows every `dial.draft_ready`.

**Design decisions worth keeping:**
- **The reviewer is offered no tools** — `driver.step(&history, &[])`. This makes "a review can never re-run the work the draft did" true *by construction*. It is therefore **not ablatable by deletion**; it is pinned by assertion (`seen_tool_counts == [0]`) and stated as structural rather than dressed up as an ablation.
- **`run_review_pass` takes `&mut dyn ModelDriver`**, not a `SocketAddr`. `run_live` still builds the real `LiveDriver`, but the pass is testable with a canned driver — which is why 13 bridge tests exist at all without a model or a socket.
- **The context-budget guard asks the driver, never re-derives.** `driver.prompt_tokens(...)` vs `context_budget_tokens()`. A count the driver cannot produce **proceeds**, because an overflowing prompt is refused by the server on its own and that refusal lands on the same keep-the-draft path.
- **Cancellation needed no new mechanism.** `set_stream_control(cancel, WORKSPACE_MODEL_STEP_TIMEOUT)` makes both cancel and the 90 s step timeout surface as `Err` from `step`, which is already the keep-the-draft arm.
- **The kill switch is enforced in exactly one place** — `resolve_dial_tier()` in `api/workspace.rs`. No caller can reach the bridge carrying a tier the switch was supposed to remove, and `dial.rs` stays pure (no env reads).

**Deviations recorded, not hidden:**
- **AD-12 ("verdict plus a patch, never a full regeneration") is implemented verdict-first**: `NO CHANGES` alone on the common path, or a corrected answer. Strict diff/patch output is unreliable from small local models, and truncating a genuine revision is worse than its cost. The bound is behavioural (decline is cheap), not a token cap. This honours AD-12's intent; it does not match its letter.
- **The time guard is turn-level, not step-level.** R2 named the 90 s step timeout, but the review is its *own* step with its own 90 s, so that risk is already covered by the `Err` path. `WORKSPACE_REVIEW_TURN_BUDGET` (10 min) + `WORKSPACE_REVIEW_TIME_FLOOR` (20 s) instead answer a different, real question: has the caller already waited long enough that doubling it is rude?

**Ablations (each fails *only* its own test):**
- Remove the kill-switch check → only `the_kill_switch_makes_a_reviewing_tier_resolve_like_a_plain_one` fails.
- Disable the context-budget guard → only `a_review_over_the_context_budget_is_skipped_before_the_model_runs` fails, while its positive-control twin `a_review_that_exactly_fits_the_context_budget_still_runs` keeps passing — so the test pins the boundary, not merely "a review never runs".

**Corrections the gates forced:**
- My first `an_unknown_tier_is_refused_rather_than_ignored` asserted `"HIGH "` was invalid. It is not — `DialTier::parse` deliberately trims and lowercases. **The test was wrong, not the code**; it now states the real contract (case and padding tolerated as transport noise, everything else refused).
- `wants_review()` was private; making it `pub` was required for the bridge and is the only visibility change in the phase.

**Environment note:** this worktree had no `frontend/node_modules`, so the first `npm run build` failed with `'vite' is not recognized` — not a code fault. `npm ci` fixed it (exit 0) and reports 3 pre-existing high-severity advisories in the lockfile, deliberately left alone rather than dragging an unrelated dependency bump into this phase.

---

### 10.10 Phase 3 result

**Shipped:** `GET /api/dial` — route, handler, response types, candidate assembly, and 8 tests, all in `src/api/mod.rs`. Whole-lane diff is now **826 insertions / 0 deletions** across `dial.rs` + `fit.rs` + `api/mod.rs` + `lib.rs`: still purely additive.

| Gate | Result |
| --- | --- |
| Tests (7 required) | **8 passed / 0 failed** |
| `cargo fmt --all -- --check` | exit 0 |
| `cargo clippy --all-targets -- -D warnings` | exit 0 |
| `cargo test --all-targets` | **65 targets — 2136 passed / 0 failed / 90 ignored** (run at `-j 2` with `CARGO_INCREMENTAL=0`, the documented guard for this box) |
| Manual model-less boot | 200; `enabled=true`; all four tiers `unavailable` / `no_models` with an honest reason; `?surface=turbo` → **HTTP 400** |
| **Ablation** | `/v1/chat/completions` **byte-identical to Phase 0** on two models: 1B-Q4_K_M `e6cd169da6f22f75`, 3B-Q4_K_M `e52f19cab4a3f904` — same prompt, `temperature=0`, `max_tokens=64` |

The ablation is a genuine before/after: the hashes were measured in Phase 0 on the **pre-Phase-3** release binary and re-measured on a freshly built one (13m36s, verified by mtime so the stale binary could not be tested by mistake).

**The live gate earned its place — it caught a defect the unit tests could not.** Running `serve --model <path outside the models directory>`, the dial reported *"No models are installed"* while the engine was actively serving that model, because candidate assembly only scanned `state.models_dir`.
- **Fix:** `dial_candidates(models_dir, also, hw)` now also accepts already-loaded paths, deduplicated by filename, with per-path construction factored into `dial_candidate_from_path`.
- **The subtlety that shaped the fix:** reading `loaded_models` with `read().await` would have broken the property test 7 depends on — the dial answering while a model transition holds every registry lock. It uses **`try_read()`** instead: best-effort enrichment that degrades to the directory scan under contention and can never block.
- Pinned by `dial_candidates_include_a_model_loaded_from_outside_the_models_dir`, which also asserts the dedupe.
- **Re-verified live on a rebuilt binary** (`BUILD2_EXIT=0`, 17m40s, mtime checked before use). With an **empty** `--models-dir` and `--model` pointing elsewhere — so the loaded-model path is the only way a model can appear — all four tiers named `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, `availability=ready`, `capacity_verified=true`, `high`/`ultra` on `self_critique`, and the reason read *"Only one model is installed, so every tier uses it."*

**R5/P7 observed live, not just predicted.** In that same run `?surface=workspace` returned `unavailable` / `no_tool_capable_model` for every tier while Chat was `ready` — the 1B Q4_K_M row is not one of the five `tool_capable` rows. The Chat/Workspace asymmetry that Phase 0 derived from the contract and Phase 1 pinned by unit test is now confirmed on a real binary.

**Design decisions worth keeping:**
- **Takes no server lock on the happy path** (only a non-blocking `try_read`), which is why `dial_answers_while_a_model_transition_holds_the_locks` can wedge `model_transition` + `loaded_models` + `active_model_id` and still get 32 × 200.
- **Footprints match the load guard exactly**: `fit_dims::dims_from_gguf_file` — verified to be literally `dims_from_gguf(read_metadata(path))`, the guard's own source — with `advisory_footprint` as the same fallback. Using the coarser pad alone would have been *more optimistic* than the guard and could offer a tier that then 422s.
- **`tool_capable` and `supported_row` resolve through the rows `/api/capabilities` already publishes** (memoized), via the same two filename sources `filename_is_supported_exact_row` consults. No second ledger.
- **`HardwareProfile::cached()`, not `detect()`** — the endpoint is pollable and `detect()` re-initializes a CUDA context. The guard keeps its live probe and stays authoritative; this is the same hint-vs-authoritative split the catalog badge already has, and it is stated in the handler doc.
- **`estimated_relative_cost` is a structural pass count (1 or 2), not a latency multiplier.** Phase 0 Result 7 measured a warm second pass far cheaper than the first, so a measured-looking number here would be a claim we have not earned. Phase 7 owns that.
- `dual_model_oracle.hardware_permits` calls Phase 2's `assess_pair` on the two largest candidates, and a second model is only ever proposed when that gate **and** the opt-in flag agree.

**Corrections the gates forced (recorded, not hidden):**
- I wrote `state.downloads` — **no such field exists**. The real registry is the process-global `active_downloads_map()`; asserting it empty would be flaky under parallel tests, so the test compares its length before/after the 50 polls.
- Three tests failed together because `CAMELID_DIAL=0` from the kill-switch test leaked into tests running in parallel (env is process-global). Fixed by adopting the repo's existing `crate::test_support::env_lock()` convention across all dial tests — a real bug in the tests, found by the tests.

---

### 10.9 Phase 2 result

**Shipped:** `PairRefusal`, `PairVerdict`, `combined_footprint_bytes`, `assess_pair_with_headroom`, `assess_pair` — all appended to `src/fit.rs`, plus 10 tests inside its existing `mod tests` (reusing that module's `profile` helper rather than duplicating it).

| Gate | Required | Result |
| --- | --- | --- |
| Tests | 10 | **10 added; `fit::` 38 passed / 0 failed** |
| Ablation | removing the budget term must fail test 2 | **verified** — failed *exactly* `combined_weights_one_byte_over_budget_are_refused` (37 passed / 1 failed) |
| `check-fit-verdict-vocabulary.mjs` | exit 0 | **exit 0** |
| `cargo fmt --all -- --check` | clean | **exit 0** |
| `cargo clippy --all-targets -- -D warnings` | clean | **exit 0** |
| `cargo test --lib` | no pre-existing test perturbed | **1707 passed / 0 failed / 84 ignored** (= Phase 1's 1697 + exactly 10; ignored unchanged) |
| Additive only | no existing signature changes | **proven**: `git diff -U0 src/fit.rs` contains **zero removed lines**; 296 insertions / 0 deletions |

**R6 mitigated concretely, not just avoided.** `check-fit-verdict-vocabulary.mjs` locates `as_str` / `is_positive_fit` / `refuses_load` by name using `fnBody`, which takes the **first** match in the file. A `PairVerdict::as_str()` placed above `FitVerdict::as_str()` (fit.rs:75) would have made the checker parse the wrong body, find no `FitVerdict::X => "y"` arms, and fail CI with "check cannot run". **`PairVerdict` therefore defines none of those three names** — it uses `admits_pair()` instead. The type's doc comment states this so a future edit cannot undo it by accident.

**Deviation from the written contract (deliberate, recorded).** The plan's signature was `assess_pair(hw, primary, oracle, cpu_weight_budget_bytes)`, but it also required activation scratch to be "counted **once**". Those are incompatible: `FitInputs` folds scratch into `kv_bytes_at_ctx` (`exact_footprint_with_scratch`, fit.rs:501), so from two `FitInputs` alone the scratch term cannot be recovered — summing them double-counts it. The signature therefore gained one parameter, `shared_scratch_bytes`, which the caller already knows because it chose the value when building the footprints. Honouring the *intent* (scratch once) was judged more important than the literal parameter list; `shared_scratch_is_counted_once_not_twice` pins it.

**Other contract refinements:**
- `OnlyPrimary(reason)` is `OnlyPrimary(PairRefusal)` — a machine-readable enum (`PrimaryOverWeightBudget` / `CombinedOverWeightBudget` / `HostMemory`), not a string, so callers never branch on prose. This is what makes plan test 4's "distinct reason" actually distinct from test 2's.
- Ordering is **host-RAM-first**, per Phase 0 correction C7: the weight-materialisation bound is checked first (it is what `load_weights_lru` evicts on), then host RAM, and VRAM only *upgrades* a passing pair to `BothResident`. A GPU-first rule would admit pairs this machine cannot stage — Phase 0 measured every model running on a CPU backend with `vram_used = 0`.
- Both budget bounds are **inclusive** (spending the budget exactly is not an overrun), tested at the boundary and one byte past it.
- `assess_pair` / `assess_pair_with_headroom` mirror the existing `assess` / `assess_with_headroom` split, so the env read stays out of the testable core.

**Consistency invariant proven, not assumed:** `admitting_a_pair_never_contradicts_the_single_model_verdict` sweeps 4 hosts × 5 × 5 sizes and asserts that whenever a pair is admitted, neither model alone is `WontFit`. It also asserts the sweep admitted at least one pair, so the test cannot pass vacuously.

---

### 10.8 Phase 1 result

**Shipped:** `src/dial.rs` (new) and one `pub mod dial;` line in `src/lib.rs`. Placed alphabetically between `diagnostics` and `diffusion_gemma` — the file's actual convention, which supersedes AD-1's "after `fit`" wording.

**Gate — every item measured, not asserted:**

| Gate | Required | Result |
| --- | --- | --- |
| Unit tests | ≥ 13 | **19 passed, 0 failed** |
| Ablation | test 8 must fail without the rule | **verified** — removing the `tool_capable` filter failed *exactly* the 2 Workspace tests, other 17 still passed |
| `cargo fmt --all -- --check` | clean | **exit 0** |
| `cargo clippy --all-targets -- -D warnings` | clean | **exit 0** |
| `cargo test --lib` | base + new, zero failures | **1697 passed / 0 failed / 84 ignored** (1697 + 84 = 1781 = 1762 base + 19 new) |
| `git diff --stat` | exactly 2 files | **`src/lib.rs` (+1), `src/dial.rs` (new)** |

**Design decision made in Phase 1 (resolves a real conflict in this document).** §2.2 says `medium` is "the best mid model that **FitsResident**", but AD-11 forbids choosing a model based on capacity. Those cannot both hold. **Resolution: tier → model selection is position-based on the deterministically sorted list and is entirely capacity-independent** (`low` = smallest, `medium`/`high` = midpoint, `ultra` = largest); capacity only ever changes the reported `Availability`. This is what makes P5 and P6 produce *identical model picks* and differ only in availability — asserted by `measured_host_reproduces_the_recorded_predictions`. §2.2's wording is superseded by AD-11; OQ-1 (task-tag-aware preference) stays open.

**Two things deliberately kept out of scope**, to avoid implementing ahead of the plan:
- The `budget_bytes` gate for `ReviewMode::SecondModel` is an honest placeholder that only compares combined weight bytes. Real admission is Phase 2's `assess_pair`; the doc comment on `ultra_review` says so explicitly rather than implying the check is sufficient.
- No `serde` derives and no env reads. The module is pure; `CAMELID_DIAL` and wire serialisation belong to Phase 3.

**Corrections to Phase 1's own tests, found by the gates (recorded because they are the reason the gates exist):**
- A `HashSet` over `ReviewMode` failed to compile. Fixed by rewriting the assertion pairwise rather than deriving `Hash` on a public type purely to satisfy a test.
- Clippy `identity_op` rejected `1 * GIB` in a test host vector. Fixed to `GIB`.

**Environment notes worth carrying forward:** `cargo test --lib` hit **LNK1104** once and passed on retry with no source change — exactly the transient documented in §7, and it must never be read as a test failure. Several `Out-File`-redirected gate runs also produced stale or empty output; every gate result above was re-read directly from a live terminal rather than trusted from a log tail.

---

### 10.1 Claim re-verification (automated, `phase0_verify.ps1`)

**35 pass / 0 fail / 0 drift.** Every `file:line` in §1.1-§1.9 matched exactly at `342b0f058`.

Absence checks (all as expected — confirms we are starting from zero):

| Check | Result |
| --- | --- |
| `assess_pair` / `PairVerdict` in `fit.rs` | 0 hits |
| `DialTier` / `camelid_dial` / `mod dial` in `src/` | 0 hits |
| `src/dial.rs` exists | False |
| `dialTier` / `DialTier` in `frontend/src/` | 0 hits |

### 10.2 Host profile (measured 2026-08-08T20:49 +05:30)

| Field | Value |
| --- | --- |
| CPU | Intel Core i9-14900HX — 24 physical / 32 logical |
| `host_ram_total_bytes` | 16,873,545,728 (**15.71 GiB**) |
| `host_ram_free_bytes` | 1,345,228,800 (**1.25 GiB**) |
| ⇒ `usable_host_ram_bytes` (80 % of free, no floor) | **1,076,183,040 (1.002 GiB)** |
| ⇒ `idle_host_ram_bytes` (80 % of total, diagnostic only) | 13,498,836,582 (12.57 GiB) |
| GPU | NVIDIA RTX 4060 Laptop, driver 577.02 |
| VRAM total / free | 8,188 MiB (7.99 GiB) / **7,957 MiB (7.77 GiB)** |
| ⇒ max VRAM alloc (free − 512 MiB headroom) | ≈ 7,806,648,320 (7.27 GiB) |

All budget-relevant env vars (`CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES`, `CAMELID_LAZY_Q8_0_LINEAR`, `CAMELID_RETAIN_Q8_0_BLOCKS`, `CAMELID_MOE_EXPERT_STORAGE`, `CAMELID_GENERATION_TIMEOUT_MS`, `CAMELID_GPU`, `CAMELID_CUDA_RESIDENT_DECODE`, `CAMELID_THREADS`, `CAMELID_DECODE_THREADS`) were **unset** ⇒ all documented defaults apply.

> RAM/VRAM above are WMI + `nvidia-smi` readings, i.e. a *proxy* for `HardwareProfile::detect()`. They agree in kind, and the tri-state rule does not depend on the last few MiB. Phase 1's host-vector test (7d) must pin the numbers the engine itself reports.

### 10.3 Model inventory (exact bytes + sha256, all 8 verified)

| # | Model | bytes | GiB | quant | sha256 (full) |
| --- | --- | --- | --- | --- | --- |
| 1 | Qwen3-0.6B-Q8_0 | 639,446,688 | 0.596 | Q8_0 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| 2 | Llama-3.2-1B-Instruct-Q4_K_M | 807,694,464 | 0.752 | Q4_K_M | `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` |
| 3 | Llama-3.2-1B-Instruct-Q6_K | 1,021,800,576 | 0.952 | Q6_K | `0f4c510daf16e0d1b3bc94931fd9296c28936bebdda2593687d4eb70c5b70628` |
| 4 | Llama-3.2-1B-Instruct-Q8_0 | 1,321,082,528 | 1.230 | Q8_0 | `3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1` |
| 5 | Llama-3.2-3B-Instruct-Q4_K_M | 2,019,377,696 | 1.881 | Q4_K_M | `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff` |
| 6 | Llama-3.2-3B-Instruct-Q5_K_M | 2,322,154,016 | 2.163 | Q5_K_M | `0b94ccd04d908304cec5246a3d942b64417a423bc5c6d47c73bc557e590b5194` |
| 7 | Qwen3-4B-Q4_K_M | 2,497,280,256 | 2.326 | Q4_K_M | `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` |
| 8 | Llama-3.2-3B-Instruct-Q8_0 | 3,421,899,296 | 3.187 | Q8_0 | `b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921` |

**Provenance cross-check (independent confirmation, not assumed):** #5 and #6 sha256 match the values hard-pinned in `capabilities` at mod.rs:5499 and mod.rs:5541 verbatim — these are the canonical artifacts, not look-alikes.

Files span three directories (`C:\models`, `C:\camelid-fork\models`, `C:\camelid-fork\target\first-run-lane\models`). A real deployment has ONE models dir; **assume `C:\camelid-fork\models` (models 2,3,4,5,7,8 = six models)** for all predictions below, and point `--models-dir` there rather than copying ~13 GB.

Quant is filename-derived; architecture is **pending `camelid inspect`** (binary still building). The resolver must read real GGUF metadata, never the filename.

### 10.4 HAND-COMPUTED TIER PREDICTION (registered BEFORE Phase 1)

Applying the verified rule from §1.13 with the §10.2 host vector. Advisory footprint = `weight_bytes × 1.25`. Every footprint fits VRAM (max 4.28 GiB ≤ 7.27 GiB), so **every model reaches the second gate**, and the verdict is decided entirely by `footprint > usable_host_ram (1,076,183,040)`:

| Model | footprint (w×1.25) | ≤ usable RAM? | **predicted `FitVerdict`** |
| --- | --- | --- | --- |
| Qwen3-0.6B-Q8_0 | 799,308,360 | yes | `FitsResident` |
| Llama-1B-Q4_K_M | 1,009,618,080 | yes (by 66 MB) | `FitsResident` |
| Llama-1B-Q6_K | 1,277,250,720 | no | `Unknown` |
| Llama-1B-Q8_0 | 1,651,353,160 | no | `Unknown` |
| Llama-3B-Q4_K_M | 2,524,222,120 | no | `Unknown` |
| Llama-3B-Q5_K_M | 2,902,692,520 | no | `Unknown` |
| Qwen3-4B-Q4_K_M | 3,121,600,320 | no | `Unknown` |
| Llama-3B-Q8_0 | 4,277,374,120 | no | `Unknown` |

**PREDICTION P1 — 6 of 8 models return `Unknown` on this machine right now**, despite 7.77 GiB of free VRAM and despite every one of them being known to run here. Cause: the host-RAM staging gate, driven by a *momentary* 1.25 GiB free-RAM reading.

**PREDICTION P2 (REVISED — see the self-correction note below).** `Unknown` does **not** refuse a load (§1.14), so the correct rendering of those six tiers is `Ready { capacity_verified: false }`, i.e. **offerable with an honest "capacity could not be confirmed" caveat** — not `Unavailable`, and not `NeedsFreeMemory` either. The original plan's `Unknown ⇒ Unavailable` rule would have rendered `medium`/`high`/`ultra` as **Unavailable** on a 16 GB / RTX 4060 laptop that runs all of them fine. That rule is R0 and is now deleted.

> **SELF-CORRECTION, recorded deliberately.** My first Phase 0 fix mapped `Unknown ⇒ NeedsFreeMemory`. Reading `FitVerdict::refuses_load` (fit.rs:104) showed that is **also wrong** — merely wrong in the opposite direction: it would nag the user to free memory for a load the engine would have accepted. Both errors came from re-deriving a policy the codebase already owns. The fix is not a better guess but a **binding to the authoritative predicate** (§1.14 invariant). Rule for the rest of this build: **never re-derive a decision the engine already makes; call it.**

**PREDICTION P3 — tier availability will be unstable over time.** The gate reads *free* RAM with no floor, so closing a browser tab can flip a tier. Any UI that renders this as a hard capability boundary will flicker. Phase 6 must present `NeedsFreeMemory` as an actionable, transient state.

**PREDICTION P4 — dual-model `ultra`: the 6 GiB budget is the WRONG lens (answer: budget-YES, RAM-NO, VRAM-PLAUSIBLE).**
- *Materialisation budget:* per §1.12, under stock env both models' quantized linears count **0**, so any pair passes 6 GiB by a wide margin. **Not the constraint.**
- *Host RAM:* staging alone already fails for one 3B model at 1.25 GiB free. A pair is **not** stageable in the current memory state.
- *VRAM:* the two largest (3.187 + 2.326 = 5.51 GiB) + ~0.9 GiB combined KV ≈ 6.4 GiB < 7.27 GiB usable ⇒ **arithmetically plausible**.
⇒ **Registered answer: a real dual-model `ultra` is plausible on this box via VRAM, and impossible via host RAM in the current state.** `assess_pair` (Phase 2) must therefore be **VRAM-first on a CUDA host**, model host-RAM staging explicitly, and treat the materialisation budget as a secondary edge condition. Phase 5 must verify empirically; if it cannot be reproduced, `ultra` ships marked experimental rather than claimed.

#### Tier→model mapping, hand-computed (models dir = `C:\camelid-fork\models`, six models)

Sorted by `weight_bytes`: 1B-Q4_K_M (807,694,464) < 1B-Q6_K (1,021,800,576) < 1B-Q8_0 (1,321,082,528) < 3B-Q4_K_M (2,019,377,696) < Qwen3-4B-Q4_K_M (2,497,280,256) < 3B-Q8_0 (3,421,899,296).

**PREDICTION P5 (REVISED) — in the CURRENT memory state, all four tiers are offerable, but only `low` has verified capacity.** 1B-Q4_K_M is `FitsResident` ⇒ `Ready{verified:true}`; the other five are `Unknown` ⇒ `Ready{verified:false}`. So the dial *is* multi-position right now — `low` = 1B-Q4_K_M, `medium` = 3B-Q4_K_M, `high` = 3B-Q4_K_M + self-critique, `ultra` = 3B-Q8_0 — with the upper three carrying an unverified-capacity caveat and a real risk of a slow or failed load at 1.25 GiB free. **Zero tiers should render as `Unavailable` or `NeedsFreeMemory` on this box.** If Phase 1 produces any `Unavailable` here, the mapping is wrong.

**PREDICTION P6 — on an idle machine (12.57 GiB idle budget), all six become `FitsResident`** and the dial becomes genuinely multi-model: `low` = 1B-Q4_K_M, `medium` = 3B-Q4_K_M, `high` = 3B-Q4_K_M + self-critique, `ultra` = 3B-Q8_0 (+ oracle if Phase 5 proves the pair). **Phase 1 must reproduce both P5 and P6 from the same pure function, differing only in the `HardwareProfile` input** — that is test 7d's real purpose.

**DESIGN QUESTION P5 FORCED (now AD-11):** when a tier's intended model is `NeedsFreeMemory` but a smaller model is `Ready`, do we silently substitute the smaller one? **No.** Silent substitution means a user on `ultra` is served a 1B and never knows — the exact dishonesty this repo's evidence culture exists to prevent. See AD-11.

#### Workspace surface — `tool_capable` is evidence-gated, and it bites hard here

`tool_capable` (mod.rs:737) is a field on the **model compatibility row**, not a runtime probe. The rows state it is "earned ONLY by" a committed `camelid.agent_eval/v1` PASS receipt. A GGUF on disk that is not a supported exact row therefore has **no** tool-capable claim.

Exactly **5 rows in the entire contract** carry `tool_capable: true`:

| Row id | Line | On this disk? |
| --- | --- | --- |
| `ornith_1_0_9b_q4_k_m` | 5155 | no |
| `Ornith1.09B` (Q8_0) | 5245 | no |
| **`llama32_3b_instruct_q8_0`** | 5436 | **yes — model #8, 3.187 GiB** |
| `qwen3_4b_instruct_q8_0` | 6477 | no (we have Q4_K_M, not Instruct-Q8_0) |
| **`qwen3_4b_q4_k_m`** | 6528 | **yes — model #7, 2.326 GiB** |

**PREDICTION P7 — Chat sees 6 candidate models, Workspace sees 2.** That is R5, made concrete and large on this exact box. Both tool-capable models are `Unknown` in the current memory state, so every Workspace tier is `Ready{verified:false}` and Workspace becomes a **two-position** dial while Chat is four-position. Phase 1 test 8 must assert exactly this shape.

**PREDICTION P8 — the natural `ultra` pair here is cross-family, which is the useful kind.** On an idle machine Workspace becomes a two-position dial (`low`/`medium` → Qwen3-4B-Q4_K_M; `high`/`ultra` → Llama-3.2-3B-Q8_0 + review), and the only valid oracle pair is **Qwen3-4B-Q4_K_M + Llama-3.2-3B-Q8_0 = 5.513 GiB weights** — the same pair P4 found to be VRAM-plausible. They are different *families* (qwen3 vs llama), so a second opinion between them is genuine cross-architecture diversity rather than one model agreeing with itself. Phase 5/7 should use this pair.

### 10.5 Baselines + P2 falsification test (measured, real release binary)

Binary: `C:\camelid-dial\target\release\camelid.exe`, built from `342b0f058` (`cargo build --release --bin camelid`, EXIT=0). One **server process per model** (repo rule: the unit of sampling is the process, not the request). Prompt sha256 `69d481e4cbd8ae7127a300a1d47807fb2efe8e2217cdfae52c98cd3cadf15c33`, `temperature=0`, `max_tokens=64`, 3 reps. **Free RAM at start: 4.21 GiB** (up from 1.25 GiB — the build had just released ~1.6 GB; see R9 note below).

| Model | tier | predicted fit | load | RSS / peak | backend | reps (ms) | tok/s | text sha16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B-Q4_K_M | `low` | `FitsResident` | **OK** 5,749 ms | 932 / 1,064 MB | `cpu_kquant_block_dot` | 8677 / 8212 / 8034 | 7.38 / 7.79 / 7.97 | `e6cd169da6f22f75` |
| Llama-3.2-3B-Q4_K_M | `medium`/`high` | `Unknown` | **OK** 5,421 ms | 2,085 / 2,447 MB | `cpu_kquant_block_dot` | 23155 / 20876 / 20841 | 2.76 / 3.07 / 3.07 | `e52f19cab4a3f904` |
| Qwen3-4B-Q4_K_M | Workspace | `Unknown` | **OK** 5,581 ms | 2,504 / 2,973 MB | `cpu_kquant_block_dot` | 27937 / 8611 / 6504 | 2.29 / 7.43 / 9.84 | `87409b826c6efdda` |
| Llama-3.2-3B-Q8_0 | `ultra` | `Unknown` | **OK** 13,078 ms | 4,035 / 4,130 MB | `cpu_q8_runtime_repack` | 14929 / 12615 / 13050 | 4.29 / 5.07 / 4.90 | `f0d1d776f0197523` |

**Result 1 — P2 CONFIRMED, not falsified.** All three `Unknown` models loaded and generated. The 3B-Q8_0 reached **4,035 MB RSS against a 3.37 GiB "usable" budget** and worked. `Unknown ⇒ Ready { capacity_verified: false }` is correct; `refuses_load()` is the right predicate to bind to.

**Result 2 — determinism holds.** Identical output sha256 across all 3 reps for every model at `temperature=0`. This is the control that makes a Phase 7 A/B possible at all.

**Result 3 — P4 IS FALSIFIED. Every model ran on CPU: `vram_used_mib = 0` for all four**, on a CUDA-by-default Windows build with an idle RTX 4060 and 7.77 GiB free VRAM. Backends were `cpu_kquant_block_dot` / `cpu_q8_runtime_repack`.

> **SUPERSEDED 2026-08-10 — see §10.12.** The `vram_used_mib = 0` half of this result is **invalid**: VRAM is allocated at first *generation*, and this harness sampled before any completion. The `backend=cpu_*` half is real but **could not be reproduced** — all four models select CUDA resident lanes today. Do not cite Result 3 as evidence that the GPU lane is unavailable.
- `assess` reached its **VRAM** branch (because `has_usable_gpu` is true) and judged capacity against VRAM — but the runtime placed the weights in **host RAM**.
- ⇒ The advisor's verdict is computed against a lane the engine did not take. P4's "dual-model `ultra` is plausible via VRAM" is **not reachable by default** and is withdrawn.
- ⇒ **`assess_pair` (Phase 2) must be HOST-RAM-first**, not VRAM-first, and must not assume GPU placement it cannot confirm. Reopened as OQ-5.

**Result 4 — measured RSS gives `assess_pair` a real basis.** Peak RSS ÷ file size = **1.21× / 1.08× / 1.05× / 1.18×**. The existing `ADVISORY_OVERHEAD_PERCENT = 25` pad is therefore a slightly conservative and *empirically sound* predictor of real memory cost — reuse it rather than inventing a new constant. For the P8 oracle pair (Qwen3-4B + 3B-Q8_0) the combined peak is **2,973 + 4,130 = 7,103 MB ≈ 6.94 GiB**, which needs an idle machine, not this one at 4.21 GiB free.

**Result 5 — the cost of a second pass is the product's central UX risk.** On CPU a 64-token answer takes **8–28 s**. A naive `high` that regenerates a full answer roughly doubles that (16–56 s). **Design consequence: the review pass must be bounded** — review the answer and emit only a verdict-plus-patch, never a full regeneration. Phase 4's contract must state this, and Phase 7 must measure it.

**Result 6 — R9 observed live, twice.** Free RAM moved 1.25 GiB (20:49) → 4.21 GiB (21:07) purely because a compile finished. At 1.25 GiB the advisor called 6 of 8 models `Unknown`; at 4.21 GiB it would call 7 of 8 `FitsResident`. **Tier state genuinely changes with unrelated background activity** — exactly R9, now evidenced rather than predicted.

**Result 7 — warmup is large and model-dependent.** Within one process: Qwen3-4B 27.9 s → 6.5 s (**4.3×**), while the others moved only 1.08–1.14×. Attributable to cold page-cache reads of file-backed weights (§1.12: quantized linears are file-backed, so a "loaded" model is not yet in RAM) and/or the exact-prompt cache. **Phase 7 must use distinct prompts per rep and discard a warmup rep**, or it will measure the cache instead of the feature.

### 10.6 Corrections applied to this plan as a result of Phase 0

| # | Correction |
| --- | --- |
| C1 | **R1 demoted.** The 6 GiB materialisation budget does not count quantized 2-D/3-D linears under stock env (§1.12). The original framing ("the single most important constraint") was wrong. |
| C2 | **R0 added as top risk**, then re-scoped: the error was *re-deriving* availability instead of binding it to the engine's own predicate. |
| C3 | **Availability is tri-state and bound to `FitVerdict::refuses_load()`** (§1.14 invariant). `Unknown ⇒ Ready { capacity_verified: false }` — **measured correct** (Result 1). |
| C4 | **Phase 1 gains tests 7, 7b, 7c (the invariant), 7d** (real host vector reproducing P5 and P6). |
| C5 | **Footprint basis must match `exact_preload_footprint`** (§1.14), not the catalog pad. |
| C6 | Noted (not fixed): `usable_host_ram_bytes` doc-comment contradicts its own code re: the 25 %-of-total floor. Candidate for a separate one-line docs PR. |
| C7 | **P4 withdrawn; `assess_pair` is host-RAM-first** (Result 3). VRAM is not the operative budget because the engine did not use the GPU. **AMENDED 2026-08-10 (§10.12): the engine DOES use the GPU. Host-RAM-first is retained but is necessary-not-sufficient — dual residency is VRAM-bound, and one model already takes 61-82% of VRAM.** |
| C8 | **AD-12 added: the review pass must be bounded** (verdict + patch), never a full regeneration (Result 5). |
| C9 | **R9 promoted from predicted to observed** (Result 6). |
| C10 | **Phase 7 protocol hardened**: distinct prompts per rep + discarded warmup rep (Result 7). |

### 10.7 Phase 0 exit gate

- [x] §1 re-verified — **35/35 claims, 0 drift**; 4/4 absence checks clean.
- [x] Model inventory recorded — 8 GGUFs, exact bytes + sha256, 2 independently cross-checked against pinned contract values.
- [x] Hand-computed tier predictions registered **before** any implementation (P1–P8).
- [x] Baseline single-pass timings recorded on the real release binary.
- [x] **Dual-model `ultra` question answered: NO on this machine as configured.** Arithmetic: the only valid oracle pair (Qwen3-4B-Q4_K_M + Llama-3.2-3B-Q8_0, both `tool_capable`, different families) needs **≈6.94 GiB combined peak RSS**; the host has 15.71 GiB total but was at 4.21 GiB free, and the GPU path — which would have made it comfortable — **is not taken by default** (Result 3). Phase 5 must therefore treat `ultra` dual-model as **experimental and hardware-gated**, never a default.
- [x] Predictions P2 (confirmed) and P4 (falsified) both resolved against measurement, not opinion.

**PHASE 0 COMPLETE.** Ready for Phase 1.
