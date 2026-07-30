# GEMMA3 → METAL GPU-RESIDENT LANE — CONDUCTOR

Campaign branch: `feat/gemma3-metal-resident` (worktree /Volumes/Untitled/Camelid-g3metal), base
`origin/main` @ d9053ec4. Scope target: the promoted `gemma-3-1b-it-Q8_0` row (id
`gemma_3_1b_it_q8_0`, src/api/mod.rs:4044) on the M4 Mac mini (16 GB). All file:line references
are into this checkout at the base commit. Sections 1-6 are the scoping report; the Phase 0
record (decisions + reachability recon, docs-only gate) follows at the end.

---

## 1. Current state

**How gemma3 serves today.** gemma3 is served exclusively through the CPU runnable lane. The serve
router fail-closes it there by architecture name: `is_runnable_serve_arch` matches
`qwen35 | gemma2 | gemma3` (src/api/mod.rs:7163-7165), with the stated rationale that "the optimized
dense binder silently drops gemma3's QK/post norms and has no GeGLU" (src/api/mod.rs:7157-7159).
Chat requests short-circuit to the runnable bridge (src/api/mod.rs:10167-10180); the bridge is
default-on since PR #547 with opt-out `CAMELID_RUNNABLE_SERVE=0` (src/api/mod.rs:7138-7155,
8555-8559). Raw `/v1/completions` fails closed with a typed 422 keyed on the same predicate
(src/api/mod.rs:7471-7513, 9815-9820). The Metal resident lane is therefore unreachable for gemma3
regardless of kernel capability.

**The forward pass.** The only correct gemma3 forward is the runnable lane's generic parametric
pre-norm f32 decoder (src/runnable/model.rs:1, switches set at load, :602-661). Per decode step it
re-dequantizes every layer's seven weight matrices to f32 and walks the 262,144-row tied LM head
row-by-row (src/runnable/model.rs:776-903, :886-891, tied-head fallback :435-445), via sequential
`Mat::matvec` (src/runnable/model.rs:49-57) — the rayon `par_matvec` is qwen35-only (:102-109).

**Measured baseline.** ~0.2 tok/s (~5 s/token) on this M4 mini, recorded in the chat-parity evidence
bundle README (qa/evidence-bundles/gemma3-1b-q8-runnable-serve-chat-parity-20260716-head-6d0d57eb/)
and reflected in the support row (src/api/mod.rs:4036-4083).

**Model shape (anchored to the bundle's dense_metadata,
qa/evidence-bundles/gemma3-1b-q8-runnable-serve-chat-parity-20260716-head-6d0d57eb/api-webui/completion.response.json):**
26 layers, 4 attention heads / 1 KV head (4:1 GQA), head_dim = rope_dim = 256, d_model 1152
(q projection 1152→1024), ffn 6912, vocab 262,144, n_ctx_train 32,768, RMSNorm eps 1e-6, tied LM
head, 999,885,952 params, 183 Q8_0 + 157 F32 tensors.

**Correctness receipts already in hand:**
- HF-transformers greedy parity, `all_greedy_match`, max logit abs diff 1.25e-4
  (qa/runnable/gemma3-parity.json:2-9, test tests/runnable_parity.rs:61-63).
- llama.cpp (pinned acd79d603) chat-parity bundle: 4/5 prompts token-and-text identical at depths
  1/5/50; one disclosed near-tie flip at position 16 of one 50-token leg, 0.3416-nat gap
  (bundle README + manifest).
- Byte-locked chat-template pack (qa/prompt-packs/gemma3-chat-template-shapes-v1.json, in-src lock
  test src/api/mod.rs:6421-6470) and a reusable parity harness (scripts/chat-parity-gemma3.mjs,
  gate pack qa/prompt-packs/gemma3-chat-gate-pack-v1.json).

**Known correctness ceiling of the current lane.** No sliding-window mask exists anywhere on the
gemma3 path; `gemma3.attention.sliding_window=512` is never read (only gemma4 metadata at
src/model.rs:492-493 and the gait inspector src/gait/mod.rs:120 read that key). Sequences at or
beyond 512 tokens are disclosed as "mathematically wrong by construction" in the support row's
blockers (src/api/mod.rs:4051), and the row's `next_step` explicitly demands a window mask or an
optimized-lane gemma3 forward before any ≥512-token context claim (src/api/mod.rs:4083). The Metal
port is that next step.

---

## 2. Gap analysis

Requirement inventory for gemma3-1b vs the Metal GPU-resident lane at d9053ec4. "Generic lane" =
`ResidentDecodeState` + `encode_attention_block`/`encode_ffn_block` driven by
`LlamaInferenceSession`; "gemma4 lane" = the separate opt-in `Gemma4ResidentModel` stack
(src/metal.rs:6543-6735), which is the in-tree template for every gemma-shaped feature.

| # | gemma3 requirement | Metal-lane status | Receipts |
|---|---|---|---|
| 1 | RMSNorm (pre-attn, pre-FFN, final), eps from GGUF | **Supported** | Generic lane norms + GPU LogitsStage final norm (src/inference/metal_resident.rs:334-550) |
| 2 | Per-head QK RMSNorm on Q and K before RoPE | **Supported** | `rms_norm_per_head_f32` kernel (src/metal.rs:1712), wired into decode (src/metal.rs:10069-10104) and prefill (src/metal.rs:11738 area); eligibility comment src/inference.rs:2725-2727; gemma-shape unit test 8x256/8x512 (src/metal.rs:15328-15355) |
| 3 | Post-attention + post-FFN sandwich norms (before each residual add) | **Missing in generic lane; wired in gemma4 lane** | Generic blocks lack them; gemma4 `encode_gemma4_ffn` "adds the extra post_ffw_norm" (src/metal.rs:8248-8253); reference semantics src/runnable/model.rs:304-307, tensor names :597-598 |
| 4 | GeGLU (gelu-tanh(gate) \* up) | **Missing in generic lane; kernel exists** | `encode_ffn_block` hardcodes SiLU (src/metal.rs:9776); `gelu_mul_f32` kernel exists and mirrors the CPU reference exactly (src/metal.rs:1760-1777), wired only into the gemma4 lane |
| 5 | NEOX split-half RoPE pairing | **Partial** | Kernel supports both pairings via runtime flag (`rope_rotate_f32`, src/metal.rs:1813-1843), but `arch_uses_neox_rope_pairing` excludes gemma3 (src/model.rs:812-814, test assert :2124) — host must force pairing=1 as the gemma4 encode does (src/metal.rs:8525) |
| 6 | Per-layer dual RoPE theta (local base 10000 on 5-of-6 layers, global freq_base 1e6 on layers 5/11/17/23) | **Missing in generic lane; template in gemma4 lane** | Generic lane builds ONE cos/sin table per forward and ropes every layer with it (src/metal.rs:11516-11554; documented lane-wide at src/inference.rs:2693-2705; single freq_base at src/inference/rope.rs:574-629). gemma4 threads per-layer tables (src/metal.rs:6982-7009, src/gemma4_runtime.rs:2766-2801). gemma3 needs only TWO tables per token + a per-layer selector — simpler than gemma4. Reference schedule src/runnable/model.rs:607-626 |
| 7 | Sliding-window mask, window 512, 5-of-6 local layers (decode) | **Missing host wiring; kernels ready, zero new MSL** | Every decode-attention kernel takes `kv_base_offset`/strides (v1 src/metal.rs:1872,1879; kv16 :1934,1941; v2 :1994,2004; split-K :2089,2100; and tree twins :4117-4384). Generic path hardcodes offset 0 (src/metal.rs:10008-10012). gemma4 lane implements the window as `position_count = filled - window_start`, `kv_base_offset = window_start * head_dim` (src/metal.rs:8452-8453, scalar write :8537), per-layer `window_start = filled.saturating_sub(window)` (src/gemma4_runtime.rs:2789-2793). Regression test locks kernel reuse at head_dim 256 (src/metal.rs:15362-15456) |
| 8 | Sliding-window in batched prefill | **Missing everywhere (real kernel gap)** | Flash-prefill `kv_base_offset` is a uniform shift (src/metal.rs:3521,3778,3892); causal mask is upper-bound-only (src/metal.rs:3476-3489, :3663) — no per-query-row lower bound exists in any batched prefill kernel. Moot for correctness: see row 10 |
| 9 | Decode attention at head_dim 256, 4:1 GQA | **Supported via v1 kernels only** | v1 `attention_decode_f32`/`_kv16` take arbitrary head_dim + integer GQA (src/metal.rs:1859-1974, kv_head = head/group :1877). v2/split-K hard-capped at head_dim ≤ 128 and the caps are memory-safety-critical (MAX_DPL=4 :2000, sh_acc[NSG\*128] :2038, k_s/v_s[PT\*128] :2118-2119; host gates :9501, :9507-9512). gemma4 lane runs 256/512-dim heads on v1 in production (comment src/metal.rs:8410) |
| 10 | Batched GPU prefill at head_dim 256 | **Missing** | `prefill_tokens` bails on head_dim > 128 (src/metal.rs:11723-11727); flash prefill kernels require ≤ 128 (src/metal.rs:3506, :12327). gemma4's answer: token-by-token prefill through the decode path (src/gemma4_runtime.rs:4919-4949), which gets per-layer windows for free |
| 11 | Speculative batched verify | **Unavailable and must stay gated** | `verify_batch_inner` bails on head_dim > 128 (src/metal.rs:12841-12843); additionally the verify scalar layouts hardcode kv_base_offset=0 (src/metal.rs:9215, 9294), so window-correct verify would need its own plumbing |
| 12 | Embedding scale sqrt(d_model) | **Missing in generic embed path (trivial)** | Reference: src/runnable/model.rs:645-650, applied :704-711, :790-795 |
| 13 | Tied LM head, 262,144-row output GEMV | **Supported** | GPU LogitsStage + Q8_0 wire GEMV handles arbitrary row counts (ragged-row guards src/metal.rs:1154,1180); GPU sampling needs the Q8_0 wire embed table for the gather (src/inference/metal_resident.rs:480-504) |
| 14 | Q8_0 GEMV/GEMM shape fit (k ∈ {1152, 1024, 256, 6912}) | **Supported** | All k dims %32 == 0; production GEMV `q8_0_block_linear_row_ksplit_f32y_wire_nsg8` (src/metal.rs:1124-1186) requires only that |
| 15 | Eligibility dims gate | **Passes** | hidden 1152 %32, q_dim 1024 %32, head_dim 256 even, ffn 6912 %32, 4 % 1 == 0 (src/inference.rs:2883-2892) |
| 16 | No logit softcap / no logit_scale (gemma3 has neither) | **Supported** | Generic lane applies neither; reference src/runnable/model.rs:348-350, :628-630 |
| 17 | GGUF parse of window/pattern/local-base keys | **Missing on the gemma3 path** | Nothing gemma3-side reads `attention.sliding_window`/`sliding_window_pattern`; runnable lane hardcodes pattern=6, local base 10000 (src/runnable/model.rs:607-626). Template: gemma4 parse (src/model.rs:451-460, :486-494) |
| 18 | Optimized dense binder carries gemma3 tensors | **Missing (mis-bound today)** | Binder drops all 104 norm tensors (26× attn_q_norm/attn_k_norm/post_attention_norm/post_ffw_norm), no GeGLU, adjacent even/odd pairing (src/api/mod.rs:7157-7159; src/model.rs:812-814) |
| 19 | Serve routing to the resident lane | **Missing** | `is_runnable_serve_arch` diverts gemma3 before the resident engine is reachable (src/api/mod.rs:7163-7165); resident decode is library-default-OFF, enabled by the CLI fast stack (src/inference.rs:12183-12186; src/main.rs:5721-5735) |
| 20 | Execution-plan row recognition | **Missing** | No gemma3 branch in `recognized_row_level` (src/execution_plan.rs:975-1012); `is_supported_exact_q8_row` (src/execution_plan.rs:1014-1021) gates Metal-resident Q8 plan selection (src/execution_plan.rs:318-320) — without this edit the resident plan is never selected even with all kernels wired |
| 21 | KV cache with per-layer window reads | **Supported by layout; plumbing missing** | Full-length per-layer buffers `[kv_head][max_positions][head_dim]`, doubling growth, no ring buffer needed — window is purely a read-range restriction (src/metal.rs:10904-10957, :11033-11061, ensure_capacity :11096-11184); gemma4 proves per-layer window/head_dim on the same layout (src/metal.rs:6615-6623, 13769-13842) |
| 22 | GPU→CPU KV mirror-back for fallback reads | **Supported** | `ensure_cpu_kv_materialized` (src/inference/metal_resident.rs:208-257) |

Net kernel verdict (adversarially verified): a **correct decode-side gemma3 port requires zero new
MSL kernels**. All gaps are host wiring, estimated ~500-850 LOC across ~6 files (src/metal.rs
~250-400; src/inference/metal_resident.rs ~100-150; src/inference/rope.rs ~40-80; src/model.rs
~40-80; src/inference.rs ~30; src/api/mod.rs ~50-100). The only genuine new-kernel candidate is an
optional batched windowed prefill with head_dim-256 support (~200-400 MSL LOC + host + tuning),
deferrable by adopting gemma4-style token-by-token prefill.

---

## 3. Phase plan

Ordered per this repo's campaign style (qwen3 PRs #266/#275/#278; GABBRO #477-#482): recon →
binder → kernels-with-self-parity → eligibility flip in the same commit → serve reachability →
evidence bundles → promotion surfaces → perf last. Correctness gates precede all perf work.

### Phase 0 — Gates and prerequisites (hours)
Deliverables:
- Reachability recon doc: run the row with `CAMELID_RESIDENT_TRACE=1` and enumerate every bail that
  fires in `resident_decode_eligible` (src/inference.rs:2679-2893). Precedent: WIN2METAL A2
  (9ad5c9e7, docs-only) and GABBRO M1 (2ad74557, evidence-only).
- Lane-architecture decision recorded in the doc: **recommended** — extend the generic lane
  (window param on `encode_attention_block` + per-layer (window_start, theta-table) threading in
  `prepare_token`, GeGLU/post-norm encodes, per gap rows 3-7/12), with gemma4-style token-by-token
  prefill. Alternative (clone `Gemma4ResidentModel` as a gemma3 runtime driver, ~1-2k LOC borrowed
  structure, near-zero metal.rs changes) is acceptable but heavier.
- Scope pin: 1B Q8_0 row only. 4B/12B/27B are out (rope-scaling rejection
  src/runnable/model.rs:395-403; attention-scale coincidence, see Risks).
- Parity-envelope policy decision: whether the GPU lane inherits the disclosed single-flip
  0.3416-nat envelope from the runnable bundle or must match the runnable lane token-exact.
  Gate: doc committed; no code.

### Phase 1 — Config + dense binder (1-2 days)
Deliverables:
- Parse `gemma3.attention.sliding_window`, `sliding_window_pattern`, and the local freq base from
  GGUF (template: gemma4 parse at src/model.rs:451-494), replacing the hardcoded pattern=6 /
  local-base-10000 constants for the resident path (reference src/runnable/model.rs:607-626).
- Teach the optimized dense binder the gemma3 tensor set: attn_q_norm/attn_k_norm,
  post_attention_norm/post_ffw_norm (104 tensors), GeGLU flag, embed scale, forced NEOX pairing,
  per-layer rope-base schedule. Precedent: qwen3 Gate 1 (c8f886f6, tests/model_binding.rs pattern).
Parity gate: binding test asserting all 104 norm tensors bind and the schedule/window metadata
round-trips from the real GGUF.

### Phase 2 — Metal resident forward, correctness-first (2-4 days)
Sub-steps, each landing with its self-parity gate in the same commit (GABBRO M3 pattern, 6b265027):
- 2a QK-norm wiring for gemma3 (mechanical replay of d56f7da2; kernel reuse, ~90-line diff class).
- 2b Sandwich post-norms in the resident attention/FFN blocks (reuse existing RMSNorm encode).
- 2c GeGLU via existing `gelu_mul_f32` (src/metal.rs:1760-1777). Do NOT fuse gate+up — the fused
  variant was previously reverted for register spill (Metal parity campaign).
- 2d Per-layer dual-theta RoPE: build TWO cos/sin tables per token (local 10000 / global 1e6) plus
  a per-layer selector; force split-half pairing host-side (gemma4 precedent src/metal.rs:8525).
- 2e Sliding-window decode mask: add `window_start` to `encode_attention_block`, apply the two-line
  gemma4 math (src/metal.rs:8452-8453), thread per-layer window starts through `prepare_token`.
  Preserve the exact off-by-one convention: window INCLUDES the current position,
  "attend [pos+1-window ..= pos]" (src/model.rs:632-635).
- 2f Embed scale sqrt(1152) on the resident embed gather; verify the GPU sampling gather path.
- Prefill: token-by-token through the decode path (gemma4 precedent src/gemma4_runtime.rs:4919-4949)
  — no new MSL; batched prefill is out of scope here (head_dim 256 exceeds every prefill kernel cap).
Parity gate: per-kernel CPU-vs-GPU self-parity tests (extend the pattern at
src/metal.rs:15362-15456), plus a full-forward logit comparison vs the runnable lane at depths
1/5/50 under 512 tokens.

### Phase 3 — Eligibility, execution plan, serve reachability (~1 day)
Deliverables:
- Flip/condition the gemma3 disqualifiers in `resident_decode_eligible` in the SAME commit that
  completes wiring (d56f7da2 precedent).
- Add a gemma3 branch to `recognized_row_level`/`is_supported_exact_q8_row`
  (src/execution_plan.rs:975-1021) so the Metal-resident Q8 plan is selectable
  (src/execution_plan.rs:318-320). This step is absent from the historical checklist and is
  load-bearing.
- Serve routing: remove gemma3 from `is_runnable_serve_arch` (src/api/mod.rs:7163-7165) with the
  runnable lane as fallback. Handle the side effects explicitly: `/v1/completions` reopens
  automatically (same predicate via `completions_unsupported_for_arch`, src/api/mod.rs:7478); the
  "runnable-runtime" backend/health label and its test (src/api/mod.rs:2571, :15759) and the
  qualified-row comment scope (src/api/mod.rs:17357-17365) go stale.
- Keep speculative decode gated off for gemma3 (naturally blocked by src/metal.rs:12841-12843, but
  add an explicit gate + test given the kv_base_offset=0 hardcodes at src/metal.rs:9215, 9294).
Parity gate: end-to-end serve smoke on the resident lane; runnable fallback verified when
`CAMELID_METAL_RESIDENT_DECODE=0`.

### Phase 4 — Parity receipts and the ≥512-token window claim (1-2 days)
Deliverables:
- Rerun scripts/chat-parity-gemma3.mjs + scripts/raw-decode-parity.mjs on the resident lane at
  depths 1/5/50 vs pinned llama.cpp acd79d603; commit
  qa/evidence-bundles/gemma3-1b-q8-gpu-resident-parity-<date>-<head>/ (README + SHA256SUMS +
  manifest.json + parity json; pattern af928fdd/65772ff2). Pass all four bundle validators
  (check-public-scrub.sh, audit-evidence-bundle-privacy.mjs --strict,
  check-evidence-bundle-checksums.sh, check-public-evidence-claims.mjs — recorded in af928fdd).
- NEW windowed-context receipt: a ≥512-token oracle pack vs llama.cpp (none exists today; bounded
  context ladders were explicitly off for this row). This is the deliverable that unlocks the
  context claim the runnable row could not make (src/api/mod.rs:4083), with the gemma4 lane as an
  internal window-semantics cross-check.
- Determinism receipt: byte-identical decode across two serve sessions.
Parity gate: bundle committed; flips (if any) within the Phase-0 envelope policy.

### Phase 5 — Promotion surfaces (1-2 days)
Deliverables:
- Capabilities row rewrite (src/api/mod.rs:4036-4083: scope, generation_runs, readiness-gate
  string, tested_context, next_step) → MANDATORY ledger regen (scripts/check-ledger-drift.mjs
  Check A; CI gate .github/workflows/ci.yml:146-150) → capabilities test → DECISIONS.md entry.
- Fix the live id mismatch: catalog_id "gemma3_1b_it_q8_0" (src/api/mod.rs:20813) vs row id
  "gemma_3_1b_it_q8_0" (src/api/mod.rs:4044) breaks `filename_is_supported_exact_row`
  (src/api/mod.rs:22103-22111) and drift Check C; precedent c2c33fb5 (qwen3-4B).
- Frontend surfaces (absent from the historical checklist): smoke fixtures
  (frontend/scripts/model-state-smoke.mjs:165-178, capability-readiness-smoke.mjs,
  model-lanes-smoke.mjs — all CI-run at .github/workflows/ci.yml:362-406,428), catalog entry
  (frontend/src/lib/supportedModels.js; gemma4 precedent at :89-116), and executionPlan.js backend
  sets if a new backend label is minted. Use real /api/capabilities row ids — the lane-gate
  fixture-drift incident came from fixtures diverging from production ids.
- Docs sweep across README.md, COMPATIBILITY.md, SUPPORT_MATRIX_v0.1.md, STATUS.md,
  CAPABILITY_MATRIX.md, DOCS.md + architecture note, honoring drift Checks D/E (full-sha256
  mentions must state the ledger sha b205840c…; new bundle indexed ONLY in STATUS.md). Do the
  sweep twice (GABBRO needed 1df8c791 as a second pass).
- Validate with `cargo test --all-targets` (integration tests in tests/ are missed by --lib/--bins).
Parity gate: CI green including ledger schema/drift, scrub, frontend smokes.

### Phase 6 — Performance (after all correctness gates; 2-5 days, open-ended)
Deliverables, in order of expected yield:
- Measured baseline of the correct resident lane (decode tok/s at depths 64/512/2048; prefill
  tok/s), plus a same-hardware llama.cpp Metal measurement for a neutral comparison.
- Free win already banked: window masking caps the v1 kernel's serial position walk at 512 for
  20 of 26 layers, so depth scaling improves vs full-causal.
- Token-by-token prefill cost assessment; if prompts dominate, the batched windowed prefill kernel
  with head_dim-256 support (~200-400 MSL + host + tuning) is the single largest kernel item.
- Decode-attention utilization: at 4 heads the v1 dispatch is 4 threadgroups × 32 lanes with each
  lane streaming 1 KB K rows — a 256-dim-capable fast variant (wider staging or dims-per-lane
  restructure of v2/split-K) is real kernel work with self-parity gates; never relax the ≤128 host
  gates without it (they prevent out-of-bounds writes).
Parity gate for every perf commit: byte-identical decode vs the Phase-4 receipts.

Total estimate: ~8-14 working days to a promoted, receipt-backed resident row (Phases 0-5), with
Phase 6 perf work incremental on top.

---

## 4. Expected performance

- **Baseline (measured):** ~0.2 tok/s (~5 s/token), runnable CPU lane, f32 with per-token
  re-dequantization and a row-walked 262k tied head (bundle README; src/runnable/model.rs:776-903).
- **Roofline ceiling:** decode is weight-bandwidth-bound. 999,885,952 params at Q8_0 wire density
  (34 bytes / 32 weights = 1.0625 B/param) ≈ 1.06 GB of weight traffic per token, tied head
  included; KV traffic is negligible beside it at ≤512-window depths. At the M4's ~120 GB/s
  unified-memory bandwidth the ceiling is ≈ 110 tok/s.
- **Reference points:** no in-repo measured Metal number exists for gemma3 (the lane does not run
  it today). The nearest structural precedent is the gemma4 resident lane, which runs the same v1
  attention kernel at head_dim 256/512 in production (src/metal.rs:8410). A same-hardware llama.cpp
  Metal measurement should be taken in Phase 6 as the neutral external reference.
- **Defensible target range: 25-60 tok/s decode** (roughly 125-300x over baseline). The discount
  from the ~110 tok/s roofline reflects: v1 attention underutilization at 4 heads (4 threadgroups
  of 32 lanes, uncoalesced 1 KB K-row streams per lane), per-token encoder overhead at 26 layers,
  and imperfect GEMV bandwidth utilization on the 262k-row head. Anything in this range is
  transformative vs 0.2 tok/s; the upper half likely requires the Phase 6 attention work.
- **Prefill:** token-by-token prefill runs at roughly decode speed, so long prompts will be
  noticeably slower than batched-prefill architectures until the optional head_dim-256 windowed
  flash prefill kernel lands (gap row 8/10).

---

## 5. Risks & landmines

1. **head_dim 256 excludes every fast kernel, and the ≤128 caps are memory-safety-critical.**
   v2/split-K would corrupt memory at 256 (fixed per-lane arrays MAX_DPL=4 src/metal.rs:2000,
   128-float staging :2038, :2118-2119); the host gates (:9501, :9507-9512, :11723-11727,
   :12841-12843) are what prevent it. Mitigation: ship on v1 (proven at 256 by the gemma4 lane),
   never relax gates without new 256-dim variants carrying self-parity tests.
2. **No batched GPU prefill at head_dim 256.** Mitigation: gemma4-style token-by-token prefill for
   correctness (src/gemma4_runtime.rs:4919-4949); treat the batched windowed prefill kernel as a
   scoped Phase 6 item, not a blocker.
3. **Window semantics have no in-repo oracle at ≥512 tokens.** The runnable reference lane has no
   mask at all (src/api/mod.rs:4051), and the off-by-one convention (window includes current
   position, src/model.rs:632-635) is easy to get subtly wrong. Mitigation: llama.cpp ≥512-token
   parity pack as the gate (Phase 4), gemma4 lane as internal cross-check, reuse the exact
   window_start math (src/metal.rs:8452-8453).
4. **Speculative-verify paths hardcode kv_base_offset=0** (src/metal.rs:9215, 9294) even though the
   tree kernels accept the offset. Mitigation: explicit gemma3 spec-decode gate + test in Phase 3
   (head_dim already blocks it, but belt-and-braces against future kernel work).
5. **Hardcoded schedule constants and the attention-scale coincidence.** pattern=6 / local base
   10000 are hardcoded (src/runnable/model.rs:607-626) and 1/sqrt(head_dim) equals gemma3-1B's
   query_pre_attn_scalar only because head_dim=256; larger sizes differ, and 4B+ carry rope scaling
   the loader rejects (src/runnable/model.rs:395-403). Mitigation: parse the GGUF keys in Phase 1;
   pin scope to the 1B row; re-scope explicitly before any multi-size claim.
6. **Promotion-surface landmines (each has bitten before):** execution-plan row gate missing
   (src/execution_plan.rs:975-1021 — resident plan never selected without it); live
   catalog_id/row-id mismatch (src/api/mod.rs:20813 vs :4044, precedent c2c33fb5); mandatory ledger
   regen on readiness-gate string edits (GABBRO fix bb8ee0f2); frontend fixtures must use real
   /api/capabilities ids (lane-gate fixture-drift incident); /v1/completions silently reopens when
   gemma3 leaves the runnable predicate (src/api/mod.rs:7478); docs drift Checks D/E wording rules.
   Mitigation: all are enumerated as explicit Phase 3/5 deliverables above.
7. **GeGLU fusion regression precedent.** The fused gate+up kernel was reverted for register spill.
   Mitigation: separate gate/up GEMVs + `gelu_mul_f32`, matching the gemma4 lane.
8. **Parity-envelope ambiguity.** The existing receipt carries one disclosed 0.3416-nat near-tie
   flip; without a pinned policy, the GPU lane's comparison result is unadjudicable. Mitigation:
   Phase 0 policy decision, recorded before any comparison runs.

---

## 6. Go/No-Go recommendation

**GO**, scoped to the gemma-3-1b-it-Q8_0 row, decode-side first with token-by-token prefill.
Verified kernel-level analysis shows a correct port needs zero new MSL — every gemma3 requirement
(QK-norm, sandwich norms, GeGLU, dual-theta RoPE, kv_base_offset windowing, head_dim-256 v1
attention) already exists in-tree with the gemma4 lane as a production-proven template, leaving
~500-850 LOC of host wiring plus the standard promotion overhead. The payoff is large and
receipt-backed on both ends: a measured 0.2 tok/s baseline, a ~110 tok/s roofline, a reusable
llama.cpp parity harness, and the port simultaneously retires the row's disclosed ≥512-token
correctness blocker by adding the sliding-window mask the CPU lane never had.

---

## 7. Phase 0 record (2026-07-29)

Docs-only gate, executed on this branch at base d9053ec4 with the prebuilt release binary of the
same main commit. Recon model file: the desktop app's `gemma-3-1b-it-Q8_0.gguf`
(1,069,306,368 bytes). No engine code was changed; no server was started (the recon vehicle is
the one-shot `bench-generate` subcommand).

### 7a. Reachability recon results

Mechanism verified before running: `CAMELID_RESIDENT_TRACE` (any value) makes
`resident_decode_eligible` print `[resident-eligible] no: <gate>` on stderr for the first gate
that declines (bail macro, src/inference.rs:2680-2689). The trace is evaluated inside the
library, but the resident lane itself is CLI-armed: `apply_default_fast_stack`
(src/main.rs:5721-5735) sets `CAMELID_METAL_RESIDENT_DECODE=1` (plus wire/NSG8/attn2/prefill/MM
defaults) for every non-deterministic subcommand, so `bench-generate` is a faithful direct-session
probe of the lane.

**Run 1 — default fast stack, trace on** (6-token greedy prompt, 8 generated tokens):

- **Zero eligibility bails fired.** Full trace-relevant stderr, verbatim:
  - `[resident-dispatch] cuda_enabled=false metal_enabled=true` (src/inference.rs:3611-3621)
  - `[resident] pos=5 layers=26 ...` through `[resident] pos=12 layers=26 ...` — the generic
    Metal resident decode lane admitted gemma3-1b and decoded every generated token on the GPU.
- Measured: 38.09 tok/s decode at trivial depth (positions 5-12), peak RSS 0.57 GB (wire pages).
- Output is garbage, as the gap analysis predicts for the mis-bound forward:
  `讖Compliance по bowels по切りごφό` (token ids 251392, 70408, 1311, 143805, 1311, 49874,
  237790, 137586).
- Every gate in `resident_decode_eligible` was walked without firing: session disable
  (src/inference.rs:2690), NoPE (:2700), runnable-tier parity verdict (:2711 — vacuous here: no
  cache key is set on this path, and the GPU-runnable tier is CUDA-only with gemma3 excluded by
  `is_gpu_runnable_arch`, src/execution_plan.rs:837-848), backend-enabled (:2716), MoE (:2719),
  logit_scale (:2722), diagnostic defaults (:2728-2732), the per-layer Q8_0 loop (:2828-2857; all
  26 layers carry wire-page Q8_0 projections, no attention biases), tied output projection
  (:2864-2866), output_norm dim (:2868-2879), and the dims gate (:2883-2892; 1152%32, q_dim
  1024%32, head_dim 256 even, 6912%32, 4%1).

**Run 2 — control, resident decode+prefill forced off** (`CAMELID_METAL_RESIDENT_DECODE=0`,
`CAMELID_METAL_RESIDENT_PREFILL=0`): confirms the trace mechanism works. One disqualifier fired,
15 times (once per prefill/decode/speculative eligibility call), verbatim:

> `[resident-eligible] no: neither CAMELID_METAL_RESIDENT_DECODE nor CAMELID_CUDA_RESIDENT_DECODE enabled`

emitted from src/inference.rs:2716-2717. CPU-lane output is also garbage
(`讖Compliance по切り میر마다 ラя`; ids 251392, 70408, 1311, 49874, 43344, 108003, 37646,
236895) and diverges from the GPU lane at generated index 3 — the wrongness is binder-level and
shared by both lanes, with lane-numeric drift on top of the already-wrong graph. Confirmed
in-source: the dense binder classifies gemma3 as neither `expects_qk_norm` nor `forbids_qk_norm`
(src/model.rs:1008-1009), so `attn_q_norm`/`attn_k_norm` bind `(None, None)` silently and the
post_attention/post_ffw sandwich norms are never requested — the mis-binding disclosed at
src/api/mod.rs:7157-7159 is live, not hypothetical.

**The one resident-side decline that did fire was silent.** Batched GPU prefill declined at
head_dim 256: `try_metal_resident_prefill` (src/inference/metal_resident.rs:67) passes
eligibility (:75), then `prefill_tokens` returns `None` on its guard `self.head_dim > 128`
(src/metal.rs:11714-11731, offending term :11727), and the host returns `Ok(false)` with no
trace line (src/inference/metal_resident.rs:154-159). Evidence: resident decode telemetry starts
at pos=5 (the last prompt token), so positions 0-4 were CPU-prefilled. This is gap rows 8/10
observed live, and it is invisible to `CAMELID_RESIDENT_TRACE`.

**Consequences recorded for the plan (amendments to sections 1-3):**

1. Section 1's "the Metal resident lane is therefore unreachable for gemma3" is true for
   **serve only**. On the CLI direct-session path (bench-generate today; any future non-serve
   session running the default fast stack), reachability is already OPEN: nothing stands between
   gemma3 and a mathematically wrong resident forward, and it decodes silently at speed. The
   serve router divert (src/api/mod.rs:7163-7165) is the only correctness guard in production.
2. Phase 3's "flip/condition the gemma3 disqualifiers in `resident_decode_eligible`" has nothing
   to flip — **no gemma3 disqualifier exists**. The work is inverted: Phase 1 must ADD a
   fail-closed arch-keyed disqualifier (gemma3 declines the resident path until the wiring is
   complete), and Phase 3 removes it in the same commit that lands the last correctness encode.
   This closes the silent-garbage CLI path for the duration of the campaign instead of only at
   its end.
3. The execution plan's fail-closed safe path does not protect this: gemma3 gets the
   "non-validated row or quant" safe plan, but plan `env_updates` never unset the CLI fast-stack
   variables, so the resident lane still engages. Also noted in passing: the startup line
   `[hw] GPU: none detected — CPU backend is the inference path` printed while the Metal lane
   decoded every token; the hardware-probe log line is CUDA-oriented and cosmetically wrong on
   this path (not a Phase 0 work item).
4. Corroborating perf datum: the (incorrect) resident forward at head_dim 256 on the v1 kernels
   already sustains ~38 tok/s at trivial depth on this M4, consistent with the section 4 target
   range (25-60 tok/s) for the corrected lane once the added encodes (QK-norm, sandwich norms,
   GeGLU, dual-theta RoPE, window) take their share.

### 7b. Lane-architecture decision

**DECIDED: extend the generic resident lane.** Concretely: a window parameter on
`encode_attention_block` plus per-layer `(window_start, theta-table)` threading in
`prepare_token`, GeGLU and sandwich post-norm encodes in the generic blocks, forced split-half
RoPE pairing host-side, and the embed scale on the resident gather — per gap rows 3-7/12 — with
gemma4-style token-by-token prefill through the decode path (no new MSL, per gap rows 8/10).

Alternative considered and **rejected for weight**: cloning `Gemma4ResidentModel`
(src/metal.rs:6543-6735) as a standalone gemma3 runtime driver. It would borrow ~1-2k LOC of
structure and leave metal.rs nearly untouched, but it duplicates the generic lane's
session/KV/dispatch machinery for one row, doubles the surface future kernel work must keep in
parity, and forfeits the generic lane's existing self-parity test pattern — the per-feature
deltas on the generic lane are each small, kernel-reusing, and individually gateable.

### 7c. Scope pin

**gemma-3-1b-it-Q8_0 row only.** 4B/12B/27B are explicitly OUT pending a re-scope: the loader
rejects their rope scaling (src/runnable/model.rs:395-403), and 1/sqrt(head_dim) equals gemma3's
query_pre_attn_scalar only at the 1B's head_dim=256 — the larger sizes break that coincidence
(risk 5). No multi-size claim, docs row, or fixture may reference them without a new scoping
pass.

### 7d. Parity-envelope policy

**The GPU lane inherits the existing receipt's envelope.** Token-exact vs the reference is the
target; disclosed near-ties are the tolerance. Specifically: near-tie flips are permitted only
if disclosed in the evidence-bundle README with their measured nat gap (precedent: the runnable
bundle's single position-16 flip at 0.3416 nats); the new ≥512-token windowed pack must be clean
— any flip in it is individually adjudicated before the bundle lands rather than waved through
under the envelope; and no undisclosed divergence of any size is acceptable. This pins the
Phase 4 adjudication rule before any comparison runs (risk 8).

---

## 8. Phase 1 record (2026-07-29)

Landed on this branch after rebasing onto origin/main @ bce31c2c (clean rebase; PR #553 merged
the fail-closed CLI/resident guard with the shared `model::is_runnable_only_arch` predicate and
the new `LlamaModelConfig.architecture` field — that PR IS amendment §7a-2's "Phase 1 must ADD a
fail-closed disqualifier" deliverable, landed ahead of this branch, so Phase 1 here keeps and
tests it rather than re-adding it).

**Config metadata (gap row 17).** New `Gemma3Metadata` on `LlamaModelConfig` (`config.gemma3`,
parsed in `from_gguf` for gemma3 only; src/model.rs): `sliding_window`,
`sliding_window_pattern`, `rope_freq_base_global`, `rope_freq_base_local`,
`layer_is_sliding` (schedule: layer i global iff (i+1) % pattern == 0 — NO forced-global final
layer, unlike gemma4), `embed_scale` = sqrt(d_model), `ffn_geglu`, `rope_neox_pairing`, plus
accessors `is_sliding_layer`/`rope_freq_base_at`/`layer_window` (window INCLUDES the current
position, same convention as `Gemma4LayerPlan::window`). Phase 2 consumes this struct for the
resident encodes.

**Key-name verification finding (deviation from the section-3 sketch).** The real row
(gemma-3-1b-it-Q8_0.gguf, 38 metadata keys dumped raw) carries ONLY two window/rope keys:
`gemma3.attention.sliding_window = 512` (u32) and `gemma3.rope.freq_base = 1e6` (f32). There is
NO sliding-window-pattern key and NO local-freq-base key in the file — no gemma3 conversion
writes them; the reference implementations hardcode pattern 6 / local base 10000 (the same
no-GGUF-key situation as smollm3's `no_rope_layer_step`). Resolution, honoring "no silent
defaults" as far as the file format allows: the two keys that exist are REQUIRED (absent or
malformed → typed parse error; a gemma3 GGUF without `attention.sliding_window` or
`rope.freq_base` no longer loads anywhere, including the runnable lane, which shares
`from_gguf`); pattern and local base are reference-pinned constants
(`Gemma3Metadata::REFERENCE_SLIDING_WINDOW_PATTERN = 6`,
`REFERENCE_LOCAL_ROPE_FREQ_BASE = 10000.0`) disclosed in the struct docs, with explicit
override keys (`gemma3.attention.sliding_window_pattern` scalar,
`gemma3.rope.freq_base_swa`) honored if present and hard-erroring if present-but-malformed —
never a silent fallback over an explicit key. The runnable lane's hardcoded schedule
(src/runnable/model.rs:607-626) is untouched; it remains the CPU reference.

**Dense binder (gap rows 5/6/18).** gemma3 moved from unclassified to `expects_qk_norm`
(alongside qwen3/command-r, with the key_length==value_length gate now arch-labeled), and a new
`expects_post_norms` (gemma3-only) requirement binds `post_attention_norm`/`post_ffw_norm` —
new `Option` fields on `LlamaLayerTensors`, shape-validated `[embedding_length]` as a
must-be-paired set. All 26×4 = 104 norm tensors now bind non-None from the real file, and a
gemma3 row missing ANY of the four fails closed at bind — mis-binding to `(None, None)` is
impossible. `arch_uses_neox_rope_pairing` (and `LlamaModelConfig::rope_neox_pairing`) are
deliberately UNCHANGED for gemma3 per the §7b lane decision: the resident lane forces
split-half pairing host-side in Phase 2 from `Gemma3Metadata.rope_neox_pairing` (gemma4-encode
precedent), leaving the guarded-off CPU dense path unperturbed.

**Safety invariant (unchanged, now co-tested with binding).** `is_runnable_only_arch` still
matches gemma3; the serve divert, the CLI direct-session guard, and the resident-eligibility
arch disqualifier (src/inference.rs, PR #553) are untouched, and PR #553's
`runnable_only_arch_disqualifies_the_resident_gpu_path` still passes. The new binding tests
additionally assert the predicate fires AFTER a successful bind — tensors available, lanes
unreachable. No serve routing, gemma2/qwen35, or Metal encode change (Phase 2/3 scope).

**Parity gate tests** (tests/model_binding.rs; real-row test env-keyed on `CAMELID_GEMMA3_GGUF`
per the `CAMELID_GEMMA4_GGUF` convention, run PASS against the real file):
- `gemma3_real_row_binds_all_104_norm_tensors_and_window_schedule` — (a) 104/104 norms bind
  with real shapes ([256] QK, [1152] sandwich), (b) window 512 / pattern 6 / globals at
  5/11/17/23 / layer 25 local / local 10000 / global 1e6 round-trip, (c) GeGLU + sqrt(1152)
  embed scale + pairing flags set, plus the guard-still-fires assertion.
- `gemma3_binds_qk_and_sandwich_norms_with_window_metadata` (synthetic twin),
  `gemma3_without_qk_norm_fails_closed`, `gemma3_without_sandwich_norms_fails_closed`,
  `gemma3_without_sliding_window_key_fails_closed`,
  `gemma3_explicit_pattern_and_local_base_keys_override_reference_constants`,
  `gemma3_malformed_pattern_key_fails_closed`.
- `model::gemma3_tests::one_b_schedule_globals_at_5_11_17_23_and_no_forced_global_final_layer`
  (unit, schedule/accessor semantics).

Gates: cargo fmt clean, clippy --all-targets -D warnings clean, cargo test --all-targets green
(with the real-row test exercised via CAMELID_GEMMA3_GGUF), check-public-scrub.sh clean.

### 8a. Phase 1b review record (2026-07-29)

Five confirmed adversarial-review findings against the Phase 1 landing, all fixed in one
commit on this branch:

**R1 (major) — swapped sandwich-norm bindings were test-invisible.** `post_attention_norm`
and `post_ffw_norm` are both `[1152]` and every Phase 1 test asserted only `.dimensions`, so
transposing the `find_tensor` lookups passed the whole suite. Fixed by NAME-pinning: the
synthetic binding test and the real-row test now assert the bound descriptor's `.name`
(`blk.{i}.post_attention_norm.weight` on the `post_attention_norm` field, and likewise for
`post_ffw_norm` and — same blindness, both `[256]` — `attn_q_norm`/`attn_k_norm`) on every
layer. Verified by temporarily transposing the lookups (sandwich pair AND QK pair): both the
synthetic and the real-row test fail on the name assertion in each case; restored, both green.

**R2 (major) — schedule derivation was only CI-exercised at block_count=1, and the 26-layer
unit test duplicated the production expression (tautology).** The fixture writer now takes a
`block_count` option (per-block tensors follow it), and two new `from_gguf`-driven tests
assert THROUGH the parsed metadata's accessors with literal expected lists (never the
`(i+1) % pattern` formula): (a) 26 layers, no override → globals exactly at 5/11/17/23,
layer 25 local; (b) 12 layers with an explicit `sliding_window_pattern = 4` override →
globals at 3/7/11, plus a `freq_base_swa = 50000` override reaching `rope_freq_base_at` —
proving the resolved (possibly overridden) pattern, not `REFERENCE_SLIDING_WINDOW_PATTERN`,
drives the derivation. The unit test's hand-built fixture now uses a literal 26-entry
schedule list instead of the formula; it remains the accessor-semantics test.

**R3 (minor, design) — override keys honored by `Gemma3Metadata` but not by the runnable
lane.** CHOICE: single source of truth (the preferred option), not the fail-closed fallback.
The runnable lane's hardcoded pattern-6/local-10000 schedule (src/runnable/model.rs) now
derives `layer_rope_base` from the SAME parsed `Gemma3Metadata` (`cfg.gemma3`, shared
`from_gguf`) via `rope_freq_base_at`, so an override-carrying row can no longer make the
runnable (CPU parity oracle) and resident lanes compute different schedules for one file.
Bit-identity for the real 1B row proven by a new env-gated test
(`runnable::model::gemma3_schedule_tests::gemma3_real_row_runnable_rope_schedule_is_the_reference_schedule`,
literal expected base list + forward-logits fingerprint over a short prompt): fingerprint
`sum_bits=0x0002eec61740012f` identical before and after the rewiring, and the Phase 1
real-row binding/schedule test still passes. Note the runnable lane still implements no
window mask (documented full-support blocker) — R3 unifies the schedule/rope-base inputs,
not the mask.

**R4 (minor) — stale "silently drops the norms" safety comments.** With Phase 1 binding the
norms (and the dense forward path applying QK norms where bound), five comments describing
the pre-Phase-1 binder as present-tense fact were rewritten to the current rationale (binds,
but does not APPLY the sandwich norms; no GeGLU, dual-theta RoPE, or sliding-window mask;
gemma2's sandwich norms still dropped at bind): src/model.rs (`is_runnable_only_arch` doc),
src/inference/tests.rs (resident-disqualifier test doc), src/main.rs
(`ensure_arch_has_direct_dense_session` doc + its bail message), src/api/mod.rs (M-A1 compat
row comment, `is_runnable_serve_arch` doc). A grep sweep caught three more the list missed:
src/inference.rs (resident disqualifier comment + bail message), src/model.rs
(`runnable_only_arch_set_is_exactly_the_serve_bridge_set` comment), src/api/mod.rs
(`completions_unsupported_for_arch` doc).

**R5 (minor) — untested fail-closed branches in `Gemma3Metadata::from_gguf`.** Fixture
options added for each; five new tests assert the typed `InvalidModelMetadata` error (not a
silent reference-constant fallback): `sliding_window == 0`, missing `rope.freq_base`,
non-positive `rope.freq_base`, wrong-typed `rope.freq_base_swa`, non-positive
`rope.freq_base_swa`.

Gates re-run after the fixes: cargo fmt clean, clippy --all-targets -D warnings clean,
cargo test --all-targets green, real-row + runnable-schedule tests green under
CAMELID_GEMMA3_GGUF, check-public-scrub.sh clean.
