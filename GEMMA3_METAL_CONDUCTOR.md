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

---

## 9. Phase 2 record (2026-07-30)

Phase 2 landed the correctness-first Metal resident forward for gemma3 in the six sub-steps
sketched in §3, each with its self-parity gate in the same commit (`8b9247d1` 2a QK-norm,
`94ae0263` 2b sandwich post-norms, `55a2e961` 2c GeGLU, `8c476e45` 2d dual-theta RoPE schedule,
`9dad6544` 2e sliding-window decode mask, `462bedec` 2f embed scale), then merged `origin/main`
@ `e28f0f76` underneath them and re-proved the whole stack against the real row. The lane
decision from §7b held: zero new MSL kernels, all host wiring in the generic resident lane.

### 9a. The merge (`origin/main` @ e28f0f76 → the branch)

Merged, not rebased. Five of the six Phase 2 commits touch the same hunks in src/metal.rs, so a
rebase would have replayed the same three-way weave five times against five different
intermediate states; a merge resolves each region once against the final state. Nine conflict
regions in exactly two files (src/metal.rs ×7, src/inference/metal_resident.rs ×2). Main's side
is PR #556 (`ResidentLinearWeight` GEMV dispatch, Q8/F16 primary KV formats, format-dispatched
embed gather, appliance-mode encode-ahead gating) plus PR #557 (prompt-prefix cache).

Three regions have a resolution that **compiles and is silently wrong**, and they are the reason
this is recorded rather than left in the commit message:

1. **FFN f32y GEMV.** Keeping the campaign's `encode_q8_matmul_f32y` and satisfying the type
   checker with `&gate_w.buffer` compiles and pushes Q4_K/Q6_K FFN weights through the Q8_0
   GEMV — garbage on exactly the K-quant rows #556 exists to serve. Resolution takes main's
   `encode_resident_matmul_f32` call *shape* and appends the campaign's GeGLU/SiLU
   `act_pipeline` binding.
2. **Attention scalar byte 28 (`kv_base_offset`).** After #556 the shared encode computes a
   conditional `kv_position_stride` — BYTES on the Q8 primary, elements on f32/f16 — and
   `kv_base_offset` shares those units. Neither side's text is acceptable: main pins byte 28 to
   0, which does not merely revert to full-causal (the caller still passes the narrowed
   `position_count`, so the kernel reads the OLDEST rows and never the current position); the
   campaign reverts bytes 20/24 to element strides, breaking the Q8 primary KV lane for every
   `head_dim <= 128` row. The correct weave is `window_start * kv_position_stride`.
3. **`prepare_token` gather scalar.** The 8-byte allocation and the shader's `buffer(4)` read
   auto-merge from the campaign, and `pool_get` classes by `bytes.max(32).next_power_of_two()`
   and never zeroes, so main's format-derived bytes-per-row alone leaves bytes 4..8 unwritten
   while the kernel reads them — every GPU-sampled token's embedding multiplied by a recycled
   stale float, on ALL resident rows, not just gemma3. The weave writes both fields.

Two more were loud-but-easy to get wrong: `ResidentDecodeState::new` needed BOTH prologues
(main's text alone silently deletes the fail-closed schedule-length check, turning a clean
`None` into an out-of-bounds panic on the first decode token), and the decode encode-ahead
tables had to be NESTED inside `resident_encode_ahead_enabled` rather than replaced (taking the
campaign hunk wholesale compiles and reinstates unconditional encode-ahead, undoing #556's
cooperative-batching head-of-line-blocking fix).

**Hardening taken while the attention region was open (was §5 landmine material, now closed):**
`encode_attention_block` no longer re-reads the process-global KV-format gates. It takes the
session's `kv16`/`kvq8` as parameters, because the call site, the KV readback and the KV seed
already use the per-session fields and the window offset now rides in the same scalar. The three
standalone (non-session) helpers pass the globals, so their behaviour is byte-identical. Residual,
recorded: the inner `encode_attention` helper still selects its pipeline from the globals — it is
shared with the gemma4 lane and the speculative-verify path, so threading it is a separate change.

### 9b. Phase 2 final gate — real-row parity

`gemma3_real_row_resident_forward_matches_runnable_oracle` drives the resident machinery directly
(the production arch disqualifier stays up until Phase 3) with every Phase 2 encode live, and
requires a token-identical greedy continuation to the runnable lane — the CPU oracle pinned to HF
transformers by qa/runnable/gemma3-parity.json. Run targeted and in `--release` with the
production GEMV configuration (f32y + wire + NSG8), because those gates are process-latched and a
full `--lib` run silently SKIPs the test:

```
CAMELID_METAL_F32Y=1 CAMELID_METAL_WIRE=1 CAMELID_METAL_WIRE_NSG8=1 \
CAMELID_GEMMA3_GGUF=<gemma-3-1b-it-Q8_0.gguf> \
  cargo test --release --lib gemma3_real_row_resident_forward -- --nocapture
```

Measured on the M4 mini against the real `gemma-3-1b-it-Q8_0` row, no SKIP line, 590.94 s:

| depth | resident argmax | oracle argmax | max abs logit diff |
|---|---|---|---|
| 1 | 108 | 108 | 6.247e-5 |
| 5 | 1077 | 1077 | 7.820e-5 |
| 50 | 578 | 578 | 9.584e-5 |

**PASS: 50/50 greedy tokens identical; overall max abs logit diff 2.122e-4.** Zero flips, so the
§7d envelope is not drawn on at all — this receipt is clean, not disclosed-flip. Per §5 landmine
below, the slot count matters: this run is the `active_slots <= 1` equivalent with encode-ahead
OFF (the test passes `next_rope: None`), and 9c's MR2 regression proves encode-ahead ON is
bit-identical to it.

The gate asserts `total < 512`, so the whole comparison sits inside the window and it cannot
distinguish a correct window base from one pinned to 0. That is 9c's job.

**Landmine found while re-applying this test: it must not arm its own gates.** As written it
opened with `std::env::set_var("CAMELID_METAL_F32Y"/"..._WIRE"/"..._WIRE_NSG8", "1")` *before*
its own SKIP checks, so it mutated the process environment on every run, including a plain
`cargo test` with no gemma3 GGUF present. Those gates are process-wide `OnceLock`s read by every
other Metal test in the binary; whichever siblings had not read them yet then latched onto the
wire path, where the standalone block helpers' 36-byte uploads are read as 34-byte wire blocks
and come back NaN. Measured: `cargo test --lib metal::tests` is green twice at the merge commit
and fails five resident/standalone tests with the gate test added; a full `cargo test
--all-targets` failed those five plus `metal_gemma4_layer_matches_cpu`. Whether a given sibling
is hit is a race, so an earlier full run happening to pass proves nothing. The test now checks
`CAMELID_GEMMA3_GGUF` first, never sets the gates, and SKIPs with the full armed invocation in
its message. **Three pre-existing tests on main still use the in-test `set_var` pattern**
(`metal_verify_gemv_batched_bit_identical`, `metal_spec_verify_bit_identical`,
`metal_tree_verify_bit_identical`) — same hazard, untouched here, worth the same treatment.

### 9c. Post-merge regressions added

- `metal_resident_window_start_beyond_512_matches_seeded_window_oracle` — pins a windowed decode
  at `filled` 256/512/513/561/600 (the row's real window of 512, `window_start` up to 88)
  bit-for-bit against a full-causal oracle seeded with exactly the window's rows. head_dim 64 is
  deliberate: it admits the v2/split-K attention geometry the 1B's head_dim 256 can never reach,
  and it makes the Q8 primary KV reachable, where `kv_base_offset` is a byte offset. Verified
  sensitive in both directions by temporarily breaking the packed word — byte 28 pinned to 0
  fails this test AND the Phase 2e self-parity; element units fail this test under
  `CAMELID_METAL_KV_DTYPE=q8`. Green on all three primaries (f32 default, q8, f16).
  The history is SEEDED rather than decoded, and that is load-bearing: the first cut walked 600
  real tokens and destabilised the whole suite. `MetalLinearKernel` owns ONE shared serial
  command queue (the `Drop` impl for `ResidentDecodeState` already warns that a gated pending
  graph "would block every future commit on the shared serial queue"), so a test holding it for
  hundreds of gated command buffers starves the others — observed as unrelated one-dispatch
  kernel tests (`metal_rms_norm_matches_cpu`, `metal_silu_mul_matches_cpu`,
  `metal_rope_rotate_matches_reference`, `metal_soft_cap_matches_cpu`, `metal_residual_add…`)
  returning their untouched input, and as NaN in the resident/standalone comparisons. Two
  command buffers per depth is the same proof at 1/300th of the occupancy (0.34 s vs 10 s).
  **Rule for future Metal tests on this lane: budget command buffers, not wall time.**
- `metal_resident_gemma3_decode_is_identical_with_encode_ahead_off` — 12/12 tokens bit-identical
  on a gemma3-shaped session (dual theta + sliding window + QK/sandwich norms) with the next
  token's tables supplied and withheld. This is the claim that makes the appliance-mode
  `(None, None)` arm safe, and it is the coverage the Phase 2 gate cannot give (that test is
  already the encode-ahead-off configuration). Note this is the **only** in-suite exercise of the
  encode-ahead pipeline: before it, every `forward_token` call in `mod tests` passed
  `next_rope: None`. The two configurations run SEQUENTIALLY, each session dropped before the
  next starts; the first cut interleaved them and deterministically broke the five gemma3
  self-parity tests plus `metal_gemma4_layer_matches_cpu` — a pre-encoded graph is committed and
  gated, so with two live sessions the second one's work (and every concurrent Metal test's)
  queues behind a command buffer that only unblocks on the next loop iteration.
- `metal_kquant_embed_gather_drops_embed_scale_so_gpu_sampling_fails_closed` — proves on the
  device that binding `buffer(4)` on `embed_row_gather_q4k` is legal and INERT, and pins the new
  host fail-closed (`gpu_sampling_tail_is_scale_safe`) that refuses the device-side sampling tail
  when a non-unit `embed_scale` meets a non-Q8_0 embedding table. Note the production caller
  already requires a Q8_0 token embedding before it builds the stage at all, so this is
  defence-in-depth at the enforcement point rather than a live bug fix.
- `resident_session_construction_sets_the_kquant_lane_at_both_sites` — `ResidentDecodeState::new`
  reads the global that `set_resident_kquant_lane` writes, and both call sites sit in
  merge-conflicting hunks; dropping either is silent (an F32 primary where F16 was intended, no
  failing assertion anywhere). A source-level count is crude but it is the only thing that fails
  when a merge quietly deletes one.

### 9d. Gates

cargo fmt clean; `clippy --all-targets -D warnings` clean (load-bearing here: it is what turns a
dead `window_start` parameter into a red build — do not silence it by underscoring the
parameter); `cargo test --all-targets` green twice in a row (1734 passed / 0 failed, against
1730/0 at the commit before these four tests); `cargo test --lib metal::tests` green three times in a row
(the filtered run is the sensitive one for queue starvation); the real-row gate green with the
numbers in 9b; check-public-scrub.sh clean.

**Standing note for anyone adding Metal tests here.** The three failure modes hit in this pass
all came from process-wide or device-wide sharing, never from the maths: (1) a test that sets a
gate env var latches sibling tests onto a different kernel; (2) a test that holds the single
shared serial command queue for hundreds of command buffers starves siblings until they read
back unwritten buffers; (3) two live resident sessions with encode-ahead park a gated command
buffer in front of each other. All three present as "unrelated test returns 0 or NaN", and (2)
and (3) look like flakes until you re-run the filtered subset. Budget command buffers, keep one
resident session live at a time, and arm gates from the shell.

### 9e. Amendments to sections 3-6 (recorded, not yet applied to those sections)

1. **§3 Phase 3 — a fourth eligibility surface.** Main added `is_gpu_runnable_arch`
   (src/execution_plan.rs), an allow-list of `llama | qwen2 | qwen3 | mistral` consumed by the Q8
   GPU-runnable tier and by K-quant plan selection, whose comment explicitly names gemma3 as a
   mirror of `resident_decode_eligible`. Without it the Metal-resident K-quant plan is never
   advertised and the Q8 GPU-runnable tier stays closed. Add it beside
   `resident_decode_eligible`, `recognized_row_level`/`is_supported_exact_q8_row` and
   `is_runnable_serve_arch`. The §3/§5 line citations for all of these are stale after main's
   +2775 lines in src/metal.rs alone — re-derive rather than trusting them.
2. **§3 Phase 3 — the prompt-prefix cache is a first-class blocker.** PR #557 did not exist when
   Phase 3 was scoped. On a non-exact cache hit the resume path rolls back to `k` and re-prefills
   the divergent suffix at `kv_position = k > 0`; the only GPU-prefill hook refuses any non-zero
   start, so the suffix is evaluated by the CPU dense forward — which has none of gemma3's
   structure and no window at all. Partial hits are admitted from 16 tokens. The failing case is
   ordinary multi-turn chat. The bypass must be a new explicit windowed-arch predicate at the
   lookup sites and the store site, NOT inside `try_metal_resident_prefill` (unreachable at
   position > 0) and NOT `CAMELID_PREFIX_CACHE_RESIDENT=0` (`prepare_for_prompt_prefix_cache`
   returns before consulting it). Related: the campaign's "token-by-token prefill through the
   decode path" bullet needs a location — it must be at the session level in
   `generate_next_token_with_history_diagnostics`, forcing a single-token prefill chunk.
3. **§3 Phase 3 — pin the flip to the Q8_0 row in the mechanism, not only the risk register.**
   The disqualifier is arch-keyed (`matches!(architecture, "gemma2" | "gemma3")`), and main has
   since opened the Metal resident lane to Q4_K/Q6_K weights. A gemma3 Q4_K_M GGUF would reach
   the resident lane, take an F16 primary, and activate the whole mirror/store/partial-hit path.
   Scope the flip to the Q8_0 exact row, or exclude gemma3 from the K-quant Metal admission,
   until a windowed K-quant lane has its own receipt.
4. **§3 Phase 2 — the "two-line gemma4 math" is no longer two lines.** After #556 the shared
   `encode_attention_block` computes `kv_position_stride` conditionally, so the window base is
   `window_start * kv_position_stride`. The gemma4 lane keeps element units legitimately (f32 KV
   only) and is no longer a copyable precedent for the shared encode.
5. **§4 / §6 — state the prefix-cache exclusion with any throughput claim.** On the Q8_0 row
   gemma3 gets ZERO prompt-prefix reuse: `prepare_for_prompt_prefix_cache` requires
   `kv_roundtrips_through_cpu_exactly()`, which is literally the F16-primary flag. Multi-turn
   chat therefore pays a full token-by-token prefill every turn. The §4 "0.2 tok/s baseline /
   ~110 tok/s roofline" framing predates #557 and must carry this caveat.
6. **§5 — two new landmines.** (a) The process-global vs per-session KV-format split inside the
   shared attention encode; closed for `encode_attention_block` in 9a, still open for the inner
   `encode_attention`. (b) Appliance mode drops encode-ahead at 2+ active slots, which makes
   `SampleStage.embed_scale` dormant — the sqrt(d_model) scale is applied twice by design (CPU
   input and GPU gather) and the two paths are never both exercised in one run. Every gemma3
   parity receipt must state its slot count; the Phase 4 bundle should carry both.
   Also: a window-aware KV mirror is tempting (25 of 26 layers can only read the trailing 512
   positions, yet the mirror copies `[0, position)` per layer — ~53 KiB/position, ~3.4 GB at the
   row's 32,768-token context) but it changes the round-trip exactness argument and must not be
   attempted before the Phase 3 blockers.
7. **§3 Phase 4 — the ≥512-token receipt has a second job.** Because the Phase 2 gate asserts
   `total < 512`, the windowed pack is the only external artifact that can distinguish a correct
   window base from a zeroed one. It is merge-correctness evidence, not only a context claim.

### 9f. Phase 3 is NOT open

The blockers in 9e-2/9e-3 (prompt-prefix-cache routing, arch-vs-quantization scope) plus the
requirement that an explicit gemma3 fail-closed on the CPU dense fallback lands in the SAME
commit that removes `is_runnable_only_arch` / the `resident_decode_eligible` disqualifier are
gates on Phase 3, not Phase 3 work. Until they are closed, the ≥512-token correctness claim this
campaign exists to retire can be re-broken by any fallback, and the serve router divert remains
the only production correctness guard.

## 10. Phase 3a record (2026-07-30)

Phase 3a closed the three blocking hazards from §9e-2/§9f so the Phase 3b routing flip
(removing the arch disqualifiers) becomes a safety-neutral change. gemma3 stayed FAIL-CLOSED
throughout: nothing here changes production routing for any arch — this is the safety plumbing
the flip will stand on. Three commits, one per hazard, each gated by
fmt / clippy --all-targets -D warnings / cargo test --all-targets.

### 10a. H1 — prompt-prefix cache bypass for windowed archs

New predicate `crate::model::arch_has_windowed_attention(&LlamaModelConfig)`
(src/model.rs:167, beside `is_runnable_only_arch` at :150): keyed on the PARSED metadata
(`config.gemma3.is_some()`), not the arch string, so gemma3-4B and any future windowed arch
inherit every guard that consults it.

Enforced at the three prompt-prefix-cache decision sites in src/api/mod.rs:

- STORE (`store_prompt_prefix_cache`, :13053): a windowed arch never stores an entry — checked
  before the position check and before `prepare_for_prompt_prefix_cache`, so no mirror cost is
  ever paid on the refusal.
- Both PARTIAL-RESUME sites now share one decision point, `resume_partial_prefix_hit` (:13116,
  extracted so the non-streaming handler and `stream_prompt_cache_prologue` cannot drift),
  which refuses a windowed arch: the divergent suffix would be re-prefilled at
  `kv_position > 0`, the resident prefill hook refuses any non-zero start, and the CPU dense
  forward has no window — the H1 failing case is ordinary multi-turn chat. Declining costs one
  cold full prefill: slower, never wrong.
- EXACT hits stay allowed: no forward runs on that path, and with the store site refusing, no
  windowed entry can exist outside a stale pool — which the resume guard also covers.

Tests: `windowed_arch_never_stores_a_prompt_prefix_entry` (store site; a non-windowed control
run proves the bypass is the thing that fired), `windowed_arch_never_takes_a_partial_prefix_resume`
(the shared resume decision point, with control), and
`stream_prologue_windowed_arch_partial_hit_falls_back_to_cold_prefill` (the streaming site
end-to-end through `CooperativeStreamDecodeJob::new` against a hand-inserted stale entry).

### 10b. H2 — session-level token-by-token prefill for windowed archs

`session_prefill_chunk_tokens(config, prefill_count)` (src/inference.rs:5510) is now the
prefill routing decision consumed by `generate_next_token_with_history_diagnostics`: a windowed
arch forces the single-token lane (chunk = 1), so every prompt token flows through
`forward_single_token_timed_internal` → `try_resident_decode_forward` — the only lane whose
forward carries the sliding-window / dual-theta schedule once the arch is admitted (the gemma4
runtime's token-by-token prefill is the semantic precedent, per §9e-2). Every other arch keeps
`prefill_chunk_token_count` verbatim — byte-identical routing, pinned by
`non_windowed_arch_prefill_chunking_is_byte_identical` next to
`windowed_arch_prefill_forces_the_single_token_lane`.

The production arch disqualifier (src/inference.rs:2779) stays up. A cfg(test)-only seam,
`TEST_ADMIT_WINDOWED_ARCH_RESIDENT` (src/inference.rs:12303, compiled out of production builds
entirely), admits the arch for the duration of one targeted test so the routing could be proven
BEFORE the flip: `gemma3_session_level_token_by_token_prefill_matches_runnable_oracle`
(src/metal.rs) drives the PRODUCTION session entry over the real 1B row with a multi-token
prompt. Measured (M4 mini, release, f32y+wire+NSG8 armed, CAMELID_METAL_RESIDENT_DECODE=1):

| depth | session argmax | oracle argmax | max abs logit diff |
|---|---|---|---|
| 1 | 108 | 108 | 6.247e-5 |
| 2 | 584 | 584 | 6.676e-5 |
| 3 | 568 | 568 | 5.627e-5 |
| 4 | 2364 | 2364 | 5.913e-5 |
| 5 | 1077 | 1077 | 7.820e-5 |

5/5 greedy tokens identical, overall max abs logit diff 7.820e-5; depth 1 matches the Phase 2
gate bit-for-bit-in-report (108 / 6.247e-5) — same forward, now reached through the session.
The routing itself is pinned by `!session.cpu_kv_authoritative()` at the end: a CPU dense
prefill of any flavor materializes the CPU KV as it goes; the resident lane leaves it hollow.

### 10c. H3 — the cache kill switch is real (arch-independent live-main bug)

`prepare_for_prompt_prefix_cache` returned `true` for a CPU-authoritative session BEFORE
consulting `CAMELID_PREFIX_CACHE_RESIDENT`, so the documented opt-out did nothing on any
CPU-authoritative session — which today is every windowed-arch session (H2) and the entire
ordinary CPU lane. The gate is now consulted FIRST
(`prepare_for_prompt_prefix_cache_gated`, src/inference/metal_resident.rs:428): with the
variable set to `0`/`false`, preparation refuses every session, making the variable a real
kill switch for cache storage (`store_prompt_prefix_cache` refuses on `false`).

Tests: `prompt_prefix_cache_preparation_env_opt_out_is_a_kill_switch` drives the parameterized
seam on a session that caches under `true`;
`prefix_cache_env_setting_parses_the_documented_opt_out` covers the env-value parse
(`prefix_cache_setting_enables`, split pure from the OnceLock). Deliberately NOT an in-test
`set_var`: the gate is a process-wide OnceLock and §9d's standing note applies — gates are
armed from the shell, never latched from inside a test.

### 10d. Gates

Per commit: cargo fmt clean; clippy --all-targets -D warnings clean; cargo test --all-targets
green (H1 commit: pipeline exit 0 under pipefail, lib suite 1387 tests started, no failures —
the tally lines were lost to an output filter; H2 commit: 1367 lib passed / 0 failed / 23
ignored plus every integration suite green, exit 0; H3 commit: recorded below in this section's
final battery). check-public-scrub.sh clean. Env-keyed battery at phase end (release,
production GEMV configuration armed from the shell): the H2 session-level gate above, and the
Phase 2 real-row final gate re-run:

H3-commit full battery: cargo test --all-targets exit 0, lib 1369 passed / 0 failed / 23
ignored, every integration suite green (60 green tallies). Phase 2 real-row final gate re-run
(release, no SKIP, 570.08 s): depth 1 argmax 108 = oracle, max abs logit diff 6.247e-5; depth 5
argmax 1077 = oracle, 7.820e-5; depth 50 argmax 578 = oracle, 9.584e-5 — 50/50 greedy tokens
identical, overall max abs logit diff 2.122e-4, bit-for-bit the §9b record. The whole Phase 2
stack is therefore proven UNCHANGED under the Phase 3a plumbing.

### 10e. What Phase 3b still owes (restated against current line numbers)

- The flip commit must remove/condition BOTH `is_runnable_only_arch` (src/model.rs:150) and
  the `resident_decode_eligible` arch disqualifier (src/inference.rs:2779) AND land the
  explicit gemma3 fail-closed on the CPU dense fallback in the SAME commit (H4, §9f).
- Scope the flip to the Q8_0 exact row in the MECHANISM (H5): main's Metal K-quant admission
  (`is_resident_quant` / `metal_only`, src/inference.rs:2894-2903) would otherwise admit a
  gemma3 Q4_K_M to an F16-primary resident lane whose K-quant gather drops `embed_scale` (the
  H6 fail-closed covers only the GPU sampling tail).
- Eligibility surfaces, current locations: `is_gpu_runnable_arch` (src/execution_plan.rs:863;
  consumed at :353 Q8 GPU-runnable tier and :375 K-quant plan selection),
  `recognized_row_level` (src/execution_plan.rs:1027) / `is_supported_exact_q8_row` (:1066),
  `is_runnable_serve_arch` (src/api/mod.rs:7337, a delegate to `is_runnable_only_arch`).
- NEW since the §3 checklist was written (the #549/#554 merges):
  - `prepare_generation` (src/api/mod.rs:11523) carries the raw-completions choke-point gate
    (`completions_unsupported_for_arch` at :11587, delegating to `is_runnable_serve_arch`)
    covering `/completion`, `/v1/completions` n>1 fan-out, `/api/generation/preflight`,
    `/api/generation/sessions`, and receipt replay; `/v1/completions` itself also gates via
    `reject_completions_for_runnable_arch` (:7684, applied at :10065). Flipping
    `is_runnable_only_arch` membership REOPENS all of these for gemma3 in the same motion —
    the flip commit must decide whether that is intended and cover it with tests (the #554
    test module `runnable_completions_gate_api_tests` pins today's behavior).
  - `/v1/responses` delegates to `chat_completions` (src/api/responses.rs:169) and so inherits
    whatever chat routing the flip leaves behind.
  - The runnable serve lane's tools threading (`runnable_request_tools`, src/api/mod.rs:14066,
    consumed by `runnable_chat_nonstreaming` :7731 and `runnable_chat_streaming` :7854) serves
    gemma3 chat today and goes dormant for gemma3 when the arch leaves
    `is_runnable_serve_arch` — tool-calling parity on the dense lane is NOT covered by any
    existing gemma3 test.
- H1's bypass keys on `arch_has_windowed_attention`, which is INDEPENDENT of the runnable
  predicates: the flip does not reopen the prompt-prefix cache for gemma3. Reopening it later
  is §9e-5/H11 territory (F32 primary never qualifies; a window-aware mirror changes the
  round-trip exactness argument) and stays out of 3b.

## 11. Phase 3b record (2026-07-30)

Phase 3b is the routing flip: gemma3 is servable on the Metal GPU-resident lane. One commit
(ba6de7f7), standing entirely on the 3a plumbing; nothing in it touches kernels or the forward.

### 11a. The capability-aware predicate

The flip is NOT a bare list edit. gemma3 left `is_runnable_only_arch` (src/model.rs, now
`qwen35 | gemma2` only), and routing keys on a new pair beside it:

- `model::arch_requires_runnable_bridge(arch)` — the live predicate serve and the CLI direct
  lanes consult. True for qwen35/gemma2 always; true for gemma3 only where the resident lane
  cannot serve it.
- `model::arch_requires_runnable_bridge_given(arch, capable)` — the pure half, so the split is
  unit-testable without env or a device.
- `inference::windowed_arch_resident_host_available()` — the host probe:
  macOS build AND `resident_decode_metal_enabled()` (live env; deterministic mode force-off)
  AND NOT `resident_decode_cuda_enabled()` (the CUDA engine has no windowed forward)
  AND a real Metal device (`detect_metal_device().available`, cached in a OnceLock).

Consumers rewired: `api::is_runnable_serve_arch` (serve router + runnable-runtime load +
`completions_unsupported_for_arch`) and `main::ensure_arch_has_direct_dense_session` both
delegate to `arch_requires_runnable_bridge`, so serve and the CLI cannot disagree. Outcome:
on a resident-capable host gemma3 chat falls through the runnable short-circuit onto the dense
lane and the resident engine serves it; on every other host (non-macOS CI legs, resident decode
opted out, deterministic mode, CUDA-resident, no device) gemma3 loads the runnable runtime and
serves exactly as before the flip — never the CPU dense forward.

### 11b. H4 — the CPU dense forward fails closed for windowed archs (same commit)

`LlamaInferenceSession::ensure_windowed_arch_off_cpu_dense` (src/inference.rs), keyed on
`arch_has_windowed_attention`, returns a typed `BackendError::UnsupportedModelArchitecture`
naming the hazard and both correct lanes. Guarded at ALL THREE CPU dense forward dispatches:
the single-token decode fallback (the else-branch after `try_resident_decode_forward`
declines), `forward_prefill_chunk_timed_fast`, and
`forward_prefill_layer_major_timed_fast_inner`. No routing mistake can silently run gemma3
full-causal. A second cfg(test)-only seam (`TEST_ADMIT_WINDOWED_ARCH_CPU_DENSE`, drop-guarded,
armed only under `env_lock`) lets the 3a prompt-prefix-cache decision tests keep driving tiny
synthetic gemma3 configs through the CPU forward mechanically; the 3a resident seam is
unchanged and now effectively covers only gemma2. Pinned by
`windowed_arch_cpu_dense_forward_fails_closed` (with a non-windowed control proving causality).

### 11c. H5 — resident admission pinned to the Q8_0 exact row (same commit)

In `resident_decode_eligible`: the arch disqualifier is now gemma2-only; windowed archs gained
(a) a CUDA-resident bail (the CUDA engine would run the window full-causal) and (b) a Q8_0 pin
— `is_resident_quant` returns `is_q8` only for windowed archs, plus an explicit pre-loop typed
decline when any layer linear is non-Q8_0 (a gemma3 Q4_K_M would otherwise ride the Metal
K-quant admission onto an F16-primary lane whose gather drops `embed_scale`, with no windowed
receipt). Serve falls back to the runnable bridge for such files.

### 11d. Execution plan

- `recognized_row_level`: gemma-3-1b-it row added at a NEW honest level string
  `supported_exact_row_smoke_sub512` (the ≥512 receipt is Phase 4's), included in
  `is_supported_exact_q8_row`; `support_level` already gates it to Q8_0 files only.
- Plan selection is platform-split for windowed archs: macOS+Metal+resident →
  `metal_resident_q8_runtime` (the load-bearing selection §3 called out); anywhere the Metal
  selection cannot fire (non-macOS, resident unset, `CAMELID_MAC_Q8_METAL_PLAN=0`) the plan
  FAILS CLOSED to `safe_q8_plan` with a windowed-arch reason instead of advertising a CPU
  dense lane H4 forbids (`select_macos_q8_plan` gained a `windowed_attention_arch` param;
  the x86 arm is bypassed via `is_windowed_attention_arch`).
- `is_gpu_runnable_arch`: gemma3 deliberately NOT added. Both consumers are non-Q8-exact
  tiers H5 forbids — the Q8 GPU-runnable tier is CUDA-resident (no windowed CUDA forward) and
  the K-quant plan selection would advertise the Metal K-quant lane. Decision recorded in the
  function comment; pinned by `gemma3_kquant_never_takes_the_metal_resident_kquant_plan`.
- New plan tests: `gemma3_q8_row_selects_metal_resident_plan_on_a_resident_mac`,
  `gemma3_q8_row_fails_closed_to_safe_plan_where_metal_resident_cannot_run`, and the K-quant
  pin above.

### 11e. Raw-completions surfaces (#554) and the dense chat renderer

The #554 chokepoints (`prepare_generation` dense gate, `reject_completions_for_runnable_arch`)
key on the capability-aware predicate, so gemma3's raw surfaces (`/completion`,
`/v1/completions` + n>1 fan-out, `/api/generation/preflight`, `/api/generation/sessions`,
receipt replay) REOPEN exactly where the resident lane serves and stay 422-gated on the
runnable fallback. `api::runnable_completions_gate_api_tests` pins the split: the always-gated
tests moved to qwen35/gemma2, and two new tests pin gemma3 both ways
(`completions_gate_stays_closed_for_gemma3_on_a_runnable_fallback_host` — env=0 under
env_lock, restores the caller's value; `completions_gate_reopens_for_gemma3_where_the_
resident_lane_serves` — macOS+device gated).

The dense chat lane gained gemma3's prompt renderer: without it the fallback renderer dropped
gemma3 chats onto the role-colon prompt. `is_gemma3_chat_template` (`<start_of_turn>` +
`<end_of_turn>` + `first_user_prefix`) routes to the SAME byte-faithful `render_gemma3_prompt`
the runnable bridge uses, with the identical encode contract (no BOS in the string,
add_special=true, parse_special=true). Pinned by
`gemma3_template_renders_through_the_shared_gemma3_renderer_on_the_dense_lane`.

### 11f. Tool calling (#549) — decision

gemma3 tool calling has never been supported on ANY lane: the row is `tool_capable: false` and
the runnable bridge returns a typed 422 (`unsupported_tools` — "no tools branch, no certified
grammar") by design. "Fixing dense-lane tool threading" would mean inventing an uncertified
tool grammar for a template that has none — the opposite of this repo's fail-closed policy.
The flip therefore PRESERVES the explicit refusal contract on the dense lane:
`render_chat_prompt_for_tokenization_with_tools` declines gemma3's template with the same
row-accurate reason (surfaced as 422 `unsupported_chat_template`), pinned by test. Tools are
never silently dropped from the prompt, and `tool_choice:"none"` still renders plain chat
(verified live). This is behavior-IDENTICAL to pre-flip from the API user's perspective.

### 11g. Serve smokes (release, this M4 mini 16 GB, no special env vars)

Resident smoke (`camelid serve --model gemma-3-1b-it-Q8_0.gguf --no-open`):
- /v1/health: `generation_ready:true`, `selected_backend:"metal_resident_q8_runtime"`,
  `decode_path:"q8_0_metal_resident_decode"`, `support_level:"supported_exact_row_smoke_sub512"`,
  backend `"llama"` (dense serve lane; NO runnable runtime loaded).
- Greedy chat ("Why is the sky blue? Answer in one sentence.", 20 prompt tok): coherent
  Rayleigh-scattering sentence, finish stop, 26 tokens. Run 1 wall 0.825 s, run 2 (warm)
  0.788 s — byte-identical token ids across runs. Warm timings: prefill 19 tok / 302.2 ms
  = 62.9 tok/s (token-by-token per H2), first token 40.4 ms, decode 25 tok / 389.4 ms
  = **64.2 tok/s decode** at short depth.
- Long greedy (33 prompt / 256 completion, two runs, byte-identical): prefill 81.5 / 86.1
  tok/s; decode 255 tok in 5662.0 / 5543.3 ms = **45.0 / 46.0 tok/s decode** at depth ~289.
  Within the §4 25-60 target band; ~0.4-0.6x of the ~110 tok/s roofline.
- Oracle check: first 8 greedy token ids [818, 7217, 7412, 3730, 1547, 529, 496, 20284]
  IDENTICAL 8/8 to the runnable oracle's (same prompt, fallback server below). No envelope
  flip needed.
- Tools (tools + tool_choice auto): typed 422 `unsupported_chat_template` — "the gemma3 chat
  template has no tools branch and no tool-call grammar is certified for this row; tool
  requests fail closed on the dense lane exactly as on the runnable bridge" (§11f).
- Raw `/v1/completions` REOPENED: "The capital of France is" → " Paris.\n\nThe largest city
  in France" (200).
- Response `lane` discloses `"experimental"`: `filename_is_supported_exact_row` still fails on
  the catalog/row id mismatch (`gemma3_1b_it_q8_0` vs `gemma_3_1b_it_q8_0`) — the KNOWN
  Phase 5 deliverable (§3 Phase 5, precedent c2c33fb5), pre-existing, not new breakage.

Fallback smoke (`CAMELID_METAL_RESIDENT_DECODE=0`, same command):
- /v1/health: backend `"runnable-runtime"`; plan fails closed to `cpu_reference` /
  `safe_cpu_decode` with the windowed-arch reason string.
- Greedy chat 8 tok: 27.07 s wall (the known ~0.2-0.3 tok/s runnable lane), token ids above.
- `/v1/completions`: 422 `unsupported_completions_lane` (gate intact verbatim).
- Tools: 422 `unsupported_tools` (runnable lane refusal unchanged).
Both servers killed by saved PID only.

### 11h. Gates

Flip commit: cargo fmt clean; clippy --all-targets -D warnings clean; cargo test --all-targets
exit 0 under pipefail (lib 1376 passed / 0 failed / 23 ignored; every integration suite green).
check-public-scrub.sh clean; scripts/check-ledger-drift.mjs passed (no capability-row or
readiness-gate string touched — the row rewrite is Phase 5's). Env-keyed battery (release,
production GEMV gates + resident decode armed from the shell, targeted names so the
env-mutating gate tests never run alongside):
- Phase 2 real-row final gate re-run: depth 1 argmax 108 = oracle, max abs logit diff
  6.247e-5; depth 5 argmax 1077 = oracle, 7.820e-5; depth 50 argmax 578 = oracle, 9.584e-5 —
  50/50 greedy tokens identical, overall max abs logit diff 2.122e-4, 564.04 s. Bit-for-bit
  the §9b/§10d record: the whole Phase 2 stack is UNCHANGED under the flip.
- 3a session-level prefill gate re-run: 5/5 greedy identical (108/584/568/2364/1077),
  overall max abs logit diff 7.820e-5, 21.92 s — bit-for-bit the §10b record.
- `gemma3_real_row_runnable_rope_schedule_is_the_reference_schedule` (release): PASS,
  fingerprint sum_bits 0x0002eec61740012f unchanged.
- `gemma3_real_row_binds_all_104_norm_tensors_and_window_schedule` (env-keyed, with the
  updated Phase 3b invariants): PASS.

### 11i. What Phase 4/5 inherit

- The ≥512-token windowed receipt (Phase 4) is now reachable over the SERVED lane — depth >512
  decode measured working here at speed (45-46 tok/s at depth ~289; §9e-7's merge-correctness
  role stands).
- Phase 5 owes: the capabilities-row rewrite (it still describes the runnable lane) + ledger
  regen; the catalog/row id mismatch fix that currently makes dense responses disclose
  `lane:"experimental"` and keeps `filename_is_supported_exact_row` false for this row;
  frontend fixtures; docs sweep. The plan's new `supported_exact_row_smoke_sub512` level
  string is deliberately scoped until the Phase 4 receipt lands.
- Multi-turn chat pays a full token-by-token prefill every turn (prefix cache stays closed for
  windowed archs, H1/§9e-5) — at 60-80 tok/s prefill this is now noticeable but not painful;
  reopening the cache stays H11 territory.
- Speculative decode: opt-in env only; on gemma3 the CPU verify path now H4-errors (typed)
  and no explicit spec gate was added — flagged for Phase 5/6 if spec decode is ever pointed
  at this row.

## 12. Phase 3c record — adversarial review of the Phase 3b flip (2026-07-30)

Phase 3b (`ba6de7f7`) was put through an adversarial review. Four findings were confirmed
blocking/major before this phase started; eight more were reported by the review's lenses but
its verification pass died on infrastructure, so each was re-verified here against the code
before being fixed or dismissed. **Every one of the twelve turned out to be real.** Four
commits, each gated by fmt / clippy --all-targets -D warnings / cargo test --all-targets.

The stakes framing that shaped every fix: on these paths a routing mistake is SILENT WRONG
OUTPUT, not a crash. The CPU dense forward has no sliding window, no sandwich norms, no GeGLU
and no dual-theta RoPE, so gemma3 on it decodes fluent-looking garbage under a supported label.

### 12a. F1 (BLOCKER) — the fail-closed moved to a choke point

H4 guarded three CPU dense entry points. The flip made four more reachable for gemma3, all
unguarded: `forward_greedy_verify_chunk` (the spec-decode CPU verify walk, live from
`src/api/mod.rs`), `forward_layer_range_from_hidden` (distribute master / activation replay),
`forward_worker_layers` (`src/distributed.rs`), `ghost_forward_one_layer` (`src/main.rs`).

Guarding those four would have left the fifth. **The choke point:
`ensure_windowed_arch_off_cpu_dense_layer` (src/inference.rs), called from
`forward_layer_timed` and `forward_prefill_layer_chunk_timed`** — the only two per-layer dense
forwards in the file. A dense walk cannot be written that skips it, because it cannot compute a
layer without one of them.

**Completeness enumeration.** Every CPU dense layer loop in src/inference.rs, and the choke
point each now hits:

Line numbers are at `b156e92c`; `ensure_windowed_arch_off_cpu_dense_layer` is
src/inference.rs:7114, called from `forward_layer_timed` (:7130) and
`forward_prefill_layer_chunk_timed` (:7898).

| # | entry point (src/inference.rs) | layer loop | per-layer forward | guard reached |
|---|---|---|---|---|
| 1 | `forward_worker_layers` :4076 (prefill arm) | :4114 | `forward_prefill_layer_chunk_timed` | choke point |
| 2 | `forward_worker_layers` :4076 (decode arm) | :4165 | `forward_layer_timed` | choke point |
| 3 | `forward_layer_range_from_hidden` :4218 | :4250 | `forward_prefill_layer_chunk_timed` | choke point |
| 4 | `ghost_forward_one_layer` :4284 | (single layer) | `forward_prefill_layer_chunk_timed` | choke point |
| 5 | `forward_prefill_chunk_timed_fast` :4353 | :4401 | `forward_prefill_layer_chunk_timed` | early guard "batch prefill", then choke point |
| 6 | `forward_greedy_verify_chunk` :4461 | :4521 | `forward_prefill_layer_chunk_timed` | choke point |
| 7 | `forward_prefill_layer_major_timed_fast_inner` :4591 | :4648 | `forward_prefill_layer_chunk_timed` | early guard "layer-major prefill", then choke point |
| 8 | `forward_single_token_timed_internal` :4786 | :4903 | `forward_layer_timed` | early guard "decode forward", then choke point |

There is no ninth: the only other `layers.iter()` sites in the file are binding/merge code, not
forwards. The three session-level guards are KEPT, demoted to an early leg that names the lane
before the walk starts; they delegate to the same free function so the message cannot drift.

The review also found all three existing guard sites were **revert-invisible** — the single H4
test asserts two substrings the choke point also emits, so deleting any guard failed nothing.
Ten tests now cover this (src/inference/tests.rs), each named for what it catches:

| test | fails when |
|---|---|
| `windowed_arch_choke_point_refuses_the_per_layer_decode_forward` | the `forward_layer_timed` choke-point call is deleted |
| `windowed_arch_choke_point_refuses_the_per_layer_prefill_chunk` | the `forward_prefill_layer_chunk_timed` choke-point call is deleted |
| `non_windowed_arch_still_computes_a_cpu_dense_layer` | the choke point starts refusing everything (causality control) |
| `windowed_arch_batch_prefill_names_its_lane_before_the_layer_walk` | the `"batch prefill"` early guard is deleted |
| `windowed_arch_layer_major_prefill_names_its_lane_before_the_layer_walk` | the `"layer-major prefill"` early guard is deleted |
| `windowed_arch_decode_fallback_names_its_lane_before_the_layer_walk` | the `"decode forward"` early guard is deleted |
| `windowed_arch_speculative_verify_chunk_inherits_the_choke_point` | entry point 6 stops being covered |
| `windowed_arch_worker_layer_shard_inherits_the_choke_point` | entry point 1/2 stops being covered |
| `windowed_arch_layer_range_replay_inherits_the_choke_point` | entry point 3 stops being covered |
| `windowed_arch_ghost_layer_probe_inherits_the_choke_point` | entry point 4 stops being covered |

### 12b. F2 (BLOCKER) — CLI lane admission is honest again

`ensure_arch_has_direct_dense_session` (src/main.rs) went capability-aware for gemma3, which
admitted it to all ten lanes it guards. Five of them walk the CPU dense layer loop directly and
can never run a windowed forward on ANY host. They now refuse before weights load, with an
actionable error naming `camelid serve`, via a `DenseLaneWindowedForward` lane class:

- `CpuDenseOnly` (refuse windowed on every host): distribute worker role in `main`,
  `run_ghost`, `load_model_drafter`, `run_bench_speculative`, `run_distribute_worker`,
  `run_distribute_master`.
- `ViaSessionDecode` (capability-aware, unchanged): `BenchAllocGate`, `gait_profile_trial`,
  `run_bench_owner_sweep`, `run_bench_generate` — all of which generate through
  `generate_next_token_with_history_diagnostics` and therefore inherit the resident lane (H2).

Tests: `cpu_dense_only_cli_lanes_refuse_a_windowed_arch_on_every_host` (with a llama control on
the same lane class, so the refusal is caused by the window schedule and not the lane),
`a_kquant_windowed_row_is_refused_even_on_a_session_decode_lane`,
`runnable_only_archs_stay_refused_on_both_lane_classes`.

### 12c. F3 (MAJOR) — routing is keyed on the FILE, not the arch string

`arch_requires_runnable_bridge_given` keyed on (arch, host) while resident admission is Q8_0-
pinned (H5). A gemma3 Q4_K_M on a resident-capable Mac routed to the dense lane, loaded no
runnable runtime, was declined by H5, and died on the H4 error for every request — a hard
regression (the bridge served every gemma3 quant pre-flip), and both the H5 decline text and
the flip's commit message promised a fallback no code implemented.

- `model::file_requires_runnable_bridge(gguf)` is the live predicate. The arch-only entry point
  is REMOVED, not defaulted: every live caller has the `GgufFile` and must pass it, so
  quant-blindness is structurally impossible rather than merely fixed.
- `model::windowed_arch_resident_quant_admissible(gguf)` decides the quant half from GGUF
  metadata pre-load, mirroring the engine pin exactly (all seven per-layer linears Q8_0; a file
  with no recognizable layer linears fails closed to the bridge).
- `arch_requires_runnable_bridge_given(arch, host_available, quant_admissible)` is the pure
  core; a windowed arch needs BOTH to take the resident lane.
- Consumers renamed in lockstep: `api::is_runnable_serve_file`,
  `api::completions_unsupported_for_file`.

**Verification method, stated because it is not a live-file proof:** no non-Q8_0 gemma3 GGUF
exists under `/Volumes/Untitled/models` (or the desktop model dir) — only
`gemma-3-1b-it-Q8_0.gguf`. F3 is therefore proven by unit tests on the predicate
(`a_non_q8_windowed_row_requires_the_runnable_bridge_on_every_host`,
`windowed_quant_admission_requires_every_layer_linear_to_be_q8_0`) plus an HTTP-level test with
a synthetic Q4_K fixture (`a_kquant_gemma3_keeps_the_runnable_bridge_on_a_resident_capable_host`,
identical to the reopen test but for the tensor type, so quantization is provably what decides
it), and the CLI test above.

### 12d. F4 (MINOR) — speculation is declined for the arch, not gated per mode

`serve --spec-decode` with the resident lane armed made every gemma3 request H4-error: CPU
speculation pins the target off the resident paths so chunk-verify rollback has
CPU-authoritative KV. Of the two options offered, neither was taken as written. Teaching
routing about spec mode couples two unrelated decisions and leaves routing env-order-sensitive;
exempting windowed archs from the spec pin is worse, because the CPU verify walk is itself the
window-less dense forward. **Decision: a windowed arch never speculates, on either verify
lane.** The GPU batched verify is no better than the CPU one here — `encode_attention`
(src/metal.rs) takes no `window_start` at all and reads `[0, position_count)` full-causal; it
is shared with the gemma4 lane and was never threaded with the window (§9a records this as an
open residual). Since lossless speculation only ever adds throughput, declining it costs
correctness nothing, and `--spec-decode` now SERVES gemma3 on the plain resident lane instead
of failing. The disqualifier set moved into `api::speculation_admissible`, which also gives the
four pre-existing disqualifiers their first test
(`speculation_is_declined_for_a_windowed_arch_on_both_verify_lanes`). This supersedes §11i's
"no explicit spec gate was added" flag.

### 12e. Triage of the eight unverified leads — all eight CONFIRMED, none dismissed

1. **Dense-lane renderer keyed on TEMPLATE SHAPE while routing admits by ARCH.** CONFIRMED.
   `is_gemma3_chat_template` tests three substrings of the template text; routing keys on
   `config.architecture`. A gemma3 GGUF whose template lacks them fell through
   `render_chat_prompt_for_tokenization_fallback` to `render_role_colon_prompt` — a `"user: …"`
   prompt the model was never trained on, served silently under a supported-row label — while
   the runnable bridge, which keys the same decision on `runtime.architecture`, rendered it
   correctly. The two lanes disagreed for exactly the files most likely to be mis-converted.
   FIXED: `reject_windowed_arch_with_unrecognized_template` fails the arch/template mismatch
   closed with a typed 422 naming the missing markers. Test
   `gemma3_arch_with_an_unrecognized_template_fails_closed_on_the_dense_lane` covers the
   near-miss (turn markers present, `first_user_prefix` absent), the role-colon template, and
   a missing template, with a recognized-template control and non-windowed controls.
2. **H5's Q8_0 admission has ZERO CI coverage.** CONFIRMED, and the mechanism is worse than
   reported: the test claiming to pin it does not short-circuit, it is *vacuous* — its own doc
   comment says the refusal may come from the backend-enabled gate instead, and its fixture's
   f32 weights are rejected by the generic layer loop first, so deleting both H5 lines fails
   nothing on any host, armed or not. The blocker is ordering: the backend-enabled bail sits
   ~120 lines earlier and fires on every ordinary `cargo test` (the flag is default-off), and
   arming it from inside a test is forbidden by §9d. FIXED by extraction:
   `windowed_arch_layers_violate_q8_pin` is the pure decision, pinned per-linear by
   `windowed_arch_q8_pin_rejects_any_non_q8_layer_linear` with an all-Q8_0 control.
3. **No gemma3 chat test on a runnable-FALLBACK host.** CONFIRMED and worse: there was no
   HTTP-level gemma3 *chat* test on ANY host — the only gemma3 HTTP tests hit `/completion`.
   FIXED: `gemma3_chat_routes_to_the_runnable_bridge_on_a_fallback_host` (plain `#[test]`, no
   cfg gate, so it runs on every non-macOS leg — the legs where it matters).
4. **Two of three H4 guard sites revert-invisible.** CONFIRMED and worse — all three were, and
   so was the choke point once it absorbed them. FIXED by the ten tests in 12a; the three early
   guards are now pinned BY LANE NAME, which is what makes their removal visible rather than
   silently absorbed.
5. **catalog_id/row_id mismatch.** CONFIRMED. `catalog_id: "gemma3_1b_it_q8_0"` vs row id
   `"gemma_3_1b_it_q8_0"` made `filename_is_supported_exact_row` false for the one promoted
   gemma3 row, so `classify_model_lane` returned `ExperimentalImplemented` — which both emitted
   `lane:"experimental"` AND whitespace-trimmed the generated text. For a supported exact row
   that text is the parity-claimed artifact and must stay byte-identical, so this was a live
   wrongness on the served surface, not only a label. FIXED here (one line, matching the rule
   the neighbouring qwen3 entry states verbatim). It needs NO ledger regen: the ledger is
   derived from the capabilities literal + API_CONFORMANCE_CASES, and `catalog_id` appears in
   neither; `check-ledger-drift.mjs` passes. The capabilities-ROW rewrite stays Phase 5's.
6. **`completions_fail_closed_for_runnable_serve_archs` reads the live predicate twice.**
   CONFIRMED: tautological (it asserted the gate equals itself) and genuinely racy on a Metal
   host, where seven sibling tests mutate `CAMELID_METAL_RESIDENT_DECODE` under a lock this
   test did not hold. FIXED by driving the pure core with explicit capability booleans — no
   env, no device, no lock needed, and it now pins behaviour instead of restating the
   implementation.
7. **Plan-vs-routing disagreement.** CONFIRMED for `CAMELID_MAC_Q8_METAL_PLAN=0`; CONFIRMED but
   differently-worded for the Safe profile (there the plan disclosed a bare "safe profile"
   reason next to CPU dense labels that for this arch fail closed at every dispatch — arguably
   a worse disclosure). FIXED: `execution_plan::macos_q8_metal_plan_selectable()` is now
   consulted by `inference::windowed_arch_resident_host_available()`, so the two agree, and the
   windowed "serve chats via the runnable bridge" reason is pushed alongside every early return
   that yields `safe_q8_plan()` rather than only the one after the Metal arm.
   **Landmine found while writing this fix, worth the record:** the first cut also consulted
   `CAMELID_MAC_Q8_REPACK`, which is a genuine third early return in `select_macos_q8_plan` —
   and a self-defeating latch. `CAMELID_MAC_Q8_REPACK` is a MANAGED_ENV_KEY that a successful
   Metal-resident selection WRITES to `"off"`, and `env_flag_disabled` reads `"off"` as
   disabled. Routing would therefore have decided the Metal plan was unselectable the instant
   `PlannerEnv::apply` ran, sending gemma3 to the bridge and killing the lane the plan had just
   selected — the whole Phase 3b result, undone by a "consistency" fix. The rule this yields:
   **a plan OUTPUT can never be a routing INPUT.** Only the two operator opt-outs
   (`CAMELID_PROFILE`, `CAMELID_MAC_Q8_METAL_PLAN`, neither plan-managed) are consulted, and
   `macos_q8_metal_plan_selectability_tracks_every_early_return` now pins that
   `CAMELID_MAC_Q8_REPACK=off` does NOT disarm routing, so it is not helpfully added back.
   Residual, OPEN: an operator who PRE-sets `CAMELID_MAC_Q8_REPACK=0` still gets a safe plan
   with resident routing. Closing it needs the plan to stop overloading one variable for both
   "operator opt-out" and "plan output".
8. **gemma3 tool refusal returns a different code/param across hosts.** CONFIRMED. Bridge: 422
   `unsupported_tools`, no param. Dense lane: raised inside template rendering and surfaced
   wrapped as `unsupported_chat_template` / param `"messages"`. Same status, different
   machine-readable identity, decided purely by host capability — a client switching on
   `error.code` degraded correctly on Linux and fell into a generic branch on the M4 mini.
   FIXED: one shared constructor (`gemma_runnable_lane_tools_rejection`), raised arch-keyed at
   the dense chokepoint ahead of tokenizer/template work.
   **§11f correction:** its claim that the dense-lane refusal is "behavior-IDENTICAL to pre-flip
   from the API user's perspective" was an overclaim — the refusal SEMANTICS were identical
   (422 class, tools never silently dropped, row stays `tool_capable: false`), the IDENTIFIERS
   were not. They are now.

### 12f. Nothing dismissed

No finding was dismissed. The one place a reported framing was corrected rather than accepted
is §12e-4 (the review said two of three guard sites were revert-invisible; all three were), and
§12e-2/3, where the confirmed defect is larger than described. F4's two suggested remedies were
both rejected in favour of a third, for the reason in 12d.

### 12g. Gates

Per commit: cargo fmt clean; `clippy --all-targets -D warnings` clean; `cargo test --all-targets`
exit 0 under pipefail. Final battery at `34bf672e`: **lib 1398 passed / 0 failed / 23 ignored**
(against 1376/0/23 at the flip commit `ba6de7f7` — +22 lib tests), 60 green integration tallies,
every suite green. `check-public-scrub.sh` clean. `scripts/check-ledger-drift.mjs` passed
(ledger == code contract; the `catalog_id` fix in 12e-5 touches neither the capabilities literal
nor API_CONFORMANCE_CASES, so no regen was needed — confirmed by the check, not assumed).

One pre-existing test needed a mechanical update: `tests/invariant_matrix_binding.rs`'s
I-cache-quant na anchor `"fn is_runnable_serve_arch"` became `"fn is_runnable_serve_file"`. The
anchor did its job — it caught the rename rather than letting the na verdict go stale.

Env-keyed battery (release, production GEMV configuration + resident decode armed FROM THE SHELL
per §9d, targeted names so the env-mutating gate tests never run alongside):

- **Phase 2 real-row final gate** (`gemma3_real_row_resident_forward_matches_runnable_oracle`),
  no SKIP, 585.18 s: depth 1 argmax 108 = oracle, max abs logit diff 6.247e-5; depth 5 argmax
  1077 = oracle, 7.820e-5; depth 50 argmax 578 = oracle, 9.584e-5 — **50/50 greedy tokens
  identical, overall max abs logit diff 2.122e-4**. Bit-for-bit the §9b / §10d / §11h record.
- **3a session-level prefill gate**
  (`gemma3_session_level_token_by_token_prefill_matches_runnable_oracle`), 22.20 s: depths 1-5
  argmax 108 / 584 / 568 / 2364 / 1077, all = oracle; **5/5 identical, overall max abs logit
  diff 7.820e-5**. Bit-for-bit the §10b / §11h record.

The whole Phase 2/3a stack is therefore proven UNCHANGED under the Phase 3c fixes.

### 12h. Serve smokes (release, this M4 mini 16 GB, no special env vars)

Resident (`camelid serve --model gemma-3-1b-it-Q8_0.gguf --no-open`):

- `/v1/health`: `generation_ready:true`, backend `"llama"` (dense serve lane, no runnable
  runtime), plan `metal_resident_q8_runtime` / `q8_0_metal_resident_decode` /
  `supported_exact_row_smoke_sub512`.
- Greedy chat, 20 prompt tok: coherent Rayleigh-scattering sentence, finish `stop`, 26 tokens.
  Wall 795.3 ms cold-ish / **684.0 ms warm** (§11g: 825 / 788 ms), token ids byte-identical
  across runs.
- Long greedy (26 prompt / 240 completion, two runs, byte-identical): wall 5486.7 / 5445.5 ms =
  43.7 / 44.1 tok/s end-to-end INCLUDING prefill; netting the ~26-token prefill leaves
  **≈46-47 tok/s decode** at depth ~266 — the §11g band (45.0 / 46.0 tok/s at depth ~289) with
  no regression.
- First 8 greedy token ids `[818, 7217, 7412, 3730, 1547, 529, 496, 20284]` — **IDENTICAL to
  the §11g oracle-verified head**, and identical to the fallback server's below.
- Tools: 422 **`unsupported_tools`, param absent** — the 12e-8 fix, verified live. §11g recorded
  `unsupported_chat_template` here; the two lanes now return the same object.
- `lane` field: **ABSENT** — the 12e-5 catalog_id fix, verified live. §11g recorded
  `lane:"experimental"`, which also meant the generated text was whitespace-trimmed; it is now
  served byte-identical, as a parity-claimed row must be.
- Raw `/v1/completions`: 200, `"The capital of France is"` → `" Paris.\n\nThe largest city in
  France is Paris.\n\n"` (the §11g continuation).

Fallback (`CAMELID_METAL_RESIDENT_DECODE=0`, same command):

- `/v1/health`: backend `"runnable-runtime"`; plan `cpu_reference` / `safe_cpu_decode` carrying
  the windowed-arch reason verbatim.
- Greedy chat 8 tok: 27.54 s (§11g: 27.07 s — the known ~0.2-0.3 tok/s runnable lane), token ids
  `[818, 7217, 7412, 3730, 1547, 529, 496, 20284]` — byte-identical to the resident lane's head,
  which is the cross-lane identity claim this campaign exists to hold.
- Tools: 422 `unsupported_tools`, param absent (unchanged; the dense lane moved TO this, not the
  other way).
- `/v1/completions`: 422 `unsupported_completions_lane` (gate intact verbatim).

Both servers killed by saved PID only.

### 12i. Still open after Phase 3c

- **Operator-set `CAMELID_MAC_Q8_REPACK=0`** still yields a safe plan with resident routing
  (12e-7). Needs the plan to stop overloading one variable for both operator opt-out and plan
  output.
- **`encode_attention` carries no `window_start`** (§9a residual). It is why speculation is
  declined for windowed archs (12d) rather than supported; threading it is the prerequisite for
  ever pointing spec decode at this row.
- **`scripts/chat-parity-gemma3.mjs` defaults `--row-id` to the old `gemma3_1b_it_q8_0`
  spelling**, so a receipt generated without an explicit flag carries a row id that matches no
  compatibility row. One line, deliberately left to Phase 5's docs/scripts sweep rather than
  widened into here.
- **`camelid pull` alias changed** as a consequence of 12e-5: the catalog id is now
  `gemma_3_1b_it_q8_0`. `README.md:227` already advertised a third spelling (`gemma3_1b`) that
  resolved to neither, so the README row is no more wrong than before — but it is wrong, and it
  belongs to the Phase 5 docs sweep.
- Phase 5's own inheritance from §11i is unchanged: the capabilities-row rewrite + ledger regen,
  frontend fixtures, docs sweep. The `lane`/trim half of that list is now CLOSED by 12e-5.

---

## 13. Phase 4 record (2026-07-30)

Phase 4 is the receipt phase: the SERVED resident lane compared against the pinned EXTERNAL
oracle, not against the runnable lane — whose oracle role expires at the 512-token window.
Bundle: `qa/evidence-bundles/gemma3-1b-q8-gpu-resident-parity-20260730-head-6eaf9053/`
(20 artifacts, README + manifest + SHA256SUMS).

### 13a. The oracle exists and is the pinned one

`llama-server` **version 9632 (`acd79d603`)** at
`acd79d603cb2e1c84c0886137b80f1ad649b6857`, clean tree, Release build; binary sha256
`382096b1dc10da68c2bf0a97e1f0dd36db90531cdea1434760d8c1a70fea1310`. Run on its **CPU** backend
(`-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 4096`) — the same comparator configuration the
frozen runnable bundle used, so the two gemma3 receipts share an oracle. Two-phase discipline
held for every capture, comparison and probe: the two engines were never resident together.

### 13b. Sub-512 receipts

**Chat (`scripts/chat-parity-gemma3.mjs`, committed gate pack, depths 1/5/50): CLEAN.**
Cross-engine prompt tokenization identical 5/5 (16/16/16/19/17 tokens); **15/15 generation legs
token-AND-text identical**; `all_pass: true`, zero flips. Note the contrast, stated carefully:
the frozen runnable bundle recorded one flip on this same pack (0.3416 nat); this bundle does
not re-adjudicate it — different host, different lane, and the runnable lane was not re-run on
the sub-512 pack here.

**Raw decode (`scripts/raw-decode-parity.mjs`, harness default 4 prompts, depths 1/5/50):
committed as-is with `all_pass: false`, 6/12 legs token-AND-text identical.** Two distinct
findings are inside that number:

1. A **harness re-encode artifact**, not an engine divergence. The harness scores camelid by
   re-encoding its TEXT; on this 262k SPM vocab a run of spaces re-encodes as single-space
   tokens (`236743`) where llama.cpp emitted merged whitespace tokens (`138`, `140`). Reading
   camelid's ACTUAL ids (`camelid.generated_token_ids`, `camelid-raw-probe.json`) removes it:
   `"Q: What is 2+2? A:"` is token-identical for all 43 generated tokens plus the `106` stop.
   The harness was NOT changed to hide this.
2. **Three real flips, all at depth 50**, each probed from both sides
   (`near-tie-analysis.json`):

| prompt | idx | ref | camelid | camelid top-2 gap | oracle top-2 gap (no-repack / repack) | oracle rank of camelid tok |
|---|---|---|---|---|---|---|
| `The capital of France is` | 44 | 9639 | 32219 | 0.0032 | 0.0431 / 0.0285 | **1** |
| `Once upon a time,` | 37 | 4658 | 11207 | 0.0173 | 0.4471 / 0.1398 | 2 |
| `def fibonacci(n):` | 5 | 2094 | 22304 | 0.0353 | 0.4402 / 0.4402 | **1** |

Two of the three are **oracle-side flips** — the same pinned binary and flags emit camelid's
token when the position is scored from a re-fed prefix and the reference token when it decodes
continuously from the raw prompt (`probe-oracle-continuous.json`); on prompt 0 the prefix-fed
oracle also flips back when only its repack kernel changes. The third (`Once upon a time,`) is
the weakest attribution: the oracle is rank-1-stable across all four kernel/thread controls AND
the continuous control, and camelid's token is oracle rank 2 at **0.4471 nat** — ABOVE the
0.33-nat Ornith line and above the frozen bundle's 0.3416. Disclosed as such, with the two
mitigating measurements stated rather than argued: camelid's own gap there is 0.0173 nat, and
the oracle's own gap moves 0.31 nat (0.4471 → 0.1398) when only its repack kernel changes.

Consequence for Phase 5, and it is a real one: **there is no token-exact depth-50 claim for raw
`/v1/completions` on this row.** Depths 1 and 5 are clean on 4/4.

### 13c. The >=512 windowed receipt — the deliverable this campaign exists for

New pack `qa/prompt-packs/gemma3-windowed-context-pack-v1.json`: three plain-English prompts
whose rendered gemma3 turns tokenize to **606 / 1205 / 2403 tokens** = 1.18x / 2.35x / 4.69x the
file's own `gemma3.attention.sliding_window = 512`, each answerable only from its first sentence.

**Resident lane vs the pinned oracle: 3/3 prompt tokenization identical, 9/9 generation legs
token-AND-text identical, ZERO flips, `all_pass: true`.** §7d required this pack to be clean and
it is — the envelope is not drawn on at all.

**The runnable lane on the same prompt, same oracle capture, same depth** (resident off,
`cpu_reference` / `safe_cpu_decode`): diverges at generated index 2 and never resynchronises —
oracle and resident emit `[818,103708,563,...]` "The Willow is the name of the river that runs
past the town."; the runnable lane emits `[818,103708,7940,236761]` "The Willow River.".
Prompt tokenization was identical (606/606) on both lanes, so they saw the byte-same input.
**And it is not a near-tie**: re-fed the identical prefix, the oracle ranks its own/resident
token `563` at logprob -0.2125 and the runnable lane's `7940` second at **1.667 nats** behind
(`probe-window-divergence.json`) — four times the largest disclosed near-tie in the bundle.
Attribution is stated as attribution: the two lanes are token-identical BELOW the window (13e's
two gates), and the one documented architectural difference is the window mask the runnable lane
does not implement.

Cost bound, recorded: the runnable lane prefills 606 tokens in ~10.8 min (~0.2 tok/s), so only
the shortest windowed prompt was run on it, at the single deepest leg. 1205/2403 on that lane
were not attempted.

### 13d. Determinism

Two fresh `camelid serve` processes, full stop/start between them, no env overrides, greedy.
Each session records 6 chat legs (incl. the 2403-token windowed prompt) and 5 raw-completion
legs carrying ACTUAL generated token ids (incl. a 2395-token windowed prompt).
`det-run1.json` and `det-run2.json` are **byte-identical files**, sha256
`632992c609941494905650a186ec255bf7d545950f4490aedc6a3c7158bf64d3`.

### 13e. Gates

- `cargo fmt --check` clean; `cargo clippy --all-targets -- -D warnings` clean;
  `cargo test --all-targets` **exit 0** under pipefail (every suite green, 0 failed).
- `scripts/check-public-scrub.sh` clean.
- All four bundle validators pass with the new bundle present:
  `check-public-scrub.sh`, `audit-evidence-bundle-privacy.mjs --strict` (0 findings),
  `check-evidence-bundle-checksums.sh`, `check-public-evidence-claims.mjs`
  (159 manifests, up from 158).
- Env-keyed gemma3 gates re-run at this head, targeted, release, production GEMV gates armed
  from the shell:
  - `gemma3_real_row_resident_forward_matches_runnable_oracle` (release, f32y+wire+NSG8):
    depth 1 argmax 108 = oracle, max |logit diff| 6.247e-5; depth 5 argmax 1077, 7.820e-5;
    depth 50 argmax 578, 9.584e-5 — **50/50 greedy tokens identical, overall max |logit diff|
    2.122e-4**, 565.79 s. **Bit-for-bit the §9b/§11h record.**
  - `gemma3_session_level_token_by_token_prefill_matches_runnable_oracle` (same gates plus
    `CAMELID_METAL_RESIDENT_DECODE=1`): 108 / 584 / 568 / 2364 / 1077, overall max |logit diff|
    **7.820e-5**, 5/5 identical, 22.05 s. **Bit-for-bit the §10b record.**

### 13f. Script fixes carried in this phase (receipt-label only, no engine behaviour)

`scripts/chat-parity-gemma3.mjs`: (1) `--row-id` now defaults to the real row id
`gemma_3_1b_it_q8_0` — this closes the §12i one-liner; (2) new `--lane-label` so the emitted
receipt names the lane it actually certified instead of hardcoding
`gemma3_marker_chat_greedy_runnable_serve`; (3) `postJson` moved from the global `fetch` to
`node:http` with a `--request-timeout-ms` flag, mirroring `scripts/raw-decode-parity.mjs` — the
undici ~5-minute header timeout aborted the client mid-request while the CPU runnable lane was
still legitimately prefilling a 606-token prompt, which is exactly the leg 13c needs.

### 13g. What Phase 5 MAY and MAY NOT claim

**MAY:**
- The row runs on the Metal GPU-resident serve lane by default and is token-AND-text identical
  to llama.cpp `acd79d603` on the committed chat gate pack at depths 1/5/50 (15/15), with 5/5
  cross-engine prompt tokenization.
- **Correct sliding-window behaviour above 512 tokens**, proven against the external oracle at
  606 / 1205 / 2403 prompt tokens, depths 1/5/50, 9/9 legs, zero flips. `tested_context` may
  move from "well under the 512-token sliding window" to a bounded chat claim of **2,403 prompt
  + 50 generated tokens**.
- Determinism: byte-identical decode across two fresh serve processes, including past the
  window.
- That the runnable CPU lane is measurably the wrong reference above the window (1.667-nat
  disagreement with the oracle at the divergence position) — i.e. the `full_support_blockers`
  sentence "context above the 512-token sliding window ... mathematically wrong by construction"
  now applies to the RUNNABLE lane only, and is CLOSED for the resident lane up to 2,403 tokens.

**MAY NOT:**
- No perf or throughput claim, and no speed comparison with llama.cpp.
- No token-exact claim for raw `/v1/completions` at depth 50 (three disclosed flips, one of them
  a 0.4471-nat stable-oracle near-tie). The clean raw-decode claim stops at depth 5.
- No context claim above 2,403 prompt tokens; the file's native 32,768 is UNMEASURED here, as is
  everything between ~2.4k and 32k.
- No claim for any other gemma3 row or quant (§7c scope pin holds).
- No multi-turn, streaming, tool-calling, speculative-decode or prefix-cache claim.
- The runnable-lane divergence leg is ONE prompt at ONE depth; it is a demonstration, not a
  runnable-lane receipt, and it must not be cited as a general runnable-lane characterization.
- The frozen runnable bundle's sub-512 0.3416-nat flip is NOT re-adjudicated by this bundle.

### 13h. Still open after Phase 4

- The raw-decode depth-50 near-ties are a live limit on the row's claimable surface; closing
  them would need either a chat-shaped raw pack or an oracle-side reduction-order study, and
  neither is scoped.
- §12i's other open items are unchanged except the `--row-id` default, which is CLOSED here.
- The runnable lane's cost makes a full windowed cross-lane matrix impractical; if a broader
  divergence table is ever wanted it needs a faster CPU reference, not more patience.
