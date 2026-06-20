# Resident draft-model path — diagnosis + fix + honest 6 GB verdict

Follow-up to Phase 1 Path B / Phase 3: "fix the resident draft path so the 0.6B stays on GPU."
Date (UTC): 2026-06-20. Machine: RTX 3060 Laptop, 6 GB (≈5122 MiB free after desktop), CUDA 12.9.

## Diagnosis (why the 0.6B fell to CPU)

With `CAMELID_RESIDENT_TRACE=1` the draft engine build failed:

```
[resident-cuda] VRAM sizing: free 482 MiB, weights 639 MiB ... headroom 512 MiB ... fits 0 pos -> cap 0
[resident-cuda] VRAM too small for resident decode even with 28/28 layers offloaded (cap 0 < 256); using CPU path
```

Root cause: the 4B target builds its resident engine FIRST and sizes its KV greedily (cap 1045,
~294 MiB), taking free VRAM down to **482 MiB**. The 0.6B draft then can't satisfy the fixed
**512 MiB headroom** the sizing reserves (and the `cuda_vram::evaluate` backstop's 512 MiB
post-load floor), so it builds 0 KV positions and falls back to the CPU forward (~253 ms/token).
Not a model/eligibility problem — purely VRAM budgeting between two engines.

## Fix (committed)

Three changes, all default-off / gated so the single-model path is byte-for-byte unchanged:

1. **Speculative-coexistence VRAM policy** (`build_resident_cuda_engine` + a process reserve set
   by `ModelDrafter::new`). When a draft is in play, the **target reserves the draft's footprint**
   (weights + a capped KV at `CAMELID_SPEC_DRAFT_CONTEXT`, default 512, + a small margin) so the
   draft can build **fully GPU-resident** beside it; the draft engine itself uses a small headroom
   and a capped KV. **Honored only when the target still fits FULLY resident after the reserve** —
   see the gate below.
2. **The offload gate.** If honoring the reserve would force the target to **offload** trailing
   layers, the reserve is dropped and the target builds full-resident (draft → CPU, the prior
   lossless behavior). Two reasons offloading the target is the wrong trade: it slows the target
   forward ~3× (measured 41 → 14 t/s with 5 layers offloaded), and it triggers (3).
3. **Verify-offload correctness guard** (`verify_drafts_gpu`). The batched verify
   (`run_batched_layer_stack`) reads each layer's resident VRAM slice directly; for an **offloaded**
   layer that slice is a 1-byte placeholder (real weights stream into scratch only on the
   single-token path). A batched verify over an offloaded target therefore read garbage and broke
   losslessness. Now `verify_drafts_gpu` returns `None` when the engine `is_offloaded()`, routing
   to the lossless CPU chunk verify. (Latent bug — the resident draft path had never actually run
   before, so it was never exposed.)

## Measurements

**The mechanism works** — with the reserve engaged (gate temporarily bypassed for the test), the
0.6B built **fully GPU-resident** and draft latency dropped **253 ms/tok → 13 ms/tok (≈19×)**:

```
[resident-cuda] VRAM sizing: free 994 MiB, weights 639 MiB (resident 639 MiB; 0/28 offloaded) -> cap 512   ← draft fully resident
draft 13.2 ms/tok | f_draft 0.55   (was 253 ms/tok, f_draft 0.90)
```

But it required the target to offload 5 layers, which (a) dropped the target to 14 t/s and (b) hit
the offloaded-verify path → with the guard, verify falls to CPU; without it, output diverged. Net
on 6 GB: **still slower than plain full-resident decode**, so the gate (2) correctly **declines**:

| path (this box) | target | draft | verify | result |
|---|---|---|---|---|
| matrix / default | full GPU-resident (41 t/s) | CPU (~253 ms/tok) | GPU batched | **lossless ✓, draft-bound (unchanged)** |
| coexist, gate ON (6 GB) | full GPU-resident | CPU (declines) | GPU batched | **lossless ✓ — same as above (no regression)** |
| coexist, gate OFF (forced) | 5 layers offloaded (14 t/s) | **GPU-resident (13 ms/tok)** | CPU (after guard) | lossless ✓ but net-slower; not a win |

Regression check (gate on): n-gram 1.09× lossless ✓; draft-gpu matrix lossless ✓, GPU verify, target
full-resident — identical to the committed Phase-1 behavior.

## Honest verdict — the 0.6B cannot stay on GPU on THIS 6 GB box

The two models' **weights alone are 4315 + 639 = 4954 MiB**; with CUDA context/kernel/scratch
overhead (~200–300 MiB) that already approaches the **5122 MiB free**, leaving no room for either
KV cache. So Camelid cannot hold both **fully** resident here — and any target **offload** to make
room is a net loss (slow target) and forces verify off the fast GPU path. The fix is therefore
**correct but dormant on 6 GB**: it engages on a GPU with enough free VRAM to hold both fully
resident (≈8 GB+), where the draft is ~free (13 ms/tok) and verify stays GPU-batched and lossless.

This matches Phase 3: llama.cpp wins on the same 6 GB box because its **f16 KV cache** (half of
Camelid's f32) plus lower overhead let both models sit resident. The real 6 GB unlock for Camelid is
**f16 resident KV** (a resident-attention kernel change) — and even then the margin is thin
(weights + overhead alone ≈ 5.2 GB). Until then, on ≤6 GB the lossless draft-model path stays
draft-bound; **n-gram remains the shippable win** (Phase 1/4).

## Reproduce

```
# draft fully resident only engages where both models fit resident (≥~8 GB free):
camelid bench-speculative <4B> --drafter draft --draft-model <0.6B> --spec-only \
  --draft-tokens 4 --workload code --max-tokens 128 --prompt-file prompts/code.txt
# CAMELID_RESIDENT_TRACE=1 shows the coexist gate decision + per-engine VRAM sizing.
# CAMELID_SPEC_DRAFT_CONTEXT caps the draft KV (default 512).
```
