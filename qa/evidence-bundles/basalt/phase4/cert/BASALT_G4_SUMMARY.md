# BASALT Gate G4 — NVFP4 CUDA decode: CERT + perf (measured, this box)

Status: **CERT PASS; perf reported (G4 has no pass/fail threshold — it is a measured receipt).**
**Awaiting Tim: scope decision — seal as-is vs the pre-registered dp4a kernel upgrade to
recover the speed the byte reduction should have bought.**

Engine: `basalt/phase4-cuda-decode` @ `8c2de5bb` (kernel impl `892672ca` + this CERT/perf).
Hardware: RTX 3060 Laptop, sm_86, 6144 MiB, driver 576.83, CUDA 12.9. All numbers
**measured on this box**, never general. Receipts: `qa/evidence-bundles/basalt/phase4/cert/`.

## 1. CERT — the wiring is correct end-to-end (PASS)

NVFP4-mm greedy, Camelid **CPU wire lane vs CUDA-resident lane**, 9-prompt lane-native pack:
**6/9 token-identical.** All 3 divergences are textbook near-tie argmax flips — in each the
CUDA token is exactly the CPU's **#2** candidate at a 0.047–0.111 raw-logit gap, attributable
to the accepted **CUDA f16-KV vs CPU f32-KV** difference (the same greedy-token contract the
shipping Q8_0 CUDA row runs under). No non-near-tie divergence → no kernel/wiring bug. The
first-ever full NVFP4 CUDA generation ran clean. Combined with the impl's 46/46 bit-identical
same-bytes gate, the kernel + raw-wire upload + dispatch + residency are proven correct.

## 2. Perf — measured (median of 5 warm runs, 128 greedy tokens)

| lane | tok/s | per-token read | achieved BW | % of 336 GB/s roofline | peak VRAM |
|---|---:|---:|---:|---:|---:|
| Q8_0 CUDA-resident (shipping baseline) | **25.80** | 5.182 GB | 133.7 GB/s | **39.8 %** | 5559 MiB |
| **NVFP4-mm CUDA-resident** (new) | **14.64** | 3.05 GB | 44.6 GB/s | **13.3 %** | **3479 MiB** |
| NVFP4-mm CPU wire lane (reference) | 1.57 | — | — | — | — |
| pin-GPU (cross-engine context) | skipped | — | — | — | — (6.06 GB full-offload unsafe on 6144 MiB) |

## 3. The honest headline — and what it means for Option B

You chose **Option B** (continue on **space/speed** grounds, quality cost disclosed). G4
splits that thesis cleanly:

- **SPACE: confirmed.** NVFP4-mm resides in **3479 MiB vs Q8_0's 5559 MiB — 2.08 GB more
  free VRAM.** Per-token weight read shrinks a measured **1.70×** (format-isolated 1.647×,
  matching the pre-registered ~1.6×; the extra comes from an incidental inp_gate/proj
  precision difference between the two files, disclosed).
- **SPEED: not at v1.** NVFP4-mm CUDA runs at **0.57× the Q8_0 lane's speed** (14.64 vs
  25.80 tok/s) *despite* moving 1.70× fewer bytes — because the v1 **scalar-LUT dequant
  kernel is compute-bound**: 13.3 % of the DRAM roofline vs Q8_0's 39.8 %. The bytes are
  there to be saved; the kernel isn't fast enough to cash them in.

This was **pre-registered**, not a surprise sprung after the fact: the Phase 4 recon (§5 Q1)
chose the scalar-LUT kernel for v1 *and* said the `__byte_perm` + `__dp4a` inner-loop upgrade
should be adopted "**only if the perf receipt shows the kernel is not already at the memory
roofline.**" The receipt shows exactly that (13.3 % ≪ roofline). The dp4a upgrade is
**parity-neutral** (identical i32 dot result — cannot move the 46/46 bit-identity) and is now
the warranted next bite.

**Claim-lint consequence (binds Phase 6):** no NVFP4 surface may say "faster." Today's
realized NVFP4 win on this box is **VRAM headroom, not decode speed** — that is the honest
statement every surface carries, alongside the G3 quality delta.

## 4. Decision for Tim (scope — I will not choose this silently)

- **(A) Seal G4 as-is.** NVFP4 on this box = correct, ~2 GB more VRAM headroom, ~0.57× the
  decode speed of Q8_0 at v1. Proceed to Phase 5 (BLOCKED-HW, no Blackwell — a record) and
  Phase 6 (surface alignment carrying the honest quality + speed story). Cleanest close; the
  speed gap is documented as a known v1 limitation with a named fix.
- **(B) Phase 4b — the dp4a kernel upgrade first.** Pre-registered, parity-neutral, and
  warranted by the 13.3 %-roofline measurement. Target: move the NVFP4 kernel toward the
  memory roofline so the 1.70× byte reduction actually shows up as speed (best case ~roofline
  would put it *above* Q8_0's tok/s since it moves fewer bytes). Then re-measure and seal G4
  on the upgraded kernel. This is the choice that makes the "speed" half of Option B real.
- **(C) The gpu_head lever** (bigger): the tied Q8_0 head is ~23 % of NVFP4-row bytes and
  stays resident. Attacking it (NVFP4 head / NVFP4-all residency) compounds the space win and
  the per-token byte reduction. Larger scope; a follow-on campaign bite, not a quick fix.

My read: **(B)** is the honest way to test what Option B was chosen for — the byte reduction
is real and the fix is pre-registered and parity-neutral, so the speed question deserves the
upgraded kernel before sealing. (A) is legitimate if "correct + VRAM-saving" already meets the
goal and speed was never the point. (C) is a separate campaign. Either way the G4 receipts
stand as measured.

## 5. Also still awaiting your signature (carried, non-blocking)
- **D-B6 (TK3)** per-tensor admission — draft banked (`scratchpad/basalt-db6/`), Option A
  recommended (~5-line change).
- **§2.4 matrix-mechanism deviation** — disclosed in DECISIONS.md, needs your explicit nod.

## 6. Safety note
~20 GPU model loads, **zero incidents**: one GPU process at all times, VRAM checked before
every load and verified freed to 0 after every one; peak 5559 MiB (Q8_0); box clean at exit.
The pin-GPU comparator was **skipped on memory-safety grounds** (6.06 GB full-offload on a
6144 MiB card is not comfortable headroom) — a deliberate omission, not a failure.
