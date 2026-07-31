# Phase 3 — gemma4 CUDA lane: default-on, gated, and honestly disclosed

## What changed

1. **Default-on.** `gemma4_cuda_enabled()` was opt-IN (`CAMELID_GEMMA4_CUDA=1`).
   It is now default-ON wherever CUDA is actually driving decode, with an
   explicit `CAMELID_GEMMA4_CUDA=0` opt-out.
2. **One admission predicate.** `execution_plan::gemma4_cuda_lane_admitted(gguf)`
   decides policy + quant + VRAM fit. Both the disclosed plan and the serve load
   site call it.
3. **A gemma4 arm in the execution plan.** gemma4 rows are served by their own
   runtime, so the generic Q8/K-quant plan arms described a lane they never take.

## The Phase 0 disclosure defect, fixed

Phase 0 measured the plan advertising `cuda_resident_kquant_runtime` /
`kquant_cuda_resident_decode` while serve ran the CPU `Gemma4Runtime` — 107 MiB of
VRAM in use while a 2.83 GB model generated. Now:

| Row | Plan `selected_backend` | Lane that ran | VRAM | Output |
|---|---|---|---|---|
| E2B Q4_0 | `gemma4_cpu_runtime` | CPU runtime | 0 MiB | `Paris` |
| E4B Q8_0 | `gemma4_cpu_runtime` | CPU runtime | 0 MiB | `Paris` |

…each with the decline reason carried in the plan's own `reasons`:

- E2B Q4_0 — *"CUDA-resident lane declined: gemma4 CUDA-resident decode is
  validated for Q8_0 layer projections only…"*
- E4B Q8_0 — *"CUDA-resident lane declined: allocation 8054 MiB exceeds free VRAM
  5122 MiB (would OOM mid-load); short by 3444 MiB including the 512 MiB min
  headroom"*

An intermediate revision of this phase split policy (plan) from quant+fit (load
site) and **immediately reproduced the Phase 0 defect** — the plan said
`gemma4_cuda_resident_runtime` for rows that fell back. That is why the predicate
is now singular, and why the split is called out in its doc comment.

## DEFECT FOUND: the gemma4 CUDA lane mis-decodes Q4_0

Making the lane default-on exposed a **pre-existing** bug that the opt-in default
had been hiding. Reproducible on an RTX 3060 Laptop, `gemma-4-E2B-it-Q4_0.gguf`,
prompt "Name the capital of France in one word.", greedy:

| Lane | Output |
|---|---|
| `gemma4_cuda_resident_runtime` | `passe dép oficialmenteynam shalthapp lenghtynam` |
| `gemma4_cpu_runtime` | `Paris` |

The lane ACCEPTS Q4_0 at load — its own `nvfp4_cuda_lane_check` lists
Q8_0/Q4_0/Q4_1/NVFP4 as covered — it just does not decode it correctly. This is
NOT caused by the default-on flip; the flip is what surfaced it. Admission is
therefore pinned to Q8_0 until the Q4_0 path earns a parity receipt.

## Outcome on THIS host (6 GB RTX 3060 Laptop)

| Row | Lane taken | Output | Wall (1 tok) | VRAM |
|---|---|---|---|---|
| E2B Q8_0 | **`gemma4_cuda_resident_runtime`** | `Paris` | **798 ms** | 2645 MiB |
| E4B Q8_0 | `gemma4_cpu_runtime` (fit decline) | `Paris` | 17,758 ms | 107 MiB |
| E2B Q4_0 | `gemma4_cpu_runtime` (quant decline) | `Paris` | 3,127 ms | 107 MiB |

E2B Q8_0 on the CPU runtime was 4,076 ms for the same request, so the GPU lane is
about 5x on this host. Recorded as an observation from a single unisolated
request, not a benchmark.

E4B Q8_0 projects 5231 MiB against 5122 MiB free and is short by 621 MiB
including the 512 MiB headroom floor — a real miss on a 6 GB card, and it would
take the GPU on an 8 GB one. The engine has no layer-offload path, so there is no
partial-residency fallback to reach for.

## The fit projection was wrong, and over-conservatism is not safety

The first version of this guard summed the WHOLE FILE's tensor bytes. For
E2B Q8_0 that projected **5055 MiB** and DECLINED the row — while the lane
actually uses **2635 MiB** and serves it in 794 ms.

gemma4 is a PLE matformer: its per-layer embedding tables (2406 MiB on E2B)
dwarf its per-layer projections (1879 MiB), and those tables do not go to VRAM.
More than half the file was being charged against a budget it never touches.

The projection is now per-layer projections plus a 1024 MiB overhead calibrated
against the measurement above, and the check is explicitly ADVISORY: the load
site falls back to the CPU runtime if `Gemma4CudaResident::load` returns an
error, so an optimistic projection costs a slower lane rather than a failed
request. That fallback did not exist in the first version either — a genuine OOM
would have propagated and 503'd a row the CPU runtime had been serving fine.

The lesson is worth keeping: a fit check that errs toward refusing silently keeps
working hardware on the CPU, and it fails on exactly the mid-size cards nobody
tests on.

## Separate defect, NOT fixed here

`serve --model <file>` runs a GPU warmup that holds **2538 MiB** (5122 → 2584
MiB free) before any model load. On this card that alone pushed E2B Q8_0 under
the fit line; loading the identical file via `/api/models/load` with no
`--model` flag left the room and took the GPU. It penalises constrained cards
specifically and deserves its own change.

## Not claimed

- No throughput number for the gemma4 CUDA lane.
- No parity receipt for gemma4 on CUDA — this phase changed routing and
  disclosure, not the forward.
- The Q4_0 defect is REPORTED, not diagnosed or fixed.
