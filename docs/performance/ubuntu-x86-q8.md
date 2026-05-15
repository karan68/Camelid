# Ubuntu x86 Q8 Performance Work

## Status

This is an active, evidence-gated performance lane. Optimized paths are default-off while validation continues.

The current goal is production-directional runtime improvement on a narrow Ubuntu x86_64 dense-Q8 lane. It is not a production-ready or broad platform claim.

## What changed

- Added and validated default-off AVX2 Q8 acceleration paths in the measured Ubuntu x86 lane.
- Added packed Q8 runtime storage work for selected dense tensors.
- Added matrix-level Q8 execution experiments and deeper FFN ownership slices.
- Separated cold materialization from warm inference.
- Documented rejected candidates when they failed parity, wall-clock, or clean-host discipline.

## Validated principles

- Warm inference should not rebuild packed rows.
- Cold materialization and warm decode are separate problems.
- `from_q8_0_bytes` is cold/reload-only on the measured Ubuntu x86 lane.
- Row-dot micro-optimizations are not enough by themselves.
- Matrix-level ownership is the current direction.
- Retain decisions require parity plus repeated wall-clock evidence on a clean host.

## Current retained paths

Only list the paths that are currently evidence-backed and default-off:

- `CAMELID_X86_Q8_REPACK=on` for the retained Ubuntu x86 runtime-packed lane used in current evidence.
- AVX2 packed-kernel work in the measured Ubuntu x86 lane where parity and bounded timing evidence support keeping the path under default-off gating.
- Packed Q8 runtime storage for the dense attention projection family plus dense FFN gate/up/down rows in the measured lane.

## Active experimental direction

Current work is focused on:

- AVX2 scaled row-dot and packed-kernel execution
- matrix-level Q8 GEMM/MUL_MAT ownership
- FFN projection optimization, especially deeper `ffn_down` decode ownership
- attention projection optimization
- reducing wrapper/callback overhead in hot inference
- keeping the default/reference path safe while experimental paths stay opt-in

## Rejected paths

Rejected paths stay documented when they fail for any of the following reasons:

- no wall-clock win
- parity fail
- contaminated host
- microbench-only improvement
- old-baseline-only improvement
- context-switch regression

Examples already treated this way include row-dot lookalikes, tile16 hsum/lane/simd-scale variants, wrapper-style GEMM detours, and contaminated benchmark runs.

## Clean-host discipline

Ubuntu x86 Q8 benchmarking now requires a clean host before major runs:

- check disk headroom
- inspect stale Camelid / perf / benchmark jobs
- clear conflicting ports
- remove abandoned scratch trees from invalid runs
- preserve retained evidence and model files

Contaminated runs are not used as retained evidence.

## How to reproduce

Use only the currently validated default-off gates for the retained baseline and keep the host clean before running:

```bash
CAMELID_X86_Q8_KERNEL=avx2_scaled_rowdot \
CAMELID_X86_Q8_REPACK=on \
CAMELID_X86_Q8_REPACK_TENSORS=ffn,attention_projection \
CAMELID_X86_Q8_GEMM=avx2_tiled \
CAMELID_X86_Q8_PACKED_TILE16=on \
CAMELID_X86_Q8_PACKED_TILE16_SERIAL_OWNER=on
```

For bounded warm-request measurement, use the same host, same model, same request shape, repeated runs, parity checks, and paired `perf stat` / `perf record` evidence.

## Evidence bundles

Primary public evidence anchors for this lane:

- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/README.md`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260515T1108Z-x86-attn-family.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260515T1235Z-x86-ffn-down-runtime.txt`
- the retained/reject notes for bounded Ubuntu x86 Q8 experiments kept under `qa/evidence-bundles/`

## Product/runtime note

Camelid is moving toward an appliance-style execution plan where validated runtime paths can be selected automatically while experimental acceleration remains opt-in.

The intended product/runtime mode split is:

- `safe`
- `auto`
- `experimental`
- `debug`

This note does **not** claim full multi-model runner orchestration today. It describes the direction for exposing validated runtime behavior more clearly without pushing env-var complexity onto normal users.
