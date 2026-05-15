# Ubuntu x86 Q8 acceleration progress — 2026-05-15

This note summarizes the current public state of Camelid's Ubuntu x86 Q8 performance investigation.

## Scope

This lane is intentionally narrow:

- Ubuntu x86_64 only
- dense Llama-family Q8_0 measurements only
- default-off experimental acceleration paths only
- no portability, production-throughput, or broader support claim

The current work is production-directional, not production-ready.

## What has improved

Recent Ubuntu x86 Q8 work has significantly improved the experimental accelerated path while keeping the default/reference path intact.

Key themes in the retained work so far:

- packed Q8 runtime storage for selected dense linears
- AVX2 packed kernel work on the measured hot path
- matrix-level execution work instead of single micro-kernel hype
- attention projection improvements
- FFN ownership/runtime-storage follow-up
- checksum and parity validation before performance claims move
- repeated benchmark discipline with explicit retain/reject notes
- safe fallback behavior preserved when experiment gates are off

## Current experimental gates

These paths remain default-off while validation continues:

- `CAMELID_X86_Q8_REPACK=on`
- `CAMELID_X86_Q8_KERNEL=avx2`

Fallback behavior remains intact when the gates are absent, disabled, or unsupported on the current host.

## Evidence-backed progress

Current evidence bundles show:

- parity-clean default-off AVX2 packed-kernel candidates
- packed runtime storage for the dense attention projection family
- FFN gate/up/down follow-on runtime-storage ownership work
- repeated timing and perf-symbol evidence for retained candidates
- rejected experiments documented instead of silently dropped

One bounded same-host smoke result in the active evidence bundle showed a large first-token improvement for the default-off accelerated path while preserving the measured output token. That result is kept as bounded evidence only; it is not a broad throughput or support claim.

## What is not being claimed

This lane does **not** currently claim:

- default-on acceleration
- broad production throughput
- portability beyond the measured Ubuntu x86_64 lane
- broader model-family support
- neighboring-row support
- production-ready status

## Current technical read

The measured warm-path bottleneck is still centered on packed Q8 dot and FFN compute work.

Recent audit work also clarified that packed-row conversion appears to be a cold materialization cost rather than a proven warm decode bug:

- packed rows are built during first-use materialization for selected tensors
- hot inference borrows backend-owned packed runtime storage
- attention projection and FFN now share the same packed ownership model in the measured lane
- the next evidence split is cold materialization versus warm inference, not more micro-variant tuning by default

## Current discipline

Camelid is intentionally keeping the bar high for retaining performance work:

- no win is retained without repeated confirmation
- checksum/text preservation must hold
- rejected candidates stay documented
- no support/API/frontend wording changes are made from performance experiments alone

## Primary evidence anchors

- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/README.md`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260515T1108Z-x86-attn-family.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260515T1235Z-x86-ffn-down-runtime.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/benchmarks/unique-chat-baseline-1tok.json`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/benchmarks/unique-chat-x86-repack-avx2-1tok.json`

## Next step

The immediate next proof is the cold/warm request matrix:

1. first cold request after model load
2. second warm request on the same resident model
3. third warm request on the same resident model
4. forced reload if supported

That split will determine whether any remaining packed-row work belongs in a cold-start lane or whether the warm-inference lane should stay focused on the AVX2 packed-dot and FFN compute path.
