# ORNITH 9B — G-PARITY receipt, resident Metal lane (qwen35 vs llama.cpp acd79d6)

**Gate:** G-PARITY — greedy token-identical vs the pinned llama.cpp oracle
(`first_divergent_generated_token_index = -1`). **Result: PASS (4/4 prompts).**
**Date:** 2026-08-06 · **Platform:** macOS 26.6 (25G72) arm64, Apple M4, 16 GB.
**Camelid lane:** runnable → **resident Metal graph** (`qwen35`), wire mode
(`CAMELID_METAL_F32Y=1`, `CAMELID_METAL_WIRE=1`), rustc 1.95.0, `--release`.
**Oracle:** llama.cpp `acd79d6` (build 9632) — **reused, not re-run**; see Method.
**Model:** `ornith-1.0-9b-Q8_0.gguf` (arch `qwen35`).

Companion to `G-PARITY-qwen35-vs-llamacpp.md`, which certified the same gate on
Windows x86_64 CPU. This one covers a different lane on a different platform.

## Method (isolates the model forward from tokenization)

Same method and the same four prompt token-ID arrays as the Windows receipt, so the
two are directly comparable:

```
[[3710,369,279,6511,314,9338,30],
 [727,73111,1393,1590],
 [760,13600,314,3882,369],
 [2427,310,4097,25,220,16,11,220,17,11,220,18,11]]
```

Greedy, `n_predict=20`, via the existing `ornith_qwen35_parity_gen` harness.

**The oracle was not re-run for this receipt.** The reference token IDs are the ones
already pinned in `G-PARITY-qwen35-vs-llamacpp.md` from the llama.cpp `acd79d6` run.
That is the point of a pinned oracle — this receipt asks whether a *new Camelid lane*
reproduces it, not whether llama.cpp is still self-consistent. Anyone re-certifying the
oracle itself should start from the Windows receipt's method section.

**The resident lane was confirmed engaged, not assumed.** `ornith_qwen35_parity_gen`
exits 0 and prints plausible token IDs even when the Metal lane declines and falls back
to CPU decode, so the run is only valid if stderr carries:

```
[qwen35] full Metal resident graph active (packed weights, attention,
         gated-delta recurrence, FFN, logits, GPU greedy, and request sampling)
```

and does **not** carry `using hybrid fallback`. Both were checked for this run.

## Results — generated token IDs, Metal resident lane vs pinned oracle

| # | prompt | tokens | match |
|---|--------|--------|-------|
| 0 | `What is the capital of France?` | `[271,760,6511,314,9338,369,11751,13,271,3710,369,279,6511,314,9564,30,271,760,6511,314]` | ✅ identical |
| 1 | `def fibonacci(n):` | `[198,262,413,307,2564,220,15,25,198,285,460,2958,198,262,4265,307,606,220,16,25]` | ✅ identical |
| 2 | `The opposite of hot is` | `[8981,13,271,248068,198,90700,8340,25,271,16,13,220,2972,2014,53983,279,5952,64700,198,262]` | ✅ identical |
| 3 | `Count to five: 1, 2, 3,` | `[220,19,11,220,20,13,4543,11,1092,3905,1727,30,271,760,1727,1324,303,279,8240,369]` | ✅ identical |

All four token-identical → `first_divergent_generated_token_index = -1`. Byte-for-byte
equal to the Windows CPU receipt's IDs, so CPU and Metal agree with each other as well
as with the oracle.

## What this earns

- The **resident Metal lane** reproduces the pinned oracle for `qwen35` Q8_0 on Apple
  Silicon. The gated-delta-net recurrence, causal conv1d, GQA head-repeat, state
  orientation, partial NEOX mRoPE and gated attention are all correct **as executed on
  the GPU**, not only in the CPU reference.
- Admitting GGUF `Q8_0` into the resident lane with no repack does not perturb the
  forward. Same 34-byte wire blocks, same argmax trajectory, all four prompts.
- A macOS arm64 data point for a gate that previously existed only on Windows x86_64.

This is lane/platform evidence for an existing row. It is **not** a support-contract
change and claims nothing about rows outside `qwen35` Q8_0.

## Caveats

- **Not a benchmark.** No warmup, and load time is inside the wall clock, so the
  numbers here are not comparable to `bench-generate` figures. For reference only: this
  run took 14.7 s end to end; the same harness on the CPU hybrid lane took 221.8 s. A
  cold process also pays one-time Metal pipeline compilation, which lands inside the
  first prefill — see the `--warmup` note in the PR discussion.
- **Wire mode must be set explicitly under the test harness.** `CAMELID_METAL_F32Y`
  and `CAMELID_METAL_WIRE` default off; only the CLI turns them on via
  `apply_default_fast_stack`. A bare `cargo test` therefore does *not* exercise this
  lane, and wire-page weights require it.
- **Known deviation carried from G-LOAD:** BPE pre-tokenizer `qwen35` is implemented as
  qwen2's single-digit split, with `\p{M}` combining-mark folding deferred. The parity
  method feeds token IDs, so this does not affect the model-forward result here.

## Reproduce

```sh
CAMELID_ORNITH_GGUF=/path/to/ornith-1.0-9b-Q8_0.gguf \
CAMELID_METAL_F32Y=1 CAMELID_METAL_WIRE=1 \
CAMELID_PARITY_TOKENS='[[3710,369,279,6511,314,9338,30],[727,73111,1393,1590],[760,13600,314,3882,369],[2427,310,4097,25,220,16,11,220,17,11,220,18,11]]' \
CAMELID_PARITY_NPREDICT=20 \
cargo test --release --lib ornith_qwen35_parity_gen -- --ignored --nocapture
```

Check stderr for `full Metal resident graph active` before trusting the IDs. To produce
the CPU control from the same binary, add `CAMELID_QWEN35_METAL=0`.
