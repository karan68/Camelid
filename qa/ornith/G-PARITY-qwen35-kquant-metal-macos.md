# ORNITH 9B Q4_K_M — G-PARITY receipt, resident Metal lane (macOS arm64)

**Gate:** G-PARITY — greedy token-identical between the resident Metal lane and the
CPU control on the same bytes. **Result: PASS (4/4 prompts).**
**Date:** 2026-08-07 · **Platform:** macOS arm64, Apple M4 (Mac16,10), 16 GB.
**Camelid lane:** runnable → **resident Metal graph** (`qwen35`, K-quant), wire mode
(`CAMELID_METAL_F32Y=1`, `CAMELID_METAL_WIRE=1`), `--release`.
**Model:** `ornith-1.0-9b-Q4_K_M.gguf`, **sha256 `5720d1f6…`**, 5,629,108,704 bytes.

Companion to `G-PARITY-qwen35-metal-macos.md` (same lane, Q8_0 weights). This one
certifies the **K-quant** admission: Q4_K/Q6_K super-blocks consumed at wire size by
the resident Metal graph with no repack.

## Artifact identity — read this first

These bytes are **NOT** the certified `ornith_1_0_9b_q4_k_m` row. That row is sha256
`2711bf1e…` / 5,629,108,416 bytes (an in-house requant, no imatrix — see
`REFERENCE_PIN_QWEN35.md`). The file measured here is the HuggingFace imatrix quant:
**different weights, not just different metadata.**

This receipt therefore covers a *distinct artifact* and makes **no claim** about the
certified row. It was run against a path renamed to break the filename match, because
row classification keys on filename alone and would otherwise report the certified
row's claims for these bytes.

## Method

Four prompts as token-ID arrays (the same four as the Q8_0 receipts, so the three are
directly comparable), greedy, `n_predict=20`, via the existing `ornith_qwen35_parity_gen`
harness. Feeding token IDs isolates the model forward from tokenization.

```
[[3710,369,279,6511,314,9338,30],
 [727,73111,1393,1590],
 [760,13600,314,3882,369],
 [2427,310,4097,25,220,16,11,220,17,11,220,18,11]]
```

**The control is camelid's own CPU lane on the same bytes and the same binary**
(`CAMELID_QWEN35_METAL=0`), following the Q3_K_M precedent. The pinned llama.cpp
`acd79d6` oracle was **not** re-run on these weights — see the frontier below.

**The lane was confirmed engaged, not assumed.** The harness exits 0 and prints
plausible IDs even when the Metal lane declines and falls back to CPU. Both runs were
checked: the Metal run carries `[qwen35] full Metal resident graph active` and no
`hybrid fallback`; the control run carries neither, and took 112.3 s against the Metal
run's 16.1 s end-to-end — a 7× wall-clock separation that independently confirms two
different lanes ran.

## Results — generated token IDs

| # | Metal vs CPU control | Metal vs Q8_0 reference IDs |
|---|---|---|
| 0 | ✅ identical | ✅ identical |
| 1 | ✅ identical | ✅ identical |
| 2 | ✅ identical | ✅ identical |
| 3 | ✅ identical | ✅ identical |

**4/4 token-identical to the CPU control** → `first_divergent_generated_token_index = -1`.
That is the gate, and it passes.

**4/4 also match the Q8_0 reference token IDs** pinned in
`G-PARITY-qwen35-vs-llamacpp.md`. This is a cross-check, **not a cross-engine
certification** — see the frontier.

## What this earns

- The resident Metal lane consumes GGUF `Q4_K`/`Q6_K` super-blocks correctly at wire
  size. The gated-delta-net recurrence, causal conv1d, GQA head-repeat, state
  orientation, partial NEOX mRoPE and gated attention are all correct as executed on
  the GPU with K-quant weights.
- Admitting K-quants to the loader does not perturb the forward: same argmax
  trajectory as the CPU path on the identical bytes, all four prompts.
- **A known sub-ULP kernel divergence does not reach the tokens.** The widened
  `metal_kquant_resident_projection_matches_cpu_oracles` fixture shows Q6_K at n_sb=48
  differing from the CPU oracle by exactly 1.2207e-4 = 2⁻¹³, bit-identical across
  `q6k_linear_simd` and `q6k_linear_tiled`. `ffn_down` is Q6_K at n_sb=48 on every
  layer, so that divergence is on the live path — and it flipped no token here.

## Documented frontier

**No direct llama.cpp receipt exists on these exact bytes.** The oracle IDs used for
the cross-check above were generated from the **Q8_0** weights. llama.cpp running
*this* Q4_K_M file could produce different IDs, so the agreement shows that the Q4_K_M
quantization preserves the Q8_0 greedy trajectory on these four prompts — evidence of
quantization robustness, not a certified cross-engine result. Closing this needs
llama.cpp `acd79d6` run on these bytes; the two engines cannot be co-resident on
16 GB, so the oracle must be shut down first.

Precedent for how a flip would be handled if one appeared: the CUDA Q4_K_M receipt is
2/5 token-identical at n=64 under the cross-backend tolerance policy, with every flip
probed and attributed to ≤0.33-nat soft positions.

## Caveats

- **Not a benchmark.** Load time is inside the wall clock and there is no warmup. For
  reference only, separately measured with `--warmup` on a quiet box, model loaded
  alone: decode **11.3 tok/s** (vs 10.3 for Q8_0 on the same machine),
  `phys_footprint` **6261 MB** (vs 9917 MB).
- **The win here is memory, not speed.** At 5.04 GB/token the decode ceiling on this
  M4 is 23.8 tok/s; 11.3 is ~47% of peak against the Q8_0 lane's ~72%. The K-quant
  decode kernel dispatches 32 threads with its f32 tail serialised on lane 0, where
  the Q8_0 path uses a 128-thread k-split. Kernel-shape work, tracked separately.
- **Prefill is unchanged and still slow.** No prefill claim is made.
- **Wire mode must be set explicitly under the test harness.** `CAMELID_METAL_F32Y`
  and `CAMELID_METAL_WIRE` default off outside the CLI, so a bare `cargo test` does
  not exercise this lane.
- **Q5_K is not admitted** — there is no `q5k` Metal kernel, so ornith Q3_K_M keeps
  failing closed to CPU. Nothing here applies to it.
- Known deviation carried from G-LOAD: BPE pre-tokenizer `qwen35` is qwen2's
  single-digit split with `\p{M}` folding deferred. The method feeds token IDs, so it
  does not affect this result.

This is lane/platform evidence for a **distinct artifact**. It is not a support-contract
change and promotes no row.

## Reproduce

```sh
CAMELID_ORNITH_GGUF=/path/to/ornith-1.0-9b-Q4_K_M.gguf \
CAMELID_METAL_F32Y=1 CAMELID_METAL_WIRE=1 \
CAMELID_PARITY_TOKENS='[[3710,369,279,6511,314,9338,30],[727,73111,1393,1590],[760,13600,314,3882,369],[2427,310,4097,25,220,16,11,220,17,11,220,18,11]]' \
CAMELID_PARITY_NPREDICT=20 \
cargo test --release --lib ornith_qwen35_parity_gen -- --ignored --nocapture
```

Check stderr for `full Metal resident graph active` before trusting the IDs. Add
`CAMELID_QWEN35_METAL=0` for the CPU control. Run the two serially — 16 GB cannot hold
both, and the process-global Metal buffer cache never evicts.
