# 26B-A4B Q4_0 ghost row — parity status against the llama.cpp oracle

**Status: NOT AT PARITY. 5 of 5 prompts diverge, on BOTH the CUDA-resident ghost lane and
the CPU ghost runtime.** Captured 2026-08-24. This file exists so the next person does not
read the presence of an oracle as evidence that the row passes it.

## What was captured

`google_gemma-4-26B-A4B-it-Q4_0.basic_v1.json` — the first committed oracle for this row.
Before it, `qa/gemma4/oracle/` held `gemma-4-26B_q4_0-it.*`, which is the **dense 26B QAT**
model, a *different row*. `gemma4_generation_parity` derives `row_id` from the model file
stem specifically to stop that mispairing; `CAMELID_GEMMA4_ROW` overrides that guard, so
only use it with an oracle that genuinely matches the file.

Reference: llama.cpp **build 10612 / commit 758443071**, `llama-server`, CPU,
`-ngl 0 --no-repack -fa off -ctk f32 -ctv f32 -ub 1`, `cache_prompt=false`, temp 0 / top_k 1.
Other oracles here were captured on the pinned `5d56eff`; this row used what is installed on
the Windows box. Re-capture reproduced all five sequences exactly, so the oracle is
deterministic and the divergences below are not capture noise.

## Result

| prompt | camelid CUDA ghost | camelid CPU ghost |
|---|---|---|
| capital-france | diverges @ idx 2 | diverges @ idx 2 |
| count-primes   | diverges @ idx 5 | diverges @ idx 5 |
| haiku-sea      | diverges @ idx 17 | diverges @ idx 11 |
| rust-fn        | diverges @ idx 1 | diverges @ idx 1 |
| translate-de   | diverges @ idx 2 | diverges @ idx 6 |

## What the divergences look like (this is the useful part)

1. **Not garbage — knife-edge flips.** `haiku-sea` on CUDA matches the oracle for 17 tokens,
   flips one (`16520…506, 6784` then `16254` vs oracle `20015`), and then **re-converges** for
   the remaining tokens. That is a near-tie argmax resolving differently, the same frontier
   class the `cpu_known_frontier` schema field was built for.
2. **★ camelid CPU and camelid CUDA agree with EACH OTHER while both differ from llama.cpp.**
   `capital-france` idx 2: both camelid runtimes emit `4800`, llama.cpp emits `236767`.
   `rust-fn` idx 1: both emit `148`, llama.cpp emits `140` (an indentation token).
   Two runtimes agreeing against the reference is **systematic**, not noise — this points at a
   camelid-vs-llama.cpp implementation difference in the gemma4 path, not at the ghost lane.
3. **CPU-vs-CUDA divergence is EXPECTED on this row and is not itself a defect.** This artifact
   carries **Q6_K on 14 of 30 `attn_q` tensors**, and repo precedent is explicit that K-quant
   GPU lanes are not bit-exact vs CPU (dense `q2k_gemv`'s own test asserts 1e-4 relative and
   only *reports* the bit-identical count). Do not gate this row on CPU/CUDA token identity.
4. Prompt tokenization matches the oracle on all five prompts — the tokenizer is not implicated.

## What this does and does not license

- It does **not** show the ghost lane is broken. The lane reproduces llama.cpp's behaviour on
  the same prompts qualitatively, and with the correct chat template the row answers correctly
  (verified: Rayleigh scattering, 26B ghost CUDA).
- It **does** mean two decisions currently rest on no clean receipt:
  - `gemma4_ghost_cuda_enabled` was flipped from opt-in to **default-on**, replacing a comment
    that said it stays opt-in "until the exact 26B row has a committed Windows parity receipt".
    That receipt now exists and the row does not pass it.
  - `Q6_K` is admitted to `RECEIPTED_PROJECTION_FORMATS` because this row carries it. The
    admission is honest about *why* (the row would otherwise be declined), but "receipted" is
    still doing work this row has not earned.

## Next step, if someone wants this closed

Find the first divergence and measure the top-2 logit gap at that position. If the gap is
knife-edge (~0.1% or less), record a `cpu_known_frontier` per prompt with the measured reason —
that is what the dense 26B row did ("2/5 full + 3/5 probe-verified frontiers"). If the gap is
wide at `rust-fn` idx 1 or `capital-france` idx 2, it is a real implementation difference and
should be chased in the gemma4 forward path, not the ghost cache.

Related: `qa/perf/HANDOFF.md` §7 item 2.
