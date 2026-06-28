# BENCHMARK_TREATY.md — binding rules for every Camelid-vs-llama.cpp perf claim

This is the governing doc the speed campaign assumed but had never been written. It is **binding**
for every run, number, and promotion. A claim without a committed `PERF_RECEIPTS/` bundle that
obeys these rules is unbacked and gets removed.

## Hosts of record

| Host | Role | Status |
|---|---|---|
| **win i7-11800H** (8C/16T Tiger Lake, AVX2+AVX-512+VNNI, ~16 GiB DDR4 ≈ 51 GB/s, RTX 3060 Laptop 6 GB) | primary | available |
| **Ubuntu validation host** | second host for promotion | **PENDING** (Tim: Windows-only baseline for now, 2026-06-28) |

A default flip (promotion) requires the win on **both** hosts. Until the Ubuntu leg exists, every
result is a **Windows-host result** and stays `ubuntu: pending`. Nothing promotes on one host.

## llama.cpp pin

`acd79d6` (2026-06-14), built with CUDA. The `llama-server.exe` / `llama-bench.exe` here report
`version: 1 (acd79d6)`. Re-pin only deliberately, with a note.

## Non-negotiables for a CPU lane

1. **CUDA hidden.** `CUDA_VISIBLE_DEVICES=-1` for BOTH Camelid and llama. Without it, llama's
   `-ngl 0` silently offloads prefill matmuls to the 3060 (measured pp512 = 740 tok/s GPU-assisted
   vs 30.9 tok/s true CPU — a ~24× inflation). This is the single most common trap.
2. **Cache defeated.** Camelid prompt-prefix-caches by default → inject a unique nonce in the
   system message per measured call (or use the in-binary `bench-generate`, which runs an
   independent cold prefill per iteration). llama-server caches across requests → `cache_prompt:false`
   AND `--cache-ram 0` (its slot LCP cache reuses near-identical prompts, `sim~1.0`, otherwise).
   `llama-bench` is inherently cold per repetition.
3. **Median-of-5.** `--repeats 5` / `-r 5` / `--iterations 5`; report the **median**, not the mean.
4. **Greedy, temp=0.** Same prompt, deterministic.
5. **Parity first.** A promoted optimized path must be a token-identical drop-in:
   `first_divergent_generated_token_index == -1`. For a CPU-mirrored kernel the per-kernel gate is
   stricter still — **bit-identity** (`to_bits()==to_bits()` vs the scalar oracle), not a 5e-4
   tolerance. The 5e-4 GEMM tests are defensive, not a license to diverge.
6. **Label the win correctly.** prefill ≠ decode; Q4 ≠ Q8; batch ≠ batch-1; UX ≠ kernel;
   capability ≠ throughput. State which.

## Two sanctioned harnesses (this repo)

- **Throughput, robust (preferred):** `bench-generate` (in-binary, applies the real execution plan,
  no HTTP/cache games) for Camelid + `llama-bench -ngl 0 -t 8 -p 512 -n 64 -r 5` for llama. Both
  CUDA-hidden, separate processes (no contention), `CAMELID_MAX_KV_CACHE_BYTES` pinned for a
  controlled run. Runner: `scripts/run-baseline` style → `cpu-baseline/v1` receipt.
- **Same-prompt A/B + parity capture:** `docs/perf-deep-dive/scripts/cpu-baseline-medN.mjs`
  (median-of-N, config-A-default vs llama, CUDA-hidden, cache-defeated). Use for camelid-vs-llama
  parity text alongside throughput.

> Methodology bug found & fixed 2026-06-28: an HTTP-harness variant timed `performance.now()` AFTER
> `await fetch()`, measuring JSON-parse time → absurd "1.18M tok/s". Always start the timer BEFORE
> the fetch. The `bench-generate`+`llama-bench` method sidesteps this entirely.

## Hard DO-NOTs (settled negatives — do not relitigate; see LANE_STATUS_LEDGER.md)

- Do **not** optimize the standalone decode dot (`q8_0_dot_rows_avx2` & friends) for speed — every
  dot variant is identical-throughput on the bandwidth-bound decode path (`p1-vnni-kernel-matrix`).
- Do **not** enable the gated `CAMELID_X86_Q8_*` packed decode kernels — measured −8…−11%,
  byte-identical, on this host. That negative receipt is load-bearing.
- Do **not** benchmark against ggml's generic `ggml_vec_dot_q8_0_q8_0` and call it llama's fast path
  — x86 Q8 prefill runs **tinyBLAS** register-tiling (`LLAMA_CPP_ARCHAEOLOGY §2`).
- Do **not** drop bit-exactness silently, reorder/defer the f32 reduction, use FMA where the oracle
  doesn't, or pre-fold scales. The property test is the tripwire.
- Do **not** generalize a ternary (TQ2_0) result to Q4_K_M / mainstream decode (both bandwidth-tied).
- Do **not** promote on one host, or report a number the harness didn't produce on a host of record.

## Negative receipts ship

A tie or a regression is a **valid, committed result**. The campaign's value is an honest map, not a
forced win.
