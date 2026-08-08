# Prism Bonsai + K-quant — Windows x86_64 CPU lane, 2026-08-07

Evidence that the packed low-bit quants run on the **CPU** lane on Windows, and that
the CPU result is greedy token-identical to the committed Windows CUDA oracle.

Two independent defects of the SAME shape are fixed here: a wire-only weight load
paired with a CPU dispatch that has no consumer for it. In both cases the model died
on its first forward with `storage=no-row-major-data, data_len=0`.

## Defect 1 — Prism Q1_0 / Q2_0 / PQ2_0 (a compile-time stand-in for a runtime check)

`LlamaLoadedWeights::load_with_ownership` chose the Prism weight representation from a
COMPILE-time `cfg!`:

```rust
if cfg!(any(target_os = "macos", all(target_os = "windows", feature = "cuda"))) {
    return store.load_prism_wire_linear(name);   // data: Vec::new()
}
```

`build.rs` emits `cargo:rustc-cfg=feature="cuda"` for every Windows target ("CUDA
builds by DEFAULT on Windows"), so that branch was taken on **every** Windows build —
including boxes with no usable NVIDIA device, and runs with `--gpu off`, Safe profile,
or a failed driver/NVRTC load. The planner, meanwhile, picks the lane from the RUNTIME
`platform.cuda_resident_active`. A loader keyed on the target triple can never agree
with a planner keyed on the device actually present.

Bonsai is a MIXED-ARCH family, confirmed by reading every artifact's header: the **4B
and 8B rows declare `qwen3`** and route to the dense engine, so those five rows hit
this path directly. The **27B pair declares `qwen35`**, and `is_runnable_only_arch`
sends qwen35 to the runnable lane instead — a different code path that was never
affected by this defect (see "27B rows" below). The catalog/README labelled all seven
`qwen35`; that metadata is fixed separately in PR #624 as a SPLIT, not a rename.

**Fix:** a streaming CPU block-dot for the whole Prism family
(`matmul_rhs_transposed_prism_block_dot` / `prism_block_dot_core` /
`accumulate_transposed_linear_row_prism`), wired into all five linear dispatch sites,
and the `cfg!` removed so packed wire is the representation everywhere. The kernel
reconstructs weights exactly as `decode_q1_0_tensor` / `decode_q2_0_tensor` do and
keeps the per-element `(coef * d) * a` multiply, so it is BIT-IDENTICAL to
dequantize-then-dot rather than merely close. This also retires the old fallback's
~16 GB f32 materialization for Q2_0/PQ2_0.

## Defect 2 — K-quants (an env var that could remove the only consumer)

`CAMELID_X86_Q4K_DECODE=0` gated the K-quant CPU block-dot at all four dispatch sites.
But `load_kquant_wire_linear` retains wire bytes and leaves `data` EMPTY, and there is
no f32 K-quant load path, so the block-dot is the ONLY consumer — the flag could not
mean "materialize f32 instead", because nothing materializes it.

Reproduced on a real `Qwen3-4B-Q4_K_M.gguf` with `--gpu off`:

```
matmul rhs-transposed rhs cannot read tensor blk.0.attn_q.weight as row-major f32:
storage=no-row-major-data, shape=[4096, 2560], data_len=0, expected_len=10485760
```

That is every Q2_K / Q3_K / Q4_K / Q5_K / Q6_K model, on CPU, one env var away.

**Fix:** `kquant_block_dot_selected(has_f32_data)` — the flag may express a kernel
preference, but a weight with no f32 data always selects the block-dot. Default
behavior is unchanged (flag defaults on). After the fix, the same command generates
`" Paris. The capital of Germany"`.

**And its disclosure, which the first cut of this fix broke.** `select_kquant_plan` in
`src/execution_plan.rs` still branched on `q4_k_cpu_block_dot_enabled()`, with an `else`
arm reporting `cpu_reference` / `safe_cpu_decode` and the reason "K-quant linears have
no CPU consumer". Once routing stopped honouring the flag, that arm described a lane no
run could take: with `CAMELID_X86_Q4K_DECODE=0` the plan would have claimed a safe CPU
path while the block-dot actually decoded correctly. That is exactly the drift this
whole change is about — a disclosure keyed on different inputs than the routing — so
the branch was deleted rather than reworded, and the doc comment naming the flag as a
route selector was corrected. The test that asserted `selected_backend == "cpu_reference"`
under the flag now asserts `cpu_kquant_block_dot`, because that is what runs.

This one was caught by an adversarial review pass over the diff, not by me writing it.

## Defect 3 — `--gpu off` could not select the CPU lane for the qwen35 rows

`src/runnable/model.rs` chose its CUDA lane from `CAMELID_QWEN35_CUDA`, defaulting to
ON for `cfg!(windows) && is_prism_low_bit()`, and consulted the GPU master switch
NOWHERE (no `gpu_accel_enabled` / `runtime_enabled` / `resident_decode_cuda_active`
reference exists in that file). `--gpu off` is documented as "Force the CPU reference
path; never use the GPU even if one is present", and `main.rs` implements it by calling
`cuda::set_gpu_accel_enabled(false)`. So on a Windows host with a working GPU there was
no way to select the CPU lane for the 27B Bonsai rows at all.

**Fix:** the master switch now gates the lane —
`cuda_requested && crate::cuda::gpu_accel_enabled()`. The env var stays an opt-IN and
does not override the switch. `gpu_accel_enabled()` lazily seeds from platform
capability, so the default (auto) path is unchanged.

**Verified end-to-end** — see the A/B under the 27B results table below.

## Why existing coverage missed both

Every Bonsai test asserts PLANNER DISCLOSURE STRINGS, which are byte-identical whether
or not the loader and the CPU agree. Three new regression tests close that gap, all of
which fail on the pre-fix code:
`prism_wire_only_linear_has_a_cpu_consumer`, `prism_block_dot_matches_decode_on_real_model`,
`kquant_block_dot_stays_selected_for_wire_only_weights`.

## Artifacts under test — all seven hash-pinned Bonsai rows

| File | sha256 | bytes | arch |
| --- | --- | --- | --- |
| `Bonsai-4B-Q1_0.gguf` | `4524b3f997f0f06444e568d1f26e2efd69effa3218c7ad3047432fb171e42168` | 572270624 | qwen3 (dense) |
| `Ternary-Bonsai-4B-Q2_0.gguf` | `4e0bf8b737b0431552f8c2c97695ab7c0cb214c94bcdeb4f5f267e67ddf28b8b` | 1074969344 | qwen3 (dense) |
| `Ternary-Bonsai-4B-PQ2_0.gguf` | `829abec7eb92f5bf464762be7c9e8a45d777c714543a1474fc90cee20e698beb` | 1074969344 | qwen3 (dense) |
| `Bonsai-8B-Q1_0.gguf` | `284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54` | 1158654496 | qwen3 (dense) |
| `Ternary-Bonsai-8B-Q2_0.gguf` | `3c8d70470a5d97e5a2b9410ddd899cb740116591462626c60cb2fead6448f60b` | 2182184672 | qwen3 (dense) |
| `Bonsai-27B-Q1_0.gguf` | `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0` | 3803452480 | **qwen35 (runnable)** |
| `Ternary-Bonsai-27B-Q2_0.gguf` | `868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757` | 7165121600 | **qwen35 (runnable)** |

Every size matches its catalog pin. Two hashes are independently confirmed by the
repo itself: the 4B Q1_0 matches the sha in
`../prism-bonsai-windows-cuda-20260802/bonsai-4b-q1-parity.json`, and the 27B Q1_0
matches `PRISM_BONSAI27B_Q1_SHA256` in `src/cuda_resident.rs`.

## Results

### The five dense-engine rows — the ones this defect actually broke

| Row | Kernel bit-exact vs decoder | CPU serve + generate | Oracle parity |
| --- | --- | --- | --- |
| Bonsai 4B Q1_0 | PASS | PASS | **12/12 token+text** |
| Ternary Bonsai 4B Q2_0 | PASS | PASS | no oracle committed |
| Ternary Bonsai 4B PQ2_0 | PASS | PASS | no oracle committed |
| Bonsai 8B Q1_0 | PASS | PASS | no oracle committed |
| Ternary Bonsai 8B Q2_0 | PASS | PASS | no oracle committed |

All five rows that this defect broke now load and generate on the CPU lane.

### The two 27B rows — a DIFFERENT lane (defect 1 never applied; defect 3 did)

| Row | Decoder bit-exact | CPU serve + generate |
| --- | --- | --- |
| Bonsai 27B Q1_0 | PASS | **PASS** — `"Paris"` via `/v1/chat/completions` |
| Ternary Bonsai 27B Q2_0 | PASS | host-limited (needs ~8.0 GB, 6.9 GB free) |

**Read this table carefully.** `qwen35` is `is_runnable_only_arch`, so the 27B rows go
to `src/runnable/model.rs`, which materializes weights per row via
`runnable::dequant::dequantize` — a path that has covered Q1_0/Q2_0/PQ2_0 all along and
was never broken by defect 1. The bit-exact PASS validates the tensor-layer decoders
against those bytes; it does NOT exercise the dense-engine kernel, which these rows
never execute. What they DID need was defect 3, without which `--gpu off` could not
select their CPU lane at all.

Runnable-lane archs fail closed on `/v1/completions` by design ("served only via
`/v1/chat/completions`; raw completion surfaces have no runnable bridge and fail closed
rather than falling through to the optimized engine"), so the smoke uses chat for them.

**Defect 3 verified by A/B:** with `CAMELID_QWEN35_CUDA=1` explicitly set AND
`--gpu off`, the 27B Q1_0 generated `"Paris"` while VRAM stayed FLAT at its 2187 MiB
baseline for the whole run — sampled every 15 s via `nvidia-smi`. A resident CUDA graph
for that row would have taken ~3.8 GB. The master switch now outranks the env opt-in.

`bonsai-4b-q1-cpu-parity.json` — `camelid.raw_decode_parity.v1`, greedy raw-prompt
completion vs the committed Windows CUDA oracle, 4 prompts x {1, 5, 50} tokens:
**`all_pass: true`**, 12/12 `token_match`, 12/12 `text_match`.

Generation smoke (`scripts/prism-cpu-smoke.sh`, prompt "The capital of France is"),
each asserting from `/v1/health` that the CPU lane is the one that RAN. Every row
marked PASS anywhere in this bundle has its line here — all six:

```
Bonsai-4B-Q1_0          | PASS | quant=Q1_0       backend=cpu_reference cuda=false | " Paris. Paris is the"
Ternary-Bonsai-4B-Q2_0  | PASS | quant=Q2_0_G128  backend=cpu_reference cuda=false | " Paris. The capital of"
Ternary-Bonsai-4B-PQ2_0 | PASS | quant=PQ2_0      backend=cpu_reference cuda=false | " Paris. The capital of"
Bonsai-8B-Q1_0          | PASS | quant=Q1_0       backend=cpu_reference cuda=false | " Paris. Paris is the"
Ternary-Bonsai-8B-Q2_0  | PASS | quant=Q2_0_G128  backend=cpu_reference cuda=false | " Paris. The capital of"
Bonsai-27B-Q1_0         | PASS | quant=Q1_0       backend=cpu_reference cuda=false | "Paris"
```

Two of those need a note on HOW they were obtained, because the run order matters:

- `Ternary-Bonsai-8B-Q2_0` first returned `SKIPPED-HOST-LIMIT | needs ~3281MB, only
  2895MB free` — the concurrent 14 GB of model downloads had transiently eaten free
  RAM. It was re-run after the downloads finished and PASSED on the second attempt;
  the line above is that second run. Nothing about the row changed, only the host.
- `Bonsai-27B-Q1_0` is a runnable-lane (`qwen35`) row, so it first returned
  `FAIL-GEN | model 'Bonsai-27B' is a runnable-lane architecture served only via
  /v1/chat/completions`. That is a deliberate fail-closed, not a defect; the smoke
  script now retries those on chat, which is their supported surface.

The RTX 3060 was physically present and unused (`--gpu off`,
`cuda_resident_active:false`) for every one of these — that is exactly the
configuration in which the compile-time `cfg!` and the runtime planner disagreed.

GPU regression check: the same 4B Q1_0 row without `--gpu off` still selects
`cuda_resident_prism_low_bit_runtime` (`cuda_resident_active:true`) and emits the same
tokens `[12095, 13, 12095, 374, 279]`. CPU and GPU now agree exactly.

## Scope and limits — what this does NOT claim

- **Six of the seven rows generate on CPU here. The seventh is a HOST limit, not a code
  limit.** `Ternary-Bonsai-27B-Q2_0` is a 7.17 GB row and this box had 6.9 GB free; it
  was never started, because swapping a 7 GB resident model thrashes the host. Its
  decoders are bit-exact against those exact bytes and its sibling 27B Q1_0 generates
  on the same lane, but no generation was run for it. A larger host should settle it.
- This is a **correctness** result, not a performance one. The Prism kernel is a
  scalar dot with rayon over output rows; there is no AVX2/NEON path yet. The 4B
  parity sweep took **11m22s for 224 generated tokens** (aggregate, incl. prefill).
  Recorded, not claimed. An AVX2 kernel like `tq2_0_row_dot_avx2` is the follow-up.
- Full oracle parity was run for the **Q1_0 4B** row only, because that is the only
  row with a committed oracle. The others have generation smoke plus bit-exact kernel
  tests, not token-level parity receipts.
- No vision, serve/WebUI, bounded-context, or throughput claim is made or implied. The
  27B rows are vision-capable; only text was exercised.
- macOS Metal and Windows CUDA behavior is unchanged: those platforms already took the
  packed-wire branch, and this change only removed the guard that decided it.
