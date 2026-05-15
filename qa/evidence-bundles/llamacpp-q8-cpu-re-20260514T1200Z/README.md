# Ubuntu x86_64 Q8_0 CPU performance investigation — Llama 3.2 3B Instruct Q8_0

Generated: 2026-05-14 UTC
LANE: `UBUNTU_X86_Q8`
Scope: Ubuntu x86_64 dense Llama Q8_0 only.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-15T12:35Z

- Small follow-on slice widened the default-off `CAMELID_X86_Q8_REPACK=on` runtime-packed loader to include dense Llama `blk.*.ffn_down.weight` in backend-owned `Q8_0RuntimeStorage::PackedRows4`.
- The new FFN-down case packs the GGUF descriptor shape `[ffn, hidden]` as directly consumable transposed runtime rows `[hidden, ffn]`, matching the existing `linear_for_role_runtime` hot path without retaining `data`, `q8_0_blocks`, file backing, or debug packed sidecars.
- Fallback is unchanged: with the x86 repack env unset/off, `CAMELID_Q8_0_BLOCK_DOT=off`, unaligned shapes, or tensors outside the selected x86 allowlist, the existing safe/load paths remain in force.
- llama.cpp grep evidence was refreshed for `q8_0`, `tinyBLAS`, `ggml_vec_dot_q8_0_q8_0`, `repack`, `MUL_MAT`, scheduling, OpenMP, AVX2, AVX512, and VNNI; selected hits plus implementation evidence are captured in `artifacts/cron-95495a91-20260515T1235Z-x86-ffn-down-runtime.txt`.
- Canonical Ubuntu x86_64 validation passed: `/home/ubuntu/.cargo/bin/cargo test --test tensor_store x86_q8_repack_loads_dense_ffn_family_as_transposed_packed_runtime -- --nocapture` in a synchronized scratch checkout (`1 passed; 0 failed; 23 filtered out`).

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-15T11:08Z

- Small follow-on slice widened the default-off `CAMELID_X86_Q8_REPACK=on` runtime-packed loader from `blk.*.attn_q.weight` to the dense attention projection family: `blk.*.attn_q.weight`, `blk.*.attn_k.weight`, `blk.*.attn_v.weight`, and `blk.*.attn_output.weight`.
- The implementation still uses backend-owned `Q8_0RuntimeStorage::PackedRows4` for selected tensors and keeps `data`, `q8_0_blocks`, file backing, and debug packed sidecars empty/absent for that path.
- Fallback is unchanged: with the x86 repack env unset/off, or for tensors outside the selected x86 allowlist, the existing safe/load paths remain in force.
- llama.cpp grep evidence was refreshed for `q8_0`, `tinyBLAS`, `ggml_vec_dot_q8_0_q8_0`, `repack`, `MUL_MAT`, scheduling, OpenMP, AVX2, AVX512, and VNNI; selected hits are captured in `artifacts/cron-95495a91-20260515T1108Z-x86-attn-family.txt`.
- Canonical Ubuntu x86_64 validation passed: `cargo fmt --check`, `cargo test -q x86_q8_repack_loads_dense_attention_family_as_packed_runtime --test tensor_store`, and `cargo test -q x86_q8_avx2_packed_rows4_i8_matches_scalar_dot --lib` using the installed Rust 1.90.0 toolchain because the host default cargo is too old for the lockfile/MSRV.

## CAMELID TPM UBUNTU X86 Q8 handoff — cron 0719640b, 2026-05-14T22:49Z

- CAMELID TPM UBUNTU X86 Q8: Active evidence root remains this directory; latest recheck artifact is `artifacts/cron-0719640b-20260514T2249Z-verification.txt`.
- CAMELID TPM UBUNTU X86 Q8: Canonical host path was re-verified as Ubuntu x86_64 on Intel Xeon Platinum 8488C with 16 vCPUs and AVX2/AVX512/VNNI/AMX hardware flags; this lane claims only the measured Ubuntu CPU path.
- CAMELID TPM UBUNTU X86 Q8: llama.cpp evidence still shows Release CPU build with `GGML_CPU_REPACK=ON`, `GGML_OPENMP=ON`, AVX/AVX2/F16C/FMA on, and AVX512/VNNI/AMX build gates off for the measured binary.
- CAMELID TPM UBUNTU X86 Q8: perf proof remains `artifacts/perf-bench-pp-symbols.txt`, dominated by `tinyBLAS_Q0_AVX<block_q8_0, block_q8_0>::gemm4xN<2>` via `llamafile_sgemm`/`ggml_compute_forward_mul_mat`; this proves the actual measured win is tiled Q8_0 MUL_MAT + OpenMP scheduling, not an AVX512/VNNI/AMX kernel.
- CAMELID TPM UBUNTU X86 Q8: same-host llama.cpp and Camelid benchmark artifacts remain under `benchmarks/`; Camelid baseline/default-parallel/parallel-off retained-block microbench stayed ~16 ms with equal checksum, while the bounded default-off `CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2` API smoke cut first-token wall from 147425.30 ms to 75650.18 ms and kept token id 8586.
- CAMELID TPM UBUNTU X86 Q8: bounded safe port slice is commit `80f6271` in `src/tensor/mod.rs`, `src/inference.rs`, and `tests/tensor_store.rs`; current Q8 path remains fallback when env gates are absent/off or AVX2 is unavailable.
- CAMELID TPM UBUNTU X86 Q8: blocker is still full end-to-end Camelid API throughput equivalence against llama.cpp; next owner should extend the default-off packed/tiled Q8 GEMM architecture to FFN down and more dense linears only after Ubuntu x86 parity/perf evidence per tensor family.

## Repositories and status

- Camelid local worktree: `main...origin/main [ahead 1]` with pre-existing unrelated dirty files; this active Ubuntu x86 evidence root records only the `UBUNTU_X86_Q8` findings/slice and should not be used as evidence for other platforms. This lane touched the x86 Q8 implementation files plus this evidence bundle; unrelated dirty evidence from other lanes was left unstaged.
- llama.cpp local/remote reference: `3e037f313c2c4cfce897d9be8f43954283a61de1` (`version: 9158`, commit `HIP: RDNA3 mma FA, faster AMD transpose, tune AMD (#22880)`).
- Canonical host: `ubuntu@54.186.43.33`, AWS Ubuntu 24.04 x86_64, Intel Xeon Platinum 8488C, 16 vCPUs.
- Model: `/home/ubuntu/models/Llama-3.2-3B-Instruct-Q8_0.gguf`.

Evidence:
- `artifacts/cron-95495a91-20260515T1235Z-x86-ffn-down-runtime.txt`
- `artifacts/cron-95495a91-20260515T1108Z-x86-attn-family.txt`
- `artifacts/cron-0719640b-20260514T2249Z-verification.txt`
- `artifacts/ubuntu-host-repos-models.txt`
- `artifacts/ubuntu-llamacpp-build-symbols.txt`
- `artifacts/llamacpp-git-grep.txt`
- `artifacts/llamacpp-git-grep-full.txt`
- `artifacts/llamacpp-repack-selected-source.txt`
- `artifacts/camelid-x86-repack-tests.txt`
- `artifacts/camelid-x86-repack-build.txt`

## llama.cpp Ubuntu x86 Q8_0 path findings

### Actual compiled capabilities on the canonical host

CPU hardware exposes AVX2, AVX512, AVX_VNNI, AVX512_VNNI, and AMX (`amx_int8`, `amx_bf16`, `amx_tile`), but the llama.cpp build used for evidence is narrower:

- `GGML_CPU_REPACK=ON`
- `GGML_OPENMP=ON`
- `GGML_AVX=ON`, `GGML_AVX2=ON`, `GGML_F16C=ON`, `GGML_FMA=ON`
- `GGML_AVX512=OFF`
- `GGML_AVX_VNNI=OFF`, `GGML_AVX512_VNNI=OFF`
- `GGML_AMX_INT8=OFF`, `GGML_AMX_BF16=OFF`, `GGML_AMX_TILE=OFF`
- `llama-cli` links `libgomp.so.1`.

Runtime `llama-server` system info likewise reported `AVX2 = 1`, `LLAMAFILE = 1`, `OPENMP = 1`, `REPACK = 1`, with no AVX512/VNNI/AMX runtime path in this build.

### Source map

Key source locations in current llama.cpp:

- `ggml/src/ggml-cpu/arch/x86/quants.c`
  - `quantize_row_q8_0`
  - `ggml_vec_dot_q8_0_q8_0`
- `ggml/src/ggml-cpu/ggml-cpu.c`
  - Q8_0 trait wiring: `.from_float = quantize_row_q8_0`, `.vec_dot = ggml_vec_dot_q8_0_q8_0`
  - `GGML_OP_MUL_MAT` scheduling/compute dispatch
- `ggml/src/ggml-cpu/ggml-cpu.cpp`
  - CPU backend extra buffer registration including `GGML_USE_CPU_REPACK`
- `ggml/src/ggml-cpu/repack.cpp` / `repack.h`
  - Q8_0 repack layouts and generic `ggml_gemv/gemm_q8_0_*_q8_0` hooks
  - graph rewrite hooks for `GGML_OP_MUL_MAT` / `GGML_OP_MUL_MAT_ID`
- `ggml/src/ggml-cpu/llamafile/sgemm.cpp`
  - `tinyBLAS_Q0_AVX`, the observed hot prompt-processing kernel

### Perf proof of actual Ubuntu path

Perf evidence is from the canonical host against the Q8_0 model with CPU-only llama.cpp.

Best hot-symbol run: `artifacts/perf-bench-pp-symbols.txt`

Top path:

```text
88.46% libggml-cpu.so.0.11.1  tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<2>
       tinyBLAS_Q0_AVX<...>::mnpack
       llamafile_sgemm
       ggml_compute_forward_mul_mat
       ggml_graph_compute_thread.isra.0
       GOMP_parallel / ggml_graph_compute / llama_context::process_ubatch
```

Other selected symbols:

- `quantize_row_q8_0`: 0.71%
- `ggml_vec_dot_q8_0_q8_0`: 0.43%
- `libgomp.so.1`: present in hot samples

Interpretation: for the measured Ubuntu x86_64 prompt-processing path, the dense Q8_0 hot loop is not AVX512/VNNI/AMX. It is AVX2-era llamafile/tinyBLAS Q8_0 x Q8_0 through `GGML_OP_MUL_MAT`, with OpenMP/GOMP scheduling. Repack support is compiled in and source-visible, but the selected hot evidence is dominated by `tinyBLAS_Q0_AVX`, not repack `q8_0_4x4/4x8/16x1` symbols.

Perf caveat: kernel symbols were restricted by host perf settings; user-space symbols were sufficient for this lane. See `artifacts/perf_event_paranoid.txt`, `artifacts/perf-bench-pp.stderr`, and `artifacts/perf-run.stderr`.

## Benchmarks

### llama.cpp same-host CPU-only

Files:
- `benchmarks/llama-bench-t16-p128-n16.json`
- `benchmarks/llama-bench-t1-p128-n16.json`

| Mode | Prompt processing | Token generation |
|---|---:|---:|
| llama.cpp `-t 16`, `p=128`, `n=16` | 90.421 tok/s | 25.635 tok/s |
| llama.cpp `-t 1`, `p=128`, `n=16` | 12.168 tok/s | 2.670 tok/s |

### Camelid Q8 hot-path same-host microbench

Command shape: `target/release/camelid bench-q8-blocks /home/ubuntu/models/Llama-3.2-3B-Instruct-Q8_0.gguf --tensor blk.0.ffn_gate.weight --swap-rank2-shape --repeats 5 --warmup 1 --all-rows-dot --single-input-row-dot`

Files:
- `benchmarks/baseline.json`
- `benchmarks/parallel_on.json`
- `benchmarks/parallel_off.json`
- `benchmarks/avx2.json`

| Camelid mode | avg all-row Q8 dot | avg single-input-row Q8 dot | checksum |
|---|---:|---:|---:|
| baseline env | 16.109 ms | 16.051 ms | `-0.05126936` |
| `CAMELID_PARALLEL_LINEAR=on` | 16.152 ms | 16.073 ms | `-0.05126936` |
| `CAMELID_PARALLEL_LINEAR=off` | 16.049 ms | 16.093 ms | `-0.05126936` |
| `CAMELID_X86_Q8_KERNEL=avx2` | 16.102 ms | 16.058 ms | `-0.05126936` |

Interpretation: the bounded AVX2 scalar-block replacement is parity-clean but does not materially improve the retained-block microbench by itself. That is expected: llama.cpp’s win is primarily a wider tiled GEMM/MUL_MAT architecture with tinyBLAS/OpenMP scheduling, not only a faster single 32-byte dot primitive.

### Camelid same-host API smoke benchmark: default vs x86 runtime repack

Command shape: `CAMELID_BIN=target/release/camelid node scripts/bench-unique-chat.mjs --start-backend --model /home/ubuntu/models/Llama-3.2-3B-Instruct-Q8_0.gguf --max-tokens 1 --repeats 1 --warmup 0`.

Files:
- `benchmarks/unique-chat-baseline-1tok.json`
- `benchmarks/unique-chat-x86-repack-avx2-1tok.json`

| Camelid API mode | output token text | avg wall | avg generate | avg layers | attention projections | FFN gate | FFN up | FFN down | FFN total | RSS after first token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline env | `Here` | 147425.30 ms | 144962.00 ms | 144515.97 ms | 36277.35 ms | 35464.56 ms | 35772.57 ms | 36891.29 ms | 108140.19 ms | 3808.82 MiB |
| `CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2` | `Here` | 75650.18 ms | 72077.00 ms | 71463.04 ms | 24606.97 ms | 5174.13 ms | 5127.87 ms | 36484.12 ms | 46796.54 ms | 3836.46 MiB |

Interpretation: this is a one-request smoke benchmark, not a production throughput claim. It does demonstrate the bounded default-off x86 runtime-repacked path is parity-clean for the measured first token (`Here`) and materially reduces the gate/up rows that were converted to backend-owned packed Q8 storage. FFN down remains on the existing path in this slice.

Full llama.cpp-vs-Camelid API harness note: `scripts/bench-llama3-same-host.mjs` was previously attempted with `max_tokens=8`, `repeats=2`, `threads=16`; the Camelid side did not produce measured output before the run was killed after several minutes. This README therefore keeps llama.cpp `llama-bench`, Camelid microbench, and Camelid API default-vs-repack smoke as separate same-host evidence, not a full end-to-end API throughput equivalence claim against llama.cpp.

## Camelid bounded default-off port slice

Implemented in `src/tensor/mod.rs`, `src/inference.rs`, and `tests/tensor_store.rs`:

- `CAMELID_X86_Q8_REPACK=on` is a default-off GGUF load/read gate for selected Llama dense Q8 linears in this slice (`blk.*.attn_q.weight`, `blk.*.attn_k.weight`, `blk.*.attn_v.weight`, `blk.*.attn_output.weight`, `blk.*.ffn_gate.weight`, `blk.*.ffn_up.weight`, `blk.*.ffn_down.weight`).
- When the gate is on, `TensorStore::{load_q8_0_file_backed_linear,load_q8_0_block_backed_linear}` build `Q8_0RuntimeStorage::PackedRows4` directly from GGUF Q8_0 bytes and return a tensor with empty `data`, no `q8_0_blocks`, and no file-backed row-major sidecar for those selected tensors.
- FFN gate/up/down descriptor shapes are packed in runtime output-row order so `linear_for_role_runtime` consumes the backend-owned packed storage directly.
- `x86_q8_kernel_avx2_enabled()` reads `CAMELID_X86_Q8_KERNEL` and accepts `avx2/on/1/true` (case variants included).
- `q8_0_i8_block_avx2()` and `q8_0_packed_4x8_block_avx2()` are `#[target_feature(enable = "avx2")]` and default-off behind both the env gate and `std::arch::is_x86_feature_detected!("avx2")`.
- Existing path fallback is preserved when the env gates are absent/off or AVX2 is not detected.
- Unit tests: `x86_q8_avx2_kernel_matches_scalar_dot`, `x86_q8_avx2_packed_rows4_i8_matches_scalar_dot`, `x86_q8_repack_loads_attn_q_as_packed_runtime_without_row_major_duplicate`, `x86_q8_repack_loads_dense_attention_family_as_packed_runtime`, `x86_q8_repack_loads_dense_ffn_family_as_transposed_packed_runtime`.

Validation:

- Ubuntu x86_64 test pass: `artifacts/camelid-x86-repack-tests.txt`
- Ubuntu x86_64 release build pass: `artifacts/camelid-x86-repack-build.txt`
- Same-host microbench parity: all retained-block Camelid modes had identical `dot_checksum = -0.05126936`.
- Same-host API smoke parity: baseline and `CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2` both emitted first-token text `Here` for the measured prompt; timings are in `benchmarks/unique-chat-*.json`.
- Non-Ubuntu test gates are not claimed here. The Ubuntu x86_64 slice compiled and passed in `/tmp/camelid-ubuntu-x86-q8-20260514T2221Z` on the canonical host.

This slice intentionally avoids a performance-mode row-major+packed duplicate for the selected runtime-packed tensors. Existing opt-in debug/parity sidecars remain separate gates.

## Pass/fail table

| Requirement | Result | Evidence |
|---|---|---|
| Verify git status before edits / avoid clobber | PASS | status checked before and after; unrelated dirty files left unstaged |
| Map current llama.cpp x86 Q8_0 source path | PASS | `artifacts/llamacpp-git-grep*.txt`, `artifacts/llamacpp-repack-selected-source.txt` |
| Prove actual Ubuntu build flags | PASS | `artifacts/ubuntu-llamacpp-build-symbols.txt` |
| Prove actual hot symbols | PASS | `artifacts/perf-bench-pp-symbols.txt` |
| Benchmark llama.cpp same host | PASS | `benchmarks/llama-bench-t16-p128-n16.json`, `benchmarks/llama-bench-t1-p128-n16.json` |
| Benchmark Camelid baseline/default-parallel/parallel-off | PASS (microbench) | `benchmarks/baseline.json`, `parallel_on.json`, `parallel_off.json` |
| Implement bounded default-off x86 slice | PASS | `src/tensor/mod.rs`, `src/inference.rs`, `tests/tensor_store.rs`; env `CAMELID_X86_Q8_REPACK=on`, `CAMELID_X86_Q8_KERNEL=avx2`; follow-on attention-family loader evidence in `artifacts/cron-95495a91-20260515T1108Z-x86-attn-family.txt`; FFN-down runtime-storage evidence in `artifacts/cron-95495a91-20260515T1235Z-x86-ffn-down-runtime.txt` |
| Parity test on Ubuntu x86_64 | PASS | `artifacts/camelid-x86-repack-tests.txt`; microbench checksum parity; API first token `Here` in both JSON files |
| Demonstrate performance win from bounded slice | PASS (bounded smoke) | `benchmarks/unique-chat-baseline-1tok.json` vs `unique-chat-x86-repack-avx2-1tok.json`; gate/up timings and total first-token wall time reduced in the one-request API smoke |
| Full end-to-end Camelid API vs llama.cpp API | BLOCKED / partial | llama.cpp-vs-Camelid API harness did not complete promptly; this bundle has llama.cpp bench plus Camelid default-vs-repack API smoke, not API equivalence vs llama.cpp |

## Recommended next slice

Continue toward the actual winning llama.cpp architecture without widening support claims:

1. Keep widening the default-off x86 runtime-packed path only with Ubuntu x86 parity/bench evidence per tensor family; FFN down now has loader/runtime-storage coverage but still needs performance measurement.
2. Add a default-off x86 Q8_0 tiled matmul/GEMM path, e.g. `CAMELID_X86_Q8_GEMM=avx2`, then consider `avx512_vnni` only after rebuilding/benchmarking llama.cpp with VNNI enabled for comparison.
3. Tile over multiple output rows and input blocks, quantize the f32 activation row once to Q8_0, and amortize env/dispatch outside the innermost 32-byte block loop.
4. Add perf evidence on Ubuntu x86 before claiming broader speedup: hot symbols should move from scalar Rust loops toward a tiled x86 kernel, with unchanged checksums/output tokens.
