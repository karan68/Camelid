# Q5_K_M support — plan & handoff

Building on the existing Q4_K_M (Q4_K + Q6_K) K-quant work. Goal: run and
certify `*-Q5_K_M.gguf` (Q5_K + Q6_K) rows the way Q4_K_M was done.

`Q5_K` is `Q4_K` **plus a fifth bit per weight** (the extra bit lives in the
176-byte super-block's `qh[32]`; codes become `0..31` instead of `0..15`).
Everything else — the packed 6-bit scale/min `kmask` unpack, the per-16 mins
subtraction, the 8-lane accumulation — is identical to Q4_K.

---

## ✅ Done (CPU, verified on this machine)

### PR 1 — CPU oracle
- `src/inference.rs`: `Q5_K_WIRE_BYTES_PER_BLOCK = 176` and
  `q5_k_wire_row_dot(weight_wire, &[Q8KBlock]) -> f32` — the Q5_K×Q8_K block-dot,
  a faithful mirror of the verified `q4_k_wire_row_dot`; only new logic is the
  `qh` fifth-bit rebuild (matches the in-tree `Q5KBlock::dequantize`).
- `src/inference/tests.rs`: `q5_k_wire_dot_consistent_with_tensor_dequant` —
  cross-checks the oracle against the independent `decode_q5_k_blocks` decoder
  (different scale-unpack path), tol `1e-4`. **Passes.**
- This is the **CPU oracle the GPU `q5k_gemv` kernel must be validated against**
  (same role `q4_k_wire_row_dot` plays for `q4k_gemv`).

### PR 2 — full CPU wiring (prefill + decode)
- `CpuTensor.q5_k_wire_bytes: Option<Arc<Vec<u8>>>` (`src/tensor/mod.rs`) + every
  struct literal + `BorrowedLinearWeight` (`src/inference.rs`).
- Loader `load_kquant_wire_linear` accepts + populates Q5_K; load routing
  (`src/inference.rs` `load_linear`) sends 2-D Q5_K linears **wire-only**.
- **Prefill (batched matmul):** `q5_k_block_dot_core` +
  `matmul_rhs_transposed_q5_k_block_dot` + the 3 matmul dispatch sites.
- **Decode (single-row):** `accumulate_transposed_linear_row_q5_k` + the Q5_K arm
  in the funnel `accumulate_transposed_linear_row_with_precision_with_plan`.
  *(This decode path is SEPARATE from the batched matmul dispatch — it was the
  gap that crashed the first serve smoke.)*
- **Embedding:** Q5_K arm in `CpuTensor::embedding_lookup`.
- **Tied output projection:** Q5_K fast-path in
  `output_projection_with_layout_with_plan` (TokenMajor), mirroring the Q6_K one.
- **GPU-safe scoping:** deliberately did NOT touch
  `binding_all_resident_quant_linears` or the GPU `is_resident_quant` closures,
  so a Q5_K_M model cannot misroute to a non-existent `q5k_gemv` on a CUDA box
  (it cleanly runs on the CPU block-dot everywhere).

### Verification (real weights, on this box)
- `src/inference/tests.rs`: `q5_k_block_dot_matches_decode_on_real_model`
  (env-gated `CAMELID_Q5KM_GGUF`, skips if unset) — `q5_k_block_dot_core` ==
  `decode_q5_k_tensor` within `1e-4` on real Llama-3.2-1B Q5_K weights. **Passes.**
- **End-to-end serve smoke** on `Llama-3.2-1B-Instruct-Q5_K_M.gguf` (CPU):
  loads (`generation_ready=true`), greedy chat → *"The capital of France is
  Paris."* (`finish=stop`). The decode path (previously crashing) works.
- Gates green: `check`, `clippy -D warnings`, `fmt`, no regressions.

**Support ledger is intentionally NOT changed** — no claim is made until the
pinned-`llama.cpp` certification below exists.

---

## ⛳ Left to do (on the GPU box)

The GPU box builds **natively** (no ARM64 emulation) — use the repo's pinned
toolchain and the normal commands; CUDA actually runs there.

### 1. GPU `q5k_gemv` kernel — validate against `q5_k_wire_row_dot`
All in `src/cuda_resident.rs`, mirroring the existing `q4k_gemv` (line ~455):
- Add the `extern "C" __global__ void q5k_gemv(...)` NVRTC kernel — reproduce
  `crate::inference::q5_k_wire_row_dot` **exactly** (176 B/super-block: `d`,
  `dmin`, `scales[12]`, `qh[32]`, `qs[128]`; codes `0..31`).
- Add `q5k_gemv: CudaFunction` to the kernels struct (~2142), register
  `q5k_gemv: f("q5k_gemv")?` (~2215).
- Add `launch_q5k_gemv` (mirror `launch_q4k_gemv` ~2468) and route `ProjQuant::Q5K`
  (add the variant) in `repack`/`set_layer`/`set_output`.
- GPU parity test `q5k_gemv_matches_oracle` in `src/cuda_resident/tests.rs`
  (mirror `q4k_gemv_matches_oracle` ~2128; oracle =
  `crate::inference::q5_k_wire_row_dot`). It's `#[ignore]` (needs a CUDA device).

### 2. Enable the GPU-resident + guard path for Q5_K_M
- Add `Q5K` to `build_resident_cuda_engine`'s `raw()` byte source
  (`src/inference.rs` ~10040) and the resident `is_q4k`/`is_resident_quant`
  closures (~1955).
- Add `Q5K` to `binding_all_resident_quant_linears` (`src/api/mod.rs` ~6963) —
  do this **together with** the kernel so it doesn't misroute. This also gives the
  CPU f32-materialization guard its wire-only bypass, unblocking **large** Q5_K_M
  models on CPU (currently only small ones fit under the 6 GiB default guard).

### 3. Pinned-`llama.cpp` certification (the promotion bar)
- Get the pinned `llama.cpp` build the repo certifies against (`acd79d6`).
- Run the K-quant parity harness (`scripts/chat-parity-qwen3-kquant.mjs` and/or
  `scripts/raw-decode-parity.mjs`) on an exact `*-Q5_K_M.gguf` row (e.g.
  `Qwen3-4B-Q5_K_M` or `Llama-3.2-3B-Q5_K_M`) at 1/5/50 tokens, GPU-resident.
- Commit the evidence bundle under `qa/evidence-bundles/…-q5_k_m-…/`.

### 4. Promote the row (only after step 3 passes)
- Update `COMPATIBILITY.md`, `SUPPORT_MATRIX_v0.1.md`, `CAPABILITY_MATRIX.md`,
  and `/api/capabilities` (the `llama_spm_q4_k_q5_k` row) — keep all surfaces in
  sync; cite the exact bundle. Exact-row only, no family-wide claim.

### 5. Housekeeping
- Optional CPU `q5_k_wire_row_dot_avx2` sibling (default-off, bit-identical test),
  like the Q6_K AVX2 lane.
- Run the full `cargo test --all-targets --all-features` for the formal green.

---

## Reproduce the CPU verification

```powershell
# unit + real-weight parity (needs a *-Q5_K_M.gguf)
$env:CAMELID_Q5KM_GGUF = "<path>\Llama-3.2-1B-Instruct-Q5_K_M.gguf"
cargo test --lib q5_k_wire_dot_consistent
cargo test --lib q5_k_block_dot_matches_decode_on_real_model

# end-to-end serve smoke
cargo build --release --bin camelid
.\target\release\camelid.exe serve --model "<path>\Llama-3.2-1B-Instruct-Q5_K_M.gguf" --no-open
# then POST /v1/chat/completions
```

> Note: this box was ARM64 Windows and had to build the `x86_64` toolchain under
> emulation (`cargo +stable-x86_64-pc-windows-msvc …`) and always `--release`
> (debug stack-overflows on the large clap enum). On the native x86_64 GPU box,
> the plain pinned toolchain + `cargo build --release` work directly.
