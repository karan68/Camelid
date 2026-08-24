# Camelid — Gemma 4 26B-A4B CUDA ghost lane: performance handoff

**Box:** RTX 3060 **Laptop** 6 GiB VRAM, PCIe 4.0 **x8**, i7-11800H (8c/16t), **16 GiB DDR4-3200**,
NVMe ~1.3 GB/s buffered / ~1.9–2.1 GB/s unbuffered, Windows 11 / WDDM, CUDA 12.9.
**Model:** `models/google_gemma-4-26B-A4B-it-Q4_0.{gguf,hot,cghost}`. hidden 2816, 30 layers,
128 experts/layer, top-8, `n_ff_exp` 704, per-expert record **3,345,408 B = 3.19 MiB**, routed
payload **~12 GiB**. 16 GiB of host RAM is a **hard product constraint** — this must work for
everyone on 16 GiB, so no "buy more RAM" solutions.

---

## 1. Where things stand

| | session start | now |
|---|---:|---:|
| Prompt processing (362-tok prompt) | 8.48 t/s | **27.53 t/s** (3.25×) |
| First-message TTFT (32-tok prompt) | 9.20 s | **3.56 s** (2.4×) |
| Decode, steady, long context | ~9 t/s | ~9–11.35 t/s (unchanged) |

**Head-to-head vs llama.cpp CUDA b10612, same box, same GGUF, cold (page cache purged):**

| | pp362 | tg128 |
|---|---:|---:|
| llama.cpp `-ngl 12` cold | 17.85 | 9.79 |
| llama.cpp `--n-cpu-moe 99` | 21.23 | 10.94 |
| **Camelid now** | **27.53** | **11.35** |

We are ahead on both **at matched conditions**. A CUDA llama.cpp build is installed at
`C:\Users\timto\tools\llama-cpp-cuda` (the older `tools\llama-cpp` is **CPU-only** — no
`ggml-cuda.dll` — do not benchmark against it). Add
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin` to PATH for its runtime DLLs.

### ⚠ The single most important measurement caveat
**`llama-bench` runs a warmup pass by default** (`--no-warmup` disables it). Its widely-quoted
102 t/s for this model is a **fully-warm page-cache** number (17.85 cold → 35.91 after one warmup
→ 102.37 after repeated runs). Camelid does **not** warm up (21.42 / 20.17 / 20.00 across three
identical prefills) because our ~9.6 GiB touched set does not fit ~10 GiB free RAM while their
`-ngl 12` host set is ~7.7 GiB and does. **Never compare your cold number to someone else's warm
one.** I made that mistake in both directions before catching it.

---

## 2. Reproduce

```powershell
Set-Location C:\Users\timto\Projects\Camelid
$env:CAMELID_GEMMA4_CUDA_BATCH_EXPERT_COPIES = "1"
$env:CAMELID_GEMMA4_SPECULATIVE = "0"
$env:CAMELID_GEMMA4_GHOST_TIER_PREFILL = "1"
.\target\release\camelid.exe gemma4-cuda-generate `
  .\models\google_gemma-4-26B-A4B-it-Q4_0.hot `
  --cghost .\models\google_gemma-4-26B-A4B-it-Q4_0.cghost `
  --expert-cache-mib 64 --prompt "..." --max-tokens 128
```

Add `$env:CAMELID_SSER_PROFILE = "1"` for the bucket breakdown. Prefill time is
`(generated in X) - (timed decode wall)`.

**Harnesses** (both written this session, use them — single runs on this box are not evidence):
- `qa/perf/paired-ab.ps1` — ABABAB, idle temp/clock gate before every arm, per-pair thermal-drift
  rejection, free-RAM floor, paired delta + sign consistency, **token-ID and hit/miss identity
  gates**, `-ColdCache` to purge artifact pages. It has already caught two of my own regressions.
- `qa/perf/prefill-bench.ps1` — prefill across 32/92/362/1442-token prompts. `-ChunkTokens`,
  `-ColdCache`, `-Label`.
- Receipts in `qa/perf/receipts/`.

---

## 3. Landed this session — all bit-exact, all gates green

Gates: `cargo fmt --all`, `cargo clippy --release --features cuda --all-targets -- -D warnings`,
`cargo build --release --features cuda`.

| # | Change | Default | Measured |
|---|---|---|---|
| 1 | **Layer-major chunked prefill** — `prefill_chunked` + `layers: Range` / `do_prep` on `forward_token_impl` | `CAMELID_GEMMA4_PREFILL_CHUNK_TOKENS=512` (0/1 = old path) | **2.18×** |
| 2 | **Phase-dependent read path** — `GhostFile.bulk_reads` + unbuffered reads during prefill | on for prefill only | **+39%** |
| 3 | Step-01 instrumentation — `SSER_PROF_SYNC_NS`, `_STORAGE_NS`/`_BYTES`, `_HEAD_NS`, miss-provenance, `sser_profile_enabled()` OnceLock | behind `CAMELID_SSER_PROFILE` | made everything else visible |
| 4 | Q4_0 SoA repack for dense/attn/FFN projections | `CAMELID_GEMMA4_Q4_0_SOA=1` | +2.35%, not separated |
| 5 | `q4_0_gemv_routed_rows` (R rows/warp ordered fold) | `CAMELID_GEMMA4_CUDA_ROUTED_ROWS=1` (off) | null |
| 6 | `f32_matvec` row interleaving | always | null |
| 7 | Whole-layer read-ahead (`prefetch_moe_layer`) | `CAMELID_GEMMA4_PREFILL_PREFETCH=0` (off) | null |
| 8 | **Sparse-source guard** in the `.cghost` writer | always | latent corruption fix |
| 9 | `q4_0_gemm_routed` kernel + bitwise parity test | **not wired** | see §4 |

**#1 detail.** `prefill_reusing_cache` was `for pos in start..n { forward_token(...) }` — a full
30-layer forward per prompt token, which overwrites the ~889-slot expert arena ~3× per token so
the next token finds layer 0 evicted. Measured **66.8 misses/prompt token** (72% miss) vs 12.2% in
decode. Layer-major runs layer L for all K tokens first; a layer's union is ≤128 experts = 408 MiB
which **fits** the ~2.8 GiB arena. Misses 24,470 → 3,633 (the floor is 3,840 = payload once).
Per-token state is one device buffer per token for `d_hidden`/`d_cos_all`/`d_sin_all`, rotated in
with `std::mem::swap` — **moves handles, not bytes**, so the reordering costs zero copies.
Bit-exact because a token's layer-L output depends only on its own residual stream and
KV[L][0..=t], and tokens are still visited in ascending order within each layer.

**#2 detail.** `UncachedReader` (FILE_FLAG_NO_BUFFERING) already existed but was reachable only
via `--ghost-strict-cache`, which also nulls the mmap. Now `open_with_options` opens it always and
`read_positioned_span_into` uses it when `strict_cache || (bulk_reads && len >= 256 KiB)`.
Read rate **1.14 → 1.94–2.11 GB/s**.

**#8 detail — real bug, was silent.** `write_cghost_moe_with_counts` never checked
`source_is_sparse`. Repacking from `models/*.hot` instead of `*.gguf` produced a mostly-zero
`.cghost` that passed **both** identity checks, because the hot shadow retains a 128-byte identity
island at exactly the offset `validate_moe_expert_payload_identity` samples. Now refuses.

---

## 4. Next task: wire `q4_0_gemm_routed`

The kernel is **landed and bitwise-certified** (`q4_0_gemm_routed_matches_gemv` — both real
geometries 1408×88 and 2816×22, ragged CSR with experts of 0/1/many tokens, permuted slot map,
`blockIdx.z` tiling path). Only the plumbing remains.

**Why.** `q4_0_gemv_routed` re-reads an expert's weights once per token routing to it. At 362
tokens that is ~22.6× redundancy — **271 GiB of VRAM reads per prefill against the 12 GiB
actually needed**. FLOPs are unchanged (1.03 TFLOP), so batching turns a memory-bound GEMV into a
compute-bound GEMM. Expect the prefill fence **5.39 s → under 1 s**.

**Scope honestly:** this is mainly a **long-prompt** win. Redundancy is 22.6× at 362 tokens but
only **2.3× at 32 tokens**, so a regular user's short first message barely benefits. Do not expect
it to move TTFT much.

**Kernel signature** (`src/cuda_resident.rs`, registered in `CudaResidentKernels`):
```
q4_0_gemm_routed(input_scales, input_quants, weight_arena, slot_ids,
                 token_offsets, token_ids, weight_stride, rows, blocks_per_row,
                 output, expert_count, tile)
```
CSR: `token_offsets[e]..token_offsets[e+1]` indexes `token_ids`. Activations laid out per token.
Output is **per assignment**: `output[(first + tile_base + j) * rows + row]` — so the caller's
route-order weighted sum keeps its exact accumulation order. grid = `(rows/warps, expert_count,
ceil(max_count/tile))`, shared = `warps * tile * blocks_per_row * 4`.

**The layer body must be split into THREE phases, not two.** `forward_token_impl`'s layer loop is
`src/gemma4_runtime.rs:10909–11722`:

| Segment | Lines | Batchable? |
|---|---|---|
| A — norm/QKV/RoPE/KV-scatter/attention/O-proj/post-attn residual | 10910–11227 | no, per token |
| A2/A3 — router DtoH + dense shared-expert branch → `d_mlp` | 11239–11416 | no, per token |
| **B — fence + `moe_layer_ffn_cached` → CPU router + routed GEMVs** | **11418–11443** | **YES** |
| C — compose `moe_acc` + `d_mlp` → residual | 11445–11492 | no, per token |
| D — `ple_output_scale` on `d_hidden` | 11645–11721 | no, per token |

⚠ **A two-phase "stop before the MoE" flag is insufficient** — segment D runs *after* the FFN and
is **not** gated. (The PLE-proper block at 11647 *is* dead here, because `prefill_chunk_tokens()`
declines rows with per-layer embeddings, but `ple_output_scale` at 11712–11721 is not.) Use
`enum LayerPhase { Full, PrefillAttn, PrefillTail }`; `Full` keeps decode and the token-major
fallback byte-for-byte.

**Retain per token** for the batched FFN: `rms_inv` (from `(mss+eps).powf(-0.5)`, computed on the
**host** — see §5), the top-8 `idx`, `route_scales` = `down_exps_scale[e] * probs[e]/wsum`, and
the Q8_0 activation row. At K=512 that is ~1.375 MiB (`in_q`) + 176 KiB (`in_s`) + ~34 KiB host.

**New K-wide device buffers, ~81.5 MiB at K=512** (halves at K=256): `y_all` 44 MiB and `gate_up`
22 MiB dominate; also `moe_acc`, `in_q`, `in_s`, `geglu_q`, `geglu_s`, `d_mlp`, plus
`token_offsets`/`token_ids`/`route_scales`/`assign_index`. Everything else in the runtime is
single-row and can stay that way.

Full design with line-level detail is in the workflow transcript `wf_93ab4518-d59`.

**Parity gate:** token IDs must be bit-identical to the `CAMELID_GEMMA4_PREFILL_CHUNK_TOKENS=0`
token-major path on the same prompt. Run it before believing any speed number.

---

## 5. Do NOT re-propose these — measured and closed

| Lever | Verdict |
|---|---|
| Removing the 30 per-layer host syncs | Commit `02f45f9f` already cut 240→30 syncs: **4.94 vs 4.96 tok/s**. The fence is not a drain — branch A runs during it. Same ~50 ms wall appears on Apple M4 Metal with no WDDM and no PCIe. |
| Moving the router to the GPU | **Not bit-exact.** `rms_inv = (mss+eps).powf(-0.5)` is passed as a *host* f32 into the device quantizer precisely because CUDA can't reproduce MSVC CRT `powf`; softmax `probs` become `route_scales`. Both are arithmetic, not selection. |
| n-gram batched speculative verify | Expert-union penalty: B=4 draws 29.1 unique experts/layer (873 pairs vs 240) = 86–88% of the whole cache. Bytes/committed token improve only above ~4/4 acceptance. In-tree n-gram gives ~1.20 committed/round on code = a regression. `verify_batch` is also unreachable from this lane. |
| Q2_K requant of routed experts | **Already built and measured** (`models/gemma-4-26B-mixq2k-it.*`): degenerate output on all 3 A/B prompts, and only 14.89 t/s anyway. |
| LZ4/ZSTD compression of experts | Byte entropy **7.58–7.68 bits/byte** → any codec caps at **1.05×**; deflate-Optimal achieves 1.040–1.053. Plan needed 1.25×. Q4_0 *is* the compression. |
| Positioned reads instead of mmap **for decode** | −13.0%, 3/3 pairs. (For **prefill** it is a +39% win — see §3 #2. The answer is per-phase.) |
| Attention shared-memory re-index | Void: grid is `(heads,1,1)` = **16 blocks on 30 SMs**, so shared memory is not the occupancy limiter. Attention is also ~1% of budget at this context. |
| `CAMELID_GEMMA4_CUDA_OVERLAP_MISS_IO=1` | −1.79% even cold. |
| Rows-per-warp ordered fold | Null at R=4 and R=8. Intra-warp latency is hidden when the grid is ~11k warps deep. |
| CPU-router interleaving / serial | Interleaving null; serial is a **56% regression** (767 vs 493 ms). |
| Cache eviction policy | LRU 68.9% / TinyLFU 61.1% / LFU-aged 34.1% / partitioned 67.6% / Belady-OPT 90.3% on the real trace. **FreeToken independently ships plain LRU as its only policy.** Closed. |
| Pinned host tier for prefill | 5 GiB tier → **5.6% hit** (fills a uniform 51/128-per-layer stripe; prefill touches ~121 in routing order). Pinning 7 GiB on 16 GiB also starves the box. |
| Permanently pinning N layers in VRAM | Only pays *across* prefills (within one, each expert is read once anyway), and at N=6 it leaves 121 slots for the other 24 layers → decode hit rate ~48% vs 87.8%. |
| Whole-layer read-ahead | Null — layer-major already made the access sequential, so OS read-ahead has nothing left to add. |
| `.cghost` v3 SoA for routed experts | ~+4% warm, and needs a format version, a dual-fingerprint identity change, gating 4 silent-corruption lanes, and a 12 GiB rewrite. Not worth it. **Note the artifact is MIXED-format: `down_exps` is Q4_1 in layers 0–6, Q4_0 in 7–29.** |

---

## 6. Traps

- **Sequential GPU A/Bs on this laptop are worthless** — thermal drift once manufactured a phantom
  1.8× win. Always paired, alternating, cooled. `paired-ab.ps1` enforces it.
- **Free host RAM at load moves steady decode by more than most changes under test.** Record it
  with every run. Numbers on this lane have ranged 4.3 → 23 tok/s on identical code.
- **The router bucket is noisy: 722 / 710 / 493 ms across three runs of near-identical code.**
  Only same-binary adjacent comparisons mean anything.
- **NVRTC compiles at model load**, so `cargo build` cannot catch a kernel error and helpers must
  be defined **before** use. Always run a 4-token load smoke after touching the kernel string.
- PowerShell 5.1: native stderr becomes `ErrorRecord`s (redirect `*>` inside the child instead);
  the host formatter hard-wraps output (`-replace '\s+',' '` before any regex); `{6,+6:N1}` is not
  valid .NET alignment.
- `python` is not on PATH; `py` is. `jq` is not installed.
- `camelid inspect` on this GGUF dumps the entire chat template. Grep narrowly.

---

## 7. Open items, ranked

1. **Wire `q4_0_gemm_routed`** (§4). Long-prompt win; kernel already certified.
2. **`repack-ghost --verify`** — nothing in the tree proves the current 12 GiB `.cghost` is
   byte-correct against the GGUF. Probes say it is not a zero artifact; that is not a proof.
3. **Decode at long context** is ~9 t/s vs ~19–23 at short context. Never investigated why.
4. **Prefill storage is still 49%** of an 18 s prefill even after the read-path fix. The structural
   answer is residency, and on 16 GiB it does not close — see the §5 rows on pinning.
