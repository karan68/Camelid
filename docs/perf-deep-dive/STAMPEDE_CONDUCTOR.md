# STAMPEDE_CONDUCTOR.md — Windows Speed Campaign v2

Campaign to close the remaining Windows CPU gap vs llama.cpp and extend the CUDA lead, using only
levers left open by the 2026-06 velocity campaign's receipts. Blunt rule inherited from
`SPEED_FIX_PLAN.md`: no phase ships without a byte-identical greedy parity receipt and a
before/after measurement on the pinned harness. `unsafe` Rust is in-budget for the kernel phases;
every `unsafe` block gets a safety comment, a scalar reference twin, and a parity test.

- **Host:** i7-11800H (8C/16T Tiger Lake, AVX2+FMA; AVX-512 excluded — downclock, measured), ~16 GiB DDR4 (~51 GB/s), RTX 3060 Laptop 6 GiB (sm_86), Win11.
- **Baseline (from `PERF_GAP_REPORT.md`, post-P0.6):** CPU prefill 23.73 tok/s (0.80× llama), CPU decode 5.97 (0.66×), CUDA decode 0.77× (parity-locked residual accepted).
- **Reference:** llama.cpp — re-pinned in Phase 0 (see below). `acd79d6` is stale.
- **Harness:** `docs/perf-deep-dive/scripts/cpu-prefill-matrix.mjs` (CUDA hidden, nonce cache-defeat, greedy temp=0) + `llama-bench` true-CPU protocol from `PERF_GAP_REPORT.md §Methodology`.

## The reframe this campaign rests on

Decode was closed as "memory-bandwidth bound, no cheap bit-exact win." The receipts actually show
**neither engine is at the DRAM wall**: Camelid decode ≈ 5.97 × ~3.4 GB ≈ 20 GB/s (~40% of peak);
llama.cpp ≈ 9.08 × 3.4 ≈ 31 GB/s (~60%). The gap is *achieved* bandwidth — access pattern,
outstanding misses, thread placement, and per-token fork-join overhead — not a physics limit.
Every decode phase below attacks utilization, not FLOPs. The prior campaign's own read agrees:
"llama's decode edge is memory-access/prefetch + ~29% non-matmul per-token overhead, not dot speed."

---

## Phase 0 — Re-pin, re-baseline, regression guard (1 session)

The comparison target moved. llama.cpp has landed, since `acd79d6`: n-gram/ngram-map speculative
methods, MoE sum operators, continued CPU repack coverage, and CUDA changes. A campaign measured
against a stale pin produces stale claims.

- P0.1 Pull current llama.cpp master, build with the same recipe (Release, AVX-512+FMA+tinyBLAS+REPACK+OpenMP CPU; CUDA FA+graphs arch 86). Record commit, `--version`, build flags in `PERF_RECEIPTS/env/`.
- P0.2 Re-run the full same-host matrix (3B Q8_0 primary; Qwen3-4B/0.6B secondary; Q4_K_M rows if KQUANT conductor has minted receipts by then). Both engines CPU-true (`CUDA_VISIBLE_DEVICES=-1`), cache-defeated.
- P0.3 Wire the P0 regression guard from `SPEED_FIX_PLAN.md` if not yet landed: every subsequent phase lands only with a guard run attached.
- P0.4 Run `rayon_region_microbench` (`src/inference.rs:1429`) and record regions/token × cost/region for the 3B decode loop. This number gates Phase 4.

**Gate:** updated `PERF_GAP_REPORT.md` table with new pin. If the CPU gap has *widened* (llama improved), that's signal, not noise — it re-ranks the phases below. **GO** regardless; this phase cannot fail, only inform.

---

## Phase 1 — Windows thread placement (cheap, bit-identical, ships first)

`configure_rayon_threads` (`src/main.rs`) sizes the Windows pool to physical cores but never
**pins**. macOS workers get a QoS class; Windows workers get nothing — the scheduler is free to
co-schedule two workers on SMT siblings of one core and to migrate threads mid-decode, both of
which hurt a bandwidth-utilization-bound loop. llama.cpp exposes exactly this via its cpumask
(`-C`); Camelid has no equivalent.

- P1.1 In the decode/prefill pool `start_handler`s, pin worker *i* to logical CPU 2*i* (`SetThreadAffinityMask`), one per physical core; optionally `SetThreadIdealProcessor` as the softer variant to A/B. `windows-sys` already carries `Win32_System_Threading`.
- P1.2 Sweep: {no pin (baseline), ideal-processor hint, hard affinity} × {decode pool, prefill pool}. The prefill pool spans logical cores by design (P0.6) — pin pairs (2i, 2i+1) per core there, don't fight the shipped win.
- P1.3 Env flag `CAMELID_WIN_PIN` default-off until validated; flip default on a GO.

**Operator knobs shipped by this phase (Windows x86-64):**

| Variable | Default | Effect |
|---|---|---|
| `CAMELID_WIN_PIN` | off | `ideal` = `SetThreadIdealProcessor` hint, `hard` = `SetThreadAffinityMask`, per physical core. Measured negative on top of a correctly sized pool — see ledger 1.1. |
| `CAMELID_WIN_HYBRID_POLICY` | on | Set to `0`, `false`, `off` or `no` to stop sizing the decode pool to the performance-core count on a hybrid Intel host and restore the raw physical-core width. Any other value leaves the policy on, so a typo cannot silently disable a shipped default. Engages only on `GenuineIntel` reporting more than one `EfficiencyClass`; every other host is already unaffected. |
| `CAMELID_DECODE_THREADS` | unset | Explicit decode-pool width; wins over both of the above. `0`/`off` disables the dedicated pool. |
| `CAMELID_THREADS` | unset | Operator-pinned global. When set, the decode pool defers to it and neither default policy applies. |

Thread count only, no math change ⇒ **bit-identical by construction**; parity receipt is still minted.

**Gate:** decode ≥ +5% → **GO** (ship, default-on). +0–5% → ship default-off, note in ledger. Negative → **KILL**, document (the 4-thread contention peak from char-20260620 may already be the scheduler accidentally doing this).

---

## Phase 2 — Q8 GEMV streaming: prefetch depth + multi-row MLP (the `unsafe` phase)

§6D was stubbed and never landed: `GaitSubstrate.stream_prefetch_depth` sits at 0
(`src/gait/mod.rs:943`). This phase lands it, plus its stronger sibling.

- P2.1 **Software prefetch.** In the Q8_0 decode GEMV inner loop, `_mm_prefetch(ptr + D, _MM_HINT_T0)` on the weight stream, D swept over {2, 4, 8, 16} cache lines (GAIT calibrates D per host later; hardcode the sweep winner now). `unsafe` raw-pointer walk is fine; the loads themselves are unchanged.
- P2.2 **Multi-row interleave (memory-level parallelism).** Process R ∈ {2, 4} output rows per thread iteration with R independent accumulator sets, round-robining the loads so R miss streams are in flight per core instead of 1. Each row's int8→i32 block accumulation order is *individually unchanged* ⇒ bit-exact per row ⇒ byte-identical output. This is the mechanism behind llama.cpp's higher achieved bandwidth, separated from its tiling.
- P2.3 Re-test the rows4 repack **under the new consumer only**. The 2026-06-20 regression receipt condemned the old consumers, not the layout; a streaming multi-row consumer may flip the verdict. If it regresses again, the layout is dead for decode — write it down and stop re-litigating.
- P2.4 Measure achieved GB/s directly (weights-touched ÷ token time) and report utilization %, not just tok/s — that's the honest metric for this phase.

**Gate:** decode achieved-bandwidth ≥ 50% of peak (≈ 7.5 tok/s on 3B Q8_0) → **GO**. 40–50% → **PIVOT** to Phase 3 with partial credit. No movement → the utilization thesis is wrong on this memory controller; **KILL** the streaming pillar, decode hope shifts entirely to Phase 5.

**Research sub-lane (strictly opt-in, own flag, safe-boot sentinel):** Windows large pages
(`MEM_LARGE_PAGES`, 2 MiB) for the weight arena to cut TLB misses on the 3.4 GB stream. Requires
`SeLockMemoryPrivilege` and physically locks memory — this is GAIT-v1-crash-adjacent territory, so
it inherits the v2 rules: never default, sentinel file, degrade silently to 4 KiB pages on any
failure. Only attempt after P2.1/P2.2 receipts exist, to isolate its contribution.

---

## Phase 3 — P1 execution: unified tiled Q8 GEMM owner (prefill 0.80× → ~1.0×)

Already scoped in `SPEED_FIX_PLAN.md §P1` and `LLAMA_CPP_ARCHAEOLOGY.md §1–2`; this campaign
executes it. Register-blocked AVX2 (core::arch, `unsafe`) Q8×Q8→i32 micro-kernel over the repack,
K-loop once per output tile, in-kernel chunk scheduler via `par_chunks` over tiles, fixed
accumulation order (int accumulation is associative-safe here ⇒ bit-exact), f16 scale product
applied in fixed order. Prefill-only pool (P0.6) stays.

- P3.1 Kernel + scalar twin + property tests (random shapes, exact vs twin).
- P3.2 Route prefill batched linears through the owner behind `CAMELID_Q8_GEMM_OWNER`; per-role bespoke paths remain the fallback.
- P3.3 AVX-512 *prefill-only* variant as a sub-experiment (compute-bound prefill may amortize the Tiger Lake downclock — the standing untested idea from ARCHAEOLOGY §9). Measured, not assumed.
- P3.4 If the owner wins on prefill, A/B the *same* tiled consumer on decode with the Phase-2 streaming tricks folded in — the two phases compound or they don't; receipts decide.

**Gate:** prefill ≥ 28 tok/s (≈0.95× of the old llama pin; re-express vs the Phase-0 re-pin) → **GO**, promote owner to default. Between 24–28 → **PIVOT**: keep owner opt-in, profile the residual. Below shipped 23.73 → **KILL** the owner, keep receipts.

---

## Phase 4 — Per-token overhead: persistent spinning decode pool (gated by P0.4)

The streaming role profile attributes ~29% of decode token time to non-matmul work, part of it
rayon fork-join (park/unpark per parallel region; Windows wakeups are microseconds each, many
regions per token). llama.cpp's counter-design is a persistent threadpool that spins between ops
and sweeps a graph. Rust-native version:

- P4.1 **Gate first:** from P0.4, if (regions/token × measured region overhead) < 5% of token time, **KILL this phase immediately** — the 29% is then qkv/rope/norm/KV work, not fork-join, and belongs to Phase 2/3 kernels.
- P4.2 Persistent decode workers with bounded spin-then-park (spin budget ~50–100 µs, then park — never burn a core while the server idles; the API engine's idle behavior is a product constraint).
- P4.3 Fuse per-layer op sequences into fewer parallel regions: fixed per-thread output-range ownership across norm→qkv→rope→attn-out→ffn within one region. Each output element computed by the same thread with the same per-element order ⇒ byte-identical.
- P4.4 Fold the Phase-3 chunk scheduler into the same pool so decode and prefill share one worker set with phase-adaptive width (preserving P0.6's win).

**Gate:** decode ≥ +8% over the Phase-2 result → **GO**. Else **KILL**; the audit precedent ("zero confirmed micro-wins") says be ruthless here.

---

## Phase 5 — Model-free speculative decode on the CPU path (the leapfrog lane)

P3-old rejected CPU spec decode because the **0.6B model drafter** needs ~70 tok/s and has ~28.
That verdict does not bind the **model-free** drafters already in-tree:
`src/inference/suffix_decoding.rs` (frequency-weighted suffix tree, zero forwards) and
`src/inference/token_recycling.rs` (adjacency drafter, zero forwards). `CAMELID_SPEC_TREE`
currently verifies only via `verify_tree_gpu` (`src/main.rs:~3512`) — the CPU box never benefits.

Why this wins where dots can't: decode is utilization-bound, and a **batched CPU verify of k
tokens costs ~one weight pass** — the prefill receipts prove CPU batching amortizes ~3.3×. At
5.97 tok/s plain, an average of just ~1.5 accepted tokens/round is ≈ 9+ tok/s effective —
**past llama.cpp's 9.08 — while staying lossless** (greedy verify is authoritative, per the
existing lane's contract). llama.cpp itself has been landing n-gram spec methods on master, so
this also keeps pace with the reference's direction.

- P5.1 Wire the existing CPU chunk verify (`forward_greedy_verify_chunk` path) as the `CAMELID_SPEC_TREE` verifier when no resident GPU engine is up; linear (k=2..4) before tree.
- P5.2 Port the acceptance-gated run-length latch policy verbatim from the GPU lane (its workload separation — repetitive GO / prose SKIP — was measured on this box and the economics are *better* on CPU because plain decode is slower relative to batched verify).
- P5.3 Measure the 4-workload matrix (repetitive/code/json/prose) CPU-only; publish accepted/round and net speedup per workload. The honest claim will be workload-dependent — say so, like the GPU lane does.
- P5.4 Cross-wire with camelid-turbo/TDGP later only if the latch shows headroom; not in this campaign's scope.

**Gate:** any workload class ≥ 1.3× with zero regression on latched-off classes → **GO** (ship default-on with the latch). All classes < 1.1× → **KILL** with the acceptance histogram as the receipt.

---

## Phase 6 — CUDA default path: P2 multi-stream overlap (~10–15% decode)

Scoped in `SPEED_FIX_PLAN.md §P2`, user-facing (default path on the GPU box), parity-safe (no
re-association — independent Q/K/V and gate/up GEMVs computed identically, just concurrently).
Execute as written: separate streams in `src/cuda_resident.rs`, event-join before dependents,
verify interaction with the graph-captured decode path (streams must be capturable or the overlap
applies to the live path only — decide with a receipt, not an assumption). Env flag default-off →
token-identical validation → flip.

**Gate:** CUDA decode ≥ +8% at low ctx, no depth regression → **GO**.

---

## Explicitly NOT in this campaign (standing KILLs honored)

- Non-bit-exact flash attention (CPU or CUDA) — losslessness contract.
- AVX-512 decode / VNNI decode — measured downclock/no-op, reverted.
- Re-enabling the old packed-rows4 *consumers* — condemned by receipt (P2.3 tests the layout under a new consumer once, then it's settled).
- Sampler/server/tokenizer/mmap — ruled out as non-bottlenecks by both engines' measurements.
- Model-drafter CPU spec decode — blocked until a P3-class kernel changes the drafter economics.

## Ledger

| Phase | Lever | Predicted | Effort | Parity risk | Status |
|---|---|---|---|---|---|
| 0 | Re-pin + baseline + guard | — | S | none | **DONE 2026-07-08** — pin b9918/0512ef1e5; receipts `stampede-p0-baseline-2b8b97c4-20260708T0715Z/`; guard `scripts/stampede-guard.mjs` |
| 2.0 (new) | **GQA QKV decode parallelization** (`inference.rs:13942` serial else-branch) | decode +15–20% | S | none (rows independent, per-row order unchanged) | **DONE 2026-07-08 (win-x86_64 defaults)** — measured **+37%** 3B Q8 (8.15→11.17, ratio 0.92×→**1.21×** AHEAD) and **+33%** Qwen3-4B Q8 (6.43→8.56, 0.84×→**1.17×** AHEAD); greedy text byte-identical OFF↔ON and vs P0 receipts; guard PASS ×2; bitwise unit test (GQA shape, both chunking modes); 15-agent adversarial review: 4 deduped minors fixed (comment equivalence caveat, test serial-degradation guard, knob-crossover documented, claim scoped). Flag `CAMELID_X86_Q8_QKV_GQA_PARALLEL_DECODE` default-on. Receipts `stampede-p20-qkv-gqa-{OFF,ON}-*-20260708.json` |
| 1 | Win thread pinning | decode +0–10% | S | none (bit-identical) | **DONE 2026-07-08 — shipped DEFAULT-OFF per gate**: ideal +1.1% / hard +1.9% decode, +1–2% prefill on 3B Q8 (inside the 0–5% band; the #362 physical-core decode pool already captures most placement benefit — on a hybrid Intel host that pool is now performance-core sized, see 1.1). `CAMELID_WIN_PIN={ideal,hard}`, per-core sibling masks from `GetLogicalProcessorInformation`; receipts `stampede-p1-winpin-{off,ideal,hard}-llama3b-q8-20260708.json` |
| 1.1 (new) | **Hybrid decode width** — size the decode pool to the performance-core count, not the raw physical count | decode +10–20% on hybrid Intel | S | none (bit-identical; changes a thread count, not the arithmetic) | **DONE 2026-07-31 — ships DEFAULT-ON, gated to `GenuineIntel` + more than one `EfficiencyClass`.** The #362 physical-core policy was derived on an i7-11800H, a non-hybrid part; on an i9-14900HX `windows_physical_core_count()` returns 24 (8P + 16E) and batch-1 decode fork-joins across all 24 every token. Sizing to the 8 performance cores measured **+5.296 tok/s (+16.7%, CI [+4.992, +5.600], 6/6)** and **−55.3% CPU seconds** on 1B Q8, 1 distinct SHA-256 across all 54 measured requests, TTFT unmoved (prefill keeps its own wider pool). The width sweep peaks at the performance-core count and falls off both sides (4: −2.976, 8: +4.915, 12: +3.664, 16: +1.958 vs the 24 default), so the `EfficiencyClass` boundary is the lever, not "something smaller". **Ships no placement**: against the same-width unpinned pool, `SetThreadIdealProcessor` scores −0.546 (n.s.) and `SetThreadAffinityMask` −1.686 (CI [−2.267, −1.105], 0/6) — `CAMELID_WIN_PIN` is untouched. AMD is gated out by CPUID rather than measured — Zen 5c and multi-CCD parts also report several classes without being a P/E split. Kill switch `CAMELID_WIN_HYBRID_POLICY=0|false|off|no` restores the previous width exactly; `CAMELID_DECODE_THREADS` and `CAMELID_THREADS` keep their existing precedence. Receipt `stampede-p1-hybrid-decode-width-14900hx-20260731.json` |
| 2 | Prefetch + multi-row MLP | decode +10–20% | M (`unsafe`) | low (order preserved) | pending |
| 3 | Tiled GEMM owner — **scope widened: Q8_0 AND Q4_K_M** (K-quant prefill is 0.15× with no owner) | prefill 2–4× | M–H (`unsafe`) | low (int assoc) | **IN PROGRESS 2026-07-08** — Lane A (Q8): re-validated at b9918 with engaged-checked paired sweep (+12.3% 3B / +11.9% 4B, CI excludes 1.0) → **default FLIPPED win-x86_64 (D15)**; prerequisite fix: bench-owner-sweep was silently measuring cached plans (fake-null trap) — uncached-plan bypass + engaged-check landed. Lane B (Q4_K): NEW batched prefill owner `CAMELID_X86_KQUANT_MATMUL_OWNER` (opt-in), v1 unpack-hoist +29% → v2 vector-accumulate **+50%** (14.94→22.39 tok/s 3B Q4_K_M, 0.15×→0.23×), bitwise twin tests + byte-identical e2e; decode-only probe exonerates decode (medN dip = prefill-coupled thermal). PIVOT band per gate. **Q6_K sibling DONE — combined owner 1.73× on 3B Q4_K_M** (13.74→23.79 single-engine, engaged counter off=0/on=392): the first flat receipts measured a PRODUCTION-UNREACHABLE dispatch (the Q6_K wrapper carried an inline duplicate of the core and never delegated — caught by adversarial review as a MAJOR; the "~4% Q6_K share" explanation is retracted). Wrapper now delegates (Q4_K pattern). Pure-Q6_K serve validation impossible — local requants route to the EXPERIMENTAL lane, never native kernels (`stampede-p3-q6k-sibling-verdict-20260708.md`). **AVX-512 VNNI main-side DONE: combined owner 1.87× (27.41 tok/s 3B Q4_K_M single-engine, +12.5% over the AVX2 inner; `CAMELID_X86_KQUANT_MATMUL_OWNER_VNNI` default-on-when-owner-on; `stampede-p3-kquant-vnni-ab-20260708.txt`).** **8-ROW REPACK DONE (Lane B v5, 2026-07-09): `CAMELID_X86_KQUANT_MATMUL_OWNER_REPACK8` opt-in — lazy budget-gated `Q4KPackedRows8` (+5.6% memory, wire bytes kept for GPU residency/embedding, `CAMELID_X86_KQUANT_REPACK8_BUDGET_MIB` default 2048), 8-row dpbusd GEMM with one 64-byte activation load shared by 8 output cells — 3B Q4_K_M prefill 39.18 tok/s = 2.82× vs off, 1.49× over the VNNI inner (gate 1.15×: PASS), row now ~0.42× of b9918 from 0.15× at campaign start (`stampede-p3-repack8-ab-20260709.txt`).** **256-BIT AVX-VNNI INNER DONE (2026-08-02): the VNNI inner above was gated on `avx512f+avx512bw+avx512vnni`, so every consumer Intel part from 12th gen onward — AVX-VNNI present, AVX-512 fused off — was pinned to the AVX2 inner. `q4_k_owner_weight_row_block_avxvnni` is a ymm-width twin selected only when the 512-bit gate is false: **+11–13% over the AVX2 inner, 44/44 paired rounds** on an i9-14900HX (no AVX-512), across two architectures and four prompt lengths — 3B Q4_K_M 1.1337×/124 tok, 1.1315×/606 tok (repeat 1.1348×), 1.1141×/2421 tok; Qwen3-4B Q4_K_M 1.1252×/123 tok — matching the +12.5% the 512-bit inner scored on an AVX-512 host. Bit-identical (engaged-checked twin test asserting `avxvnni_taken > 0` before comparing; 1 distinct SHA-256 e2e). AVX-512 hosts are provably untouched (the branch requires `!use_vnni`); repack8 untouched. Engagement is exactly **6 of 7** owner matmuls per layer on BOTH models (28×7=196 vs 28×6=168; 36×7=252 vs 36×6=216) — the 7th is the per-layer Q6_K tensor, and `q6_k_owner_prefill_tiled` has **no VNNI path on any host**, which is the next lever. Note the same 512-bit-only gate sat on the **Q8 owner** (`q8_owner_avx512vnni_available`, `inference.rs:16777`), which is already default-on — so Q8 on 12th-gen-and-later consumer Intel was on its AVX2 inner, unmeasured. That is the Lane A entry immediately below, landed 2026-08-03 via #598; this note is kept for the causal record, not as an open item. Receipt `stampede-p3-kquant-owner-avxvnni256-14900hx-20260802.json`.** **256-BIT AVX-VNNI INNER FOR LANE A DONE (2026-08-03): the Q8 owner's VNNI microkernel was gated on `avx512f+avx512bw+avx512vnni`, so every consumer Intel part from 12th gen onward — AVX-VNNI present, AVX-512 fused off — ran the AVX2 `maddubs`+`madd` inner. Unlike Lane B, **this owner is DEFAULT-ON for win-x86_64**, so the slow inner was shipping. `q8_0_packed_rows4_gemm4_accumulate_block_avxvnni` is a ymm twin selected only when the 512-bit gate is false: **+5–7% over the AVX2 inner, significant in all six measured cells, 57/58 paired rounds** on an i9-14900HX (no AVX-512) — 1B Q8_0 1.0735/1.0681/1.0459 and 3B Q8_0 1.0698/1.0694/1.0581 at 124/606/2421 prompt tokens. Byte-identical (microkernel test vs BOTH the scalar oracle and the AVX2 inner, ablated; 1 distinct SHA-256 per model across off/avx2/avxvnni e2e). AVX-512 hosts provably untouched (`!use_vnni`); the 4x8 tile is AVX-512-only and unreachable here. **The bigger result is the owner-vs-off flip: on 3B, `owner_avx2` was LOSING to `off` on this host (off/avx2 = 1.0396/1.0336/1.0188) and the new inner turns that into a 2.5–3.9% WIN.** **OPEN — do not treat as settled:** on 1B the owner is still break-even at short prompts and a **significant 2.7% LOSS vs off at 2421 tokens (0/8 rounds)** even with the new inner; the owner's default-on receipt was taken on a 3B model on a non-hybrid i7-11800H, so neither the small-model nor the hybrid-core case was ever covered. Prerequisite fix: the Q8 sweep hard-coded `owner_vnni4x4`/`4x8` with no capability branch, so on a non-AVX-512 host those arms measured the AVX2 inner against itself (fake null); it is now capability-aware. Receipt `stampede-p3-q8-owner-avxvnni256-14900hx-20260803.json`.** **LANE B PROMOTED TO DEFAULT-ON (win-x86_64, 2026-08-04): the gate above said prefill ≥ 28 tok/s → GO / 24–28 → PIVOT; Lane B was parked in the PIVOT band at v2 and the flag was never revisited after the lane got faster four more times (Q6_K sibling, AVX-512 VNNI, repack8, and the 256-bit AVX-VNNI inner in #593).** Re-checked as ratio-vs-shipped-default with zero regression: **1.56×–1.88× over `off`, 9/9 cells significant, 66/66 paired rounds**, across Llama-3.2-1B/3B and Qwen3-4B Q4_K_M at 124/606/2421 prompt tokens — and `owner_avx2` alone is 1.41×–1.61×, so the promotion does not depend on AVX-VNNI. Byte-identical output (one SHA per model off↔on), peak RSS within ±0.25% (noise), decode untouched (owner declines <4 rows). **The deciding test was the small-model case**: the Q8 owner REGRESSES on 1B (0.9730, 0/8) so Llama-3.2-1B Q4_K_M was pulled specifically to look for it — it instead shows the largest wins measured (1.8768/1.7488/1.5619). One contaminated cell (qwen4b p2421, `off` spanning 10.18–24.12 tok/s) was re-run clean rather than averaged in. `kquant_prefill_owner` now resolves via `env_flag_default_on` on win-x86_64 only and stays opt-in elsewhere; kill switch `CAMELID_X86_KQUANT_MATMUL_OWNER=0|false|off|no|disabled` verified on the shipped binary to restore the pre-promotion timings exactly, and any other value leaves it on so a typo cannot disable a measured default. Receipt `stampede-p3-kquant-owner-default-on-14900hx-20260804.json`.** Receipts `q8-prefill-owner-b9918-revalidation-20260708/` + `stampede-p3-kquant-owner-*.json` + `stampede-p3-q6k-single-engine-ab-20260708.txt` |
| 4 | Spinning pool / region fusion | — | — | — | **KILLED by P0.4**: fork-join = 0.4% (hot) to 5.5% (all-cold bound) of token time; census receipt `stampede-p04-region-census-2b8b97c4-20260708.md` |
| 5 | Model-free CPU spec decode | decode 1.3–2× (workload-dep) | M | none (lossless verify) | **Stage A LANDED 2026-07-09, economics KILLED at current verify cost (ships default-off)** — P5.1 CPU chunk-verify + rollback wired into the spec-tree lane (primary-chain flatten, one-way resident ratchet, `CAMELID_SPEC_CPU_VERIFY=0` kill-switch); P5.2 latch extracted to `speculative::SpecLatch` (5 unit tests); LOSSLESS on all 5 workload cells; acceptance PROVEN (5.45 drafts/round on repetitive, 64%); blocker: 8-row chunk verify costs **7.1× a decode step** (premise ≲3× from large-M receipts — small-M amortization is ~nil), capping even perfect acceptance at 1.13×. **Small-M fixes LANDED 2026-07-09**: gate/up bespoke arms decline 2..=16-row batches (flow to owner, 6×→owner-level), verify chunk on the prefill pool — repetitive **1.08→1.265×** lossless, matrix 0.905-0.952 elsewhere, large-chunk prefill unaffected (sanity sweep matches Lane A receipts). Remaining lever to the 1.3× bar: batched small-M chunk attention (actx = 60% of verify layers). `stampede-p5-cpu-spec-verdict-20260709.md` |
| 6 | CUDA multi-stream | GPU decode +10–15% | M | none (no re-assoc) | **IMPLEMENTED 2026-07-09, measured NULL on WDDM — ships default-OFF** (the risk the design named). K/V + FFN-up overlap per plan (`CAMELID_CUDA_STREAMS`, Full layers, live path only, kernels launch unchanged; lazy side-stream construction keeps flag-off in single-stream mode). Correctness: byte-identical greedy OFF==ON on 5 legs × 2 rungs (3B Q8 low+depth/split-K, Qwen3-4B Q8 QK-norm, Qwen3-4B Q4_K_M K-quant, NO_FUSION, qwen35 device loop), engaged-checked; device tests 34/34 both states. Perf at the gate (≥+8% low-ctx both models): **Rung A (tracking on) −9.5%/−2.5% low ctx; Rung B (`disable_event_tracking`) −9.0%/−4.2%** — the A/B localizes the cost to WDDM launch-batching/co-scheduling, not cudarc bookkeeping (review correction: the context is in cudarc multi-stream mode even flag-off — the engine stream is a `new_stream` — so Rung B's ON legs paid LESS host bookkeeping than their OFF baselines and still regressed); depth flat at Rung B (Rung A's one valid depth cell, 3B, was −7.6% with tracking on). Falsifiable follow-up: Linux/TCC should show the predicted win (flag exists, one-session A/B on a capable host). Verdict + receipts: `qa/perf/stampede-p6-cuda-streams-verdict-20260709.md`, `stampede-p6-cuda-streams-*-20260709{,-rungB}.json` (invalid Rung A 4B-depth cell renamed `*.INVALID-cap-overflow.json`). Side hardening: AB probe hard-fails on missing content + node:http (undici 300s trap) + setEncoding, validate script asserts corpus key counts, bench runner refuses a dirty GPU/CPU (committed in the script), forward_pass drains on Err while overlap is armed — after three real contamination events (concurrent-session checkout swap, leaked GPU-resident server, zombied bench tree). |

### Phase-0 gate outcome (2026-07-08) — the re-rank

Baseline moved on BOTH sides since the brief: llama.cpp b9918 CPU prefill improved ~68% on Q8
(repack GEMM progress) and Camelid decode improved ~30% (#362 win-default promotion). At the new
pin: decode is nearly closed (0.84–1.08×; 0.6B already AHEAD at 1.08×) while prefill is the
campaign: 0.42–0.46× on Q8_0, **0.15–0.16× on Q4_K_M**. Re-ranked execution order:

1. **Phase 2.0** — parallelize GQA QKV decode (serial single-thread today; ~13.7% of weight
   stream). Small, parity-safe by construction, modeled ≈ +20% decode → likely puts 3B/4B decode
   at ≥ 1.0×.
2. **Phase 1** — thread pinning sweep (cheap, bit-identical; compounds with 2.0).
3. **Phase 3** — tiled GEMM owner, now covering Q8_0 + Q4_K_M prefill (the 0.15× row is the
   single biggest prize in the matrix).
4. **Phase 2** — prefetch/multi-row streaming on whatever decode gap remains.
5. **Phase 5 / Phase 6** — unchanged.

Original decode target (~8.5–9.5 tok/s on 3B Q8) is now within reach of Phase 2.0+1 alone;
Phase 5 remains the lane that can exceed the reference. Prefill target re-expressed vs b9918:
Q8 ≥ 0.9× (≈ 46 tok/s), Q4_K_M ≥ 0.6× (≈ 54 tok/s) this campaign.
Every number above is a prediction, not a claim — receipts or it didn't happen.
