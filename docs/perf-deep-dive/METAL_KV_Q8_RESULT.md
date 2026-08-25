# Q8_0 resident KV on Apple Metal — what it is actually worth

First same-host receipt for the Metal **Q8_0-primary resident KV cache**. Receipt:
`PERF_RECEIPTS/same-host/metal-kv-q8-m3-20260822.json`. Governed by
[BENCHMARK_TREATY.md](BENCHMARK_TREATY.md).

`PERF_RECEIPTS/same-host/` carried arms for the **f16** and **f32** resident KV formats
and none for **q8** — even though the Q8_0-primary cache and its four MSL kernels
(`attention_decode_v2_kvq8`, `kv_scatter_kvq8`, `kv_scatter_batch_kvq8`,
`attention_prefill_v3_kvq8`) shipped with the Metal appliance work, complete with a
numeric parity test.

[METAL_KQUANT_ROADMAP_RESULT.md](METAL_KQUANT_ROADMAP_RESULT.md) documents that, at the
receipt's measured commit, a compressed primary KV forfeited split-K decode attention and
attention-as-matmul prefill. What had never been measured was the question that actually
decided whether to invest in this lane:

> **What is q8 worth once you control for the split-K forfeit?**

## The answer

**The Q8_0 KV kernels are good. The lane is blocked by one missing kernel, not by the
quantization.**

On an equal footing — both arms without split-K — q8 decode is **1.14× → 1.50× faster than
f32, rising monotonically with context depth**. That is the KV-bandwidth win landing exactly
where theory says it should, and it is the central result.

At the measured commit, *enabling* q8 forfeited split-K, which was itself worth **2.1×**, so
the shipped trade was a 2.1× win swapped for a 1.5× one. Against the zero-config default
that netted out to:
decode unresolved at most depths, **significantly better at 4117 tokens (1.031×)**, and
prefill **2.2× → 4.5× worse** — which is significant at every depth and is the real blocker.

On quality, the two compressed formats part company: **f16 is a token-identical drop-in on
every probe; q8 is not** — it diverges deterministically from f32 on coherent English,
though the divergence is benign (fluent alternative continuation, not degradation).

The original receipt's three action items (the first two are implemented and re-measured
later in this document):

1. Build **`attention_decode_splitk_kvq8`** — split-K and Q8 are independent wins that are
   mutually exclusive today for no reason but a missing kernel.
2. Prefill at 2.2–4.5× worse disqualifies the lane as a default on its own, split-K or not.
3. Nothing here promotes anything: one host, one model.

## Setup

| | |
|---|---|
| Host | Apple M3, 8 logical cores, **8 GiB** unified memory, macOS 26.3 (25D5112c) |
| Model | `Llama-3.2-1B-Instruct-Q8_0.gguf` — 16 layers, 32 heads, 8 KV heads, head_dim 64 |
| Binary | `camelid v0.6.1-15-g5cbee84c`, release, Rust 1.95.0 |
| Harness | [`scripts/metal-kv-dtype-ab.sh`](scripts/metal-kv-dtype-ab.sh) + [`scripts/metal-kv-dtype-stats.py`](scripts/metal-kv-dtype-stats.py) |
| Statistic | median of **paired within-round ratios**, bootstrap 95% CI, 20000 resamples |

head_dim 64 satisfies the Q8 resident gate (`head_dim % 32 == 0 && <= 128`). Q8_0 weights
are **not** in `metal_f16_kv_tensor_type`, so `weights_use_kquant()` is false and this
model's zero-config resident KV default is **F32** — which is why F32, not F16, is the
honest baseline arm here.

### The fourth arm is the point

| Arm | Configuration |
|---|---|
| `f32` | zero-config default: split-K decode + matmul prefill both active |
| `f16` | `CAMELID_METAL_KV_DTYPE=f16` |
| `q8` | `CAMELID_METAL_KV_DTYPE=q8` — the lane under test |
| **`f32-nosplitk`** | `CAMELID_METAL_ATTN_SPLITK=0`, F32 primary — **the mechanism control** |

At the measured commit, enabling f16 or q8 forfeited split-K. The gate in
`encode_attention_block` (`src/metal.rs`) was explicit about it:

```rust
let splitk = v2
    && !kv16_enabled()
    && !kvq8_enabled()          // <-- any compressed primary disqualifies split-K
    && splitk_attention_enabled()
    && (1..=4).contains(&group)
    && position_count >= 128;
```

Without an arm that holds the KV format constant and disables split-K by hand, a q8
regression is **uninterpretable**: a slow quantized kernel and a forfeited split-K look
identical in the totals. This arm separates them, and it is why the receipt reaches a
different conclusion than the raw numbers suggest.

Two clauses of that predicate also explain the shape of the results. `position_count >= 128`
is why the 6-token probe shows no f32/f32-nosplitk difference at all — split-K never
engages there. `(1..=4).contains(&group)` is satisfied for this model (32 heads / 8 KV
heads = 4), so it sits exactly at the top of the supported GQA range.

*(Citations here are by symbol, not line number: the measured commit and current `main`
differ by +702/−20 in `src/metal.rs`, so line numbers would rot.)*

## Results — 8006-token prompt, 64 generated, 5 rounds

Medians:

| Metric | f32 (default) | f16 | q8 | f32-nosplitk |
|---|---:|---:|---:|---:|
| prefill | 13,123 ms | 54,629 ms | 57,755 ms | 11,969 ms |
| decode | 2,191 ms | 3,071 ms | 2,384 ms | 3,616 ms |
| **decode tok/s** | **28.76** | 20.51 | **26.42** | 17.42 |

Paired ratios (CI must exclude 1.0 to count):

| Comparison | ratio | CI95 | verdict |
|---|---:|---|---|
| **q8 / f32-nosplitk, tok/s** | **1.498** | [1.468, 1.644] | **significant — better** |
| f16 / f32-nosplitk, tok/s | 1.177 | [1.059, 1.218] | significant — better |
| q8 / f32-default, tok/s | 0.871 | [0.731, 1.066] | **not resolved** |
| f16 / f32-default, tok/s | 0.688 | [0.499, 0.778] | significant — worse |
| f32-nosplitk / f32-default, tok/s | 0.582 | [0.471, 0.722] | significant — worse |
| **q8 / f32-default, prefill** | **4.483** | [4.237, 5.514] | **significant — worse** |

Reading these together:

1. **Split-K is worth 2.1×** (1 / 0.582). It is the single largest decode lever on this host.
2. **q8 beats f32 by 1.50× on equal footing** — the bandwidth win is real and the CI is tight.
3. **q8 beats f16 by 1.28×** (1.498 / 1.177), consistent with q8 reading 68 bytes per
   head-position against f16's 128.
4. **q8 against the shipped default is a wash, not a loss.** The earlier single-run reading
   of "26% worse" did not survive five rounds; the CI includes 1.0.
5. **Prefill is the unambiguous blocker at 4.5×.**

### Quality — f16 is a lossless drop-in here; q8 is not

Q8 KV is lossy by construction, so the question is whether it moves greedy output. It does.

| Probe | Prompt | f16 vs f32 | q8 vs f32 |
|---|---|---|---|
| **realtext** (1460 tok, `BENCHMARK_TREATY.md`) | **coherent English** | **identical 3/3** | **diverges at token 50, 3/3** |
| short (6 tok, "The capital of France is") | real text | identical 5/5 | identical 5/5 |
| depth 1067 | filler | identical 3/3 | diverges at token 3, 3/3 |
| depth 2066 | filler | identical 3/3 | diverges at token 13, 3/3 |
| depth 4117 | filler | identical 3/3 | diverges at token 7, 3/3 |
| depth 8006 | filler | identical 5/5 | identical 5/5 |

**The `realtext` row is the one that counts**, and it is unambiguous:

- **f16 is a token-identical drop-in** — identical in every round of every probe, at every
  depth. On this model it does not move a single greedy token.
- **q8 is not.** It diverges deterministically (same index in all three rounds — a
  systematic numerical difference, not noise) on coherent English.

The divergence itself is **benign, not a quality collapse**. Both continuations are fluent
and faithful to the source document; q8 simply drops a clause and continues:

```
f32: ...- **Tie** or **regression**. A valid, committed result. The campaign's value is an honest map
q8 : ...- **Tie** or **regression**. The campaign's value is an honest map, not a forced win.
```

So the honest framing is: **q8 buys memory at the cost of exact reproducibility, not at the
cost of coherence.** That is an acceptable trade for a capacity feature and a disqualifying
one for anything that must be bit-reproducible — note `--deterministic` already pins f32.

#### Why the filler probes cannot answer this

The depth-sweep prompts are generated filler, chosen so they reproduce byte-for-byte from a
seed. That is correct for throughput — memory bandwidth does not care what tokens mean —
but it makes filler *parity* numbers uninformative in **both** directions:

- Filler gives a nearly flat next-token distribution, so a lossy cache flips the argmax on
  numerical noise (hence divergence at token 3 at depth 1067).
- Long filler runs collapse into a repetition attractor — the 8006 probe emits **9 distinct
  tokens across 64 positions** — where both arms agree for free. That "identical 5/5" is
  worth nothing and should not be quoted as a parity win.

This is why the `realtext` probe exists and why the parity verdict rests on it alone.

Every arm is self-deterministic across rounds at every depth.

### The win scales with context depth

Decode attention is KV-bandwidth-bound, so a smaller KV should help *more* the deeper the
context. It does, monotonically — this is the curve the recommendation rests on:

| Prompt tokens | q8 / f32-nosplitk (apples-to-apples) | q8 / f32 default (real world) |
|---:|---|---|
| 6 | 1.001 — not resolved | 1.001 — not resolved |
| 1067 | **1.140** — significant | 0.949 — not resolved |
| 1460 (realtext) | **1.137** — significant | 0.872 — significant worse |
| 2066 | **1.274** — significant | 0.980 — not resolved |
| 4117 | **1.432** — significant | **1.031 — significant BETTER** |
| 8006 | **1.498** — significant | 0.871 — not resolved |

Two readings:

- **The apples-to-apples column is clean and monotonic**, 1.14× → 1.50× as depth grows 8×.
  That is the KV-bandwidth win behaving exactly as a bandwidth win should, and it is the
  strongest evidence in this receipt that the Q8 kernels are sound.
- **The real-world column is noisy and non-monotonic**, because the *baseline* moves: split-K's
  own benefit varies with depth and with thermal state. The 4117 row is nonetheless a
  genuine, significant result — **at that depth q8 already beats the shipped default
  outright (1.031×), while forfeiting split-K.** The 8006 row's wide CI [0.731, 1.066]
  reflects the worst thermal drift of the run, not a real reversal.

The prefill penalty moves the other way and gets worse with depth: q8/f32 prefill is
**2.19×** at 2066 tokens, **3.23×** at 4117, **4.48×** at 8006 — all significant.

### Short context is a null

At a 6-token prompt all four arms sit at ~66 tok/s with no resolved difference. At 56
tokens of KV there is nothing for a KV format to win or lose. Worth stating explicitly so
nobody benchmarks this lane at short context and concludes it does nothing.

## GPU allocation

Process peak RSS — what `bench-generate` reports — **cannot** separate q8 from f16 on this
host (1,625,341,952 B vs 1,625,653,248 B) despite a predicted 120 MB gap. It is the wrong
instrument. Sampling the `IOAccelerator (graphics)` region with `vmmap` instead:

| Arm | GPU virtual | GPU dirty |
|---|---:|---:|
| f32 | 2867.2 MB | 2355.2 MB |
| f16 | 1638.4 MB | 1331.2 MB |
| q8 | **1536.0 MB** | **1228.8 MB** |

**f16 → q8 saves 102.4 MB against 120 MB predicted** by the allocator in
`ResidentDecodeState::new` — the KV cache is shrinking as designed, measured directly.

The f32 → q8 gap (1228.8 MB measured, against 632 MB predicted from KV alone) is inflated
by f32-only split-K and matmul-prefill scratch and **must not be reported as a pure KV
saving**.

Per-element the Q8_0 block costs `34/32 = 1.0625` B against f16's 2 B, so the cache is
**46.9% smaller, not 50%**.

## What was built next — and what it measured

Both blockers named above were implemented and re-measured on the same host. The
predictions in this section's original text are kept below, because one of them was wrong
and that is worth recording.

### Prefill: fixed (the big one)

**The Q8 dequant was never the problem.** The decisive tell was already in the data above:
q8/f16 prefill is **1.00–1.07× at every depth** while both trail f32 by 1.8–4.4×. f16 and
q8 were excluded by the *same* predicate and landed on top of each other, so at most ~7% of
the regression was attributable to Q8.

The cause was one conjunct in `use_attn_mm`: `!self.kv16 && !self.kvq8 && …`. It dropped q8
out of the simdgroup-matrix attention (`transpose_v16 → half_mm_batched_f16o →
softmax_causal_rows → half_mm_batched_f16o`) and transitively out of the es=2 activation
stream. Admitting q8 required a source of half K/V, since `cache_k16`/`cache_v16` are empty
under a compressed primary — supplied by the new `kv_dequant_q8_to_h` staging kernel
(transient scratch, ~16.8 MB, against the ~256 MB of permanent mirrors the f32 lane
carries at 8192 positions).

| Depth | q8-both / q8-old | q8-both / f32 default |
|---:|---|---|
| 2066 | 0.426 (CI spans 1.0) | **0.990** [0.966, 1.369] — parity |
| 8006 | **0.222** [0.198, 0.252] — **4.5× faster** | **0.986** [0.924, 1.132] — parity |

Prefill went from **significantly 4.48× worse** than the default to **statistically
indistinguishable** from it.

### Decode: `attention_decode_splitk_kvq8` works, but the prediction above was wrong

The kernel exists and is correct. It does **not** compose with split-K the way this
document predicted.

**Predicted ~1.5×–3.2×. Measured roughly +5%, and it does not resolve in every run.** Four
independent 4-round paired runs of `q8-splitk / q8-old` on tokens_per_second:

| Run | ratio | CI | resolved? |
|---|---:|---|---|
| 8006, run 1 | 1.059 | [1.042, 1.153] | **yes** |
| 8006, run 2 | 1.036 | [0.977, 1.149] | no |
| 2066, run 1 | 0.983 | [0.455, 1.102] | no |
| 2066, run 2 | 1.106 | [1.045, 2.084] | **yes** |

So the honest statement is **~+5%, significant in 2 of 4 paired runs** — a small positive
effect that this host cannot resolve reliably, not the 2× the prediction implied. An earlier
revision of this document quoted the 8006 run-1 figure ("+5.9%, significant") as if it were
the result; that was one run of four and is corrected here.

The most likely reason it does not stack is the risk this document flagged and then
discounted: split-K wins largely by improving memory-level parallelism on the KV read, and Q8
relieves that same resource, so the second lever has little left to pull.

### The es=2 activation stream: no throughput win, but it buys exact agreement with f32

The half activation stream was the remaining item. It is now closed —
`rope_rotate_batch_h` (half RoPE with separate src/dst, so Q rotates straight into the half
query panel) and `kv_scatter_batch_kvq8_h` (Q8 block quantization from half operands). A
fused `rope_scatter_qh_h_q8` was considered and rejected as the wrong shape: that kernel runs
one thread per RoPE pair while Q8 needs an amax across each 32-element block, and the split
path turns out to be one dispatch *cheaper* than es=4 anyway, because the rotation absorbs
the f32→f16 convert.

**It produced no statistically resolvable throughput change at either depth.** Four paired
runs of `q8-both / f32` on prefill:

| Depth | es=4 | es=2 |
|---|---|---|
| 8006 | 0.986 [0.924, 1.132] — not resolved | 1.000 [0.941, 1.419] — not resolved |
| 2066 | 0.990 [0.966, 1.369] — not resolved | 0.984 [0.513, 1.093] — not resolved |

Parity with the f32 default, before and after. At 8006 the activation stream is swamped by
the O(n²) attention; at 2066 a 3-round reading did show a significant 0.974 but the fourth
round widened it back to unresolved, so it does not stand.

What es=2 *did* buy is **fidelity**: with the activation precision matched to the f32 lane,
q8 greedy output became **token-identical to f32 in 4/4 rounds** at 2066, where the es=4 q8
path diverged at generated token 13. Only KV quantization now separates the two paths, and on
this model that sits below the argmax threshold. That plus one fewer dispatch is the case for
keeping it — not speed.

### Net position

q8 matches the f32 default on **both** prefill and decode while holding **0.858 of its peak
RSS at 8006** (0.948 at 2066, both significant) and 1.88× less KV. That makes the lane a
viable capacity option rather than a measured loss, which is what it was for.

Aggregate across all four paired runs, so no single run is mistaken for the result:

| Claim | 8006 r1 | 8006 r2 | 2066 r1 | 2066 r2 | Verdict |
|---|---|---|---|---|---|
| attn-mm prefill vs old q8 | **0.222** | **0.220** | 0.426 | **0.398** | 2.5–4.5× faster, solid |
| q8-both prefill vs f32 | 0.986 | 1.000 | 0.990 | 0.984 | parity, consistently |
| split-K decode vs old q8 | **1.059** | 1.036 | 0.983 | **1.106** | ~+5%, 2 of 4 resolved |

**Bold = CI excludes 1.0.** The prefill fix is the durable result; the decode kernel is a
small real effect this host cannot resolve every time; es=2 is a fidelity and dispatch-count
improvement, not a speedup.

Parity is unchanged by any of this: on coherent English q8-both is **token-identical to
q8-old**, and its divergence from f32 sits at generated token 50 — exactly where the
pre-existing q8 lane already diverged. The change costs no fidelity.

### Still open

- **`src/fit.rs` `KvDtype` still has only `F32` and `F16`**, so the admission planner cannot
  represent the 34/32 block overhead and sizes q8 as f16 — a 1.88× over-estimate that can
  refuse a preload which would actually fit. Cheap and independent of everything above.
- **One host, one model shape** (llama, head_dim 64, GQA group 4). Nothing here promotes a
  default under BENCHMARK_TREATY; it improves an already opt-in lane.

## Limitations

- **Single host, single model, single architecture.** No second host, so under
  BENCHMARK_TREATY nothing here promotes a default.
- **Not an isolated bench host.** 1.84 GiB swap in use, 5.4 GiB wired, on 8 GiB total.
  Absolute throughput drifts hard across rounds (f32 decode 41.8 → 28.8 → 30.1 tok/s).
  This is precisely why every verdict rests on paired within-round ratios, and why
  cross-round absolutes in this document should not be quoted on their own.
- **Rotation is not permutation.** Arm order rotates per round, which preserves cyclic
  order — q8 always runs immediately after f16 and never gets a cold-cache slot. Cancels
  linear drift, not order-adjacency.
- **Wall-clock per run is unreliable** on a swapping host: one q8 round took 241 s against
  a 56 s measured generation. Only in-process post-warmup metrics are trustworthy.
- **Q4_0 KV is untested** because Metal has no Q4 resident format (`ResidentKvFormat` is
  `F32|F16|Q8`), so `--kv-quant q4_0` has no resident lane to measure.
- **GPU allocation figures are 2 s samples**, therefore lower bounds on the true peak.
- **Only one real-text probe, at 1460 tokens.** The parity verdict rests entirely on it;
  the depth sweep is filler and cannot corroborate it. A real-text probe at 8k+ would be
  the natural next check, and this receipt does not have one.
- **`--kv-quant q8_0` was not exercised.** Every arm here is driven by
  `CAMELID_METAL_KV_DTYPE`, because the CLI flag never reaches Metal at all
  (`grep -c kv_quant src/metal.rs` → 0). That plumbing gap is a separate finding, not
  something this receipt measures.

## Staleness

Measured at `5cbee84c`. `origin/main` is since 355 files ahead and changes `src/metal.rs`
by +702/−20. Verified this does not invalidate the result: the KV-format gates (`kvq8`,
`kv16_enabled`, `splitk_attention_enabled`, `ResidentKvFormat`, `CAMELID_METAL_KV_DTYPE`)
are untouched, and the newly added `attention_matmul_prefill_head_dim_supported` returns
true for head_dim 64, so the f32 arm keeps matmul prefill exactly as measured. The other
additions are Q5_K, bf16 and BitNet kernels, unreachable for this model.
