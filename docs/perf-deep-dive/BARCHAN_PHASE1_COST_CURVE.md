# BARCHAN — Phase 1: the verify-cost curve

**Verdict: GATE 1 = KILL.** Per-round verify cost at k=15 is **9.55×** the k=1 cost, against a
KILL threshold of 3×. The width thesis is dead on this host.

**Consequence, per conductor Gate 1:** publish the curve, **drop Phases 3 and 4**, record the
negative. No latch re-tune, no branch sweep, no serve wiring.

---

## 1. What was measured

`bench-speculative`, Metal resident tree lane, `Llama-3.2-3B-Instruct-Q8_0`, column
`repetitive_extraction` (the column with the most verify rounds to average over).

`--draft-tokens k` sets the per-round tree budget to `(min(k+1,16), k)` = `(max_nodes, max_depth)`
(`src/main.rs`, `full_tree`). So the sweep is a **tree-width sweep**:

| `--draft-tokens` | 1 | 3 | 5 | 7 | 11 | 15 |
|---|---|---|---|---|---|---|
| max_nodes | 2 | 4 | 6 | 8 | 12 | **16** |

k=5 is the shipped default. k=15 saturates `TREE_MAX_NODES`.

- Binary `camelid v0.3.1-155-g38e0aaf`, **clean worktree**,
  sha256 `2efbbbc123aa627335693a87efb9800de4f6aa74795a2100dd0e07eb1c0d05cd`.
- Env: conductor §2.1 block, `CAMELID_SPEC_TREE_GATE=0` (ungated), `CAMELID_SPEC_CPU_VERIFY=0`.
- Configs **interleaved within each repetition**, `--warmup` on every run.
- Per-round verify cost uses the **A4-corrected** `verify_ms` (commit `38e0aaf`). The pre-A4 field
  conflated plain-step time and would have flattened this curve toward a wrong answer.

**Sample size caveat.** 2 clean repetitions, not the N≥5 the conductor requires. A concurrent
session began building and running GPU benchmarks on the same host at run 18 of 30; runs from that
point were discarded rather than reported (`records-INADMISSIBLE-*`, and reps 3+ of the clean
sweep). Reps 1–2 completed before contention began and are retained. This is stated as a limitation,
not waved away — but see §4 on why it does not put the verdict in question.

---

## 2. The curve

Every cell: `lossless = true`, `first_divergent = -1`, `cpu_verify_rounds = 0`.

| k | max_nodes | actual mean n | verify ms/round (median) | ms per node | acc/round | s_sync |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2.00 | 59.3 | 29.7 | 1.81 | **1.066** |
| 3 | 4 | 4.00 | 117.9 | 29.5 | 2.94 | 0.928 |
| 5 | 6 | 5.92 | 193.0 | 32.6 | 3.62 | 0.741 |
| 7 | 8 | 7.77 | 290.6 | 37.4 | 4.27 | 0.611 |
| 11 | 12 | 11.52 | 434.6 | 37.7 | 4.48 | 0.446 |
| 15 | 16 | 14.95 | 566.5 | 37.9 | 4.95 | 0.393 |

The drafter genuinely fills the requested width — mean n tracks max_nodes to within 7% at every
point — so this is a real width sweep, not a saturated one.

**Internal control:** `normal_step_ms / normal_steps` held at **37.4 – 40.0 ms** across all six
configurations, and `plain_tokens_per_second` at **26.8 – 27.4**. Conditions were homogeneous
across the sweep; the verify curve moved while its own control did not.

### Least-squares fit over actual mean node count

```
verify_ms_per_round = 40.18 × nodes − 32.18
```

- **Marginal cost per additional verified node: 40.18 ms**
- **Fixed per-round cost (intercept): ≈ 0** (slightly negative)

---

## 3. What this refutes

**The §0.3 thesis is refuted.** It predicted that on a memory-bound M4, a k-row verify reads the
same weights as a 1-row decode, so verifying up to ~15 tokens should cost about what verifying 1
costs. Measured: cost is **linear in k**, at **40.18 ms per row against a 38.0 ms plain decode
step**. Each additional verified row costs *slightly more than an entire independent decode*. The
batched verify delivers **zero weight-read amortization**. A k-row tree verify is strictly worse
than k sequential decodes.

**The PIVOT hypothesis is also refuted.** The conductor's 1.5–3× branch — and my own Phase 0
prime suspect, per-round resident-engine teardown/reseed, plus `compact_tree_kv_path` — predicted a
*fixed* per-round cost dominating. The intercept is ≈ 0. There is essentially **no** fixed
per-round cost. Whatever is expensive is paid **per row**, not per round. That rules out the
mechanisms the PIVOT branch was designed to catch, and it is why this is a KILL rather than a
redirect at KV compaction.

**This also explains the 3060's 1.28×** without needing a drafter explanation: if per-row verify
cost is near a full decode there too, acceptance can never outrun it.

**Not yet attributed: GPU vs host.** The conductor's KILL wording names "the residual (not
gpu_busy) as the driver". This sweep establishes the *shape* (pure per-row, zero intercept) but
does not split GPU-busy from host time — the `gpu_busy_us` instrumentation was not built, because
the curve answered the gate without it. Both a non-amortizing GEMM and per-row host work are
consistent with linearity. **The attribution is open; the verdict is not.** No mechanism that costs
a full decode per row can make width pay.

---

## 4. Why the verdict is not sample-size-limited

The gate is 3×. The measurement is **9.55×** — a 3.2× margin over the threshold. Supporting this:

- Between-rep spread is tight: k=15 gave 556–577 ms (±1.9%), k=1 gave 59–60 ms.
- s_sync reproduced per rep: k=1 → 1.085 / 1.047; k=15 → 0.385 / 0.401.
- The curve is monotone across six configurations with a clean linear fit.
- The internal control (plain-step ms) was flat throughout.

Completing to N=5 would tighten the confidence interval on 40.18 ms/node. It cannot move 9.55
below 3. The remaining reps are owed for the receipt, not for the decision.

---

## 5. The one positive result

**k=1 — a 2-node tree — is the only configuration that beats plain decode**, at
**s_sync ≈ 1.066** (1.085 / 1.047), i.e. a ~5–8% win. The arithmetic is consistent: 59.3 ms buys
1.81 emitted tokens = **32.8 ms/token against 38.0 ms/token plain**.

So the lane is not worthless — but **width is the wrong knob, and the optimum is the minimum
width**, which is the exact opposite of the campaign's premise. Every widening step past 2 nodes
loses, monotonically, because acceptance grows sublinearly (1.81 → 4.95, a 2.7× gain for 8× the
nodes) while cost grows linearly (9.55×).

Note the shipped default is k=5, which measures **0.741** on this column — i.e. the current default
is a **26% regression** versus plain decode whenever the ungated tree lane actually engages. The
`SpecLatch` exists to skip exactly these rounds, which is why the shipped gated path does not show
this; but it means the latch is doing load-bearing damage control, not fine-tuning.

---

## 6. Recommendation

1. **Close the width lane.** Do not run Phases 3–4. Record the negative.
2. **Do not ship a width change.** k=1 beating k=5 is a real signal, but it is one column on one
   host at N=2, and the gated path already suppresses the loss. Any default change needs its own
   evidence across the full column pack.
3. **If the lane is ever reopened, the question is "why does a verified row cost a full decode?"**
   That is a single, well-posed instrumentation question (`gpu_busy_us` in `verify_batch_tree`),
   and it is the only thing that could revive tree speculation on Apple Silicon. It is a *kernel*
   question, and conductor §1 currently rules the Q8 Metal kernel lane closed — so reopening the
   width lane requires reopening that first.

---

## 7. Artifacts

`target/barchan-p1-sweep/` — `records/` (18 JSON records; **only k{1,3,5,7,11,15}-rep{1,2} are
admissible**), `logs/` (per-run stderr with `[metal-tree-verify]` and `[spec-tree]` traces),
`run-k-sweep.sh`, `sweep.log`, and `records-INADMISSIBLE-dirty-binary/` (13 runs discarded for
being built from a dirty worktree — retained as a process artifact only).
