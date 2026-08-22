#!/usr/bin/env python3
"""Summarize the Metal KV-dtype A/B: per-arm medians plus paired per-round ratios
with a bootstrap 95% CI, following docs/perf-deep-dive/BENCHMARK_TREATY.md.

Pairing matters here: the arms cannot be swept inside one process (the KV format
is frozen in a OnceLock), so the comparison is cross-invocation and exposed to
thermal drift. Comparing arms within the same round cancels the slow drift; a
result only counts if the bootstrap CI excludes 1.0.
"""
import json
import statistics as st
import sys
import random

random.seed(677)

METRICS = [
    ("prefill_ms", "lower"),
    ("ttft_ms", "lower"),
    ("decode_ms", "lower"),
    ("tokens_per_second", "higher"),
    ("peak_memory_bytes", "lower"),
]
ARMS = ["f32", "f16", "q8", "f32-nosplitk"]


def first_divergence(a, b):
    """BENCHMARK_TREATY parity gate: -1 means token-identical."""
    if a is None or b is None:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def get(rec, key):
    """Scalar metric accessor. bench-generate emits one record per iteration, but a
    multi-iteration receipt may carry arrays; unwrap a single-element array to a scalar."""
    r = rec.get("record")
    if not r:
        return None
    v = r.get(key)
    if isinstance(v, list):
        return v[0] if v else None
    return v


def get_seq(rec, key):
    """List-valued accessor (output_token_ids) — must NOT be unwrapped by get()."""
    r = rec.get("record")
    if not r:
        return None
    v = r.get(key)
    return v if isinstance(v, list) else None


def boot_ci(pairs, n=20000):
    """Bootstrap CI of the median per-round ratio."""
    if not pairs:
        return (float("nan"), float("nan"))
    out = []
    k = len(pairs)
    for _ in range(n):
        s = [pairs[random.randrange(k)] for _ in range(k)]
        out.append(st.median(s))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def main(path, label):
    rows = load(path)
    bad = [r for r in rows if r.get("rc") != 0 or "record" not in r]
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"runs: {len(rows)}  failed/unparsed: {len(bad)}")
    for b in bad[:4]:
        tail = (b.get("stderr_tail") or b.get("stdout_tail") or "")[-400:]
        print(f"  !! arm={b['arm']} round={b['round']} rc={b.get('rc')}: {tail.strip()[:400]}")

    ok = [r for r in rows if r.get("rc") == 0 and "record" in r]
    if not ok:
        print("no usable runs")
        return

    by = {a: [r for r in ok if r["arm"] == a] for a in ARMS}

    # Sanity: prompt/generated token counts must match across arms.
    for key in ("prompt_tokens", "generated_tokens"):
        vals = {a: sorted({get(r, key) for r in by[a]}) for a in ARMS if by[a]}
        print(f"{key}: {vals}")

    print(f"\n{'metric':<28}" + "".join(f"{a:>16}" for a in ARMS))
    med = {}
    for m, _dir in METRICS:
        med[m] = {}
        line = f"{m:<28}"
        for a in ARMS:
            vs = [get(r, m) for r in by[a] if get(r, m) is not None]
            if vs:
                med[m][a] = st.median(vs)
                line += f"{st.median(vs):>16,.2f}"
            else:
                line += f"{'-':>16}"
        print(line)

    # Paired per-round ratios vs the f32 baseline (the zero-config default for a
    # Q8_0 model: weights_use_kquant() is false, so resident_kv_format() -> F32).
    def paired(baseline, arms, title):
        print(f"\n{title}")
        for m, direction in METRICS:
            for a in arms:
                pairs = []
                for rnd in sorted({r["round"] for r in ok}):
                    base = [get(r, m) for r in by[baseline] if r["round"] == rnd]
                    cur = [get(r, m) for r in by[a] if r["round"] == rnd]
                    if base and cur and base[0] and cur[0]:
                        pairs.append(cur[0] / base[0])
                if not pairs:
                    continue
                lo, hi = boot_ci(pairs)
                r = st.median(pairs)
                sig = "SIGNIFICANT" if (lo > 1.0 or hi < 1.0) else "not resolved"
                better = (r < 1.0) if direction == "lower" else (r > 1.0)
                arrow = "better" if better else "worse"
                if sig == "not resolved":
                    arrow = ""
                tag = f"{a}/{baseline}"
                print(f"  {m:<24} {tag:>22} = {r:6.4f}  [{lo:6.4f}, {hi:6.4f}]  {sig:<13} {arrow}")

    # Real-world comparison: against the zero-config default (f32, split-K on).
    paired("f32", ["f16", "q8", "f32-nosplitk"],
           "paired per-round ratio vs f32 DEFAULT (median, bootstrap 95% CI); CI must exclude 1.0")
    # Mechanism comparison: q8 vs f32 on the same no-split-K footing. This isolates
    # the KV-bandwidth effect from the split-K forfeit that enabling q8 currently causes.
    paired("f32-nosplitk", ["q8", "f16"],
           "paired per-round ratio vs f32-nosplitk (APPLES-TO-APPLES: isolates KV bandwidth)")

    # Quality: greedy output must be token-identical to be a lossless drop-in.
    # Q8 KV is lossy by construction, so this measures HOW lossy, per round.
    print("\ngreedy output parity vs f32 (first divergent generated token index; -1 = identical)")
    for a in ("f16", "q8", "f32-nosplitk"):
        idxs = []
        for rnd in sorted({r["round"] for r in ok}):
            base = [get_seq(r, "output_token_ids") for r in by["f32"] if r["round"] == rnd]
            cur = [get_seq(r, "output_token_ids") for r in by[a] if r["round"] == rnd]
            if base and cur:
                idxs.append(first_divergence(base[0], cur[0]))
        if idxs:
            n_ident = sum(1 for i in idxs if i == -1)
            print(f"  {a:>4} vs f32: {idxs}   identical in {n_ident}/{len(idxs)} rounds")

    # Self-consistency: is each arm even deterministic across rounds at temp 0?
    print("\nself-determinism across rounds (first divergence vs that arm's round-1 output)")
    for a in ARMS:
        rounds = sorted({r["round"] for r in by[a]})
        if not rounds:
            continue
        ref = [get_seq(r, "output_token_ids") for r in by[a] if r["round"] == rounds[0]]
        if not ref:
            continue
        idxs = []
        for rnd in rounds[1:]:
            cur = [get_seq(r, "output_token_ids") for r in by[a] if r["round"] == rnd]
            if cur:
                idxs.append(first_divergence(ref[0], cur[0]))
        print(f"  {a:>4}: {idxs}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
