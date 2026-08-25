#!/usr/bin/env python3
"""Validate and summarize the Metal resident-KV same-host campaign.

No statistic or parity result is emitted until the input is a complete rectangular
campaign: exactly one successful, schema-valid record for every required arm/round.
New harness output carries a campaign header. Headerless four-arm input is ambiguous
with a truncated five-arm run and is accepted only with --legacy-four-arm.
Timing equivalence is established only when the complete bootstrap interval lies
inside the predeclared ratio band [0.95, 1.05]; merely spanning 1.0 is unresolved.
"""

import argparse
import json
import math
import random
import statistics as st
import sys


SCHEMA = "camelid.metal-kv-dtype-ab/v2"
ORDER_DESIGN = "paired-reverse-williams-v1"
EQUIVALENCE_LOWER = 0.95
EQUIVALENCE_UPPER = 1.05
METRICS = [
    ("prefill_ms", "lower"),
    ("ttft_ms", "lower"),
    ("decode_ms", "lower"),
    ("tokens_per_second", "higher"),
    ("peak_memory_bytes", "lower"),
]
REQUIRED_SCALARS = ["load_ms", *(name for name, _ in METRICS)]
IDENTITY_FIELDS = ["runtime", "commit", "model", "quantization"]

ARM_SETS = {
    "full": {
        "arms": ["f32", "f16", "q8", "f32-nosplitk", "q8-nosplitk"],
        "configs": {
            "f32": {"kv_dtype": "f32", "splitk": "1", "q8_attn_mm": "1"},
            "f16": {"kv_dtype": "f16", "splitk": "1", "q8_attn_mm": "1"},
            "q8": {"kv_dtype": "q8", "splitk": "1", "q8_attn_mm": "1"},
            "f32-nosplitk": {"kv_dtype": "f32", "splitk": "0", "q8_attn_mm": "1"},
            "q8-nosplitk": {"kv_dtype": "q8", "splitk": "0", "q8_attn_mm": "1"},
        },
    },
    "prefill": {
        "arms": ["q8", "q8-noattnmm"],
        "configs": {
            "q8": {"kv_dtype": "q8", "splitk": "1", "q8_attn_mm": "1"},
            "q8-noattnmm": {"kv_dtype": "q8", "splitk": "1", "q8_attn_mm": "0"},
        },
    },
}
LEGACY_ARMS = ["f32", "f16", "q8", "f32-nosplitk"]
LEGACY_CONFIGS = {
    arm: {key: value for key, value in ARM_SETS["full"]["configs"][arm].items() if key != "q8_attn_mm"}
    for arm in LEGACY_ARMS
}


class CampaignError(ValueError):
    pass


def is_int(value):
    return type(value) is int


def is_positive_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def load_jsonl(path):
    entries = []
    try:
        source = open(path, encoding="utf-8")
    except OSError as exc:
        raise CampaignError(f"cannot open {path}: {exc}") from exc
    with source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CampaignError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise CampaignError(f"line {line_number}: top-level JSON value must be an object")
            entries.append((line_number, value))
    if not entries:
        raise CampaignError("input has no JSON records")
    return entries


def infer_rounds(records):
    rounds = sorted({row.get("round") for _, row in records if is_int(row.get("round"))})
    if not rounds:
        raise CampaignError("headerless input has no integer round IDs")
    expected = list(range(1, rounds[-1] + 1))
    if rounds != expected:
        raise CampaignError(f"round IDs must be consecutive from 1; found {rounds}")
    if len(rounds) < 4:
        raise CampaignError(f"at least 4 complete rounds are required; found {len(rounds)}")
    return len(rounds)


def select_campaign(entries, legacy_four_arm):
    headers = [(line, row) for line, row in entries if row.get("type") == "campaign"]
    records = [(line, row) for line, row in entries if row.get("type") != "campaign"]
    if len(headers) > 1:
        raise CampaignError(f"expected at most one campaign header; found {len(headers)}")
    if not records:
        raise CampaignError("campaign has no run records")

    if headers:
        line, header = headers[0]
        if legacy_four_arm:
            raise CampaignError("--legacy-four-arm cannot be used with a schema-bearing campaign")
        if header.get("schema") != SCHEMA:
            raise CampaignError(f"line {line}: unsupported campaign schema {header.get('schema')!r}")
        arm_set = header.get("arm_set")
        if arm_set not in ARM_SETS:
            raise CampaignError(f"line {line}: unknown arm_set {arm_set!r}")
        spec = ARM_SETS[arm_set]
        if header.get("arms") != spec["arms"]:
            raise CampaignError(
                f"line {line}: arms must be exactly {spec['arms']}; found {header.get('arms')!r}"
            )
        if header.get("arm_configs") != spec["configs"]:
            raise CampaignError(f"line {line}: arm_configs do not match the {arm_set} contract")
        rounds = header.get("rounds")
        if arm_set == "full":
            valid_rounds = is_int(rounds) and rounds >= 10 and rounds % 10 == 0
            rounds_contract = "a multiple of 10 and at least 10"
        else:
            valid_rounds = is_int(rounds) and rounds >= 10 and rounds % 2 == 0
            rounds_contract = "an even integer and at least 10"
        if not valid_rounds:
            raise CampaignError(
                f"line {line}: {arm_set} rounds must be {rounds_contract}; found {rounds!r}"
            )
        if header.get("order_design") != ORDER_DESIGN:
            raise CampaignError(f"line {line}: order_design must be {ORDER_DESIGN!r}")
        return {
            "kind": arm_set,
            "arms": spec["arms"],
            "configs": spec["configs"],
            "rounds": rounds,
            "records": records,
            "schema_bearing": True,
        }

    seen_arms = {row.get("arm") for _, row in records if isinstance(row.get("arm"), str)}
    legacy_set = set(LEGACY_ARMS)
    full_set = set(ARM_SETS["full"]["arms"])
    if legacy_four_arm:
        if seen_arms != legacy_set:
            raise CampaignError(
                f"--legacy-four-arm requires exactly {LEGACY_ARMS}; found {sorted(seen_arms)}"
            )
        return {
            "kind": "legacy-four-arm",
            "arms": LEGACY_ARMS,
            "configs": LEGACY_CONFIGS,
            "rounds": infer_rounds(records),
            "records": records,
            "schema_bearing": False,
        }
    if seen_arms == legacy_set:
        raise CampaignError(
            "headerless four-arm input is ambiguous with a failed/truncated current five-arm "
            "campaign; use --legacy-four-arm only for a known pre-Q8-split-K receipt"
        )
    if seen_arms != full_set:
        raise CampaignError(
            f"headerless current input requires exactly {ARM_SETS['full']['arms']}; "
            f"found {sorted(seen_arms)}"
        )
    return {
        "kind": "full",
        "arms": ARM_SETS["full"]["arms"],
        "configs": ARM_SETS["full"]["configs"],
        "rounds": infer_rounds(records),
        "records": records,
        "schema_bearing": False,
    }


def validate_campaign(entries, legacy_four_arm=False):
    campaign = select_campaign(entries, legacy_four_arm)
    arms = campaign["arms"]
    rounds = campaign["rounds"]
    expected_cells = {(arm, rnd) for arm in arms for rnd in range(1, rounds + 1)}
    cells = {}
    errors = []

    for line, row in campaign["records"]:
        if campaign["schema_bearing"] and row.get("type") != "run":
            errors.append(f"line {line}: campaign record type must be 'run'")
        arm = row.get("arm")
        rnd = row.get("round")
        if arm not in arms:
            errors.append(f"line {line}: unknown or missing arm {arm!r}")
            continue
        if not is_int(rnd) or not 1 <= rnd <= rounds:
            errors.append(f"line {line}: round must be an integer in 1..{rounds}; found {rnd!r}")
            continue
        cells.setdefault((arm, rnd), []).append((line, row))

    for arm, rnd in sorted(expected_cells, key=lambda cell: (cell[1], arms.index(cell[0]))):
        count = len(cells.get((arm, rnd), []))
        if count != 1:
            errors.append(
                f"arm={arm} round={rnd} requires exactly one successful record; found {count} rows"
            )

    validated = []
    for (arm, rnd), rows in cells.items():
        if (arm, rnd) not in expected_cells or len(rows) != 1:
            continue
        line, row = rows[0]
        if not is_int(row.get("rc")) or row.get("rc") != 0:
            errors.append(f"line {line}: arm={arm} round={rnd} child rc must be integer 0")
            continue
        record = row.get("record")
        if not isinstance(record, dict):
            errors.append(f"line {line}: arm={arm} round={rnd} has no parsed benchmark record")
            continue

        expected_config = campaign["configs"][arm]
        if not campaign["schema_bearing"]:
            expected_config = {
                key: value for key, value in expected_config.items() if key != "q8_attn_mm"
            }
        for key, expected in expected_config.items():
            if row.get(key) != expected:
                errors.append(
                    f"line {line}: arm={arm} {key} must be {expected!r}; found {row.get(key)!r}"
                )

        for key in IDENTITY_FIELDS:
            value = record.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"line {line}: record.{key} must be a nonempty string")
        if not is_int(record.get("iteration")) or record.get("iteration") != 0:
            errors.append(f"line {line}: record.iteration must be integer 0")

        for key in ("prompt_tokens", "generated_tokens"):
            if not is_int(record.get(key)) or record.get(key) <= 0:
                errors.append(f"line {line}: record.{key} must be a positive integer")
        for key in REQUIRED_SCALARS:
            if not is_positive_number(record.get(key)):
                errors.append(f"line {line}: record.{key} must be a finite positive number")

        token_ids = record.get("output_token_ids")
        if not isinstance(token_ids, list):
            errors.append(f"line {line}: record.output_token_ids must be a list")
        else:
            if any(not is_int(token) or not 0 <= token <= 0xFFFFFFFF for token in token_ids):
                errors.append(f"line {line}: record.output_token_ids must contain u32 integers")
            generated = record.get("generated_tokens")
            if is_int(generated) and len(token_ids) != generated:
                errors.append(
                    f"line {line}: generated_tokens={generated} but output_token_ids has {len(token_ids)} entries"
                )
        validated.append(row)

    if len(validated) == len(expected_cells):
        identity_values = {
            key: {row["record"].get(key) for row in validated}
            for key in IDENTITY_FIELDS
        }
        for key, values in identity_values.items():
            if len(values) != 1:
                found = sorted(repr(value) for value in values)
                errors.append(f"record.{key} must match across the campaign; found {found!r}")
        for key in ("prompt_tokens", "generated_tokens"):
            values = {row["record"].get(key) for row in validated}
            if len(values) != 1:
                found = sorted(repr(value) for value in values)
                errors.append(f"record.{key} must match across every arm/round; found {found!r}")

    if errors:
        shown = errors[:20]
        if len(errors) > len(shown):
            shown.append(f"... and {len(errors) - len(shown)} more validation errors")
        raise CampaignError("\n  - ".join(["campaign is incomplete or invalid", *shown]))

    by = {arm: {} for arm in arms}
    for row in validated:
        by[row["arm"]][row["round"]] = row
    campaign["by"] = by
    return campaign


def first_divergence(a, b):
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return -1 if len(a) == len(b) else min(len(a), len(b))


def boot_ci(pairs, n=20000):
    rng = random.Random(677)
    samples = []
    count = len(pairs)
    for _ in range(n):
        samples.append(st.median(pairs[rng.randrange(count)] for _ in range(count)))
    samples.sort()
    return samples[int(0.025 * n)], samples[int(0.975 * n)]


def metric(row, key):
    return row["record"][key]


def summarize(campaign, label):
    arms = campaign["arms"]
    rounds = campaign["rounds"]
    by = campaign["by"]
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"validated runs: {len(arms) * rounds}  arm-set: {campaign['kind']}  rounds: {rounds}")
    print(f"prompt_tokens: {metric(by[arms[0]][1], 'prompt_tokens')}")
    print(f"generated_tokens: {metric(by[arms[0]][1], 'generated_tokens')}")

    print(f"\n{'metric':<28}" + "".join(f"{arm:>16}" for arm in arms))
    for name, _direction in METRICS:
        line = f"{name:<28}"
        for arm in arms:
            values = [metric(by[arm][rnd], name) for rnd in range(1, rounds + 1)]
            line += f"{st.median(values):>16,.2f}"
        print(line)

    def paired(baseline, candidates, title):
        print(f"\n{title}")
        for name, direction in METRICS:
            for candidate in candidates:
                ratios = [
                    metric(by[candidate][rnd], name) / metric(by[baseline][rnd], name)
                    for rnd in range(1, rounds + 1)
                ]
                lo, hi = boot_ci(ratios)
                ratio = st.median(ratios)
                if lo >= EQUIVALENCE_LOWER and hi <= EQUIVALENCE_UPPER:
                    resolution = "equivalent within +/-5%"
                elif lo > 1.0 or hi < 1.0:
                    resolution = "CI EXCLUDES 1"
                else:
                    resolution = "not resolved; equivalence not established"
                is_better = ratio < 1.0 if direction == "lower" else ratio > 1.0
                verdict = "better" if is_better else "worse"
                if resolution != "CI EXCLUDES 1":
                    verdict = ""
                tag = f"{candidate}/{baseline}"
                print(
                    f"  {name:<24} {tag:>26} = {ratio:6.4f}  "
                    f"[{lo:6.4f}, {hi:6.4f}]  {resolution} {verdict}"
                )

    if campaign["kind"] == "full":
        paired(
            "f32",
            ["f16", "q8", "f32-nosplitk", "q8-nosplitk"],
            "paired per-round ratio vs f32 DEFAULT (median, bootstrap 95% CI)",
        )
        paired(
            "f32-nosplitk",
            ["q8-nosplitk", "f16"],
            "paired per-round ratio vs f32-nosplitk (explicit no-split-K controls)",
        )
        paired(
            "q8-nosplitk",
            ["q8"],
            "paired per-round Q8 split-K effect (q8/q8-nosplitk)",
        )
        parity_baseline = "f32"
    elif campaign["kind"] == "prefill":
        paired(
            "q8-noattnmm",
            ["q8"],
            "paired per-round Q8 attention-matmul prefill effect (q8/q8-noattnmm)",
        )
        parity_baseline = "q8-noattnmm"
    else:
        paired(
            "f32",
            ["f16", "q8", "f32-nosplitk"],
            "paired per-round ratio vs f32 DEFAULT (legacy four-arm receipt)",
        )
        paired(
            "f32-nosplitk",
            ["q8", "f16"],
            "paired per-round ratio vs f32-nosplitk (legacy Q8 predates Q8 split-K)",
        )
        parity_baseline = "f32"

    print(
        f"\ngreedy output parity vs {parity_baseline} "
        "(first divergent generated token index; -1 = identical)"
    )
    baseline_ids = {
        rnd: metric(by[parity_baseline][rnd], "output_token_ids")
        for rnd in range(1, rounds + 1)
    }
    for arm in arms:
        if arm == parity_baseline:
            continue
        divergences = [
            first_divergence(baseline_ids[rnd], metric(by[arm][rnd], "output_token_ids"))
            for rnd in range(1, rounds + 1)
        ]
        identical = sum(index == -1 for index in divergences)
        print(f"  {arm:>14} vs {parity_baseline}: {divergences}   identical in {identical}/{rounds} rounds")

    print("\nself-determinism across rounds (first divergence vs that arm's round-1 output)")
    for arm in arms:
        reference = metric(by[arm][1], "output_token_ids")
        divergences = [
            first_divergence(reference, metric(by[arm][rnd], "output_token_ids"))
            for rnd in range(2, rounds + 1)
        ]
        print(f"  {arm:>14}: {divergences}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-four-arm",
        action="store_true",
        help="accept a known pre-Q8-split-K headerless four-arm receipt",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate completeness/schema but emit no statistics or parity",
    )
    parser.add_argument("path", help="raw JSONL from metal-kv-dtype-ab.sh")
    parser.add_argument("label", nargs="?", help="report label")
    args = parser.parse_args(argv)
    if not args.validate_only and not args.label:
        parser.error("label is required unless --validate-only is used")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        entries = load_jsonl(args.path)
        campaign = validate_campaign(entries, legacy_four_arm=args.legacy_four_arm)
    except CampaignError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    if args.validate_only:
        print(
            f"validated {len(campaign['arms']) * campaign['rounds']} runs "
            f"({campaign['kind']}, {campaign['rounds']} rounds)"
        )
        return 0
    summarize(campaign, args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
