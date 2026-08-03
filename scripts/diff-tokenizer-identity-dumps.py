#!/usr/bin/env python3
"""Byte-identical token-id gate: baseline dump vs after dump, per family.

Exit 0 only if every (case, add_special, parse_special) triple yields exactly
the same id sequence in both dumps, for every family present in both dirs.
"""
import json
import os
import sys

BASE = sys.argv[1]
AFTER = sys.argv[2]

families = sorted(f for f in os.listdir(BASE) if f.endswith(".json"))
missing = []
failures = 0
grand_ids = 0
grand_cases = 0

print(f"{'family':<14} {'vocab':>8} {'model':<10} {'cases':>6} {'ids':>9}  verdict")
print("-" * 70)

for fname in families:
    bp = os.path.join(BASE, fname)
    ap = os.path.join(AFTER, fname)
    if not os.path.exists(ap):
        missing.append(fname)
        continue
    b = json.load(open(bp))
    a = json.load(open(ap))

    problems = []
    for key in ("vocab", "tokenizer_model", "add_space_prefix", "cases", "total_ids"):
        if b.get(key) != a.get(key):
            problems.append(f"{key}: {b.get(key)!r} -> {a.get(key)!r}")

    br, ar = b["results"], a["results"]
    if len(br) != len(ar):
        problems.append(f"result count {len(br)} -> {len(ar)}")
    else:
        for i, (rb, ra) in enumerate(zip(br, ar)):
            if (rb["name"], rb["add_special"], rb["parse_special"]) != (
                ra["name"], ra["add_special"], ra["parse_special"]
            ):
                problems.append(f"case order diverged at {i}")
                break
            if rb["ids"] != ra["ids"]:
                problems.append(
                    f"IDS DIVERGED case={rb['name']!r} "
                    f"add_special={rb['add_special']} parse_special={rb['parse_special']}\n"
                    f"    before ({len(rb['ids'])}): {rb['ids'][:24]}...\n"
                    f"    after  ({len(ra['ids'])}): {ra['ids'][:24]}..."
                )
                if len(problems) > 3:
                    break

    verdict = "IDENTICAL" if not problems else "DIVERGED"
    if problems:
        failures += 1
    grand_ids += b.get("total_ids", 0)
    grand_cases += b.get("cases", 0)
    print(
        f"{fname[:-5]:<14} {b.get('vocab', 0):>8} {b.get('tokenizer_model', '?'):<10} "
        f"{b.get('cases', 0):>6} {b.get('total_ids', 0):>9}  {verdict}"
    )
    for p in problems:
        print(f"    {p}")

print("-" * 70)
print(f"{len(families) - len(missing)} families compared, "
      f"{grand_cases} encode results each, {grand_ids} token ids total")
if missing:
    print(f"MISSING from after/: {', '.join(missing)}")
if failures or missing:
    print("GATE: FAIL")
    sys.exit(1)
print("GATE: PASS — token ids byte-identical in both parse_special modes")
