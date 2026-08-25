#!/bin/zsh
# metal-kv-dtype-ab.sh — same-host A/B of the Metal resident KV primary format.
#
# Governed by ../BENCHMARK_TREATY.md. Produces the JSONL that
# metal-kv-dtype-stats.py validates and summarizes.
#
#   ./metal-kv-dtype-ab.sh <camelid-binary> <model.gguf> <probe> <max_tokens> <rounds> <out.jsonl> [arm-set]
#
# <arm-set> is `full` (the default five-arm KV/decode campaign) or `prefill`
# (a targeted q8-vs-old-prefill campaign). Full campaigns require a multiple
# of 10 rounds to complete the five-arm Williams/reverse design; targeted
# two-arm campaigns require an even count of at least 10.
#
# <probe> is `short` (a 6-token prompt), `long` (~8k tokens), `tokN` for a prompt of about
# N tokens (tok512, tok2048, tok8192), or `realtext` (see below). The tokN form is what
# produces the context-depth sweep: decode attention is KV-bandwidth-bound, so the q8
# advantage is a FUNCTION OF DEPTH, and a single context length cannot show that.
#
# WHY `realtext` EXISTS. The generated prompts are random filler, which is correct for
# throughput but useless for the parity gate. Filler gives a nearly flat next-token
# distribution, so a lossy KV cache flips the argmax on numerical noise, and conversely
# a long filler run can collapse into an attractor where both arms agree for free.
# `realtext` uses this repo's own BENCHMARK_TREATY.md as a coherent in-tree prompt.
#
# WHY CROSS-PROCESS. `resident_kv_format_override()` caches CAMELID_METAL_KV_DTYPE in a
# OnceLock (src/metal.rs), so one process can only ever measure one arm.
#
# WHY PAIRED-REVERSE. Each odd round uses a row from a deterministic Williams-style
# schedule and the following even round uses its exact reverse. Thus every pair of arms
# appears in both relative orders in every two-round block, cancelling monotonic within-
# round drift. Across ten full-campaign rounds each arm also occupies each position twice
# and ordered first-order carryovers are balanced. A plain rotating round-robin does not.
#
# FULL ARMS
#   f32          zero-config F32 primary; split-K decode enabled.
#   f16          opt-in half KV; split-K requested.
#   q8           opt-in Q8_0 KV; split-K decode and Q8 attention-matmul prefill enabled.
#   f32-nosplitk F32 with split-K decode forced off.
#   q8-nosplitk  Q8_0 with split-K decode forced off.
#
# PREFILL ARMS
#   q8           Q8_0 with Q8 attention-matmul prefill enabled.
#   q8-noattnmm  Q8_0 with CAMELID_METAL_Q8_ATTN_MM=0, reproducing the old prefill path.

set -euo pipefail

usage() {
  echo "usage: $0 <camelid-binary> <model.gguf> <probe> <max_tokens> <rounds> <out.jsonl> [full|prefill]" >&2
}

if (( $# < 6 || $# > 7 )); then
  usage
  exit 2
fi

BIN="$1"
MODEL="$2"
PROBE="$3"
MAXTOK="$4"
ROUNDS="$5"
OUT="$6"
ARM_SET="${7:-full}"

if [[ ! "$MAXTOK" =~ '^[1-9][0-9]*$' ]] || (( MAXTOK < 2 )); then
  echo "max_tokens must be an integer >= 2 so decode metrics are positive: $MAXTOK" >&2
  exit 2
fi
if [[ ! "$ROUNDS" =~ '^[1-9][0-9]*$' ]]; then
  echo "rounds must be a positive integer: $ROUNDS" >&2
  exit 2
fi
if [[ "$ARM_SET" != "full" && "$ARM_SET" != "prefill" ]]; then
  echo "arm-set must be 'full' or 'prefill': $ARM_SET" >&2
  exit 2
fi
if [[ "$ARM_SET" == "full" ]] && (( ROUNDS < 10 || ROUNDS % 10 != 0 )); then
  echo "full campaigns require a multiple of 10 rounds for complete Williams/reverse balance: $ROUNDS" >&2
  exit 2
fi
if [[ "$ARM_SET" == "prefill" ]] && (( ROUNDS < 10 || ROUNDS % 2 != 0 )); then
  echo "prefill campaigns require an even number of rounds >= 10: $ROUNDS" >&2
  exit 2
fi

PROMPT="$(dirname "$OUT")/prompt-$PROBE.txt"
SCRIPT_DIR="${0:A:h}"

# Prompts are derived, not committed: the filler probes regenerate byte-for-byte from
# seed 677, and `realtext` reads an in-tree doc.
python3 - "$PROBE" "$PROMPT" "$SCRIPT_DIR" <<'PY'
import random, sys
probe, path, script_dir = sys.argv[1], sys.argv[2], sys.argv[3]
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron "
         "pi rho sigma tau upsilon phi chi psi omega tensor kernel buffer cache decode "
         "prefill quantize residency throughput latency bandwidth attention softmax residual "
         "embedding projection scatter gather stride occupancy simdgroup threadgroup").split()
CHARS_PER_TOKEN = 5.46


def filler(target_chars):
    random.seed(677)
    out, n = [], 0
    while n < target_chars:
        k = random.randint(8, 18)
        s = " ".join(random.choice(WORDS) for _ in range(k)).capitalize() + "."
        out.append(s)
        n += len(s) + 1
    return " ".join(out)


if probe == "short":
    open(path, "w").write("The capital of France is")
elif probe == "realtext":
    import os
    src = os.path.join(script_dir, "..", "BENCHMARK_TREATY.md")
    open(path, "w").write(open(src, encoding="utf-8").read())
elif probe.startswith("tok") and probe[3:].isdigit() and int(probe[3:]) > 0:
    open(path, "w").write(filler(int(int(probe[3:]) * CHARS_PER_TOKEN)))
elif probe == "long":
    open(path, "w").write(filler(int(8000 * CHARS_PER_TOKEN)))
else:
    raise SystemExit(f"unsupported probe: {probe}")
PY

if [[ "$ARM_SET" == "full" ]]; then
  ARMS=("f32:f32:1:1" "f16:f16:1:1" "q8:q8:1:1" "f32-nosplitk:f32:0:1" "q8-nosplitk:q8:0:1")
  # One-based form of [0, 1, 4, 2, 3], the first row of the odd-N Williams design.
  BASE_ORDER=(1 2 5 3 4)
else
  ARMS=("q8:q8:1:1" "q8-noattnmm:q8:1:0")
  BASE_ORDER=(1 2)
fi
N=${#ARMS[@]}

# A schema-bearing campaign header makes a missing fifth arm distinguishable from a
# deliberate pre-split-K four-arm legacy input. The stats tool refuses ambiguous
# headerless four-arm input unless its explicit --legacy-four-arm switch is supplied.
python3 - "$ARM_SET" "$ROUNDS" "${ARMS[@]}" > "$OUT" <<'PY'
import json, sys
arm_set, rounds, *specs = sys.argv[1:]
arms = []
configs = {}
for spec in specs:
    label, kv, splitk, attn_mm = spec.split(":")
    arms.append(label)
    configs[label] = {"kv_dtype": kv, "splitk": splitk, "q8_attn_mm": attn_mm}
print(json.dumps({
    "type": "campaign",
    "schema": "camelid.metal-kv-dtype-ab/v2",
    "arm_set": arm_set,
    "arms": arms,
    "arm_configs": configs,
    "rounds": int(rounds),
    "order_design": "paired-reverse-williams-v1",
}))
PY

campaign_failed=0
stdout_f=""
stderr_f=""
cleanup() {
  [[ -z "$stdout_f" ]] || rm -f "$stdout_f"
  [[ -z "$stderr_f" ]] || rm -f "$stderr_f"
}
on_signal() {
  signal_status="$1"
  cleanup
  trap - EXIT
  exit "$signal_status"
}
trap cleanup EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

for r in $(seq 1 "$ROUNDS"); do
  pair_index=$(( (r - 1) / 2 ))
  shift=$(( pair_index % N ))
  order=()
  for base_idx in "${BASE_ORDER[@]}"; do
    order+=( $(( (base_idx - 1 + shift) % N + 1 )) )
  done
  if (( r % 2 == 0 )); then
    reversed=()
    for pos in $(seq "$N" -1 1); do
      reversed+=( "${order[$pos]}" )
    done
    order=( "${reversed[@]}" )
  fi

  for idx in "${order[@]}"; do
    spec="${ARMS[$idx]}"
    label="${spec%%:*}"
    rest="${spec#*:}"
    kv="${rest%%:*}"
    rest="${rest#*:}"
    sk="${rest%%:*}"
    attn_mm="${rest##*:}"
    echo "[round $r] arm=$label (KV=$kv SPLITK=$sk Q8_ATTN_MM=$attn_mm)" >&2
    stdout_f=$(mktemp)
    stderr_f=$(mktemp)
    SECONDS=0
    # env -i prevents stray CAMELID_* settings from leaking into an arm. Every
    # experiment variable, including the default-on Q8 prefill gate, is explicit.
    if env -i HOME="$HOME" PATH="$PATH" \
        CAMELID_METAL_KV_DTYPE="$kv" \
        CAMELID_METAL_ATTN_SPLITK="$sk" \
        CAMELID_METAL_Q8_ATTN_MM="$attn_mm" \
      "$BIN" bench-generate "$MODEL" \
        --prompt-file "$PROMPT" \
        --max-tokens "$MAXTOK" \
        --temperature 0 \
        --iterations 1 \
        --warmup \
        --json \
      >"$stdout_f" 2>"$stderr_f"; then
      rc=0
    else
      rc=$?
    fi
    wall=$SECONDS

    # The command contract is exactly one JSON object on stdout. Record every failed
    # attempt for diagnosis, but make the campaign itself fail after all arms finish.
    if ! python3 - "$label" "$r" "$rc" "$stdout_f" "$stderr_f" "$kv" "$sk" "$attn_mm" "$wall" >> "$OUT" <<'PY'
import json, sys
label, rnd, rc, so, se, kv, sk, attn_mm, wall = sys.argv[1:10]
raw = open(so, encoding="utf-8", errors="replace").read()
err = open(se, encoding="utf-8", errors="replace").read()
objects = []
parse_errors = []
for line_number, line in enumerate(raw.splitlines(), 1):
    line = line.strip()
    if not line:
        continue
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        parse_errors.append(f"line {line_number}: {exc.msg}")
        continue
    if isinstance(value, dict):
        objects.append(value)
    else:
        parse_errors.append(f"line {line_number}: JSON value is not an object")

rec = {
    "type": "run",
    "arm": label,
    "round": int(rnd),
    "rc": int(rc),
    "kv_dtype": kv,
    "splitk": sk,
    "q8_attn_mm": attn_mm,
    "wall_seconds": int(wall),
    "stderr_tail": err[-1500:],
}
if int(rc) == 0 and len(objects) == 1 and not parse_errors:
    rec["record"] = objects[0]
else:
    if raw:
        rec["stdout_tail"] = raw[-2000:]
    details = list(parse_errors)
    if len(objects) != 1:
        details.append(f"expected exactly one JSON object, found {len(objects)}")
    if int(rc) != 0:
        details.append(f"child exited {rc}")
    rec["parse_errors"] = details
print(json.dumps(rec))
raise SystemExit(0 if "record" in rec else 1)
PY
    then
      campaign_failed=1
    fi
    cleanup
    stdout_f=""
    stderr_f=""
  done
done

# Validate the complete campaign before calling the JSONL usable. This catches a valid
# JSON object with missing IDs/metrics, duplicate/missing arm-round cells, or mismatched
# token counts. Statistics and parity are never emitted for incomplete data.
if ! python3 "$SCRIPT_DIR/metal-kv-dtype-stats.py" --validate-only "$OUT"; then
  campaign_failed=1
fi

if (( campaign_failed != 0 )); then
  echo "campaign failed validation; diagnostic JSONL retained at $OUT" >&2
  exit 1
fi

echo "wrote validated campaign $OUT" >&2
