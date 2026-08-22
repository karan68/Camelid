#!/bin/zsh
# metal-kv-dtype-ab.sh — same-host A/B of the Metal resident KV primary format.
#
# Governed by ../BENCHMARK_TREATY.md. Produces the JSONL that
# metal-kv-dtype-stats.py turns into a receipt.
#
#   ./metal-kv-dtype-ab.sh <camelid-binary> <model.gguf> <short|long> <max_tokens> <rounds> <out.jsonl>
#
# WHY CROSS-PROCESS.  `resident_kv_format_override()` caches CAMELID_METAL_KV_DTYPE in a
# OnceLock (src/metal.rs), so the KV format is frozen at first read. `camelid
# bench-owner-sweep` — the treaty's preferred harness for sub-5% effects — re-reads the
# runtime plan from env per call, which means it CANNOT sweep KV dtype: one process can
# only ever measure one arm. That forces the cross-invocation A/B the treaty warns is
# drift-prone.
#
# WHY INTERLEAVED.  To recover what pairing would have given us, arms are run round-robin
# with the starting arm rotated per round, and compared WITHIN a round. That cancels the
# slow thermal drift a 5-in-a-row-per-arm layout would bake in.
#
# ARMS
#   f32          zero-config default for a Q8_0 model (weights_use_kquant() == false, so
#                resident_kv_format() returns F32).
#   f16          opt-in half KV.
#   q8           opt-in Q8_0 KV — the lane under test.
#   f32-nosplitk mechanism control: f32 with split-K decode forced off. Enabling f16/q8
#                already forfeits split-K (the gate requires an F32 primary), so this is
#                the apples-to-apples baseline for q8; the plain f32 arm is the
#                real-world one. Without this arm a q8 regression is uninterpretable —
#                you cannot tell a slow KV kernel from a forfeited split-K.

set -u
BIN="$1"; MODEL="$2"; PROBE="$3"; MAXTOK="$4"; ROUNDS="$5"; OUT="$6"

PROMPT="$(dirname "$OUT")/prompt-$PROBE.txt"

# Prompts are GENERATED, not committed: seed 677 reproduces them byte-for-byte.
python3 - "$PROBE" "$PROMPT" <<'PY'
import random, sys
probe, path = sys.argv[1], sys.argv[2]
if probe == "short":
    open(path, "w").write("The capital of France is")
else:
    random.seed(677)
    words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron "
             "pi rho sigma tau upsilon phi chi psi omega tensor kernel buffer cache decode "
             "prefill quantize residency throughput latency bandwidth attention softmax residual "
             "embedding projection scatter gather stride occupancy simdgroup threadgroup").split()
    sents = []
    for _ in range(480):
        n = random.randint(8, 18)
        sents.append(" ".join(random.choice(words) for _ in range(n)).capitalize() + ".")
    open(path, "w").write(" ".join(sents))
PY

ARMS=("f32:f32:1" "f16:f16:1" "q8:q8:1" "f32-nosplitk:f32:0")
N=${#ARMS[@]}
: > "$OUT"

for r in $(seq 1 "$ROUNDS"); do
  for i in $(seq 0 $((N - 1))); do
    idx=$(( (i + r - 1) % N + 1 ))
    spec="${ARMS[$idx]}"
    label="${spec%%:*}"; rest="${spec#*:}"
    kv="${rest%%:*}"; sk="${rest##*:}"
    echo "[round $r] arm=$label (KV=$kv SPLITK=$sk)" >&2
    stdout_f=$(mktemp); stderr_f=$(mktemp)
    SECONDS=0
    # env -i: no stray CAMELID_* leaks into an arm, matching
    # metal-kquant-m4-postfix-three-way-20260730.json.
    env -i HOME="$HOME" PATH="$PATH" \
        CAMELID_METAL_KV_DTYPE="$kv" CAMELID_METAL_ATTN_SPLITK="$sk" \
      "$BIN" bench-generate "$MODEL" \
        --prompt-file "$PROMPT" \
        --max-tokens "$MAXTOK" \
        --temperature 0 \
        --iterations 1 \
        --warmup \
        --json \
      >"$stdout_f" 2>"$stderr_f"
    rc=$?
    wall=$SECONDS
    python3 - "$label" "$r" "$rc" "$stdout_f" "$stderr_f" "$kv" "$sk" "$wall" >> "$OUT" <<'PY'
import json, sys
label, rnd, rc, so, se, kv, sk, wall = sys.argv[1:9]
raw = open(so).read()
err = open(se).read()
rec = {"arm": label, "round": int(rnd), "rc": int(rc),
       "kv_dtype": kv, "splitk": sk, "wall_seconds": int(wall)}
got = None
for line in raw.splitlines():
    line = line.strip()
    if line.startswith("{"):
        try:
            got = json.loads(line)
        except Exception:
            pass
if got:
    rec["record"] = got
else:
    rec["stdout_tail"] = raw[-2000:]
rec["stderr_tail"] = err[-1500:]
print(json.dumps(rec))
PY
    rm -f "$stdout_f" "$stderr_f"
  done
done
echo "wrote $OUT" >&2
