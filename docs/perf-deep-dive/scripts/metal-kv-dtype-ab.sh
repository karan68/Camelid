#!/bin/zsh
# metal-kv-dtype-ab.sh — same-host A/B of the Metal resident KV primary format.
#
# Governed by ../BENCHMARK_TREATY.md. Produces the JSONL that
# metal-kv-dtype-stats.py turns into a receipt.
#
#   ./metal-kv-dtype-ab.sh <camelid-binary> <model.gguf> <probe> <max_tokens> <rounds> <out.jsonl>
#
# <probe> is `short` (a 6-token prompt), `long` (~8k tokens), `tokN` for a prompt of about
# N tokens (tok512, tok2048, tok8192), or `realtext` (see below). The tokN form is what
# produces the context-depth sweep: decode attention is KV-bandwidth-bound, so the q8
# advantage is a FUNCTION OF DEPTH, and a single context length cannot show that.
#
# WHY `realtext` EXISTS.  The generated prompts are random filler, which is correct for
# throughput -- memory bandwidth does not care what the tokens mean -- but useless for the
# PARITY gate. Filler gives a nearly flat next-token distribution, so a lossy KV cache
# flips the argmax on numerical noise, and conversely a long filler run collapses into a
# repetition attractor where both arms agree for free. Neither outcome says anything about
# quality. `realtext` uses this repo's own BENCHMARK_TREATY.md as the prompt: coherent
# English, in-tree, identical for anyone who reruns it.
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
# Measured on Llama-3.2-1B-Instruct-Q8_0: this filler tokenizes at ~5.46 chars/token.
# Only used to SIZE the prompt; the receipt records the exact prompt_tokens the engine
# reported, never this estimate.
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
elif probe.startswith("tok"):
    open(path, "w").write(filler(int(int(probe[3:]) * CHARS_PER_TOKEN)))
else:
    open(path, "w").write(filler(int(8000 * CHARS_PER_TOKEN)))
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
