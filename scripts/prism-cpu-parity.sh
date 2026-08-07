#!/usr/bin/env bash
# Bonsai CPU-lane parity: run a Prism packed low-bit row with the GPU forced OFF
# and check greedy raw-completion parity against a committed oracle.
#
# This is the lane that used to be unreachable on Windows: the loader kept the
# packed Q1_0/Q2_0 wire (a compile-time `cfg!` that is true for every Windows
# build) while the planner routed decode to the CPU, and no CPU kernel consumed
# those bytes. `--gpu off` is the user-facing way into exactly that state.
#
# Usage:
#   scripts/prism-cpu-parity.sh <model.gguf> <model-id> <row-id> <oracle.json> <out.json>
set -euo pipefail

MODEL="$1"; MODEL_ID="$2"; ROW_ID="$3"; ORACLE="$4"; OUT="$5"
PORT="${PORT:-8185}"
BIN="${BIN:-target/release/camelid.exe}"
LOG="${LOG:-$(dirname "$OUT")/serve-$ROW_ID.log}"

mkdir -p "$(dirname "$OUT")"

# Free-RAM guard: these rows stream packed wire, but refuse to start a run that
# would push the box into swap (this repo has crashed a host that way twice).
FREE_MB=$(powershell -NoProfile -Command \
  "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB)")
echo "free RAM: ${FREE_MB} MB"
if [ "$FREE_MB" -lt 2500 ]; then
  echo "REFUSING: under 2.5 GB free RAM" >&2
  exit 1
fi

"$BIN" serve --model "$MODEL" --gpu off --addr "127.0.0.1:$PORT" >"$LOG" 2>&1 &
SERVE_PID=$!
# Kill only the server we started -- never a blanket taskkill, a concurrent
# session may be holding its own engine.
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "serve died during startup; log:" >&2; tail -40 "$LOG" >&2; exit 1
  fi
  sleep 2
done

# Record the lane the engine says it took, so the receipt cannot claim a CPU
# result that actually ran on the GPU.
curl -fsS "http://127.0.0.1:$PORT/v1/health" > "$(dirname "$OUT")/health-$ROW_ID.json"

node scripts/raw-decode-parity.mjs \
  --camelid "http://127.0.0.1:$PORT" \
  --model-id "$MODEL_ID" \
  --row-id "$ROW_ID" \
  --reference-in "$ORACLE" \
  --variant "prism_low_bit_cpu_block_dot" \
  --proof-chain "camelid CPU streaming Prism block-dot (matmul_rhs_transposed_prism_block_dot over packed Q1_0/Q2_0 wire, GPU forced off with --gpu off) vs the committed Windows-CUDA oracle for the same exact GGUF; raw-prompt greedy token+text parity." \
  --out "$OUT"
