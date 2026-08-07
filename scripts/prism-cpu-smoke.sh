#!/usr/bin/env bash
# CPU-lane generation smoke for one Prism/Bonsai row.
#
# Starts `camelid serve --gpu off`, asserts from /v1/health that the CPU lane is the
# one that actually ran (not just that a GPU was absent), generates greedily, and
# prints a one-line result. Refuses to start if the row cannot fit in free RAM --
# these are packed-wire loads, but a 7 GB row on a box with 4 GB free will swap the
# host to death rather than fail cleanly.
#
# Usage: scripts/prism-cpu-smoke.sh <model.gguf> <port> [max_tokens]
set -uo pipefail

MODEL="$1"; PORT="$2"; MAXTOK="${3:-5}"
BIN="${BIN:-target/release/camelid.exe}"
NAME="$(basename "$MODEL" .gguf)"
LOG="${SMOKE_LOG_DIR:-${TMPDIR:-/tmp}}/smoke-$NAME.log"

SIZE_MB=$(( $(stat -c %s "$MODEL") / 1048576 ))
FREE_MB=$(powershell -NoProfile -Command \
  "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB)")
# Wire stays packed, so RSS tracks file size; keep ~1.2 GB for activations/KV/OS.
NEED_MB=$(( SIZE_MB + 1200 ))
if [ "$FREE_MB" -lt "$NEED_MB" ]; then
  echo "$NAME | SKIPPED-HOST-LIMIT | needs ~${NEED_MB}MB, only ${FREE_MB}MB free"
  exit 3
fi

"$BIN" serve --model "$MODEL" --gpu off --addr "127.0.0.1:$PORT" >"$LOG" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null; taskkill //PID "$SERVE_PID" //F >/dev/null 2>&1' EXIT

READY=""
for _ in $(seq 1 450); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/health" 2>/dev/null | grep -q '"generation_ready":true'; then
    READY=1; break
  fi
  kill -0 "$SERVE_PID" 2>/dev/null || break
  sleep 2
done
if [ -z "$READY" ]; then
  echo "$NAME | FAIL-LOAD | never became generation_ready; see $LOG"
  exit 1
fi

H=$(curl -fsS "http://127.0.0.1:$PORT/v1/health")
BACKEND=$(echo "$H" | grep -o '"selected_backend":"[^"]*"' | cut -d'"' -f4)
CUDA=$(echo "$H"  | grep -o '"cuda_resident_active":[a-z]*' | cut -d: -f2)
QUANT=$(echo "$H" | grep -o '"quant_type":"[^"]*"' | head -1 | cut -d'"' -f4)
MID=$(curl -fsS "http://127.0.0.1:$PORT/v1/models" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)

if [ "$CUDA" != "false" ] || [ "$BACKEND" = "cuda_resident_prism_low_bit_runtime" ]; then
  echo "$NAME | FAIL-LANE | expected the CPU lane, got backend=$BACKEND cuda_resident_active=$CUDA"
  exit 1
fi

R=$(curl -sS -X POST "http://127.0.0.1:$PORT/v1/completions" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MID\",\"prompt\":\"The capital of France is\",\"max_tokens\":$MAXTOK,\"temperature\":0,\"top_k\":1,\"seed\":0,\"stream\":false}")
TEXT=$(echo "$R" | grep -o '"text":"[^"]*"' | head -1 | cut -d'"' -f4)
ERR=$(echo "$R"  | grep -o '"message":"[^"]*"' | head -1 | cut -d'"' -f4)

# Runnable-lane architectures (qwen35 / gemma2 / lfm2) deliberately fail closed on the
# raw completion surface -- there is no runnable bridge there, and falling through to
# the optimized engine would silently run the wrong forward. Retry those on chat,
# which is their supported surface. Not a fallback for real errors: only this one.
if [ -n "$ERR" ] && echo "$ERR" | grep -q "runnable-lane architecture"; then
  R=$(curl -sS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one word.\"}],\"max_tokens\":$MAXTOK,\"temperature\":0,\"top_k\":1,\"seed\":0,\"stream\":false}")
  TEXT=$(echo "$R" | grep -o '"content":"[^"]*"' | tail -1 | cut -d'"' -f4)
  ERR=$(echo "$R"  | grep -o '"message":"[^"]*"' | head -1 | cut -d'"' -f4)
  [ -n "$TEXT" ] && ERR=""
fi

if [ -n "$ERR" ]; then
  echo "$NAME | FAIL-GEN | quant=$QUANT backend=$BACKEND | $ERR"
  exit 1
fi
echo "$NAME | PASS | quant=$QUANT backend=$BACKEND cuda=$CUDA | \"$TEXT\""
