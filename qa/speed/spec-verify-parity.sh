#!/usr/bin/env bash
# spec-verify-parity.sh — Apple-Silicon twin of spec-verify-parity.ps1.
#
# SPEC-VERIFY LOSSLESSNESS GATE: greedy speculative decode must be token-identical
# to this build's own plain greedy decode, on EVERY workload column (including the
# spec-hostile mandatory-report columns). The denominator is always Camelid's own
# plain greedy stream — no llama.cpp comparison.
#
# APPLE-SILICON / PHASE-0 REALITY  (see docs/perf-deep-dive/WIN2METAL_RECON.md §A)
#   The Metal GPU speculative-verify kernels do NOT exist yet: on a non-CUDA build
#   `verify_drafts_gpu` returns Ok(None) immediately, so every verify round falls
#   back to the CPU chunk verify (forward_greedy_verify_chunk + KV rollback), and
#   `bench-speculative` runs on the CPU forward path (the Metal resident engine is
#   not engaged by this harness on Mac — recon §A2).
#
#   FINDING (Phase 0, base 28f224b, M4): that CPU fallback is NOT bit-exact here.
#   Whenever a verify round fires, the spec stream diverges from plain greedy
#   (recon §A1 — the batched forward_greedy_verify_chunk is not byte-identical to
#   single-token decode). So this gate currently reports FAIL on the linear lane,
#   by design: it is correctly catching a real pre-existing defect, NOT a harness
#   bug. The gate goes green only once (a) the CPU verify forward is made byte-exact
#   and/or (b) a byte-exact Metal verify_batch becomes the active path AND the
#   harness is wired to engage the Metal resident engine on Mac (recon §A2). The
#   GPU-verify lane status is derived honestly from gpu_verify_rounds /
#   cpu_verify_rounds and will flip to "gpu-resident" once that path is reached.
#
# LANES (mirrors the .ps1): both run through bench-speculative, which executes a
# plain greedy baseline and a greedy speculative run back-to-back in one process
# and emits a single JSON record carrying the intra-Camelid lossless verdict.
#   linear : default verify path (CAMELID_SPEC_TREE unset)
#   tree   : tree-batched verify path (CAMELID_SPEC_TREE=1)
# On this build both lanes degrade to the same CPU chunk verify.
#
# Usage:
#   qa/speed/spec-verify-parity.sh
#   BIN=/path/to/camelid MODEL=/path/to.gguf qa/speed/spec-verify-parity.sh
#
# Exits non-zero if ANY (column x lane) pair diverges from plain greedy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- pins (all overridable by env) -----------------------------------------
BIN="${BIN:-/Volumes/Untitled/camelid-target/release/camelid}"
MODEL="${MODEL:-/Volumes/Untitled/models/Llama-3.2-1B-Instruct-Q8_0.gguf}"
PROMPTS_JSON="${PROMPTS_JSON:-$SCRIPT_DIR/prompts.json}"
DRAFTER="${DRAFTER:-ngram}"   # greedy n-gram drafter (no draft model needed)

# --- preflight --------------------------------------------------------------
[ -x "$BIN" ]            || { echo "[spec-verify-parity] bin not found/executable: $BIN" >&2; exit 2; }
[ -f "$MODEL" ]         || { echo "[spec-verify-parity] model not found: $MODEL" >&2; exit 2; }
[ -f "$PROMPTS_JSON" ]  || { echo "[spec-verify-parity] prompts not found: $PROMPTS_JSON" >&2; exit 2; }
command -v node >/dev/null || { echo "[spec-verify-parity] node is required (JSON parsing)" >&2; exit 2; }

# --- teardown ---------------------------------------------------------------
WORK="$(mktemp -d "${TMPDIR:-/tmp}/spec_verify_parity.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT INT TERM

# --- materialize each column's prompt verbatim, emit "id<TAB>n_gen" lines ----
# Iterates EVERY column in prompts.json (same set the .ps1 walks), so the
# mandatory >512-context longctx_splitk column is always included.
node -e '
  const fs = require("fs");
  const pack = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
  const dir = process.argv[2];
  const out = [];
  for (const c of pack.columns) {
    fs.writeFileSync(`${dir}/${c.id}.txt`, String(c.prompt));   // exact bytes, no template
    out.push(`${c.id}\t${c.n_gen}`);
  }
  process.stdout.write(out.join("\n") + "\n");
' "$PROMPTS_JSON" "$WORK" > "$WORK/_columns.tsv"

echo "[spec-verify-parity] bin=$BIN"
echo "[spec-verify-parity] model=$(basename "$MODEL")  drafter=$DRAFTER  columns=$(grep -c . "$WORK/_columns.tsv")"
echo "[spec-verify-parity] greedy (temperature 0 / argmax); GPU verify expected unavailable on this build (CPU chunk fallback)"
echo

# Run one lane of bench-speculative and print its single stdout JSON record.
#   $1 = workload id, $2 = prompt file, $3 = max-tokens (n_gen), $4 = lane
run_lane() {
  local id="$1" pf="$2" ngen="$3" lane="$4" tree=""
  [ "$lane" = "tree" ] && tree="1"
  # bench-speculative runs plain greedy + greedy spec back-to-back and writes the
  # JSON record to stdout (human summary goes to stderr). Capture the last { line.
  CAMELID_SPEC_TREE="$tree" "$BIN" bench-speculative "$MODEL" \
    --drafter "$DRAFTER" --workload "$id" --prompt-file "$pf" \
    --max-tokens "$ngen" --warmup 2>"$WORK/${id}.${lane}.err" \
    | grep -E '^[[:space:]]*\{' | tail -n1
}

fail=0
total=0
while IFS=$'\t' read -r id ngen; do
  [ -n "$id" ] || continue
  pf="$WORK/${id}.txt"
  for lane in linear tree; do
    total=$((total + 1))
    json="$(run_lane "$id" "$pf" "$ngen" "$lane" || true)"
    if [ -z "$json" ]; then
      printf '  %-22s %-6s : NO JSON (see %s)\n' "$id" "$lane" "$WORK/${id}.${lane}.err" >&2
      tail -n 3 "$WORK/${id}.${lane}.err" >&2 || true
      fail=$((fail + 1))
      continue
    fi
    # verdict\tdiv_idx\taccept%\tgpu_rounds\tcpu_rounds
    parsed="$(node -e '
      const r = JSON.parse(process.argv[1]);
      const div = r.first_divergent_generated_token_index;
      const ok = (r.lossless === true && div < 0);
      const pct = Math.round((r.accept_rate || 0) * 100);
      process.stdout.write([
        ok ? "LOSSLESS" : "DIVERGE", div, pct,
        r.gpu_verify_rounds || 0, r.cpu_verify_rounds || 0,
      ].join("\t"));
    ' "$json")"
    IFS=$'\t' read -r verdict div pct gpu_rounds cpu_rounds <<<"$parsed"

    # GPU-verify lane status, derived honestly from the round counters.
    if [ "$gpu_rounds" -gt 0 ]; then
      gpu_status="gpu-resident (gpu_rounds=$gpu_rounds)"
    elif [ "$cpu_rounds" -gt 0 ]; then
      gpu_status="unavailable (CPU chunk fallback)"
    else
      gpu_status="n/a (no spec rounds — all normal steps)"
    fi

    printf '  %-22s %-6s : %-8s  gpu-verify=%s  (div_idx=%s, accept=%s%%, cpu_rounds=%s)\n' \
      "$id" "$lane" "$verdict" "$gpu_status" "$div" "$pct" "$cpu_rounds"
    [ "$verdict" = "LOSSLESS" ] || fail=$((fail + 1))
  done
done < "$WORK/_columns.tsv"

echo
if [ "$fail" -gt 0 ]; then
  echo "FAIL: $fail/$total (column x lane) pair(s) diverged from plain greedy (or produced no record)."
  exit 1
fi
echo "PASS: greedy spec decode is token-identical to plain greedy on all $total (column x lane) pairs; GPU-verify unavailable on this build (CPU chunk fallback) — Phase-0 expected."
