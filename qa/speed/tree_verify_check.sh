#!/usr/bin/env bash
# Independent lossless + speed cross-check for the GPU tree-verify path.
# The tree-verify path (CAMELID_SPEC_TREE on) MUST emit a byte-identical greedy
# token-id sequence to plain decode (gate off) — that's the lossless receipt.
# Usage: bash tree_verify_check.sh "<gate value e.g. suffix|merge|ngram>" <prompt-file> [maxtok]
# Run only when the GPU is free (no other camelid.exe). Reuses parity_check.mjs.
set -u
cd "$(git rev-parse --show-toplevel)"
MODEL="${MODEL:-../models/Qwen3-4B-Q8_0.gguf}"  # override with MODEL=/path/to.gguf
GATE="${1:-suffix}"
PROMPT="${2:-qa/speed/depth_prompt.txt}"
MAXTOK="${3:-128}"
mkdir -p qa/speed/tv

echo "=== baseline: plain greedy (CAMELID_SPEC_TREE unset) ==="
unset CAMELID_SPEC_TREE
./target/release/camelid.exe bench-generate "$MODEL" --prompt-file "$PROMPT" \
  --max-tokens "$MAXTOK" --temperature 0 --iterations 2 2>/dev/null > qa/speed/tv/bl.json
echo "=== tree-verify (CAMELID_SPEC_TREE=$GATE) ==="
CAMELID_SPEC_TREE="$GATE" ./target/release/camelid.exe bench-generate "$MODEL" --prompt-file "$PROMPT" \
  --max-tokens "$MAXTOK" --temperature 0 --iterations 2 2>/dev/null > qa/speed/tv/tree.json

echo "=== LOSSLESS + SPEED ==="
# parity_check.mjs: PASS iff token-id sha identical (lossless); prints tok/s + ratio.
node qa/speed/parity_check.mjs qa/speed/tv/bl.json qa/speed/tv/tree.json
echo "(tree/baseline t/s ratio above = the speculative speedup; PASS = lossless)"
