#!/usr/bin/env bash
# Evidence for the flat-score SPM fix, from the committed head: release build, corpus and
# edge-pack comparisons against TinyLlama's own tokenizer, and the GGUF metadata facts.
set -u
cd "$HOME/Camelid" || exit 1
source ~/.cargo/env
HEAD=$(git rev-parse --short HEAD)
echo "commit=$HEAD tracked_changes=$(git status --porcelain --untracked-files=no | wc -l)"
echo "tokenizer diff main..0477691e: $(git diff --stat c7ea906a 0477691e -- src/tokenizer | wc -l) lines"
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release --bin camelid --target-dir target/eval > /tmp/f2r/tok-release2.log 2>&1
echo "BUILD_EXIT=$?"
cp target/eval/release/camelid /tmp/f2r/bin/camelid-spm-$HEAD
MAIN=/tmp/f2h/bin/camelid-base FIX=/tmp/f2r/bin/camelid-spm-$HEAD
$MAIN --version; $FIX --version

export GGUF="$HOME/Camelid/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf" HF_HOME=/mnt/disks/data/hf PYTHONPATH=/tmp/f2r/pylib
PY=/mnt/disks/data/research/venv/bin/python
OUT=/tmp/f2r/spm-evidence
rm -rf $OUT && mkdir -p $OUT
$PY /tmp/f2r/compare_spm.py /tmp/f2r/spm-corpus.json $OUT/corpus-comparison.json \
  "camelid:camelid $($MAIN --version | cut -d' ' -f2):$MAIN" \
  "camelid:camelid $($FIX --version | cut -d' ' -f2):$FIX" \
  "llamacpp:llama.cpp acd79d603:/mnt/disks/data/research/llama.cpp-acd79d6/build/bin/llama-tokenize" 2>&1 | grep -v -i warn
for pair in main:$MAIN fix:$FIX; do
  $PY /tmp/f2r/edge_pack_sp.py fixtures/tokenizer/tinyllama-chat-template-edge-cases.json "${pair#*:}" "$GGUF" $OUT/edge-pack-${pair%%:*}.json > /dev/null 2>&1
  echo "edge ${pair%%:*}: $(python3 -c "import json; c=json.load(open('$OUT/edge-pack-${pair%%:*}.json'))['cases']; print(sum(x['sentencepiece_vs_camelid'] for x in c), 'of', len(c), 'match; llama.cpp ids match', sum(x['sentencepiece_vs_llama_cpp'] for x in c))")"
done
python3 /tmp/f2r/gguf_scores.py models/*.gguf > $OUT/gguf-tokenizer-metadata.txt
cat $OUT/gguf-tokenizer-metadata.txt
python3 -c "
import hashlib, json
texts = json.load(open('/tmp/f2r/spm-corpus.json')); sources = json.load(open('/tmp/f2r/spm-corpus-sources.json'))
counts = {}
for s in sources: counts[s] = counts.get(s, 0) + 1
json.dump({'texts': len(texts), 'by_source': counts, 'corpus_sha256': hashlib.sha256(open('/tmp/f2r/spm-corpus.json','rb').read()).hexdigest()}, open('$OUT/corpus-provenance.json', 'w'), indent=1)
"
echo SPM_EVIDENCE_DONE
