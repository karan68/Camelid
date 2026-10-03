#!/usr/bin/env bash
# Re-run only the corpus comparison (binaries already built), now recording pairwise identical ids.
cd "$HOME/Camelid" || exit 1
export GGUF="$HOME/Camelid/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf" HF_HOME=/mnt/disks/data/hf PYTHONPATH=/tmp/f2r/pylib
MAIN=/tmp/f2h/bin/camelid-base FIX=/tmp/f2r/bin/camelid-spm-cd81d7fb
/mnt/disks/data/research/venv/bin/python /tmp/f2r/compare_spm.py /tmp/f2r/spm-corpus.json /tmp/f2r/spm-evidence/corpus-comparison.json \
  "camelid:camelid $($MAIN --version | cut -d' ' -f2):$MAIN" \
  "camelid:camelid $($FIX --version | cut -d' ' -f2):$FIX" \
  "llamacpp:llama.cpp acd79d603:/mnt/disks/data/research/llama.cpp-acd79d6/build/bin/llama-tokenize" 2>&1 | grep -v -i warn
echo SPM_COMPARE2_DONE
