#!/usr/bin/env bash
# Final checks for the tokenizer fix before committing.
cd "$HOME/Camelid" || exit 1
source ~/.cargo/env
echo "branch=$(git rev-parse --abbrev-ref HEAD) changed=$(git status --porcelain --untracked-files=no | wc -l)"
cargo fmt --all -- --check > /tmp/f2r/tok-fmt.log 2>&1; echo "FMT_EXIT=$?"
cargo test --all-features --test tokenizer --test runnable_tokenizer > /tmp/f2r/tok-test2.log 2>&1; echo "TOK_TEST_EXIT=$?"
grep -E "^test result:|FAILED|encode_mismatches" /tmp/f2r/tok-test2.log

cp src/tokenizer/mod.rs /tmp/f2r/tok-mod.keep
sed -i 's/let spm_ignores_merges = model_name == "llama" \&\& !scores_are_flat;/let spm_ignores_merges = model_name == "llama";/' src/tokenizer/mod.rs
cargo test --all-features --test tokenizer --test runnable_tokenizer > /tmp/f2r/tok-causal2.log 2>&1; echo "CAUSAL_EXIT=$? (must be non-zero)"
grep -E "^test .* FAILED|encode_mismatches|edge parity failed for" /tmp/f2r/tok-causal2.log | cut -c1-160
cp /tmp/f2r/tok-mod.keep src/tokenizer/mod.rs
git diff --stat | tail -1

cargo clippy --all-targets --all-features -- -D warnings > /tmp/f2r/tok-clippy-all.log 2>&1; echo "CLIPPY_ALL_EXIT=$?"
cargo clippy --all-targets -- -D warnings > /tmp/f2r/tok-clippy-default.log 2>&1; echo "CLIPPY_DEFAULT_EXIT=$?"
bash /tmp/f2h/run_gates.sh
HF_HOME=/mnt/disks/data/hf PYTHONPATH=/tmp/f2r/pylib /mnt/disks/data/research/venv/bin/python scripts/gen-tinyllama-sentencepiece-reference.py --check 2>&1 | grep -v -i warn

for bin in /tmp/f2h/bin/camelid-head /tmp/f2r/bin/camelid-spmfix; do
  echo "== runnable-smoke $($bin --version)"
  timeout 900 $bin runnable-smoke --model models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf > /tmp/f2r/smoke-$(basename $bin).log 2>&1
  echo "SMOKE_EXIT=$?"; tail -4 /tmp/f2r/smoke-$(basename $bin).log | cut -c1-300
done
echo TOK_FINAL_DONE
