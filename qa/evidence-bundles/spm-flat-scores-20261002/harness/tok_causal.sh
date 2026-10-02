#!/usr/bin/env bash
# Causal check across both tokenizer test targets, and the TinyLlama runnable smoke on both builds.
cd "$HOME/Camelid" || exit 1
source ~/.cargo/env
cp src/tokenizer/mod.rs /tmp/f2r/tok-mod.keep
sed -i 's/let spm_ignores_merges = model_name == "llama" \&\& !scores_are_flat;/let spm_ignores_merges = model_name == "llama";/' src/tokenizer/mod.rs
cargo test --no-fail-fast --all-features --test tokenizer --test runnable_tokenizer > /tmp/f2r/tok-causal3.log 2>&1; echo "CAUSAL_EXIT=$? (must be non-zero)"
grep -E "^test .* FAILED|encode_mismatches|edge parity failed for|^test result" /tmp/f2r/tok-causal3.log | cut -c1-170
cp /tmp/f2r/tok-mod.keep src/tokenizer/mod.rs
echo "tracked diff after restore: $(git diff --stat | tail -1)"
for bin in /tmp/f2h/bin/camelid-head /tmp/f2r/bin/camelid-spmfix; do
  echo "== runnable-smoke $($bin --version)"
  timeout 900 $bin runnable-smoke models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf > /tmp/f2r/smoke-$(basename $bin).log 2>&1
  echo "SMOKE_EXIT=$?"; tail -6 /tmp/f2r/smoke-$(basename $bin).log | cut -c1-400
done
echo TOK_CAUSAL_DONE
