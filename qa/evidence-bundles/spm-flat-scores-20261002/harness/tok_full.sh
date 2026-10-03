#!/usr/bin/env bash
# Full test suite on the tokenizer fix (TinyLlama present), to find every test the change affects.
cd "$HOME/Camelid" || exit 1
source ~/.cargo/env
echo "branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD) changed=$(git status --porcelain --untracked-files=no | wc -l)"
cargo test --all-targets --all-features --no-fail-fast -- --skip smoke_admits_tinyllama > /tmp/f2r/tok-full.log 2>&1
echo "TEST_EXIT=$?"
grep -E "^test result:" /tmp/f2r/tok-full.log | awk '{p+=$4; f+=$6; i+=$8} END {print "passed", p, "failed", f, "ignored", i, "binaries", NR}'
grep -E "^test .* FAILED$" /tmp/f2r/tok-full.log | sort -u
echo TOK_FULL_DONE
