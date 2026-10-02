# TinyLlama's tokenizer: flat scores, merge ranks

Everything here was produced on one machine, with the release builds and the
tools named below. `harness/check_claims.py` re-derives every number in this
README from the files in this bundle, checks `SHA256SUMS`, and fails if any of
them disagrees.

## The problem

`tests/runnable_tokenizer.rs::spm_tokenizer_matches_hf` failed on every
machine that has `models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf`. CI has no model
files, so CI never ran it.

`data/gguf-tokenizer-metadata.jsonl` shows why. The TinyLlama GGUF declares
`tokenizer.ggml.model = llama` (SentencePiece) and carries 61,249 merges, but
its 32,000 scores hold 1 distinct value: every score is 0.0. It was converted
from a HF `tokenizer.json`, which has merges and no scores. Since `e78dc0a2`,
camelid ignores the merges of a `llama` GGUF and merges pieces by score, the
way llama.cpp does. With every score equal, that falls back to merging the
leftmost pair first. Mistral 7B v0.3, the other `llama` GGUF here, has 31,728
distinct scores and 0 merges.

## The fix

A `llama` GGUF whose scores are all equal now merges by its merge ranks.
Otherwise nothing changes: real scores decide, and merges are ignored.

## Measured against the model's own tokenizer

The reference is TinyLlama's own `tokenizer.model`, run by sentencepiece
0.2.2. The corpus is 1362 texts already on this machine
(`data/corpus-provenance.json`, built by `harness/build_spm_corpus.py`):

- the 30 cases of the HF fixture;
- 27 hand-written probes (whitespace, Unicode, the words named in `e78dc0a2`);
- the 405 off-topic chat messages from the library-search calibration;
- 300 SciFact claims and 300 FiQA questions;
- 150 lines each of Rust and JavaScript from this repository.

Only the third-party texts' provenance is kept here, not the texts.

| Tokenizer | Texts tokenized like the model's tokenizer |
| --- | --- |
| camelid `v0.7.8-5-gc7ea906a` (main) | 322 of 1362 |
| llama.cpp `acd79d603` | 322 of 1362 |
| camelid `v0.7.8-6-gcd81d7fb` (this change) | 1361 of 1362 |

`data/corpus-comparison.json` has the counts by source and the first
disagreements of each tokenizer. camelid on main and llama.cpp produce
identical ids for 1361 of the 1362 texts: camelid had been a faithful copy of
llama.cpp, and both mis-tokenize this file.

The one text the fix still gets wrong is a line of Rust that contains a literal
`<s>`. camelid inserts a dummy space after control-token text even when special
tokens are not parsed. That is separate, existing behaviour, and this change
does not touch it.

## The chat-template edge pack

`fixtures/tokenizer/tinyllama-chat-template-edge-cases.json` pinned
llama.cpp's ids for five chat prompts. Tokenized the same way (BOS, `<s>` and
`</s>` as control tokens), the model's own tokenizer agrees with those ids on
0 of 5 (`data/edge-pack-main.json`). For example, the model's tokenizer splits
"camelid" as `▁cam|el|id`, and llama.cpp as `▁came|li|d`. With this change,
camelid agrees with the model's tokenizer on 5 of 5 (`data/edge-pack-fix.json`).

The fixture keeps llama.cpp's ids and gains `model_tokenizer_tokens`, written
by `scripts/gen-tinyllama-sentencepiece-reference.py` (its `--check` mode
reproduces the file). The test now holds camelid to those ids.

**This ends token-for-token parity with llama.cpp for this file.** The
TinyLlama row's llama.cpp parity evidence was produced with llama.cpp's
tokenization of this GGUF, which this bundle shows does not match the model.
Whether that row's claims need to be re-audited is the owner's decision; this
change does not edit the ledger.

## Tests

- `tests/full-suite.txt`: the full suite with TinyLlama present, run before
  the edge-pack test was pointed at the model's tokenizer:
  3149 passed, 1 failed, 159 ignored, across 70 test binaries. The failure was
  that test, still comparing against llama.cpp's ids.
- `tests/tokenizer-targets.txt`: after that change, `--test tokenizer` (28)
  and `--test runnable_tokenizer` (2) all pass.
- `tests/causal.txt`: with the gate reverted (merges ignored again), three
  tests fail: `spm_tokenizer_matches_hf` (13 of 30 cases mismatch),
  `llama_spm_with_flat_scores_merges_by_rank`, and the edge-pack test (5 of 5).
- `tests/checks.txt`: `cargo fmt --check` clean, `cargo clippy` clean with and
  without `--all-features`, and all 47 validation gates pass.
- `tests/smoke-main.json` and `tests/smoke-fix.json`: `camelid runnable-smoke`
  on the TinyLlama GGUF passes with both builds. Its prompt tokenizes
  identically under both, and so the two answers are the same.

## Reproducing

`harness/build_llamacpp.sh` builds llama.cpp's `llama-tokenize` at
`acd79d603`. `harness/build_spm_corpus.py` builds the corpus.
`harness/spm_evidence.sh` builds this change, then runs `compare_spm.py` and
`edge_pack_sp.py`; `spm_compare2.sh` re-runs the comparison alone.
`harness/spm_rank_probe.py` is the first probe, which merged the fixture's
texts by rank in Python before any Rust changed.
