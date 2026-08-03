# Tokenizer special-token scan — O(chars x vocab) removed

Executed 2026-07-31 on a 16 GB Apple M4 Mac mini (10 cores), base `2c13e52e`.

## 0. Verdict

`Tokenizer::encode` scanned the **entire vocabulary at every character** of the
input whenever the special-token partition was active. On the gemma-3-1b-it-Q8_0
row (262,144 entries) that cost **0.53 ms per output token** — about 500x the
plain path — and it sat on the critical path of every `/v1/chat/completions`
request, outside the inference engine.

Replacing the scan with a first-byte-bucketed index built once at tokenizer
construction removes the vocabulary term entirely. The special-aware path is now
**indistinguishable from the plain path** (delta 0.0 us/token, within noise) on
every family measured, and token ids are **byte-identical** — 3,547,564 ids over
13 tokenizer rows.

The measurement that opened this was taken on gemma-3, which is the only family
in tree whose `parse_special = false` path was already fast. **Every other family
paid the same scan in BOTH modes**, so this was never a chat-only problem — plain
`/v1/completions` on llama3, qwen3, mistral, tinyllama and phi-4-mini paid it
too. See section 3.

## 1. The defect

`longest_control_token_at` (src/tokenizer/mod.rs) answered "is there a special
token starting at this byte offset?" by filtering `self.tokens`:

```rust
self.tokens
    .iter()
    .filter(|token| matches!(token.kind, TokenKind::UserDefined)
        || (include_control && matches!(token.kind, TokenKind::Control)))
    .filter(|token| !token.text.is_empty())
    .filter(|token| text[byte_start..].starts_with(&token.text))
    .max_by_key(|token| token.text.len())
```

`next_control_token_start` then called it **once per character**:

```rust
text[byte_start..].char_indices()
    .map(|(offset, _)| byte_start + offset)
    .find(|idx| self.longest_control_token_at(text, *idx, include_control).is_some())
```

so finding the next special in a 10,765-character prompt on a 262k-entry vocab
performed ~2.8 x 10^9 iterations. That is the whole of the reported
"0.525 ms/token, strictly linear, no fixed component" shape: the cost is per
character, and only *looks* per-token because tokens/character is roughly
constant for natural text.

Two entry points reach it:

| caller | `include_control` | families affected |
|---|---|---|
| `encode_piece` (SPM) | `true`, only when `parse_special` | gemma3, gemma4, mistral, tinyllama, phi-3 |
| `encode_bpe_text` (GPT-2/BPE) | `parse_special` — **runs in both modes** | llama3, qwen3, qwen2.5, phi-4-mini |
| `add_dummy_prefix_after_control_tokens` (SPM, `parse_special = false`) | `true` | any SPM row with `add_space_prefix = true` |

The third row is why gemma-3 looked different: gemma-3 carries
`add_space_prefix = false`, so its plain path returns before the scan. Mistral
and TinyLlama carry `add_space_prefix = true` and paid it with
`parse_special = false`.

A third full-vocabulary scan, `chat_control_marker_rstrips`, ran once per
special token matched.

## 2. The fix

`SpecialsIndex`, built once from `tokens`:

- **first-byte bitset**, one per `include_control` mode — an O(1) reject that
  is the whole per-character fast path. A byte whose bit is clear cannot start
  any eligible pattern.
- **first-byte buckets**, each ordered longest-first, so the first kind-eligible
  hit IS the longest match. Buckets are tiny: the largest in tree is gemma-3's
  `<` bucket at 6,321 patterns, and it is only consulted at an actual `<`.
- **rstrip marker set**, replacing the `chat_control_marker_rstrips` scan.

`next_control_token_start` now scans raw bytes rather than `char_indices`. That
is equivalent, not an approximation: a pattern's text is a `&str`, so its first
byte is never a UTF-8 continuation byte, so the filter cannot fire at a
non-boundary offset — and `longest_at` re-checks `is_char_boundary` before
slicing regardless.

The index is a `OnceLock` field. `from_gguf` seeds it eagerly so real rows pay
the build at load time; struct-literal construction (test fixtures, and the two
call sites in src/api/mod.rs that replace `tokens` after construction) leaves it
empty and it is built on first use, so those stay correct. A `debug_assert` on
the recorded vocabulary length catches a stale index in test builds.

Nothing about the *answer* changed: same kind filter, same `starts_with`, same
longest-match rule.

## 3. Before / after — in-process `encode`

`examples/tokenizer_probe.rs bench`, best of 3, same content string per row, no
HTTP. "before" is the same example built against unmodified `2c13e52e`.

#### 2,730-char prompt

| family | tokens | before plain ms | before special ms | after plain ms | after special ms | special speedup |
|---|---|---|---|---|---|---|
| gemma3-1b | 589 | 0.40 | 318.01 | 0.387 | 0.355 | **896x** |
| mistral-7b | 673 | 38.27 | 38.25 | 0.368 | 0.320 | **120x** |
| tinyllama | 673 | 35.62 | 35.48 | 0.270 | 0.260 | **136x** |
| llama3-8b | 589 | 172.66 | 175.82 | 0.674 | 0.660 | **266x** |
| qwen3-0.6b | 589 | 171.05 | 170.34 | 0.676 | 0.673 | **253x** |
| phi4-mini | 589 | 228.12 | 215.69 | 0.655 | 0.654 | **330x** |

#### 21,450-char prompt

| family | tokens | before plain ms | before special ms | after plain ms | after special ms | special speedup |
|---|---|---|---|---|---|---|
| gemma3-1b | 4621 | 3.63 | 2494.89 | 3.700 | 3.793 | **658x** |
| mistral-7b | 5281 | 381.15 | 394.38 | 3.593 | 3.494 | **113x** |
| tinyllama | 5281 | 321.16 | 273.91 | 2.292 | 2.219 | **123x** |
| llama3-8b | 4621 | 1150.62 | 1174.78 | 5.144 | 5.040 | **233x** |
| qwen3-0.6b | 4621 | 1338.27 | 1304.88 | 5.100 | 5.243 | **249x** |
| phi4-mini | 4621 | 1810.66 | 1818.52 | 5.110 | 5.117 | **355x** |

Read the "before plain" column: on every family except gemma-3 it is as large as
"before special". The scan was never gated on `parse_special` for those rows.

Per-token delta between the two modes, gemma-3-1b — the quantity originally
reported at 0.525 ms/token:

| chars | tokens | before delta us/token | after delta us/token |
|---|---|---|---|
| 130 | 29 | 530.6 | -0.1 |
| 650 | 141 | 534.2 | -0.1 |
| 2730 | 589 | 539.2 | -0.1 |
| 5460 | 1177 | 539.4 | -0.1 |
| 10725 | 2311 | 540.5 | 0.0 |
| 21450 | 4621 | 539.1 | 0.0 |

The mode delta is gone. What remains is the shared cost of the encoder itself.

gemma-4 is unchanged by this work and remains ~13x slower per character than
gemma-3 in **both** modes (4.7 ms vs 0.36 ms at 2,730 chars): that is the
rank-based symbol-merge path in `encode_spm_segment`, a separate lever. It has
only 23 special tokens, so it never had a scan problem.

## 4. Before / after — `/tokenize` endpoint

Same content string per row in both modes, best of 3, one server alive at a time,
each killed by saved PID and confirmed dead before the next started. "before" is
an untouched `2c13e52e` release build,
sha256 `66552fc25537df22993a4abe5c40c6d03a163613febef8de355ffeae4428ee7f`. The
~19–23 ms floor is the HTTP + JSON round trip of the endpoint itself, not
tokenizer time.

| chars | tokens | before plain ms | before special ms | after plain ms | after special ms | before delta us/tok | after delta us/tok |
|---|---|---|---|---|---|---|---|
| 38 | 10 | 18.80 | 25.06 | 20.19 | 18.95 | 625.6 | -124.0 |
| 152 | 37 | 19.45 | 37.81 | 18.89 | 18.89 | 496.2 | -0.0 |
| 304 | 73 | 18.86 | 55.88 | 19.52 | 19.32 | 507.0 | -2.7 |
| 722 | 172 | 19.08 | 105.60 | 19.46 | 19.20 | 503.0 | -1.5 |
| 1444 | 343 | 19.21 | 191.49 | 19.46 | 19.40 | 502.3 | -0.2 |
| 2888 | 685 | 19.40 | 364.68 | 19.42 | 19.69 | 504.1 | 0.4 |
| 5776 | 1369 | 20.18 | 708.76 | 19.97 | 20.07 | 503.0 | 0.1 |
| 11362 | 2692 | 20.90 | 1370.91 | 21.35 | 20.93 | 501.5 | -0.2 |
| 22724 | 5383 | 22.94 | 2727.67 | 23.30 | 23.41 | 502.5 | 0.0 |

Before, `parse_special = true` cost **118.9x** `parse_special = false` at the top
of the ladder. After, the ratio is **1.00x**: both modes are pinned to the same
HTTP floor and the special path no longer has a slope. Reproduces the published
shape (flat plain path, ~0.5 ms/token special path) before the change and
removes it after.

Endpoint token ids: 9 sizes x 2 modes, 21,528 ids, identical before and after.

### Chat path, end to end

The quantity the write-up called "1.35 s of pure pre-engine latency". Streaming
`/v1/chat/completions`, `CAMELID_STREAM_TIMING_DIAGNOSTICS=1`, 11,362-char user
message (2,700 prompt tokens), greedy, `max_tokens = 1`, 3 runs each, same
session, one server at a time:

| build | `timings_ms.tokenize` | wall clock |
|---|---|---|
| before | 1499 / 1622 / 1676 ms | 53.33 / 53.35 / 53.78 s |
| after | 2 / 2 / 2 ms | 51.60 / 51.87 / 51.73 s |

**~1.6 s removed from TTFT on every chat request on this row**, and it shows up
one-for-one in wall clock. It is still hidden behind a ~51 s prefill today; after
the gemma-3 Metal prefill work lands it would have been the largest single
remaining term.

## 5. Identity gate

Tokenization identity is load-bearing for every parity receipt in this repo, so
the gate is exact equality of token ids, not a tolerance.

`examples/tokenizer_probe.rs dump` encodes 876 cases under all four
`(add_special, parse_special)` combinations and writes the ids to JSON. The same
876 cases were run against the unmodified `2c13e52e` build and the fixed build,
on the same GGUF artifacts, and diffed.

Cases: every string value found anywhere in the committed prompt packs
(`qa/prompt-packs/*.json`, 529 cases — including all three packs the task names,
`gemma3-chat-template-shapes-v1`, `gemma3-chat-gate-pack-v1`,
`gemma3-windowed-context-pack-v1`), plus 347 synthetic adversarial strings:
every marker shape in tree at string start/end/adjacent/wrapped, partial and
overlapping markers, markers beside multi-byte and astral characters, and long
repeated chat scaffolds.

| family | vocab | model | specials | encode results | token ids | verdict |
|---|---|---|---|---|---|---|
| gemma3-1b | 262144 | llama_spm | 6414 | 3504 | 274520 | IDENTICAL |
| gemma4-E2B | 262144 | llama_spm | 23 | 3504 | 276178 | IDENTICAL |
| gemma4-E4B | 262144 | llama_spm | 23 | 3504 | 276178 | IDENTICAL |
| llama3-8b | 128256 | gpt2_bpe | 256 | 3504 | 236174 | IDENTICAL |
| llama32-1b | 128256 | gpt2_bpe | 256 | 3504 | 236174 | IDENTICAL |
| mistral-7b | 32768 | llama_spm | 770 | 3504 | 302928 | IDENTICAL |
| mixtral-8x7b | 32000 | llama_spm | 2 | 3504 | 303126 | IDENTICAL |
| phi3-mini | 32064 | llama_spm | 13 | 3504 | 299354 | IDENTICAL |
| phi4-mini | 200064 | gpt2_bpe | 14 | 3504 | 235106 | IDENTICAL |
| qwen25-05b | 151936 | gpt2_bpe | 22 | 3504 | 269276 | IDENTICAL |
| qwen3-06b | 151936 | gpt2_bpe | 26 | 3504 | 269016 | IDENTICAL |
| qwen3-8b | 151936 | gpt2_bpe | 26 | 3504 | 269016 | IDENTICAL |
| tinyllama | 32000 | llama_spm | 2 | 3504 | 300518 | IDENTICAL |

**13 families, 45,552 encode results each side, 3,547,564 token ids, zero
divergence.**

In-tree, permanently: `specials_index_reference_scan` and friends in
`src/tokenizer/mod.rs` keep the replaced vocabulary scan verbatim as
`reference_longest_control_token_at` / `reference_next_control_token_start` /
`reference_chat_control_marker_rstrips`, and assert the index agrees at **every
byte offset** of an adversarial corpus, in both modes, on a fixture vocabulary
built to hit every branch (prefix overlaps, the same shape as USER_DEFINED in one
entry and CONTROL in another, multi-byte patterns, an empty-text entry).
`specials_index_matches_reference_on_real_vocab_when_available` replays the same
comparison against real 262k-entry rows when the artifacts are reachable.

Suites run: `cargo test --lib` 1414 passed / 0 failed; `--test tokenizer` 27
passed; `--test runnable_tokenizer`, `--test dg_tokenizer_parity`,
`--test gemma4_template_shapes` all pass. `cargo fmt --all --check` clean,
`cargo clippy --all-targets -- -D warnings` clean.

## 6. Reproduce

```bash
cargo build --release --example tokenizer_probe

# 1. build the case list from the committed prompt packs + adversarial strings
python3 scripts/gen-tokenizer-identity-cases.py qa/prompt-packs cases.json

# 2. dump ids per row, once per build under test
target/release/examples/tokenizer_probe dump <row.gguf> cases.json before/<row>.json

# 3. the gate — exits non-zero on any divergence
python3 scripts/diff-tokenizer-identity-dumps.py before after

# throughput, both parse_special modes over a content ladder
target/release/examples/tokenizer_probe bench <row.gguf> 5
# special-token population and first-byte distribution for a row
target/release/examples/tokenizer_probe stats <row.gguf>
```

## 7. Out of scope, found on the way

- **`Tokenizer::from_gguf` takes 72 s on Phi-4-mini-instruct-Q4_K_M** (200,064
  entries). Unrelated to the special-token scan — it is in construction, not
  encode, and is unchanged by this work. Every other row builds in under 1.4 s.

  **Resolved after this report.** The cost was not per-token work over the
  vocabulary: `is_exact_phi4_mini_q4km` SHA-256'd the entire 2.5 GB artifact to
  check the pin admitting the `gpt-4o` pre-tokenizer, and that volume reads at
  ~29 MB/s. Narrowing the pin to the GGUF header region `[0, data_start_offset)`
  — 8,250,976 bytes, everything that can move a token id — took construction to
  65 ms warm (#577).

  Loading the row was still slow afterwards for a *second*, independent
  full-file read: `build_loaded_model` hashes the whole GGUF to name the receipt
  lane. That digest is now memoized across process starts on
  (path, len, mtime, dev, ino), measured 65,098 ms cold → 0.1 ms warm on the
  same artifact. Two full-file reads per load became zero on every start after
  the first.
- **gemma-4's `encode_spm_segment` rank-merge path** is ~13x slower per
  character than gemma-3's, in both modes.
