# Gemma 3 1B-It Q8_0 — window-edge parity harness, Phase 1 baseline and mutation receipts

Row: `gemma_3_1b_it_q8_0` — exact file `gemma-3-1b-it-Q8_0.gguf`, sha256
`b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`, 1,069,306,368 B.
Branch `feat/gemma3-batched-prefill`, head `a82dd41a`, off `main` at `a5945f8a`.
Long-prompt TTFT campaign, Phase 1 — see `GEMMA3_METAL_CONDUCTOR.md` §16.

**This bundle makes NO performance claim of any kind and contains no speed comparison with
any other engine.** The pinned llama.cpp build is used only as a correctness oracle and was
deliberately run on its CPU backend for a stable reduction order, which makes any timing
comparison against it meaningless.

## What this bundle is for

Phase 1 of the campaign builds the parity harness *before* any kernel work, because the
existing windowed evidence has near-zero power against the error class a batched prefill
introduces. These four files are the receipts for that harness: what the current
(token-by-token) lane scores on the new pack, and what the deliberate-defect run found.

## Oracle

llama.cpp `llama-server` **version 9632 (`acd79d603`)**, CPU backend, flags
`-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 4096`, greedy `/completion`
(`temperature 0`, `top_k 1`, `seed 0`, `cache_prompt false`, `samplers ["top_k"]`,
`return_tokens true`, `n_probs 2`). Two-phase throughout: the oracle capture and the pack
build ran with **only** llama-server alive, and every camelid leg with **only** camelid
alive, each started and killed by saved PID with death verified.

`n_probs: 2` is a deviation from the frozen Phase 4 captures, which used no `n_probs`. It is
opt-in in the harness (`--top-logprobs`, default 0) precisely so this is a recorded choice
rather than a silent change of the request both engines answer. It buys per-position top-2
margins, without which a passing leg's sensitivity cannot be estimated afterwards.

## 1. `oracle-window-edge.json` — the captured reference

`scripts/chat-parity-gemma3.mjs --mode capture` over the new pack
`qa/prompt-packs/gemma3-window-edge-pack-v1.json` (24 items) at greedy depths 1/5/50.
Cross-check: the pack's own `prompt_tokens` matched the oracle's `/tokenize` length on
**24/24** items — 63/64/65, 127/128/129, 255/256/257, 511/512/513, 1023/1024/1025,
1536, 2400/2432/2433 and four at 1024.

## 2. `window-edge-baseline-parity.json` — the current lane's baseline

`--mode compare` against the Metal GPU-resident serve lane
(`selected_backend: metal_resident_q8_runtime`, `prefill_path:
q8_0_metal_resident_prefill`, `prefill_runtime_policy:
resident_single_command_buffer_prefill`; `health-resident.json`). Token identity is scored on
camelid's **own** `camelid.generated_token_ids`, not on re-encoded output text — see §16d of
the conductor for why that mattered.

- **Cross-engine prompt tokenization identical 24/24.**
- **70/72 generation legs token-AND-text identical.** `all_pass: false`.
- **Every depth-1 and depth-5 leg is clean (48/48).** Both failures are depth-50.
- **All six anchored window items (`w-edge-q-510/511/512/513`, `w-multi-1536`,
  `w-multi-2400`) are clean at every depth**, with per-leg minimum top-2 margins of
  3.45–7.81 nats. Those are the items the campaign's window gates rest on.

The two failures, disclosed with their margins:

| item | depth | diverges at | camelid top-2 gap there | oracle top-2 gap there | window_power |
|---|---|---:|---:|---:|---|
| `w-len-256` | 50 | generated index 13 | **0.468 nat** | 0.235 nat | none |
| `w-len-513` | 50 | generated index 5 | **0.0696 nat** | 0.314 nat | minimal |

Both are unanchored ladder items carrying the open-ended question "name one item mentioned
above", which invites an arbitrary choice among many equally good continuations; both flips
are inside the near-tie band this row's earlier bundle already disclosed (flips at
0.0032 / 0.0173 / 0.0353 nat, largest disclosed oracle-side gap 0.447 nat). `w-len-256`
is camelid ending a list where the oracle continues it; `w-len-513` is the two engines
quoting different sentences from the body. Neither is a window item and neither is claimed
as clean.

**A finding worth recording because it is negative:** `text_reencode_artifact` fired on
**0/72** legs, and the old text round-trip disagreed with the engine's ids on 0 legs where
the ids matched. On this pack the fixed harness reaches the same verdict as the old one. The
fix is a *power* fix for the defect class ahead, not a correction of a wrong result here.

## 3. `window-mutation-harness.json` — the deliberate-defect run (gate G7)

`gemma3_real_row_window_mutation_harness`, driving `ResidentDecodeState` directly on the real
row, over the pack's 9-item mutation subset, 12 greedy tokens per leg, under the production
schedule and seven deliberate defects expressed as `ResidentLayerSchedule` perturbations.
Positive control first: two baseline runs of `w-len-513` produced identical tokens and
**bit-identical KV** (digest `9c51bfa9b0c9eef9`), so a green matrix cannot come from a
harness that distinguishes nothing. Wall clock 1761 s in one process.

**All seven mutants caught; `survivors` is empty.** But *which observable* caught them is the
result that matters:

| mutant | caught by TOKEN identity | caught by KV equivalence |
|---|---:|---:|
| `window_minus_one` (w-1) | **0 / 9** | 9 / 9 |
| `window_plus_one` (w+1) | **0 / 9** | 9 / 9 |
| `window_on_all_layers` | 6 / 9 | 9 / 9 |
| `layer_pattern_shift_by_one` | 8 / 9 | 9 / 9 |
| `no_lower_bound` (full causal) | 9 / 9 | 9 / 9 |
| `window_on_wrong_layers` | 9 / 9 | 9 / 9 |
| `rope_tables_swapped` | 9 / 9 | 9 / 9 |

**A one-position window error changed not one generated token on any of the nine items,
including the four built specifically to make it visible.** Its KV signature is
unmistakable — 6.8M to 24.4M differing cache elements, max |ΔKV| 0.43 to 24.4, against a
2.122e-4 reduction-noise floor — but it is invisible to any argmax-only gate at every prompt
length tested, from 513 to 2400 tokens.

### Reading the numbers

`kv_differing_elements` is counted against `kv_compared_elements` **plus 262 144**: the
comparison also covers the first-token logits, which the receipt carries in the snapshot's
`final_hidden` slot but which `kv_compared_elements` (caches only) does not count. The code
was corrected after this run so future receipts share one denominator; the numbers here are
unchanged and correct, only the printed ratio is caches-only in its denominator.

`w-len-513` is the sharpest illustration of why the outlier half of the Tier B bound matters:
its `kv_median_position_max_abs` is **0.0** for both off-by-one mutants (only 1–2 of 513
query positions clip, so the median across positions is zero) while 287 743 elements differ.
A scalar bound alone would have to be set absurdly tight to see that; the per-position
outlier test sees it immediately.

Consistency check visible in the data: at N=513, `window_plus_one` and `no_lower_bound`
produce **identical** KV numbers (274 944 differing, max 4.2887e-1, hidden 1.4681). They must
— at 513 positions a 513-wide window never clips, so w+1 *is* full causal there. The harness
reproduces that identity without being told about it.

## 4. `health-resident.json`

`/health` from the serving process that produced §2, confirming the lane rather than assuming
it. Host paths are `/Volumes/...` build paths only; no home directory appears in any file in
this bundle (checked).

## What this bundle does NOT claim

- **No performance or throughput claim, and no speed comparison with llama.cpp.**
- **No `all_pass: true`** for the baseline: 70/72, with two depth-50 near-tie flips
  adjudicated above and not excused.
- **No claim about a batched prefill.** None exists yet. These are the receipts the
  batched-prefill work will be compared *against*.
- **No claim that a correct implementation cannot recall a fact from outside one layer's
  window.** 22 sliding layers stack to a receptive field of ~22×511 positions. The pack's
  power is output *sensitivity* to the mask, not semantic failure — see the pack's
  `conventions.limits`.
- **No context claim above 2 433 prompt tokens**, and nothing about any other gemma3 row,
  quant, or lane.

## Reproduce

```
# Phase 1 — oracle only (pinned llama.cpp build, DYLD_LIBRARY_PATH set to its bin dir):
llama-server -m <models>/gemma-3-1b-it-Q8_0.gguf -c 4096 --port 8090 --host 127.0.0.1 \
  -ngl 0 -ctk f32 -ctv f32 -fa off --no-repack
node scripts/build-gemma3-window-edge-pack.mjs --tokenizer http://127.0.0.1:8090 \
  --out qa/prompt-packs/gemma3-window-edge-pack-v1.json
node scripts/chat-parity-gemma3.mjs --mode capture --llama http://127.0.0.1:8090 \
  --oracle oracle-window-edge.json \
  --prompts-file qa/prompt-packs/gemma3-window-edge-pack-v1.json \
  --token-counts 1,5,50 --top-logprobs 2

# ... stop llama-server, verify dead ...

# Phase 2 — camelid only, resident lane is the default selection:
camelid serve --addr 127.0.0.1:8399 --no-open --model <models>/gemma-3-1b-it-Q8_0.gguf
node scripts/chat-parity-gemma3.mjs --mode compare --camelid http://127.0.0.1:8399 \
  --oracle oracle-window-edge.json --model-id "Gemma 3 1b It" --row-id gemma_3_1b_it_q8_0 \
  --lane-label gemma3_marker_chat_greedy_metal_resident_serve --top-logprobs 2 \
  --out window-edge-baseline-parity.json

# ... stop camelid, verify dead ...

# Phase 3 — the mutation harness, one process, no server:
CAMELID_METAL_F32Y=1 CAMELID_METAL_WIRE=1 CAMELID_METAL_WIRE_NSG8=1 \
CAMELID_GEMMA3_GGUF=<models>/gemma-3-1b-it-Q8_0.gguf \
CAMELID_GEMMA3_MUTATION_OUT=window-mutation-harness.json \
  cargo test --release --lib gemma3_real_row_window_mutation -- --nocapture
```

## Environment

Apple M4 Mac mini, 10 cores, 16 GiB RAM. Release build of this branch. The `camelid serve`
binary that produced §2 was built from the tree that became `d11bc7fd`; the mutation harness
in §3 was built from `a82dd41a`, which adds that harness. The difference between the two is
`#[cfg(test)]` code only, so the serving binary is identical in both — stated rather than
glossed. One
model-loading process alive at any time, always killed by saved PID with death verified by
`ps -p` and a port check. The 1-minute load average during the camelid legs ranged 2.4–20.5
(other sessions on this host run their own builds); that affects wall clock only, and no wall
clock is claimed here.
