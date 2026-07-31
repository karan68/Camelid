# Gemma 3 1B-It Q8_0 — Metal GPU-resident lane parity, including the >=512-token window

Row: `gemma_3_1b_it_q8_0` (filename-anchored) — exact file `gemma-3-1b-it-Q8_0.gguf`,
sha256 `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`, 1,069,306,368 B,
upstream-verified exact against ggml-org/gemma-3-1b-it-GGUF (license gemma). Source head
`6eaf9053` (branch `feat/gemma3-metal-resident`), GEMMA3_METAL_CONDUCTOR.md Phase 4.

Lane under test: the **Metal GPU-resident Q8_0 serve lane**, selected by default —
`selected_backend: "metal_resident_q8_runtime"`, `decode_path: "q8_0_metal_resident_decode"`,
`prefill_path: "q8_0_metal_resident_prefill"`, dense serve backend `"llama"` with no runnable
runtime loaded (`health-resident.json`). This is a different lane from the one the frozen
`gemma3-1b-q8-runnable-serve-chat-parity-20260716-head-6d0d57eb/` bundle certified (the CPU
runnable bridge), and it is the first gemma3 lane that implements the sliding-window mask.

## Oracle

llama.cpp `llama-server` **version 9632 (`acd79d603`)**, CPU backend, binary sha256
`382096b1dc10da68c2bf0a97e1f0dd36db90531cdea1434760d8c1a70fea1310`, flags
`-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 4096`, greedy `/completion`
(`temperature 0`, `top_k 1`, `seed 0`, `cache_prompt false`, `samplers ["top_k"]`).
Two-phase throughout: every oracle capture and every probe was taken with **only**
llama-server running, and every camelid leg with **only** camelid running.

## 1. Sub-512 chat parity — CLEAN (`gemma3-gpu-resident-chat-parity.json`)

`scripts/chat-parity-gemma3.mjs` over the committed 5-prompt gate pack
(`qa/prompt-packs/gemma3-chat-gate-pack-v1.json`) at greedy depths 1/5/50; rendered prompts
16-19 tokens, far inside the window.

- **Cross-engine prompt tokenization identical 5/5** (llama `/tokenize` vs camelid encode).
- **15/15 generation legs token-AND-text identical**; `all_pass: true`, zero flips.

The frozen runnable-lane bundle recorded one near-tie flip on this same pack (position 16 of
the "sky colour" 50-token leg, 0.3416 nat). This bundle does **not** re-adjudicate that flip:
it was captured on a different host and a different lane, and the runnable lane was not re-run
on this pack here. What is recorded is only that the resident lane is clean on all 15 legs.

## 2. Sub-512 raw-decode parity — 3 DISCLOSED NEAR-TIE FLIPS (`gemma3-gpu-resident-raw-decode-parity.json`)

`scripts/raw-decode-parity.mjs` (raw `/v1/completions`, no chat template) over the harness's
own default 4-prompt set at depths 1/5/50. Committed **as-is with `all_pass: false`**.

As the harness scores it: **6/12 legs token-AND-text identical**. Broken out:

- Depth 1: 4/4 token-AND-text identical.
- Depth 5: 4/4 **text** identical; 2/4 also token-identical as the harness scores tokens.
- Depth 50: 0/4 as the harness scores it — 1/4 is text-identical and token-identical once
  camelid's actual emitted ids are read, and 3/4 flip once each inside the 50-token leg.

Two separate things are mixed into that `all_pass: false`, and they are not the same finding:

**(a) A harness re-encode artifact, not an engine divergence.** The harness compares the
oracle's generated token ids against camelid's *text re-encoded by camelid's tokenizer*. On this
262k SPM vocab a run of spaces re-encodes as single-space tokens (`236743`) where llama.cpp
generated the merged whitespace tokens (`138` = two spaces, `140` = four spaces), so
`"Q: What is 2+2? A:"` and `"def fibonacci(n):"` report `token_match: false` at depth 5 while
`text_match` is **true**. `camelid-raw-probe.json` re-runs the same prompts and reads camelid's
ACTUAL emitted ids (`camelid.generated_token_ids`), which removes the artifact: on
`"Q: What is 2+2? A:"` the true token streams are identical for all 43 generated tokens plus the
`<end_of_turn>` (106) stop that llama.cpp strips from its content tokens. That leg is a PASS at
the token level; the harness's `token_match: false` for it is a re-encode artifact only. No
tolerance was edited and the harness was not changed to hide it.

**(b) Three real near-tie flips**, each individually probed and disclosed below.

## 3. The >=512-token windowed receipt — CLEAN (`gemma3-gpu-resident-windowed-parity.json`)

This is the claim the CPU runnable lane cannot make. That lane implements no sliding-window
mask (`src/runnable/model.rs`: "this lane implements no window mask — a documented full-support
blocker"), so past the 512-token gemma3 window it is wrong by construction; the resident lane
carries the 5:1 local/global schedule and the window mask. The window is the file's own:
`gemma3.attention.sliding_window = 512`, `gemma3.block_count = 26`, `gemma3.context_length =
32768` (read from this exact GGUF's metadata).

New committed pack `qa/prompt-packs/gemma3-windowed-context-pack-v1.json`, three prompts whose
rendered gemma3 turns tokenize to **606 / 1205 / 2403 tokens** — 1.18x, 2.35x and 4.69x the
512-token window — each ending in a question whose answer appears only in the first sentence.
Greedy depths 1/5/50 against the same pinned oracle.

- **Cross-engine prompt tokenization identical 3/3** (606/1205/2403).
- **9/9 generation legs token-AND-text identical**; `all_pass: true`, **zero flips**.
- The oracle and camelid both answer from the far-past sentence at every depth, e.g. at 2403
  prompt tokens both emit `"The name of the river is the Willow."`.

Per the conductor's 7d envelope this pack had to be clean, and it is: the envelope is not drawn
on at all here.

### 3a. The runnable lane on the same prompt and the same oracle (`gemma3-runnable-windowed-parity.json`)

The same 606-token prompt, the same committed oracle capture, the same greedy 50-token depth —
with the resident lane switched off (`CAMELID_METAL_RESIDENT_DECODE=0`, plan `cpu_reference` /
`safe_cpu_decode`, backend `runnable-runtime`):

| lane | generated tokens (depth 50) | text |
| --- | --- | --- |
| llama.cpp `acd79d603` (oracle) | `[818,103708,563,506,1463,529,506,8858,600,8784,3068,506,5148,236761]` | `The Willow is the name of the river that runs past the town.` |
| camelid Metal GPU-resident | **identical, 14/14** | identical |
| camelid runnable CPU | `[818,103708,7940,236761]` | `The Willow River.` |

Cross-engine prompt tokenization was identical (606/606) on both lanes, so the two camelid lanes
saw the byte-same 606 tokens; they diverge at **generated token index 2** (oracle `563`, runnable
`7940`) and never resynchronise — the runnable lane stops 10 tokens early with a different
sentence. `all_pass: false` for that leg, which is the expected and intended result.

This is the demonstration the campaign exists for, and it is stated as attribution, not as proof
of mechanism: the two camelid lanes are token-identical *below* the window (the in-src gates
`gemma3_real_row_resident_forward_matches_runnable_oracle`, 50/50 greedy tokens, and
`gemma3_session_level_token_by_token_prefill_matches_runnable_oracle`, 5/5, both re-run green at
this head), and the one documented architectural difference between them is the sliding-window
mask that the runnable lane does not implement. Past the window the runnable lane is therefore
the wrong reference, and this leg shows it behaving that way against a third-party oracle.

**And it is not a near-tie.** `probe-window-divergence.json` re-feeds the oracle the identical
606-token prompt plus the two shared generated tokens and reads its distribution at the
divergence position: the oracle's rank-1 is `563` " is" at logprob `-0.2125` — the token both the
oracle and the resident lane emit — and the runnable lane's `7940` " River" is rank 2, **1.667
nats** behind. Compare that with the largest disclosed near-tie in this bundle (0.447 nat): the
runnable lane's >=512 output is a substantive disagreement with the external oracle, not a soft
position.

Bounded by cost: the runnable lane prefills a 606-token prompt in ~10.8 min at ~0.2 tok/s, so
only the shortest of the three windowed prompts was run on it, at the single deepest leg.

## 4. Determinism — byte-identical across two fresh serve processes (`det-run1.json`, `det-run2.json`)

Two independent `camelid serve` processes (full stop/start between them, same command, no env
overrides), same prompts, greedy. Each session records:

- 6 chat legs (`/v1/chat/completions`, 24 tokens): the 5 gate prompts plus the 2403-token
  windowed prompt.
- 5 raw-completion legs (`/v1/completions`, 24 tokens) that carry the ACTUAL generated token
  ids, including one 2395-token windowed prompt.

`det-run1.json` and `det-run2.json` are **byte-identical files** (sha256
`632992c609941494905650a186ec255bf7d545950f4490aedc6a3c7158bf64d3` for both): every generated
token id and every chat string matches, at short depth and past the window.

## Near-tie disclosure (read before citing this bundle)

Three positions in the raw-decode legs are not token-identical. All three are in the depth-50
legs; every depth-1 and depth-5 leg of every harness is clean, and the >=512 windowed pack has
**zero** flips. Full data in `near-tie-analysis.json`, `camelid-raw-probe.json`,
`probe-oracle-{default,t4,t2,repack}.json` and `probe-oracle-continuous.json`.

Each flip was probed from both sides: camelid's own top-5 at the position, and the pinned oracle
re-fed the identical token prefix under four configurations (`--no-repack` with default threads,
`-t 4`, `-t 2`, and the runtime-repack kernel path), plus a continuous-decode control that
re-runs the oracle exactly as the capture ran it.

| prompt | index | reference token | camelid token | camelid top-2 gap | oracle top-2 gap (no-repack / repack) | oracle rank of camelid's token |
| --- | --- | --- | --- | --- | --- | --- |
| `The capital of France is` | 44 | `9639` " famous" | `32219` " charming" | **0.0032 nat** | 0.0431 / 0.0285 | **1** |
| `Once upon a time,` | 37 | `4658` " anything" | `11207` " broken" | **0.0173 nat** | 0.4471 / 0.1398 | 2 |
| `def fibonacci(n):` | 5 | `2094` "This" | `22304` "Calcul" | **0.0353 nat** | 0.4402 / 0.4402 | **1** |

Two of the three are **oracle-side flips**, which is the strongest attribution available: the same
pinned binary with the same flags emits camelid's token when the position is scored from a re-fed
prefix, and the reference token when it decodes continuously from the raw prompt
(`probe-oracle-continuous.json`). On `The capital of France is` the prefix-fed oracle also flips
back to the reference token when only its repack kernel changes. Those positions are not stable
on the oracle side, so no camelid-side defect is implied by them.

The third — `Once upon a time,` at index 37 — is the weakest attribution here and is disclosed as
such. The oracle is stable at rank 1 across all four kernel/thread controls AND the continuous
control, and camelid's token sits at oracle rank 2 with a **0.4471-nat** gap under the capture's
own `--no-repack` configuration. That is ABOVE the 0.33-nat Ornith soft-position line and above
the frozen runnable bundle's 0.3416-nat disclosure. What softens it, stated as measurement rather
than excuse: camelid's own view of the position puts the two tokens **0.0173 nat** apart, and the
oracle's own gap at that position moves from 0.4471 to 0.1398 nat when only its repack kernel
changes — a 0.31-nat swing inside one pinned build. It is disclosed here, in the manifest, and it
is why this bundle does not offer a token-exact depth-50 claim for raw `/v1/completions`.

Under the conductor's 7d envelope: the chat pack and the >=512 windowed pack are clean receipts
that do not draw on the envelope at all; the raw-decode depth-50 legs draw on it and are
individually adjudicated above.

## What this bundle does NOT claim

- **No performance claim of any kind**, and no speed comparison with llama.cpp. The oracle is
  used only as a correctness reference; it was deliberately run on its CPU backend for a stable
  reduction order, which makes any timing comparison against it meaningless.
- **No token-exact claim for raw `/v1/completions` at depth 50** — see section 2(b). Depths 1
  and 5 are clean on all four raw prompts; depth 50 carries three disclosed near-ties.
- **No context claim above 2,403 prompt tokens.** The model's native context is 32,768; nothing
  between 2,453 total tokens and that ceiling was measured here.
- **No claim for any other gemma3 row** — 4B/12B/27B stay out of scope (conductor 7c), as do
  other quants of the 1B.
- **No multi-turn, streaming, tool-calling, speculative-decode or prefix-cache claim.** Every
  leg is a single-user-turn non-streaming greedy request; tools fail closed on this row and the
  prompt-prefix cache stays closed for windowed archs.
- The runnable-lane comparison in 3a is a **single prompt at a single depth**, bounded by that
  lane's ~0.2 tok/s cost; it is a divergence demonstration, not a runnable-lane receipt.

## Reproduce

```
# Phase 1 — oracle only (pinned llama.cpp build, DYLD_LIBRARY_PATH set to its bin dir):
llama-server -m <models>/gemma-3-1b-it-Q8_0.gguf -c 4096 --port 8090 --host 127.0.0.1 \
  -ngl 0 -ctk f32 -ctv f32 -fa off --no-repack
node scripts/chat-parity-gemma3.mjs --mode capture --llama http://127.0.0.1:8090 \
  --oracle oracle-short.json --prompts-file qa/prompt-packs/gemma3-chat-gate-pack-v1.json
node scripts/chat-parity-gemma3.mjs --mode capture --llama http://127.0.0.1:8090 \
  --oracle oracle-windowed.json \
  --prompts-file qa/prompt-packs/gemma3-windowed-context-pack-v1.json
node scripts/raw-decode-parity.mjs --llama http://127.0.0.1:8090 \
  --reference-out raw-reference.json --token-counts 1,5,50 --stop 1,106

# ... stop llama-server ...

# Phase 2 — camelid only, resident lane is the default selection:
camelid serve --addr 127.0.0.1:8185 --no-open --model <models>/gemma-3-1b-it-Q8_0.gguf
node scripts/chat-parity-gemma3.mjs --mode compare --camelid http://127.0.0.1:8185 \
  --oracle oracle-short.json --model-id "Gemma 3 1b It" --row-id gemma_3_1b_it_q8_0 \
  --lane-label gemma3_marker_chat_greedy_metal_resident_serve --out <bundle>/gemma3-gpu-resident-chat-parity.json
node scripts/chat-parity-gemma3.mjs --mode compare --camelid http://127.0.0.1:8185 \
  --oracle oracle-windowed.json --model-id "Gemma 3 1b It" --row-id gemma_3_1b_it_q8_0 \
  --lane-label gemma3_marker_chat_greedy_metal_resident_serve --out <bundle>/gemma3-gpu-resident-windowed-parity.json
node scripts/raw-decode-parity.mjs --camelid http://127.0.0.1:8185 \
  --reference-in raw-reference.json --model-id "Gemma 3 1b It" --row-id gemma_3_1b_it_q8_0 \
  --token-counts 1,5,50 --stop 1,106 --variant q8_0_metal_gpu_resident \
  --out <bundle>/gemma3-gpu-resident-raw-decode-parity.json

# The divergence leg reuses the same oracle with the resident lane switched off:
CAMELID_METAL_RESIDENT_DECODE=0 camelid serve --addr 127.0.0.1:8185 --no-open \
  --model <models>/gemma-3-1b-it-Q8_0.gguf
```

## Environment

Apple M4 Mac mini, 10 cores, 16 GiB RAM, macOS 26.5 (build 25F5058e). Release build of head
`6eaf9053`. Both harnesses were run from that tree with three Phase 4 fixes applied in the same
commit series: `--row-id` now defaults to the real row id `gemma_3_1b_it_q8_0`, a `--lane-label`
flag records which served lane produced the camelid side, and `postJson` moved from the global
`fetch` to `node:http` (undici's ~5-minute header timeout aborted the client while the CPU
runnable lane was still legitimately prefilling a 606-token prompt). None of the three touches
engine behaviour.
