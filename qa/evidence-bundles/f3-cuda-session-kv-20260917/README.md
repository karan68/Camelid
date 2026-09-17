# F3 — CUDA session KV cache: live evidence

Captured 2026-09-17 on `feat/f3-cuda-session-kv` (build `v0.7.8-6-g73e2fdbc`).

Host: NVIDIA L4 (sm_89, 22 GiB), Ubuntu 22.04, release build, CUDA feature on.
Model: `Llama-3.2-3B-Instruct-Q4_K_M.gguf`.

Everything here came out of one `camelid serve` process answering real requests.
Nothing is mocked, replayed or hand-edited; the trace is the server's own
`CAMELID_RESIDENT_TRACE=1` output with absolute host paths scrubbed.

## screenshots/camelid-chat-session-kv-reuse.png

The shipped chat UI, driven through a headless browser against that server. The
third turn's telemetry row reads:

    tokens in 287 · out 231 · first token 509ms · 62 tok/s · 4.2s

287 prompt tokens answered in 509 ms. On this lane a cold prefill runs at about
14 ms/token, so prefilling those 287 tokens from scratch would cost ~4 s. The
matching trace line below shows why it did not: only 23 tokens were prefilled.

## resident-prefill-trace.txt

The same three turns, from the engine's side:

    reused  21 of  22 positions, prefilled  1   ->  14 ms
    reused  74 of  97 positions, prefilled 23   -> 302 ms
    reused 263 of 286 positions, prefilled 23   -> 310 ms

Each turn prefills only its new question. Prefill wall clock stays flat while the
conversation grows, which is the whole point of F3.

The first pair of lines in the file is deliberately included and is NOT a good
result:

    reused 178 of 693 positions, prefilled 515  -> 7361 ms

That request switched to a DIFFERENT conversation than the one the engine was
holding, so only the shared 178-token prefix matched. The device-side record
holds one conversation at a time (the spec's "one exact active session" scope),
so alternating between chats costs a re-prefill. Kept here because it is the
honest boundary of what this change buys.

## api-replay-timings.txt

The same shape through `/v1/chat/completions`, with no browser involved:

    turn 1: prompt_tokens=23  wall=1.09s
    turn 2: prompt_tokens=98  wall=1.63s
    turn 3: prompt_tokens=199 wall=1.63s
    turn 4: prompt_tokens=298 wall=1.62s

Wall clock is flat from turn 2 onward while the prompt grows 98 -> 298 tokens.

## What this evidence does NOT show

- It does not show the spec's "turn-10 TTFT within 20% of turn-1" gate being met.
  It is not: measured 1.59x on this host. The reason is recorded in the doc
  comment on `f3_turn_ten_prefills_only_the_new_question`, with the four prefill
  lanes that were A/B'd.
- It is a single host and a single model row. The correctness claims rest on the
  five `f3_*` gates in `src/inference/tests.rs`, not on these timings.
