# Phi-3 hold evidence

Why `phi3_mini_4k_instruct_q8_0` is NOT advertised as supported.

## Cleared 2026-07-27

Prompt-token parity now PASSES on the pinned artifact `Phi-3-mini-4k-instruct-Q8_0.gguf`:

- `prompt-token-parity-q8-20260727.json` - 8/8 raw prompts, `all_match=true`
- `chat-prompt-token-parity-q8-20260727.json` - 3/3 rendered chat prompts, `all_match=true`

This closed the tokenizer half of the hold. Three defects were fixed: no rstrip of the
whitespace following `<|...|>` markers; the SPM dummy prefix emitted as a standalone token
id instead of prepended as a character for SPM to merge; and raw text bypassing the SPM
algorithm entirely for a longest-match encoder. The superseded pre-fix captures
(`prompt-token-parity.json`, `phi3-chat-parity.json`) are retained for history.

## STILL BLOCKING

`generation-divergence-q8-20260727.json` - the forward pass diverges from the reference.
On the 9-token prefix `The capital of France is Paris.\n` the reference is ~99.1%
confident of `<|assistant|>` (32001) while camelid ranks it -4.944 and picks `===`.
camelid also disagrees with ITSELF between fresh prefill and incremental decode at that
position. Tokenizer, RoPE pairing, padded vocab, sampling policy and detokenization are
all ruled out in that file; the attention / KV-cache path is the open suspect.

Generation parity, the bounded-context packs, and API/WebUI evidence remain outstanding.
The row stays `active_validation_blocked_parity` and fail-closed in the frontend.
