# LFM2.5 2.6B Q8_0 macOS resident-Metal qualification

This bundle records the completed Apple M4 handoff for one exact artifact:
`LiquidAI/LFM2.5-2.6B-GGUF@b421ad1d549afeda6a0fb2ad3a697cb5a7879adc`
`LFM2.5-2.6B-Q8_0.gguf` (2,874,779,456 bytes, SHA-256
`36587fdf27bdfc69caf2637273679a0870ec155162161bde6fd16e8c70bdb757`).

Every qualification artifact in this bundle was refreshed from clean source
head `d31e5cb02bd3633407cbd3e6e89aeac5bacde617`. The earlier handoff anchor was
`35ca855f22658db2a6b2d15a7d0194b2cc00384c`; the manifest retains that lineage
while the bundle name and runtime identity point to the final qualified head.

The checked result on one Mac16,10 with Apple M4, 16 GiB unified memory, macOS
26.5 build 25F5058e, and arm64 is:

- four deterministic short prompts matched pinned llama.cpp b9632
  (`acd79d603`) for all 96 generated token IDs;
- a chat receipt contained exactly 512 rendered prompt tokens and matched the
  same CPU oracle for all eight generated token IDs and text;
- the final execution plan reported `LFM2.5-2.6B-Q8_0.gguf`,
  `supported_exact_row_smoke`, `metal_resident_lfm2_runtime`,
  `lfm2_metal_resident_prefill`, and `lfm2_metal_resident_decode`;
- the API/Models-page smoke reported the exact filename, byte size, SHA-256,
  `lane_class=supported`, and exactly one active model with catalog ID
  `lfm2_5_2_6b_q8_0`;
- non-streaming chat returned eight generated tokens; the frontend smoke
  returned HTTP 200 and its 128-token SSE request stopped naturally after 43
  completion tokens with finish, terminal usage, and `[DONE]` events;
- the final backend launch did not set `CAMELID_SKIP_FIT_CHECK`; reloading the
  already-resident exact artifact under its catalog ID succeeded as an
  idempotent load.

`context/parity-receipt.json` is the captured Metal result, while
`context/verify.log` records the independent pinned-oracle rerun. Verification
used `--reference-only`, so the verifier correctly says "partially verified":
the receipt digest, exact lane identity, and llama.cpp rerun were checked, while
a second Camelid self-replay was intentionally skipped.

`api-webui/compatibility-row.json` is an explicit post-qualification projection
of the finalized `/api/capabilities` row, not a raw runtime response. It carries
the final host- and execution-lane-scoped contract while the adjacent health and
execution-plan files preserve the dynamic d31e5cb0 runtime capture. The
projection declares and omits `latest_checked_output`: the runtime response
necessarily preceded publication of this durable bundle filename, while the
source contract now points at this bundle.

Legacy `/v1/completions` was intentionally skipped because LFM2 is qualified
on the runnable chat surface. Timing summarization was also intentionally
skipped because this runnable response does not emit dense-engine timing
diagnostics; neither skip is treated as a support gate.

This is not broad LFM2 or broad Apple support. The claim is limited to this
exact hash-pinned Q8_0 file, this Apple M4/macOS 26.5 arm64 host, deterministic
greedy generation, the exact 512-token chat bucket, and the recorded
API/Models-page/WebUI smoke. It does not claim other Apple hardware, Linux or
CUDA portability, neighboring files or quants, context above 512, non-greedy
sampling, tools, raw completions, production throughput, or an SLA.
