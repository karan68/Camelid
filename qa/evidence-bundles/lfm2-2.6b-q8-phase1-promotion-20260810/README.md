# LFM2.5 2.6B Q8_0 Phase 1 promotion

This bundle closes the Phase 1 qualification gates for one exact artifact:
`LiquidAI/LFM2.5-2.6B-GGUF@b421ad1d549afeda6a0fb2ad3a697cb5a7879adc`
`LFM2.5-2.6B-Q8_0.gguf` (2,874,779,456 bytes, SHA-256
`36587fdf27bdfc69caf2637273679a0870ec155162161bde6fd16e8c70bdb757`).

The Windows deterministic CPU lane passed all Phase 1 exact-row gates:

- the frozen tokenizer/template fixtures and four 24-token raw prompts pass;
  all 96 generated token IDs match llama.cpp b9632 (`acd79d603`);
- a native runnable-chat receipt contains exactly 512 rendered prompt tokens;
  all eight generated IDs and text match the same pinned llama.cpp CPU oracle
  with f32 K/V cache, flash attention off, and repacking off;
- the API/Models-page smoke reports the exact filename, size, SHA, and
  `lane_class=supported`, and resolves compatibility row
  `lfm2_5_2_6b_q8_0` as `supported_exact_row_smoke`;
- non-streaming runnable chat passes; legacy `/v1/completions` remains
  intentionally refused for this runnable-only architecture;
- streaming chat ran with a 128-token ceiling, stopped naturally after 43
  completion tokens, and emitted one finish frame, one terminal usage frame,
  then `[DONE]`, with visible answer text;
- runnable responses do not expose the dense timing-diagnostics object, so the
  timing-only summarizer is explicitly recorded as skipped and is not treated
  as a support gate.

The runtime binary reports source head `15ab4ddac5a21ac0f98f6ccba5e3ddff4b51a9b5`.
The final promotion harness ran from clean head
`2cd9fa78f8dcaa78e011914076ecd1473ae99d0a`; the two later commits only make
the generic promotion harness handle missing runnable timing diagnostics and
explicitly replace an auto-selected resident model.

This is exact-row smoke support, not broad LFM2 support. Tools remain typed
fail-closed; sampling beyond the deterministic greedy lane, context above the
checked 512-token bucket, neighboring sizes/quants, production throughput,
CUDA, and a resident Apple-Silicon Metal rerun remain unclaimed. The Mac handoff
is documented in `MAC_HANDOFF.md`.
