# Mac handoff — completed

The handoff from branch `codex/huggingface-model-catchup` is complete for the
exact file below on one Apple M4 host running macOS 26.5 arm64. The independently
committed receipt is
`qa/evidence-bundles/lfm2-2.6b-q8-macos-metal-20260810-head-35ca855f/`.

That receipt records 96/96 short greedy generated IDs and an exact
512-rendered-prompt-token plus 8-generated-ID/text match against pinned llama.cpp b9632
(`acd79d603`). `/v1/health` asserts
`selected_backend=metal_resident_lfm2_runtime`,
`prefill_path=lfm2_metal_resident_prefill`, and
`decode_path=lfm2_metal_resident_decode`. The exact-row API/Models-page/WebUI
non-streaming smoke and SSE smoke with a 128-token ceiling also pass.

The original Windows x86_64 CPU/runnable promotion in this bundle remains an
independent supported lane; the M4 result neither replaces nor weakens it. The
recipe used to close the handoff is retained below for reproducibility.

1. Acquire the immutable artifact if it is not already present, then verify it
   before loading (the download is about 2.87 GB):

   ```sh
   mkdir -p models
   curl --fail --location \
     "https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/b421ad1d549afeda6a0fb2ad3a697cb5a7879adc/LFM2.5-2.6B-Q8_0.gguf?download=true" \
     --output models/LFM2.5-2.6B-Q8_0.gguf
   shasum -a 256 models/LFM2.5-2.6B-Q8_0.gguf
   # 36587fdf27bdfc69caf2637273679a0870ec155162161bde6fd16e8c70bdb757
   ```

2. Build the checked-out branch and run the frozen short gate:

   ```sh
   cargo test --release --test lfm2_parity --locked -- --nocapture
   ```

3. Repeat the exact 512-token reference gate with the pinned llama.cpp b9632
   (`acd79d603`) binary and one engine resident at a time. The established Mac
   pin may be substituted for the example path below; do not use an unpinned
   llama.cpp build:

   ```sh
   node qa/capability/context_parity.mjs \
     --request-mode chat \
     --gguf models/LFM2.5-2.6B-Q8_0.gguf \
     --row lfm2_5_2_6b_q8_0 \
     --label "LFM2.5 2.6B Q8_0" \
     --target-tokens 681 \
     --expect-actual-tokens 512 \
     --kv-bytes-per-token 32768 \
     --llama-ctx 640 \
     --max-gen 8 \
     --verify-mode reference-only \
     --camelid-lane metal \
     --expect-selected-backend metal_resident_lfm2_runtime \
     --expect-prefill-path lfm2_metal_resident_prefill \
     --expect-decode-path lfm2_metal_resident_decode \
     --camelid-exe target/release/camelid \
     --llama-server target/reference/llama.cpp-b9632/bin/llama-server \
     --camelid-port 8231 \
     --llama-port 8233 \
     --out target/model-qualification/mac/lfm2-context-512
   ```

4. Run the promotion smoke against a fresh backend/frontend pair. Use the same
   exact SHA assertions and `--replace-loaded-model`; on Apple Silicon record
   the selected execution lane from `/api/models/current` and `/v1/health`:

   ```sh
   node scripts/model-promotion-smoke-bundle.mjs \
     --api http://127.0.0.1:8251 \
     --frontend http://127.0.0.1:4175 \
     --model models/LFM2.5-2.6B-Q8_0.gguf \
     --model-id lfm2_5_2_6b_q8_0 \
     --out-dir target/model-qualification/mac/lfm2-api-webui \
     --chat-only \
     --replace-loaded-model \
     --max-tokens 8 \
     --stream-max-tokens 128 \
     --expect-compatibility-row lfm2_5_2_6b_q8_0 \
     --expect-compatibility-status supported_exact_row_smoke \
     --expect-contract-supported true \
     --expect-webui-chat enabled \
     --expect-local-lane-class supported \
     --expect-selected-backend metal_resident_lfm2_runtime \
     --expect-prefill-path lfm2_metal_resident_prefill \
     --expect-decode-path lfm2_metal_resident_decode \
     --expect-gguf-sha256 36587fdf27bdfc69caf2637273679a0870ec155162161bde6fd16e8c70bdb757
   ```

The completed Mac claim is limited to this exact GGUF, Apple M4, macOS 26.5,
arm64, the asserted resident-Metal execution plan, deterministic greedy parity,
the exact 512-token prompt bucket, and the recorded chat/API/WebUI smoke. Do not
widen it to other Apple hardware, broader platform portability, other LFM2
files or quants, context above 512, non-greedy sampling, tools, CUDA, or
throughput.
