# Mac handoff

Start from branch `codex/huggingface-model-catchup`. The Windows CPU proof is
complete for the exact file below; the Mac task is a portability/resident-Metal
follow-up, not a prerequisite for the Windows exact-row smoke claim.

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
     --expect-gguf-sha256 36587fdf27bdfc69caf2637273679a0870ec155162161bde6fd16e8c70bdb757
   ```

Do not widen the result to other LFM2 files, other quants, context above 512,
sampling, tools, or a throughput claim. If the Mac selects a resident Metal
lane, its token IDs/text still need their own pinned-oracle receipt before a
Metal-specific parity claim is added.
