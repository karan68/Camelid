# PrismML Bonsai Windows CUDA receipt

This bundle records Windows x86_64 CUDA validation for the same seven exact, hash-pinned Bonsai model artifacts covered by the macOS Metal receipt. The comparison reference for parity checks is the Windows CUDA `llama-server.exe` pinned by `PrismML-Eng/Bonsai-demo` (`9fcaed7`), not an unrelated upstream build. The implementation measured here is source commit `f297857f`.

The important capacity result is the 27B Q2 row on a 6 GiB RTX 3060 Laptop GPU. An all-resident upload correctly failed with `CUDA_ERROR_OUT_OF_MEMORY`; the capacity plan keeps recurrent mixers on CUDA and streams the FFN/full-attention weights of 36 trailing layers from pinned host RAM. Its cold graph build completed in 13.1 seconds, its checked eight-token text prefix matched the vendor-validated 27B output, and a real PNG completed through `/v1/chat/completions` with `vision_ready=true`.

The 4B Q1 row is the strongest cross-engine token receipt. The 4B Q2 row matches every 1- and 5-token leg and three of four 50-token legs; the remaining leg diverges late at token 46 and is retained as a disclosed low-bit numerical frontier. The 27B text checks are token-exact. Vision is a bounded functional/semantic smoke: prompt-token counts align at the 128-token image cap and both engines describe the image coherently, but generated visual wording is not claimed token-exact.

The image picker accepts one local PNG or JPEG, sends it as an OpenAI-compatible `image_url` data part, and remains hidden or blocked unless the active 27B runtime reports `vision_ready=true`. Multiple or remote images and other multimodal types still fail closed.

## Final 27B Q1 CUDA decode A/B

The final same-host A/B isolates the artifact-gated POPC Q1 decode route. Each configuration ran the identical 142-prompt-token frontier-image request three times, emitted 24 completion tokens, and produced the same generated text in all six runs. The checked-in receipts preserve the repository-relative image path `target/reference/Bonsai-demo/assets/frontier.png`.

| Configuration | Mean image TTFT | Mean post-first decode time | Mean decode rate |
| --- | ---: | ---: | ---: |
| POPC enabled (default) | 2888.223 ms | 1075.742 ms | 21.3805924341 tok/s |
| POPC disabled (`CAMELID_PRISM_CUDA_NO_POPC=1`) | 2844.230 ms | 1396.996 ms | 16.4639010553 tok/s |

At the receipt's four-decimal rate display precision, the default route improves post-first decode throughput by 29.864%; the ratio of the full-precision means is 29.863465%. It reduces the 23-token post-first interval by 22.996%. Image TTFT is a separate measurement that includes image projection, prompt prefill, and production of the first token; it was 1.547% slower in this A/B and is not presented as a POPC speedup. End-to-end latency fell by 6.537% (3963.966 ms versus 4241.226 ms).

`prism-popc-default.json` and `prism-popc-disabled.json` are the unabridged three-run benchmark receipts. `bonsai-27b-q1-cuda-performance.json` is their compact comparison. This receipt makes an observed output-identity claim for these six requests; it does not claim that every internal kernel or intermediate logit is bit-exact to a legacy serial or BMMA implementation.

See `manifest.json` for exact model hashes, hardware, evidence files, and non-claims. `windows-exact-row-smoke.json` records the deterministic seven-row matrix and the constrained-VRAM 27B Q2 capacity/image run. `SHA256SUMS` covers every bundle file except itself.
