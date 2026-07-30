# Windows K-quant roadmap on Apple Metal

Implementation result for the five-item Windows-to-Metal roadmap. The
same-host receipt is accepted for Camelid's CLI/server appliance policy:
qualified Q4_K/Q6_K models use the Metal path automatically on Apple Silicon,
while library embedders retain conservative defaults.

## What landed

1. **Resident Q4_K/Q6_K decode.** Metal consumes GGUF wire weights directly,
   quantizes activations to the same Q8_K representation as the CPU/CUDA
   oracle, and dispatches Q4_K or Q6_K tiled kernels for attention, FFN, and
   the output projection. Token embedding uses format-specific GPU gather
   kernels. Mixed Q4_K_M models may switch format per tensor.
2. **Parity and performance gates.** The Q8_K quantizer is compiled separately
   with Metal fast math disabled so its scales and integer codes match the
   CPU/CUDA oracle. Unit tests cover both K-quant formats before an
   end-to-end benchmark is allowed to count.
3. **Metal-native K-quant prefill.** Batched rows use the same resident wire
   weights and a four-token tile, amortizing each weight tile across prompt
   rows. Existing Q8_0 simdgroup-matrix prefill remains unchanged.
4. **Compressed resident KV.** F16-primary and Q8_0-primary caches support
   prefill scatter and decode attention without maintaining an F32 primary
   copy. Q8 cache rows use 32-value blocks with a half scale and signed bytes.
5. **Continuous streaming batches.** The production engine can retain
   multiple streaming sessions and rotate them one token step at a time on
   its sole compute thread. A lone session retains single-request encode-ahead;
   contended rounds stop enqueueing future session-local graphs so one stream
   cannot head-of-line block another on the shared command queue.

## Rollout controls

| Variable | Values | Default | Effect |
|---|---|---:|---|
| `CAMELID_METAL_KQUANT` | `0`, `1` | `1` in the macOS CLI; library default `0` | Admit resident Q4_K/Q6_K weights and native K-quant prefill/decode. Unsupported mixes fall back. |
| `CAMELID_METAL_NOCOPY` | `0`, `1` | `1` in qualified macOS serve/bench runs | Read Q8_0/Q4_K/Q6_K weights into page-aligned storage which Metal wraps without a second upload. |
| `CAMELID_METAL_KV_DTYPE` | `f32`, `f16`, `q8` | `f16` for K-quant; `f32` otherwise | Select the resident KV primary representation. |
| `CAMELID_METAL_KV16` | `0`, `1` | `0` | Legacy alias for `CAMELID_METAL_KV_DTYPE=f16`. |
| `CAMELID_CONTINUOUS_BATCH_SLOTS` | `1..256` | `2` | Maximum active cooperative streaming sessions. Set `1` for legacy run-to-completion scheduling. |

`--deterministic` forces `CAMELID_METAL_KQUANT=0` and
`CAMELID_METAL_KV_DTYPE=f32` along with the rest of the GPU-off policy.

## Fail-closed boundaries

- Metal K-quant admission requires every resident dense projection to be
  Q8_0, Q4_K, or Q6_K with a valid aligned wire layout. Unsupported K-quants
  stay on their existing CPU/CUDA route.
- Q4_K/Q6_K input dimensions must be multiples of 256.
- Q8 KV requires a head dimension divisible by 32 and no larger than 128.
- Tree verification currently falls back when a compressed primary KV cache
  is selected; linear speculative verification supports compressed KV.
- Continuous batching applies only to streaming generation. Non-streaming
  work and management jobs remain exclusive.
- Q5_K/Q2_K/Q3_K/IQ4_XS mixes are not labeled or admitted as Metal K-quant;
  they keep their existing wire-only CPU/CUDA routes.

## Merge gate

Run these in a release profile on the same Mac and exact GGUF:

1. `metal_kquant_resident_projection_matches_cpu_oracles`
2. `metal_q8_primary_kv_scatter_and_attention_match_dequantized_reference`
3. `metal_attention_decode_splitk_kv16_matches_cpu_reference`
4. `cooperative_jobs_interleave_one_step_per_round`
5. the full library suite
6. a cold, greedy, median-of-five `bench-generate` before/after comparison

The benchmark receipt must record the commit, host, model path and hash,
prompt, environment, raw iterations, medians, and generated token IDs.
The merge receipt records the first generated-token divergence, if any.
F16-primary must pass the predeclared confident probe; Q8-primary is explicitly
lossy and is compared against the dequantized-Q8 oracle rather than claimed as
token-identical to F16/F32. A speed tie or regression is reported as such and
leaves the new path default-off. Continuous batching additionally requires a
live two-client streaming probe: both sessions must emit all requested tokens,
alternate after admission, and complete without a shared-queue stall.
