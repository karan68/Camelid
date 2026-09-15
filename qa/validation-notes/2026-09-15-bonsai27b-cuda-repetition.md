# Bonsai 27B Q1 Windows CUDA repetition

## Reproduction

The installed v0.7.4 engine on Windows, RTX 3060 Laptop 6 GB, driver
576.83, produced ` is: in Haber Haber Haber Haber Haber` for a fresh
`Can you code` chat at temperature 0 and max_tokens 8. The repeated token
was 214867. The prompt contained 15 tokens, with thinking disabled.

The GGUF was rehashed from disk and matched the certified artifact:
`17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.

Each comparison used a fresh engine process and the same file and prompt:

| Override | Result |
| --- | --- |
| None | Repeated Haber |
| `CAMELID_PRISM_CUDA_NO_GRAPH=1` | Identical bad token sequence |
| `CAMELID_PRISM_CUDA_NO_DEVICE_DECODE=1` | Four Haber tokens, then existing host-loop guard stopped |
| `CAMELID_CUDA_SAFE_EVENTS=1` | Identical bad token sequence |
| `CAMELID_PRISM_CUDA_NO_POPC=1` | Repeated special token 248050, no visible answer |
| `CAMELID_PRISM_CUDA_NO_Q1T128=1` | Repeated special token 248050, no visible answer |
| `CAMELID_PRISM_CUDA_STRICT=1` | `Yes, I can code! I can` |

A separate strict-path request, `Write a Python function add(a, b) that
returns their sum. Give only the code.`, completed naturally in 17 tokens:

```python
def add(a, b):
    return a + b
```

## Change and limits

Default the hash-identified Bonsai-27B Q1 artifact to strict f32 CUDA
contractions on Windows. Keep other artifacts/platforms unchanged and retain
explicit `CAMELID_PRISM_CUDA_STRICT=0` for diagnosing the fast path. This
contains a demonstrated fast-path correctness failure; it does not claim to
repair or requalify its underlying arithmetic. No new throughput claim is made.

Also apply the existing short-token-cycle guard to the CUDA text device loop,
which previously bypassed it. This limits runaway repetition independently
of the arithmetic policy; it cannot turn incorrect logits into a good answer.

The execution-plan prefill policy now uses a neutral packed-CUDA label rather
than claiming tensor-core prefill regardless of the selected arithmetic.

## Patched engine validation

The initial optimized Windows build (source base `04938343`, plus this fix), SHA256
`c178b65a43f583f78cabea8a8c46e26b023e9227cda2b06176e30d41a0935ecf`,
passed these checks with no arithmetic, graph, or device-input overrides:

- `Can you code`, max_tokens 32: coherent response beginning `Yes, I can code!`.
- Coding prompt above, max_tokens 32: complete correct function, natural stop.
- Repeat the first request in the same process: identical generated token IDs.
- Stream the coding request through SSE: content identical to the non-streaming response.

`CAMELID_RESIDENT_TRACE=1` recorded `Bonsai-27B Q1 fast_q1=false`.
CUDA graph replay remained enabled. Raw responses are retained in
[`bonsai27b-haber-windows-20260915`](../evidence-bundles/bonsai27b-haber-windows-20260915/).

Build command: `cargo build --release --bin camelid --features cuda -j 2
--config profile.release.lto=false --config profile.release.codegen-units=16`.
Frontend assets were built with `npm run build` before embedding.

Both `prism_cuda_fast` policy unit tests and
`short_exact_repetition_cycles_stop_but_normal_lists_do_not` passed in the
optimized Windows test binary. The installed engine's SHA256 was verified
against the tested build; after desktop restart, Bonsai reported loaded and
generation-ready.

## Integration onto v0.7.4 main

The fixes were rebased onto main `641af7ff` and rebuilt at `85337b80`.
The resulting optimized Windows engine, SHA256
`c402515ceda26df3f1fdd66373d7f02f77bffa93767c562b2a70f5f21956f76a`,
passed the same four requests: greeting, correct Python function with natural
stop, deterministic repeated greeting, and matching streamed function.
No arithmetic, graph, or device-input overrides were set; the trace again
recorded `fast_q1=false`. The `current-main-*` responses in the same bundle
preserve this rerun separately from the initial build. These are bounded
functional checks, not broad parity or performance qualification.

All 19 desktop tests passed after integration, as did `npm ci`, `npm run build`,
and formatting. The native close probe passed after readiness and without
waiting for readiness. Its main-window selection was updated to exclude the
newer hidden Spotlight window. The three lifecycle tests separately cover
shutdown racing a late startup and retry.

The subsequent v0.7.5 version bump changes package metadata only; release
packaging and the cross-platform standard gates are checked by CI.
