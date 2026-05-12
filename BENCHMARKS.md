# Camelid Benchmarks

Last updated: 2026-05-12

This file is Camelid's public performance snapshot.

It is intentionally narrower than a marketing benchmark page:

- it reports only sanitized, reproducible numbers already backed by committed evidence bundles
- it separates **runtime measurements** from **support claims**
- it does **not** treat one good number as broad model-family proof

If a row or host is not listed here, Camelid is not claiming a benchmark result for it yet.

## Reading rules

- **Exact row only.** Numbers apply only to the exact GGUF row named in the table.
- **Bounded workload only.** These are short, bounded validation or microbenchmark lanes, not production multi-user load tests.
- **Same-host comparison only.** A fair Camelid-vs-llama.cpp throughput claim requires the same prompt shape, same model file, same host, same thread settings, and the same token budget.
- **Parity first.** Camelid treats 1:1 token parity with llama.cpp as the prerequisite for performance claims, not a substitute for them.

## Benchmark snapshot: current committed numbers

### Ubuntu bounded unique-chat envelope

These results come from `qa/evidence-bundles/llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/manifest.json`.

Method summary:

- endpoint: `/v1/chat/completions`
- warmup: 2
- measured repeats: 4
- max tokens: 5
- prompt style: unique prompts, average prompt token count `22.5`
- weight cache: hot on measured runs
- prompt cache: no hits on measured runs

| Exact row | Avg wall ms | Avg generate ms | Approx ms / output token | Max backend RSS MiB | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Llama 3.2 1B Instruct Q8_0 | 7379.73 | 7065.25 | 1413.05 | 274.31 | Exact-row bounded unique-chat envelope only |
| Llama 3.2 3B Instruct Q8_0 | 19762.21 | 19449.25 | 3889.85 | 287.21 | Exact-row bounded unique-chat envelope only |

### Ubuntu first-token direction probe

This result comes from `qa/evidence-bundles/llama32-3b-parallel-q8-first-token-20260505T140400Z-head-ffc22b85214f/manifest.json`.

Method summary:

- endpoint: `/v1/chat/completions`
- warmup: 1
- measured repeats: 1
- max tokens: 1
- comparison: default serial file-backed Q8 path vs opt-in parallel output-row partitioning

| Exact row | Mode | Avg generate ms | Avg generate ms / prompt token | Max backend RSS MiB | Delta |
| --- | --- | ---: | ---: | ---: | --- |
| Llama 3.2 3B Instruct Q8_0 | serial Q8 path | 13960 | 775.56 | 283.57 | baseline |
| Llama 3.2 3B Instruct Q8_0 | opt-in parallel Q8 path | 12200 | 677.78 | 282.97 | `-12.61%` generate time |

This closes only the exact 3B **first-token direction** sub-box. It is not a broad production-throughput claim.

### Ubuntu compact smoke timing snapshot

These results come from `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json`.

Method summary:

- compact smoke / API + WebUI oriented validation lane
- `hello` message, `max_tokens=1`
- same sanitized Ubuntu validation host

| Exact row | Model load ms | Chat completion ms | Backend generate ms | Max RSS KiB |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama 1.1B Chat Q8_0 | 134 | 40558 | 40487 | 136588 |
| Llama 3.2 1B Instruct Q8_0 | 559 | 20973 | 20674 | 347316 |
| Llama 3.2 3B Instruct Q8_0 | 566 | 45138 | 44830 | 559572 |
| Llama 3 8B Instruct Q8_0 | 566 | 81086 | 80794 | 566980 |

These are useful for release-audit snapshots, not for broad “fastest local runtime” marketing.

## Llama.cpp comparison policy

Camelid already publishes bounded **parity** against llama.cpp for the exact supported rows, but the repo does **not yet** publish a fully normalized same-host throughput table against llama.cpp for those rows.

That means Camelid should say this plainly today:

- **Yes:** Camelid has exact-row 1:1 bounded parity with llama.cpp where cited.
- **Not yet:** Camelid has not published a repo-safe apples-to-apples throughput table versus llama.cpp on the same host for every headline row.

That missing table is an execution gap, not a copywriting gap.

## What should be added next

The next benchmark slice worth publishing is:

1. **Same-host Camelid vs llama.cpp on the exact 3B row**
   - same model SHA
   - same prompt pack
   - same token budget
   - same thread settings
   - report TTFT, decode tok/s, and ms/token
2. **Mac benchmark snapshot**
   - especially the exact Llama 3.2 3B Instruct Q8_0 row that Tim is actively feeling in the UI
3. **8B bounded same-host comparison**
   - only after the same-host run is reproducible and repo-safe

## Claim boundary

This file is a benchmark snapshot, not a support matrix.

It does not widen any support claim beyond `COMPATIBILITY.md`, `STATUS.md`, and the cited evidence bundles.