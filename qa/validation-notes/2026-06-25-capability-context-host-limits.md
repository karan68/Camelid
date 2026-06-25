# Capability conductor — `context.full_length` host limits + a KV predict-and-abort gap (Windows CPU)

Date: 2026-06-25 · Platform: Windows x86_64 (MSVC), CPU backend · Oracle: llama.cpp `acd79d6`
Scope: the `context.full_length` and `context.rope_scaling` lanes of `MODEL_CAPABILITY_COVERAGE_CONDUCTOR.md`, validated on this development host.

This note records why two `context.full_length` cells (Llama 3.2 1B and 3B) are **NOT** promoted to `done` on this box, and surfaces a real safety gap. It is the honest counterpart to the receipts that *did* land. Per conductor §9 (predict-and-abort) and the runnable-lane memory policy, a length the model genuinely supports but that this host cannot materialize is a **host limit**, not a model limit or a refusal — and it must be documented, never discovered by crashing.

## Host facts

- Total RAM: **15.74 GiB** (16,905,969,664 bytes); ~7 GiB free at measurement.
- Resident Q8_0 weights (default lazy file-backed): TinyLlama ~1.17 GB, Llama 3.2 1B ~1.32 GB, Llama 3.2 3B ~3.42 GB.
- CPU KV cache is **f32** (`src/inference/kv_cache.rs`), shape `[layers, seq, kv_heads, head_dim]` for each of K and V. Per-token bytes = `2 · n_layers · n_kv_heads · head_dim · 4`.

| Row | n_layers | n_kv_heads | head_dim | KV bytes/token | trained ctx | KV at trained ctx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyLlama 1.1B | 22 | 4 | 64 | 45,056 (44 KiB) | 2048 | **88 MiB** ✅ |
| Llama 3.2 1B | 16 | 8 | 64 | 65,536 (64 KiB) | 131072 | **8.0 GiB** ⚠ no headroom |
| Llama 3.2 3B | 28 | 8 | 128 | 229,376 (224 KiB) | 131072 | **28.0 GiB** ❌ infeasible |

(The 3B figure corrects an in-flight reader error that had claimed 1.75 MiB/token; the `inspect`-measured head counts give 224 KiB/token, confirmed by recompute. This is load-bearing: it is why 3B rope-scaling at >8192 positions IS feasible here, while 3B *full* 131072 is not.)

## What was validated (class D, bit-exact vs llama.cpp `acd79d6`)

Harness `qa/capability/context_parity.mjs`: a long prompt is decoded greedily; Camelid emits a server-sealed `camelid.parity-receipt/v1`; `camelid verify-receipt` feeds Camelid's exact prompt token ids to llama.cpp `/completion` and asserts the continuation matches bit-exact (prompt PINNED → proves KV + rope correctness across the whole context, token-for-token). The harness itself projects KV bytes and **aborts before requesting an unsafe context** — the host is never discovered by crashing it.

- **TinyLlama `context.full_length` → done.** Bit-exact at **1953 tokens (95% of the trained 2048)**; full-length KV ≈ 88 MiB, fully reachable. Receipt: `capability-receipt.context.full_length.tinyllama-1.1b-chat-q8_0.json`.
- **Llama 3.2 1B `context.rope_scaling` → done.** Bit-exact at **8511 tokens (> 8192, the llama3-scaled regime; full self+reference verify)**. Receipt: `capability-receipt.context.rope_scaling.llama-3.2-1b-instruct-q8_0.json`.
- **Llama 3.2 3B `context.rope_scaling` → done.** Bit-exact at **8304 tokens (> 8192; reference-only verify — 1.9 GiB KV)**. Receipt: `capability-receipt.context.rope_scaling.llama-3.2-3b-instruct-q8_0.json`.

## What stays `wip` (host-limited, NOT a model limit)

- **Llama 3.2 1B `context.full_length`** — validated bit-exact across a long context (feasible frontier, **8511 tokens**; also an earlier 7958-token run), but the **trained 131072** materializes an 8.0 GiB f32 KV cache, which fits total RAM only with no safe headroom (and exceeds free memory). Not validated at the trained length on this box. Needs a larger-RAM host.
- **Llama 3.2 3B `context.full_length`** — validated bit-exact across a long context (feasible frontier, **8304 tokens**), but the **trained 131072** needs a **28 GiB** f32 KV cache — ~12 GiB beyond this box's entire RAM. Decisively infeasible here.

These two cells remain `wip` with this host-limit annotation. They are NOT `n/a` (the models genuinely support 131072) and NOT `done` (unvalidated at the trained length here).

## Safety gap surfaced (conductor §9)

Camelid has **no pre-flight KV predict-and-abort**. The context cap is the GGUF trained length (`model.rs:141` → `kv_cache.rs:23`), but the KV cache grows **lazily/incrementally** (`kv_cache.rs:135-136`, `keys.resize`/`values.resize`, chunks of `CAMELID_KV_CACHE_GROW_TOKENS`=256). A request approaching 131072 on a host that cannot hold the projected KV would **OOM mid-generation**, not fail closed. The existing weight-materialization guard (`CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES`, default 6 GiB, `api/mod.rs:6533`) covers **weights only** and estimates ≈0 for lazy-Q8 file-backed linears, so it never fires for KV.

This is exactly the rail conductor §9 calls load-bearing on Windows ("never discover the ceiling by crashing the host"). The `context_parity.mjs` harness models the missing guard externally (projects KV, aborts before request). A clean in-engine fix would project `requested_positions · kv_bytes_per_token` against an available-memory budget at request admission / KV-plan time and return a typed error before the first oversized `resize`. **Proposed as a follow-up; not implemented in this lane** (it touches the core request/KV path and warrants its own validation + gate run).
