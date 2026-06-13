# CLUSTER_BENCH.md — distributed parity lane, measured cost

This file records the latency cost of distributed inference **plainly and without spin**.
Distributed pipeline-parallel inference is *slower* than single-node would be when the model
fits — that is expected and is the point: the lane buys *capability* (a model spread across
nodes' RAM), not speed. The defensible result is token-identity to a single-node reference
with a receipt, never a throughput win.

## Phase 3 — two Macs over LAN (2026-06-13)

- Config: `two-mac-tinyllama-q8`, TinyLlama 1.1B Chat Q8_0
  (sha256 `a4c9bb1d…`, byte-identical on both nodes — verified).
- Topology: `mac-m4-coordinator` owns embedding + layers `[0,11)`; `mini2` (Mac mini M4,
  16 GB) owns layers `[11,22)` + output head + greedy sampling. One hidden-state vector
  (`[1,2048]` f32 = 8192 bytes) crosses the wire per token, one hop per token.
- Lane: CPU (resident paths disabled) on both nodes for deterministic, comparable math
  (DECISIONS D4). Build: the **same** `parity_node` binary on both nodes.
- Gate: token-identical to single-node `camelid` at 50 tokens, **two consecutive runs**.
  Result: **PASS** — both runs token-identical; sealed receipt
  `qa/distributed/two-mac-tinyllama-q8.json`, `receipt_id 33b79d8d…`,
  `first_divergent_generated_token_index = -1`. The receipt id is identical across runs
  (deterministic).

### Measured per-token cost (run 2, decode steps)

| Metric | Value |
|---|---|
| activation bytes / token (coord→worker) | 8192 |
| TTFT (prefill hop round-trip) | ~220 ms |
| avg coordinator-local compute / token (layers [0,11)) | ~72 ms |
| avg hop round-trip / token | ~111 ms |
| **distributed decode throughput** | **~5.5 tok/s** |

**Honest caveats (do not read these numbers as a speed claim):**
- The "hop round-trip" bundles the **worker's compute** (its 11 layers + output head, also
  ~70 ms on the CPU lane) **and** the wire. It is not pure network time. To isolate pure
  link latency, use `camelid bench-network` (Thunderbolt RTT measured here was ~0.6 ms; the
  Wi-Fi LAN is higher and variable).
- This is the **CPU lane** by construction (parity requires determinism). It is far slower
  than the production GPU-resident decode path; these numbers say nothing about camelid's
  normal single-node speed.
- Single-node would be faster than this two-node split for a model this size: one node runs
  all 22 layers with **no** per-token network hop. Distributing a 1.1B model is strictly a
  cost here — the demonstration is parity, not performance. The payoff (capability) only
  appears in Phase 4, where the model does not fit on one node.

### Operational notes (environment, not the engine)

- The Thunderbolt bridge (`169.254.0.0/16`) offers ~0.6 ms RTT but its link-local address
  reassigned mid-session; the gated runs used the stable LAN IP `192.168.86.50`.
- mini2's worker, launched over SSH, was initially throttled by macOS App Nap / Wi-Fi
  power-save between runs (TCP `accept()` stalled → connect timeouts). Running the worker
  under `caffeinate -dimsu` resolved it. The coordinator also uses bounded connect-retry +
  whole-run retry so a transport flake retries the run and never relaxes the parity gate
  (operating rule: fix the transport, never the threshold).

## Phase 4 — heterogeneous (Mac + Pi): partial (2026-06-13)

Setup: 3× Pi 5 (16 GB, aarch64 Linux) found and used; camelid built **natively on camelid1**
(`cargo build --release --example parity_node`, 2m10s, zero errors) — proving aarch64-Linux
portability. llama-2-13b + TinyLlama copied to the Pi (byte-identical).

**Result: cross-ARM parity NOT yet measured — the Pi worker crashes at model load.** Every
`parity_node worker --gguf …` on the Pi dies instantly (hard signal during load, before any
output; reproduces with the 1.2 GB TinyLlama, so it is not OOM). The same binary/source runs
on the Macs. See DECISIONS D7. The cross-ARM (Apple vs Cortex) numeric-parity verdict is
therefore **open**, not faked. Next: capture the crash on the Pi console / via a core dump to
localize the aarch64-Linux load-path bug, then re-run; if the Pis then introduce numeric
divergence, it will be documented here, not smoothed over.
