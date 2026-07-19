<div align="center">

# 🐪 Camelid

**A Rust-native local LLM inference engine — GGUF in, OpenAI-style API out, every claim backed by reproducible evidence.**

[![CI][ci-badge]][ci-workflow]
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Language: Rust](https://img.shields.io/badge/language-Rust-orange.svg)
![Platform: macOS · Linux · Windows](https://img.shields.io/badge/platform-macOS%20·%20Linux%20·%20Windows-lightgrey.svg)

</div>

Camelid loads GGUF models directly, serves them over a local OpenAI-style API, and gates every optimized path on token-for-token parity with a reference implementation. It is **not** a wrapper around Ollama or llama.cpp — the tokenizer, GGUF loader, CPU kernels, and the Metal (Apple Silicon) and CUDA (NVIDIA) GPU paths are all implemented in this repository, shipping as a single static Rust binary with no Python.

![Camelid WebUI chat surface](docs/assets/camelid-readme-chat-surface-dark.png)

<div align="center"><sub>The local web frontend — a dark, collapsed-rail chat surface that unlocks chat only for model rows the compatibility contract recognizes.</sub></div>

---

## Install

**Two ways to run Camelid — both use the same engine and the same models. Pick what fits:**

| | 🪟 Camelid Desktop | ⚙️ Camelid engine |
|---|---|---|
| **What it is** | A native Windows app | The prebuilt `camelid` binary |
| **Best for** | Just chatting on your own PC — the easy button | Sharing on a network, the API, scripting |
| **How you chat** | A native window (no browser, no terminal) | In your **browser**, or a **server** others connect to |
| **Install** | Double-click the signed installer | Unzip and run `camelid.exe` |
| **Runs on** | Windows | Windows · macOS · Linux |

The desktop app simply wraps the same engine in a native window — same models, same support gate, same GPU acceleration. **If you just want to chat, get the desktop app. If you want to share it or use the API, get the engine.**

### 🪟 Camelid Desktop (Windows) — easiest

The simplest way to run a model locally: a native app, no browser tab, no command line.

![Camelid Desktop — native Windows app](docs/assets/camelid-desktop-window.png)

<div align="center"><sub>Camelid Desktop on Windows — the same chat surface in a native window, with the supported Llama 3.2 3B row loaded and ready.</sub></div>

1. Download the **Camelid Desktop installer** (`Camelid.Desktop_<version>_x64-setup.exe`) from the [latest release](https://github.com/timtoole02/Camelid/releases/latest).
2. Double-click it and follow the prompts. It's **code-signed** (verified publisher); if Windows SmartScreen warns on a fresh download, click *More info → Run anyway*.
3. Launch **Camelid Desktop** from the Start menu. It starts the engine for you and opens the chat window — pick a model to download, and chat.

Installs just for you under `%LOCALAPPDATA%\Camelid Desktop` — **no admin rights needed**. GPU acceleration works on any NVIDIA card with only the normal driver (the CUDA runtime is bundled); no GPU, it runs on the CPU.

### ⚙️ Camelid engine — run it in a browser, or serve it to share

The prebuilt binary, with the web UI baked in. Run it to chat in your **browser**, or serve it so **other people or apps** can connect over an OpenAI-style API — on Windows, macOS, or Linux. Get it from the [latest release](https://github.com/timtoole02/Camelid/releases/latest).

**Windows (x86_64):**

1. Download **`camelid-windows-x64.zip`** and right-click → **Extract All…** to a folder (your Desktop is fine).
2. Run **`.\camelid.exe serve`** in a terminal (or double-click `camelid.exe`).

The chat UI opens automatically at <http://127.0.0.1:8181> in your default browser. The binary is **Authenticode code-signed**, and GPU acceleration works out of the box on any NVIDIA card (normal driver only — the CUDA runtime is bundled). No Python, Node, Docker, or CUDA Toolkit to install.

**macOS (Apple Silicon) / Linux (x86_64):**

```bash
# macOS (Apple Silicon)
curl -L https://github.com/timtoole02/Camelid/releases/latest/download/camelid-macos-arm64.tar.gz | tar -xz
cd camelid-macos-arm64
xattr -d com.apple.quarantine ./camelid 2>/dev/null || true   # allow the unsigned binary to run

# Linux (x86_64): same, with camelid-linux-x86_64.tar.gz
```

**Then: download a model and chat (any OS):**

```bash
./camelid pull llama32_3b      # download a supported model into ./models
./camelid serve --model models/Llama-3.2-3B-Instruct-Q8_0.gguf
```

`serve` runs one static binary serving the OpenAI-style API and the web UI on the same port. **Sharing on a network?** `camelid serve --addr 0.0.0.0:8181` lets anyone on your LAN open the same chat UI and API. Prefer to build from source? See [Build from source](#quickstart).

---

## Why Camelid is different

Most local runtimes optimize for breadth — "point it at any GGUF." Camelid optimizes for **trust**, and treats the boundary as the feature:

- **Every claim is backed by a re-runnable receipt.** Support is per *exact* model row — a specific GGUF at a specific quant — and an optimized path ships only after it matches a reference token-for-token. No "same family, probably fine."
- **It fails closed, on purpose.** Point it at an unsupported model and you get a typed error, not a silent wrong answer. The honest boundary *is* the product.
- **One Rust binary, no Python.** Tokenizer, GGUF loader, CPU kernels, and the Metal and CUDA GPU paths all live in this repo and ship as a single static binary — `serve` even embeds the web UI.
- **Numbers come with logs or they don't ship.** Every published benchmark links to a committed bundle with raw logs, exact commands, and versions. No raw log, no claim.

---

## More ways to run

You already have the web chat UI from `serve` (see [Install](#install) above). Camelid also runs **in your terminal** and as a **sandboxed agent** — covered just below.

Want a single command that proves the whole path end to end? `scripts/smoke.sh` pulls TinyLlama, serves it, does one real chat round-trip, and asserts on the reply — no mocks. See [Quickstart](#quickstart).

### Run in the terminal

Prefer the keyboard? `camelid chat` is a full-screen terminal app — Markdown-rendered replies that stream in live, a scrollable chat pane, a settings sidebar with a context gauge, a `/` command palette, and instant switching between models already loaded in the server. It attaches to a running `camelid serve` or spawns one for you. (Over a pipe, SSH without a TTY, or with `--plain`, it falls back to a scrollback-friendly line REPL.)

```bash
./camelid pull tinyllama        # the baseline supported row (or any pull alias)
./camelid chat                  # full-screen TUI; opens the model browser, or:
./camelid chat --model models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
```

Type **`/`** to open the command palette and browse everything (filter as you type, **↑↓** to pick, **Tab**/**Enter** to run). Highlights: **`/models`** browses loaded + downloadable models, **`/switch`** flips instantly between models already loaded in the server (no reload), **`/set <temperature|top_p|top_k|max_tokens|seed|stream> <value>`** tunes sampling live, **`/system`** sets a prompt, **`/save`/`/load`** persist a session, **`/copy`** yanks the last reply to the clipboard, **`/theme`** restyles, **`/retry`** regenerates. **Tab** toggles the sidebar, **PgUp/PgDn** and the wheel scroll, **Ctrl-C** stops a stream, **Ctrl-D** quits (**F1** for the full key/command list). The model browser is built from the live support ledger (`/api/capabilities`) — it lists only **supported** rows and shows which are already downloaded. Pointing `--model` at a GGUF whose architecture Camelid doesn't support is refused with the same typed error the rest of the engine uses — the terminal is not a backdoor around the support contract. Gemma 4 12B/26B remain **two-Mac distributed only** and are not single-node chat rows.

### Agent mode (preview)

`camelid chat --agent --model <gguf>` runs a **sandboxed tool-calling loop**: the model reads/writes/searches files, runs shell commands, and (opt-in) fetches URLs, observing each result and iterating toward your goal — every write/exec/network action behind an **approval prompt** (`y` once · `a` this tool for the session · `n` deny · `q` abort). File tools are confined to a canonical workspace root (`--workdir`, default the current directory); path escapes (`..`, outside-absolute, escaping symlinks) are refused in code, not just discouraged. Tool results are treated as **untrusted data** — an instruction hidden in a file or web page can never make the agent escalate or run a prohibited action. The network tool is **off unless `--allow-net`**; `--auto-approve` exists for power users but warns loudly and still enforces the sandbox.

**Requires a tool-capable supported row.** Agent mode is gated to models the ledger marks `tool_capable`, promoted only with a real tool-call round-trip as evidence — the same bar as the support gate. The engine renders tool definitions through each model's own chat template (canonical flat-function form, matching llama.cpp/vLLM — see [`TOOLCALL_DIAG.md`](TOOLCALL_DIAG.md)) and the loop parses the tool-call output back out; that plumbing ships and is tested.

Promotion is decided by the **`camelid agent-eval --model <gguf>`** harness, which runs a fixed tool-use battery against a fixture and reports one of three outcomes with a receipt artifact: **`PASS`** (clean round-trip — eligible for promotion), **`FAIL`** (loaded but the model can't produce usable tool calls), or **`INCONCLUSIVE`** (didn't load in budget — a contended box, *not* a capability failure; re-run on a quiet host). A row's `tool_capable` flag is flipped true **only** after a `PASS` receipt — never a lucky run.

**`Llama 3.2 3B Instruct Q8_0` is the first promoted row** — it earned a `PASS` receipt ([`qa/agent-eval/`](qa/agent-eval/)): with the corrected render it emits well-formed tool calls, reads the fixture, and answers correctly. So `camelid chat --agent --model models/Llama-3.2-3B-Instruct-Q8_0.gguf` runs the live loop. (The 1B is too weak — it `FAIL`s the harness with malformed args even with the correct render — so it stays gated, as does any row without a PASS receipt.) The capability moves only on harness evidence, never a claim.

### Native Windows desktop app (add-on)

Prefer a desktop window over a browser tab? **[Camelid Desktop](camelid-desktop/README.md)** is an additive native Windows app (Tauri v2 + WebView2) that embeds the **same** `camelid` engine — it spawns `camelid serve` as a loopback sidecar and hosts the existing web UI in a native window. It inherits the **identical** support contract and the same runtime-ready + exact-supported-row chat gate (it talks to the same `/api/capabilities`); it makes no broader claims about supported models or performance, and any tokens/sec readout is sourced from real generation events. The web path remains canonical. See [`camelid-desktop/README.md`](camelid-desktop/README.md).

---

## Which model should I try first?

Every row below is a **supported exact row** with committed evidence; the caveat column is the real support envelope from [`STATUS.md`](STATUS.md), not marketing. The three quick-pick rows below are the frictionless path — pick one and you're chatting in two commands (or run `scripts/smoke.sh` for the zero-decision path). `camelid pull` also carries the Mistral, Qwen3 (0.6B/1.7B/4B/8B), and Gemma 4 E2B rows listed under the table.

| If you want… | Try this row | One command | First-run reality (from STATUS.md) |
|---|---|---|---|
| **The fastest "does it work" check** | TinyLlama 1.1B Chat Q8_0 | `camelid pull tinyllama` | The baseline gate — ~1.2 GB, single-node, runs anywhere. This is exactly what `scripts/smoke.sh` exercises. |
| **A solid single-node default** | Llama 3.2 3B Instruct Q8_0 | `camelid pull llama32_3b` | Exact-row smoke + API/WebUI (historical, prior upload — pending re-anchor). Verified context is **bounded to the anchored 512/1024/2048/4096/8192 raw-decode ladder** on the row's current canonical GGUF (fully GPU-resident on the 6 GiB reference card) — model-native/longer contexts still aren't a support claim. |
| **A small Gemma 4** | Gemma 4 E4B-It Q8_0 | `camelid pull gemma4_e4b` | 5/5 greedy parity vs the pinned llama.cpp oracle on the CPU runtime **and** the Metal GPU-resident runtime. Windows NVIDIA CUDA decode is **experimental** (first-token argmax matches the CPU oracle; not token-for-token gated). Multimodal input fails closed by design. |

**Also in `camelid pull`** — curated one-command downloads (resolve by any unique id fragment):

- **Most capable on a 16 GB Mac — Mistral 7B Instruct v0.3 Q8_0** · `camelid pull mistral`. Exact-row smoke with **bounded context 512→8192** and GPU-vs-CPU greedy parity; the 7B parity receipt re-verifies on a 16 GB host.
- **The Qwen3 family — 0.6B / 1.7B / 4B / 8B Q8_0** · `camelid pull qwen3_1_7b` (or `qwen3_0_6b` / `qwen3_4b` / `qwen3_8b`). **ChatML** with token-and-text parity at 1/5/50 tokens plus API smoke (thinking-disabled is the parity-locked mode); the **GPU-resident decode + single-shot prefill** path is validated on Windows CUDA, and 1.7B is token-and-text-identical to llama.cpp at a **15,373-token single-shot prefill context** (ceilings: 16,384 single-shot prefill / 40,960 KV). **Thinking mode** is opt-in (`camelid_enable_thinking:true`), token-identical for the leading reasoning trace before the documented f32 frontier. (The Qwen3-4B and Llama-3.2-3B **K-quant** exact rows aren't in `camelid pull` yet — bring the GGUF and point `serve` at it.)
- **A smaller Gemma 4 — Gemma 4 E2B-It Q8_0** · `camelid pull gemma4_e2b`. 5/5 greedy parity (CPU + Metal) vs the pinned reference, plus **bounded context 512→8192**.

> **Mostly a two-machine setup:** **Gemma 4 12B-It** is in **active validation only** through the **two-Mac distributed serve lane** (not promoted to full support) — single-node on a 16 GB CPU host is memory-bound and **unsupported**. Treat it as a deliberate two-machine setup ([`docs/gemma4-two-mac-cluster.md`](docs/gemma4-two-mac-cluster.md)). The **26B-A4B MoE** now *also* runs **single-node on a 6 GB Windows NVIDIA card** via the opt-in SSER (self-specializing expert residency) expert-offload lane — **experimental** (bit-exact greedy vs the CPU oracle in-tree; no committed evidence bundle yet).

Anything not in [Supported models](#supported-models) is refused or clearly labeled: unimplemented architectures fail closed with a typed error, and a file whose architecture is implemented but whose exact row isn't supported runs only in a clearly-marked **experimental lane** — every reply flagged as unverified, with no parity claim. That's the contract, not a limitation to work around.

---

## At a glance

| | |
|---|---|
| 🦀 **Rust-native** | Tokenizer, GGUF loader, CPU kernels, and the Metal and CUDA GPU paths live in this repo. One static binary, no Python. |
| 📦 **Direct GGUF** | Point it at a `.gguf` file — no conversion or import step. |
| 🔌 **OpenAI-style API** | `/v1/chat/completions` and `/v1/completions` with SSE streaming, served locally. |
| ✅ **Correctness-first** | Optimized paths ship only after token-for-token parity with a reference; unsupported configs fail closed with typed errors. |
| 🧾 **Proof-carrying** | Any request can emit a sealed *parity receipt* — exact GGUF (SHA-256), exact input, exact tokens — independently re-verifiable against llama.cpp on your own machine, including 7B receipts on a 16 GB Mac. |
| 📊 **Evidence-gated** | Every published number comes from a committed bundle with raw logs, commands, and versions. No raw log, no claim. |
| ⚡ **Apple Silicon path** | A Metal-resident pipeline (GPU prefill, GPU decode with on-GPU greedy sampling) measured head-to-head against llama.cpp and MLX-LM — wins, ties, and losses all stated. |
| 🖥️ **NVIDIA CUDA path (Windows)** | A from-scratch NVRTC GPU-resident decode + prefill engine — no vendored llama.cpp — token-parity-validated per exact row, with fully-GPU-resident 9B rows on a 6 GB card. |
| 🚀 **Fast model loading** | On Apple Silicon the server maps Q8_0 weights for the GPU to read in place instead of reading and copying them, so reloads are quick and peak memory stays lower. |

---

## Supported models

Support is **per exact model row** (a specific GGUF at a specific quantization), each backed by committed evidence. Anything not listed either fails closed (unimplemented architecture) or runs only in the clearly-marked experimental lane (implemented architecture, no supported row, no parity claim).

| Model row | Quant | Serve lane | Evidence |
|---|---|---|---|
| TinyLlama 1.1B Chat | Q8_0 | single-node | Current verified gate |
| **Gemma 3 1B-It** | Q8_0 | single-node (runnable serve lane, CPU — `CAMELID_RUNNABLE_SERVE=1`) | Exact-row **chat** parity smoke (gemma3 marker renderer, byte-locked against the pinned oracle's `/apply-template` by a committed shapes pack) — greedy token+text parity vs llama.cpp `acd79d603` on the committed 5-prompt gate pack: **4/5 prompts identical at every 1/5/50 depth** (incl. a natural early-stop leg), fifth identical at 1/5 with a single documented 50-token near-tie (oracle-#2, **0.3416-nat gap stated** — above the Ornith soft line, disclosed); cross-engine prompt tokenization identical 5/5. Scope: sequences well under the **512-token sliding window** (unmasked in this lane); ~5 s/token CPU observed (recorded, not a claim); tools fail closed |
| Llama 3.2 1B Instruct | Q8_0 | single-node | Exact-row + bounded context 512→8192 |
| **Llama 3.2 1B Instruct** | IQ4_XS | single-node (Windows CUDA GPU-resident; CPU wire-streaming) | Exact-row **raw-completion** parity smoke (first i-quant row) — IQ4_XS 4.25bpw linears + K-quant tied embed/lm_head, streaming the 136-byte IQ4_XS wire super-blocks with no f32 materialisation, all-16-layer GPU-resident (`iq4xs_gemv`, ~2.35 GB VRAM) and CPU wire-streamed (~2.05 GB RSS); greedy vs llama.cpp `acd79d6`: GPU-resident 2/4 probe prompts token-identical at 24 tokens, 2/4 match a probe-verified prefix then diverge at a knife-edge near-tie at index 1 (the reference token is camelid's immediate #2); prompt tokenization identical on all 4. No chat-template/serve/WebUI closure, no bounded context. Receipt: `qa/iquant/iq4xs-llama3.2-1b-parity-receipt.json` |
| **Llama 3.2 1B Instruct** | Q4_K_M | single-node (Windows CUDA GPU-resident) | Exact-row **raw-decode** parity vs llama.cpp `acd79d603` — token+text-identical on 5/8 committed-pack prompts at every 1/5/50 depth (incl. code completion and a 1,867-token long-context continuation to depth 50; prompt tokenization identical 8/8); the 3 open-ended flips are attributed benign near-ties (each the oracle's immediate #2 at ≤0.106 nat, camelid CPU-lane cross-backend control). **Local file, upstream provenance unresolved** (SHA-anchored — see `MUSTER_ACQUISITION.md`). Per-quant API/WebUI/context is a follow-up |
| Llama 3.2 3B Instruct | Q8_0 | single-node | Exact-row smoke + anchored bounded context 512→8192 (raw-decode ladder, current canonical GGUF) |
| **Llama 3.2 3B Instruct** | Q4_K_M | single-node (Windows CUDA GPU-resident) | Exact-row **raw-decode** parity vs llama.cpp `acd79d6` — token+text-identical on 5/8 confident probes at 1/5/50 (incl. code + a ~3.5k-token long-context continuation to depth 50); 3 open-ended probes are documented benign near-ties (same bar as the Q8_0 row). Per-quant API/WebUI/context is a follow-up |
| **Llama 3.2 3B Instruct** | Q5_K_M | single-node (Windows CUDA GPU-resident) | Exact-row **raw-decode** parity — token+text-identical to llama.cpp `acd79d603` at 1/5/50 (`all_pass`). GGUF not on the dev disk; rests on the committed bundle (captured on an RTX 4060 Laptop) |
| Llama 3 8B Instruct | Q8_0 | single-node | Exact-row + bounded context 512→2048 |
| Mistral 7B Instruct v0.3 | Q8_0 | single-node | Exact-row smoke + bounded context 512→8192 + GPU/CPU parity |
| **Qwen3 1.7B** | Q8_0 | single-node (CPU + CUDA) | Exact-row ChatML (thinking-disabled) — token+text parity at 1/5/50 tokens + API smoke; macOS GPU-resident decode+prefill validated to a 15,373-token context (vs llama.cpp); **Windows CUDA GPU-resident decode+prefill** parity (== cpu_reference/llama.cpp at 1/5/50, RTX 3060 Laptop); **bounded context 512/1024/2048/4096** (446/968/2013/4088 tokens) fully-GPU-resident == llama.cpp `acd79d603` at 50 tokens (8192 held as a documented benign near-tie — both reach CMLD-8192, camelid one token longer); thinking mode opt-in (leading-trace parity) |
| **Qwen3 0.6B** | Q8_0 | single-node (CPU + CUDA) | Exact-row ChatML (thinking-disabled) — token+text parity at 1/5/50 tokens (explicit head_dim path); **Windows CUDA GPU-resident** parity (== cpu_reference/llama.cpp); **bounded context 512/2048/4096/8192** (446/2013/4088/8253 tokens) fully-GPU-resident == llama.cpp `acd79d603` at 50 tokens — engine-correctness parity (the tiny 0.6B degenerates at context; not a text-quality claim), 1024 held as an isolated benign near-tie; thinking mode opt-in (leading-trace parity, 6–126-token envelope) |
| **Qwen3 4B** | Q8_0 | single-node (CPU + CUDA) | Exact-row ChatML (thinking-disabled) — token+text parity at 1/5/50 on confident prompts (explicit head_dim); one probe is a documented first-token near-tie; **Windows CUDA GPU-resident** parity (== cpu_reference/llama.cpp); **bounded context 512/1024/2048** GPU-resident == llama.cpp `acd79d603` at 1/5/50, plus **4096/8192** (4088/8253 tokens) == llama.cpp at 50 tokens (on 6 GiB these exceed the 2090-pos VRAM KV cap → GPU-resident prefix + CPU-fallback tail, identical tokens); thinking mode opt-in (leading-trace parity, 35–235-token envelope) |
| **Qwen3 4B** | Q4_K_M | single-node (Windows CUDA GPU-resident) | Exact-row ChatML (thinking-disabled) — GPU-resident CUDA decode (`q4k_gemv` + `q6k_gemv`, fully VRAM-resident on a 6 GB card) token+text-identical to llama.cpp `acd79d6` at 1/5/50 on all 3 prompts; a default-on CPU K-quant block-dot lane also decodes it (confident-probe parity); **bounded context 512/1024** (446/968 tokens) GPU-resident == llama.cpp `acd79d603` at 50 tokens (in the same sweep 4096/8192 also matched but are held for a contiguous ladder — the 2048 bucket hits a documented benign 0.08-nat Q4 greedy near-tie). **`tool_capable`** — earned a committed agent-eval PASS on this exact GGUF (the Llama-3.2-3B K-quants FAIL the same battery, so they stay tool-gated) |
| **Qwen3 8B** | Q8_0 | single-node (CPU + CUDA) | Exact-row ChatML (thinking-disabled) — token+text parity at 1/5/50 tokens (untied embeddings); on the macOS GPU-resident decode+prefill path; **Windows CUDA** via VRAM+host-RAM offload (16/36 layers resident on a 6 GB card, parity == cpu_reference/llama.cpp at 1/5/50); thinking mode opt-in (template-shape byte parity + host-bounded leading-trace) |
| **Ornith 1.0 9B** (`qwen35` hybrid state-space/SSM) | Q8_0 (CPU) · Q4_K_M, Q3_K_M (**Windows CUDA, fully GPU-resident on 6 GB**) | single-node (CPU + CUDA) | From-scratch `qwen35` engine (gated DeltaNet linear attention + sparse full attention, 24+8 layers). Greedy token parity vs pinned llama.cpp `acd79d6`: 4/4 committed at n=20 (Q8_0 CPU) + five-prompt n=64 extension with every flip probed and attributed to sub-0.33-nat near-ties (the oracle's own backends flip at the same positions); byte-exact BPE tokenizer gate (45 fixtures × 2 modes incl. NFD/Devanagari/ChatML adversarial); `tool_capable` on **Q8_0 and Q4_K_M** (a committed agent-eval battery PASS on each — Q3_K_M has no agent-eval receipt yet and is not tool-capable); `reasoning_content` + `qwen3_xml` → `tool_calls` serving incl. SSE streaming with `include_usage`; **Q3_K_M runs 16K context fully resident** (4.7 GB peak, ≥1.3 GB headroom) at ~15 tok/s, Q4_K_M @8K at ~19 tok/s, greedy == CPU oracle. Receipts under `qa/ornith/` |
| **Ternary Bonsai 4B** (community ternary, `qwen3` arch) | TQ2_0 (+ Q6_K tied head) | single-node (CPU completion smoke **only**) | Exact-row CPU completion smoke — streams the TQ2_0 wire blocks + Q6_K tied head (4B in ~3.1 GB RSS, no f32 materialization); greedy parity vs llama.cpp `acd79d6`: 3/4 probe prompts token-identical at 24 tokens, 1 documented benign near-tie; decode ~11.3 tok/s ≈ 0.53× llama.cpp (recorded, not a perf claim). No serve/WebUI/frontend gate, no bounded context, not tool-capable, not in `camelid pull`; community-sourced GGUF (superkaiii/Ternary-Bonsai-4B). Receipt: `qa/ternary/tq2_0-bonsai-parity-receipt.json` |
| **Gemma 4 E2B-It** | Q8_0 | single-node (CPU + Metal + Windows CUDA) | 5/5 greedy parity (CPU + Metal) vs pinned llama.cpp 5d56eff **+ bounded context 512→8192** (committed context-pass bundle, token+text-identical at every bucket). **Windows NVIDIA CUDA** GPU-resident lane is token+text-identical to the oracle on basic_v1 + context 512→8192 (and deep_v1 to the same near-tie frontier the CPU lane hits) — GPU-verified, committed bundle, token-for-token gated; not experimental (contrast E4B) |
| **Gemma 4 E4B-It** | Q8_0 | single-node (CPU + Metal); CUDA experimental | 5/5 greedy parity (CPU + Metal) vs the pinned reference **+ bounded context 512→8192** (committed bundle, token+text-identical at every bucket). **Windows NVIDIA CUDA** decode is **experimental** — first-token argmax matches the CPU oracle (in-tree gate, RTX 3060 Laptop); not token-for-token gated, no committed evidence bundle yet |
| **Gemma 4 12B-It** | Q8_0 | two-Mac distributed | **Active validation** (two-Mac distributed ONLY, not promoted): distributed output == single-node + 3/5 full-budget vs the pinned reference + serve/WebUI smoke |
| **Gemma 4 26B-A4B-It QAT** | Q4_0 (128-expert MoE) | two-Mac distributed; **single-node Windows CUDA (experimental)** | **Active validation** (two-Mac distributed): 2/5 full-budget + 3/5 frontiers vs the pinned reference + serve/WebUI smoke. **Windows NVIDIA CUDA single-node GPU-resident** now runs this 26B MoE on a **6 GB card** via the opt-in **SSER** expert-offload lane (attention + dense on the GPU; the 128 experts stay host-resident with an adaptive VRAM expert cache): **experimental** — greedy tokens bit-identical to the CPU oracle on content (in-tree, RTX 3060 Laptop; within the f16-KV floor), ~6–7 tok/s, opt-in via `CAMELID_GEMMA4_CUDA=1` + `CAMELID_SSER_CACHE=1`; no committed evidence bundle yet |
| **DiffusionGemma 26B-A4B-It** | Q4_K_M | single-node (CPU; Windows CUDA) | **Experimental** — bit-exact through the full chat path (Phases 0–6) vs the pinned reference (Apple Silicon); **verified running on Windows x86_64 (MSVC)** with model receipts (determinism pairs + a full 2-block 512-token multi-canvas answer), pure-Rust (no C/C++). Run via `camelid diffusion-gemma-chat` (`--max-steps N` bounds the denoise). GPU (CUDA, on by default on Windows): the self-conditioning matmul runs on the GPU (**2.7× wall**, documented non-bit-exact; `CAMELID_DG_CUDA_SC=0` opts out) and the MoE experts run from a **VRAM-resident expert pool that is bit-exact** — an experts-on-GPU run is byte-identical to the CPU-pure oracle (fits a 6 GB card: 4.6 GiB peak) |

> **Fails closed (by design):** Mixtral-8x7B v0.1 (validation-in-progress, one-token runtime only); other Qwen3 sizes (14B/32B), base variants, Qwen3-MoE (A3B), and full-trace Qwen3 thinking-mode token-parity (thinking is available opt-in with leading-trace parity); Gemma 4 26B-A4B **Q8_0** (26.9 GB) and 31B (over the 2×16 GB envelope); Gemma 4 MTP/drafter rows; **DiffusionGemma 26B-A4B on the autoregressive engine** (a discrete block-diffusion model cannot run an AR forward — the AR engine fails closed and redirects to the dedicated `diffusion-gemma-chat` lane, which **is** supported; see below); multimodal input; and all quantization rows not named above (implemented formats load experimental-only with no parity claim; unimplemented formats fail closed — see [Quantization support](#quantization-support)).

### Quantization support

Quantization breadth is tiered, never a flat list — a quant is only ever supported as an exact (model file × quant × lane) row with a committed receipt:

| Tier | Formats | What it means |
|---|---|---|
| **Certified per exact row** | Q8_0 (most rows) · Q4_K_M (Qwen3 4B, Llama 3.2 3B + 1B, Ornith 9B) · Q5_K_M (Llama 3.2 3B) · Q3_K_M (Ornith 9B) · Q4_0 QAT (Gemma 4 26B-A4B distributed, E4B Metal) · TQ2_0 (Ternary Bonsai 4B, CPU smoke) · IQ4_XS (Llama 3.2 1B, GPU-resident/CPU raw-completion smoke) · the DiffusionGemma Q4_K_M lane | Parity-certified against a pinned reference, listed in the table above with its evidence bundle. Support never spreads to neighboring files or quants. |
| **Runnable/experimental lane** | F32, F16, Q8_0, Q4_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, BF16 (for the covered architectures; BF16 joined the covered set as an exact-decode type via D-B6, f32-materializes on load); NVFP4 (**gemma4-E4B pilot only, Windows + macOS** — macOS CPU wire lane on `serve`, plus an opt-in Metal GPU resident lane via `gemma4-generate-gpu` as of GABBRO M3-followup (self-parity-proven vs the CPU oracle, no macOS perf or real-artifact claim); narrower than the other formats here) | The engine runs the file in a clearly-marked experimental lane — every reply flagged unverified, **no parity claim** (`src/runnable/admit.rs`). NVFP4 admission is a pilot-model carve-out and fails closed on targets other than Windows/macOS, on sidecar-bearing files, and on NaN-sentinel scale bytes — its produced pilot file (gemma4-E4B) admits fully as of D-B6 (2026-07-17): BF16 is now a covered exact-decode type, so the one BF16 tensor (`per_layer_model_proj`) that was the prior admission blocker no longer refuses. It executes via `gemma4_runtime` (CPU wire + CUDA-resident lanes), not the generic runnable serve bridge. |
| **Load-only** | Q4_1, Q5_0, Q5_1, Q2_K, Q8_K, IQ4_NL, TQ1_0 (and TQ2_0 outside the certified Bonsai row) | CPU dequant-to-f32 exists (some also have GPU kernels) — an engine fact, not a support claim; loadable for implemented architectures in the experimental lane only. |
| **Fail-closed** | Q8_1, integer/F64 tensors, unrecognized type ids | Typed error at load; nothing runs. |

Per-row receipts live in [`SUPPORT_MATRIX_v0.1.md`](SUPPORT_MATRIX_v0.1.md), [`COMPATIBILITY.md`](COMPATIBILITY.md), and `/api/capabilities`.

### Experimental lanes

| Model row | Quant | Status | Evidence |
|---|---|---|---|
| DiffusionGemma 26B-A4B-It | Q4_K_M | **Supported (experimental) via the dedicated diffusion lane** (`camelid diffusion-gemma-chat`, `--max-steps N`). Pure-Rust on macOS / Linux / **Windows** (the expert-argsort C++ shim was removed). CPU multi-step is slow (the self-conditioning matmul dominates). **GPU (Windows CUDA, evidence-backed):** the SC matmul offloads for 2.7× wall (non-bit-exact f32 accumulation, `CAMELID_DG_CUDA_SC=0` opts out; `CAMELID_DG_CUDA=0` for CPU-pure) and the MoE expert pool holds ~2.9 GiB of experts VRAM-resident with **bit-exact** GEMV kernels — byte-identical output vs the CPU-pure run (recon §8e). The autoregressive engine fails closed and redirects here by design. | End-to-end bit-exact vs the pinned llama.cpp diffusion reference at zero tolerance, [recon](docs/recon/DIFFUSIONGEMMA_RECON.md): Phase 0.5 lazy dequant (5 quant formats) + Phase 1 tokenizer (12/12, 100% token-id match) + Phase 2 encoder checkpoints (242/242, 510/510 expert selections) + Phase 3 single denoise step (all 67,108,864 canvas logits + host-RNG streams + every EB step-0 output) + Phase 4 full EB denoise loop (S=48, live self-conditioning, 268M logits) + Phase 5 multi-canvas block-autoregressive loop (2 blocks, 512-token response byte-identical to the reference) + Phase 6 chat wrapper (render+tokenize and detokenize parity vs the reference chat path). CPU-pure pinned configuration |
| Gemma 3 1B-It | Q8_0 | **PROMOTED (2026-07-16, MUSTER M-A1)** — this row now carries a supported exact-row contract entry (`gemma_3_1b_it_q8_0`, runnable serve chat smoke, `CAMELID_RUNNABLE_SERVE=1`; see the supported-models table above) and no longer runs as unverified-experimental. Support stays scoped to the exact file, chat sequences under the 512-token sliding window, and no tools/perf/context claims. | Promotion bundle `qa/evidence-bundles/gemma3-1b-q8-runnable-serve-chat-parity-20260716-head-6d0d57eb/`; the earlier HF-reference receipt (`qa/runnable/gemma3-parity.json`) remains as independent engine-math evidence. (The Phi-3-mini Q8_0 catalog entry exited MUSTER via a committed HOLD receipt — see its own row below.) |
| Phi-3-mini-4k-instruct | Q8_0 | **Experimental — HOLD (MUSTER M-A2, 2026-07-16)**: no supported row; both candidate parity surfaces are blocked by SPM tokenizer divergences (the specials `rstrip` seam on chat prompts; the documented merge-order divergence on 6/8 raw pack prompts), which are the pre-existing `SPM_MERGE_ORDER_CONDUCTOR.md` campaign, not a row-sized fix. **The engine forward is healthy**: MUSTER's probe-proven NEOX RoPE flip turned long generation from degenerate (the old 92029b7e limitation) to coherent, matched-token prompts run 40+ tokens identical to the pinned oracle, and the remaining flip is a 0.0995-nat benign near-tie. Replies remain flagged unverified. | HOLD receipt `qa/muster/HOLD-phi3-mini-4k-instruct-q8_0.json` + evidence `qa/muster/phi3-hold-evidence/` (both parity reports committed as-is, prompt-token parity, rope-probe artifacts, oracle n_probs); renderer byte-locked by `qa/prompt-packs/phi3-chat-template-shapes-v1.json` for whenever the SPM campaign clears the blockers. |
| Gemma 4 E4B-It | Q4_0 (mixed QAT) | **Experimental — Windows NVIDIA CUDA, off by default** (`CAMELID_GEMMA4_CUDA`). The mixed-quant export (Q4_0 projections, Q4_1 ffn_down, Q4_K tied head, Q5_K per_layer_token_embd, BF16 proj) runs GPU-resident in ~2.5 GB — fits a 6 GB card with headroom — at ~18 tok/s decode. **Not a supported row:** the first-token argmax matches the CPU oracle and every projection GEMV is bit-exact, but the fp-reassociated attention/Per-Layer-Embedding (PLE)/norm reductions flip later near-ties on the coarse Q4 logits, so it is **argmax-stable, not token-for-token** greedy parity (unlike the E2B Q8_0 CUDA row, which is token-for-token gated on the checked packs; the E4B Q8_0 CUDA lane remains experimental). | Per-kernel bit-exact unit tests (q4_0 / q4_1 / q4_k GEMV, 96/96 rows each) + an in-tree first-token parity gate vs the CPU oracle loading the same file (RTX 3060 Laptop, 6 GB) |
| Gemma 4 E4B-It | NVFP4 (mm; gemma4-E4B pilot) | **Pilot lane (gemma4 wire CPU + CUDA on Windows; CPU wire lane on macOS), Windows + macOS — macOS is now a supported_exact_row_smoke row (GABBRO; Metal GPU-resident lane); the Windows/CUDA lane stays receipted engine facts. A current-engine near-tie vs Q4_K, NOT quality-competitive beyond the 2pp GO tolerance — a space/speed quant.** NVFP4 4-bit weights, gemma-4-E4B pilot, Windows + macOS (CPU wire lane on `serve`; the macOS Metal GPU resident lane also runs NVFP4 as of GABBRO M3-followup, opt-in via the macOS-only `gemma4-generate-gpu` subcommand — self-parity-proven vs the CPU oracle, fail-closed on NaN-sentinel/sidecar scales, run end-to-end on the byte-exact real artifact; isolated 128-tok decode 12.12 tok/s, 1.45x the Q8_0 parent). Engine facts: bit-exact CPU decode, validated on x86 and on Apple Silicon/ARM (GABBRO Gate G-M1), + a Windows CUDA dp4a GEMV kernel (46/46 bit-identical) — its produced pilot file (gemma4-E4B) admits fully as of D-B6 (2026-07-17): BF16 is now a covered exact-decode type, so the one BF16 tensor (`per_layer_model_proj`) that was the prior admission blocker no longer refuses. It executes via `gemma4_runtime` (CPU wire + CUDA-resident lanes), not the generic runnable serve bridge. Measured vs the Q8_0 parent at matched 4.5 bpw: behind Q4_K on quality (G3 NO-GO, 88.5% vs 92.6% top-1 agreement; 0.111 vs 0.065 mean-KL nats), but 1.03x faster than Q8_0 CUDA decode (26.51 vs 25.80 tok/s) and 2.08 GB lighter VRAM on an RTX 3060 Laptop (decode-only, Windows/this box — no macOS perf claim). macOS: now a supported_exact_row_smoke row (a current-engine near-tie vs Q4_K; the frozen G3 NO-GO stands as history); NOT quality-competitive beyond the 2pp GO tolerance — a space/speed quant. Sidecar-bearing and NaN-sentinel files fail closed; targets other than Windows/macOS refuse with the typed TK2 error; Phase 5 (Blackwell) BLOCKED-HW. | G3 quality (NO-GO): `qa/evidence-bundles/basalt/phase3/BASALT_G3_SUMMARY.md`; G4 CERT + dp4a perf: `qa/evidence-bundles/basalt/phase4/cert/BASALT_G4_SUMMARY.md`; macOS/ARM bit-exact decode: `qa/evidence-bundles/gabbro/phase1/GABBRO_M1_SUMMARY.md`; unit gate `nvfp4_gemv_matches_oracle` (46/46 bit-identical); spec `docs/architecture/NVFP4_FORMAT.md` |

**Windows bring-up, perf & GPU (2026-06/07).** The diffusion lane builds and **runs verified on Windows x86_64 (MSVC)** with **zero C/C++**: the expert-argsort `std::sort` shim was ported to pure Rust, leaving only macOS-only Apple framework bindings (`vDSP` / `__sincosf_stret`) elsewhere. The portable sort breaks an exact expert-probability tie by lower index rather than reproducing the reference's libc++ introsort tie-order, so re-validate the Apple-Silicon encoder/decode parity gates if exact-tie ordering matters. **GPU offload now has committed evidence** (recon §8e; RTX 3060 Laptop 6 GB): CPU-pure and GPU determinism run-pairs both byte-identical; the self-conditioning matmul on GPU cuts a 2-step leg from 551 s to 200 s (**2.7×**, the one non-bit-exact stage — f32 accumulation, per-stage opt-out `CAMELID_DG_CUDA_SC=0`); the **MoE expert pool** keeps ~2.9 GiB of expert weights VRAM-resident behind bit-exact Q4_K/Q8_0 GEMV kernels (unit-gated, and an experts-on-GPU run is **byte-identical to the CPU-pure oracle**) — a capacity lever that also lets the 16.8 GB model coexist with a 16 GB-RAM host; and a natural-stop 2-block multi-canvas answer (512 tokens, grown prefix 297 matching the Apple-sealed Phase 5 shape) completed end-to-end at 4.6 GiB VRAM peak. Windows claims are a this-host contract (determinism + bit-exact GPU stages vs the local CPU oracle), not a bitwise claim against the Apple-sealed artifacts (off-macOS sincos + tie-order notes in the recon).

Per-row detail and the exact evidence artifacts live in [`SUPPORT_MATRIX_v0.1.md`](SUPPORT_MATRIX_v0.1.md) and [`COMPATIBILITY.md`](COMPATIBILITY.md).

---

## Engine status

| Capability | Status | Notes |
|---|---|---|
| GGUF loading | ✅ Working | Direct load with metadata/tensor inspection (`camelid inspect`). |
| Q8_0 inference | ✅ Working | The most broadly validated quantization; support is per exact row (see above). K-quant, Q4_0/QAT, and TQ2_0 rows are certified per exact row too (see below). |
| Gemma 4 engine | ✅ Working | From-scratch `gemma4` engine — see [Gemma 4](#gemma-4) below. |
| OpenAI-style API | ✅ Working | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, plus capability/health routes. |
| Streaming chat | ✅ Working | SSE streaming on the chat endpoint, including OpenAI `stream_options.include_usage` — opt in for a terminal `usage` chunk whose `prompt_tokens`/`completion_tokens`/`total_tokens` are identical to the non-streaming response. |
| Apple Silicon Metal path | ✅ Working | GPU-resident prefill and decode, auto-selected when a Metal device is present; CPU fallback otherwise. |
| NVIDIA CUDA path (Windows) | ✅ Working | GPU-resident decode + single-shot prefill (`--features cuda`), auto-selected when a CUDA device is present; token-parity-validated on the dense Qwen3 Q8_0 rows (RTX 3060 Laptop). The **Gemma 4 E2B-It Q8_0** row is **token-for-token gated** on this path (basic_v1 + context 512→8192 strict, deep_v1 to the shared near-tie frontier; GPU-verified, committed bundle) — opt-in via `CAMELID_GEMMA4_CUDA=1`. The **Gemma 4 E4B-It Q8_0** row also runs on this path but is **experimental** (first-token argmax matches the CPU oracle; not token-for-token gated, no committed bundle yet). Models that exceed VRAM use automatic VRAM+host-RAM layer offload; CPU fallback otherwise. Results are GPU/driver/CUDA-version specific. |
| Web frontend | ✅ Working | Local React/Vite chat surface, embedded in the binary and served at the same address; unlocks chat only for recognized model rows. |
| Parity receipts | ✅ Working | Opt-in sealed record of one request; `camelid verify-receipt` re-checks it against llama.cpp (incl. 7B on a 16 GB host). |
| Two-Mac distributed serve | ✅ Working | Layer sharding over TCP for rows too large for one 16 GB host (Gemma 4 12B, 26B-A4B). |
| K-quant inference (Q4_K_M / Q5_K_M / Q3_K_M) | ✅ Working (per exact row) | Parity-certified on named rows only (Qwen3 4B Q4_K_M, Llama 3.2 3B Q4_K_M/Q5_K_M, Ornith 9B Q4_K_M/Q3_K_M — see the table above); GPU-resident CUDA `q4k`/`q5k`/`q6k`/`q3k` kernels plus a default-on CPU K-quant block-dot lane. Neighboring K-quant files are not supported until separately certified. |
| Q4_0 / QAT inference | ✅ Working (per exact row) | Gemma 4 QAT rows: 26B-A4B MoE (Q4_0 experts + Q6_K head) in two-Mac distributed active validation, and the E4B QAT row on the Metal GPU-resident path, token-identical to the CPU runtime. Parity-gated Q4_0 wire GEMVs on CUDA and Metal. |
| Ternary (TQ2_0) inference | ✅ Working (one exact row) | Ternary Bonsai 4B TQ2_0 (`qwen3` arch) is a supported exact-row **CPU completion-smoke** lane (streamed wire blocks, 3/4 probe parity + 1 documented near-tie); no serve/WebUI/context/perf claim. |
| Other quantizations | 🧪 Load-only / experimental | CPU dequant-to-f32 exists for Q4_1, Q5_0, Q5_1, Q2_K, Q8_K, BF16, IQ4_NL, TQ1_0 (plus CUDA `q2k`/`q4_1` GEMV kernels) — loadable in the experimental lane for implemented architectures, unverified, **no parity claim**. Unrecognized formats fail closed with a typed error. |
| Ghost mode (layer streaming) | 🧪 Experimental | `ghost-run` executes one block at a time for a strict memory ceiling; trades throughput for memory. |

---

## Ornith 1.0 9B (qwen35)

Camelid implements the `qwen35` architecture (Qwen3.5 / [Ornith-1.0-9B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF), a hybrid of gated-DeltaNet linear-attention (state-space, SSM) layers and sparse full attention — 24 SSM + 8 full-attention layers) from scratch: the gated delta-rule recurrence, causal conv1d, partial NEOX mRoPE, gated attention, and a hybrid cache (fixed-size recurrent state for DeltaNet layers, f16 KV for the full-attention layers — context length does not grow the SSM state). The reference pin is llama.cpp `acd79d6` with CUDA ([`REFERENCE_PIN_QWEN35.md`](REFERENCE_PIN_QWEN35.md)).

The model is a reasoning model with tool calling: turns open with `<think>…</think>` (split into `reasoning_content`, message and SSE deltas) and tools are called in its custom `qwen3_xml` format, lifted into OpenAI `tool_calls`. The **Q8_0 and Q4_K_M** rows are `tool_capable`, each with a committed agent-eval PASS receipt (Q8_0: three `camelid.agent_eval/v1` PASS receipts on the runnable lane; Q4_K_M: a full read/list/write agent battery PASS on the GPU-resident file). The **Q3_K_M** row carries no agent-eval receipt yet and is not tool-capable.

On the recorded 6 GB Windows CUDA card (RTX 3060 Laptop), **Q4_K_M and Q3_K_M run fully GPU-resident** — sparse KV keeps 16K context inside 6 GB on Q3_K_M (4.7 GB peak) — with a device-side decode loop (on-GPU embedding gather + resident rope tables; the host syncs once per token chunk). Decode is ~19 tok/s (Q4_K_M @8K) / ~15 tok/s (Q3_K_M @16K), greedy-token-identical to the CPU oracle lane. The Q3_K_M and IQ-family quants, an imatrix calibrated on an agentic-coding corpus, and a quality × residency table are documented under [`qa/ornith/constrained-vram/`](qa/ornith/constrained-vram/QUANT_QUALITY_TABLE.md); a speculative split-precision lane was evaluated and closed with a measured NO-GO receipt (acceptance 0.87 passed, the speed bar did not — see `RECEIPT_ITEM5_acceptance_economics.json`).

## Gemma 4

Camelid implements Gemma 4 from scratch in the `gemma4` engine: per-layer-type sliding/global attention (the GGUF `sliding_window_pattern` is authoritative; E2B is 4:1), per-layer FFN widths and KV-head counts, QK-norm, dual-θ RoPE, GeGLU, Per-Layer-Embeddings, cross-layer KV sharing, and the `<|turn>`/`<turn|>` chat markers with thinking-channel suppression. Multimodal input fails closed with a typed error.

**E2B-It & E4B-It (Q8_0, single-node).** Five-prompt greedy parity against the pinned llama.cpp oracle on **both** the CPU and the Metal GPU-resident runtime. A reproducible bounded-context harness (512 / 1024 / 2048 / 4096 / 8192) and pinned-comparator oracles are committed under `qa/gemma4/`, but **bounded context is not a promoted support claim** for these rows — the committed evidence bundles record exact-row text-token generation within the `basic_v1` prompt-pack envelope only ("no bounded-context promotion"). The chat template is locked byte- and token-exact (`qa/gemma4/template_shapes_v1.json`, both thinking modes). A Metal GPU-resident decode path (`camelid gemma4-generate-gpu`) runs the full E4B forward on the GPU at the memory-bandwidth wall. The **QAT row (`gemma-4-E4B_q4_0-it`, Q4_0 layers + Q6_K tied head) runs on the same GPU-resident path** — the Q4_0 projections decode on the GPU (parity-gated wire GEMVs) and the Q6_K tied head runs on the CPU; on an M4 it is token-for-token identical to the CPU runtime and ~25 % faster warm (15.2 vs 12.2 tok/s). The per-block GPU↔CPU parity is gated in CI; the end-to-end GPU==CPU check runs locally (no GPU model in CI). See [`docs/performance/gemma4-qat-gpu-2026-06-11.md`](docs/performance/gemma4-qat-gpu-2026-06-11.md). The committed CPU QAT parity (E4B QAT `basic_v1`: 3/5 full-budget + 2 probe-verified frontiers) is unchanged.

**E4B-It on Windows (CPU and NVIDIA GPU).** Gemma 4 E4B-It Q8_0 runs on Windows on the **CPU** and now **also on the NVIDIA CUDA GPU**. The CUDA lane is a from-scratch GPU-resident decode path for the full E4B forward — resident layer weights, a captured decode graph, the tied head, sliding/global attention, dual-θ RoPE, QK-norm, GeGLU, and Per-Layer-Embedding injection all on-device — wired into `serve` behind a `--features cuda` build and opt-in via `CAMELID_GEMMA4_CUDA=1` — it is **not** auto-engaged; the CPU lane serves gemma4 unless the flag is set. Its first-token argmax matches the CPU `Gemma4Runtime` oracle, asserted by the in-tree gate (`gemma4_cuda_matches_cpu_greedy`), which is `#[ignore]` and checks first-token argmax stability — later-token divergence is permitted on this lane, so it is **not** token-for-token greedy parity. As with the other CUDA rows, results are GPU/driver/CUDA-version specific, so the E4B lane stays experimental beyond the recorded GPU (RTX 3060 Laptop, 6 GB) and the CPU path remains the correctness reference; there is no committed CUDA evidence bundle for the E4B row yet.

**E2B-It on Windows CUDA (token-for-token, committed bundle).** Unlike E4B, the `gemma-4-E2B-it-Q8_0` row's CUDA-resident lane *is* token-AND-text identical to the pinned llama.cpp `5d56eff` oracle — across `basic_v1`, `deep_v1` (3/4 full-budget strict + 1 bounded to the oracle's own `applies_to_gpu` near-tie frontier), and the full context ladder 512 / 1024 / 2048 / 4096 / 8192 (strict). GPU-resident execution is empirically verified (nvidia-smi 2.7 GB VRAM, 90–100 % utilization on the RTX 3060 Laptop; not a CPU fallback), gated by the in-tree `tests/gemma4_generation_parity.rs` CUDA branch (`CAMELID_GEMMA4_CUDA=1`) and committed under `qa/evidence-bundles/gemma4-e2b-q8-cuda-resident-parity-*`. This upgrades the **E2B** Windows-CUDA lane from experimental (first-token argmax) to token-for-token parity on the checked packs; results stay specific to the recorded GPU and the CPU path remains the correctness reference. It does **not** promote the E2B row beyond `supported_exact_row_smoke` (performance/RSS and WebUI-promotion evidence still gate any wider claim), and it says nothing about the E4B CUDA lane.

**12B-It (Q8_0) & 26B-A4B-It QAT (Q4_0, MoE) — two-Mac distributed.** These rows are too large for a single 16 GB host, so the lane under active validation is distributed layer sharding over TCP (not promoted to single-node support): `gemma4-master`/`gemma4-worker` split one row across two machines with a versioned handshake and per-packet checksums, and distributed greedy output is asserted token-identical to single-node (`tests/gemma4_distributed_parity.rs`). The 26B row is a 128-expert MoE (Q4_0 experts + Q6_K tied head) with the dense shared-expert + sparse top-8 branch implemented end to end.

Proven on two 16 GB M4 Mac minis, full `basic_v1` pack vs the pinned reference:

| Row | Distributed = single-node | vs. reference |
|---|---|---|
| 12B-It Q8_0 | 5/5 token-identical | 3/5 full-budget + recorded comparator frontiers |
| 26B-A4B-It QAT Q4_0 | identical (f32 wire) | 2/5 full-budget token-identical + 3/5 probe-verified knife-edge frontiers |

Both rows serve over HTTP through the same lane — set `CAMELID_GEMMA4_SERVE=1` plus `CAMELID_GEMMA4_WORKER`/`CAMELID_GEMMA4_SPLIT`, and `/v1/chat/completions` (incl. SSE) and `/v1/completions` route through a persistent master shard with per-request worker sessions (wire protocol v1). The distributed serve/WebUI promotion smoke is green for both. Evidence bundles are under [`qa/evidence-bundles/`](qa/evidence-bundles/); setup is in [`docs/gemma4-two-mac-cluster.md`](docs/gemma4-two-mac-cluster.md).

> Scope guardrails: these are exact-row claims only — no Gemma-family-wide support, and no model-native/larger context beyond the checked packs.

---

## Quickstart

Already have a binary from [Install](#install)? Skip to "Get a model" below. To build from source instead — the web UI is compiled into the binary, so build the frontend first and it gets embedded (one binary, no separate Node process at runtime):

```bash
(cd frontend && npm ci && npm run build)   # bundles the web UI
cargo build --release                       # embeds it into the binary
```

### Build from source on Windows (x86_64, MSVC)

Windows `x86_64-pc-windows-msvc` is a tracked platform (see [`COMPATIBILITY.md`](COMPATIBILITY.md) → Platform support). Most users should grab the prebuilt **signed** Windows download in [Install](#install) above (GPU acceleration included); build from source only if you want to modify Camelid. Prerequisites: the **MSVC** toolchain (Visual Studio Build Tools with the C++ workload — *not* MinGW), Rust via `rustup` with the `x86_64-pc-windows-msvc` host, and Node.js for the embedded web UI. Then, in PowerShell:

```powershell
cd frontend; npm ci; npm run build; cd ..   # bundles the web UI
cargo build --release                       # embeds it into the binary
.\target\release\camelid.exe pull tinyllama # the baseline supported row
.\target\release\camelid.exe serve --model models\tinyllama-1.1b-chat-v1.0.Q8_0.gguf
```

The server behaves exactly as on the other platforms (listens on `127.0.0.1:8181`, same OpenAI-style API + web UI). The TinyLlama 1.1B Chat Q8_0 baseline gate is verified on Windows with the same parity evidence as macOS/Ubuntu.

> **GPU (NVIDIA/CUDA) on Windows.** `cargo build --release --features cuda` adds a CUDA backend with a GPU-resident decode engine (weights uploaded once, single-shot GPU prefill, on-device greedy/temperature sampling) implemented from scratch in NVRTC kernels — no vendored llama.cpp. It auto-engages when a CUDA device is present for the validated Qwen3 rows below; the Gemma 4 lane is the exception — it stays opt-in behind `CAMELID_GEMMA4_CUDA=1` and is **not** auto-engaged. The **dense Qwen3 Q8_0 rows (0.6B / 1.7B / 4B / 8B Instruct, thinking-disabled ChatML)** are validated on it: GPU decode + single-shot prefill are token-AND-text-identical to the camelid `cpu_reference` (transitively llama.cpp 9632) at 1/5/50 generated tokens, greedy. **Gemma 4 E4B-It Q8_0** also runs on this CUDA lane (enabled with `CAMELID_GEMMA4_CUDA`), greedy-parity with the CPU `Gemma4Runtime` oracle via the in-tree gate (`gemma4_cuda_matches_cpu_greedy`); it has no committed evidence bundle yet, so it stays experimental beyond the recorded GPU. `/api/capabilities` reports the live path (`selected_backend=cuda_resident_q8_runtime`, `cuda_resident_active=true`). Validated on an **RTX 3060 Laptop (6 GB), driver 576.83, CUDA 12.9**; 0.6B/1.7B/4B are fully VRAM-resident and 8B runs via automatic VRAM+host-RAM layer offload. Results are **GPU/driver/CUDA-version specific** (f32 reduction order is GPU-specific), so the lane stays experimental beyond the recorded GPU; the CPU path remains the default and correctness reference. See [`COMPATIBILITY.md`](COMPATIBILITY.md) → *Windows CUDA* and the `qa/evidence-bundles/qwen3-*-windows-cuda-resident-parity-*` bundles. Building the feature needs the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (12.x; libraries loaded at runtime); running it needs an NVIDIA GPU + driver.

Get a model. `pull` fetches a known-good row into `./models` — support stays per exact row (mostly the validated Q8_0 rows; the certified K-quant rows aren't in `pull` yet — bring the GGUF and point `serve` at it). Catalog presence is not a support claim: the Gemma 3 1B and Phi-3-mini Q8_0 catalog entries have no supported row and run in the experimental/runnable lane only (see [Experimental lanes](#experimental-lanes)). A GGUF whose architecture is implemented but whose exact row isn't supported runs in the clearly-marked experimental lane; unimplemented architectures and formats fail closed:

```bash
./target/release/camelid pull              # list the supported models
./target/release/camelid pull llama32_3b   # download Llama 3.2 3B Instruct Q8_0
```

Serve it (`pull` prints the exact command to run; the model is in `./models`):

```bash
./target/release/camelid serve \
  --model models/Llama-3.2-3B-Instruct-Q8_0.gguf \
  --threads 4
```

The server listens on `127.0.0.1:8181` and **opens the chat UI in your browser automatically** (pass `--no-open` to disable). The same address serves the OpenAI-style API. List the loaded model (its `id` comes from the GGUF metadata):

```bash
curl -s http://127.0.0.1:8181/v1/models
```

Chat (replace the model `id` with the one returned above; add `"stream": true` for SSE):

```bash
curl -s http://127.0.0.1:8181/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama 3.2 3B Instruct",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

The web frontend is served by the binary itself at the same address — no extra step. For hot-reloading frontend development, run the Vite dev server separately (it proxies to a running `camelid serve`):

```bash
cd frontend && npm ci && npm run dev
```

---

## Evidence

Benchmark claims are listed only when raw logs or reproducible commands are committed. If there is no raw log, there is no benchmark claim.

Same-host snapshot on one Apple M4 (10-core GPU, 16 GB), Llama 3.2 3B Instruct Q8_0, greedy sampling, three same-session rounds with alternating runtime order (medians):

| Lane | Camelid | llama.cpp (Metal) | MLX-LM (8-bit) |
| --- | ---: | ---: | ---: |
| Prefill, 601-token prompt (tok/s) | **587.3** | 543.7 | 577.9 |
| Decode, short context (tok/s) | **29.7** | 29.1 | 29.1 |

> **Reading boundary:** a same-session result on one exact model row and one machine, with narrow margins — not a durable or general claim. Some lanes read below the comparators (decode at long context trails MLX-LM), and deeper prompt depths use single warm probes rather than protocol-grade rounds. Full methods, raw logs, per-round detail, and the lanes where Camelid loses are in [`BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md) and the bundles under [`qa/evidence-bundles/`](qa/evidence-bundles/).

Correctness evidence (token-parity gates, per-row validation artifacts) is indexed in [`COMPATIBILITY.md`](COMPATIBILITY.md) and [`CORRECTNESS_v0.1.md`](docs/release/CORRECTNESS_v0.1.md).

### Parity receipts

A parity receipt is a verifiable record of one request: the exact GGUF (by SHA-256), the exact input, and the exact tokens produced. Opt in with `"camelid_receipt": true` on `/v1/chat/completions` or `/v1/completions`, then check it on any machine:

```bash
camelid verify-receipt receipt.json --gguf path/to/exact-model.Q8_0.gguf
```

The verifier recomputes the receipt's digest, confirms your GGUF is the named file, replays the request through Camelid, and re-runs it through llama.cpp — in two isolated passes so each model loads within one model's memory footprint, which lets a 7B receipt verify on a 16 GB Mac. Receipts exist only for deterministic (greedy) runs; sampled runs are stamped `reproducible: false`. **A receipt verifies a single request; it does not change the release ledger or promote any lane.** Details in [`RECEIPTS.md`](RECEIPTS.md).

To measure *any* local runtime — not only Camelid — by determinism, cross-runtime agreement, tokenizer parity, and provability on the same model bytes, see the [conformance suite](docs/CONFORMANCE.md).

---

## Under the hood

For the reader who wants the engineering, not the pitch — a few of the genuinely interesting artifacts:

- **The token-major `output.weight` guardrail.** TinyLlama had perfect tokenizer parity but *wrong first-token logits* until the final vocab projection was read as token-major rows. The fix, the rationale, and the regression guard are pinned in [`DECISIONS.md` D0007](docs/architecture/DECISIONS.md).
- **Reproduce any supported row's parity yourself.** Each row's greedy parity is re-runnable with a committed harness against pinned llama.cpp — methodology and per-row reproduction steps in [`CORRECTNESS_v0.1.md`](docs/release/CORRECTNESS_v0.1.md).
- **One four-row story across every surface.** README, `STATUS.md`, `/api/capabilities`, and the UI are held to the same ledger by the readiness-gate inventory in [`VALIDATION_MATRIX.md`](docs/VALIDATION_MATRIX.md) — drift on any surface is treated as a bug.
- **Sealed, portable parity receipts.** Any greedy request can emit a SHA-256-anchored receipt that re-verifies against llama.cpp on a different machine (incl. a 7B receipt on a 16 GB Mac) — [`RECEIPTS.md`](RECEIPTS.md).
- **Engine internals.** The from-scratch tokenizer, GGUF loader, CPU kernels, and Metal-resident pipeline are mapped in [`ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md).

---

## Documentation

| Doc | What's in it |
|---|---|
| [`SUPPORT_MATRIX_v0.1.md`](SUPPORT_MATRIX_v0.1.md) | Which exact model rows are supported, and with what evidence |
| [`COMPATIBILITY.md`](COMPATIBILITY.md) | The durable support contract |
| [`BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md) | Benchmark snapshots and claim rules |
| [`RECEIPTS.md`](RECEIPTS.md) | Verifiable single-request parity receipts |
| [`docs/CONFORMANCE.md`](docs/CONFORMANCE.md) | Measure any runtime by one ruler |
| [`STATUS.md`](STATUS.md) | Current evidence snapshot and blockers |
| [`ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md) | Implementation architecture |
| [`docs/gemma4-two-mac-cluster.md`](docs/gemma4-two-mac-cluster.md) | Two-Mac distributed serve setup |
| [`RELEASE_NOTES_v0.1.md`](docs/release/RELEASE_NOTES_v0.1.md) | v0.1 release notes |
| [`ROADMAP.md`](ROADMAP.md) | Planned engineering sequence |

Validation for code changes:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

---

## License

Camelid is licensed under the [MIT License](LICENSE).

Camelid's tokenizer, reference compatibility layouts, and validation benchmarks are inspired by and checked against [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (Copyright © 2023–2026 The ggml authors, MIT License). Camelid maintains its own Rust-native codebase while crediting the reference work of the `ggml` ecosystem. Full third-party attributions are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

[ci-badge]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/timtoole02/Camelid/actions/workflows/ci.yml

## Star History

<a href="https://www.star-history.com/?repos=timtoole02%2FCamelid&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=timtoole02/Camelid&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=timtoole02/Camelid&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=timtoole02/Camelid&type=date&legend=top-left" />
 </picture>
</a>
