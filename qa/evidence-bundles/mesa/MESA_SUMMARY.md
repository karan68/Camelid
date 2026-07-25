# MESA — Linux GPU Parity: summary

Bring x86_64 Linux to the Windows plateau: CUDA compiled into the **default** build,
the GPU used automatically when a device is present, and an explicit on/off switch —
without touching the CPU parity reference, the Windows wiring, or the aarch64 (Pi /
NanoCamelid) opt-in path.

Branch: `feat/mesa-linux-gpu-default` (from `main` b9f9a403). Build-wiring + CLI only —
**no kernel, dispatch, quant, or parity-logic changes** (all the detect → default-on →
UI-toggle → `/api/runtime/gpu` machinery already existed; it was just dark on Linux).

## What changed (4 files, additive)

1. **Cargo.toml** — `cudarc` is now **non-optional** on `cfg(all(target_os="linux",
   target_arch="x86_64"))` (mirrors the Windows dep). A second block keeps it `optional`
   for every other non-macOS/non-Windows target, so `cuda = ["dep:cudarc"]` still
   resolves and aarch64/BSD stay opt-in.
2. **build.rs** — injects `cargo:rustc-cfg=feature="cuda"` for x86_64 Linux (same as the
   Windows block), with **no** Optimus/`/STACK` link args (those stay Windows-only).
3. **src/cuda.rs** — adds a sibling `compile_error!` guard so a bare x86_64-Linux build
   can never silently drop CUDA (symmetric to the Windows guard; G4 = included).
4. **src/main.rs** — `serve --gpu auto|on|off` (env `CAMELID_GPU`) seeds the runtime GPU
   switches at startup for headless/agent runs. Also honored on the double-click launch
   path via `GpuMode::from_env()`.

## Precedence contract (G3)

`--gpu` / `CAMELID_GPU` defaults to **`auto`**. In `auto` the atomics are left
uninitialised so they lazy-seed exactly as today (`gpu_accel` from `is_available()`,
hybrid Q8 matmul from `CAMELID_CUDA_Q8`). `on`/`off` are **authoritative at startup** and
override the env seed. The UI `POST /api/runtime/gpu` can still flip state live
afterwards. Deterministic mode and `CAMELID_CUDA_RESIDENT_DECODE=0` still force the GPU
off at their own call sites regardless of the seed.
*(Decision point flagged in the campaign: flag-wins-over-env chosen, per the campaign's
default recommendation. Invert if CI reproducibility should make env win.)*

## §4 consequences (called out honestly)

- **Every** x86_64-Linux build now compiles cudarc (Windows-identical). Runtime CPU-only
  stays fully available via `--gpu off` or a GPU-less host (cudarc no-ops via
  `fallback-dynamic-loading`). A *compile-time* exclusion of cudarc is no longer a default
  option; if ever needed that is a dedicated future feature flag, not MESA.
- aarch64 Linux (Pi / NanoCamelid), macOS (Metal), and Windows are provably unchanged.
- CI ubuntu already ran `--all-features`; after MESA a bare ubuntu `cargo build`/`test`
  also pulls cudarc. Note this graph change in the PR.

## Evidence (this bundle)

| Gate | File | Result |
|------|------|--------|
| G0/G1 | `before_after_cudarc_tree.txt` | x86_64-linux default graph flips to cudarc-present; aarch64 stays clean |
| G2 | `default_build_links_cuda.txt` | default Linux check compiles the CUDA backend (cudarc.rmeta built; guard-didn't-fire proof) |
| G3 | `serve_gpu_help_and_states.json` | `--gpu auto/on/off` verified live on RTX 3060: enabled follows the mode |
| G5 | `validation_suite.txt` | fmt/clippy(-D)/858 lib + 121 bin tests/doc green (Windows-debug stack artifact documented) |
| G6 | `windows_unaffected.md` | Windows Cargo.toml/build.rs/cuda.rs regions byte-identical; Windows default build green |

## Open hardware gate (§6 — cannot close on this Windows host or in CI)

On the RTX Linux box, after MESA lands: confirm a default `cargo build --release` (no
`--features`) yields a binary where `GET /api/runtime/gpu` reports
`available:true, enabled:true` out of the box; re-run the `cuda_parity` gate for
token-identical greedy output vs the CPU/llama.cpp oracle under the default build; and
spot-check `--gpu off` forces the CPU path. File the receipt under
`qa/evidence-bundles/mesa/parity/`.

**Status: landed + locally/cross-checked green on Windows; GPU-runtime parity on x86_64
Linux pending the hardware gate.** (Same honesty bar as the project's other HW-blocked
work.)
