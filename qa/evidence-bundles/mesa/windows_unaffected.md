# MESA — G6: Windows byte-unaffected (diff audit)

**Hard invariant (§4):** the shipped Windows server binary and its dependency graph
must be unaffected. MESA only *adds* an x86_64-Linux target block and a Linux cfg
injection; it touches nothing Windows-scoped.

## Cargo.toml
- `[target.'cfg(windows)'.dependencies]` (memmap2, windows-sys, the non-optional
  Windows `cudarc`): **untouched**. `git diff Cargo.toml | grep 'cfg(windows)'` returns
  nothing — no Windows dependency line changed.
- The only Cargo.toml edits: (1) the old single `cfg(all(not(macos), not(windows)))`
  Linux block was split into a new x86_64-linux **non-optional** block + a reworded
  "other non-macos/non-windows" **optional** block; (2) the `[features]` doc comment was
  updated to say "Windows AND x86_64 Linux" (prose only — `default = []` and
  `cuda = ["dep:cudarc"]` values unchanged).

## build.rs
- The `if target_os == "windows" { ... }` block (cuda cfg + NvOptimus/AmdPowerXpress
  `/EXPORT` + `/STACK:8388608`): **byte-identical**. The MESA insertion is a separate
  `if target_os == "linux" && target_arch == "x86_64"` block placed *after* the Windows
  block and *before* the existing early-return; it deliberately carries **no** Optimus /
  `/STACK` link args (those stay Windows-only). Diff is purely additive.

## src/cuda.rs
- The existing `#[cfg(all(windows, not(feature="cuda")))] compile_error!` Windows guard:
  **untouched**. MESA adds a sibling `#[cfg(all(linux, x86_64, not(feature="cuda")))]`
  guard directly below it.

## Runtime proof
- `cargo build --bin camelid` (DEFAULT, no `--features`) on Windows: **green** — the
  Windows default build is still CUDA-on, and the resulting binary passed the G3
  runtime test on the RTX 3060 (auto/on/off).
- The Windows `compile_error!` guard still compiles satisfied (cuda cfg present), i.e.
  the default-on-Windows guarantee is intact.

## Result
Windows build wiring, dependency graph, and shipped-binary behavior are unchanged.
`git diff --stat`: Cargo.toml, build.rs, src/cuda.rs, src/main.rs — additive only.
