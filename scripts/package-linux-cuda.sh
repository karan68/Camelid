#!/usr/bin/env bash
#
# Stage the NVRTC redistributables next to the `camelid` binary so a downloaded
# x86_64 Linux release runs on the GPU with only the NVIDIA *driver* installed
# (no CUDA Toolkit required on the end user's machine).
#
# Why this is needed at all: camelid links cudarc with `fallback-dynamic-loading`,
# so it dlopen's the CUDA driver (libcuda.so.1, shipped with the GPU driver) and
# the NVRTC runtime compiler at startup. The resident decode engine compiles its
# kernels through NVRTC — and NVRTC ships with the CUDA *Toolkit*, not the driver.
# Since CUDA is compiled into the DEFAULT x86_64 Linux build, without this pair the
# common driver-only host detects the GPU and then quietly runs on the CPU.
#
# How the binary finds them: it is linked with `-Wl,-rpath,$ORIGIN`
# (.cargo/config.toml), so its own directory is on the run-time search path and
# dlopen resolves these without the user setting LD_LIBRARY_PATH. The Windows
# equivalent (`ensure_cuda_runtime_on_path`, which prepends the exe directory to
# PATH) cannot work here: glibc fixes the loader search path at process start, so
# changing the environment in-process has no effect on later dlopen calls.
#
# The CUDA driver (libcuda.so.1) is NOT redistributable and is NOT copied — it
# comes from the user's NVIDIA driver install.
#
# Usage: scripts/package-linux-cuda.sh <dist-dir> [cuda-lib-dir]
set -euo pipefail

dist="${1:?usage: package-linux-cuda.sh <dist-dir> [cuda-lib-dir]}"
[ -d "$dist" ] || { echo "package-linux-cuda: '$dist' is not a directory" >&2; exit 1; }

# Prefer an explicit dir, then the toolkit CUDA_PATH the installer action exports,
# then the usual system locations.
candidates=()
[ "${2:-}" != "" ] && candidates+=("$2")
[ "${CUDA_PATH:-}" != "" ] && candidates+=("$CUDA_PATH/lib64" "$CUDA_PATH/lib")
candidates+=(/usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)

src=""
for dir in "${candidates[@]}"; do
  if [ -d "$dir" ] && compgen -G "$dir/libnvrtc.so*" > /dev/null; then
    src="$dir"
    break
  fi
done

if [ -z "$src" ]; then
  echo "package-linux-cuda: no libnvrtc.so* found in: ${candidates[*]}" >&2
  echo "package-linux-cuda: install the CUDA nvrtc component before packaging." >&2
  exit 1
fi

echo "package-linux-cuda: staging NVRTC from $src"
copied=0
for pattern in 'libnvrtc.so*' 'libnvrtc-builtins.so*'; do
  for lib in "$src"/$pattern; do
    [ -e "$lib" ] || continue
    # `.alt.` variants are an alternate JIT backend cudarc never asks for; skipping
    # them keeps ~90 MB out of every download.
    case "$(basename "$lib")" in *.alt.so*) continue ;; esac
    # -P, NOT -L. A toolkit ships libnvrtc.so.<major> as a SYMLINK to
    # libnvrtc.so.<major>.<minor>.<patch>, and this glob matches both, so
    # dereferencing copied the same 106 MB library twice — v0.4.3 shipped 113 MB of
    # duplication that way. Preserving the link keeps one real file plus a cheap
    # symlink beside it, which resolves identically under $ORIGIN.
    cp -P "$lib" "$dist/"
    echo "  + $(basename "$lib")"
    copied=$((copied + 1))
  done
done

# libnvrtc.so.<major> is the soname cudarc actually dlopen's; a stage that missed it
# would produce a tarball that still cannot reach the GPU, so fail rather than ship.
if ! compgen -G "$dist/libnvrtc.so.*" > /dev/null; then
  echo "package-linux-cuda: staged $copied file(s) but no libnvrtc.so.<major> soname" >&2
  exit 1
fi

# A symlink that resolves nowhere is worse than no symlink: dlopen would fail and the
# GPU would silently never engage. Every staged link must land on a staged file.
for link in "$dist"/libnvrtc*.so*; do
  [ -L "$link" ] || continue
  if [ ! -e "$dist/$(basename "$(readlink "$link")")" ]; then
    echo "package-linux-cuda: $(basename "$link") -> $(readlink "$link") is dangling" >&2
    exit 1
  fi
done

# Regression guard for the v0.4.3 duplication: two staged REAL files of identical size
# means a symlink was dereferenced again. Cheap to check, and it fails the release
# rather than quietly doubling every download.
dupes=$(find "$dist" -maxdepth 1 -type f -name 'libnvrtc*.so*' -printf '%s\n' | sort | uniq -d | wc -l)
if [ "$dupes" -ne 0 ]; then
  echo "package-linux-cuda: staged duplicate NVRTC payloads (same-size real files);" >&2
  echo "package-linux-cuda: symlinks must be preserved with cp -P, not followed." >&2
  find "$dist" -maxdepth 1 -name 'libnvrtc*.so*' -printf '  %y %10s %f\n' >&2
  exit 1
fi

echo "package-linux-cuda: staged $copied NVRTC file(s) into $dist"
du -sh "$dist" | awk '{print "package-linux-cuda: dist is " $1}'
