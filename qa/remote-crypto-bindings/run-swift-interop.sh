#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${CAMELID_REMOTE_BINDING_TARGET:-/tmp/camelid-remote-binding-target}"
OUT="$TARGET/swift-bindings"
APP="$TARGET/swift-interop"

if [[ -f "$HOME/.cargo/env" ]]; then source "$HOME/.cargo/env"; fi
if [[ -f "${SWIFTLY_HOME_DIR:-$HOME/.local/share/swiftly}/env.sh" ]]; then
  source "${SWIFTLY_HOME_DIR:-$HOME/.local/share/swiftly}/env.sh"
fi
SWIFT_TOOLCHAIN_BIN="${SWIFT_TOOLCHAIN_BIN:-/opt/camelid-toolchains/swift-6.3.3/usr/bin}"
if [[ -d "$SWIFT_TOOLCHAIN_BIN" ]]; then
  export PATH="$SWIFT_TOOLCHAIN_BIN:$PATH"
fi
cd "$ROOT"
rm -rf "$OUT" "$APP"
mkdir -p "$OUT"

case "$(uname -s)" in
  Darwin)
    LIBRARY="$TARGET/debug/libcamelid_remote_crypto_ffi.dylib"
    RUNTIME_PATH_VAR="DYLD_LIBRARY_PATH"
    ;;
  Linux)
    LIBRARY="$TARGET/debug/libcamelid_remote_crypto_ffi.so"
    RUNTIME_PATH_VAR="LD_LIBRARY_PATH"
    ;;
  *)
    echo "unsupported Swift interoperability host" >&2
    exit 1
    ;;
esac

CARGO_TARGET_DIR="$TARGET" cargo build -p camelid-remote-crypto-ffi
CARGO_TARGET_DIR="$TARGET" cargo run -p camelid-remote-crypto-bindgen -- \
  generate --no-format \
  --config tools/remote-crypto-bindgen/uniffi-global.toml \
  --language swift \
  --out-dir "$OUT" \
  "$LIBRARY"

swiftc \
  -I "$OUT" \
  -Xcc "-fmodule-map-file=$OUT/CamelidRemoteCryptoFFI.modulemap" \
  -L "$TARGET/debug" \
  -lcamelid_remote_crypto_ffi \
  "$OUT/CamelidRemoteCrypto.swift" \
  qa/remote-crypto-bindings/swift/main.swift \
  -o "$APP"

if [[ "$RUNTIME_PATH_VAR" == "DYLD_LIBRARY_PATH" ]]; then
  DYLD_LIBRARY_PATH="$TARGET/debug${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" "$APP"
else
  LD_LIBRARY_PATH="$TARGET/debug${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$APP"
fi