#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${CAMELID_REMOTE_BINDING_TARGET:-/tmp/camelid-remote-binding-target}"
OUT="$TARGET/kotlin-bindings"
APP="$TARGET/kotlin-interop.jar"
KOTLINC="${KOTLINC:-/opt/camelid-toolchains/kotlin-compiler-2.4.10/bin/kotlinc}"
JNA_JAR="${JNA_JAR:-/opt/camelid-libs/jna-5.19.1.jar}"

source "$HOME/.cargo/env"
cd "$ROOT"
rm -rf "$OUT" "$APP"
mkdir -p "$OUT"

CARGO_TARGET_DIR="$TARGET" cargo build -p camelid-remote-crypto-ffi
CARGO_TARGET_DIR="$TARGET" cargo run -p camelid-remote-crypto-bindgen -- \
  generate --no-format \
  --config tools/remote-crypto-bindgen/uniffi-global-jvm.toml \
  --language kotlin \
  --out-dir "$OUT" \
  "$TARGET/debug/libcamelid_remote_crypto_ffi.so"

KOTLIN_SOURCE="$(find "$OUT" -type f -name '*.kt' -print -quit)"
test -n "$KOTLIN_SOURCE"
"$KOTLINC" \
  "$KOTLIN_SOURCE" \
  qa/remote-crypto-bindings/kotlin/Main.kt \
  -classpath "$JNA_JAR" \
  -include-runtime \
  -d "$APP"

LD_LIBRARY_PATH="$TARGET/debug${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  java \
  -Djna.library.path="$TARGET/debug" \
  -classpath "$APP:$JNA_JAR" \
  MainKt