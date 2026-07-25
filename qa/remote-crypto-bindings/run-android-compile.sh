#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${CAMELID_REMOTE_BINDING_TARGET:-/tmp/camelid-remote-binding-target}"
OUT="$TARGET/android-bindings"
APP="$TARGET/android-binding-compile.jar"
KOTLINC="${KOTLINC:-/opt/camelid-toolchains/kotlin-compiler-2.4.10/bin/kotlinc}"
JNA_JAR="${JNA_JAR:-/opt/camelid-libs/jna-5.19.1.jar}"
ANDROID_JAR="${ANDROID_JAR:-/opt/camelid-libs/android-all-17-robolectric-15733970.jar}"
ANNOTATION_JAR="${ANNOTATION_JAR:-/opt/camelid-libs/annotation-jvm-1.10.0.jar}"

check_sha256() {
  local path="$1" expected="$2"
  local actual
  actual="$(sha256sum "$path" | cut -d' ' -f1)"
  test "$actual" = "$expected" || {
    echo "checksum mismatch for $path" >&2
    exit 1
  }
}

check_sha256 "$ANDROID_JAR" "f6a41ad548bb45cccd3b1d4774cb50d57826dd319b6e5accd6b6269876e12d71"
check_sha256 "$ANNOTATION_JAR" "ddd072ddbb553178e205517ce777b2f05aa9e412982d9ecb4eedc74f1212f697"

source "$HOME/.cargo/env"
cd "$ROOT"
rm -rf "$OUT" "$APP"
mkdir -p "$OUT"

CARGO_TARGET_DIR="$TARGET" cargo build -p camelid-remote-crypto-ffi
CARGO_TARGET_DIR="$TARGET" cargo run -p camelid-remote-crypto-bindgen -- \
  generate --no-format \
  --config tools/remote-crypto-bindgen/uniffi-global.toml \
  --language kotlin \
  --out-dir "$OUT" \
  "$TARGET/debug/libcamelid_remote_crypto_ffi.so"

KOTLIN_SOURCE="$(find "$OUT" -type f -name '*.kt' -print -quit)"
test -n "$KOTLIN_SOURCE"
grep -q '^package ai\.camelid\.remote\.crypto$' "$KOTLIN_SOURCE"
grep -q 'android\.system\.SystemCleaner' "$KOTLIN_SOURCE"

"$KOTLINC" \
  "$KOTLIN_SOURCE" \
  -classpath "$JNA_JAR:$ANDROID_JAR:$ANNOTATION_JAR" \
  -d "$APP"

echo "ANDROID_REMOTE_CRYPTO_BINDING_COMPILE=PASS"