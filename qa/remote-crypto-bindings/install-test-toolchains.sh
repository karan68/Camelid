#!/usr/bin/env bash
set -euo pipefail

KOTLIN_VERSION="2.4.10"
KOTLIN_SHA256="473dd66c7a3ef4b182065b3da670466c1bf2773a9dbb0ed8b33a39fe9d4f876d"
JNA_VERSION="5.19.1"
JNA_SHA256="4fb141dd8ef6b0585ffceea4bc49602fbc6312fa977e2c488794ea3e6aafecae"
ANDROID_ALL_VERSION="17-robolectric-15733970"
ANDROID_ALL_SHA256="f6a41ad548bb45cccd3b1d4774cb50d57826dd319b6e5accd6b6269876e12d71"
ANNOTATION_VERSION="1.10.0"
ANNOTATION_SHA256="ddd072ddbb553178e205517ce777b2f05aa9e412982d9ecb4eedc74f1212f697"

DOWNLOADS="/opt/camelid-downloads"
TOOLCHAINS="/opt/camelid-toolchains"
LIBS="/opt/camelid-libs"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run as root so test-only toolchains can be installed under /opt" >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg unzip
install -d "$DOWNLOADS" "$TOOLCHAINS" "$LIBS"

download_verified() {
  local url="$1" path="$2" expected="$3"
  curl -fL --retry 4 --retry-all-errors -o "$path" "$url"
  printf '%s  %s\n' "$expected" "$path" | sha256sum -c -
}

kotlin_archive="$DOWNLOADS/kotlin-compiler-$KOTLIN_VERSION.zip"
download_verified \
  "https://github.com/JetBrains/kotlin/releases/download/v$KOTLIN_VERSION/kotlin-compiler-$KOTLIN_VERSION.zip" \
  "$kotlin_archive" \
  "$KOTLIN_SHA256"
rm -rf "$TOOLCHAINS/kotlin-compiler-$KOTLIN_VERSION" "$TOOLCHAINS/kotlin-unpack"
unzip -q "$kotlin_archive" -d "$TOOLCHAINS/kotlin-unpack"
mv "$TOOLCHAINS/kotlin-unpack/kotlinc" "$TOOLCHAINS/kotlin-compiler-$KOTLIN_VERSION"
rmdir "$TOOLCHAINS/kotlin-unpack"

download_verified \
  "https://repo1.maven.org/maven2/net/java/dev/jna/jna/$JNA_VERSION/jna-$JNA_VERSION.jar" \
  "$LIBS/jna-$JNA_VERSION.jar" \
  "$JNA_SHA256"
download_verified \
  "https://repo1.maven.org/maven2/org/robolectric/android-all/$ANDROID_ALL_VERSION/android-all-$ANDROID_ALL_VERSION.jar" \
  "$LIBS/android-all-$ANDROID_ALL_VERSION.jar" \
  "$ANDROID_ALL_SHA256"
download_verified \
  "https://dl.google.com/dl/android/maven2/androidx/annotation/annotation-jvm/$ANNOTATION_VERSION/annotation-jvm-$ANNOTATION_VERSION.jar" \
  "$LIBS/annotation-jvm-$ANNOTATION_VERSION.jar" \
  "$ANNOTATION_SHA256"

"$TOOLCHAINS/kotlin-compiler-$KOTLIN_VERSION/bin/kotlinc" -version
echo "REMOTE_BINDING_TEST_TOOLCHAINS_BOOTSTRAPPED=PASS"