#!/usr/bin/env bash
set -euo pipefail

SWIFT_VERSION="6.3.3"
SWIFT_PLATFORM="ubuntu24.04"
DOWNLOADS="/opt/camelid-downloads"
TOOLCHAINS="/opt/camelid-toolchains"
ARCHIVE_NAME="swift-$SWIFT_VERSION-RELEASE-$SWIFT_PLATFORM.tar.gz"
BASE_URL="https://download.swift.org/swift-$SWIFT_VERSION-release/ubuntu2404/swift-$SWIFT_VERSION-RELEASE"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run as root so the test-only Swift toolchain can be installed under /opt" >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg
install -d "$DOWNLOADS" "$TOOLCHAINS"

archive="$DOWNLOADS/$ARCHIVE_NAME"
signature="$archive.sig"
keys="$DOWNLOADS/swift-org-keys.asc"
curl -fL --retry 4 --retry-all-errors -o "$archive" "$BASE_URL/$ARCHIVE_NAME"
curl -fL --retry 4 --retry-all-errors -o "$signature" "$BASE_URL/$ARCHIVE_NAME.sig"
curl -fL --retry 4 --retry-all-errors --compressed -o "$keys" \
  "https://www.swift.org/keys/all-keys.asc"

keyring="$DOWNLOADS/swift-toolchain-keyring.gpg"
rm -f "$keyring"
gpg --batch --no-default-keyring --keyring "$keyring" --import "$keys"
gpg --batch --no-default-keyring --keyring "$keyring" --verify "$signature" "$archive"

destination="$TOOLCHAINS/swift-$SWIFT_VERSION"
rm -rf "$destination" "$TOOLCHAINS/swift-unpack"
install -d "$TOOLCHAINS/swift-unpack"
tar -xzf "$archive" -C "$TOOLCHAINS/swift-unpack" --strip-components=1
mv "$TOOLCHAINS/swift-unpack" "$destination"
test -x "$destination/usr/bin/swiftc"
"$destination/usr/bin/swift" --version
echo "REMOTE_BINDING_SWIFT_TOOLCHAIN_INSTALLED=PASS"