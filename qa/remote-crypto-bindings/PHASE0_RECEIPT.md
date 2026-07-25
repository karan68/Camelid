# Remote Control Phase 0 Receipt

**Date:** 2026-07-24
**Scope:** Development protocol and cross-language crypto gate only.
**Non-claim:** No remote host, relay, mobile app, real-device qualification, or shipped capability.

## Pinned Inputs

- Rust toolchain/MSRV gate: 1.89.0
- Repository toolchain used for local Linux execution: 1.95.0
- Noise: `snow` 0.10.0, minimal fixed-suite features
- Binding generator/runtime: UniFFI 0.32.0
- Kotlin: 2.4.10, archive SHA-256
  `473dd66c7a3ef4b182065b3da670466c1bf2773a9dbb0ed8b33a39fe9d4f876d`
- JNA: 5.19.1, SHA-256
  `4fb141dd8ef6b0585ffceea4bc49602fbc6312fa977e2c488794ea3e6aafecae`
- Swift: official 6.3.3 Ubuntu 24.04 archive with detached Swift.org signature verification;
  executed on the newer Ubuntu 26.04 WSL userspace as a Linux binding gate only
- Android compile framework: Robolectric Android 17 `15733970`, SHA-256
  `f6a41ad548bb45cccd3b1d4774cb50d57826dd319b6e5accd6b6269876e12d71`
- AndroidX annotation JVM 1.10.0, SHA-256
  `ddd072ddbb553178e205517ce777b2f05aa9e412982d9ecb4eedc74f1212f697`

## Executable Results

- Remote protocol: 14 tests passed.
- Fixed-suite Rust crypto core: IK identity, bidirectional transport, wrong pinned host,
  every handshake/transport byte position tamper, reconnect freshness, bounds, and synchronized
  rekey passed (6 tests).
- UniFFI adapter: 4 tests passed on Rust 1.89; strict Clippy passed.
- Kotlin generated binding: `KOTLIN_REMOTE_CRYPTO_INTEROP=PASS`, including the committed canonical
  start-turn fixture.
- Swift generated binding: `SWIFT_REMOTE_CRYPTO_INTEROP=PASS`, including the committed canonical
  start-turn fixture.
- Android-optimized Kotlin source: `ANDROID_REMOTE_CRYPTO_BINDING_COMPILE=PASS`.
- Protocol schemas: 6 valid checks, 3 invalid checks, approval digest
  `sha256:995876c30076bb24a3273e215dcd0839eceef86fcb97bce4c9a1a8038fab6fdd`.
- RustSec: zero vulnerabilities after compatible lockfile upgrades to `crossbeam-epoch` 0.9.20,
  `plist` 1.10.0, and `quick-xml` 0.41.0. The scan reports 21 allowed informational warnings in
  pre-existing root/Tauri graphs; none belongs to the remote protocol/crypto/FFI packages.

Final consolidated package/test counts and generated-binding hashes are produced by the branch CI
and the committed binding manifest. This receipt deliberately does not substitute Linux language
execution for iOS/Android real-device evidence.