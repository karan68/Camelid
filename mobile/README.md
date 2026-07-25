# Camelid Remote Mobile

Development-only React Native controller foundation for Camelid remote agent sessions.

## Current Gate

Implemented and locally validated:

- Expo SDK 57 development-build project for iOS and Android;
- strict QR parser and QR-facing relay URL derivation;
- protected host metadata through Expo SecureStore;
- pure replay/live event reducer with gap and duplicate handling;
- strict pairing response, event batch, replay end, and command result decoders;
- canonical start, cancel, approval, and replay request builders;
- strong-biometric approval gate;
- host list and QR pairing preview UI;
- local Expo module boundary with branded opaque key/handshake/transport handles;
- sealed Android Rust/UniFFI artifacts with Android Keystore-backed private-key wrapping;
- binary WebSocket initial pairing flow with strict PairResponse and rollback;
- post-pair fresh-IK reconnect, replay request, and authenticated binary chunk transport;
- session, history, activity, settings, and exact-action approval screens;
- bounded host-scoped session catalogs with isolated per-history replay projections;
- atomic new-session creation and explicit continuation of dormant remote histories;
- real-time catalog refresh after committed host event bursts;
- arm64 Android development APK and API 36 emulator instrumentation.
- physical Android pair/revoke/re-pair, restart/replay, multi-history switching, and contextual continuation.

Not yet implemented or evidenced:

- iOS Rust artifact linking and Keychain persistence;
- push provider integration;
- iOS development build;
- retention/delete controls;
- the remaining biometric, lock/relaunch, accessibility, and network-transition matrix.

The app must not be run through Expo Go as a security qualification surface. Use a development build after native artifacts are linked.

## Local Checks

```powershell
npm.cmd ci --ignore-scripts
npm.cmd test
.\node_modules\.bin\tsc.cmd --noEmit
npm.cmd run lint
npm.cmd audit --omit=dev --audit-level=moderate
npx.cmd expo config --type public
npx.cmd expo-modules-autolinking search --platform android
npm.cmd run verify:native
```

Android build/emulator details and artifact hashes are recorded in `docs/architecture/REMOTE_AGENT_CONTROL_PHASE4_ANDROID_RECEIPT.md`.

`npm` uses a dependency-scoped override from `xcode@3.0.1` to API-compatible `uuid@11.1.1`, closing the upstream UUID advisory without changing Expo SDK versions.
