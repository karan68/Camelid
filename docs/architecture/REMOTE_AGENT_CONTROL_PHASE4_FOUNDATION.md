# Remote Agent Control Phase 4 Foundation Receipt

**Date:** 2026-07-25
**Status:** Foundation in progress; Phase 4 gate not complete
**Environment:** Windows host with pinned Android SDK/NDK and API 36 emulator; no Apple SDK, Xcode, or attached real mobile device

## Implemented

- Expo SDK 57 React Native development-build project under `mobile/`.
- SDK-matched camera, SecureStore, local authentication, dev client, Router, fonts, Jest, and ESLint dependencies.
- Local Expo module for Android and Apple with an opaque handle contract. JavaScript cannot request or receive device private key bytes.
- Android links sealed arm64/x86_64 Rust and generated Kotlin UniFFI artifacts. Swift remains fail-closed until Apple artifacts are linked.
- Android static private keys are wrapped by per-device AES-256-GCM keys in Android Keystore and stored outside backup.
- QR parser matching the v1 strict fields, bounds, secure relay scheme, host UUID, route token, host public key, pairing secret, and exclusive expiry.
- QR-facing `/v1/connect/:route_id` relay alias with the same blind device bridge.
- Protected host metadata excluding pairing secrets and private keys. Replay cursors cannot move backwards.
- Pure reducer for duplicate/gap handling, unknown observations, terminal ordering, approval settlement, approval schema pinning, conflicting approval identity failure, and cancellation settlement.
- Strict mobile protocol decoders for pairing response, event batch, replay end, and command result.
- Canonical JSON and strict builders for start, cancel, approval decision, and replay request.
- Strong-biometric approval gate with no device fallback when enabled.
- Host-list and QR-pairing preview UI with stable 390 x 844 layout and no overflow in a web-render smoke.
- Binary WebSocket pairing coordinator with strict host binding, timeout, rollback, and opaque-handle cleanup.
- Post-pair WebSocket transport with fresh IK, immediate replay request, binary Noise records, authenticated v1 chunk framing/reassembly, strict envelopes, and terminal cleanup.
- Composed host pairing operation from blind relay through Noise, one-time secret, local confirmation, durable grant, strict PairResponse, and authenticated post-pair traffic.
- First-class development-only `camelid agent host` CLI with no yolo, auto-approve, unrestricted filesystem, MCP, subagent, or GUI controls.
- Windows DPAPI-protected host static key and relay host bearer; SQLite stores only public identity, route metadata, and opaque secret references.
- Authenticated per-device host dispatch with strict chunk reassembly, replay, start/cancel/approval commands, one model worker, durable accepted/applied command results, and no offline queue.
- Exact identity-bound session reuse, interrupted-turn invalidation, protected route reuse, and optional relay route persistence through `CAMELID_RELAY_STATE`.
- Scannable terminal pairing QR, caller-supplied bounded reconnect/backoff policy, clean cancellation during backoff, and replay partitioning by both 256-event and 1,114,112-byte limits.
- Durable `host.capabilities`/`session.armed` bootstrap events expose exact workspace, model artifact, tools, file scope, shell enforcement, and network-tool posture as descriptive mobile context.
- Mobile session controller with strict envelope identity checks, live/replay race buffering, monotonic cursor persistence, foreground fresh-IK replay, and background transport teardown.
- Native Session, Activity, Settings, and full exact-action Approval surfaces; Allow Once requires the strong-biometric gate and controls become inert after settlement/disconnect.

## Executable Evidence

- Rust shared protocol: 17 tests and strict Clippy.
- Relay after QR connect alias: 10 tests and strict Clippy.
- Root local pairing: 3 tests and strict root Clippy.
- Mobile: 59 Jest tests, TypeScript `--noEmit`, Expo lint, Expo config, Android autolinking search, and three-file native artifact seal.
- Root remote namespace: 30 tests plus 2 CLI contract tests; remote store: 19 tests; relay: 11 tests; root and relay strict Clippy pass.
- Windows host binary compile and DPAPI encrypted-file round trip: PASS.
- npm production audit: zero vulnerabilities after an API-compatible dependency-scoped `xcode -> uuid@11.1.1` override.
- Android arm64 development APK: PASS; only `arm64-v8a`, sealed Rust transform verified.
- API 36 x86_64 emulator instrumentation: 1 test, 0 failures; Keystore + Noise + transport + deletion refusal PASS.
- Native Android app launch and host/pair/session/activity/settings screen hierarchy: PASS; `Protected transport ready` observed. Session-route controls fit the API 36 `1080 x 2400` viewport with no hierarchy overlap.
- Detailed Android receipt: `docs/architecture/REMOTE_AGENT_CONTROL_PHASE4_ANDROID_RECEIPT.md`.

## Open Gate

Phase 4 is not complete. Android emulator/build evidence now exists, but Windows cannot produce an iOS/Xcode build and no physical mobile device is attached. The following remain required:

- link the generated Swift UniFFI binding and Apple Rust XCFramework;
- implement native iOS Keychain device-only storage;
- implement generic push registration/delivery and notification reconciliation;
- add local host management, device revocation, emergency disable, and retention controls;
- compile the iOS development build;
- run every Phase 4 scenario on real devices, including network switch, lock, kill/relaunch, stale approval, overflow, dropped push, and lost-phone revocation.

The web render remains layout evidence only. Android emulator evidence is native build/runtime evidence but is not hardware-backed, biometric, lifecycle, or real-device evidence.
