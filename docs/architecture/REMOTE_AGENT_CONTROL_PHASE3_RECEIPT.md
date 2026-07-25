# Remote Agent Control Phase 3 Development Receipt

**Date:** 2026-07-24
**Status:** Development gate complete; not shipped or promoted
**Environment:** WSL Ubuntu, repository Rust 1.95.0 toolchain, isolated Cargo target

## Scope

Phase 3 adds the internal relay and end-to-end transport foundation:

- standalone bounded blind relay with Axum HTTP/WebSocket adapter;
- separate host bearer and 128-bit QR `route_id` device routing capability;
- no queue while the host is offline;
- strict binary frame bounds and bounded Tokio channels;
- fixed-category push registration and notification capabilities;
- outbound host WebSocket connection with cancellation-aware bounded reconnect;
- fixed-suite Noise IK sessions keyed by relay connection ID;
- short-lived, single-use pairing secret and explicit local confirmation;
- durable device registration/revocation and immediate live connection closure;
- exact replay of host-authoritative events after transport or process loss.

The relay never imports or parses the Camelid inner protocol. The device route token permits only a relay connection attempt. Noise authentication and the local SQLite device grant remain required before command bytes can reach the local host.

## Gate Command

```text
cargo fmt --all
cargo fmt --all -- --check
cargo test -p camelid-remote-protocol -p camelid-remote-crypto -p camelid-remote-store -p camelid-relay
cargo test --lib chat::remote_pairing::tests -- --nocapture
cargo test --lib chat::remote_transport::tests -- --nocapture
cargo test --lib chat::remote_host::tests -- --nocapture
cargo clippy -p camelid-remote-protocol -p camelid-remote-crypto -p camelid-remote-store -p camelid-relay --all-targets -- -D warnings
cargo clippy --lib --tests -- -D warnings
```

## Result

All commands passed.

- relay: 10 tests;
- shared crypto: 6 tests;
- protocol: 15 tests;
- durable store: 15 tests;
- local pairing coordinator: 3 tests;
- host transport and Noise integration: 8 tests;
- durable local host: 13 tests;
- total: 70 explicitly executed tests;
- formatting: clean;
- strict Clippy: clean.

## Scenario Evidence

- Relay ciphertext opacity: `opaque_noise_records_round_trip_without_plaintext_at_the_relay`.
- Modified ciphertext is terminal: `noise_sessions_authorize_tamper_terminally_and_revoke_live_connections`.
- Wrong pinned host key fails IK: `wrong_host_key_and_unregistered_device_cannot_create_noise_sessions`.
- Expired, over-attempted, cancelled, rejected, and consumed pairing offers fail closed: pairing coordinator and transport tests.
- Local confirmation is bound to relay connection ID and Noise transcript fingerprint.
- Revocation removes durable authority, drops Noise state, and closes the exact open relay device socket.
- A valid stolen route token with an unregistered device key cannot create a command-bearing session.
- Host reconnect uses bounded exponential delay with jitter, honors cancellation, and resets old Noise state.
- Committed host events replay exactly after complete host destruction and SQLite reopen; relay state is never authoritative.
- Oversized and slow-client queues fail within fixed bounds.
- Offline host refuses device connections without queueing frames.
- Push registration requires the exact QR route token; the provider receives only a fixed enum category and platform token.

## Non-Claims

This receipt does not add or validate:

- a public `camelid agent host` command;
- production host key storage adapters for Windows Credential Manager, macOS Keychain, or Linux secret service;
- a production relay operator, domain, deployment, account model, rate policy, or push provider;
- iOS or Android product UI or real-device qualification;
- packaging, upgrade, recovery, or clean-machine install behavior;
- independent cryptographic/security review;
- a public capability row, release claim, or support promotion.

Those remain later-phase gates. `snow` remains explicitly unaudited upstream and cannot be promoted without the independent Phase 5 review.
