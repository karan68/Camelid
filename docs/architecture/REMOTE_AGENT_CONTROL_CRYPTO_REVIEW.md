# Remote Agent Control Crypto Dependency Review

**Status:** Phase 0 development gate complete. No production security claim.
**Reviewed:** 2026-07-24.
**Parent design:** `REMOTE_AGENT_CONTROL.md`.

## Decision

Use `snow` 0.10.0 as one shared Rust Noise core for the Camelid host and thin iOS/Android
native bindings. Keep `Noise_IK_25519_ChaChaPoly_BLAKE2s`. Disable default features and enable
only `use-curve25519`, `use-chacha20poly1305`, `use-blake2`, and `use-getrandom`.
The `std` feature is intentionally absent because in `snow` 0.10.0 it also enables the optional
`ring` dependency's standard-library feature even though Camelid does not select the ring resolver.

This is a development dependency decision, not an audit result. `snow` explicitly states that
it has not received a formal audit. Remote control remains unshipped and cannot be promoted
until the independent security review in the parent design passes.

## Candidate Evidence

| Candidate | Exact suite | Mobile posture | Decision |
|---|---:|---|---|
| `mcginty/snow` 0.10.0 | Yes | Rust builds for native targets; thin bindings required | Selected shared core |
| `swift-libp2p/swift-noise` at `09e447da` | No BLAKE2s | iOS 13+, IK and Noise vectors, no release | Rejected as crypto owner |
| `samueltangz/swift-noise-protocol` 0.2.1 | No BLAKE2s | iOS 13+, IK; older package | Rejected as crypto owner |
| `sander/noise-kotlin` 1.0.1 | SHA256 only; IK not shipped | Android/JVM primitives supplied by caller; explicitly unaudited | Rejected as crypto owner |
| `rweather/noise-java` at `49377b6` | Yes | Plain Java reference, no published release, inactive since 2022 | Rejected as production dependency |
| `ChainSafe/js-libp2p-noise` 17.0.0 | XX/SHA256 only | React Native support but libp2p-specific | Rejected: wrong protocol |
| `holepunchto/noise-handshake` 4.2.0 | IK/BLAKE2b only | Native Sodium dependency | Rejected: wrong suite |

## Selected Dependency

- Source: `https://github.com/mcginty/snow`
- Crate: `snow` 0.10.0
- License: MIT OR Apache-2.0
- Noise specification: revision 34
- Required primitives: Curve25519, ChaChaPoly, BLAKE2s, OS CSPRNG
- Release: v0.10.0, published 2025-07-19
- Reviewed repository head: `8ac60f51cfe3e010c84f0a454cc575ad9204fa12`
- Known caveat: upstream states that the library has not received a formal audit

The lockfile, not repository head, is the reproducible production input. Updating the crate or
its cryptographic transitive dependencies requires rerunning the interoperability fixtures,
advisory scan, license review, and external-review delta assessment.

## Binding Boundary

UniFFI 0.32.0 (MPL-2.0) generates the Swift and Kotlin adapters from a separate FFI crate. The
generator is a pinned build tool and does not enter the Camelid server or crypto-core runtime
graphs. Generated sources are committed with SHA-256 manifests; foreign callers receive opaque
objects, fieldless bounded errors, and a one-shot private-key handoff rather than a key record.

The mobile bindings may expose only:

- static key generation/import through platform secure storage adapters;
- IK initiator construction with a QR-pinned host public key;
- IK responder construction for host-side tests;
- bounded handshake read/write;
- ordered transport encrypt/decrypt;
- explicit rekey;
- handshake hash and authenticated remote static public key;
- deterministic invalidation plus generated foreign-object disposal.

They must not expose arbitrary Noise pattern or cipher-suite selection to application code. They
must not log keys, plaintext, handshake frames, transport frames, route capabilities, or pairing
secrets. All buffers cross the binding with explicit lengths and maximums.

## Phase 0 Gate

The standalone spike must prove:

1. Rust host responder and Swift/Kotlin initiators produce the same handshake hash.
2. Both directions exchange the committed canonical application fixture after transport split.
3. Flipping every byte class in handshake and transport frames fails authentication.
4. A different QR-pinned host public key fails the first IK message.
5. Reconnection uses fresh ephemeral material and a different transcript/transport ciphertext.
6. Rekey agrees on both peers and rejects pre-rekey ciphertext afterward.
7. Private/static/ephemeral key bytes never enter errors, debug output, or test logs.
8. The core uses OS randomness and the binding exposes only a one-shot private-key handoff.

Gate results on 2026-07-24:

| Gate | Result | Evidence boundary |
|---|---|---|
| Strict protocol/schema/invalid fixtures | PASS | Rust unit tests plus independent AJV/Node validator |
| Noise record/chunk bounds | PASS | Maximum-size round trip and malformed sequence tests |
| Fixed-suite Rust IK/tamper/wrong-key/reconnect/rekey | PASS | `camelid-remote-crypto` tests, including every frame byte position |
| Opaque UniFFI adapter and one-shot key | PASS | Rust 1.89 tests and strict Clippy |
| Generated Kotlin interoperability | PASS | Kotlin 2.4.10/JVM 21 against Linux shared core |
| Generated Swift interoperability | PASS | Swift 6.3.3 against Linux shared core |
| Android-optimized generated source | PASS | Kotlin compiler against pinned Android framework/AndroidX stubs |
| Real iOS/Android native library packaging | DEFERRED | Phase 4, requires target SDKs/signing decisions |
| Keychain/Keystore adapter and real-device key protection | DEFERRED | Phase 4 real-device gate |
| Independent cryptographic/authorization review | BLOCKING RELEASE | Phase 5 |

The final 2026-07-24 RustSec scan reports zero vulnerabilities. Compatible lockfile-only upgrades
closed newly published advisories in the pre-existing Rayon and Tauri graphs (`crossbeam-epoch`
0.9.20, `plist` 1.10.0, `quick-xml` 0.41.0). Twenty-one informational warnings remain in existing
root/Tauri dependencies; none is reachable from the remote protocol, crypto, FFI, or generator
package graphs.

Passing this gate proves protocol validation, shared-core interoperability, generated wrapper
behavior, and language compilation. It does not prove iOS/Android packaging, secure-storage
integration, real-device lifecycle, or formal cryptographic safety. Independent review remains a
release gate.