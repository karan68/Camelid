# Remote Agent Control Physical Qualification Receipt

**Date:** 2026-07-25
**PR:** `timtoole02/Camelid#506`
**Base candidate:** `097fa7e05aa4ef6b1b66030560abf27f7dc07ef7`
**Physical UI follow-up:** `699fc2b9` (`mobile: keep approval actions above system navigation`)
**Status:** bounded physical Android pass with one model-context quality finding; full manual matrix remains open

## Environment

- Host: Windows 11, NVIDIA RTX 4060 Laptop GPU;
- device: Realme RMX5003 (`FAT4UCGQSCPN6LGQ`), Android 16, `1080 x 2400`;
- app package: `com.camelid.remote.development`, exact PR-head Metro bundle;
- model: `Qwen3-4B-Q4_K_M.gguf`;
- model SHA-256: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`;
- disposable workspace: `C:\camelid-fork\target\physical-097fa7e0\workspace`;
- disposable authority database: `C:\camelid-fork\target\physical-097fa7e0\authority.sqlite3`;
- temporary Cloudflare quick tunnel to an exact-head local relay. This is not production relay evidence.

Outside-root canary SHA-256 before and after every tested action:
`089fb397c6e42a9e2cf21275b2ab8df13fccb534b9a2602f05208a26d65f6624`.

## Physical Results

- physical phone was connected and independently identified by ADB: PASS;
- stale protected host record was removed through the app; phone returned to zero hosts: PASS;
- fresh QR scanned with the real phone camera; a distinct authorized device
  `7f5c6ec7-d772-4af3-8ee0-4c55b8767a23` was created: PASS;
- fresh Noise IK authenticated; durable `last_seen_at` populated; phone reached Idle with composer enabled: PASS;
- exact capability snapshot showed the disposable canonical workspace, exact model/hash, shell disabled,
  network disabled, and the remote-safe tool list: PASS;
- read-only prompt invoked one `read_file` call/result, advanced cursor `2 -> 10`, returned
  `physical-original`, settled idle, and changed no file: PASS for transport/tool/state safety;
- the model prefixed the requested exact-only answer with explanatory text. This is a model instruction-
  adherence observation, not a protocol or filesystem failure;
- exact write approval showed complete canonical/workspace target, full content `APPROVED-SECOND`, risk,
  digest, and canonical action record; file did not exist before settlement: PASS;
- strong-biometric `Allow once` path was required by code (`biometricsSecurityLevel: strong`, device
  fallback disabled); durable `allow_once` settlement preceded exactly one successful write: PASS;
- resulting `approved.txt` bytes were exactly `APPROVED-SECOND`; outside canary unchanged: PASS;
- one edit approval was allowed to time out while investigating UI geometry; durable settlement was
  `expired`, turn aborted, and content remained unchanged: PASS for expiry safety;
- a fresh edit approval was explicitly denied; durable settlement was `deny`, tool result recorded denial,
  and content remained `APPROVED-SECOND`: PASS;
- a layout-only edit approval was explicitly `abort_turn`; no mutation occurred: PASS;
- New created active session `9d69a77d-3283-41ae-ad67-755edda41816` at cursor 2 while the prior history
  remained at cursor 43: PASS;
- selecting the dormant history replayed its transcript, hid the composer, and displayed
  `Replay-only history`: PASS;
- explicit Continue restored session `60ae6d1c-fafa-4845-9d89-8bbec49bdb42` as active: PASS;
- continuation advanced the restored session `43 -> 48`, while the dormant session remained cursor 2:
  PASS for history activation, isolation, and real-time catalog refresh;
- the context-dependent continuation answer incorrectly claimed `DENIED-FAST` had been written, although
  the edit was denied and the file remained `APPROVED-SECOND`: **MODEL-CONTEXT QUALITY FINDING**;
- exact-head host restart restored the same host `2fa8f53f-5fab-497e-ac4a-ea42fea22e41`, active session,
  relay route, cursor 48, device grant, and complete transcript after fresh reconnect: PASS;
- physical testing exposed approval Deny/Abort buttons partially covered by Android system navigation.
  Commit `699fc2b9` applies `useSafeAreaInsets()` to the absolute action tray. All three controls are now
  fully visible and tappable above navigation at `1080 x 2400`: PASS after fix;
- fixed approval-modal PNG SHA-256:
  `8ff096443515b74313bf941fe1ba7b7999472cf70d068ad7545f20b13c89caa4`.

The local model worker logged one `aborted` turn caused by a compressed ADB input sequence during test
setup. Repeating the flow with keyboard dismissal and submission as separate actions reached the expected
approval state. This is not counted as a product failure.

## Automated Candidate Gates

Before the physical follow-up:

- root remote tests: 42/42;
- protocol: 19/19;
- store: 25/25;
- relay: 11/11 plus Windows persistence;
- crypto/FFI: 10/10;
- mobile: 66/66, TypeScript, Expo lint, native artifact seal, production audit;
- frontend build, schema fixtures, strict Clippy, and rustfmt;
- GitHub Actions: 11/11 jobs successful across macOS, Ubuntu, and Windows.

After the safe-area follow-up `699fc2b9`:

- mobile tests: 66/66;
- TypeScript: PASS;
- Expo lint: PASS;
- editor diagnostics: clean;
- physical fixed-modal geometry: PASS.

## Remaining Non-Claims and Required Work

This run does not complete the 48-scenario manual plan. The PR must remain `[DON'T MERGE]` until the
remaining priority scenarios are exercised, especially:

- fingerprint comparison independently witnessed during pairing;
- biometric cancellation/failure, lock/unlock, and app kill/relaunch during approval;
- Wi-Fi/cellular transitions and long-idle relay behavior;
- two-device races for prompts, approvals, and active-session switches;
- outside-root/symlink traversal on a physical run;
- disk-full, corrupt/newer database, app upgrade/reinstall, and clean-machine packaging;
- resolution or explicit product acceptance of the model-context quality finding;
- independent cryptographic and authorization review.
