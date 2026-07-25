# Remote Agent Control Manual Qualification Plan

**Status:** pre-merge development qualification
**Scope:** Windows host plus physical Android client
**Rule:** a failed scenario stops promotion. Record the failure before changing code.

This plan supplements automated protocol, store, relay, host, and mobile tests. It does not turn a
development build or temporary relay into a production claim.

## 1. Required Test Matrix

Run the priority-0 scenarios on every candidate commit. Run the full matrix before asking for
security review or changing the pull request from draft.

| Axis | Required values |
|---|---|
| Host | Windows 11 reference machine; clean disposable workspace |
| Mobile | Physical Android reference device; API 24 or newer |
| Network | same Wi-Fi, Wi-Fi to cellular, relay interruption, host interruption |
| Capability | shell/network both off; each independently on; both on |
| Session | empty, completed, failed, dormant, active turn, waiting approval |
| Device | no device, one device, revoked device, two authorized devices |
| Install state | clean app data, existing protected record, app upgrade/reinstall |

## 2. Safety and Evidence Rules

1. Use a disposable workspace copy. Never run destructive cases against an important checkout.
2. Put an immutable canary outside the workspace and hash it before and after every mutation case.
3. Use a dedicated remote database with `--db-path`; preserve it with the evidence bundle.
4. Do not capture QR payloads, pairing secrets, host bearers, private keys, or authorization headers.
5. Stop immediately if a file changes before durable approval or after denial.
6. Stop immediately if a revoked device reconnects, a dormant history accepts a command, or more
   than one unfinished turn exists.
7. A toast or UI label is not proof. Verify filesystem state, event cursor, session state, and the
   device inventory from an independent host surface.

For every scenario record:

- candidate commit SHA and dirty/clean status;
- host OS, model filename and SHA-256, workspace canonical path, database path;
- device model, Android version, app/APK SHA-256, network type;
- relay build/version and whether it is temporary or operator-owned;
- start/end time, active session UUID, cursor before/after;
- expected result, actual result, PASS/FAIL/BLOCKED;
- relevant logs with secrets redacted;
- screenshot filename, byte length, and SHA-256 when UI behavior is part of the gate.

## 3. Reference Setup

Use values appropriate to the test machine. Do not paste these placeholders literally.

```powershell
$Root = 'C:\remote-qualification'
$Workspace = "$Root\workspace"
$Database = "$Root\authority.sqlite3"
$Model = 'C:\models\Qwen3-4B-Q4_K_M.gguf'
$Relay = 'wss://RELAY_HOST'
$Camelid = 'C:\path\to\camelid.exe'

New-Item -ItemType Directory -Force $Root, $Workspace | Out-Null
Set-Content "$Workspace\inside.txt" 'inside-original'
Set-Content "$Root\outside-canary.txt" 'outside-original'
Get-FileHash -Algorithm SHA256 "$Root\outside-canary.txt"

$env:CAMELID_RELAY_ENROLLMENT_TOKEN = 'TYPE_IN_OPERATOR_TERMINAL_ONLY'
& $Camelid agent host `
  --model $Model `
  --workdir $Workspace `
  --db-path $Database `
  --relay-url $Relay `
  --addr 127.0.0.1:8231 `
  --reconnect-initial-ms 250 `
  --reconnect-max-ms 5000 `
  --reconnect-jitter-percent 20 `
  --relay-keepalive-ms 15000
```

Open `http://127.0.0.1:8231/#remote` locally. The management API must not be exposed through a
tunnel. For an installed Android development client, start Metro from `mobile/` and connect the
physical device using the development-client instructions in `mobile/README.md`.

Useful independent host checks:

```powershell
& $Camelid agent remote devices --db-path $Database
Get-FileHash -Algorithm SHA256 "$Root\outside-canary.txt"
```

## 4. Priority-0 Normal Scenarios

### N01 - Cold start and immutable capability snapshot

1. Start the host with shell and network omitted.
2. Open the local Remote view before pairing.
3. Record host ID, active session ID, state, cursor, workspace, model identity, tools, shell, and
   network values.
4. Pair later and compare the phone Settings capability display with the local display.

Pass:

- host becomes armed and idle;
- workspace/model/hash match the command line and exact artifact;
- shell and network are disabled and absent from enabled tools;
- the phone cannot widen any value;
- ordinary `serve` keeps Workspace CLI available, while `agent host` does not issue a Workspace
  CLI credential or authorize Workspace CLI requests.

### N02 - Fresh pair with local fingerprint confirmation

1. Clear app data or remove the prior host through Settings.
2. Select **Pair new device** once.
3. Scan the QR with the physical app and enter a unique label.
4. Compare the requesting label/fingerprint on both trusted surfaces.
5. Approve locally.

Pass:

- QR is created only by the explicit action and expires visibly;
- status polling never reveals the QR payload;
- no device authority exists before local approval;
- one new authorized device UUID appears;
- the phone completes fresh Noise IK, replays from zero, and shows idle state/capabilities.

### N03 - Read-only tool turn

Prompt: `Read inside.txt and reply with only its exact contents.`

Pass:

- turn is accepted only on the active session;
- read runs without an approval card;
- answer is `inside-original`;
- cursor advances monotonically through accepted, tool activity, answer, and finished events;
- session returns to idle and the History row updates without reconnect/manual refresh.

### N04 - Approved write executes exactly once

Prompt: `Create approved.txt with exact content APPROVED-REMOTE-01, then read it back.`

1. Inspect the approval card without approving.
2. Verify `approved.txt` does not exist.
3. Complete the strong-biometric gate and choose **Allow once**.

Pass:

- card shows exact resolved path and complete content, not a truncated summary;
- no mutation occurs before durable settlement;
- file appears once with exact bytes after approval;
- replay shows one approval settlement, one mutation result, and the verification read;
- the approval does not become a persistent grant.

### N05 - Denied write causes no mutation

Prompt: `Replace inside.txt with DENIED-REMOTE-01.` Choose **Deny**.

Pass:

- original file hash/content is unchanged;
- outside canary is unchanged;
- denial is replayable;
- session remains reusable for a later read-only prompt.

### N06 - New session and dormant history isolation

1. Finish N03.
2. In History select **New**.
3. Record the new UUID and bootstrap cursor.
4. Select the old history without choosing Continue.

Pass:

- new UUID becomes the sole active session and starts with bootstrap events only;
- old transcript remains intact;
- selecting old history is replay-only: composer and approvals are unavailable;
- events from either history never appear in the other projection.

### N07 - Contextual continuation of old history

1. From the old N03 history choose **Continue**.
2. Verify it becomes active in both phone and local management view.
3. Prompt: `Reply with only the contents you read in the previous turn.`

Pass:

- old transcript replays before the prompt;
- answer is `inside-original` without another read request;
- composer returns only after authenticated catalog activation;
- old cursor advances; dormant new session cursor does not change.

### N08 - Host restart restores authority and replay

1. Record host/session/route/device IDs and cursor.
2. Stop the host cleanly and restart with identical model, workspace, database, and relay URL.
3. Reopen the app and reconnect.

Pass:

- host identity and compatible active session are restored;
- phone performs fresh IK, not transport-state reuse;
- complete transcript and exact cursor recover;
- no historical tool call executes again;
- any old approval card is inert unless the store still has that exact pending approval.

## 5. Pairing and Identity Failure Scenarios

### P01 - Reject pairing

Submit a valid phone request, then choose **Reject** locally.

Pass: no authorized row is created, pending Noise state is destroyed, and retry requires a new
pairing flow.

### P02 - Cancel unscanned offer

Create and cancel an offer before scanning. Scan the captured/old QR afterward.

Pass: connection is refused and no pending confirmation or device row appears.

### P03 - Expired QR

Allow the five-minute offer to expire, then scan it.

Pass: request fails closed; expiry cannot be extended by scanning or reconnecting.

### P04 - Active offer replacement

Create one offer, then attempt **Pair new device** again from another browser tab/session.

Pass: existing offer is not silently replaced; operator must cancel or wait for expiry.

### P05 - Wrong label/fingerprint

When the trusted surfaces do not show the expected label/fingerprint, do not approve.

Pass: rejection/cancel leaves no authority. Any mismatch that still pairs is a critical failure.

### P06 - Reuse QR after successful pairing

Attempt to pair a second fresh app identity using the already consumed QR.

Pass: single-use secret cannot authorize a second device.

### P07 - Remove local protected identity

Use Settings -> **Remove from this device**, confirm, then try to reconnect before host revocation.

Pass: key reference and local record are gone; reconnect cannot occur without pairing again.

## 6. Authorization and Sandbox Scenarios

### S01 - Outside-root absolute path

Prompt the model to read and then overwrite the absolute outside-canary path.

Pass: both actions are refused before approval; canary hash is unchanged.

### S02 - Parent traversal and escaping symlink/junction

Create an in-workspace symlink/junction that resolves outside the root. Prompt reads/writes through
it and through `..` traversal.

Pass: canonical-prefix enforcement refuses all attempts; no approval can legalize an escaped path.

### S03 - Protected `.camelid` state

Prompt a write/edit under `.camelid/sessions` and `.camelid/checkpoints`.

Pass: model cannot mutate authority/session/checkpoint storage.

### S04 - Shell and network disabled

Prompt shell execution and a web fetch while N01 capabilities are active.

Pass: tools are not advertised or accepted; model prose cannot manufacture either authority.

### S05 - Explicit shell/network capability variants

Restart separate disposable runs with `--allow-shell`, then `--allow-net`, then both.

Pass: only locally selected tools appear; shell reports the actual enforced sandbox layers; each
dangerous action requires its defined approval; restarting without flags removes the capabilities.

### S06 - Forbidden unattended flags

Try `agent host` with each of `--yolo`, `--allow-fs`, `--allow-mcp`, and `--auto-approve`.

Pass: CLI parsing fails and no host is armed.

### S07 - Loopback management origin guard

Attempt management requests using a non-loopback Host, mismatched Origin, cross-site fetch, and a
remote tunnel. Repeat from the same-origin loopback UI.

Pass: all non-local/mismatched requests return forbidden; same-origin loopback succeeds. No secret
bearer is accepted as a substitute.

### S08 - Revoked device

Revoke the connected device from the local UI or `agent remote revoke`, then press Reconnect.

Pass: live socket closes, fresh IK is rejected, no command is accepted, row remains auditable as
revoked, and re-pair creates a different device UUID.

### S09 - Emergency disable

With an authorized device and, separately, with a turn awaiting approval, use **Emergency disable**.

Pass: all devices become revoked, sockets close, controlled work cancels, pending approvals become
inert, and no completed local file change is rolled back.

## 7. Command, Session, and Concurrency Edge Cases

### C01 - Rapid duplicate Start tap

Double-tap Start or replay the same command ID through the test client.

Pass: one durable command/turn exists; duplicate gets the stored result; no duplicate mutation.

### C02 - New/Continue while running

Start a slow read/tool turn and immediately tap New or Continue on another history.

Pass: controls are disabled or host returns a stable rejection; active pointer/generation does not
change until the unfinished turn settles.

### C03 - Two devices submit prompts simultaneously

Pair two physical/emulated identities and submit at nearly the same time.

Pass: exactly one turn is accepted; the other receives `session_busy`; both converge by replay.

### C04 - Two devices settle one approval

Open the same pending approval on two authorized devices. Allow on one and deny on the other.

Pass: first valid settlement wins; the late decision is rejected/inert; mutation matches the winner
exactly once.

### C05 - Two devices switch active session

Device A views dormant history while device B activates another session.

Pass: authenticated catalog updates authority on both devices; A is not forcibly navigated away
from the history it is viewing; commands remain available only for the new active session.

### C06 - Rapid duplicate New/Continue

Double-tap New and Continue under network latency.

Pass: idempotency/digest binding prevents duplicate sessions or conflicting command reuse.

### C07 - Catalog pagination changes mid-read

Using a seeded test database with more than 64 histories, request page one, mutate the catalog from
another device, then request page two with the old revision.

Pass: pagination fails closed with a refresh-required error; pages from different revisions are
never merged.

### C08 - Per-history cursor eviction

Seed more than 256 histories, visit enough to evict an old protected cursor, then revisit it.

Pass: replay safely restarts from zero and reconstructs the transcript; no cross-history events.

## 8. Lifecycle and Network Scenarios

### L01 - App background/foreground while idle

Background for longer than one heartbeat, then foreground.

Pass: old transport is discarded, fresh IK runs, replay converges, no duplicate events.

### L02 - App kill during running turn

Force-stop after `turn.accepted`; reopen after the host finishes.

Pass: host authority remains local, turn reaches a valid terminal state, app reconstructs complete
history by replay, and partial assistant output is not persisted as a final answer.

### L03 - App kill during pending approval

Force-stop on an approval card; reopen before and after approval expiry.

Pass: exact pending card recovers before expiry; expired card becomes inert; no mutation occurs
without a durable decision.

### L04 - Host process loss during running turn

Terminate the host process after acceptance, then restart with the same database.

Pass: interrupted work becomes failed; it is never resumed or re-executed; session can be explicitly
rearmed only through supported recovery.

### L05 - Relay interruption

Stop relay/tunnel while idle and while running, then restore it.

Pass: host uses bounded reconnect, clears connection-bound Noise/pairing state, queues no commands,
and device establishes fresh Noise before replay.

### L06 - Wi-Fi to cellular and back

Switch the phone network during idle, running, and waiting approval.

Pass: no command executes twice; state converges after reconnect; pending approval identity remains
exact or the card is invalidated.

### L07 - Long idle heartbeat

Leave host and phone idle beyond every relay/proxy idle timeout used in deployment.

Pass: heartbeat keeps the path alive or bounded reconnect recovers; reconnect never widens authority.

### L08 - Lock/unlock with biometric approval

Lock on an approval card, unlock, and attempt Allow once with success, failure, and cancellation of
the device biometric prompt.

Pass: only successful strong authentication can send Allow once; failures/cancel leave approval
pending and files unchanged.

## 9. Bounds, Display, and Storage Scenarios

### B01 - Maximum prompt and Unicode

Submit 4096 characters containing emoji, combining marks, RTL text, quotes, backslashes, and newlines.

Pass: exact text survives canonical encoding/replay, layout remains usable, and 4097+ is refused or
prevented without truncation-based authority.

### B02 - Long approval preview

Request a write near the protocol content bound with distinctive prefix/suffix sentinels.

Pass: the complete action is digest-bound; UI permits inspecting both sentinels; approval is never
computed from clipped text.

### B03 - Large tool output and chunk boundary

Read a file that forces multiple encrypted chunks and test the maximum supported inner message in
the protocol harness.

Pass: ordered chunks reconstruct exactly; oversized, duplicate, omitted, reordered, or mixed-message
chunks close the connection and cause no state change.

### B04 - Small screen, font scaling, and accessibility

Test the smallest supported viewport, 200% font scale, screen reader focus order, keyboard open,
long UUID/path/tool names, and approval modal scrolling.

Pass: no action is hidden behind system navigation, no overlap/truncation changes meaning, all
controls have names/roles, and dangerous confirmation remains explicit.

### B05 - Disk full/read-only database

Use a disposable volume/quota to force SQLite writes to fail during command acceptance and approval
settlement.

Pass: no event is broadcast as committed, no tool mutation occurs, UI reports failure, and restart
does not invent the missing command/decision.

### B06 - Newer/corrupt schema

Copy the database, set an unsupported newer schema version and, separately, corrupt it.

Pass: host fails closed without overwriting the database or arming remote authority.

### B07 - Model/workspace/capability identity change

Restart against a different workspace, model ID/hash, or capability snapshot using the same database.

Pass: incompatible history remains replay-only or is refused; it cannot become execution context or
inherit authority.

### B08 - Clean install and upgrade

Test clean app install, upgrade over a paired development build, uninstall/reinstall, and OS
Keystore invalidation.

Pass: protected records survive only when platform semantics permit and remain usable; missing or
invalidated key material fails closed and requires explicit re-pairing.

## 10. Tamper/Adversarial Harness Scenarios

These are manual operator-triggered runs of the committed protocol/transport harnesses, not UI-only
tests:

- wrong pinned host key;
- unregistered or revoked device static key;
- altered Noise handshake/transport ciphertext;
- non-binary WebSocket frame;
- oversized relay frame and slow consumer;
- stolen route ID without device key;
- unknown protocol kind/field and malformed privileged DTO;
- approval with mismatched tool/action digest;
- command ID reused with different payload;
- replay request for another workspace or unknown session.

Pass: each attempt is connection-terminal or command-rejected as specified, produces no mutation,
and leaks no plaintext, key, pairing secret, or executable argument into relay logs.

## 11. Exit Criteria

The feature is not ready to merge merely because all priority-0 cases pass. Before removing
`[DON'T MERGE]`, require:

1. every priority-0 scenario passing on the candidate commit;
2. the complete abnormal/security/concurrency matrix passing or explicitly blocked with owner/date;
3. physical Wi-Fi/cellular, lock/relaunch, biometric, and accessibility evidence;
4. clean-machine packaging/install evidence for the supported Android and host targets;
5. production relay load, retention, privacy/logging, backup, and rollback evidence;
6. independent review of cryptographic wrapper, key lifecycle, authorization boundaries, and relay;
7. public claims reconciled to the evidence and remaining non-claims.

Any critical safety failure requires a new candidate commit and a fresh run of all priority-0
scenarios; do not carry forward earlier physical evidence as proof of the changed build.