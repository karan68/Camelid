# Coding reliability

The current Code implementation adds verification and bounded recovery, helper completion delivery, active corrections and queued follow-ups, a pinned task record, static previews, engine binding, reusable workflows, and grouped file Undo. [Code in Chat](AGENTIC_CODING.md) describes the controls, HTTP interface, validation, and operational limits.

## Verification and recovery

`verify_project` prepares the configured command and current changed-file fingerprints before asking for exact approval. It rechecks the fingerprints before execution and records the real result. Later changes make an earlier pass stale. A finishing lead that changed files receives a verification boundary when a check is configured; at most three attempts are allowed per turn. Failed checks provide concrete output for repair. Denials do not request a different execution route.

The optional Chromium check captures page-load errors and unhandled rejections. Static previews inline bounded local assets and run in an opaque iframe with project-specific storage. Neither a preview nor a page-load pass proves interactions work; those require an actual project test command.

Validation and file errors receive at most two targeted recovery hints. Repeated promises, identical reads, and alternating calls with unchanged results are bounded. The model can still fail a task or write incorrect code. An accepted answer is labeled **Turn finished**, not independently certified success.

## Coordination and corrections

`wait_for_helpers` releases the model while read-only helpers work. Each completion includes its parent run, outcome, findings, and observed file references. Delivery and the matching transcript are persisted together. Reconnect cannot duplicate delivery; a stale parent result cannot enter a newer run. A final answer encounters a collection boundary while helpers remain active. Stop cancels and joins workers; restart retains observations without resuming execution.

The composer offers **Add to current task** and **Queue follow-up**. Incoming messages carry the expected run and an idempotent request ID. Corrections invalidate pending approval and any unlaunched proposal based on the older task revision. An already-admitted command can finish; steering cannot undo it. Final-answer acceptance closes correction admission under the same control lock, preventing a last-second accepted correction from being lost at completion.

Queued follow-ups start once after a normally finished turn while the session retains engine ownership. Failure, Stop, and restart leave the queue visible for explicit continuation. Persisted queues never grant restored execution authority.

OpenClaw's distinction between yielding for results and inspecting status informed the helper boundary. Its separate steering and queue operations informed the active composer. [Sub-agent tools](https://docs.openclaw.ai/tools/subagents/tool-reference), [command queue](https://docs.openclaw.ai/concepts/queue).

## Task state and project controls

A deterministic task record preserves the original objective, current request, consumed corrections, plan claims, applied-change references, delivered helper findings, and recent check evidence. It is inserted into the pinned context after compaction. Required context that cannot fit stops the run rather than silently dropping constraints. Execution permissions remain in server state. This adapts the task-preservation concerns described in OpenClaw's [compaction documentation](https://docs.openclaw.ai/concepts/compaction).

Sessions bind to the connected engine's persistent identity and host name. They cannot silently continue on a different machine. Model generation is serialized across lead and helpers. Approved commands default Cargo, CMake, and OpenMP to one job; these are environment defaults, not OS resource limits. The total wall-clock budget defaults to 30 minutes and explicitly includes approval and pause time.

Users can save a project workflow after a current passing check. The named workflow carries notes and verification/preview settings; it never enables commands or automatic approvals. Task checkpoints group journaled edits and preflight all file versions before restoring in reverse order. They preserve later manual edits and support retry after a partial restoration. Command side effects remain outside Undo.

## Remaining work

The connected-engine runner is not a general execution-backend interface, fleet manager, or OS sandbox. Broader unattended execution requires an isolated backend with explicit filesystem, network, environment, and cancellation capabilities. Exact command approval and execution isolation remain separate responsibilities.

Richer browser interactions, independent helper budgets, and broader model scheduling remain future work. Validate lifecycle invariants with deterministic fixtures and report exact-artifact live-model failures separately. A tool-capability gate does not certify coding quality.

## Managed static previews

Code tasks can use `start_preview`, `open_preview`, and `stop_preview`, including
when shell commands are disabled. `open_preview` starts or reuses a local static
HTTP server and opens Google Chrome on the engine's computer. These agent actions
use the existing per-action approval dialog; automatic file approval does not
approve them. The Preview panel also has direct Start, Open in Chrome, and Stop
controls.

Set the project's Preview entry (for example `tiny-board/index.html`), or pass
`entry` to the tool. With no entry configured, Camelid uses `index.html` at the
workspace root or the unique immediate subfolder containing it. Ambiguous sites
require an explicit entry. The containing folder is the site's document root.

The server binds only to `127.0.0.1` on an available port. It serves HTML, CSS,
JavaScript, JSON, images, fonts, and other supported static assets with their MIME
types, including absolute asset paths within that site. Hidden paths, traversal,
paths escaping the site, and assets over 16 MB are refused. Changed files are read
on reload; there is no build step or hot-reload injection. Framework development
servers and application backends are outside this static-preview feature.

The preview survives agent completion and command timeouts. Repeated starts reuse
its URL. Stop closes the listener without closing Chrome tabs; engine shutdown or
deleting the task also closes it. A restart never replays a browser launch or
pretends a saved URL is still live. Opening Chrome is not recorded as a passing
browser test. The API supports `GET` and `POST` at
`/api/agent/coding/sessions/:id/preview-server`. POST accepts `action` (`start`,
`open`, or `stop`) and an optional `entry` such as `tiny-board/index.html`;
management remains same-origin and loopback-only.
