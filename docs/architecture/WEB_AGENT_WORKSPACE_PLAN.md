# Web Agent Workspace Plan

Status: executable plan. Each stage must pass its gate before the next stage starts.

Execution status (2026-07-17):

- Stage 0: complete — contract and adversarial architecture review.
- Stage 1: complete — shared agent core, exact five-tool profile, bounded bridge,
  approval identity/timeout/cancellation tests.
- Stage 2: complete — loopback-authorized one-session API, exact earned-model
  gate, SSE/decision/cancel routes, and model-transition exclusion.
- Stage 3: complete — first-class operational Workspace view, exact eligibility,
  responsive timeline, and production frontend build.
- Stage 4: complete — live event reducer, approval modal, intentional-close
  handling, desktop/mobile overflow and control-size gates.
- Stage 5: complete — exact `Qwen3-4B-Q4_K_M.gguf` was acquired from the pinned
  official revision and verified at 2,497,280,256 bytes / SHA-256
  `7485fe6f…fdf5`. The fail-closed harness passed multi-step
  list/read/search, denied-write no-mutation, approved exact-content write, an
  unchanged outside-root canary, and real approval/terminal WebUI capture. The
  bundle is `qa/evidence-bundles/workspace-qwen3-4b-q4km-20260717T165404Z-head-8c2a2b74/`.

## Product boundary

Camelid Workspace lets a user give a tool-capable local model one canonical
directory and one goal. The model may inspect that directory, propose changes,
and apply a write only after explicit user approval. The browser is a view and
approval surface; all path validation and tool execution remain server-side.

The first release supports:

- one active workspace session per Camelid process;
- exact model rows whose capability contract has `tool_capable: true`;
- directory listing, bounded file reads, bounded search, and approval-gated
  file writes inside one canonical workspace root;
- a bounded plan-act-observe loop, live activity events, cancellation, and a
  final answer;
- `allow once`, `always allow this tool for this session`, `deny`, and `abort`
  decisions at the existing approval-policy chokepoint.

The first release does not expose shell execution, unrestricted filesystem
access, network access, GUI control, subagents, unattended approval, or
background persistence. Those capabilities remain available only on their
existing CLI/TUI surfaces and are not implied by Workspace.

## Invariants

1. **Exact capability gate.** Starting a session requires the active loaded
   model to match a supported compatibility row with `tool_capable: true`.
2. **Loopback management boundary.** Workspace routes are unavailable when the
  server is bound to a non-loopback address. Every route requires a loopback
  `Host` plus either `Sec-Fetch-Site: same-origin` or an `Origin` whose host is
  loopback. This mirrors the reviewed local-management predicate from PR #447,
  including the separate loopback frontend/backend ports used in development.
3. **Server-owned sandbox.** The server canonicalizes the workspace root once.
   Clients never authorize paths or submit prevalidated actions.
4. **One approval chokepoint.** `agent::run_loop` remains the only place that
   decides whether an action executes. The HTTP bridge implements `Approver`;
   it does not bypass `ApprovalPolicy` or call tools directly.
5. **Fail closed on disconnect.** If the event consumer disappears while an
   approval is pending, the action is denied and the session is aborted.
6. **Bounded operation.** Session count, steps, event backlog, read/output
  sizes, and model output remain bounded. At most one session record exists;
  its terminal state remains inspectable until the next session replaces it.
7. **Model lifecycle exclusion.** An active Workspace session prevents every
  load or unload through the current main load/unload chokepoints. Stage 2 adds
  this Workspace-owned gate directly; it does not depend on PR #447 merging.
  If #447 lands first, the gate composes with its broader lifecycle lease
  instead of creating a second model-lifecycle protocol.
8. **No support widening.** Workspace availability is a product surface over
   existing `tool_capable` evidence. It does not promote a model, quant,
   platform, backend, or context window.

## HTTP contract

All routes are under `/api/agent/workspace` and return Camelid's existing error
envelope on failure.

### Create

`POST /api/agent/workspace/sessions`

```json
{
  "workspace": "C:\\projects\\example",
  "goal": "Find the failing test and make the smallest repair.",
  "max_steps": 12,
  "max_tokens": 800,
  "temperature": 0.0
}
```

The server returns `201` with the session id, canonical workspace display path,
active model id, state, and limits. A second active session returns typed `409`.

### Events

`GET /api/agent/workspace/sessions/:id/events`

SSE event types:

- `session.started`
- `model.delta`
- `tool.call`
- `approval.required`
- `tool.result`
- `model.answer`
- `session.notice`
- `session.finished`
- `session.error`

Every event has a monotonically increasing sequence number and session id.
Approval events include an opaque approval id, risk, tool name, and validated
human-readable detail. Raw tool arguments are never used as authorization.

### Decision

`POST /api/agent/workspace/sessions/:id/decisions`

```json
{ "approval_id": "...", "decision": "allow_once" }
```

Accepted decisions are `allow_once`, `always_tool`, `deny`, and `abort`. A
stale, duplicate, or wrong-session approval id returns typed `409`.

### Cancel and inspect

- `DELETE /api/agent/workspace/sessions/:id` requests cooperative cancellation.
- `GET /api/agent/workspace/sessions/:id` returns state and non-sensitive
  counters, never tool output or raw arguments.

## Stages and gates

### Stage 0 - Contract and architecture

- Freeze this plan and verify the current agent, API, lifecycle, and frontend
  ownership boundaries.
- Record conflicts with active upstream work before code starts.

Gate: plan review finds no route that executes a tool outside `run_loop`, no
browser-owned path validation, and no capability claim beyond existing rows.

### Stage 1 - Reusable bridge core

- Move or expose only the minimum agent-core types needed by both the CLI/TUI
  and server without making UI code a library dependency.
- Add an explicit tool profile to `run_loop`: the existing CLI/TUI profile is
  unchanged, while `WorkspaceFiles` contains exactly `read_file`, `list_dir`,
  `search`, `write_file`, and `edit_file`. The profile is the source of the
  advertised schemas and validation allowlist; global subagent state and
  platform system tools cannot widen it.
- Add a bounded event reporter and channel-backed approver around `run_loop`.
- Keep the bridge independent of Axum and test it with a mock model driver.

Gate: focused tests prove that reads may auto-run, writes block, denial does not
execute, explicit approval executes once, cancellation stops the loop, and the
step cap is enforced.

### Stage 2 - Secured session API

- Add the one-session manager to `AppState`.
- Add create/events/decision/status/cancel routes.
- Enforce loopback bind plus the reviewed local-management authorization
  predicate (`Host` loopback and either same-origin fetch metadata or a
  loopback `Origin`).
- Gate the active model against the compatibility contract.
- Add a Workspace operation lease and check it at
  `load_model_from_path_with_activation` and `unload_model`, the current main
  transition chokepoints, for the complete session.

Gate: focused router tests cover authorization, malformed requests, inaccessible
roots, non-tool-capable models, active-session conflict, stale decisions,
cancellation, event ordering, and disconnect during approval. Rust format,
Clippy for touched targets, and focused library tests pass.

### Stage 3 - Workspace UI shell

- Add a first-class `Workspace` view beside Chat, not a marketing page.
- Show the selected root, active exact model, capability state, goal composer,
  limits, activity timeline, and a quiet empty state.
- Match the existing typography, spacing, controls, evidence language, and
  responsive shell. Do not add decorative cards, gradients, or explanatory
  feature copy.

Gate: frontend state and rendering smokes cover eligible, ineligible, running,
terminal, reconnect, and error states. Production build passes.

### Stage 4 - Approval and live execution UX

- Connect SSE events to the timeline.
- Present approval as a focused modal with exact validated action detail.
- Implement allow once, always for this tool, deny, abort, and stop.
- Prevent duplicate decisions and make reconnect behavior explicit.

Gate: end-to-end browser tests prove no write before approval, denied writes do
not occur, approved writes remain inside the root, stop prevents later tools,
and ordinary Chat is unchanged. Desktop and 390x844 layouts have no overflow or
overlap.

### Stage 5 - Real-model closure and release truth

- Run a certified `tool_capable` exact row through read, search, denied-write,
  and approved-write scenarios against a disposable workspace.
- Capture a privacy-scrubbed evidence bundle with request/event summaries,
  filesystem before/after hashes, model identity, and UI screenshots.
- Add the bounded Workspace API feature to the capability contract and
  regenerate the ledger only after the real-model gate passes.
- Update README and compatibility wording with explicit non-claims.

Gate: full Rust gates, ledger drift/schema checks, frontend build/smokes,
privacy scrub, and the disposable real-model scenario all pass. No user files
outside the disposable root are touched.

## Design review checklist

- Workspace is an operational surface, not a landing page.
- The activity timeline is dense and scannable; individual tool operations are
  rows, not nested cards.
- Risk is communicated by action-specific language and icons, not alarm colors
  everywhere.
- Approval controls have stable dimensions and remain reachable on mobile.
- File paths wrap safely and never force horizontal page scrolling.
- Empty, loading, reconnecting, denied, cancelled, capped, failed, and completed
  states are all designed explicitly.
- The UI never labels an unverified model as agent-capable and never hides why a
  session cannot start.

## Rollback

Workspace is additive. Removing its routes, manager field, view, and capability
entry restores the previous product without changing chat generation, model
loading, tool execution, or the CLI/TUI agent.