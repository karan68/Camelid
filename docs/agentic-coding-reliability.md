# Coding reliability follow-up

Status: proposed next iterations of Code in Chat. These capabilities are not implemented by the current UI and loop refresh.

The current runner owns sessions, exact approvals, file journals, bounded read-only helpers, and restart recovery. Live Tiny Tasks checks also show its limits: a model can spend steps polling or rereading, lose track of a failed call, and finish after creating invalid code. Recovery prompts bound some failures; coordination and verification need explicit runtime state.

## 1. Helper completion and active-run corrections

Add a bounded `wait_for_helpers` operation and a `waiting_for_helpers` activity state. Completion should wake the waiting lead without polling the model. Release the inference slot while waiting so a helper can use the resident model. Keep status inspection available for the sidebar.

Persist a completion record keyed by session, parent run, and helper run. Include terminal outcome, findings, observed file references, and delivery sequence. Insert each result into the parent history once and persist its delivery cursor with that history. A crash between notification and delivery must neither lose the result nor duplicate it. Reject results belonging to an older parent run. Stop cancels and joins workers; restart retains observations but does not automatically resume execution. An ordinary lead answer with unfinished helpers should enter a bounded collection boundary before declaring the turn finished.

OpenClaw separates yielding for announced child results from status inspection, which is a useful coordination pattern to adapt to Camelid's single resident model. [Sub-agent tool reference](https://docs.openclaw.ai/tools/subagents/tool-reference)

Enable the running composer with explicit **Add to current task** and **Queue follow-up** choices. Bind each request to the expected run and an idempotent message ID. Show separate accepted and consumed events. At a boundary before the next model decision or tool launch, persist incoming corrections, discard any unlaunched proposal based on the old task revision, then regenerate. Let an already-running tool finish and record its result; steering cannot undo it.

An outstanding approval must be invalidated when steering supersedes its proposal. Check the task revision under the same control boundary as approval consumption and execution admission, so an approval arriving concurrently cannot authorize an obsolete action. User text never changes the session's command permission or artifact identity. Queued follow-ups are distinct from steering; restart leaves them visible but requires explicit continuation.

OpenClaw documents steering, queued follow-ups, and interruption as separate operations. Camelid can start with the first two without adopting every queue mode. [Command queue](https://docs.openclaw.ai/concepts/queue)

Acceptance checks: delayed helper completion with no status polling; two results delivered once; stop while waiting; stale completion after continuation; steering during generation, tool execution, and approval; duplicate/conflicting message IDs; restart at the delivery/consumption boundary.

## 2. Task state through compaction

Store a bounded, versioned task record alongside authoritative execution state. Preserve the objective, accepted user constraints, decisions, unresolved work, helper findings, and verification references. Build execution evidence deterministically from tool outcomes and journal entries. Label model-authored decisions as claims; an applied edit is not a successful test. Preserve exact file and run identifiers, cap excerpts, and treat helper/file content as untrusted observations.

Reinsert this record after compaction and continuation, retaining recent tool-call/result pairs. Keep permissions and approvals exclusively in server state. When required task state cannot fit the context budget, stop with an explicit limit instead of silently dropping constraints. OpenClaw's persisted summaries and identifier preservation provide relevant examples. [Compaction](https://docs.openclaw.ai/concepts/compaction)

Acceptance checks: repeated compaction preserves a user correction, failed call, helper finding, unresolved requirement, and successful test reference; restart after an applied edit preserves evidence and Undo without replaying authority.

## 3. Execution isolation and liveness

Introduce an execution backend with explicit filesystem, network, environment, process cancellation, and isolation capabilities. Preserve exact approvals while adding an isolated backend; the current account-permission command runner must never be labeled sandboxed. Backend identity, working directory, command arguments, and relevant executable identity belong to the approval binding. Unattended command execution remains out of scope until isolation is available and verified.

Keep the conservative inference limit, but expose queued-for-model, generating, running-tool, waiting-for-helper, and waiting-for-user separately. Guard worker updates with their run identity. Add a total active execution budget covering inference queues, generation, tools, and helper waits. Approval/pause waits should use explicit separate expiry rules rather than quietly renewing execution budgets. Record elapsed time and the reason for waiting so slow generation is distinguishable from a stalled run.

Validate deterministic lifecycle invariants first, then run the correction, delayed-helper, compaction, and restart scenarios sequentially against the exact supported local model artifacts. Report model failures separately from application test results. A tool-capability gate is not a coding-quality certification.
