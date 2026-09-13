# Agentic coding in Camelid Chat

Reviewed 2026-09-13 against GitHub main at [`0f53c46b`](https://github.com/timtoole02/Camelid/tree/0f53c46b60c66866b16cfe8b35c6d3bc63ebb4aa). The original local checkout was `c4b7d604`; current main was fetched and inspected in an isolated worktree. This records the original feasibility review and design proposal. The implementation and current operating limits are documented in [Code in Chat](AGENTIC_CODING.md).

## Recommendation

Add a **Chat / Code** switch to the existing chat header. Code opens a project-bound coding conversation and a collapsible right sidebar. Keep the recent composer, project context, connected Tools picker, conversation rendering, and file-review behavior.

The sidebar should answer three questions immediately: who owns the task, what are they doing now, and what needs attention? Each agent gets a name/role, assigned goal, observed status, latest action, and relevant files. Selecting an agent opens its assignment and activity details. A neighboring Changes tab opens the existing review workflow without losing the conversation.

Use one lead agent for edits and command execution, with up to two read-only helpers for exploration and review. This matches the current child-worker policy. It also avoids concurrent edits escaping the parent's undo journal. Show queued and waiting states explicitly; sharing one local model is not evidence of simultaneous inference or a speed increase.

The interactive preview includes selecting agents, Chat/Code switching, sidebar collapse, sample pause/resume, file review, approval/rejection, undo, command review, and follow-up messages. All activity, code, statuses, and results in that preview are illustrative. No agent, command, or file mutation is triggered by it.

## Recent changes to build on

| Existing change | How Code should use it |
| --- | --- |
| September 12 composer Tools picker and Connections | Preserve the single Tools entry, saved selections, and explicit approval for external tool calls. |
| September 12 Projects and conversation context | Reuse project instructions and references; additionally bind a coding session to a verified local folder. Current project records do not contain a repository path. |
| September 12 output previews and reviewed file changes with undo | Route generated edits through the existing prepare → review → apply → undo flow, and show outputs in conversation. |
| September 10 conversation organization and alternate replies | Keep normal chat behavior; coding sessions need their own run identity and tool transcript so reply variants never re-execute tools. |
| Earlier chat polish, dark/light tokens, navigation groups | Keep the compact navigation rail, steel-blue controls, existing typography, and plain status language. Copper remains reserved for verified claims. |

Primary files at the reviewed commit:

- [`ChatWorkspace.jsx`](https://github.com/timtoole02/Camelid/blob/0f53c46b60c66866b16cfe8b35c6d3bc63ebb4aa/frontend/src/views/ChatWorkspace.jsx)
- [`ConnectedTools.jsx`](https://github.com/timtoole02/Camelid/blob/0f53c46b60c66866b16cfe8b35c6d3bc63ebb4aa/frontend/src/components/mcp/ConnectedTools.jsx)
- [`projectContext.js`](https://github.com/timtoole02/Camelid/blob/0f53c46b60c66866b16cfe8b35c6d3bc63ebb4aa/frontend/src/lib/projectContext.js)
- [`ChangesView.jsx`](https://github.com/timtoole02/Camelid/blob/0f53c46b60c66866b16cfe8b35c6d3bc63ebb4aa/frontend/src/views/ChangesView.jsx) and [`changes.rs`](https://github.com/timtoole02/Camelid/blob/0f53c46b60c66866b16cfe8b35c6d3bc63ebb4aa/src/api/changes.rs)

## What already executes work

- The Rust agent loop already selects and executes tools with policy checks, approval decisions, cancellation, and session support.
- The Web Workspace bridge already streams model/tool events, transports approval decisions, and restores saved conversations. Its production configuration remains `WorkspaceReadOnly` with shell disabled, and the session API rejects `allow_writes: true`.
- Chat's connected-MCP runner already performs bounded tool rounds with approval. Manual function definitions are a separate request-display path; displaying a tool call does not execute it.
- CLI subagent orchestration already launches and tracks children. Current workers are unconditionally read-only, cannot run commands or spawn descendants, and carry the parent's model artifact identity. Their status does not yet provide the complete, session-scoped event feed the proposed sidebar needs.

Sources: [`agent.rs`](https://github.com/timtoole02/Camelid/blob/0f53c46b60c66866b16cfe8b35c6d3bc63ebb4aa/src/chat/agent.rs), [`workspace_bridge.rs`](https://github.com/timtoole02/Camelid/blob/0f53c46b60c66866b16cfe8b35c6d3bc63ebb4aa/src/chat/workspace_bridge.rs), [`workspace.rs`](https://github.com/timtoole02/Camelid/blob/0f53c46b60c66866b16cfe8b35c6d3bc63ebb4aa/src/api/workspace.rs), [`mcp.js`](https://github.com/timtoole02/Camelid/blob/0f53c46b60c66866b16cfe8b35c6d3bc63ebb4aa/frontend/src/lib/mcp.js), [`subagent.rs`](https://github.com/timtoole02/Camelid/blob/0f53c46b60c66866b16cfe8b35c6d3bc63ebb4aa/src/chat/subagent.rs).

## Work needed for real coding

1. **Create an explicit coding session.** Bind it to a canonical workspace root, conversation, exact tool-capable model artifact, context budget, and permissions. Keep the existing read-only Workspace contract. Use an explicit coding tool profile; do not expose every tool in `Full` just to unlock editing. Preserve the existing local-management and restricted LAN-chat boundaries.
2. **Make the server own the run.** Keep the Rust loop as the execution owner and give the UI a run snapshot plus ordered events. Each event should carry `session_id`, `run_id`, `agent_id`, optional `parent_agent_id`, sequence, and time. Add structured assignment, status, tool-call, review, result, and terminal events; do not infer assignments or progress percentages from assistant prose. Reconnect/reload should restore state without repeating tool execution.
3. **Connect mutations to durable reviews.** Let the lead prepare file changes through the existing review manager and link those review IDs to the run. Revalidate file contents at apply/undo time. The present API reviews individual files; the UI must show individual outcomes and partial failure rather than pretending a batch is atomic. Tool-originated edits must not bypass the journal. Shell commands can also mutate files, so file-review undo must not imply that arbitrary command side effects can be undone.
4. **Expose read-only helper work.** Adapt the subagent registry to a coding-session lifetime and publish assignment/status events. The current registry is process-global, and Workspace currently admits one active session; avoid advertising concurrent independent coding sessions until isolation is designed. Cancel/reap children with the parent. Keep stale/disconnected/failed states visible.
5. **Provide an explicit command runner.** Reuse approval, timeout, cancellation, and honest platform reporting. The default shell sandbox currently refuses execution on macOS; choose and validate a supported execution path before offering “Run tests” there. Windows has different confinement from Linux. External MCP execution keeps its own approval flow and does not inherit file-review undo.
6. **Wire the frontend.** Add a mode switch and coding-session controller beside the existing chat send path, plus agent sidebar and inline review detail components. Preserve normal chat readiness and enforce coding/tool readiness separately on the server. A pending run must retain its identity across navigation, and changing a project/model must not silently retarget it.

Platform source: [`shell_sandbox.rs`](https://github.com/timtoole02/Camelid/blob/0f53c46b60c66866b16cfe8b35c6d3bc63ebb4aa/src/chat/shell_sandbox.rs). Model tool gating: [`toolCalling.js`](https://github.com/timtoole02/Camelid/blob/0f53c46b60c66866b16cfe8b35c6d3bc63ebb4aa/frontend/src/lib/toolCalling.js).

## Practical delivery sequence

1. A complete single-lead flow: select project folder → inspect → propose a change → review/apply → run an approved check → report outcome. Show the lead in the sidebar immediately.
2. Add read-only Explorer and Reviewer children with live assignment/status events and cancellation propagation.
3. Add durable run restoration, richer review navigation, and measured scheduling behavior. Broader parallel writing would require a separate isolation/merge design; it is not part of this proposal.

The first two increments deliver the requested coding experience. A UI toggle alone would not.

## Acceptance checks

- Normal Chat, connected MCP tools, and read-only Workspace continue working.
- A real supported tool-capable model completes the single-lead edit/check flow; fixture-only tests are not presented as model capability evidence.
- The sidebar distinguishes working, queued, waiting for approval, done, failed, cancelled, and disconnected using server state.
- Denied or stale approvals cannot execute; reconnect and transcript restore do not repeat actions; cancellation reaches tools and children.
- Helper writes and command execution remain rejected; reviewed edits retain conflict detection and undo behavior.
- Dark/light layouts and keyboard interactions remain usable at desktop and mobile widths. The concept stacks the agent panel below the conversation on small screens.

## Review verification

The latest main frontend built successfully, with existing bundle-size/dynamic-import warnings. The repository's MCP browser smoke passed its picker, saved sets, approval/denial, cancellation, persistence, and responsive checks. The concept has separate browser checks for agent selection, mode switching, pause/resume, sidebar collapse, follow-ups, per-file apply/reject/undo, and simulated command review. The concept checks passed at 1024, 736, 390, and 320 pixels in both themes, with no script errors or horizontal overflow. These validate the reviewed UI and preview interactions, not a newly implemented coding backend.
