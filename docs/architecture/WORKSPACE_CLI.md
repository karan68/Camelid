# Workspace CLI

The Workspace CLI is a thin client for the existing Web Workspace API. It does not own an agent
loop, persistence format, retrieval policy, or tool registry.

## Commands

| CLI | Existing API |
| --- | --- |
| `workspace ask` | create session, then consume its Workspace SSE stream |
| `workspace threads` | list durable threads for a canonical root and active model |
| `workspace show` | retrieve one bounded durable transcript |
| `workspace compact [--undo]` | compact or undo compaction through the existing store |
| `workspace delete` | delete one durable thread through the existing ownership checks |

`ask --thread <id>` resumes a durable thread. Human mode writes the answer to stdout and progress
to stderr. `--json` writes compact JSON; streaming asks use JSON Lines.

## Authentication

Workspace has two independent local authorization paths:

1. Browser requests must retain the existing same-origin provenance check.
2. CLI requests must present a process-rotated bearer capability.

Both paths additionally require a loopback listener and a loopback `Host`. A bearer token never
enables Workspace on a non-loopback bind.

The server creates the credential only after it has bound the requested endpoint and completed
startup model loading. The endpoint-scoped token file is then discoverable by another process
running as the same OS user. It is mode `0600` on Unix; on Windows its default location inherits
the current user's LocalAppData ACL. `CAMELID_WORKSPACE_TOKEN_FILE` is an explicit path override
and must point to a location private to that user.

The token is replaced on every successful server start. A stale file after a crash grants nothing
because no server is listening with that in-memory token; the next server rotates it before serving
CLI requests. Cleanup on drop is best-effort and never deletes a replacement token written by a
newer server.

This design protects the loopback API from cross-site browser requests without pretending to
isolate mutually untrusted processes under the same OS account. A same-user process that can read
the model files and Workspace database is already inside Camelid's local authority boundary.

## Invariants

- The CLI accepts only loopback socket addresses.
- The browser does not read or receive the bearer credential.
- Authorization headers reject newline injection.
- Workspace SSE must return `text/event-stream` and valid JSON envelopes.
- An unexpected approval request fails closed and cancels the read-only session.
- File access, grounding, context fitting, memory, cancellation, and compaction remain server-owned.

## Non-goals

- No remote Workspace API authentication.
- No automatic server or model startup.
- No write, shell, network, GUI, MCP, or subagent tools.
- No second CLI-specific conversation database or context compiler.