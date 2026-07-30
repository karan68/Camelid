# Responses and Conversations storage

Camelid's OpenAI-compatible Responses surface is stateless by default and uses
an opt-in SQLite store for durable local state. This store is deliberately
separate from Workspace memory: Workspace indexes flattened human-readable
turns, while Responses must preserve canonical JSON items such as
`function_call`, `function_call_output`, `call_id`, and JSON argument strings.

## Persistence rules

- Omitted `store` and `store:false` do not create a retrievable Response row.
- `store:true` commits the terminal Response plus canonical input, output, and
  reconstructed context.
- `previous_response_id` loads the referenced stored context and appends only
  the current input. Prior `instructions` are not carried forward.
- A `conversation` id loads its ordered items and atomically appends the current
  input and generated output. This happens even with `store:false`, because
  conversation use is itself a request to mutate durable state.
- `previous_response_id` and `conversation` are mutually exclusive.
- A client-supplied `Idempotency-Key` is accepted only with `store:true`.
  Reusing it with the same request returns the stored Response; using it with a
  different request returns `409 idempotency_key_conflict`.
- A streaming request is committed before its terminal `response.completed` or
  `response.incomplete` event. Disconnecting before the terminal event does not
  leave a partial Response or conversation turn.

## Database

The path resolves in this order:

1. `CAMELID_RESPONSES_DB`
2. the platform user-data directory
3. the system temporary directory as a last resort

The initial schema contains:

- `conversations`: identity, timestamps, and metadata
- `conversation_items`: canonical JSON items with a stable per-conversation
  ordering
- `responses`: terminal response JSON, current input/output, reconstructed
  context, parent response id, optional conversation id, and optional
  idempotency key

Foreign keys, WAL mode, and a five-second busy timeout are enabled on every
connection. Conversation item appends and optional Response insertion share
one transaction. On Unix, Camelid applies mode `0700` when it creates the
database directory and `0600` to the database file; it does not change an
existing caller-selected parent directory.

## Concurrency and bounds

Within a Camelid process, conversation mutations are serialized by conversation
id and idempotent requests are serialized by idempotency key. SQLite remains
the cross-process integrity boundary, but Camelid does not claim cross-process
request serialization.

Context reconstruction fails closed at:

- 512 canonical items
- 1 MiB for one item
- 8 MiB for the complete reconstructed JSON context

These are storage/reconstruction bounds, not model context limits. The existing
tokenization and model context gate runs after reconstruction and remains
authoritative for inference.

## HTTP surface

- `POST /v1/responses`
- `GET|DELETE /v1/responses/:id`
- `POST /v1/conversations`
- `GET|POST|DELETE /v1/conversations/:id`
- `GET|POST /v1/conversations/:id/items`
- `GET|DELETE /v1/conversations/:id/items/:item_id`

Conversation item listing supports `order=asc|desc`, `after=<item_id>`, and
`limit=1..100`. Deleting a conversation cascades to its items and detaches
stored Response rows; deleting a stored Response does not delete conversation
items.
