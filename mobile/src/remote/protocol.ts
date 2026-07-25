import type { RemoteEvent } from './reducer';

const PROTOCOL = 'camelid.remote/v1';
const MAX_INNER_MESSAGE_BYTES = 1_114_112;
const MAX_REPLAY_EVENTS = 256;
const MAX_RESULT_MESSAGE_BYTES = 2048;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const TOKEN = /^[a-z0-9._-]+$/;
const SHA256 = /^sha256:[0-9a-f]{64}$/;

export type MessageKind =
  | 'command'
  | 'command_result'
  | 'event_batch'
  | 'replay_request'
  | 'replay_end'
  | 'session_catalog_request'
  | 'session_catalog'
  | 'ping'
  | 'pong'
  | 'error';

export interface RemoteEnvelope {
  protocol: typeof PROTOCOL;
  message_id: string;
  kind: MessageKind;
  host_id: string;
  device_id: string;
  session_id: string | null;
  sent_at_unix_ms: number;
  payload: Readonly<Record<string, unknown>>;
}

export interface PairResponse {
  v: 1;
  host_id: string;
  device_id: string;
  session_id: string;
  supported_capabilities: readonly string[];
}

export interface ReplayEnd {
  lastSequence: number;
  hasMore: boolean;
  sessionState: 'armed' | 'idle' | 'running' | 'waiting_approval' | 'cancelling' | 'failed' | 'closed';
}

export interface CommandResult {
  commandId: string;
  status: 'accepted' | 'applied' | 'rejected';
  code: string;
  message: string;
  currentEventSequence: number;
}

export interface SessionCatalogCursor {
  updatedAtUnixMs: number;
  historyId: string;
}

export interface SessionSummary {
  historyId: string;
  source: 'remote' | 'agent_saved';
  title: string;
  state: ReplayEnd['sessionState'];
  canonicalRoot: string;
  modelId: string;
  modelSha256: string | null;
  createdAtUnixMs: number;
  updatedAtUnixMs: number;
  lastEventSequence: number;
  active: boolean;
  continuable: boolean;
  refusalCode: string | null;
}

export interface SessionCatalog {
  activeSessionId: string;
  revision: string;
  sessions: readonly SessionSummary[];
  nextCursor: SessionCatalogCursor | null;
}

export class RemoteProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RemoteProtocolError';
  }
}

export function parsePairResponse(input: string, expectedHostId: string): PairResponse {
  const value = parseJsonObject(input, 4096, 'Pairing response');
  exactKeys(value, ['v', 'host_id', 'device_id', 'session_id', 'supported_capabilities']);
  if (value.v !== 1) throw new RemoteProtocolError('Pairing response version is unsupported.');
  const hostId = uuid(value, 'host_id');
  if (hostId !== expectedHostId) throw new RemoteProtocolError('Pairing response host identity changed.');
  const capabilities = stringArray(value.supported_capabilities, 'supported_capabilities', 16);
  validateUniqueTokens(capabilities);
  return {
    v: 1,
    host_id: hostId,
    device_id: uuid(value, 'device_id'),
    session_id: uuid(value, 'session_id'),
    supported_capabilities: capabilities,
  };
}

export function parseRemoteEnvelope(input: string): RemoteEnvelope {
  const value = parseJsonObject(input, MAX_INNER_MESSAGE_BYTES, 'Remote message');
  if (value.protocol !== PROTOCOL) throw new RemoteProtocolError('Remote protocol is unsupported.');
  const kind = value.kind;
  if (!isMessageKind(kind)) throw new RemoteProtocolError('Remote message kind is unsupported.');
  const sessionId = value.session_id;
  if (sessionId !== null && (typeof sessionId !== 'string' || !UUID.test(sessionId))) {
    throw new RemoteProtocolError('Remote session identity is invalid.');
  }
  if (!Number.isSafeInteger(value.sent_at_unix_ms) || (value.sent_at_unix_ms as number) < 0) {
    throw new RemoteProtocolError('Remote timestamp is invalid.');
  }
  if (!isRecord(value.payload)) throw new RemoteProtocolError('Remote payload is invalid.');
  return {
    protocol: PROTOCOL,
    message_id: uuid(value, 'message_id'),
    kind,
    host_id: uuid(value, 'host_id'),
    device_id: uuid(value, 'device_id'),
    session_id: sessionId,
    sent_at_unix_ms: value.sent_at_unix_ms as number,
    payload: value.payload,
  };
}

export function parseEventBatch(message: RemoteEnvelope): readonly RemoteEvent[] {
  requireKind(message, 'event_batch');
  requireSession(message);
  exactKeys(message.payload, ['events']);
  if (!Array.isArray(message.payload.events)) throw new RemoteProtocolError('Event batch is invalid.');
  if (message.payload.events.length < 1 || message.payload.events.length > MAX_REPLAY_EVENTS) {
    throw new RemoteProtocolError('Event batch size is invalid.');
  }
  const events = message.payload.events.map(parseEvent);
  for (let index = 1; index < events.length; index += 1) {
    if (events[index].sequence !== events[index - 1].sequence + 1) {
      throw new RemoteProtocolError('Event batch sequence is not contiguous.');
    }
  }
  return events;
}

export function parseReplayEnd(message: RemoteEnvelope): ReplayEnd {
  requireKind(message, 'replay_end');
  requireSession(message);
  exactKeys(message.payload, ['last_sequence', 'has_more', 'session_state']);
  const lastSequence = safeSequence(message.payload.last_sequence, true);
  if (typeof message.payload.has_more !== 'boolean') {
    throw new RemoteProtocolError('Replay continuation flag is invalid.');
  }
  const state = message.payload.session_state;
  if (!isSessionState(state)) throw new RemoteProtocolError('Replay session state is invalid.');
  return { lastSequence, hasMore: message.payload.has_more, sessionState: state };
}

export function parseCommandResult(message: RemoteEnvelope): CommandResult {
  requireKind(message, 'command_result');
  requireSession(message);
  exactKeys(message.payload, [
    'command_id',
    'status',
    'code',
    'message',
    'current_event_sequence',
  ]);
  const status = message.payload.status;
  if (status !== 'accepted' && status !== 'applied' && status !== 'rejected') {
    throw new RemoteProtocolError('Command result status is invalid.');
  }
  const code = token(message.payload, 'code');
  const text = requiredString(message.payload, 'message');
  if (utf8Length(text) > MAX_RESULT_MESSAGE_BYTES) {
    throw new RemoteProtocolError('Command result message is too large.');
  }
  return {
    commandId: uuid(message.payload, 'command_id'),
    status,
    code,
    message: text,
    currentEventSequence: safeSequence(message.payload.current_event_sequence, true),
  };
}

export function parseSessionCatalog(message: RemoteEnvelope): SessionCatalog {
  requireKind(message, 'session_catalog');
  if (message.session_id !== null) throw new RemoteProtocolError('Session catalog must be host scoped.');
  exactKeys(message.payload, ['active_session_id', 'revision', 'sessions', 'next_cursor']);
  const revision = requiredString(message.payload, 'revision');
  if (!SHA256.test(revision)) throw new RemoteProtocolError('Session catalog revision is invalid.');
  if (!Array.isArray(message.payload.sessions) || message.payload.sessions.length > 64) {
    throw new RemoteProtocolError('Session catalog entries are invalid.');
  }
  const sessions = message.payload.sessions.map(parseSessionSummary);
  for (let index = 1; index < sessions.length; index += 1) {
    const previous = sessions[index - 1];
    const current = sessions[index];
    if (
      previous.updatedAtUnixMs < current.updatedAtUnixMs ||
      (previous.updatedAtUnixMs === current.updatedAtUnixMs && previous.historyId >= current.historyId)
    ) {
      throw new RemoteProtocolError('Session catalog ordering is invalid.');
    }
  }
  const cursorValue = message.payload.next_cursor;
  const nextCursor = cursorValue === null ? null : parseCatalogCursor(cursorValue);
  return {
    activeSessionId: uuid(message.payload, 'active_session_id'),
    revision,
    sessions,
    nextCursor,
  };
}

function parseSessionSummary(value: unknown): SessionSummary {
  if (!isRecord(value)) throw new RemoteProtocolError('Session summary is invalid.');
  exactKeys(value, [
    'history_id', 'source', 'title', 'state', 'canonical_root', 'model_id', 'model_sha256',
    'created_at_unix_ms', 'updated_at_unix_ms', 'last_event_sequence', 'active', 'continuable',
    'refusal_code',
  ]);
  if (value.source !== 'remote' && value.source !== 'agent_saved') {
    throw new RemoteProtocolError('Session history source is invalid.');
  }
  if (!isSessionState(value.state)) throw new RemoteProtocolError('Session state is invalid.');
  const title = requiredString(value, 'title');
  const canonicalRoot = requiredString(value, 'canonical_root');
  const modelId = token(value, 'model_id');
  const modelSha256 = value.model_sha256;
  if (modelSha256 !== null && (typeof modelSha256 !== 'string' || !SHA256.test(modelSha256))) {
    throw new RemoteProtocolError('Session model digest is invalid.');
  }
  if (typeof value.active !== 'boolean' || typeof value.continuable !== 'boolean') {
    throw new RemoteProtocolError('Session authority flags are invalid.');
  }
  const refusalCode = value.refusal_code;
  if (refusalCode !== null && (typeof refusalCode !== 'string' || !TOKEN.test(refusalCode))) {
    throw new RemoteProtocolError('Session refusal code is invalid.');
  }
  if ((value.continuable && refusalCode !== null) || (!value.continuable && refusalCode === null)) {
    throw new RemoteProtocolError('Session continuation status is inconsistent.');
  }
  if (value.continuable && modelSha256 === null) {
    throw new RemoteProtocolError('Continuable session model identity is incomplete.');
  }
  return {
    historyId: uuid(value, 'history_id'),
    source: value.source,
    title,
    state: value.state,
    canonicalRoot,
    modelId,
    modelSha256,
    createdAtUnixMs: safeSequence(value.created_at_unix_ms, true),
    updatedAtUnixMs: safeSequence(value.updated_at_unix_ms, true),
    lastEventSequence: safeSequence(value.last_event_sequence, true),
    active: value.active,
    continuable: value.continuable,
    refusalCode,
  };
}

function parseCatalogCursor(value: unknown): SessionCatalogCursor {
  if (!isRecord(value)) throw new RemoteProtocolError('Session catalog cursor is invalid.');
  exactKeys(value, ['updated_at_unix_ms', 'history_id']);
  return {
    updatedAtUnixMs: safeSequence(value.updated_at_unix_ms, true),
    historyId: uuid(value, 'history_id'),
  };
}

function parseEvent(value: unknown): RemoteEvent {
  if (!isRecord(value)) throw new RemoteProtocolError('Remote event is invalid.');
  exactKeys(value, ['sequence', 'event_id', 'turn_id', 'event', 'created_at_unix_ms', 'payload']);
  const turnId = value.turn_id;
  if (turnId !== null && (typeof turnId !== 'string' || !UUID.test(turnId))) {
    throw new RemoteProtocolError('Event turn identity is invalid.');
  }
  uuid(value, 'event_id');
  safeSequence(value.created_at_unix_ms, true);
  if (!isRecord(value.payload)) throw new RemoteProtocolError('Event payload is invalid.');
  return {
    sequence: safeSequence(value.sequence, false),
    type: token(value, 'event'),
    payload: value.payload,
    turnId,
  };
}

function parseJsonObject(input: string, maxBytes: number, name: string): Readonly<Record<string, unknown>> {
  if (utf8Length(input) > maxBytes) throw new RemoteProtocolError(`${name} is too large.`);
  let value: unknown;
  try {
    value = JSON.parse(input);
  } catch {
    throw new RemoteProtocolError(`${name} is not valid JSON.`);
  }
  if (!isRecord(value)) throw new RemoteProtocolError(`${name} must be an object.`);
  return value;
}

function exactKeys(value: Readonly<Record<string, unknown>>, allowed: readonly string[]): void {
  const expected = new Set(allowed);
  if (Object.keys(value).length !== allowed.length || Object.keys(value).some((key) => !expected.has(key))) {
    throw new RemoteProtocolError('Privileged payload fields are invalid.');
  }
}

function requiredString(value: Readonly<Record<string, unknown>>, key: string): string {
  const field = value[key];
  if (typeof field !== 'string') throw new RemoteProtocolError(`${key} is invalid.`);
  return field;
}

function uuid(value: Readonly<Record<string, unknown>>, key: string): string {
  const field = requiredString(value, key);
  if (!UUID.test(field)) throw new RemoteProtocolError(`${key} is invalid.`);
  return field;
}

function token(value: Readonly<Record<string, unknown>>, key: string): string {
  const field = requiredString(value, key);
  if (utf8Length(field) > 64 || !TOKEN.test(field)) throw new RemoteProtocolError(`${key} is invalid.`);
  return field;
}

function stringArray(value: unknown, name: string, maximum: number): string[] {
  if (!Array.isArray(value) || value.length > maximum || value.some((item) => typeof item !== 'string')) {
    throw new RemoteProtocolError(`${name} is invalid.`);
  }
  return value as string[];
}

function validateUniqueTokens(values: readonly string[]): void {
  if (new Set(values).size !== values.length) throw new RemoteProtocolError('Capabilities repeat.');
  for (const value of values) {
    if (utf8Length(value) > 64 || !TOKEN.test(value)) {
      throw new RemoteProtocolError('Capability token is invalid.');
    }
  }
}

function safeSequence(value: unknown, allowZero: boolean): number {
  if (!Number.isSafeInteger(value) || (value as number) < (allowZero ? 0 : 1)) {
    throw new RemoteProtocolError('Sequence is invalid.');
  }
  return value as number;
}

function requireKind(message: RemoteEnvelope, kind: MessageKind): void {
  if (message.kind !== kind) throw new RemoteProtocolError(`Expected ${kind} message.`);
}

function requireSession(message: RemoteEnvelope): void {
  if (message.session_id === null) throw new RemoteProtocolError('Session identity is required.');
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isMessageKind(value: unknown): value is MessageKind {
  return ['command', 'command_result', 'event_batch', 'replay_request', 'replay_end', 'session_catalog_request', 'session_catalog', 'ping', 'pong', 'error'].includes(
    value as string,
  );
}

function isSessionState(value: unknown): value is ReplayEnd['sessionState'] {
  return ['armed', 'idle', 'running', 'waiting_approval', 'cancelling', 'failed', 'closed'].includes(
    value as string,
  );
}

function utf8Length(value: string): number {
  let bytes = 0;
  for (const character of value) {
    const point = character.codePointAt(0) ?? 0;
    bytes += point <= 0x7f ? 1 : point <= 0x7ff ? 2 : point <= 0xffff ? 3 : 4;
  }
  return bytes;
}
