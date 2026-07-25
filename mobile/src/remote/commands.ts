import { canonicalJson } from './canonicalJson';

const PROTOCOL = 'camelid.remote/v1';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SHA256 = /^sha256:[0-9a-f]{64}$/;
const MAX_TURN_TEXT_BYTES = 4096;
const MAX_REPLAY_EVENTS = 256;
const MAX_SESSION_CATALOG_ENTRIES = 64;

export interface MessageScope {
  messageId: string;
  hostId: string;
  deviceId: string;
  sessionId: string;
  sentAtUnixMs: number;
}

export interface CommandScope extends MessageScope {
  commandId: string;
  turnId: string;
}

export type HostMessageScope = Omit<MessageScope, 'sessionId'>;

export type ApprovalDecision = 'allow_once' | 'deny' | 'abort_turn';

export class RemoteCommandError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RemoteCommandError';
  }
}

export function startTurn(scope: CommandScope, text: string): string {
  if (text.trim().length === 0) throw new RemoteCommandError('Turn text is empty.');
  if (utf8Length(text) > MAX_TURN_TEXT_BYTES) throw new RemoteCommandError('Turn text is too large.');
  return command(scope, { command: 'start_turn', command_id: scope.commandId, turn_id: scope.turnId, text });
}

export function cancelTurn(scope: CommandScope): string {
  return command(scope, { command: 'cancel_turn', command_id: scope.commandId, turn_id: scope.turnId });
}

export function createSession(scope: MessageScope & { commandId: string }, sessionId: string): string {
  requireUuid(sessionId, 'session ID');
  return sessionCommand(scope, {
    command: 'create_session',
    command_id: scope.commandId,
    session_id: sessionId,
  });
}

export function activateSession(scope: MessageScope & { commandId: string }, sessionId: string): string {
  requireUuid(sessionId, 'session ID');
  return sessionCommand(scope, {
    command: 'activate_session',
    command_id: scope.commandId,
    session_id: sessionId,
  });
}

export function approvalDecision(
  scope: CommandScope,
  approval: { callId: string; approvalId: string; actionDigest: string; decision: ApprovalDecision },
): string {
  requireUuid(approval.callId, 'call ID');
  requireUuid(approval.approvalId, 'approval ID');
  if (!SHA256.test(approval.actionDigest)) throw new RemoteCommandError('Action digest is invalid.');
  return command(scope, {
    command: 'approval_decision',
    command_id: scope.commandId,
    turn_id: scope.turnId,
    call_id: approval.callId,
    approval_id: approval.approvalId,
    action_digest: approval.actionDigest,
    decision: approval.decision,
  });
}

export function replayRequest(scope: MessageScope, afterSequence: number, limit = MAX_REPLAY_EVENTS): string {
  if (!Number.isSafeInteger(afterSequence) || afterSequence < 0) {
    throw new RemoteCommandError('Replay sequence is invalid.');
  }
  if (!Number.isInteger(limit) || limit < 1 || limit > MAX_REPLAY_EVENTS) {
    throw new RemoteCommandError('Replay limit is invalid.');
  }
  return envelope(scope, 'replay_request', { after_sequence: afterSequence, limit });
}

export function sessionCatalogRequest(
  scope: HostMessageScope,
  options: {
    cursor?: { updatedAtUnixMs: number; historyId: string };
    revision?: string;
    limit?: number;
  } = {},
): string {
  const limit = options.limit ?? MAX_SESSION_CATALOG_ENTRIES;
  if (!Number.isInteger(limit) || limit < 1 || limit > MAX_SESSION_CATALOG_ENTRIES) {
    throw new RemoteCommandError('Session catalog limit is invalid.');
  }
  const hasCursor = options.cursor !== undefined;
  const hasRevision = options.revision !== undefined;
  if (hasCursor !== hasRevision) {
    throw new RemoteCommandError('Session catalog cursor and revision must be provided together.');
  }
  if (options.cursor !== undefined) {
    if (!Number.isSafeInteger(options.cursor.updatedAtUnixMs) || options.cursor.updatedAtUnixMs < 0) {
      throw new RemoteCommandError('Session catalog cursor time is invalid.');
    }
    requireUuid(options.cursor.historyId, 'session history ID');
  }
  if (options.revision !== undefined && !SHA256.test(options.revision)) {
    throw new RemoteCommandError('Session catalog revision is invalid.');
  }
  return hostEnvelope(scope, 'session_catalog_request', {
    cursor: options.cursor === undefined ? null : {
      updated_at_unix_ms: options.cursor.updatedAtUnixMs,
      history_id: options.cursor.historyId,
    },
    limit,
    revision: options.revision ?? null,
  });
}

function command(scope: CommandScope, payload: Readonly<Record<string, unknown>>): string {
  requireUuid(scope.commandId, 'command ID');
  requireUuid(scope.turnId, 'turn ID');
  return envelope(scope, 'command', payload);
}

function sessionCommand(
  scope: MessageScope & { commandId: string },
  payload: Readonly<Record<string, unknown>>,
): string {
  requireUuid(scope.commandId, 'command ID');
  return envelope(scope, 'command', payload);
}

function envelope(
  scope: MessageScope,
  kind: 'command' | 'replay_request',
  payload: Readonly<Record<string, unknown>>,
): string {
  requireUuid(scope.messageId, 'message ID');
  requireUuid(scope.hostId, 'host ID');
  requireUuid(scope.deviceId, 'device ID');
  requireUuid(scope.sessionId, 'session ID');
  if (!Number.isSafeInteger(scope.sentAtUnixMs) || scope.sentAtUnixMs < 0) {
    throw new RemoteCommandError('Message timestamp is invalid.');
  }
  return canonicalJson({
    protocol: PROTOCOL,
    message_id: scope.messageId,
    kind,
    host_id: scope.hostId,
    device_id: scope.deviceId,
    session_id: scope.sessionId,
    sent_at_unix_ms: scope.sentAtUnixMs,
    payload,
  });
}

function hostEnvelope(
  scope: HostMessageScope,
  kind: 'session_catalog_request',
  payload: Readonly<Record<string, unknown>>,
): string {
  requireUuid(scope.messageId, 'message ID');
  requireUuid(scope.hostId, 'host ID');
  requireUuid(scope.deviceId, 'device ID');
  if (!Number.isSafeInteger(scope.sentAtUnixMs) || scope.sentAtUnixMs < 0) {
    throw new RemoteCommandError('Message timestamp is invalid.');
  }
  return canonicalJson({
    protocol: PROTOCOL,
    message_id: scope.messageId,
    kind,
    host_id: scope.hostId,
    device_id: scope.deviceId,
    session_id: null,
    sent_at_unix_ms: scope.sentAtUnixMs,
    payload,
  });
}

function requireUuid(value: string, field: string): void {
  if (!UUID.test(value)) throw new RemoteCommandError(`${field} is invalid.`);
}

function utf8Length(value: string): number {
  let bytes = 0;
  for (const character of value) {
    const point = character.codePointAt(0) ?? 0;
    bytes += point <= 0x7f ? 1 : point <= 0x7ff ? 2 : point <= 0xffff ? 3 : 4;
  }
  return bytes;
}
