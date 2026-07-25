import { canonicalJson, CanonicalJsonError } from './canonicalJson';
import {
  approvalDecision,
  activateSession,
  cancelTurn,
  CommandScope,
  RemoteCommandError,
  replayRequest,
  sessionCatalogRequest,
  startTurn,
  createSession,
} from './commands';

const scope: CommandScope = {
  messageId: '10000000-0000-4000-8000-000000000001',
  hostId: '20000000-0000-4000-8000-000000000002',
  deviceId: '30000000-0000-4000-8000-000000000003',
  sessionId: '40000000-0000-4000-8000-000000000004',
  commandId: '50000000-0000-4000-8000-000000000005',
  turnId: '60000000-0000-4000-8000-000000000006',
  sentAtUnixMs: 10,
};

describe('canonical remote commands', () => {
  test('sorts object keys recursively using JavaScript UTF-16 order', () => {
    expect(canonicalJson({ z: 1, a: { y: true, b: 'x' }, list: [{ d: 4, c: 3 }] })).toBe(
      '{"a":{"b":"x","y":true},"list":[{"c":3,"d":4}],"z":1}',
    );
    expect(() => canonicalJson(Number.NaN)).toThrow(CanonicalJsonError);
    expect(() => canonicalJson(1.5)).toThrow(CanonicalJsonError);
    expect(canonicalJson({ '\uE000': 1, '\u{10000}': 2 })).toBe(
      '{"𐀀":2,"":1}',
    );
  });

  test('builds a strict canonical start command', () => {
    const encoded = startTurn(scope, 'Run focused tests');
    expect(encoded).toBe(canonicalJson(JSON.parse(encoded)));
    expect(JSON.parse(encoded)).toMatchObject({
      kind: 'command',
      payload: {
        command: 'start_turn',
        command_id: scope.commandId,
        turn_id: scope.turnId,
        text: 'Run focused tests',
      },
    });
  });

  test('enforces UTF-8 prompt bounds and exact replay limits', () => {
    expect(() => startTurn(scope, '  ')).toThrow(RemoteCommandError);
    expect(() => startTurn(scope, 'é'.repeat(2049))).toThrow('too large');
    expect(() => replayRequest(scope, -1)).toThrow('sequence is invalid');
    expect(() => replayRequest(scope, 0, 257)).toThrow('limit is invalid');
    expect(JSON.parse(replayRequest(scope, 42)).payload).toEqual({ after_sequence: 42, limit: 256 });
  });

  test('allows only one-shot approval vocabulary and a full lowercase digest', () => {
    const encoded = approvalDecision(scope, {
      callId: '70000000-0000-4000-8000-000000000007',
      approvalId: '80000000-0000-4000-8000-000000000008',
      actionDigest: `sha256:${'a'.repeat(64)}`,
      decision: 'allow_once',
    });
    expect(JSON.parse(encoded).payload.decision).toBe('allow_once');
    expect(() =>
      approvalDecision(scope, {
        callId: '70000000-0000-4000-8000-000000000007',
        approvalId: '80000000-0000-4000-8000-000000000008',
        actionDigest: `sha256:${'A'.repeat(64)}`,
        decision: 'allow_once',
      }),
    ).toThrow('digest is invalid');
  });

  test('cancel command carries no steering or approval fields', () => {
    expect(JSON.parse(cancelTurn(scope)).payload).toEqual({
      command: 'cancel_turn',
      command_id: scope.commandId,
      turn_id: scope.turnId,
    });
  });

  test('session catalog request is host scoped and revision pinned after page one', () => {
    const first = JSON.parse(sessionCatalogRequest(scope));
    expect(first.session_id).toBeNull();
    expect(first.payload).toEqual({ cursor: null, limit: 64, revision: null });
    const next = JSON.parse(sessionCatalogRequest(scope, {
      cursor: { updatedAtUnixMs: 20, historyId: scope.sessionId },
      revision: `sha256:${'b'.repeat(64)}`,
      limit: 16,
    }));
    expect(next.payload.cursor).toEqual({
      updated_at_unix_ms: 20,
      history_id: scope.sessionId,
    });
    expect(() => sessionCatalogRequest(scope, {
      cursor: { updatedAtUnixMs: 20, historyId: scope.sessionId },
    })).toThrow('provided together');
  });

  test('create and activate session commands carry only the new session identity', () => {
    const nextSession = '70000000-0000-4000-8000-000000000007';
    expect(JSON.parse(createSession(scope, nextSession)).payload).toEqual({
      command: 'create_session',
      command_id: scope.commandId,
      session_id: nextSession,
    });
    expect(JSON.parse(activateSession(scope, nextSession)).payload).toEqual({
      command: 'activate_session',
      command_id: scope.commandId,
      session_id: nextSession,
    });
  });
});
