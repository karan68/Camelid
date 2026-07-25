import {
  parseCommandResult,
  parseEventBatch,
  parsePairResponse,
  parseRemoteEnvelope,
  parseReplayEnd,
  parseSessionCatalog,
  RemoteProtocolError,
} from './protocol';

const HOST_ID = '20000000-0000-4000-8000-000000000002';
const DEVICE_ID = '30000000-0000-4000-8000-000000000003';
const SESSION_ID = '40000000-0000-4000-8000-000000000004';
const MESSAGE_ID = '50000000-0000-4000-8000-000000000005';

function envelope(kind: string, payload: Record<string, unknown>) {
  return JSON.stringify({
    protocol: 'camelid.remote/v1',
    message_id: MESSAGE_ID,
    kind,
    host_id: HOST_ID,
    device_id: DEVICE_ID,
    session_id: SESSION_ID,
    sent_at_unix_ms: 10,
    payload,
  });
}

function remoteEvent(sequence: number, type = 'session.notice') {
  return {
    sequence,
    event_id: '60000000-0000-4000-8000-000000000006',
    turn_id: null,
    event: type,
    created_at_unix_ms: 10,
    payload: { content: 'ready' },
  };
}

describe('mobile remote protocol', () => {
  test('parses strict pairing response and binds the scanned host identity', () => {
    expect(
      parsePairResponse(
        JSON.stringify({
          v: 1,
          host_id: HOST_ID,
          device_id: DEVICE_ID,
          session_id: SESSION_ID,
          supported_capabilities: ['agent_events'],
        }),
        HOST_ID,
      ),
    ).toMatchObject({ host_id: HOST_ID, device_id: DEVICE_ID, session_id: SESSION_ID });
    expect(() =>
      parsePairResponse(
        JSON.stringify({
          v: 1,
          host_id: DEVICE_ID,
          device_id: DEVICE_ID,
          session_id: SESSION_ID,
          supported_capabilities: [],
        }),
        HOST_ID,
      ),
    ).toThrow('host identity changed');
  });

  test('accepts additive envelope metadata but rejects unknown message kinds', () => {
    const raw = JSON.parse(envelope('replay_end', {}));
    raw.trace_id = 'observation-only';
    expect(parseRemoteEnvelope(JSON.stringify(raw)).kind).toBe('replay_end');
    raw.kind = 'grant_admin';
    expect(() => parseRemoteEnvelope(JSON.stringify(raw))).toThrow(RemoteProtocolError);
  });

  test('rejects session-bound payloads when the envelope has no session identity', () => {
    const raw = JSON.parse(
      envelope('replay_end', {
        last_sequence: 0,
        has_more: false,
        session_state: 'idle',
      }),
    );
    raw.session_id = null;
    expect(() => parseReplayEnd(parseRemoteEnvelope(JSON.stringify(raw)))).toThrow(
      'Session identity is required',
    );
  });

  test('parses contiguous event batches and rejects gaps or privileged extra fields', () => {
    const parsed = parseRemoteEnvelope(
      envelope('event_batch', { events: [remoteEvent(4), remoteEvent(5, 'future.observation')] }),
    );
    expect(parseEventBatch(parsed).map((event) => event.sequence)).toEqual([4, 5]);

    expect(() =>
      parseEventBatch(
        parseRemoteEnvelope(envelope('event_batch', { events: [remoteEvent(4), remoteEvent(6)] })),
      ),
    ).toThrow('not contiguous');
    expect(() =>
      parseEventBatch(
        parseRemoteEnvelope(
          envelope('event_batch', { events: [{ ...remoteEvent(4), approval: 'always' }] }),
        ),
      ),
    ).toThrow('fields are invalid');
  });

  test('parses replay end and command result without trusting human text', () => {
    const end = parseReplayEnd(
      parseRemoteEnvelope(
        envelope('replay_end', {
          last_sequence: 42,
          has_more: false,
          session_state: 'waiting_approval',
        }),
      ),
    );
    expect(end).toEqual({ lastSequence: 42, hasMore: false, sessionState: 'waiting_approval' });

    const result = parseCommandResult(
      parseRemoteEnvelope(
        envelope('command_result', {
          command_id: MESSAGE_ID,
          status: 'rejected',
          code: 'stale_approval',
          message: 'Refresh state',
          current_event_sequence: 42,
        }),
      ),
    );
    expect(result.code).toBe('stale_approval');
    expect(result.status).toBe('rejected');
  });

  test('rejects duplicate capabilities, unsafe numbers, and unknown pairing authority', () => {
    expect(() =>
      parsePairResponse(
        JSON.stringify({
          v: 1,
          host_id: HOST_ID,
          device_id: DEVICE_ID,
          session_id: SESSION_ID,
          supported_capabilities: ['agent_events', 'agent_events'],
        }),
        HOST_ID,
      ),
    ).toThrow('Capabilities repeat');
    expect(() =>
      parseReplayEnd(
        parseRemoteEnvelope(
          envelope('replay_end', {
            last_sequence: Number.MAX_SAFE_INTEGER + 1,
            has_more: false,
            session_state: 'idle',
          }),
        ),
      ),
    ).toThrow(RemoteProtocolError);
    expect(() =>
      parsePairResponse(
        JSON.stringify({
          v: 1,
          host_id: HOST_ID,
          device_id: DEVICE_ID,
          session_id: SESSION_ID,
          supported_capabilities: [],
          persistent_approval: true,
        }),
        HOST_ID,
      ),
    ).toThrow('fields are invalid');
  });

  test('parses a host-scoped ordered session catalog and rejects inconsistent continuation', () => {
    const raw = JSON.parse(envelope('session_catalog', {
      active_session_id: SESSION_ID,
      revision: `sha256:${'a'.repeat(64)}`,
      sessions: [{
        history_id: SESSION_ID,
        source: 'remote',
        title: 'Fix parser',
        state: 'idle',
        canonical_root: 'C:\\work',
        model_id: 'qwen3_4b_q4_k_m',
        model_sha256: `sha256:${'b'.repeat(64)}`,
        created_at_unix_ms: 10,
        updated_at_unix_ms: 20,
        last_event_sequence: 4,
        active: true,
        continuable: true,
        refusal_code: null,
      }],
      next_cursor: null,
    }));
    raw.session_id = null;
    const catalog = parseSessionCatalog(parseRemoteEnvelope(JSON.stringify(raw)));
    expect(catalog.activeSessionId).toBe(SESSION_ID);
    expect(catalog.sessions[0]).toMatchObject({ title: 'Fix parser', active: true });
    raw.payload.sessions[0].continuable = false;
    expect(() => parseSessionCatalog(parseRemoteEnvelope(JSON.stringify(raw)))).toThrow(
      'continuation status is inconsistent',
    );
  });
});
