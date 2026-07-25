import type { StoredHost } from './hostStore';
import type { RemoteEnvelope, SessionCatalog } from './protocol';
import { canControlSelectedSession, mergeSessionCatalogPage, messageTargetsSelectedSession } from './sessionSelection';

const ACTIVE_SESSION = '40000000-0000-4000-8000-000000000004';
const HISTORY_SESSION = '50000000-0000-4000-8000-000000000005';

const host: StoredHost = {
  hostId: '20000000-0000-4000-8000-000000000002',
  label: 'Workstation',
  relayUrl: 'wss://relay.example.test/v1/connect',
  routeId: 'AAAAAAAAAAAAAAAAAAAAAA',
  hostNoisePublic: 'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
  keyReference: '30000000-0000-4000-8000-000000000003',
  deviceId: '60000000-0000-4000-8000-000000000006',
  sessionId: ACTIVE_SESSION,
  lastAppliedSequence: 0,
  supportedCapabilities: ['agent_events', 'session_catalog_v1'],
};

const catalog: SessionCatalog = {
  activeSessionId: ACTIVE_SESSION,
  revision: `sha256:${'a'.repeat(64)}`,
  sessions: [],
  nextCursor: null,
};

function envelope(sessionId: string | null): RemoteEnvelope {
  return {
    protocol: 'camelid.remote/v1',
    message_id: '70000000-0000-4000-8000-000000000007',
    kind: 'event_batch',
    host_id: host.hostId,
    device_id: host.deviceId,
    session_id: sessionId,
    sent_at_unix_ms: 10,
    payload: {},
  };
}

describe('mobile session selection authority', () => {
  test('routes events only into their selected history projection', () => {
    expect(messageTargetsSelectedSession(envelope(HISTORY_SESSION), HISTORY_SESSION)).toBe(true);
    expect(messageTargetsSelectedSession(envelope(ACTIVE_SESSION), HISTORY_SESSION)).toBe(false);
    expect(messageTargetsSelectedSession(envelope(null), HISTORY_SESSION)).toBe(false);
  });

  test('exposes command authority only for the catalog active session', () => {
    expect(canControlSelectedSession(host, catalog, ACTIVE_SESSION)).toBe(true);
    expect(canControlSelectedSession(host, catalog, HISTORY_SESSION)).toBe(false);
    expect(canControlSelectedSession(host, null, ACTIVE_SESSION)).toBe(true);
    expect(canControlSelectedSession(null, catalog, ACTIVE_SESSION)).toBe(false);
  });

  test('replaces unsolicited catalog snapshots but revision-pins pagination', () => {
    const oldPage = {
      ...catalog,
      sessions: [{ historyId: ACTIVE_SESSION } as SessionCatalog['sessions'][number]],
      nextCursor: {
        updatedAtUnixMs: 10,
        historyId: ACTIVE_SESSION,
      },
    };
    const refreshed = {
      ...catalog,
      revision: `sha256:${'b'.repeat(64)}`,
      sessions: [{ historyId: HISTORY_SESSION } as SessionCatalog['sessions'][number]],
    };
    expect(mergeSessionCatalogPage(oldPage, refreshed, null)).toEqual(refreshed);

    expect(() => mergeSessionCatalogPage(oldPage, refreshed, oldPage.revision)).toThrow(
      'Remote session catalog changed during pagination.',
    );
  });
});
