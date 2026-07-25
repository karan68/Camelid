import { HostStore, HostStoreError, ProtectedValueStore, StoredHost } from './hostStore';

class MemoryValues implements ProtectedValueStore {
  readonly values = new Map<string, string>();

  async get(key: string): Promise<string | null> {
    return this.values.get(key) ?? null;
  }

  async set(key: string, value: string): Promise<void> {
    this.values.set(key, value);
  }

  async remove(key: string): Promise<void> {
    this.values.delete(key);
  }
}

const HOST_ID = '20000000-0000-4000-8000-000000000002';
const DEVICE_ID = '30000000-0000-4000-8000-000000000003';
const SESSION_ID = '40000000-0000-4000-8000-000000000004';
const KEY_REFERENCE = '50000000-0000-4000-8000-000000000005';

function host(overrides: Partial<StoredHost> = {}): StoredHost {
  return {
    hostId: HOST_ID,
    label: 'Workstation',
    relayUrl: 'wss://relay.example.test',
    routeId: 'AAAAAAAAAAAAAAAAAAAAAA',
    hostNoisePublic: 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
    keyReference: KEY_REFERENCE,
    deviceId: DEVICE_ID,
    sessionId: SESSION_ID,
    lastAppliedSequence: 0,
    ...overrides,
  };
}

describe('protected host store', () => {
  test('persists only public pairing metadata and replay position', async () => {
    const values = new MemoryValues();
    const store = new HostStore(values);
    await store.save(host());

    expect(await store.list()).toEqual([host({
      supportedCapabilities: ['agent_events'],
      sessionCursors: { [SESSION_ID]: 0 },
    })]);
    const serialized = [...values.values.values()].join('\n');
    expect(serialized).not.toContain('pairing_secret');
    expect(serialized).not.toContain('private');
  });

  test('refuses replay cursor rollback', async () => {
    const store = new HostStore(new MemoryValues());
    await store.save(host());
    await store.updateLastAppliedSequence(HOST_ID, SESSION_ID, 42);
    await expect(store.updateLastAppliedSequence(HOST_ID, SESSION_ID, 41)).rejects.toThrow(
      'cannot move backwards',
    );
    expect((await store.list())[0].lastAppliedSequence).toBe(42);
    expect((await store.list())[0].sessionCursors?.[SESSION_ID]).toBe(42);
  });

  test('keeps independent replay cursors and resets the active alias on session switch', async () => {
    const store = new HostStore(new MemoryValues());
    const otherSession = '60000000-0000-4000-8000-000000000006';
    await store.save(host());
    await store.updateLastAppliedSequence(HOST_ID, SESSION_ID, 12);
    await store.updateLastAppliedSequence(HOST_ID, otherSession, 7);
    const switched = await store.updateActiveSession(HOST_ID, otherSession);
    expect(switched.lastAppliedSequence).toBe(7);
    expect(switched.sessionCursors).toEqual({ [SESSION_ID]: 12, [otherSession]: 7 });
  });

  test('fails closed on corrupt, incomplete, mismatched, or authority-bearing metadata', async () => {
    const indexes: string[] = [
      '{',
      JSON.stringify([HOST_ID, HOST_ID]),
    ];
    for (const index of indexes) {
      const values = new MemoryValues();
      values.values.set('camelid.remote.hosts.v1', index);
      await expect(new HostStore(values).list()).rejects.toBeInstanceOf(HostStoreError);
    }

    const values = new MemoryValues();
    values.values.set('camelid.remote.hosts.v1', JSON.stringify([HOST_ID]));
    values.values.set(
      `camelid.remote.host.v1.${HOST_ID}`,
      JSON.stringify({ ...host(), hostId: DEVICE_ID, privateKey: 'must-not-load' }),
    );
    await expect(new HostStore(values).list()).rejects.toBeInstanceOf(HostStoreError);

    const duplicateCapabilities = new HostStore(new MemoryValues());
    await expect(duplicateCapabilities.save(host({
      supportedCapabilities: ['agent_events', 'agent_events'],
    }))).rejects.toThrow('capabilities are invalid');
  });

  test('removes host metadata and index entry together', async () => {
    const values = new MemoryValues();
    const store = new HostStore(values);
    await store.save(host());
    await store.remove(HOST_ID);
    expect(await store.list()).toEqual([]);
    expect([...values.values.keys()].some((key) => key.endsWith(HOST_ID))).toBe(false);
  });
});
