const INDEX_KEY = 'camelid.remote.hosts.v1';
const MAX_HOSTS = 16;
const MAX_SESSION_CURSORS = 256;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const BASE64URL = /^[A-Za-z0-9_-]+$/;

export interface ProtectedValueStore {
  get(key: string): Promise<string | null>;
  set(key: string, value: string): Promise<void>;
  remove(key: string): Promise<void>;
}

export interface StoredHost {
  hostId: string;
  label: string;
  relayUrl: string;
  routeId: string;
  hostNoisePublic: string;
  keyReference: string;
  deviceId: string;
  sessionId: string;
  lastAppliedSequence: number;
  sessionCursors?: Readonly<Record<string, number>>;
  supportedCapabilities?: readonly string[];
}

export class HostStoreError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'HostStoreError';
  }
}

export class HostStore {
  constructor(private readonly values: ProtectedValueStore) {}

  async list(): Promise<readonly StoredHost[]> {
    const ids = await this.readIndex();
    const hosts: StoredHost[] = [];
    for (const hostId of ids) {
      const raw = await this.values.get(hostKey(hostId));
      if (raw === null) throw new HostStoreError('Protected host metadata is incomplete.');
      hosts.push(parseStoredHost(raw, hostId));
    }
    return hosts;
  }

  async save(host: StoredHost): Promise<void> {
    const normalized = normalizeHost(host);
    const ids = await this.readIndex();
    const nextIds = ids.includes(normalized.hostId) ? ids : [...ids, normalized.hostId];
    if (nextIds.length > MAX_HOSTS) throw new HostStoreError('Paired host limit reached.');

    await this.values.set(hostKey(normalized.hostId), JSON.stringify(normalized));
    await this.values.set(INDEX_KEY, JSON.stringify(nextIds));
  }

  async updateLastAppliedSequence(
    hostId: string,
    sessionId: string,
    sequence: number,
  ): Promise<StoredHost> {
    if (!UUID.test(sessionId)) throw new HostStoreError('Session identity is invalid.');
    if (!Number.isSafeInteger(sequence) || sequence < 0) {
      throw new HostStoreError('Replay sequence is invalid.');
    }
    const raw = await this.values.get(hostKey(hostId));
    if (raw === null) throw new HostStoreError('Paired host does not exist.');
    const host = parseStoredHost(raw, hostId);
    const previous = host.sessionCursors?.[sessionId] ?? 0;
    if (sequence < previous) {
      throw new HostStoreError('Replay sequence cannot move backwards.');
    }
    const cursors = { ...host.sessionCursors, [sessionId]: sequence };
    while (Object.keys(cursors).length > MAX_SESSION_CURSORS) {
      const evicted = Object.keys(cursors)
        .sort()
        .find((candidate) => candidate !== host.sessionId && candidate !== sessionId);
      if (evicted === undefined) throw new HostStoreError('Session replay cursor limit reached.');
      delete cursors[evicted];
    }
    const next = {
      ...host,
      lastAppliedSequence: sessionId === host.sessionId ? sequence : host.lastAppliedSequence,
      sessionCursors: cursors,
    };
    await this.values.set(hostKey(hostId), JSON.stringify(next));
    return next;
  }

  async updateActiveSession(hostId: string, sessionId: string): Promise<StoredHost> {
    if (!UUID.test(sessionId)) throw new HostStoreError('Session identity is invalid.');
    const raw = await this.values.get(hostKey(hostId));
    if (raw === null) throw new HostStoreError('Paired host does not exist.');
    const host = parseStoredHost(raw, hostId);
    const next = {
      ...host,
      sessionId,
      lastAppliedSequence: host.sessionCursors?.[sessionId] ?? 0,
      sessionCursors: { ...host.sessionCursors, [sessionId]: host.sessionCursors?.[sessionId] ?? 0 },
    };
    await this.values.set(hostKey(hostId), JSON.stringify(next));
    return next;
  }

  async remove(hostId: string): Promise<void> {
    if (!UUID.test(hostId)) throw new HostStoreError('Host identity is invalid.');
    const ids = await this.readIndex();
    await this.values.remove(hostKey(hostId));
    await this.values.set(INDEX_KEY, JSON.stringify(ids.filter((id) => id !== hostId)));
  }

  private async readIndex(): Promise<string[]> {
    const raw = await this.values.get(INDEX_KEY);
    if (raw === null) return [];
    let decoded: unknown;
    try {
      decoded = JSON.parse(raw);
    } catch {
      throw new HostStoreError('Protected host index is corrupt.');
    }
    if (
      !Array.isArray(decoded) ||
      decoded.length > MAX_HOSTS ||
      decoded.some((value) => typeof value !== 'string' || !UUID.test(value)) ||
      new Set(decoded).size !== decoded.length
    ) {
      throw new HostStoreError('Protected host index is invalid.');
    }
    return decoded;
  }
}

function normalizeHost(host: StoredHost): StoredHost {
  if (
    typeof host.hostId !== 'string' ||
    typeof host.keyReference !== 'string' ||
    typeof host.deviceId !== 'string' ||
    !UUID.test(host.hostId) ||
    typeof host.sessionId !== 'string' ||
    !UUID.test(host.keyReference) ||
    !UUID.test(host.deviceId) ||
    !UUID.test(host.sessionId)
  ) {
    throw new HostStoreError('Host or device identity is invalid.');
  }
  if (typeof host.label !== 'string' || host.label.trim().length === 0 || host.label.length > 128) {
    throw new HostStoreError('Host label is invalid.');
  }
  if (typeof host.relayUrl !== 'string') throw new HostStoreError('Relay URL is invalid.');
  let relay: URL;
  try {
    relay = new URL(host.relayUrl);
  } catch {
    throw new HostStoreError('Relay URL is invalid.');
  }
  if (
    relay.protocol !== 'wss:' ||
    relay.hostname.length === 0 ||
    relay.username ||
    relay.password ||
    relay.hash
  ) {
    throw new HostStoreError('Relay URL is not secure.');
  }
  if (typeof host.routeId !== 'string' || host.routeId.length !== 22 || !BASE64URL.test(host.routeId)) {
    throw new HostStoreError('Route identity is invalid.');
  }
  if (
    typeof host.hostNoisePublic !== 'string' ||
    host.hostNoisePublic.length !== 43 ||
    !BASE64URL.test(host.hostNoisePublic)
  ) {
    throw new HostStoreError('Host public key is invalid.');
  }
  if (!Number.isSafeInteger(host.lastAppliedSequence) || host.lastAppliedSequence < 0) {
    throw new HostStoreError('Replay sequence is invalid.');
  }
  const supportedCapabilities = host.supportedCapabilities ?? ['agent_events'];
  const sessionCursors = host.sessionCursors ?? { [host.sessionId]: host.lastAppliedSequence };
  const cursorEntries = Object.entries(sessionCursors);
  if (
    cursorEntries.length > MAX_SESSION_CURSORS ||
    cursorEntries.some(([sessionId, sequence]) =>
      !UUID.test(sessionId) || !Number.isSafeInteger(sequence) || sequence < 0
    )
  ) {
    throw new HostStoreError('Session replay cursors are invalid.');
  }
  if (
    !Array.isArray(supportedCapabilities) ||
    supportedCapabilities.length > 16 ||
    supportedCapabilities.some(
      (value) => typeof value !== 'string' || !/^[a-z0-9._-]{1,64}$/.test(value),
    ) ||
    new Set(supportedCapabilities).size !== supportedCapabilities.length
  ) {
    throw new HostStoreError('Host capabilities are invalid.');
  }
  return {
    hostId: host.hostId,
    label: host.label.trim(),
    relayUrl: host.relayUrl,
    routeId: host.routeId,
    hostNoisePublic: host.hostNoisePublic,
    keyReference: host.keyReference,
    deviceId: host.deviceId,
    sessionId: host.sessionId,
    lastAppliedSequence: host.lastAppliedSequence,
    sessionCursors,
    supportedCapabilities,
  };
}

function parseStoredHost(raw: string, expectedHostId: string): StoredHost {
  let decoded: unknown;
  try {
    decoded = JSON.parse(raw);
  } catch {
    throw new HostStoreError('Protected host metadata is corrupt.');
  }
  if (!isRecord(decoded)) throw new HostStoreError('Protected host metadata is invalid.');
  const allowed = new Set([
    'hostId',
    'label',
    'relayUrl',
    'routeId',
    'hostNoisePublic',
    'keyReference',
    'deviceId',
    'sessionId',
    'lastAppliedSequence',
    'sessionCursors',
    'supportedCapabilities',
  ]);
  if (Object.keys(decoded).some((key) => !allowed.has(key))) {
    throw new HostStoreError('Protected host metadata contains unsupported fields.');
  }
  const host = normalizeHost(decoded as unknown as StoredHost);
  if (host.hostId !== expectedHostId) throw new HostStoreError('Protected host identity mismatch.');
  return host;
}

function hostKey(hostId: string): string {
  if (!UUID.test(hostId)) throw new HostStoreError('Host identity is invalid.');
  return `camelid.remote.host.v1.${hostId}`;
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
