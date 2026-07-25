import type {
  DeviceIdentity,
  HandshakeHandle,
  KeyReference,
  TransportHandle,
} from '../../modules/camelid-remote-crypto';

import { HostStore, ProtectedValueStore } from './hostStore';
import { PairingClientError, PairingCryptoPort, PairingSocket, pairHost } from './pairingClient';
import type { PairingQr } from './pairingQr';

const HOST_ID = '20000000-0000-4000-8000-000000000002';
const DEVICE_ID = '30000000-0000-4000-8000-000000000003';
const SESSION_ID = '40000000-0000-4000-8000-000000000004';
const KEY_REFERENCE = '50000000-0000-4000-8000-000000000005' as KeyReference;
const HANDSHAKE = '60000000-0000-4000-8000-000000000006' as HandshakeHandle;
const TRANSPORT = '70000000-0000-4000-8000-000000000007' as TransportHandle;
const NOW = 1_780_000_000_000;

class MemoryValues implements ProtectedValueStore {
  readonly values = new Map<string, string>();
  async get(key: string) { return this.values.get(key) ?? null; }
  async set(key: string, value: string) { this.values.set(key, value); }
  async remove(key: string) { this.values.delete(key); }
}

class FakeSocket implements PairingSocket {
  binaryType = '';
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  sent: ArrayBuffer | null = null;
  closed = false;
  response: unknown = Uint8Array.from([4, 5, 6]).buffer;

  send(data: ArrayBuffer) {
    this.sent = data;
    this.onmessage?.({ data: this.response });
  }

  close() { this.closed = true; }
}

function qr(): PairingQr {
  return {
    v: 1,
    relay_url: 'wss://relay.example.test/v1/connect',
    route_id: 'AAAAAAAAAAAAAAAAAAAAAA',
    host_id: HOST_ID,
    host_noise_public: 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
    pairing_secret: 'BBBBBBBBBBBBBBBBBBBBBB',
    expires_at_unix_ms: NOW + 300_000,
  };
}

function crypto(responseHostId = HOST_ID) {
  const identity: DeviceIdentity = {
    keyReference: KEY_REFERENCE,
    publicKey: 'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC',
    protection: 'android_keystore_wrapped',
  };
  const port: PairingCryptoPort = {
    createDeviceIdentityAsync: jest.fn(async () => identity),
    removeDeviceIdentityAsync: jest.fn(async () => undefined),
    startInitiatorAsync: jest.fn(async () => HANDSHAKE),
    handshakeWriteAsync: jest.fn(async () => 'AQID'),
    handshakeReadAsync: jest.fn(async (_handle, record) => {
      expect(record).toBe('BAUG');
      return JSON.stringify({
        v: 1,
        host_id: responseHostId,
        device_id: DEVICE_ID,
        session_id: SESSION_ID,
        supported_capabilities: ['agent_events'],
      });
    }),
    finishHandshakeAsync: jest.fn(async () => TRANSPORT),
    invalidateAsync: jest.fn(async () => undefined),
  };
  return port;
}

function socketFactory(socket: FakeSocket) {
  return jest.fn((url: string) => {
    expect(url).toBe('wss://relay.example.test/v1/connect/AAAAAAAAAAAAAAAAAAAAAA');
    queueMicrotask(() => socket.onopen?.());
    return socket;
  });
}

describe('mobile pairing coordinator', () => {
  test('pairs through binary Noise records, persists public metadata, and closes transport state', async () => {
    const values = new MemoryValues();
    const hostStore = new HostStore(values);
    const native = crypto();
    const socket = new FakeSocket();
    const host = await pairHost({
      qr: qr(),
      deviceLabel: '  My phone  ',
      crypto: native,
      hostStore,
      socketFactory: socketFactory(socket),
      nowUnixMs: NOW,
    });

    expect(new Uint8Array(socket.sent ?? new ArrayBuffer(0))).toEqual(Uint8Array.from([1, 2, 3]));
    expect(socket.closed).toBe(true);
    expect(host).toMatchObject({
      hostId: HOST_ID,
      deviceId: DEVICE_ID,
      sessionId: SESSION_ID,
      keyReference: KEY_REFERENCE,
      label: 'My phone',
    });
    expect(await hostStore.list()).toEqual([host]);
    expect(native.invalidateAsync).toHaveBeenCalledWith(TRANSPORT);
    expect(native.removeDeviceIdentityAsync).not.toHaveBeenCalled();
  });

  test('wrong host identity removes the new key and invalidates the pending handshake', async () => {
    const native = crypto(DEVICE_ID);
    await expect(
      pairHost({
        qr: qr(),
        deviceLabel: 'Phone',
        crypto: native,
        hostStore: new HostStore(new MemoryValues()),
        socketFactory: socketFactory(new FakeSocket()),
        nowUnixMs: NOW,
      }),
    ).rejects.toThrow('host identity changed');
    expect(native.invalidateAsync).toHaveBeenCalledWith(HANDSHAKE);
    expect(native.removeDeviceIdentityAsync).toHaveBeenCalledWith(KEY_REFERENCE);
  });

  test('rejects non-binary relay data and rolls back local key authority', async () => {
    const native = crypto();
    const socket = new FakeSocket();
    socket.response = 'not binary';
    await expect(
      pairHost({
        qr: qr(),
        deviceLabel: 'Phone',
        crypto: native,
        hostStore: new HostStore(new MemoryValues()),
        socketFactory: socketFactory(socket),
        nowUnixMs: NOW,
      }),
    ).rejects.toBeInstanceOf(PairingClientError);
    expect(native.removeDeviceIdentityAsync).toHaveBeenCalledWith(KEY_REFERENCE);
  });

  test('rejects expired QR and oversized UTF-8 label before native key generation', async () => {
    const native = crypto();
    await expect(
      pairHost({
        qr: qr(),
        deviceLabel: 'é'.repeat(65),
        crypto: native,
        hostStore: new HostStore(new MemoryValues()),
        nowUnixMs: NOW,
      }),
    ).rejects.toThrow('label is invalid');
    await expect(
      pairHost({
        qr: qr(),
        deviceLabel: 'Phone',
        crypto: native,
        hostStore: new HostStore(new MemoryValues()),
        nowUnixMs: NOW + 300_000,
      }),
    ).rejects.toThrow('expired');
    expect(native.createDeviceIdentityAsync).not.toHaveBeenCalled();
  });
});
