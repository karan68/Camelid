import { createHash } from 'node:crypto';

import type { HandshakeHandle, TransportHandle } from '../../modules/camelid-remote-crypto';

import { bytesToBase64Url } from './binary';
import { encodeChunks } from './chunks';
import type { StoredHost } from './hostStore';
import { SessionCryptoPort, SessionSocket, connectSession } from './sessionTransport';
import { encodeUtf8 } from './utf8';

const HANDSHAKE = '60000000-0000-4000-8000-000000000006' as HandshakeHandle;
const TRANSPORT = '70000000-0000-4000-8000-000000000007' as TransportHandle;
const REPLAY_ID = '80000000-0000-4000-8000-000000000008';
const sha256 = async (value: Uint8Array) =>
  new Uint8Array(createHash('sha256').update(value).digest());

const host: StoredHost = {
  hostId: '20000000-0000-4000-8000-000000000002',
  label: 'Workstation',
  relayUrl: 'wss://relay.example.test/v1/connect',
  routeId: 'AAAAAAAAAAAAAAAAAAAAAA',
  hostNoisePublic: 'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
  keyReference: '30000000-0000-4000-8000-000000000003',
  deviceId: '40000000-0000-4000-8000-000000000004',
  sessionId: '50000000-0000-4000-8000-000000000005',
  lastAppliedSequence: 42,
  supportedCapabilities: ['agent_events', 'session_catalog_v1'],
};

class FakeSocket implements SessionSocket {
  binaryType = '';
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  closed = false;
  sent: ArrayBuffer[] = [];
  send(data: ArrayBuffer) { this.sent.push(data); }
  close() { this.closed = true; }
  emit(data: unknown) { this.onmessage?.({ data }); }
}

function crypto(): SessionCryptoPort {
  return {
    startInitiatorAsync: jest.fn(async (reference, hostPublic) => {
      expect(reference).toBe(host.keyReference);
      expect(hostPublic).toBe(host.hostNoisePublic);
      return HANDSHAKE;
    }),
    handshakeWriteAsync: jest.fn(async () => bytesToBase64Url(Uint8Array.of(1, 2, 3))),
    handshakeReadAsync: jest.fn(async () => ''),
    finishHandshakeAsync: jest.fn(async () => TRANSPORT),
    sealAsync: jest.fn(async (_handle, plaintext) => plaintext),
    openAsync: jest.fn(async (_handle, ciphertext) => ciphertext),
    invalidateAsync: jest.fn(async () => undefined),
  };
}

async function connectFixture() {
  const socket = new FakeSocket();
  const native = crypto();
  const messages: unknown[] = [];
  const connected = connectSession({
    host,
    crypto: native,
    sha256,
    messageId: () => REPLAY_ID,
    nowUnixMs: () => 100,
    socketFactory: (url) => {
      expect(url).toBe('wss://relay.example.test/v1/connect/AAAAAAAAAAAAAAAAAAAAAA');
      queueMicrotask(() => socket.onopen?.());
      return socket;
    },
    onMessage: (message) => { messages.push(message); },
  });
  await eventually(() => socket.sent.length === 1);
  expect(new Uint8Array(socket.sent[0])).toEqual(Uint8Array.of(1, 2, 3));
  socket.emit(Uint8Array.of(4, 5).buffer);
  const transport = await connected;
  await eventually(() => socket.sent.length >= 3);
  return { socket, native, messages, transport };
}

describe('post-pair session transport', () => {
  test('uses fresh IK and sends replay plus negotiated session catalog request', async () => {
    const { socket, native, transport } = await connectFixture();
    expect(native.finishHandshakeAsync).toHaveBeenCalledWith(HANDSHAKE);
    expect(socket.sent.length).toBeGreaterThanOrEqual(2);
    const replayChunk = new Uint8Array(socket.sent[1]);
    expect(replayChunk.slice(0, 4)).toEqual(Uint8Array.from([0x43, 0x4d, 0x52, 0x31]));
    const replayText = new TextDecoder().decode(replayChunk.slice(64));
    expect(JSON.parse(replayText).payload).toEqual({ after_sequence: 42, limit: 256 });
    const catalogChunk = new Uint8Array(socket.sent[2]);
    const catalogText = new TextDecoder().decode(catalogChunk.slice(64));
    expect(JSON.parse(catalogText)).toMatchObject({
      kind: 'session_catalog_request',
      session_id: null,
      payload: { cursor: null, limit: 64, revision: null },
    });
    await transport.close();
    expect(native.invalidateAsync).toHaveBeenCalledWith(TRANSPORT);
    expect(socket.closed).toBe(true);
  });

  test('opens authenticated chunks and emits only complete strict envelopes', async () => {
    const { socket, messages, transport } = await connectFixture();
    const json = JSON.stringify({
      protocol: 'camelid.remote/v1',
      message_id: REPLAY_ID,
      kind: 'replay_end',
      host_id: host.hostId,
      device_id: host.deviceId,
      session_id: host.sessionId,
      sent_at_unix_ms: 101,
      payload: { last_sequence: 42, has_more: false, session_state: 'idle' },
    });
    const chunks = await encodeChunks(REPLAY_ID, encodeUtf8(json), sha256);
    for (const chunk of chunks) socket.emit(chunk.slice().buffer);
    await eventually(() => messages.length === 1);
    expect(messages).toHaveLength(1);
    expect(messages[0]).toMatchObject({ kind: 'replay_end', session_id: host.sessionId });
    await transport.close();
  });

  test('non-binary transport frame closes terminally', async () => {
    const onClose = jest.fn();
    const socket = new FakeSocket();
    const native = crypto();
    const connected = connectSession({
      host,
      crypto: native,
      sha256,
      messageId: () => REPLAY_ID,
      socketFactory: () => {
        queueMicrotask(() => socket.onopen?.());
        return socket;
      },
      onMessage: () => undefined,
      onClose,
    });
    await eventually(() => socket.sent.length === 1);
    socket.emit(Uint8Array.of(4).buffer);
    await connected;
    await eventually(() => socket.sent.length >= 2);
    socket.emit('text frame');
    await eventually(() => onClose.mock.calls.length === 1);
    expect(onClose).toHaveBeenCalledWith(expect.any(Error));
    expect(native.invalidateAsync).toHaveBeenCalledWith(TRANSPORT);
  });
});

async function eventually(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await Promise.resolve();
  }
  throw new Error('Expected asynchronous state was not reached.');
}
