import type {
  DeviceIdentity,
  HandshakeHandle,
  KeyReference,
  TransportHandle,
} from '../../modules/camelid-remote-crypto';

import { canonicalJson } from './canonicalJson';
import { base64UrlToBytes, bytesToBase64Url, toArrayBuffer } from './binary';
import type { HostStore, StoredHost } from './hostStore';
import type { PairingQr } from './pairingQr';
import { pairingSocketUrl } from './pairingQr';
import { parsePairResponse } from './protocol';

const PAIRING_NETWORK_TIMEOUT_MS = 30_000;
const MAX_DEVICE_LABEL_BYTES = 128;

export interface PairingCryptoPort {
  createDeviceIdentityAsync(hostId: string): Promise<DeviceIdentity>;
  removeDeviceIdentityAsync(keyReference: KeyReference): Promise<void>;
  startInitiatorAsync(keyReference: KeyReference, pinnedHostPublic: string): Promise<HandshakeHandle>;
  handshakeWriteAsync(handle: HandshakeHandle, payload: string): Promise<string>;
  handshakeReadAsync(handle: HandshakeHandle, record: string): Promise<string>;
  finishHandshakeAsync(handle: HandshakeHandle): Promise<TransportHandle>;
  invalidateAsync(handle: HandshakeHandle | TransportHandle): Promise<void>;
}

export interface PairingSocket {
  binaryType: string;
  onopen: (() => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onerror: (() => void) | null;
  onclose: (() => void) | null;
  send(data: ArrayBuffer): void;
  close(): void;
}

export type PairingSocketFactory = (url: string) => PairingSocket;

export class PairingClientError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PairingClientError';
  }
}

export async function pairHost(options: {
  qr: PairingQr;
  deviceLabel: string;
  crypto: PairingCryptoPort;
  hostStore: HostStore;
  socketFactory?: PairingSocketFactory;
  nowUnixMs?: number;
}): Promise<StoredHost> {
  const label = options.deviceLabel.trim();
  if (label.length === 0 || utf8Length(label) > MAX_DEVICE_LABEL_BYTES) {
    throw new PairingClientError('Device label is invalid.');
  }
  const nowUnixMs = options.nowUnixMs ?? Date.now();
  const remaining = options.qr.expires_at_unix_ms - nowUnixMs;
  if (remaining <= 0) throw new PairingClientError('Pairing QR has expired.');

  const identity = await options.crypto.createDeviceIdentityAsync(options.qr.host_id);
  let handshake: HandshakeHandle | null = null;
  let transport: TransportHandle | null = null;
  let stored = false;
  try {
    handshake = await options.crypto.startInitiatorAsync(
      identity.keyReference,
      options.qr.host_noise_public,
    );
    const request = canonicalJson({
      pairing_secret: options.qr.pairing_secret,
      device_label: label,
      app_protocol_version: 1,
      supported_capabilities: ['agent_events'],
    });
    const firstRecord = await options.crypto.handshakeWriteAsync(handshake, request);
    const secondRecord = await exchangePairingRecord(
      pairingSocketUrl(options.qr),
      firstRecord,
      Math.min(remaining, PAIRING_NETWORK_TIMEOUT_MS),
      options.socketFactory ?? defaultSocketFactory,
    );
    const responseJson = await options.crypto.handshakeReadAsync(handshake, secondRecord);
    const response = parsePairResponse(responseJson, options.qr.host_id);
    if (!response.supported_capabilities.includes('agent_events')) {
      throw new PairingClientError('Host does not support remote agent events.');
    }
    transport = await options.crypto.finishHandshakeAsync(handshake);
    handshake = null;
    const host: StoredHost = {
      hostId: response.host_id,
      label,
      relayUrl: options.qr.relay_url,
      routeId: options.qr.route_id,
      hostNoisePublic: options.qr.host_noise_public,
      keyReference: identity.keyReference,
      deviceId: response.device_id,
      sessionId: response.session_id,
      lastAppliedSequence: 0,
      sessionCursors: { [response.session_id]: 0 },
      supportedCapabilities: response.supported_capabilities,
    };
    await options.hostStore.save(host);
    stored = true;
    return host;
  } finally {
    if (handshake !== null) await ignoreFailure(options.crypto.invalidateAsync(handshake));
    if (transport !== null) await ignoreFailure(options.crypto.invalidateAsync(transport));
    if (!stored) await ignoreFailure(options.crypto.removeDeviceIdentityAsync(identity.keyReference));
  }
}

function exchangePairingRecord(
  url: string,
  firstRecord: string,
  timeoutMs: number,
  factory: PairingSocketFactory,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const socket = factory(url);
    socket.binaryType = 'arraybuffer';
    let settled = false;
    let finish: (complete: () => void) => void;
    const timeout = setTimeout(
      () => finish(() => reject(new PairingClientError('Pairing timed out.'))),
      timeoutMs,
    );

    finish = (complete) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      socket.onopen = null;
      socket.onmessage = null;
      socket.onerror = null;
      socket.onclose = null;
      socket.close();
      complete();
    };

    socket.onopen = () => {
      try {
        socket.send(toArrayBuffer(base64UrlToBytes(firstRecord)));
      } catch {
        finish(() => reject(new PairingClientError('Pairing transport failed.')));
      }
    };
    socket.onmessage = (event) => {
      const data = event.data;
      if (!(data instanceof ArrayBuffer)) {
        finish(() => reject(new PairingClientError('Pairing relay returned a non-binary frame.')));
        return;
      }
      finish(() => resolve(bytesToBase64Url(new Uint8Array(data))));
    };
    socket.onerror = () => finish(() => reject(new PairingClientError('Pairing transport failed.')));
    socket.onclose = () => finish(() => reject(new PairingClientError('Pairing transport closed.')));
  });
}

function defaultSocketFactory(url: string): PairingSocket {
  return new WebSocket(url) as unknown as PairingSocket;
}

async function ignoreFailure(operation: Promise<unknown>): Promise<void> {
  try {
    await operation;
  } catch {
    // Cleanup is best effort after the primary error has already failed closed.
  }
}

function utf8Length(value: string): number {
  let bytes = 0;
  for (const character of value) {
    const point = character.codePointAt(0) ?? 0;
    bytes += point <= 0x7f ? 1 : point <= 0x7ff ? 2 : point <= 0xffff ? 3 : 4;
  }
  return bytes;
}
