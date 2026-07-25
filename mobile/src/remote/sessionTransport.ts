import type {
  Base64UrlData,
  HandshakeHandle,
  KeyReference,
  TransportHandle,
} from '../../modules/camelid-remote-crypto';

import { base64UrlToBytes, bytesToBase64Url, toArrayBuffer } from './binary';
import { ChunkReassembler, encodeChunks, Sha256 } from './chunks';
import { replayRequest, sessionCatalogRequest } from './commands';
import type { StoredHost } from './hostStore';
import { relaySocketUrl } from './pairingQr';
import { parseRemoteEnvelope, RemoteEnvelope } from './protocol';
import { decodeUtf8, encodeUtf8 } from './utf8';

const CONNECT_TIMEOUT_MS = 20_000;

export interface SessionCryptoPort {
  startInitiatorAsync(keyReference: KeyReference, pinnedHostPublic: string): Promise<HandshakeHandle>;
  handshakeWriteAsync(handle: HandshakeHandle, payload: string): Promise<string>;
  handshakeReadAsync(handle: HandshakeHandle, record: string): Promise<string>;
  finishHandshakeAsync(handle: HandshakeHandle): Promise<TransportHandle>;
  sealAsync(handle: TransportHandle, plaintext: Base64UrlData): Promise<Base64UrlData>;
  openAsync(handle: TransportHandle, ciphertext: Base64UrlData): Promise<Base64UrlData>;
  invalidateAsync(handle: HandshakeHandle | TransportHandle): Promise<void>;
}

export interface SessionSocket {
  binaryType: string;
  onopen: (() => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onerror: (() => void) | null;
  onclose: (() => void) | null;
  send(data: ArrayBuffer): void;
  close(): void;
}

export interface SessionTransport {
  send(messageId: string, canonicalJson: string): Promise<void>;
  close(): Promise<void>;
}

export class SessionTransportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SessionTransportError';
  }
}

export async function connectSession(options: {
  host: StoredHost;
  crypto: SessionCryptoPort;
  sha256: Sha256;
  messageId: () => string;
  nowUnixMs?: () => number;
  socketFactory?: (url: string) => SessionSocket;
  onMessage: (message: RemoteEnvelope) => void | Promise<void>;
  onClose?: (error: Error | null) => void;
  replayAfterSequence?: number;
}): Promise<SessionTransport> {
  const socket = (options.socketFactory ?? defaultSocketFactory)(
    relaySocketUrl(options.host.relayUrl, options.host.routeId),
  );
  socket.binaryType = 'arraybuffer';
  let handshake: HandshakeHandle | null = null;
  let transport: TransportHandle | null = null;
  let closed = false;
  let inbound = Promise.resolve();
  const reassembler = new ChunkReassembler();

  const close = async (error: Error | null) => {
    if (closed) return;
    closed = true;
    socket.onopen = null;
    socket.onmessage = null;
    socket.onerror = null;
    socket.onclose = null;
    socket.close();
    reassembler.reset();
    if (handshake !== null) await ignoreFailure(options.crypto.invalidateAsync(handshake));
    if (transport !== null) await ignoreFailure(options.crypto.invalidateAsync(transport));
    handshake = null;
    transport = null;
    options.onClose?.(error);
  };

  const ready = new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      const error = new SessionTransportError('Remote connection timed out.');
      void close(error).then(() => reject(error));
    }, CONNECT_TIMEOUT_MS);
    const fail = (error: Error) => {
      clearTimeout(timeout);
      void close(error).then(() => reject(error));
    };

    socket.onopen = () => {
      void (async () => {
        handshake = await options.crypto.startInitiatorAsync(
          options.host.keyReference as KeyReference,
          options.host.hostNoisePublic,
        );
        const first = await options.crypto.handshakeWriteAsync(handshake, '');
        socket.send(toArrayBuffer(base64UrlToBytes(first)));
      })().catch((error) => fail(asError(error)));
    };
    socket.onerror = () => fail(new SessionTransportError('Remote transport failed.'));
    socket.onclose = () => fail(new SessionTransportError('Remote transport closed.'));
    socket.onmessage = (event) => {
      const data = event.data;
      if (!(data instanceof ArrayBuffer) || handshake === null) {
        fail(new SessionTransportError('Remote handshake frame is invalid.'));
        return;
      }
      void (async () => {
        await options.crypto.handshakeReadAsync(handshake as HandshakeHandle, bytesToBase64Url(new Uint8Array(data)));
        transport = await options.crypto.finishHandshakeAsync(handshake as HandshakeHandle);
        handshake = null;
        clearTimeout(timeout);
        installTransportHandler();
        resolve();
      })().catch((error) => fail(asError(error)));
    };
  });

  const send = async (messageId: string, canonicalJson: string) => {
    if (closed || transport === null) throw new SessionTransportError('Remote transport is not connected.');
    const chunks = await encodeChunks(messageId, encodeUtf8(canonicalJson), options.sha256);
    for (const chunk of chunks) {
      const record = await options.crypto.sealAsync(transport, bytesToBase64Url(chunk));
      socket.send(toArrayBuffer(base64UrlToBytes(record)));
    }
  };

  const installTransportHandler = () => {
    socket.onmessage = (event) => {
      inbound = inbound.then(async () => {
        const data = event.data;
        if (!(data instanceof ArrayBuffer) || transport === null) {
          throw new SessionTransportError('Remote transport frame is invalid.');
        }
        const opened = await options.crypto.openAsync(
          transport,
          bytesToBase64Url(new Uint8Array(data)),
        );
        const complete = await reassembler.push(base64UrlToBytes(opened), options.sha256);
        if (complete !== null) await options.onMessage(parseRemoteEnvelope(decodeUtf8(complete)));
      }).catch((error) => close(asError(error)));
    };
    socket.onerror = () => void close(new SessionTransportError('Remote transport failed.'));
    socket.onclose = () => void close(new SessionTransportError('Remote transport closed.'));
  };

  await ready;
  const replayId = options.messageId();
  await send(
    replayId,
    replayRequest(
      {
        messageId: replayId,
        hostId: options.host.hostId,
        deviceId: options.host.deviceId,
        sessionId: options.host.sessionId,
        sentAtUnixMs: (options.nowUnixMs ?? Date.now)(),
      },
      options.replayAfterSequence ?? options.host.lastAppliedSequence,
    ),
  );
  if (options.host.supportedCapabilities?.includes('session_catalog_v1')) {
    const catalogId = options.messageId();
    await send(
      catalogId,
      sessionCatalogRequest({
        messageId: catalogId,
        hostId: options.host.hostId,
        deviceId: options.host.deviceId,
        sentAtUnixMs: (options.nowUnixMs ?? Date.now)(),
      }),
    );
  }

  return {
    send,
    close: () => close(null),
  };
}

function defaultSocketFactory(url: string): SessionSocket {
  return new WebSocket(url) as unknown as SessionSocket;
}

function asError(value: unknown): Error {
  return value instanceof Error ? value : new SessionTransportError('Remote transport failed.');
}

async function ignoreFailure(operation: Promise<unknown>): Promise<void> {
  try { await operation; } catch { /* primary failure remains authoritative */ }
}
