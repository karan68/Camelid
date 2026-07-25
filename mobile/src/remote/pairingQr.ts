const MAX_PAIRING_QR_BYTES = 2560;
const BASE64URL = /^[A-Za-z0-9_-]+$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export interface PairingQr {
  v: 1;
  relay_url: string;
  route_id: string;
  host_id: string;
  host_noise_public: string;
  pairing_secret: string;
  expires_at_unix_ms: number;
}

export class PairingQrError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PairingQrError';
  }
}

export function parsePairingQr(input: string, nowUnixMs: number): PairingQr {
  if (utf8Length(input) > MAX_PAIRING_QR_BYTES) throw new PairingQrError('Pairing QR is too large.');
  if (!Number.isSafeInteger(nowUnixMs) || nowUnixMs < 0) {
    throw new PairingQrError('Device time is invalid.');
  }

  let decoded: unknown;
  try {
    decoded = JSON.parse(input);
  } catch {
    throw new PairingQrError('Pairing QR is not valid JSON.');
  }
  if (!isRecord(decoded)) throw new PairingQrError('Pairing QR must be an object.');

  const allowed = new Set([
    'v',
    'relay_url',
    'route_id',
    'host_id',
    'host_noise_public',
    'pairing_secret',
    'expires_at_unix_ms',
  ]);
  if (Object.keys(decoded).some((key) => !allowed.has(key))) {
    throw new PairingQrError('Pairing QR contains unsupported fields.');
  }

  if (decoded.v !== 1) throw new PairingQrError('Pairing QR version is not supported.');
  const relayUrl = requiredString(decoded, 'relay_url');
  validateRelayUrl(relayUrl);
  const routeId = exactBase64Url(decoded, 'route_id', 22);
  const hostId = requiredString(decoded, 'host_id');
  if (!UUID.test(hostId)) throw new PairingQrError('Host identity is invalid.');
  const hostNoisePublic = exactBase64Url(decoded, 'host_noise_public', 43);
  const pairingSecret = exactBase64Url(decoded, 'pairing_secret', 22);
  const expiresAt = decoded.expires_at_unix_ms;
  if (!Number.isSafeInteger(expiresAt) || (expiresAt as number) <= nowUnixMs) {
    throw new PairingQrError('Pairing QR has expired.');
  }

  return {
    v: 1,
    relay_url: relayUrl,
    route_id: routeId,
    host_id: hostId,
    host_noise_public: hostNoisePublic,
    pairing_secret: pairingSecret,
    expires_at_unix_ms: expiresAt as number,
  };
}

export function pairingSocketUrl(qr: PairingQr): string {
  return relaySocketUrl(qr.relay_url, qr.route_id);
}

export function relaySocketUrl(relayUrl: string, routeId: string): string {
  const parsed = new URL(relayUrl);
  parsed.pathname = `${parsed.pathname.replace(/\/$/, '')}/${routeId}`;
  return parsed.toString();
}

function validateRelayUrl(value: string): void {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new PairingQrError('Relay URL is invalid.');
  }
  if (
    parsed.protocol !== 'wss:' ||
    parsed.hostname.length === 0 ||
    parsed.username.length !== 0 ||
    parsed.password.length !== 0 ||
    parsed.hash.length !== 0
  ) {
    throw new PairingQrError('Relay URL must be a credential-free wss URL.');
  }
}

function exactBase64Url(
  value: Readonly<Record<string, unknown>>,
  key: string,
  length: number,
): string {
  const token = requiredString(value, key);
  if (token.length !== length || !BASE64URL.test(token)) {
    throw new PairingQrError(`${key} is not valid base64url.`);
  }
  return token;
}

function requiredString(value: Readonly<Record<string, unknown>>, key: string): string {
  const field = value[key];
  if (typeof field !== 'string' || field.length === 0) {
    throw new PairingQrError(`${key} is missing.`);
  }
  return field;
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function utf8Length(value: string): number {
  let bytes = 0;
  for (const character of value) {
    const point = character.codePointAt(0) ?? 0;
    bytes += point <= 0x7f ? 1 : point <= 0x7ff ? 2 : point <= 0xffff ? 3 : 4;
  }
  return bytes;
}
