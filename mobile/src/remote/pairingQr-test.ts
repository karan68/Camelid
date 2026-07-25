import { PairingQrError, pairingSocketUrl, parsePairingQr } from './pairingQr';

const NOW = 1_780_000_000_000;

function validQr(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({
    v: 1,
    relay_url: 'wss://relay.example.test/v1/connect',
    route_id: 'AAAAAAAAAAAAAAAAAAAAAA',
    host_id: '20000000-0000-4000-8000-000000000002',
    host_noise_public: 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
    pairing_secret: 'BBBBBBBBBBBBBBBBBBBBBB',
    expires_at_unix_ms: NOW + 300_000,
    ...overrides,
  });
}

describe('pairing QR parser', () => {
  test('accepts the strict v1 fixture while preserving the pinned host key', () => {
    const parsed = parsePairingQr(validQr(), NOW);
    expect(parsed.route_id).toBe('AAAAAAAAAAAAAAAAAAAAAA');
    expect(parsed.host_noise_public).toHaveLength(43);
    expect(parsed.expires_at_unix_ms).toBe(NOW + 300_000);
    expect(pairingSocketUrl(parsed)).toBe(
      'wss://relay.example.test/v1/connect/AAAAAAAAAAAAAAAAAAAAAA',
    );
  });

  test.each([
    ['unknown field', { admin: true }],
    ['unsupported version', { v: 2 }],
    ['non-wss relay', { relay_url: 'https://relay.example.test/v1/connect' }],
    ['relay credentials', { relay_url: 'wss://user:secret@relay.example.test/v1/connect' }],
    ['relay fragment', { relay_url: 'wss://relay.example.test/v1/connect#secret' }],
    ['short route', { route_id: 'short' }],
    ['invalid host identity', { host_id: 'not-a-uuid' }],
    ['short host key', { host_noise_public: 'short' }],
    ['short pairing secret', { pairing_secret: 'short' }],
    ['expired payload', { expires_at_unix_ms: NOW }],
    ['unsafe expiry', { expires_at_unix_ms: Number.MAX_SAFE_INTEGER + 1 }],
  ])('rejects %s', (_label, override) => {
    expect(() => parsePairingQr(validQr(override), NOW)).toThrow(PairingQrError);
  });

  test('rejects arrays, invalid JSON, and overlarge UTF-8 input before use', () => {
    expect(() => parsePairingQr('[]', NOW)).toThrow('must be an object');
    expect(() => parsePairingQr('{', NOW)).toThrow('not valid JSON');
    expect(() => parsePairingQr(`{"padding":"${'é'.repeat(1300)}"}`, NOW)).toThrow('too large');
  });
});
