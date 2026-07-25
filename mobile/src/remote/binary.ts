import type { Base64UrlData } from '../../modules/camelid-remote-crypto';

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';

export function base64UrlToBytes(value: string): Uint8Array {
  if (value.length === 0) return new Uint8Array();
  if (!/^[A-Za-z0-9_-]+$/.test(value)) throw new Error('Invalid base64url data.');
  let accumulator = 0;
  let bits = 0;
  const output: number[] = [];
  for (const character of value) {
    const index = ALPHABET.indexOf(character);
    accumulator = (accumulator << 6) | index;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      output.push((accumulator >> bits) & 0xff);
    }
  }
  if (bits > 0 && (accumulator & ((1 << bits) - 1)) !== 0) {
    throw new Error('Non-canonical base64url data.');
  }
  return Uint8Array.from(output);
}

export function bytesToBase64Url(bytes: Uint8Array): Base64UrlData {
  let accumulator = 0;
  let bits = 0;
  let output = '';
  for (const byte of bytes) {
    accumulator = (accumulator << 8) | byte;
    bits += 8;
    while (bits >= 6) {
      bits -= 6;
      output += ALPHABET[(accumulator >> bits) & 0x3f];
    }
  }
  if (bits > 0) output += ALPHABET[(accumulator << (6 - bits)) & 0x3f];
  return output as Base64UrlData;
}

export function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}
