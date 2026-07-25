import * as Crypto from 'expo-crypto';

import type { Sha256 } from './chunks';

export const nativeSha256: Sha256 = async (value) => {
  const input = new Uint8Array(new ArrayBuffer(value.byteLength));
  input.set(value);
  const digest = await Crypto.digest(Crypto.CryptoDigestAlgorithm.SHA256, input);
  return new Uint8Array(digest);
};

export function randomUuid(): string {
  return Crypto.randomUUID();
}
