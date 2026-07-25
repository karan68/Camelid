import { createHash } from 'node:crypto';

import {
  ChunkError,
  ChunkReassembler,
  MAX_INNER_MESSAGE_BYTES,
  MAX_TRANSPORT_PLAINTEXT_BYTES,
  encodeChunks,
} from './chunks';

const MESSAGE_ID = '10000000-0000-4000-8000-000000000001';
const OTHER_ID = '20000000-0000-4000-8000-000000000002';
const sha256 = async (value: Uint8Array) =>
  new Uint8Array(createHash('sha256').update(value).digest());

async function reassemble(chunks: readonly Uint8Array[]) {
  const reassembler = new ChunkReassembler();
  let result: Uint8Array | null = null;
  for (const chunk of chunks) result = await reassembler.push(chunk, sha256);
  return result;
}

describe('authenticated transport chunks', () => {
  test('round trips small and maximum inner messages within Noise record bounds', async () => {
    const small = new TextEncoder().encode('{"kind":"replay_request"}');
    const smallChunks = await encodeChunks(MESSAGE_ID, small, sha256);
    expect(smallChunks).toHaveLength(1);
    expect(await reassemble(smallChunks)).toEqual(small);

    const maximum = new Uint8Array(MAX_INNER_MESSAGE_BYTES).fill(0x5a);
    const maximumChunks = await encodeChunks(MESSAGE_ID, maximum, sha256);
    expect(maximumChunks).toHaveLength(18);
    expect(maximumChunks.every((chunk) => chunk.byteLength <= MAX_TRANSPORT_PLAINTEXT_BYTES)).toBe(
      true,
    );
    expect(await reassemble(maximumChunks)).toEqual(maximum);
  });

  test('rejects reordered, duplicate, and cross-message chunks and resets terminally', async () => {
    const message = new Uint8Array(100_000).fill(7);
    const chunks = await encodeChunks(MESSAGE_ID, message, sha256);
    const other = await encodeChunks(OTHER_ID, message, sha256);
    const reassembler = new ChunkReassembler();

    await expect(reassembler.push(chunks[1], sha256)).rejects.toBeInstanceOf(ChunkError);
    expect(await reassembler.push(chunks[0], sha256)).toBeNull();
    await expect(reassembler.push(chunks[0], sha256)).rejects.toBeInstanceOf(ChunkError);
    expect(await reassembler.push(chunks[0], sha256)).toBeNull();
    await expect(reassembler.push(other[1], sha256)).rejects.toBeInstanceOf(ChunkError);
    expect(await reassemble(chunks)).toEqual(message);
  });

  test('rejects header and whole-message tampering', async () => {
    const message = new Uint8Array(70_000).fill(9);
    const chunks = (await encodeChunks(MESSAGE_ID, message, sha256)).map((chunk) => chunk.slice());
    const badHeader = chunks[0].slice();
    badHeader[4] = 2;
    await expect(new ChunkReassembler().push(badHeader, sha256)).rejects.toBeInstanceOf(ChunkError);

    chunks[1][chunks[1].length - 1] ^= 1;
    const reassembler = new ChunkReassembler();
    expect(await reassembler.push(chunks[0], sha256)).toBeNull();
    await expect(reassembler.push(chunks[1], sha256)).rejects.toBeInstanceOf(ChunkError);
  });

  test('rejects empty, oversized, invalid UUID, and invalid digest provider', async () => {
    await expect(encodeChunks(MESSAGE_ID, new Uint8Array(), sha256)).rejects.toThrow('size');
    await expect(
      encodeChunks(MESSAGE_ID, new Uint8Array(MAX_INNER_MESSAGE_BYTES + 1), sha256),
    ).rejects.toThrow('size');
    await expect(encodeChunks('not-a-uuid', Uint8Array.of(1), sha256)).rejects.toThrow('Message ID');
    await expect(
      encodeChunks(MESSAGE_ID, Uint8Array.of(1), async () => new Uint8Array(31)),
    ).rejects.toThrow('invalid digest');
  });
});
