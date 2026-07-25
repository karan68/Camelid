const CHUNK_MAGIC = Uint8Array.from([0x43, 0x4d, 0x52, 0x31]);
const CHUNK_VERSION = 1;
export const CHUNK_HEADER_BYTES = 64;
export const MAX_NOISE_RECORD_BYTES = 65_535;
export const NOISE_TAG_BYTES = 16;
export const MAX_TRANSPORT_PLAINTEXT_BYTES = MAX_NOISE_RECORD_BYTES - NOISE_TAG_BYTES;
export const MAX_CHUNK_DATA_BYTES = MAX_TRANSPORT_PLAINTEXT_BYTES - CHUNK_HEADER_BYTES;
export const MAX_MESSAGE_CHUNKS = 18;
export const MAX_INNER_MESSAGE_BYTES = 1_114_112;

export type Sha256 = (value: Uint8Array) => Promise<Uint8Array>;

export class ChunkError extends Error {
  constructor(message = 'Invalid authenticated transport chunk.') {
    super(message);
    this.name = 'ChunkError';
  }
}

export async function encodeChunks(
  messageId: string,
  message: Uint8Array,
  sha256: Sha256,
): Promise<readonly Uint8Array[]> {
  if (message.byteLength < 1 || message.byteLength > MAX_INNER_MESSAGE_BYTES) {
    throw new ChunkError('Inner message size is invalid.');
  }
  const id = uuidBytes(messageId);
  const count = Math.ceil(message.byteLength / MAX_CHUNK_DATA_BYTES);
  if (count < 1 || count > MAX_MESSAGE_CHUNKS) throw new ChunkError();
  const digest = await checkedDigest(message, sha256);
  const chunks: Uint8Array[] = [];
  for (let index = 0; index < count; index += 1) {
    const start = index * MAX_CHUNK_DATA_BYTES;
    const data = message.subarray(start, Math.min(start + MAX_CHUNK_DATA_BYTES, message.byteLength));
    const chunk = new Uint8Array(CHUNK_HEADER_BYTES + data.byteLength);
    chunk.set(CHUNK_MAGIC, 0);
    chunk[4] = CHUNK_VERSION;
    chunk.set(id, 8);
    const view = new DataView(chunk.buffer);
    view.setUint16(24, index, false);
    view.setUint16(26, count, false);
    view.setUint32(28, message.byteLength, false);
    chunk.set(digest, 32);
    chunk.set(data, CHUNK_HEADER_BYTES);
    chunks.push(chunk);
  }
  return chunks;
}

export class ChunkReassembler {
  private state: ReassemblyState | null = null;

  async push(frame: Uint8Array, sha256: Sha256): Promise<Uint8Array | null> {
    try {
      const decoded = decodeChunk(frame);
      if (this.state === null) {
        if (decoded.index !== 0) throw new ChunkError();
        this.state = {
          messageId: decoded.messageId,
          count: decoded.count,
          total: decoded.total,
          digest: decoded.digest,
          nextIndex: 0,
          received: 0,
          parts: [],
        };
      }
      const state = this.state;
      if (
        state === null ||
        decoded.messageId !== state.messageId ||
        decoded.count !== state.count ||
        decoded.total !== state.total ||
        decoded.index !== state.nextIndex ||
        !equalBytes(decoded.digest, state.digest) ||
        state.received + decoded.data.byteLength > state.total
      ) {
        throw new ChunkError();
      }
      state.parts.push(decoded.data.slice());
      state.received += decoded.data.byteLength;
      state.nextIndex += 1;
      if (state.nextIndex < state.count) return null;
      if (state.received !== state.total) throw new ChunkError();
      const message = concatenate(state.parts, state.total);
      const actual = await checkedDigest(message, sha256);
      if (!equalBytes(actual, state.digest)) throw new ChunkError();
      this.state = null;
      return message;
    } catch (error) {
      this.state = null;
      throw error instanceof ChunkError ? error : new ChunkError();
    }
  }

  reset(): void {
    this.state = null;
  }
}

interface ReassemblyState {
  messageId: string;
  count: number;
  total: number;
  digest: Uint8Array;
  nextIndex: number;
  received: number;
  parts: Uint8Array[];
}

interface DecodedChunk {
  messageId: string;
  index: number;
  count: number;
  total: number;
  digest: Uint8Array;
  data: Uint8Array;
}

function decodeChunk(frame: Uint8Array): DecodedChunk {
  if (
    frame.byteLength < CHUNK_HEADER_BYTES ||
    frame.byteLength > MAX_TRANSPORT_PLAINTEXT_BYTES ||
    !equalBytes(frame.subarray(0, 4), CHUNK_MAGIC) ||
    frame[4] !== CHUNK_VERSION ||
    frame[5] !== 0 ||
    frame[6] !== 0 ||
    frame[7] !== 0
  ) {
    throw new ChunkError();
  }
  const view = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
  const index = view.getUint16(24, false);
  const count = view.getUint16(26, false);
  const total = view.getUint32(28, false);
  if (
    count < 1 ||
    count > MAX_MESSAGE_CHUNKS ||
    index >= count ||
    total < 1 ||
    total > MAX_INNER_MESSAGE_BYTES ||
    count !== Math.ceil(total / MAX_CHUNK_DATA_BYTES) ||
    frame.byteLength - CHUNK_HEADER_BYTES < 1
  ) {
    throw new ChunkError();
  }
  return {
    messageId: bytesUuid(frame.subarray(8, 24)),
    index,
    count,
    total,
    digest: frame.slice(32, 64),
    data: frame.subarray(CHUNK_HEADER_BYTES),
  };
}

function uuidBytes(value: string): Uint8Array {
  const hex = value.replaceAll('-', '');
  if (!/^[0-9a-fA-F]{32}$/.test(hex)) throw new ChunkError('Message ID is invalid.');
  const bytes = new Uint8Array(16);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  if (bytesUuid(bytes).toLowerCase() !== value.toLowerCase()) {
    throw new ChunkError('Message ID is not canonical UUID text.');
  }
  return bytes;
}

function bytesUuid(bytes: Uint8Array): string {
  if (bytes.byteLength !== 16) throw new ChunkError();
  const hex = [...bytes].map((byte) => byte.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

async function checkedDigest(value: Uint8Array, sha256: Sha256): Promise<Uint8Array> {
  const digest = await sha256(value);
  if (digest.byteLength !== 32) throw new ChunkError('SHA-256 provider returned an invalid digest.');
  return digest;
}

function concatenate(parts: readonly Uint8Array[], total: number): Uint8Array {
  const value = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    value.set(part, offset);
    offset += part.byteLength;
  }
  return value;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= left[index] ^ right[index];
  }
  return difference === 0;
}
