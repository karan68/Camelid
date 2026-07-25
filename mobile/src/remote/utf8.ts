export function encodeUtf8(value: string): Uint8Array {
  const bytes: number[] = [];
  for (let index = 0; index < value.length; index += 1) {
    const first = value.charCodeAt(index);
    let point = first;
    if (first >= 0xd800 && first <= 0xdbff) {
      const second = value.charCodeAt(index + 1);
      if (second < 0xdc00 || second > 0xdfff) throw new Error('Invalid UTF-16 input.');
      point = 0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00);
      index += 1;
    } else if (first >= 0xdc00 && first <= 0xdfff) {
      throw new Error('Invalid UTF-16 input.');
    }
    if (point <= 0x7f) bytes.push(point);
    else if (point <= 0x7ff) bytes.push(0xc0 | (point >> 6), 0x80 | (point & 0x3f));
    else if (point <= 0xffff) {
      bytes.push(0xe0 | (point >> 12), 0x80 | ((point >> 6) & 0x3f), 0x80 | (point & 0x3f));
    } else {
      bytes.push(
        0xf0 | (point >> 18),
        0x80 | ((point >> 12) & 0x3f),
        0x80 | ((point >> 6) & 0x3f),
        0x80 | (point & 0x3f),
      );
    }
  }
  return Uint8Array.from(bytes);
}

export function decodeUtf8(bytes: Uint8Array): string {
  let value = '';
  for (let index = 0; index < bytes.length; ) {
    const first = bytes[index];
    let point: number;
    let count: number;
    if (first <= 0x7f) {
      point = first;
      count = 1;
    } else if (first >= 0xc2 && first <= 0xdf) {
      point = first & 0x1f;
      count = 2;
    } else if (first >= 0xe0 && first <= 0xef) {
      point = first & 0x0f;
      count = 3;
    } else if (first >= 0xf0 && first <= 0xf4) {
      point = first & 0x07;
      count = 4;
    } else throw new Error('Invalid UTF-8 input.');
    if (index + count > bytes.length) throw new Error('Truncated UTF-8 input.');
    for (let offset = 1; offset < count; offset += 1) {
      const next = bytes[index + offset];
      if ((next & 0xc0) !== 0x80) throw new Error('Invalid UTF-8 continuation.');
      point = (point << 6) | (next & 0x3f);
    }
    if (
      (count === 3 && first === 0xe0 && bytes[index + 1] < 0xa0) ||
      (count === 3 && first === 0xed && bytes[index + 1] >= 0xa0) ||
      (count === 4 && first === 0xf0 && bytes[index + 1] < 0x90) ||
      (count === 4 && first === 0xf4 && bytes[index + 1] >= 0x90)
    ) throw new Error('Non-canonical UTF-8 input.');
    value += point <= 0xffff
      ? String.fromCharCode(point)
      : String.fromCharCode(0xd800 + ((point - 0x10000) >> 10), 0xdc00 + ((point - 0x10000) & 0x3ff));
    index += count;
  }
  return value;
}
