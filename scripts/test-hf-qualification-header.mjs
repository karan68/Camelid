#!/usr/bin/env node
import assert from 'node:assert/strict'
import { fetchHeaderPrefix, parseContentRange, summarizeInspection } from './hf-qualification-header.mjs'

const lock = {
  download_url: 'https://example.invalid/model.gguf',
  size_bytes: 100,
}

assert.deepEqual(parseContentRange('bytes 0-15/100'), { start: 0, end: 15, total: 100 })
assert.throws(() => parseContentRange('0-15/100'), /invalid or missing Content-Range/)

const calls = []
const ranged = await fetchHeaderPrefix(lock, {
  prefixBytes: 16,
  fetchImpl: async (url, options) => {
    calls.push({ url, options })
    return new Response(Buffer.alloc(16, 7), {
      status: 206,
      headers: { 'content-range': 'bytes 0-15/100' },
    })
  },
})
assert.equal(ranged.bytes.length, 16)
assert.equal(calls[0].options.headers.Range, 'bytes=0-15')

await assert.rejects(
  fetchHeaderPrefix(lock, {
    prefixBytes: 16,
    fetchImpl: async () => new Response(Buffer.alloc(100), { status: 200 }),
  }),
  /refusing a possible full-model download/,
)

await assert.rejects(
  fetchHeaderPrefix(lock, {
    prefixBytes: 16,
    fetchImpl: async () => new Response(Buffer.alloc(16), {
      status: 206,
      headers: { 'content-range': 'bytes 0-15/101' },
    }),
  }),
  /range identity mismatch/,
)

await assert.rejects(
  fetchHeaderPrefix(lock, {
    prefixBytes: 16,
    fetchImpl: async () => new Response(Buffer.alloc(17), {
      status: 206,
      headers: { 'content-range': 'bytes 0-15/100' },
    }),
  }),
  /exceeded the 16-byte request budget/,
)

const summary = summarizeInspection({
  version: 3,
  tensor_count: 2,
  metadata_count: 3,
  alignment: 32,
  data_start_offset: 4096,
  metadata: {
    'general.architecture': { String: 'qwen3moe' },
    'tokenizer.ggml.tokens': { Array: ['x'.repeat(70 * 1024)] },
  },
  tensors: [
    { name: 'a', dimensions: [2, 3], tensor_type: 'Q8_0', relative_offset: 0, absolute_offset: 4096, n_bytes: 6 },
    { name: 'b', dimensions: [1], tensor_type: 'F32', relative_offset: 32, absolute_offset: 4128, n_bytes: 4 },
  ],
})
assert.deepEqual(summary.metadata['general.architecture'], { String: 'qwen3moe' })
assert.equal(summary.omitted_metadata[0].key, 'tokenizer.ggml.tokens')
assert.equal(summary.tensor_inventory.types.Q8_0, 1)
assert.equal(summary.tensor_inventory.total_n_bytes, 10)
assert.match(summary.tensor_inventory.sha256, /^[0-9a-f]{64}$/)
assert.equal(JSON.stringify(summary).includes('absolute_offset'), false)

console.log('test-hf-qualification-header: all checks passed')
