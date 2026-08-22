#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  BenchGenerateParseError,
  parseBenchGenerateJsonl,
} from './lib/bench-generate.mjs'

const root = resolve(fileURLToPath(new URL('.', import.meta.url)))
const fixtureRoot = resolve(root, 'fixtures/bench-generate')
const directText = await fixture('valid-direct.jsonl')
const runnableText = await fixture('valid-runnable.jsonl')
const truncatedText = await fixture('invalid-truncated.jsonl')
const negativeMetricText = await fixture('invalid-negative-metric.jsonl')

const direct = parseBenchGenerateJsonl(directText, {
  expectedCommit: '2222222222222222222222222222222222222222',
  expectedModel: '<model>/synthetic-direct-Q8_0.gguf',
})
assert.equal(direct.length, 1)
assert.equal(direct[0].iteration, 0)
assert.equal(direct[0].generated_tokens, 4)
assert.equal(Object.hasOwn(direct[0], 'offload'), false)

const runnable = parseBenchGenerateJsonl(runnableText)
assert.equal(runnable.length, 1)
assert.equal(runnable[0].prefill_ms, runnable[0].ttft_ms)
assert.equal(runnable[0].offload.source, 'none')

await rejects(
  () => parseBenchGenerateJsonl(''),
  'empty_output',
  /contained no records/,
)
await rejects(
  () => parseBenchGenerateJsonl('[bench-generate] warmup\n'),
  'invalid_parse',
  /line 1 is not JSON/,
)
await rejects(
  () => parseBenchGenerateJsonl(truncatedText),
  'invalid_parse',
  /line 1 is not JSON/,
)
await rejects(
  () => parseBenchGenerateJsonl(negativeMetricText),
  'invalid_contract',
  /tokens_per_second must be a non-negative finite number/,
)
await rejects(
  () => parseBenchGenerateJsonl(directText, { expectedCommit: '3333333333333333333333333333333333333333' }),
  'invalid_identity',
  /does not match/,
)

const duplicate = `${directText.trim()}\n${directText.trim()}\n`
await rejects(
  () => parseBenchGenerateJsonl(duplicate),
  'invalid_sequence',
  /iteration 0 appeared at record 1; expected 1/,
)

const wrongThroughput = mutateRecord(directText, (record) => {
  record.tokens_per_second = 99
})
await rejects(
  () => parseBenchGenerateJsonl(wrongThroughput),
  'invalid_contract',
  /does not match 3 decode tokens/,
)

const unknownField = mutateRecord(directText, (record) => {
  record.stderr = 'must never be accepted as a metric field'
})
await rejects(
  () => parseBenchGenerateJsonl(unknownField),
  'invalid_contract',
  /stderr is not allowed/,
)

console.log('benchmark Phase 1 bench-generate parser: PASS')

async function fixture(name) {
  return readFile(resolve(fixtureRoot, name), 'utf8')
}

function mutateRecord(text, mutate) {
  const record = JSON.parse(text.trim())
  mutate(record)
  return `${JSON.stringify(record)}\n`
}

async function rejects(action, expectedCode, expectedMessage) {
  try {
    await action()
    assert.fail('expected action to reject')
  } catch (error) {
    assert.ok(error instanceof BenchGenerateParseError)
    assert.equal(error.code, expectedCode)
    assert.match(error.message, expectedMessage)
  }
}
