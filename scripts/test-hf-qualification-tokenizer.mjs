#!/usr/bin/env node

import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import {
  GEMMA2_CASES,
  assertGemma2TokenizerMetadata,
  buildCamelidArgs,
  buildLlamaArgs,
  classifyCamelidProvenance,
  makeVocabOnlyGguf,
  parseLlamaIds,
  parseLlamaVersionOutput,
  sourceSelectionForRow,
} from './hf-qualification-tokenizer.mjs'
import { validateLockAgainstSelection } from './hf-qualification-source.mjs'

const prefix = Buffer.alloc(64)
prefix.write('GGUF', 0, 'ascii')
prefix.writeUInt32LE(3, 4)
prefix.writeBigInt64LE(464n, 8)
prefix.writeBigInt64LE(26n, 16)
const derived = makeVocabOnlyGguf(prefix)
assert.equal(derived.original_tensor_count, 464)
assert.equal(derived.metadata_count, 26)
assert.equal(derived.patched_offset, 8)
assert.equal(derived.bytes.readBigInt64LE(8), 0n)
assert.equal(prefix.readBigInt64LE(8), 464n, 'derivation must not mutate the source prefix')
assert.deepEqual(derived.bytes.subarray(24), prefix.subarray(24), 'metadata/body bytes must be unchanged')
assert.throws(() => makeVocabOnlyGguf(Buffer.alloc(8)), /too short/)
assert.throws(() => makeVocabOnlyGguf(Buffer.alloc(64)), /invalid magic/)

assert.deepEqual(parseLlamaIds('[2, 1, 42]\n'), [2, 1, 42])
assert.deepEqual(parseLlamaIds('diagnostic\n[2,3]\n'), [2, 3])
assert.throws(() => parseLlamaIds('no ids'), /did not emit/)
assert.throws(() => parseLlamaIds('[2,-1]'), /invalid ID/)
assert.deepEqual(
  parseLlamaVersionOutput('version: 9632 (acd79d603)\nbuilt with Clang'),
  { build: 9632, revision: 'acd79d603' },
)
assert.throws(
  () => parseLlamaVersionOutput('version: 9632 (deadbeef0)'),
  /does not match pin/,
)
assert.throws(() => parseLlamaVersionOutput('unknown version'), /parseable build revision/)

assert.deepEqual(classifyCamelidProvenance({
  version: 'camelid v0.6.1-24-g96db3486',
  sourceHead: '96db34867b0402f7775670d4a767fae73b6b19d9',
  sourceTrackedDirty: false,
}), {
  status: 'clean_current_head_receipt',
  gate_requires_clean_current_head: true,
  source_head: '96db34867b0402f7775670d4a767fae73b6b19d9',
  source_tracked_dirty: false,
  binary_commit_abbrev: '96db3486',
  binary_reports_dirty: false,
  binary_matches_source_head: true,
  clean_current_head: true,
})
assert.equal(classifyCamelidProvenance({
  version: 'camelid v0.6.1-24-g96db3486-dirty',
  sourceHead: '96db34867b0402f7775670d4a767fae73b6b19d9',
  sourceTrackedDirty: true,
}).clean_current_head, false)
assert.equal(classifyCamelidProvenance({
  version: 'camelid v0.6.1-23-g35ca855f',
  sourceHead: '96db34867b0402f7775670d4a767fae73b6b19d9',
  sourceTrackedDirty: false,
}).binary_matches_source_head, false)

const camelidArgs = buildCamelidArgs({
  prefixPath: 'header.gguf',
  declaredLength: 9_827_149_312,
  inputPath: 'inputs.json',
  addSpecial: false,
  parseSpecial: true,
})
assert.deepEqual(camelidArgs, [
  'tokenize',
  '--model', 'header.gguf',
  '--declared-len', '9827149312',
  '--file', 'inputs.json',
  '--parse-special',
  '--no-add-special',
])

const llamaArgs = buildLlamaArgs({
  modelPath: 'vocab-only.gguf',
  promptPath: 'prompt.txt',
  addSpecial: false,
  parseSpecial: false,
})
assert.deepEqual(llamaArgs, [
  '--model', 'vocab-only.gguf',
  '--file', 'prompt.txt',
  '--ids',
  '--no-escape',
  '--log-disable',
  '--no-bos',
  '--no-parse-special',
])

const fixture = JSON.parse(readFileSync(
  new URL('../qa/model-qualification/fixtures/gemma2-it-chat-template-v1.json', import.meta.url),
  'utf8',
))
const tokenArray = Array(256_000).fill('x')
const metadata = {
  'general.architecture': 'gemma2',
  'tokenizer.ggml.model': 'llama',
  'tokenizer.ggml.pre': 'default',
  'tokenizer.ggml.bos_token_id': 2,
  'tokenizer.ggml.eos_token_id': 1,
  'tokenizer.ggml.unknown_token_id': 3,
  'tokenizer.ggml.add_bos_token': true,
  'tokenizer.ggml.add_eos_token': false,
  'tokenizer.ggml.add_space_prefix': false,
  'tokenizer.ggml.tokens': tokenArray,
  'tokenizer.ggml.scores': Array(256_000).fill(0),
  'tokenizer.ggml.token_type': Array(256_000).fill(1),
  'tokenizer.chat_template': fixture.source_template,
}
const summary = assertGemma2TokenizerMetadata({ metadata })
assert.deepEqual(summary, {
  token_count: 256_000,
  score_count: 256_000,
  token_type_count: 256_000,
  chat_template_utf8_bytes: 591,
  chat_template_sha256: 'ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6',
})
assert.throws(
  () => assertGemma2TokenizerMetadata({ metadata: { ...metadata, 'tokenizer.ggml.scores': [] } }),
  /score array/,
)
assert.throws(
  () => assertGemma2TokenizerMetadata({ metadata: { ...metadata, 'tokenizer.ggml.pre': 'gemma' } }),
  /tokenizer.ggml.pre mismatch/,
)

assert.equal(GEMMA2_CASES.length, 7)
assert(GEMMA2_CASES.some((testCase) => testCase.parse_special))
assert(GEMMA2_CASES.some((testCase) => !testCase.parse_special))
assert(GEMMA2_CASES.some((testCase) => !testCase.add_special))

const selected = sourceSelectionForRow({
  id: 'gemma2_9b_it_q8_0',
  identity: { size_bytes: 1234, sha256: 'a'.repeat(64) },
  source: {
    repo: 'org/gemma',
    file: 'gemma.gguf',
    revision: '1'.repeat(40),
    license: 'gemma',
  },
})
const lock = {
  repo: selected.repo,
  file: selected.file,
  revision: selected.revision,
  size_bytes: selected.expected.size_bytes,
  sha256: selected.expected.sha256,
  license: selected.expected.license,
}
assert.doesNotThrow(() => validateLockAgainstSelection(lock, selected))
assert.throws(
  () => validateLockAgainstSelection({ ...lock, sha256: 'b'.repeat(64) }, selected),
  /sha256/,
  'the tokenizer harness must reject remote bytes that drift from the roster identity',
)

const receipt = JSON.parse(readFileSync(
  new URL('../qa/model-qualification/gemma2-9b-it-q8-header-tokenizer-parity.json', import.meta.url),
  'utf8',
))
assert.equal(receipt.schema, 'camelid.header-tokenizer-parity/v1')
assert.equal(receipt.result.case_count, GEMMA2_CASES.length)
assert.equal(receipt.result.exact_match_count, GEMMA2_CASES.length)
assert.equal(receipt.result.all_token_ids_match, true)
assert.equal(receipt.provenance.status, 'preparatory_requires_clean_current_head_rerun')
assert.equal(receipt.provenance.gate_requires_clean_current_head, true)
assert.equal(receipt.provenance.binary_matches_source_head, true)
assert.equal(receipt.provenance.clean_current_head, false)
assert.equal(receipt.provenance.binary_reports_dirty, true)
assert(receipt.cases.every((testCase) => testCase.exact_match))
assert.match(receipt.oracle.input, /tensor_count is zeroed/)
assert.match(receipt.bounded_fetch.scope_note, /opaque initial tensor payload bytes/)
assert(!/[A-Za-z]:[\\/]/.test(JSON.stringify(receipt)), 'receipt must not expose an absolute Windows path')

console.log('test-hf-qualification-tokenizer: all checks passed')
