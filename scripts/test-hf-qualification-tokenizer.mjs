#!/usr/bin/env node

import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import {
  GEMMA2_CASES,
  SMOLLM3_CASES,
  TokenizerQualificationError,
  assessTokenizerReceipt,
  assertGemma2TokenizerMetadata,
  assertSmolLM3TokenizerMetadata,
  buildCamelidArgs,
  buildLlamaArgs,
  classifyCamelidProvenance,
  classifyTokenizerQualificationError,
  inspectRemoteTokenizer,
  makeVocabOnlyGguf,
  normalizeTokenizerPrefixBytes,
  parseLlamaIds,
  parseLlamaVersionOutput,
  runTokenizerCli,
  sourceSelectionForRow,
  tokenizerPackAvailable,
  tokenizerPrefixBytesForRow,
  validateGemma2TokenizerReceipt,
  validateSmolLM3TokenizerReceipt,
} from './hf-qualification-tokenizer.mjs'
import { validateLockAgainstSelection } from './hf-qualification-source.mjs'

const sha256 = (value) => createHash('sha256').update(Buffer.from(value)).digest('hex')

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
assert.equal(tokenizerPackAvailable('gemma2_9b_it_q8_0'), true)
assert.equal(tokenizerPackAvailable('smollm3_3b_q8_0'), true)
assert.equal(tokenizerPackAvailable('not_a_pack'), false)
assert.equal(tokenizerPrefixBytesForRow('gemma2_9b_it_q8_0'), 32 * 1024 * 1024)
assert.equal(tokenizerPrefixBytesForRow('not_a_pack'), null)

const typedTokenizerFailure = classifyTokenizerQualificationError(
  new TokenizerQualificationError(
    'tokenizer_oracle_unavailable',
    'fail',
    'C:\\private\\llama-tokenize.exe test-secret-token',
  ),
)
assert.deepEqual(typedTokenizerFailure, {
  status: 'blocked',
  error_code: 'tokenizer_oracle_unavailable',
  reason: 'the pinned llama.cpp tokenizer oracle is unavailable',
})
const mutatedTokenizerError = new TokenizerQualificationError(
  'tokenizer_probe_failed',
  'fail',
  'safe',
)
mutatedTokenizerError.code = 'forged'
mutatedTokenizerError.status = 'pass'
mutatedTokenizerError.message = 'C:\\private\\model.gguf bearer-token'
assert.deepEqual(classifyTokenizerQualificationError(mutatedTokenizerError), {
  status: 'fail',
  error_code: 'tokenizer_probe_failed',
  reason: 'an exact-row tokenizer probe failed to produce a valid result',
})
const mutatedKnownTokenizerError = new TokenizerQualificationError(
  'tokenizer_oracle_unavailable',
  'blocked',
  'safe',
)
mutatedKnownTokenizerError.code = 'tokenizer_source_identity_mismatch'
mutatedKnownTokenizerError.status = 'fail'
mutatedKnownTokenizerError.message = 'C:\\private\\model.gguf bearer-token'
assert.deepEqual(classifyTokenizerQualificationError(mutatedKnownTokenizerError), {
  status: 'blocked',
  error_code: 'tokenizer_oracle_unavailable',
  reason: 'the pinned llama.cpp tokenizer oracle is unavailable',
})
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

assert.equal(normalizeTokenizerPrefixBytes('gemma2_9b_it_q8_0', 32 * 1024 * 1024), 32 * 1024 * 1024)
assert.equal(normalizeTokenizerPrefixBytes('smollm3_3b_q8_0', '33554432'), 32 * 1024 * 1024)
assert.throws(
  () => normalizeTokenizerPrefixBytes('smollm3_3b_q8_0', 16 * 1024 * 1024),
  /requires exactly 33554432 prefix bytes/,
)

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

const smolTokens = Array(128_256).fill('x')
const smolTypes = Array(128_256).fill(1)
for (let id = 128_000; id < 128_256; id += 1) smolTypes[id] = 3
for (const id of [128_002, 128_003, 128_013, 128_014, 128_015, 128_016, 128_017, 128_018]) {
  smolTypes[id] = 4
}
for (const [id, text] of [
  [128_000, '<|begin_of_text|>'],
  [128_001, '<|end_of_text|>'],
  [128_002, '<think>'],
  [128_003, '</think>'],
  [128_006, '<|start_header_id|>'],
  [128_007, '<|end_header_id|>'],
  [128_008, '<|eom_id|>'],
  [128_009, '<|eot_id|>'],
  [128_010, '<|python_tag|>'],
  [128_011, '<|im_start|>'],
  [128_012, '<|im_end|>'],
  [128_013, '<tool_response>'],
  [128_014, '</tool_response>'],
  [128_015, '<tool_call>'],
  [128_016, '</tool_call>'],
  [128_017, '<code>'],
  [128_018, '</code>'],
]) smolTokens[id] = text
const smolMetadata = {
  'general.architecture': 'smollm3',
  'general.license': 'apache-2.0',
  'smollm3.vocab_size': 128_256,
  'tokenizer.ggml.model': 'gpt2',
  'tokenizer.ggml.pre': 'smaug-bpe',
  'tokenizer.ggml.bos_token_id': 128_000,
  'tokenizer.ggml.eos_token_id': 128_012,
  'tokenizer.ggml.padding_token_id': 128_012,
  'tokenizer.ggml.tokens': smolTokens,
  'tokenizer.ggml.merges': Array(280_147).fill('a b'),
  'tokenizer.ggml.token_type': smolTypes,
  'tokenizer.chat_template': 'intentionally not the pinned 5493-byte template',
}
assert.throws(
  () => assertSmolLM3TokenizerMetadata({ metadata: smolMetadata }),
  /chat template does not match/,
  'all grounded SmolLM3 metadata must reach the final exact template check',
)
for (const key of [
  'tokenizer.ggml.add_bos_token',
  'tokenizer.ggml.add_eos_token',
  'tokenizer.ggml.add_space_prefix',
]) {
  assert.throws(
    () => assertSmolLM3TokenizerMetadata({ metadata: { ...smolMetadata, [key]: false } }),
    new RegExp(`unexpectedly declares ${key.replaceAll('.', '\\.')}`),
    `the exact row pins ${key} absence, not an invented explicit false value`,
  )
}
const driftedSmolTypes = [...smolTypes]
driftedSmolTypes[128_255] = 1
assert.throws(
  () => assertSmolLM3TokenizerMetadata({
    metadata: { ...smolMetadata, 'tokenizer.ggml.token_type': driftedSmolTypes },
  }),
  /special-token type counts mismatch/,
)

assert.equal(GEMMA2_CASES.length, 7)
assert(GEMMA2_CASES.some((testCase) => testCase.parse_special))
assert(GEMMA2_CASES.some((testCase) => !testCase.parse_special))
assert(GEMMA2_CASES.some((testCase) => !testCase.add_special))
assert.equal(SMOLLM3_CASES.length, 10)
assert(SMOLLM3_CASES.some((testCase) => testCase.id === 'empty_with_add_special'))
assert(SMOLLM3_CASES.some((testCase) => testCase.id === 'plain_ascii_with_add_special'))
assert(SMOLLM3_CASES.some((testCase) => /unicode/.test(testCase.id)))
assert(SMOLLM3_CASES.some((testCase) => /contractions/.test(testCase.id)))
assert(SMOLLM3_CASES.some((testCase) => testCase.parse_special))
assert(SMOLLM3_CASES.some((testCase) => !testCase.parse_special))

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
const roster = JSON.parse(readFileSync(
  new URL('../qa/model-qualification/phase1-roster.json', import.meta.url),
  'utf8',
))
const receiptRow = roster.rows.find((row) => row.id === 'gemma2_9b_it_q8_0')
assert.deepEqual(
  validateGemma2TokenizerReceipt(receipt, receiptRow, roster.defaults),
  [],
  'durable Gemma2 tokenizer evidence must be derived from exact identities and token arrays',
)
assert.equal(receipt.schema, 'camelid.header-tokenizer-parity/v1')
assert.equal(receipt.result.case_count, GEMMA2_CASES.length)
assert.equal(receipt.result.exact_match_count, GEMMA2_CASES.length)
assert.equal(receipt.result.all_token_ids_match, true)
assert.equal(receipt.provenance.status, 'clean_current_head_receipt')
assert.equal(receipt.provenance.gate_requires_clean_current_head, true)
assert.equal(receipt.provenance.binary_matches_source_head, true)
assert.equal(receipt.provenance.clean_current_head, true)
assert.equal(receipt.provenance.source_tracked_dirty, false)
assert.equal(receipt.provenance.binary_reports_dirty, false)
assert(receipt.cases.every((testCase) => testCase.exact_match))
assert.match(receipt.oracle.input, /tensor_count is zeroed/)
assert.match(receipt.bounded_fetch.scope_note, /opaque initial tensor payload bytes/)
assert(!/[A-Za-z]:[\\/]/.test(JSON.stringify(receipt)), 'receipt must not expose an absolute Windows path')

const boundAssessment = assessTokenizerReceipt(receipt, receiptRow, roster.defaults, {
  expectedSourceHead: receipt.provenance.source_head,
})
assert.deepEqual(boundAssessment.errors, [])
assert.deepEqual(boundAssessment.parity_errors, [])
assert.equal(boundAssessment.all_token_ids_match, true)
const staleHeadAssessment = assessTokenizerReceipt(receipt, receiptRow, roster.defaults, {
  expectedSourceHead: 'f'.repeat(40),
})
assert(
  staleHeadAssessment.errors.some((error) => error.includes('expected source HEAD')),
  'factory receipt assessment must bind the Camelid binary to the report source HEAD',
)

const honestMismatch = structuredClone(receipt)
honestMismatch.cases[0].llama_cpp_ids[0] = (honestMismatch.cases[0].llama_cpp_ids[0] + 1) % 256_000
honestMismatch.cases[0].exact_match = false
honestMismatch.result.exact_match_count -= 1
honestMismatch.result.all_token_ids_match = false
const honestMismatchAssessment = assessTokenizerReceipt(
  honestMismatch,
  receiptRow,
  roster.defaults,
  { expectedSourceHead: receipt.provenance.source_head },
)
assert.deepEqual(honestMismatchAssessment.errors, [], 'an honest mismatch remains a valid receipt')
assert.equal(honestMismatchAssessment.all_token_ids_match, false)
assert.equal(honestMismatchAssessment.parity_errors.length, 1)

const forgedMismatchFlags = structuredClone(honestMismatch)
forgedMismatchFlags.cases[0].exact_match = true
forgedMismatchFlags.result.exact_match_count = forgedMismatchFlags.cases.length
forgedMismatchFlags.result.all_token_ids_match = true
const forgedMismatchAssessment = assessTokenizerReceipt(
  forgedMismatchFlags,
  receiptRow,
  roster.defaults,
  { expectedSourceHead: receipt.provenance.source_head },
)
assert(forgedMismatchAssessment.errors.some((error) => error.includes('exact_match is not derived')))
assert(forgedMismatchAssessment.errors.some((error) => error.includes('result exact_match_count mismatch')))

const callableLock = {
  repo: receiptRow.source.repo,
  file: receiptRow.source.file,
  revision: receiptRow.source.revision,
  size_bytes: receiptRow.identity.size_bytes,
  sha256: receiptRow.identity.sha256,
  license: receiptRow.source.license,
  download_url: 'https://huggingface.co/example/unused-in-injected-tests',
}
const callableHead = '1'.repeat(40)
const callableSource = { sourceHead: callableHead, sourceTrackedDirty: false }
const callableProvenance = classifyCamelidProvenance({
  version: 'camelid v0.6.1-g11111111',
  ...callableSource,
})
const callableIdentity = {
  version: 'camelid v0.6.1-g11111111',
  binary_sha256: 'a'.repeat(64),
  provenance: callableProvenance,
}

let dirtyPreflightFetched = false
await assert.rejects(
  inspectRemoteTokenizer(callableLock, {
    row: receiptRow,
    defaults: roster.defaults,
    binary: 'camelid',
    llamaTokenize: 'llama-tokenize',
    sourceProvenanceImpl: async () => ({ sourceHead: callableHead, sourceTrackedDirty: true }),
    fetchPrefixImpl: async () => { dirtyPreflightFetched = true },
  }),
  (error) => error instanceof TokenizerQualificationError
    && error.code === 'tokenizer_inspector_not_clean_current_head',
)
assert.equal(dirtyPreflightFetched, false, 'tracked-dirty source must block before the range fetch')

let invalidBudgetSourceCalled = false
await assert.rejects(
  inspectRemoteTokenizer(callableLock, {
    row: receiptRow,
    defaults: roster.defaults,
    binary: 'camelid',
    llamaTokenize: 'llama-tokenize',
    prefixBytes: 16 * 1024 * 1024,
    sourceProvenanceImpl: async () => { invalidBudgetSourceCalled = true },
  }),
  (error) => error instanceof TokenizerQualificationError
    && error.code === 'tokenizer_prefix_budget_invalid',
)
assert.equal(invalidBudgetSourceCalled, false, 'invalid pack budget must fail before local or network probes')

let staleOracleFetched = false
await assert.rejects(
  inspectRemoteTokenizer(callableLock, {
    row: receiptRow,
    defaults: roster.defaults,
    binary: 'camelid',
    llamaTokenize: 'llama-tokenize',
    sourceProvenanceImpl: async () => callableSource,
    camelidIdentityImpl: async () => callableIdentity,
    llamaPackageImpl: async () => ({
      revision: roster.defaults.llama_cpp.revision,
      build: Number(roster.defaults.llama_cpp.build.slice(1)),
      binary_sha256: '0'.repeat(64),
      companion_binary_sha256: '0'.repeat(64),
    }),
    fetchPrefixImpl: async () => { staleOracleFetched = true },
  }),
  (error) => error instanceof TokenizerQualificationError
    && error.code === 'tokenizer_oracle_identity_mismatch',
)
assert.equal(staleOracleFetched, false, 'oracle identity drift must fail before the range fetch')

let invalidCliBudgetResolved = false
await assert.rejects(
  runTokenizerCli([
    '--row', 'gemma2_9b_it_q8_0',
    '--prefix-bytes', '1',
  ], {
    sourceResolver: async () => {
      invalidCliBudgetResolved = true
      return callableLock
    },
  }),
  (error) => error instanceof TokenizerQualificationError
    && error.code === 'tokenizer_prefix_budget_invalid'
    && classifyTokenizerQualificationError(error).status === 'fail',
)
assert.equal(invalidCliBudgetResolved, false, 'invalid CLI budget must fail before HF source lookup')

let sourceDriftPreflightCalled = false
await assert.rejects(
  inspectRemoteTokenizer({ ...callableLock, sha256: '0'.repeat(64) }, {
    row: receiptRow,
    defaults: roster.defaults,
    binary: 'camelid',
    llamaTokenize: 'llama-tokenize',
    sourceProvenanceImpl: async () => { sourceDriftPreflightCalled = true },
  }),
  (error) => error instanceof TokenizerQualificationError
    && error.code === 'tokenizer_source_identity_mismatch',
)
assert.equal(sourceDriftPreflightCalled, false, 'source drift must fail before local or network probes')

const callablePrefix = Buffer.alloc(32 * 1024 * 1024)
callablePrefix.write('GGUF', 0, 'ascii')
callablePrefix.writeUInt32LE(3, 4)
callablePrefix.writeBigInt64LE(464n, 8)
callablePrefix.writeBigInt64LE(26n, 16)
const callableMetadataSummary = {
  token_count: 256_000,
  score_count: 256_000,
  token_type_count: 256_000,
  chat_template_utf8_bytes: 591,
  chat_template_sha256: 'ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6',
}
const callableLlamaPackage = {
  revision: roster.defaults.llama_cpp.revision,
  build: Number(roster.defaults.llama_cpp.build.slice(1)),
  executable: 'llama-tokenize',
  binary_sha256: 'a44a4d7e1445d22a4cffb0d38f6efa8f1d81e84ae2c3d481af857c5e331b8c7a',
  companion_executable: 'llama-cli',
  companion_binary_sha256: '2ec09da0b81d0201ce5b21810caefb4e77fd108f383b30c15ca493c5a70f7731',
}
const callableOptions = (overrides = {}) => ({
  row: receiptRow,
  defaults: roster.defaults,
  binary: 'camelid',
  llamaTokenize: 'llama-tokenize',
  sourceRoot: 'virtual-source-root',
  prefixBytes: 32 * 1024 * 1024,
  token: 'test-secret-token',
  sourceProvenanceImpl: async () => callableSource,
  camelidIdentityImpl: async () => callableIdentity,
  llamaPackageImpl: async () => callableLlamaPackage,
  fetchPrefixImpl: async (_lock, options) => {
    assert.equal(options.prefixBytes, 32 * 1024 * 1024)
    assert.equal(options.token, 'test-secret-token')
    return {
      bytes: callablePrefix,
      requested_bytes: 32 * 1024 * 1024,
      content_range: {
        start: 0,
        end: 32 * 1024 * 1024 - 1,
        total: receiptRow.identity.size_bytes,
      },
      prefix_sha256: 'b2bcc601c188ffc7c306f0011944a7a5492bfde490c34ddc390b69424c09a5e5',
    }
  },
  prefixSha256Impl: () => 'b2bcc601c188ffc7c306f0011944a7a5492bfde490c34ddc390b69424c09a5e5',
  mkdtempImpl: async () => 'virtual-tokenizer-temp',
  writeFileImpl: async () => {},
  inspectImpl: async () => ({ tensor_count: 464 }),
  metadataValidatorImpl: () => callableMetadataSummary,
  deriveImpl: () => ({
    bytes: Buffer.from('vocab-only-derivative'),
    original_tensor_count: 464,
    metadata_count: 26,
    patched_offset: 8,
  }),
  camelidCaseImpl: async () => ({ ids: [42], decoded: 'probe' }),
  llamaCaseImpl: async () => [42],
  now: () => new Date('2026-08-10T19:30:00.000Z'),
  ...overrides,
})

let successCleanupCalls = 0
let successSourceCalls = 0
const callableSuccessReceipt = await inspectRemoteTokenizer(callableLock, callableOptions({
  sourceProvenanceImpl: async () => {
    successSourceCalls += 1
    return callableSource
  },
  rmImpl: async (path, options) => {
    successCleanupCalls += 1
    assert.equal(path, 'virtual-tokenizer-temp')
    assert.deepEqual(options, { recursive: true, force: true })
  },
}))
assert.equal(successSourceCalls, 2, 'successful qualification must inspect source before and after probes')
assert.equal(successCleanupCalls, 1, 'successful qualification must delete its temporary directory')
assert.equal(callableSuccessReceipt.bounded_fetch.temporary_files_deleted, true)
assert.equal(callableSuccessReceipt.result.all_token_ids_match, true)
assert.equal(JSON.stringify(callableSuccessReceipt).includes('test-secret-token'), false)
const callableSuccessAssessment = assessTokenizerReceipt(
  callableSuccessReceipt,
  receiptRow,
  roster.defaults,
  { expectedSourceHead: callableHead },
)
assert.deepEqual(callableSuccessAssessment.errors, [])
assert.deepEqual(callableSuccessAssessment.parity_errors, [])

let changedCamelidIdentityCalls = 0
let changedCamelidCleanupCalls = 0
await assert.rejects(
  inspectRemoteTokenizer(callableLock, callableOptions({
    camelidIdentityImpl: async () => {
      changedCamelidIdentityCalls += 1
      return changedCamelidIdentityCalls === 1
        ? callableIdentity
        : { ...callableIdentity, binary_sha256: 'b'.repeat(64) }
    },
    rmImpl: async () => { changedCamelidCleanupCalls += 1 },
  })),
  (error) => error instanceof TokenizerQualificationError
    && error.code === 'tokenizer_inspector_changed',
)
assert.equal(changedCamelidIdentityCalls, 2, 'Camelid identity must be checked before and after probes')
assert.equal(changedCamelidCleanupCalls, 1, 'Camelid identity drift must still clean temporary files')

let changedOracleIdentityCalls = 0
let changedOracleCleanupCalls = 0
await assert.rejects(
  inspectRemoteTokenizer(callableLock, callableOptions({
    llamaPackageImpl: async () => {
      changedOracleIdentityCalls += 1
      return changedOracleIdentityCalls === 1
        ? callableLlamaPackage
        : { ...callableLlamaPackage, binary_sha256: 'b'.repeat(64) }
    },
    rmImpl: async () => { changedOracleCleanupCalls += 1 },
  })),
  (error) => error instanceof TokenizerQualificationError
    && error.code === 'tokenizer_oracle_changed',
)
assert.equal(changedOracleIdentityCalls, 2, 'oracle identity must be checked before and after probes')
assert.equal(changedOracleCleanupCalls, 1, 'oracle identity drift must still clean temporary files')

const callableMismatchReceipt = await inspectRemoteTokenizer(callableLock, callableOptions({
  llamaCaseImpl: async (_binary, _model, _temporary, testCase) => (
    testCase.id === GEMMA2_CASES[0].id ? [43] : [42]
  ),
  rmImpl: async () => {},
}))
assert.equal(callableMismatchReceipt.result.all_token_ids_match, false)
assert.equal(callableMismatchReceipt.result.exact_match_count, GEMMA2_CASES.length - 1)
const callableMismatchAssessment = assessTokenizerReceipt(
  callableMismatchReceipt,
  receiptRow,
  roster.defaults,
  { expectedSourceHead: callableHead },
)
assert.deepEqual(callableMismatchAssessment.errors, [])
assert.equal(callableMismatchAssessment.parity_errors.length, 1)

let driftSourceCalls = 0
let driftCleanupCalls = 0
await assert.rejects(
  inspectRemoteTokenizer(callableLock, callableOptions({
    sourceProvenanceImpl: async () => {
      driftSourceCalls += 1
      return driftSourceCalls === 1
        ? callableSource
        : { sourceHead: '2'.repeat(40), sourceTrackedDirty: false }
    },
    rmImpl: async () => { driftCleanupCalls += 1 },
  })),
  (error) => error instanceof TokenizerQualificationError
    && error.code === 'tokenizer_source_changed',
)
assert.equal(driftSourceCalls, 2, 'post-probe source identity must be re-read')
assert.equal(driftCleanupCalls, 1, 'post-probe source drift must still clean temporary files')

await assert.rejects(
  inspectRemoteTokenizer(callableLock, callableOptions({
    rmImpl: async () => { throw new Error('C:\\private\\locked test-secret-token') },
  })),
  (error) => error instanceof TokenizerQualificationError
    && error.code === 'tokenizer_cleanup_failed'
    && !error.message.includes('private')
    && !error.message.includes('test-secret-token'),
)

const forgedIds = structuredClone(receipt)
forgedIds.cases[0].llama_cpp_ids = [999]
forgedIds.cases[0].exact_match = true
assert(
  validateGemma2TokenizerReceipt(forgedIds, receiptRow, roster.defaults)
    .some((error) => error.includes('token IDs diverge')),
  'receipt-authored exact_match must not hide divergent IDs',
)
const allEmptyGemma = structuredClone(receipt)
for (const testCase of allEmptyGemma.cases) {
  testCase.camelid_ids = []
  testCase.llama_cpp_ids = []
  testCase.exact_match = true
}
allEmptyGemma.result.exact_match_count = allEmptyGemma.cases.length
allEmptyGemma.result.all_token_ids_match = true
assert(
  validateGemma2TokenizerReceipt(allEmptyGemma, receiptRow, roster.defaults)
    .some((error) => error.includes('invalid token IDs')),
  'paired empty Gemma2 outputs must not qualify as exact tokenizer parity',
)
const driftedSource = structuredClone(receipt)
driftedSource.source.sha256 = '0'.repeat(64)
assert(
  validateGemma2TokenizerReceipt(driftedSource, receiptRow, roster.defaults)
    .some((error) => error.includes('source.sha256 mismatch')),
  'receipt source identity must remain bound to the roster row',
)
const driftedOracle = structuredClone(receipt)
driftedOracle.oracle.revision = 'deadbeef0'
assert(
  validateGemma2TokenizerReceipt(driftedOracle, receiptRow, roster.defaults)
    .some((error) => error.includes('llama.cpp revision mismatch')),
  'receipt oracle identity must remain bound to the roster pin',
)

const smolRow = roster.rows.find((row) => row.id === 'smollm3_3b_q8_0')
const smolReceipt = JSON.parse(readFileSync(
  new URL('../qa/model-qualification/smollm3-3b-q8-header-tokenizer-parity.json', import.meta.url),
  'utf8',
))
assert.deepEqual(
  validateSmolLM3TokenizerReceipt(smolReceipt, smolRow, roster.defaults),
  [],
  'durable SmolLM3 tokenizer evidence must remain bound to the exact row and oracle pack',
)
assert.equal(smolReceipt.result.case_count, SMOLLM3_CASES.length)
assert.equal(smolReceipt.result.exact_match_count, SMOLLM3_CASES.length)
assert.equal(smolReceipt.result.all_token_ids_match, true)
assert.equal(smolReceipt.result.support_decision, 'smollm3_exact_row_tokenizer_gate_only')
assert.equal(smolReceipt.provenance.status, 'clean_current_head_receipt')
assert.equal(smolReceipt.provenance.clean_current_head, true)
assert.deepEqual(
  smolReceipt.cases.find((testCase) => testCase.id === 'empty_with_add_special').camelid_ids,
  [],
)
assert.deepEqual(
  smolReceipt.cases.find((testCase) => testCase.id === 'plain_ascii_with_add_special').camelid_ids,
  [9_906],
)
assert.deepEqual(
  smolReceipt.cases.find(
    (testCase) => testCase.id === 'user_defined_tool_tags_without_parse_special',
  ).camelid_ids,
  [128_015, 5_018, 609, 3_332, 15_561, 9_388, 128_016],
)

const syntheticSmolCases = SMOLLM3_CASES.map((testCase) => {
  let ids = [42]
  if (testCase.id === 'empty_with_add_special') ids = []
  if (testCase.id === 'plain_ascii_with_add_special'
    || testCase.id === 'plain_ascii_without_add_special') ids = [9_906]
  if (testCase.id === 'single_user_chat_controls'
    || testCase.id === 'multi_turn_chat_controls') ids = [128_011, 882, 128_012]
  if (testCase.id === 'chat_controls_as_ordinary_text') ids = [27, 91, 318]
  if (testCase.id === 'user_defined_tool_tags_with_parse_special'
    || testCase.id === 'user_defined_tool_tags_without_parse_special') ids = [128_015, 5018, 128_016]
  return {
    id: testCase.id,
    text_utf8_bytes: Buffer.byteLength(testCase.text),
    text_sha256: sha256(testCase.text),
    add_special: testCase.add_special,
    parse_special: testCase.parse_special,
    camelid_ids: ids,
    llama_cpp_ids: [...ids],
    exact_match: true,
    camelid_decoded_sha256: 'd'.repeat(64),
  }
})
const syntheticSmolReceipt = {
  schema: 'camelid.header-tokenizer-parity/v1',
  generated_at: '2026-08-10T18:00:00.000Z',
  provenance: structuredClone(receipt.provenance),
  row_id: smolRow.id,
  host: { platform: 'win32-x64', hostname_redacted: true },
  source: {
    repo: smolRow.source.repo,
    file: smolRow.source.file,
    revision: smolRow.source.revision,
    size_bytes: smolRow.identity.size_bytes,
    sha256: smolRow.identity.sha256,
    license: smolRow.source.license,
  },
  bounded_fetch: {
    requested_bytes: 32 * 1024 * 1024,
    received_bytes: 32 * 1024 * 1024,
    content_range: { start: 0, end: 32 * 1024 * 1024 - 1, total: smolRow.identity.size_bytes },
    prefix_sha256: '2d043b2114b89100c7ba464e57375a6f32c06c04729542d54ed684b5e8c5016e',
    temporary_paths_redacted: true,
    temporary_files_deleted: true,
  },
  grounding: {
    header_receipt: 'qa/model-qualification/smollm3-3b-q8-header-inspection.json',
    tokenizer_pre_fixture: 'qa/model-qualification/fixtures/smollm3-tokenizer-pre-v1.json',
  },
  tokenizer_metadata: {
    token_count: 128_256,
    merge_count: 280_147,
    token_type_count: 128_256,
    normal_token_count: 128_000,
    special_token_count: 256,
    control_token_count: 248,
    user_defined_token_count: 8,
    bos_token_id: 128_000,
    eos_token_id: 128_012,
    padding_token_id: 128_012,
    declared_add_bos_token: 'absent',
    declared_add_eos_token: 'absent',
    declared_add_space_prefix: 'absent',
    oracle_resolved_add_bos_token: false,
    oracle_resolved_add_eos_token: false,
    chat_control_token_ids: { im_start: 128_011, im_end: 128_012 },
    chat_template_utf8_bytes: 5_493,
    chat_template_sha256: 'b9b66f04c64fbb8695cf5b35c37780efd0b8e0829fbfe3e30fafb9f469b7d30e',
  },
  camelid: structuredClone(receipt.camelid),
  oracle: {
    ...structuredClone(receipt.oracle),
    derivative: {
      ...structuredClone(receipt.oracle.derivative),
      original_tensor_count: 326,
      metadata_count: 26,
      patch_offset: 8,
      sha256: 'e'.repeat(64),
      persisted: false,
    },
  },
  cases: syntheticSmolCases,
  result: {
    case_count: syntheticSmolCases.length,
    exact_match_count: syntheticSmolCases.length,
    all_token_ids_match: true,
    support_decision: 'smollm3_exact_row_tokenizer_gate_only',
  },
}
assert.deepEqual(
  validateSmolLM3TokenizerReceipt(syntheticSmolReceipt, smolRow, roster.defaults),
  [],
  'SmolLM3 durable receipt validator accepts only the grounded exact-row pack',
)

const smolPrefixDrift = structuredClone(syntheticSmolReceipt)
smolPrefixDrift.bounded_fetch.prefix_sha256 = '0'.repeat(64)
assert(
  validateSmolLM3TokenizerReceipt(smolPrefixDrift, smolRow, roster.defaults)
    .some((error) => error.includes('grounded exact-row prefix')),
)
const smolLicenseDrift = structuredClone(syntheticSmolReceipt)
smolLicenseDrift.source.license = 'other'
assert(
  validateSmolLM3TokenizerReceipt(smolLicenseDrift, smolRow, roster.defaults)
    .some((error) => error.includes('source.license mismatch')),
)
const smolFalseBos = structuredClone(syntheticSmolReceipt)
smolFalseBos.cases[0].camelid_ids = [128_000]
smolFalseBos.cases[0].llama_cpp_ids = [128_000]
assert(
  validateSmolLM3TokenizerReceipt(smolFalseBos, smolRow, roster.defaults)
    .some((error) => error.includes('did not resolve false for empty input')),
)
const smolControlLeak = structuredClone(syntheticSmolReceipt)
const ordinaryControl = smolControlLeak.cases.find((testCase) => testCase.id === 'chat_controls_as_ordinary_text')
ordinaryControl.camelid_ids = [128_011, 128_012]
ordinaryControl.llama_cpp_ids = [128_011, 128_012]
assert(
  validateSmolLM3TokenizerReceipt(smolControlLeak, smolRow, roster.defaults)
    .some((error) => error.includes('parsed despite parse_special=false')),
)
const smolUserDefinedDrift = structuredClone(syntheticSmolReceipt)
for (const testCase of smolUserDefinedDrift.cases.filter((candidate) => candidate.id.startsWith('user_defined_tool_tags_'))) {
  testCase.camelid_ids = [77]
  testCase.llama_cpp_ids = [77]
}
assert(
  validateSmolLM3TokenizerReceipt(smolUserDefinedDrift, smolRow, roster.defaults)
    .some((error) => error.includes('lost exact boundary IDs')),
)

const smolOutOfRangeIds = structuredClone(syntheticSmolReceipt)
for (const testCase of smolOutOfRangeIds.cases.filter((candidate) => candidate.id.startsWith('plain_ascii_'))) {
  testCase.camelid_ids = [999_999]
  testCase.llama_cpp_ids = [999_999]
}
assert(
  validateSmolLM3TokenizerReceipt(smolOutOfRangeIds, smolRow, roster.defaults)
    .some((error) => error.includes('has invalid token IDs')),
  'paired token IDs outside the pinned vocabulary must fail closed',
)
const smolEmptyHelloIds = structuredClone(syntheticSmolReceipt)
for (const testCase of smolEmptyHelloIds.cases.filter((candidate) => candidate.id.startsWith('plain_ascii_'))) {
  testCase.camelid_ids = []
  testCase.llama_cpp_ids = []
}
assert(
  validateSmolLM3TokenizerReceipt(smolEmptyHelloIds, smolRow, roster.defaults)
    .some((error) => error.includes('did not preserve exact Hello token IDs')),
  'paired empty Hello outputs must not masquerade as absent-BOS parity',
)

const smolMissingEmptyIds = structuredClone(syntheticSmolReceipt)
delete smolMissingEmptyIds.cases.find(
  (testCase) => testCase.id === 'empty_with_add_special',
).camelid_ids
assert(
  validateSmolLM3TokenizerReceipt(smolMissingEmptyIds, smolRow, roster.defaults)
    .some((error) => error.includes('did not resolve false for empty input')),
  'missing empty-input IDs must fail closed without throwing',
)
const smolScalarParsedIds = structuredClone(syntheticSmolReceipt)
smolScalarParsedIds.cases.find(
  (testCase) => testCase.id === 'single_user_chat_controls',
).camelid_ids = 128_011
assert(
  validateSmolLM3TokenizerReceipt(smolScalarParsedIds, smolRow, roster.defaults)
    .some((error) => error.includes('were not parsed to their exact IDs')),
  'scalar parsed-control IDs must fail closed without throwing',
)
const smolMissingOrdinaryIds = structuredClone(syntheticSmolReceipt)
delete smolMissingOrdinaryIds.cases.find(
  (testCase) => testCase.id === 'chat_controls_as_ordinary_text',
).llama_cpp_ids
assert(
  validateSmolLM3TokenizerReceipt(smolMissingOrdinaryIds, smolRow, roster.defaults)
    .some((error) => error.includes('parsed despite parse_special=false')),
  'missing ordinary-control IDs must fail closed without throwing',
)
const smolNullCase = structuredClone(syntheticSmolReceipt)
smolNullCase.cases[4] = null
assert(
  validateSmolLM3TokenizerReceipt(smolNullCase, smolRow, roster.defaults)
    .some((error) => error.includes('id/order mismatch')),
  'null case entries must fail closed without throwing',
)

console.log('test-hf-qualification-tokenizer: all checks passed')
