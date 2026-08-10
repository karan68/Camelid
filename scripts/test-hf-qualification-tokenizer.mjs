#!/usr/bin/env node

import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import {
  GEMMA2_CASES,
  SMOLLM3_CASES,
  assertGemma2TokenizerMetadata,
  assertSmolLM3TokenizerMetadata,
  buildCamelidArgs,
  buildLlamaArgs,
  classifyCamelidProvenance,
  makeVocabOnlyGguf,
  normalizeTokenizerPrefixBytes,
  parseLlamaIds,
  parseLlamaVersionOutput,
  sourceSelectionForRow,
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

const forgedIds = structuredClone(receipt)
forgedIds.cases[0].llama_cpp_ids = [999]
forgedIds.cases[0].exact_match = true
assert(
  validateGemma2TokenizerReceipt(forgedIds, receiptRow, roster.defaults)
    .some((error) => error.includes('token IDs diverge')),
  'receipt-authored exact_match must not hide divergent IDs',
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
