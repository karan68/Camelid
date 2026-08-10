#!/usr/bin/env node
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile, stat } from 'node:fs/promises'
import {
  HeaderInspectionError,
  MAX_PREFIX_BYTES,
  classifyHeaderInspectionError,
  fetchHeaderPrefix,
  inspectBinaryIdentity,
  inspectPrefix,
  inspectRemoteHeader,
  normalizePrefixBytes,
  parseContentRange,
  summarizeInspection,
  validateImmutableLock,
} from './hf-qualification-header.mjs'

const revision = 'a'.repeat(40)
const lock = {
  schema: 'camelid.hf-source-lock/v1',
  repo: 'example/model',
  file: 'model.gguf',
  revision,
  size_bytes: 100,
  sha256: 'b'.repeat(64),
  license: 'apache-2.0',
  access: { gated: false, private: false, disabled: false },
  download_url: `https://huggingface.co/example/model/resolve/${revision}/model.gguf?download=true`,
}

const inspection = {
  version: 3,
  tensor_count: 2,
  metadata_count: 7,
  alignment: 32,
  data_start_offset: 96,
  metadata: {
    'general.architecture': 'qwen3moe',
    'general.file_type': 7,
    'tokenizer.ggml.model': 'gpt2',
    'tokenizer.ggml.pre': 'qwen2',
    'tokenizer.chat_template': 'private C:\\operator\\template.jinja test-secret-token',
    'upstream.private.note': '/Users/operator/model.gguf',
    'tokenizer.ggml.tokens': ['x'.repeat(70 * 1024)],
  },
  tensors: [
    { name: 'a', dimensions: [2, 3], tensor_type: 'Q8_0', relative_offset: 0, absolute_offset: 96, n_bytes: 6 },
    { name: 'b', dimensions: [1], tensor_type: 'F32', relative_offset: 32, absolute_offset: 128, n_bytes: 4 },
  ],
}
const sourceHead = `12345678${'9'.repeat(32)}`
const inspectorIdentity = {
  version: 'camelid v0.6.1-27-g12345678',
  binary_sha256: 'e'.repeat(64),
  source_head: sourceHead,
  binary_commit_abbrev: '12345678',
  binary_reports_dirty: false,
  binary_matches_source_head: true,
  clean_current_head: true,
}

assert.deepEqual(parseContentRange('bytes 0-15/100'), { start: 0, end: 15, total: 100 })
assert.throws(() => parseContentRange('0-15/100'), /invalid or missing Content-Range/)
assert.throws(() => parseContentRange('bytes 15-0/100'), /invalid or missing Content-Range/)
assert.throws(() => parseContentRange('bytes 0-100/100'), /invalid or missing Content-Range/)
assert.equal(normalizePrefixBytes('16'), 16)
assert.throws(() => normalizePrefixBytes(0), /prefix byte budget/)
assert.throws(() => normalizePrefixBytes(MAX_PREFIX_BYTES + 1), /prefix byte budget/)
assert.doesNotThrow(() => validateImmutableLock(lock))
assert.throws(
  () => validateImmutableLock({ ...lock, download_url: 'https://example.invalid/model.gguf' }),
  (error) => error instanceof HeaderInspectionError && error.code === 'source_lock_invalid',
)
assert.throws(
  () => validateImmutableLock({ ...lock, download_url: 'https://huggingface.co/example/model/resolve/main/model.gguf' }),
  (error) => error instanceof HeaderInspectionError && error.code === 'source_lock_invalid',
)
assert.throws(
  () => validateImmutableLock({
    ...lock,
    download_url: `https://huggingface.co/example/model/resolve/${revision}/different.gguf`,
  }),
  (error) => error instanceof HeaderInspectionError && error.code === 'source_lock_invalid',
)

const calls = []
const ranged = await fetchHeaderPrefix(lock, {
  prefixBytes: 16,
  token: 'test-secret-token',
  fetchImpl: async (url, options) => {
    calls.push({ url, options })
    return new Response(Buffer.alloc(16, 7), {
      status: 206,
      headers: { 'content-range': 'bytes 0-15/100' },
    })
  },
})
assert.equal(ranged.bytes.length, 16)
assert.equal(ranged.prefix_sha256, createHash('sha256').update(Buffer.alloc(16, 7)).digest('hex'))
assert.equal(calls[0].options.headers.Range, 'bytes=0-15')
assert.equal(calls[0].options.headers['Accept-Encoding'], 'identity')
assert.equal(calls[0].options.headers.Authorization, 'Bearer test-secret-token')
assert.equal(JSON.stringify(ranged).includes('test-secret-token'), false)

await assert.rejects(
  fetchHeaderPrefix(lock, {
    prefixBytes: 16,
    fetchImpl: async () => new Response(Buffer.alloc(100), { status: 200 }),
  }),
  (error) => error instanceof HeaderInspectionError
    && error.code === 'header_range_unavailable'
    && error.status === 'blocked',
)

await assert.rejects(
  fetchHeaderPrefix(lock, {
    prefixBytes: 16,
    fetchImpl: async () => new Response(Buffer.alloc(8), {
      status: 206,
      headers: { 'content-range': 'bytes 0-7/100' },
    }),
  }),
  (error) => error instanceof HeaderInspectionError
    && error.code === 'header_range_incomplete'
    && error.status === 'blocked',
)

await assert.rejects(
  fetchHeaderPrefix(lock, {
    prefixBytes: 16,
    fetchImpl: async () => new Response(Buffer.alloc(16), {
      status: 206,
      headers: { 'content-range': 'bytes 0-15/101' },
    }),
  }),
  (error) => error instanceof HeaderInspectionError
    && error.code === 'header_range_identity_mismatch'
    && error.status === 'fail',
)

await assert.rejects(
  fetchHeaderPrefix(lock, {
    prefixBytes: 16,
    fetchImpl: async () => new Response(Buffer.alloc(17), {
      status: 206,
      headers: { 'content-range': 'bytes 0-16/100' },
    }),
  }),
  (error) => error instanceof HeaderInspectionError
    && error.code === 'header_body_budget_exceeded'
    && error.status === 'blocked',
)

await assert.rejects(
  fetchHeaderPrefix(lock, {
    prefixBytes: 16,
    fetchImpl: async () => new Response(Buffer.alloc(15), {
      status: 206,
      headers: { 'content-range': 'bytes 0-15/100' },
    }),
  }),
  (error) => error instanceof HeaderInspectionError
    && error.code === 'header_body_length_mismatch'
    && error.status === 'fail',
)

let fetchError
try {
  await fetchHeaderPrefix(lock, {
    prefixBytes: 16,
    fetchImpl: async () => { throw new Error('test-secret-token at C:\\private\\model.gguf') },
  })
} catch (error) {
  fetchError = classifyHeaderInspectionError(error)
}
assert.equal(fetchError.error_code, 'header_fetch_unavailable')
assert.equal(JSON.stringify(fetchError).includes('test-secret-token'), false)
assert.equal(JSON.stringify(fetchError).includes('private'), false)
const injectedDetailsFailure = classifyHeaderInspectionError(new HeaderInspectionError(
  'header_fetch_unavailable',
  'blocked',
  'remote header fetch is blocked',
  { status: 'pass', reason: 'C:\\private\\model.gguf test-secret-token' },
))
assert.deepEqual(injectedDetailsFailure, {
  status: 'blocked',
  error_code: 'header_fetch_unavailable',
  reason: 'remote header fetch is blocked',
})
const forgedTypedFailure = classifyHeaderInspectionError(new HeaderInspectionError(
  'forged',
  'pass',
  'C:\\private\\model.gguf bearer-token',
))
assert.deepEqual(forgedTypedFailure, {
  status: 'blocked',
  error_code: 'header_inspection_error',
  reason: 'remote header inspection could not complete',
})
assert.equal(JSON.stringify(forgedTypedFailure).includes('private'), false)
assert.equal(JSON.stringify(forgedTypedFailure).includes('bearer-token'), false)
const mutatedTypedError = new HeaderInspectionError(
  'header_fetch_unavailable',
  'blocked',
  'ignored',
)
mutatedTypedError.code = 'forged'
mutatedTypedError.status = 'pass'
mutatedTypedError.message = 'C:\\private\\model.gguf bearer-token'
const mutatedTypedFailure = classifyHeaderInspectionError(mutatedTypedError)
assert.deepEqual(mutatedTypedFailure, {
  status: 'blocked',
  error_code: 'header_inspection_error',
  reason: 'remote header inspection could not complete',
})
assert.equal(JSON.stringify(mutatedTypedFailure).includes('private'), false)
assert.equal(JSON.stringify(mutatedTypedFailure).includes('bearer-token'), false)

const summary = summarizeInspection(inspection)
assert.deepEqual(summary.observed, {
  architecture: 'qwen3moe',
  general_file_type: 7,
  tokenizer_model: 'gpt2',
  tokenizer_pre: 'qwen2',
  headline_quant: 'Q8_0',
})
assert.equal(summary.tensor_inventory.types.Q8_0, 1)
assert.equal(summary.tensor_inventory.total_n_bytes, 10)
assert.match(summary.tensor_inventory.sha256, /^[0-9a-f]{64}$/)
assert.equal(Object.hasOwn(summary.tensor_inventory, 'tensors'), false)
assert.equal(Object.hasOwn(summary, 'metadata'), false)
assert.equal(JSON.stringify(summary).includes('absolute_offset'), false)
assert.equal(JSON.stringify(summary).includes('test-secret-token'), false)
assert.equal(JSON.stringify(summary).includes('operator'), false)
const mixedCaseTensorTypeInspection = structuredClone(inspection)
mixedCaseTensorTypeInspection.tensors[0].tensor_type = 'Tq1_0'
const mixedCaseTensorTypeSummary = summarizeInspection(mixedCaseTensorTypeInspection)
assert.equal(mixedCaseTensorTypeSummary.observed.headline_quant, 'Tq1_0')
assert.equal(mixedCaseTensorTypeSummary.tensor_inventory.types.Tq1_0, 1)
const maliciousObservedInspection = structuredClone(inspection)
maliciousObservedInspection.metadata['general.architecture'] = 'C:\\private\\model.gguf test-secret-token'
maliciousObservedInspection.metadata['tokenizer.ggml.model'] = '/Users/operator/tokenizer.json'
const maliciousObservedSummary = summarizeInspection(maliciousObservedInspection)
assert.equal(maliciousObservedSummary.observed.architecture, '<invalid>')
assert.equal(maliciousObservedSummary.observed.tokenizer_model, '<invalid>')
assert.equal(JSON.stringify(maliciousObservedSummary).includes('test-secret-token'), false)
assert.equal(JSON.stringify(maliciousObservedSummary).includes('operator'), false)
assert.throws(
  () => summarizeInspection({ ...inspection, tensor_count: 3 }),
  (error) => error instanceof HeaderInspectionError && error.code === 'header_inspection_contract_invalid',
)

const parsedInspection = await inspectPrefix('<camelid>', '<prefix>', 100, {
  execImpl: async () => ({ stdout: JSON.stringify(inspection) }),
})
assert.equal(parsedInspection.tensor_count, 2)
const inspectedIdentity = await inspectBinaryIdentity('<camelid>', {
  execImpl: async (binary, args) => {
    assert.equal(binary, '<camelid>')
    assert.deepEqual(args, ['--version'])
    return { stdout: 'camelid v0.6.1-27-g12345678\n' }
  },
  hashImpl: async (binary) => {
    assert.equal(binary, '<camelid>')
    return 'e'.repeat(64)
  },
  gitImpl: async (binary, args, options) => {
    assert.equal(binary, 'git')
    assert.deepEqual(args, ['rev-parse', 'HEAD'])
    assert.equal(options.cwd, '<source-root>')
    return { stdout: `${sourceHead}\n` }
  },
  sourceRoot: '<source-root>',
})
assert.deepEqual(inspectedIdentity, inspectorIdentity)

const dirtyInspectedIdentity = await inspectBinaryIdentity('<camelid>', {
  execImpl: async () => ({ stdout: 'camelid v0.6.1-27-g12345678-dirty\n' }),
  hashImpl: async () => 'e'.repeat(64),
  gitImpl: async () => ({ stdout: `${sourceHead}\n` }),
})
assert.equal(dirtyInspectedIdentity.binary_reports_dirty, true)
assert.equal(dirtyInspectedIdentity.binary_matches_source_head, true)
assert.equal(dirtyInspectedIdentity.clean_current_head, false)

const staleInspectedIdentity = await inspectBinaryIdentity('<camelid>', {
  execImpl: async () => ({ stdout: 'camelid v0.6.1-27-gabcdef12\n' }),
  hashImpl: async () => 'e'.repeat(64),
  gitImpl: async () => ({ stdout: `${sourceHead}\n` }),
})
assert.equal(staleInspectedIdentity.binary_reports_dirty, false)
assert.equal(staleInspectedIdentity.binary_matches_source_head, false)
assert.equal(staleInspectedIdentity.clean_current_head, false)
await assert.rejects(
  inspectPrefix(undefined, '<prefix>', 100, {
    execImpl: async () => { throw new Error('must not execute') },
  }),
  (error) => error instanceof HeaderInspectionError
    && error.code === 'header_inspector_unavailable'
    && error.status === 'blocked',
)
await assert.rejects(
  inspectPrefix('<camelid>', '<prefix>', 100, {
    execImpl: async () => { throw Object.assign(new Error('C:\\private\\camelid.exe test-secret-token'), { code: 'ENOENT' }) },
  }),
  (error) => error instanceof HeaderInspectionError
    && error.code === 'header_inspector_unavailable'
    && !error.message.includes('private')
    && !error.message.includes('test-secret-token'),
)

let temporaryPrefix
const remoteReceipt = await inspectRemoteHeader(lock, {
  binary: '<camelid>',
  rowId: 'fixture_row',
  prefixBytes: 16,
  token: 'test-secret-token',
  sourceRoot: '<source-root>',
  identityImpl: async (binary, options) => {
    assert.equal(binary, '<camelid>')
    assert.equal(options.sourceRoot, '<source-root>')
    return inspectorIdentity
  },
  now: () => new Date('2026-08-10T12:34:56.000Z'),
  fetchImpl: async () => new Response(Buffer.alloc(16, 11), {
    status: 206,
    headers: { 'content-range': 'bytes 0-15/100' },
  }),
  inspectImpl: async (binary, prefixPath, declaredLength) => {
    temporaryPrefix = prefixPath
    assert.equal(binary, '<camelid>')
    assert.equal(declaredLength, 100)
    assert.deepEqual(await readFile(prefixPath), Buffer.alloc(16, 11))
    return inspection
  },
})
assert.equal(remoteReceipt.range.prefix_sha256, createHash('sha256').update(Buffer.alloc(16, 11)).digest('hex'))
assert.equal(remoteReceipt.row_id, 'fixture_row')
assert.equal(remoteReceipt.generated_at, '2026-08-10T12:34:56.000Z')
assert.equal(remoteReceipt.host.hostname_redacted, true)
assert.equal(Object.hasOwn(remoteReceipt.host, 'hostname'), false)
assert.deepEqual(remoteReceipt.inspector, {
  ...inspectorIdentity,
  binary_path_redacted: true,
  command: ['<camelid>', 'inspect-prefix', '<remote-gguf-prefix>', '--declared-len', '100'],
})
assert.equal(remoteReceipt.scope.prefix_sha256, 'all_received_prefix_bytes')
assert.equal(remoteReceipt.scope.tensor_payload, 'partially_range_fetched_opaque')
assert.equal(remoteReceipt.scope.full_artifact_sha256, 'not_run')
assert.equal(remoteReceipt.scope.tensor_payload_interpretation, 'not_run')
assert.equal(remoteReceipt.support_claim, false)
assert.match(remoteReceipt.note, /opaque initial tensor payload/)
assert.match(remoteReceipt.note, /prefix SHA-256 covers every received byte/)
assert.match(remoteReceipt.note, /loaded as model weights/)
assert.doesNotMatch(remoteReceipt.note, /does not hash tensor payload bytes/)
assert.equal(Object.hasOwn(remoteReceipt.source, 'download_url'), false)
assert.equal(JSON.stringify(remoteReceipt).includes('test-secret-token'), false)
assert.equal(JSON.stringify(remoteReceipt).includes('operator'), false)
await assert.rejects(stat(temporaryPrefix), (error) => error.code === 'ENOENT')

let missingBinaryFetched = false
await assert.rejects(
  inspectRemoteHeader(lock, {
    binary: undefined,
    prefixBytes: 16,
    fetchImpl: async () => {
      missingBinaryFetched = true
      throw new Error('must not fetch')
    },
  }),
  (error) => error instanceof HeaderInspectionError
    && error.code === 'header_inspector_unavailable'
    && error.status === 'blocked',
)
assert.equal(missingBinaryFetched, false, 'a missing inspector binary must block before any range request')

for (const identity of [
  {
    ...inspectorIdentity,
    version: 'camelid v0.6.1-27-g12345678-dirty',
    binary_reports_dirty: true,
    clean_current_head: false,
  },
  {
    ...inspectorIdentity,
    version: 'camelid v0.6.1-27-gabcdef12',
    binary_commit_abbrev: 'abcdef12',
    binary_matches_source_head: false,
    clean_current_head: false,
  },
]) {
  let uncleanIdentityFetched = false
  await assert.rejects(
    inspectRemoteHeader(lock, {
      binary: '<camelid>',
      prefixBytes: 16,
      identityImpl: async () => identity,
      fetchImpl: async () => {
        uncleanIdentityFetched = true
        throw new Error('must not fetch')
      },
    }),
    (error) => error instanceof HeaderInspectionError
      && error.code === 'header_inspector_not_clean_current_head'
      && error.status === 'blocked',
  )
  assert.equal(uncleanIdentityFetched, false, 'dirty or stale inspector identity must block before range fetch')
}

let forgedIdentityFetched = false
await assert.rejects(
  inspectRemoteHeader(lock, {
    binary: '<camelid>',
    prefixBytes: 16,
    identityImpl: async () => ({
      ...inspectorIdentity,
      binary_commit_abbrev: 'forged',
    }),
    fetchImpl: async () => {
      forgedIdentityFetched = true
      throw new Error('must not fetch')
    },
  }),
  (error) => error instanceof HeaderInspectionError
    && error.code === 'header_inspector_identity_invalid'
    && error.status === 'fail',
)
assert.equal(forgedIdentityFetched, false, 'inconsistent inspector fields must fail before range fetch')

let failedTemporaryPrefix
let inspectionFailure
try {
  await inspectRemoteHeader(lock, {
    binary: '<camelid>',
    prefixBytes: 16,
    identityImpl: async () => inspectorIdentity,
    fetchImpl: async () => new Response(Buffer.alloc(16), {
      status: 206,
      headers: { 'content-range': 'bytes 0-15/100' },
    }),
    inspectImpl: async (_binary, prefixPath) => {
      failedTemporaryPrefix = prefixPath
      throw new Error('test-secret-token at C:\\private\\header.gguf')
    },
  })
} catch (error) {
  inspectionFailure = classifyHeaderInspectionError(error)
}
assert.equal(inspectionFailure.error_code, 'header_inspection_error')
assert.equal(JSON.stringify(inspectionFailure).includes('test-secret-token'), false)
assert.equal(JSON.stringify(inspectionFailure).includes('private'), false)
await assert.rejects(stat(failedTemporaryPrefix), (error) => error.code === 'ENOENT')

const unknownFailure = classifyHeaderInspectionError(new Error('C:\\secret\\path test-secret-token'))
assert.equal(unknownFailure.error_code, 'header_inspection_error')
assert.equal(JSON.stringify(unknownFailure).includes('secret'), false)

console.log('test-hf-qualification-header: all checks passed')
