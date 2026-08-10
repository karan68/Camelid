#!/usr/bin/env node
import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import { validateQualificationReport } from './check-model-qualification-report.mjs'
import { HeaderInspectionError, MAX_PREFIX_BYTES } from './hf-qualification-header.mjs'
import {
  artifactForRow,
  defaultCamelidBinary,
  firstUnresolvedStage,
  metadataStageFromHeader,
  metadataStageFromHeaderError,
  publicRosterLabel,
  resolveSourcePreflight,
  resolveSourceStage,
  runFactory,
  selectRows,
  sourceLookupErrorCode,
  sourceSelectionForRow,
  summarizeHeaderInspections,
  summarizeSourceResolution,
  summarizeReports,
} from './model-qualification-factory.mjs'

const execFileAsync = promisify(execFile)
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const currentSourceHead = (await execFileAsync('git', ['rev-parse', 'HEAD'], {
  cwd: root,
  windowsHide: true,
})).stdout.trim()
const currentCommitAbbrev = currentSourceHead.slice(0, 8)
assert.equal(
  defaultCamelidBinary(),
  process.platform === 'win32' ? 'target/release/camelid.exe' : 'target/release/camelid',
)
const roster = JSON.parse(await readFile(resolve(root, 'qa/model-qualification/phase1-roster.json'), 'utf8'))
const qwen = roster.rows.find((row) => row.id === 'qwen2_5_0_5b_instruct_q8_0')
const qwenMoe = roster.rows.find((row) => row.id === 'qwen3_30b_a3b_q8_0')

assert.deepEqual(
  selectRows(roster, ['qwen3_30b_a3b_q8_0', 'lfm2_5_2_6b_q8_0']).map((row) => row.id),
  ['lfm2_5_2_6b_q8_0', 'qwen3_30b_a3b_q8_0'],
  'selected rows must preserve qualification priority',
)
assert.throws(() => selectRows(roster, ['not_a_row']), /unknown qualification rows/)
assert.equal(
  artifactForRow(qwen, resolve(root, 'models')),
  resolve(root, 'models', qwen.identity.gguf_filename),
)
assert.equal(
  artifactForRow(qwenMoe, resolve(root, 'models')),
  resolve(root, 'models', qwenMoe.identity.gguf_filename),
  'newly anchored Qwen3 MoE must resolve its official exact filename',
)
const unanchored = structuredClone(qwenMoe)
unanchored.identity.gguf_filename = null
unanchored.source.file = null
assert.equal(artifactForRow(unanchored, resolve(root, 'models')), null, 'unanchored rows must not invent an artifact filename')
assert.equal(artifactForRow(qwenMoe, null, resolve(root, 'manual.gguf')), resolve(root, 'manual.gguf'))
assert.equal(
  publicRosterLabel(root, resolve(root, 'qa/model-qualification/phase1-roster.json')),
  'qa/model-qualification/phase1-roster.json',
)
assert.equal(
  publicRosterLabel(root, resolve(root, '..', 'private', 'secret-roster.json')),
  '<external-roster>',
  'a scrubbed factory index must not copy an external absolute roster path',
)

const selectedSource = sourceSelectionForRow(qwen)
assert.deepEqual(selectedSource, {
  row_id: qwen.id,
  repo: qwen.source.repo,
  file: qwen.source.file,
  revision: qwen.source.revision,
  expected: {
    size_bytes: qwen.identity.size_bytes,
    sha256: qwen.identity.sha256,
    license: qwen.source.license,
  },
})
const lockFor = (row) => ({
  schema: 'camelid.hf-source-lock/v1',
  repo: row.source.repo,
  file: row.source.file,
  revision: row.source.revision,
  size_bytes: row.identity.size_bytes,
  sha256: row.identity.sha256,
  license: row.source.license,
  access: { gated: false, private: false, disabled: false },
  download_url: `https://huggingface.co/${row.source.repo}/resolve/${row.source.revision}/${row.source.file}?download=true`,
})
const headerReceiptFor = (row, overrides = {}) => ({
  schema: 'camelid.remote-gguf-header-inspection/v1',
  row_id: row.id,
  generated_at: '2026-08-10T12:34:56.000Z',
  host: {
    hostname_redacted: true,
    platform: 'win32',
    release: '10.0.26200',
    arch: 'x64',
  },
  inspector: {
    version: `camelid v0.6.1-27-g${currentCommitAbbrev}`,
    binary_sha256: 'e'.repeat(64),
    source_head: currentSourceHead,
    binary_commit_abbrev: currentCommitAbbrev,
    binary_reports_dirty: false,
    binary_matches_source_head: true,
    clean_current_head: true,
    binary_path_redacted: true,
    command: [
      '<camelid>',
      'inspect-prefix',
      '<remote-gguf-prefix>',
      '--declared-len',
      String(row.identity.size_bytes),
    ],
  },
  source: {
    repo: row.source.repo,
    file: row.source.file,
    revision: row.source.revision,
    size_bytes: row.identity.size_bytes,
    sha256: row.identity.sha256,
  },
  scope: {
    prefix_sha256: 'all_received_prefix_bytes',
    tensor_payload: 'partially_range_fetched_opaque',
    full_artifact_sha256: 'not_run',
    tensor_payload_interpretation: 'not_run',
    load: 'not_run',
    generation: 'not_run',
  },
  range: {
    requested_bytes: 16,
    received_bytes: 16,
    content_range: { start: 0, end: 15, total: row.identity.size_bytes },
    prefix_sha256: 'c'.repeat(64),
  },
  inspection: {
    version: 3,
    tensor_count: 5,
    metadata_count: 5,
    alignment: 32,
    data_start_offset: 4096,
    observed: {
      architecture: row.identity.architecture,
      general_file_type: 7,
      tokenizer_model: row.expected.tokenizer_model,
      tokenizer_pre: row.expected.tokenizer_pre,
      headline_quant: row.identity.quantization,
    },
    tensor_inventory: {
      sha256: 'd'.repeat(64),
      total_n_bytes: row.identity.size_bytes - 4096,
      types: { [row.identity.quantization]: 4, F32: 1 },
    },
  },
  support_claim: false,
  note: 'test receipt',
  ...overrides,
})

for (const [rowId, receiptPath] of [
  ['gemma2_9b_it_q8_0', 'qa/model-qualification/gemma2-9b-it-q8-header-inspection.json'],
  ['smollm3_3b_q8_0', 'qa/model-qualification/smollm3-3b-q8-header-inspection.json'],
]) {
  const durableRow = roster.rows.find((row) => row.id === rowId)
  assert.ok(durableRow, `committed header receipt row ${rowId} must remain in the roster`)
  const durableReceipt = JSON.parse(await readFile(resolve(root, receiptPath), 'utf8'))
  const durableStage = metadataStageFromHeader(durableRow, durableReceipt)
  assert.equal(
    durableStage.status,
    'pass',
    `${receiptPath} must remain a valid clean-head exact-row remote-header receipt`,
  )
}

let resolverCalls = 0
const resolvedStage = await resolveSourceStage(qwen, {
  token: 'test-token-not-recorded',
  resolver: async (selection) => {
    resolverCalls += 1
    assert.equal(selection.token, 'test-token-not-recorded')
    return lockFor(qwen)
  },
})
assert.equal(resolverCalls, 1)
assert.equal(resolvedStage.status, 'pass')
assert.equal(resolvedStage.resolution, 'live_huggingface')
assert.equal(JSON.stringify(resolvedStage).includes('download_url'), false, 'reports must omit signed/public download URLs')
assert.equal(JSON.stringify(resolvedStage).includes('test-token'), false, 'reports must never include the HF token')

const retainedPreflight = await resolveSourcePreflight(qwen, {
  token: 'test-token-not-recorded',
  resolver: async () => lockFor(qwen),
})
assert.equal(retainedPreflight.stage.status, 'pass')
assert.equal(retainedPreflight.lock.download_url, lockFor(qwen).download_url, 'the raw lock remains internal for the header fetch')
assert.equal(JSON.stringify(retainedPreflight.stage).includes('download_url'), false, 'the serialized source stage must still omit the URL')

const passingHeaderStage = metadataStageFromHeader(qwen, headerReceiptFor(qwen))
assert.equal(passingHeaderStage.status, 'pass')
assert.equal(passingHeaderStage.mode, 'remote_immutable_prefix')
assert.equal(passingHeaderStage.observed.architecture, qwen.identity.architecture)
assert.equal(passingHeaderStage.observed.tensor_inventory_sha256, 'd'.repeat(64))
assert.equal(passingHeaderStage.inspection_generated_at, '2026-08-10T12:34:56.000Z')
assert.equal(passingHeaderStage.host.hostname_redacted, true)
assert.equal(Object.hasOwn(passingHeaderStage.host, 'hostname'), false)
assert.equal(passingHeaderStage.inspector.version, `camelid v0.6.1-27-g${currentCommitAbbrev}`)
assert.equal(passingHeaderStage.inspector.binary_sha256, 'e'.repeat(64))
assert.equal(passingHeaderStage.inspector.source_head, currentSourceHead)
assert.equal(passingHeaderStage.inspector.binary_commit_abbrev, currentCommitAbbrev)
assert.equal(passingHeaderStage.inspector.binary_reports_dirty, false)
assert.equal(passingHeaderStage.inspector.binary_matches_source_head, true)
assert.equal(passingHeaderStage.inspector.clean_current_head, true)
assert.equal(passingHeaderStage.scope.prefix_sha256, 'all_received_prefix_bytes')
assert.equal(passingHeaderStage.scope.full_artifact_sha256, 'not_run')
assert.equal(passingHeaderStage.scope.tensor_payload, 'partially_range_fetched_opaque')
assert.equal(passingHeaderStage.scope.opaque_tensor_payload_prefix_bytes, 0)
assert.equal(passingHeaderStage.scope.tensor_payload_interpretation, 'not_run')
assert.equal(passingHeaderStage.scope.support_claim, false)
assert.equal(Object.hasOwn(passingHeaderStage, 'note'), false, 'arbitrary receipt prose must not enter the report stage')
assert.equal(Object.hasOwn(passingHeaderStage.observed, 'types'), false, 'the report stage remains compact')

const maliciousHeaderReceipt = headerReceiptFor(qwen)
maliciousHeaderReceipt.inspection.observed.architecture = 'C:\\private\\model.gguf test-secret-token'
const mismatchedHeaderStage = metadataStageFromHeader(qwen, maliciousHeaderReceipt)
assert.equal(mismatchedHeaderStage.status, 'fail')
assert.equal(mismatchedHeaderStage.observed.architecture, '<invalid>')
assert.equal(JSON.stringify(mismatchedHeaderStage).includes('test-secret-token'), false)
assert.equal(JSON.stringify(mismatchedHeaderStage).includes('private'), false)

const identityDriftReceipt = headerReceiptFor(qwen)
identityDriftReceipt.source.sha256 = '0'.repeat(64)
const identityDriftHeaderStage = metadataStageFromHeader(qwen, identityDriftReceipt)
assert.equal(identityDriftHeaderStage.status, 'fail')
assert.equal(identityDriftHeaderStage.error_code, 'header_source_identity_mismatch')

const missingInspectorReceipt = headerReceiptFor(qwen)
delete missingInspectorReceipt.inspector
const missingInspectorStage = metadataStageFromHeader(qwen, missingInspectorReceipt)
assert.equal(missingInspectorStage.status, 'fail')
assert.equal(missingInspectorStage.error_code, 'header_receipt_invalid')

const maliciousInspectorReceipt = headerReceiptFor(qwen)
maliciousInspectorReceipt.inspector.version = 'C:\\private\\camelid.exe test-secret-token'
const maliciousInspectorStage = metadataStageFromHeader(qwen, maliciousInspectorReceipt)
assert.equal(maliciousInspectorStage.status, 'fail')
assert.equal(JSON.stringify(maliciousInspectorStage).includes('test-secret-token'), false)
assert.equal(JSON.stringify(maliciousInspectorStage).includes('private'), false)

const dirtyInspectorReceipt = headerReceiptFor(qwen)
dirtyInspectorReceipt.inspector.version += '-dirty'
dirtyInspectorReceipt.inspector.binary_reports_dirty = true
dirtyInspectorReceipt.inspector.clean_current_head = false
const dirtyInspectorStage = metadataStageFromHeader(qwen, dirtyInspectorReceipt)
assert.equal(dirtyInspectorStage.status, 'fail')
assert.equal(dirtyInspectorStage.error_code, 'header_receipt_invalid')

const staleInspectorReceipt = headerReceiptFor(qwen)
staleInspectorReceipt.inspector.version = 'camelid v0.6.1-27-gabcdef12'
staleInspectorReceipt.inspector.binary_commit_abbrev = 'abcdef12'
staleInspectorReceipt.inspector.binary_matches_source_head = false
staleInspectorReceipt.inspector.clean_current_head = false
const staleInspectorStage = metadataStageFromHeader(qwen, staleInspectorReceipt)
assert.equal(staleInspectorStage.status, 'fail')
assert.equal(staleInspectorStage.error_code, 'header_receipt_invalid')

const forgedInspectorReceipt = headerReceiptFor(qwen)
forgedInspectorReceipt.inspector.binary_reports_dirty = true
const forgedInspectorStage = metadataStageFromHeader(qwen, forgedInspectorReceipt)
assert.equal(forgedInspectorStage.status, 'fail')
assert.equal(forgedInspectorStage.error_code, 'header_receipt_invalid')

const forgedSourceHeadReceipt = headerReceiptFor(qwen)
forgedSourceHeadReceipt.inspector.source_head = `${currentCommitAbbrev}${'f'.repeat(32)}`
const forgedSourceHeadStage = metadataStageFromHeader(qwen, forgedSourceHeadReceipt, {
  expectedSourceHead: currentSourceHead,
})
assert.equal(forgedSourceHeadStage.status, 'fail')
assert.equal(forgedSourceHeadStage.error_code, 'header_receipt_invalid')

const typedHeaderFailure = metadataStageFromHeaderError(new HeaderInspectionError(
  'header_body_budget_exceeded',
  'blocked',
  'range body exceeded the 16-byte request budget',
))
assert.deepEqual(typedHeaderFailure, {
  status: 'blocked',
  mode: 'remote_immutable_prefix',
  error_code: 'header_body_budget_exceeded',
  reason: 'remote header body exceeded its byte budget',
})
const scrubbedUnknownHeaderFailure = metadataStageFromHeaderError(
  new Error('test-secret-token at C:\\private\\header.gguf'),
)
assert.equal(scrubbedUnknownHeaderFailure.error_code, 'header_inspection_error')
assert.equal(JSON.stringify(scrubbedUnknownHeaderFailure).includes('test-secret-token'), false)
assert.equal(JSON.stringify(scrubbedUnknownHeaderFailure).includes('private'), false)

const driftStage = await resolveSourceStage(qwen, {
  resolver: async () => ({ ...lockFor(qwen), sha256: '0'.repeat(64) }),
})
assert.equal(driftStage.status, 'fail', 'an exact-byte identity drift is a hard failure')
assert.match(driftStage.reason, /does not match roster row/)

const maliciousDriftStage = await resolveSourceStage(qwen, {
  resolver: async () => ({
    ...lockFor(qwen),
    license: 'other C:\\private\\operator\\secret.txt bearer-token',
  }),
})
assert.equal(maliciousDriftStage.status, 'fail')
assert.match(maliciousDriftStage.reason, /license/)
assert.equal(maliciousDriftStage.expected.license, qwen.source.license)
assert.equal(JSON.stringify(maliciousDriftStage).includes('bearer-token'), false)
assert.equal(JSON.stringify(maliciousDriftStage).includes('operator'), false)

const offlineStage = await resolveSourceStage(qwen, {
  token: 'test-secret-token',
  resolver: async () => {
    throw new Error('ENOENT C:\\unrelated-private\\operator\\model.gguf using test-secret-token')
  },
})
assert.equal(offlineStage.status, 'blocked', 'network/infrastructure failure must block rather than pass')
assert.equal(offlineStage.error_code, 'source_lookup_error')
assert.equal(offlineStage.reason, 'live Hugging Face source resolution could not complete (source_lookup_error)')
assert.equal(JSON.stringify(offlineStage).includes('test-secret-token'), false, 'resolver errors must scrub the HF token')
assert.equal(JSON.stringify(offlineStage).includes('unrelated-private'), false, 'resolver errors must never copy arbitrary local paths')
assert.equal(sourceLookupErrorCode(Object.assign(new Error('socket stalled'), { code: 'ETIMEDOUT' })), 'timeout')
assert.equal(sourceLookupErrorCode(new Error('Hugging Face model-info request failed (404 Not Found)')), 'http_404')

const incomplete = structuredClone(qwen)
incomplete.identity.sha256 = null
let incompleteResolverCalled = false
const incompleteStage = await resolveSourceStage(incomplete, {
  resolver: async () => {
    incompleteResolverCalled = true
    return lockFor(qwen)
  },
})
assert.equal(incompleteStage.status, 'blocked')
assert.match(incompleteStage.reason, /sha256/)
assert.equal(incompleteResolverCalled, false, 'an incomplete roster identity must fail closed before network access')

const blockedReport = {
  overall_status: 'blocked',
  stages: {
    artifact: { status: 'pass' },
    source: { status: 'pass' },
    metadata: { status: 'pass' },
    tokenizer: { status: 'blocked' },
  },
}
assert.equal(firstUnresolvedStage(blockedReport, roster.gate_order), 'tokenizer')
assert.equal(
  firstUnresolvedStage({ overall_status: 'blocked', stages: { artifact: { status: 'blocked' } } }, roster.gate_order),
  'artifact',
)
assert.deepEqual(
  summarizeReports([{ row: qwen, report: blockedReport, reportFile: 'qwen.json' }], roster.gate_order),
  {
    counts: { pass: 0, fail: 0, blocked: 1 },
    rows: [{
      priority: qwen.priority,
      row_id: qwen.id,
      disposition: qwen.disposition,
      overall_status: 'blocked',
      first_unresolved_stage: 'tokenizer',
      report_file: 'qwen.json',
    }],
  },
)

assert.deepEqual(
  summarizeSourceResolution([
    { report: { stages: { source: { status: 'pass' } } } },
    { report: { stages: { source: { status: 'blocked' } } } },
    { report: { stages: { source: { status: 'fail' } } } },
  ]),
  { mode: 'live_huggingface', counts: { pass: 1, fail: 1, blocked: 1 } },
)

assert.deepEqual(
  summarizeHeaderInspections([
    { headerOutcome: passingHeaderStage },
    { headerOutcome: typedHeaderFailure },
    { headerOutcome: { status: 'not_run' } },
  ], 16),
  {
    mode: 'remote_immutable_prefix',
    per_row_byte_budget: 16,
    verified_receipt_requested_bytes: 16,
    verified_receipt_received_bytes: 16,
    counts: { pass: 1, fail: 0, blocked: 1, not_run: 1 },
  },
)

const factoryOut = await mkdtemp(join(tmpdir(), 'camelid-qualification-factory-test-'))
try {
  let offlineResolverCalled = false
  let offlineHeaderInspectorCalled = false
  const offlineIndex = await runFactory({
    root,
    rows: [qwen.id],
    outDir: join(factoryOut, 'offline-default'),
    sourceResolver: async () => {
      offlineResolverCalled = true
      throw new Error('must not run')
    },
    headerInspector: async () => {
      offlineHeaderInspectorCalled = true
      throw new Error('must not run')
    },
  })
  assert.equal(offlineResolverCalled, false, 'the default factory mode must remain fully offline')
  assert.equal(offlineHeaderInspectorCalled, false, 'the default factory mode must not inspect remote headers')
  assert.equal(Object.hasOwn(offlineIndex, 'source_resolution'), false, 'default index shape must remain unchanged')
  assert.equal(Object.hasOwn(offlineIndex, 'header_inspection'), false, 'default index shape must not gain a header summary')

  const requested = [qwen.id, qwenMoe.id]
  const index = await runFactory({
    root,
    rows: requested,
    outDir: factoryOut,
    resolveSource: true,
    sourceResolver: async ({ repo }) => {
      if (repo === qwen.source.repo) {
        throw new Error(`injected row-local outage under C:\\outside-workspace\\private.gguf using ${process.env.HF_TOKEN || 'no-token'}`)
      }
      return lockFor(qwenMoe)
    },
  })
  assert.deepEqual(index.source_resolution, {
    mode: 'live_huggingface',
    counts: { pass: 1, fail: 0, blocked: 1 },
  })
  assert.equal(index.rows.length, 2, 'one failed source lookup must not abort the remaining batch')
  assert.equal(index.rows.find((row) => row.row_id === qwen.id).first_unresolved_stage, 'source')
  const blockedReport = JSON.parse(await readFile(join(factoryOut, `${qwen.id}.json`), 'utf8'))
  assert.equal(blockedReport.stages.source.status, 'blocked')
  assert.equal(blockedReport.stages.source.error_code, 'source_lookup_error')
  assert.equal(blockedReport.overall_status, 'blocked')
  assert.equal(JSON.stringify(blockedReport).includes(root), false, 'factory reports must scrub the workspace path')
  assert.equal(JSON.stringify(blockedReport).includes('outside-workspace'), false, 'factory reports must not retain arbitrary resolver paths')
  assert.deepEqual(validateQualificationReport(blockedReport), [], 'live-source reports must satisfy the scrubbed report contract')
  const continuedReport = JSON.parse(await readFile(join(factoryOut, `${qwenMoe.id}.json`), 'utf8'))
  assert.equal(continuedReport.stages.source.status, 'pass')
  assert.deepEqual(validateQualificationReport(continuedReport), [], 'continued batch reports must satisfy the report contract')

  const driftOut = join(factoryOut, 'malicious-source-drift')
  await runFactory({
    root,
    rows: [qwen.id],
    outDir: driftOut,
    resolveSource: true,
    sourceResolver: async () => ({
      ...lockFor(qwen),
      license: 'other C:\\private\\operator\\secret.txt bearer-token',
    }),
  })
  const maliciousDriftReport = JSON.parse(await readFile(join(driftOut, `${qwen.id}.json`), 'utf8'))
  assert.equal(maliciousDriftReport.stages.source.status, 'fail')
  assert.equal(JSON.stringify(maliciousDriftReport).includes('bearer-token'), false)
  assert.equal(JSON.stringify(maliciousDriftReport).includes('operator'), false)
  assert.deepEqual(validateQualificationReport(maliciousDriftReport), [])

  const headerOut = join(factoryOut, 'header-batch')
  let headerCalls = 0
  let headerResolverCalls = 0
  const headerIndex = await runFactory({
    root,
    rows: requested,
    outDir: headerOut,
    inspectHeader: true,
    prefixBytes: 16,
    hfToken: 'test-secret-token',
    sourceResolver: async ({ repo, token }) => {
      headerResolverCalls += 1
      assert.equal(token, 'test-secret-token')
      return repo === qwen.source.repo ? lockFor(qwen) : lockFor(qwenMoe)
    },
    headerInspector: async (sourceLock, options) => {
      headerCalls += 1
      assert.equal(options.prefixBytes, 16)
      assert.equal(options.token, 'test-secret-token')
      assert.equal(options.sourceRoot, root)
      assert.equal(options.rowId, sourceLock.repo === qwen.source.repo ? qwen.id : qwenMoe.id)
      assert.match(
        options.binary,
        process.platform === 'win32'
          ? /target[\\/]release[\\/]camelid\.exe$/
          : /target[\\/]release[\\/]camelid$/,
      )
      if (sourceLock.repo === qwen.source.repo) {
        throw new Error('injected header outage at C:\\outside-workspace\\private.gguf using test-secret-token')
      }
      return headerReceiptFor(qwenMoe)
    },
  })
  assert.equal(headerResolverCalls, 2, '--inspect-header must imply one source preflight per row')
  assert.equal(headerCalls, 2, 'a row-local header failure must not abort the remaining batch')
  assert.deepEqual(headerIndex.source_resolution, {
    mode: 'live_huggingface',
    counts: { pass: 2, fail: 0, blocked: 0 },
  })
  assert.deepEqual(headerIndex.header_inspection, {
    mode: 'remote_immutable_prefix',
    per_row_byte_budget: 16,
    verified_receipt_requested_bytes: 16,
    verified_receipt_received_bytes: 16,
    counts: { pass: 1, fail: 0, blocked: 1, not_run: 0 },
  })
  const headerBlockedReport = JSON.parse(await readFile(join(headerOut, `${qwen.id}.json`), 'utf8'))
  assert.equal(headerBlockedReport.stages.source.status, 'pass')
  assert.equal(headerBlockedReport.stages.artifact.status, 'blocked')
  assert.equal(headerBlockedReport.stages.metadata.status, 'blocked')
  assert.equal(headerBlockedReport.stages.metadata.error_code, 'header_inspection_error')
  assert.equal(headerBlockedReport.overall_status, 'blocked')
  assert.equal(JSON.stringify(headerBlockedReport).includes('test-secret-token'), false)
  assert.equal(JSON.stringify(headerBlockedReport).includes('outside-workspace'), false)
  assert.equal(JSON.stringify(headerBlockedReport).includes('download_url'), false)
  assert.deepEqual(validateQualificationReport(headerBlockedReport), [])

  const headerPassedReport = JSON.parse(await readFile(join(headerOut, `${qwenMoe.id}.json`), 'utf8'))
  assert.equal(headerPassedReport.stages.source.status, 'pass')
  assert.equal(headerPassedReport.stages.artifact.status, 'blocked', 'header evidence must not claim a full artifact')
  assert.equal(headerPassedReport.stages.metadata.status, 'pass')
  assert.equal(headerPassedReport.stages.metadata.mode, 'remote_immutable_prefix')
  assert.equal(headerPassedReport.stages.metadata.inspection_generated_at, '2026-08-10T12:34:56.000Z')
  assert.equal(
    headerPassedReport.stages.metadata.inspector.version,
    `camelid v0.6.1-27-g${currentCommitAbbrev}`,
  )
  assert.equal(headerPassedReport.stages.metadata.inspector.binary_sha256, 'e'.repeat(64))
  assert.equal(headerPassedReport.stages.metadata.inspector.source_head, currentSourceHead)
  assert.equal(headerPassedReport.stages.metadata.inspector.binary_commit_abbrev, currentCommitAbbrev)
  assert.equal(headerPassedReport.stages.metadata.inspector.binary_reports_dirty, false)
  assert.equal(headerPassedReport.stages.metadata.inspector.binary_matches_source_head, true)
  assert.equal(headerPassedReport.stages.metadata.inspector.clean_current_head, true)
  assert.equal(headerPassedReport.stages.metadata.range.prefix_sha256, 'c'.repeat(64))
  assert.equal(headerPassedReport.stages.metadata.scope.prefix_sha256, 'all_received_prefix_bytes')
  assert.equal(headerPassedReport.stages.metadata.scope.full_artifact_sha256, 'not_run')
  assert.equal(headerPassedReport.stages.metadata.scope.tensor_payload, 'partially_range_fetched_opaque')
  assert.equal(headerPassedReport.stages.metadata.scope.tensor_payload_interpretation, 'not_run')
  assert.equal(headerPassedReport.stages.metadata.scope.load, 'not_run')
  assert.equal(headerPassedReport.stages.metadata.scope.generation, 'not_run')
  assert.equal(headerPassedReport.stages.metadata.scope.support_claim, false)
  assert.equal(headerPassedReport.stages.tokenizer.status, 'blocked', 'header metadata must not advance downstream gates')
  assert.equal(headerPassedReport.stages.load_smoke.status, 'blocked')
  assert.equal(headerPassedReport.stages.parity.status, 'blocked')
  assert.equal(headerPassedReport.overall_status, 'blocked')
  assert.equal(JSON.stringify(headerPassedReport).includes('download_url'), false)
  assert.equal(JSON.stringify(headerPassedReport).includes('test-secret-token'), false)
  assert.deepEqual(validateQualificationReport(headerPassedReport), [])

  let invalidBudgetResolverCalled = false
  let invalidBudgetInspectorCalled = false
  await assert.rejects(
    runFactory({
      root,
      rows: [qwen.id],
      outDir: join(factoryOut, 'invalid-budget'),
      inspectHeader: true,
      prefixBytes: MAX_PREFIX_BYTES + 1,
      sourceResolver: async () => {
        invalidBudgetResolverCalled = true
        return lockFor(qwen)
      },
      headerInspector: async () => {
        invalidBudgetInspectorCalled = true
        return headerReceiptFor(qwen)
      },
    }),
    /prefix byte budget/,
  )
  assert.equal(invalidBudgetResolverCalled, false, 'an invalid global budget must fail before network resolution')
  assert.equal(invalidBudgetInspectorCalled, false, 'an invalid global budget must fail before range inspection')
} finally {
  await rm(factoryOut, { recursive: true, force: true })
}

console.log('test-model-qualification-factory: all checks passed')
