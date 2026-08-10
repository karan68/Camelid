#!/usr/bin/env node
import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { validateQualificationReport } from './check-model-qualification-report.mjs'
import {
  artifactForRow,
  firstUnresolvedStage,
  publicRosterLabel,
  resolveSourceStage,
  runFactory,
  selectRows,
  sourceLookupErrorCode,
  sourceSelectionForRow,
  summarizeSourceResolution,
  summarizeReports,
} from './model-qualification-factory.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
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
  download_url: 'https://example.invalid/not-recorded-in-report',
})
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

const driftStage = await resolveSourceStage(qwen, {
  resolver: async () => ({ ...lockFor(qwen), sha256: '0'.repeat(64) }),
})
assert.equal(driftStage.status, 'fail', 'an exact-byte identity drift is a hard failure')
assert.match(driftStage.reason, /does not match roster row/)

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

const factoryOut = await mkdtemp(join(tmpdir(), 'camelid-qualification-factory-test-'))
try {
  let offlineResolverCalled = false
  const offlineIndex = await runFactory({
    root,
    rows: [qwen.id],
    outDir: join(factoryOut, 'offline-default'),
    sourceResolver: async () => {
      offlineResolverCalled = true
      throw new Error('must not run')
    },
  })
  assert.equal(offlineResolverCalled, false, 'the default factory mode must remain fully offline')
  assert.equal(Object.hasOwn(offlineIndex, 'source_resolution'), false, 'default index shape must remain unchanged')

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
} finally {
  await rm(factoryOut, { recursive: true, force: true })
}

console.log('test-model-qualification-factory: all checks passed')
