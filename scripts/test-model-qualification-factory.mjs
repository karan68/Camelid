#!/usr/bin/env node
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { execFile } from 'node:child_process'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import { validateQualificationReport } from './check-model-qualification-report.mjs'
import { HeaderInspectionError, MAX_PREFIX_BYTES } from './hf-qualification-header.mjs'
import { SmolLM3TemplateQualificationError } from './hf-qualification-smollm3-template.mjs'
import { TokenizerQualificationError } from './hf-qualification-tokenizer.mjs'
import {
  artifactForRow,
  candidateIdForSelection,
  candidateMetadataStageFromHeader,
  candidateSelectorDigest,
  captureCandidateWorkspaceProvenance,
  defaultCamelidBinary,
  defaultLlamaTemplateAnalyzerBinary,
  defaultLlamaTokenizerBinary,
  firstUnresolvedStage,
  metadataStageFromHeader,
  metadataStageFromHeaderError,
  parseArgs,
  publicRosterLabel,
  resolveCandidateSourcePreflight,
  resolveSourcePreflight,
  resolveSourceStage,
  runFactory,
  runCandidateFactory,
  selectRows,
  sourceLookupErrorCode,
  sourceSelectionForRow,
  summarizeBatchSourceHeads,
  summarizeHeaderInspections,
  summarizeSourceResolution,
  summarizeReports,
  summarizeTemplatePreparations,
  summarizeTokenizerInspections,
  templatePreparationStageFromError,
  templatePreparationStageFromPack,
  tokenizerStageFromError,
  tokenizerStageFromReceipt,
} from './model-qualification-factory.mjs'
import { qualify } from './model-qualification-runner.mjs'

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
assert.equal(
  defaultLlamaTokenizerBinary(),
  process.platform === 'win32'
    ? 'target/reference/llama.cpp-b9632/bin/llama-tokenize.exe'
    : 'target/reference/llama.cpp-b9632/bin/llama-tokenize',
)
assert.equal(
  defaultLlamaTemplateAnalyzerBinary(),
  process.platform === 'win32'
    ? 'target/reference/llama.cpp-b9632/bin/llama-template-analysis.exe'
    : 'target/reference/llama.cpp-b9632/bin/llama-template-analysis',
)
assert.deepEqual(
  summarizeBatchSourceHeads([
    { report: { source_head: 'a'.repeat(40) } },
    { report: { source_head: 'a'.repeat(40) } },
  ]),
  { source_head: 'a'.repeat(40), state: 'uniform' },
)
assert.deepEqual(
  summarizeBatchSourceHeads([
    { report: { source_head: 'a'.repeat(40) } },
    { report: { source_head: 'b'.repeat(40) } },
  ]),
  { source_head: null, state: 'mixed' },
  'a multi-row batch must not claim the first row HEAD when later rows differ',
)
assert.deepEqual(
  summarizeBatchSourceHeads([
    { report: { source_head: 'a'.repeat(40) } },
    { report: { source_head: null } },
  ]),
  { source_head: null, state: 'unknown' },
  'one unknown per-row HEAD must not be masked by an earlier valid row',
)
const roster = JSON.parse(await readFile(resolve(root, 'qa/model-qualification/phase1-roster.json'), 'utf8'))
const qwen = roster.rows.find((row) => row.id === 'qwen2_5_0_5b_instruct_q8_0')
const qwenMoe = roster.rows.find((row) => row.id === 'qwen3_30b_a3b_q8_0')
const gemma = roster.rows.find((row) => row.id === 'gemma2_9b_it_q8_0')
const smol = roster.rows.find((row) => row.id === 'smollm3_3b_q8_0')
const candidateSelection = {
  repo: 'example-org/Phase2-Candidate-GGUF',
  file: 'weights/phase2-candidate-Q8_0.gguf',
  revision: null,
}
const candidateSelectorSha256 = createHash('sha256')
  .update(JSON.stringify(candidateSelection))
  .digest('hex')
const candidateId = `hf_selector_${candidateSelectorSha256.slice(0, 24)}`
const opaqueRunIdentity = '01234567-89ab-4cde-8fab-0123456789ab'
const opaqueCandidateId = 'hf_candidate_run_0123456789ab4cde8fab0123456789ab'
const candidateLock = (overrides = {}) => ({
  schema: 'camelid.hf-source-lock/v1',
  repo: candidateSelection.repo,
  file: candidateSelection.file,
  revision: '7'.repeat(40),
  size_bytes: 100,
  sha256: '8'.repeat(64),
  license: 'apache-2.0',
  access: { gated: false, private: false, disabled: false },
  download_url: `https://huggingface.co/${candidateSelection.repo}/resolve/${'7'.repeat(40)}/${candidateSelection.file}?download=true`,
  ...overrides,
})
const committedSmolTemplatePack = JSON.parse(await readFile(
  resolve(root, 'qa/prompt-packs/smollm3-chat-template-shapes-v1.json'),
  'utf8',
))

const templatePackForFactory = () => {
  const pack = structuredClone(committedSmolTemplatePack)
  pack.inspector = {
    ...pack.inspector,
    version: `camelid v0.6.1-test-g${currentCommitAbbrev}`,
    binary_sha256: '9'.repeat(64),
    source_head: currentSourceHead,
    source_tracked_dirty: false,
    binary_commit_abbrev: currentCommitAbbrev,
    binary_reports_dirty: false,
    binary_matches_source_head: true,
    clean_current_head: true,
    binary_path_redacted: true,
  }
  return pack
}
const templateInspectorForFactory = () => structuredClone(templatePackForFactory().inspector)
const templateBinaryIdentityForFactory = () => {
  const identity = templateInspectorForFactory()
  delete identity.source_tracked_dirty
  delete identity.binary_path_redacted
  return identity
}

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
assert.equal(candidateSelectorDigest(candidateSelection), candidateSelectorSha256)
assert.equal(candidateIdForSelection(candidateSelection), candidateId)
assert.deepEqual([...parseArgs([
  '--repo=org/model',
  '--file',
  'weights/model=Q8_0.gguf',
  '--inspect-header',
])], [
  ['repo', 'org/model'],
  ['file', 'weights/model=Q8_0.gguf'],
  ['inspect-header', true],
])
for (const argv of [
  ['positional'],
  ['--'],
  ['--unknown'],
  ['--repo='],
  ['--repo=   '],
  ['--repo'],
  ['--repo', '   '],
  ['--repo', '--file', 'model.gguf'],
  ['--repo', 'org/model', '--repo', 'other/model'],
  ['--inspect-header=true'],
  ['--inspect-header', 'true'],
  ['--repo', 'org/model', 'extra'],
]) {
  assert.throws(() => parseArgs(argv), /positional|unknown option|non-empty value|exactly one|duplicate|does not accept/)
}
for (const invalidPromptLimit of ['1junk', '01', '+1', '1.0', '1e2', '9007199254740992']) {
  assert.throws(
    () => parseArgs(['--prompt-limit', invalidPromptLimit]),
    /canonical positive integer/,
    `--prompt-limit ${invalidPromptLimit} must fail before any roster or source work`,
  )
}
assert.equal(parseArgs(['--prompt-limit', '1']).get('prompt-limit'), '1')
for (const invalidSelection of [
  { repo: 'missing-owner', file: 'model.gguf', revision: null },
  { repo: 'org/model', file: '../model.gguf', revision: null },
  { repo: 'org/model', file: 'C:\\private\\model.gguf', revision: null },
  { repo: 'org/model', file: 'model.safetensors', revision: null },
  { repo: 'org/model', file: 'model.gguf', revision: 'main' },
]) {
  assert.throws(() => candidateIdForSelection(invalidSelection), /--repo|--file|--revision/)
}

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
    source_tracked_dirty: false,
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
      general_file_type: row.identity.quantization === 'Q4_K_M' ? 15 : 7,
      tokenizer_model: row.expected.tokenizer_model,
      tokenizer_pre: row.expected.tokenizer_pre,
      headline_quant: row.identity.quantization === 'Q4_K_M' ? 'Q4_K' : row.identity.quantization,
    },
    tensor_inventory: {
      sha256: 'd'.repeat(64),
      total_n_bytes: row.identity.size_bytes - 4096,
      types: row.identity.quantization === 'Q4_K_M'
        ? { Q4_K: 3, Q6_K: 1, F32: 1 }
        : { [row.identity.quantization]: 4, F32: 1 },
    },
  },
  support_claim: false,
  note: 'test receipt',
  ...overrides,
})
const currentHeadMetadataStage = (row, receipt) => metadataStageFromHeader(row, receipt, {
  expectedSourceHead: currentSourceHead,
})
const candidateHeaderReceipt = (overrides = {}) => {
  const lock = candidateLock()
  const row = {
    id: candidateId,
    source: {
      repo: lock.repo,
      file: lock.file,
      revision: lock.revision,
    },
    identity: {
      size_bytes: lock.size_bytes,
      sha256: lock.sha256,
      architecture: 'qwen2',
      quantization: 'Q8_0',
    },
    expected: { tokenizer_model: 'gpt2', tokenizer_pre: 'qwen2' },
  }
  const receipt = headerReceiptFor(row)
  receipt.range = {
    requested_bytes: 99,
    received_bytes: 99,
    content_range: { start: 0, end: 98, total: lock.size_bytes },
    prefix_sha256: 'c'.repeat(64),
  }
  receipt.inspection.data_start_offset = 96
  receipt.inspection.tensor_inventory.total_n_bytes = 4
  return { ...receipt, ...overrides }
}

const tokenizerReceiptPaths = new Map([
  [gemma.id, 'qa/model-qualification/gemma2-9b-it-q8-header-tokenizer-parity.json'],
  [smol.id, 'qa/model-qualification/smollm3-3b-q8-header-tokenizer-parity.json'],
  [qwenMoe.id, 'qa/model-qualification/qwen3-30b-a3b-q8-header-tokenizer-parity.json'],
])
const durableTokenizerReceipts = new Map(await Promise.all(
  [...tokenizerReceiptPaths].map(async ([rowId, path]) => [
    rowId,
    JSON.parse(await readFile(resolve(root, path), 'utf8')),
  ]),
))
const tokenizerReceiptFor = (row) => {
  const receipt = structuredClone(durableTokenizerReceipts.get(row.id))
  receipt.generated_at = '2026-08-10T19:00:00.000Z'
  receipt.provenance = {
    status: 'clean_current_head_receipt',
    gate_requires_clean_current_head: true,
    source_head: currentSourceHead,
    source_tracked_dirty: false,
    binary_commit_abbrev: currentCommitAbbrev,
    binary_reports_dirty: false,
    binary_matches_source_head: true,
    clean_current_head: true,
  }
  receipt.camelid.version = `camelid v0.6.1-test-g${currentCommitAbbrev}`
  receipt.camelid.binary_sha256 = 'a'.repeat(64)
  return receipt
}
const durableQwenTokenizerBytes = await readFile(resolve(
  root,
  tokenizerReceiptPaths.get(qwenMoe.id),
))
assert.equal(
  createHash('sha256').update(durableQwenTokenizerBytes).digest('hex'),
  '021dbe0b4f6a94f7140daa8e02969106dab941e205d184ee60f683d58f13ea37',
  'factory evidence must remain byte-bound to the clean-head Qwen3 tokenizer receipt',
)

for (const [rowId, receiptPath] of [
  ['qwen3_30b_a3b_q8_0', 'qa/model-qualification/qwen3-30b-a3b-q8-header-inspection.json'],
  ['gemma2_9b_it_q8_0', 'qa/model-qualification/gemma2-9b-it-q8-header-inspection.json'],
  ['smollm3_3b_q8_0', 'qa/model-qualification/smollm3-3b-q8-header-inspection.json'],
]) {
  const durableRow = roster.rows.find((row) => row.id === rowId)
  assert.ok(durableRow, `committed header receipt row ${rowId} must remain in the roster`)
  const durableReceipt = JSON.parse(await readFile(resolve(root, receiptPath), 'utf8'))
  const durableStage = metadataStageFromHeader(durableRow, durableReceipt, {
    expectedSourceHead: durableReceipt.inspector.source_head,
  })
  assert.equal(
    durableStage.status,
    'fail',
    `${receiptPath} predates tracked-worktree provenance and must remain preparatory`,
  )
  assert.equal(durableStage.error_code, 'header_receipt_invalid')
  if (rowId === qwenMoe.id) {
    assert.equal(
      durableReceipt.range.prefix_sha256,
      '55c565264523c5862247d983f857b9034c04d762ee14fecfd68a827cdbb2d566',
    )
    assert.equal(durableReceipt.inspection.tensor_count, 579)
    assert.equal(durableReceipt.inspection.metadata_count, 31)
    assert.equal(durableReceipt.inspection.data_start_offset, 5_969_408)
    assert.deepEqual(durableReceipt.inspection.observed, {
      architecture: 'qwen3moe',
      general_file_type: 7,
      tokenizer_model: 'gpt2',
      tokenizer_pre: 'qwen2',
      headline_quant: 'Q8_0',
    })
    assert.deepEqual(durableReceipt.inspection.tensor_inventory, {
      sha256: '8cf88e2856a5aff086e8ea0f2767e7af63a9feac1b721e7df676b472d9f2cdcf',
      total_n_bytes: 32_477_962_240,
      types: { Q8_0: 338, F32: 241 },
    })
    assert.equal(durableReceipt.support_claim, false)
    assert.equal(durableRow.gates.tokenizer.status, 'pass')
    assert(durableRow.gates.tokenizer.evidence.includes(
      'qa/model-qualification/qwen3-30b-a3b-q8-header-tokenizer-parity.json',
    ))
    assert.notEqual(durableRow.gates.template.status, 'pass')
  }
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

const passingHeaderStage = metadataStageFromHeader(qwen, headerReceiptFor(qwen), {
  expectedSourceHead: currentSourceHead,
})
assert.equal(passingHeaderStage.status, 'pass')
assert.equal(passingHeaderStage.mode, 'remote_immutable_prefix')
assert.equal(passingHeaderStage.observed.architecture, qwen.identity.architecture)
assert.equal(passingHeaderStage.observed.general_file_type, 7)
assert.equal(passingHeaderStage.observed.declared_quantization, 'Q8_0')
assert.equal(passingHeaderStage.observed.tensor_inventory_sha256, 'd'.repeat(64))
assert.equal(passingHeaderStage.inspection_generated_at, '2026-08-10T12:34:56.000Z')
assert.equal(passingHeaderStage.host.hostname_redacted, true)
assert.equal(Object.hasOwn(passingHeaderStage.host, 'hostname'), false)
assert.equal(passingHeaderStage.inspector.version, `camelid v0.6.1-27-g${currentCommitAbbrev}`)
assert.equal(passingHeaderStage.inspector.binary_sha256, 'e'.repeat(64))
assert.equal(passingHeaderStage.inspector.source_head, currentSourceHead)
assert.equal(passingHeaderStage.inspector.source_tracked_dirty, false)
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
const unboundHeaderStage = metadataStageFromHeader(qwen, headerReceiptFor(qwen))
assert.equal(unboundHeaderStage.status, 'fail', 'header receipts must be bound to an explicit source HEAD')
assert.equal(unboundHeaderStage.error_code, 'header_receipt_invalid')

const q4kmRow = structuredClone(qwen)
q4kmRow.identity.quantization = 'Q4_K_M'
const q4kmStage = currentHeadMetadataStage(q4kmRow, headerReceiptFor(q4kmRow))
assert.equal(q4kmStage.status, 'pass', 'Q4_K_M is a declared file recipe, not a tensor type')
assert.equal(q4kmStage.observed.general_file_type, 15)
assert.equal(q4kmStage.observed.declared_quantization, 'Q4_K_M')
assert.equal(q4kmStage.observed.headline_quant, 'Q4_K')

const q4kmMissingMixReceipt = headerReceiptFor(q4kmRow)
q4kmMissingMixReceipt.inspection.tensor_inventory.types = { Q4_K: 4, F32: 1 }
const q4kmMissingMixStage = currentHeadMetadataStage(q4kmRow, q4kmMissingMixReceipt)
assert.equal(q4kmMissingMixStage.status, 'fail')
assert.match(q4kmMissingMixStage.reason, /both Q4_K and Q6_K/)

const passingCandidateHeaderStage = candidateMetadataStageFromHeader(
  candidateId,
  candidateLock(),
  candidateHeaderReceipt(),
  { expectedSourceHead: currentSourceHead },
)
assert.equal(passingCandidateHeaderStage.status, 'pass')
assert.equal(passingCandidateHeaderStage.assessment, 'bounded_header_descriptor_inspection_only')
assert.equal(passingCandidateHeaderStage.range.received_bytes, 99)
assert.deepEqual(passingCandidateHeaderStage.observed.tensor_type_counts, { F32: 1, Q8_0: 4 })
assert.equal(passingCandidateHeaderStage.scope.opaque_tensor_payload_prefix_bytes, 3)
assert.equal(passingCandidateHeaderStage.scope.runtime_compatibility, 'not_run')
assert.equal(passingCandidateHeaderStage.scope.support_claim, false)

for (const [mutate, expectedErrorCode] of [
  [(receipt) => { receipt.inspection.alignment = 24 }, 'header_descriptor_invariants_invalid'],
  [(receipt) => { receipt.inspection.data_start_offset = 95 }, 'header_descriptor_invariants_invalid'],
  [(receipt) => { receipt.inspection.alignment = 16; receipt.inspection.data_start_offset = 16 }, 'header_descriptor_invariants_invalid'],
  [(receipt) => { receipt.inspection.tensor_inventory.total_n_bytes = 5 }, 'header_descriptor_invariants_invalid'],
  [(receipt) => { receipt.inspection.tensor_inventory.total_n_bytes = 3 }, 'header_descriptor_invariants_invalid'],
  [(receipt) => { delete receipt.inspection.tensor_inventory.types }, 'header_receipt_invalid'],
  [(receipt) => { receipt.inspection.tensor_inventory.types = { F32: 1, Q8_0: 3 } }, 'header_receipt_invalid'],
  [(receipt) => {
    receipt.range.requested_bytes = 96
    receipt.range.received_bytes = 96
    receipt.range.content_range.end = 95
  }, 'header_descriptor_invariants_invalid'],
]) {
  const receipt = candidateHeaderReceipt()
  mutate(receipt)
  const rejected = candidateMetadataStageFromHeader(
    candidateId,
    candidateLock(),
    receipt,
    { expectedSourceHead: currentSourceHead },
  )
  assert.equal(rejected.status, 'fail')
  assert.equal(rejected.error_code, expectedErrorCode)
}
const forgedOpaqueCountReceipt = candidateHeaderReceipt()
forgedOpaqueCountReceipt.scope.opaque_tensor_payload_prefix_bytes = 999
assert.equal(
  candidateMetadataStageFromHeader(
    candidateId,
    candidateLock(),
    forgedOpaqueCountReceipt,
    { expectedSourceHead: currentSourceHead },
  ).scope.opaque_tensor_payload_prefix_bytes,
  3,
  'the compact report must derive opaque bytes instead of trusting a receipt-supplied count',
)

const fullCandidateReceipt = candidateHeaderReceipt()
fullCandidateReceipt.range.requested_bytes = 100
fullCandidateReceipt.range.received_bytes = 100
fullCandidateReceipt.range.content_range.end = 99
const forbiddenFullCandidateStage = candidateMetadataStageFromHeader(
  candidateId,
  candidateLock(),
  fullCandidateReceipt,
  { expectedSourceHead: currentSourceHead },
)
assert.equal(forbiddenFullCandidateStage.status, 'fail')
assert.equal(forbiddenFullCandidateStage.error_code, 'header_full_artifact_forbidden')

const maliciousHeaderReceipt = headerReceiptFor(qwen)
maliciousHeaderReceipt.inspection.observed.architecture = 'C:\\private\\model.gguf test-secret-token'
const mismatchedHeaderStage = currentHeadMetadataStage(qwen, maliciousHeaderReceipt)
assert.equal(mismatchedHeaderStage.status, 'fail')
assert.equal(mismatchedHeaderStage.observed.architecture, '<invalid>')
assert.equal(JSON.stringify(mismatchedHeaderStage).includes('test-secret-token'), false)
assert.equal(JSON.stringify(mismatchedHeaderStage).includes('private'), false)

const identityDriftReceipt = headerReceiptFor(qwen)
identityDriftReceipt.source.sha256 = '0'.repeat(64)
const identityDriftHeaderStage = currentHeadMetadataStage(qwen, identityDriftReceipt)
assert.equal(identityDriftHeaderStage.status, 'fail')
assert.equal(identityDriftHeaderStage.error_code, 'header_source_identity_mismatch')

const missingInspectorReceipt = headerReceiptFor(qwen)
delete missingInspectorReceipt.inspector
const missingInspectorStage = currentHeadMetadataStage(qwen, missingInspectorReceipt)
assert.equal(missingInspectorStage.status, 'fail')
assert.equal(missingInspectorStage.error_code, 'header_receipt_invalid')

const maliciousInspectorReceipt = headerReceiptFor(qwen)
maliciousInspectorReceipt.inspector.version = 'C:\\private\\camelid.exe test-secret-token'
const maliciousInspectorStage = currentHeadMetadataStage(qwen, maliciousInspectorReceipt)
assert.equal(maliciousInspectorStage.status, 'fail')
assert.equal(JSON.stringify(maliciousInspectorStage).includes('test-secret-token'), false)
assert.equal(JSON.stringify(maliciousInspectorStage).includes('private'), false)

const dirtyInspectorReceipt = headerReceiptFor(qwen)
dirtyInspectorReceipt.inspector.version += '-dirty'
dirtyInspectorReceipt.inspector.binary_reports_dirty = true
dirtyInspectorReceipt.inspector.clean_current_head = false
const dirtyInspectorStage = currentHeadMetadataStage(qwen, dirtyInspectorReceipt)
assert.equal(dirtyInspectorStage.status, 'fail')
assert.equal(dirtyInspectorStage.error_code, 'header_receipt_invalid')

const staleInspectorReceipt = headerReceiptFor(qwen)
staleInspectorReceipt.inspector.version = 'camelid v0.6.1-27-gabcdef12'
staleInspectorReceipt.inspector.binary_commit_abbrev = 'abcdef12'
staleInspectorReceipt.inspector.binary_matches_source_head = false
staleInspectorReceipt.inspector.clean_current_head = false
const staleInspectorStage = currentHeadMetadataStage(qwen, staleInspectorReceipt)
assert.equal(staleInspectorStage.status, 'fail')
assert.equal(staleInspectorStage.error_code, 'header_receipt_invalid')

const forgedInspectorReceipt = headerReceiptFor(qwen)
forgedInspectorReceipt.inspector.binary_reports_dirty = true
const forgedInspectorStage = currentHeadMetadataStage(qwen, forgedInspectorReceipt)
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

const passingTokenizerStage = tokenizerStageFromReceipt(
  smol,
  tokenizerReceiptFor(smol),
  roster.defaults,
  { expectedSourceHead: currentSourceHead },
)
assert.equal(passingTokenizerStage.status, 'pass')
assert.equal(passingTokenizerStage.mode, 'remote_immutable_prefix_tokenizer')
assert.equal(passingTokenizerStage.result.all_token_ids_match, true)
assert.equal(passingTokenizerStage.result.case_count, 10)
assert.equal(passingTokenizerStage.inspector.source_head, currentSourceHead)
assert.equal(passingTokenizerStage.inspector.clean_current_head, true)
assert.equal(passingTokenizerStage.range.requested_bytes, 32 * 1024 * 1024)
assert.equal(passingTokenizerStage.scope.full_artifact_sha256, 'not_run')
assert.equal(passingTokenizerStage.scope.template_rendering, 'not_run')
assert.equal(passingTokenizerStage.scope.load, 'not_run')
assert.equal(passingTokenizerStage.scope.generation, 'not_run')
assert.equal(passingTokenizerStage.scope.api_webui, 'not_run')
assert.equal(passingTokenizerStage.scope.context, 'not_run')
assert.equal(passingTokenizerStage.scope.support_claim, false)
assert.equal(Object.hasOwn(passingTokenizerStage, 'cases'), false, 'full token arrays must stay out of factory reports')
assert.equal(Object.hasOwn(passingTokenizerStage, 'does_not_prove'), false, 'arbitrary receipt prose must stay out of factory reports')

const passingQwenMoeTokenizerStage = tokenizerStageFromReceipt(
  qwenMoe,
  tokenizerReceiptFor(qwenMoe),
  roster.defaults,
  { expectedSourceHead: currentSourceHead },
)
assert.equal(passingQwenMoeTokenizerStage.status, 'pass')
assert.equal(passingQwenMoeTokenizerStage.result.case_count, 13)
assert.equal(passingQwenMoeTokenizerStage.result.exact_match_count, 13)
assert.equal(passingQwenMoeTokenizerStage.result.all_token_ids_match, true)
assert.equal(passingQwenMoeTokenizerStage.observed.token_count, 151_936)
assert.equal(passingQwenMoeTokenizerStage.observed.declared_add_bos_token, false)
assert.equal(passingQwenMoeTokenizerStage.scope.template_rendering, 'not_run')
assert.equal(passingQwenMoeTokenizerStage.scope.load, 'not_run')
assert.equal(passingQwenMoeTokenizerStage.scope.generation, 'not_run')
assert.equal(passingQwenMoeTokenizerStage.scope.api_webui, 'not_run')
assert.equal(passingQwenMoeTokenizerStage.scope.context, 'not_run')
assert.equal(passingQwenMoeTokenizerStage.scope.support_claim, false)
assert.equal(Object.hasOwn(passingQwenMoeTokenizerStage, 'cases'), false)

const staleTokenizerStage = tokenizerStageFromReceipt(
  smol,
  durableTokenizerReceipts.get(smol.id),
  roster.defaults,
  { expectedSourceHead: currentSourceHead },
)
assert.equal(staleTokenizerStage.status, 'fail')
assert.equal(staleTokenizerStage.error_code, 'tokenizer_receipt_invalid')

const honestTokenizerMismatch = tokenizerReceiptFor(gemma)
honestTokenizerMismatch.cases[0].llama_cpp_ids[0] = (
  honestTokenizerMismatch.cases[0].llama_cpp_ids[0] + 1
) % 256_000
honestTokenizerMismatch.cases[0].exact_match = false
honestTokenizerMismatch.result.exact_match_count -= 1
honestTokenizerMismatch.result.all_token_ids_match = false
const mismatchTokenizerStage = tokenizerStageFromReceipt(
  gemma,
  honestTokenizerMismatch,
  roster.defaults,
  { expectedSourceHead: currentSourceHead },
)
assert.equal(mismatchTokenizerStage.status, 'fail')
assert.equal(mismatchTokenizerStage.error_code, 'tokenizer_parity_mismatch')
assert.equal(mismatchTokenizerStage.result.all_token_ids_match, false)

const forgedTokenizerMatch = structuredClone(honestTokenizerMismatch)
forgedTokenizerMatch.cases[0].exact_match = true
forgedTokenizerMatch.result.exact_match_count = forgedTokenizerMatch.cases.length
forgedTokenizerMatch.result.all_token_ids_match = true
const forgedTokenizerStage = tokenizerStageFromReceipt(
  gemma,
  forgedTokenizerMatch,
  roster.defaults,
  { expectedSourceHead: currentSourceHead },
)
assert.equal(forgedTokenizerStage.status, 'fail')
assert.equal(forgedTokenizerStage.error_code, 'tokenizer_receipt_invalid')

const safeTokenizerFailure = tokenizerStageFromError(new TokenizerQualificationError(
  'tokenizer_oracle_unavailable',
  'fail',
  'C:\\outside-workspace\\llama-tokenize.exe test-secret-token',
))
assert.deepEqual(safeTokenizerFailure, {
  status: 'blocked',
  mode: 'remote_immutable_prefix_tokenizer',
  error_code: 'tokenizer_oracle_unavailable',
  reason: 'the pinned llama.cpp tokenizer oracle is unavailable',
})
const unknownTokenizerFailure = tokenizerStageFromError(
  new Error('C:\\private\\prefix.gguf test-secret-token'),
)
assert.equal(unknownTokenizerFailure.error_code, 'tokenizer_qualification_error')
assert.equal(JSON.stringify(unknownTokenizerFailure).includes('private'), false)
assert.equal(JSON.stringify(unknownTokenizerFailure).includes('test-secret-token'), false)

const passingTemplatePreparation = templatePreparationStageFromPack(
  smol,
  templatePackForFactory(),
  {
    expectedSourceHead: currentSourceHead,
    expectedInspector: templateInspectorForFactory(),
  },
)
assert.equal(passingTemplatePreparation.status, 'blocked')
assert.equal(passingTemplatePreparation.error_code, 'smollm3_template_runtime_hold')
assert.equal(passingTemplatePreparation.preparation.status, 'pass')
assert.equal(passingTemplatePreparation.range.requested_bytes, 32 * 1024 * 1024)
assert.equal(passingTemplatePreparation.range.received_bytes, 32 * 1024 * 1024)
assert.equal(passingTemplatePreparation.inspector.source_head, currentSourceHead)
assert.equal(passingTemplatePreparation.scope.runtime_chat, 'blocked')
assert.equal(passingTemplatePreparation.scope.template_gate, 'blocked')
assert.equal(passingTemplatePreparation.scope.support_claim, false)
assert.equal(passingTemplatePreparation.cases.length, 2)
assert.equal(passingTemplatePreparation.cases.every((testCase) => testCase.oracle_exact_match), true)
const compactTemplatePreparation = JSON.stringify(passingTemplatePreparation)
assert.ok(Buffer.byteLength(compactTemplatePreparation) < 4 * 1024, 'factory projection must remain compact')
for (const forbidden of [
  'source_template',
  'normalized_prompt"',
  'messages',
  'header_receipt',
  'tokenizer_receipt',
  'download_url',
  'llama-template-analysis.exe',
  '<|im_start|>',
]) {
  assert.equal(
    compactTemplatePreparation.includes(forbidden),
    false,
    `factory template preparation must omit ${forbidden}`,
  )
}

const staleTemplatePreparation = templatePreparationStageFromPack(
  smol,
  templatePackForFactory(),
  {
    expectedSourceHead: 'f'.repeat(40),
    expectedInspector: templateInspectorForFactory(),
  },
)
assert.equal(staleTemplatePreparation.status, 'fail')
assert.equal(staleTemplatePreparation.error_code, 'template_preparation_receipt_invalid')

const forgedTemplatePack = templatePackForFactory()
const forgedTemplateHead = `11111111${'1'.repeat(32)}`
forgedTemplatePack.inspector = {
  ...forgedTemplatePack.inspector,
  version: 'camelid v0.6.1-99-g11111111',
  binary_sha256: '2'.repeat(64),
  source_head: forgedTemplateHead,
  source_tracked_dirty: false,
  binary_commit_abbrev: '11111111',
  binary_reports_dirty: false,
  binary_matches_source_head: true,
  clean_current_head: true,
  binary_path_redacted: true,
}
const forgedTemplatePreparation = templatePreparationStageFromPack(
  smol,
  forgedTemplatePack,
  {
    expectedSourceHead: forgedTemplateHead,
    expectedInspector: templateInspectorForFactory(),
  },
)
assert.equal(forgedTemplatePreparation.status, 'fail')
assert.equal(forgedTemplatePreparation.error_code, 'template_preparation_receipt_invalid')

const tamperedTemplatePack = templatePackForFactory()
tamperedTemplatePack.source_template.text = tamperedTemplatePack.source_template.text.replace(
  '<|im_start|>',
  '<|im_finish|>',
)
tamperedTemplatePack.runtime_chat_enabled = 'C:\\private\\template.gguf test-secret-token'
const tamperedTemplatePreparation = templatePreparationStageFromPack(
  smol,
  tamperedTemplatePack,
  {
    expectedSourceHead: currentSourceHead,
    expectedInspector: templateInspectorForFactory(),
  },
)
assert.equal(tamperedTemplatePreparation.status, 'fail')
assert.equal(tamperedTemplatePreparation.error_code, 'template_preparation_receipt_invalid')
assert.equal(JSON.stringify(tamperedTemplatePreparation).includes('<|im_finish|>'), false)
assert.equal(JSON.stringify(tamperedTemplatePreparation).includes('test-secret-token'), false)
assert.equal(JSON.stringify(tamperedTemplatePreparation).includes('template.gguf'), false)

const safeTemplateFailure = templatePreparationStageFromError(
  new SmolLM3TemplateQualificationError('smollm3_template_oracle_unavailable'),
)
assert.deepEqual(safeTemplateFailure, {
  status: 'blocked',
  mode: 'remote_immutable_prefix_smollm3_template_preparation',
  error_code: 'smollm3_template_oracle_unavailable',
  reason: 'the pinned llama.cpp template analyzer package is unavailable',
  preparation: { status: 'blocked' },
})
const unknownTemplateFailure = templatePreparationStageFromError(
  new Error('C:\\private\\template.gguf test-secret-token'),
)
assert.equal(unknownTemplateFailure.error_code, 'smollm3_template_qualification_error')
assert.equal(JSON.stringify(unknownTemplateFailure).includes('private'), false)
assert.equal(JSON.stringify(unknownTemplateFailure).includes('test-secret-token'), false)

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

const candidatePreflight = await resolveCandidateSourcePreflight(candidateSelection, {
  token: 'test-secret-token',
  resolver: async (selection) => {
    assert.deepEqual(selection, { ...candidateSelection, token: 'test-secret-token' })
    return candidateLock()
  },
})
assert.equal(candidatePreflight.stage.status, 'pass')
assert.equal(candidatePreflight.stage.requested_revision, null)
assert.equal(candidatePreflight.stage.revision, '7'.repeat(40))
assert.equal(candidatePreflight.lock.download_url, candidateLock().download_url)
assert.equal(JSON.stringify(candidatePreflight.stage).includes('download_url'), false)
assert.equal(JSON.stringify(candidatePreflight.stage).includes('test-secret-token'), false)

const pinnedCandidateSelection = { ...candidateSelection, revision: '7'.repeat(40) }
const pinnedCandidatePreflight = await resolveCandidateSourcePreflight(pinnedCandidateSelection, {
  resolver: async () => candidateLock(),
})
assert.equal(pinnedCandidatePreflight.stage.status, 'pass')
assert.equal(pinnedCandidatePreflight.stage.requested_revision, '7'.repeat(40))
assert.equal(pinnedCandidatePreflight.stage.revision, '7'.repeat(40))

const gatedPublicCandidate = await resolveCandidateSourcePreflight(candidateSelection, {
  resolver: async () => candidateLock({ access: { gated: true, private: false, disabled: false } }),
})
assert.equal(gatedPublicCandidate.stage.status, 'pass')
assert.equal(gatedPublicCandidate.stage.access.gated, true)

const candidateWrongLock = await resolveCandidateSourcePreflight(candidateSelection, {
  resolver: async () => candidateLock({ repo: 'malicious/private-path' }),
})
assert.equal(candidateWrongLock.stage.status, 'fail')
assert.equal(candidateWrongLock.stage.error_code, 'source_identity_invalid')
assert.equal(candidateWrongLock.lock, null)
assert.equal(JSON.stringify(candidateWrongLock.stage).includes('malicious'), false)

const candidateWrongUrl = await resolveCandidateSourcePreflight(candidateSelection, {
  resolver: async () => candidateLock({
    download_url: `${candidateLock().download_url}&token=test-secret-token`,
  }),
})
assert.equal(candidateWrongUrl.stage.status, 'fail')
assert.equal(candidateWrongUrl.stage.error_code, 'source_identity_invalid')
assert.equal(JSON.stringify(candidateWrongUrl.stage).includes('test-secret-token'), false)

const candidateUnsafeLicense = await resolveCandidateSourcePreflight(candidateSelection, {
  resolver: async () => candidateLock({ license: 'C:\\private\\license.txt bearer-token' }),
})
assert.equal(candidateUnsafeLicense.stage.status, 'blocked')
assert.equal(candidateUnsafeLicense.stage.error_code, 'source_license_unavailable')
assert.equal(JSON.stringify(candidateUnsafeLicense.stage).includes('bearer-token'), false)

const privateCandidate = await resolveCandidateSourcePreflight(candidateSelection, {
  resolver: async () => candidateLock({ access: { gated: true, private: true, disabled: false } }),
})
assert.equal(privateCandidate.stage.status, 'blocked')
assert.equal(privateCandidate.stage.error_code, 'private_source_not_persisted')
assert.equal(privateCandidate.stage.selector_redacted, true)
assert.equal(Object.hasOwn(privateCandidate.stage, 'repo'), false)
assert.equal(privateCandidate.lock, null)

const disabledCandidate = await resolveCandidateSourcePreflight(candidateSelection, {
  resolver: async () => candidateLock({ access: { gated: false, private: false, disabled: true } }),
})
assert.equal(disabledCandidate.stage.status, 'blocked')
assert.equal(disabledCandidate.stage.error_code, 'source_disabled')
assert.equal(disabledCandidate.lock, null)

const candidateAccessSmuggling = await resolveCandidateSourcePreflight(candidateSelection, {
  resolver: async () => candidateLock({
    access: { gated: false, private: false, disabled: false, accessToken: 'secret' },
  }),
})
assert.equal(candidateAccessSmuggling.stage.status, 'fail')
assert.equal(candidateAccessSmuggling.stage.error_code, 'source_identity_invalid')
assert.equal(JSON.stringify(candidateAccessSmuggling.stage).includes('accessToken'), false)

const unavailableCandidate = await resolveCandidateSourcePreflight(candidateSelection, {
  token: 'test-secret-token',
  resolver: async () => { throw new Error('C:\\private\\model.gguf test-secret-token') },
})
assert.equal(unavailableCandidate.stage.status, 'blocked')
assert.equal(unavailableCandidate.stage.error_code, 'source_lookup_error')
assert.equal(JSON.stringify(unavailableCandidate.stage).includes('test-secret-token'), false)
assert.equal(JSON.stringify(unavailableCandidate.stage).includes('private'), false)

const capturedCandidateProvenance = await captureCandidateWorkspaceProvenance(root, {
  gitImpl: async (_binary, args) => {
    if (args[0] === 'rev-parse') return { stdout: `${currentSourceHead}\n` }
    if (args.includes('--untracked-files=no')) return { stdout: '' }
    return { stdout: '?? harmless-untracked-file.txt\n' }
  },
})
assert.deepEqual(capturedCandidateProvenance, {
  source_head: currentSourceHead,
  source_dirty: true,
  source_tracked_dirty: false,
  source_inspection: 'observed',
})
assert.deepEqual(
  await captureCandidateWorkspaceProvenance(root, {
    gitImpl: async () => { throw new Error('C:\\private\\repo test-secret-token') },
  }),
  {
    source_head: null,
    source_dirty: null,
    source_tracked_dirty: null,
    source_inspection: 'unknown',
  },
)

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
  summarizeSourceResolution([{
    sourcePreflight: null,
    report: { stages: { source: { status: 'pass' } } },
  }]),
  { mode: 'live_huggingface', counts: { pass: 0, fail: 0, blocked: 0, not_run: 1 } },
  'a skipped per-row preflight must not be mislabeled as a live Hugging Face pass',
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

assert.deepEqual(
  summarizeTokenizerInspections([
    { tokenizerOutcome: passingTokenizerStage },
    { tokenizerOutcome: safeTokenizerFailure },
    { tokenizerOutcome: { status: 'not_run' } },
  ]),
  {
    mode: 'remote_immutable_prefix_tokenizer',
    per_row_byte_budget: 32 * 1024 * 1024,
    verified_receipt_requested_bytes: 32 * 1024 * 1024,
    verified_receipt_received_bytes: 32 * 1024 * 1024,
    counts: { pass: 1, fail: 0, blocked: 1, not_run: 1 },
  },
)

assert.deepEqual(
  summarizeTemplatePreparations([
    { templatePreparationOutcome: passingTemplatePreparation },
    { templatePreparationOutcome: safeTemplateFailure },
    { templatePreparationOutcome: null },
  ]),
  {
    mode: 'remote_immutable_prefix_smollm3_template_preparation',
    per_row_byte_budget: 32 * 1024 * 1024,
    verified_receipt_requested_bytes: 32 * 1024 * 1024,
    verified_receipt_received_bytes: 32 * 1024 * 1024,
    counts: { pass: 0, fail: 0, blocked: 2, not_run: 1 },
    preparation_results: { pass: 1, fail: 0, blocked: 1, not_run: 1 },
    runtime_template_gate: 'blocked',
    support_claim: false,
  },
)

const factoryOut = await mkdtemp(join(tmpdir(), 'camelid-qualification-factory-test-'))
try {
  let offlineResolverCalled = false
  let offlineHeaderInspectorCalled = false
  let offlineTokenizerInspectorCalled = false
  let offlineTemplateInspectorCalled = false
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
    tokenizerInspector: async () => {
      offlineTokenizerInspectorCalled = true
      throw new Error('must not run')
    },
    templateInspector: async () => {
      offlineTemplateInspectorCalled = true
      throw new Error('must not run')
    },
  })
  assert.equal(offlineResolverCalled, false, 'the default factory mode must remain fully offline')
  assert.equal(offlineHeaderInspectorCalled, false, 'the default factory mode must not inspect remote headers')
  assert.equal(offlineTokenizerInspectorCalled, false, 'the default factory mode must not inspect remote tokenizers')
  assert.equal(offlineTemplateInspectorCalled, false, 'the default factory mode must not inspect remote templates')
  assert.equal(Object.hasOwn(offlineIndex, 'source_resolution'), false, 'default index shape must remain unchanged')
  assert.equal(Object.hasOwn(offlineIndex, 'header_inspection'), false, 'default index shape must not gain a header summary')
  assert.equal(Object.hasOwn(offlineIndex, 'tokenizer_inspection'), false, 'default index shape must not gain a tokenizer summary')
  assert.equal(Object.hasOwn(offlineIndex, 'template_preparation'), false, 'default index shape must not gain a template summary')

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
  let headerOnlyTokenizerCalls = 0
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
    tokenizerInspector: async () => {
      headerOnlyTokenizerCalls += 1
      throw new Error('must not run')
    },
  })
  assert.equal(headerResolverCalls, 2, '--inspect-header must imply one source preflight per row')
  assert.equal(headerCalls, 2, 'a row-local header failure must not abort the remaining batch')
  assert.equal(headerOnlyTokenizerCalls, 0, '--inspect-header alone must not run tokenizer qualification')
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

  const tokenizerOut = join(factoryOut, 'tokenizer-batch')
  let tokenizerResolverCalls = 0
  let tokenizerHeaderCalls = 0
  let tokenizerCalls = 0
  const tokenizerIndex = await runFactory({
    root,
    rows: [gemma.id, smol.id],
    outDir: tokenizerOut,
    inspectTokenizer: true,
    prefixBytes: 16,
    hfToken: 'test-secret-token',
    sourceResolver: async ({ repo, token }) => {
      tokenizerResolverCalls += 1
      assert.equal(token, 'test-secret-token')
      return repo === gemma.source.repo ? lockFor(gemma) : lockFor(smol)
    },
    headerInspector: async (sourceLock, options) => {
      tokenizerHeaderCalls += 1
      assert.equal(options.prefixBytes, 16)
      assert.equal(options.token, 'test-secret-token')
      if (sourceLock.repo === gemma.source.repo) {
        throw new Error('C:\\private\\header.gguf test-secret-token')
      }
      return headerReceiptFor(smol)
    },
    tokenizerInspector: async (sourceLock, options) => {
      tokenizerCalls += 1
      assert.equal(sourceLock.repo, smol.source.repo, 'metadata-blocked rows must not reach tokenizer probes')
      assert.equal(options.row.id, smol.id)
      assert.deepEqual(options.defaults, roster.defaults)
      assert.equal(options.sourceRoot, root)
      assert.equal(options.prefixBytes, 32 * 1024 * 1024)
      assert.equal(options.token, 'test-secret-token')
      assert.match(
        options.binary,
        process.platform === 'win32'
          ? /target[\\/]release[\\/]camelid\.exe$/
          : /target[\\/]release[\\/]camelid$/,
      )
      assert.match(
        options.llamaTokenize,
        process.platform === 'win32'
          ? /target[\\/]reference[\\/]llama\.cpp-b9632[\\/]bin[\\/]llama-tokenize\.exe$/
          : /target[\\/]reference[\\/]llama\.cpp-b9632[\\/]bin[\\/]llama-tokenize$/,
      )
      return tokenizerReceiptFor(smol)
    },
  })
  assert.equal(tokenizerResolverCalls, 2, '--inspect-tokenizer must imply source preflight')
  assert.equal(tokenizerHeaderCalls, 2, '--inspect-tokenizer must imply bounded header inspection')
  assert.equal(tokenizerCalls, 1, 'one metadata-blocked row must not abort a later tokenizer row')
  assert.deepEqual(tokenizerIndex.tokenizer_inspection, {
    mode: 'remote_immutable_prefix_tokenizer',
    per_row_byte_budget: 32 * 1024 * 1024,
    verified_receipt_requested_bytes: 32 * 1024 * 1024,
    verified_receipt_received_bytes: 32 * 1024 * 1024,
    counts: { pass: 1, fail: 0, blocked: 1, not_run: 0 },
  })
  const tokenizerBlockedReport = JSON.parse(await readFile(
    join(tokenizerOut, `${gemma.id}.json`),
    'utf8',
  ))
  assert.equal(tokenizerBlockedReport.stages.metadata.status, 'blocked')
  assert.equal(tokenizerBlockedReport.stages.tokenizer.status, 'blocked')
  assert.equal(
    tokenizerBlockedReport.stages.tokenizer.error_code,
    'tokenizer_metadata_preflight_blocked',
  )
  assert.equal(JSON.stringify(tokenizerBlockedReport).includes('test-secret-token'), false)
  assert.equal(JSON.stringify(tokenizerBlockedReport).includes('header.gguf'), false)
  assert.deepEqual(validateQualificationReport(tokenizerBlockedReport), [])

  const tokenizerPassedReport = JSON.parse(await readFile(
    join(tokenizerOut, `${smol.id}.json`),
    'utf8',
  ))
  assert.equal(tokenizerPassedReport.stages.artifact.status, 'blocked')
  assert.equal(tokenizerPassedReport.stages.metadata.status, 'pass')
  assert.equal(tokenizerPassedReport.stages.tokenizer.status, 'pass')
  assert.equal(tokenizerPassedReport.stages.template.status, 'blocked')
  assert.equal(tokenizerPassedReport.stages.load_smoke.status, 'blocked')
  assert.equal(tokenizerPassedReport.stages.parity.status, 'blocked')
  assert.equal(tokenizerPassedReport.stages.api_webui.status, 'blocked')
  assert.equal(tokenizerPassedReport.stages.context.status, 'blocked')
  assert.equal(tokenizerPassedReport.overall_status, 'blocked')
  assert.equal(Object.hasOwn(tokenizerPassedReport.stages.tokenizer, 'cases'), false)
  assert.equal(JSON.stringify(tokenizerPassedReport).includes('download_url'), false)
  assert.equal(JSON.stringify(tokenizerPassedReport).includes('test-secret-token'), false)
  assert.deepEqual(validateQualificationReport(tokenizerPassedReport), [])

  let localFailHeaderCalls = 0
  let localFailTokenizerCalls = 0
  const localFailOut = join(factoryOut, 'tokenizer-local-fail-preserved')
  const localFailBase = structuredClone(tokenizerPassedReport)
  localFailBase.stages.artifact = { status: 'pass' }
  localFailBase.stages.tokenizer = {
    status: 'fail',
    error_code: 'local_tokenizer_parity_mismatch',
    reason: 'the exact local tokenizer fixture diverged',
  }
  await runFactory({
    root,
    rows: [smol.id],
    outDir: localFailOut,
    inspectTokenizer: true,
    qualifier: async () => structuredClone(localFailBase),
    sourceResolver: async () => lockFor(smol),
    headerInspector: async () => { localFailHeaderCalls += 1 },
    tokenizerInspector: async () => { localFailTokenizerCalls += 1; return tokenizerReceiptFor(smol) },
  })
  assert.equal(localFailHeaderCalls, 0, 'an exact local artifact keeps its authoritative metadata lane')
  assert.equal(localFailTokenizerCalls, 0, 'remote pass must not overwrite an authoritative local tokenizer failure')
  const localFailReport = JSON.parse(await readFile(
    join(localFailOut, `${smol.id}.json`),
    'utf8',
  ))
  assert.equal(localFailReport.stages.tokenizer.status, 'fail')
  assert.equal(localFailReport.stages.tokenizer.error_code, 'local_tokenizer_parity_mismatch')
  assert.equal(localFailReport.overall_status, 'fail')

  let sourceBlockedHeaderCalls = 0
  let sourceBlockedTokenizerCalls = 0
  const sourceBlockedOut = join(factoryOut, 'tokenizer-source-blocked')
  await runFactory({
    root,
    rows: [smol.id],
    outDir: sourceBlockedOut,
    inspectTokenizer: true,
    sourceResolver: async () => { throw new Error('C:\\private\\source test-secret-token') },
    headerInspector: async () => { sourceBlockedHeaderCalls += 1 },
    tokenizerInspector: async () => { sourceBlockedTokenizerCalls += 1 },
  })
  assert.equal(sourceBlockedHeaderCalls, 0)
  assert.equal(sourceBlockedTokenizerCalls, 0)
  const sourceBlockedTokenizerReport = JSON.parse(await readFile(
    join(sourceBlockedOut, `${smol.id}.json`),
    'utf8',
  ))
  assert.equal(sourceBlockedTokenizerReport.stages.tokenizer.error_code, 'tokenizer_source_preflight_blocked')
  assert.equal(JSON.stringify(sourceBlockedTokenizerReport).includes('test-secret-token'), false)

  let unsupportedTokenizerCalls = 0
  const unsupportedOut = join(factoryOut, 'tokenizer-pack-unavailable')
  await runFactory({
    root,
    rows: [qwen.id],
    outDir: unsupportedOut,
    inspectTokenizer: true,
    prefixBytes: 16,
    sourceResolver: async () => lockFor(qwen),
    headerInspector: async () => headerReceiptFor(qwen),
    tokenizerInspector: async () => { unsupportedTokenizerCalls += 1 },
  })
  assert.equal(unsupportedTokenizerCalls, 0, 'unsupported rows must block before tokenizer inspection')
  const unsupportedTokenizerReport = JSON.parse(await readFile(
    join(unsupportedOut, `${qwen.id}.json`),
    'utf8',
  ))
  assert.equal(unsupportedTokenizerReport.stages.metadata.status, 'pass')
  assert.equal(unsupportedTokenizerReport.stages.tokenizer.status, 'blocked')
  assert.equal(unsupportedTokenizerReport.stages.tokenizer.error_code, 'tokenizer_pack_unavailable')
  assert.deepEqual(validateQualificationReport(unsupportedTokenizerReport), [])

  let exclusiveResolverCalls = 0
  let exclusiveTemplateCalls = 0
  await assert.rejects(
    runFactory({
      root,
      rows: [smol.id],
      outDir: join(factoryOut, 'template-mutually-exclusive'),
      inspectHeader: true,
      inspectTemplate: true,
      sourceResolver: async () => {
        exclusiveResolverCalls += 1
        return lockFor(smol)
      },
      templateInspector: async () => {
        exclusiveTemplateCalls += 1
        return templatePackForFactory()
      },
    }),
    /cannot be combined/,
  )
  assert.equal(exclusiveResolverCalls, 0, 'mutually exclusive lanes must fail before source resolution')
  assert.equal(exclusiveTemplateCalls, 0, 'mutually exclusive lanes must fail before a prefix probe')

  let unsupportedTemplateResolverCalls = 0
  let unsupportedTemplateCalls = 0
  const unsupportedTemplateOut = join(factoryOut, 'template-pack-unavailable')
  const unsupportedTemplateIndex = await runFactory({
    root,
    rows: [qwen.id],
    outDir: unsupportedTemplateOut,
    inspectTemplate: true,
    sourceResolver: async () => {
      unsupportedTemplateResolverCalls += 1
      return lockFor(qwen)
    },
    templateInspector: async () => {
      unsupportedTemplateCalls += 1
      return templatePackForFactory()
    },
  })
  assert.equal(
    unsupportedTemplateResolverCalls,
    0,
    'unsupported template rows must block before network source resolution',
  )
  assert.equal(unsupportedTemplateCalls, 0, 'unsupported rows must block before template inspection')
  assert.deepEqual(unsupportedTemplateIndex.source_resolution, {
    mode: 'live_huggingface',
    counts: { pass: 0, fail: 0, blocked: 0, not_run: 1 },
  })
  assert.equal(
    unsupportedTemplateIndex.rows[0].first_unresolved_stage,
    'artifact',
    'a skipped source preflight must not be reported as the first unresolved source gate',
  )
  assert.deepEqual(unsupportedTemplateIndex.template_preparation, {
    mode: 'remote_immutable_prefix_smollm3_template_preparation',
    per_row_byte_budget: 32 * 1024 * 1024,
    verified_receipt_requested_bytes: 0,
    verified_receipt_received_bytes: 0,
    counts: { pass: 0, fail: 0, blocked: 1, not_run: 0 },
    preparation_results: { pass: 0, fail: 0, blocked: 1, not_run: 0 },
    runtime_template_gate: 'blocked',
    support_claim: false,
  })
  const unsupportedTemplateReport = JSON.parse(await readFile(
    join(unsupportedTemplateOut, `${qwen.id}.json`),
    'utf8',
  ))
  assert.equal(unsupportedTemplateReport.stages.template.status, 'blocked')
  assert.equal(unsupportedTemplateReport.stages.template.error_code, 'template_pack_unavailable')
  assert.deepEqual(validateQualificationReport(unsupportedTemplateReport), [])

  const cleanQualifier = async (options) => {
    const report = await qualify(options)
    report.source_head = currentSourceHead
    report.source_dirty = false
    report.source_inspection = 'observed'
    return report
  }
  for (const localTemplateStatus of ['pass', 'fail']) {
    let unsupportedLocalResolverCalls = 0
    let unsupportedLocalTemplateCalls = 0
    const unsupportedLocalOut = join(
      factoryOut,
      `template-unsupported-local-${localTemplateStatus}`,
    )
    const unsupportedLocalIndex = await runFactory({
      root,
      rows: [qwen.id],
      outDir: unsupportedLocalOut,
      inspectTemplate: true,
      qualifier: async (options) => {
        const report = await qualify(options)
        report.stages.artifact = { status: 'pass' }
        report.stages.template = localTemplateStatus === 'pass'
          ? { status: 'pass' }
          : {
            status: 'fail',
            error_code: 'local_template_mismatch',
            reason: 'the exact local artifact failed its authoritative template comparison',
          }
        return report
      },
      sourceResolver: async () => {
        unsupportedLocalResolverCalls += 1
        return lockFor(qwen)
      },
      templateInspector: async () => {
        unsupportedLocalTemplateCalls += 1
        return templatePackForFactory()
      },
    })
    assert.equal(unsupportedLocalResolverCalls, 0)
    assert.equal(unsupportedLocalTemplateCalls, 0)
    assert.deepEqual(unsupportedLocalIndex.template_preparation.preparation_results, {
      pass: 0,
      fail: 0,
      blocked: 0,
      not_run: 1,
    })
    assert.deepEqual(unsupportedLocalIndex.source_resolution, {
      mode: 'live_huggingface',
      counts: { pass: 0, fail: 0, blocked: 0, not_run: 1 },
    })
    assert.equal(
      unsupportedLocalIndex.rows[0].first_unresolved_stage,
      'metadata',
      'an authoritative local template result must not imply a skipped live source failure',
    )
    const unsupportedLocalReport = JSON.parse(await readFile(
      join(unsupportedLocalOut, `${qwen.id}.json`),
      'utf8',
    ))
    assert.equal(unsupportedLocalReport.stages.template.status, localTemplateStatus)
    assert.notEqual(unsupportedLocalReport.stages.template.error_code, 'template_pack_unavailable')
  }
  const trustedTemplateBinaryInspector = async (binary, { sourceRoot }) => {
    assert.equal(sourceRoot, root)
    assert.match(
      binary,
      process.platform === 'win32'
        ? /target[\\/]release[\\/]camelid\.exe$/
        : /target[\\/]release[\\/]camelid$/,
    )
    return templateBinaryIdentityForFactory()
  }
  let templateResolverCalls = 0
  let templateCalls = 0
  const templateOut = join(factoryOut, 'template-preparation-pass')
  const templateIndex = await runFactory({
    root,
    rows: [smol.id],
    outDir: templateOut,
    inspectTemplate: true,
    hfToken: 'test-secret-token',
    qualifier: cleanQualifier,
    sourceResolver: async ({ repo, file, revision, token }) => {
      templateResolverCalls += 1
      assert.equal(repo, smol.source.repo)
      assert.equal(file, smol.source.file)
      assert.equal(revision, smol.source.revision)
      assert.equal(token, 'test-secret-token')
      return lockFor(smol)
    },
    templateBinaryInspector: trustedTemplateBinaryInspector,
    templateInspector: async (options) => {
      templateCalls += 1
      assert.equal(options.root, root)
      assert.equal(options.rosterPath, resolve(root, 'qa/model-qualification/phase1-roster.json'))
      assert.equal(options.prefixBytes, 32 * 1024 * 1024)
      assert.equal(options.token, 'test-secret-token')
      assert.deepEqual(options.initialLock, lockFor(smol))
      assert.equal(typeof options.sourceResolver, 'function')
      assert.match(
        options.binary,
        process.platform === 'win32'
          ? /target[\\/]release[\\/]camelid\.exe$/
          : /target[\\/]release[\\/]camelid$/,
      )
      assert.match(
        options.analyzer,
        process.platform === 'win32'
          ? /target[\\/]reference[\\/]llama\.cpp-b9632[\\/]bin[\\/]llama-template-analysis\.exe$/
          : /target[\\/]reference[\\/]llama\.cpp-b9632[\\/]bin[\\/]llama-template-analysis$/,
      )
      return templatePackForFactory()
    },
  })
  assert.equal(templateResolverCalls, 1, 'factory must retain one exact-row source lock for the template lane')
  assert.equal(templateCalls, 1)
  assert.deepEqual(templateIndex.source_resolution, {
    mode: 'live_huggingface',
    counts: { pass: 1, fail: 0, blocked: 0 },
  })
  assert.deepEqual(templateIndex.template_preparation, {
    mode: 'remote_immutable_prefix_smollm3_template_preparation',
    per_row_byte_budget: 32 * 1024 * 1024,
    verified_receipt_requested_bytes: 32 * 1024 * 1024,
    verified_receipt_received_bytes: 32 * 1024 * 1024,
    counts: { pass: 0, fail: 0, blocked: 1, not_run: 0 },
    preparation_results: { pass: 1, fail: 0, blocked: 0, not_run: 0 },
    runtime_template_gate: 'blocked',
    support_claim: false,
  })
  const templateReport = JSON.parse(await readFile(join(templateOut, `${smol.id}.json`), 'utf8'))
  assert.equal(templateReport.stages.artifact.status, 'blocked')
  assert.equal(templateReport.stages.template.status, 'blocked')
  assert.equal(templateReport.stages.template.error_code, 'smollm3_template_runtime_hold')
  assert.equal(templateReport.stages.template.preparation.status, 'pass')
  assert.equal(templateReport.stages.template.scope.template_gate, 'blocked')
  assert.equal(templateReport.stages.template.scope.runtime_chat, 'blocked')
  assert.equal(templateReport.stages.template.scope.support_claim, false)
  assert.equal(templateReport.overall_status, 'blocked')
  const serializedTemplateReport = JSON.stringify(templateReport)
  for (const forbidden of [
    'test-secret-token',
    'download_url',
    'source_template',
    'normalized_prompt"',
    'messages',
    '<|im_start|>',
    'llama-template-analysis.exe',
  ]) {
    assert.equal(serializedTemplateReport.includes(forbidden), false)
  }
  assert.deepEqual(validateQualificationReport(templateReport), [])

  const unavailableTemplateIdentityOut = join(factoryOut, 'template-binary-identity-unavailable')
  await runFactory({
    root,
    rows: [smol.id],
    outDir: unavailableTemplateIdentityOut,
    inspectTemplate: true,
    qualifier: cleanQualifier,
    sourceResolver: async () => lockFor(smol),
    templateInspector: async () => templatePackForFactory(),
    templateBinaryInspector: async () => null,
  })
  const unavailableTemplateIdentityReport = JSON.parse(await readFile(
    join(unavailableTemplateIdentityOut, `${smol.id}.json`),
    'utf8',
  ))
  assert.equal(unavailableTemplateIdentityReport.stages.template.status, 'blocked')
  assert.equal(
    unavailableTemplateIdentityReport.stages.template.error_code,
    'template_inspector_identity_unavailable',
  )
  assert.deepEqual(validateQualificationReport(unavailableTemplateIdentityReport), [])

  let untrackedNoiseTemplateCalls = 0
  const untrackedNoiseTemplateOut = join(factoryOut, 'template-untracked-noise')
  await runFactory({
    root,
    rows: [smol.id],
    outDir: untrackedNoiseTemplateOut,
    inspectTemplate: true,
    qualifier: async (options) => {
      const report = await cleanQualifier(options)
      report.source_dirty = true
      return report
    },
    sourceResolver: async () => lockFor(smol),
    templateBinaryInspector: trustedTemplateBinaryInspector,
    templateInspector: async () => {
      untrackedNoiseTemplateCalls += 1
      return templatePackForFactory()
    },
  })
  assert.equal(
    untrackedNoiseTemplateCalls,
    1,
    'runner dirty state includes untracked files; the harness owns the tracked-clean pre-fetch gate',
  )
  const untrackedNoiseTemplateReport = JSON.parse(await readFile(
    join(untrackedNoiseTemplateOut, `${smol.id}.json`),
    'utf8',
  ))
  assert.equal(untrackedNoiseTemplateReport.source_dirty, true)
  assert.equal(untrackedNoiseTemplateReport.stages.template.status, 'blocked')
  assert.equal(untrackedNoiseTemplateReport.stages.template.preparation.status, 'pass')

  let authoritativeTemplateCalls = 0
  const authoritativeTemplateOut = join(factoryOut, 'template-authoritative-local-fail')
  await runFactory({
    root,
    rows: [smol.id],
    outDir: authoritativeTemplateOut,
    inspectTemplate: true,
    qualifier: async (options) => {
      const report = await cleanQualifier(options)
      report.stages.artifact = { status: 'pass' }
      report.stages.template = {
        status: 'fail',
        error_code: 'local_template_mismatch',
        reason: 'the exact local artifact failed its authoritative template comparison',
      }
      return report
    },
    sourceResolver: async () => lockFor(smol),
    templateInspector: async () => {
      authoritativeTemplateCalls += 1
      return templatePackForFactory()
    },
  })
  assert.equal(
    authoritativeTemplateCalls,
    0,
    'preparation evidence must not overwrite an authoritative local template result',
  )
  const authoritativeTemplateReport = JSON.parse(await readFile(
    join(authoritativeTemplateOut, `${smol.id}.json`),
    'utf8',
  ))
  assert.equal(authoritativeTemplateReport.stages.template.status, 'fail')
  assert.equal(authoritativeTemplateReport.stages.template.error_code, 'local_template_mismatch')

  let failedTemplateBatchCalls = 0
  let failedTemplateBatchResolverCalls = 0
  const failedTemplateBatchOut = join(factoryOut, 'template-row-local-failure')
  const failedTemplateBatchIndex = await runFactory({
    root,
    rows: [smol.id, qwenMoe.id],
    outDir: failedTemplateBatchOut,
    inspectTemplate: true,
    qualifier: cleanQualifier,
    sourceResolver: async () => {
      failedTemplateBatchResolverCalls += 1
      return lockFor(smol)
    },
    templateInspector: async () => {
      failedTemplateBatchCalls += 1
      throw new Error('test-secret-token at C:\\private\\template.gguf')
    },
  })
  assert.equal(failedTemplateBatchResolverCalls, 1, 'unsupported rows must skip source lookup in a mixed batch')
  assert.equal(failedTemplateBatchCalls, 1, 'row-local analyzer failure must not abort a later row')
  assert.equal(failedTemplateBatchIndex.rows.length, 2)
  assert.deepEqual(failedTemplateBatchIndex.source_resolution, {
    mode: 'live_huggingface',
    counts: { pass: 1, fail: 0, blocked: 0, not_run: 1 },
  })
  assert.deepEqual(failedTemplateBatchIndex.template_preparation.counts, {
    pass: 0,
    fail: 0,
    blocked: 2,
    not_run: 0,
  })
  const failedTemplateReport = JSON.parse(await readFile(
    join(failedTemplateBatchOut, `${smol.id}.json`),
    'utf8',
  ))
  const continuedTemplateReport = JSON.parse(await readFile(
    join(failedTemplateBatchOut, `${qwenMoe.id}.json`),
    'utf8',
  ))
  assert.equal(failedTemplateReport.stages.template.error_code, 'smollm3_template_qualification_error')
  assert.equal(continuedTemplateReport.stages.template.error_code, 'template_pack_unavailable')
  assert.equal(JSON.stringify(failedTemplateBatchIndex).includes('test-secret-token'), false)
  assert.equal(JSON.stringify(failedTemplateReport).includes('template.gguf'), false)
  assert.equal(JSON.stringify(failedTemplateReport).includes('C:\\private'), false)
  assert.deepEqual(validateQualificationReport(failedTemplateReport), [])
  assert.deepEqual(validateQualificationReport(continuedTemplateReport), [])

  let candidateResolverCalls = 0
  let candidateHeaderCalls = 0
  const candidateOut = join(factoryOut, 'selector-candidate-pass')
  const candidateReport = await runFactory({
    root,
    candidate: candidateSelection,
    outDir: candidateOut,
    inspectHeader: true,
    prefixBytes: 1024,
    hfToken: 'test-secret-token',
    now: () => new Date('2026-08-10T20:00:00.000Z'),
    candidateWorkspaceInspector: async () => ({
      source_head: currentSourceHead,
      source_dirty: true,
      source_tracked_dirty: false,
      source_inspection: 'observed',
    }),
    sourceResolver: async (selection) => {
      candidateResolverCalls += 1
      assert.deepEqual(selection, { ...candidateSelection, token: 'test-secret-token' })
      return candidateLock()
    },
    headerInspector: async (lock, options) => {
      candidateHeaderCalls += 1
      assert.deepEqual(lock, candidateLock())
      assert.equal(options.rowId, candidateId)
      assert.equal(options.prefixBytes, 99, 'candidate range must remain strictly smaller than the artifact')
      assert.equal(options.token, 'test-secret-token')
      assert.equal(options.sourceRoot, root)
      return candidateHeaderReceipt()
    },
  })
  assert.equal(candidateResolverCalls, 1)
  assert.equal(candidateHeaderCalls, 1)
  assert.equal(candidateReport.qualification_mode, 'unrostered_hf_selector')
  assert.equal(candidateReport.row_id, candidateId)
  assert.equal(candidateReport.candidate.identity_mode, 'public_selector_digest')
  assert.equal(candidateReport.candidate.selector_sha256, candidateSelectorSha256)
  assert.equal(candidateReport.candidate.selector_redacted, true)
  assert.equal(candidateReport.stages.source.status, 'pass')
  assert.equal(candidateReport.stages.source.requested_revision, null)
  assert.equal(candidateReport.stages.metadata.status, 'pass')
  assert.equal(candidateReport.stages.metadata.assessment, 'bounded_header_descriptor_inspection_only')
  assert.equal(candidateReport.stages.metadata.range.received_bytes, 99)
  assert.deepEqual(candidateReport.stages.metadata.observed.tensor_type_counts, { F32: 1, Q8_0: 4 })
  assert.equal(candidateReport.source_dirty, true)
  assert.equal(candidateReport.source_tracked_dirty, false)
  assert.equal(candidateReport.stages.artifact.status, 'blocked')
  assert.equal(candidateReport.stages.tokenizer.status, 'blocked')
  assert.equal(candidateReport.stages.template.status, 'blocked')
  assert.equal(candidateReport.stages.load_smoke.status, 'blocked')
  assert.equal(candidateReport.stages.parity.status, 'blocked')
  assert.equal(candidateReport.stages.api_webui.status, 'blocked')
  assert.equal(candidateReport.stages.context.status, 'blocked')
  assert.equal(candidateReport.overall_status, 'blocked')
  assert.equal(candidateReport.support_claim, false)
  const serializedCandidate = JSON.stringify(candidateReport)
  assert.equal(serializedCandidate.includes('test-secret-token'), false)
  assert.equal(serializedCandidate.includes('download_url'), false)
  assert.equal(serializedCandidate.includes('source_template'), false)
  assert.equal(serializedCandidate.includes('test receipt'), false)
  assert.deepEqual(validateQualificationReport(candidateReport), [])
  assert.deepEqual(
    JSON.parse(await readFile(join(candidateOut, `${candidateId}-report.json`), 'utf8')),
    candidateReport,
  )

  let privateCandidateHeaderCalled = false
  const privateCandidateReport = await runCandidateFactory({
    root,
    candidate: candidateSelection,
    outDir: join(factoryOut, 'selector-private-blocked'),
    inspectHeader: true,
    candidateRunIdentity: opaqueRunIdentity,
    candidateWorkspaceInspector: async () => ({
      source_head: currentSourceHead,
      source_dirty: false,
      source_tracked_dirty: false,
      source_inspection: 'observed',
    }),
    sourceResolver: async () => candidateLock({
      access: { gated: true, private: true, disabled: false },
    }),
    headerInspector: async () => {
      privateCandidateHeaderCalled = true
      throw new Error('must not inspect a private source')
    },
  })
  assert.equal(privateCandidateHeaderCalled, false)
  assert.equal(privateCandidateReport.row_id, opaqueCandidateId)
  assert.equal(privateCandidateReport.candidate.identity_mode, 'opaque_run')
  assert.equal(privateCandidateReport.candidate.run_id, opaqueRunIdentity)
  assert.equal(Object.hasOwn(privateCandidateReport.candidate, 'selector_sha256'), false)
  assert.equal(Object.hasOwn(privateCandidateReport.candidate, 'selector_id'), false)
  assert.equal(privateCandidateReport.stages.source.error_code, 'private_source_not_persisted')
  assert.equal(privateCandidateReport.stages.metadata.error_code, 'header_source_preflight_blocked')
  assert.equal(Object.hasOwn(privateCandidateReport.stages.source, 'repo'), false)
  assert.equal(JSON.stringify(privateCandidateReport).includes(candidateSelection.repo), false)
  assert.deepEqual(validateQualificationReport(privateCandidateReport), [])
  assert.deepEqual(
    JSON.parse(await readFile(
      join(factoryOut, 'selector-private-blocked', `${opaqueCandidateId}-report.json`),
      'utf8',
    )),
    privateCandidateReport,
  )

  const lookupRunIdentity = 'fedcba98-7654-4321-8fed-cba987654321'
  const lookupOpaqueId = 'hf_candidate_run_fedcba98765443218fedcba987654321'
  const lookupFailureReport = await runCandidateFactory({
    root,
    candidate: candidateSelection,
    outDir: join(factoryOut, 'selector-lookup-blocked'),
    inspectHeader: true,
    candidateRunIdentity: lookupRunIdentity,
    candidateWorkspaceInspector: async () => ({
      source_head: currentSourceHead,
      source_dirty: false,
      source_tracked_dirty: false,
      source_inspection: 'observed',
    }),
    sourceResolver: async () => {
      throw new Error(`C:\\private\\${candidateSelection.file} hf_secret`)
    },
  })
  assert.equal(lookupFailureReport.row_id, lookupOpaqueId)
  assert.equal(lookupFailureReport.candidate.identity_mode, 'opaque_run')
  assert.equal(Object.hasOwn(lookupFailureReport.candidate, 'selector_sha256'), false)
  assert.equal(JSON.stringify(lookupFailureReport).includes(candidateSelection.repo), false)
  assert.equal(JSON.stringify(lookupFailureReport).includes(candidateSelection.file), false)
  assert.equal(JSON.stringify(lookupFailureReport).includes('hf_secret'), false)
  assert.deepEqual(validateQualificationReport(lookupFailureReport), [])

  const headerFailureReport = await runCandidateFactory({
    root,
    candidate: candidateSelection,
    outDir: join(factoryOut, 'selector-header-blocked'),
    inspectHeader: true,
    hfToken: 'test-secret-token',
    candidateWorkspaceInspector: async () => ({
      source_head: currentSourceHead,
      source_dirty: false,
      source_tracked_dirty: false,
      source_inspection: 'observed',
    }),
    sourceResolver: async () => candidateLock(),
    headerInspector: async () => {
      throw new Error('C:\\private\\header.gguf test-secret-token')
    },
  })
  assert.equal(headerFailureReport.stages.metadata.status, 'blocked')
  assert.equal(headerFailureReport.stages.metadata.error_code, 'header_inspection_error')
  assert.equal(JSON.stringify(headerFailureReport).includes('test-secret-token'), false)
  assert.equal(JSON.stringify(headerFailureReport).includes('C:\\private'), false)
  assert.equal(JSON.stringify(headerFailureReport).includes('header.gguf'), false)
  assert.deepEqual(validateQualificationReport(headerFailureReport), [])

  let tinyHeaderCalled = false
  const tinyCandidateReport = await runCandidateFactory({
    root,
    candidate: candidateSelection,
    outDir: join(factoryOut, 'selector-tiny-blocked'),
    inspectHeader: true,
    candidateWorkspaceInspector: async () => ({
      source_head: currentSourceHead,
      source_dirty: false,
      source_tracked_dirty: false,
      source_inspection: 'observed',
    }),
    sourceResolver: async () => candidateLock({ size_bytes: 1 }),
    headerInspector: async () => { tinyHeaderCalled = true },
  })
  assert.equal(tinyHeaderCalled, false)
  assert.equal(tinyCandidateReport.stages.source.status, 'pass')
  assert.equal(tinyCandidateReport.stages.metadata.error_code, 'header_partial_range_unavailable')
  assert.deepEqual(validateQualificationReport(tinyCandidateReport), [])

  let unknownHeadHeaderCalled = false
  const unknownHeadReport = await runCandidateFactory({
    root,
    candidate: candidateSelection,
    outDir: join(factoryOut, 'selector-head-blocked'),
    inspectHeader: true,
    candidateWorkspaceInspector: async () => ({
      source_head: null,
      source_dirty: null,
      source_tracked_dirty: null,
      source_inspection: 'unknown',
    }),
    sourceResolver: async () => candidateLock(),
    headerInspector: async () => { unknownHeadHeaderCalled = true },
  })
  assert.equal(unknownHeadHeaderCalled, false)
  assert.equal(unknownHeadReport.stages.metadata.error_code, 'header_source_head_unavailable')
  assert.deepEqual(validateQualificationReport(unknownHeadReport), [])

  let trackedDirtyHeaderCalled = false
  const trackedDirtyReport = await runCandidateFactory({
    root,
    candidate: candidateSelection,
    outDir: join(factoryOut, 'selector-tracked-dirty-blocked'),
    inspectHeader: true,
    candidateWorkspaceInspector: async () => ({
      source_head: currentSourceHead,
      source_dirty: true,
      source_tracked_dirty: true,
      source_inspection: 'observed',
    }),
    sourceResolver: async () => candidateLock(),
    headerInspector: async () => { trackedDirtyHeaderCalled = true },
  })
  assert.equal(trackedDirtyHeaderCalled, false)
  assert.equal(trackedDirtyReport.stages.metadata.error_code, 'header_source_tracked_dirty')
  assert.deepEqual(validateQualificationReport(trackedDirtyReport), [])

  for (const conflict of [
    { rows: [] },
    { roster: undefined },
    { modelsDir: '' },
    { artifact: '' },
    { inspectTokenizer: false },
    { inspectTemplate: false },
    { runSmoke: false },
    { runGeneration: false },
    { promptLimit: null },
    { llamaTokenize: '' },
    { llamaTemplateAnalysis: '' },
  ]) {
    let conflictResolverCalled = false
    await assert.rejects(
      runCandidateFactory({
        root,
        candidate: candidateSelection,
        outDir: join(factoryOut, 'selector-conflict'),
        inspectHeader: true,
        ...conflict,
        sourceResolver: async () => {
          conflictResolverCalled = true
          return candidateLock()
        },
      }),
      /cannot be combined/,
    )
    assert.equal(conflictResolverCalled, false)
  }
  await assert.rejects(
    runCandidateFactory({ root, candidate: candidateSelection }),
    /requires --inspect-header/,
  )
  await assert.rejects(
    runCandidateFactory({
      root,
      candidate: candidateSelection,
      inspectHeader: true,
      candidateRunIdentity: 'not-a-v4-uuid',
      sourceResolver: async () => { throw new Error('offline') },
    }),
    /version-4 UUID/,
  )
  let invalidSelectorResolverCalled = false
  await assert.rejects(
    runCandidateFactory({
      root,
      candidate: { repo: 'org/model', file: '../private.gguf', revision: null },
      inspectHeader: true,
      sourceResolver: async () => { invalidSelectorResolverCalled = true },
    }),
    /--file/,
  )
  assert.equal(invalidSelectorResolverCalled, false)

  let rosteredSelectorResolverCalled = false
  await assert.rejects(
    runCandidateFactory({
      root,
      candidate: {
        repo: qwen.source.repo,
        file: qwen.source.file,
        revision: qwen.source.revision,
      },
      inspectHeader: true,
      sourceResolver: async () => { rosteredSelectorResolverCalled = true },
    }),
    /already present in the Phase 1 roster/,
  )
  assert.equal(rosteredSelectorResolverCalled, false)

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
