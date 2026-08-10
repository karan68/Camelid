#!/usr/bin/env node
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { basename, isAbsolute, relative, resolve, sep } from 'node:path'
import { pathToFileURL } from 'node:url'
import { validateRoster } from './check-model-qualification-roster.mjs'
import {
  DEFAULT_PREFIX_BYTES,
  MAX_PREFIX_BYTES,
  classifyHeaderInspectionError,
  inspectRemoteHeader,
  normalizePrefixBytes,
} from './hf-qualification-header.mjs'
import {
  DEFAULT_TOKENIZER_PREFIX_BYTES,
  assessTokenizerReceipt,
  classifyTokenizerQualificationError,
  inspectRemoteTokenizer,
  tokenizerPackAvailable,
  tokenizerPrefixBytesForRow,
} from './hf-qualification-tokenizer.mjs'
import { resolveHfSource, validateLockAgainstSelection } from './hf-qualification-source.mjs'
import { deriveOverall, qualify, redactLocalPaths } from './model-qualification-runner.mjs'

const GGUF_FILE_TYPE_LABELS = new Map([
  [0, 'F32'], [1, 'F16'], [2, 'Q4_0'], [3, 'Q4_1'], [7, 'Q8_0'],
  [8, 'Q5_0'], [9, 'Q5_1'], [10, 'Q2_K'], [11, 'Q3_K_S'],
  [12, 'Q3_K_M'], [13, 'Q3_K_L'], [14, 'Q4_K_S'], [15, 'Q4_K_M'],
  [16, 'Q5_K_S'], [17, 'Q5_K_M'], [18, 'Q6_K'], [32, 'BF16'],
  [36, 'TQ1_0'], [37, 'TQ2_0'], [39, 'NVFP4'], [40, 'Q1_0'], [41, 'Q2_0'],
])

function parseArgs(argv) {
  const args = new Map()
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    if (!arg.startsWith('--')) continue
    const [key, inline] = arg.slice(2).split('=', 2)
    const next = argv[index + 1]
    const value = inline ?? (next && !next.startsWith('--') ? argv[++index] : 'true')
    args.set(key, value)
  }
  return args
}

function selectRows(roster, requested) {
  if (!requested?.length) return [...roster.rows].sort((a, b) => a.priority - b.priority)
  const requestedSet = new Set(requested)
  const unknown = [...requestedSet].filter((id) => !roster.rows.some((row) => row.id === id))
  if (unknown.length) throw new Error(`unknown qualification rows: ${unknown.join(', ')}`)
  return roster.rows
    .filter((row) => requestedSet.has(row.id))
    .sort((a, b) => a.priority - b.priority)
}

function artifactForRow(row, modelsDir, explicitArtifact = null) {
  if (explicitArtifact) return resolve(explicitArtifact)
  const filename = row.identity.gguf_filename || row.source.file
  if (!modelsDir || !filename) return null
  return resolve(modelsDir, basename(filename))
}

function defaultCamelidBinary() {
  return process.platform === 'win32'
    ? 'target/release/camelid.exe'
    : 'target/release/camelid'
}

function defaultLlamaTokenizerBinary() {
  return process.platform === 'win32'
    ? 'target/reference/llama.cpp-b9632/bin/llama-tokenize.exe'
    : 'target/reference/llama.cpp-b9632/bin/llama-tokenize'
}

function firstUnresolvedStage(report, gateOrder) {
  if (report.stages.artifact?.status !== 'pass') return 'artifact'
  return gateOrder.find((name) => report.stages[name]?.status !== 'pass') || null
}

function publicRosterLabel(root, rosterPath) {
  const candidate = relative(root, rosterPath)
  if (!candidate || candidate === '.') return '<workspace-roster>'
  if (isAbsolute(candidate) || candidate === '..' || candidate.startsWith('..' + sep)) {
    return '<external-roster>'
  }
  return candidate.split(sep).join('/')
}

function sourceSelectionForRow(row) {
  return {
    row_id: row.id,
    repo: row.source.repo,
    file: row.source.file,
    revision: row.source.revision,
    expected: {
      size_bytes: row.identity.size_bytes,
      sha256: row.identity.sha256,
      license: row.source.license,
    },
  }
}

function sourceLookupErrorCode(error) {
  const name = typeof error?.name === 'string' ? error.name.toLowerCase() : ''
  const code = typeof error?.code === 'string' ? error.code.toLowerCase() : ''
  const message = error instanceof Error ? error.message : String(error)
  if (name === 'aborterror' || name === 'timeouterror' || ['etimedout', 'esockettimedout'].includes(code)) {
    return 'timeout'
  }
  const httpStatus = /(?:request failed|http)\D*(\d{3})/i.exec(message)?.[1]
  if (httpStatus) return `http_${httpStatus}`
  if (/immutable 40-character revision/i.test(message)) return 'missing_immutable_revision'
  if (/file .* is absent from/i.test(message)) return 'file_not_found_at_revision'
  if (/positive byte size/i.test(message)) return 'missing_file_size'
  if (/LFS SHA-256/i.test(message)) return 'missing_lfs_sha256'
  if (['econnreset', 'econnrefused', 'enotfound', 'eai_again'].includes(code)) return 'network_error'
  return 'source_lookup_error'
}

function sourceLockMismatchFields(lock, selected) {
  const mismatches = []
  for (const field of ['repo', 'file', 'revision']) {
    if (lock?.[field] !== selected[field]) mismatches.push(field)
  }
  for (const field of ['size_bytes', 'sha256']) {
    if (lock?.[field] !== selected.expected?.[field]) mismatches.push(field)
  }
  const expectedLicense = selected.expected?.license
  if (expectedLicense
    && String(expectedLicense).trim().toLowerCase() !== String(lock?.license || '').trim().toLowerCase()) {
    mismatches.push('license')
  }
  return mismatches
}

async function resolveSourcePreflight(row, {
  resolver = resolveHfSource,
  token = process.env.HF_TOKEN || null,
} = {}) {
  const selected = sourceSelectionForRow(row)
  const missing = [
    ['repo', selected.repo],
    ['file', selected.file],
    ['revision', selected.revision],
    ['size_bytes', selected.expected.size_bytes],
    ['sha256', selected.expected.sha256],
    ['license', selected.expected.license],
  ].filter(([, value]) => value === null || value === undefined || value === '')
    .map(([field]) => field)
  if (missing.length) {
    return {
      lock: null,
      stage: {
        status: 'blocked',
        resolution: 'live_huggingface',
        reason: `roster source selector is not fully pinned: ${missing.join(', ')}`,
      },
    }
  }

  let lock
  try {
    lock = await resolver({
      repo: selected.repo,
      file: selected.file,
      revision: selected.revision,
      token,
    })
  } catch (error) {
    const errorCode = sourceLookupErrorCode(error)
    return {
      lock: null,
      stage: {
        status: 'blocked',
        resolution: 'live_huggingface',
        error_code: errorCode,
        reason: `live Hugging Face source resolution could not complete (${errorCode})`,
      },
    }
  }
  if (!lock || typeof lock !== 'object' || Array.isArray(lock)) {
    return {
      lock: null,
      stage: {
        status: 'blocked',
        resolution: 'live_huggingface',
        reason: 'live Hugging Face source resolution returned no source-lock object',
      },
    }
  }

  const access = {
    gated: Boolean(lock.access?.gated),
    private: Boolean(lock.access?.private),
    disabled: Boolean(lock.access?.disabled),
  }

  try {
    validateLockAgainstSelection(lock, selected)
  } catch {
    const mismatchFields = sourceLockMismatchFields(lock, selected)
    return {
      lock: null,
      stage: {
        status: 'fail',
        resolution: 'live_huggingface',
        reason: `resolved source does not match roster row ${row.id}: ${mismatchFields.join(', ') || 'unknown identity field'}`,
        expected: {
          repo: selected.repo,
          file: selected.file,
          revision: selected.revision,
          size_bytes: selected.expected.size_bytes,
          sha256: selected.expected.sha256,
          license: selected.expected.license,
          access,
        },
      },
    }
  }

  return {
    lock,
    stage: {
      status: 'pass',
      resolution: 'live_huggingface',
      repo: lock.repo,
      file: lock.file,
      revision: lock.revision,
      size_bytes: lock.size_bytes,
      sha256: lock.sha256,
      license: lock.license,
      access,
    },
  }
}

async function resolveSourceStage(row, options = {}) {
  return (await resolveSourcePreflight(row, options)).stage
}

function safeObservedToken(value) {
  return typeof value === 'string' && /^[A-Za-z0-9_.:+-]{1,128}$/.test(value)
    ? value
    : value === null || value === undefined
      ? null
      : '<invalid>'
}

function validReceiptTimestamp(value) {
  return typeof value === 'string'
    && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(value)
    && !Number.isNaN(Date.parse(value))
}

function validHostToken(value) {
  return typeof value === 'string' && /^[A-Za-z0-9_.:+-]{1,128}$/.test(value)
}

function deriveInspectorProvenance(inspector) {
  if (!inspector || typeof inspector !== 'object' || Array.isArray(inspector)
    || typeof inspector.version !== 'string'
    || !/^[A-Za-z0-9 ._+()-]{1,128}$/.test(inspector.version)
    || !/^[0-9a-f]{40}$/.test(inspector.source_head || '')
    || typeof inspector.source_tracked_dirty !== 'boolean') return null
  const versionMatch = /^camelid [A-Za-z0-9._+()-]+-g([0-9a-f]{7,40})(-dirty)?$/.exec(inspector.version)
  if (!versionMatch) return null
  const derived = {
    source_head: inspector.source_head,
    source_tracked_dirty: inspector.source_tracked_dirty,
    binary_commit_abbrev: versionMatch[1],
    binary_reports_dirty: Boolean(versionMatch[2]),
    binary_matches_source_head: inspector.source_head.startsWith(versionMatch[1]),
  }
  derived.clean_current_head = derived.binary_matches_source_head
    && !derived.binary_reports_dirty
    && !derived.source_tracked_dirty
  return derived
}

function metadataStageFromHeader(row, receipt, { expectedSourceHead = null } = {}) {
  const host = receipt?.host
  const inspector = receipt?.inspector
  const inspectorProvenance = deriveInspectorProvenance(inspector)
  const source = receipt?.source
  const range = receipt?.range
  const inspection = receipt?.inspection
  const observed = inspection?.observed
  const inventory = inspection?.tensor_inventory
  const contentRange = range?.content_range
  const inventoryTypes = inventory?.types
  const inventoryTypesValid = inventoryTypes
    && typeof inventoryTypes === 'object'
    && !Array.isArray(inventoryTypes)
    && Object.keys(inventoryTypes).length > 0
    && Object.entries(inventoryTypes).every(([type, count]) => (
      /^[A-Za-z][A-Za-z0-9_]{0,31}$/.test(type)
      && Number.isSafeInteger(count)
      && count > 0
    ))
    && Object.values(inventoryTypes).reduce((sum, count) => sum + count, 0) === inspection?.tensor_count
  const expectedCommand = [
    '<camelid>',
    'inspect-prefix',
    '<remote-gguf-prefix>',
    '--declared-len',
    String(row.identity.size_bytes),
  ]
  const receiptShapeValid = receipt?.schema === 'camelid.remote-gguf-header-inspection/v1'
    && receipt.row_id === row.id
    && validReceiptTimestamp(receipt.generated_at)
    && host && typeof host === 'object' && !Array.isArray(host)
    && host.hostname_redacted === true
    && !Object.hasOwn(host, 'hostname')
    && validHostToken(host.platform)
    && validHostToken(host.release)
    && validHostToken(host.arch)
    && inspector && typeof inspector === 'object' && !Array.isArray(inspector)
    && inspectorProvenance
    && /^[0-9a-f]{64}$/.test(inspector.binary_sha256 || '')
    && inspector.source_head === inspectorProvenance.source_head
    && inspector.binary_commit_abbrev === inspectorProvenance.binary_commit_abbrev
    && inspector.binary_reports_dirty === inspectorProvenance.binary_reports_dirty
    && inspector.binary_matches_source_head === inspectorProvenance.binary_matches_source_head
    && inspector.clean_current_head === inspectorProvenance.clean_current_head
    && inspector.clean_current_head === true
    && /^[0-9a-f]{40}$/.test(expectedSourceHead || '')
    && inspector.source_head === expectedSourceHead
    && inspector.binary_path_redacted === true
    && JSON.stringify(inspector.command) === JSON.stringify(expectedCommand)
    && source && typeof source === 'object'
    && range && typeof range === 'object'
    && inspection && typeof inspection === 'object'
    && observed && typeof observed === 'object'
    && inventory && typeof inventory === 'object'
    && contentRange && typeof contentRange === 'object'
    && Number.isSafeInteger(range.requested_bytes) && range.requested_bytes > 0
    && range.requested_bytes <= MAX_PREFIX_BYTES
    && Number.isSafeInteger(range.received_bytes) && range.received_bytes > 0
    && range.received_bytes === range.requested_bytes
    && Number.isSafeInteger(contentRange.start) && contentRange.start === 0
    && Number.isSafeInteger(contentRange.end) && contentRange.end >= 0
    && Number.isSafeInteger(contentRange.total) && contentRange.total > contentRange.end
    && contentRange.end + 1 === range.received_bytes
    && /^[0-9a-f]{64}$/.test(range.prefix_sha256 || '')
    && [2, 3].includes(inspection.version)
    && Number.isSafeInteger(inspection.tensor_count) && inspection.tensor_count >= 0
    && Number.isSafeInteger(inspection.metadata_count) && inspection.metadata_count >= 0
    && Number.isSafeInteger(inspection.alignment) && inspection.alignment > 0
    && Number.isSafeInteger(inspection.data_start_offset) && inspection.data_start_offset >= 0
    && Number.isSafeInteger(inventory.total_n_bytes) && inventory.total_n_bytes >= 0
    && /^[0-9a-f]{64}$/.test(inventory.sha256 || '')
    && inventoryTypesValid
    && receipt.scope?.prefix_sha256 === 'all_received_prefix_bytes'
    && receipt.scope?.tensor_payload === 'partially_range_fetched_opaque'
    && receipt.scope?.full_artifact_sha256 === 'not_run'
    && receipt.scope?.tensor_payload_interpretation === 'not_run'
    && receipt.scope?.load === 'not_run'
    && receipt.scope?.generation === 'not_run'
    && receipt.support_claim === false
  if (!receiptShapeValid) {
    return {
      status: 'fail',
      mode: 'remote_immutable_prefix',
      error_code: 'header_receipt_invalid',
      reason: 'remote header inspector returned an invalid compact receipt',
    }
  }

  const sourceMismatches = []
  for (const [field, expected] of [
    ['repo', row.source.repo],
    ['file', row.source.file],
    ['revision', row.source.revision],
    ['size_bytes', row.identity.size_bytes],
    ['sha256', row.identity.sha256],
  ]) {
    if (source[field] !== expected) sourceMismatches.push(field)
  }
  if (contentRange.total !== row.identity.size_bytes) sourceMismatches.push('content_range.total')
  if (sourceMismatches.length) {
    return {
      status: 'fail',
      mode: 'remote_immutable_prefix',
      error_code: 'header_source_identity_mismatch',
      reason: `remote header receipt does not match the roster identity: ${sourceMismatches.join(', ')}`,
    }
  }

  const compactObserved = {
    gguf_version: inspection.version,
    architecture: safeObservedToken(observed.architecture),
    tokenizer_model: safeObservedToken(observed.tokenizer_model),
    tokenizer_pre: safeObservedToken(observed.tokenizer_pre),
    general_file_type: Number.isSafeInteger(observed.general_file_type)
      ? observed.general_file_type
      : null,
    declared_quantization: GGUF_FILE_TYPE_LABELS.get(observed.general_file_type) || null,
    headline_quant: safeObservedToken(observed.headline_quant),
    tensor_count: inspection.tensor_count,
    metadata_count: inspection.metadata_count,
    alignment: inspection.alignment,
    data_start_offset: inspection.data_start_offset,
    tensor_payload_n_bytes: inventory.total_n_bytes,
    tensor_inventory_sha256: inventory.sha256,
  }
  const mismatches = []
  if (compactObserved.architecture !== row.identity.architecture) {
    mismatches.push(`architecture ${JSON.stringify(compactObserved.architecture)} != ${JSON.stringify(row.identity.architecture)}`)
  }
  if (compactObserved.tokenizer_model !== row.expected.tokenizer_model) {
    mismatches.push(`tokenizer model ${JSON.stringify(compactObserved.tokenizer_model)} != ${JSON.stringify(row.expected.tokenizer_model)}`)
  }
  if (row.expected.tokenizer_pre !== null && compactObserved.tokenizer_pre !== row.expected.tokenizer_pre) {
    mismatches.push(`tokenizer pre ${JSON.stringify(compactObserved.tokenizer_pre)} != ${JSON.stringify(row.expected.tokenizer_pre)}`)
  }
  if (compactObserved.declared_quantization !== row.identity.quantization) {
    mismatches.push(`declared quant ${JSON.stringify(compactObserved.declared_quantization)} != ${JSON.stringify(row.identity.quantization)}`)
  }
  if (row.identity.quantization === 'Q4_K_M') {
    if (!Number.isSafeInteger(inventoryTypes.Q4_K) || inventoryTypes.Q4_K <= 0
      || !Number.isSafeInteger(inventoryTypes.Q6_K) || inventoryTypes.Q6_K <= 0) {
      mismatches.push('Q4_K_M tensor inventory must contain both Q4_K and Q6_K tensors')
    }
  } else if (compactObserved.headline_quant !== row.identity.quantization) {
    mismatches.push(`headline quant ${JSON.stringify(compactObserved.headline_quant)} != ${JSON.stringify(row.identity.quantization)}`)
  }

  const details = {
    mode: 'remote_immutable_prefix',
    inspection_generated_at: receipt.generated_at,
    host: {
      hostname_redacted: true,
      platform: host.platform,
      release: host.release,
      arch: host.arch,
    },
    inspector: {
      version: inspector.version,
      binary_sha256: inspector.binary_sha256,
      source_head: inspectorProvenance.source_head,
      source_tracked_dirty: inspectorProvenance.source_tracked_dirty,
      binary_commit_abbrev: inspectorProvenance.binary_commit_abbrev,
      binary_reports_dirty: inspectorProvenance.binary_reports_dirty,
      binary_matches_source_head: inspectorProvenance.binary_matches_source_head,
      clean_current_head: inspectorProvenance.clean_current_head,
      binary_path_redacted: true,
      command: expectedCommand,
    },
    source: {
      repo: row.source.repo,
      file: row.source.file,
      revision: row.source.revision,
      size_bytes: row.identity.size_bytes,
      sha256: row.identity.sha256,
    },
    range: {
      requested_bytes: range.requested_bytes,
      received_bytes: range.received_bytes,
      content_range: {
        start: contentRange.start,
        end: contentRange.end,
        total: contentRange.total,
      },
      prefix_sha256: range.prefix_sha256,
    },
    observed: compactObserved,
    scope: {
      prefix_sha256: 'all_received_prefix_bytes',
      tensor_payload: 'partially_range_fetched_opaque',
      opaque_tensor_payload_prefix_bytes: Math.max(0, range.received_bytes - inspection.data_start_offset),
      full_artifact_sha256: 'not_run',
      tensor_payload_interpretation: 'not_run',
      load: 'not_run',
      generation: 'not_run',
      support_claim: false,
    },
  }
  return mismatches.length
    ? { status: 'fail', ...details, reason: mismatches.join('; ') }
    : { status: 'pass', ...details }
}

function metadataStageFromHeaderError(error) {
  const failure = classifyHeaderInspectionError(error)
  return {
    status: failure.status,
    mode: 'remote_immutable_prefix',
    error_code: failure.error_code,
    reason: failure.reason,
  }
}

function tokenizerStageFromReceipt(row, receipt, defaults, {
  expectedSourceHead,
} = {}) {
  if (!/^[0-9a-f]{40}$/.test(expectedSourceHead || '')) {
    return {
      status: 'fail',
      mode: 'remote_immutable_prefix_tokenizer',
      error_code: 'tokenizer_receipt_invalid',
      reason: 'bounded tokenizer receipt is not bound to the factory source HEAD',
    }
  }
  const assessed = assessTokenizerReceipt(receipt, row, defaults, { expectedSourceHead })
  if (assessed.errors.length) {
    return {
      status: 'fail',
      mode: 'remote_immutable_prefix_tokenizer',
      error_code: 'tokenizer_receipt_invalid',
      reason: 'bounded tokenizer inspector returned an invalid compact receipt',
    }
  }

  const provenance = receipt.provenance
  const bounded = receipt.bounded_fetch
  const contentRange = bounded.content_range
  const details = {
    mode: 'remote_immutable_prefix_tokenizer',
    inspection_generated_at: receipt.generated_at,
    host: {
      hostname_redacted: true,
      platform: receipt.host.platform,
    },
    inspector: {
      version: receipt.camelid.version,
      binary_sha256: receipt.camelid.binary_sha256,
      source_head: provenance.source_head,
      binary_commit_abbrev: provenance.binary_commit_abbrev,
      binary_reports_dirty: provenance.binary_reports_dirty,
      binary_matches_source_head: provenance.binary_matches_source_head,
      source_tracked_dirty: provenance.source_tracked_dirty,
      clean_current_head: provenance.clean_current_head,
      binary_path_redacted: true,
      prefix_mode: 'tokenize --declared-len',
    },
    source: {
      repo: row.source.repo,
      file: row.source.file,
      revision: row.source.revision,
      size_bytes: row.identity.size_bytes,
      sha256: row.identity.sha256,
      license: row.source.license,
    },
    range: {
      requested_bytes: bounded.requested_bytes,
      received_bytes: bounded.received_bytes,
      content_range: {
        start: contentRange.start,
        end: contentRange.end,
        total: contentRange.total,
      },
      prefix_sha256: bounded.prefix_sha256,
    },
    oracle: {
      project: 'ggml-org/llama.cpp',
      revision: receipt.oracle.revision,
      build: receipt.oracle.build,
      binary_sha256: receipt.oracle.binary_sha256,
      companion_binary_sha256: receipt.oracle.companion_binary_sha256,
      derivative_sha256: receipt.oracle.derivative.sha256,
      derivative_persisted: false,
    },
    observed: assessed.tokenizer_metadata,
    result: {
      case_count: assessed.case_count,
      exact_match_count: assessed.exact_match_count,
      all_token_ids_match: assessed.all_token_ids_match,
    },
    scope: {
      prefix_sha256: 'all_received_prefix_bytes',
      tokenizer_metadata: 'unchanged_from_immutable_prefix',
      tensor_payload: 'partially_range_fetched_opaque',
      full_artifact_sha256: 'not_run',
      tensor_payload_interpretation: 'not_run',
      template_rendering: 'not_run',
      load: 'not_run',
      generation: 'not_run',
      api_webui: 'not_run',
      context: 'not_run',
      support_claim: false,
    },
  }
  if (assessed.parity_errors.length || !assessed.all_token_ids_match) {
    return {
      status: 'fail',
      ...details,
      error_code: 'tokenizer_parity_mismatch',
      reason: 'one or more exact-row tokenizer probes diverged from the pinned oracle',
    }
  }
  return { status: 'pass', ...details }
}

function tokenizerStageFromError(error) {
  const failure = classifyTokenizerQualificationError(error)
  return {
    status: failure.status,
    mode: 'remote_immutable_prefix_tokenizer',
    error_code: failure.error_code,
    reason: failure.reason,
  }
}

function summarizeHeaderInspections(reports, prefixBytes) {
  const counts = { pass: 0, fail: 0, blocked: 0, not_run: 0 }
  let requestedBytes = 0
  let receivedBytes = 0
  for (const { headerOutcome } of reports) {
    const status = headerOutcome?.status || 'not_run'
    counts[status] = (counts[status] || 0) + 1
    requestedBytes += headerOutcome?.range?.requested_bytes || 0
    receivedBytes += headerOutcome?.range?.received_bytes || 0
  }
  return {
    mode: 'remote_immutable_prefix',
    per_row_byte_budget: prefixBytes,
    verified_receipt_requested_bytes: requestedBytes,
    verified_receipt_received_bytes: receivedBytes,
    counts,
  }
}

function summarizeTokenizerInspections(reports) {
  const counts = { pass: 0, fail: 0, blocked: 0, not_run: 0 }
  let requestedBytes = 0
  let receivedBytes = 0
  for (const { tokenizerOutcome } of reports) {
    const status = tokenizerOutcome?.status || 'not_run'
    counts[status] = (counts[status] || 0) + 1
    requestedBytes += tokenizerOutcome?.range?.requested_bytes || 0
    receivedBytes += tokenizerOutcome?.range?.received_bytes || 0
  }
  return {
    mode: 'remote_immutable_prefix_tokenizer',
    per_row_byte_budget: DEFAULT_TOKENIZER_PREFIX_BYTES,
    verified_receipt_requested_bytes: requestedBytes,
    verified_receipt_received_bytes: receivedBytes,
    counts,
  }
}

function summarizeSourceResolution(reports) {
  const counts = { pass: 0, fail: 0, blocked: 0 }
  for (const { report } of reports) {
    const status = report.stages.source?.status
    counts[status] = (counts[status] || 0) + 1
  }
  return { mode: 'live_huggingface', counts }
}

function summarizeBatchSourceHeads(reports) {
  const heads = reports.map(({ report }) => report?.source_head)
  if (!heads.length || heads.some((head) => !/^[0-9a-f]{40}$/.test(head || ''))) {
    return { source_head: null, state: 'unknown' }
  }
  const unique = new Set(heads)
  return unique.size === 1
    ? { source_head: heads[0], state: 'uniform' }
    : { source_head: null, state: 'mixed' }
}

function summarizeReports(reports, gateOrder, { sourcePreflight = false } = {}) {
  const counts = { pass: 0, fail: 0, blocked: 0 }
  const rows = reports.map(({ row, report, reportFile }) => {
    counts[report.overall_status] = (counts[report.overall_status] || 0) + 1
    return {
      priority: row.priority,
      row_id: row.id,
      disposition: row.disposition,
      overall_status: report.overall_status,
      first_unresolved_stage: sourcePreflight && report.stages.source?.status !== 'pass'
        ? 'source'
        : firstUnresolvedStage(report, gateOrder),
      report_file: reportFile,
    }
  })
  return { counts, rows }
}

async function runFactory(options) {
  const root = resolve(options.root || '.')
  const rosterPath = resolve(root, options.roster || 'qa/model-qualification/phase1-roster.json')
  const roster = JSON.parse(await readFile(rosterPath, 'utf8'))
  const rosterErrors = validateRoster(roster, rosterPath)
  if (rosterErrors.length) throw new Error(`roster is invalid:\n${rosterErrors.join('\n')}`)

  const rows = selectRows(roster, options.rows)
  if (options.artifact && rows.length !== 1) {
    throw new Error('--artifact is only valid when exactly one row is selected')
  }
  const inspectTokenizer = Boolean(options.inspectTokenizer)
  const inspectHeader = Boolean(options.inspectHeader || inspectTokenizer)
  const sourcePreflightEnabled = Boolean(options.resolveSource || inspectHeader)
  const prefixBytes = inspectHeader
    ? normalizePrefixBytes(options.prefixBytes ?? DEFAULT_PREFIX_BYTES)
    : DEFAULT_PREFIX_BYTES
  const hfToken = options.hfToken ?? process.env.HF_TOKEN ?? null

  const configuredModelsDir = options.modelsDir
    || process.env[roster.defaults.models_dir_env]
    || null
  const modelsDir = configuredModelsDir ? resolve(root, configuredModelsDir) : null
  const outDir = resolve(root, options.outDir || 'target/model-qualification/factory')
  await mkdir(outDir, { recursive: true })

  const reports = []
  for (const row of rows) {
    const artifact = artifactForRow(row, modelsDir, options.artifact)
    const sourcePreflight = sourcePreflightEnabled
      ? await resolveSourcePreflight(row, {
        resolver: options.sourceResolver || resolveHfSource,
        token: hfToken,
      })
      : null
    const resolvedSource = sourcePreflight?.stage || null
    const sourcePassed = !resolvedSource || resolvedSource.status === 'pass'
    let report = await (options.qualifier || qualify)({
      root,
      roster: options.roster,
      row: row.id,
      artifact: sourcePassed ? artifact : null,
      camelid: options.camelid,
      runSmoke: sourcePassed && options.runSmoke,
      runGeneration: sourcePassed && options.runGeneration,
      promptLimit: options.promptLimit,
    })
    let headerOutcome = null
    let tokenizerOutcome = null
    if (resolvedSource) {
      report.stages.source = resolvedSource
      if (!sourcePassed && artifact) {
        report.stages.artifact = {
          status: 'blocked',
          reason: 'live source preflight did not pass; the local artifact was not inspected',
        }
      }
    }
    if (inspectHeader) {
      if (!sourcePassed || !sourcePreflight?.lock) {
        headerOutcome = {
          status: 'blocked',
          mode: 'remote_immutable_prefix',
          error_code: 'header_source_preflight_blocked',
          reason: 'remote header inspection is downstream of a passing live source preflight',
        }
        report.stages.metadata = headerOutcome
      } else if (report.stages.artifact?.status === 'pass') {
        headerOutcome = {
          status: 'not_run',
          reason: 'an exact local artifact passed identity and remains the authoritative metadata lane',
        }
      } else {
        try {
          const receipt = await (options.headerInspector || inspectRemoteHeader)(sourcePreflight.lock, {
            binary: resolve(root, options.camelid || defaultCamelidBinary()),
            rowId: row.id,
            sourceRoot: root,
            prefixBytes,
            token: hfToken,
          })
          headerOutcome = metadataStageFromHeader(row, receipt, {
            expectedSourceHead: report.source_head,
          })
        } catch (error) {
          headerOutcome = metadataStageFromHeaderError(error)
        }
        report.stages.metadata = headerOutcome
      }
    }
    if (inspectTokenizer) {
      if (!sourcePassed || !sourcePreflight?.lock) {
        tokenizerOutcome = {
          status: 'blocked',
          mode: 'remote_immutable_prefix_tokenizer',
          error_code: 'tokenizer_source_preflight_blocked',
          reason: 'bounded tokenizer qualification is downstream of a passing live source preflight',
        }
        report.stages.tokenizer = tokenizerOutcome
      } else if (report.stages.metadata?.status !== 'pass') {
        tokenizerOutcome = {
          status: 'blocked',
          mode: 'remote_immutable_prefix_tokenizer',
          error_code: 'tokenizer_metadata_preflight_blocked',
          reason: 'bounded tokenizer qualification is downstream of a passing metadata inspection',
        }
        report.stages.tokenizer = tokenizerOutcome
      } else if (!/^[0-9a-f]{40}$/.test(report.source_head || '')) {
        tokenizerOutcome = {
          status: 'blocked',
          mode: 'remote_immutable_prefix_tokenizer',
          error_code: 'tokenizer_source_head_unavailable',
          reason: 'bounded tokenizer qualification requires an observed factory source HEAD',
        }
        report.stages.tokenizer = tokenizerOutcome
      } else if (report.stages.artifact?.status === 'pass'
        && ['pass', 'fail'].includes(report.stages.tokenizer?.status)) {
        tokenizerOutcome = {
          status: 'not_run',
          reason: report.stages.tokenizer.status === 'pass'
            ? 'an exact local artifact already passed the authoritative tokenizer lane'
            : 'an exact local artifact already failed the authoritative tokenizer lane; remote evidence cannot overwrite that failure',
        }
      } else if (!tokenizerPackAvailable(row.id)) {
        tokenizerOutcome = {
          status: 'blocked',
          mode: 'remote_immutable_prefix_tokenizer',
          error_code: 'tokenizer_pack_unavailable',
          reason: 'no bounded tokenizer pack is defined for this exact row',
        }
        report.stages.tokenizer = tokenizerOutcome
      } else {
        try {
          const receipt = await (options.tokenizerInspector || inspectRemoteTokenizer)(
            sourcePreflight.lock,
            {
              row,
              defaults: roster.defaults,
              binary: resolve(root, options.camelid || defaultCamelidBinary()),
              llamaTokenize: resolve(
                root,
                options.llamaTokenize || defaultLlamaTokenizerBinary(),
              ),
              sourceRoot: root,
              prefixBytes: tokenizerPrefixBytesForRow(row.id),
              token: hfToken,
            },
          )
          tokenizerOutcome = tokenizerStageFromReceipt(row, receipt, roster.defaults, {
            expectedSourceHead: report.source_head,
          })
        } catch (error) {
          tokenizerOutcome = tokenizerStageFromError(error)
        }
        report.stages.tokenizer = tokenizerOutcome
      }
    }
    report.overall_status = deriveOverall(report.stages)
    report = redactLocalPaths(report, [
      [artifact, '<artifact>'],
      [root, '<repo>'],
    ])
    const reportFile = `${row.id}.json`
    await writeFile(resolve(outDir, reportFile), `${JSON.stringify(report, null, 2)}\n`)
    reports.push({ row, report, reportFile, headerOutcome, tokenizerOutcome })
  }

  const summary = summarizeReports(reports, roster.gate_order, {
    sourcePreflight: sourcePreflightEnabled,
  })
  const sourceDirtyValues = reports.map(({ report }) => report.source_dirty)
  const sourceHeadSummary = summarizeBatchSourceHeads(reports)
  const index = {
    schema: 'camelid.model-qualification-index/v1',
    generated_at: new Date().toISOString(),
    roster: publicRosterLabel(root, rosterPath),
    roster_schema: roster.schema,
    models_dir_env: roster.defaults.models_dir_env,
    models_dir_configured: Boolean(modelsDir),
    source_head: sourceHeadSummary.source_head,
    ...(sourceHeadSummary.state === 'uniform'
      ? {}
      : { source_head_state: sourceHeadSummary.state }),
    source_dirty: sourceDirtyValues.some((value) => value === null)
      ? null
      : sourceDirtyValues.some(Boolean),
    ...(sourcePreflightEnabled
      ? { source_resolution: summarizeSourceResolution(reports) }
      : {}),
    ...(inspectHeader
      ? { header_inspection: summarizeHeaderInspections(reports, prefixBytes) }
      : {}),
    ...(inspectTokenizer
      ? { tokenizer_inspection: summarizeTokenizerInspections(reports) }
      : {}),
    ...summary,
  }
  await writeFile(resolve(outDir, 'index.json'), `${JSON.stringify(index, null, 2)}\n`)
  return index
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  if (args.has('help')) {
    console.log(`Usage: node scripts/model-qualification-factory.mjs [options]

Options:
  --roster <path>          Qualification roster (default: Phase 1)
  --rows <id,id>           Run only these rows, preserving roster priority
  --models-dir <path>      Artifact directory (default: roster env variable)
  --artifact <path>        Exact artifact override; requires one selected row
  --camelid <path>         Camelid binary (default: target/release/camelid)
  --out-dir <path>         Scrubbed reports/index output directory
  --resolve-source         Live-resolve each pinned HF selector before local probes
  --inspect-header         Bounded immutable-prefix inspection; implies --resolve-source
  --prefix-bytes <n>       Per-row header range budget, max 64 MiB (default: 32 MiB)
  --inspect-tokenizer      Exact 32 MiB tokenizer/oracle lane; implies header + source
  --llama-tokenize <path>  Pinned llama-tokenize binary/package
  --run-smoke              Execute configured smoke probes
  --run-generation         Execute pinned greedy parity probes
  --prompt-limit <n>       Deliberately partial parity run (remains blocked)
`)
    return
  }
  const requestedRows = args.get('rows')
    ? args.get('rows').split(',').map((id) => id.trim()).filter(Boolean)
    : []
  const promptLimitRaw = args.get('prompt-limit')
  const promptLimit = promptLimitRaw ? Number.parseInt(promptLimitRaw, 10) : undefined
  if (promptLimitRaw && (!Number.isInteger(promptLimit) || promptLimit < 1)) {
    throw new Error('--prompt-limit must be a positive integer')
  }
  const inspectTokenizer = args.has('inspect-tokenizer')
  const inspectHeader = args.has('inspect-header') || inspectTokenizer
  const prefixBytes = args.has('prefix-bytes')
    ? normalizePrefixBytes(args.get('prefix-bytes'))
    : DEFAULT_PREFIX_BYTES
  const index = await runFactory({
    root: '.',
    roster: args.get('roster'),
    rows: requestedRows,
    modelsDir: args.get('models-dir'),
    artifact: args.get('artifact'),
    camelid: args.get('camelid'),
    llamaTokenize: args.get('llama-tokenize'),
    outDir: args.get('out-dir'),
    resolveSource: args.has('resolve-source') || inspectHeader,
    inspectHeader,
    inspectTokenizer,
    prefixBytes,
    runSmoke: args.has('run-smoke'),
    runGeneration: args.has('run-generation'),
    promptLimit,
  })
  process.stdout.write(`${JSON.stringify(index, null, 2)}\n`)
}

export {
  artifactForRow,
  defaultCamelidBinary,
  defaultLlamaTokenizerBinary,
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
  summarizeBatchSourceHeads,
  summarizeHeaderInspections,
  summarizeTokenizerInspections,
  summarizeSourceResolution,
  summarizeReports,
  tokenizerStageFromError,
  tokenizerStageFromReceipt,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
