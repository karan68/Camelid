#!/usr/bin/env node
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { basename, isAbsolute, relative, resolve, sep } from 'node:path'
import { pathToFileURL } from 'node:url'
import { validateRoster } from './check-model-qualification-roster.mjs'
import {
  DEFAULT_PREFIX_BYTES,
  MAX_PREFIX_BYTES,
  classifyHeaderInspectionError,
  inspectBinaryIdentity,
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
import {
  PREFIX_BYTES as SMOLLM3_TEMPLATE_PREFIX_BYTES,
  classifySmolLM3TemplateQualificationError,
  qualifySmolLM3Template,
  smollm3TemplatePackAvailable,
  smollm3TemplatePrefixBytesForRow,
  validateShapePack as validateSmolLM3TemplateShapePack,
} from './hf-qualification-smollm3-template.mjs'
import { resolveHfSource, validateLockAgainstSelection } from './hf-qualification-source.mjs'
import { deriveOverall, qualify, redactLocalPaths } from './model-qualification-runner.mjs'

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

function defaultLlamaTemplateAnalyzerBinary() {
  return process.platform === 'win32'
    ? 'target/reference/llama.cpp-b9632/bin/llama-template-analysis.exe'
    : 'target/reference/llama.cpp-b9632/bin/llama-template-analysis'
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
    || !/^[0-9a-f]{40}$/.test(inspector.source_head || '')) return null
  const versionMatch = /^camelid [A-Za-z0-9._+()-]+-g([0-9a-f]{7,40})(-dirty)?$/.exec(inspector.version)
  if (!versionMatch) return null
  const derived = {
    source_head: inspector.source_head,
    binary_commit_abbrev: versionMatch[1],
    binary_reports_dirty: Boolean(versionMatch[2]),
    binary_matches_source_head: inspector.source_head.startsWith(versionMatch[1]),
  }
  derived.clean_current_head = derived.binary_matches_source_head && !derived.binary_reports_dirty
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
    && (expectedSourceHead === null
      || (/^[0-9a-f]{40}$/.test(expectedSourceHead) && inspector.source_head === expectedSourceHead))
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
  if (compactObserved.headline_quant !== row.identity.quantization) {
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

function templatePreparationStageFromPack(row, pack, {
  expectedSourceHead,
  expectedInspector,
} = {}) {
  if (!/^[0-9a-f]{40}$/.test(expectedSourceHead || '')
    || !expectedInspector
    || typeof expectedInspector !== 'object'
    || Array.isArray(expectedInspector)) {
    return {
      status: 'fail',
      mode: 'remote_immutable_prefix_smollm3_template_preparation',
      error_code: 'template_preparation_receipt_invalid',
      reason: 'bounded template preparation is not bound to the factory source HEAD',
      preparation: { status: 'fail' },
    }
  }
  const errors = validateSmolLM3TemplateShapePack(pack, {
    expectedInspector,
  })
  if (errors.length
    || pack?.row_id !== row.id
    || expectedInspector.source_head !== expectedSourceHead
    || pack?.inspector?.source_head !== expectedSourceHead) {
    return {
      status: 'fail',
      mode: 'remote_immutable_prefix_smollm3_template_preparation',
      error_code: 'template_preparation_receipt_invalid',
      reason: 'bounded template preparation returned an invalid compact receipt',
      preparation: { status: 'fail' },
    }
  }

  return {
    status: 'blocked',
    mode: 'remote_immutable_prefix_smollm3_template_preparation',
    error_code: 'smollm3_template_runtime_hold',
    reason: 'bounded template preparation passed, but runtime chat remains on the architecture-wide typed HOLD',
    preparation: {
      status: 'pass',
      schema: pack.schema,
      pack_id: pack.pack_id,
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
      requested_bytes: pack.bounded_prefix.requested_bytes,
      received_bytes: pack.bounded_prefix.received_bytes,
      content_range: {
        start: pack.bounded_prefix.content_range.start,
        end: pack.bounded_prefix.content_range.end,
        total: pack.bounded_prefix.content_range.total,
      },
      prefix_sha256: pack.bounded_prefix.prefix_sha256,
    },
    inspector: {
      version: pack.inspector.version,
      binary_sha256: pack.inspector.binary_sha256,
      source_head: pack.inspector.source_head,
      binary_commit_abbrev: pack.inspector.binary_commit_abbrev,
      binary_reports_dirty: pack.inspector.binary_reports_dirty,
      binary_matches_source_head: pack.inspector.binary_matches_source_head,
      source_tracked_dirty: pack.inspector.source_tracked_dirty,
      clean_current_head: pack.inspector.clean_current_head,
      binary_path_redacted: true,
    },
    oracle: {
      project: pack.oracle.project,
      build: pack.oracle.build,
      revision: pack.oracle.revision,
      reported_revision: pack.oracle.reported_revision,
      analyzer_binary_sha256: pack.oracle.analyzer_binary_sha256,
      companion_binary_sha256: pack.oracle.companion_binary_sha256,
      package_manifest_sha256: pack.oracle.package_manifest_sha256,
      archive_sha256: pack.oracle.archive_sha256,
      executable_paths_redacted: true,
      mode: pack.oracle.mode,
    },
    template: {
      utf8_bytes: pack.source_template.utf8_bytes,
      sha256: pack.source_template.sha256,
    },
    cases: pack.cases.map((testCase) => ({
      id: testCase.id,
      normalized_prompt_utf8_bytes: testCase.normalized_prompt_utf8_bytes,
      normalized_prompt_sha256: testCase.normalized_prompt_sha256,
      oracle_exact_match: testCase.oracle_exact_match_after_date_normalization
        ?? testCase.oracle_core_exact_match_after_date_normalization,
    })),
    scope: {
      prefix_sha256: 'all_received_prefix_bytes',
      full_artifact_sha256: 'not_run',
      tensor_payload_interpretation: 'not_run',
      load: 'not_run',
      generation: 'not_run',
      http_apply_template: 'not_run',
      runtime_chat: 'blocked',
      template_gate: 'blocked',
      support_claim: false,
    },
  }
}

function templatePreparationStageFromError(error) {
  const failure = classifySmolLM3TemplateQualificationError(error)
  return {
    status: failure.status,
    mode: 'remote_immutable_prefix_smollm3_template_preparation',
    error_code: failure.error_code,
    reason: failure.reason,
    preparation: { status: failure.status },
  }
}

function blockedTemplatePreparation(errorCode, reason) {
  return {
    status: 'blocked',
    mode: 'remote_immutable_prefix_smollm3_template_preparation',
    error_code: errorCode,
    reason,
    preparation: { status: 'blocked' },
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

function summarizeTemplatePreparations(reports) {
  const counts = { pass: 0, fail: 0, blocked: 0, not_run: 0 }
  const preparationResults = { pass: 0, fail: 0, blocked: 0, not_run: 0 }
  let requestedBytes = 0
  let receivedBytes = 0
  for (const { templatePreparationOutcome } of reports) {
    const status = templatePreparationOutcome?.status || 'not_run'
    const preparationStatus = templatePreparationOutcome?.preparation?.status || 'not_run'
    counts[status] = (counts[status] || 0) + 1
    preparationResults[preparationStatus] = (preparationResults[preparationStatus] || 0) + 1
    if (preparationStatus === 'pass') {
      requestedBytes += templatePreparationOutcome?.range?.requested_bytes || 0
      receivedBytes += templatePreparationOutcome?.range?.received_bytes || 0
    }
  }
  return {
    mode: 'remote_immutable_prefix_smollm3_template_preparation',
    per_row_byte_budget: SMOLLM3_TEMPLATE_PREFIX_BYTES,
    verified_receipt_requested_bytes: requestedBytes,
    verified_receipt_received_bytes: receivedBytes,
    counts,
    preparation_results: preparationResults,
    runtime_template_gate: 'blocked',
    support_claim: false,
  }
}

function summarizeSourceResolution(reports) {
  const counts = { pass: 0, fail: 0, blocked: 0 }
  for (const item of reports) {
    const status = Object.hasOwn(item, 'sourcePreflight')
      ? item.sourcePreflight?.stage?.status || 'not_run'
      : item.report?.stages?.source?.status
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
  const rows = reports.map((item) => {
    const { row, report, reportFile } = item
    const sourcePreflightAttempted = sourcePreflight && (
      Object.hasOwn(item, 'sourcePreflight')
        ? item.sourcePreflight !== null
        : Boolean(report.stages.source)
    )
    counts[report.overall_status] = (counts[report.overall_status] || 0) + 1
    return {
      priority: row.priority,
      row_id: row.id,
      disposition: row.disposition,
      overall_status: report.overall_status,
      first_unresolved_stage: sourcePreflightAttempted && report.stages.source?.status !== 'pass'
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
  const inspectTemplate = Boolean(options.inspectTemplate)
  if (inspectTemplate && inspectHeader) {
    throw new Error('--inspect-template cannot be combined with --inspect-header or --inspect-tokenizer; each lane owns a bounded prefix fetch')
  }
  const baseSourcePreflightEnabled = Boolean(options.resolveSource || inspectHeader)
  const sourceSummaryEnabled = Boolean(baseSourcePreflightEnabled || inspectTemplate)
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
    const templatePackAvailable = smollm3TemplatePackAvailable(row.id)
    const templatePrefixBytes = smollm3TemplatePrefixBytesForRow(row.id)
    const rowSourcePreflightEnabled = baseSourcePreflightEnabled
      || (inspectTemplate && templatePackAvailable)
    const sourcePreflight = rowSourcePreflightEnabled
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
    let templatePreparationOutcome = null
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
    if (inspectTemplate) {
      if (report.stages.artifact?.status === 'pass'
        && ['pass', 'fail'].includes(report.stages.template?.status)) {
        templatePreparationOutcome = {
          status: 'not_run',
          reason: report.stages.template.status === 'pass'
            ? 'an exact local artifact already passed the authoritative template lane'
            : 'an exact local artifact already failed the authoritative template lane; preparation evidence cannot overwrite that failure',
          preparation: { status: 'not_run' },
        }
      } else if (!templatePackAvailable) {
        templatePreparationOutcome = blockedTemplatePreparation(
          'template_pack_unavailable',
          'no bounded template-preparation pack is defined for this exact row',
        )
        report.stages.template = templatePreparationOutcome
      } else if (!sourcePassed || !sourcePreflight?.lock) {
        templatePreparationOutcome = blockedTemplatePreparation(
          'template_source_preflight_blocked',
          'bounded template preparation is downstream of a passing live source preflight',
        )
        report.stages.template = templatePreparationOutcome
      } else if (!/^[0-9a-f]{40}$/.test(report.source_head || '')) {
        templatePreparationOutcome = blockedTemplatePreparation(
          'template_source_head_unavailable',
          'bounded template preparation requires an observed factory source HEAD',
        )
        report.stages.template = templatePreparationOutcome
      } else {
        try {
          const binary = resolve(root, options.camelid || defaultCamelidBinary())
          const pack = await (options.templateInspector || qualifySmolLM3Template)({
            root,
            rosterPath,
            binary,
            analyzer: resolve(
              root,
              options.llamaTemplateAnalysis || defaultLlamaTemplateAnalyzerBinary(),
            ),
            prefixBytes: templatePrefixBytes,
            token: hfToken,
            initialLock: sourcePreflight.lock,
            sourceResolver: options.sourceResolver || resolveHfSource,
          })
          let inspectedBinary = null
          try {
            const candidate = await (options.templateBinaryInspector || inspectBinaryIdentity)(
              binary,
              { sourceRoot: root },
            )
            if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) {
              throw new Error('invalid Camelid inspector identity')
            }
            inspectedBinary = candidate
          } catch {
            templatePreparationOutcome = blockedTemplatePreparation(
              'template_inspector_identity_unavailable',
              'the Camelid template inspector identity could not be independently rechecked',
            )
          }
          if (inspectedBinary) {
            templatePreparationOutcome = templatePreparationStageFromPack(row, pack, {
              expectedSourceHead: report.source_head,
              expectedInspector: {
                version: inspectedBinary.version,
                binary_sha256: inspectedBinary.binary_sha256,
                source_head: inspectedBinary.source_head,
                source_tracked_dirty: false,
                binary_commit_abbrev: inspectedBinary.binary_commit_abbrev,
                binary_reports_dirty: inspectedBinary.binary_reports_dirty,
                binary_matches_source_head: inspectedBinary.binary_matches_source_head,
                clean_current_head: inspectedBinary.clean_current_head,
                binary_path_redacted: true,
              },
            })
          }
        } catch (error) {
          templatePreparationOutcome = templatePreparationStageFromError(error)
        }
        report.stages.template = templatePreparationOutcome
      }
    }
    report.overall_status = deriveOverall(report.stages)
    report = redactLocalPaths(report, [
      [artifact, '<artifact>'],
      [root, '<repo>'],
    ])
    const reportFile = `${row.id}.json`
    await writeFile(resolve(outDir, reportFile), `${JSON.stringify(report, null, 2)}\n`)
    reports.push({
      row,
      report,
      reportFile,
      sourcePreflight,
      headerOutcome,
      tokenizerOutcome,
      templatePreparationOutcome,
    })
  }

  const summary = summarizeReports(reports, roster.gate_order, {
    sourcePreflight: sourceSummaryEnabled,
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
    ...(sourceSummaryEnabled
      ? { source_resolution: summarizeSourceResolution(reports) }
      : {}),
    ...(inspectHeader
      ? { header_inspection: summarizeHeaderInspections(reports, prefixBytes) }
      : {}),
    ...(inspectTokenizer
      ? { tokenizer_inspection: summarizeTokenizerInspections(reports) }
      : {}),
    ...(inspectTemplate
      ? { template_preparation: summarizeTemplatePreparations(reports) }
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
  --inspect-template       Exact 32 MiB SmolLM3 template-preparation lane; implies source
                           and is mutually exclusive with header/tokenizer inspection
  --llama-template-analysis <path>
                           Pinned llama-template-analysis binary/package
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
  const inspectTemplate = args.has('inspect-template')
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
    llamaTemplateAnalysis: args.get('llama-template-analysis'),
    outDir: args.get('out-dir'),
    resolveSource: args.has('resolve-source') || inspectHeader,
    inspectHeader,
    inspectTokenizer,
    inspectTemplate,
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
  defaultLlamaTemplateAnalyzerBinary,
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
  summarizeTemplatePreparations,
  summarizeSourceResolution,
  summarizeReports,
  tokenizerStageFromError,
  tokenizerStageFromReceipt,
  templatePreparationStageFromError,
  templatePreparationStageFromPack,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
