#!/usr/bin/env node
import { createHash } from 'node:crypto'
import { execFile } from 'node:child_process'
import { createReadStream } from 'node:fs'
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { arch, platform, release, tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import { validateRoster } from './check-model-qualification-roster.mjs'
import { resolveHfSource, validateLockAgainstSelection } from './hf-qualification-source.mjs'

const execFileAsync = promisify(execFile)
const DEFAULT_PREFIX_BYTES = 32 * 1024 * 1024
const MAX_PREFIX_BYTES = 64 * 1024 * 1024
const REVISION_RE = /^[0-9a-f]{40}$/
const SHA256_RE = /^[0-9a-f]{64}$/
const HEADER_ERROR_CONTRACTS = Object.freeze({
  header_inspector_identity_invalid: ['fail', 'camelid inspector identity is absent or invalid'],
  header_inspector_not_clean_current_head: ['blocked', 'camelid inspector is not a clean build of current source HEAD'],
  header_inspector_unavailable: ['blocked', 'camelid inspector is unavailable'],
  source_lock_invalid: ['blocked', 'immutable source lock is absent or invalid'],
  header_fetch_unavailable: ['blocked', 'remote header fetch is blocked'],
  header_range_unavailable: ['blocked', 'remote header range response is unavailable'],
  header_range_invalid: ['fail', 'remote header range response is invalid'],
  header_range_identity_mismatch: ['fail', 'remote header range identity does not match the source lock'],
  header_range_incomplete: ['blocked', 'remote header range response is incomplete'],
  header_body_budget_exceeded: ['blocked', 'remote header body exceeded its byte budget'],
  header_body_missing: ['blocked', 'remote header range response has no body'],
  header_body_unavailable: ['blocked', 'remote header range body could not be read'],
  header_body_length_mismatch: ['fail', 'remote header body length does not match its range declaration'],
  header_inspection_contract_invalid: ['fail', 'camelid inspect-prefix returned an invalid inspection object'],
  header_inspector_timeout: ['blocked', 'camelid inspect-prefix timed out'],
  header_inspector_output_budget: ['blocked', 'camelid inspect-prefix exceeded its output budget'],
  header_parse_failed: ['fail', 'camelid inspect-prefix rejected the remote GGUF header'],
  header_inspector_output_invalid: ['fail', 'camelid inspect-prefix did not emit valid JSON'],
  header_row_id_invalid: ['fail', 'remote header inspection row id is invalid'],
  header_inspection_error: ['blocked', 'remote header inspection could not complete'],
  header_receipt_time_invalid: ['fail', 'remote header receipt time could not be recorded'],
})
const FALLBACK_HEADER_ERROR = ['blocked', 'remote header inspection could not complete']

class HeaderInspectionError extends Error {
  constructor(code, _status, _message, _details = {}) {
    const knownCode = typeof code === 'string' && Object.hasOwn(HEADER_ERROR_CONTRACTS, code)
    const contract = knownCode ? HEADER_ERROR_CONTRACTS[code] : FALLBACK_HEADER_ERROR
    super(contract[1])
    this.name = 'HeaderInspectionError'
    this.code = knownCode ? code : 'header_inspection_error'
    this.status = contract[0]
  }
}

function headerError(code, status, message, details = {}) {
  return new HeaderInspectionError(code, status, message, details)
}

function classifyHeaderInspectionError(error) {
  if (error instanceof HeaderInspectionError) {
    const knownCode = typeof error.code === 'string'
      && Object.hasOwn(HEADER_ERROR_CONTRACTS, error.code)
    const code = knownCode ? error.code : 'header_inspection_error'
    const contract = knownCode ? HEADER_ERROR_CONTRACTS[code] : FALLBACK_HEADER_ERROR
    return {
      status: contract[0],
      error_code: code,
      reason: contract[1],
    }
  }
  return {
    status: 'blocked',
    error_code: 'header_inspection_error',
    reason: 'remote header inspection could not complete (header_inspection_error)',
  }
}

function normalizePrefixBytes(value = DEFAULT_PREFIX_BYTES) {
  const prefixBytes = typeof value === 'number' ? value : Number(value)
  if (!Number.isSafeInteger(prefixBytes) || prefixBytes <= 0 || prefixBytes > MAX_PREFIX_BYTES) {
    throw new Error(`prefix byte budget must be between 1 and ${MAX_PREFIX_BYTES}`)
  }
  return prefixBytes
}

function safeMetadataToken(value) {
  return typeof value === 'string' && /^[A-Za-z0-9_.:+-]{1,128}$/.test(value)
    ? value
    : value === null || value === undefined
      ? null
      : '<invalid>'
}

function safeHostToken(value) {
  return typeof value === 'string' && /^[A-Za-z0-9_.:+-]{1,128}$/.test(value)
    ? value
    : 'redacted'
}

function parseBinaryVersionProvenance(version) {
  if (typeof version !== 'string' || !/^[A-Za-z0-9 ._+()-]{1,128}$/.test(version)) return null
  const match = /^camelid [A-Za-z0-9._+()-]+-g([0-9a-f]{7,40})(-dirty)?$/.exec(version)
  if (!match) return null
  return {
    binary_commit_abbrev: match[1],
    binary_reports_dirty: Boolean(match[2]),
  }
}

function normalizeInspectorIdentity(identity) {
  const parsedVersion = parseBinaryVersionProvenance(identity?.version)
  if (!identity || typeof identity !== 'object' || Array.isArray(identity)
    || !parsedVersion
    || !SHA256_RE.test(identity.binary_sha256 || '')
    || !REVISION_RE.test(identity.source_head || '')) {
    throw headerError(
      'header_inspector_identity_invalid',
      'fail',
      'camelid inspector identity is absent or invalid',
    )
  }
  const derived = {
    version: identity.version,
    binary_sha256: identity.binary_sha256,
    source_head: identity.source_head,
    binary_commit_abbrev: parsedVersion.binary_commit_abbrev,
    binary_reports_dirty: parsedVersion.binary_reports_dirty,
    binary_matches_source_head: identity.source_head.startsWith(parsedVersion.binary_commit_abbrev),
  }
  derived.clean_current_head = derived.binary_matches_source_head && !derived.binary_reports_dirty
  for (const field of [
    'binary_commit_abbrev',
    'binary_reports_dirty',
    'binary_matches_source_head',
    'clean_current_head',
  ]) {
    if (Object.hasOwn(identity, field) && identity[field] !== derived[field]) {
      throw headerError(
        'header_inspector_identity_invalid',
        'fail',
        'camelid inspector identity contains inconsistent provenance fields',
      )
    }
  }
  return derived
}

async function sha256File(path) {
  const hash = createHash('sha256')
  await new Promise((resolvePromise, reject) => {
    const input = createReadStream(path)
    input.on('data', (chunk) => hash.update(chunk))
    input.once('error', reject)
    input.once('end', resolvePromise)
  })
  return hash.digest('hex')
}

async function inspectBinaryIdentity(binary, {
  execImpl = execFileAsync,
  hashImpl = sha256File,
  gitImpl = execFileAsync,
  sourceRoot = resolve('.'),
} = {}) {
  if (typeof binary !== 'string' || !binary.trim()) {
    throw headerError('header_inspector_unavailable', 'blocked', 'camelid inspect-prefix binary is not configured')
  }
  let stdout
  try {
    ({ stdout } = await execImpl(binary, ['--version'], {
      timeout: 10_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
    }))
  } catch {
    throw headerError('header_inspector_unavailable', 'blocked', 'camelid inspector identity could not be read')
  }
  let binarySha256
  try { binarySha256 = await hashImpl(binary) }
  catch {
    throw headerError('header_inspector_unavailable', 'blocked', 'camelid inspector binary could not be hashed')
  }
  let sourceHead
  try {
    const result = await gitImpl('git', ['rev-parse', 'HEAD'], {
      cwd: sourceRoot,
      timeout: 10_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
    })
    sourceHead = String(result.stdout || '').trim()
  } catch {
    throw headerError('header_inspector_unavailable', 'blocked', 'current source HEAD could not be read')
  }
  return normalizeInspectorIdentity({
    version: String(stdout || '').trim(),
    binary_sha256: binarySha256,
    source_head: sourceHead,
  })
}

function validateImmutableLock(lock) {
  if (!lock || typeof lock !== 'object' || Array.isArray(lock)) {
    throw headerError('source_lock_invalid', 'blocked', 'immutable source lock is absent or invalid')
  }
  if (typeof lock.repo !== 'string' || !lock.repo
    || typeof lock.file !== 'string' || !lock.file
    || !REVISION_RE.test(lock.revision || '')
    || !Number.isSafeInteger(lock.size_bytes) || lock.size_bytes <= 0
    || !SHA256_RE.test(lock.sha256 || '')) {
    throw headerError('source_lock_invalid', 'blocked', 'immutable source lock is incomplete or invalid')
  }
  let downloadUrl
  try { downloadUrl = new URL(lock.download_url) }
  catch {
    throw headerError('source_lock_invalid', 'blocked', 'immutable source lock has no valid download URL')
  }
  if (downloadUrl.protocol !== 'https:' || downloadUrl.hostname !== 'huggingface.co') {
    throw headerError('source_lock_invalid', 'blocked', 'immutable source lock download URL is not an HTTPS Hugging Face URL')
  }
  const expectedPath = `/${lock.repo.split('/').map(encodeURIComponent).join('/')}`
    + `/resolve/${lock.revision}/`
    + lock.file.split('/').map(encodeURIComponent).join('/')
  if (downloadUrl.pathname !== expectedPath) {
    throw headerError('source_lock_invalid', 'blocked', 'source lock download URL does not match its repo, file, and immutable revision')
  }
}

function parseArgs(argv) {
  const args = new Map()
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    if (!arg.startsWith('--')) continue
    const [key, inline] = arg.slice(2).split('=', 2)
    const next = argv[index + 1]
    args.set(key, inline ?? (next && !next.startsWith('--') ? argv[++index] : 'true'))
  }
  return args
}

function parseContentRange(value) {
  const match = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(value || '')
  if (!match) throw new Error(`invalid or missing Content-Range: ${JSON.stringify(value)}`)
  const parsed = { start: Number(match[1]), end: Number(match[2]), total: Number(match[3]) }
  if (!Object.values(parsed).every(Number.isSafeInteger)
    || parsed.start < 0 || parsed.end < parsed.start || parsed.total <= parsed.end) {
    throw new Error(`invalid or missing Content-Range: ${JSON.stringify(value)}`)
  }
  return parsed
}

async function fetchHeaderPrefix(lock, {
  prefixBytes = DEFAULT_PREFIX_BYTES,
  token = null,
  fetchImpl = fetch,
} = {}) {
  validateImmutableLock(lock)
  prefixBytes = normalizePrefixBytes(prefixBytes)
  const requested = Math.min(prefixBytes, lock.size_bytes)
  const headers = { Range: `bytes=0-${requested - 1}`, 'Accept-Encoding': 'identity' }
  if (token) headers.Authorization = `Bearer ${token}`
  let response
  try {
    response = await fetchImpl(lock.download_url, {
      headers,
      redirect: 'follow',
      signal: AbortSignal.timeout(60_000),
    })
  } catch {
    throw headerError(
      'header_fetch_unavailable',
      'blocked',
      'remote header range request could not complete (header_fetch_unavailable)',
    )
  }
  if (response.status !== 206) {
    await response.body?.cancel?.()
    throw headerError(
      'header_range_unavailable',
      'blocked',
      `range request returned HTTP ${response.status}; refusing a possible full-model download`,
    )
  }
  let contentRange
  try {
    contentRange = parseContentRange(response.headers.get('content-range'))
  } catch {
    await response.body?.cancel?.()
    throw headerError(
      'header_range_invalid',
      'fail',
      'range response has an invalid or missing Content-Range header',
    )
  }
  if (contentRange.start !== 0 || contentRange.total !== lock.size_bytes) {
    await response.body?.cancel?.()
    throw headerError(
      'header_range_identity_mismatch',
      'fail',
      'range response identity does not match the immutable source lock',
    )
  }
  const expectedLength = contentRange.end - contentRange.start + 1
  if (expectedLength < requested) {
    await response.body?.cancel?.()
    throw headerError(
      'header_range_incomplete',
      'blocked',
      `range response returned ${expectedLength} bytes, shorter than the ${requested}-byte request`,
    )
  }
  if (expectedLength > requested) {
    await response.body?.cancel?.()
    throw headerError(
      'header_body_budget_exceeded',
      'blocked',
      `range response exceeds the ${requested}-byte request budget`,
    )
  }
  if (!response.body) {
    throw headerError('header_body_missing', 'blocked', 'range response has no body')
  }
  const chunks = []
  let received = 0
  const reader = response.body.getReader()
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      received += value.byteLength
      if (received > requested) {
        await reader.cancel()
        throw headerError(
          'header_body_budget_exceeded',
          'blocked',
          `range body exceeded the ${requested}-byte request budget`,
        )
      }
      chunks.push(Buffer.from(value))
    }
  } catch (error) {
    if (error instanceof HeaderInspectionError) throw error
    throw headerError(
      'header_body_unavailable',
      'blocked',
      'remote header range body could not be read (header_body_unavailable)',
    )
  }
  const bytes = Buffer.concat(chunks, received)
  if (bytes.length !== expectedLength || bytes.length > requested) {
    throw headerError(
      'header_body_length_mismatch',
      'fail',
      `range body length ${bytes.length} does not match Content-Range length ${expectedLength}`,
    )
  }
  return {
    bytes,
    content_range: contentRange,
    requested_bytes: requested,
    prefix_sha256: createHash('sha256').update(bytes).digest('hex'),
  }
}

function summarizeInspection(inspection) {
  if (!inspection || typeof inspection !== 'object' || Array.isArray(inspection)
    || !Array.isArray(inspection.tensors)
    || !inspection.metadata || typeof inspection.metadata !== 'object' || Array.isArray(inspection.metadata)) {
    throw headerError('header_inspection_contract_invalid', 'fail', 'inspect-prefix returned an invalid inspection object')
  }
  if (![2, 3].includes(inspection.version)
    || !Number.isSafeInteger(inspection.tensor_count) || inspection.tensor_count < 0
    || inspection.tensor_count !== inspection.tensors.length
    || !Number.isSafeInteger(inspection.metadata_count) || inspection.metadata_count < 0
    || !Number.isSafeInteger(inspection.alignment) || inspection.alignment <= 0
    || !Number.isSafeInteger(inspection.data_start_offset) || inspection.data_start_offset < 0) {
    throw headerError('header_inspection_contract_invalid', 'fail', 'inspect-prefix returned inconsistent header counts or offsets')
  }
  const tensors = inspection.tensors.map((tensor) => {
    if (!tensor || typeof tensor !== 'object' || Array.isArray(tensor)
      || typeof tensor.name !== 'string'
      || !Array.isArray(tensor.dimensions)
      || !tensor.dimensions.every((value) => Number.isSafeInteger(value) && value >= 0)
      || typeof tensor.tensor_type !== 'string'
      || !/^[A-Za-z][A-Za-z0-9_]{0,31}$/.test(tensor.tensor_type)
      || !Number.isSafeInteger(tensor.relative_offset) || tensor.relative_offset < 0
      || !Number.isSafeInteger(tensor.n_bytes) || tensor.n_bytes < 0) {
      throw headerError('header_inspection_contract_invalid', 'fail', 'inspect-prefix returned an invalid tensor descriptor')
    }
    return {
      name: tensor.name,
      dimensions: tensor.dimensions,
      tensor_type: tensor.tensor_type,
      relative_offset: tensor.relative_offset,
      n_bytes: tensor.n_bytes,
    }
  })
  const tensorTypes = {}
  let tensorBytes = 0
  for (const tensor of tensors) {
    tensorTypes[tensor.tensor_type] = (tensorTypes[tensor.tensor_type] || 0) + 1
    tensorBytes += tensor.n_bytes
    if (!Number.isSafeInteger(tensorBytes)) {
      throw headerError('header_inspection_contract_invalid', 'fail', 'tensor inventory byte count exceeds the safe integer range')
    }
  }
  const inventorySha256 = createHash('sha256')
    .update(JSON.stringify(tensors))
    .digest('hex')
  const quantCounts = Object.entries(tensorTypes)
    .filter(([type]) => !['F32', 'F16', 'BF16'].includes(type))
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
  const metadata = inspection.metadata
  return {
    version: inspection.version,
    tensor_count: inspection.tensor_count,
    metadata_count: inspection.metadata_count,
    alignment: inspection.alignment,
    data_start_offset: inspection.data_start_offset,
    observed: {
      architecture: safeMetadataToken(metadata['general.architecture']),
      general_file_type: Number.isSafeInteger(metadata['general.file_type'])
        ? metadata['general.file_type']
        : null,
      tokenizer_model: safeMetadataToken(metadata['tokenizer.ggml.model']),
      tokenizer_pre: safeMetadataToken(metadata['tokenizer.ggml.pre']),
      headline_quant: quantCounts[0]?.[0] || null,
    },
    tensor_inventory: {
      sha256: inventorySha256,
      total_n_bytes: tensorBytes,
      types: tensorTypes,
    },
  }
}

async function selectRow(root, rosterPath, rowId) {
  const absolute = resolve(root, rosterPath)
  const roster = JSON.parse(await readFile(absolute, 'utf8'))
  const errors = validateRoster(roster, absolute)
  if (errors.length) throw new Error(`roster is invalid:\n${errors.join('\n')}`)
  const row = roster.rows.find((candidate) => candidate.id === rowId)
  if (!row) throw new Error(`unknown --row ${JSON.stringify(rowId)}`)
  if (!row.source.repo || !row.source.file || !row.source.revision) {
    throw new Error(`${row.id} does not have a fully pinned Hugging Face source`)
  }
  return row
}

async function inspectPrefix(binary, prefixPath, declaredLength, { execImpl = execFileAsync } = {}) {
  if (typeof binary !== 'string' || !binary.trim()) {
    throw headerError('header_inspector_unavailable', 'blocked', 'camelid inspect-prefix binary is not configured')
  }
  let stdout
  try {
    ({ stdout } = await execImpl(binary, [
      'inspect-prefix', prefixPath, '--declared-len', String(declaredLength),
    ], { timeout: 90_000, maxBuffer: 256 * 1024 * 1024, windowsHide: true }))
  } catch (error) {
    const code = typeof error?.code === 'string' ? error.code.toLowerCase() : error?.code
    if (error?.killed || error?.signal || code === 'etimedout') {
      throw headerError('header_inspector_timeout', 'blocked', 'camelid inspect-prefix timed out')
    }
    if (['enoent', 'eacces', 'eperm'].includes(code)) {
      throw headerError('header_inspector_unavailable', 'blocked', 'camelid inspect-prefix could not start')
    }
    if (code === 'err_child_process_stdio_maxbuffer') {
      throw headerError('header_inspector_output_budget', 'blocked', 'camelid inspect-prefix exceeded its output budget')
    }
    throw headerError('header_parse_failed', 'fail', 'camelid inspect-prefix rejected the remote GGUF header')
  }
  try { return JSON.parse(stdout) }
  catch {
    throw headerError('header_inspector_output_invalid', 'fail', 'camelid inspect-prefix did not emit valid JSON')
  }
}

async function inspectRemoteHeader(lock, {
  binary,
  rowId = null,
  prefixBytes = DEFAULT_PREFIX_BYTES,
  token = null,
  fetchImpl = fetch,
  inspectImpl = inspectPrefix,
  identityImpl = inspectBinaryIdentity,
  sourceRoot = resolve('.'),
  now = () => new Date(),
} = {}) {
  validateImmutableLock(lock)
  if (typeof binary !== 'string' || !binary.trim()) {
    throw headerError('header_inspector_unavailable', 'blocked', 'camelid inspect-prefix binary is not configured')
  }
  if (rowId !== null && (typeof rowId !== 'string' || !/^[a-z0-9_]{1,128}$/.test(rowId))) {
    throw headerError('header_row_id_invalid', 'fail', 'remote header inspection row id is invalid')
  }
  let inspector
  try { inspector = normalizeInspectorIdentity(await identityImpl(binary, { sourceRoot })) }
  catch (error) {
    if (error instanceof HeaderInspectionError) throw error
    throw headerError('header_inspector_unavailable', 'blocked', 'camelid inspector identity could not be read')
  }
  if (!inspector.clean_current_head) {
    throw headerError(
      'header_inspector_not_clean_current_head',
      'blocked',
      'camelid inspector is not a clean build of current source HEAD',
    )
  }
  const ranged = await fetchHeaderPrefix(lock, { prefixBytes, token, fetchImpl })
  const temporary = await mkdtemp(join(tmpdir(), 'camelid-hf-header-'))
  try {
    const prefixPath = join(temporary, 'header.gguf')
    await writeFile(prefixPath, ranged.bytes)
    let inspection
    try {
      inspection = await inspectImpl(binary, prefixPath, lock.size_bytes)
    } catch (error) {
      if (error instanceof HeaderInspectionError) throw error
      throw headerError(
        'header_inspection_error',
        'blocked',
        'remote header inspection could not complete (header_inspection_error)',
      )
    }
    let generatedAt
    try { generatedAt = now().toISOString() }
    catch {
      throw headerError('header_receipt_time_invalid', 'fail', 'remote header receipt time could not be recorded')
    }
    return {
      schema: 'camelid.remote-gguf-header-inspection/v1',
      ...(rowId ? { row_id: rowId } : {}),
      generated_at: generatedAt,
      host: {
        hostname_redacted: true,
        platform: safeHostToken(platform()),
        release: safeHostToken(release()),
        arch: safeHostToken(arch()),
      },
      inspector: {
        ...inspector,
        binary_path_redacted: true,
        command: [
          '<camelid>',
          'inspect-prefix',
          '<remote-gguf-prefix>',
          '--declared-len',
          String(lock.size_bytes),
        ],
      },
      source: {
        repo: lock.repo,
        file: lock.file,
        revision: lock.revision,
        size_bytes: lock.size_bytes,
        sha256: lock.sha256,
      },
      range: {
        requested_bytes: ranged.requested_bytes,
        received_bytes: ranged.bytes.length,
        content_range: ranged.content_range,
        prefix_sha256: ranged.prefix_sha256,
      },
      inspection: summarizeInspection(inspection),
      scope: {
        prefix_sha256: 'all_received_prefix_bytes',
        tensor_payload: 'partially_range_fetched_opaque',
        full_artifact_sha256: 'not_run',
        tensor_payload_interpretation: 'not_run',
        load: 'not_run',
        generation: 'not_run',
      },
      support_claim: false,
      note: 'Remote header inspection validates metadata and tensor descriptors against the pinned full length. The prefix SHA-256 covers every received byte, including any opaque initial tensor payload after data_start_offset; the remaining payload is not fetched, and no tensor payload is interpreted, loaded as model weights, or used for inference or generation.',
    }
  } finally {
    await rm(temporary, { recursive: true, force: true })
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  if (args.has('help') || !args.get('row')) {
    console.log(`Usage:
  node scripts/hf-qualification-header.mjs --row <roster-id> [options]

Options:
  --roster <path>        Roster path (default: Phase 1)
  --camelid <path>       Camelid binary (default: CAMELID_BIN or target/debug/camelid)
  --prefix-bytes <n>     Range budget, max 64 MiB (default: 32 MiB)
  --out <path>           Write the scrubbed inspection receipt
  HF_TOKEN               Optional token for gated/private rows
`)
    process.exit(args.has('help') ? 0 : 1)
  }

  const root = resolve('.')
  const row = await selectRow(root, args.get('roster') || 'qa/model-qualification/phase1-roster.json', args.get('row'))
  const lock = await resolveHfSource({
    repo: row.source.repo,
    file: row.source.file,
    revision: row.source.revision,
    token: process.env.HF_TOKEN || null,
  })
  validateLockAgainstSelection(lock, {
    row_id: row.id,
    repo: row.source.repo,
    file: row.source.file,
    revision: row.source.revision,
    expected: {
      size_bytes: row.identity.size_bytes,
      sha256: row.identity.sha256,
      license: row.source.license,
    },
  })
  const defaultBinary = process.platform === 'win32' ? 'target/debug/camelid.exe' : 'target/debug/camelid'
  const report = await inspectRemoteHeader(lock, {
    binary: resolve(args.get('camelid') || process.env.CAMELID_BIN || defaultBinary),
    rowId: row.id,
    sourceRoot: root,
    prefixBytes: normalizePrefixBytes(args.get('prefix-bytes') || DEFAULT_PREFIX_BYTES),
    token: process.env.HF_TOKEN || null,
  })
  const rendered = `${JSON.stringify(report, null, 2)}\n`
  if (args.get('out')) {
    const out = resolve(args.get('out'))
    await mkdir(dirname(out), { recursive: true })
    await writeFile(out, rendered)
  }
  process.stdout.write(rendered)
}

export {
  DEFAULT_PREFIX_BYTES,
  HeaderInspectionError,
  MAX_PREFIX_BYTES,
  classifyHeaderInspectionError,
  fetchHeaderPrefix,
  inspectBinaryIdentity,
  inspectPrefix,
  inspectRemoteHeader,
  normalizeInspectorIdentity,
  normalizePrefixBytes,
  parseContentRange,
  summarizeInspection,
  validateImmutableLock,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    if (error instanceof HeaderInspectionError) {
      const failure = classifyHeaderInspectionError(error)
      console.error(`${failure.error_code}: ${failure.reason}`)
    } else {
      console.error(error)
    }
    process.exit(1)
  })
}
