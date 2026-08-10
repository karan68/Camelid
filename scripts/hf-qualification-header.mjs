#!/usr/bin/env node
import { createHash } from 'node:crypto'
import { execFile } from 'node:child_process'
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import { validateRoster } from './check-model-qualification-roster.mjs'
import { resolveHfSource } from './hf-qualification-source.mjs'

const execFileAsync = promisify(execFile)
const DEFAULT_PREFIX_BYTES = 32 * 1024 * 1024
const MAX_PREFIX_BYTES = 64 * 1024 * 1024

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
  return { start: Number(match[1]), end: Number(match[2]), total: Number(match[3]) }
}

async function fetchHeaderPrefix(lock, {
  prefixBytes = DEFAULT_PREFIX_BYTES,
  token = null,
  fetchImpl = fetch,
} = {}) {
  if (!Number.isSafeInteger(prefixBytes) || prefixBytes <= 0 || prefixBytes > MAX_PREFIX_BYTES) {
    throw new Error(`prefix byte budget must be between 1 and ${MAX_PREFIX_BYTES}`)
  }
  const requested = Math.min(prefixBytes, lock.size_bytes)
  const headers = { Range: `bytes=0-${requested - 1}` }
  if (token) headers.Authorization = `Bearer ${token}`
  const response = await fetchImpl(lock.download_url, {
    headers,
    redirect: 'follow',
    signal: AbortSignal.timeout(60_000),
  })
  if (response.status !== 206) {
    await response.body?.cancel?.()
    throw new Error(`range request returned HTTP ${response.status}; refusing a possible full-model download`)
  }
  const contentRange = parseContentRange(response.headers.get('content-range'))
  if (contentRange.start !== 0 || contentRange.total !== lock.size_bytes) {
    await response.body?.cancel?.()
    throw new Error(`range identity mismatch: ${JSON.stringify(contentRange)} for ${lock.size_bytes} bytes`)
  }
  if (!response.body) throw new Error('range response has no body')
  const chunks = []
  let received = 0
  const reader = response.body.getReader()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    received += value.byteLength
    if (received > requested) {
      await reader.cancel()
      throw new Error(`range body exceeded the ${requested}-byte request budget`)
    }
    chunks.push(Buffer.from(value))
  }
  const bytes = Buffer.concat(chunks, received)
  const expectedLength = contentRange.end - contentRange.start + 1
  if (bytes.length !== expectedLength || bytes.length > requested) {
    throw new Error(`range body length ${bytes.length} does not match Content-Range length ${expectedLength}`)
  }
  return { bytes, content_range: contentRange, requested_bytes: requested }
}

function summarizeInspection(inspection) {
  const omittedMetadata = []
  const metadata = {}
  for (const [key, value] of Object.entries(inspection.metadata || {})) {
    const encodedLength = Buffer.byteLength(JSON.stringify(value))
    if (encodedLength > 64 * 1024) {
      omittedMetadata.push({ key, encoded_bytes: encodedLength })
    } else {
      metadata[key] = value
    }
  }
  const tensors = (inspection.tensors || []).map((tensor) => ({
    name: tensor.name,
    dimensions: tensor.dimensions,
    tensor_type: tensor.tensor_type,
    relative_offset: tensor.relative_offset,
    n_bytes: tensor.n_bytes,
  }))
  const tensorTypes = {}
  let tensorBytes = 0
  for (const tensor of tensors) {
    tensorTypes[tensor.tensor_type] = (tensorTypes[tensor.tensor_type] || 0) + 1
    tensorBytes += tensor.n_bytes
  }
  const inventorySha256 = createHash('sha256')
    .update(JSON.stringify(tensors))
    .digest('hex')
  return {
    version: inspection.version,
    tensor_count: inspection.tensor_count,
    metadata_count: inspection.metadata_count,
    alignment: inspection.alignment,
    data_start_offset: inspection.data_start_offset,
    metadata,
    omitted_metadata: omittedMetadata,
    tensor_inventory: {
      sha256: inventorySha256,
      total_n_bytes: tensorBytes,
      types: tensorTypes,
      tensors,
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

async function inspectPrefix(binary, prefixPath, declaredLength) {
  const { stdout } = await execFileAsync(binary, [
    'inspect-prefix', prefixPath, '--declared-len', String(declaredLength),
  ], { timeout: 90_000, maxBuffer: 256 * 1024 * 1024, windowsHide: true })
  return JSON.parse(stdout)
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
  const ranged = await fetchHeaderPrefix(lock, {
    prefixBytes: Number(args.get('prefix-bytes') || DEFAULT_PREFIX_BYTES),
    token: process.env.HF_TOKEN || null,
  })
  const temporary = await mkdtemp(join(tmpdir(), 'camelid-hf-header-'))
  try {
    const prefixPath = join(temporary, 'header.gguf')
    await writeFile(prefixPath, ranged.bytes)
    const defaultBinary = process.platform === 'win32' ? 'target/debug/camelid.exe' : 'target/debug/camelid'
    const inspection = await inspectPrefix(resolve(args.get('camelid') || process.env.CAMELID_BIN || defaultBinary), prefixPath, lock.size_bytes)
    const report = {
      schema: 'camelid.remote-gguf-header-inspection/v1',
      row_id: row.id,
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
      },
      inspection: summarizeInspection(inspection),
      support_claim: false,
      note: 'Remote header inspection validates metadata and tensor descriptors against the pinned full length; it does not hash tensor payload bytes, load weights, or prove generation.',
    }
    const rendered = `${JSON.stringify(report, null, 2)}\n`
    if (args.get('out')) {
      const out = resolve(args.get('out'))
      await mkdir(dirname(out), { recursive: true })
      await writeFile(out, rendered)
    }
    process.stdout.write(rendered)
  } finally {
    await rm(temporary, { recursive: true, force: true })
  }
}

export { fetchHeaderPrefix, parseContentRange, summarizeInspection }

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
