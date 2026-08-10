#!/usr/bin/env node
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { validateRoster } from './check-model-qualification-roster.mjs'

const SHA256_RE = /^[0-9a-f]{64}$/
const REVISION_RE = /^[0-9a-f]{40}$/

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

function encodedRepo(repo) {
  return repo.split('/').map(encodeURIComponent).join('/')
}

function modelInfoUrl(repo, revision = null) {
  const revisionPath = revision ? `/revision/${encodeURIComponent(revision)}` : ''
  return `https://huggingface.co/api/models/${encodedRepo(repo)}${revisionPath}?blobs=true`
}

function resolveLicense(info) {
  const cardLicense = info.cardData?.license
  if (typeof cardLicense === 'string' && cardLicense) {
    if (cardLicense.toLowerCase() === 'other'
      && typeof info.cardData?.license_name === 'string'
      && info.cardData.license_name) {
      return info.cardData.license_name
    }
    return cardLicense.toLowerCase()
  }
  if (Array.isArray(cardLicense) && cardLicense.length === 1) return String(cardLicense[0]).toLowerCase()
  const licenseTag = (info.tags || []).find((tag) => typeof tag === 'string' && tag.startsWith('license:'))
  return licenseTag ? licenseTag.slice('license:'.length).toLowerCase() : null
}

function sourceLockFromModelInfo({ repo, file, requestedRevision = null, info }) {
  if (!info || typeof info !== 'object') throw new Error('Hugging Face model info response is not an object')
  if (!REVISION_RE.test(info.sha || '')) throw new Error(`Hub did not return an immutable 40-character revision: ${JSON.stringify(info.sha)}`)
  if (requestedRevision && info.sha !== requestedRevision) {
    throw new Error(`Hub resolved revision ${info.sha}, expected ${requestedRevision}`)
  }
  const sibling = (info.siblings || []).find((candidate) => candidate.rfilename === file)
  if (!sibling) throw new Error(`file ${JSON.stringify(file)} is absent from ${repo}@${info.sha}`)
  const sizeBytes = sibling.lfs?.size ?? sibling.size ?? null
  const sha256 = sibling.lfs?.sha256 ?? null
  if (!Number.isInteger(sizeBytes) || sizeBytes <= 0) {
    throw new Error(`Hub did not return a positive byte size for ${file}; files metadata is required`)
  }
  if (!SHA256_RE.test(sha256 || '')) {
    throw new Error(`Hub did not return an LFS SHA-256 for ${file}; exact-byte qualification cannot proceed`)
  }
  return {
    schema: 'camelid.hf-source-lock/v1',
    repo,
    file,
    revision: info.sha,
    size_bytes: sizeBytes,
    sha256,
    license: resolveLicense(info),
    access: {
      gated: Boolean(info.gated),
      private: Boolean(info.private),
      disabled: Boolean(info.disabled),
    },
    download_url: `https://huggingface.co/${encodedRepo(repo)}/resolve/${info.sha}/${file.split('/').map(encodeURIComponent).join('/')}?download=true`,
  }
}

async function resolveHfSource({ repo, file, revision = null, token = null, fetchImpl = fetch }) {
  if (!repo || !file) throw new Error('repo and file are required')
  const headers = token ? { Authorization: `Bearer ${token}` } : {}
  const response = await fetchImpl(modelInfoUrl(repo, revision), {
    headers,
    signal: AbortSignal.timeout(30_000),
  })
  if (!response.ok) {
    throw new Error(`Hugging Face model-info request failed (${response.status} ${response.statusText})`)
  }
  const info = await response.json()
  return sourceLockFromModelInfo({ repo, file, requestedRevision: revision, info })
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

async function verifyArtifact(path, lock) {
  const absolute = resolve(path)
  let artifactStat
  try { artifactStat = await stat(absolute) }
  catch (error) {
    return { status: 'blocked', path_redacted: true, reason: `artifact is absent (${error.code || 'unreadable'})` }
  }
  const sha256 = await sha256File(absolute)
  const mismatches = []
  if (artifactStat.size !== lock.size_bytes) mismatches.push(`size ${artifactStat.size} != ${lock.size_bytes}`)
  if (sha256 !== lock.sha256) mismatches.push(`sha256 ${sha256} != ${lock.sha256}`)
  return mismatches.length
    ? { status: 'fail', path_redacted: true, size_bytes: artifactStat.size, sha256, reason: mismatches.join('; ') }
    : { status: 'pass', path_redacted: true, size_bytes: artifactStat.size, sha256 }
}

async function rowSelector(root, rosterPath, rowId) {
  const absoluteRoster = resolve(root, rosterPath)
  const roster = JSON.parse(await readFile(absoluteRoster, 'utf8'))
  const errors = validateRoster(roster, absoluteRoster)
  if (errors.length) throw new Error(`roster is invalid:\n${errors.join('\n')}`)
  const row = roster.rows.find((candidate) => candidate.id === rowId)
  if (!row) throw new Error(`unknown --row ${JSON.stringify(rowId)}`)
  if (!row.source.repo || !row.source.file) throw new Error(`${row.id} has no exact Hugging Face repo/file selector`)
  return {
    row_id: row.id,
    repo: row.source.repo,
    file: row.source.file,
    revision: row.source.revision,
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  if (args.has('help') || (!args.get('row') && (!args.get('repo') || !args.get('file')))) {
    console.log(`Usage:
  node scripts/hf-qualification-source.mjs --row <roster-id> [options]
  node scripts/hf-qualification-source.mjs --repo <org/repo> --file <path> [options]

Options:
  --roster <path>       Roster path (default: Phase 1)
  --revision <sha>      Expected immutable revision; unpinned selectors resolve current HEAD
  --artifact <path>     Hash an existing local artifact against the source lock
  --out <path>          Write the scrubbed source lock/result JSON
  HF_TOKEN              Optional token for gated/private model metadata
`)
    process.exit(args.has('help') ? 0 : 1)
  }

  const root = resolve('.')
  const selected = args.get('row')
    ? await rowSelector(root, args.get('roster') || 'qa/model-qualification/phase1-roster.json', args.get('row'))
    : { row_id: null, repo: args.get('repo'), file: args.get('file'), revision: args.get('revision') || null }
  if (args.get('revision')) selected.revision = args.get('revision')
  const lock = await resolveHfSource({ ...selected, token: process.env.HF_TOKEN || null })
  const result = {
    ...lock,
    row_id: selected.row_id,
    resolved_at: new Date().toISOString(),
    artifact: args.get('artifact') ? await verifyArtifact(args.get('artifact'), lock) : { status: 'not_checked' },
  }
  const rendered = `${JSON.stringify(result, null, 2)}\n`
  if (args.get('out')) {
    const out = resolve(args.get('out'))
    await mkdir(dirname(out), { recursive: true })
    await writeFile(out, rendered)
  }
  process.stdout.write(rendered)
}

export {
  modelInfoUrl,
  resolveHfSource,
  sourceLockFromModelInfo,
  verifyArtifact,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
