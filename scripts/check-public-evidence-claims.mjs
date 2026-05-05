#!/usr/bin/env node
import { access, readdir, readFile } from 'node:fs/promises'
import { join, relative, resolve } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const rootDir = resolve(args.get('root') || 'qa/evidence-bundles')
const failures = []
let checkedBundles = 0
let checkedSummaries = 0

const manifestPaths = await findManifestPaths(rootDir)
for (const manifestPath of manifestPaths) {
  checkedBundles += 1
  await validateBundle(manifestPath)
}

if (failures.length > 0) {
  console.error(`public evidence claim check failed with ${failures.length} finding(s):`)
  for (const failure of failures) console.error(`- ${failure}`)
  process.exit(1)
}

console.log(`public evidence claim check passed: ${checkedBundles} manifest(s), ${checkedSummaries} summary file(s)`)

async function validateBundle(manifestPath) {
  const bundleDir = manifestPath.slice(0, -'/manifest.json'.length)
  const bundleRel = relative(process.cwd(), bundleDir) || '.'
  const manifest = await readJson(manifestPath)

  if (!manifest || typeof manifest !== 'object') {
    fail(bundleRel, 'manifest.json is not a JSON object')
    return
  }
  const schema = typeof manifest.schema === 'string' ? manifest.schema : ''

  const summaryPath = join(bundleDir, 'summary.json')
  const summaryExists = await exists(summaryPath)
  if (summaryExists) {
    checkedSummaries += 1
    const summary = await readJson(summaryPath)
    validateSummaryAgreement(bundleRel, manifest, summary)
  }

  if (schema === 'camelid.four_row_context_512_public_evidence.v1') {
    if (!summaryExists) fail(bundleRel, 'four-row context-512 bundle must include summary.json')
    await validateFourRowContext512(bundleRel, manifest, summaryExists ? await readJson(summaryPath) : null)
  }
}

function validateSummaryAgreement(bundleRel, manifest, summary) {
  if (!summary || typeof summary !== 'object') {
    fail(bundleRel, 'summary.json is not a JSON object')
    return
  }
  if (summary.source_manifest !== undefined && summary.source_manifest !== 'manifest.json') {
    fail(bundleRel, `summary.json source_manifest must be manifest.json, got ${JSON.stringify(summary.source_manifest)}`)
  }
  if (typeof manifest.passed === 'boolean' && typeof summary.passed === 'boolean' && manifest.passed !== summary.passed) {
    fail(bundleRel, `manifest passed=${manifest.passed} disagrees with summary passed=${summary.passed}`)
  }
  if (typeof manifest.claim_boundary === 'string' && typeof summary.claim_boundary === 'string' && manifest.claim_boundary !== summary.claim_boundary) {
    fail(bundleRel, 'manifest and summary claim_boundary differ')
  }
  if (Array.isArray(manifest.rows) && Array.isArray(summary.rows)) {
    if (manifest.rows.length !== summary.rows.length) {
      fail(bundleRel, `manifest has ${manifest.rows.length} row(s) but summary has ${summary.rows.length}`)
      return
    }
    for (let index = 0; index < manifest.rows.length; index += 1) {
      const manifestRow = manifest.rows[index]
      const summaryRow = summary.rows[index]
      const rowId = manifestRow.row_id || `row-${index}`
      compareRowField(bundleRel, rowId, manifestRow, summaryRow, 'row_id')
      compareRowField(bundleRel, rowId, manifestRow, summaryRow, 'context_window')
      compareRowField(bundleRel, rowId, manifestRow, summaryRow, 'reference_prompt_token_count')
      compareRowField(bundleRel, rowId, manifestRow, summaryRow, 'max_tokens')
      compareRowField(bundleRel, rowId, manifestRow, summaryRow, 'max_resident_set_kib')
      if (typeof summaryRow.passed === 'boolean') {
        const manifestPassed = rowPassed(manifestRow)
        if (manifestPassed !== summaryRow.passed) fail(bundleRel, `${rowId} summary passed=${summaryRow.passed} disagrees with manifest row checks`)
      }
    }
  }
}

function rowPassed(row) {
  if (typeof row?.passed === 'boolean') return row.passed
  if (
    typeof row?.prompt_tokens_match === 'boolean' ||
    typeof row?.generated_tokens_match === 'boolean' ||
    typeof row?.generated_text_match === 'boolean'
  ) {
    return row.prompt_tokens_match === true && row.generated_tokens_match === true && row.generated_text_match === true
  }
  if (
    typeof row?.prompt_tokens_all_match === 'boolean' ||
    typeof row?.generated_tokens_all_match === 'boolean' ||
    typeof row?.generated_text_all_match === 'boolean'
  ) {
    return row.prompt_tokens_all_match === true && row.generated_tokens_all_match === true && row.generated_text_all_match === true
  }
  return undefined
}

async function validateFourRowContext512(bundleRel, manifest, summary) {
  const expectedRows = [
    'tinyllama_1_1b_chat_q8_0',
    'llama32_1b_instruct_q8_0',
    'llama32_3b_instruct_q8_0',
    'llama3_8b_instruct_q8_0',
  ]

  if (manifest.passed !== true) fail(bundleRel, 'four-row context-512 manifest must be passed=true')
  if (manifest.checkout_clean !== true) fail(bundleRel, 'four-row context-512 manifest must record checkout_clean=true')
  if (manifest.pack?.target_context_window !== 512) fail(bundleRel, 'four-row context-512 manifest pack target_context_window must be 512')
  if (manifest.pack?.max_tokens !== 5) fail(bundleRel, 'four-row context-512 manifest pack max_tokens must be 5')
  if (manifest.pack?.source_prompt_pack !== 'qa/prompt-packs/llama3-context-512-smoke.json') {
    fail(bundleRel, 'four-row context-512 manifest source_prompt_pack must stay on the checked llama3 context pack')
  }
  if (!boundaryIsNarrow(manifest.claim_boundary)) fail(bundleRel, 'four-row context-512 claim_boundary must explicitly avoid broader/full-family promotion')

  validateChecksObject(bundleRel, 'summary.checks', summary?.checks, [
    'checkout_clean',
    'prompt_tokens_all_match',
    'generated_tokens_all_match',
    'generated_text_all_match',
    'all_rows_have_bounded_rss',
  ])

  if (!Array.isArray(manifest.rows)) {
    fail(bundleRel, 'four-row context-512 manifest rows must be an array')
    return
  }
  const rowIds = manifest.rows.map((row) => row.row_id)
  if (JSON.stringify(rowIds) !== JSON.stringify(expectedRows)) {
    fail(bundleRel, `four-row context-512 row order changed: ${JSON.stringify(rowIds)}`)
  }
  for (const row of manifest.rows) validateContext512Row(bundleRel, row)
}

function validateContext512Row(bundleRel, row) {
  const rowId = row.row_id || '<missing row_id>'
  if (row.context_window !== 512) fail(bundleRel, `${rowId} context_window must be 512`)
  if (row.max_tokens !== 5) fail(bundleRel, `${rowId} max_tokens must be 5`)
  if (row.prompt_tokens_match !== true) fail(bundleRel, `${rowId} prompt_tokens_match must be true`)
  if (row.generated_tokens_match !== true) fail(bundleRel, `${rowId} generated_tokens_match must be true`)
  if (row.generated_text_match !== true) fail(bundleRel, `${rowId} generated_text_match must be true`)
  if (row.first_generated_token_diff_index !== -1) fail(bundleRel, `${rowId} first_generated_token_diff_index must be -1`)
  if (!Number.isInteger(row.reference_prompt_token_count) || row.reference_prompt_token_count <= 0) fail(bundleRel, `${rowId} reference_prompt_token_count must be positive`)
  if (!Number.isInteger(row.max_resident_set_kib) || row.max_resident_set_kib <= 0) fail(bundleRel, `${rowId} max_resident_set_kib must be positive`)
  if (typeof row.model_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(row.model_sha256)) fail(bundleRel, `${rowId} model_sha256 must be a 64-character lowercase sha256`)
  if (typeof row.raw_artifact !== 'string' || row.raw_artifact.startsWith('/') || row.raw_artifact.includes('..')) fail(bundleRel, `${rowId} raw_artifact must be a safe relative path`)
}

function validateChecksObject(bundleRel, label, checks, requiredKeys) {
  if (!checks || typeof checks !== 'object') {
    fail(bundleRel, `${label} must be present`)
    return
  }
  for (const key of requiredKeys) {
    if (checks[key] !== true) fail(bundleRel, `${label}.${key} must be true`)
  }
}

function boundaryIsNarrow(boundary) {
  if (typeof boundary !== 'string') return false
  return /does not promote/i.test(boundary) && /full Llama-family support/i.test(boundary)
}

function compareRowField(bundleRel, rowId, manifestRow, summaryRow, field) {
  if (summaryRow[field] === undefined) return
  if (manifestRow[field] !== summaryRow[field]) {
    fail(bundleRel, `${rowId} ${field} mismatch: manifest=${JSON.stringify(manifestRow[field])} summary=${JSON.stringify(summaryRow[field])}`)
  }
}

async function findManifestPaths(root) {
  const paths = []
  await walk(root)
  return paths.sort()

  async function walk(currentDir) {
    const entries = await readdir(currentDir, { withFileTypes: true })
    for (const entry of entries) {
      const fullPath = join(currentDir, entry.name)
      if (entry.isDirectory()) {
        await walk(fullPath)
      } else if (entry.isFile() && entry.name === 'manifest.json') {
        paths.push(fullPath)
      }
    }
  }
}

async function readJson(path) {
  try {
    return JSON.parse(await readFile(path, 'utf8'))
  } catch (error) {
    fail(relative(process.cwd(), path), `failed to read JSON: ${error.message}`)
    return null
  }
}

async function exists(path) {
  try {
    await access(path)
    return true
  } catch {
    return false
  }
}

function fail(bundleRel, message) {
  failures.push(`${bundleRel}: ${message}`)
}

function parseArgs(argv) {
  const parsed = new Map()
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (!arg.startsWith('--')) continue
    const [key, inline] = arg.slice(2).split('=', 2)
    const next = argv[i + 1]
    if (inline !== undefined) {
      parsed.set(key, inline)
      continue
    }
    if (!next || next.startsWith('--')) {
      parsed.set(key, 'true')
      continue
    }
    parsed.set(key, next)
    i += 1
  }
  return parsed
}
