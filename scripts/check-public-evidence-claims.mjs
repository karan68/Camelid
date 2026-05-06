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

  if (schema === 'camelid.llama3_8b_context_1024_2048_current_head_public_evidence.v1') {
    if (!summaryExists) fail(bundleRel, 'Llama 3 8B context-1024/2048 bundle must include summary.json')
    validateLlama3_8bContext1024And2048(bundleRel, manifest)
  }

  const singleRowContext = singleRowContextSchema(schema)
  if (singleRowContext) validateSingleRowContextBundle(bundleRel, manifest, singleRowContext)

  const legacyPublicContext = legacyPublicContextSchema(manifest)
  if (legacyPublicContext) validateLegacyPublicContextBundle(bundleRel, manifest, legacyPublicContext)
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
  validateContextRow(bundleRel, row, {
    contextWindow: 512,
    maxTokens: 5,
    minPromptTokens: 1,
  })
}

function validateLlama3_8bContext1024And2048(bundleRel, manifest) {
  if (manifest.passed !== true) fail(bundleRel, 'Llama 3 8B context-1024/2048 manifest must be passed=true')
  if (manifest.checkout_clean !== true) fail(bundleRel, 'Llama 3 8B context-1024/2048 manifest must record checkout_clean=true')
  if (!boundaryIsNarrow(manifest.claim_boundary)) {
    fail(bundleRel, 'Llama 3 8B context-1024/2048 claim_boundary must explicitly avoid broader/full-family promotion')
  }
  const expectedPackIds = ['llama3-context-1024-smoke-v1', 'llama3-context-2048-smoke-v1']
  const expectedSourcePacks = ['qa/prompt-packs/llama3-context-1024-smoke.json', 'qa/prompt-packs/llama3-context-2048-smoke.json']
  if (manifest.pack !== undefined) {
    if (manifest.pack?.max_tokens !== 5) fail(bundleRel, 'Llama 3 8B context-1024/2048 pack max_tokens must be 5')
    if (JSON.stringify(manifest.pack?.ids) !== JSON.stringify(expectedPackIds)) {
      fail(bundleRel, `Llama 3 8B context-1024/2048 pack ids changed: ${JSON.stringify(manifest.pack?.ids)}`)
    }
    if (JSON.stringify(manifest.pack?.source_prompt_packs) !== JSON.stringify(expectedSourcePacks)) {
      fail(bundleRel, `Llama 3 8B context-1024/2048 source_prompt_packs changed: ${JSON.stringify(manifest.pack?.source_prompt_packs)}`)
    }
  }
  if (!Array.isArray(manifest.rows) || manifest.rows.length !== 2) {
    fail(bundleRel, 'Llama 3 8B context-1024/2048 manifest must include exactly two rows')
    return
  }
  validateContextRow(bundleRel, manifest.rows[0], {
    rowId: 'llama3_8b_instruct_q8_0',
    contextWindow: 1024,
    maxTokens: 5,
    minPromptTokens: 513,
    promptId: 'roughly-1024-token-recall',
  })
  validateContextText(bundleRel, 'llama3_8b_instruct_q8_0 context-1024', manifest.rows[0], 'CMLD-102')
  validateContextRow(bundleRel, manifest.rows[1], {
    rowId: 'llama3_8b_instruct_q8_0',
    contextWindow: 2048,
    maxTokens: 5,
    minPromptTokens: 1025,
    promptId: 'roughly-2048-token-recall',
  })
  validateContextText(bundleRel, 'llama3_8b_instruct_q8_0 context-2048', manifest.rows[1], 'CMLD-204')
}

function validateContextText(bundleRel, label, row, expected) {
  const generatedText = row.generated_text ?? row.backend_text
  const llamaText = row.llama_text ?? generatedText
  if (generatedText !== expected) fail(bundleRel, `${label} generated text must stay ${JSON.stringify(expected)}, got ${JSON.stringify(generatedText)}`)
  if (llamaText !== expected) fail(bundleRel, `${label} llama text must stay ${JSON.stringify(expected)}, got ${JSON.stringify(llamaText)}`)
}

function validateSingleRowContextBundle(bundleRel, manifest, expected) {
  if (manifest.passed !== true) fail(bundleRel, `${expected.rowId} context-${expected.contextWindow} manifest must be passed=true`)
  if (manifest.checkout_clean !== true) fail(bundleRel, `${expected.rowId} context-${expected.contextWindow} manifest must record checkout_clean=true`)
  if (manifest.pack?.target_context_window !== expected.contextWindow) {
    fail(bundleRel, `${expected.rowId} pack target_context_window must be ${expected.contextWindow}`)
  }
  if (manifest.pack?.max_tokens !== expected.maxTokens) fail(bundleRel, `${expected.rowId} pack max_tokens must be ${expected.maxTokens}`)
  if (manifest.pack?.source_prompt_pack !== expected.sourcePromptPack) {
    fail(bundleRel, `${expected.rowId} source_prompt_pack must be ${expected.sourcePromptPack}`)
  }
  if (manifest.pack?.prompt_count !== 1) fail(bundleRel, `${expected.rowId} context-${expected.contextWindow} pack prompt_count must be 1`)
  if (!boundaryIsNarrow(manifest.claim_boundary)) {
    fail(bundleRel, `${expected.rowId} context-${expected.contextWindow} claim_boundary must explicitly avoid broader/full-family promotion`)
  }
  if (!Array.isArray(manifest.rows) || manifest.rows.length !== 1) {
    fail(bundleRel, `${expected.rowId} context-${expected.contextWindow} manifest must include exactly one row`)
    return
  }
  validateContextRow(bundleRel, manifest.rows[0], expected)
}

function validateContextRow(bundleRel, row, expected) {
  const rowId = row?.row_id || '<missing row_id>'
  if (expected.rowId && rowId !== expected.rowId) fail(bundleRel, `row_id must be ${expected.rowId}, got ${rowId}`)
  if (row.context_window !== expected.contextWindow) fail(bundleRel, `${rowId} context_window must be ${expected.contextWindow}`)
  if (row.max_tokens !== expected.maxTokens) fail(bundleRel, `${rowId} max_tokens must be ${expected.maxTokens}`)
  if (expected.promptId && row.prompt_id !== expected.promptId) fail(bundleRel, `${rowId} prompt_id must be ${expected.promptId}`)
  if (expected.generatedText && row.generated_text !== expected.generatedText) {
    fail(bundleRel, `${rowId} generated_text must stay ${JSON.stringify(expected.generatedText)}`)
  }
  if (row.passed !== undefined && row.passed !== true) fail(bundleRel, `${rowId} passed must be true`)
  if (row.prompt_tokens_match !== true) fail(bundleRel, `${rowId} prompt_tokens_match must be true`)
  if (row.generated_tokens_match !== true) fail(bundleRel, `${rowId} generated_tokens_match must be true`)
  if (row.generated_text_match !== true) fail(bundleRel, `${rowId} generated_text_match must be true`)
  if (row.first_generated_token_diff_index !== -1) fail(bundleRel, `${rowId} first_generated_token_diff_index must be -1`)
  if (!Number.isInteger(row.reference_prompt_token_count) || row.reference_prompt_token_count < expected.minPromptTokens) {
    fail(bundleRel, `${rowId} reference_prompt_token_count must be at least ${expected.minPromptTokens}`)
  }
  if (!Number.isInteger(row.max_resident_set_kib) || row.max_resident_set_kib <= 0) fail(bundleRel, `${rowId} max_resident_set_kib must be positive`)
  if (typeof row.model_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(row.model_sha256)) fail(bundleRel, `${rowId} model_sha256 must be a 64-character lowercase sha256`)
  if (typeof row.raw_artifact !== 'string' || row.raw_artifact.startsWith('/') || row.raw_artifact.includes('..')) fail(bundleRel, `${rowId} raw_artifact must be a safe relative path`)
}

function singleRowContextSchema(schema) {
  const schemas = {
    'camelid.llama32_1b_context_1024_public_evidence.v1': {
      rowId: 'llama32_1b_instruct_q8_0',
      contextWindow: 1024,
      maxTokens: 5,
      minPromptTokens: 513,
      sourcePromptPack: 'qa/prompt-packs/llama3-context-1024-smoke.json',
      promptId: 'roughly-1024-token-recall',
      generatedText: 'CMLD-102',
    },
    'camelid.llama32_3b_context_1024_public_evidence.v1': {
      rowId: 'llama32_3b_instruct_q8_0',
      contextWindow: 1024,
      maxTokens: 5,
      minPromptTokens: 513,
      sourcePromptPack: 'qa/prompt-packs/llama3-context-1024-smoke.json',
      promptId: 'roughly-1024-token-recall',
      generatedText: 'CMLD-102',
    },
    'camelid.llama32_3b_context_2048_public_evidence.v1': {
      rowId: 'llama32_3b_instruct_q8_0',
      contextWindow: 2048,
      maxTokens: 5,
      minPromptTokens: 1025,
      sourcePromptPack: 'qa/prompt-packs/llama3-context-2048-smoke.json',
      promptId: 'roughly-2048-token-recall',
      generatedText: 'CMLD-204',
    },
    'camelid.llama3_8b_context_1024_public_evidence.v1': {
      rowId: 'llama3_8b_instruct_q8_0',
      contextWindow: 1024,
      maxTokens: 5,
      minPromptTokens: 513,
      sourcePromptPack: 'qa/prompt-packs/llama3-context-1024-smoke.json',
      promptId: 'roughly-1024-token-recall',
      generatedText: 'CMLD-102',
    },
    'camelid.llama3_8b_context_2048_public_evidence.v1': {
      rowId: 'llama3_8b_instruct_q8_0',
      contextWindow: 2048,
      maxTokens: 5,
      minPromptTokens: 1025,
      sourcePromptPack: 'qa/prompt-packs/llama3-context-2048-smoke.json',
      promptId: 'roughly-2048-token-recall',
      generatedText: 'CMLD-204',
    },
  }
  return schemas[schema]
}

function legacyPublicContextSchema(manifest) {
  if (manifest.schema !== 'camelid.public-evidence-bundle.v1') return undefined
  if (
    manifest.model_row === 'llama32_1b_instruct_q8_0' &&
    manifest.pack_id === 'llama3-context-2048-smoke-v1' &&
    manifest.target_context_window === 2048
  ) {
    return {
      rowId: 'llama32_1b_instruct_q8_0',
      contextWindow: 2048,
      maxTokens: 5,
      minPromptTokens: 1025,
      generatedText: 'CMLD-204',
      generatedTokens: [34, 2735, 35, 12, 7854],
    }
  }
  return undefined
}

function validateLegacyPublicContextBundle(bundleRel, manifest, expected) {
  if (manifest.model_row !== expected.rowId) fail(bundleRel, `model_row must be ${expected.rowId}, got ${manifest.model_row}`)
  if (manifest.target_context_window !== expected.contextWindow) fail(bundleRel, `${expected.rowId} target_context_window must be ${expected.contextWindow}`)
  if (manifest.max_tokens !== expected.maxTokens) fail(bundleRel, `${expected.rowId} max_tokens must be ${expected.maxTokens}`)
  if (!Number.isInteger(manifest.reference_prompt_token_count) || manifest.reference_prompt_token_count < expected.minPromptTokens) {
    fail(bundleRel, `${expected.rowId} reference_prompt_token_count must be at least ${expected.minPromptTokens}`)
  }
  if (typeof manifest.boundary !== 'string' || !boundaryIsNarrow(manifest.boundary)) {
    fail(bundleRel, `${expected.rowId} context-${expected.contextWindow} boundary must explicitly avoid broader/full-family promotion`)
  }
  const result = manifest.result || {}
  if (result.passed !== true) fail(bundleRel, `${expected.rowId} context-${expected.contextWindow} result.passed must be true`)
  if (result.prompt_tokens_all_match !== true) fail(bundleRel, `${expected.rowId} prompt_tokens_all_match must be true`)
  if (result.generated_tokens_all_match !== true) fail(bundleRel, `${expected.rowId} generated_tokens_all_match must be true`)
  if (result.generated_text_all_match !== true) fail(bundleRel, `${expected.rowId} generated_text_all_match must be true`)
  compareExactJson(bundleRel, `${expected.rowId} backend_generated_tokens`, result.backend_generated_tokens, expected.generatedTokens)
  compareExactJson(bundleRel, `${expected.rowId} reference_generated_tokens`, result.reference_generated_tokens, expected.generatedTokens)
  if (result.backend_text !== expected.generatedText) fail(bundleRel, `${expected.rowId} backend_text must stay ${JSON.stringify(expected.generatedText)}`)
  if (result.reference_text !== expected.generatedText) fail(bundleRel, `${expected.rowId} reference_text must stay ${JSON.stringify(expected.generatedText)}`)
  if (!Array.isArray(manifest.primary_artifacts) || manifest.primary_artifacts.length === 0) {
    fail(bundleRel, `${expected.rowId} primary_artifacts must list the raw pack artifacts`)
  } else {
    for (const artifact of manifest.primary_artifacts) {
      if (typeof artifact !== 'string' || artifact.startsWith('/') || artifact.includes('..')) {
        fail(bundleRel, `${expected.rowId} primary_artifacts must be safe relative paths`)
      }
    }
  }
}

function compareExactJson(bundleRel, label, actual, expected) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    fail(bundleRel, `${label} must stay ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`)
  }
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
