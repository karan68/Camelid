import { mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { join, relative, resolve, sep } from 'node:path'

import { buildComparison } from './aggregate.mjs'
import { validatePlan, validateRuntimeSample } from './lib/contracts.mjs'
import { canonicalJson, sha256File } from './lib/digest.mjs'

export async function writeBenchmarkBundle(input, options = {}) {
  const { plan, preparedArms, samples, executions } = input
  const preparationModes = new Set(['built_from_plan', 'supplied_local_ablation'])
  if (!preparationModes.has(input.preparationMode)) {
    throw new TypeError(`preparationMode must be one of ${[...preparationModes].join(', ')}`)
  }
  validatePlan(plan)
  samples.forEach(validateRuntimeSample)
  const outputDir = resolve(input.outputDir)
  const generatedUtc = options.generatedUtc ?? new Date().toISOString()
  const comparison = buildComparison(plan, samples, options.stats)
  await mkdir(outputDir, { recursive: true })
  await writeCanonical(join(outputDir, 'plan.json'), plan)
  await writeCanonical(join(outputDir, 'prepared-arms.json'), preparedArms)

  const sampleFiles = []
  for (const sample of samples) {
    const path = join(
      outputDir,
      'runtime',
      sample.workload_id,
      sample.arm_id,
      `block-${String(sample.process_block).padStart(3, '0')}.json`,
    )
    await writeCanonical(path, sample)
    sampleFiles.push(relativePath(outputDir, path))
  }
  await writeCanonical(join(outputDir, 'comparison.json'), comparison)

  const invalidSamples = samples.filter((sample) => sample.validity !== 'valid')
  const manifest = {
    schema: 'camelid.benchmark.bundle/v1',
    campaign_id: plan.campaign_id,
    generated_utc: generatedUtc,
    state: invalidSamples.length === 0 ? 'COMPLETE_VALID' : 'COMPLETE_WITH_FINDINGS',
    plan_sha256: await sha256File(join(outputDir, 'plan.json')),
    prepared_arm_count: preparedArms.length,
    sample_count: samples.length,
    valid_sample_count: samples.length - invalidSamples.length,
    invalid_sample_count: invalidSamples.length,
    sample_files: sampleFiles.sort(),
    raw_execution_count: executions.length,
    preparation_mode: input.preparationMode,
    comparison_file: 'comparison.json',
    claim_boundary: 'Local informational Phase 1 runtime comparison only; no numeric gate or public performance claim.',
  }
  await writeCanonical(join(outputDir, 'manifest.json'), manifest)
  await writeFile(join(outputDir, 'summary.md'), renderSummary(manifest, comparison), 'utf8')
  await writeChecksums(outputDir)
  return { manifest, comparison, outputDir }
}

export async function verifyBundleChecksums(outputDir) {
  const root = resolve(outputDir)
  const text = await readFile(join(root, 'SHA256SUMS'), 'utf8')
  const failures = []
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (line.length === 0) continue
    const match = line.match(/^([0-9a-f]{64})  (.+)$/)
    if (!match) {
      failures.push(`line ${index + 1} is malformed`)
      continue
    }
    const [, expected, relativeFile] = match
    let actual
    try {
      actual = await sha256File(join(root, ...relativeFile.split('/')))
    } catch (error) {
      failures.push(`${relativeFile}: ${error.message}`)
      continue
    }
    if (actual !== expected) failures.push(`${relativeFile}: expected ${expected}, got ${actual}`)
  }
  return { ok: failures.length === 0, failures }
}

async function writeChecksums(outputDir) {
  const sumsPath = join(outputDir, 'SHA256SUMS')
  await rm(sumsPath, { force: true })
  const files = await walk(outputDir)
  const lines = []
  for (const path of files) {
    const relativeFile = relativePath(outputDir, path)
    lines.push(`${await sha256File(path)}  ${relativeFile}`)
  }
  await writeFile(sumsPath, `${lines.join('\n')}\n`, 'utf8')
}

async function walk(directory) {
  const files = []
  const entries = await readdir(directory, { withFileTypes: true })
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) files.push(...await walk(path))
    else if (entry.isFile()) files.push(path)
  }
  return files
}

async function writeCanonical(path, value) {
  await mkdir(resolve(path, '..'), { recursive: true })
  await writeFile(path, canonicalJson(value), 'utf8')
}

function renderSummary(manifest, comparison) {
  const lines = [
    '# Camelid Phase 1 Runtime Benchmark',
    '',
    `- Campaign: \`${manifest.campaign_id}\``,
    `- State: **${manifest.state}**`,
    `- Samples: ${manifest.valid_sample_count} valid / ${manifest.invalid_sample_count} invalid`,
    `- Claim boundary: ${manifest.claim_boundary}`,
    '',
    '| Workload | Metric | Valid pairs | Excluded | Base median | Head median | Head/base | 95% bootstrap CI | Direction | Verdict |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |',
  ]
  for (const row of comparison.runtime) {
    lines.push(`| ${cell(row.workload_id)} | ${cell(row.metric)} | ${row.valid_pairs} | ${row.excluded_pairs.length} | ${number(row.base_median)} | ${number(row.head_median)} | ${number(row.median_ratio_head_over_base)} | ${row.bootstrap_ci95 ? row.bootstrap_ci95.map(number).join(' - ') : '-'} | ${row.observed_direction} | ${row.verdict} |`)
  }
  lines.push('', 'Invalid and excluded samples remain in the machine-readable bundle.', '')
  return lines.join('\n')
}

function cell(value) {
  return String(value).replaceAll('|', '\\|')
}

function number(value) {
  return value === null ? '-' : Number(value).toFixed(6)
}

function relativePath(root, path) {
  return relative(root, path).split(sep).join('/')
}
