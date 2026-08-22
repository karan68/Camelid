#!/usr/bin/env node
import assert from 'node:assert/strict'
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { writeBenchmarkBundle, verifyBundleChecksums } from './bundle.mjs'

const root = resolve(fileURLToPath(new URL('.', import.meta.url)))
const plan = JSON.parse(await readFile(resolve(root, 'fixtures/schemas/valid-plan.json'), 'utf8'))
const sampleFixture = JSON.parse(await readFile(resolve(root, 'fixtures/schemas/valid-runtime-sample.json'), 'utf8'))
const temp = await mkdtemp(join(tmpdir(), 'camelid-benchmark-bundle-'))

try {
  const samples = []
  for (let block = 0; block < 2; block += 1) {
    samples.push(sample('base', block, 10 + block))
    samples.push(sample('head', block, 11 + block * 1.1))
  }
  const preparedArms = plan.source_arms.map((arm, index) => ({
    arm_id: arm.id,
    source_sha: arm.git_sha,
    binary_path: arm.build.binary_path,
    binary_sha256: String(index + 1).repeat(64),
    binary_size_bytes: 42,
    reported_version: 'camelid 0.6.1',
  }))
  const result = await writeBenchmarkBundle({
    plan,
    preparedArms,
    samples,
    executions: [],
    outputDir: temp,
    preparationMode: 'built_from_plan',
  }, {
    generatedUtc: '2026-08-23T00:00:00Z',
    stats: { seed: 7, bootstrapSamples: 1000 },
  })

  assert.equal(result.manifest.state, 'COMPLETE_VALID')
  assert.equal(result.comparison.runtime[0].observed_direction, 'head_faster')
  assert.equal(result.comparison.runtime[0].verdict, 'INCONCLUSIVE_NOISE')
  assert.equal(result.comparison.runtime[0].valid_pairs, 2)
  for (const file of ['plan.json', 'prepared-arms.json', 'executions.json', 'comparison.json', 'manifest.json', 'summary.md', 'SHA256SUMS']) {
    await access(join(temp, file))
  }
  const verification = await verifyBundleChecksums(temp)
  assert.deepEqual(verification, { ok: true, failures: [] })
  const summary = await readFile(join(temp, 'summary.md'), 'utf8')
  assert.match(summary, /Local informational Phase 1/)
  assert.match(summary, /head_faster/)

  await writeFile(join(temp, 'summary.md'), `${summary}\ntampered\n`, 'utf8')
  const tampered = await verifyBundleChecksums(temp)
  assert.equal(tampered.ok, false)
  assert.match(tampered.failures.join('\n'), /summary.md/)
} finally {
  await rm(temp, { recursive: true, force: true })
}

console.log('benchmark Phase 1 aggregate bundle and checksums: PASS')

function sample(armId, block, throughput) {
  const value = structuredClone(sampleFixture)
  value.arm_id = armId
  value.process_block = block
  value.metrics.tokens_per_second = throughput
  value.correctness.output_token_ids_sha256 = String(block + 1).repeat(64)
  return value
}
