#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  ContractError,
  validateBenchGenerateRecord,
  validateCampaign,
  validateComparison,
  validatePlan,
  validateRuntimeSample,
} from './lib/contracts.mjs'

const systemRoot = resolve(fileURLToPath(new URL('.', import.meta.url)))

const schemaCases = [
  ['campaign-v1.schema.json', 'https://camelid.ai/schemas/benchmark/campaign-v1.schema.json'],
  ['plan-v1.schema.json', 'https://camelid.ai/schemas/benchmark/plan-v1.schema.json'],
  ['runtime-sample-v1.schema.json', 'https://camelid.ai/schemas/benchmark/runtime-sample-v1.schema.json'],
  ['comparison-v1.schema.json', 'https://camelid.ai/schemas/benchmark/comparison-v1.schema.json'],
]
for (const [name, expectedId] of schemaCases) {
  const schema = await json(`schemas/${name}`)
  assert.equal(schema.$schema, 'https://json-schema.org/draft/2020-12/schema')
  assert.equal(schema.$id, expectedId)
  assert.equal(schema.additionalProperties, false)
}

const campaign = await json('fixtures/schemas/valid-campaign.json')
const plan = await json('fixtures/schemas/valid-plan.json')
const runtimeSample = await json('fixtures/schemas/valid-runtime-sample.json')
const invalidRuntimeSample = await json('fixtures/schemas/valid-invalid-runtime-sample.json')
const comparison = await json('fixtures/schemas/valid-comparison.json')
assert.equal(validateCampaign(campaign), campaign)
assert.equal(validatePlan(plan), plan)
assert.equal(validateRuntimeSample(runtimeSample), runtimeSample)
assert.equal(validateRuntimeSample(invalidRuntimeSample), invalidRuntimeSample)
assert.equal(validateComparison(comparison), comparison)

const missingHash = structuredClone(plan)
delete missingHash.models[0].artifact_sha256
fails(() => validatePlan(missingHash), 'artifact_sha256 is required')

const duplicateArm = structuredClone(plan)
duplicateArm.source_arms[1].id = duplicateArm.source_arms[0].id
fails(() => validatePlan(duplicateArm), 'duplicates "base"')

const wrongArmName = structuredClone(campaign)
wrongArmName.source_arms[1].id = 'candidate'
wrongArmName.workloads[0].model_id = campaign.workloads[0].model_id
fails(() => validateCampaign(wrongArmName), 'exactly two arms named base and head')

const unknownSchema = structuredClone(plan)
unknownSchema.schema = 'camelid.benchmark.plan/v2'
fails(() => validatePlan(unknownSchema), 'must equal "camelid.benchmark.plan/v1"')

const invalidRuntime = structuredClone(runtimeSample)
invalidRuntime.metrics.tokens_per_second = -1
fails(() => validateRuntimeSample(invalidRuntime), 'tokens_per_second must be a non-negative finite number')

const nonFiniteRuntime = structuredClone(runtimeSample)
nonFiniteRuntime.metrics.ttft_ms = Number.NaN
fails(() => validateRuntimeSample(nonFiniteRuntime), 'ttft_ms must be a non-negative finite number')

const invalidComparison = structuredClone(comparison)
invalidComparison.runtime[0].verdict = 'PASS'
fails(() => validateComparison(invalidComparison), '$.runtime[0].verdict must be one of')

for (const fixture of ['valid-direct.jsonl', 'valid-runnable.jsonl']) {
  const records = await jsonLines(`fixtures/bench-generate/${fixture}`)
  assert.ok(records.length > 0)
  for (const record of records) validateBenchGenerateRecord(record)
}

await assert.rejects(
  jsonLines('fixtures/bench-generate/invalid-truncated.jsonl'),
  /invalid JSON on line 1/,
)
const negativeMetric = await jsonLines('fixtures/bench-generate/invalid-negative-metric.jsonl')
fails(
  () => validateBenchGenerateRecord(negativeMetric[0]),
  '$.tokens_per_second must be a non-negative finite number',
)

console.log('benchmark Phase 1 schema contracts: PASS')

async function json(relativePath) {
  return JSON.parse(await readFile(resolve(systemRoot, relativePath), 'utf8'))
}

async function jsonLines(relativePath) {
  const text = await readFile(resolve(systemRoot, relativePath), 'utf8')
  const records = []
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (line.trim().length === 0) continue
    try {
      records.push(JSON.parse(line))
    } catch (error) {
      throw new Error(`${relativePath}: invalid JSON on line ${index + 1}: ${error.message}`)
    }
  }
  return records
}

function fails(action, expected) {
  assert.throws(action, (error) => {
    assert.ok(error instanceof ContractError)
    assert.match(error.message, new RegExp(escapeRegExp(expected)))
    return true
  })
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
