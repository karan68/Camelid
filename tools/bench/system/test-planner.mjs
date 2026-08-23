#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { validatePlan } from './lib/contracts.mjs'
import { balancedArmOrder, PlanResolutionError, resolveCampaignPlan, serializePlan } from './planner.mjs'

const root = resolve(fileURLToPath(new URL('.', import.meta.url)))
const campaign = JSON.parse(await readFile(resolve(root, 'fixtures/schemas/valid-campaign.json'), 'utf8'))
const original = structuredClone(campaign)
const dependencies = fakeDependencies()

const plan = await resolveCampaignPlan(campaign, dependencies)
assert.equal(validatePlan(plan), plan)
assert.deepEqual(campaign, original, 'planner must not mutate its request')
assert.equal(plan.repository_root, '<resolved-repo>')
assert.equal(plan.source_arms[0].cargo_lock_sha256, 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
assert.match(plan.source_arms[0].build.binary_path, /base[\\/]release[\\/]camelid-test$/)
assert.equal(plan.models[0].size_bytes, 1321082528)
assert.equal(plan.workloads[0].prompt_sha256, 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
assert.deepEqual(plan.workloads[0].order, ['base', 'head', 'head', 'base'])
assert.equal(serializePlan(plan), serializePlan(structuredClone(plan)))

const reordered = Object.fromEntries(Object.entries(campaign).reverse())
const reorderedPlan = await resolveCampaignPlan(reordered, fakeDependencies())
assert.equal(serializePlan(plan), serializePlan(reorderedPlan))

const wrongSource = structuredClone(campaign)
wrongSource.source_arms[0].expected_git_sha = '9999999999999999999999999999999999999999'
await rejects(
  () => resolveCampaignPlan(wrongSource, fakeDependencies()),
  'invalid_hash',
  /source arm base resolved to/,
)

const wrongModel = structuredClone(campaign)
wrongModel.models[0].expected_sha256 = '9999999999999999999999999999999999999999999999999999999999999999'
await rejects(
  () => resolveCampaignPlan(wrongModel, fakeDependencies()),
  'invalid_hash',
  /model synthetic-q8-row resolved to/,
)

const dirtyDependencies = fakeDependencies()
dirtyDependencies.inspectSource = async (arm) => ({
  ...(await fakeDependencies().inspectSource(arm)),
  treeDirty: arm.id === 'head',
})
await rejects(
  () => resolveCampaignPlan(campaign, dirtyDependencies),
  'invalid_environment',
  /source arm head is dirty/,
)

assert.deepEqual(balancedArmOrder(['a', 'b', 'c'], 3), [
  'a', 'b', 'c',
  'b', 'c', 'a',
  'c', 'a', 'b',
])
assert.throws(() => balancedArmOrder([], 1), /at least one arm/)
assert.throws(() => balancedArmOrder(['a', 'a'], 1), /must be unique/)
assert.throws(() => balancedArmOrder(['a'], 0), /positive safe integer/)

console.log('benchmark Phase 1 deterministic planner: PASS')

function fakeDependencies() {
  return {
    binaryName: 'camelid-test',
    resolveDirectory: async () => '<resolved-repo>',
    inspectSource: async (arm) => ({
      sourceDir: `<resolved-${arm.id}>`,
      gitSha: arm.id === 'base'
        ? '1111111111111111111111111111111111111111'
        : '2222222222222222222222222222222222222222',
      treeDirty: false,
      cargoLockSha256: arm.id === 'base'
        ? 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
        : 'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
    }),
    inspectArtifact: async () => ({
      path: '<resolved-model>',
      sha256: 'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd',
      sizeBytes: 1321082528,
    }),
    inspectPrompt: async () => ({
      path: '<resolved-prompt>',
      sha256: 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
    }),
  }
}

async function rejects(action, code, message) {
  try {
    await action()
    assert.fail('expected action to reject')
  } catch (error) {
    assert.ok(error instanceof PlanResolutionError)
    assert.equal(error.code, code)
    assert.match(error.message, message)
  }
}
