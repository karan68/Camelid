#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { ContractError, validateAgentAttempt, validateAgentTask } from './lib/contracts.mjs'

const systemRoot = resolve(fileURLToPath(new URL('.', import.meta.url)))
const schema = await json('schemas/agent-task-v1.schema.json')
assert.equal(schema.$schema, 'https://json-schema.org/draft/2020-12/schema')
assert.equal(schema.$id, 'https://camelid.ai/schemas/benchmark/agent-task-v1.schema.json')
assert.equal(schema.additionalProperties, false)
const attemptSchema = await json('schemas/agent-attempt-v1.schema.json')
assert.equal(attemptSchema.$schema, 'https://json-schema.org/draft/2020-12/schema')
assert.equal(attemptSchema.$id, 'https://camelid.ai/schemas/benchmark/agent-attempt-v1.schema.json')
assert.equal(attemptSchema.additionalProperties, false)
const taskCheckSchema = await json('schemas/task-check-v1.schema.json')
assert.equal(taskCheckSchema.$schema, 'https://json-schema.org/draft/2020-12/schema')
assert.equal(taskCheckSchema.$id, 'https://camelid.ai/schemas/benchmark/task-check-v1.schema.json')
assert.equal(taskCheckSchema.additionalProperties, false)

const task = await json('fixtures/schemas/valid-agent-task.json')
assert.equal(validateAgentTask(task), task)
const attempt = await json('fixtures/schemas/valid-agent-attempt.json')
assert.equal(validateAgentAttempt(attempt), attempt)

const unknownSchema = structuredClone(task)
unknownSchema.schema = 'camelid.benchmark.agent-task/v2'
fails(() => validateAgentTask(unknownSchema), 'must equal "camelid.benchmark.agent-task/v1"')

const duplicateCapability = structuredClone(task)
duplicateCapability.required_capabilities.push('read')
fails(() => validateAgentTask(duplicateCapability), 'duplicates "read"')

const unsafePattern = structuredClone(task)
unsafePattern.allowed_mutations = ['../controller/**']
fails(() => validateAgentTask(unsafePattern), 'cannot contain a parent segment')

const windowsPattern = structuredClone(task)
windowsPattern.forbidden_mutations = ['tests\\**']
fails(() => validateAgentTask(windowsPattern), 'must use forward slashes')

const networkedTask = structuredClone(task)
networkedTask.network = 'explicit'
fails(() => validateAgentTask(networkedTask), '$.network must equal "deny"')

const unsupportedRating = structuredClone(task)
unsupportedRating.difficulty = { label: 'easy', evidence: 'Maintainer intuition only.' }
fails(() => validateAgentTask(unsupportedRating), 'must cite observed completion data')

const duplicateCheck = structuredClone(task)
duplicateCheck.required_checks.push(structuredClone(duplicateCheck.required_checks[0]))
fails(() => validateAgentTask(duplicateCheck), 'duplicates "target_behavior"')

const evalCommand = structuredClone(task)
evalCommand.setup_command = ['node', '--eval', 'process.exit(0)']
fails(() => validateAgentTask(evalCommand), 'must contain exactly two items')

const escapingCommand = structuredClone(task)
escapingCommand.required_checks[0].command = ['node', '../test.mjs']
fails(() => validateAgentTask(escapingCommand), 'cannot contain a parent segment')

const broadGlob = structuredClone(task)
broadGlob.allowed_mutations = ['src/*.cjs']
fails(() => validateAgentTask(broadGlob), 'may use wildcards only as a trailing /**')

const missingControl = structuredClone(task)
missingControl.negative_controls.pop()
fails(() => validateAgentTask(missingControl), 'must include scorer_immutable')

const inconsistentPass = structuredClone(attempt)
inconsistentPass.score.passed_checks = 1
fails(() => validateAgentAttempt(inconsistentPass), 'must equal required_checks for PASS_COMPARABLE')

const wrongComparability = structuredClone(attempt)
wrongComparability.comparability = 'noncomparable'
fails(() => validateAgentAttempt(wrongComparability), '$.comparability must equal "comparable"')

const missingUsageReason = structuredClone(attempt)
missingUsageReason.usage.unavailable_reason = null
fails(() => validateAgentAttempt(missingUsageReason), '$.usage.unavailable_reason must be a non-empty string')

const timeout = structuredClone(attempt)
timeout.comparability = 'noncomparable'
timeout.terminal = { class: 'timed_out', exit_code: null, reason: 'wall timeout expired' }
timeout.score = {
  outcome: 'INCONCLUSIVE_TIMEOUT',
  required_checks: 2,
  passed_checks: 0,
  diff_sha256: attempt.score.diff_sha256,
}
assert.equal(validateAgentAttempt(timeout), timeout)

console.log('benchmark Phase 2 agent task contracts: PASS')

async function json(relativePath) {
  return JSON.parse(await readFile(resolve(systemRoot, relativePath), 'utf8'))
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