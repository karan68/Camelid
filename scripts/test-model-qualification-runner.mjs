#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  compareIds,
  declaredQuant,
  deriveOverall,
  gitDirtyState,
  headlineQuant,
  promoteLoadFromFullParity,
  redactLocalPaths,
  validateOracleFixture,
} from './model-qualification-runner.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const roster = JSON.parse(await readFile(join(root, 'qa', 'model-qualification', 'phase1-roster.json'), 'utf8'))
const qwenRow = roster.rows.find((row) => row.id === 'qwen2_5_0_5b_instruct_q8_0')
const qwenFixture = JSON.parse(await readFile(resolve(root, qwenRow.probes.oracle_fixture), 'utf8'))

assert.deepEqual(
  compareIds([1, 2, 3], [1, 2, 3]),
  { match: true, first_divergence: null, actual_length: 3, expected_length: 3 },
)
assert.equal(compareIds([1, 9, 3], [1, 2, 3]).first_divergence, 1)
assert.equal(compareIds([1, 2], [1, 2, 3]).first_divergence, 2)

assert.equal(
  headlineQuant([
    { tensor_type: 'F32' },
    { tensor_type: 'Q8_0' },
    { tensor_type: 'Q8_0' },
    { tensor_type: 'Q4_K' },
  ]),
  'Q8_0',
)
assert.equal(headlineQuant([{ tensor_type: 'F16' }]), null)
assert.equal(declaredQuant(7), 'Q8_0')
assert.equal(declaredQuant(15), 'Q4_K_M')
assert.equal(declaredQuant(999), null)

assert.equal(deriveOverall({ a: { status: 'pass' }, b: { status: 'pass' } }), 'pass')
assert.equal(deriveOverall({ a: { status: 'pass' }, b: { status: 'blocked' } }), 'blocked')
assert.equal(deriveOverall({ a: { status: 'fail' }, b: { status: 'blocked' } }), 'fail')
assert.equal(deriveOverall({ a: { status: 'skipped' } }), 'blocked')
assert.equal(deriveOverall({ a: { status: 'skipped', required: false }, b: { status: 'pass' } }), 'pass')
assert.equal(gitDirtyState(''), false)
assert.equal(gitDirtyState(' M src/model.rs'), true)
assert.equal(gitDirtyState(null), null, 'a failed git inspection must never be reported as clean')

const paritySupersedesSmoke = {
  load_smoke: { status: 'blocked', reason: 'combo not yet anchored' },
  parity: { status: 'pass', probes: [{ match: true }] },
}
promoteLoadFromFullParity(paritySupersedesSmoke)
assert.equal(paritySupersedesSmoke.load_smoke.status, 'pass')
assert.equal(paritySupersedesSmoke.load_smoke.qualified_by, 'full_raw_greedy_parity_replay')
assert.equal(paritySupersedesSmoke.load_smoke.superseded_standalone_smoke.status, 'blocked')
const partialParityDoesNotPromote = {
  load_smoke: { status: 'blocked' },
  parity: { status: 'blocked', reason: 'partial probe' },
}
promoteLoadFromFullParity(partialParityDoesNotPromote)
assert.equal(partialParityDoesNotPromote.load_smoke.status, 'blocked')

assert.deepEqual(
  validateOracleFixture(qwenFixture, qwenRow, roster.defaults),
  [],
  'the pinned Qwen fixture must identify the exact selected row and oracle',
)
const wrongFixture = structuredClone(qwenFixture)
wrongFixture.row_id = 'different_exact_row'
wrongFixture.artifact.sha256 = '0'.repeat(64)
wrongFixture.oracle.build = 'different-build'
const fixtureErrors = validateOracleFixture(wrongFixture, qwenRow, roster.defaults)
assert.ok(fixtureErrors.some((error) => error.startsWith('row_id ')), 'a fixture for another roster row must be rejected')
assert.ok(fixtureErrors.some((error) => error.startsWith('artifact.sha256 ')), 'a fixture for another artifact must be rejected')
assert.ok(fixtureErrors.some((error) => error.startsWith('oracle.build ')), 'a fixture from another oracle build must be rejected')
const tokenizerOnlyFixture = structuredClone(qwenFixture)
for (const prompt of tokenizerOnlyFixture.raw_prompts) {
  delete prompt.generated_ids
  delete prompt.generated_text
}
assert.deepEqual(
  validateOracleFixture(tokenizerOnlyFixture, qwenRow, roster.defaults),
  [],
  'a tokenizer-only exact-row fixture may defer generation evidence without weakening token parity',
)
const emptyGenerationFixture = structuredClone(tokenizerOnlyFixture)
emptyGenerationFixture.raw_prompts[0].generated_ids = []
assert.ok(
  validateOracleFixture(emptyGenerationFixture, qwenRow, roster.defaults)
    .some((error) => error.includes('generated_ids must be a non-empty')),
  'an empty generated-id array must never masquerade as generation parity',
)

assert.deepEqual(
  redactLocalPaths(
    { command: ['C:\\private\\repo\\camelid.exe', 'C:\\private\\models\\row.gguf'] },
    [
      ['C:\\private\\repo', '<repo>'],
      ['C:\\private\\models\\row.gguf', '<artifact>'],
    ],
  ),
  { command: ['<repo>\\camelid.exe', '<artifact>'] },
  'generated reports must scrub local repository and model paths',
)

console.log('test-model-qualification-runner: all checks passed')
