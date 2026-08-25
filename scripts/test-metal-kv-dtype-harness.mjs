#!/usr/bin/env node
// Self-test for the Metal KV campaign harness and validator. The validation-scripts
// CI job runs every scripts/test-*.mjs, so fail-open evidence regressions are gated.

import assert from 'node:assert/strict'
import { chmod, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawnSync } from 'node:child_process'

const stats = 'docs/perf-deep-dive/scripts/metal-kv-dtype-stats.py'
const harness = 'docs/perf-deep-dive/scripts/metal-kv-dtype-ab.sh'
const schema = 'camelid.metal-kv-dtype-ab/v2'
const orderDesign = 'paired-reverse-williams-v1'
const armSets = {
  full: {
    arms: ['f32', 'f16', 'q8', 'f32-nosplitk', 'q8-nosplitk'],
    configs: {
      f32: { kv_dtype: 'f32', splitk: '1', q8_attn_mm: '1' },
      f16: { kv_dtype: 'f16', splitk: '1', q8_attn_mm: '1' },
      q8: { kv_dtype: 'q8', splitk: '1', q8_attn_mm: '1' },
      'f32-nosplitk': { kv_dtype: 'f32', splitk: '0', q8_attn_mm: '1' },
      'q8-nosplitk': { kv_dtype: 'q8', splitk: '0', q8_attn_mm: '1' },
    },
  },
  prefill: {
    arms: ['q8', 'q8-noattnmm'],
    configs: {
      q8: { kv_dtype: 'q8', splitk: '1', q8_attn_mm: '1' },
      'q8-noattnmm': { kv_dtype: 'q8', splitk: '1', q8_attn_mm: '0' },
    },
  },
}

const temp = await mkdtemp(join(tmpdir(), 'camelid-metal-kv-harness-'))
let fixtureSequence = 0

function benchmarkRecord(overrides = {}) {
  return {
    runtime: 'camelid',
    commit: '0123456789abcdef',
    model: 'stub-model.gguf',
    quantization: 'Q8_0',
    iteration: 0,
    prompt_tokens: 6,
    generated_tokens: 4,
    load_ms: 10,
    prefill_ms: 20,
    ttft_ms: 22,
    decode_ms: 30,
    tokens_per_second: 100,
    peak_memory_bytes: 1024,
    output_text: 'stub',
    output_token_ids: [10, 11, 12, 13],
    ...overrides,
  }
}

function campaignRows(armSet = 'full', rounds = 10) {
  const spec = armSets[armSet]
  const rows = [{
    type: 'campaign',
    schema,
    arm_set: armSet,
    arms: spec.arms,
    arm_configs: spec.configs,
    rounds,
    order_design: orderDesign,
  }]
  for (let round = 1; round <= rounds; round += 1) {
    for (const arm of spec.arms) {
      rows.push({
        type: 'run',
        arm,
        round,
        rc: 0,
        ...spec.configs[arm],
        wall_seconds: 1,
        stderr_tail: '',
        record: benchmarkRecord(),
      })
    }
  }
  return rows
}

async function writeRows(rows, stem = 'fixture') {
  fixtureSequence += 1
  const path = join(temp, `${stem}-${fixtureSequence}.jsonl`)
  await writeFile(path, `${rows.map(row => JSON.stringify(row)).join('\n')}\n`)
  return path
}

function invokeStats(path, args = ['--validate-only']) {
  return spawnSync('python3', [stats, ...args, path], {
    cwd: process.cwd(),
    encoding: 'utf8',
  })
}

async function expectInvalid(name, rows, pattern) {
  const path = await writeRows(rows, name)
  const result = invokeStats(path)
  assert.notEqual(result.status, 0, `${name} unexpectedly validated`)
  assert.match(result.stderr, pattern)
  assert.doesNotMatch(result.stdout, /CI EXCLUDES 1|equivalent within|greedy output parity/)
}

function cloneRows(rows) {
  return structuredClone(rows)
}

async function makeStub(name, failureBody = '', jsonCopies = 1) {
  const path = join(temp, `${name}.sh`)
  const json = JSON.stringify(benchmarkRecord())
  const emissions = Array.from({ length: jsonCopies }, () => `printf '%s\\n' '${json}'`).join('\n')
  await writeFile(path, `#!/bin/sh
printf '%s/%s/%s\\n' "$CAMELID_METAL_KV_DTYPE" "$CAMELID_METAL_ATTN_SPLITK" "$CAMELID_METAL_Q8_ATTN_MM" >> "$2"
${failureBody}
${emissions}
`)
  await chmod(path, 0o755)
  return path
}

async function invokeHarness(name, stub, armSet = 'full', rounds = '10') {
  const log = join(temp, `${name}.log`)
  const output = join(temp, `${name}.jsonl`)
  const result = spawnSync('bash', [harness, stub, log, 'short', '4', rounds, output, armSet], {
    cwd: process.cwd(),
    encoding: 'utf8',
    timeout: 30_000,
  })
  return { result, log, output }
}

try {
  const validPath = await writeRows(campaignRows(), 'valid')
  const valid = invokeStats(validPath, [])
  assert.equal(valid.status, 2, 'a report label is mandatory outside --validate-only')
  const validReport = spawnSync('python3', [stats, validPath, 'valid fixture'], {
    cwd: process.cwd(),
    encoding: 'utf8',
  })
  assert.equal(validReport.status, 0, validReport.stderr)
  assert.match(validReport.stdout, /validated runs: 50/)
  assert.match(validReport.stdout, /equivalent within \+\/-5%/)
  assert.match(validReport.stdout, /greedy output parity vs f32/)

  const allFailed = campaignRows()
  for (const row of allFailed.slice(1)) {
    row.rc = 7
    delete row.record
  }
  await expectInvalid('all-failed', allFailed, /child rc must be integer 0/)

  const underpowered = campaignRows()
  underpowered[0].rounds = 4
  await expectInvalid('underpowered-current-campaign', underpowered, /multiple of 10 and at least 10/)

  const oneControlFailed = campaignRows()
  const failedControl = oneControlFailed.find(row => row.arm === 'q8-nosplitk' && row.round === 2)
  failedControl.rc = 9
  delete failedControl.record
  await expectInvalid('one-control-failed', oneControlFailed, /arm=q8-nosplitk round=2 child rc/)

  const missing = campaignRows().filter(row => !(row.arm === 'q8-nosplitk' && row.round === 3))
  await expectInvalid('missing-cell', missing, /arm=q8-nosplitk round=3 requires exactly one successful record; found 0/)

  const duplicate = campaignRows()
  duplicate.push(cloneRows(duplicate.find(row => row.arm === 'f16' && row.round === 1)))
  await expectInvalid('duplicate-cell', duplicate, /arm=f16 round=1 requires exactly one successful record; found 2/)

  const tokenMismatch = campaignRows()
  tokenMismatch.find(row => row.arm === 'q8' && row.round === 4).record.prompt_tokens = 7
  await expectInvalid('token-mismatch', tokenMismatch, /record.prompt_tokens must match across every arm\/round/)

  const tokenIdMismatch = campaignRows()
  tokenIdMismatch.find(row => row.arm === 'q8' && row.round === 1).record.output_token_ids.pop()
  await expectInvalid('token-id-count-mismatch', tokenIdMismatch, /generated_tokens=4 but output_token_ids has 3 entries/)

  const missingIdentity = campaignRows()
  delete missingIdentity[1].record.commit
  await expectInvalid('missing-identity', missingIdentity, /record.commit must be a nonempty string/)

  const missingMetric = campaignRows()
  delete missingMetric[1].record.prefill_ms
  await expectInvalid('missing-metric', missingMetric, /record.prefill_ms must be a finite positive number/)

  const nonpositiveMetric = campaignRows()
  nonpositiveMetric[1].record.tokens_per_second = 0
  await expectInvalid('nonpositive-metric', nonpositiveMetric, /record.tokens_per_second must be a finite positive number/)

  const legacy = campaignRows()
    .filter(row => row.type !== 'campaign' && row.arm !== 'q8-nosplitk')
    .map(row => {
      const copy = cloneRows(row)
      delete copy.type
      delete copy.q8_attn_mm
      return copy
    })
  const legacyPath = await writeRows(legacy, 'legacy-four-arm')
  const ambiguousLegacy = invokeStats(legacyPath)
  assert.notEqual(ambiguousLegacy.status, 0)
  assert.match(ambiguousLegacy.stderr, /headerless four-arm input is ambiguous/)
  const explicitLegacy = invokeStats(legacyPath, ['--validate-only', '--legacy-four-arm'])
  assert.equal(explicitLegacy.status, 0, explicitLegacy.stderr)
  assert.match(explicitLegacy.stdout, /legacy-four-arm/)

  const validStub = await makeStub('valid-stub')
  const underpoweredHarness = await invokeHarness('underpowered-harness', validStub, 'full', '4')
  assert.notEqual(underpoweredHarness.result.status, 0)
  assert.match(underpoweredHarness.result.stderr, /multiple of 10 rounds/)
  const fullHarness = await invokeHarness('full-harness', validStub)
  assert.equal(fullHarness.result.status, 0, fullHarness.result.stderr)
  const fullOrder = (await readFile(fullHarness.log, 'utf8')).trim().split('\n')
  assert.equal(fullOrder.length, 50)
  for (let offset = 0; offset < fullOrder.length; offset += 10) {
    assert.deepEqual(fullOrder.slice(offset + 5, offset + 10), fullOrder.slice(offset, offset + 5).reverse())
  }
  const fullSequences = Array.from({ length: 10 }, (_, round) => fullOrder.slice(round * 5, round * 5 + 5))
  for (const arm of new Set(fullOrder)) {
    const positionCounts = Array.from({ length: 5 }, (_, position) =>
      fullSequences.filter(sequence => sequence[position] === arm).length)
    assert.deepEqual(positionCounts, [2, 2, 2, 2, 2], `${arm} must occupy every position twice`)
    for (const successor of new Set(fullOrder)) {
      if (successor === arm) continue
      const carryovers = fullSequences.reduce((count, sequence) =>
        count + sequence.slice(0, -1).filter((value, index) => value === arm && sequence[index + 1] === successor).length, 0)
      assert.equal(carryovers, 2, `${arm} -> ${successor} must occur twice`)
    }
  }
  const fullHeader = JSON.parse((await readFile(fullHarness.output, 'utf8')).split('\n')[0])
  assert.deepEqual(fullHeader.arms, armSets.full.arms)
  assert.equal(fullHeader.order_design, orderDesign)

  const prefillHarness = await invokeHarness('prefill-harness', validStub, 'prefill')
  assert.equal(prefillHarness.result.status, 0, prefillHarness.result.stderr)
  const prefillOrder = (await readFile(prefillHarness.log, 'utf8')).trim().split('\n')
  assert.equal(prefillOrder.length, 20)
  assert.ok(prefillOrder.includes('q8/1/1'))
  assert.ok(prefillOrder.includes('q8/1/0'), 'q8-noattnmm must explicitly disable Q8 attention matmul')

  const failedStub = await makeStub('failed-stub', 'exit 7')
  const failedHarness = await invokeHarness('failed-harness', failedStub)
  assert.notEqual(failedHarness.result.status, 0, 'all child failures must fail the harness')
  assert.match(failedHarness.result.stderr, /campaign failed validation/)

  const controlFailure = '[ "$CAMELID_METAL_KV_DTYPE" = q8 ] && [ "$CAMELID_METAL_ATTN_SPLITK" = 0 ] && exit 8'
  const controlFailStub = await makeStub('control-fail-stub', controlFailure)
  const controlFailHarness = await invokeHarness('control-fail-harness', controlFailStub)
  assert.notEqual(controlFailHarness.result.status, 0, 'a failed q8-nosplitk control must fail the harness')

  const doubleJsonStub = await makeStub('double-json-stub', '', 2)
  const doubleJsonHarness = await invokeHarness('double-json-harness', doubleJsonStub, 'prefill')
  assert.notEqual(doubleJsonHarness.result.status, 0, 'multiple parsed child records must fail the harness')
  assert.match(await readFile(doubleJsonHarness.output, 'utf8'), /expected exactly one JSON object, found 2/)
} finally {
  await rm(temp, { recursive: true, force: true })
}
