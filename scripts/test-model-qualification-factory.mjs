#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  artifactForRow,
  firstUnresolvedStage,
  publicRosterLabel,
  selectRows,
  summarizeReports,
} from './model-qualification-factory.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const roster = JSON.parse(await readFile(resolve(root, 'qa/model-qualification/phase1-roster.json'), 'utf8'))
const qwen = roster.rows.find((row) => row.id === 'qwen2_5_0_5b_instruct_q8_0')
const qwenMoe = roster.rows.find((row) => row.id === 'qwen3_30b_a3b_q8_0')

assert.deepEqual(
  selectRows(roster, ['qwen3_30b_a3b_q8_0', 'lfm2_5_2_6b_q8_0']).map((row) => row.id),
  ['lfm2_5_2_6b_q8_0', 'qwen3_30b_a3b_q8_0'],
  'selected rows must preserve qualification priority',
)
assert.throws(() => selectRows(roster, ['not_a_row']), /unknown qualification rows/)
assert.equal(
  artifactForRow(qwen, resolve(root, 'models')),
  resolve(root, 'models', qwen.identity.gguf_filename),
)
assert.equal(
  artifactForRow(qwenMoe, resolve(root, 'models')),
  resolve(root, 'models', qwenMoe.identity.gguf_filename),
  'newly anchored Qwen3 MoE must resolve its official exact filename',
)
const unanchored = structuredClone(qwenMoe)
unanchored.identity.gguf_filename = null
unanchored.source.file = null
assert.equal(artifactForRow(unanchored, resolve(root, 'models')), null, 'unanchored rows must not invent an artifact filename')
assert.equal(artifactForRow(qwenMoe, null, resolve(root, 'manual.gguf')), resolve(root, 'manual.gguf'))
assert.equal(
  publicRosterLabel(root, resolve(root, 'qa/model-qualification/phase1-roster.json')),
  'qa/model-qualification/phase1-roster.json',
)
assert.equal(
  publicRosterLabel(root, resolve(root, '..', 'private', 'secret-roster.json')),
  '<external-roster>',
  'a scrubbed factory index must not copy an external absolute roster path',
)

const blockedReport = {
  overall_status: 'blocked',
  stages: {
    artifact: { status: 'pass' },
    source: { status: 'pass' },
    metadata: { status: 'pass' },
    tokenizer: { status: 'blocked' },
  },
}
assert.equal(firstUnresolvedStage(blockedReport, roster.gate_order), 'tokenizer')
assert.equal(
  firstUnresolvedStage({ overall_status: 'blocked', stages: { artifact: { status: 'blocked' } } }, roster.gate_order),
  'artifact',
)
assert.deepEqual(
  summarizeReports([{ row: qwen, report: blockedReport, reportFile: 'qwen.json' }], roster.gate_order),
  {
    counts: { pass: 0, fail: 0, blocked: 1 },
    rows: [{
      priority: qwen.priority,
      row_id: qwen.id,
      disposition: qwen.disposition,
      overall_status: 'blocked',
      first_unresolved_stage: 'tokenizer',
      report_file: 'qwen.json',
    }],
  },
)

console.log('test-model-qualification-factory: all checks passed')
