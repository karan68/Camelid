#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { getChatGateState } from '../src/lib/chatGate.js'

const gemma426bRow = {
  id: 'gemma4_26b_a4b_it_q4_0',
  family: 'gemma4_a4b_moe_decoder',
  quantization: 'Q4_0',
  status: 'supported_exact_row_smoke',
  support_scope: 'exact_row_distributed_serve_smoke_only',
  evidence: 'distributed serve receipt',
}

const residentRow = {
  id: 'gemma4_e4b_it_q8_0',
  family: 'gemma4_decoder',
  quantization: 'Q8_0',
  status: 'supported_exact_row_smoke',
  support_scope: 'exact_row_resident_serve_smoke_only',
  evidence: 'resident serve receipt',
}

// Keep the resident row first: generic Gemma family fallback ordering must not
// make the 26B lane inherit or display the wrong row.
const capabilities = { model_compatibility: [residentRow, gemma426bRow] }
const model26b = {
  id: 'gemma-4-26B_q4_0-it.gguf',
  runtime_model_name: 'gemma-4-26B_q4_0-it.gguf',
  catalog_id: gemma426bRow.id,
  name: 'Gemma 4 26B A4B Q4_0',
  quant: 'Q4_0',
  lane_class: 'supported',
  provider_kind: 'local',
  model_path: 'models/gemma-4-26B_q4_0-it.gguf',
  status: 'ready',
  loaded_now: true,
  generation_ready: true,
}
const baseRuntime = {
  status: 'online',
  loaded_now: true,
  generation_ready: true,
  active_model_id: model26b.runtime_model_name,
  backend: 'gemma4-runtime',
}

for (const gemma4_serve_lane of ['ghost_moe', 'local', 'cuda', undefined, 'future_lane']) {
  const gate = getChatGateState(
    capabilities,
    model26b,
    { ...baseRuntime, gemma4_serve_lane },
  )
  assert.equal(gate.contractSupported, false, `${gemma4_serve_lane || 'absent'} must not inherit distributed support`)
  assert.equal(gate.chatUnlocked, false)
  assert.equal(gate.experimentalUnlocked, true)
  assert.equal(gate.chatMode, 'experimental')
  assert.equal(gate.hint?.target?.status, 'experimental_runtime_lane')
  assert.match(gate.label, /experimental runtime lane/i)
  assert.match(gate.copy, /experimental/i)
}

const distributed = getChatGateState(
  capabilities,
  model26b,
  { ...baseRuntime, gemma4_serve_lane: 'distributed' },
)
assert.equal(distributed.contractSupported, true, 'the evidenced distributed lane stays supported')
assert.equal(distributed.chatUnlocked, true)
assert.equal(distributed.experimentalUnlocked, false)
assert.equal(distributed.chatMode, 'supported')
assert.equal(distributed.hint?.target?.status, 'supported_exact_row_smoke')

const residentModel = {
  ...model26b,
  id: 'gemma-4-E4B-it-Q8_0.gguf',
  runtime_model_name: 'gemma-4-E4B-it-Q8_0.gguf',
  catalog_id: residentRow.id,
  name: 'Gemma 4 E4B Q8_0',
  quant: 'Q8_0',
  model_path: 'models/gemma-4-E4B-it-Q8_0.gguf',
}
const resident = getChatGateState(
  capabilities,
  residentModel,
  {
    ...baseRuntime,
    active_model_id: residentModel.runtime_model_name,
    gemma4_serve_lane: 'local',
  },
)
assert.equal(resident.contractSupported, true, 'a genuinely supported resident row is unchanged')
assert.equal(resident.chatMode, 'supported')
assert.equal(resident.chatUnlocked, true)

const dashboardSource = readFileSync(
  new URL('../src/hooks/useDashboardData.js', import.meta.url),
  'utf8',
)
assert.match(
  dashboardSource,
  /gemma4_serve_lane:\s*optionalString\(health\?\.gemma4_serve_lane\)/,
  'health lane must survive the dashboard projection used by every chat gate',
)

console.log('ghost-moe chat gate smoke: ok')
