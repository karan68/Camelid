#!/usr/bin/env node
import assert from 'node:assert/strict'
import { getChatGateState } from '../src/lib/chatGate.js'

const row = {
  id: 'lfm2_5_2_6b_q8_0',
  family: 'lfm2_decoder',
  quantization: 'Q8_0',
  status: 'supported_exact_row_smoke',
  support_scope: 'windows_x86_64_runnable_cpu_or_mac16_10_macos_26_5_resident_metal',
  evidence: 'two explicitly scoped receipts',
}
const capabilities = { model_compatibility: [row] }
const model = {
  id: 'lfm2_5_2_6b_q8_0',
  runtime_model_name: 'lfm2_5_2_6b_q8_0',
  catalog_id: row.id,
  name: 'LFM2.5 2.6B Q8_0',
  quant: 'Q8_0',
  lane_class: 'supported',
  provider_kind: 'local',
  model_path: 'models/LFM2.5-2.6B-Q8_0.gguf',
  status: 'ready',
  loaded_now: true,
  generation_ready: true,
}
const runtime = {
  status: 'online',
  loaded_now: true,
  generation_ready: true,
  active_model_id: model.runtime_model_name,
  backend: 'runnable-runtime',
}
const macPlan = {
  operating_system: 'macos',
  architecture: 'aarch64',
  cpu_model: 'Apple M4',
  model_family: 'lfm2',
  quant_type: 'Q8_0',
  exact_model_row: 'LFM2.5-2.6B-Q8_0.gguf',
  support_level: 'supported_exact_row_smoke',
  selected_backend: 'metal_resident_lfm2_runtime',
  prefill_path: 'lfm2_metal_resident_prefill',
  decode_path: 'lfm2_metal_resident_decode',
}
const windowsPlan = {
  operating_system: 'windows',
  architecture: 'x86_64',
  cpu_model: 'AMD Ryzen fixture',
  model_family: 'lfm2',
  quant_type: 'Q8_0',
  exact_model_row: 'LFM2.5-2.6B-Q8_0.gguf',
  support_level: 'supported_exact_row_smoke',
  selected_backend: 'cpu_reference',
  prefill_path: 'safe_cpu_prefill',
  decode_path: 'safe_cpu_decode',
}

for (const execution_plan of [macPlan, windowsPlan]) {
  const gate = getChatGateState(capabilities, model, { ...runtime, execution_plan })
  assert.equal(gate.contractSupported, true, 'each receipted lane must remain supported')
  assert.equal(gate.chatUnlocked, true)
  assert.equal(gate.experimentalUnlocked, false)
  assert.equal(gate.chatMode, 'supported')
}

for (const execution_plan of [
  null,
  { ...macPlan, operating_system: 'linux' },
  { ...macPlan, cpu_model: 'Apple M3' },
  { ...macPlan, support_level: 'unknown_or_unvalidated' },
  { ...macPlan, selected_backend: 'cpu_reference' },
  { ...macPlan, prefill_path: 'safe_cpu_prefill' },
  { ...macPlan, decode_path: 'safe_cpu_decode' },
  { ...windowsPlan, architecture: 'aarch64' },
  { ...windowsPlan, selected_backend: 'cuda_resident_q8_runtime_runnable_unvalidated' },
]) {
  const gate = getChatGateState(capabilities, model, { ...runtime, execution_plan })
  assert.equal(gate.contractSupported, false, `unreceipted plan must fail closed: ${JSON.stringify(execution_plan)}`)
  assert.equal(gate.chatUnlocked, false)
  assert.equal(gate.experimentalUnlocked, true)
  assert.equal(gate.chatMode, 'experimental')
  assert.equal(gate.hint?.target?.status, 'experimental_runtime_lane')
}

const explicitExperimental = getChatGateState(
  capabilities,
  { ...model, lane_class: 'experimental_implemented' },
  { ...runtime, execution_plan: macPlan },
)
assert.equal(explicitExperimental.contractSupported, false, 'backend artifact verdict remains authoritative')
assert.equal(explicitExperimental.chatMode, 'experimental')

const missingVerdict = getChatGateState(
  capabilities,
  { ...model, lane_class: undefined },
  { ...runtime, execution_plan: macPlan },
)
assert.equal(missingVerdict.contractSupported, false, 'LFM must not inherit a static green row without a backend artifact verdict')
assert.equal(missingVerdict.chatMode, 'experimental')

console.log('lfm2 chat gate smoke: ok')
