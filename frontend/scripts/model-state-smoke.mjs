#!/usr/bin/env node
import assert from 'node:assert/strict'

import {
  capabilityStatusTone,
  compatibilityHintCopy,
  compatibilityHintLabel,
  findCompatibilityHint,
  formatCapabilityStatus,
  getCurrentCompatibilityTarget,
  guardedCapabilityCopy,
  isCompatibilitySupportedForModel,
  isGuardedCapabilityStatus,
  isSupportedCapabilityStatus,
  quantLabelFromGgufFileType,
  summarizeCapabilityItems,
} from '../src/lib/capabilities.js'

import {
  canLoadIntoRuntime,
  describeModelState,
  getModelStatusLabel,
  hasLocalModelPath,
  isExternalModel,
  isHostedRoutingAvailable,
  isModelGenerationReady,
  isModelLoadedNow,
  isRunnableInCurrentRuntime,
  isRunnableModel,
} from '../src/lib/modelState.js'

const localLoadedReady = {
  id: 'tiny-generation',
  name: 'Tiny generation',
  provider_kind: 'local',
  status: 'ready',
  model_path: '/tmp/tiny-generation.gguf',
  loaded_now: true,
  generation_ready: true,
}

assert.equal(isExternalModel(localLoadedReady), false)
assert.equal(hasLocalModelPath(localLoadedReady), true)
assert.equal(isModelLoadedNow(localLoadedReady), true)
assert.equal(isModelGenerationReady(localLoadedReady), true)
assert.equal(isRunnableModel(localLoadedReady), true)
assert.equal(isRunnableInCurrentRuntime(localLoadedReady, { active_model_id: 'tiny-generation', generation_ready: true }), true)
assert.equal(isRunnableInCurrentRuntime(localLoadedReady, { active_model_id: 'other-model', generation_ready: true }), false, 'a local model is not runnable for chat if a different model is active in Camelid')
assert.equal(isRunnableInCurrentRuntime(localLoadedReady, { active_model_id: 'tiny-generation', generation_ready: false }), false, 'loaded_now alone is not enough without runtime generation_ready=true')
assert.equal(getModelStatusLabel(localLoadedReady), 'Loaded + generation-ready')
assert.match(describeModelState(localLoadedReady), /generation_ready=true/)

const nestedLoadedReady = {
  ...localLoadedReady,
  loaded_now: false,
  generation_ready: false,
  backendinference: { loaded_now: true, generation_ready: true },
}
assert.equal(isModelLoadedNow(nestedLoadedReady), true)
assert.equal(isModelGenerationReady(nestedLoadedReady), true)
assert.equal(isRunnableModel(nestedLoadedReady), true, 'nested backend readiness should unlock chat when the local GGUF path is present')

const localSavedPath = {
  ...localLoadedReady,
  status: 'registered',
  loaded_now: false,
  generation_ready: false,
  backendinference: { loaded_now: false, generation_ready: false },
}
assert.equal(canLoadIntoRuntime(localSavedPath), true)
assert.equal(isRunnableModel(localSavedPath), false)
assert.equal(getModelStatusLabel(localSavedPath), 'Local path saved')
assert.match(describeModelState(localSavedPath), /Use Load now/)

const localLoadedNotReady = {
  ...localLoadedReady,
  loaded_now: true,
  generation_ready: false,
  backendinference: { loaded_now: true, generation_ready: false },
}
assert.equal(isRunnableModel(localLoadedNotReady), false)
assert.equal(getModelStatusLabel(localLoadedNotReady), 'Loaded, not generation-ready')
assert.match(describeModelState(localLoadedNotReady), /generation_ready=false/)
assert.match(describeModelState(localLoadedNotReady), /materialization budget/)

const staleReadyRecord = {
  ...localLoadedReady,
  loaded_now: false,
  backendinference: { loaded_now: false, generation_ready: true },
}
assert.equal(isRunnableModel(staleReadyRecord), false, 'a stale saved record is not runnable unless it is loaded now')
assert.equal(isRunnableInCurrentRuntime(staleReadyRecord, { active_model_id: 'tiny-generation', generation_ready: true }), false)

const hostedPlanned = {
  id: 'openai-gpt-4o-mini',
  name: 'OpenAI GPT-4o mini',
  provider_kind: 'external',
  status: 'ready',
  api_base: 'https://api.openai.com/v1',
  runtime_model_name: 'gpt-4o-mini',
  api_key_configured: true,
}
assert.equal(isExternalModel(hostedPlanned), true)
assert.equal(isHostedRoutingAvailable(hostedPlanned), false)
assert.equal(isRunnableModel(hostedPlanned), false)
assert.equal(canLoadIntoRuntime(hostedPlanned), false)
assert.equal(getModelStatusLabel(hostedPlanned), 'API routing planned')
assert.match(describeModelState(hostedPlanned), /not wired yet/)

const hostedReady = { ...hostedPlanned, hosted_routing_ready: true }
assert.equal(isHostedRoutingAvailable(hostedReady), true)
assert.equal(isRunnableModel(hostedReady), true)
assert.equal(getModelStatusLabel(hostedReady), 'API routing ready')

assert.equal(formatCapabilityStatus('planned_phase_11_12'), 'planned phase 11 12')
assert.equal(quantLabelFromGgufFileType(7), 'Q8_0')
assert.equal(quantLabelFromGgufFileType('15'), 'Q4_K_M')
assert.equal(quantLabelFromGgufFileType(32), 'BF16')
assert.equal(quantLabelFromGgufFileType('unknown'), null)
assert.equal(isSupportedCapabilityStatus('supported_current_gate'), true)
assert.equal(isGuardedCapabilityStatus('future'), true)
assert.equal(capabilityStatusTone('blocked_until_tensor_load_and_parity'), 'warm')
assert.match(summarizeCapabilityItems([{ id: 'Q8_0', status: 'supported_current_gate' }]), /Q8_0: supported current gate/)
assert.match(guardedCapabilityCopy({ notes: 'Multi-choice is not implemented yet' }, 'API controls'), /API controls should stay disabled.*typed backend refusals.*not silently drop/)
assert.equal(getCurrentCompatibilityTarget({ model_compatibility: [{ id: 'planned', status: 'planned' }, { id: 'tiny', status: 'supported_current_gate' }] }).id, 'tiny')

const capabilityFixture = {
  planned_model_families: [
    { id: 'larger_llama_instruct', status: 'planned', notes: 'progressively larger LLaMA-family instruct models' },
  ],
  model_compatibility: [
    { id: 'tinyllama_1_1b_chat_q8_0', family: 'llama_spm_decoder', quantization: 'Q8_0', status: 'supported_current_gate', evidence: 'TinyLlama Q8_0 evidence' },
    { id: 'llama_spm_q4_k_q5_k', family: 'llama_spm_decoder', quantization: 'Q4_K_M/Q5_K_M', status: 'planned_phase_10', next_step: 'implement K-quant support' },
    { id: 'llama32_1b_instruct_q8_0', family: 'llama_bpe_decoder', quantization: 'Q8_0', status: 'evidence_only', evidence: '1B compact-header evidence only' },
    { id: 'llama32_3b_instruct_q8_0', family: 'llama_bpe_decoder', quantization: 'Q8_0', status: 'acceptance_target_blocked_before_first_token', next_step: 'clear the first-token memory blocker before parity work' },
    { id: 'llama3_8b_instruct_gguf', family: 'llama_bpe_decoder', quantization: 'Q8_0', status: 'planned_phase_11_12', next_step: 'safe lazy execution first' },
  ],
}
const tinyQ8Hint = findCompatibilityHint(capabilityFixture, { name: 'TinyLlama 1.1B Chat', quant: 'Q8_0' })
assert.equal(tinyQ8Hint.target.id, 'tinyllama_1_1b_chat_q8_0')
assert.equal(compatibilityHintLabel(tinyQ8Hint), 'tinyllama_1_1b_chat_q8_0: supported current gate')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'TinyLlama 1.1B Chat', quant: 'Q8_0' }), true)
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'TinyLlama 1.1B Chat', quant: 'file_type 7' }), true, 'GGUF file_type labels should map to exact quant rows')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'TinyLlama 1.1B Chat' }), false, 'chat should not unlock from a family/name match without quant evidence')
const tinyKQuantHint = findCompatibilityHint(capabilityFixture, { name: 'TinyLlama 1.1B Chat', quant: 'Q4_K_M' })
assert.equal(tinyKQuantHint.target.id, 'llama_spm_q4_k_q5_k', 'TinyLlama family names must not inherit Q8 support for a K-quant entry')
assert.equal(compatibilityHintLabel(tinyKQuantHint), 'llama_spm_q4_k_q5_k: planned phase 10')
assert.match(compatibilityHintCopy(tinyKQuantHint), /runtime generation still requires loaded_now=true and generation_ready=true/)
const llama3Q4Hint = findCompatibilityHint(capabilityFixture, { name: 'Meta Llama 3 8B Instruct', quant: 'Q4_K_M' })
assert.equal(llama3Q4Hint.kind, 'quant_mismatch')
assert.match(compatibilityHintCopy(llama3Q4Hint), /Do not inherit the supported gate|wait for an exact COMPATIBILITY\.md row/)
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Meta Llama 3 8B Instruct', quant: 'Q8_0' }), false, 'planned Llama 3 8B rows must not unlock chat')
const llama32OneBHint = findCompatibilityHint(capabilityFixture, { name: 'Llama 3.2 1B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama32OneBHint.target.id, 'llama32_1b_instruct_q8_0', 'Llama 3.2 1B must match its exact evidence-only row')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Llama 3.2 1B Instruct Q8_0', quant: 'Q8_0' }), false, 'evidence-only 1B rows must not unlock chat')
const llama32ThreeBHint = findCompatibilityHint(capabilityFixture, { name: 'Llama 3.2 3B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama32ThreeBHint.target.id, 'llama32_3b_instruct_q8_0', 'Llama 3.2 3B must match its exact row rather than inheriting the 8B row')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Llama 3.2 3B Instruct Q8_0', quant: 'Q8_0' }), false, 'blocked 3B rows must not unlock chat')
const noExactThreeBHint = findCompatibilityHint({ ...capabilityFixture, model_compatibility: capabilityFixture.model_compatibility.filter((row) => row.id !== 'llama32_3b_instruct_q8_0') }, { name: 'Llama 3.2 3B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(noExactThreeBHint.kind, 'family', 'Llama 3.2 3B must stay at planned-family evidence when no exact compatibility row exists')
assert.match(compatibilityHintCopy(noExactThreeBHint), /not chat-ready support until a concrete compatibility row is validated/)

console.log('✓ model-state smoke passed')
