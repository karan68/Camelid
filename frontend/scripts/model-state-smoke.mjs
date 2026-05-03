#!/usr/bin/env node
import assert from 'node:assert/strict'

import {
  LLAMA32_3B_ACCEPTANCE_AVAILABILITY,
  LLAMA32_3B_ACCEPTANCE_GATING_NOTE,
  LLAMA32_3B_ACCEPTANCE_SUMMARY,
} from '../src/lib/acceptanceTargets.js'

import {
  capabilityStatusTone,
  compatibilityHintCopy,
  compatibilityHintLabel,
  findCompatibilityHint,
  formatCapabilityStatus,
  getCurrentCompatibilityTarget,
  getTrackedCompatibilityTargets,
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

import { getChatGateState } from '../src/lib/chatGate.js'

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
assert.equal(capabilityStatusTone('groundwork_backend_evidence_only'), 'warm')
assert.match(summarizeCapabilityItems([{ id: 'Q8_0', status: 'supported_current_gate' }]), /Q8_0: supported current gate/)
assert.match(guardedCapabilityCopy({ notes: 'Multi-choice is not implemented yet' }, 'API controls'), /API controls should stay disabled.*typed backend refusals.*not silently drop/)
assert.equal(getCurrentCompatibilityTarget({ model_compatibility: [{ id: 'planned', status: 'planned' }, { id: 'tiny', status: 'supported_current_gate' }] }).id, 'tiny')
assert.equal(getCurrentCompatibilityTarget({ model_compatibility: [{ id: 'planned', status: 'planned' }] }), null, 'a planned/evidence row must not become the current supported gate fallback')

const capabilityFixture = {
  planned_model_families: [
    { id: 'larger_llama_instruct', status: 'planned', notes: 'progressively larger LLaMA-family instruct models' },
  ],
  model_compatibility: [
    { id: 'tinyllama_1_1b_chat_q8_0', family: 'llama_spm_decoder', quantization: 'Q8_0', status: 'supported_current_gate', evidence: 'TinyLlama Q8_0 evidence' },
    { id: 'llama_spm_q4_k_q5_k', family: 'llama_spm_decoder', quantization: 'Q4_K_M/Q5_K_M', status: 'planned_phase_10', next_step: 'implement K-quant support' },
    { id: 'llama32_1b_instruct_q8_0', family: 'llama_bpe_decoder', quantization: 'Q8_0', status: 'supported_exact_row_smoke', evidence: '1B exact-row load, completion, chat, frontend smoke, and prompt-pack evidence' },
    { id: 'llama32_3b_instruct_q8_0', family: 'llama_bpe_decoder', quantization: 'Q8_0', status: 'supported_exact_row_smoke', evidence: '3B exact-row load, completion, chat, frontend smoke, compact parity, and broader prompt-pack evidence' },
    { id: 'llama3_8b_instruct_gguf', family: 'llama_bpe_decoder', quantization: 'Q8_0', status: 'supported_exact_row_smoke', evidence: '8B exact-row API/frontend smoke plus compact and prompt-pack parity evidence' },
  ],
}
assert.deepEqual(
  getTrackedCompatibilityTargets(capabilityFixture).map((target) => target.id),
  ['llama32_1b_instruct_q8_0', 'llama32_3b_instruct_q8_0', 'llama3_8b_instruct_gguf'],
  'tracked promotion rows should stay pinned to the exact 1B/3B/8B ids in /api/capabilities order',
)

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
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Meta Llama 3 8B Instruct', quant: 'Q8_0' }), true, 'exact promoted 8B rows are supported only with exact size/instruct/quant evidence')
const llama32OneBHint = findCompatibilityHint(capabilityFixture, { name: 'Llama 3.2 1B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama32OneBHint.target.id, 'llama32_1b_instruct_q8_0', 'Llama 3.2 1B must match its exact promoted row')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Llama 3.2 1B Instruct Q8_0', quant: 'Q8_0' }), true, 'exact promoted 1B rows are supported only with exact size/instruct/quant evidence')
assert.deepEqual(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'llama32-1b', name: 'Llama 3.2 1B Instruct Q8_0', quant: 'Q8_0' }, { active_model_id: 'llama32-1b', loaded_now: true, generation_ready: true }),
  {
    hint: llama32OneBHint,
    runtimeReady: true,
    runtimeLoaded: true,
    runtimeGenerationReady: true,
    contractSupported: true,
    guardedLlamaEvaluation: false,
    chatUnlocked: true,
    chatMode: 'supported',
    label: 'llama32_1b_instruct_q8_0: supported exact row smoke',
    copy: compatibilityHintCopy(llama32OneBHint),
  },
  'Llama 3.2 1B runtime-green exact rows should unlock supported WebUI chat without broad family claims',
)
const llama32OneBQuantMissingHint = findCompatibilityHint(capabilityFixture, { name: 'Llama 3.2 1B Instruct' })
assert.equal(llama32OneBQuantMissingHint.kind, 'quant_missing', 'Llama 3.2 exact-size matches must not become compatibility matches without quant evidence')
assert.equal(compatibilityHintLabel(llama32OneBQuantMissingHint), 'llama32_1b_instruct_q8_0: quant not verified')
assert.match(compatibilityHintCopy(llama32OneBQuantMissingHint), /does not expose a quant label yet/)
const promotedOneBFixture = {
  ...capabilityFixture,
  model_compatibility: capabilityFixture.model_compatibility.map((row) => row.id === 'llama32_1b_instruct_q8_0' ? { ...row, status: 'supported_current_gate' } : row),
}
assert.equal(isCompatibilitySupportedForModel(promotedOneBFixture, { name: 'Llama 3.2 1B Instruct' }), false, 'exact-size Llama rows still need quant evidence even after promotion')
const llama32ThreeBHint = findCompatibilityHint(capabilityFixture, { name: 'Llama 3.2 3B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama32ThreeBHint.target.id, 'llama32_3b_instruct_q8_0', 'Llama 3.2 3B must match its exact row rather than inheriting the 8B row')
assert.equal(compatibilityHintLabel(llama32ThreeBHint), 'llama32_3b_instruct_q8_0: supported exact row smoke')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Llama 3.2 3B Instruct Q8_0', quant: 'Q8_0' }), true, 'exact promoted 3B rows are supported only with exact size/instruct/quant evidence')
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'llama32-3b', name: 'Llama 3.2 3B Instruct Q8_0', quant: 'Q8_0' }, { active_model_id: 'llama32-3b', loaded_now: true, generation_ready: true }).chatUnlocked,
  true,
  'Llama 3.2 3B exact rows should unlock supported WebUI chat when runtime-green',
)
const llama3EightBHint = findCompatibilityHint(capabilityFixture, { name: 'Meta Llama 3 8B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama3EightBHint.target.id, 'llama3_8b_instruct_gguf', 'Llama 3 8B must match its exact supported row')
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'llama3-8b', name: 'Meta Llama 3 8B Instruct Q8_0', quant: 'Q8_0' }, { active_model_id: 'llama3-8b', loaded_now: true, generation_ready: true }).chatUnlocked,
  true,
  'Llama 3 8B exact rows should unlock supported WebUI chat when runtime-green',
)
const llama32NoSizeHint = findCompatibilityHint(capabilityFixture, { name: 'Llama 3.2 Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama32NoSizeHint, null, 'Llama 3.2 names without exact 1B/3B size must not inherit a tracked row or family readiness hint')
const llama32EightBHint = findCompatibilityHint(capabilityFixture, { name: 'Llama 3.2 8B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama32EightBHint, null, 'Llama 3.2 8B must not inherit the Llama 3 8B row or a family readiness hint')
const llama3OneBHint = findCompatibilityHint(capabilityFixture, { name: 'Meta Llama 3 1B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama3OneBHint, null, 'Llama 3 1B must not inherit the Llama 3.2 1B row or a family readiness hint')
const llama32OneBBaseHint = findCompatibilityHint(capabilityFixture, { name: 'Llama 3.2 1B Base Q8_0', quant: 'Q8_0' })
assert.equal(llama32OneBBaseHint, null, 'Llama 3.2 1B non-instruct names must not inherit the exact Instruct row')
const noExactThreeBHint = findCompatibilityHint({ ...capabilityFixture, model_compatibility: capabilityFixture.model_compatibility.filter((row) => row.id !== 'llama32_3b_instruct_q8_0') }, { name: 'Llama 3.2 3B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(noExactThreeBHint, null, 'Llama 3.2 3B must not show family readiness when no exact compatibility row exists')
assert.match(compatibilityHintCopy(noExactThreeBHint), /No exact COMPATIBILITY\.md row matched/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /smoke-supported for short local chat/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /\/api\/models\/load, \/v1\/completions, \/v1\/chat\/completions, frontend smoke, compact parity/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /five-prompt API smoke pack/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /does not promote neighboring Llama sizes/)
assert.match(LLAMA32_3B_ACCEPTANCE_AVAILABILITY, /does not currently show the exact 3B row/)
assert.doesNotMatch(LLAMA32_3B_ACCEPTANCE_AVAILABILITY, /not present locally yet/)
assert.match(LLAMA32_3B_ACCEPTANCE_GATING_NOTE, /loaded_now=true and generation_ready=true/)
assert.match(LLAMA32_3B_ACCEPTANCE_GATING_NOTE, /exact supported Llama 3\.2 3B Q8_0 compatibility row/)

console.log('✓ model-state smoke passed')
