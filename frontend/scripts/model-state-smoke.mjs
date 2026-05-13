#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

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
  isExactCompatibilityHint,
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
  modelRuntimeIdMatches,
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
const localReadyWithRuntimeName = { ...localLoadedReady, id: 'browser-alias', runtime_model_name: 'backend-runtime-id' }
assert.equal(modelRuntimeIdMatches(localReadyWithRuntimeName, { active_model_id: 'backend-runtime-id' }), true, 'API/support readiness should accept the backend runtime model id when it differs from the browser alias')
assert.equal(isRunnableInCurrentRuntime(localReadyWithRuntimeName, { active_model_id: 'backend-runtime-id', generation_ready: true }), true, 'runtime-name matches keep chat/API gating tied to the exact loaded backend row')
assert.equal(getChatGateState({ model_compatibility: [] }, localReadyWithRuntimeName, { active_model_id: 'backend-runtime-id', loaded_now: true, generation_ready: true }).runtimeReady, true, 'chat gate runtime readiness should use the same runtime id matcher as the API view')
assert.equal(getModelStatusLabel(localLoadedReady), 'Loaded + generation-ready')
assert.match(describeModelState(localLoadedReady), /generation_ready=true/)

const nestedLoadedReady = {
  ...localLoadedReady,
  loaded_now: false,
  generation_ready: false,
  camelid: { loaded_now: true, generation_ready: true },
}
assert.equal(isModelLoadedNow(nestedLoadedReady), true)
assert.equal(isModelGenerationReady(nestedLoadedReady), true)
assert.equal(isRunnableModel(nestedLoadedReady), true, 'nested backend readiness should unlock chat when the local GGUF path is present')

const localSavedPath = {
  ...localLoadedReady,
  status: 'registered',
  loaded_now: false,
  generation_ready: false,
  camelid: { loaded_now: false, generation_ready: false },
}
assert.equal(canLoadIntoRuntime(localSavedPath), true)
assert.equal(isRunnableModel(localSavedPath), false)
assert.equal(getModelStatusLabel(localSavedPath), 'Local path saved')
assert.match(describeModelState(localSavedPath), /Use Load now/)

const localLoadedNotReady = {
  ...localLoadedReady,
  loaded_now: true,
  generation_ready: false,
  camelid: { loaded_now: true, generation_ready: false },
}
assert.equal(isRunnableModel(localLoadedNotReady), false)
assert.equal(getModelStatusLabel(localLoadedNotReady), 'Loaded, not generation-ready')
assert.match(describeModelState(localLoadedNotReady), /generation_ready=false/)
assert.match(describeModelState(localLoadedNotReady), /materialization budget/)

const staleReadyRecord = {
  ...localLoadedReady,
  loaded_now: false,
  camelid: { loaded_now: false, generation_ready: true },
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
assert.equal(isSupportedCapabilityStatus('validated'), false, 'validated evidence must not be treated as a support status')
assert.equal(isSupportedCapabilityStatus('measured'), false, 'measurement evidence must not be treated as a support status')
assert.equal(isGuardedCapabilityStatus('future'), true)
assert.equal(capabilityStatusTone('blocked_until_tensor_load_and_parity'), 'warm')
assert.equal(capabilityStatusTone('groundwork_backend_evidence_only'), 'warm')
assert.equal(capabilityStatusTone('blocked_unsupported_bringup'), 'warm')
assert.equal(capabilityStatusTone('validated_second_pack'), 'ready')
assert.equal(capabilityStatusTone('validated_bounded_pack_not_promoted'), 'warm')
assert.equal(capabilityStatusTone('fail-closed_until_promotion'), 'warm')
assert.equal(capabilityStatusTone('supported_exact_row_smoke'), 'ready')
assert.match(summarizeCapabilityItems([{ id: 'Q8_0', status: 'supported_current_gate' }]), /Q8_0: supported current gate/)
assert.match(guardedCapabilityCopy({ notes: 'Multi-choice is not implemented yet' }, 'API controls'), /API controls should stay disabled.*typed backend refusals.*not silently drop/)
assert.equal(getCurrentCompatibilityTarget({ model_compatibility: [{ id: 'planned', status: 'planned' }, { id: 'tiny', status: 'supported_current_gate' }] }).id, 'tiny')
assert.equal(getCurrentCompatibilityTarget({ model_compatibility: [{ id: 'planned', status: 'planned' }] }), null, 'a planned/evidence row must not become the current supported gate fallback')

const capabilityFixture = {
  planned_model_families: [
    { id: 'larger_llama_instruct', status: 'planned', notes: 'progressively larger LLaMA-family instruct models' },
  ],
  model_compatibility: [
    { id: 'tinyllama_1_1b_chat_q8_0', family: 'llama_spm_decoder', quantization: 'Q8_0', status: 'supported_current_gate', support_scope: 'current_full_gate_exact_row', full_support_status: 'current_gate_refresh_under_stricter_bar', full_support_blockers: 'do not widen beyond TinyLlama 1.1B Chat Q8_0 without repeated current-head API/WebUI/parity/RSS/context evidence under the stricter bar', frontend_readiness_gate: 'green only when this exact Q8_0 row is loaded_now=true, generation_ready=true, and selected by active_model_id', bounded_context_512_pack: 'validated_bounded_pack', bounded_context_1024_pack: 'not_promoted', bounded_context_2048_pack: 'not_promoted', latest_checked_bucket: 'direct_chat_smoke', latest_checked_result: 'pass', latest_checked_output: 'Certainly! Here', evidence: 'TinyLlama Q8_0 evidence' },
    { id: 'llama_spm_q4_k_q5_k', family: 'llama_spm_decoder', quantization: 'Q4_K_M/Q5_K_M', status: 'planned_phase_10', next_step: 'implement K-quant support' },
    { id: 'llama32_1b_instruct_q8_0', family: 'llama_bpe_decoder', quantization: 'Q8_0', status: 'supported_exact_row_smoke', full_support_status: 'blocked_pending_normalized_full_support', full_support_blockers: 'model-native/larger context beyond checked packs, arbitrary/Jinja templates, production throughput, portability, and durable repeated current-head bundles remain missing', frontend_readiness_gate: 'green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id', bounded_context_1024_pack: 'validated_second_pack', bounded_context_2048_pack: 'validated_third_pack', latest_checked_bucket: 'llama3-context-2048-smoke-v1', latest_checked_result: 'pass', latest_checked_output: 'CMLD-204', evidence: '1B exact-row load, completion, chat, frontend smoke, second 1024-context evidence, and third 2048-context evidence after the RoPE factor fix' },
    { id: 'llama32_3b_instruct_q8_0', family: 'llama_bpe_decoder', quantization: 'Q8_0', status: 'supported_exact_row_smoke', full_support_status: 'blocked_pending_normalized_full_support', full_support_blockers: 'model-native/larger context beyond checked packs, arbitrary/Jinja templates, production throughput, portability, and durable repeated current-head bundles remain missing', frontend_readiness_gate: 'green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id', bounded_context_1024_pack: 'validated_second_pack', bounded_context_2048_pack: 'validated_third_pack', latest_checked_bucket: 'llama3-context-2048-smoke-v1', latest_checked_result: 'pass', latest_checked_output: 'CMLD-204', evidence: '3B exact-row load, completion, chat, frontend smoke, compact parity, broader prompt-pack, first 512-context, second 1024-context, and third 2048-context evidence' },
    { id: 'llama3_8b_instruct_q8_0', family: 'llama_bpe_decoder', quantization: 'Q8_0', status: 'supported_exact_row_smoke', support_scope: 'exact_row_smoke_only', full_support_status: 'blocked_pending_normalized_full_support', full_support_blockers: 'model-native/larger context beyond the checked 512/1024/2048 packs, arbitrary templates, throughput, portability, repeated current-head evidence, and durable normalized full-support bundles remain missing', frontend_readiness_gate: 'green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id', bounded_context_512_pack: 'validated_first_pack', bounded_context_1024_pack: 'validated_second_pack', bounded_context_2048_pack: 'validated_third_pack', latest_checked_bucket: 'llama3-context-2048-smoke-v1', latest_checked_result: 'pass', latest_checked_output: 'CMLD-204', evidence: '8B exact-row API/frontend smoke plus compact 50-token, broader 50-token, checked 512/1024/2048-context packs, compact template-shapes pack evidence, bounded memory/hot-path measurements, and current-head 1024/2048 PASS evidence. No model-native/larger context or broader/full support is implied.' },
    { id: 'mistral_7b_instruct_v0_3_q8_0', family: 'mistral', quantization: 'Q8_0', status: 'active_validation_unsupported', support_scope: 'bringup_exact_row_unsupported', full_support_status: 'blocked_unsupported_bringup', full_support_blockers: 'API/WebUI readiness, RSS/timing, current-head promotion sync, scrubbed manifest posture, support-surface proof, and durable promotion bundle evidence remain incomplete; exact tokenizer/template references plus row-specific 1-token/bounded/broader parity evidence alone do not promote support', evidence: 'Mistral v0.3 active validation only; exact tokenizer/template, 1-token, broader 50-token, and bounded context evidence are green but support-promotion evidence remains fail-closed' },
    { id: 'mixtral_8x7b_instruct_v0_1_q8_0', family: 'mixtral_moe', quantization: 'Q8_0', status: 'active_validation_partial_runtime', support_scope: 'exact_row_bounded_moe_runtime_only', full_support_status: 'blocked_later_generation_divergence', full_support_blockers: 'later short-prompt generation still diverges from llama.cpp; API/WebUI readiness, long-context evidence, production throughput, portability, and durable broad prompt coverage are missing', frontend_readiness_gate: 'fail-closed for broad readiness: exact row may be described only as bounded one-token backend runtime evidence until later-generation parity and API/WebUI gates close', evidence: 'Mixtral bounded one-token backend MoE runtime evidence only; later-generation divergence keeps frontend/API/WebUI support blocked' },
    { id: 'qwen25_7b_instruct_q8_0', family: 'qwen2', quantization: 'Q8_0', status: 'planned_unsupported', support_scope: 'future_exact_row_planning_only', full_support_status: 'not_applicable_until_runtime_support', full_support_blockers: 'qwen2 runtime, tokenizer/pre-tokenizer fixtures, ChatML parity, bounded load/readiness, API/WebUI, RSS/timing, context, and durable bundle evidence are missing', evidence: 'Qwen 2.5 planning row only; no support evidence exists' },
    { id: 'gemma2_9b_it_q8_0', family: 'gemma2', quantization: 'Q8_0', status: 'planned_unsupported', support_scope: 'future_exact_row_planning_only', full_support_status: 'not_applicable_until_runtime_support', full_support_blockers: 'gemma2 runtime, control-token/template fixtures, bounded load/readiness, API/WebUI, RSS/timing, context, and durable bundle evidence are missing', evidence: 'Gemma 2 planning row only; no support evidence exists' },
  ],
}
const modelsViewSource = readFileSync(new URL('../src/views/ModelsView.jsx', import.meta.url), 'utf8')
assert.match(
  modelsViewSource,
  /pin-badge ready[^>]*>8B 1024\/2048 bounded packs passed</,
  'ModelsView should show 8B 1024/2048 ready only after fresh current-head evidence and docs/API/frontend alignment land',
)
assert.doesNotMatch(
  modelsViewSource,
  /pin-badge warm[^>]*>8B 1024\/2048 needs fresh current-head PASS</,
  'ModelsView must not keep stale warm 8B 1024/2048 copy after fresh current-head PASS and alignment land',
)

const trackedTargets = getTrackedCompatibilityTargets(capabilityFixture)
assert.deepEqual(
  trackedTargets.map((target) => target.id),
  ['tinyllama_1_1b_chat_q8_0', 'llama32_1b_instruct_q8_0', 'llama32_3b_instruct_q8_0', 'llama3_8b_instruct_q8_0'],
  'tracked full-support hardening rows should stay pinned to the exact TinyLlama/1B/3B/8B ids in /api/capabilities order',
)
assert.deepEqual(
  trackedTargets.map((target) => [target.id, target.full_support_status, Boolean(target.full_support_blockers), Boolean(target.frontend_readiness_gate)]),
  [
    ['tinyllama_1_1b_chat_q8_0', 'current_gate_refresh_under_stricter_bar', true, true],
    ['llama32_1b_instruct_q8_0', 'blocked_pending_normalized_full_support', true, true],
    ['llama32_3b_instruct_q8_0', 'blocked_pending_normalized_full_support', true, true],
    ['llama3_8b_instruct_q8_0', 'blocked_pending_normalized_full_support', true, true],
  ],
  'all current rows should carry an explicit stricter full-support bar and fail-closed frontend readiness gate',
)
assert.deepEqual(
  trackedTargets.map((target) => [target.id, target.bounded_context_1024_pack]),
  [
    ['tinyllama_1_1b_chat_q8_0', 'not_promoted'],
    ['llama32_1b_instruct_q8_0', 'validated_second_pack'],
    ['llama32_3b_instruct_q8_0', 'validated_second_pack'],
    ['llama3_8b_instruct_q8_0', 'validated_second_pack'],
  ],
  'frontend tracked rows should preserve the API 1024-context boundary: TinyLlama not promoted; exact 1B/3B/8B promoted only for their checked bounded packs',
)
assert.deepEqual(
  trackedTargets.map((target) => [target.id, target.bounded_context_2048_pack]),
  [
    ['tinyllama_1_1b_chat_q8_0', 'not_promoted'],
    ['llama32_1b_instruct_q8_0', 'validated_third_pack'],
    ['llama32_3b_instruct_q8_0', 'validated_third_pack'],
    ['llama3_8b_instruct_q8_0', 'validated_third_pack'],
  ],
  'frontend tracked rows should preserve the API 2048-context boundary: TinyLlama not promoted; exact 1B/3B/8B promoted only for their checked bounded packs',
)
assert.deepEqual(
  trackedTargets.map((target) => [target.id, target.latest_checked_bucket, target.latest_checked_result, target.latest_checked_output]),
  [
    ['tinyllama_1_1b_chat_q8_0', 'direct_chat_smoke', 'pass', 'Certainly! Here'],
    ['llama32_1b_instruct_q8_0', 'llama3-context-2048-smoke-v1', 'pass', 'CMLD-204'],
    ['llama32_3b_instruct_q8_0', 'llama3-context-2048-smoke-v1', 'pass', 'CMLD-204'],
    ['llama3_8b_instruct_q8_0', 'llama3-context-2048-smoke-v1', 'pass', 'CMLD-204'],
  ],
  'frontend tracked rows should surface the API latest bounded checks without implying broad/full support or model-native/larger-context support',
)

const tinyQ8Hint = findCompatibilityHint(capabilityFixture, { name: 'TinyLlama 1.1B Chat', quant: 'Q8_0' })
assert.equal(tinyQ8Hint.target.id, 'tinyllama_1_1b_chat_q8_0')
assert.equal(compatibilityHintLabel(tinyQ8Hint), 'tinyllama_1_1b_chat_q8_0: supported current gate')
assert.equal(isExactCompatibilityHint(tinyQ8Hint), true, 'TinyLlama support should come from its exact row, not a broad family row')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'TinyLlama 1.1B Chat', quant: 'Q8_0' }), true)
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'TinyLlama 1.1B Chat', quant: 'file_type 7' }), true, 'GGUF file_type labels should map to exact quant rows')
const tinyNoQuantHint = findCompatibilityHint(capabilityFixture, { name: 'TinyLlama 1.1B Chat' })
assert.equal(tinyNoQuantHint.kind, 'quant_missing', 'TinyLlama current gate still needs exact Q8_0 evidence before chat unlocks')
assert.equal(compatibilityHintLabel(tinyNoQuantHint), 'tinyllama_1_1b_chat_q8_0: quant not verified')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'TinyLlama 1.1B Chat' }), false, 'chat should not unlock from a family/name match without quant evidence')
const tinyKQuantHint = findCompatibilityHint(capabilityFixture, { name: 'TinyLlama 1.1B Chat', quant: 'Q4_K_M' })
assert.equal(tinyKQuantHint.kind, 'family', 'TinyLlama K-quant should be shown as a guarded family row, not exact-row evidence')
assert.equal(tinyKQuantHint.target.id, 'llama_spm_q4_k_q5_k', 'TinyLlama family names must not inherit Q8 support for a K-quant entry')
assert.equal(compatibilityHintLabel(tinyKQuantHint), 'llama_spm_q4_k_q5_k: planned phase 10')
assert.equal(isExactCompatibilityHint(tinyKQuantHint), false)
assert.match(compatibilityHintCopy(tinyKQuantHint), /not chat-ready support|concrete exact compatibility row/)
const llama3Q4Hint = findCompatibilityHint(capabilityFixture, { name: 'Meta Llama 3 8B Instruct', quant: 'Q4_K_M' })
assert.equal(llama3Q4Hint.kind, 'quant_mismatch')
assert.match(compatibilityHintCopy(llama3Q4Hint), /Do not inherit the supported gate|wait for an exact COMPATIBILITY\.md row/)
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Meta Llama 3 8B Instruct', quant: 'Q8_0' }), true, 'exact promoted 8B rows are supported only with exact size/instruct/quant evidence')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Meta Llama 3 8B Instruct', quant: 'file_type 7' }), true, 'GGUF file_type evidence should map the 8B exact row back to Q8_0 support only')
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
    chatUnlocked: true,
    chatMode: 'supported',
    label: 'llama32_1b_instruct_q8_0: supported exact row smoke',
    copy: compatibilityHintCopy(llama32OneBHint),
  },
  'Llama 3.2 1B runtime-green exact rows should unlock supported WebUI chat without broad family claims',
)
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'llama32-1b', name: 'Llama 3.2 1B Instruct Q8_0', quant: 'Q8_0' }, { active_model_id: 'llama32-1b', loaded_now: false, generation_ready: true }).chatUnlocked,
  false,
  'exact supported rows still require runtime loaded_now=true before chat unlocks',
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
assert.equal(llama3EightBHint.target.id, 'llama3_8b_instruct_q8_0', 'Llama 3 8B must match its exact supported row')
assert.match(compatibilityHintCopy(llama3EightBHint), /checked 512\/1024\/2048-context packs, compact template-shapes pack evidence, bounded memory\/hot-path measurements, and current-head 1024\/2048 PASS evidence/)
assert.match(compatibilityHintCopy(llama3EightBHint), /No model-native\/larger context or broader\/full support is implied/)
const llama3HyphenEightBHint = findCompatibilityHint(capabilityFixture, { name: 'Meta-Llama-3-8B-Instruct-Q8_0', quant: 'Q8_0' })
assert.equal(llama3HyphenEightBHint.target.id, 'llama3_8b_instruct_q8_0', 'Llama-3-8B filenames should match the exact Llama 3 8B row')
const llama3EightBQuantMissingHint = findCompatibilityHint(capabilityFixture, { name: 'Meta Llama 3 8B Instruct' })
assert.equal(llama3EightBQuantMissingHint.kind, 'quant_missing', 'Llama 3 8B must not unlock from a size/instruct match without Q8_0 evidence')
assert.equal(compatibilityHintLabel(llama3EightBQuantMissingHint), 'llama3_8b_instruct_q8_0: quant not verified')
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'llama3-8b', name: 'Meta Llama 3 8B Instruct Q8_0', quant: 'Q8_0' }, { active_model_id: 'llama3-8b', loaded_now: true, generation_ready: true }).chatUnlocked,
  true,
  'Llama 3 8B exact rows should unlock supported WebUI chat when runtime-green',
)
const llama31EightBHint = findCompatibilityHint(capabilityFixture, { name: 'Meta Llama 3.1 8B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama31EightBHint, null, 'Llama 3.1 8B must not inherit the Llama 3 8B row')
const llama33EightBHint = findCompatibilityHint(capabilityFixture, { name: 'Meta Llama 3.3 8B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama33EightBHint, null, 'Llama 3.3 8B must not inherit the Llama 3 8B row')
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
const evidenceOnly1BFixture = {
  ...capabilityFixture,
  model_compatibility: capabilityFixture.model_compatibility.map((row) => row.id === 'llama32_1b_instruct_q8_0' ? { ...row, status: 'groundwork_backend_evidence_only' } : row),
}
const evidenceOnly1BGate = getChatGateState(evidenceOnly1BFixture, { ...localLoadedReady, id: 'llama32-1b', name: 'Llama 3.2 1B Instruct Q8_0', quant: 'Q8_0' }, { active_model_id: 'llama32-1b', loaded_now: true, generation_ready: true })
assert.equal(evidenceOnly1BGate.runtimeReady, true, 'runtime readiness should be visible even for evidence-only rows')
assert.equal(evidenceOnly1BGate.contractSupported, false, 'evidence-only rows are not exact supported rows')
assert.equal(evidenceOnly1BGate.chatUnlocked, false, 'WebUI chat must remain blocked unless runtime readiness and an exact supported compatibility row both pass')
const validatedOnly1BFixture = {
  ...capabilityFixture,
  model_compatibility: capabilityFixture.model_compatibility.map((row) => row.id === 'llama32_1b_instruct_q8_0' ? { ...row, status: 'validated' } : row),
}
const validatedOnly1BGate = getChatGateState(validatedOnly1BFixture, { ...localLoadedReady, id: 'llama32-1b', name: 'Llama 3.2 1B Instruct Q8_0', quant: 'Q8_0' }, { active_model_id: 'llama32-1b', loaded_now: true, generation_ready: true })
assert.equal(validatedOnly1BGate.contractSupported, false, 'validated rows are evidence boundaries only, not support statuses')
assert.equal(validatedOnly1BGate.chatUnlocked, false, 'WebUI chat must not unlock from a generic validated row status')
const mistralExactHint = findCompatibilityHint(capabilityFixture, { name: 'Mistral-7B-Instruct-v0.3 Q8_0', quant: 'Q8_0' })
assert.equal(mistralExactHint.kind, 'compatibility', 'the future Mistral lane should identify only the exact v0.3 7B Instruct Q8_0 row')
assert.equal(mistralExactHint.target.id, 'mistral_7b_instruct_v0_3_q8_0')
assert.equal(mistralExactHint.target.status, 'active_validation_unsupported', 'Mistral exact-row matching must still advertise unsupported active-validation status')
assert.equal(mistralExactHint.target.full_support_status, 'blocked_unsupported_bringup', 'Mistral exact-row matching must still advertise unsupported bring-up status')
assert.match(mistralExactHint.target.full_support_blockers, /API\/WebUI readiness|RSS\/timing|current-head promotion sync|durable promotion bundle/i, 'Mistral exact-row matching must carry its remaining blocking evidence list')
assert.doesNotMatch(mistralExactHint.target.full_support_blockers, /source\/SHA\/license|1-token generation parity .*not complete/i, 'Mistral exact-row matching must not mark already-green row-specific evidence as missing')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Mistral-7B-Instruct-v0.3 Q8_0', quant: 'Q8_0' }), false, 'Mistral acceptance-target evidence must not unlock chat')
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'mistral-v03', name: 'Mistral-7B-Instruct-v0.3 Q8_0', quant: 'Q8_0' }, { active_model_id: 'mistral-v03', loaded_now: true, generation_ready: true }).chatUnlocked,
  false,
  'even runtime-green Mistral v0.3 remains blocked until /api/capabilities promotes the exact row to supported',
)
const mistralNoQuantHint = findCompatibilityHint(capabilityFixture, { name: 'Mistral-7B-Instruct-v0.3' })
assert.equal(mistralNoQuantHint.kind, 'quant_missing', 'Mistral exact-row support must still require quant evidence')
const mistralV02Hint = findCompatibilityHint(capabilityFixture, { name: 'Mistral-7B-Instruct-v0.2 Q8_0', quant: 'Q8_0' })
assert.equal(mistralV02Hint.kind, 'family', 'Mistral v0.2 must not inherit the v0.3 exact-row lane')
assert.match(compatibilityHintCopy(mistralV02Hint), /not chat-ready support|not support/i)
const mixtralHint = findCompatibilityHint(capabilityFixture, { name: 'Mixtral-8x7B-Instruct-v0.1 Q8_0', quant: 'Q8_0' })
assert.equal(mixtralHint.kind, 'compatibility', 'Mixtral should match only its exact active-validation row, not a Mistral exact-row match')
assert.equal(mixtralHint.target.id, 'mixtral_8x7b_instruct_v0_1_q8_0')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Mixtral-8x7B-Instruct-v0.1 Q8_0', quant: 'Q8_0' }), false)
const mixtralNoQuantHint = findCompatibilityHint(capabilityFixture, { name: 'Mixtral-8x7B-Instruct-v0.1' })
assert.equal(mixtralNoQuantHint.kind, 'quant_missing', 'Mixtral exact-row support must still require quant evidence')
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'mixtral-v01', name: 'Mixtral-8x7B-Instruct-v0.1 Q8_0', quant: 'Q8_0' }, { active_model_id: 'mixtral-v01', loaded_now: true, generation_ready: true }).chatUnlocked,
  false,
  'runtime-green Mixtral v0.1 exact Q8_0 row stays blocked while /api/capabilities keeps it active-validation unsupported',
)
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'mixtral-v01', name: 'Mixtral-8x7B-Instruct-v0.1 Q8_0', quant: 'Q8_0' }, { active_model_id: 'mixtral-v01', loaded_now: true, generation_ready: false }).chatUnlocked,
  false,
  'Mixtral v0.1 exact Q8_0 row remains blocked when runtime generation_ready is false',
)
const qwenHint = findCompatibilityHint(capabilityFixture, { name: 'Qwen2.5-7B-Instruct-Q8_0', quant: 'Q8_0' })
assert.equal(qwenHint.kind, 'compatibility', 'Qwen should match only its exact future planning row')
assert.equal(qwenHint.target.id, 'qwen25_7b_instruct_q8_0')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Qwen2.5-7B-Instruct-Q8_0', quant: 'Q8_0' }), false)
const qwenQ4Hint = findCompatibilityHint(capabilityFixture, { name: 'Qwen2.5-7B-Instruct-Q4_K_M', quant: 'Q4_K_M' })
assert.equal(qwenQ4Hint.kind, 'quant_mismatch', 'Qwen planning rows must not absorb different quantizations')
const gemmaHint = findCompatibilityHint(capabilityFixture, { name: 'gemma-2-9b-it-Q8_0', quant: 'Q8_0' })
assert.equal(gemmaHint.kind, 'compatibility', 'Gemma should match only its exact future planning row')
assert.equal(gemmaHint.target.id, 'gemma2_9b_it_q8_0')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'gemma-2-9b-it-Q8_0', quant: 'Q8_0' }), false)
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'gemma2-9b', name: 'gemma-2-9b-it-Q8_0', quant: 'Q8_0' }, { active_model_id: 'gemma2-9b', loaded_now: true, generation_ready: true }).chatUnlocked,
  false,
  'even runtime-green Gemma planning rows remain blocked until /api/capabilities promotes the exact row to supported',
)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /smoke-supported for local chat/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /\/api\/models\/load, \/v1\/completions, \/v1\/chat\/completions, frontend smoke, compact parity/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /five-prompt API smoke pack, and bounded 512\/1024\/2048-context parity packs/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /does not promote neighboring Llama sizes/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /model-native\/larger contexts beyond the checked packs/)
assert.match(LLAMA32_3B_ACCEPTANCE_AVAILABILITY, /does not currently show the exact 3B row/)
assert.doesNotMatch(LLAMA32_3B_ACCEPTANCE_AVAILABILITY, /not present locally yet/)
assert.match(LLAMA32_3B_ACCEPTANCE_GATING_NOTE, /loaded_now=true and generation_ready=true/)
assert.match(LLAMA32_3B_ACCEPTANCE_GATING_NOTE, /exact supported Llama 3\.2 3B Q8_0 compatibility row/)

console.log('✓ model-state smoke passed')
