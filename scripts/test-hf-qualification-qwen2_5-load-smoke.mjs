#!/usr/bin/env node

import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

import {
  BINARY_PROFILE,
  DOES_NOT_PROVE,
  EXACT_ROW,
  EXECUTION_PLAN_EXACT_MODEL_ROW,
  LIMITS,
  Qwen25LoadSmokeError,
  RAW_REQUEST,
  RECEIPT_SCHEMA,
  ROW_ID,
  SAFE_CAMELID_ENV,
  SERVER_ADDR,
  STEP_CONTRACT,
  buildChildEnv,
  buildServeArgs,
  classifyQwen25LoadSmokeError,
  inspectExactArtifactIdentity,
  normalizeGeneration,
  normalizeHealth,
  parseArgs,
  receiptCommand,
  runPreflight,
  runQwen25LoadSmoke,
  validateLoadSmokeReceipt,
  writeReceiptAtomic,
} from './hf-qualification-qwen2_5-load-smoke.mjs'
import {
  SmolLM3LoadSmokeError,
} from './hf-qualification-smollm3-load-smoke.mjs'

const head = '5'.repeat(40)
const sourceDescribe = 'v0.6.1-50-g55555555'
const binarySha256 = '6'.repeat(64)
const binaryVersion = `camelid ${sourceDescribe}`
const root = resolve('qualification-root')
const binary = resolve('qualification-bin', 'camelid.exe')
const artifact = resolve('qualification-artifacts', EXACT_ROW.source.file)
const cwd = resolve('qualification-run', 'work')
const modelsDir = resolve('qualification-run', 'empty-models')
const durableReceiptBytes = await readFile(resolve(
  'qa/model-qualification/qwen2.5-0.5b-q8-windows-cpu-load-smoke.json',
))
const durableReceipt = JSON.parse(durableReceiptBytes)
const fileSha256 = bytes => createHash('sha256').update(bytes).digest('hex')

function deepClone(value) {
  return structuredClone(value)
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]))
  }
  return value
}

function reseal(receipt) {
  const { receipt_id: _old, ...body } = receipt
  return {
    schema: body.schema,
    receipt_id: createHash('sha256').update(JSON.stringify(canonical(body))).digest('hex'),
    ...Object.fromEntries(Object.entries(body).filter(([key]) => key !== 'schema')),
  }
}

async function expectCode(promise, code) {
  await assert.rejects(promise, (error) => {
    const classified = classifyQwen25LoadSmokeError(error)
    assert.equal(classified.error_code, code)
    return true
  })
}

assert.equal(RECEIPT_SCHEMA, 'camelid.model-qualification.load-smoke/v1')
assert.equal(ROW_ID, 'qwen2_5_0_5b_instruct_q8_0')
assert.equal(BINARY_PROFILE, 'release-fat-lto')
assert.equal(EXECUTION_PLAN_EXACT_MODEL_ROW, 'qwen2.5-0.5b-instruct')
assert.equal(EXACT_ROW.disposition, 'active_validation')
assert.equal(EXACT_ROW.source.size_bytes, 675_710_816)
assert.equal(EXACT_ROW.source.sha256, 'ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e')
assert.equal(LIMITS.preflight_physical_bytes, 4 * 1024 ** 3)
assert.equal(LIMITS.preflight_disk_bytes, 4 * 1024 ** 3)
assert.equal(LIMITS.low_memory_abort_bytes, 1024 ** 3)
assert.equal(LIMITS.child_working_set_abort_bytes, 2 * 1024 ** 3)
assert.equal(LIMITS.consecutive_abort_samples, 2)
assert.equal(
  fileSha256(durableReceiptBytes),
  'f74bde1366aabce927ab808a9a8d229e0221cbdb20cd6a9eedd78d3319fa3870',
)
assert.equal(
  durableReceipt.receipt_id,
  'af388a3ef19951dab0da657e59963ad2d725136518a4aac28a1e23451ffa864b',
)
assert.deepEqual(validateLoadSmokeReceipt(durableReceipt), [])
assert.deepEqual(durableReceipt.row, EXACT_ROW)
assert.equal(durableReceipt.provenance.runtime_head,
  '188631159b07f76ca3b081fd1b401090edfa1e21')
assert.equal(durableReceipt.provenance.source_describe, 'v0.6.1-51-g18863115')
assert.equal(durableReceipt.provenance.tracked_files_clean, true)
assert.equal(durableReceipt.provenance.untracked_files_excluded, true)
assert.equal(durableReceipt.provenance.binary.profile, BINARY_PROFILE)
assert.equal(durableReceipt.provenance.binary.sha256,
  '5c4584186fded5fcbb7e848a938afc006f597b3302ed31942562faa908ca3346')
assert.equal(durableReceipt.provenance.binary.version, 'camelid v0.6.1-51-g18863115')
assert.equal(durableReceipt.provenance.binary.health_build, 'v0.6.1-51-g18863115')
assert.equal(durableReceipt.provenance.binary.built_from_clean_tracked_head, true)
assert.equal(durableReceipt.provenance.artifact.size_bytes, EXACT_ROW.source.size_bytes)
assert.equal(durableReceipt.provenance.artifact.sha256, EXACT_ROW.source.sha256)
assert.equal(durableReceipt.provenance.artifact.verified_after_lock_acquisition, true)
assert.equal(durableReceipt.provenance.artifact.verified_after_generation, true)
assert.equal(durableReceipt.provenance.platform, 'windows-x86_64')
assert.equal(durableReceipt.provenance.paths_redacted, true)
assert.equal(durableReceipt.provenance.hostname_redacted, true)

const durableLoaded = durableReceipt.steps
  .find((step) => step.name === 'loaded_health').evidence
const durableRaw = durableReceipt.steps
  .find((step) => step.name === 'raw_first_forward').evidence
const durableGpuSteps = durableReceipt.steps
  .filter((step) => step.name === 'baseline_gpu' || step.name === 'final_gpu')
assert.equal(durableReceipt.runtime_contract.requests.raw_first_forward.max_tokens, 1)
assert.equal(durableReceipt.runtime_contract.requests.raw_first_forward.stream, false)
assert.equal(durableReceipt.runtime_contract.requests.camelid_receipt_requested, false)
assert.equal(durableRaw.choice_count, 1)
assert.equal(durableRaw.generated_token_ids.length, 1)
assert.equal(durableRaw.timings.weight_cache_hit, false)
assert.equal(durableRaw.timings.prompt_cache_hit, false)
assert.equal(durableRaw.timings.first_token_evaluated, true)
assert.equal(durableRaw.camelid_receipt_present, false)
for (const phase of durableRaw.memory_phases) {
  assert.ok(phase.forward_passes > 0)
  assert.equal(phase.materialization.has_lazy_q8_0_file_backing, true)
  assert.ok(phase.materialization.q8_0_file_backed_tensor_count > 0)
  assert.ok(phase.materialization.q8_0_file_backed_storage_bytes > 0)
  assert.equal(phase.materialization.has_q8_0_f32_materialization, false)
  assert.equal(phase.materialization.q8_0_f32_materialized_tensor_count, 0)
  assert.equal(phase.materialization.q8_0_f32_materialized_bytes, 0)
  assert.equal(phase.materialization.has_retained_q8_0_blocks, false)
  assert.equal(phase.materialization.q8_0_retained_block_tensor_count, 0)
  assert.equal(phase.materialization.q8_0_retained_block_bytes, 0)
  assert.ok(phase.q8_file_reads.read_calls > 0)
  assert.ok(phase.q8_file_reads.read_bytes > 0)
  for (const [key, value] of Object.entries(phase.q8_file_reads)) {
    if (key.startsWith('cache_')) assert.equal(value, 0)
  }
}
assert.equal(durableLoaded.execution_plan.selected_backend, 'cpu_reference')
assert.equal(durableLoaded.execution_plan.cuda_resident_active, false)
assert.equal(durableLoaded.execution_plan.selected_q8_path, 'safe_dense_or_q8_cpu')
for (const { evidence } of durableGpuSteps) {
  assert.equal(evidence.enabled, false)
  assert.equal(evidence.run_count, 0)
  assert.equal(evidence.device_redacted, true)
}
assert.equal(durableReceipt.resource_observations.thresholds_tripped, false)
assert.ok(durableReceipt.resource_observations.monitor_samples > 0)
assert.ok(durableReceipt.resource_observations.preflight_available_physical_bytes
  >= LIMITS.preflight_physical_bytes)
assert.ok(durableReceipt.resource_observations.preflight_available_disk_bytes
  >= LIMITS.preflight_disk_bytes)
assert.ok(durableReceipt.resource_observations.minimum_available_physical_bytes
  >= LIMITS.low_memory_abort_bytes)
assert.ok(durableReceipt.resource_observations.peak_child_working_set_bytes
  < LIMITS.child_working_set_abort_bytes)
assert.equal(durableReceipt.isolation.startup_warmup_markers.raw_output_persisted, false)
assert.equal(durableRaw.generated_text.redacted, true)
const durableReceiptText = durableReceiptBytes.toString('utf8')
assert.doesNotMatch(durableReceiptText, /[A-Za-z]:[\\/]/)
assert.doesNotMatch(durableReceiptText, /target[\\/]model-qualification/i)
assert.deepEqual(durableReceipt.gate_decision.authorized_roster_scope, ['gates.load_smoke'])
assert.equal(durableReceipt.gate_decision.load_smoke, 'pass')
assert.equal(durableReceipt.gate_decision.support_claim, false)
assert.equal(durableReceipt.gate_decision.existing_parity_gate, 'fail_unchanged')
assert.equal(durableReceipt.gate_decision.other_gates_unchanged, true)
assert.equal(durableReceipt.gate_decision.disposition, EXACT_ROW.disposition)
assert.equal(durableReceipt.gate_decision.target_tier, EXACT_ROW.target_tier)
assert.deepEqual(STEP_CONTRACT.map(([name]) => name), [
  'baseline_health', 'baseline_gpu', 'load', 'verify_identity',
  'loaded_health', 'raw_first_forward', 'final_health', 'final_gpu',
])
assert.equal(STEP_CONTRACT.some(([name]) => name === 'props' || name.includes('chat')), false)
assert.ok(DOES_NOT_PROVE.some((claim) => claim.includes('parity gate remains failed')))
assert.ok(DOES_NOT_PROVE.some((claim) => claim.includes('streaming SSE')))

const privateQwenError = new Qwen25LoadSmokeError('load_smoke_resource_abort')
assert.equal(privateQwenError.code, 'load_smoke_resource_abort')
assert.throws(() => { privateQwenError.code = 'load_smoke_http_failed' }, TypeError)
assert.equal(classifyQwen25LoadSmokeError(privateQwenError).error_code,
  'load_smoke_resource_abort')
const forgedPlainError = new Error('forged')
forgedPlainError.code = 'load_smoke_resource_abort'
assert.equal(classifyQwen25LoadSmokeError(forgedPlainError).error_code,
  'load_smoke_http_failed')
const sharedSmolError = new SmolLM3LoadSmokeError('load_smoke_resource_abort')
sharedSmolError.code = 'load_smoke_http_failed'
assert.equal(classifyQwen25LoadSmokeError(sharedSmolError).error_code,
  'load_smoke_resource_abort')

const serveArgs = buildServeArgs(modelsDir)
assert.deepEqual(serveArgs, [
  'serve', '--addr', SERVER_ADDR, '--models-dir', modelsDir, '--threads', '4',
  '--gpu', 'off', '--deterministic', '--kv-quant', 'f16', '--no-open',
  '--max-prompt-tokens', '1024', '--max-generation-tokens', '1',
])
assert.equal(serveArgs.includes('--model'), false)
assert.deepEqual(receiptCommand(), [
  '<camelid>', 'serve', '--addr', SERVER_ADDR, '--models-dir', '<empty-models-dir>',
  '--threads', '4', '--gpu', 'off', '--deterministic', '--kv-quant', 'f16',
  '--no-open', '--max-prompt-tokens', '1024', '--max-generation-tokens', '1',
])

const childEnv = buildChildEnv({
  PATH: 'safe-path',
  CAMELID_GPU: 'cuda',
  CAMELID_PROFILE: 'experimental',
  camelid_secret: 'must-be-cleared',
})
assert.equal(childEnv.PATH, 'safe-path')
assert.equal(childEnv.CAMELID_GPU, undefined)
assert.equal(childEnv.camelid_secret, undefined)
for (const [key, value] of Object.entries(SAFE_CAMELID_ENV)) assert.equal(childEnv[key], value)

assert.deepEqual([...parseArgs(['--help']).entries()], [['help', true]])
assert.deepEqual([...parseArgs([
  '--root=.', '--binary', 'camelid.exe', '--artifact=model.gguf', '--cwd', 'run',
  '--models-dir=models', '--binary-profile', 'release-fat-lto', '--out=receipt.json',
]).entries()], [
  ['root', '.'], ['binary', 'camelid.exe'], ['artifact', 'model.gguf'], ['cwd', 'run'],
  ['models-dir', 'models'], ['binary-profile', 'release-fat-lto'], ['out', 'receipt.json'],
])
for (const argv of [
  [], ['--help', '--out=x'], ['--artifact'], ['--bogus=x'],
  ['--artifact=x', '--artifact=y', '--cwd=x', '--models-dir=y'],
]) {
  assert.throws(() => parseArgs(argv), (error) => error?.code === 'load_smoke_options_invalid')
}

const provenance = {
  runtime_head: head,
  source_describe: sourceDescribe,
  tracked_files_clean: true,
  untracked_files_excluded: true,
  binary_profile: BINARY_PROFILE,
  binary_sha256: binarySha256,
  binary_version: binaryVersion,
}

const autoSelectRoots = [
  ['configured_models_dir', true],
  ['executable_models_dir', false],
  ['executable_dir', true],
  ['cwd_models_dir', false],
  ['cwd', true],
].map(([kind, exists]) => ({
  kind,
  exists,
  path_redacted: true,
  gguf_candidates: 0,
  default_preference_present: false,
}))

const fileStats = {
  isFile: () => true,
  isSymbolicLink: () => false,
  size: EXACT_ROW.source.size_bytes,
}

const preflightDeps = {
  platformInfo: () => ({ platform: 'win32', arch: 'x64' }),
  lstatImpl: async () => fileStats,
  statImpl: async () => fileStats,
  checkIgnoredImpl: async () => true,
  inspectProvenanceImpl: async () => provenance,
  assertAutoSelectRootsEmptyImpl: async () => autoSelectRoots,
  assertPortFreeImpl: async () => {},
  llamaServerRunningImpl: async () => false,
  freePhysicalBytesImpl: async () => 6 * 1024 ** 3,
  diskFreeBytesImpl: async () => 9 * 1024 ** 3,
}

const preflightOptions = { root, binary, artifact, cwd, modelsDir, binaryProfile: BINARY_PROFILE }
const preflight = await runPreflight(preflightOptions, preflightDeps)
assert.equal(preflight.platform, 'windows-x86_64')
assert.equal(preflight.artifact.hash_recomputed, false)
assert.equal(preflight.artifact.ignored, true)
assert.deepEqual(preflight.provenance, provenance)
assert.deepEqual(preflight.auto_select_roots, autoSelectRoots)

await expectCode(runPreflight(preflightOptions, {
  ...preflightDeps,
  lstatImpl: async () => ({ ...fileStats, isSymbolicLink: () => true }),
}), 'load_smoke_artifact_identity_mismatch')
await expectCode(runPreflight(preflightOptions, {
  ...preflightDeps,
  checkIgnoredImpl: async () => false,
}), 'load_smoke_artifact_not_ignored')
await expectCode(runPreflight(preflightOptions, {
  ...preflightDeps,
  freePhysicalBytesImpl: async () => LIMITS.preflight_physical_bytes - 1,
}), 'load_smoke_resources_low')
await expectCode(runPreflight(preflightOptions, {
  ...preflightDeps,
  inspectProvenanceImpl: async () => ({ ...provenance, binary_version: 'camelid stale' }),
}), 'load_smoke_binary_stale')
await expectCode(runPreflight(preflightOptions, {
  ...preflightDeps,
  assertAutoSelectRootsEmptyImpl: async () => {
    throw new Qwen25LoadSmokeError('load_smoke_auto_select_candidate_present')
  },
}), 'load_smoke_auto_select_candidate_present')
await expectCode(runPreflight(preflightOptions, {
  ...preflightDeps,
  llamaServerRunningImpl: async () => true,
}), 'load_smoke_llama_server_present')

assert.deepEqual(await inspectExactArtifactIdentity(artifact, {
  lstatImpl: async () => fileStats,
  statImpl: async () => fileStats,
  sha256FileImpl: async () => EXACT_ROW.source.sha256,
}), { size_bytes: EXACT_ROW.source.size_bytes, sha256: EXACT_ROW.source.sha256 })
await expectCode(inspectExactArtifactIdentity(artifact, {
  lstatImpl: async () => fileStats,
  statImpl: async () => fileStats,
  sha256FileImpl: async () => '0'.repeat(64),
}), 'load_smoke_artifact_identity_mismatch')

function q8Runtime() {
  return {
    policy: 'forced_lazy_file_backed_q8',
    lazy_q8_linear: true,
    retain_q8_blocks: false,
    file_cache_bytes: 0,
    note: 'raw response field intentionally normalized away',
  }
}

function executionPlan() {
  return {
    profile: 'safe',
    operating_system: 'windows',
    architecture: 'x86_64',
    model_family: 'qwen2',
    quant_type: 'Q8_0',
    exact_model_row: EXECUTION_PLAN_EXACT_MODEL_ROW,
    support_level: 'unknown_or_unvalidated',
    selected_backend: 'cpu_reference',
    selected_q8_path: 'safe_dense_or_q8_cpu',
    prefill_path: 'safe_cpu_prefill',
    prefill_runtime_policy: 'always_retained_reference_path',
    decode_path: 'safe_cpu_decode',
    diagnostics_status: 'operator-requested RSS timings enabled; performance claims disabled',
    fallback_path: 'safe_cpu_reference_path',
    cuda_resident_active: false,
  }
}

function healthBody(loaded, { final = false, plan = executionPlan() } = {}) {
  return {
    ok: true,
    engine: 'camelid',
    version: '0.6.1',
    build: sourceDescribe,
    loaded_now: loaded,
    generation_ready: loaded,
    vision_ready: false,
    active_model_id: loaded ? ROW_ID : null,
    backend: loaded ? 'llama' : 'none',
    model_family: loaded ? 'llama-family' : null,
    q8_runtime: q8Runtime(),
    execution_plan: loaded ? plan : null,
    engine_queue_depth: 0,
    engine_queued_tasks: 0,
    engine_active_task_id: null,
    engine_active_generated_tokens: 0,
    continuous_batch_slots: 1,
    executable: 'camelid.exe',
    listen_addr: SERVER_ADDR,
    engine_active_elapsed_seconds: final ? 0 : 0,
    engine_stalled_seconds: final ? 0 : 0,
  }
}

function materialization(overrides = {}) {
  return {
    tensor_count: 291,
    dense_f32_tensor_count: 37,
    dense_f32_bytes: 1024,
    q8_0_source_tensor_count: 254,
    q8_0_f32_materialized_tensor_count: 0,
    q8_0_f32_materialized_bytes: 0,
    q8_0_file_backed_tensor_count: 254,
    q8_0_file_backed_storage_bytes: 600_000_000,
    q8_0_file_backed_f32_bytes_avoided: 2_000_000_000,
    q8_0_file_backed_retained_block_bytes_if_enabled: 635_000_000,
    q8_0_file_handle_cached_count: 1,
    q8_0_retained_block_tensor_count: 0,
    q8_0_retained_block_bytes: 0,
    has_q8_0_f32_materialization: false,
    has_lazy_q8_0_file_backing: true,
    has_retained_q8_0_blocks: false,
    ...overrides,
  }
}

function q8Reads(overrides = {}) {
  return {
    read_calls: 9,
    read_bytes: 4096,
    cache_hits: 0,
    cache_hit_bytes: 0,
    cache_misses: 9,
    cache_miss_bytes: 4096,
    cache_inserts: 0,
    cache_insert_bytes: 0,
    cache_evictions: 0,
    cache_evicted_bytes: 0,
    cache_merges: 0,
    cache_merged_bytes: 0,
    cache_decoded_scale_hits: 0,
    cache_decoded_scale_hit_blocks: 0,
    cache_entries: 0,
    cache_bytes: 0,
    cache_capacity_bytes: 0,
    ...overrides,
  }
}

function memoryPhase(overrides = {}) {
  return {
    forward_passes: 1,
    materialization: materialization(),
    q8_file_reads: q8Reads(),
    peak_rss_kib: 420_000,
    ...overrides,
  }
}

function generationBody(overrides = {}) {
  const body = {
    model: ROW_ID,
    choices: [{ text: ' Paris', finish_reason: 'length' }],
    usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
    camelid: {
      prompt_token_ids: [785, 6722, 315, 9625, 374],
      generated_token_ids: [12095],
      top_logits: [{ token_id: 12095, logit: 18.25, probability: 0.62, rank: 1, selected: false }],
      step_top_logits: [],
      timings_ms: {
        weight_load: 1,
        weight_cache_hit: false,
        prompt_cache_hit: false,
        prompt_evaluation: {
          first_token_evaluated: true,
          prefill_memory: memoryPhase(),
          first_token_memory: memoryPhase(),
        },
        generation: { forward_total: 17 },
        memory: memoryPhase(),
      },
    },
  }
  return Object.assign(body, overrides)
}

assert.equal(normalizeHealth(healthBody(false), { loaded: false }).execution_plan, null)
const normalizedLoaded = normalizeHealth(healthBody(true), { loaded: true })
assert.equal(normalizedLoaded.execution_plan.model_family, 'qwen2')
assert.equal(normalizedLoaded.execution_plan.selected_q8_path, 'safe_dense_or_q8_cpu')
assert.equal(Object.hasOwn(normalizedLoaded.execution_plan, 'prefill_runtime_policy'), false)
const normalizedGeneration = normalizeGeneration(generationBody())
assert.equal(normalizedGeneration.generated_token_ids[0], 12095)
assert.equal(normalizedGeneration.timings.forward_total, 17)
assert.equal(normalizedGeneration.timings.weight_cache_hit, false)
assert.equal(normalizedGeneration.memory_phases.length, 3)
assert.equal(Object.hasOwn(normalizedGeneration.generated_text, 'text'), false)

for (const mutate of [
  (body) => { body.usage.completion_tokens = 2 },
  (body) => { body.usage.prompt_tokens = 4; body.usage.total_tokens = 5 },
  (body) => { body.camelid.generated_token_ids = [12095, 13] },
  (body) => { body.camelid.top_logits[0].logit = Number.NaN },
  (body) => { body.camelid.top_logits[0].token_id = 13 },
  (body) => { body.camelid.timings_ms.weight_cache_hit = true },
  (body) => { body.camelid.timings_ms.weight_load = 0 },
  (body) => { body.camelid.timings_ms.generation.forward_total = 0 },
  (body) => { body.camelid.timings_ms.prompt_evaluation.prefill_memory.forward_passes = 0 },
  (body) => {
    for (const phase of ['prefill_memory', 'first_token_memory']) {
      body.camelid.timings_ms.prompt_evaluation[phase].materialization.q8_0_f32_materialized_tensor_count = 1
    }
    body.camelid.timings_ms.memory.materialization.q8_0_f32_materialized_tensor_count = 1
  },
  (body) => {
    for (const phase of ['prefill_memory', 'first_token_memory']) {
      body.camelid.timings_ms.prompt_evaluation[phase].q8_file_reads.read_calls = 0
      body.camelid.timings_ms.prompt_evaluation[phase].q8_file_reads.read_bytes = 0
    }
    body.camelid.timings_ms.memory.q8_file_reads.read_calls = 0
    body.camelid.timings_ms.memory.q8_file_reads.read_bytes = 0
  },
]) {
  const body = generationBody()
  mutate(body)
  assert.throws(() => normalizeGeneration(body), (error) => error?.code === 'load_smoke_raw_invalid')
}

const wrongPlanHealth = healthBody(true)
wrongPlanHealth.execution_plan.model_family = 'qwen3'
assert.throws(() => normalizeHealth(wrongPlanHealth, { loaded: true }),
  (error) => error?.code === 'load_smoke_health_invalid')
const wrongGpuPlanHealth = healthBody(true)
wrongGpuPlanHealth.execution_plan.cuda_resident_active = true
assert.throws(() => normalizeHealth(wrongGpuPlanHealth, { loaded: true }),
  (error) => error?.code === 'load_smoke_health_invalid')
const wrongDiagnosticsHealth = healthBody(true)
wrongDiagnosticsHealth.execution_plan.diagnostics_status = 'standard diagnostics; RSS timings disabled by default'
assert.throws(() => normalizeHealth(wrongDiagnosticsHealth, { loaded: true }),
  (error) => error?.code === 'load_smoke_health_invalid')

function gpuBody(overrides = {}) {
  return { available: true, enabled: false, backend: 'cuda', run_count: 0, ...overrides }
}

function loadBody(overrides = {}) {
  return {
    data: {
      id: ROW_ID,
      path: null,
      status: { value: 'loaded' },
      camelid: { generation_ready: true, model_path_redacted: true },
    },
    camelid: {
      model_path_redacted: true,
      compatibility: 'partial_llama_server_models_load_local_path',
      scope: 'single_local_model_load_alias',
    },
    ...overrides,
  }
}

function verifyBody(overrides = {}) {
  return {
    model_id: ROW_ID,
    gguf_sha256: EXACT_ROW.source.sha256,
    eligible: false,
    profile_id: null,
    report: null,
    ...overrides,
  }
}

function makeLock(events) {
  let exited = false
  let held = true
  let exitResolve
  let closeResolve
  const exitedPromise = new Promise((resolvePromise) => { exitResolve = resolvePromise })
  const closedPromise = new Promise((resolvePromise) => { closeResolve = resolvePromise })
  const status = { error: false, code: 0, signal: null }
  return {
    acquired: true,
    exited: exitedPromise,
    closed: closedPromise,
    isExited: () => exited,
    exitStatus: () => exited ? status : null,
    assertHeld() {
      if (!held || exited) throw new Error('lock lost')
    },
    forceExit(code = 65) {
      if (exited) return
      exited = true
      held = false
      status.code = code
      events.push('lock-exited')
      exitResolve(status)
      closeResolve(status)
    },
    async release() {
      events.push('lock-release')
      if (!held || exited) throw new Error('not held')
      held = false
      exited = true
      exitResolve(status)
      closeResolve(status)
      return { observed: true, released_token_observed: true, exit_code: 0 }
    },
  }
}

function makeHandle(events, { warmup = false } = {}) {
  let exited = false
  let closed = false
  let exitResolve
  let closeResolve
  const exitedPromise = new Promise((resolvePromise) => { exitResolve = resolvePromise })
  const closedPromise = new Promise((resolvePromise) => { closeResolve = resolvePromise })
  return {
    pid: 4242,
    exited: exitedPromise,
    closed: closedPromise,
    kill: () => true,
    isExited: () => exited,
    isClosed: () => closed,
    exitStatus: () => exited ? { error: false, code: 0, signal: 'SIGTERM' } : null,
    logMarkers: () => ({
      warming_up_seen: warmup,
      generation_warmup_complete_seen: warmup,
      output_captured_only_for_markers: true,
      raw_output_persisted: false,
      observed_output_bytes: 128,
    }),
    terminate() {
      events.push('child-terminate')
      exited = true
      closed = true
      const status = { error: false, code: 0, signal: 'SIGTERM' }
      exitResolve(status)
      closeResolve(status)
    },
  }
}

function responseSequence(overrides = {}) {
  const responses = [
    healthBody(false),
    gpuBody(),
    loadBody(),
    verifyBody(),
    healthBody(true),
    generationBody(),
    healthBody(true, { final: true }),
    gpuBody(),
  ]
  for (const [index, body] of Object.entries(overrides)) responses[Number(index)] = body
  return responses
}

function makeHarness({
  responseOverrides = {},
  warmup = false,
  guardError = null,
  terminateFailure = false,
  lockLostAtRequest = null,
  postflightMutation = null,
} = {}) {
  const events = []
  const lock = makeLock(events)
  const handle = makeHandle(events, { warmup })
  const responses = responseSequence(responseOverrides)
  let requestIndex = 0
  let now = 1_000
  let guardStopped = false
  const preflightReceipt = {
    platform: 'windows-x86_64',
    artifact: {
      size_bytes: EXACT_ROW.source.size_bytes,
      expected_sha256: EXACT_ROW.source.sha256,
      hash_recomputed: false,
      ignored: true,
      path_redacted: true,
    },
    provenance,
    auto_select_roots: autoSelectRoots,
    available_physical_bytes: 6 * 1024 ** 3,
    available_disk_bytes: 9 * 1024 ** 3,
    qualification_port_unbound: true,
    llama_server_absent: true,
  }
  const postflight = {
    provenance,
    auto_select_roots: autoSelectRoots,
    artifact: {
      size_bytes: EXACT_ROW.source.size_bytes,
      sha256: EXACT_ROW.source.sha256,
      verified_after_generation: true,
      path_redacted: true,
      ...(postflightMutation || {}),
    },
  }
  const deps = {
    inheritedEnv: { PATH: 'safe', CAMELID_GPU: 'cuda' },
    preflightImpl: async () => {
      events.push('preflight')
      return preflightReceipt
    },
    acquireArtifactLockImpl: async () => {
      events.push('lock-acquire')
      return lock
    },
    preloadArtifactIdentityImpl: async () => {
      events.push('preload-hash')
      return { size_bytes: EXACT_ROW.source.size_bytes, sha256: EXACT_ROW.source.sha256 }
    },
    startProcessImpl: async () => {
      events.push('child-start')
      return handle
    },
    createResourceGuardImpl: async () => ({
      signal: new AbortController().signal,
      throwIfAborted() {
        if (guardError) throw guardError
      },
      async stop() {
        events.push('guard-stop')
        guardStopped = true
        return { observed: true }
      },
      summary: () => ({
        samples: 8,
        minimum_available_physical_bytes: 5 * 1024 ** 3,
        peak_child_working_set_bytes: 700 * 1024 ** 2,
        thresholds_tripped: false,
      }),
    }),
    httpJsonImpl: async ({ method, endpoint, body }) => {
      events.push(`request:${requestIndex}:${method}:${endpoint}`)
      assert.equal(guardStopped, false)
      if (lockLostAtRequest === requestIndex) lock.forceExit()
      if (endpoint === '/models/load') {
        assert.deepEqual(body, { path: artifact, id: ROW_ID })
      } else if (endpoint === '/v1/completions') {
        assert.deepEqual(body, RAW_REQUEST)
      } else {
        assert.equal(body, undefined)
      }
      const response = { status: 200, body: responses[requestIndex] }
      requestIndex += 1
      return response
    },
    terminateChildImpl: async () => {
      if (terminateFailure) throw new Error('termination failed')
      handle.terminate()
      return { observed: true, already_exited: false, termination_requested: true }
    },
    postflightImpl: async () => {
      events.push('postflight')
      return postflight
    },
    sleepImpl: async () => {},
    yieldImpl: async () => {},
    nowMsImpl: () => { now += 7; return now },
    nowIsoImpl: () => '2026-08-11T00:00:00.000Z',
  }
  return { deps, events, lock, handle, requestCount: () => requestIndex }
}

const runOptions = { root, binary, artifact, cwd, modelsDir, binaryProfile: BINARY_PROFILE }
const malformedLockHarness = makeHarness()
malformedLockHarness.deps.acquireArtifactLockImpl = async () => {
  malformedLockHarness.events.push('malformed-lock-acquire')
  return {
    acquired: true,
    // Deliberately missing exited/closed/assertHeld/isExited/exitStatus.
    async release() {
      malformedLockHarness.events.push('malformed-lock-release-observed')
      return { observed: true, released_token_observed: true, exit_code: 0 }
    },
  }
}
await expectCode(runQwen25LoadSmoke(runOptions, malformedLockHarness.deps),
  'load_smoke_artifact_lock_failed')
assert.deepEqual(malformedLockHarness.events, [
  'preflight', 'malformed-lock-acquire', 'malformed-lock-release-observed',
])

const successHarness = makeHarness()
const receipt = await runQwen25LoadSmoke(runOptions, successHarness.deps)
assert.equal(successHarness.requestCount(), STEP_CONTRACT.length)
assert.deepEqual(validateLoadSmokeReceipt(receipt), [])
assert.equal(receipt.schema, RECEIPT_SCHEMA)
assert.equal(receipt.row.id, ROW_ID)
assert.equal(receipt.row.disposition, 'active_validation')
assert.equal(receipt.provenance.artifact.sha256, EXACT_ROW.source.sha256)
assert.equal(receipt.provenance.artifact.mutation_guard.held_through_post_generation_hash, true)
assert.equal(receipt.runtime_contract.requests.raw_first_forward.max_tokens, 1)
assert.equal(Object.hasOwn(receipt.runtime_contract.requests, 'chat_followup'), false)
assert.equal(receipt.steps.length, 8)
assert.deepEqual(receipt.steps.map((step) => step.name), STEP_CONTRACT.map(([name]) => name))
assert.equal(receipt.steps.find((step) => step.name === 'raw_first_forward')
  .evidence.timings.weight_cache_hit, false)
assert.equal(receipt.gate_decision.disposition, 'active_validation')
assert.equal(receipt.gate_decision.existing_parity_gate, 'fail_unchanged')
assert.equal(receipt.gate_decision.support_claim, false)
assert.deepEqual(receipt.gate_decision.authorized_roster_scope, ['gates.load_smoke'])
assert.deepEqual(successHarness.events.slice(0, 4), [
  'preflight', 'lock-acquire', 'preload-hash', 'child-start',
])
assert.ok(successHarness.events.indexOf('child-terminate') < successHarness.events.indexOf('postflight'))
assert.ok(successHarness.events.indexOf('postflight') < successHarness.events.indexOf('lock-release'))

const receiptWithoutId = deepClone(receipt)
delete receiptWithoutId.receipt_id
assert.ok(validateLoadSmokeReceipt(receiptWithoutId).length > 0)
const unsealedTamper = deepClone(receipt)
unsealedTamper.gate_decision.support_claim = true
assert.ok(validateLoadSmokeReceipt(unsealedTamper).some((error) => error.includes('seal')))
const promoted = deepClone(receipt)
promoted.gate_decision.support_claim = true
promoted.gate_decision.disposition = 'promoted'
assert.ok(validateLoadSmokeReceipt(reseal(promoted))
  .some((error) => error.includes('load-smoke-only')))
const parityWaived = deepClone(receipt)
parityWaived.gate_decision.existing_parity_gate = 'pass'
assert.ok(validateLoadSmokeReceipt(reseal(parityWaived))
  .some((error) => error.includes('parity failed')))
const extraStep = deepClone(receipt)
extraStep.steps.push(deepClone(extraStep.steps.at(-1)))
assert.ok(validateLoadSmokeReceipt(reseal(extraStep))
  .some((error) => error.includes('step count')))
const reordered = deepClone(receipt)
;[reordered.steps[0], reordered.steps[1]] = [reordered.steps[1], reordered.steps[0]]
assert.ok(validateLoadSmokeReceipt(reseal(reordered))
  .some((error) => error.includes('exact sequence')))
const retained = deepClone(receipt)
retained.steps.find((step) => step.name === 'raw_first_forward')
  .evidence.memory_phases[0].materialization.q8_0_retained_block_tensor_count = 1
assert.ok(validateLoadSmokeReceipt(reseal(retained))
  .some((error) => error.includes('lazy-Q8')))
const cacheHit = deepClone(receipt)
cacheHit.steps.find((step) => step.name === 'raw_first_forward')
  .evidence.timings.weight_cache_hit = true
assert.ok(validateLoadSmokeReceipt(reseal(cacheHit))
  .some((error) => error.includes('uncached')))
const promptCountMismatch = deepClone(receipt)
const promptCountEvidence = promptCountMismatch.steps
  .find((step) => step.name === 'raw_first_forward').evidence
promptCountEvidence.usage.prompt_tokens -= 1
promptCountEvidence.usage.total_tokens -= 1
assert.ok(validateLoadSmokeReceipt(reseal(promptCountMismatch))
  .some((error) => error.includes('one-token, finite, uncached')))
const diagnosticsTamper = deepClone(receipt)
diagnosticsTamper.steps.find((step) => step.name === 'loaded_health')
  .evidence.execution_plan.diagnostics_status = 'RSS maybe enabled'
assert.ok(validateLoadSmokeReceipt(reseal(diagnosticsTamper))
  .some((error) => error.includes('Safe CPU plan')))
const zeroForwardPasses = deepClone(receipt)
zeroForwardPasses.steps.find((step) => step.name === 'raw_first_forward')
  .evidence.memory_phases[0].forward_passes = 0
assert.ok(validateLoadSmokeReceipt(reseal(zeroForwardPasses))
  .some((error) => error.includes('positive forward passes')))
const absolutePathLeak = deepClone(receipt)
absolutePathLeak.provenance.binary.raw_log = 'C:\\private\\camelid.log'
const pathLeakErrors = validateLoadSmokeReceipt(reseal(absolutePathLeak))
assert.ok(pathLeakErrors.some((error) => error.includes('keys must be exact')))
assert.ok(pathLeakErrors.some((error) => error.includes('forbidden key')))
assert.ok(pathLeakErrors.some((error) => error.includes('absolute local path')))
const credentialLeak = deepClone(receipt)
credentialLeak.does_not_prove[0] = 'authorization: Bearer very-secret-value'
assert.ok(validateLoadSmokeReceipt(reseal(credentialLeak))
  .some((error) => error.includes('credential-like')))
const githubCredentialLeak = deepClone(receipt)
githubCredentialLeak.does_not_prove[0] = 'opaque value ghp_0123456789abcdefghijklmnop'
assert.ok(validateLoadSmokeReceipt(reseal(githubCredentialLeak))
  .some((error) => error.includes('credential-like')))
const extraTopLevel = deepClone(receipt)
extraTopLevel.hostname = 'private-host'
const extraErrors = validateLoadSmokeReceipt(reseal(extraTopLevel))
assert.ok(extraErrors.some((error) => error.includes('receipt keys must be exact')))
assert.ok(extraErrors.some((error) => error.includes('forbidden key')))

let accessorInvocations = 0
const accessorReceipt = deepClone(receipt)
Object.defineProperty(accessorReceipt.provenance.binary, 'version', {
  enumerable: true,
  configurable: true,
  get() {
    accessorInvocations += 1
    throw new Error('this accessor represents a potentially hanging getter')
  },
})
const accessorErrors = validateLoadSmokeReceipt(accessorReceipt)
assert.equal(accessorInvocations, 0)
assert.ok(accessorErrors.some((error) => error.includes('uses an accessor')))

let prototypeAccessorInvocations = 0
const prototypeAccessorReceipt = deepClone(receipt)
delete prototypeAccessorReceipt.schema
const hostilePrototype = Object.create(Object.prototype, {
  schema: {
    enumerable: true,
    get() {
      prototypeAccessorInvocations += 1
      throw new Error('prototype getter must never execute')
    },
  },
})
Object.setPrototypeOf(prototypeAccessorReceipt, hostilePrototype)
const prototypeAccessorErrors = validateLoadSmokeReceipt(prototypeAccessorReceipt)
assert.equal(prototypeAccessorInvocations, 0)
assert.ok(prototypeAccessorErrors.some((error) => error.includes('unexpected prototype')))

let transparentProxyTrapInvocations = 0
const transparentProxyTarget = deepClone(receipt)
const transparentProxyReceipt = new Proxy(transparentProxyTarget, {
  get(target, key, receiver) {
    transparentProxyTrapInvocations += 1
    return Reflect.get(target, key, receiver)
  },
  getOwnPropertyDescriptor(target, key) {
    transparentProxyTrapInvocations += 1
    return Reflect.getOwnPropertyDescriptor(target, key)
  },
  getPrototypeOf(target) {
    transparentProxyTrapInvocations += 1
    return Reflect.getPrototypeOf(target)
  },
  ownKeys(target) {
    transparentProxyTrapInvocations += 1
    return Reflect.ownKeys(target)
  },
})
const transparentProxyErrors = validateLoadSmokeReceipt(transparentProxyReceipt)
assert.equal(transparentProxyTrapInvocations, 0)
assert.ok(transparentProxyErrors.some((error) => error.includes('is a Proxy')))

let throwingProxyTrapInvocations = 0
const throwingProxy = new Proxy({ version: binaryVersion }, {
  get() {
    throwingProxyTrapInvocations += 1
    throw new Error('proxy get trap must never execute')
  },
  getOwnPropertyDescriptor() {
    throwingProxyTrapInvocations += 1
    throw new Error('proxy descriptor trap must never execute')
  },
  getPrototypeOf() {
    throwingProxyTrapInvocations += 1
    throw new Error('proxy prototype trap must never execute')
  },
  ownKeys() {
    throwingProxyTrapInvocations += 1
    throw new Error('proxy ownKeys trap must never execute')
  },
})
const throwingProxyReceipt = deepClone(receipt)
throwingProxyReceipt.provenance.binary = throwingProxy
const throwingProxyErrors = validateLoadSmokeReceipt(throwingProxyReceipt)
assert.equal(throwingProxyTrapInvocations, 0)
assert.ok(throwingProxyErrors.some((error) => error.includes('is a Proxy')))

const writeEvents = []
await writeReceiptAtomic(resolve('receipts', 'qwen.json'), receipt, {
  mkdirImpl: async (_path, options) => { writeEvents.push(['mkdir', options]) },
  writeFileImpl: async (path, contents, options) => {
    writeEvents.push(['write', path, contents, options])
  },
  renameImpl: async (from, to) => { writeEvents.push(['rename', from, to]) },
  rmImpl: async () => { throw new Error('must not remove on success') },
})
assert.equal(writeEvents[0][0], 'mkdir')
assert.equal(writeEvents[1][0], 'write')
assert.equal(writeEvents[1][3].flag, 'wx')
assert.equal(JSON.parse(writeEvents[1][2]).receipt_id, receipt.receipt_id)
assert.equal(writeEvents[2][0], 'rename')

let invalidWriteCalled = false
await expectCode(writeReceiptAtomic(resolve('receipts', 'bad.json'), promoted, {
  mkdirImpl: async () => { invalidWriteCalled = true },
}), 'load_smoke_receipt_invalid')
assert.equal(invalidWriteCalled, false)

let removedTemporary = false
await expectCode(writeReceiptAtomic(resolve('receipts', 'failed.json'), receipt, {
  mkdirImpl: async () => {},
  writeFileImpl: async () => { throw new Error('disk full') },
  renameImpl: async () => {},
  rmImpl: async () => { removedTemporary = true },
}), 'load_smoke_output_failed')
assert.equal(removedTemporary, true)

const wrongPlanHarness = makeHarness({
  responseOverrides: { 4: healthBody(true, { plan: { ...executionPlan(), model_family: 'qwen3' } }) },
})
await expectCode(runQwen25LoadSmoke(runOptions, wrongPlanHarness.deps), 'load_smoke_health_invalid')
assert.ok(wrongPlanHarness.events.includes('child-terminate'))
assert.ok(wrongPlanHarness.events.includes('lock-release'))

const gpuRunHarness = makeHarness({ responseOverrides: { 7: gpuBody({ run_count: 1 }) } })
await expectCode(runQwen25LoadSmoke(runOptions, gpuRunHarness.deps), 'load_smoke_gpu_invalid')
assert.ok(gpuRunHarness.events.includes('child-terminate'))
assert.ok(gpuRunHarness.events.includes('lock-release'))

const badRaw = generationBody()
badRaw.camelid.timings_ms.weight_cache_hit = true
// A response-level cached first forward is already fail-closed by the normalizer.
assert.throws(() => normalizeGeneration(badRaw), (error) => error?.code === 'load_smoke_raw_invalid')

const warmupHarness = makeHarness({ warmup: true })
await expectCode(runQwen25LoadSmoke(runOptions, warmupHarness.deps), 'load_smoke_warmup_detected')
assert.ok(warmupHarness.events.includes('child-terminate'))
assert.ok(warmupHarness.events.includes('lock-release'))

const resourceAbortHarness = makeHarness({
  guardError: new Qwen25LoadSmokeError('load_smoke_resource_abort'),
})
await expectCode(runQwen25LoadSmoke(runOptions, resourceAbortHarness.deps), 'load_smoke_resource_abort')
assert.ok(resourceAbortHarness.events.includes('child-terminate'))
assert.ok(resourceAbortHarness.events.includes('lock-release'))

const terminationHarness = makeHarness({ terminateFailure: true })
await expectCode(runQwen25LoadSmoke(runOptions, terminationHarness.deps), 'load_smoke_termination_failed')
assert.ok(terminationHarness.events.includes('lock-release'))

const mutationHarness = makeHarness({ postflightMutation: { sha256: '0'.repeat(64) } })
await expectCode(runQwen25LoadSmoke(runOptions, mutationHarness.deps),
  'load_smoke_artifact_identity_mismatch')
assert.ok(mutationHarness.events.includes('lock-release'))

const lockLostHarness = makeHarness({ lockLostAtRequest: 3 })
await expectCode(runQwen25LoadSmoke(runOptions, lockLostHarness.deps), 'load_smoke_artifact_lock_lost')
assert.ok(lockLostHarness.events.includes('child-terminate'))
assert.equal(lockLostHarness.events.includes('postflight'), false)

assert.deepEqual(classifyQwen25LoadSmokeError(new Error('unknown')), {
  status: 'blocked',
  error_code: 'load_smoke_http_failed',
  reason: 'an isolated loopback request failed or timed out',
})

console.log('qwen2.5 guarded load-smoke foundation tests passed')
