const ID = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/
const SHA256 = /^[0-9a-f]{64}$/
const GIT_SHA = /^[0-9a-f]{40}$/

const PLAN_MODES = new Set(['pr', 'main_trend', 'release', 'external_comparator', 'ablation'])
const NETWORK_POLICIES = new Set(['deny', 'loopback_only', 'explicit'])
const TRUST_TIERS = new Set(['hosted_validation', 'trusted_benchmark', 'release', 'local_exploratory'])
const SAMPLE_VALIDITIES = new Set([
  'valid',
  'invalid_parse',
  'invalid_hash',
  'invalid_backend',
  'invalid_environment',
  'invalid_concurrent_load',
  'invalid_timeout',
  'invalid_cleanup',
  'invalid_correctness',
  'unsupported',
])
const PROCESS_STATES = new Set([
  'not_started',
  'running',
  'exited',
  'timed_out',
  'cancelled',
  'killed',
  'spawn_failed',
  'cleanup_failed',
])
const RUNTIME_VERDICTS = new Set([
  'IMPROVEMENT',
  'NO_MATERIAL_CHANGE',
  'REGRESSION',
  'INCONCLUSIVE_NOISE',
  'INVALID_CORRECTNESS',
  'INVALID_BACKEND',
  'INVALID_INFRASTRUCTURE',
  'UNSUPPORTED',
])
const BACKENDS = new Set(['cpu_deterministic'])
const BACKEND_ASSERTIONS = new Set(['deterministic_no_offload'])

export class ContractError extends Error {
  constructor(contract, issues) {
    super(`${contract} contract failed: ${issues.join('; ')}`)
    this.name = 'ContractError'
    this.contract = contract
    this.issues = issues
  }
}

export function validateCampaign(value) {
  const issues = []
  object(value, '$', issues)
  if (issues.length > 0) fail('campaign/v1', issues)
  exactKeys(value, '$', [
    'schema', 'campaign_id', 'mode', 'created_utc', 'controller',
    'repository_root', 'source_arms', 'models', 'workloads', 'resources', 'security',
  ], [], issues)
  equal(value.schema, 'camelid.benchmark.campaign/v1', '$.schema', issues)
  id(value.campaign_id, '$.campaign_id', issues)
  member(value.mode, PLAN_MODES, '$.mode', issues)
  rfc3339(value.created_utc, '$.created_utc', issues)
  nonEmpty(value.repository_root, '$.repository_root', issues)
  nestedObject(value.controller, '$.controller', ['source_manifest_sha256', 'version'], issues, (controller) => {
    sha256(controller.source_manifest_sha256, '$.controller.source_manifest_sha256', issues)
    nonEmpty(controller.version, '$.controller.version', issues)
  })

  nonEmptyArray(value.source_arms, '$.source_arms', issues)
  const armIds = new Set()
  const campaignTargetDirs = new Set()
  if (Array.isArray(value.source_arms)) {
    value.source_arms.forEach((arm, index) => {
      const path = `$.source_arms[${index}]`
      nestedObject(arm, path, [
        'id', 'source_dir', 'expected_git_sha', 'cargo_path', 'toolchain',
        'features', 'target_dir',
      ], issues, (item) => {
        id(item.id, `${path}.id`, issues)
        unique(item.id, armIds, `${path}.id`, issues)
        nonEmpty(item.source_dir, `${path}.source_dir`, issues)
        gitSha(item.expected_git_sha, `${path}.expected_git_sha`, issues)
        nonEmpty(item.cargo_path, `${path}.cargo_path`, issues)
        nonEmpty(item.toolchain, `${path}.toolchain`, issues)
        stringArray(item.features, `${path}.features`, issues, true)
        nonEmpty(item.target_dir, `${path}.target_dir`, issues)
        unique(item.target_dir, campaignTargetDirs, `${path}.target_dir`, issues)
      })
    })
  }
  validatePhase1Arms(armIds, issues)

  nonEmptyArray(value.models, '$.models', issues)
  const modelIds = new Set()
  if (Array.isArray(value.models)) {
    value.models.forEach((model, index) => {
      const path = `$.models[${index}]`
      nestedObject(model, path, ['id', 'artifact_path', 'expected_sha256', 'quantization'], issues, (item) => {
        id(item.id, `${path}.id`, issues)
        unique(item.id, modelIds, `${path}.id`, issues)
        nonEmpty(item.artifact_path, `${path}.artifact_path`, issues)
        sha256(item.expected_sha256, `${path}.expected_sha256`, issues)
        nonEmpty(item.quantization, `${path}.quantization`, issues)
      })
    })
  }

  nonEmptyArray(value.workloads, '$.workloads', issues)
  const workloadIds = new Set()
  if (Array.isArray(value.workloads)) {
    value.workloads.forEach((workload, index) => {
      const path = `$.workloads[${index}]`
      nestedObject(workload, path, [
        'id', 'adapter', 'model_id', 'prompt_file', 'max_tokens', 'warmup',
        'deterministic', 'threads', 'backend', 'primary_metrics', 'schedule',
        'repetitions', 'timeout_ms',
      ], issues, (item) => {
        id(item.id, `${path}.id`, issues)
        unique(item.id, workloadIds, `${path}.id`, issues)
        equal(item.adapter, 'runtime-camelid', `${path}.adapter`, issues)
        id(item.model_id, `${path}.model_id`, issues)
        if (!modelIds.has(item.model_id)) issues.push(`${path}.model_id references unknown model ${JSON.stringify(item.model_id)}`)
        nonEmpty(item.prompt_file, `${path}.prompt_file`, issues)
        positiveInteger(item.max_tokens, `${path}.max_tokens`, issues)
        boolean(item.warmup, `${path}.warmup`, issues)
        equal(item.deterministic, true, `${path}.deterministic`, issues)
        if (item.threads !== null) positiveInteger(item.threads, `${path}.threads`, issues)
        validateBackendContract(item.backend, `${path}.backend`, issues)
        nonEmptyStringArray(item.primary_metrics, `${path}.primary_metrics`, issues, true)
        equal(item.schedule, 'balanced_rotation', `${path}.schedule`, issues)
        positiveInteger(item.repetitions, `${path}.repetitions`, issues)
        positiveInteger(item.timeout_ms, `${path}.timeout_ms`, issues)
      })
    })
  }
  validateResources(value.resources, issues)
  validateSecurity(value.security, issues)
  if (issues.length > 0) fail('campaign/v1', issues)
  return value
}

export function validatePlan(value) {
  const issues = []
  object(value, '$', issues)
  if (issues.length > 0) fail('plan/v1', issues)
  exactKeys(value, '$', [
    'schema', 'campaign_id', 'mode', 'created_utc', 'controller', 'repository_root',
    'source_arms', 'models', 'workloads', 'resources', 'security',
  ], [], issues)
  equal(value.schema, 'camelid.benchmark.plan/v1', '$.schema', issues)
  id(value.campaign_id, '$.campaign_id', issues)
  member(value.mode, PLAN_MODES, '$.mode', issues)
  rfc3339(value.created_utc, '$.created_utc', issues)
  nonEmpty(value.repository_root, '$.repository_root', issues)

  nestedObject(value.controller, '$.controller', ['source_manifest_sha256', 'version'], issues, (controller) => {
    sha256(controller.source_manifest_sha256, '$.controller.source_manifest_sha256', issues)
    nonEmpty(controller.version, '$.controller.version', issues)
  })

  nonEmptyArray(value.source_arms, '$.source_arms', issues)
  const armIds = new Set()
  const planTargetDirs = new Set()
  if (Array.isArray(value.source_arms)) {
    value.source_arms.forEach((arm, index) => {
      const path = `$.source_arms[${index}]`
      nestedObject(arm, path, ['id', 'source_dir', 'git_sha', 'tree_dirty', 'cargo_lock_sha256', 'build'], issues, (item) => {
        id(item.id, `${path}.id`, issues)
        unique(item.id, armIds, `${path}.id`, issues)
        nonEmpty(item.source_dir, `${path}.source_dir`, issues)
        gitSha(item.git_sha, `${path}.git_sha`, issues)
        boolean(item.tree_dirty, `${path}.tree_dirty`, issues)
        sha256(item.cargo_lock_sha256, `${path}.cargo_lock_sha256`, issues)
        nestedObject(item.build, `${path}.build`, ['cargo_path', 'toolchain', 'profile', 'features', 'target_dir', 'binary_path'], issues, (build) => {
          nonEmpty(build.cargo_path, `${path}.build.cargo_path`, issues)
          nonEmpty(build.toolchain, `${path}.build.toolchain`, issues)
          equal(build.profile, 'release', `${path}.build.profile`, issues)
          stringArray(build.features, `${path}.build.features`, issues, true)
          nonEmpty(build.target_dir, `${path}.build.target_dir`, issues)
          unique(build.target_dir, planTargetDirs, `${path}.build.target_dir`, issues)
          nonEmpty(build.binary_path, `${path}.build.binary_path`, issues)
        })
      })
    })
  }
  validatePhase1Arms(armIds, issues)

  nonEmptyArray(value.models, '$.models', issues)
  const modelIds = new Set()
  if (Array.isArray(value.models)) {
    value.models.forEach((model, index) => {
      const path = `$.models[${index}]`
      nestedObject(model, path, ['id', 'artifact_path', 'artifact_sha256', 'size_bytes', 'quantization'], issues, (item) => {
        id(item.id, `${path}.id`, issues)
        unique(item.id, modelIds, `${path}.id`, issues)
        nonEmpty(item.artifact_path, `${path}.artifact_path`, issues)
        sha256(item.artifact_sha256, `${path}.artifact_sha256`, issues)
        positiveInteger(item.size_bytes, `${path}.size_bytes`, issues)
        nonEmpty(item.quantization, `${path}.quantization`, issues)
      })
    })
  }

  nonEmptyArray(value.workloads, '$.workloads', issues)
  const workloadIds = new Set()
  if (Array.isArray(value.workloads)) {
    value.workloads.forEach((workload, index) => {
      const path = `$.workloads[${index}]`
      nestedObject(workload, path, [
        'id', 'adapter', 'model_id', 'prompt_file', 'prompt_sha256', 'prompt_policy', 'max_tokens',
        'warmup', 'deterministic', 'threads', 'backend', 'primary_metrics',
        'order', 'repetitions', 'timeout_ms',
      ], issues, (item) => {
        id(item.id, `${path}.id`, issues)
        unique(item.id, workloadIds, `${path}.id`, issues)
        equal(item.adapter, 'runtime-camelid', `${path}.adapter`, issues)
        id(item.model_id, `${path}.model_id`, issues)
        if (!modelIds.has(item.model_id)) issues.push(`${path}.model_id references unknown model ${JSON.stringify(item.model_id)}`)
        nonEmpty(item.prompt_file, `${path}.prompt_file`, issues)
        sha256(item.prompt_sha256, `${path}.prompt_sha256`, issues)
        equal(item.prompt_policy, 'front_block_marker_v1', `${path}.prompt_policy`, issues)
        positiveInteger(item.max_tokens, `${path}.max_tokens`, issues)
        boolean(item.warmup, `${path}.warmup`, issues)
        equal(item.deterministic, true, `${path}.deterministic`, issues)
        if (item.threads !== null) positiveInteger(item.threads, `${path}.threads`, issues)
        validateBackendContract(item.backend, `${path}.backend`, issues)
        nonEmptyStringArray(item.primary_metrics, `${path}.primary_metrics`, issues, true)
        validateBalancedOrder(item.order, item.repetitions, armIds, `${path}.order`, issues)
        positiveInteger(item.repetitions, `${path}.repetitions`, issues)
        positiveInteger(item.timeout_ms, `${path}.timeout_ms`, issues)
      })
    })
  }

  validateResources(value.resources, issues)
  validateSecurity(value.security, issues)

  if (value.mode === 'release' && Array.isArray(value.source_arms)) {
    value.source_arms.forEach((arm, index) => {
      if (arm?.tree_dirty === true) issues.push(`$.source_arms[${index}].tree_dirty must be false in release mode`)
    })
  }
  if (issues.length > 0) fail('plan/v1', issues)
  return value
}

export function validateRuntimeSample(value) {
  const issues = []
  object(value, '$', issues)
  if (issues.length > 0) fail('runtime-sample/v1', issues)
  exactKeys(value, '$', [
    'schema', 'campaign_id', 'workload_id', 'arm_id', 'process_block',
    'request_index', 'validity', 'invalid_reason', 'identity', 'backend',
    'metrics', 'metrics_unavailable_reason', 'correctness', 'process',
  ], [], issues)
  equal(value.schema, 'camelid.benchmark.runtime-sample/v1', '$.schema', issues)
  id(value.campaign_id, '$.campaign_id', issues)
  id(value.workload_id, '$.workload_id', issues)
  id(value.arm_id, '$.arm_id', issues)
  nonNegativeInteger(value.process_block, '$.process_block', issues)
  nonNegativeInteger(value.request_index, '$.request_index', issues)
  member(value.validity, SAMPLE_VALIDITIES, '$.validity', issues)
  if (value.validity === 'valid') nullableReason(value.invalid_reason, '$.invalid_reason', issues, false)
  else nullableReason(value.invalid_reason, '$.invalid_reason', issues, true)

  nestedObject(value.identity, '$.identity', ['source_sha', 'binary_sha256', 'model_sha256', 'prompt_sha256'], issues, (identity) => {
    gitSha(identity.source_sha, '$.identity.source_sha', issues)
    sha256(identity.binary_sha256, '$.identity.binary_sha256', issues)
    sha256(identity.model_sha256, '$.identity.model_sha256', issues)
    sha256(identity.prompt_sha256, '$.identity.prompt_sha256', issues)
  })
  nestedObject(value.backend, '$.backend', ['requested', 'observed', 'assertion_passed'], issues, (backend) => {
    nonEmpty(backend.requested, '$.backend.requested', issues)
    if (backend.observed !== null) nonEmpty(backend.observed, '$.backend.observed', issues)
    boolean(backend.assertion_passed, '$.backend.assertion_passed', issues)
  })
  if (value.metrics === null) {
    nonEmpty(value.metrics_unavailable_reason, '$.metrics_unavailable_reason', issues)
  } else {
    if (value.metrics_unavailable_reason !== null) issues.push('$.metrics_unavailable_reason must be null when metrics are available')
    nestedObject(value.metrics, '$.metrics', [
      'load_ms', 'prefill_ms', 'ttft_ms', 'decode_ms', 'tokens_per_second',
      'prompt_tokens', 'generated_tokens', 'peak_rss_bytes', 'peak_vram_bytes',
      'peak_vram_unavailable_reason',
    ], issues, (metrics) => {
      for (const name of ['load_ms', 'prefill_ms', 'ttft_ms', 'decode_ms', 'tokens_per_second']) {
        nonNegativeNumber(metrics[name], `$.metrics.${name}`, issues)
      }
      positiveInteger(metrics.prompt_tokens, '$.metrics.prompt_tokens', issues)
      positiveInteger(metrics.generated_tokens, '$.metrics.generated_tokens', issues)
      positiveInteger(metrics.peak_rss_bytes, '$.metrics.peak_rss_bytes', issues)
      if (metrics.peak_vram_bytes === null) {
        nonEmpty(metrics.peak_vram_unavailable_reason, '$.metrics.peak_vram_unavailable_reason', issues)
      } else {
        nonNegativeInteger(metrics.peak_vram_bytes, '$.metrics.peak_vram_bytes', issues)
        if (metrics.peak_vram_unavailable_reason !== null) issues.push('$.metrics.peak_vram_unavailable_reason must be null when peak_vram_bytes is available')
      }
    })
  }
  nestedObject(value.correctness, '$.correctness', [
    'output_token_ids_sha256', 'parity_required', 'parity_passed', 'unavailable_reason',
    'parity_unavailable_reason',
  ], issues, (correctness) => {
    if (correctness.output_token_ids_sha256 === null) {
      nonEmpty(correctness.unavailable_reason, '$.correctness.unavailable_reason', issues)
    } else {
      sha256(correctness.output_token_ids_sha256, '$.correctness.output_token_ids_sha256', issues)
      if (correctness.unavailable_reason !== null) issues.push('$.correctness.unavailable_reason must be null when output token IDs are available')
    }
    boolean(correctness.parity_required, '$.correctness.parity_required', issues)
    if (correctness.parity_passed === null) {
      nonEmpty(correctness.parity_unavailable_reason, '$.correctness.parity_unavailable_reason', issues)
    } else {
      boolean(correctness.parity_passed, '$.correctness.parity_passed', issues)
      if (correctness.parity_unavailable_reason !== null) issues.push('$.correctness.parity_unavailable_reason must be null when parity is available')
    }
  })
  nestedObject(value.process, '$.process', ['state', 'exit_code', 'timed_out', 'cleanup_passed'], issues, (process) => {
    member(process.state, PROCESS_STATES, '$.process.state', issues)
    if (process.exit_code !== null && !Number.isInteger(process.exit_code)) issues.push('$.process.exit_code must be an integer or null')
    boolean(process.timed_out, '$.process.timed_out', issues)
    boolean(process.cleanup_passed, '$.process.cleanup_passed', issues)
  })

  if (value.validity === 'valid') {
    if (value.metrics === null) issues.push('$.metrics must be available for a valid sample')
    if (value.metrics_unavailable_reason !== null) issues.push('$.metrics_unavailable_reason must be null for a valid sample')
    if (value.backend?.assertion_passed !== true) issues.push('$.backend.assertion_passed must be true for a valid sample')
    if (value.correctness?.output_token_ids_sha256 === null) issues.push('$.correctness.output_token_ids_sha256 must be available for a valid sample')
    if (value.correctness?.unavailable_reason !== null) issues.push('$.correctness.unavailable_reason must be null for a valid sample')
    if (value.correctness?.parity_unavailable_reason !== null) issues.push('$.correctness.parity_unavailable_reason must be null for a valid sample')
    if (value.correctness?.parity_required === true && value.correctness?.parity_passed !== true) issues.push('$.correctness.parity_passed must be true when parity is required for a valid sample')
    if (value.process?.state !== 'exited') issues.push('$.process.state must be exited for a valid sample')
    if (value.process?.exit_code !== 0) issues.push('$.process.exit_code must be 0 for a valid sample')
    if (value.process?.timed_out !== false) issues.push('$.process.timed_out must be false for a valid sample')
    if (value.process?.cleanup_passed !== true) issues.push('$.process.cleanup_passed must be true for a valid sample')
  }
  if (issues.length > 0) fail('runtime-sample/v1', issues)
  return value
}

export function validateComparison(value) {
  const issues = []
  object(value, '$', issues)
  if (issues.length > 0) fail('comparison/v1', issues)
  exactKeys(value, '$', ['schema', 'campaign_id', 'runtime', 'agents'], [], issues)
  equal(value.schema, 'camelid.benchmark.comparison/v1', '$.schema', issues)
  id(value.campaign_id, '$.campaign_id', issues)
  array(value.runtime, '$.runtime', issues)
  array(value.agents, '$.agents', issues)
  const workloadIds = new Set()
  if (Array.isArray(value.runtime)) {
    value.runtime.forEach((record, index) => {
      const path = `$.runtime[${index}]`
      nestedObject(record, path, [
        'workload_id', 'metric', 'valid_pairs', 'excluded_pairs', 'base_median',
        'head_median', 'median_ratio_head_over_base', 'bootstrap_ci95',
        'practical_margin', 'observed_direction', 'bootstrap_seed',
        'bootstrap_samples', 'verdict',
      ], issues, (item) => {
        id(item.workload_id, `${path}.workload_id`, issues)
        unique(item.workload_id, workloadIds, `${path}.workload_id`, issues)
        nonEmpty(item.metric, `${path}.metric`, issues)
        nonNegativeInteger(item.valid_pairs, `${path}.valid_pairs`, issues)
        validateExcludedPairs(item.excluded_pairs, `${path}.excluded_pairs`, issues)
        nullablePositiveNumber(item.base_median, `${path}.base_median`, issues)
        nullablePositiveNumber(item.head_median, `${path}.head_median`, issues)
        nullablePositiveNumber(item.median_ratio_head_over_base, `${path}.median_ratio_head_over_base`, issues)
        if (item.bootstrap_ci95 === null) {
          if (item.valid_pairs > 0) issues.push(`${path}.bootstrap_ci95 cannot be null when valid pairs exist`)
        } else if (!Array.isArray(item.bootstrap_ci95) || item.bootstrap_ci95.length !== 2) {
          issues.push(`${path}.bootstrap_ci95 must be null or contain exactly two numbers`)
        } else {
          positiveNumber(item.bootstrap_ci95[0], `${path}.bootstrap_ci95[0]`, issues)
          positiveNumber(item.bootstrap_ci95[1], `${path}.bootstrap_ci95[1]`, issues)
          if (item.bootstrap_ci95[0] > item.bootstrap_ci95[1]) issues.push(`${path}.bootstrap_ci95 must be ordered low to high`)
        }
        if (item.valid_pairs === 0) {
          for (const field of ['base_median', 'head_median', 'median_ratio_head_over_base']) {
            if (item[field] !== null) issues.push(`${path}.${field} must be null when valid_pairs is zero`)
          }
        }
        if (item.practical_margin !== null) nonNegativeNumber(item.practical_margin, `${path}.practical_margin`, issues)
        member(item.observed_direction, new Set(['head_faster', 'head_slower', 'no_clear_direction', 'insufficient_data']), `${path}.observed_direction`, issues)
        nonNegativeInteger(item.bootstrap_seed, `${path}.bootstrap_seed`, issues)
        if (Number.isInteger(item.bootstrap_seed) && item.bootstrap_seed > 0xffff_ffff) issues.push(`${path}.bootstrap_seed must be a u32 integer`)
        positiveInteger(item.bootstrap_samples, `${path}.bootstrap_samples`, issues)
        member(item.verdict, RUNTIME_VERDICTS, `${path}.verdict`, issues)
      })
    })
  }
  const taskIds = new Set()
  if (Array.isArray(value.agents)) {
    value.agents.forEach((record, index) => {
      const path = `$.agents[${index}]`
      nestedObject(record, path, ['task_id', 'native', 'pi', 'comparative_claim_allowed', 'reason'], issues, (item) => {
        id(item.task_id, `${path}.task_id`, issues)
        unique(item.task_id, taskIds, `${path}.task_id`, issues)
        agentCount(item.native, `${path}.native`, issues)
        agentCount(item.pi, `${path}.pi`, issues)
        boolean(item.comparative_claim_allowed, `${path}.comparative_claim_allowed`, issues)
        nonEmpty(item.reason, `${path}.reason`, issues)
      })
    })
  }
  if (issues.length > 0) fail('comparison/v1', issues)
  return value
}

export function validateBenchGenerateRecord(value) {
  const issues = []
  object(value, '$', issues)
  if (issues.length > 0) fail('bench-generate/current', issues)
  exactKeys(value, '$', [
    'runtime', 'commit', 'model', 'quantization', 'iteration', 'prompt_tokens',
    'generated_tokens', 'load_ms', 'prefill_ms', 'ttft_ms', 'decode_ms',
    'tokens_per_second', 'peak_memory_bytes', 'output_text', 'output_token_ids',
  ], ['offload'], issues)
  equal(value.runtime, 'camelid', '$.runtime', issues)
  nonEmpty(value.commit, '$.commit', issues)
  nonEmpty(value.model, '$.model', issues)
  nonEmpty(value.quantization, '$.quantization', issues)
  nonNegativeInteger(value.iteration, '$.iteration', issues)
  positiveInteger(value.prompt_tokens, '$.prompt_tokens', issues)
  positiveInteger(value.generated_tokens, '$.generated_tokens', issues)
  for (const name of ['load_ms', 'prefill_ms', 'ttft_ms', 'decode_ms', 'tokens_per_second']) {
    nonNegativeNumber(value[name], `$.${name}`, issues)
  }
  positiveInteger(value.peak_memory_bytes, '$.peak_memory_bytes', issues)
  if (typeof value.output_text !== 'string') issues.push('$.output_text must be a string')
  if (!Array.isArray(value.output_token_ids) || value.output_token_ids.length !== value.generated_tokens) {
    issues.push('$.output_token_ids length must equal $.generated_tokens')
  } else {
    value.output_token_ids.forEach((token, index) => {
      if (!Number.isInteger(token) || token < 0 || token > 0xffff_ffff) issues.push(`$.output_token_ids[${index}] must be a u32 integer`)
    })
  }
  if (Object.hasOwn(value, 'offload')) validateOffload(value.offload, '$.offload', issues)
  if (issues.length > 0) fail('bench-generate/current', issues)
  return value
}

function validateOffload(value, path, issues) {
  nestedObject(value, path, [
    'total_layers', 'layers_resident', 'layers_offloaded', 'per_layer_bytes',
    'free_vram_bytes', 'pcie_gbps', 'source',
  ], issues, (offload) => {
    positiveInteger(offload.total_layers, `${path}.total_layers`, issues)
    nonNegativeInteger(offload.layers_resident, `${path}.layers_resident`, issues)
    nonNegativeInteger(offload.layers_offloaded, `${path}.layers_offloaded`, issues)
    nonNegativeInteger(offload.per_layer_bytes, `${path}.per_layer_bytes`, issues)
    nonNegativeInteger(offload.free_vram_bytes, `${path}.free_vram_bytes`, issues)
    if (offload.pcie_gbps !== null) positiveNumber(offload.pcie_gbps, `${path}.pcie_gbps`, issues)
    member(offload.source, new Set(['forced', 'auto', 'none']), `${path}.source`, issues)
    if (Number.isInteger(offload.total_layers)
      && Number.isInteger(offload.layers_resident)
      && Number.isInteger(offload.layers_offloaded)
      && offload.layers_resident + offload.layers_offloaded !== offload.total_layers) {
      issues.push(`${path}.layers_resident + layers_offloaded must equal total_layers`)
    }
  })
}

function agentCount(value, path, issues) {
  nestedObject(value, path, ['passes', 'valid_attempts'], issues, (count) => {
    nonNegativeInteger(count.passes, `${path}.passes`, issues)
    nonNegativeInteger(count.valid_attempts, `${path}.valid_attempts`, issues)
    if (Number.isInteger(count.passes) && Number.isInteger(count.valid_attempts) && count.passes > count.valid_attempts) {
      issues.push(`${path}.passes cannot exceed valid_attempts`)
    }
  })
}

function nestedObject(value, path, required, issues, check) {
  object(value, path, issues)
  if (!isObject(value)) return
  exactKeys(value, path, required, [], issues)
  check(value)
}

function exactKeys(value, path, required, optional, issues) {
  if (!isObject(value)) return
  const allowed = new Set([...required, ...optional])
  for (const key of required) {
    if (!Object.hasOwn(value, key)) issues.push(`${path}.${key} is required`)
  }
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) issues.push(`${path}.${key} is not allowed`)
  }
}

function object(value, path, issues) {
  if (!isObject(value)) issues.push(`${path} must be an object`)
}

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function array(value, path, issues) {
  if (!Array.isArray(value)) issues.push(`${path} must be an array`)
}

function nonEmptyArray(value, path, issues) {
  if (!Array.isArray(value) || value.length === 0) issues.push(`${path} must be a non-empty array`)
}

function stringArray(value, path, issues, uniqueValues) {
  if (!Array.isArray(value)) {
    issues.push(`${path} must be an array`)
    return
  }
  const seen = new Set()
  value.forEach((item, index) => {
    if (typeof item !== 'string') issues.push(`${path}[${index}] must be a string`)
    if (uniqueValues && seen.has(item)) issues.push(`${path}[${index}] duplicates ${JSON.stringify(item)}`)
    seen.add(item)
  })
}

function nonEmptyStringArray(value, path, issues, uniqueValues) {
  if (!Array.isArray(value) || value.length === 0) {
    issues.push(`${path} must be a non-empty array`)
    return
  }
  const seen = new Set()
  value.forEach((item, index) => {
    nonEmpty(item, `${path}[${index}]`, issues)
    if (uniqueValues && seen.has(item)) issues.push(`${path}[${index}] duplicates ${JSON.stringify(item)}`)
    seen.add(item)
  })
}

function unique(value, seen, path, issues) {
  if (seen.has(value)) issues.push(`${path} duplicates ${JSON.stringify(value)}`)
  seen.add(value)
}

function equal(value, expected, path, issues) {
  if (value !== expected) issues.push(`${path} must equal ${JSON.stringify(expected)}`)
}

function member(value, values, path, issues) {
  if (!values.has(value)) issues.push(`${path} must be one of ${[...values].join(', ')}`)
}

function id(value, path, issues) {
  if (typeof value !== 'string' || !ID.test(value)) issues.push(`${path} must be a valid identifier`)
}

function nonEmpty(value, path, issues) {
  if (typeof value !== 'string' || value.length === 0) issues.push(`${path} must be a non-empty string`)
}

function sha256(value, path, issues) {
  if (typeof value !== 'string' || !SHA256.test(value)) issues.push(`${path} must be 64 lowercase hex characters`)
}

function gitSha(value, path, issues) {
  if (typeof value !== 'string' || !GIT_SHA.test(value)) issues.push(`${path} must be 40 lowercase hex characters`)
}

function rfc3339(value, path, issues) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/.test(value) || Number.isNaN(Date.parse(value))) {
    issues.push(`${path} must be a UTC RFC3339 timestamp`)
  }
}

function boolean(value, path, issues) {
  if (typeof value !== 'boolean') issues.push(`${path} must be a boolean`)
}

function positiveInteger(value, path, issues) {
  if (!Number.isSafeInteger(value) || value <= 0) issues.push(`${path} must be a positive safe integer`)
}

function nonNegativeInteger(value, path, issues) {
  if (!Number.isSafeInteger(value) || value < 0) issues.push(`${path} must be a non-negative safe integer`)
}

function positiveNumber(value, path, issues) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) issues.push(`${path} must be a positive finite number`)
}

function nonNegativeNumber(value, path, issues) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) issues.push(`${path} must be a non-negative finite number`)
}

function nullableReason(value, path, issues, required) {
  if (required) nonEmpty(value, path, issues)
  else if (value !== null) issues.push(`${path} must be null for a valid record`)
}

function fail(contract, issues) {
  throw new ContractError(contract, issues)
}

function validateBackendContract(value, path, issues) {
  nestedObject(value, path, ['requested', 'assertion'], issues, (backend) => {
    member(backend.requested, BACKENDS, `${path}.requested`, issues)
    member(backend.assertion, BACKEND_ASSERTIONS, `${path}.assertion`, issues)
  })
}

function validatePhase1Arms(armIds, issues) {
  const sorted = [...armIds].sort()
  if (sorted.length !== 2 || sorted[0] !== 'base' || sorted[1] !== 'head') {
    issues.push('$.source_arms must contain exactly two arms named base and head')
  }
}

function validateBalancedOrder(value, repetitions, armIds, path, issues) {
  nonEmptyStringArray(value, path, issues, false)
  if (!Array.isArray(value)) return
  const counts = new Map([...armIds].map((armId) => [armId, 0]))
  value.forEach((armId, index) => {
    if (!armIds.has(armId)) {
      issues.push(`${path}[${index}] references unknown arm ${JSON.stringify(armId)}`)
      return
    }
    counts.set(armId, counts.get(armId) + 1)
  })
  if (Number.isSafeInteger(repetitions) && repetitions > 0) {
    for (const [armId, count] of counts) {
      if (count !== repetitions) issues.push(`${path} must contain arm ${JSON.stringify(armId)} exactly ${repetitions} times; found ${count}`)
    }
  }
}

function validateSecurity(value, issues) {
  nestedObject(value, '$.security', ['network', 'trust_tier'], issues, (security) => {
    member(security.network, NETWORK_POLICIES, '$.security.network', issues)
    member(security.trust_tier, TRUST_TIERS, '$.security.trust_tier', issues)
  })
}

function validateResources(value, issues) {
  nestedObject(value, '$.resources', ['minimum_free_disk_bytes'], issues, (resources) => {
    positiveInteger(resources.minimum_free_disk_bytes, '$.resources.minimum_free_disk_bytes', issues)
  })
}

function validateExcludedPairs(value, path, issues) {
  array(value, path, issues)
  if (!Array.isArray(value)) return
  value.forEach((excluded, index) => {
    const itemPath = `${path}[${index}]`
    nestedObject(excluded, itemPath, ['process_block', 'reason'], issues, (item) => {
      nonNegativeInteger(item.process_block, `${itemPath}.process_block`, issues)
      nonEmpty(item.reason, `${itemPath}.reason`, issues)
    })
  })
}

function nullablePositiveNumber(value, path, issues) {
  if (value !== null) positiveNumber(value, path, issues)
}
