import { spawnSync } from 'node:child_process'
import { readFile, realpath, stat } from 'node:fs/promises'
import { join, resolve } from 'node:path'

import { validateCampaign, validatePlan } from './lib/contracts.mjs'
import { canonicalJson, sha256Bytes, sha256File } from './lib/digest.mjs'

export class PlanResolutionError extends Error {
  constructor(code, message, options = {}) {
    super(message, options.cause ? { cause: options.cause } : undefined)
    this.name = 'PlanResolutionError'
    this.code = code
  }
}

export async function resolveCampaignPlan(request, dependencies = {}) {
  validateCampaign(request)
  const inspectSource = dependencies.inspectSource ?? defaultInspectSource
  const inspectArtifact = dependencies.inspectArtifact ?? defaultInspectArtifact
  const inspectPrompt = dependencies.inspectPrompt ?? defaultInspectPrompt
  const resolveDirectory = dependencies.resolveDirectory ?? defaultResolveDirectory
  const binaryName = dependencies.binaryName ?? (process.platform === 'win32' ? 'camelid.exe' : 'camelid')

  const repositoryRoot = await resolveDirectory(request.repository_root)
  const sourceArms = []
  for (const arm of request.source_arms) {
    const inspected = await inspectSource(arm)
    if (inspected.gitSha !== arm.expected_git_sha) {
      throw new PlanResolutionError(
        'invalid_hash',
        `source arm ${arm.id} resolved to ${inspected.gitSha}, expected ${arm.expected_git_sha}`,
      )
    }
    if (inspected.treeDirty) {
      throw new PlanResolutionError(
        'invalid_environment',
        `source arm ${arm.id} is dirty; Phase 1 does not benchmark uncommitted source`,
      )
    }
    const targetDir = resolve(arm.target_dir)
    sourceArms.push({
      id: arm.id,
      source_dir: inspected.sourceDir,
      git_sha: inspected.gitSha,
      tree_dirty: inspected.treeDirty,
      cargo_lock_sha256: inspected.cargoLockSha256,
      build: {
        cargo_path: resolve(arm.cargo_path),
        toolchain: arm.toolchain,
        profile: 'release',
        features: [...arm.features],
        target_dir: targetDir,
        binary_path: join(targetDir, 'release', binaryName),
      },
    })
  }

  const models = []
  for (const model of request.models) {
    const inspected = await inspectArtifact(model.artifact_path)
    if (inspected.sha256 !== model.expected_sha256) {
      throw new PlanResolutionError(
        'invalid_hash',
        `model ${model.id} resolved to ${inspected.sha256}, expected ${model.expected_sha256}`,
      )
    }
    models.push({
      id: model.id,
      artifact_path: inspected.path,
      artifact_sha256: inspected.sha256,
      size_bytes: inspected.sizeBytes,
      quantization: model.quantization,
    })
  }

  const workloads = []
  for (const workload of request.workloads) {
    const prompt = await inspectPrompt(workload.prompt_file)
    workloads.push({
      id: workload.id,
      adapter: workload.adapter,
      model_id: workload.model_id,
      prompt_file: prompt.path,
      prompt_sha256: prompt.sha256,
      prompt_policy: 'front_block_marker_v1',
      max_tokens: workload.max_tokens,
      warmup: workload.warmup,
      deterministic: workload.deterministic,
      threads: workload.threads,
      backend: { ...workload.backend },
      primary_metrics: [...workload.primary_metrics],
      order: balancedArmOrder(sourceArms.map((arm) => arm.id), workload.repetitions),
      repetitions: workload.repetitions,
      timeout_ms: workload.timeout_ms,
    })
  }

  const plan = {
    schema: 'camelid.benchmark.plan/v1',
    campaign_id: request.campaign_id,
    mode: request.mode,
    created_utc: request.created_utc,
    controller: { ...request.controller },
    repository_root: repositoryRoot,
    source_arms: sourceArms,
    models,
    workloads,
    resources: { ...request.resources },
    security: { ...request.security },
  }
  validatePlan(plan)
  return plan
}

export function serializePlan(plan) {
  validatePlan(plan)
  return canonicalJson(plan)
}

async function defaultInspectSource(arm) {
  const sourceDir = await defaultResolveDirectory(arm.source_dir)
  const gitSha = git(['rev-parse', 'HEAD'], sourceDir)
  const status = git(['status', '--porcelain=v1', '--untracked-files=all'], sourceDir)
  const cargoLockSha256 = await sha256File(join(sourceDir, 'Cargo.lock'))
  return {
    sourceDir,
    gitSha,
    treeDirty: status.length > 0,
    cargoLockSha256,
  }
}

async function defaultInspectArtifact(path) {
  const absolute = await realpath(resolve(path))
  const info = await stat(absolute)
  if (!info.isFile()) throw new PlanResolutionError('invalid_environment', `artifact is not a file: ${absolute}`)
  return {
    path: absolute,
    sha256: await sha256File(absolute),
    sizeBytes: info.size,
  }
}

async function defaultInspectPrompt(path) {
  const absolute = await realpath(resolve(path))
  const bytes = await readFile(absolute)
  try {
    new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch (error) {
    throw new PlanResolutionError('invalid_environment', `prompt is not valid UTF-8: ${absolute}`, { cause: error })
  }
  return {
    path: absolute,
    sha256: sha256Bytes(bytes),
  }
}

async function defaultResolveDirectory(path) {
  const absolute = await realpath(resolve(path))
  const info = await stat(absolute)
  if (!info.isDirectory()) throw new PlanResolutionError('invalid_environment', `path is not a directory: ${absolute}`)
  return absolute
}

function git(args, cwd) {
  const result = spawnSync('git', args, {
    cwd,
    encoding: 'utf8',
    windowsHide: true,
  })
  if (result.status !== 0) {
    const detail = (result.stderr || result.stdout || `exit ${result.status}`).trim()
    throw new PlanResolutionError('invalid_environment', `git ${args.join(' ')} failed in ${cwd}: ${detail}`)
  }
  return result.stdout.trim()
}

export function balancedArmOrder(armIds, repetitions) {
  if (!Array.isArray(armIds) || armIds.length === 0) {
    throw new RangeError('balanced arm order requires at least one arm')
  }
  if (armIds.some((armId) => typeof armId !== 'string' || armId.length === 0)) {
    throw new TypeError('balanced arm IDs must be non-empty strings')
  }
  if (new Set(armIds).size !== armIds.length) {
    throw new RangeError('balanced arm IDs must be unique')
  }
  if (!Number.isSafeInteger(repetitions) || repetitions < 1) {
    throw new RangeError('balanced arm repetitions must be a positive safe integer')
  }
  const order = []
  for (let round = 0; round < repetitions; round += 1) {
    for (let offset = 0; offset < armIds.length; offset += 1) {
      order.push(armIds[(round + offset) % armIds.length])
    }
  }
  return order
}
