import { stat } from 'node:fs/promises'
import { resolve } from 'node:path'

import { validatePlan } from './lib/contracts.mjs'
import { sha256File } from './lib/digest.mjs'
import { runProcess } from './process/runner.mjs'

export class PreparationError extends Error {
  constructor(code, message, options = {}) {
    super(message, options.cause ? { cause: options.cause } : undefined)
    this.name = 'PreparationError'
    this.code = code
    this.armId = options.armId ?? null
  }
}

export async function prepareArms(plan, options = {}) {
  validatePlan(plan)
  const execute = options.execute ?? runProcess
  const inspectBinary = options.inspectBinary ?? defaultInspectBinary
  const buildTimeoutMs = options.buildTimeoutMs ?? 45 * 60 * 1000
  const versionTimeoutMs = options.versionTimeoutMs ?? 30 * 1000
  if (!Number.isSafeInteger(buildTimeoutMs) || buildTimeoutMs < 1) {
    throw new RangeError('buildTimeoutMs must be a positive safe integer')
  }
  if (!Number.isSafeInteger(versionTimeoutMs) || versionTimeoutMs < 1) {
    throw new RangeError('versionTimeoutMs must be a positive safe integer')
  }

  const prepared = []
  for (const arm of plan.source_arms) {
    const args = buildArgs(arm)
    const build = await execute({
      file: arm.build.cargo_path,
      args,
      cwd: arm.source_dir,
      env: isolatedBuildEnv(plan.security, process.env),
      timeoutMs: buildTimeoutMs,
      stdoutFile: options.logDir ? resolve(options.logDir, `${arm.id}.build.stdout.log`) : null,
      stderrFile: options.logDir ? resolve(options.logDir, `${arm.id}.build.stderr.log`) : null,
    })
    requireSuccess(build, arm.id, 'build')

    const binary = await inspectBinary(arm.build.binary_path)
    const versionRun = await execute({
      file: arm.build.binary_path,
      args: ['--version'],
      cwd: arm.source_dir,
      env: process.env,
      timeoutMs: versionTimeoutMs,
      stdoutFile: options.logDir ? resolve(options.logDir, `${arm.id}.version.stdout.log`) : null,
      stderrFile: options.logDir ? resolve(options.logDir, `${arm.id}.version.stderr.log`) : null,
    })
    requireSuccess(versionRun, arm.id, 'version check')
    const reportedVersion = versionRun.stdout.preview.trim()
    if (reportedVersion.length === 0) {
      throw new PreparationError('invalid_binary', `arm ${arm.id} produced an empty --version response`, { armId: arm.id })
    }

    prepared.push({
      arm_id: arm.id,
      source_sha: arm.git_sha,
      binary_path: binary.path,
      binary_sha256: binary.sha256,
      binary_size_bytes: binary.sizeBytes,
      reported_version: reportedVersion,
      build: summarizeProcess(build),
      version_check: summarizeProcess(versionRun),
    })
  }
  return prepared
}

export function buildArgs(arm) {
  const args = [
    `+${arm.build.toolchain}`,
    'build',
    '--release',
    '--locked',
    '--bin',
    'camelid',
    '--target-dir',
    arm.build.target_dir,
  ]
  if (arm.build.features.length > 0) {
    args.push('--features', arm.build.features.join(','))
  }
  return args
}

export function isolatedBuildEnv(security, sourceEnv) {
  const blocked = /^(?:CAMELID_|CARGO_TARGET_DIR$|CARGO_ENCODED_RUSTFLAGS$|RUSTFLAGS$|RUSTDOCFLAGS$)/i
  const env = Object.fromEntries(Object.entries(sourceEnv).filter(([key]) => !blocked.test(key)))
  env.CARGO_INCREMENTAL = '0'
  if (security.network === 'deny') env.CARGO_NET_OFFLINE = 'true'
  return env
}

async function defaultInspectBinary(path) {
  const absolute = resolve(path)
  let info
  try {
    info = await stat(absolute)
  } catch (error) {
    throw new PreparationError('invalid_binary', `prepared binary is not accessible: ${absolute}`, { cause: error })
  }
  if (!info.isFile()) throw new PreparationError('invalid_binary', `prepared binary is not a file: ${absolute}`)
  return {
    path: absolute,
    sha256: await sha256File(absolute),
    sizeBytes: info.size,
  }
}

function requireSuccess(result, armId, step) {
  if (result.state !== 'exited' || result.exitCode !== 0 || result.cleanupPassed !== true) {
    const detail = result.error || result.stderr?.preview?.trim() || result.cleanupDetail || `${result.state} exit ${result.exitCode}`
    throw new PreparationError('build_failed', `arm ${armId} ${step} failed: ${detail}`, { armId })
  }
}

function summarizeProcess(result) {
  return {
    state: result.state,
    exit_code: result.exitCode,
    timed_out: result.timedOut,
    duration_ms: result.durationMs,
    cleanup_passed: result.cleanupPassed,
    stdout_bytes: result.stdout.totalBytes,
    stderr_bytes: result.stderr.totalBytes,
  }
}
