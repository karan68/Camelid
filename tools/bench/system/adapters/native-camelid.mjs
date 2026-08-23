import { createServer } from 'node:net'
import { resolve } from 'node:path'

import { validateAgentAttempt } from '../lib/contracts.mjs'
import { sha256File } from '../lib/digest.mjs'
import { runProcess } from '../process/runner.mjs'
import { loadTaskPackage, materializeTask, scoreTaskAttempt } from '../tasks/package.mjs'

const FORBIDDEN_SHARED_FLAGS = new Set(['--allow-net', '--allow-fs', '--allow-mcp'])

export async function runNativeAgentAttempt(options) {
  const normalized = validateOptions(options)
  const taskPackage = await loadTaskPackage(normalized.taskRoot)
  if (taskPackage.task.network !== 'deny') {
    throw new NativeAdapterError('INVALID_FIXTURE', 'shared native tasks must deny network access')
  }
  if (normalized.timeoutMs > taskPackage.task.budgets.wall_ms) {
    throw new NativeAdapterError('INVALID_FIXTURE', 'native adapter timeout cannot exceed the task wall budget')
  }

  // Resolve immutable inputs before materializing any writable task state.
  const [binarySha256, modelSha256] = await Promise.all([
    sha256File(normalized.binaryPath),
    sha256File(normalized.modelPath),
  ])

  const materialized = await materializeTask(taskPackage, normalized.workspaceRoot)
  const addr = await reserveLoopbackAddress()
  const args = nativeExecArgs({
    task: taskPackage.task,
    modelPath: normalized.modelPath,
    workdir: materialized.attemptRoot,
    addr,
  })
  const startedAt = performance.now()
  const execution = await runProcess({
    file: normalized.binaryPath,
    args: [...normalized.syntheticCandidatePrefix, ...args],
    cwd: materialized.attemptRoot,
    env: isolatedNativeEnv(normalized.env),
    timeoutMs: normalized.timeoutMs,
  })
  const wallMs = performance.now() - startedAt

  // The scorer runs only after runProcess has observed a terminal child and,
  // on timeout, completed exact descendant cleanup.
  const repositoryScore = await scoreTaskAttempt(normalized.taskRoot, materialized.workspaceRoot)
  const terminal = terminalFromExecution(execution)
  const outcome = attemptOutcome(terminal, execution.cleanupPassed, repositoryScore.outcome)
  const attemptRecord = validateAgentAttempt({
    schema: 'camelid.benchmark.agent-attempt/v1',
    campaign_id: normalized.campaignId,
    task_id: taskPackage.task.id,
    adapter: 'camelid-native',
    attempt: normalized.attempt,
    comparability: 'comparable',
    terminal,
    score: {
      outcome,
      required_checks: Math.max(1, repositoryScore.required_checks),
      passed_checks: repositoryScore.passed_checks,
      diff_sha256: repositoryScore.diff_sha256,
    },
    usage: {
      model_steps: null,
      tool_calls: null,
      input_tokens: null,
      output_tokens: null,
      unavailable_reason: 'native structured events are not implemented (O9)',
    },
    timing: {
      wall_ms: wallMs,
      model_ms: null,
      ttft_ms: null,
    },
    process: {
      cleanup_passed: execution.cleanupPassed,
    },
  })

  return {
    identity: {
      source_sha: normalized.sourceSha,
      binary_sha256: binarySha256,
      model_sha256: modelSha256,
    },
    address: 'loopback_ephemeral',
    args,
    execution,
    repository_score: repositoryScore,
    attempt: attemptRecord,
    workspace_root: materialized.workspaceRoot,
    attempt_root: materialized.attemptRoot,
  }
}

export function nativeExecArgs({ task, modelPath, workdir, addr }) {
  const args = [
    'agent',
    'exec',
    task.goal,
    '--model',
    resolve(modelPath),
    '--addr',
    addr,
    '--workdir',
    resolve(workdir),
    '--max-steps',
    String(task.budgets.max_steps),
    '--max-tokens',
    String(task.budgets.max_output_tokens_per_step),
    '--shell-sandbox',
    'sandboxed',
    '--shell-timeout',
    String(Math.max(1, Math.ceil(task.budgets.command_ms / 1000))),
    '--today-is-a-good-day-to-die',
  ]
  for (const flag of FORBIDDEN_SHARED_FLAGS) {
    if (args.includes(flag)) throw new NativeAdapterError('INVALID_FIXTURE', `shared task enabled forbidden flag ${flag}`)
  }
  return args
}

export async function reserveLoopbackAddress() {
  const server = createServer()
  await new Promise((resolveListen, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolveListen)
  })
  const address = server.address()
  await new Promise((resolveClose, reject) => server.close((error) => error ? reject(error) : resolveClose()))
  if (address === null || typeof address === 'string') throw new NativeAdapterError('INVALID_INFRASTRUCTURE', 'could not reserve a loopback TCP port')
  return `127.0.0.1:${address.port}`
}

export class NativeAdapterError extends Error {
  constructor(outcome, message) {
    super(message)
    this.name = 'NativeAdapterError'
    this.outcome = outcome
  }
}

function validateOptions(options) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('native adapter options must be an object')
  }
  if (options.disposableBoundary !== true) {
    throw new NativeAdapterError('INVALID_INFRASTRUCTURE', 'unattended native agent execution requires an explicit disposable boundary')
  }
  for (const name of ['taskRoot', 'workspaceRoot', 'binaryPath', 'modelPath']) {
    if (typeof options[name] !== 'string' || options[name].length === 0) throw new TypeError(`${name} must be a non-empty string`)
  }
  if (typeof options.campaignId !== 'string' || options.campaignId.length === 0) throw new TypeError('campaignId must be non-empty')
  if (typeof options.sourceSha !== 'string' || !/^[0-9a-f]{40}$/.test(options.sourceSha)) throw new TypeError('sourceSha must be 40 lowercase hex characters')
  if (!Number.isSafeInteger(options.attempt) || options.attempt < 0) throw new TypeError('attempt must be a non-negative safe integer')
  if (!Number.isSafeInteger(options.timeoutMs) || options.timeoutMs < 1) throw new TypeError('timeoutMs must be a positive safe integer')
  return {
    taskRoot: resolve(options.taskRoot),
    workspaceRoot: resolve(options.workspaceRoot),
    binaryPath: resolve(options.binaryPath),
    modelPath: resolve(options.modelPath),
    campaignId: options.campaignId,
    sourceSha: options.sourceSha,
    attempt: options.attempt,
    timeoutMs: options.timeoutMs,
    env: options.env ?? process.env,
    syntheticCandidatePrefix: syntheticPrefix(options),
  }
}

function syntheticPrefix(options) {
  if (options.syntheticCandidatePrefix === undefined) return []
  if (options.syntheticCandidate !== true) {
    throw new NativeAdapterError('INVALID_INFRASTRUCTURE', 'a synthetic candidate prefix is test-only')
  }
  if (!Array.isArray(options.syntheticCandidatePrefix)
    || options.syntheticCandidatePrefix.length === 0
    || options.syntheticCandidatePrefix.some((item) => typeof item !== 'string' || item.length === 0)) {
    throw new TypeError('syntheticCandidatePrefix must be a non-empty string array')
  }
  return [...options.syntheticCandidatePrefix]
}

function terminalFromExecution(execution) {
  if (execution.timedOut) return { class: 'timed_out', exit_code: execution.exitCode, reason: 'native agent wall timeout expired' }
  if (execution.state !== 'exited') return { class: 'adapter_error', exit_code: execution.exitCode, reason: execution.error ?? execution.state }
  if (execution.exitCode === 0) return { class: 'answered', exit_code: 0, reason: 'agent exec returned answered' }
  if (execution.exitCode === 1) return { class: 'failed', exit_code: 1, reason: 'agent exec returned failed or blocked' }
  if (execution.exitCode === 3) return { class: 'inconclusive', exit_code: 3, reason: 'agent exec returned inconclusive; exact reason unavailable without O9 events' }
  return { class: 'adapter_error', exit_code: execution.exitCode, reason: `unexpected agent exec exit ${execution.exitCode}` }
}

function attemptOutcome(terminal, cleanupPassed, repositoryOutcome) {
  if (!cleanupPassed) return 'INVALID_INFRASTRUCTURE'
  if (terminal.class === 'timed_out') return 'INCONCLUSIVE_TIMEOUT'
  if (terminal.class === 'adapter_error') return 'INVALID_INFRASTRUCTURE'
  if (terminal.class !== 'answered') return 'FAIL_AGENT_TERMINAL'
  return repositoryOutcome
}

export function isolatedNativeEnv(source) {
  const allowed = new Set([
    'APPDATA',
    'HOME',
    'LOCALAPPDATA',
    'PATH',
    'PATHEXT',
    'PROGRAMDATA',
    'SYSTEMROOT',
    'TEMP',
    'TMP',
    'TMPDIR',
    'USERPROFILE',
    'WINDIR',
  ])
  return Object.fromEntries(Object.entries(source).filter(([key]) => allowed.has(key.toUpperCase())))
}