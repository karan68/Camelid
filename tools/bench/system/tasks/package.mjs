import { lstat, mkdir, readFile, readdir, writeFile, copyFile } from 'node:fs/promises'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'

import { validateAgentTask, validateTaskCheck } from '../lib/contracts.mjs'
import { canonicalJson, sha256Bytes, sha256File } from '../lib/digest.mjs'
import { runProcess } from '../process/runner.mjs'

const BINARY_EXTENSIONS = new Set(['.a', '.bin', '.dll', '.dylib', '.exe', '.lib', '.o', '.obj', '.so', '.wasm'])
const SCORE_SCHEMA = 'camelid.benchmark.task-check/v1'

export async function loadTaskPackage(taskRoot) {
  const { root, task, taskDefinitionSha256, fixture, scorer, canaries } = await taskPackageDigests(taskRoot)
  if (fixture.sha256 !== task.fixture_manifest_sha256) {
    throw new TaskPackageError('INVALID_FIXTURE', `fixture manifest is ${fixture.sha256}, task pins ${task.fixture_manifest_sha256}`)
  }
  if (scorer.sha256 !== task.scorer_manifest_sha256) {
    throw new TaskPackageError('INVALID_FIXTURE', `scorer manifest is ${scorer.sha256}, task pins ${task.scorer_manifest_sha256}`)
  }
  validateScorerCommand(task.scorer_command)
  for (const [index, canary] of task.canaries.entries()) {
    if (canary.location === 'attempt' || canary.location.startsWith('attempt/')) {
      throw new TaskPackageError('INVALID_FIXTURE', `canary ${canary.id} must be outside the writable attempt root`)
    }
    const actual = canaries[index].sha256
    if (actual !== canary.sha256) {
      throw new TaskPackageError('INVALID_FIXTURE', `canary ${canary.id} digest is ${actual}, task pins ${canary.sha256}`)
    }
  }
  return { root, task, taskDefinitionSha256, fixture, scorer }
}

export async function taskPackageDigests(taskRoot) {
  const root = resolve(taskRoot)
  await requireRealDirectory(root, 'task package')
  const taskPath = join(root, 'task.json')
  await requireRealFile(taskPath, 'task manifest')
  const taskBytes = await readFile(taskPath)
  const task = validateAgentTask(JSON.parse(taskBytes.toString('utf8')))
  const taskDefinitionSha256 = sha256Bytes(taskBytes)
  const fixture = await treeManifest(root, 'fixture')
  const scorer = await treeManifest(root, 'scorer', task.scorer_command)
  const canaries = task.canaries.map((canary) => ({
    id: canary.id,
    sha256: sha256Bytes(canaryBytes(task.id, canary.id)),
  }))
  return { root, task, taskDefinitionSha256, fixture, scorer, canaries }
}

export async function materializeTask(taskPackage, workspaceRoot) {
  const root = resolve(workspaceRoot)
  const attemptRoot = join(root, 'attempt')
  await requireAbsent(root, 'workspace root')
  await copyTree(join(taskPackage.root, 'fixture'), attemptRoot)
  for (const canary of taskPackage.task.canaries) {
    const path = confined(root, canary.location)
    await mkdir(dirname(path), { recursive: true })
    await writeFile(path, canaryBytes(taskPackage.task.id, canary.id))
  }
  const setup = await runTaskCommand(taskPackage.task.setup_command, attemptRoot, taskPackage.task.budgets.command_ms)
  if (!succeeded(setup)) {
    throw new TaskPackageError('INVALID_FIXTURE', `setup failed: ${processDetail(setup)}`)
  }
  return { workspaceRoot: root, attemptRoot }
}

export async function scoreTaskAttempt(taskRoot, workspaceRoot) {
  let taskPackage
  try {
    taskPackage = await loadTaskPackage(taskRoot)
  } catch (error) {
    if (error instanceof TaskPackageError) return invalidScore(error.outcome, error.message)
    return invalidScore('INVALID_FIXTURE', error.message)
  }
  const attemptRoot = join(resolve(workspaceRoot), 'attempt')
  let attempt
  try {
    attempt = await treeManifest(resolve(workspaceRoot), 'attempt')
  } catch (error) {
    return invalidScore('INVALID_INFRASTRUCTURE', error.message)
  }
  const mutations = compareManifests(taskPackage.fixture.manifest.files, attempt.manifest.files)
  const forbidden = mutationViolations(mutations, taskPackage.task)
  const canaryChecks = []
  for (const canary of taskPackage.task.canaries) {
    const path = confined(resolve(workspaceRoot), canary.location)
    let actual = null
    try {
      actual = await sha256File(path)
    } catch (error) {
      canaryChecks.push({ id: canary.id, passed: false, detail: error.message })
      continue
    }
    canaryChecks.push({
      id: canary.id,
      passed: actual === canary.sha256,
      detail: actual === canary.sha256 ? 'unchanged' : `expected ${canary.sha256}, found ${actual}`,
    })
  }
  const diffSha256 = sha256Bytes(Buffer.from(canonicalJson(mutations), 'utf8'))
  if (forbidden.length > 0 || canaryChecks.some((check) => !check.passed)) {
    return score('FAIL_FORBIDDEN_MUTATION', diffSha256, mutations, canaryChecks, [], null, forbidden)
  }

  const requiredChecks = []
  for (const check of taskPackage.task.required_checks) {
    const execution = await runTaskCommand(check.command, attemptRoot, taskPackage.task.budgets.command_ms)
    requiredChecks.push({
      id: check.id,
      passed: succeeded(execution),
      state: execution.state,
      exit_code: execution.exitCode,
      timed_out: execution.timedOut,
      cleanup_passed: execution.cleanupPassed,
      detail: processDetail(execution),
    })
  }
  if (requiredChecks.some((check) => check.timed_out || !check.cleanup_passed || check.state !== 'exited')) {
    return score('INVALID_SCORER', diffSha256, mutations, canaryChecks, requiredChecks, null, ['a required check did not exit cleanly'])
  }
  if (requiredChecks.some((check) => !check.passed)) {
    return score('FAIL_BEHAVIOR', diffSha256, mutations, canaryChecks, requiredChecks, null, [])
  }

  const scorerExecution = await runScorerCommand(taskPackage, attemptRoot)
  if (!succeeded(scorerExecution)) {
    return score('INVALID_SCORER', diffSha256, mutations, canaryChecks, requiredChecks, null, [processDetail(scorerExecution)])
  }
  let hiddenCheck
  try {
    hiddenCheck = parseScorerOutput(scorerExecution.stdout.preview)
    validateScorerChecks(hiddenCheck, taskPackage.task.scorer_checks)
  } catch (error) {
    return score('INVALID_SCORER', diffSha256, mutations, canaryChecks, requiredChecks, null, [error.message])
  }
  let finalPackage
  try {
    finalPackage = await loadTaskPackage(taskRoot)
  } catch (error) {
    return score('INVALID_SCORER', diffSha256, mutations, canaryChecks, requiredChecks, hiddenCheck, [error.message])
  }
  if (finalPackage.taskDefinitionSha256 !== taskPackage.taskDefinitionSha256
    || finalPackage.fixture.sha256 !== taskPackage.fixture.sha256
    || finalPackage.scorer.sha256 !== taskPackage.scorer.sha256) {
    return score('INVALID_SCORER', diffSha256, mutations, canaryChecks, requiredChecks, hiddenCheck, ['task package changed while the scorer ran'])
  }
  return score(hiddenCheck.passed ? 'PASS_COMPARABLE' : 'FAIL_BEHAVIOR', diffSha256, mutations, canaryChecks, requiredChecks, hiddenCheck, [])
}

export async function applyTaskOverlay(taskRoot, overlay, attemptRoot) {
  const source = confined(resolve(taskRoot), overlay)
  await copyTree(source, resolve(attemptRoot))
}

export function canaryBytes(taskId, canaryId) {
  return Buffer.from(`camelid-benchmark-canary/v1\n${taskId}\n${canaryId}\n`, 'utf8')
}

export class TaskPackageError extends Error {
  constructor(outcome, message) {
    super(message)
    this.name = 'TaskPackageError'
    this.outcome = outcome
  }
}

async function treeManifest(packageRoot, directory, command = null) {
  const root = confined(packageRoot, directory)
  await requireRealDirectory(root, directory)
  const files = []
  await walk(root, root, files)
  files.sort((left, right) => left.path.localeCompare(right.path))
  const manifest = {
    schema: 'camelid.benchmark.file-manifest/v1',
    files,
    ...(command === null ? {} : { command: [...command] }),
  }
  return { manifest, sha256: sha256Bytes(Buffer.from(canonicalJson(manifest), 'utf8')) }
}

async function walk(root, directory, files) {
  const entries = await readdir(directory, { withFileTypes: true })
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const path = join(directory, entry.name)
    if (entry.isSymbolicLink()) throw new TaskPackageError('INVALID_FIXTURE', `symbolic links are not allowed: ${portable(relative(root, path))}`)
    if (entry.isDirectory()) {
      await walk(root, path, files)
      continue
    }
    if (!entry.isFile()) throw new TaskPackageError('INVALID_FIXTURE', `special files are not allowed: ${portable(relative(root, path))}`)
    const info = await lstat(path)
    files.push({
      path: portable(relative(root, path)),
      size_bytes: info.size,
      sha256: await sha256File(path),
      executable: (info.mode & 0o111) !== 0,
    })
  }
  rejectCaseCollisions(files)
}

function compareManifests(baselineFiles, attemptFiles) {
  const baseline = new Map(baselineFiles.map((file) => [file.path, file]))
  const attempt = new Map(attemptFiles.map((file) => [file.path, file]))
  const paths = [...new Set([...baseline.keys(), ...attempt.keys()])].sort()
  return paths.flatMap((path) => {
    const before = baseline.get(path)
    const after = attempt.get(path)
    if (!before) {
      return [{
        path,
        kind: 'added',
        before_sha256: null,
        after_sha256: after.sha256,
        before_executable: null,
        after_executable: after.executable,
      }]
    }
    if (!after) {
      return [{
        path,
        kind: 'deleted',
        before_sha256: before.sha256,
        after_sha256: null,
        before_executable: before.executable,
        after_executable: null,
      }]
    }
    if (before.sha256 !== after.sha256 || before.executable !== after.executable) {
      return [{
        path,
        kind: 'modified',
        before_sha256: before.sha256,
        after_sha256: after.sha256,
        before_executable: before.executable,
        after_executable: after.executable,
      }]
    }
    return []
  })
}

function mutationViolations(mutations, task) {
  const violations = []
  for (const mutation of mutations) {
    if (task.forbidden_mutations.some((pattern) => matchesPath(pattern, mutation.path))) {
      violations.push(`${mutation.path} matches a forbidden mutation pattern`)
    } else if (!task.allowed_mutations.some((pattern) => matchesPath(pattern, mutation.path))) {
      violations.push(`${mutation.path} is outside the allowed mutation set`)
    }
    const dot = mutation.path.lastIndexOf('.')
    const extension = dot >= 0 ? mutation.path.slice(dot).toLowerCase() : ''
    if (mutation.kind !== 'deleted' && BINARY_EXTENSIONS.has(extension)) {
      violations.push(`${mutation.path} is an unexpected binary artifact`)
    }
    if (mutation.kind !== 'deleted' && mutation.after_executable === true) {
      violations.push(`${mutation.path} is an unexpected executable artifact`)
    }
  }
  return violations
}

function matchesPath(pattern, path) {
  if (!pattern.endsWith('/**')) return path === pattern
  const prefix = pattern.slice(0, -3)
  return path === prefix || path.startsWith(`${prefix}/`)
}

async function runTaskCommand(command, cwd, timeoutMs) {
  validateNodeCommand(command)
  return runProcess({
    file: process.execPath,
    args: command.slice(1),
    cwd,
    env: isolatedEnvironment(process.env),
    timeoutMs,
  })
}

async function runScorerCommand(taskPackage, attemptRoot) {
  const command = taskPackage.task.scorer_command
  const script = confined(taskPackage.root, command[1])
  const scorerRoot = join(taskPackage.root, 'scorer')
  if (relative(scorerRoot, script).startsWith('..')) throw new TaskPackageError('INVALID_FIXTURE', 'scorer script must stay under scorer/')
  return runProcess({
    file: process.execPath,
    args: [script, attemptRoot, ...command.slice(2)],
    cwd: taskPackage.root,
    env: isolatedEnvironment(process.env),
    timeoutMs: taskPackage.task.budgets.command_ms,
  })
}

function validateScorerCommand(command) {
  validateNodeCommand(command)
  if (!command[1].startsWith('scorer/')) {
    throw new TaskPackageError('INVALID_FIXTURE', 'scorer_command must name a script under scorer/')
  }
}

function validateNodeCommand(command) {
  if (!Array.isArray(command) || command[0] !== 'node') {
    throw new TaskPackageError('INVALID_FIXTURE', 'Phase 2 model-free commands must use the pinned Node runtime')
  }
  let script
  if (command[1] === '--check') {
    if (command.length !== 3) throw new TaskPackageError('INVALID_FIXTURE', 'Node --check commands must contain exactly three items')
    script = command[2]
  } else {
    if (command.length !== 2) throw new TaskPackageError('INVALID_FIXTURE', 'Node task commands must contain exactly two items')
    script = command[1]
  }
  if (typeof script !== 'string' || !/[.](?:cjs|mjs|js)$/.test(script)
    || isAbsolute(script) || script.includes('\\') || script.split('/').includes('..')) {
    throw new TaskPackageError('INVALID_FIXTURE', `Node task command has an unsafe script path: ${script}`)
  }
}

function validateScorerChecks(result, expectedIds) {
  const actual = result.checks.map((check) => check.id)
  if (new Set(actual).size !== actual.length) throw new Error('scorer check IDs must be unique')
  if (actual.length !== expectedIds.length || actual.some((id, index) => id !== expectedIds[index])) {
    throw new Error(`scorer checks must be exactly ${expectedIds.join(', ')}; found ${actual.join(', ')}`)
  }
}

function parseScorerOutput(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0)
  if (lines.length !== 1) throw new Error(`scorer must emit exactly one JSON line; found ${lines.length}`)
  const result = JSON.parse(lines[0])
  if (result?.schema !== SCORE_SCHEMA) throw new Error(`scorer schema must be ${SCORE_SCHEMA}`)
  return validateTaskCheck(result)
}

function isolatedEnvironment(source) {
  const allowed = new Set(['HOME', 'SYSTEMROOT', 'TEMP', 'TMP', 'TMPDIR', 'USERPROFILE', 'WINDIR'])
  return Object.fromEntries(Object.entries(source).filter(([key]) => allowed.has(key.toUpperCase())))
}

function score(outcome, diffSha256, mutations, canaries, requiredChecks, hiddenCheck, errors) {
  return {
    outcome,
    required_checks: requiredChecks.length,
    passed_checks: requiredChecks.filter((check) => check.passed).length,
    diff_sha256: diffSha256,
    mutations,
    canaries,
    required_check_results: requiredChecks,
    hidden_check: hiddenCheck,
    errors,
  }
}

function invalidScore(outcome, error) {
  return score(outcome, sha256Bytes(Buffer.from(canonicalJson([]), 'utf8')), [], [], [], null, [error])
}

function succeeded(execution) {
  return execution.state === 'exited' && execution.exitCode === 0 && !execution.timedOut && execution.cleanupPassed
}

function processDetail(execution) {
  const stderr = execution.stderr.preview.trim()
  const stdout = execution.stdout.preview.trim()
  return stderr || stdout || `${execution.state} with exit ${execution.exitCode}`
}

async function copyTree(source, destination) {
  const info = await lstat(source)
  if (info.isSymbolicLink()) throw new TaskPackageError('INVALID_FIXTURE', `symbolic links are not allowed: ${source}`)
  if (info.isFile()) {
    await mkdir(dirname(destination), { recursive: true })
    await copyFile(source, destination)
    return
  }
  if (!info.isDirectory()) throw new TaskPackageError('INVALID_FIXTURE', `special files are not allowed: ${source}`)
  await mkdir(destination, { recursive: true })
  const entries = await readdir(source, { withFileTypes: true })
  for (const entry of entries) await copyTree(join(source, entry.name), join(destination, entry.name))
}

async function requireAbsent(path, label) {
  try {
    await lstat(path)
  } catch (error) {
    if (error.code === 'ENOENT') return
    throw error
  }
  throw new TaskPackageError('INVALID_INFRASTRUCTURE', `${label} already exists: ${path}`)
}

function confined(root, path) {
  if (typeof path !== 'string' || path.length === 0 || isAbsolute(path)) throw new TaskPackageError('INVALID_FIXTURE', `path must be relative: ${path}`)
  const resolved = resolve(root, path)
  const relation = relative(root, resolved)
  if (relation === '..' || relation.startsWith(`..${sep}`) || isAbsolute(relation)) {
    throw new TaskPackageError('INVALID_FIXTURE', `path escapes its root: ${path}`)
  }
  return resolved
}

function portable(path) {
  return path.split(sep).join('/')
}

function rejectCaseCollisions(files) {
  const seen = new Map()
  for (const file of files) {
    const folded = file.path.toLowerCase()
    const prior = seen.get(folded)
    if (prior && prior !== file.path) throw new TaskPackageError('INVALID_FIXTURE', `case-folding path collision: ${prior} and ${file.path}`)
    seen.set(folded, file.path)
  }
}

async function requireRealDirectory(path, label) {
  const info = await lstat(path)
  if (info.isSymbolicLink()) throw new TaskPackageError('INVALID_FIXTURE', `${label} cannot be a symbolic link`)
  if (!info.isDirectory()) throw new TaskPackageError('INVALID_FIXTURE', `${label} must be a directory`)
}

async function requireRealFile(path, label) {
  const info = await lstat(path)
  if (info.isSymbolicLink()) throw new TaskPackageError('INVALID_FIXTURE', `${label} cannot be a symbolic link`)
  if (!info.isFile()) throw new TaskPackageError('INVALID_FIXTURE', `${label} must be a file`)
}