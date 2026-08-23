#!/usr/bin/env node
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { cp, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { applyTaskOverlay } from './tasks/package.mjs'

const systemRoot = resolve(fileURLToPath(new URL('.', import.meta.url)))
const repositoryRoot = resolve(systemRoot, '../../..')
const cli = resolve(systemRoot, 'cli.mjs')
const taskRoot = resolve(repositoryRoot, 'qa/benchmarks/agent/tasks/agent_local_logic_fix')
const tempRoot = await mkdtemp(join(tmpdir(), 'camelid-agent-task-cli-'))

try {
  const verified = run(['task-verify', '--task', taskRoot])
  assert.equal(verified.status, 0, verified.stderr)
  const verification = JSON.parse(verified.stdout)
  assert.equal(verification.schema, 'camelid.benchmark.task-package-verification/v1')
  assert.equal(verification.task_id, 'agent_local_logic_fix')
  assert.equal(verification.verified, true)

  const workspace = join(tempRoot, 'workspace')
  const materialized = run(['task-materialize', '--task', taskRoot, '--workspace', workspace])
  assert.equal(materialized.status, 0, materialized.stderr)
  const materialization = JSON.parse(materialized.stdout)
  assert.equal(materialization.setup_passed, true)
  assert.equal(materialization.task_id, 'agent_local_logic_fix')

  const untouched = run(['task-score', '--task', taskRoot, '--workspace', workspace])
  assert.equal(untouched.status, 0, untouched.stderr)
  assert.equal(JSON.parse(untouched.stdout).outcome, 'FAIL_BEHAVIOR')

  await applyTaskOverlay(taskRoot, 'expected/solution', resolve(workspace, 'attempt'))
  const scorePath = join(tempRoot, 'score.json')
  const solved = run(['task-score', '--task', taskRoot, '--workspace', workspace, '--out', scorePath])
  assert.equal(solved.status, 0, solved.stderr)
  const stdoutScore = JSON.parse(solved.stdout)
  const fileScore = JSON.parse(await readFile(scorePath, 'utf8'))
  assert.deepEqual(stdoutScore, fileScore)
  assert.equal(stdoutScore.outcome, 'PASS_COMPARABLE')

  const rematerialize = run(['task-materialize', '--task', taskRoot, '--workspace', workspace])
  assert.equal(rematerialize.status, 1)
  assert.match(rematerialize.stderr, /workspace root already exists/)

  const missingTask = run(['task-verify'])
  assert.equal(missingTask.status, 1)
  assert.match(missingTask.stderr, /--task is required/)

  const invalidTask = join(tempRoot, 'invalid-task')
  await cp(taskRoot, invalidTask, { recursive: true })
  await writeFile(join(invalidTask, 'task.json'), '{not-json\n')
  const invalidScore = run(['task-score', '--task', invalidTask, '--workspace', workspace])
  assert.equal(invalidScore.status, 1, invalidScore.stderr)
  assert.equal(JSON.parse(invalidScore.stdout).outcome, 'INVALID_FIXTURE')
} finally {
  await rm(tempRoot, { recursive: true, force: true })
}

console.log('benchmark Phase 2 task CLI integration: PASS')

function run(args) {
  return spawnSync(process.execPath, [cli, ...args], {
    cwd: repositoryRoot,
    encoding: 'utf8',
    windowsHide: true,
  })
}