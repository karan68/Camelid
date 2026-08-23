#!/usr/bin/env node
import assert from 'node:assert/strict'
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:net'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { NativeAdapterError, runNativeAgentAttempt } from './adapters/native-camelid.mjs'

const repositoryRoot = resolve(fileURLToPath(new URL('../../..', import.meta.url)))
const taskRoot = resolve(repositoryRoot, 'qa/benchmarks/agent/tasks/agent_local_logic_fix')
const tempRoot = await mkdtemp(join(tmpdir(), 'camelid-native-adapter-'))
const modelPath = join(tempRoot, 'model.gguf')
await writeFile(modelPath, 'synthetic model bytes\n')

try {
  await assert.rejects(
    run('answered', false),
    (error) => error instanceof NativeAdapterError && error.outcome === 'INVALID_INFRASTRUCTURE',
  )
  await assert.rejects(
    run('over-budget', true, 300001),
    (error) => error instanceof NativeAdapterError && error.outcome === 'INVALID_FIXTURE',
  )
  assert.equal(await exists(join(tempRoot, 'workspace-over-budget')), false)

  const answered = await run('answered', true)
  assert.equal(answered.attempt.terminal.class, 'answered')
  assert.equal(answered.attempt.terminal.exit_code, 0)
  assert.equal(answered.attempt.score.outcome, 'PASS_COMPARABLE')
  assert.equal(answered.repository_score.outcome, 'PASS_COMPARABLE')
  assertSecurityArgs(answered.args)

  const failed = await run('failed', true)
  assert.equal(failed.attempt.terminal.class, 'failed')
  assert.equal(failed.attempt.terminal.exit_code, 1)
  assert.equal(failed.attempt.score.outcome, 'FAIL_AGENT_TERMINAL')

  const inconclusive = await run('inconclusive', true)
  assert.equal(inconclusive.attempt.terminal.class, 'inconclusive')
  assert.equal(inconclusive.attempt.terminal.exit_code, 3)
  assert.equal(inconclusive.attempt.score.outcome, 'FAIL_AGENT_TERMINAL')

  const timedOut = await run('timeout', true, 500)
  assert.equal(timedOut.attempt.terminal.class, 'timed_out')
  assert.equal(timedOut.attempt.score.outcome, 'INCONCLUSIVE_TIMEOUT')
  assert.equal(timedOut.execution.cleanupPassed, true)
  await assertPortReusable(addrFromArgs(timedOut.args))
  assert.equal(await readFile(join(timedOut.workspace_root, 'canary', 'outside.txt'), 'utf8'), 'camelid-benchmark-canary/v1\nagent_local_logic_fix\noutside_task_root\n')
} finally {
  await rm(tempRoot, { recursive: true, force: true })
}

console.log('benchmark Phase 3 native adapter canned lifecycle: PASS')

async function run(mode, disposableBoundary, timeoutMs = 5000) {
  const scriptPath = await fakeCandidate(mode)
  return runNativeAgentAttempt({
    taskRoot,
    workspaceRoot: join(tempRoot, `workspace-${mode}`),
    binaryPath: process.execPath,
    modelPath,
    campaignId: `native-${mode}`,
    sourceSha: 'a'.repeat(40),
    attempt: 0,
    timeoutMs,
    disposableBoundary,
    env: {
      ...process.env,
      CAMELID_API_KEY: 'must-not-reach-candidate',
      CAMELID_PRODUCTION: 'must-not-reach-candidate',
    },
    syntheticCandidate: true,
    syntheticCandidatePrefix: [scriptPath],
  })
}

async function fakeCandidate(mode) {
  const scriptPath = join(tempRoot, `fake-${mode}.mjs`)
  await writeFile(scriptPath, candidateSource(mode))
  return scriptPath
}

function candidateSource(mode) {
  return `
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { readFile, writeFile } from 'node:fs/promises'

const args = process.argv.slice(2)
assert.deepEqual(args.slice(0, 2), ['agent', 'exec'])
assert.equal(process.env.CAMELID_API_KEY, undefined)
assert.equal(process.env.CAMELID_PRODUCTION, undefined)
for (const forbidden of ['--allow-net', '--allow-fs', '--allow-mcp']) assert.equal(args.includes(forbidden), false)
for (const required of ['--model', '--addr', '--workdir', '--max-steps', '--max-tokens', '--shell-sandbox', '--shell-timeout', '--today-is-a-good-day-to-die']) assert.ok(args.includes(required), required)
assert.equal(value('--shell-sandbox'), 'sandboxed')
const workdir = value('--workdir')
if (${JSON.stringify(mode)} === 'answered') {
  const path = workdir + '/src/pricing.cjs'
  const source = await readFile(path, 'utf8')
  await writeFile(path, source.replace('subtotalCents > 10000', 'subtotalCents >= 10000'))
  process.stdout.write('done\\n')
  process.exit(0)
}
if (${JSON.stringify(mode)} === 'failed') process.exit(1)
if (${JSON.stringify(mode)} === 'inconclusive') process.exit(3)
if (${JSON.stringify(mode)} === 'timeout') {
  const child = spawn(process.execPath, ['-e', \`require('net').createServer().listen(\${JSON.stringify(value('--addr').split(':').at(-1) * 1)}, '127.0.0.1'); setInterval(() => {}, 1000)\`], { stdio: 'ignore' })
  child.unref()
  setInterval(() => {}, 1000)
}
function value(flag) { const index = args.indexOf(flag); assert.ok(index >= 0); return args[index + 1] }
`
}

function assertSecurityArgs(args) {
  for (const flag of ['--allow-net', '--allow-fs', '--allow-mcp']) assert.equal(args.includes(flag), false)
  assert.ok(args.includes('--today-is-a-good-day-to-die'))
  assert.equal(args[args.indexOf('--shell-sandbox') + 1], 'sandboxed')
}

function addrFromArgs(args) {
  return args[args.indexOf('--addr') + 1]
}

async function assertPortReusable(addr) {
  const port = Number(addr.split(':').at(-1))
  const server = createServer()
  await new Promise((resolveListen, reject) => {
    server.once('error', reject)
    server.listen(port, '127.0.0.1', resolveListen)
  })
  await new Promise((resolveClose) => server.close(resolveClose))
}

async function exists(path) {
  try {
    await access(path)
    return true
  } catch (error) {
    if (error.code === 'ENOENT') return false
    throw error
  }
}