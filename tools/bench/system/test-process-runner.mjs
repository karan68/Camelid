#!/usr/bin/env node
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import net from 'node:net'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { killProcessTree, runProcess } from './process/runner.mjs'

const root = resolve(fileURLToPath(new URL('.', import.meta.url)))
const fixture = resolve(root, 'fixtures/process/child.mjs')
const temp = await mkdtemp(join(tmpdir(), 'camelid-bench-process-'))
let unrelated

try {
  const normal = await run('normal')
  assert.equal(normal.state, 'exited')
  assert.equal(normal.exitCode, 0)
  assert.equal(normal.stdout.preview, 'normal stdout\n')
  assert.equal(normal.stderr.preview, 'normal stderr\n')
  assert.equal(normal.cleanupPassed, true)

  const failed = await run('fail')
  assert.equal(failed.state, 'exited')
  assert.equal(failed.exitCode, 7)
  assert.match(failed.stderr.preview, /intentional failure/)

  const stdoutFile = join(temp, 'large.stdout.log')
  const large = await run('large', { maxCaptureBytes: 1024, stdoutFile })
  assert.equal(large.exitCode, 0)
  assert.equal(large.stdout.capturedBytes, 1024)
  assert.equal(large.stdout.totalBytes, 128 * 1024)
  assert.equal(large.stdout.truncated, true)
  assert.equal((await readFile(stdoutFile)).length, 128 * 1024)

  const hung = await run('hang', { timeoutMs: 250 })
  assert.equal(hung.state, 'timed_out')
  assert.equal(hung.timedOut, true)
  assert.equal(hung.cleanupPassed, true)

  unrelated = spawn(process.execPath, [fixture, 'hang'], {
    stdio: 'ignore',
    windowsHide: true,
    detached: process.platform !== 'win32',
  })
  const descendant = await run('descendant', { timeoutMs: 1000 })
  assert.equal(descendant.state, 'timed_out')
  assert.equal(descendant.cleanupPassed, true)
  const match = descendant.stdout.preview.match(/BENCH_CHILD_PORT=(\d+)/)
  assert.ok(match, `descendant did not report a port: ${descendant.stdout.preview}`)
  await assertPortClosed(Number(match[1]))
  assert.equal(processStillExists(unrelated.pid), true, 'cleanup must not kill unrelated processes')

  const missing = await runProcess({
    file: resolve(temp, 'missing-executable'),
    args: [],
    timeoutMs: 1000,
  })
  assert.equal(missing.state, 'spawn_failed')
  assert.equal(missing.cleanupPassed, true)

  await assert.rejects(() => runProcess({ file: '', timeoutMs: 1 }), /non-empty/)
  await assert.rejects(() => runProcess({ file: process.execPath, timeoutMs: 0 }), /positive/)
} finally {
  if (unrelated?.pid && processStillExists(unrelated.pid)) killProcessTree(unrelated.pid)
  await rm(temp, { recursive: true, force: true })
}

console.log('benchmark Phase 1 owned process runner: PASS')

function run(mode, overrides = {}) {
  return runProcess({
    file: process.execPath,
    args: [fixture, mode],
    timeoutMs: 5000,
    ...overrides,
  })
}

function processStillExists(pid) {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    if (error.code === 'ESRCH') return false
    throw error
  }
}

async function assertPortClosed(port) {
  await new Promise((resolveClosed, reject) => {
    const socket = net.createConnection({ host: '127.0.0.1', port })
    socket.setTimeout(1000)
    socket.once('connect', () => {
      socket.destroy()
      reject(new Error(`descendant listener still accepts connections on ${port}`))
    })
    socket.once('error', () => resolveClosed())
    socket.once('timeout', () => {
      socket.destroy()
      reject(new Error(`descendant listener did not close within the audit window on ${port}`))
    })
  })
}
