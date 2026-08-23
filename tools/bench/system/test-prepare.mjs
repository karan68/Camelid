#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { validatePlan } from './lib/contracts.mjs'
import { buildArgs, isolatedBuildEnv, PreparationError, prepareArms } from './prepare.mjs'

const root = resolve(fileURLToPath(new URL('.', import.meta.url)))
const plan = JSON.parse(await readFile(resolve(root, 'fixtures/schemas/valid-plan.json'), 'utf8'))
validatePlan(plan)

const calls = []
const prepared = await prepareArms(plan, {
  execute: async (options) => {
    calls.push(options)
    return success(options.args[0] === '--version' ? 'camelid 0.6.1\n' : '')
  },
  inspectBinary: async (path) => ({
    path,
    sha256: path.includes('base')
      ? '1111111111111111111111111111111111111111111111111111111111111111'
      : '2222222222222222222222222222222222222222222222222222222222222222',
    sizeBytes: 42,
  }),
})

assert.equal(prepared.length, 2)
assert.deepEqual(calls.map((call) => call.args[0]), ['+1.89.0', '--version', '+1.89.0', '--version'])
assert.deepEqual(calls[0].args, buildArgs(plan.source_arms[0]))
assert.equal(calls[0].cwd, '<repo>/base')
assert.equal(calls[2].cwd, '<repo>/head')
assert.equal(prepared[0].binary_sha256, '1111111111111111111111111111111111111111111111111111111111111111')
assert.equal(prepared[1].reported_version, 'camelid 0.6.1')
assert.equal(calls[0].env.CARGO_INCREMENTAL, '0')
assert.equal(calls[0].env.CARGO_NET_OFFLINE, 'true')

const isolated = isolatedBuildEnv({ network: 'deny' }, {
  PATH: '<path>',
  CAMELID_GPU: 'on',
  CARGO_TARGET_DIR: '<wrong-target>',
  CARGO_ENCODED_RUSTFLAGS: '<flags>',
  RUSTFLAGS: '<flags>',
})
assert.deepEqual(isolated, {
  PATH: '<path>',
  CARGO_INCREMENTAL: '0',
  CARGO_NET_OFFLINE: 'true',
})

const failedCalls = []
await assert.rejects(
  () => prepareArms(plan, {
    execute: async (options) => {
      failedCalls.push(options)
      return failure('compiler failed')
    },
    inspectBinary: async () => assert.fail('binary inspection must not run after build failure'),
  }),
  (error) => {
    assert.ok(error instanceof PreparationError)
    assert.equal(error.code, 'build_failed')
    assert.equal(error.armId, 'base')
    assert.match(error.message, /compiler failed/)
    return true
  },
)
assert.equal(failedCalls.length, 1, 'preparation must stop at the first failed arm')

const sharedTarget = structuredClone(plan)
sharedTarget.source_arms[1].build.target_dir = sharedTarget.source_arms[0].build.target_dir
await assert.rejects(
  () => prepareArms(sharedTarget),
  /target_dir duplicates/,
)

console.log('benchmark Phase 1 isolated arm preparation: PASS')

function success(stdout) {
  return result('exited', 0, stdout, '')
}

function failure(stderr) {
  return result('exited', 1, '', stderr)
}

function result(state, exitCode, stdout, stderr) {
  return {
    state,
    exitCode,
    signal: null,
    timedOut: false,
    durationMs: 10,
    cleanupPassed: true,
    cleanupDetail: null,
    error: null,
    stdout: output(stdout),
    stderr: output(stderr),
  }
}

function output(text) {
  return {
    preview: text,
    totalBytes: Buffer.byteLength(text),
    capturedBytes: Buffer.byteLength(text),
    truncated: false,
  }
}
