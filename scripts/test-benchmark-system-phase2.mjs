#!/usr/bin/env node
import { spawnSync } from 'node:child_process'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(fileURLToPath(new URL('..', import.meta.url)))
const tests = [
  'tools/bench/system/test-agent-contracts.mjs',
  'tools/bench/system/test-task-foundation.mjs',
  'tools/bench/system/test-agent-task-cli.mjs',
]

for (const test of tests) {
  console.log(`== ${test}`)
  const result = spawnSync(process.execPath, [resolve(root, test)], {
    cwd: root,
    stdio: 'inherit',
    windowsHide: true,
  })
  if (result.status !== 0) process.exit(result.status ?? 1)
}

console.log(`benchmark Phase 2 validation: PASS (${tests.length} tests)`)