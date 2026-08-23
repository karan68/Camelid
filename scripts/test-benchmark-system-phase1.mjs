#!/usr/bin/env node
import { spawnSync } from 'node:child_process'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(fileURLToPath(new URL('..', import.meta.url)))
const tests = [
  'tools/bench/system/test-schemas.mjs',
  'tools/bench/system/test-bench-generate-parser.mjs',
  'tools/bench/system/test-stats.mjs',
  'tools/bench/system/test-planner.mjs',
  'tools/bench/system/test-prepare.mjs',
  'tools/bench/system/test-process-runner.mjs',
  'tools/bench/system/test-runtime-adapter.mjs',
  'tools/bench/system/test-bundle.mjs',
  'tools/bench/system/test-cli.mjs',
  'tools/bench/system/test-safety.mjs',
  'tools/bench/test-v0.1-benchmark-harness.mjs',
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

console.log(`benchmark Phase 1 validation: PASS (${tests.length} tests)`)
