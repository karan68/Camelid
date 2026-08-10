#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expectedOverall, validateQualificationReport } from './check-model-qualification-report.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const path = resolve(root, 'qa/model-qualification/qwen2.5-0.5b-q8-bootstrap-report.json')
const report = JSON.parse(await readFile(path, 'utf8'))
const mutated = (change) => {
  const candidate = structuredClone(report)
  change(candidate)
  return validateQualificationReport(candidate, 'test')
}

assert.deepEqual(validateQualificationReport(report, 'bootstrap'), [])
assert.equal(expectedOverall(report.stages), 'fail')
assert.ok(mutated((candidate) => { candidate.host.hostname = 'private-machine' }).some((error) => error.includes('raw hostname')))
assert.ok(mutated((candidate) => { candidate.artifact.path = 'C:\\Users\\private\\model.gguf' }).some((error) => error.includes('absolute local path')))
assert.ok(mutated((candidate) => { candidate.source_dirty = false; candidate.overall_status = 'pass' }).some((error) => error.includes('fail-closed')))
assert.ok(mutated((candidate) => { delete candidate.stages.context }).some((error) => error.includes('missing stage object')))

console.log('test-check-model-qualification-report: all checks passed')
