#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readdir, readFile } from 'node:fs/promises'
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

const reportDir = resolve(root, 'qa/model-qualification')
const committedReports = (await readdir(reportDir)).filter((name) => name.endsWith('-report.json'))
assert.ok(committedReports.length >= 2, 'the committed Phase 1 source/bootstrap reports must stay under validation')
for (const name of committedReports) {
  const candidate = JSON.parse(await readFile(resolve(reportDir, name), 'utf8'))
  assert.deepEqual(validateQualificationReport(candidate, name), [], `${name} must satisfy the report contract`)
}
assert.equal(expectedOverall(report.stages), 'fail')
assert.ok(mutated((candidate) => { candidate.host.hostname = 'private-machine' }).some((error) => error.includes('raw hostname')))
assert.ok(mutated((candidate) => { candidate.artifact.path = 'C:\\Users\\private\\model.gguf' }).some((error) => error.includes('absolute local path')))
for (const privatePath of [
  '/tmp/private/model.gguf',
  '//server/private/model.gguf',
  '///tmp/private/model.gguf',
  '/var/tmp/private/model.gguf',
  '/mnt/models/private.gguf',
  'file://localhost/tmp/private/model.gguf',
]) {
  assert.ok(
    mutated((candidate) => { candidate.artifact.path = privatePath }).some((error) => error.includes('absolute local path')),
    'Unix local absolute paths must be rejected',
  )
}
assert.ok(mutated((candidate) => { candidate.stages.metadata.command = ['<camelid>', 'inspect', '/workspace/models/private.gguf'] }).some((error) => error.includes('absolute local path')))
for (const assignedPath of [
  '--model=/tmp/private/model.gguf',
  '--model=C:\\Users\\private\\model.gguf',
  '--model=file:///tmp/private/model.gguf',
]) {
  assert.ok(
    mutated((candidate) => { candidate.stages.metadata.command = assignedPath }).some((error) => error.includes('absolute local path')),
    'absolute command-assignment paths must be rejected',
  )
}
assert.ok(
  mutated((candidate) => { candidate.artifact.path = '/api/private/model.gguf' }).some((error) => error.includes('absolute local path')),
  'an artifact path must not bypass privacy checks by starting with an API-looking directory',
)
assert.ok(
  mutated((candidate) => { candidate.artifact.path = '/completion' }).some((error) => error.includes('absolute local path')),
  'a known route literal is safe only in a route/endpoint/command field',
)
assert.deepEqual(
  mutated((candidate) => {
    candidate.stages.api_webui.public_routes = [
      '/api/models',
      '/api/models/example-id/status',
      '/v1/chat/completions',
    ]
    candidate.stages.api_webui.command = [
      '/completion',
      '/health',
      '/apply-template',
      '/tokenize',
    ]
    candidate.source_url = 'https://huggingface.co/org/repo'
  }),
  [],
  'public URLs and API route literals must not be mistaken for host paths',
)
assert.ok(mutated((candidate) => { candidate.source_dirty = false; candidate.overall_status = 'pass' }).some((error) => error.includes('fail-closed')))
assert.ok(mutated((candidate) => { delete candidate.stages.context }).some((error) => error.includes('missing stage object')))

console.log('test-check-model-qualification-report: all checks passed')
