#!/usr/bin/env node

import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { basename, dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const scriptDir = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(scriptDir, '..')
const bundleScript = join(scriptDir, 'model-promotion-smoke-bundle.mjs')
const tempDir = await mkdtemp(join(tmpdir(), 'camelid-model-promotion-smoke-'))
const modelPath = join(tempDir, 'fixture-supported-Q8_0.gguf')
const modelBytes = Buffer.from('exact promotion fixture bytes')
await writeFile(modelPath, modelBytes)
const modelStats = await stat(modelPath)
const ggufSha256 = createHash('sha256').update(modelBytes).digest('hex')

let laneClass = 'supported'
let completionCalls = 0
let chatCalls = 0
let loadedId = 'fixture-row'
let loadedPath = modelPath
let includeTimings = true
let lastLoadReplace = null

const server = createServer(async (request, response) => {
  try {
    const body = await readJsonBody(request)
    if (request.url === '/api/models/load' && request.method === 'POST') {
      loadedId = body.id
      loadedPath = body.path
      lastLoadReplace = body.replace ?? null
      return json(response, 200, currentModel())
    }
    if (request.url === '/v1/health') {
      return json(response, 200, {
        status: 'ok',
        active_model_id: loadedId,
        loaded_now: true,
        generation_ready: true,
        execution_plan: {
          selected_backend: 'metal_resident_fixture_runtime',
          prefill_path: 'fixture_metal_resident_prefill',
          decode_path: 'fixture_metal_resident_decode',
        },
      })
    }
    if (request.url === '/api/models/current') return json(response, 200, currentModel())
    if (request.url === '/api/models/local') {
      return json(response, 200, {
        models_dir: tempDir,
        models: [{
          filename: basename(modelPath),
          size_bytes: modelStats.size,
          architecture: 'fixture',
          quantization: 'Q8_0',
          lane_class: laneClass,
        }],
      })
    }
    if (request.url === '/v1/models') {
      return json(response, 200, { object: 'list', data: [{ id: loadedId, object: 'model' }] })
    }
    if (request.url === '/api/capabilities') return json(response, 200, { model_compatibility: [] })
    if (request.url === '/v1/completions') {
      completionCalls += 1
      return json(response, 500, { error: { message: 'legacy completions must be skipped by this test' } })
    }
    if (request.url === '/v1/chat/completions') {
      chatCalls += 1
      return json(response, 200, generationResponse())
    }
    return json(response, 404, { error: { message: `unhandled ${request.method} ${request.url}` } })
  } catch (error) {
    return json(response, 500, { error: { message: error instanceof Error ? error.message : String(error) } })
  }
})

await new Promise((resolvePromise, reject) => {
  server.once('error', reject)
  server.listen(0, '127.0.0.1', resolvePromise)
})

try {
  const address = server.address()
  assert.ok(address && typeof address === 'object')
  const apiBase = `http://127.0.0.1:${address.port}`
  const outDir = join(tempDir, 'pass')
  const { stdout } = await execFileAsync(process.execPath, [
    bundleScript,
    '--api', apiBase,
    '--model', modelPath,
    '--model-id', 'fixture-row',
    '--out-dir', outDir,
    '--chat-only',
    '--replace-loaded-model',
    '--skip-frontend',
    '--expect-local-lane-class', 'supported',
    '--expect-gguf-sha256', ggufSha256,
    '--expect-selected-backend', 'metal_resident_fixture_runtime',
    '--expect-prefill-path', 'fixture_metal_resident_prefill',
    '--expect-decode-path', 'fixture_metal_resident_decode',
  ], { cwd: repoRoot })
  assert.equal(stdout.includes('legacy completions'), false)
  assert.equal(completionCalls, 0, 'chat-only mode must not probe /v1/completions')
  assert.equal(chatCalls, 1, 'chat-only mode must still probe /v1/chat/completions')
  assert.equal(lastLoadReplace, true, 'explicit replacement must reach the backend load request')

  const summary = await readJson(join(outDir, 'summary.json'))
  assert.equal(summary.passed, true)
  assert.equal(summary.skip_completions, true)
  assert.equal(summary.steps.v1_completions.skipped, true)
  assert.equal(summary.steps.exact_model_identity.ok, true)
  assert.equal(summary.steps.execution_plan.ok, true)
  assert.equal(summary.steps.execution_plan_final.ok, true)
  const skipped = await readJson(join(outDir, 'completion.skipped.json'))
  assert.equal(skipped.skipped, true)
  const identity = await readJson(join(outDir, 'exact-model-identity.json'))
  assert.equal(identity.passed, true)
  assert.equal(identity.local_model.lane_class, 'supported')
  assert.equal(identity.current.lane.gguf_sha256, ggufSha256)
  const timings = await readJson(join(outDir, 'generation-timings.summary.json'))
  assert.equal(timings.inputs.length, 1, 'chat-only timing summary must contain only the chat response')
  assert.equal(timings.inputs[0].file, 'chat.response.json')
  const executionPlan = await readJson(join(outDir, 'execution-plan.json'))
  assert.equal(executionPlan.passed, true)
  assert.equal(executionPlan.actual.selected_backend, 'metal_resident_fixture_runtime')
  const executionPlanFinal = await readJson(join(outDir, 'execution-plan-final.json'))
  assert.equal(executionPlanFinal.passed, true)

  includeTimings = false
  const noTimingsDir = join(tempDir, 'no-timings')
  await execFileAsync(process.execPath, [
    bundleScript,
    '--api', apiBase,
    '--model', modelPath,
    '--model-id', 'fixture-row',
    '--out-dir', noTimingsDir,
    '--chat-only',
    '--skip-frontend',
    '--expect-local-lane-class', 'supported',
    '--expect-gguf-sha256', ggufSha256,
  ], { cwd: repoRoot })
  const noTimingsSummary = await readJson(join(noTimingsDir, 'summary.json'))
  assert.equal(noTimingsSummary.passed, true)
  assert.equal(noTimingsSummary.steps.generation_timings.skipped, true)
  const skippedTimings = await readJson(join(noTimingsDir, 'generation-timings.skipped.json'))
  assert.match(skippedTimings.reason, /did not emit camelid\.timings_ms/)

  laneClass = 'experimental_implemented'
  const failDir = join(tempDir, 'lane-mismatch')
  await assert.rejects(
    () => execFileAsync(process.execPath, [
      bundleScript,
      '--api', apiBase,
      '--model', modelPath,
      '--model-id', 'fixture-row',
      '--out-dir', failDir,
      '--skip-completions',
      '--skip-frontend',
      '--expect-contract-supported', 'true',
      '--expect-gguf-sha256', ggufSha256,
    ], { cwd: repoRoot }),
  )
  const failedSummary = await readJson(join(failDir, 'summary.json'))
  assert.equal(failedSummary.passed, false)
  assert.match(failedSummary.error, /local lane class.*supported.*experimental_implemented/)

  laneClass = 'supported'
  const executionPlanFailDir = join(tempDir, 'execution-plan-mismatch')
  await assert.rejects(
    () => execFileAsync(process.execPath, [
      bundleScript,
      '--api', apiBase,
      '--model', modelPath,
      '--model-id', 'fixture-row',
      '--out-dir', executionPlanFailDir,
      '--chat-only',
      '--skip-frontend',
      '--expect-local-lane-class', 'supported',
      '--expect-gguf-sha256', ggufSha256,
      '--expect-selected-backend', 'cpu_reference',
    ], { cwd: repoRoot }),
  )
  const executionPlanFailedSummary = await readJson(join(executionPlanFailDir, 'summary.json'))
  assert.equal(executionPlanFailedSummary.passed, false)
  assert.match(executionPlanFailedSummary.error, /execution plan selected_backend.*cpu_reference.*metal_resident_fixture_runtime/)

  console.log('model-promotion-smoke-bundle self-test passed')
} finally {
  await new Promise(resolvePromise => server.close(resolvePromise))
  await rm(tempDir, { recursive: true, force: true })
}

function currentModel() {
  return {
    id: loadedId,
    path: loadedPath,
    lane: {
      model_id: loadedId,
      gguf_filename: basename(modelPath),
      gguf_sha256: ggufSha256,
      architecture: 'fixture',
      quantization: 'Q8_0',
      tokenizer_kind: 'gpt2_bpe',
      camelid_version: 'test',
      camelid_commit: 'test',
    },
  }
}

function generationResponse() {
  const response = {
    id: 'chatcmpl-test',
    object: 'chat.completion',
    model: loadedId,
    choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
    camelid: {},
  }
  if (includeTimings) {
    response.camelid.timings_ms = {
        tokenize: 0.1,
        weight_load: 0,
        weight_cache_hit: true,
        session_create: 0.1,
        generate: 0.5,
        generation: { forward_total: 0.4, layers_total: 0.3, logits: 0.1, sample: 0 },
        layers: [],
    }
  }
  return response
}

async function readJsonBody(request) {
  const chunks = []
  for await (const chunk of request) chunks.push(chunk)
  const text = Buffer.concat(chunks).toString('utf8')
  return text ? JSON.parse(text) : null
}

function json(response, status, payload) {
  response.writeHead(status, { 'content-type': 'application/json' })
  response.end(JSON.stringify(payload))
}

async function readJson(path) {
  return JSON.parse(await readFile(path, 'utf8'))
}
