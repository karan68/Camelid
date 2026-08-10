#!/usr/bin/env node

import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdtemp, rm, stat, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { basename, dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const scriptDir = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(scriptDir, '..')
const smokeScript = join(scriptDir, 'smoke.mjs')
const tempDir = await mkdtemp(join(tmpdir(), 'camelid-frontend-smoke-harness-'))
const modelPath = join(tempDir, 'fixture-supported-Q8_0.gguf')
const modelBytes = Buffer.from('frontend exact identity fixture')
await writeFile(modelPath, modelBytes)
const modelStats = await stat(modelPath)
const ggufSha256 = createHash('sha256').update(modelBytes).digest('hex')

let loadedId = 'fixture-row'
let loadedPath = modelPath
let includeTerminalUsage = true
let localLaneClass = 'supported'
const streamRequests = []

const server = createServer(async (request, response) => {
  try {
    const body = await readJsonBody(request)
    if (request.url === '/' && request.method === 'GET') {
      response.writeHead(200, { 'content-type': 'text/html' })
      return response.end('<!doctype html><title>Camelid smoke fixture</title>')
    }
    if (request.url === '/api/models/load' && request.method === 'POST') {
      loadedId = body.id
      loadedPath = body.path
      return json(response, 200, currentModel())
    }
    if (request.url === '/v1/health') {
      return json(response, 200, {
        status: 'ok',
        active_model_id: loadedId,
        loaded_now: true,
        generation_ready: true,
      })
    }
    if (request.url === '/v1/models') {
      return json(response, 200, { object: 'list', data: [{ id: loadedId, object: 'model' }] })
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
          admitted: true,
          oracle_qualified: true,
          chat_capable: true,
          generation_capable: true,
          lane_class: localLaneClass,
        }],
      })
    }
    if (request.url === '/api/capabilities') {
      return json(response, 200, {
        support_contract: { current_gate: 'fixture' },
        model_compatibility: [],
        supported_model_families: [],
        planned_model_families: [],
        api_features: [],
      })
    }
    if (request.url === '/v1/chat/completions' && request.method === 'POST' && body?.stream === true) {
      streamRequests.push(body)
      response.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
      })
      response.write('data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
      response.write('data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n')
      response.write('data: {"choices":[{"delta":{"content":"answer"}}]}\n\n')
      response.write('data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
      if (includeTerminalUsage) {
        response.write('data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}\n\n')
      }
      response.write('data: [DONE]\n\n')
      return response.end()
    }
    if (request.url === '/v1/chat/completions' && request.method === 'POST') {
      return json(response, 200, {
        id: 'chatcmpl-test',
        object: 'chat.completion',
        model: loadedId,
        choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
      })
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
  const base = `http://127.0.0.1:${address.port}`
  const commonArgs = [
    smokeScript,
    '--api', base,
    '--frontend', base,
    '--model', modelPath,
    '--model-id', 'fixture-row',
    '--require-generation',
    '--require-local-model',
    '--expect-local-lane-class', 'supported',
    '--expect-gguf-sha256', ggufSha256,
    '--expect-contract-supported', 'true',
    '--expect-webui-chat', 'enabled',
    '--stream-max-tokens', '77',
  ]

  const { stdout } = await execFileAsync(process.execPath, commonArgs, { cwd: frontendRoot })
  assert.match(stdout, /exact local identity .*lane_class=supported/)
  assert.match(stdout, /stream_events=.*finish,usage,done/)
  assert.match(stdout, /terminal_finish_reason=stop; terminal_usage=/)
  assert.equal(streamRequests.length, 1)
  assert.equal(streamRequests[0].max_tokens, 77, 'frontend stream max must be configurable')
  assert.deepEqual(streamRequests[0].stream_options, { include_usage: true }, 'frontend smoke must request terminal usage')

  localLaneClass = 'experimental_implemented'
  const experimentalArgs = replaceOption(
    replaceOption(
      replaceOption(commonArgs, '--expect-local-lane-class', 'experimental_implemented'),
      '--expect-contract-supported',
      'false',
    ),
    '--expect-webui-chat',
    'experimental',
  )
  const experimental = await execFileAsync(process.execPath, experimentalArgs, { cwd: frontendRoot })
  assert.match(experimental.stdout, /lane_class=experimental_implemented/)
  assert.equal(streamRequests.length, 2, 'experimental WebUI chat mode must exercise the stream path')
  localLaneClass = 'supported'

  includeTerminalUsage = false
  await assert.rejects(
    () => execFileAsync(process.execPath, commonArgs, { cwd: frontendRoot }),
    error => {
      assert.match(error.stderr, /exactly one include_usage event; got 0/)
      return true
    },
    'frontend smoke must fail closed when [DONE] arrives without terminal usage',
  )

  console.log('frontend smoke harness self-test passed')
} finally {
  await new Promise(resolvePromise => server.close(resolvePromise))
  await rm(tempDir, { recursive: true, force: true })
}

function currentModel() {
  return {
    id: loadedId,
    path: loadedPath,
    gguf: { metadata: { general: { file_type: 7 } } },
    tokenizer: { status: 'available' },
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

function replaceOption(args, flag, value) {
  const replaced = [...args]
  const index = replaced.indexOf(flag)
  assert.notEqual(index, -1, `missing fixture option ${flag}`)
  replaced[index + 1] = value
  return replaced
}
