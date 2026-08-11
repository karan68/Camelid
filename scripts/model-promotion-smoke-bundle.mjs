#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { mkdir, stat, writeFile } from 'node:fs/promises'
import { basename, resolve, join } from 'node:path'

const args = parseArgs(process.argv.slice(2))

if (args.has('help') || args.has('h')) {
  console.log(`Usage: node scripts/model-promotion-smoke-bundle.mjs [options]

Capture one exact-row API/WebUI promotion smoke bundle against a running Camelid backend/frontend.

Required options:
  --model <path>                       Exact GGUF model path to load
  --out-dir <path>                     Output artifact directory

Common options:
  --api <url>                          Camelid API base (default: CAMELID_API_BASE or http://127.0.0.1:8181)
  --frontend <url>                     Frontend URL (default: CAMELID_FRONTEND_URL or http://127.0.0.1:4175)
  --model-id <id>                      Runtime model id to load (default: CAMELID_SMOKE_MODEL_ID or smoke-model)
  --message <text>                     Prompt/chat message (default: hello)
  --max-tokens <n>                     Positive token budget (default: 1)
  --stream-max-tokens <n>              Frontend streaming token budget (default: 24)
  --temperature <number>               Sampling temperature (default: 0)
  --skip-completions, --chat-only       Do not call the unsupported legacy /v1/completions route
  --replace-loaded-model                Replace an auto-selected/resident model while loading the qualified row
  --skip-frontend                      Capture API artifacts only
  --allow-guarded-chat                 Let frontend smoke pass guarded-chat state instead of requiring generation
  --frontend-script <path>             Frontend smoke script (default: frontend/scripts/smoke.mjs)
  --timings-script <path>              Timing summary script (default: scripts/summarize-generation-timings.mjs)
  --expect-compatibility-row <id>      Assert exact frontend/API compatibility row
  --expect-compatibility-status <text> Assert exact compatibility status
  --expect-contract-supported <bool>   Assert frontend contract support state
  --expect-webui-chat <state>          Assert WebUI chat state, e.g. enabled
  --expect-local-lane-class <class>    Assert /api/models/local lane_class for the exact file
  --expect-gguf-sha256 <hex>           Assert the loaded lane's exact GGUF SHA-256
  --expect-selected-backend <name>    Assert /v1/health execution_plan.selected_backend
  --expect-prefill-path <name>        Assert /v1/health execution_plan.prefill_path
  --expect-decode-path <name>         Assert /v1/health execution_plan.decode_path
  --help, -h                           Print this help without writing files
`)
  process.exit(0)
}

const apiBase = (args.get('api') || args.get('backend') || process.env.CAMELID_API_BASE || 'http://127.0.0.1:8181').replace(/\/$/, '')
const frontendUrl = (args.get('frontend') || process.env.CAMELID_FRONTEND_URL || 'http://127.0.0.1:4175').replace(/\/$/, '')
const modelPath = args.get('model') ? resolve(args.get('model')) : null
const modelId = args.get('model-id') || process.env.CAMELID_SMOKE_MODEL_ID || 'smoke-model'
const outDir = args.get('out-dir') ? resolve(args.get('out-dir')) : null
const message = args.get('message') ?? 'hello'
const maxTokens = parsePositiveInt('max-tokens', args.get('max-tokens') || '1')
const streamMaxTokens = parsePositiveInt('stream-max-tokens', args.get('stream-max-tokens') || '24')
const temperature = Number.parseFloat(args.get('temperature') || '0')
const skipCompletions = args.has('skip-completions') || args.has('skip-completion') || args.has('chat-only')
const replaceLoadedModel = args.has('replace-loaded-model') || args.has('replace')
const skipFrontend = args.has('skip-frontend')
const allowGuardedChat = args.has('allow-guarded-chat')
const frontendScript = resolve(args.get('frontend-script') || 'frontend/scripts/smoke.mjs')
const timingsScript = resolve(args.get('timings-script') || 'scripts/summarize-generation-timings.mjs')
const expectCompatibilityRow = args.get('expect-compatibility-row') || ''
const expectCompatibilityStatus = args.get('expect-compatibility-status') || ''
const expectContractSupported = args.get('expect-contract-supported') || ''
const expectWebUiChat = args.get('expect-webui-chat') || ''
const expectedContractSupported = parseOptionalBoolean('expect-contract-supported', expectContractSupported)
const expectLocalLaneClass = args.get('expect-local-lane-class') || (expectedContractSupported === true ? 'supported' : '')
const expectGgufSha256 = normalizeSha256(args.get('expect-gguf-sha256') || '')
const expectedExecutionPlan = {
  selected_backend: args.get('expect-selected-backend'),
  prefill_path: args.get('expect-prefill-path'),
  decode_path: args.get('expect-decode-path'),
}

if (!modelPath) throw new Error('--model is required')
if (!outDir) throw new Error('--out-dir is required')
if (!Number.isFinite(temperature)) throw new Error(`--temperature must be numeric, got ${args.get('temperature')}`)
if (expectLocalLaneClass && !['supported', 'experimental_implemented', 'unsupported'].includes(expectLocalLaneClass)) {
  throw new Error(`--expect-local-lane-class must be one of supported, experimental_implemented, unsupported; got ${expectLocalLaneClass}`)
}

await mkdir(outDir, { recursive: true })

const summary = {
  schema: 'camelid.model-promotion.smoke-bundle.v1',
  generated_utc: new Date().toISOString(),
  api_base: apiBase,
  frontend_url: frontendUrl,
  model_path: modelPath,
  model_id: modelId,
  message,
  max_tokens: maxTokens,
  stream_max_tokens: streamMaxTokens,
  temperature,
  skip_completions: skipCompletions,
  replace_loaded_model: replaceLoadedModel,
  allow_guarded_chat: allowGuardedChat,
  skip_frontend: skipFrontend,
  steps: {},
  passed: false,
}

try {
  const healthBefore = await tryFetchJson(`${apiBase}/v1/health`)
  await recordStep('health_before', healthBefore, join(outDir, 'health-before.json'))

  const loadRequest = { path: modelPath, id: modelId, ...(replaceLoadedModel ? { replace: true } : {}) }
  await writeJson(join(outDir, 'load.request.json'), loadRequest)
  const loadResponse = await fetchJson(`${apiBase}/api/models/load`, {
    method: 'POST',
    body: JSON.stringify(loadRequest),
  })
  await recordStep('load', loadResponse, join(outDir, 'load.response.json'))

  const current = await fetchJson(`${apiBase}/api/models/current`)
  await recordStep('current_model', current, join(outDir, 'current-model.json'))

  const localModels = await fetchJson(`${apiBase}/api/models/local`)
  await recordStep('local_models', localModels, join(outDir, 'local-models.json'))

  const exactIdentity = await exactModelIdentityEvidence({
    modelPath,
    modelId,
    current,
    localModels,
    expectLocalLaneClass,
    expectGgufSha256,
  })
  await recordStep('exact_model_identity', exactIdentity, join(outDir, 'exact-model-identity.json'))

  const models = await fetchJson(`${apiBase}/v1/models`)
  await recordStep('v1_models', models, join(outDir, 'v1-models.json'))

  const capabilities = await tryFetchJson(`${apiBase}/api/capabilities`)
  await recordStep('capabilities', capabilities, join(outDir, 'capabilities.json'))

  const timingInputs = []
  if (skipCompletions) {
    await recordStep('v1_completions', {
      skipped: true,
      reason: 'chat-only qualification requested; this model does not claim legacy /v1/completions support',
    }, join(outDir, 'completion.skipped.json'))
  } else {
    const completionRequest = {
      model: modelId,
      prompt: message,
      max_tokens: maxTokens,
      stream: false,
      temperature,
    }
    await writeJson(join(outDir, 'completion.request.json'), completionRequest)
    const completionResponsePath = join(outDir, 'completion.response.json')
    const completionResponse = await fetchJson(`${apiBase}/v1/completions`, {
      method: 'POST',
      body: JSON.stringify(completionRequest),
    })
    await recordStep('v1_completions', completionResponse, completionResponsePath)
    if (completionResponse?.camelid?.timings_ms) timingInputs.push(completionResponsePath)
  }

  const chatRequest = {
    model: modelId,
    messages: [{ role: 'user', content: message }],
    max_tokens: maxTokens,
    stream: false,
    temperature,
  }
  await writeJson(join(outDir, 'chat.request.json'), chatRequest)
  const chatResponsePath = join(outDir, 'chat.response.json')
  const chatResponse = await fetchJson(`${apiBase}/v1/chat/completions`, {
    method: 'POST',
    body: JSON.stringify(chatRequest),
  })
  await recordStep('v1_chat_completions', chatResponse, chatResponsePath)
  if (chatResponse?.camelid?.timings_ms) timingInputs.push(chatResponsePath)

  if (timingInputs.length > 0) {
    const timingsReportPath = join(outDir, 'generation-timings.summary.json')
    const timingsCommand = [
      process.execPath,
      timingsScript,
      '--out', timingsReportPath,
      ...timingInputs,
    ]
    await writeFile(join(outDir, 'generation-timings.command.txt'), `${shellJoin(timingsCommand)}\n`)
    const timingsRun = await run(timingsCommand[0], timingsCommand.slice(1))
    await writeFile(join(outDir, 'generation-timings.stdout.log'), timingsRun.stdout)
    await writeFile(join(outDir, 'generation-timings.stderr.log'), timingsRun.stderr)
    const timingsSummary = {
      command: timingsCommand,
      exit_code: timingsRun.code,
      summary_path: timingsReportPath,
    }
    if (timingsRun.code !== 0) {
      timingsSummary.__error = `generation timing summary exited ${timingsRun.code}`
    }
    await recordStep('generation_timings', timingsSummary, join(outDir, 'generation-timings.run.json'))
  } else {
    await recordStep('generation_timings', {
      skipped: true,
      reason: 'the selected generation lane did not emit camelid.timings_ms; timing diagnostics are not a support gate',
    }, join(outDir, 'generation-timings.skipped.json'))
  }

  const healthAfter = await tryFetchJson(`${apiBase}/v1/health`)
  await recordStep('health_after', healthAfter, join(outDir, 'health-after.json'))
  if (Object.values(expectedExecutionPlan).some(value => value !== undefined)) {
    const executionPlan = executionPlanEvidence(healthAfter, expectedExecutionPlan)
    await recordStep('execution_plan', executionPlan, join(outDir, 'execution-plan.json'))
  }

  if (!skipFrontend) {
    const frontendCommand = [
      process.execPath,
      frontendScript,
      '--api', apiBase,
      '--frontend', frontendUrl,
      '--model', modelPath,
      '--model-id', modelId,
      '--chat-repeats', '1',
      '--stream-max-tokens', String(streamMaxTokens),
      '--message', message,
      '--require-local-model',
      '--require-generation',
    ]
    if (allowGuardedChat) {
      frontendCommand.pop()
      frontendCommand.push('--allow-guarded-chat')
    }
    if (expectCompatibilityRow) frontendCommand.push('--expect-compatibility-row', expectCompatibilityRow)
    if (expectCompatibilityStatus) frontendCommand.push('--expect-compatibility-status', expectCompatibilityStatus)
    if (expectContractSupported) frontendCommand.push('--expect-contract-supported', expectContractSupported)
    if (expectWebUiChat) frontendCommand.push('--expect-webui-chat', expectWebUiChat)
    if (expectLocalLaneClass) frontendCommand.push('--expect-local-lane-class', expectLocalLaneClass)
    if (expectGgufSha256) frontendCommand.push('--expect-gguf-sha256', expectGgufSha256)
    if (replaceLoadedModel) frontendCommand.push('--replace-loaded-model')
    await writeFile(join(outDir, 'frontend.command.txt'), `${shellJoin(frontendCommand)}\n`)
    const frontendRun = await run(frontendCommand[0], frontendCommand.slice(1))
    await writeFile(join(outDir, 'frontend.stdout.log'), frontendRun.stdout)
    await writeFile(join(outDir, 'frontend.stderr.log'), frontendRun.stderr)
    const frontendSummary = {
      command: frontendCommand,
      exit_code: frontendRun.code,
      mode: allowGuardedChat ? 'allow_guarded_chat' : 'require_generation',
    }
    if (frontendRun.code !== 0) {
      frontendSummary.__error = `frontend smoke exited ${frontendRun.code}`
    }
    await recordStep('frontend_smoke', frontendSummary, join(outDir, 'frontend.summary.json'))
  }

  const healthFinal = await tryFetchJson(`${apiBase}/v1/health`)
  await recordStep('health_final', healthFinal, join(outDir, 'health-final.json'))
  if (Object.values(expectedExecutionPlan).some(value => value !== undefined)) {
    const executionPlanFinal = executionPlanEvidence(healthFinal, expectedExecutionPlan)
    await recordStep('execution_plan_final', executionPlanFinal, join(outDir, 'execution-plan-final.json'))
  }

  summary.passed = true
} catch (error) {
  summary.error = error instanceof Error ? error.message : String(error)
} finally {
  await writeJson(join(outDir, 'summary.json'), summary)
}

if (!summary.passed) process.exitCode = 1

async function recordStep(name, payload, path) {
  await writeJson(path, payload)
  summary.steps[name] = {
    ok: !payload?.__error,
    skipped: payload?.skipped === true,
    path,
  }
  if (payload?.__error) throw new Error(`${name}: ${payload.__error}`)
}

async function tryFetchJson(url, options = {}) {
  try {
    return await fetchJson(url, options)
  } catch (error) {
    return { __error: error instanceof Error ? error.message : String(error), url }
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {
      'content-type': 'application/json',
      ...(options.headers || {}),
    },
  })
  const text = await response.text()
  const body = text ? JSON.parse(text) : null
  if (!response.ok) {
    throw new Error(`${url}: ${response.status} ${response.statusText}: ${body?.error?.message || text}`)
  }
  return body
}

async function writeJson(path, payload) {
  await writeFile(path, `${JSON.stringify(payload, null, 2)}\n`)
}

function parseArgs(argv) {
  const parsed = new Map()
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (!arg.startsWith('--')) continue
    const [key, inline] = arg.slice(2).split('=', 2)
    const next = argv[i + 1]
    const value = inline ?? (next && !next.startsWith('--') ? argv[++i] : 'true')
    parsed.set(key, value)
  }
  return parsed
}

function parsePositiveInt(name, value) {
  const parsed = Number.parseInt(value, 10)
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`--${name} must be a positive integer, got ${value}`)
  return parsed
}

function parseOptionalBoolean(name, value) {
  if (value === undefined || value === null || value === '') return null
  const normalized = String(value).trim().toLowerCase()
  if (['1', 'true', 'yes'].includes(normalized)) return true
  if (['0', 'false', 'no'].includes(normalized)) return false
  throw new Error(`--${name} must be true or false, got ${value}`)
}

function normalizeSha256(value) {
  if (!value) return ''
  const normalized = String(value).trim().toLowerCase()
  if (!/^[a-f0-9]{64}$/.test(normalized)) {
    throw new Error(`--expect-gguf-sha256 must be 64 hexadecimal characters, got ${value}`)
  }
  return normalized
}

async function exactModelIdentityEvidence({ modelPath, modelId, current, localModels, expectLocalLaneClass, expectGgufSha256 }) {
  const modelStats = await stat(modelPath)
  const expectedFilename = basename(modelPath)
  const entries = Array.isArray(localModels?.models) ? localModels.models : []
  const exactMatches = entries.filter(entry => entry?.filename === expectedFilename)
  const local = exactMatches.length === 1 ? exactMatches[0] : null
  const actualSha256 = String(current?.lane?.gguf_sha256 || '').toLowerCase()
  const checks = [
    check('current model id', current?.id, modelId),
    check('current lane model id', current?.lane?.model_id, modelId),
    check('current model path', normalizedPath(current?.path), normalizedPath(modelPath)),
    check('current lane filename', current?.lane?.gguf_filename, expectedFilename),
    check('local exact filename match count', exactMatches.length, 1),
    check('local models directory path', normalizedPath(localModels?.models_dir && local ? join(localModels.models_dir, local.filename) : null), normalizedPath(modelPath)),
    check('local exact file size', local?.size_bytes, modelStats.size),
    check('loaded lane SHA-256 shape', /^[a-f0-9]{64}$/.test(actualSha256), true),
  ]
  if (expectLocalLaneClass) checks.push(check('local lane class', local?.lane_class, expectLocalLaneClass))
  if (expectGgufSha256) checks.push(check('loaded lane SHA-256', actualSha256, expectGgufSha256))
  const failed = checks.filter(item => !item.passed)
  return {
    schema: 'camelid.model-promotion.exact-model-identity.v1',
    expected: {
      model_id: modelId,
      model_path: modelPath,
      filename: expectedFilename,
      size_bytes: modelStats.size,
      lane_class: expectLocalLaneClass || null,
      gguf_sha256: expectGgufSha256 || null,
    },
    current: {
      id: current?.id ?? null,
      path: current?.path ?? null,
      lane: current?.lane ?? null,
    },
    local_model: local,
    checks,
    passed: failed.length === 0,
    ...(failed.length ? { __error: failed.map(item => `${item.name}: expected ${JSON.stringify(item.expected)}, got ${JSON.stringify(item.actual)}`).join('; ') } : {}),
  }
}

function normalizedPath(value) {
  if (!value) return null
  const normalized = resolve(String(value))
  return process.platform === 'win32' ? normalized.toLowerCase() : normalized
}

function check(name, actual, expected) {
  return { name, expected, actual: actual ?? null, passed: actual === expected }
}

function executionPlanEvidence(health, expected) {
  const actual = health?.execution_plan || null
  const checks = Object.entries(expected)
    .filter(([, value]) => value !== undefined)
    .map(([field, value]) => check(`execution plan ${field}`, actual?.[field], value))
  const failed = checks.filter(item => !item.passed)
  return {
    schema: 'camelid.model-promotion.execution-plan.v1',
    expected,
    actual,
    checks,
    passed: failed.length === 0,
    ...(failed.length ? { __error: failed.map(item => `${item.name}: expected ${JSON.stringify(item.expected)}, got ${JSON.stringify(item.actual)}`).join('; ') } : {}),
  }
}

function run(command, commandArgs) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, commandArgs, { stdio: ['ignore', 'pipe', 'pipe'] })
    let stdout = ''
    let stderr = ''
    child.stdout.on('data', chunk => { stdout += chunk })
    child.stderr.on('data', chunk => { stderr += chunk })
    child.once('error', reject)
    child.once('close', code => resolvePromise({ code: code ?? 1, stdout, stderr }))
  })
}

function shellJoin(parts) {
  return parts.map(shellEscape).join(' ')
}

function shellEscape(value) {
  if (/^[A-Za-z0-9_/:=.,-]+$/.test(value)) return value
  return `'${String(value).replace(/'/g, `'\\''`)}'`
}
