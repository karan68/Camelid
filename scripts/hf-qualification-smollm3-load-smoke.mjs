#!/usr/bin/env node

import { createHash, randomBytes } from 'node:crypto'
import { execFile } from 'node:child_process'
import { createReadStream } from 'node:fs'
import {
  mkdir,
  lstat,
  readdir,
  rename,
  rm,
  stat,
  statfs,
  writeFile,
} from 'node:fs/promises'
import { freemem } from 'node:os'
import { createServer } from 'node:net'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { pathToFileURL } from 'node:url'
import { spawn } from 'node:child_process'
import { setImmediate as yieldImmediate, setTimeout as sleep } from 'node:timers/promises'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)

const RECEIPT_SCHEMA = 'camelid.model-qualification.load-smoke/v1'
const ROW_ID = 'smollm3_3b_q8_0'
const BINARY_PROFILE = 'release-fat-lto'
const EXECUTION_PLAN_EXACT_MODEL_ROW = 'SmolLM3-Q8_0.gguf'
const SERVER_ADDR = '127.0.0.1:8297'
const SERVER_ORIGIN = `http://${SERVER_ADDR}`
const TEMPLATE_UTF8_BYTES = 5_493
const TEMPLATE_SHA256 = 'b9b66f04c64fbb8695cf5b35c37780efd0b8e0829fbfe3e30fafb9f469b7d30e'

const EXACT_ROW = Object.freeze({
  id: ROW_ID,
  family: 'smollm3',
  architecture: 'smollm3',
  quantization: 'Q8_0',
  target_tier: 'experimental_exact_row',
  disposition: 'hold',
  source: Object.freeze({
    repo: 'ggml-org/SmolLM3-3B-GGUF',
    file: 'SmolLM3-Q8_0.gguf',
    revision: '4965cb60b150737b68a0408c36aeefb65078f894',
    size_bytes: 3_275_574_624,
    sha256: '8aa8cc74656137174a1988d993b00828e65a86fd68773412b632a75aa1373248',
    license: 'apache-2.0',
  }),
})

const SAFE_CAMELID_ENV = Object.freeze({
  CAMELID_PROFILE: 'safe',
  CAMELID_LAZY_Q8_0_LINEAR: '1',
  CAMELID_X86_Q8_REPACK: 'off',
  CAMELID_Q8_0_FILE_CACHE_BYTES: '0',
  CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES: '0',
  CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES: '1073741824',
  CAMELID_MAX_KV_CACHE_BYTES: '268435456',
  CAMELID_KV_POOL_BUDGET_BYTES: '268435456',
  CAMELID_FORWARD_RSS_TIMINGS: 'on',
  CAMELID_GENERATION_TIMEOUT_MS: '1800000',
  CAMELID_QUEUE_DEPTH: '1',
  CAMELID_CONTINUOUS_BATCH_SLOTS: '1',
  CAMELID_NO_REMOTE_DIMS: '1',
  CAMELID_PREFIX_CACHE_MIN_TOKENS: '1024',
})

const LIMITS = Object.freeze({
  startup_timeout_ms: 60_000,
  load_timeout_ms: 20 * 60_000,
  generation_timeout_ms: 31 * 60_000,
  ordinary_request_timeout_ms: 30_000,
  monitor_interval_ms: 1_000,
  low_memory_abort_bytes: 1 * 1024 ** 3,
  child_working_set_abort_bytes: 2 * 1024 ** 3,
  consecutive_abort_samples: 2,
  preflight_disk_bytes: 4 * 1024 ** 3,
  preflight_physical_bytes: 4 * 1024 ** 3,
  max_response_bytes: 16 * 1024 ** 2,
})

const RAW_REQUEST = Object.freeze({
  model: ROW_ID,
  prompt: 'The capital of France is',
  max_tokens: 1,
  temperature: 0,
  stream: false,
})

const CHAT_REQUEST = Object.freeze({
  model: ROW_ID,
  messages: Object.freeze([
    Object.freeze({ role: 'user', content: 'Hello, please help me.' }),
  ]),
  max_tokens: 1,
  temperature: 0,
  stream: false,
})

const EXPECTED_TEMPLATE_CAPS = Object.freeze({
  available: true,
  requires_loaded_model: true,
  source: 'tokenizer.chat_template',
  detected_format: 'smollm3_exact_default_thinking_text_qualified',
  length: TEMPLATE_UTF8_BYTES,
  supported_operations: Object.freeze(['render_prompt']),
  render_prompt_envelope: Object.freeze({
    public_surfaces: Object.freeze({
      '/apply-template': Object.freeze({ thinking: Object.freeze(['omitted_effective_true']) }),
      '/v1/chat/completions': Object.freeze({
        thinking: Object.freeze(['omitted_defaults_true', 'explicit_true']),
        streaming: Object.freeze([false, true]),
      }),
    }),
    content: 'text_only',
    roles: Object.freeze(['user', 'assistant']),
    history: 'strict_alternation_ending_user',
    add_generation_prompt: true,
    today_date: 'system_local_english_dd_month_yyyy',
  }),
  unsupported: Object.freeze([
    'system_messages',
    'custom_instructions',
    'system_override',
    'tools',
    'tool_messages',
    'thinking_disabled',
    'invalid_roles',
    'non_alternating_history',
    'history_not_ending_user',
    'multimodal_content',
    'non_text_content',
    'arbitrary_template_kwargs',
    'full_llama_server_template_parity',
  ]),
})

const STEP_CONTRACT = Object.freeze([
  Object.freeze(['baseline_health', 'GET', '/v1/health']),
  Object.freeze(['baseline_gpu', 'GET', '/api/runtime/gpu']),
  Object.freeze(['load', 'POST', '/models/load']),
  Object.freeze(['verify_identity', 'GET', '/api/models/verify']),
  Object.freeze(['loaded_health', 'GET', '/v1/health']),
  Object.freeze(['props', 'GET', '/props']),
  Object.freeze(['raw_first_forward', 'POST', '/v1/completions']),
  Object.freeze(['post_raw_health', 'GET', '/v1/health']),
  Object.freeze(['post_raw_gpu', 'GET', '/api/runtime/gpu']),
  Object.freeze(['chat_followup', 'POST', '/v1/chat/completions']),
  Object.freeze(['final_health', 'GET', '/v1/health']),
  Object.freeze(['final_gpu', 'GET', '/api/runtime/gpu']),
])

const DOES_NOT_PROVE = Object.freeze([
  'llama.cpp token or text parity',
  'output correctness beyond non-degenerate one-token execution',
  'full chat-template parity',
  'the blocked template gate',
  'api_webui qualification',
  'context-512 qualification',
  'performance or throughput',
  'GPU execution',
  'support or promotion',
  'adjacent sizes, variants, or quantizations',
  'system, tools, no-think, multimodal, or non-text chat shapes',
])

const LEGACY_STORAGE_LABELS = Object.freeze([
  'execution_plan.prefill_runtime_policy',
  'execution_plan.fallback_path',
])

const ERROR_CONTRACTS = Object.freeze({
  load_smoke_options_invalid: ['blocked', 'the load-smoke invocation is incomplete or unsafe'],
  load_smoke_platform_invalid: ['blocked', 'the exact gate requires Windows x86_64'],
  load_smoke_artifact_unavailable: ['blocked', 'the ignored exact SmolLM3 artifact is unavailable'],
  load_smoke_artifact_identity_mismatch: ['fail', 'the artifact size does not match the exact source lock'],
  load_smoke_artifact_lock_failed: ['blocked', 'the exact artifact could not be protected by a Windows read-share lock'],
  load_smoke_artifact_lock_lost: ['fail', 'the Windows artifact read-share lock exited before evidence collection completed'],
  load_smoke_artifact_lock_release_failed: ['blocked', 'the Windows artifact read-share lock could not be released and observed cleanly'],
  load_smoke_artifact_not_ignored: ['blocked', 'the full artifact is not contained under an ignored path'],
  load_smoke_source_dirty: ['blocked', 'tracked source files must be clean before freezing runtime provenance'],
  load_smoke_binary_stale: ['blocked', 'the frozen binary version does not exactly match the clean source describe'],
  load_smoke_source_changed: ['blocked', 'source or binary provenance changed during the gate'],
  load_smoke_auto_select_root_invalid: ['blocked', 'an auto-selection root could not be verified'],
  load_smoke_auto_select_candidate_present: ['blocked', 'an auto-selection root contains a model candidate or saved selector'],
  load_smoke_port_in_use: ['blocked', 'the isolated loopback qualification port is already in use'],
  load_smoke_llama_server_present: ['blocked', 'a llama-server process must not overlap this first-forward gate'],
  load_smoke_resources_low: ['blocked', 'preflight disk or physical memory is below the fixed safety budget'],
  load_smoke_process_start_failed: ['blocked', 'the isolated Camelid child could not start'],
  load_smoke_process_exited: ['fail', 'the isolated Camelid child exited before the gate completed'],
  load_smoke_startup_timeout: ['blocked', 'the no-model server did not become healthy within the startup budget'],
  load_smoke_http_failed: ['blocked', 'an isolated loopback request failed or timed out'],
  load_smoke_health_invalid: ['fail', 'health did not match the exact unloaded or loaded runtime contract'],
  load_smoke_gpu_invalid: ['fail', 'GPU telemetry did not remain disabled and unused'],
  load_smoke_load_invalid: ['fail', 'the local load alias did not return the exact redacted readiness contract'],
  load_smoke_verify_invalid: ['fail', 'GET model verification did not bind the active model to the exact GGUF identity'],
  load_smoke_props_invalid: ['fail', 'props did not match the exact privacy-safe SmolLM3 template contract'],
  load_smoke_raw_invalid: ['fail', 'the raw first-forward response did not meet the exact one-token evidence contract'],
  load_smoke_chat_invalid: ['fail', 'the chat follow-up did not meet the exact experimental-lane contract'],
  load_smoke_resource_abort: ['blocked', 'the child crossed a fixed memory safety abort threshold'],
  load_smoke_resource_telemetry_unavailable: ['blocked', 'the required child resource telemetry became unavailable'],
  load_smoke_warmup_detected: ['fail', 'startup generation warm-up was observed in the no-model run'],
  load_smoke_termination_failed: ['blocked', 'the exact spawned child could not be terminated'],
  load_smoke_receipt_invalid: ['fail', 'the compact load-smoke receipt failed its durable contract'],
  load_smoke_output_failed: ['blocked', 'the sealed receipt could not be written atomically'],
})

const ERROR_CODES = new WeakMap()

class SmolLM3LoadSmokeError extends Error {
  constructor(code) {
    const canonical = Object.hasOwn(ERROR_CONTRACTS, code)
      ? code
      : 'load_smoke_http_failed'
    super(ERROR_CONTRACTS[canonical][1])
    this.name = 'SmolLM3LoadSmokeError'
    this.code = canonical
    this.status = ERROR_CONTRACTS[canonical][0]
    ERROR_CODES.set(this, canonical)
  }
}

function loadSmokeError(code) {
  return new SmolLM3LoadSmokeError(code)
}

function classifySmolLM3LoadSmokeError(error) {
  if (error instanceof SmolLM3LoadSmokeError) {
    const code = ERROR_CODES.get(error)
    if (Object.hasOwn(ERROR_CONTRACTS, code)) {
      return { status: ERROR_CONTRACTS[code][0], error_code: code, reason: ERROR_CONTRACTS[code][1] }
    }
  }
  return {
    status: ERROR_CONTRACTS.load_smoke_http_failed[0],
    error_code: 'load_smoke_http_failed',
    reason: ERROR_CONTRACTS.load_smoke_http_failed[1],
  }
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

function canonicalValue(value) {
  if (Array.isArray(value)) return value.map(canonicalValue)
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, canonicalValue(value[key])]),
    )
  }
  return value
}

function canonicalJson(value) {
  return JSON.stringify(canonicalValue(value))
}

function sameJson(left, right) {
  return canonicalJson(left) === canonicalJson(right)
}

function sealReceipt(body) {
  const receiptId = sha256(Buffer.from(canonicalJson(body), 'utf8'))
  const { schema, ...rest } = body
  return { schema, receipt_id: receiptId, ...rest }
}

function receiptBody(receipt) {
  const { receipt_id: _receiptId, ...body } = receipt
  return body
}

function finiteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value)
}

function nonNegativeInteger(value) {
  return Number.isSafeInteger(value) && value >= 0
}

function expect(condition, code) {
  if (!condition) throw loadSmokeError(code)
}

function buildChildEnv(inherited = process.env) {
  const clean = {}
  for (const [key, value] of Object.entries(inherited)) {
    if (!key.toUpperCase().startsWith('CAMELID_') && value !== undefined) clean[key] = value
  }
  return { ...clean, ...SAFE_CAMELID_ENV }
}

function buildServeArgs(modelsDir) {
  expect(typeof modelsDir === 'string' && isAbsolute(modelsDir), 'load_smoke_options_invalid')
  return [
    'serve',
    '--addr', SERVER_ADDR,
    '--models-dir', modelsDir,
    '--threads', '4',
    '--gpu', 'off',
    '--deterministic',
    '--kv-quant', 'f16',
    '--no-open',
    '--max-prompt-tokens', '1024',
    '--max-generation-tokens', '1',
  ]
}

function receiptCommand() {
  return ['<camelid>', ...buildServeArgs(resolve('<empty-models-dir>')).map((value) => (
    value === resolve('<empty-models-dir>') ? '<empty-models-dir>' : value
  ))]
}

function childCamelidEnv(env) {
  return Object.fromEntries(
    Object.entries(env)
      .filter(([key]) => key.toUpperCase().startsWith('CAMELID_'))
      .sort(([left], [right]) => left.localeCompare(right)),
  )
}

function autoSelectRoots({ binary, cwd, modelsDir }) {
  return [
    { kind: 'configured_models_dir', path: resolve(modelsDir) },
    { kind: 'executable_models_dir', path: join(dirname(resolve(binary)), 'models') },
    { kind: 'executable_dir', path: dirname(resolve(binary)) },
    { kind: 'cwd_models_dir', path: join(resolve(cwd), 'models') },
    { kind: 'cwd', path: resolve(cwd) },
  ]
}

async function assertAutoSelectRootsEmpty(options, { readdirImpl = readdir } = {}) {
  const results = []
  for (const root of autoSelectRoots(options)) {
    let entries
    try {
      entries = await readdirImpl(root.path, { withFileTypes: true })
    } catch (error) {
      if (error?.code === 'ENOENT') {
        results.push({
          kind: root.kind,
          exists: false,
          path_redacted: true,
          gguf_candidates: 0,
          default_preference_present: false,
        })
        continue
      }
      throw loadSmokeError('load_smoke_auto_select_root_invalid')
    }
    const names = entries.map((entry) => String(entry.name))
    const ggufCandidates = names.filter((name) => name.toLowerCase().endsWith('.gguf'))
    const preferencePresent = names.some((name) => name.toLowerCase() === '.camelid-default-model')
    if (ggufCandidates.length || preferencePresent) {
      throw loadSmokeError('load_smoke_auto_select_candidate_present')
    }
    results.push({
      kind: root.kind,
      exists: true,
      path_redacted: true,
      gguf_candidates: 0,
      default_preference_present: false,
    })
  }
  return results
}

async function sha256File(path) {
  const digest = createHash('sha256')
  await new Promise((resolvePromise, rejectPromise) => {
    const input = createReadStream(path)
    input.on('data', (chunk) => digest.update(chunk))
    input.once('end', resolvePromise)
    input.once('error', rejectPromise)
  })
  return digest.digest('hex')
}

async function inspectExactArtifactIdentity(path, {
  lstatImpl = lstat,
  statImpl = stat,
  sha256FileImpl = sha256File,
} = {}) {
  let artifactLinkStats
  let artifactStats
  let artifactSha256
  try {
    artifactLinkStats = await lstatImpl(path)
    artifactStats = await statImpl(path)
    artifactSha256 = await sha256FileImpl(path)
  } catch {
    throw loadSmokeError('load_smoke_artifact_unavailable')
  }
  expect(artifactLinkStats.isFile()
    && artifactLinkStats.isSymbolicLink() === false
    && artifactStats.isFile()
    && artifactStats.size === EXACT_ROW.source.size_bytes
    && artifactSha256 === EXACT_ROW.source.sha256,
  'load_smoke_artifact_identity_mismatch')
  return { size_bytes: artifactStats.size, sha256: artifactSha256 }
}

const ARTIFACT_LOCK_SCRIPT = String.raw`
$ErrorActionPreference = 'Stop'
$stream = $null
try {
  $line = [Console]::In.ReadLine()
  if ($null -eq $line) { exit 64 }
  $request = $line | ConvertFrom-Json
  $path = [string]$request.path
  $nonce = [string]$request.nonce
  if ([string]::IsNullOrWhiteSpace($path) -or $nonce -notmatch '^[0-9a-f]{32}$') { exit 64 }
  $stream = [System.IO.File]::Open(
    $path,
    [System.IO.FileMode]::Open,
    [System.IO.FileAccess]::Read,
    [System.IO.FileShare]::Read
  )
  [Console]::Out.WriteLine(('LOCKED:{0}' -f $nonce))
  [Console]::Out.Flush()
  $release = [Console]::In.ReadLine()
  if ($release -ne ('RELEASE:{0}' -f $nonce)) { exit 66 }
  $stream.Dispose()
  $stream = $null
  [Console]::Out.WriteLine(('RELEASED:{0}' -f $nonce))
  [Console]::Out.Flush()
  exit 0
} catch {
  exit 65
} finally {
  if ($null -ne $stream) { $stream.Dispose() }
}
`

function boundedTimeout(ms, value) {
  let timer
  const promise = new Promise((resolvePromise) => {
    timer = setTimeout(() => resolvePromise(value), ms)
    timer.unref?.()
  })
  return { promise, cancel: () => clearTimeout(timer) }
}

async function acquireWindowsArtifactReadLock(path, {
  spawnImpl = spawn,
  acquireTimeoutMs = 10_000,
  releaseTimeoutMs = 10_000,
} = {}) {
  expect(process.platform === 'win32' && typeof path === 'string' && isAbsolute(path),
    'load_smoke_artifact_lock_failed')
  const nonce = randomBytes(16).toString('hex')
  const lockedToken = `LOCKED:${nonce}`
  const releasedToken = `RELEASED:${nonce}`
  const encodedScript = Buffer.from(ARTIFACT_LOCK_SCRIPT, 'utf16le').toString('base64')
  let child
  try {
    child = spawnImpl('powershell.exe', [
      '-NoLogo', '-NoProfile', '-NonInteractive', '-EncodedCommand', encodedScript,
    ], {
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
    })
  } catch {
    throw loadSmokeError('load_smoke_artifact_lock_failed')
  }
  if (!(child?.stdin && child?.stdout && child?.stderr && typeof child.kill === 'function')) {
    try { child?.kill?.('SIGKILL') } catch { /* invalid helper is already failed closed */ }
    throw loadSmokeError('load_smoke_artifact_lock_failed')
  }

  let stdoutBuffer = ''
  let stderrText = ''
  let lockedSeen = false
  let releasedSeen = false
  let releaseRequested = false
  let protocolState = 'waiting_locked'
  let protocolError = false
  let exitStatus
  let resolveLocked
  let resolveReleased
  let resolveExited
  let resolveClosed
  let resolveProtocolFailed
  const locked = new Promise((resolvePromise) => { resolveLocked = resolvePromise })
  const released = new Promise((resolvePromise) => { resolveReleased = resolvePromise })
  const exited = new Promise((resolvePromise) => { resolveExited = resolvePromise })
  const closed = new Promise((resolvePromise) => { resolveClosed = resolvePromise })
  const protocolFailed = new Promise((resolvePromise) => { resolveProtocolFailed = resolvePromise })
  const failProtocol = () => {
    if (protocolError) return
    protocolError = true
    resolveProtocolFailed(true)
  }
  const observeStdout = (chunk) => {
    stdoutBuffer += String(chunk)
    if (Buffer.byteLength(stdoutBuffer) > 512) {
      failProtocol()
      return
    }
    while (stdoutBuffer.includes('\n')) {
      const newline = stdoutBuffer.indexOf('\n')
      const line = stdoutBuffer.slice(0, newline).replace(/\r$/, '')
      stdoutBuffer = stdoutBuffer.slice(newline + 1)
      if (protocolState === 'waiting_locked' && line === lockedToken) {
        lockedSeen = true
        protocolState = 'held'
        resolveLocked(true)
      } else if (protocolState === 'release_requested' && line === releasedToken) {
        releasedSeen = true
        protocolState = 'released'
        resolveReleased(true)
      } else {
        failProtocol()
      }
    }
  }
  child.stdout.on('data', observeStdout)
  child.stderr.on('data', (chunk) => {
    stderrText += String(chunk)
    if (Buffer.byteLength(stderrText) > 4_096) failProtocol()
  })
  // Windows PowerShell may emit a bounded CLIXML module-initialization progress
  // record on stderr even for a successful non-interactive encoded command.
  // It is never interpreted as protocol; only exact nonce-bound stdout lines are.
  const stderrWithinBound = () => Buffer.byteLength(stderrText) <= 4_096
  const finish = (status) => {
    if (exitStatus !== undefined) return
    exitStatus = status
    resolveExited(status)
  }
  child.once('error', () => finish({ error: true, code: null, signal: null }))
  child.once('exit', (code, signal) => finish({ error: false, code, signal }))
  child.once('close', () => {
    if (stdoutBuffer.length > 0) failProtocol()
    resolveClosed(true)
  })

  const writeLine = (line, code) => new Promise((resolvePromise, rejectPromise) => {
    child.stdin.write(`${line}\n`, (error) => {
      if (error) rejectPromise(loadSmokeError(code))
      else resolvePromise()
    })
  })
  const terminateHelper = async () => {
    if (exitStatus === undefined) {
      try { child.stdin.end() } catch { /* best effort after a failed protocol */ }
      try { child.kill('SIGKILL') } catch { return false }
    }
    const timeout = boundedTimeout(2_000, null)
    const outcome = await Promise.race([
      Promise.all([exited, closed]).then(() => true),
      timeout.promise,
    ])
    timeout.cancel()
    return outcome === true
  }

  try {
    await writeLine(JSON.stringify({ path, nonce }), 'load_smoke_artifact_lock_failed')
    const timeout = boundedTimeout(acquireTimeoutMs, { timeout: true })
    const outcome = await Promise.race([
      locked.then(() => ({ locked: true })),
      exited.then((status) => ({ exited: status })),
      protocolFailed.then(() => ({ protocol_error: true })),
      timeout.promise,
    ])
    timeout.cancel()
    expect(outcome.locked === true
      && protocolState === 'held'
      && protocolError === false
      && exitStatus === undefined
      && stderrWithinBound(),
      'load_smoke_artifact_lock_failed')
  } catch (error) {
    const terminated = await terminateHelper()
    if (!terminated) throw loadSmokeError('load_smoke_artifact_lock_release_failed')
    if (error instanceof SmolLM3LoadSmokeError) throw error
    throw loadSmokeError('load_smoke_artifact_lock_failed')
  }

  return {
    acquired: true,
    exited,
    closed,
    isExited: () => exitStatus !== undefined,
    exitStatus: () => exitStatus ?? null,
    assertHeld() {
      if (!lockedSeen || protocolState !== 'held' || protocolError
        || releaseRequested || exitStatus !== undefined) {
        throw loadSmokeError('load_smoke_artifact_lock_lost')
      }
    },
    async release() {
      if (releaseRequested || protocolState !== 'held' || protocolError || exitStatus !== undefined) {
        throw loadSmokeError('load_smoke_artifact_lock_release_failed')
      }
      releaseRequested = true
      protocolState = 'release_requested'
      try {
        await writeLine(`RELEASE:${nonce}`, 'load_smoke_artifact_lock_release_failed')
        const timeout = boundedTimeout(releaseTimeoutMs, { timeout: true })
        const outcome = await Promise.race([
          Promise.all([released, exited, closed]).then(([, status]) => ({ released: true, status })),
          protocolFailed.then(() => ({ protocol_error: true })),
          timeout.promise,
        ])
        timeout.cancel()
        expect(outcome.released === true
          && outcome.status?.error === false
          && outcome.status?.code === 0
          && releasedSeen === true
          && protocolState === 'released'
          && protocolError === false
          && stderrWithinBound(),
        'load_smoke_artifact_lock_release_failed')
        return { observed: true, released_token_observed: true, exit_code: 0 }
      } catch (error) {
        const terminated = await terminateHelper()
        if (!terminated) throw loadSmokeError('load_smoke_artifact_lock_release_failed')
        if (error instanceof SmolLM3LoadSmokeError) throw error
        throw loadSmokeError('load_smoke_artifact_lock_release_failed')
      }
    },
  }
}

async function inspectProvenance({ root, binary, binaryProfile = BINARY_PROFILE }, {
  execFileImpl = execFileAsync,
  sha256FileImpl = sha256File,
} = {}) {
  expect(binaryProfile === BINARY_PROFILE, 'load_smoke_options_invalid')
  let head
  let tracked
  let describe
  try {
    [
      { stdout: head },
      { stdout: tracked },
      { stdout: describe },
    ] = await Promise.all([
      execFileImpl('git', ['-C', root, 'rev-parse', 'HEAD'], { timeout: 10_000 }),
      execFileImpl('git', ['-C', root, 'status', '--porcelain', '--untracked-files=no'], { timeout: 10_000 }),
      execFileImpl('git', ['-C', root, 'describe', '--tags', '--always'], { timeout: 10_000 }),
    ])
  } catch {
    throw loadSmokeError('load_smoke_source_dirty')
  }
  const runtimeHead = String(head).trim().toLowerCase()
  const trackedClean = String(tracked).trim() === ''
  const sourceDescribe = String(describe).trim()
  expect(/^[0-9a-f]{40}$/.test(runtimeHead), 'load_smoke_source_dirty')
  expect(trackedClean, 'load_smoke_source_dirty')
  expect(isCleanSourceDescribe(sourceDescribe)
    && sourceDescribeMatchesHead(sourceDescribe, runtimeHead),
  'load_smoke_source_dirty')

  let version
  let binarySha256
  try {
    [{ stdout: version }, binarySha256] = await Promise.all([
      execFileImpl(binary, ['--version'], { timeout: 10_000, windowsHide: true }),
      sha256FileImpl(binary),
    ])
  } catch {
    throw loadSmokeError('load_smoke_binary_stale')
  }
  const binaryVersion = String(version).trim()
  try {
    expect(/^[0-9a-f]{64}$/.test(binarySha256), 'load_smoke_binary_stale')
    expect(binaryVersion === `camelid ${sourceDescribe}`, 'load_smoke_binary_stale')
    return {
      runtime_head: runtimeHead,
      source_describe: sourceDescribe,
      tracked_files_clean: true,
      untracked_files_excluded: true,
      binary_profile: binaryProfile,
      binary_sha256: binarySha256,
      binary_version: binaryVersion,
    }
  } catch (error) {
    if (error instanceof SmolLM3LoadSmokeError) throw error
    throw loadSmokeError('load_smoke_binary_stale')
  }
}

function isCleanSourceDescribe(value) {
  return typeof value === 'string'
    && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value)
    && !/-dirty/i.test(value)
}

function sourceDescribeMatchesHead(sourceDescribe, runtimeHead) {
  if (!isCleanSourceDescribe(sourceDescribe) || !/^[0-9a-f]{40}$/.test(runtimeHead || '')) return false
  const match = /^(?:([0-9a-f]{7,40})|.*-g([0-9a-f]{7,40}))$/i.exec(sourceDescribe)
  if (!match) return false
  const abbreviation = match[1] || match[2]
  return runtimeHead.startsWith(abbreviation.toLowerCase())
}

function assertFrozenProvenance(provenance) {
  expect(exactKeys(provenance, [
    'runtime_head', 'source_describe', 'tracked_files_clean', 'untracked_files_excluded',
    'binary_profile', 'binary_sha256', 'binary_version',
  ]), 'load_smoke_source_dirty')
  expect(/^[0-9a-f]{40}$/.test(provenance.runtime_head)
    && isCleanSourceDescribe(provenance.source_describe)
    && sourceDescribeMatchesHead(provenance.source_describe, provenance.runtime_head)
    && provenance.tracked_files_clean === true
    && provenance.untracked_files_excluded === true,
  'load_smoke_source_dirty')
  expect(provenance.binary_profile === BINARY_PROFILE
    && /^[0-9a-f]{64}$/.test(provenance.binary_sha256)
    && provenance.binary_version === `camelid ${provenance.source_describe}`,
  'load_smoke_binary_stale')
  return provenance
}

async function gitPathIgnored(root, path, { execFileImpl = execFileAsync } = {}) {
  try {
    await execFileImpl('git', ['-C', root, 'check-ignore', '--quiet', '--', path], { timeout: 10_000 })
    return true
  } catch {
    return false
  }
}

async function diskFreeBytes(path, { statfsImpl = statfs } = {}) {
  const stats = await statfsImpl(path, { bigint: true })
  return Number(stats.bavail * stats.bsize)
}

async function assertPortFree({ host = '127.0.0.1', port = 8297, createServerImpl = createServer } = {}) {
  await new Promise((resolvePromise, rejectPromise) => {
    const server = createServerImpl()
    server.unref?.()
    server.once('error', () => rejectPromise(loadSmokeError('load_smoke_port_in_use')))
    server.listen({ host, port, exclusive: true }, () => {
      server.close((error) => {
        if (error) rejectPromise(loadSmokeError('load_smoke_port_in_use'))
        else resolvePromise()
      })
    })
  })
}

async function llamaServerRunning({ execFileImpl = execFileAsync } = {}) {
  try {
    const { stdout } = await execFileImpl('powershell.exe', [
      '-NoProfile',
      '-NonInteractive',
      '-Command',
      "@(Get-Process -Name 'llama-server' -ErrorAction SilentlyContinue).Count",
    ], { timeout: 10_000, windowsHide: true })
    return Number(String(stdout).trim()) > 0
  } catch {
    throw loadSmokeError('load_smoke_auto_select_root_invalid')
  }
}

function pathInside(parent, candidate) {
  const rel = relative(resolve(parent), resolve(candidate))
  return rel === '' || (!rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel))
}

async function runPreflight(options, deps = {}) {
  const platform = deps.platformInfo?.() || { platform: process.platform, arch: process.arch }
  expect(platform.platform === 'win32' && platform.arch === 'x64', 'load_smoke_platform_invalid')
  for (const path of [options.root, options.binary, options.artifact, options.cwd, options.modelsDir]) {
    expect(typeof path === 'string' && isAbsolute(path), 'load_smoke_options_invalid')
  }
  expect(resolve(options.cwd) !== resolve(options.modelsDir), 'load_smoke_options_invalid')
  expect(dirname(resolve(options.binary)) !== resolve(options.cwd), 'load_smoke_options_invalid')
  expect(dirname(resolve(options.binary)) !== resolve(options.modelsDir), 'load_smoke_options_invalid')
  for (const candidate of autoSelectRoots(options)) {
    expect(!pathInside(candidate.path, options.artifact), 'load_smoke_options_invalid')
  }

  let artifactLinkStats
  let artifactStats
  try {
    artifactLinkStats = await (deps.lstatImpl || lstat)(options.artifact)
    artifactStats = await (deps.statImpl || stat)(options.artifact)
  }
  catch { throw loadSmokeError('load_smoke_artifact_unavailable') }
  expect(artifactLinkStats.isFile()
    && artifactLinkStats.isSymbolicLink() === false
    && artifactStats.isFile()
    && artifactStats.size === EXACT_ROW.source.size_bytes,
    'load_smoke_artifact_identity_mismatch')
  const ignored = deps.checkIgnoredImpl
    ? await deps.checkIgnoredImpl(options.root, options.artifact)
    : await gitPathIgnored(options.root, options.artifact, deps)
  expect(ignored === true, 'load_smoke_artifact_not_ignored')

  const provenance = deps.inspectProvenanceImpl
    ? await deps.inspectProvenanceImpl(options)
    : await inspectProvenance(options, deps)
  assertFrozenProvenance(provenance)
  const roots = await assertAutoSelectRootsEmpty(options, deps)
  if (deps.assertPortFreeImpl) await deps.assertPortFreeImpl()
  else await assertPortFree({ createServerImpl: deps.createServerImpl })
  const llamaPresent = deps.llamaServerRunningImpl
    ? await deps.llamaServerRunningImpl()
    : await llamaServerRunning(deps)
  expect(llamaPresent === false, 'load_smoke_llama_server_present')

  const availablePhysicalBytes = deps.freePhysicalBytesImpl
    ? await deps.freePhysicalBytesImpl()
    : freemem()
  const availableDiskBytes = deps.diskFreeBytesImpl
    ? await deps.diskFreeBytesImpl(options.cwd)
    : await diskFreeBytes(options.cwd, deps)
  expect(availablePhysicalBytes >= LIMITS.preflight_physical_bytes
    && availableDiskBytes >= LIMITS.preflight_disk_bytes, 'load_smoke_resources_low')

  return {
    platform: 'windows-x86_64',
    artifact: {
      size_bytes: artifactStats.size,
      expected_sha256: EXACT_ROW.source.sha256,
      hash_recomputed: false,
      ignored: true,
      path_redacted: true,
    },
    provenance,
    auto_select_roots: roots,
    available_physical_bytes: availablePhysicalBytes,
    available_disk_bytes: availableDiskBytes,
    qualification_port_unbound: true,
    llama_server_absent: true,
  }
}

async function runPostflight(options, preflight, deps = {}) {
  const provenance = deps.inspectProvenanceImpl
    ? await deps.inspectProvenanceImpl(options)
    : await inspectProvenance(options, deps)
  assertFrozenProvenance(provenance)
  const roots = await assertAutoSelectRootsEmpty(options, deps)
  const artifactIdentity = await inspectExactArtifactIdentity(options.artifact, deps)
  expect(sameJson(provenance, preflight.provenance), 'load_smoke_source_changed')
  expect(sameJson(roots, preflight.auto_select_roots), 'load_smoke_source_changed')
  return {
    provenance,
    auto_select_roots: roots,
    artifact: {
      size_bytes: artifactIdentity.size_bytes,
      sha256: artifactIdentity.sha256,
      verified_after_generation: true,
      path_redacted: true,
    },
  }
}

function startCamelidProcess({ binary, args, cwd, env }, { spawnImpl = spawn } = {}) {
  let child
  try {
    child = spawnImpl(binary, args, {
      cwd,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    })
  } catch {
    throw loadSmokeError('load_smoke_process_start_failed')
  }
  let tail = ''
  let outputBytes = 0
  let warmingMarker = false
  let completionMarker = false
  const observe = (chunk) => {
    const text = `${tail}${String(chunk)}`
    outputBytes += Buffer.byteLength(chunk)
    warmingMarker ||= /Warming up/i.test(text)
    completionMarker ||= /generation warm-up complete/i.test(text)
    tail = text.slice(-128)
  }
  child.stdout?.on('data', observe)
  child.stderr?.on('data', observe)
  let exitStatus = null
  const exited = new Promise((resolvePromise) => {
    const finish = (status) => {
      if (exitStatus !== null) return
      exitStatus = status
      resolvePromise(status)
    }
    child.once('error', () => finish({ error: true, code: null, signal: null }))
    child.once('exit', (code, signal) => finish({ error: false, code, signal }))
  })
  let closeStatus = null
  const closed = new Promise((resolvePromise) => {
    child.once('close', (code, signal) => {
      closeStatus = { code, signal }
      resolvePromise(closeStatus)
    })
  })
  return {
    pid: child.pid,
    kill: (signal) => child.kill(signal),
    exited,
    closed,
    isExited: () => exitStatus !== null,
    isClosed: () => closeStatus !== null,
    exitStatus: () => exitStatus,
    closeStatus: () => closeStatus,
    logMarkers: () => ({
      warming_up_seen: warmingMarker,
      generation_warmup_complete_seen: completionMarker,
      output_captured_only_for_markers: true,
      raw_output_persisted: false,
      observed_output_bytes: outputBytes,
    }),
  }
}

async function terminateSpawnedChild(handle, { sleepImpl = sleep } = {}) {
  if (!handle || typeof handle.kill !== 'function' || !handle.exited || !handle.closed) {
    throw loadSmokeError('load_smoke_termination_failed')
  }
  const waitForExitAndClose = async (timeoutMs) => {
    const stopped = await Promise.race([
      Promise.all([handle.exited, handle.closed]).then(([status]) => ({ done: true, status })),
      sleepImpl(timeoutMs).then(() => ({ done: false })),
    ])
    expect(stopped.done === true && handle.isExited?.() === true && handle.isClosed?.() === true,
      'load_smoke_termination_failed')
    return stopped.status
  }
  if (handle.isExited?.()) {
    const status = await waitForExitAndClose(5_000)
    return {
      observed: true,
      already_exited: true,
      termination_requested: false,
      status,
    }
  }
  let gracefulRequested
  try { gracefulRequested = handle.kill('SIGTERM') } catch { throw loadSmokeError('load_smoke_termination_failed') }
  if (gracefulRequested !== true) {
    if (handle.isExited?.()) {
      const status = await waitForExitAndClose(5_000)
      return {
        observed: true,
        already_exited: true,
        termination_requested: false,
        status,
      }
    }
    throw loadSmokeError('load_smoke_termination_failed')
  }
  const graceful = await Promise.race([
    handle.exited.then((status) => ({ done: true, status })),
    sleepImpl(10_000).then(() => ({ done: false })),
  ])
  if (graceful.done) {
    const status = await waitForExitAndClose(5_000)
    return {
      observed: true,
      already_exited: false,
      termination_requested: true,
      status,
    }
  }
  let forcedRequested
  try { forcedRequested = handle.kill('SIGKILL') } catch { throw loadSmokeError('load_smoke_termination_failed') }
  expect(forcedRequested === true, 'load_smoke_termination_failed')
  const forced = await Promise.race([
    Promise.all([handle.exited, handle.closed]).then(([status]) => ({ done: true, status })),
    sleepImpl(5_000).then(() => ({ done: false })),
  ])
  if (!forced.done || handle.isExited?.() !== true || handle.isClosed?.() !== true) {
    throw loadSmokeError('load_smoke_termination_failed')
  }
  return {
    observed: true,
    already_exited: false,
    termination_requested: true,
    status: forced.status,
  }
}

async function sampleWindowsResources(pid, { execFileImpl = execFileAsync } = {}) {
  expect(Number.isSafeInteger(pid) && pid > 0, 'load_smoke_resource_telemetry_unavailable')
  const script = `$p=Get-Process -Id ${pid} -ErrorAction Stop; `
    + '$o=Get-CimInstance Win32_OperatingSystem; '
    + '[Console]::Out.Write([string]::Format("{0},{1}",'
    + '([int64]$o.FreePhysicalMemory*1024),([int64]$p.WorkingSet64)))'
  try {
    const { stdout } = await execFileImpl('powershell.exe', [
      '-NoProfile', '-NonInteractive', '-Command', script,
    ], { timeout: 10_000, windowsHide: true })
    const [availablePhysicalBytes, childWorkingSetBytes] = String(stdout).trim().split(',').map(Number)
    expect(nonNegativeInteger(availablePhysicalBytes) && nonNegativeInteger(childWorkingSetBytes),
      'load_smoke_resource_telemetry_unavailable')
    return { available_physical_bytes: availablePhysicalBytes, child_working_set_bytes: childWorkingSetBytes }
  } catch (error) {
    if (error instanceof SmolLM3LoadSmokeError) throw error
    throw loadSmokeError('load_smoke_resource_telemetry_unavailable')
  }
}

function createResourceGuard(handle, {
  sampleImpl = (pid) => sampleWindowsResources(pid),
  sleepImpl = sleep,
  limits = LIMITS,
} = {}) {
  const controller = new AbortController()
  let stopped = false
  let fatal = null
  let samples = 0
  let lowMemoryStreak = 0
  let highWorkingSetStreak = 0
  let minimumAvailablePhysicalBytes = null
  let peakChildWorkingSetBytes = 0

  const done = (async () => {
    while (!stopped) {
      try {
        const sample = await sampleImpl(handle.pid)
        samples += 1
        minimumAvailablePhysicalBytes = minimumAvailablePhysicalBytes === null
          ? sample.available_physical_bytes
          : Math.min(minimumAvailablePhysicalBytes, sample.available_physical_bytes)
        peakChildWorkingSetBytes = Math.max(peakChildWorkingSetBytes, sample.child_working_set_bytes)
        lowMemoryStreak = sample.available_physical_bytes < limits.low_memory_abort_bytes
          ? lowMemoryStreak + 1
          : 0
        highWorkingSetStreak = sample.child_working_set_bytes > limits.child_working_set_abort_bytes
          ? highWorkingSetStreak + 1
          : 0
        if (lowMemoryStreak >= limits.consecutive_abort_samples
          || highWorkingSetStreak >= limits.consecutive_abort_samples) {
          fatal = loadSmokeError('load_smoke_resource_abort')
          controller.abort(fatal)
          break
        }
      } catch (error) {
        fatal = error instanceof SmolLM3LoadSmokeError
          ? error
          : loadSmokeError('load_smoke_resource_telemetry_unavailable')
        controller.abort(fatal)
        break
      }
      if (!stopped) await sleepImpl(limits.monitor_interval_ms)
    }
  })()

  return {
    signal: controller.signal,
    throwIfAborted() { if (fatal) throw fatal },
    async stop() { stopped = true; await done; return { observed: true } },
    summary() {
      return {
        samples,
        minimum_available_physical_bytes: minimumAvailablePhysicalBytes,
        peak_child_working_set_bytes: peakChildWorkingSetBytes,
        thresholds_tripped: fatal?.code === 'load_smoke_resource_abort',
      }
    },
  }
}

async function httpJson({ method, endpoint, body, timeoutMs, signal, fetchImpl = fetch }) {
  expect(/^\/(?:api\/|v1\/|models\/|props$)/.test(endpoint), 'load_smoke_http_failed')
  if (signal?.aborted) {
    if (signal.reason instanceof SmolLM3LoadSmokeError) throw signal.reason
    throw loadSmokeError('load_smoke_http_failed')
  }
  const controller = new AbortController()
  const abort = () => controller.abort(signal?.reason || loadSmokeError('load_smoke_http_failed'))
  signal?.addEventListener('abort', abort, { once: true })
  // Register first, then re-check. This closes the window where the resource
  // guard aborts between the initial state read and listener subscription.
  if (signal?.aborted) abort()
  const timeout = setTimeout(() => controller.abort(loadSmokeError('load_smoke_http_failed')), timeoutMs)
  try {
    if (controller.signal.aborted) {
      if (signal?.reason instanceof SmolLM3LoadSmokeError) throw signal.reason
      throw loadSmokeError('load_smoke_http_failed')
    }
    const response = await fetchImpl(`${SERVER_ORIGIN}${endpoint}`, {
      method,
      headers: body === undefined ? undefined : { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    })
    const text = await response.text()
    expect(Buffer.byteLength(text) <= LIMITS.max_response_bytes, 'load_smoke_http_failed')
    let parsed
    try { parsed = JSON.parse(text) } catch { throw loadSmokeError('load_smoke_http_failed') }
    return { status: response.status, body: parsed }
  } catch (error) {
    if (signal?.aborted && signal.reason instanceof SmolLM3LoadSmokeError) throw signal.reason
    if (error instanceof SmolLM3LoadSmokeError) throw error
    throw loadSmokeError('load_smoke_http_failed')
  } finally {
    clearTimeout(timeout)
    signal?.removeEventListener('abort', abort)
  }
}

function normalizeQ8Runtime(q8, code) {
  expect(q8 && q8.policy === 'forced_lazy_file_backed_q8'
    && q8.lazy_q8_linear === true
    && q8.retain_q8_blocks === false
    && q8.file_cache_bytes === 0, code)
  return {
    policy: q8.policy,
    lazy_q8_linear: q8.lazy_q8_linear,
    retain_q8_blocks: q8.retain_q8_blocks,
    file_cache_bytes: q8.file_cache_bytes,
  }
}

function normalizeHealth(body, { loaded, final = false } = {}) {
  const code = 'load_smoke_health_invalid'
  expect(body && body.ok === true && body.engine === 'camelid', code)
  expect(typeof body.version === 'string' && typeof body.build === 'string', code)
  expect(body.loaded_now === loaded && body.generation_ready === loaded, code)
  expect(body.vision_ready === false, code)
  expect(body.active_model_id === (loaded ? ROW_ID : null), code)
  expect(body.backend === (loaded ? 'llama' : 'none'), code)
  expect(body.model_family === (loaded ? 'llama-family' : null), code)
  expect(body.engine_queue_depth === 0
    && body.engine_queued_tasks === 0
    && body.engine_active_task_id === null
    && body.engine_active_generated_tokens === 0
    && body.continuous_batch_slots === 1, code)
  expect(typeof body.executable === 'string' && body.executable.length > 0, code)
  expect(body.listen_addr === SERVER_ADDR, code)
  const executionPlan = body.execution_plan
  if (loaded) {
    expect(executionPlan
      && executionPlan.profile === 'safe'
      && executionPlan.operating_system === 'windows'
      && executionPlan.architecture === 'x86_64'
      && executionPlan.model_family === 'smollm3'
      && executionPlan.quant_type === 'Q8_0'
      && executionPlan.support_level === 'unknown_or_unvalidated'
      && executionPlan.selected_backend === 'cpu_reference'
      && executionPlan.selected_q8_path === 'safe_dense_or_q8_cpu'
      && executionPlan.exact_model_row === EXECUTION_PLAN_EXACT_MODEL_ROW
      && typeof executionPlan.diagnostics_status === 'string'
      && executionPlan.cuda_resident_active === false, code)
  } else {
    expect(executionPlan === null, code)
  }
  if (final) expect(body.engine_active_elapsed_seconds === 0 && body.engine_stalled_seconds === 0, code)
  return {
    ok: true,
    engine: 'camelid',
    version: body.version,
    build: body.build,
    loaded_now: loaded,
    generation_ready: loaded,
    vision_ready: false,
    active_model_id: body.active_model_id,
    backend: body.backend,
    model_family: body.model_family,
    q8_runtime: normalizeQ8Runtime(body.q8_runtime, code),
    execution_plan: loaded ? {
      profile: executionPlan.profile,
      operating_system: executionPlan.operating_system,
      architecture: executionPlan.architecture,
      model_family: executionPlan.model_family,
      quant_type: executionPlan.quant_type,
      exact_model_row: executionPlan.exact_model_row,
      support_level: executionPlan.support_level,
      selected_backend: executionPlan.selected_backend,
      selected_q8_path: executionPlan.selected_q8_path,
      diagnostics_status: executionPlan.diagnostics_status,
      cuda_resident_active: executionPlan.cuda_resident_active,
      legacy_storage_labels_excluded: [...LEGACY_STORAGE_LABELS],
    } : null,
    queue: {
      depth: 0,
      queued_tasks: 0,
      active_task: false,
      active_generated_tokens: 0,
      continuous_batch_slots: 1,
    },
    executable_redacted: true,
    listen_addr: SERVER_ADDR,
  }
}

function normalizeGpu(body) {
  expect(body && typeof body.available === 'boolean'
    && body.enabled === false
    && body.run_count === 0
    && typeof body.backend === 'string', 'load_smoke_gpu_invalid')
  return { available: body.available, enabled: false, backend: body.backend, run_count: 0, device_redacted: true }
}

function normalizeLoad(body) {
  const code = 'load_smoke_load_invalid'
  expect(body?.data?.id === ROW_ID
    && body.data.path === null
    && body.data.status?.value === 'loaded'
    && body.data.camelid?.generation_ready === true
    && body.data.camelid?.model_path_redacted === true
    && body.camelid?.model_path_redacted === true
    && body.camelid?.compatibility === 'partial_llama_server_models_load_local_path'
    && body.camelid?.scope === 'single_local_model_load_alias', code)
  return {
    request: { path_redacted: true, id: ROW_ID },
    id: ROW_ID,
    path: null,
    status: 'loaded',
    generation_ready: true,
    model_path_redacted: true,
    compatibility: body.camelid.compatibility,
    scope: body.camelid.scope,
  }
}

function normalizeVerify(body) {
  expect(body?.model_id === ROW_ID
    && body.gguf_sha256 === EXACT_ROW.source.sha256
    && body.eligible === false
    && body.profile_id === null
    && body.report === null, 'load_smoke_verify_invalid')
  return {
    model_id: body.model_id,
    gguf_sha256: body.gguf_sha256,
    eligible: false,
    profile_id: null,
    report: null,
  }
}

function normalizeProps(body) {
  const code = 'load_smoke_props_invalid'
  expect(body?.model_path === null
    && body.model_id === ROW_ID
    && body.camelid?.generation_ready === true
    && body.camelid?.model_path_redacted === true
    && body.modalities?.vision === false
    && body.total_slots === 1
    && body.default_generation_settings?.is_processing === false
    && body.default_generation_settings?.next_token?.has_next_token === true
    && typeof body.chat_template === 'string'
    && Buffer.byteLength(body.chat_template, 'utf8') === TEMPLATE_UTF8_BYTES
    && sha256(Buffer.from(body.chat_template, 'utf8')) === TEMPLATE_SHA256
    && sameJson(body.chat_template_caps, EXPECTED_TEMPLATE_CAPS), code)
  return {
    model_path: null,
    model_id: ROW_ID,
    generation_ready: true,
    model_path_redacted: true,
    modalities: { vision: false },
    total_slots: 1,
    is_processing: false,
    has_next_token: true,
    chat_template: {
      redacted: true,
      utf8_bytes: TEMPLATE_UTF8_BYTES,
      sha256: TEMPLATE_SHA256,
    },
    chat_template_caps: structuredClone(EXPECTED_TEMPLATE_CAPS),
  }
}

function normalizeMaterialization(value, code) {
  const numeric = [
    'tensor_count',
    'dense_f32_tensor_count',
    'dense_f32_bytes',
    'q8_0_source_tensor_count',
    'q8_0_f32_materialized_tensor_count',
    'q8_0_f32_materialized_bytes',
    'q8_0_file_backed_tensor_count',
    'q8_0_file_backed_storage_bytes',
    'q8_0_file_backed_f32_bytes_avoided',
    'q8_0_file_backed_retained_block_bytes_if_enabled',
    'q8_0_file_handle_cached_count',
    'q8_0_retained_block_tensor_count',
    'q8_0_retained_block_bytes',
  ]
  expect(value && numeric.every((key) => nonNegativeInteger(value[key])), code)
  expect(typeof value.has_q8_0_f32_materialization === 'boolean'
    && typeof value.has_lazy_q8_0_file_backing === 'boolean'
    && typeof value.has_retained_q8_0_blocks === 'boolean', code)
  return Object.fromEntries([
    ...numeric.map((key) => [key, value[key]]),
    ['has_q8_0_f32_materialization', value.has_q8_0_f32_materialization],
    ['has_lazy_q8_0_file_backing', value.has_lazy_q8_0_file_backing],
    ['has_retained_q8_0_blocks', value.has_retained_q8_0_blocks],
  ])
}

function normalizeQ8Reads(value, code) {
  const keys = [
    'read_calls', 'read_bytes', 'cache_hits', 'cache_hit_bytes', 'cache_misses',
    'cache_miss_bytes', 'cache_inserts', 'cache_insert_bytes', 'cache_evictions',
    'cache_evicted_bytes', 'cache_merges', 'cache_merged_bytes',
    'cache_decoded_scale_hits', 'cache_decoded_scale_hit_blocks', 'cache_entries',
    'cache_bytes', 'cache_capacity_bytes',
  ]
  expect(value && keys.every((key) => nonNegativeInteger(value[key])), code)
  return Object.fromEntries(keys.map((key) => [key, value[key]]))
}

function normalizeMemoryPhase(phase, value, code) {
  expect(value && nonNegativeInteger(value.forward_passes), code)
  const peakRssKib = value.peak_rss_kib ?? null
  expect(peakRssKib === null || nonNegativeInteger(peakRssKib), code)
  return {
    phase,
    forward_passes: value.forward_passes,
    materialization: normalizeMaterialization(value.materialization, code),
    q8_file_reads: normalizeQ8Reads(value.q8_file_reads, code),
    peak_rss_kib: peakRssKib,
  }
}

function collectMemoryPhases(timings, code) {
  const candidates = [
    ['prefill', timings?.prompt_evaluation?.prefill_memory],
    ['first_token', timings?.prompt_evaluation?.first_token_memory],
    ['generation', timings?.memory],
  ]
  return candidates
    .filter(([, value]) => value !== undefined && value !== null)
    .map(([phase, value]) => normalizeMemoryPhase(phase, value, code))
}

function assertRawMemory(phases) {
  const code = 'load_smoke_raw_invalid'
  expect(phases.length > 0, code)
  const materializations = phases.map((phase) => phase.materialization)
  expect(materializations.some((value) => value.q8_0_source_tensor_count > 0), code)
  expect(materializations.every((value) => value.q8_0_f32_materialized_tensor_count === 0
    && value.q8_0_f32_materialized_bytes === 0
    && value.q8_0_retained_block_tensor_count === 0
    && value.q8_0_retained_block_bytes === 0
    && value.has_q8_0_f32_materialization === false
    && value.has_lazy_q8_0_file_backing === true
    && value.has_retained_q8_0_blocks === false), code)
  expect(materializations.some((value) => value.q8_0_file_backed_tensor_count > 0
    && value.q8_0_file_backed_storage_bytes > 0
    && value.q8_0_file_backed_f32_bytes_avoided > 0), code)
  expect(phases.reduce((sum, phase) => sum + phase.q8_file_reads.read_calls, 0) > 0
    && phases.reduce((sum, phase) => sum + phase.q8_file_reads.read_bytes, 0) > 0, code)
  expect(phases.every((phase) => phase.q8_file_reads.cache_entries === 0
    && phase.q8_file_reads.cache_bytes === 0
    && phase.q8_file_reads.cache_capacity_bytes === 0), code)
}

function normalizeGeneration(body, { chat }) {
  const code = chat ? 'load_smoke_chat_invalid' : 'load_smoke_raw_invalid'
  expect(body && body.model === ROW_ID && Array.isArray(body.choices) && body.choices.length === 1, code)
  expect(body.usage?.completion_tokens === 1
    && nonNegativeInteger(body.usage.prompt_tokens)
    && body.usage.prompt_tokens > 0
    && body.usage.total_tokens === body.usage.prompt_tokens + 1, code)
  expect(!Object.hasOwn(body, 'camelid_receipt'), code)
  if (chat) expect(body.lane === 'experimental', code)
  const diagnostics = body.camelid
  expect(diagnostics
    && Array.isArray(diagnostics.prompt_token_ids) && diagnostics.prompt_token_ids.length > 0
    && diagnostics.prompt_token_ids.every(nonNegativeInteger)
    && Array.isArray(diagnostics.generated_token_ids) && diagnostics.generated_token_ids.length === 1
    && diagnostics.generated_token_ids.every(nonNegativeInteger), code)
  const text = chat ? body.choices[0]?.message?.content : body.choices[0]?.text
  expect(typeof text === 'string' && Buffer.byteLength(text, 'utf8') > 0, code)
  expect(['length', 'stop'].includes(body.choices[0]?.finish_reason), code)
  const stepTopLogits = Object.hasOwn(diagnostics, 'step_top_logits')
    ? diagnostics.step_top_logits
    : []
  expect(Array.isArray(diagnostics.top_logits) && diagnostics.top_logits.length > 0
    && Array.isArray(stepTopLogits)
    && stepTopLogits.every(Array.isArray), code)
  const logitGroups = [diagnostics.top_logits, ...stepTopLogits]
  const logits = logitGroups.flat()
  expect(logits.length > 0 && logits.every((entry) => nonNegativeInteger(entry?.token_id)
    && finiteNumber(entry.logit) && finiteNumber(entry.probability)
    && Number.isSafeInteger(entry.rank) && entry.rank >= 1
    && entry.selected === false), code)
  expect(logitGroups.every((group) => new Set(group.map((entry) => entry.rank)).size === group.length), code)
  const greedyTop = diagnostics.top_logits.filter((entry) => entry.rank === 1)
  expect(greedyTop.length === 1
    && greedyTop[0].token_id === diagnostics.generated_token_ids[0], code)
  const timings = diagnostics.timings_ms
  expect(timings
    && nonNegativeInteger(timings.weight_load)
    && timings.weight_cache_hit === chat
    && timings.prompt_cache_hit === false
    && timings.prompt_evaluation?.first_token_evaluated === true, code)
  const forwardTotal = Number(timings.prompt_evaluation.prefill?.forward_total || 0)
    + Number(timings.prompt_evaluation.first_token?.forward_total || 0)
    + Number(timings.generation?.forward_total || 0)
  expect(finiteNumber(forwardTotal) && forwardTotal > 0, code)
  if (!chat) expect(timings.weight_load > 0, code)
  const memoryPhases = collectMemoryPhases(timings, code)
  if (!chat) assertRawMemory(memoryPhases)
  return {
    model: ROW_ID,
    lane: chat ? 'experimental' : null,
    support_semantics: 'experimental_unverified_no_support_or_parity_claim',
    choice_count: 1,
    finish_reason: body.choices[0].finish_reason,
    usage: {
      prompt_tokens: body.usage.prompt_tokens,
      completion_tokens: 1,
      total_tokens: body.usage.total_tokens,
    },
    prompt_token_ids: [...diagnostics.prompt_token_ids],
    generated_token_ids: [...diagnostics.generated_token_ids],
    generated_text: {
      redacted: true,
      utf8_bytes: Buffer.byteLength(text, 'utf8'),
      sha256: sha256(Buffer.from(text, 'utf8')),
    },
    logits: {
      emitted_count: logits.length,
      all_finite: true,
      greedy_top: {
        token_id: greedyTop[0].token_id,
        logit: greedyTop[0].logit,
        probability: greedyTop[0].probability,
        rank: greedyTop[0].rank,
      },
    },
    timings: {
      weight_load: timings.weight_load,
      weight_cache_hit: timings.weight_cache_hit,
      prompt_cache_hit: false,
      first_token_evaluated: true,
      forward_total: forwardTotal,
    },
    memory_phases: memoryPhases,
    camelid_receipt_present: false,
  }
}

function normalizeResponse(name, body) {
  switch (name) {
    case 'baseline_health': return normalizeHealth(body, { loaded: false })
    case 'baseline_gpu':
    case 'post_raw_gpu':
    case 'final_gpu': return normalizeGpu(body)
    case 'load': return normalizeLoad(body)
    case 'verify_identity': return normalizeVerify(body)
    case 'loaded_health':
    case 'post_raw_health': return normalizeHealth(body, { loaded: true })
    case 'props': return normalizeProps(body)
    case 'raw_first_forward': return normalizeGeneration(body, { chat: false })
    case 'chat_followup': return normalizeGeneration(body, { chat: true })
    case 'final_health': return normalizeHealth(body, { loaded: true, final: true })
    default: throw loadSmokeError('load_smoke_receipt_invalid')
  }
}

function requestContract() {
  return {
    load: { path_redacted: true, id: ROW_ID, unsupported_fields_omitted: true },
    raw_first_forward: structuredClone(RAW_REQUEST),
    chat_followup: {
      ...structuredClone(CHAT_REQUEST),
      camelid_enable_thinking_omitted: true,
    },
    camelid_receipt_requested: false,
  }
}

function buildReceipt({
  preflight,
  postflightArtifact,
  steps,
  resources,
  logMarkers,
  healthBuild,
  createdUtc,
}) {
  const body = {
    schema: RECEIPT_SCHEMA,
    created_utc: createdUtc,
    gate: 'load_smoke',
    row: structuredClone(EXACT_ROW),
    provenance: {
      runtime_head: preflight.provenance.runtime_head,
      source_describe: preflight.provenance.source_describe,
      tracked_files_clean: true,
      untracked_files_excluded: true,
      binary: {
        profile: preflight.provenance.binary_profile,
        sha256: preflight.provenance.binary_sha256,
        version: preflight.provenance.binary_version,
        health_build: healthBuild,
        built_from_clean_tracked_head: true,
      },
      artifact: postflightArtifact,
      platform: preflight.platform,
      paths_redacted: true,
      hostname_redacted: true,
    },
    isolation: {
      no_startup_model: true,
      auto_select_roots: preflight.auto_select_roots,
      loopback_only: true,
      address: SERVER_ADDR,
      qualification_port_unbound_before_start: true,
      llama_server_absent: true,
      harness_request_sequence_exclusive: true,
      inherited_camelid_env_cleared: true,
      child_handle_only_termination: true,
      child_termination_observed: true,
      startup_warmup_markers: {
        warming_up_seen: logMarkers.warming_up_seen,
        generation_warmup_complete_seen: logMarkers.generation_warmup_complete_seen,
        raw_output_persisted: false,
      },
    },
    runtime_contract: {
      command: receiptCommand(),
      cwd_redacted: true,
      environment: structuredClone(SAFE_CAMELID_ENV),
      requests: requestContract(),
      limits: structuredClone(LIMITS),
      readiness_semantics: 'header_config_tokenizer_attemptability_not_forward_proof',
      first_forward_proof: 'raw_completion_is_first_generative_request_and_reports_weight_cache_hit_false',
      excluded_legacy_storage_labels: [...LEGACY_STORAGE_LABELS],
    },
    steps,
    resource_observations: {
      preflight_available_physical_bytes: preflight.available_physical_bytes,
      preflight_available_disk_bytes: preflight.available_disk_bytes,
      monitor_samples: resources.samples,
      minimum_available_physical_bytes: resources.minimum_available_physical_bytes,
      peak_child_working_set_bytes: resources.peak_child_working_set_bytes,
      thresholds_tripped: resources.thresholds_tripped,
    },
    gate_decision: {
      load_smoke: 'pass',
      support_claim: false,
      disposition: 'hold',
      target_tier: 'experimental_exact_row',
      authorized_roster_scope: ['gates.load_smoke'],
      other_gates_unchanged: true,
    },
    does_not_prove: [...DOES_NOT_PROVE],
  }
  return sealReceipt(body)
}

const HEALTH_EVIDENCE_KEYS = Object.freeze([
  'ok', 'engine', 'version', 'build', 'loaded_now', 'generation_ready', 'vision_ready',
  'active_model_id', 'backend', 'model_family', 'q8_runtime', 'execution_plan', 'queue',
  'executable_redacted', 'listen_addr',
])
const Q8_RUNTIME_KEYS = Object.freeze([
  'policy', 'lazy_q8_linear', 'retain_q8_blocks', 'file_cache_bytes',
])
const EXECUTION_PLAN_KEYS = Object.freeze([
  'profile', 'operating_system', 'architecture', 'model_family', 'quant_type',
  'exact_model_row', 'support_level', 'selected_backend', 'selected_q8_path',
  'diagnostics_status', 'cuda_resident_active', 'legacy_storage_labels_excluded',
])
const QUEUE_KEYS = Object.freeze([
  'depth', 'queued_tasks', 'active_task', 'active_generated_tokens', 'continuous_batch_slots',
])
const GPU_EVIDENCE_KEYS = Object.freeze([
  'available', 'enabled', 'backend', 'run_count', 'device_redacted',
])
const GENERATION_EVIDENCE_KEYS = Object.freeze([
  'model', 'lane', 'support_semantics', 'choice_count', 'finish_reason', 'usage', 'prompt_token_ids',
  'generated_token_ids', 'generated_text', 'logits', 'timings', 'memory_phases',
  'camelid_receipt_present',
])
const MATERIALIZATION_KEYS = Object.freeze([
  'tensor_count', 'dense_f32_tensor_count', 'dense_f32_bytes', 'q8_0_source_tensor_count',
  'q8_0_f32_materialized_tensor_count', 'q8_0_f32_materialized_bytes',
  'q8_0_file_backed_tensor_count', 'q8_0_file_backed_storage_bytes',
  'q8_0_file_backed_f32_bytes_avoided',
  'q8_0_file_backed_retained_block_bytes_if_enabled', 'q8_0_file_handle_cached_count',
  'q8_0_retained_block_tensor_count', 'q8_0_retained_block_bytes',
  'has_q8_0_f32_materialization', 'has_lazy_q8_0_file_backing',
  'has_retained_q8_0_blocks',
])
const Q8_READ_KEYS = Object.freeze([
  'read_calls', 'read_bytes', 'cache_hits', 'cache_hit_bytes', 'cache_misses',
  'cache_miss_bytes', 'cache_inserts', 'cache_insert_bytes', 'cache_evictions',
  'cache_evicted_bytes', 'cache_merges', 'cache_merged_bytes',
  'cache_decoded_scale_hits', 'cache_decoded_scale_hit_blocks', 'cache_entries',
  'cache_bytes', 'cache_capacity_bytes',
])

function privacyErrors(value) {
  const errors = []
  const bannedKeys = new Set([
    'hostname', 'pid', 'process_id', 'artifact_path', 'binary_path', 'executable_path',
    'raw_log', 'raw_logs', 'authorization', 'cookie', 'password', 'secret', 'token',
  ])
  const allowedRoutes = new Set(STEP_CONTRACT.map(([, , endpoint]) => endpoint))
  const seen = new WeakSet()
  const stack = [{ node: value, path: '$', depth: 0 }]
  const maxNodes = 50_000
  const maxDepth = 128
  let visited = 0
  while (stack.length) {
    const { node, path, depth } = stack.pop()
    visited += 1
    if (visited > maxNodes) {
      errors.push('privacy scan exceeded its bounded node budget')
      break
    }
    if (typeof node === 'string') {
      if (/[A-Za-z]:[\\/]/.test(node) || /\\\\[^\\]/.test(node) || /\bfile:\/\//i.test(node)) {
        errors.push(`${path} contains an absolute local path`)
      }
      if (/\bhf_[A-Za-z0-9]{8,}\b/.test(node)
        || /\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+/i.test(node)
        || /(?:^|[?&#;,\s])(?:access[_-]?token|auth[_-]?token|authorization|bearer[_-]?token|client[_-]?secret|credential|hf[_-]?token|id[_-]?token|api[_-]?key|password|private[_-]?key|refresh[_-]?token|secret|signed[_-]?(?:url|uri)|token)\s*[:=]\s*[^\s&#;,]+/i.test(node)) {
        errors.push(`${path} contains credential-like data`)
      }
      if (node.startsWith('/') && !allowedRoutes.has(node)) {
        errors.push(`${path} contains an unexpected absolute path`)
      }
      if (node.length > 4_096) errors.push(`${path} contains oversized raw text`)
      continue
    }
    if (typeof node === 'number') {
      if (!Number.isFinite(node)) errors.push(`${path} contains a non-finite number`)
      continue
    }
    if (node === null || typeof node === 'boolean') continue
    if (typeof node !== 'object') {
      errors.push(`${path} contains a non-JSON value`)
      continue
    }
    if (depth >= maxDepth) {
      errors.push(`${path} exceeds the bounded privacy depth`)
      continue
    }
    if (seen.has(node)) {
      errors.push(`${path} contains a cycle or repeated object reference`)
      continue
    }
    seen.add(node)
    let descriptors
    let symbols
    try {
      descriptors = Object.getOwnPropertyDescriptors(node)
      symbols = Object.getOwnPropertySymbols(node)
    } catch {
      errors.push(`${path} could not be inspected safely`)
      continue
    }
    if (symbols.length) errors.push(`${path} contains symbol-keyed data`)
    if (Array.isArray(node)) {
      const elementKeys = Object.keys(descriptors).filter((key) => key !== 'length')
      const canonicalElementKeys = elementKeys.filter((key) => (
        /^(?:0|[1-9][0-9]*)$/.test(key) && Number(key) < node.length
      ))
      if (canonicalElementKeys.length !== elementKeys.length) {
        errors.push(`${path} contains an unexpected array property`)
      }
      if (node.length > maxNodes || canonicalElementKeys.length !== node.length) {
        errors.push(`${path} contains a sparse or oversized array`)
      }
    }
    for (const [key, descriptor] of Object.entries(descriptors)) {
      if (Array.isArray(node) && key === 'length') continue
      const childPath = `${path}.${key}`
      if (!Object.hasOwn(descriptor, 'value')) {
        errors.push(`${childPath} uses an accessor`)
        continue
      }
      if (!descriptor.enumerable) {
        errors.push(`${childPath} is non-enumerable`)
        continue
      }
      if (bannedKeys.has(key.toLowerCase())) errors.push(`${childPath} uses a forbidden key`)
      stack.push({ node: descriptor.value, path: childPath, depth: depth + 1 })
    }
  }
  return errors
}

function exactKeys(value, keys) {
  return value && typeof value === 'object' && !Array.isArray(value)
    && sameJson(Object.keys(value).sort(), [...keys].sort())
}

function validateLoadSmokeReceiptUnsafe(receipt) {
  const errors = []
  const check = (condition, message) => { if (!condition) errors.push(message) }
  const close = (value, keys, path) => check(exactKeys(value, keys), `${path} keys must be exact`)
  check(exactKeys(receipt, [
    'schema', 'receipt_id', 'created_utc', 'gate', 'row', 'provenance', 'isolation',
    'runtime_contract', 'steps', 'resource_observations', 'gate_decision', 'does_not_prove',
  ]), 'top-level keys must match the load-smoke schema')
  check(receipt?.schema === RECEIPT_SCHEMA, 'schema must be exact')
  check(/^[0-9a-f]{64}$/.test(receipt?.receipt_id || ''), 'receipt_id must be lowercase SHA-256')
  check(receipt?.receipt_id === sha256(Buffer.from(canonicalJson(receiptBody(receipt)), 'utf8')),
    'receipt_id must seal the canonical body')
  check(typeof receipt?.created_utc === 'string'
    && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(receipt.created_utc)
    && !Number.isNaN(Date.parse(receipt.created_utc))
    && new Date(receipt.created_utc).toISOString() === receipt.created_utc,
    'created_utc must be an ISO timestamp')
  check(receipt?.gate === 'load_smoke', 'gate must remain load_smoke')
  check(sameJson(receipt?.row, EXACT_ROW), 'row identity must match the exact source lock')
  close(receipt?.provenance, [
    'runtime_head', 'source_describe', 'tracked_files_clean', 'untracked_files_excluded',
    'binary', 'artifact', 'platform', 'paths_redacted', 'hostname_redacted',
  ], 'provenance')
  close(receipt?.provenance?.binary, [
    'profile', 'sha256', 'version', 'health_build', 'built_from_clean_tracked_head',
  ], 'provenance.binary')
  close(receipt?.provenance?.artifact, [
    'size_bytes', 'sha256', 'verified_after_lock_acquisition', 'verified_after_generation',
    'mutation_guard', 'path_redacted',
  ], 'provenance.artifact')
  close(receipt?.provenance?.artifact?.mutation_guard, [
    'mechanism', 'read_access', 'write_access', 'delete_access', 'rename_access',
    'symbolic_links_rejected', 'acquired_before_preload_hash',
    'held_through_post_generation_hash', 'released_token_observed', 'helper_exit_code',
    'artifact_path_in_helper_argv',
  ], 'provenance.artifact.mutation_guard')
  check(/^[0-9a-f]{40}$/.test(receipt?.provenance?.runtime_head || ''), 'runtime HEAD must be exact')
  check(isCleanSourceDescribe(receipt?.provenance?.source_describe),
    'source describe must be exact and clean')
  check(sourceDescribeMatchesHead(
    receipt?.provenance?.source_describe,
    receipt?.provenance?.runtime_head,
  ), 'source describe must bind the runtime HEAD')
  check(receipt?.provenance?.tracked_files_clean === true
    && receipt?.provenance?.untracked_files_excluded === true
    && receipt?.provenance?.paths_redacted === true
    && receipt?.provenance?.hostname_redacted === true, 'provenance must be clean and redacted')
  check(/^[0-9a-f]{64}$/.test(receipt?.provenance?.binary?.sha256 || ''), 'binary SHA-256 must be exact')
  check(receipt?.provenance?.binary?.profile === BINARY_PROFILE
    && receipt?.provenance?.binary?.built_from_clean_tracked_head === true
    && receipt?.provenance?.binary?.version === `camelid ${receipt?.provenance?.source_describe}`,
  'binary provenance must bind the exact clean source describe')
  check(receipt?.provenance?.binary?.health_build === receipt?.provenance?.source_describe,
    'health build must equal the clean source describe')
  check(receipt?.provenance?.platform === 'windows-x86_64', 'platform must be Windows x86_64')
  check(receipt?.provenance?.artifact?.size_bytes === EXACT_ROW.source.size_bytes
    && receipt?.provenance?.artifact?.sha256 === EXACT_ROW.source.sha256
    && receipt?.provenance?.artifact?.verified_after_lock_acquisition === true
    && receipt?.provenance?.artifact?.verified_after_generation === true
    && receipt?.provenance?.artifact?.path_redacted === true
    && receipt?.provenance?.artifact?.mutation_guard?.mechanism === 'windows_file_stream_share_read'
    && receipt?.provenance?.artifact?.mutation_guard?.read_access === 'allowed'
    && receipt?.provenance?.artifact?.mutation_guard?.write_access === 'denied'
    && receipt?.provenance?.artifact?.mutation_guard?.delete_access === 'denied'
    && receipt?.provenance?.artifact?.mutation_guard?.rename_access === 'denied'
    && receipt?.provenance?.artifact?.mutation_guard?.symbolic_links_rejected === true
    && receipt?.provenance?.artifact?.mutation_guard?.acquired_before_preload_hash === true
    && receipt?.provenance?.artifact?.mutation_guard?.held_through_post_generation_hash === true
    && receipt?.provenance?.artifact?.mutation_guard?.released_token_observed === true
    && receipt?.provenance?.artifact?.mutation_guard?.helper_exit_code === 0
    && receipt?.provenance?.artifact?.mutation_guard?.artifact_path_in_helper_argv === false,
  'post-generation artifact identity must match the exact source lock')

  const isolation = receipt?.isolation
  close(isolation, [
    'no_startup_model', 'auto_select_roots', 'loopback_only', 'address',
    'qualification_port_unbound_before_start', 'llama_server_absent',
    'harness_request_sequence_exclusive', 'inherited_camelid_env_cleared',
    'child_handle_only_termination', 'child_termination_observed', 'startup_warmup_markers',
  ], 'isolation')
  close(isolation?.startup_warmup_markers, [
    'warming_up_seen', 'generation_warmup_complete_seen', 'raw_output_persisted',
  ], 'isolation.startup_warmup_markers')
  check(isolation?.no_startup_model === true
    && isolation?.loopback_only === true
    && isolation?.address === SERVER_ADDR
    && isolation?.qualification_port_unbound_before_start === true
    && isolation?.llama_server_absent === true
    && isolation?.harness_request_sequence_exclusive === true
    && isolation?.inherited_camelid_env_cleared === true
    && isolation?.child_handle_only_termination === true
    && isolation?.child_termination_observed === true, 'isolation contract must be exact')
  check(Array.isArray(isolation?.auto_select_roots)
    && isolation.auto_select_roots.length === 5
    && sameJson(isolation.auto_select_roots.map((root) => root.kind), autoSelectRoots({
      binary: resolve('bin/camelid.exe'), cwd: resolve('run'), modelsDir: resolve('models'),
    }).map((root) => root.kind))
    && isolation.auto_select_roots.every((root) => root.path_redacted === true
      && root.gguf_candidates === 0 && root.default_preference_present === false),
  'all five auto-selection roots must be candidate-free and redacted')
  if (Array.isArray(isolation?.auto_select_roots)) {
    isolation.auto_select_roots.forEach((root, index) => close(root, [
      'kind', 'exists', 'path_redacted', 'gguf_candidates', 'default_preference_present',
    ], `isolation.auto_select_roots.${index}`))
    isolation.auto_select_roots.forEach((root, index) => check(
      typeof root?.exists === 'boolean'
        && root?.path_redacted === true
        && root?.gguf_candidates === 0
        && root?.default_preference_present === false,
      `isolation.auto_select_roots.${index} scalar contract must remain exact`,
    ))
  }
  check(isolation?.startup_warmup_markers?.warming_up_seen === false
    && isolation?.startup_warmup_markers?.generation_warmup_complete_seen === false
    && isolation?.startup_warmup_markers?.raw_output_persisted === false,
  'startup warm-up markers must remain absent')
  close(receipt?.runtime_contract, [
    'command', 'cwd_redacted', 'environment', 'requests', 'limits', 'readiness_semantics',
    'first_forward_proof', 'excluded_legacy_storage_labels',
  ], 'runtime_contract')
  close(receipt?.runtime_contract?.requests, [
    'load', 'raw_first_forward', 'chat_followup', 'camelid_receipt_requested',
  ], 'runtime_contract.requests')
  close(receipt?.runtime_contract?.requests?.load, [
    'path_redacted', 'id', 'unsupported_fields_omitted',
  ], 'runtime_contract.requests.load')
  check(receipt?.runtime_contract?.cwd_redacted === true
    && receipt?.runtime_contract?.readiness_semantics
      === 'header_config_tokenizer_attemptability_not_forward_proof'
    && receipt?.runtime_contract?.first_forward_proof
      === 'raw_completion_is_first_generative_request_and_reports_weight_cache_hit_false',
  'runtime semantics must remain exact')
  check(receipt?.runtime_contract?.requests?.load?.path_redacted === true
    && receipt?.runtime_contract?.requests?.load?.id === ROW_ID
    && receipt?.runtime_contract?.requests?.load?.unsupported_fields_omitted === true,
  'load request contract must remain exact and redacted')
  check(sameJson(receipt?.runtime_contract?.limits, LIMITS), 'runtime limits must remain exact')
  check(sameJson(receipt?.runtime_contract?.command, receiptCommand()), 'serve command must be exact and no-model')
  check(!receipt?.runtime_contract?.command?.includes('--model'), 'serve command must not contain --model')
  check(sameJson(receipt?.runtime_contract?.environment, SAFE_CAMELID_ENV), 'Camelid environment allowlist must be exact')
  check(receipt?.runtime_contract?.requests?.camelid_receipt_requested === false,
    'camelid_receipt must remain disabled')
  check(sameJson(receipt?.runtime_contract?.requests?.raw_first_forward, RAW_REQUEST),
    'raw first-forward request must be exact')
  const chatContract = receipt?.runtime_contract?.requests?.chat_followup
  check(chatContract?.camelid_enable_thinking_omitted === true
    && sameJson(Object.fromEntries(Object.entries(chatContract || {})
      .filter(([key]) => key !== 'camelid_enable_thinking_omitted')), CHAT_REQUEST),
  'chat request must omit the thinking override and remain exact')
  check(sameJson(receipt?.runtime_contract?.excluded_legacy_storage_labels, LEGACY_STORAGE_LABELS),
    'legacy execution-plan storage labels must be excluded')

  check(Array.isArray(receipt?.steps) && receipt.steps.length === STEP_CONTRACT.length,
    'step count must match the exact sequence')
  if (Array.isArray(receipt?.steps)) {
    STEP_CONTRACT.forEach(([name, method, endpoint], index) => {
      const step = receipt.steps[index]
      close(step, ['ordinal', 'name', 'method', 'endpoint', 'http_status', 'elapsed_ms', 'evidence'],
        `steps.${index}`)
      check(step?.ordinal === index + 1 && step?.name === name
        && step?.method === method && step?.endpoint === endpoint
        && step?.http_status === 200 && nonNegativeInteger(step?.elapsed_ms),
      `step ${index + 1} must match the exact sequence`)
    })
    const byName = Object.fromEntries(receipt.steps.map((step) => [step.name, step.evidence]))
    for (const name of ['baseline_health', 'loaded_health', 'post_raw_health', 'final_health']) {
      const evidence = byName[name]
      close(evidence, HEALTH_EVIDENCE_KEYS, `steps.${name}.evidence`)
      close(evidence?.q8_runtime, Q8_RUNTIME_KEYS, `steps.${name}.evidence.q8_runtime`)
      close(evidence?.queue, QUEUE_KEYS, `steps.${name}.evidence.queue`)
      if (evidence?.execution_plan !== null) {
        close(evidence?.execution_plan, EXECUTION_PLAN_KEYS,
          `steps.${name}.evidence.execution_plan`)
      }
      check(evidence?.ok === true
        && evidence?.engine === 'camelid'
        && typeof evidence?.version === 'string' && evidence.version.length > 0
        && typeof evidence?.build === 'string' && evidence.build.length > 0
        && evidence?.vision_ready === false
        && evidence?.q8_runtime?.policy === 'forced_lazy_file_backed_q8'
        && evidence?.q8_runtime?.lazy_q8_linear === true
        && evidence?.q8_runtime?.retain_q8_blocks === false
        && evidence?.q8_runtime?.file_cache_bytes === 0
        && evidence?.queue?.depth === 0
        && evidence?.queue?.queued_tasks === 0
        && evidence?.queue?.active_task === false
        && evidence?.queue?.active_generated_tokens === 0
        && evidence?.queue?.continuous_batch_slots === 1
        && evidence?.executable_redacted === true
        && evidence?.listen_addr === SERVER_ADDR,
      `${name} scalar health contract must remain exact`)
    }
    const baselineBuild = byName.baseline_health?.build
    const baselineVersion = byName.baseline_health?.version
    check(baselineBuild === receipt?.provenance?.source_describe
      && receipt?.provenance?.binary?.health_build === baselineBuild,
    'binary health build must equal the clean source describe and every health observation exactly')
    check(['loaded_health', 'post_raw_health', 'final_health'].every((name) => (
      byName[name]?.build === baselineBuild && byName[name]?.version === baselineVersion
    )), 'all health observations must come from one exact build')
    check(byName.baseline_health?.loaded_now === false
      && byName.baseline_health?.generation_ready === false
      && byName.baseline_health?.active_model_id === null
      && byName.baseline_health?.backend === 'none'
      && byName.baseline_health?.model_family === null
      && byName.baseline_health?.execution_plan === null,
    'baseline health must be exactly unloaded')
    for (const name of ['loaded_health', 'post_raw_health', 'final_health']) {
      const evidence = byName[name]
      const plan = evidence?.execution_plan
      check(evidence?.loaded_now === true
        && evidence?.generation_ready === true
        && evidence?.active_model_id === ROW_ID
        && evidence?.backend === 'llama'
        && evidence?.model_family === 'llama-family'
        && plan?.profile === 'safe'
        && plan?.operating_system === 'windows'
        && plan?.architecture === 'x86_64'
        && plan?.model_family === 'smollm3'
        && plan?.quant_type === 'Q8_0'
        && plan?.exact_model_row === EXECUTION_PLAN_EXACT_MODEL_ROW
        && plan?.support_level === 'unknown_or_unvalidated'
        && plan?.selected_backend === 'cpu_reference'
        && plan?.selected_q8_path === 'safe_dense_or_q8_cpu'
        && typeof plan?.diagnostics_status === 'string' && plan.diagnostics_status.length > 0
        && plan?.cuda_resident_active === false
        && sameJson(plan?.legacy_storage_labels_excluded, LEGACY_STORAGE_LABELS),
      `${name} must remain the exact unsupported SmolLM3 CPU plan`)
    }
    for (const name of ['baseline_gpu', 'post_raw_gpu', 'final_gpu']) {
      close(byName[name], GPU_EVIDENCE_KEYS, `steps.${name}.evidence`)
      check(typeof byName[name]?.available === 'boolean'
        && byName[name]?.enabled === false
        && typeof byName[name]?.backend === 'string' && byName[name].backend.length > 0
        && byName[name]?.run_count === 0
        && byName[name]?.device_redacted === true,
      `${name} scalar GPU contract must remain exact`)
    }
    close(byName.load, [
      'request', 'id', 'path', 'status', 'generation_ready', 'model_path_redacted',
      'compatibility', 'scope',
    ], 'steps.load.evidence')
    close(byName.load?.request, ['path_redacted', 'id'], 'steps.load.evidence.request')
    check(byName.load?.request?.path_redacted === true
      && byName.load?.request?.id === ROW_ID
      && byName.load?.id === ROW_ID
      && byName.load?.path === null
      && byName.load?.status === 'loaded'
      && byName.load?.generation_ready === true
      && byName.load?.model_path_redacted === true
      && byName.load?.compatibility === 'partial_llama_server_models_load_local_path'
      && byName.load?.scope === 'single_local_model_load_alias',
    'load evidence scalars must remain exact and path-redacted')
    close(byName.verify_identity, [
      'model_id', 'gguf_sha256', 'eligible', 'profile_id', 'report',
    ], 'steps.verify_identity.evidence')
    check(byName.verify_identity?.model_id === ROW_ID
      && byName.verify_identity?.gguf_sha256 === EXACT_ROW.source.sha256
      && byName.verify_identity?.eligible === false
      && byName.verify_identity?.profile_id === null
      && byName.verify_identity?.report === null,
    'verification scalars must bind the exact unsupported artifact')
    close(byName.props, [
      'model_path', 'model_id', 'generation_ready', 'model_path_redacted', 'modalities',
      'total_slots', 'is_processing', 'has_next_token', 'chat_template', 'chat_template_caps',
    ], 'steps.props.evidence')
    close(byName.props?.modalities, ['vision'], 'steps.props.evidence.modalities')
    close(byName.props?.chat_template, [
      'redacted', 'utf8_bytes', 'sha256',
    ], 'steps.props.evidence.chat_template')
    check(sameJson(byName.props?.chat_template_caps, EXPECTED_TEMPLATE_CAPS),
      'props chat-template caps must remain exact')
    check(byName.props?.model_path === null
      && byName.props?.model_id === ROW_ID
      && byName.props?.generation_ready === true
      && byName.props?.model_path_redacted === true
      && byName.props?.modalities?.vision === false
      && byName.props?.total_slots === 1
      && byName.props?.is_processing === false
      && byName.props?.has_next_token === true
      && byName.props?.chat_template?.redacted === true
      && byName.props?.chat_template?.utf8_bytes === TEMPLATE_UTF8_BYTES
      && byName.props?.chat_template?.sha256 === TEMPLATE_SHA256,
    'props scalars must bind the exact loaded model and redacted template')
    for (const name of ['raw_first_forward', 'chat_followup']) {
      const evidence = byName[name]
      close(evidence, GENERATION_EVIDENCE_KEYS, `steps.${name}.evidence`)
      close(evidence?.usage, ['prompt_tokens', 'completion_tokens', 'total_tokens'],
        `steps.${name}.evidence.usage`)
      close(evidence?.generated_text, ['redacted', 'utf8_bytes', 'sha256'],
        `steps.${name}.evidence.generated_text`)
      close(evidence?.logits, ['emitted_count', 'all_finite', 'greedy_top'],
        `steps.${name}.evidence.logits`)
      close(evidence?.logits?.greedy_top, ['token_id', 'logit', 'probability', 'rank'],
        `steps.${name}.evidence.logits.greedy_top`)
      close(evidence?.timings, [
        'weight_load', 'weight_cache_hit', 'prompt_cache_hit', 'first_token_evaluated',
        'forward_total',
      ], `steps.${name}.evidence.timings`)
      if (Array.isArray(evidence?.memory_phases)) {
        const phaseNames = evidence.memory_phases.map((phase) => phase?.phase)
        const canonicalPhaseNames = ['prefill', 'first_token', 'generation']
          .filter((phase) => phaseNames.includes(phase))
        check(sameJson(phaseNames, canonicalPhaseNames),
          `${name} memory phases must be unique and in canonical order`)
        evidence.memory_phases.forEach((phase, index) => {
          close(phase, ['phase', 'forward_passes', 'materialization', 'q8_file_reads', 'peak_rss_kib'],
            `steps.${name}.evidence.memory_phases.${index}`)
          close(phase?.materialization, MATERIALIZATION_KEYS,
            `steps.${name}.evidence.memory_phases.${index}.materialization`)
          close(phase?.q8_file_reads, Q8_READ_KEYS,
            `steps.${name}.evidence.memory_phases.${index}.q8_file_reads`)
          check(nonNegativeInteger(phase?.forward_passes)
            && (phase?.peak_rss_kib === null || nonNegativeInteger(phase?.peak_rss_kib)),
          `${name} memory phase ${index} counters must be non-negative`)
          const materializationNumericKeys = MATERIALIZATION_KEYS.filter((key) => !key.startsWith('has_'))
          check(materializationNumericKeys.every((key) => nonNegativeInteger(phase?.materialization?.[key]))
            && typeof phase?.materialization?.has_q8_0_f32_materialization === 'boolean'
            && typeof phase?.materialization?.has_lazy_q8_0_file_backing === 'boolean'
            && typeof phase?.materialization?.has_retained_q8_0_blocks === 'boolean',
          `${name} memory phase ${index} materialization telemetry must be typed`)
          check(Q8_READ_KEYS.every((key) => nonNegativeInteger(phase?.q8_file_reads?.[key])),
            `${name} memory phase ${index} Q8 read telemetry must be non-negative`)
        })
      }
      check(evidence?.model === ROW_ID
        && evidence?.support_semantics === 'experimental_unverified_no_support_or_parity_claim'
        && evidence?.choice_count === 1
        && ['length', 'stop'].includes(evidence?.finish_reason)
        && nonNegativeInteger(evidence?.usage?.prompt_tokens)
        && evidence?.usage?.prompt_tokens > 0
        && evidence?.usage?.completion_tokens === 1
        && evidence?.usage?.total_tokens === evidence?.usage?.prompt_tokens + 1
        && Array.isArray(evidence?.prompt_token_ids) && evidence.prompt_token_ids.length > 0
        && evidence.prompt_token_ids.every(nonNegativeInteger)
        && Array.isArray(evidence?.generated_token_ids) && evidence.generated_token_ids.length === 1
        && evidence.generated_token_ids.every(nonNegativeInteger)
        && evidence?.generated_text?.redacted === true
        && nonNegativeInteger(evidence?.generated_text?.utf8_bytes)
        && evidence.generated_text.utf8_bytes > 0
        && /^[0-9a-f]{64}$/.test(evidence?.generated_text?.sha256 || '')
        && evidence?.logits?.all_finite === true
        && nonNegativeInteger(evidence?.logits?.emitted_count)
        && evidence.logits.emitted_count > 0
        && evidence?.logits?.greedy_top?.token_id === evidence?.generated_token_ids?.[0]
        && finiteNumber(evidence?.logits?.greedy_top?.logit)
        && finiteNumber(evidence?.logits?.greedy_top?.probability)
        && evidence.logits.greedy_top.probability >= 0
        && evidence.logits.greedy_top.probability <= 1
        && evidence?.logits?.greedy_top?.rank === 1
        && nonNegativeInteger(evidence?.timings?.weight_load)
        && evidence?.timings?.prompt_cache_hit === false
        && evidence?.timings?.first_token_evaluated === true
        && finiteNumber(evidence?.timings?.forward_total)
        && evidence.timings.forward_total > 0,
      `${name} compact generation evidence must remain internally consistent`)
    }
    check(byName.baseline_health?.loaded_now === false
      && byName.baseline_health?.generation_ready === false
      && byName.baseline_health?.active_model_id === null, 'baseline must be unloaded')
    for (const name of ['loaded_health', 'post_raw_health', 'final_health']) {
      check(byName[name]?.loaded_now === true
        && byName[name]?.generation_ready === true
        && byName[name]?.active_model_id === ROW_ID
        && byName[name]?.q8_runtime?.policy === 'forced_lazy_file_backed_q8'
        && byName[name]?.q8_runtime?.lazy_q8_linear === true,
      `${name} must prove loaded forced-lazy health`)
    }
    check(byName.load?.request?.path_redacted === true && byName.load?.path === null,
      'load path must be redacted')
    check(byName.verify_identity?.gguf_sha256 === EXACT_ROW.source.sha256
      && byName.verify_identity?.eligible === false, 'GET verification must bind exact unsupported bytes')
    check(byName.props?.chat_template?.redacted === true
      && byName.props?.chat_template?.utf8_bytes === TEMPLATE_UTF8_BYTES
      && byName.props?.chat_template?.sha256 === TEMPLATE_SHA256,
    'props must persist only the exact template identity')
    check(byName.raw_first_forward?.timings?.weight_cache_hit === false
      && byName.raw_first_forward?.timings?.weight_load > 0
      && byName.raw_first_forward?.timings?.forward_total > 0
      && byName.raw_first_forward?.generated_token_ids?.length === 1
      && byName.raw_first_forward?.camelid_receipt_present === false,
    'raw request must prove the first materializing forward')
    check(byName.raw_first_forward?.lane === null
      && byName.raw_first_forward?.support_semantics
        === 'experimental_unverified_no_support_or_parity_claim',
    'raw evidence must remain explicitly unsupported and unverified')
    check(Array.isArray(byName.raw_first_forward?.memory_phases), 'raw memory telemetry must be present')
    try { assertRawMemory(byName.raw_first_forward?.memory_phases || []) } catch { errors.push('raw lazy-Q8 telemetry must remain exact') }
    check(byName.chat_followup?.lane === 'experimental'
      && byName.chat_followup?.timings?.weight_cache_hit === true
      && nonNegativeInteger(byName.chat_followup?.timings?.weight_load)
      && byName.chat_followup?.support_semantics
        === 'experimental_unverified_no_support_or_parity_claim'
      && byName.chat_followup?.generated_token_ids?.length === 1
      && byName.chat_followup?.camelid_receipt_present === false,
    'chat must remain a cached experimental follow-up')
    for (const name of ['baseline_gpu', 'post_raw_gpu', 'final_gpu']) {
      check(byName[name]?.enabled === false && byName[name]?.run_count === 0,
        `${name} must prove GPU did not run`)
    }
  }
  close(receipt?.resource_observations, [
    'preflight_available_physical_bytes', 'preflight_available_disk_bytes', 'monitor_samples',
    'minimum_available_physical_bytes', 'peak_child_working_set_bytes', 'thresholds_tripped',
  ], 'resource_observations')
  check(receipt?.resource_observations?.thresholds_tripped === false
    && nonNegativeInteger(receipt?.resource_observations?.preflight_available_physical_bytes)
    && receipt.resource_observations.preflight_available_physical_bytes >= LIMITS.preflight_physical_bytes
    && nonNegativeInteger(receipt?.resource_observations?.preflight_available_disk_bytes)
    && receipt.resource_observations.preflight_available_disk_bytes >= LIMITS.preflight_disk_bytes
    && nonNegativeInteger(receipt?.resource_observations?.monitor_samples)
    && receipt.resource_observations.monitor_samples > 0
    && nonNegativeInteger(receipt?.resource_observations?.peak_child_working_set_bytes)
    && receipt.resource_observations.peak_child_working_set_bytes > 0
    && nonNegativeInteger(receipt?.resource_observations?.minimum_available_physical_bytes),
  'resource telemetry must be observed without a tripped threshold')
  close(receipt?.gate_decision, [
    'load_smoke', 'support_claim', 'disposition', 'target_tier',
    'authorized_roster_scope', 'other_gates_unchanged',
  ], 'gate_decision')
  check(receipt?.gate_decision?.load_smoke === 'pass'
    && receipt?.gate_decision?.support_claim === false
    && receipt?.gate_decision?.disposition === 'hold'
    && receipt?.gate_decision?.target_tier === 'experimental_exact_row'
    && sameJson(receipt?.gate_decision?.authorized_roster_scope, ['gates.load_smoke'])
    && receipt?.gate_decision?.other_gates_unchanged === true,
  'gate decision must remain load-smoke-only without support')
  check(sameJson(receipt?.does_not_prove, DOES_NOT_PROVE), 'scope exclusions must remain exact')
  errors.push(...privacyErrors(receipt))
  return [...new Set(errors)]
}

function validateLoadSmokeReceipt(receipt) {
  try {
    return validateLoadSmokeReceiptUnsafe(receipt)
  } catch {
    return ['receipt validation could not safely inspect malformed input']
  }
}

function assertValidLoadSmokeReceipt(receipt) {
  if (validateLoadSmokeReceipt(receipt).length) throw loadSmokeError('load_smoke_receipt_invalid')
  return receipt
}

async function waitForBaselineHealth({ request, handle, artifactLock, guard, nowMs, sleepImpl }) {
  const deadline = nowMs() + LIMITS.startup_timeout_ms
  while (nowMs() < deadline) {
    guard.throwIfAborted()
    artifactLock.assertHeld()
    const attempt = await Promise.race([
      request().then((response) => ({ response })).catch(() => ({ response: null })),
      handle.exited.then((status) => ({ exited: status })),
      artifactLock.exited.then((status) => ({ lock_exited: status })),
    ])
    if (attempt.lock_exited) throw loadSmokeError('load_smoke_artifact_lock_lost')
    if (attempt.exited) throw loadSmokeError('load_smoke_process_exited')
    if (attempt.response?.status === 200) return attempt.response
    await sleepImpl(100)
  }
  throw loadSmokeError('load_smoke_startup_timeout')
}

function requestBodyForStep(name, artifact) {
  if (name === 'load') return { path: artifact, id: ROW_ID }
  if (name === 'raw_first_forward') return structuredClone(RAW_REQUEST)
  if (name === 'chat_followup') return structuredClone(CHAT_REQUEST)
  return undefined
}

function timeoutForStep(name) {
  if (name === 'load') return LIMITS.load_timeout_ms
  if (name === 'raw_first_forward' || name === 'chat_followup') return LIMITS.generation_timeout_ms
  return LIMITS.ordinary_request_timeout_ms
}

async function runSmolLM3LoadSmoke(rawOptions, deps = {}) {
  const options = {
    root: resolve(rawOptions?.root || '.'),
    binary: resolve(rawOptions?.binary || ''),
    artifact: resolve(rawOptions?.artifact || ''),
    cwd: resolve(rawOptions?.cwd || ''),
    modelsDir: resolve(rawOptions?.modelsDir || ''),
    binaryProfile: rawOptions?.binaryProfile || BINARY_PROFILE,
  }
  expect(rawOptions?.binary && rawOptions?.artifact && rawOptions?.cwd && rawOptions?.modelsDir,
    'load_smoke_options_invalid')
  expect(options.binaryProfile === BINARY_PROFILE, 'load_smoke_options_invalid')
  const env = buildChildEnv(deps.inheritedEnv || process.env)
  expect(sameJson(childCamelidEnv(env), Object.fromEntries(Object.entries(SAFE_CAMELID_ENV)
    .sort(([left], [right]) => left.localeCompare(right)))), 'load_smoke_options_invalid')
  const args = buildServeArgs(options.modelsDir)
  expect(!args.includes('--model'), 'load_smoke_options_invalid')
  const preflight = deps.preflightImpl
    ? await deps.preflightImpl(options, { env, args })
    : await runPreflight(options, deps)
  assertFrozenProvenance(preflight?.provenance)

  const nowMs = deps.nowMsImpl || Date.now
  const sleepImpl = deps.sleepImpl || sleep
  const yieldImpl = deps.yieldImpl || yieldImmediate
  const requestImpl = deps.httpJsonImpl || ((requestOptions) => httpJson({ ...requestOptions, fetchImpl: deps.fetchImpl }))
  let artifactLock
  try {
    artifactLock = deps.acquireArtifactLockImpl
      ? await deps.acquireArtifactLockImpl(options.artifact)
      : await acquireWindowsArtifactReadLock(options.artifact, deps)
  } catch (error) {
    if (error instanceof SmolLM3LoadSmokeError) throw error
    throw loadSmokeError('load_smoke_artifact_lock_failed')
  }
  expect(artifactLock?.acquired === true
    && artifactLock.exited
    && artifactLock.closed
    && typeof artifactLock.isExited === 'function'
    && typeof artifactLock.exitStatus === 'function'
    && typeof artifactLock.assertHeld === 'function'
    && typeof artifactLock.release === 'function',
  'load_smoke_artifact_lock_failed')
  const assertArtifactLockHeld = () => {
    try { artifactLock.assertHeld() }
    catch { throw loadSmokeError('load_smoke_artifact_lock_lost') }
  }
  const whileArtifactLockHeld = async (operation) => {
    assertArtifactLockHeld()
    const operationResult = Promise.resolve()
      .then(operation)
      .then((value) => ({ value }), (error) => ({ error }))
    const outcome = await Promise.race([
      operationResult,
      artifactLock.exited.then((status) => ({ lock_exited: status })),
    ])
    if (outcome.lock_exited) throw loadSmokeError('load_smoke_artifact_lock_lost')
    if (outcome.error) throw outcome.error
    assertArtifactLockHeld()
    return outcome.value
  }

  let handle
  let guard
  let steps = []
  let resources
  let logMarkers
  let primaryError = null
  let cleanupError = null
  let terminationObserved = false
  let lockedPreloadArtifact
  let postflight
  let healthBuild
  let evidenceError = null

  try {
    lockedPreloadArtifact = await whileArtifactLockHeld(() => (
      deps.preloadArtifactIdentityImpl
        ? deps.preloadArtifactIdentityImpl(options.artifact)
        : inspectExactArtifactIdentity(options.artifact, deps)
    ))
    expect(lockedPreloadArtifact?.size_bytes === EXACT_ROW.source.size_bytes
      && lockedPreloadArtifact?.sha256 === EXACT_ROW.source.sha256,
    'load_smoke_artifact_identity_mismatch')

    try {
      let startedHandle
      try {
        startedHandle = deps.startProcessImpl
          ? await deps.startProcessImpl({ binary: options.binary, args, cwd: options.cwd, env })
          : startCamelidProcess({ binary: options.binary, args, cwd: options.cwd, env }, deps)
      } catch {
        throw loadSmokeError('load_smoke_process_start_failed')
      }
      expect(startedHandle && startedHandle.exited && startedHandle.closed
        && typeof startedHandle.kill === 'function'
        && typeof startedHandle.isExited === 'function'
        && typeof startedHandle.isClosed === 'function'
        && typeof startedHandle.exitStatus === 'function'
        && typeof startedHandle.logMarkers === 'function',
      'load_smoke_process_start_failed')
      handle = startedHandle
      let startedGuard
      try {
        startedGuard = deps.createResourceGuardImpl
          ? await deps.createResourceGuardImpl(handle)
          : createResourceGuard(handle, deps)
      } catch {
        throw loadSmokeError('load_smoke_resource_telemetry_unavailable')
      }
      expect(startedGuard && startedGuard.signal && typeof startedGuard.throwIfAborted === 'function'
        && typeof startedGuard.stop === 'function' && typeof startedGuard.summary === 'function',
      'load_smoke_resource_telemetry_unavailable')
      guard = startedGuard

      const assertNoWarmup = () => {
        const observed = handle.logMarkers()
        expect(observed.warming_up_seen === false
          && observed.generation_warmup_complete_seen === false, 'load_smoke_warmup_detected')
        return observed
      }

      const call = async (index, baseline = false) => {
        const [name, method, endpoint] = STEP_CONTRACT[index]
        const body = requestBodyForStep(name, options.artifact)
        const started = nowMs()
        let response
        assertArtifactLockHeld()
        if (baseline) {
          response = await waitForBaselineHealth({
            request: () => requestImpl({
              method, endpoint, body, timeoutMs: 2_000, signal: guard.signal,
            }),
            handle,
            artifactLock,
            guard,
            nowMs,
            sleepImpl,
          })
        } else {
          guard.throwIfAborted()
          const requestPromise = Promise.resolve()
            .then(() => requestImpl({
              method, endpoint, body, timeoutMs: timeoutForStep(name), signal: guard.signal,
            }))
            .then((value) => ({ response: value }), (error) => ({ request_error: error }))
          const outcome = await Promise.race([
            requestPromise,
            handle.exited.then((status) => ({ exited: status })),
            artifactLock.exited.then((status) => ({ lock_exited: status })),
          ])
          if (outcome.lock_exited) throw loadSmokeError('load_smoke_artifact_lock_lost')
          if (outcome.exited) throw loadSmokeError('load_smoke_process_exited')
          if (outcome.request_error) {
            const exitObservation = await Promise.race([
              handle.exited.then(() => 'process'),
              artifactLock.exited.then(() => 'lock'),
              sleepImpl(0).then(() => 'none'),
            ])
            if (exitObservation === 'lock' || artifactLock.isExited()) {
              throw loadSmokeError('load_smoke_artifact_lock_lost')
            }
            if (exitObservation === 'process' || handle.isExited()) {
              throw loadSmokeError('load_smoke_process_exited')
            }
            throw outcome.request_error
          }
          response = outcome.response
          if (response?.status !== 200) {
            await sleepImpl(0)
            if (artifactLock.isExited()) throw loadSmokeError('load_smoke_artifact_lock_lost')
            if (handle.isExited()) throw loadSmokeError('load_smoke_process_exited')
          }
        }
        guard.throwIfAborted()
        assertArtifactLockHeld()
        if (!baseline && handle.isExited()) throw loadSmokeError('load_smoke_process_exited')
        expect(response?.status === 200 && response.body && typeof response.body === 'object',
          'load_smoke_http_failed')
        const evidence = normalizeResponse(name, response.body)
        const elapsed = Math.max(0, Math.round(nowMs() - started))
        steps.push({
          ordinal: index + 1,
          name,
          method,
          endpoint,
          http_status: 200,
          elapsed_ms: elapsed,
          evidence,
        })
      }

      await call(0, true)
      // A startup `--model` or auto-selected model performs a hidden generation
      // warm-up. Check immediately after the unloaded baseline and again directly
      // before the load alias, so a marker can never be discovered only after we
      // have already materialized weights or executed a forward.
      assertNoWarmup()
      await call(1)
      assertNoWarmup()
      for (let index = 2; index < STEP_CONTRACT.length; index += 1) await call(index)
      logMarkers = assertNoWarmup()
    } catch (error) {
      try { guard?.throwIfAborted() } catch (guardError) { primaryError = guardError }
      if (!primaryError) {
        primaryError = error instanceof SmolLM3LoadSmokeError
          ? error
          : loadSmokeError('load_smoke_http_failed')
      }
    } finally {
      if (guard) {
        try {
          const stopped = await guard.stop()
          expect(stopped?.observed === true, 'load_smoke_resource_telemetry_unavailable')
          resources = guard.summary()
        } catch {
          cleanupError = loadSmokeError('load_smoke_resource_telemetry_unavailable')
        }
        if (!cleanupError) {
          try { guard.throwIfAborted() } catch (guardError) { primaryError = guardError }
        }
      }
      if (handle) {
        try {
          // Give a request-associated natural exit one event-loop turn to become
          // observable before the harness starts its own teardown.
          if (primaryError && !handle.isExited()) await yieldImpl()
          if (handle.isExited()) primaryError = loadSmokeError('load_smoke_process_exited')
          const terminated = deps.terminateChildImpl
            ? await deps.terminateChildImpl(handle)
            : await terminateSpawnedChild(handle, { sleepImpl })
          expect(terminated?.observed === true
            && typeof terminated.already_exited === 'boolean'
            && typeof terminated.termination_requested === 'boolean',
          'load_smoke_termination_failed')
          const closeTimeout = boundedTimeout(5_000, { timeout: true })
          const closeObservation = await Promise.race([
            handle.closed.then(() => ({ closed: true })),
            closeTimeout.promise,
          ])
          closeTimeout.cancel()
          expect(closeObservation.closed === true && handle.isClosed() === true,
            'load_smoke_termination_failed')
          if (terminated.already_exited || terminated.termination_requested === false) {
            primaryError = loadSmokeError('load_smoke_process_exited')
          } else {
            terminationObserved = true
          }
        } catch {
          cleanupError = loadSmokeError('load_smoke_termination_failed')
        }
      }
      if (handle && !cleanupError) {
        try {
          // `close` is the stdio-drain boundary. Only this post-close snapshot can
          // prove that a late warm-up marker was not omitted from the receipt.
          const observed = handle.logMarkers()
          expect(observed.warming_up_seen === false
            && observed.generation_warmup_complete_seen === false,
          'load_smoke_warmup_detected')
          logMarkers = observed
        } catch (error) {
          primaryError = error instanceof SmolLM3LoadSmokeError
            ? error
            : loadSmokeError('load_smoke_warmup_detected')
        }
      }
    }

    if (!cleanupError) {
      try { assertArtifactLockHeld() } catch (error) { primaryError = error }
    }
    if (cleanupError) throw cleanupError
    if (primaryError) throw primaryError
    expect(terminationObserved
      && resources?.samples > 0
      && resources.peak_child_working_set_bytes > 0
      && resources.thresholds_tripped === false,
      'load_smoke_resource_telemetry_unavailable')
    postflight = await whileArtifactLockHeld(() => (
      deps.postflightImpl
        ? deps.postflightImpl(options, preflight)
        : runPostflight(options, preflight, deps)
    ))
    expect(sameJson(postflight.provenance, preflight.provenance)
      && sameJson(postflight.auto_select_roots, preflight.auto_select_roots),
    'load_smoke_source_changed')
    expect(postflight?.artifact?.size_bytes === lockedPreloadArtifact.size_bytes
      && postflight.artifact.sha256 === lockedPreloadArtifact.sha256
      && postflight.artifact.verified_after_generation === true
      && postflight.artifact.path_redacted === true,
    'load_smoke_artifact_identity_mismatch')
    healthBuild = steps.find((step) => step.name === 'baseline_health').evidence.build
    expect(steps.filter((step) => step.evidence?.build)
      .every((step) => step.evidence.build === healthBuild), 'load_smoke_source_changed')
    expect(healthBuild === preflight.provenance.source_describe, 'load_smoke_source_changed')
  } catch (error) {
    evidenceError = error instanceof SmolLM3LoadSmokeError
      ? error
      : loadSmokeError('load_smoke_http_failed')
  }

  let lockRelease
  if (artifactLock.isExited()) {
    try {
      const timeout = boundedTimeout(2_000, null)
      const observation = await Promise.race([
        Promise.all([artifactLock.exited, artifactLock.closed]),
        timeout.promise,
      ])
      timeout.cancel()
      const observedExit = observation?.[0]
      expect(observedExit && sameJson(observedExit, artifactLock.exitStatus()),
        'load_smoke_artifact_lock_release_failed')
    } catch {
      throw loadSmokeError('load_smoke_artifact_lock_release_failed')
    }
    const cleanupCodes = new Set([
      'load_smoke_resource_telemetry_unavailable', 'load_smoke_termination_failed',
    ])
    if (!cleanupCodes.has(evidenceError?.code)) {
      evidenceError = loadSmokeError('load_smoke_artifact_lock_lost')
    }
  } else {
    try {
      lockRelease = await artifactLock.release()
      expect(lockRelease?.observed === true
        && lockRelease?.released_token_observed === true
        && lockRelease?.exit_code === 0,
      'load_smoke_artifact_lock_release_failed')
    } catch {
      throw loadSmokeError('load_smoke_artifact_lock_release_failed')
    }
  }
  if (evidenceError) throw evidenceError

  const receiptArtifact = {
    size_bytes: postflight.artifact.size_bytes,
    sha256: postflight.artifact.sha256,
    verified_after_lock_acquisition: true,
    verified_after_generation: true,
    mutation_guard: {
      mechanism: 'windows_file_stream_share_read',
      read_access: 'allowed',
      write_access: 'denied',
      delete_access: 'denied',
      rename_access: 'denied',
      symbolic_links_rejected: true,
      acquired_before_preload_hash: true,
      held_through_post_generation_hash: true,
      released_token_observed: true,
      helper_exit_code: 0,
      artifact_path_in_helper_argv: false,
    },
    path_redacted: true,
  }
  const receipt = buildReceipt({
    preflight,
    postflightArtifact: receiptArtifact,
    steps,
    resources,
    logMarkers,
    healthBuild,
    createdUtc: (deps.nowIsoImpl || (() => new Date().toISOString()))(),
  })
  return assertValidLoadSmokeReceipt(receipt)
}

async function writeReceiptAtomic(path, receipt, {
  mkdirImpl = mkdir,
  writeFileImpl = writeFile,
  renameImpl = rename,
  rmImpl = rm,
} = {}) {
  assertValidLoadSmokeReceipt(receipt)
  const target = resolve(path)
  const temporary = `${target}.tmp-${randomBytes(8).toString('hex')}`
  try {
    await mkdirImpl(dirname(target), { recursive: true })
    await writeFileImpl(temporary, `${JSON.stringify(receipt, null, 2)}\n`, { flag: 'wx' })
    await renameImpl(temporary, target)
  } catch {
    try { await rmImpl(temporary, { force: true }) } catch { /* output remains failed closed */ }
    throw loadSmokeError('load_smoke_output_failed')
  }
}

const CLI_VALUE_OPTIONS = new Set([
  'root', 'binary', 'artifact', 'cwd', 'models-dir', 'binary-profile', 'out',
])
const CLI_BOOLEAN_OPTIONS = new Set(['help'])

function parseArgs(argv) {
  expect(Array.isArray(argv), 'load_smoke_options_invalid')
  const args = new Map()
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index]
    expect(typeof value === 'string' && value.startsWith('--') && value.length > 2,
      'load_smoke_options_invalid')
    const equals = value.indexOf('=')
    const key = value.slice(2, equals < 0 ? undefined : equals)
    const inline = equals < 0 ? undefined : value.slice(equals + 1)
    expect(CLI_VALUE_OPTIONS.has(key) || CLI_BOOLEAN_OPTIONS.has(key), 'load_smoke_options_invalid')
    expect(!args.has(key), 'load_smoke_options_invalid')
    if (CLI_BOOLEAN_OPTIONS.has(key)) {
      expect(inline === undefined, 'load_smoke_options_invalid')
      args.set(key, true)
      continue
    }
    let argument = inline
    if (argument === undefined) {
      const next = argv[index + 1]
      expect(typeof next === 'string' && next.length > 0 && !next.startsWith('--'),
        'load_smoke_options_invalid')
      argument = next
      index += 1
    }
    expect(argument.trim().length > 0, 'load_smoke_options_invalid')
    args.set(key, argument)
  }
  expect(!args.has('help') || args.size === 1, 'load_smoke_options_invalid')
  if (!args.has('help')) {
    for (const required of ['artifact', 'cwd', 'models-dir']) {
      expect(args.has(required), 'load_smoke_options_invalid')
    }
  }
  return args
}

async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv)
  if (args.has('help')) {
    process.stdout.write(`Usage: node scripts/hf-qualification-smollm3-load-smoke.mjs [options]\n\n`
      + '  --binary <path>       Frozen optimized Camelid binary\n'
      + '  --artifact <path>     Ignored exact SmolLM3 Q8_0 GGUF\n'
      + '  --cwd <path>          Dedicated candidate-free run directory\n'
      + '  --models-dir <path>   Dedicated candidate-free models directory\n'
      + '  --root <path>         Clean tracked Camelid source root (default: .)\n'
      + '  --binary-profile <id> Provenance label (default: release-fat-lto)\n'
      + '  --out <path>          Atomically write the privacy-safe sealed receipt\n')
    return
  }
  const root = resolve(args.get('root') || '.')
  const defaultBinary = process.platform === 'win32'
    ? 'target/model-qualification/phase1/bin/camelid-opt3.exe'
    : 'target/model-qualification/phase1/bin/camelid-opt3'
  const receipt = await runSmolLM3LoadSmoke({
    root,
    binary: resolve(root, args.get('binary') || defaultBinary),
    artifact: resolve(root, args.get('artifact') || ''),
    cwd: resolve(root, args.get('cwd') || ''),
    modelsDir: resolve(root, args.get('models-dir') || ''),
    binaryProfile: args.get('binary-profile') || BINARY_PROFILE,
  })
  if (args.get('out')) await writeReceiptAtomic(resolve(root, args.get('out')), receipt)
  process.stdout.write(`${JSON.stringify(receipt, null, 2)}\n`)
}

export {
  BINARY_PROFILE,
  CHAT_REQUEST,
  DOES_NOT_PROVE,
  EXECUTION_PLAN_EXACT_MODEL_ROW,
  EXACT_ROW,
  EXPECTED_TEMPLATE_CAPS,
  LEGACY_STORAGE_LABELS,
  LIMITS,
  RAW_REQUEST,
  RECEIPT_SCHEMA,
  ROW_ID,
  SAFE_CAMELID_ENV,
  SERVER_ADDR,
  STEP_CONTRACT,
  SmolLM3LoadSmokeError,
  acquireWindowsArtifactReadLock,
  assertAutoSelectRootsEmpty,
  assertValidLoadSmokeReceipt,
  buildChildEnv,
  buildReceipt,
  buildServeArgs,
  canonicalJson,
  classifySmolLM3LoadSmokeError,
  createResourceGuard,
  httpJson,
  inspectProvenance,
  inspectExactArtifactIdentity,
  normalizeGeneration,
  normalizeHealth,
  normalizeProps,
  parseArgs,
  receiptCommand,
  runPostflight,
  runPreflight,
  runSmolLM3LoadSmoke,
  sealReceipt,
  startCamelidProcess,
  terminateSpawnedChild,
  validateLoadSmokeReceipt,
  writeReceiptAtomic,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    const failure = classifySmolLM3LoadSmokeError(error)
    console.error(`${failure.error_code}: ${failure.reason}`)
    process.exit(1)
  })
}
