#!/usr/bin/env node

import { createHash, randomBytes } from 'node:crypto'
import { execFile } from 'node:child_process'
import { createReadStream } from 'node:fs'
import { lstat, mkdir, rename, rm, stat, statfs, writeFile } from 'node:fs/promises'
import { freemem } from 'node:os'
import { createServer } from 'node:net'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { pathToFileURL } from 'node:url'
import { setImmediate as yieldImmediate, setTimeout as sleep } from 'node:timers/promises'
import { promisify, types as utilTypes } from 'node:util'

// These are row-neutral lifecycle primitives already exercised by the guarded
// SmolLM3 gate. Every Qwen identity, request, response, receipt, and privacy
// contract remains owned and exact-closed in this file.
import {
  acquireWindowsArtifactReadLock,
  assertAutoSelectRootsEmpty,
  canonicalJson,
  createResourceGuard,
  classifySmolLM3LoadSmokeError,
  inspectProvenance,
  sealReceipt,
  SmolLM3LoadSmokeError,
  startCamelidProcess,
  terminateSpawnedChild,
} from './hf-qualification-smollm3-load-smoke.mjs'

const execFileAsync = promisify(execFile)

const RECEIPT_SCHEMA = 'camelid.model-qualification.load-smoke/v2'
const ROW_ID = 'qwen2_5_0_5b_instruct_q8_0'
const BINARY_PROFILE = 'release-fat-lto'
const EXECUTION_PLAN_EXACT_MODEL_ROW = 'qwen2.5-0.5b-instruct'
const SERVER_ADDR = '127.0.0.1:8298'
const SERVER_ORIGIN = `http://${SERVER_ADDR}`
const DIAGNOSTICS_STATUS = 'operator-requested RSS timings enabled; performance claims disabled'

const EXACT_ROW = Object.freeze({
  id: ROW_ID,
  family: 'qwen2_5',
  architecture: 'qwen2',
  quantization: 'Q8_0',
  target_tier: 'experimental_exact_row',
  disposition: 'active_validation',
  source: Object.freeze({
    repo: 'Qwen/Qwen2.5-0.5B-Instruct-GGUF',
    file: 'qwen2.5-0.5b-instruct-q8_0.gguf',
    revision: '9217f5db79a29953eb74d5343926648285ec7e67',
    size_bytes: 675_710_816,
    sha256: 'ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e',
    license: 'apache-2.0',
  }),
})

// Keep the already-audited safety floor. The 0.5B row is smaller than the
// SmolLM3 row, but a smaller row is not a reason to weaken preflight/abort
// protections or enable a faster unqualified runtime lane.
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

const WINDOWS_CHILD_ENV_ALLOWLIST = Object.freeze([
  'COMSPEC',
  'PATH',
  'PATHEXT',
  'SYSTEMDRIVE',
  'SYSTEMROOT',
  'TEMP',
  'TMP',
  'WINDIR',
])
const CHILD_ENV_COMMITMENT_DOMAIN = 'camelid.model-qualification.child-environment/v1'

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

const STEP_CONTRACT = Object.freeze([
  Object.freeze(['baseline_health', 'GET', '/v1/health']),
  Object.freeze(['baseline_gpu', 'GET', '/api/runtime/gpu']),
  Object.freeze(['load', 'POST', '/models/load']),
  Object.freeze(['verify_identity', 'GET', '/api/models/verify']),
  Object.freeze(['loaded_health', 'GET', '/v1/health']),
  Object.freeze(['raw_first_forward', 'POST', '/v1/completions']),
  Object.freeze(['final_health', 'GET', '/v1/health']),
  Object.freeze(['final_gpu', 'GET', '/api/runtime/gpu']),
])

const DOES_NOT_PROVE = Object.freeze([
  'llama.cpp token or text parity; the existing strict parity gate remains failed',
  'output correctness beyond one finite greedy token',
  'chat-completions, streaming SSE, repeated requests, Models-page, or WebUI readiness',
  'chat-template coverage beyond the separately pinned one-user-turn shape',
  'the separately qualified 512-token context bucket or any larger context',
  'performance or throughput',
  'GPU execution',
  'support or promotion',
  'adjacent sizes, variants, or quantizations',
  'system, tools, multi-turn, multimodal, or non-text chat shapes',
])

const LEGACY_STORAGE_LABELS = Object.freeze([
  'execution_plan.prefill_runtime_policy',
  'execution_plan.fallback_path',
])

const ERROR_CONTRACTS = Object.freeze({
  load_smoke_options_invalid: ['blocked', 'the Qwen2.5 load-smoke invocation is incomplete or unsafe'],
  load_smoke_platform_invalid: ['blocked', 'the exact gate requires Windows x86_64'],
  load_smoke_artifact_unavailable: ['blocked', 'the ignored exact Qwen2.5 artifact is unavailable'],
  load_smoke_artifact_identity_mismatch: ['fail', 'the artifact does not match the exact Qwen2.5 source lock'],
  load_smoke_artifact_lock_failed: ['blocked', 'the exact artifact could not be protected by a Windows read-share lock'],
  load_smoke_artifact_lock_lost: ['fail', 'the Windows artifact read-share lock exited before evidence collection completed'],
  load_smoke_artifact_lock_release_failed: ['blocked', 'the Windows artifact read-share lock could not be released and observed cleanly'],
  load_smoke_artifact_not_ignored: ['blocked', 'the full artifact is not contained under an ignored path'],
  load_smoke_source_dirty: ['blocked', 'tracked source files must be clean before freezing runtime provenance'],
  load_smoke_binary_stale: ['blocked', 'the frozen binary version does not exactly match the clean source describe'],
  load_smoke_source_changed: ['blocked', 'source, binary, selector roots, or provenance changed during the gate'],
  load_smoke_auto_select_root_invalid: ['blocked', 'an auto-selection root could not be verified'],
  load_smoke_auto_select_candidate_present: ['blocked', 'an auto-selection root contains a model candidate or saved selector'],
  load_smoke_port_in_use: ['blocked', 'the isolated loopback qualification port is already in use'],
  load_smoke_llama_server_present: ['blocked', 'a llama-server process must not overlap this first-forward gate'],
  load_smoke_resources_low: ['blocked', 'preflight disk or physical memory is below the fixed safety budget'],
  load_smoke_process_start_failed: ['blocked', 'the isolated Camelid child could not start'],
  load_smoke_process_exited: ['fail', 'the isolated Camelid child exited before the gate completed'],
  load_smoke_startup_timeout: ['blocked', 'the no-model server did not become healthy within the startup budget'],
  load_smoke_http_failed: ['blocked', 'an isolated loopback request failed or timed out'],
  load_smoke_health_invalid: ['fail', 'health did not match the exact unloaded or Qwen2.5 loaded contract'],
  load_smoke_gpu_invalid: ['fail', 'GPU telemetry did not remain disabled and unused'],
  load_smoke_load_invalid: ['fail', 'the local load alias did not return the exact redacted readiness contract'],
  load_smoke_verify_invalid: ['fail', 'GET model verification did not bind the active model to the exact GGUF identity'],
  load_smoke_raw_invalid: ['fail', 'the raw first-forward response did not meet the exact one-token evidence contract'],
  load_smoke_resource_abort: ['blocked', 'the child crossed a fixed memory safety abort threshold'],
  load_smoke_resource_telemetry_unavailable: ['blocked', 'the required child resource telemetry became unavailable'],
  load_smoke_warmup_detected: ['fail', 'startup generation warm-up was observed in the no-model run'],
  load_smoke_termination_failed: ['blocked', 'the exact spawned child could not be terminated'],
  load_smoke_receipt_invalid: ['fail', 'the compact Qwen2.5 load-smoke receipt failed its durable contract'],
  load_smoke_output_failed: ['blocked', 'the sealed receipt could not be written atomically'],
})

const QWEN_ERROR_CODES = new WeakMap()

class Qwen25LoadSmokeError extends Error {
  constructor(code) {
    const canonical = Object.hasOwn(ERROR_CONTRACTS, code) ? code : 'load_smoke_http_failed'
    super(ERROR_CONTRACTS[canonical][1])
    this.name = 'Qwen25LoadSmokeError'
    this.status = ERROR_CONTRACTS[canonical][0]
    QWEN_ERROR_CODES.set(this, canonical)
    Object.defineProperty(this, 'code', {
      configurable: false,
      enumerable: true,
      get: () => QWEN_ERROR_CODES.get(this),
    })
  }
}

function loadSmokeError(code) {
  return new Qwen25LoadSmokeError(code)
}

function errorCode(error) {
  if (error instanceof Qwen25LoadSmokeError) {
    const code = QWEN_ERROR_CODES.get(error)
    return Object.hasOwn(ERROR_CONTRACTS, code) ? code : null
  }
  if (error instanceof SmolLM3LoadSmokeError) {
    const bridged = classifySmolLM3LoadSmokeError(error)
    return Object.hasOwn(ERROR_CONTRACTS, bridged.error_code) ? bridged.error_code : null
  }
  return null
}

function classifyQwen25LoadSmokeError(error) {
  const code = errorCode(error) || 'load_smoke_http_failed'
  return { status: ERROR_CONTRACTS[code][0], error_code: code, reason: ERROR_CONTRACTS[code][1] }
}

function expect(condition, code) {
  if (!condition) throw loadSmokeError(code)
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

function sameJson(left, right) {
  return canonicalJson(left) === canonicalJson(right)
}

function finiteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value)
}

function nonNegativeInteger(value) {
  return Number.isSafeInteger(value) && value >= 0
}

function buildChildEnv(inherited = process.env) {
  const clean = {}
  const seenAllowlistedKeys = new Set()
  for (const [key, value] of Object.entries(inherited)) {
    const canonicalKey = key.toUpperCase()
    if (!WINDOWS_CHILD_ENV_ALLOWLIST.includes(canonicalKey)) continue
    expect(!seenAllowlistedKeys.has(canonicalKey), 'load_smoke_options_invalid')
    seenAllowlistedKeys.add(canonicalKey)
    if (typeof value === 'string') clean[canonicalKey] = value
  }
  return Object.freeze({ ...clean, ...SAFE_CAMELID_ENV })
}

function describeChildEnvironment(env) {
  expect(Object.isFrozen(env), 'load_smoke_options_invalid')
  expect(Object.values(env).every((value) => typeof value === 'string'),
    'load_smoke_options_invalid')
  const presentOsKeys = WINDOWS_CHILD_ENV_ALLOWLIST.filter((key) => Object.hasOwn(env, key))
  const inheritedAllowlisted = Object.fromEntries(
    presentOsKeys.map((key) => [key, env[key]]),
  )
  const modelOverrides = Object.fromEntries(
    Object.entries(SAFE_CAMELID_ENV).sort(([left], [right]) => left.localeCompare(right)),
  )
  const effectiveKeys = [...new Set([...presentOsKeys, ...Object.keys(modelOverrides)])].sort()
  expect(sameJson(Object.keys(env).sort(), effectiveKeys)
    && sameJson(childCamelidEnv(env), modelOverrides),
  'load_smoke_options_invalid')
  return {
    model_overrides: structuredClone(modelOverrides),
    windows_allowlist: [...WINDOWS_CHILD_ENV_ALLOWLIST],
    present_os_keys: presentOsKeys,
    inherited_values_redacted: true,
    inherited_allowlisted_values_commitment: {
      domain: CHILD_ENV_COMMITMENT_DOMAIN,
      algorithm: 'sha256',
      digest: sha256(Buffer.from(
        `${CHILD_ENV_COMMITMENT_DOMAIN}\0${canonicalJson(inheritedAllowlisted)}`,
        'utf8',
      )),
    },
    effective_keys: effectiveKeys,
  }
}

function childCamelidEnv(env) {
  return Object.fromEntries(
    Object.entries(env)
      .filter(([key]) => key.toUpperCase().startsWith('CAMELID_'))
      .sort(([left], [right]) => left.localeCompare(right)),
  )
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
  const placeholder = resolve('<empty-models-dir>')
  return ['<camelid>', ...buildServeArgs(placeholder).map((value) => (
    value === placeholder ? '<empty-models-dir>' : value
  ))]
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
  let linkStats
  let fileStats
  try {
    linkStats = await lstatImpl(path)
  } catch (error) {
    if (errorCode(error)) throw error
    throw loadSmokeError('load_smoke_artifact_unavailable')
  }
  expect(linkStats.isFile?.() === true && linkStats.isSymbolicLink?.() === false,
    'load_smoke_artifact_identity_mismatch')
  try {
    fileStats = await statImpl(path)
  } catch (error) {
    if (errorCode(error)) throw error
    throw loadSmokeError('load_smoke_artifact_unavailable')
  }
  expect(fileStats.isFile?.() === true && fileStats.size === EXACT_ROW.source.size_bytes,
    'load_smoke_artifact_identity_mismatch')
  let fileSha256
  try {
    fileSha256 = await sha256FileImpl(path)
  } catch (error) {
    if (errorCode(error)) throw error
    throw loadSmokeError('load_smoke_artifact_unavailable')
  }
  expect(fileSha256 === EXACT_ROW.source.sha256, 'load_smoke_artifact_identity_mismatch')
  return { size_bytes: fileStats.size, sha256: fileSha256 }
}

function pathInside(parent, candidate) {
  const rel = relative(resolve(parent), resolve(candidate))
  return rel === '' || (!rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel))
}

function autoSelectRoots({ binary, cwd, modelsDir }) {
  return [
    resolve(modelsDir),
    join(dirname(resolve(binary)), 'models'),
    dirname(resolve(binary)),
    join(resolve(cwd), 'models'),
    resolve(cwd),
  ]
}

async function gitPathIgnored(root, path, { execFileImpl = execFileAsync } = {}) {
  try {
    await execFileImpl('git', ['-C', root, 'check-ignore', '--quiet', '--', path], {
      timeout: 10_000,
      windowsHide: true,
    })
    return true
  } catch {
    return false
  }
}

async function assertPortFree({ createServerImpl = createServer } = {}) {
  await new Promise((resolvePromise, rejectPromise) => {
    const server = createServerImpl()
    server.unref?.()
    server.once('error', () => rejectPromise(loadSmokeError('load_smoke_port_in_use')))
    server.listen(Number(SERVER_ADDR.split(':')[1]), '127.0.0.1', () => {
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
      '-NoProfile', '-NonInteractive', '-Command',
      "@(Get-Process -Name 'llama-server' -ErrorAction SilentlyContinue).Count",
    ], { timeout: 10_000, windowsHide: true })
    return Number(String(stdout).trim()) > 0
  } catch {
    throw loadSmokeError('load_smoke_auto_select_root_invalid')
  }
}

async function diskFreeBytes(path, { statfsImpl = statfs } = {}) {
  try {
    const info = await statfsImpl(path)
    const bytes = Number(info.bavail) * Number(info.bsize)
    expect(nonNegativeInteger(bytes), 'load_smoke_resources_low')
    return bytes
  } catch (error) {
    if (errorCode(error)) throw error
    throw loadSmokeError('load_smoke_resources_low')
  }
}

function assertFrozenProvenance(provenance) {
  const sourceMatch = /^(?:([0-9a-f]{7,40})|.*-g([0-9a-f]{7,40}))$/i
    .exec(provenance?.source_describe || '')
  const sourceAbbreviation = sourceMatch?.[1] || sourceMatch?.[2] || ''
  expect(provenance
    && /^[0-9a-f]{40}$/.test(provenance.runtime_head || '')
    && typeof provenance.source_describe === 'string'
    && !/-dirty/i.test(provenance.source_describe)
    && sourceAbbreviation.length >= 7
    && provenance.runtime_head.startsWith(sourceAbbreviation.toLowerCase())
    && provenance.tracked_files_clean === true
    && provenance.untracked_files_excluded === true,
  'load_smoke_source_dirty')
  expect(provenance.binary_profile === BINARY_PROFILE
    && /^[0-9a-f]{64}$/.test(provenance.binary_sha256 || '')
    && provenance.binary_version === `camelid ${provenance.source_describe}`,
  'load_smoke_binary_stale')
  return provenance
}

async function runPreflight(options, deps = {}) {
  const platform = deps.platformInfo?.() || { platform: process.platform, arch: process.arch }
  expect(platform.platform === 'win32' && platform.arch === 'x64', 'load_smoke_platform_invalid')
  for (const path of [options.root, options.binary, options.artifact, options.cwd, options.modelsDir]) {
    expect(typeof path === 'string' && isAbsolute(path), 'load_smoke_options_invalid')
  }
  expect(resolve(options.cwd) !== resolve(options.modelsDir), 'load_smoke_options_invalid')
  expect(dirname(resolve(options.binary)) !== resolve(options.cwd)
    && dirname(resolve(options.binary)) !== resolve(options.modelsDir),
  'load_smoke_options_invalid')
  for (const root of autoSelectRoots(options)) {
    expect(!pathInside(root, options.artifact), 'load_smoke_options_invalid')
  }

  let linkStats
  let fileStats
  try {
    linkStats = await (deps.lstatImpl || lstat)(options.artifact)
  } catch {
    throw loadSmokeError('load_smoke_artifact_unavailable')
  }
  expect(linkStats.isFile?.() === true && linkStats.isSymbolicLink?.() === false,
  'load_smoke_artifact_identity_mismatch')
  try {
    fileStats = await (deps.statImpl || stat)(options.artifact)
  } catch {
    throw loadSmokeError('load_smoke_artifact_unavailable')
  }
  expect(fileStats.isFile?.() === true && fileStats.size === EXACT_ROW.source.size_bytes,
  'load_smoke_artifact_identity_mismatch')
  const ignored = deps.checkIgnoredImpl
    ? await deps.checkIgnoredImpl(options.root, options.artifact)
    : await gitPathIgnored(options.root, options.artifact, deps)
  expect(ignored === true, 'load_smoke_artifact_not_ignored')

  const provenance = deps.inspectProvenanceImpl
    ? await deps.inspectProvenanceImpl(options)
    : await inspectProvenance(options, deps)
  assertFrozenProvenance(provenance)
  const roots = deps.assertAutoSelectRootsEmptyImpl
    ? await deps.assertAutoSelectRootsEmptyImpl(options)
    : await assertAutoSelectRootsEmpty(options, deps)
  if (deps.assertPortFreeImpl) await deps.assertPortFreeImpl()
  else await assertPortFree(deps)
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
  expect(nonNegativeInteger(availablePhysicalBytes)
    && availablePhysicalBytes >= LIMITS.preflight_physical_bytes
    && nonNegativeInteger(availableDiskBytes)
    && availableDiskBytes >= LIMITS.preflight_disk_bytes,
  'load_smoke_resources_low')

  return {
    platform: 'windows-x86_64',
    artifact: {
      size_bytes: fileStats.size,
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
  const roots = deps.assertAutoSelectRootsEmptyImpl
    ? await deps.assertAutoSelectRootsEmptyImpl(options)
    : await assertAutoSelectRootsEmpty(options, deps)
  const artifact = await inspectExactArtifactIdentity(options.artifact, deps)
  expect(sameJson(provenance, preflight.provenance)
    && sameJson(roots, preflight.auto_select_roots), 'load_smoke_source_changed')
  return {
    provenance,
    auto_select_roots: roots,
    artifact: { ...artifact, verified_after_generation: true, path_redacted: true },
  }
}

async function readResponseTextBounded(response, controller) {
  const failure = () => (
    errorCode(controller.signal.reason)
      ? controller.signal.reason
      : loadSmokeError('load_smoke_http_failed')
  )
  const declaredHeader = response?.headers?.get?.('content-length')
  if (declaredHeader !== null && declaredHeader !== undefined) {
    const normalized = String(declaredHeader).trim()
    const declared = Number(normalized)
    if (!/^\d+$/.test(normalized)
      || !Number.isSafeInteger(declared)
      || declared > LIMITS.max_response_bytes) {
      const error = loadSmokeError('load_smoke_http_failed')
      controller.abort(error)
      try { await response?.body?.cancel?.(error) } catch { /* bounded failure */ }
      throw error
    }
  }

  if (!response?.body || typeof response.body.getReader !== 'function') {
    const error = loadSmokeError('load_smoke_http_failed')
    controller.abort(error)
    try { await response?.body?.cancel?.(error) } catch { /* bounded failure */ }
    throw error
  }

  const reader = response.body.getReader()
  const chunks = []
  let bytes = 0
  let rejectAbort
  let cancelPromise = Promise.resolve()
  const abortPromise = new Promise((_resolvePromise, rejectPromise) => {
    rejectAbort = rejectPromise
  })
  void abortPromise.catch(() => {})
  const cancelForAbort = () => {
    const error = failure()
    cancelPromise = Promise.resolve(reader.cancel(error)).catch(() => {})
    rejectAbort(error)
  }
  controller.signal.addEventListener('abort', cancelForAbort, { once: true })
  if (controller.signal.aborted) cancelForAbort()
  try {
    while (true) {
      const { done, value } = await Promise.race([reader.read(), abortPromise])
      if (done) break
      expect(ArrayBuffer.isView(value), 'load_smoke_http_failed')
      if (value.byteLength > LIMITS.max_response_bytes - bytes) {
        const error = loadSmokeError('load_smoke_http_failed')
        controller.abort(error)
        await cancelPromise
        throw error
      }
      bytes += value.byteLength
      chunks.push(Buffer.from(value.buffer, value.byteOffset, value.byteLength))
    }
    if (controller.signal.aborted) throw failure()
    return Buffer.concat(chunks, bytes).toString('utf8')
  } finally {
    controller.signal.removeEventListener('abort', cancelForAbort)
    if (controller.signal.aborted) await cancelPromise
    try { reader.releaseLock?.() } catch { /* already cancelled */ }
  }
}

async function httpJson({ method, endpoint, body, timeoutMs, signal, fetchImpl = fetch }) {
  expect(/^\/(?:api\/|v1\/|models\/)/.test(endpoint), 'load_smoke_http_failed')
  if (signal?.aborted) throw signal.reason || loadSmokeError('load_smoke_http_failed')
  const controller = new AbortController()
  const abort = () => controller.abort(signal?.reason || loadSmokeError('load_smoke_http_failed'))
  signal?.addEventListener('abort', abort, { once: true })
  if (signal?.aborted) abort()
  const timeout = setTimeout(() => controller.abort(loadSmokeError('load_smoke_http_failed')), timeoutMs)
  try {
    const response = await fetchImpl(`${SERVER_ORIGIN}${endpoint}`, {
      method,
      headers: body === undefined ? undefined : { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    })
    const text = await readResponseTextBounded(response, controller)
    let parsed
    try { parsed = JSON.parse(text) } catch { throw loadSmokeError('load_smoke_http_failed') }
    return { status: response.status, body: parsed }
  } catch (error) {
    if (signal?.aborted && signal.reason) throw signal.reason
    if (errorCode(error)) throw error
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
    lazy_q8_linear: true,
    retain_q8_blocks: false,
    file_cache_bytes: 0,
  }
}

function normalizeHealth(body, { loaded, final = false } = {}) {
  const code = 'load_smoke_health_invalid'
  expect(body && body.ok === true && body.engine === 'camelid'
    && typeof body.version === 'string' && body.version.length > 0
    && typeof body.build === 'string' && body.build.length > 0, code)
  expect(body.loaded_now === loaded && body.generation_ready === loaded
    && body.vision_ready === false
    && body.active_model_id === (loaded ? ROW_ID : null)
    && body.backend === (loaded ? 'llama' : 'none')
    && body.model_family === (loaded ? 'llama-family' : null), code)
  expect(body.engine_queue_depth === 0
    && body.engine_queued_tasks === 0
    && body.engine_active_task_id === null
    && body.engine_active_generated_tokens === 0
    && body.continuous_batch_slots === 1, code)
  expect(typeof body.executable === 'string' && body.executable.length > 0
    && body.listen_addr === SERVER_ADDR, code)

  const plan = body.execution_plan
  if (loaded) {
    expect(plan
      && plan.profile === 'safe'
      && plan.operating_system === 'windows'
      && plan.architecture === 'x86_64'
      && plan.model_family === 'qwen2'
      && plan.quant_type === 'Q8_0'
      && plan.exact_model_row === EXECUTION_PLAN_EXACT_MODEL_ROW
      && plan.support_level === 'unknown_or_unvalidated'
      && plan.selected_backend === 'cpu_reference'
      && plan.selected_q8_path === 'safe_dense_or_q8_cpu'
      && plan.diagnostics_status === DIAGNOSTICS_STATUS
      && plan.cuda_resident_active === false, code)
  } else {
    expect(plan === null, code)
  }
  if (final) {
    expect(body.engine_active_elapsed_seconds === 0 && body.engine_stalled_seconds === 0, code)
  }

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
      profile: plan.profile,
      operating_system: plan.operating_system,
      architecture: plan.architecture,
      model_family: plan.model_family,
      quant_type: plan.quant_type,
      exact_model_row: plan.exact_model_row,
      support_level: plan.support_level,
      selected_backend: plan.selected_backend,
      selected_q8_path: plan.selected_q8_path,
      diagnostics_status: plan.diagnostics_status,
      cuda_resident_active: false,
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
    && typeof body.backend === 'string' && body.backend.length > 0,
  'load_smoke_gpu_invalid')
  return {
    available: body.available,
    enabled: false,
    backend: body.backend,
    run_count: 0,
    device_redacted: true,
  }
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
    model_id: ROW_ID,
    gguf_sha256: EXACT_ROW.source.sha256,
    eligible: false,
    profile_id: null,
    report: null,
  }
}

const MATERIALIZATION_NUMERIC_KEYS = Object.freeze([
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
])

const Q8_READ_KEYS = Object.freeze([
  'read_calls', 'read_bytes', 'cache_hits', 'cache_hit_bytes', 'cache_misses',
  'cache_miss_bytes', 'cache_inserts', 'cache_insert_bytes', 'cache_evictions',
  'cache_evicted_bytes', 'cache_merges', 'cache_merged_bytes',
  'cache_decoded_scale_hits', 'cache_decoded_scale_hit_blocks', 'cache_entries',
  'cache_bytes', 'cache_capacity_bytes',
])

function normalizeMaterialization(value, code) {
  expect(value && MATERIALIZATION_NUMERIC_KEYS.every((key) => nonNegativeInteger(value[key]))
    && typeof value.has_q8_0_f32_materialization === 'boolean'
    && typeof value.has_lazy_q8_0_file_backing === 'boolean'
    && typeof value.has_retained_q8_0_blocks === 'boolean', code)
  return Object.fromEntries([
    ...MATERIALIZATION_NUMERIC_KEYS.map((key) => [key, value[key]]),
    ['has_q8_0_f32_materialization', value.has_q8_0_f32_materialization],
    ['has_lazy_q8_0_file_backing', value.has_lazy_q8_0_file_backing],
    ['has_retained_q8_0_blocks', value.has_retained_q8_0_blocks],
  ])
}

function normalizeQ8Reads(value, code) {
  expect(value && Q8_READ_KEYS.every((key) => nonNegativeInteger(value[key])), code)
  return Object.fromEntries(Q8_READ_KEYS.map((key) => [key, value[key]]))
}

function normalizeMemoryPhase(phase, value, code) {
  expect(value && Number.isSafeInteger(value.forward_passes) && value.forward_passes > 0, code)
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
  return [
    ['prefill', timings?.prompt_evaluation?.prefill_memory],
    ['first_token', timings?.prompt_evaluation?.first_token_memory],
    ['generation', timings?.memory],
  ].filter(([, value]) => value !== undefined && value !== null)
    .map(([phase, value]) => normalizeMemoryPhase(phase, value, code))
}

function assertRawMemory(phases) {
  const code = 'load_smoke_raw_invalid'
  expect(Array.isArray(phases) && phases.length > 0, code)
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
    && phase.q8_file_reads.cache_capacity_bytes === 0
    && phase.q8_file_reads.cache_hits === 0
    && phase.q8_file_reads.cache_hit_bytes === 0
    && phase.q8_file_reads.cache_inserts === 0
    && phase.q8_file_reads.cache_insert_bytes === 0
    && phase.q8_file_reads.cache_evictions === 0
    && phase.q8_file_reads.cache_evicted_bytes === 0
    && phase.q8_file_reads.cache_merges === 0
    && phase.q8_file_reads.cache_merged_bytes === 0
    && phase.q8_file_reads.cache_decoded_scale_hits === 0
    && phase.q8_file_reads.cache_decoded_scale_hit_blocks === 0), code)
}

function normalizeGeneration(body) {
  const code = 'load_smoke_raw_invalid'
  expect(body && body.model === ROW_ID
    && Array.isArray(body.choices) && body.choices.length === 1, code)
  expect(body.usage?.completion_tokens === 1
    && nonNegativeInteger(body.usage.prompt_tokens) && body.usage.prompt_tokens > 0
    && body.usage.total_tokens === body.usage.prompt_tokens + 1, code)
  expect(!Object.hasOwn(body, 'camelid_receipt'), code)
  const diagnostics = body.camelid
  expect(diagnostics
    && Array.isArray(diagnostics.prompt_token_ids) && diagnostics.prompt_token_ids.length > 0
    && diagnostics.prompt_token_ids.every(nonNegativeInteger)
    && Array.isArray(diagnostics.generated_token_ids) && diagnostics.generated_token_ids.length === 1
    && diagnostics.generated_token_ids.every(nonNegativeInteger), code)
  expect(body.usage.prompt_tokens === diagnostics.prompt_token_ids.length, code)
  const text = body.choices[0]?.text
  expect(typeof text === 'string' && Buffer.byteLength(text, 'utf8') > 0
    && ['length', 'stop'].includes(body.choices[0]?.finish_reason), code)

  const stepTopLogits = Object.hasOwn(diagnostics, 'step_top_logits')
    ? diagnostics.step_top_logits
    : []
  expect(Array.isArray(diagnostics.top_logits) && diagnostics.top_logits.length > 0
    && Array.isArray(stepTopLogits) && stepTopLogits.every(Array.isArray), code)
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
    && nonNegativeInteger(timings.weight_load) && timings.weight_load > 0
    && timings.weight_cache_hit === false
    && timings.prompt_cache_hit === false
    && timings.prompt_evaluation?.first_token_evaluated === true, code)
  const forwardTotal = Number(timings.generation?.forward_total || 0)
  expect(finiteNumber(forwardTotal) && forwardTotal > 0, code)
  const memoryPhases = collectMemoryPhases(timings, code)
  assertRawMemory(memoryPhases)

  return {
    model: ROW_ID,
    support_semantics: 'load_only_existing_strict_parity_failure_unchanged',
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
        rank: 1,
      },
    },
    timings: {
      weight_load: timings.weight_load,
      weight_cache_hit: false,
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
    case 'final_gpu': return normalizeGpu(body)
    case 'load': return normalizeLoad(body)
    case 'verify_identity': return normalizeVerify(body)
    case 'loaded_health': return normalizeHealth(body, { loaded: true })
    case 'raw_first_forward': return normalizeGeneration(body)
    case 'final_health': return normalizeHealth(body, { loaded: true, final: true })
    default: throw loadSmokeError('load_smoke_receipt_invalid')
  }
}

function requestContract() {
  return {
    load: { path_redacted: true, id: ROW_ID, unsupported_fields_omitted: true },
    raw_first_forward: structuredClone(RAW_REQUEST),
    camelid_receipt_requested: false,
  }
}

function buildReceipt({
  preflight,
  postflightArtifact,
  environmentContract,
  steps,
  resources,
  logMarkers,
  healthBuild,
  createdUtc,
}) {
  return sealReceipt({
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
      environment: structuredClone(environmentContract),
      requests: requestContract(),
      limits: structuredClone(LIMITS),
      readiness_semantics: 'load_and_generation_ready_are_attemptability_not_forward_proof',
      first_forward_proof: 'raw_completion_is_the_only_generative_request_and_reports_weight_cache_hit_false',
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
      disposition: 'active_validation',
      target_tier: 'experimental_exact_row',
      authorized_roster_scope: ['gates.load_smoke'],
      existing_parity_gate: 'fail_unchanged',
      other_gates_unchanged: true,
    },
    does_not_prove: [...DOES_NOT_PROVE],
  })
}

function exactKeys(value, keys) {
  return value && typeof value === 'object' && !Array.isArray(value)
    && sameJson(Object.keys(value).sort(), [...keys].sort())
}

const HEALTH_KEYS = Object.freeze([
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
const GPU_KEYS = Object.freeze([
  'available', 'enabled', 'backend', 'run_count', 'device_redacted',
])
const GENERATION_KEYS = Object.freeze([
  'model', 'support_semantics', 'choice_count', 'finish_reason', 'usage', 'prompt_token_ids',
  'generated_token_ids', 'generated_text', 'logits', 'timings', 'memory_phases',
  'camelid_receipt_present',
])
const MATERIALIZATION_KEYS = Object.freeze([
  ...MATERIALIZATION_NUMERIC_KEYS,
  'has_q8_0_f32_materialization', 'has_lazy_q8_0_file_backing', 'has_retained_q8_0_blocks',
])

// Inspect the descriptor graph before the validator reads even `receipt.schema`.
// A receipt is JSON data, so proxies, accessors, cycles, symbols, sparse arrays,
// and non-enumerable properties all fail closed without invoking
// attacker-controlled traps or getters.
function structuralErrors(value) {
  const errors = []
  const seen = new WeakSet()
  const stack = [{ node: value, path: '$', depth: 0 }]
  let visited = 0
  while (stack.length) {
    const { node, path, depth } = stack.pop()
    visited += 1
    if (visited > 50_000) {
      errors.push('structural scan exceeded its bounded node budget')
      break
    }
    if (node === null || ['string', 'number', 'boolean'].includes(typeof node)) continue
    if (typeof node !== 'object') {
      errors.push(`${path} contains a non-JSON value`)
      continue
    }
    // Node's native proxy detector does not consult the handler. Reject before
    // even descriptor, prototype, array, or property inspection can run a trap.
    if (utilTypes.isProxy(node)) {
      errors.push(`${path} is a Proxy and cannot be inspected safely`)
      continue
    }
    if (depth >= 128) {
      errors.push(`${path} exceeds the bounded structural depth`)
      continue
    }
    if (seen.has(node)) {
      errors.push(`${path} contains a cycle or repeated object reference`)
      continue
    }
    seen.add(node)
    let descriptors
    let symbols
    let prototype
    try {
      descriptors = Object.getOwnPropertyDescriptors(node)
      symbols = Object.getOwnPropertySymbols(node)
      prototype = Object.getPrototypeOf(node)
    } catch {
      errors.push(`${path} could not be inspected safely`)
      continue
    }
    if (symbols.length) errors.push(`${path} contains symbol-keyed data`)
    const isArray = Array.isArray(node)
    if (isArray ? prototype !== Array.prototype : prototype !== Object.prototype && prototype !== null) {
      errors.push(`${path} has an unexpected prototype`)
    }
    const arrayLength = isArray ? descriptors.length?.value : null
    if (isArray && !nonNegativeInteger(arrayLength)) {
      errors.push(`${path} has an invalid array length descriptor`)
    }
    if (isArray) {
      const elementKeys = Object.keys(descriptors).filter((key) => key !== 'length')
      const canonicalKeys = elementKeys.filter((key) => (
        /^(?:0|[1-9][0-9]*)$/.test(key) && Number(key) < arrayLength
      ))
      if (canonicalKeys.length !== elementKeys.length) {
        errors.push(`${path} contains an unexpected array property`)
      }
      if (arrayLength > 50_000 || canonicalKeys.length !== arrayLength) {
        errors.push(`${path} contains a sparse or oversized array`)
      }
    }
    for (const [key, descriptor] of Object.entries(descriptors)) {
      if (isArray && key === 'length') continue
      const childPath = `${path}.${key}`
      if (!Object.hasOwn(descriptor, 'value')) {
        errors.push(`${childPath} uses an accessor`)
        continue
      }
      if (!descriptor.enumerable) errors.push(`${childPath} is non-enumerable`)
      stack.push({ node: descriptor.value, path: childPath, depth: depth + 1 })
    }
  }
  return [...new Set(errors)]
}

function privacyErrors(value) {
  const errors = []
  const bannedKeys = new Set([
    'hostname', 'pid', 'process_id', 'artifact_path', 'binary_path', 'executable_path',
    'raw_log', 'raw_logs', 'authorization', 'cookie', 'password', 'secret', 'token',
  ])
  const allowedRoutes = new Set(STEP_CONTRACT.map(([, , endpoint]) => endpoint))
  const seen = new WeakSet()
  const stack = [{ node: value, path: '$', depth: 0 }]
  let visited = 0
  while (stack.length) {
    const { node, path, depth } = stack.pop()
    visited += 1
    if (visited > 50_000) {
      errors.push('privacy scan exceeded its bounded node budget')
      break
    }
    if (typeof node === 'string') {
      if (/[A-Za-z]:[\\/]/.test(node) || /\\\\[^\\]/.test(node) || /\bfile:\/\//i.test(node)) {
        errors.push(`${path} contains an absolute local path`)
      }
      if (/\bhf_[A-Za-z0-9]{8,}\b/.test(node)
        || /\b(?:gh[pousr]_[A-Za-z0-9]{12,}|github_pat_[A-Za-z0-9_]{12,})\b/.test(node)
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
    if (depth >= 128) {
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
      const canonicalKeys = elementKeys.filter((key) => (
        /^(?:0|[1-9][0-9]*)$/.test(key) && Number(key) < node.length
      ))
      if (canonicalKeys.length !== elementKeys.length) {
        errors.push(`${path} contains an unexpected array property`)
      }
      if (node.length > 50_000 || canonicalKeys.length !== node.length) {
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

function validateLoadSmokeReceiptUnsafe(receipt) {
  const unsafeStructure = structuralErrors(receipt)
  if (unsafeStructure.length) return unsafeStructure
  const errors = []
  const check = (condition, message) => { if (!condition) errors.push(message) }
  const close = (value, keys, path) => check(exactKeys(value, keys), `${path} keys must be exact`)

  close(receipt, [
    'schema', 'receipt_id', 'created_utc', 'gate', 'row', 'provenance', 'isolation',
    'runtime_contract', 'steps', 'resource_observations', 'gate_decision', 'does_not_prove',
  ], 'receipt')
  check(receipt?.schema === RECEIPT_SCHEMA, 'schema must remain exact')
  check(/^[0-9a-f]{64}$/.test(receipt?.receipt_id || ''), 'receipt_id must be lowercase SHA-256')
  if (receipt && typeof receipt === 'object') {
    const { receipt_id: _receiptId, ...body } = receipt
    check(receipt.receipt_id === sha256(Buffer.from(canonicalJson(body), 'utf8')),
      'receipt_id must seal the canonical body')
  }
  check(typeof receipt?.created_utc === 'string'
    && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(receipt.created_utc)
    && !Number.isNaN(Date.parse(receipt.created_utc))
    && new Date(receipt.created_utc).toISOString() === receipt.created_utc,
  'created_utc must be canonical UTC')
  check(receipt?.gate === 'load_smoke' && sameJson(receipt?.row, EXACT_ROW),
    'receipt must bind the exact Qwen2.5 load-smoke row')

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
    'symbolic_links_rejected', 'acquired_before_preload_hash', 'held_through_post_generation_hash',
    'released_token_observed', 'helper_exit_code', 'artifact_path_in_helper_argv',
  ], 'provenance.artifact.mutation_guard')
  const receiptSourceMatch = /^(?:([0-9a-f]{7,40})|.*-g([0-9a-f]{7,40}))$/i
    .exec(receipt?.provenance?.source_describe || '')
  const receiptSourceAbbreviation = receiptSourceMatch?.[1] || receiptSourceMatch?.[2] || ''
  check(/^[0-9a-f]{40}$/.test(receipt?.provenance?.runtime_head || '')
    && typeof receipt?.provenance?.source_describe === 'string'
    && !/-dirty/i.test(receipt.provenance.source_describe)
    && receiptSourceAbbreviation.length >= 7
    && receipt.provenance.runtime_head.startsWith(receiptSourceAbbreviation.toLowerCase())
    && receipt?.provenance?.tracked_files_clean === true
    && receipt?.provenance?.untracked_files_excluded === true
    && receipt?.provenance?.binary?.profile === BINARY_PROFILE
    && /^[0-9a-f]{64}$/.test(receipt?.provenance?.binary?.sha256 || '')
    && receipt?.provenance?.binary?.version === `camelid ${receipt?.provenance?.source_describe}`
    && receipt?.provenance?.binary?.health_build === receipt?.provenance?.source_describe
    && receipt?.provenance?.binary?.built_from_clean_tracked_head === true,
  'binary provenance must remain clean and exact')
  const artifact = receipt?.provenance?.artifact
  const mutationGuard = artifact?.mutation_guard
  check(artifact?.size_bytes === EXACT_ROW.source.size_bytes
    && artifact?.sha256 === EXACT_ROW.source.sha256
    && artifact?.verified_after_lock_acquisition === true
    && artifact?.verified_after_generation === true
    && artifact?.path_redacted === true
    && mutationGuard?.mechanism === 'windows_file_stream_share_read'
    && mutationGuard?.read_access === 'allowed'
    && mutationGuard?.write_access === 'denied'
    && mutationGuard?.delete_access === 'denied'
    && mutationGuard?.rename_access === 'denied'
    && mutationGuard?.symbolic_links_rejected === true
    && mutationGuard?.acquired_before_preload_hash === true
    && mutationGuard?.held_through_post_generation_hash === true
    && mutationGuard?.released_token_observed === true
    && mutationGuard?.helper_exit_code === 0
    && mutationGuard?.artifact_path_in_helper_argv === false
    && receipt?.provenance?.platform === 'windows-x86_64'
    && receipt?.provenance?.paths_redacted === true
    && receipt?.provenance?.hostname_redacted === true,
  'artifact provenance and mutation guard must remain exact')

  close(receipt?.isolation, [
    'no_startup_model', 'auto_select_roots', 'loopback_only', 'address',
    'qualification_port_unbound_before_start', 'llama_server_absent',
    'harness_request_sequence_exclusive', 'inherited_camelid_env_cleared',
    'child_handle_only_termination', 'child_termination_observed', 'startup_warmup_markers',
  ], 'isolation')
  close(receipt?.isolation?.startup_warmup_markers, [
    'warming_up_seen', 'generation_warmup_complete_seen', 'raw_output_persisted',
  ], 'isolation.startup_warmup_markers')
  check(receipt?.isolation?.no_startup_model === true
    && Array.isArray(receipt?.isolation?.auto_select_roots)
    && receipt.isolation.auto_select_roots.length === 5
    && sameJson(receipt.isolation.auto_select_roots.map((root) => root?.kind), [
      'configured_models_dir', 'executable_models_dir', 'executable_dir', 'cwd_models_dir', 'cwd',
    ])
    && receipt.isolation.auto_select_roots.every((root) => exactKeys(root, [
      'kind', 'exists', 'path_redacted', 'gguf_candidates', 'default_preference_present',
    ]) && typeof root.exists === 'boolean' && root.path_redacted === true
      && root.gguf_candidates === 0 && root.default_preference_present === false)
    && receipt?.isolation?.loopback_only === true
    && receipt?.isolation?.address === SERVER_ADDR
    && receipt?.isolation?.qualification_port_unbound_before_start === true
    && receipt?.isolation?.llama_server_absent === true
    && receipt?.isolation?.harness_request_sequence_exclusive === true
    && receipt?.isolation?.inherited_camelid_env_cleared === true
    && receipt?.isolation?.child_handle_only_termination === true
    && receipt?.isolation?.child_termination_observed === true
    && receipt?.isolation?.startup_warmup_markers?.warming_up_seen === false
    && receipt?.isolation?.startup_warmup_markers?.generation_warmup_complete_seen === false
    && receipt?.isolation?.startup_warmup_markers?.raw_output_persisted === false,
  'isolation contract must remain exact')

  close(receipt?.runtime_contract, [
    'command', 'cwd_redacted', 'environment', 'requests', 'limits', 'readiness_semantics',
    'first_forward_proof', 'excluded_legacy_storage_labels',
  ], 'runtime_contract')
  const environment = receipt?.runtime_contract?.environment
  const environmentCommitment = environment?.inherited_allowlisted_values_commitment
  close(environment, [
    'model_overrides', 'windows_allowlist', 'present_os_keys', 'inherited_values_redacted',
    'inherited_allowlisted_values_commitment', 'effective_keys',
  ], 'runtime_contract.environment')
  close(environmentCommitment, [
    'domain', 'algorithm', 'digest',
  ], 'runtime_contract.environment.inherited_allowlisted_values_commitment')
  const presentOsKeys = Array.isArray(environment?.present_os_keys)
    ? environment.present_os_keys
    : []
  const expectedEffectiveKeys = [
    ...new Set([...presentOsKeys, ...Object.keys(SAFE_CAMELID_ENV)]),
  ].sort()
  check(sameJson(environment?.model_overrides, SAFE_CAMELID_ENV)
    && sameJson(environment?.windows_allowlist, WINDOWS_CHILD_ENV_ALLOWLIST)
    && Array.isArray(environment?.present_os_keys)
    && sameJson(presentOsKeys, [...new Set(presentOsKeys)].sort())
    && presentOsKeys.every((key) => WINDOWS_CHILD_ENV_ALLOWLIST.includes(key))
    && environment?.inherited_values_redacted === true
    && environmentCommitment?.domain === CHILD_ENV_COMMITMENT_DOMAIN
    && environmentCommitment?.algorithm === 'sha256'
    && /^[0-9a-f]{64}$/.test(environmentCommitment?.digest || '')
    && sameJson(environment?.effective_keys, expectedEffectiveKeys),
  'child environment contract must remain exact, committed, and value-redacted')
  close(receipt?.runtime_contract?.requests, [
    'load', 'raw_first_forward', 'camelid_receipt_requested',
  ], 'runtime_contract.requests')
  close(receipt?.runtime_contract?.requests?.load, [
    'path_redacted', 'id', 'unsupported_fields_omitted',
  ], 'runtime_contract.requests.load')
  check(receipt?.runtime_contract?.cwd_redacted === true
    && sameJson(receipt?.runtime_contract?.command, receiptCommand())
    && !receipt?.runtime_contract?.command?.includes('--model')
    && sameJson(receipt?.runtime_contract?.limits, LIMITS)
    && sameJson(receipt?.runtime_contract?.requests?.raw_first_forward, RAW_REQUEST)
    && receipt?.runtime_contract?.requests?.load?.path_redacted === true
    && receipt?.runtime_contract?.requests?.load?.id === ROW_ID
    && receipt?.runtime_contract?.requests?.load?.unsupported_fields_omitted === true
    && receipt?.runtime_contract?.requests?.camelid_receipt_requested === false
    && receipt?.runtime_contract?.readiness_semantics
      === 'load_and_generation_ready_are_attemptability_not_forward_proof'
    && receipt?.runtime_contract?.first_forward_proof
      === 'raw_completion_is_the_only_generative_request_and_reports_weight_cache_hit_false'
    && sameJson(receipt?.runtime_contract?.excluded_legacy_storage_labels, LEGACY_STORAGE_LABELS),
  'runtime request and safety contract must remain exact')

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
    for (const name of ['baseline_health', 'loaded_health', 'final_health']) {
      const evidence = byName[name]
      close(evidence, HEALTH_KEYS, `steps.${name}.evidence`)
      close(evidence?.q8_runtime, Q8_RUNTIME_KEYS, `steps.${name}.evidence.q8_runtime`)
      close(evidence?.queue, QUEUE_KEYS, `steps.${name}.evidence.queue`)
      if (evidence?.execution_plan !== null) {
        close(evidence?.execution_plan, EXECUTION_PLAN_KEYS,
          `steps.${name}.evidence.execution_plan`)
      }
      check(evidence?.ok === true && evidence?.engine === 'camelid'
        && typeof evidence?.version === 'string' && evidence.version.length > 0
        && evidence?.build === receipt?.provenance?.source_describe
        && evidence?.vision_ready === false
        && evidence?.q8_runtime?.policy === 'forced_lazy_file_backed_q8'
        && evidence?.q8_runtime?.lazy_q8_linear === true
        && evidence?.q8_runtime?.retain_q8_blocks === false
        && evidence?.q8_runtime?.file_cache_bytes === 0
        && evidence?.queue?.depth === 0 && evidence?.queue?.queued_tasks === 0
        && evidence?.queue?.active_task === false
        && evidence?.queue?.active_generated_tokens === 0
        && evidence?.queue?.continuous_batch_slots === 1
        && evidence?.executable_redacted === true && evidence?.listen_addr === SERVER_ADDR,
      `${name} scalar health contract must remain exact`)
    }
    check(byName.loaded_health?.version === byName.baseline_health?.version
      && byName.final_health?.version === byName.baseline_health?.version,
    'all health observations must come from one exact binary version')
    check(byName.baseline_health?.loaded_now === false
      && byName.baseline_health?.generation_ready === false
      && byName.baseline_health?.active_model_id === null
      && byName.baseline_health?.backend === 'none'
      && byName.baseline_health?.model_family === null
      && byName.baseline_health?.execution_plan === null,
    'baseline health must be exactly unloaded')
    for (const name of ['loaded_health', 'final_health']) {
      const evidence = byName[name]
      const plan = evidence?.execution_plan
      check(evidence?.loaded_now === true && evidence?.generation_ready === true
        && evidence?.active_model_id === ROW_ID
        && evidence?.backend === 'llama' && evidence?.model_family === 'llama-family'
        && plan?.profile === 'safe' && plan?.operating_system === 'windows'
        && plan?.architecture === 'x86_64' && plan?.model_family === 'qwen2'
        && plan?.quant_type === 'Q8_0'
        && plan?.exact_model_row === EXECUTION_PLAN_EXACT_MODEL_ROW
        && plan?.support_level === 'unknown_or_unvalidated'
        && plan?.selected_backend === 'cpu_reference'
        && plan?.selected_q8_path === 'safe_dense_or_q8_cpu'
        && plan?.diagnostics_status === DIAGNOSTICS_STATUS
        && plan?.cuda_resident_active === false
        && sameJson(plan?.legacy_storage_labels_excluded, LEGACY_STORAGE_LABELS),
      `${name} must remain the exact unqualified Qwen2.5 Safe CPU plan`)
    }
    for (const name of ['baseline_gpu', 'final_gpu']) {
      close(byName[name], GPU_KEYS, `steps.${name}.evidence`)
      check(typeof byName[name]?.available === 'boolean'
        && byName[name]?.enabled === false
        && typeof byName[name]?.backend === 'string' && byName[name].backend.length > 0
        && byName[name]?.run_count === 0 && byName[name]?.device_redacted === true,
      `${name} must prove the GPU remained unused`)
    }
    close(byName.load, [
      'request', 'id', 'path', 'status', 'generation_ready', 'model_path_redacted',
      'compatibility', 'scope',
    ], 'steps.load.evidence')
    close(byName.load?.request, ['path_redacted', 'id'], 'steps.load.evidence.request')
    check(byName.load?.request?.path_redacted === true && byName.load?.request?.id === ROW_ID
      && byName.load?.id === ROW_ID && byName.load?.path === null
      && byName.load?.status === 'loaded' && byName.load?.generation_ready === true
      && byName.load?.model_path_redacted === true
      && byName.load?.compatibility === 'partial_llama_server_models_load_local_path'
      && byName.load?.scope === 'single_local_model_load_alias',
    'load evidence must remain exact and path-redacted')
    close(byName.verify_identity, [
      'model_id', 'gguf_sha256', 'eligible', 'profile_id', 'report',
    ], 'steps.verify_identity.evidence')
    check(byName.verify_identity?.model_id === ROW_ID
      && byName.verify_identity?.gguf_sha256 === EXACT_ROW.source.sha256
      && byName.verify_identity?.eligible === false
      && byName.verify_identity?.profile_id === null
      && byName.verify_identity?.report === null,
    'verification must bind the exact unsupported artifact')

    const generation = byName.raw_first_forward
    close(generation, GENERATION_KEYS, 'steps.raw_first_forward.evidence')
    close(generation?.usage, ['prompt_tokens', 'completion_tokens', 'total_tokens'],
      'steps.raw_first_forward.evidence.usage')
    close(generation?.generated_text, ['redacted', 'utf8_bytes', 'sha256'],
      'steps.raw_first_forward.evidence.generated_text')
    close(generation?.logits, ['emitted_count', 'all_finite', 'greedy_top'],
      'steps.raw_first_forward.evidence.logits')
    close(generation?.logits?.greedy_top, ['token_id', 'logit', 'probability', 'rank'],
      'steps.raw_first_forward.evidence.logits.greedy_top')
    close(generation?.timings, [
      'weight_load', 'weight_cache_hit', 'prompt_cache_hit', 'first_token_evaluated',
      'forward_total',
    ], 'steps.raw_first_forward.evidence.timings')
    check(generation?.model === ROW_ID
      && generation?.support_semantics === 'load_only_existing_strict_parity_failure_unchanged'
      && generation?.choice_count === 1 && ['length', 'stop'].includes(generation?.finish_reason)
      && nonNegativeInteger(generation?.usage?.prompt_tokens) && generation.usage.prompt_tokens > 0
      && generation.usage.prompt_tokens === generation?.prompt_token_ids?.length
      && generation?.usage?.completion_tokens === 1
      && generation?.usage?.total_tokens === generation?.usage?.prompt_tokens + 1
      && Array.isArray(generation?.prompt_token_ids) && generation.prompt_token_ids.length > 0
      && generation.prompt_token_ids.every(nonNegativeInteger)
      && Array.isArray(generation?.generated_token_ids)
      && generation.generated_token_ids.length === 1
      && generation.generated_token_ids.every(nonNegativeInteger)
      && generation?.generated_text?.redacted === true
      && nonNegativeInteger(generation?.generated_text?.utf8_bytes)
      && generation.generated_text.utf8_bytes > 0
      && /^[0-9a-f]{64}$/.test(generation?.generated_text?.sha256 || '')
      && generation?.logits?.all_finite === true
      && nonNegativeInteger(generation?.logits?.emitted_count)
      && generation.logits.emitted_count > 0
      && generation?.logits?.greedy_top?.token_id === generation?.generated_token_ids?.[0]
      && finiteNumber(generation?.logits?.greedy_top?.logit)
      && finiteNumber(generation?.logits?.greedy_top?.probability)
      && generation.logits.greedy_top.probability >= 0
      && generation.logits.greedy_top.probability <= 1
      && generation?.logits?.greedy_top?.rank === 1
      && nonNegativeInteger(generation?.timings?.weight_load)
      && generation.timings.weight_load > 0
      && generation?.timings?.weight_cache_hit === false
      && generation?.timings?.prompt_cache_hit === false
      && generation?.timings?.first_token_evaluated === true
      && finiteNumber(generation?.timings?.forward_total)
      && generation.timings.forward_total > 0
      && generation?.camelid_receipt_present === false,
    'raw first-forward evidence must remain one-token, finite, uncached, and unsupported')
    if (Array.isArray(generation?.memory_phases)) {
      const names = generation.memory_phases.map((phase) => phase?.phase)
      const canonicalNames = ['prefill', 'first_token', 'generation']
        .filter((phase) => names.includes(phase))
      check(sameJson(names, canonicalNames), 'memory phases must be unique and canonical')
      generation.memory_phases.forEach((phase, index) => {
        close(phase, ['phase', 'forward_passes', 'materialization', 'q8_file_reads', 'peak_rss_kib'],
          `steps.raw_first_forward.evidence.memory_phases.${index}`)
        close(phase?.materialization, MATERIALIZATION_KEYS,
          `steps.raw_first_forward.evidence.memory_phases.${index}.materialization`)
        close(phase?.q8_file_reads, Q8_READ_KEYS,
          `steps.raw_first_forward.evidence.memory_phases.${index}.q8_file_reads`)
        check(Number.isSafeInteger(phase?.forward_passes) && phase.forward_passes > 0
          && (phase?.peak_rss_kib === null || nonNegativeInteger(phase?.peak_rss_kib))
          && MATERIALIZATION_NUMERIC_KEYS.every((key) => (
            nonNegativeInteger(phase?.materialization?.[key])
          ))
          && typeof phase?.materialization?.has_q8_0_f32_materialization === 'boolean'
          && typeof phase?.materialization?.has_lazy_q8_0_file_backing === 'boolean'
          && typeof phase?.materialization?.has_retained_q8_0_blocks === 'boolean'
          && Q8_READ_KEYS.every((key) => nonNegativeInteger(phase?.q8_file_reads?.[key])),
        `memory phase ${index} telemetry must be typed with positive forward passes and non-negative counters`)
      })
    }
    try { assertRawMemory(generation?.memory_phases || []) } catch {
      errors.push('raw lazy-Q8 telemetry must remain exact')
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
    && nonNegativeInteger(receipt?.resource_observations?.minimum_available_physical_bytes)
    && nonNegativeInteger(receipt?.resource_observations?.peak_child_working_set_bytes)
    && receipt.resource_observations.peak_child_working_set_bytes > 0,
  'resource telemetry must be observed without a tripped threshold')

  close(receipt?.gate_decision, [
    'load_smoke', 'support_claim', 'disposition', 'target_tier', 'authorized_roster_scope',
    'existing_parity_gate', 'other_gates_unchanged',
  ], 'gate_decision')
  check(receipt?.gate_decision?.load_smoke === 'pass'
    && receipt?.gate_decision?.support_claim === false
    && receipt?.gate_decision?.disposition === 'active_validation'
    && receipt?.gate_decision?.target_tier === 'experimental_exact_row'
    && sameJson(receipt?.gate_decision?.authorized_roster_scope, ['gates.load_smoke'])
    && receipt?.gate_decision?.existing_parity_gate === 'fail_unchanged'
    && receipt?.gate_decision?.other_gates_unchanged === true,
  'gate decision must remain load-smoke-only with strict parity failed')
  check(sameJson(receipt?.does_not_prove, DOES_NOT_PROVE), 'scope exclusions must remain exact')
  errors.push(...privacyErrors(receipt))
  return [...new Set(errors)]
}

function validateLoadSmokeReceipt(receipt) {
  try { return validateLoadSmokeReceiptUnsafe(receipt) }
  catch { return ['receipt validation could not safely inspect malformed input'] }
}

function assertValidLoadSmokeReceipt(receipt) {
  if (validateLoadSmokeReceipt(receipt).length) throw loadSmokeError('load_smoke_receipt_invalid')
  return receipt
}

function boundedTimeout(ms, value) {
  let timer
  const promise = new Promise((resolvePromise) => {
    timer = setTimeout(() => resolvePromise(value), ms)
    timer.unref?.()
  })
  return { promise, cancel: () => clearTimeout(timer) }
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
  return undefined
}

function timeoutForStep(name) {
  if (name === 'load') return LIMITS.load_timeout_ms
  if (name === 'raw_first_forward') return LIMITS.generation_timeout_ms
  return LIMITS.ordinary_request_timeout_ms
}

async function runQwen25LoadSmoke(rawOptions, deps = {}) {
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
  const env = buildChildEnv(deps.inheritedEnv ?? process.env)
  const expectedCamelidEnv = Object.fromEntries(
    Object.entries(SAFE_CAMELID_ENV).sort(([left], [right]) => left.localeCompare(right)),
  )
  expect(sameJson(childCamelidEnv(env), expectedCamelidEnv), 'load_smoke_options_invalid')
  const args = buildServeArgs(options.modelsDir)
  expect(!args.includes('--model'), 'load_smoke_options_invalid')
  const preflight = deps.preflightImpl
    ? await deps.preflightImpl(options, { env, args })
    : await runPreflight(options, deps)
  assertFrozenProvenance(preflight?.provenance)
  expect(preflight?.platform === 'windows-x86_64'
    && preflight?.artifact?.size_bytes === EXACT_ROW.source.size_bytes
    && preflight?.artifact?.expected_sha256 === EXACT_ROW.source.sha256
    && preflight?.artifact?.ignored === true
    && preflight?.artifact?.path_redacted === true
    && Array.isArray(preflight?.auto_select_roots) && preflight.auto_select_roots.length === 5
    && preflight?.qualification_port_unbound === true
    && preflight?.llama_server_absent === true
    && nonNegativeInteger(preflight?.available_physical_bytes)
    && preflight.available_physical_bytes >= LIMITS.preflight_physical_bytes
    && nonNegativeInteger(preflight?.available_disk_bytes)
    && preflight.available_disk_bytes >= LIMITS.preflight_disk_bytes,
  'load_smoke_options_invalid')

  const nowMs = deps.nowMsImpl || Date.now
  const sleepImpl = deps.sleepImpl || sleep
  const yieldImpl = deps.yieldImpl || yieldImmediate
  const requestImpl = deps.httpJsonImpl
    || ((requestOptions) => httpJson({ ...requestOptions, fetchImpl: deps.fetchImpl }))

  let artifactLock
  try {
    artifactLock = deps.acquireArtifactLockImpl
      ? await deps.acquireArtifactLockImpl(options.artifact)
      : await acquireWindowsArtifactReadLock(options.artifact, deps)
  } catch (error) {
    if (errorCode(error)) throw error
    throw loadSmokeError('load_smoke_artifact_lock_failed')
  }
  const artifactLockValid = artifactLock?.acquired === true
    && artifactLock.exited && artifactLock.closed
    && typeof artifactLock.isExited === 'function'
    && typeof artifactLock.exitStatus === 'function'
    && typeof artifactLock.assertHeld === 'function'
    && typeof artifactLock.release === 'function'
  if (!artifactLockValid) {
    if (artifactLock?.acquired === true) {
      if (typeof artifactLock.release !== 'function') {
        throw loadSmokeError('load_smoke_artifact_lock_release_failed')
      }
      try {
        const released = await artifactLock.release()
        expect(released?.observed === true
          && released?.released_token_observed === true
          && released?.exit_code === 0,
        'load_smoke_artifact_lock_release_failed')
      } catch {
        throw loadSmokeError('load_smoke_artifact_lock_release_failed')
      }
    }
    throw loadSmokeError('load_smoke_artifact_lock_failed')
  }

  const assertArtifactLockHeld = () => {
    try { artifactLock.assertHeld() }
    catch { throw loadSmokeError('load_smoke_artifact_lock_lost') }
  }
  const whileArtifactLockHeld = async (operation) => {
    assertArtifactLockHeld()
    const result = Promise.resolve().then(operation)
      .then((value) => ({ value }), (error) => ({ error }))
    const outcome = await Promise.race([
      result,
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
  let lockedPreloadArtifact
  let postflight
  let healthBuild
  let primaryError = null
  let cleanupError = null
  let evidenceError = null
  let environmentContract
  let terminationObserved = false

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
      try {
        environmentContract = describeChildEnvironment(env)
        handle = deps.startProcessImpl
          ? await deps.startProcessImpl({ binary: options.binary, args, cwd: options.cwd, env })
          : startCamelidProcess({ binary: options.binary, args, cwd: options.cwd, env }, deps)
      } catch {
        throw loadSmokeError('load_smoke_process_start_failed')
      }
      expect(handle && handle.exited && handle.closed
        && typeof handle.kill === 'function'
        && typeof handle.isExited === 'function'
        && typeof handle.isClosed === 'function'
        && typeof handle.exitStatus === 'function'
        && typeof handle.logMarkers === 'function',
      'load_smoke_process_start_failed')
      try {
        guard = deps.createResourceGuardImpl
          ? await deps.createResourceGuardImpl(handle)
          : createResourceGuard(handle, {
            sampleImpl: deps.sampleResourceImpl,
            sleepImpl,
            limits: LIMITS,
          })
      } catch {
        throw loadSmokeError('load_smoke_resource_telemetry_unavailable')
      }
      expect(guard && guard.signal && typeof guard.throwIfAborted === 'function'
        && typeof guard.stop === 'function' && typeof guard.summary === 'function',
      'load_smoke_resource_telemetry_unavailable')

      const assertNoWarmup = () => {
        const observed = handle.logMarkers()
        expect(observed.warming_up_seen === false
          && observed.generation_warmup_complete_seen === false,
        'load_smoke_warmup_detected')
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
          const requestResult = Promise.resolve().then(() => requestImpl({
            method, endpoint, body, timeoutMs: timeoutForStep(name), signal: guard.signal,
          })).then((value) => ({ response: value }), (error) => ({ request_error: error }))
          const outcome = await Promise.race([
            requestResult,
            handle.exited.then((status) => ({ exited: status })),
            artifactLock.exited.then((status) => ({ lock_exited: status })),
          ])
          if (outcome.lock_exited) throw loadSmokeError('load_smoke_artifact_lock_lost')
          if (outcome.exited) throw loadSmokeError('load_smoke_process_exited')
          if (outcome.request_error) {
            await sleepImpl(0)
            if (artifactLock.isExited()) throw loadSmokeError('load_smoke_artifact_lock_lost')
            if (handle.isExited()) throw loadSmokeError('load_smoke_process_exited')
            throw outcome.request_error
          }
          response = outcome.response
        }
        guard.throwIfAborted()
        assertArtifactLockHeld()
        if (!baseline && handle.isExited()) throw loadSmokeError('load_smoke_process_exited')
        expect(response?.status === 200 && response.body && typeof response.body === 'object',
          'load_smoke_http_failed')
        steps.push({
          ordinal: index + 1,
          name,
          method,
          endpoint,
          http_status: 200,
          elapsed_ms: Math.max(0, Math.round(nowMs() - started)),
          evidence: normalizeResponse(name, response.body),
        })
      }

      await call(0, true)
      assertNoWarmup()
      await call(1)
      assertNoWarmup()
      for (let index = 2; index < STEP_CONTRACT.length; index += 1) await call(index)
      logMarkers = assertNoWarmup()
    } catch (error) {
      try { guard?.throwIfAborted() } catch (guardError) { primaryError = guardError }
      if (!primaryError) primaryError = errorCode(error) ? error : loadSmokeError('load_smoke_http_failed')
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
          if (primaryError && !handle.isExited()) await yieldImpl()
          if (handle.isExited()) primaryError = loadSmokeError('load_smoke_process_exited')
          const terminated = deps.terminateChildImpl
            ? await deps.terminateChildImpl(handle)
            : await terminateSpawnedChild(handle, { sleepImpl })
          expect(terminated?.observed === true
            && typeof terminated.already_exited === 'boolean'
            && typeof terminated.termination_requested === 'boolean',
          'load_smoke_termination_failed')
          const timeout = boundedTimeout(5_000, { timeout: true })
          const closeObservation = await Promise.race([
            handle.closed.then(() => ({ closed: true })),
            timeout.promise,
          ])
          timeout.cancel()
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
          const observed = handle.logMarkers()
          expect(observed.warming_up_seen === false
            && observed.generation_warmup_complete_seen === false,
          'load_smoke_warmup_detected')
          logMarkers = observed
        } catch (error) {
          primaryError = errorCode(error) ? error : loadSmokeError('load_smoke_warmup_detected')
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
    expect(sameJson(postflight?.provenance, preflight.provenance)
      && sameJson(postflight?.auto_select_roots, preflight.auto_select_roots),
    'load_smoke_source_changed')
    expect(postflight?.artifact?.size_bytes === lockedPreloadArtifact.size_bytes
      && postflight.artifact.sha256 === lockedPreloadArtifact.sha256
      && postflight.artifact.verified_after_generation === true
      && postflight.artifact.path_redacted === true,
    'load_smoke_artifact_identity_mismatch')
    healthBuild = steps.find((step) => step.name === 'baseline_health')?.evidence?.build
    expect(steps.filter((step) => step.evidence?.build)
      .every((step) => step.evidence.build === healthBuild)
      && healthBuild === preflight.provenance.source_describe,
    'load_smoke_source_changed')
  } catch (error) {
    evidenceError = errorCode(error) ? error : loadSmokeError('load_smoke_http_failed')
  }

  if (artifactLock.isExited()) {
    try {
      const timeout = boundedTimeout(2_000, null)
      const observation = await Promise.race([
        Promise.all([artifactLock.exited, artifactLock.closed]),
        timeout.promise,
      ])
      timeout.cancel()
      expect(observation?.[0] && sameJson(observation[0], artifactLock.exitStatus()),
        'load_smoke_artifact_lock_release_failed')
    } catch {
      throw loadSmokeError('load_smoke_artifact_lock_release_failed')
    }
    if (!['load_smoke_resource_telemetry_unavailable', 'load_smoke_termination_failed']
      .includes(errorCode(evidenceError))) {
      evidenceError = loadSmokeError('load_smoke_artifact_lock_lost')
    }
  } else {
    try {
      const released = await artifactLock.release()
      expect(released?.observed === true
        && released?.released_token_observed === true
        && released?.exit_code === 0,
      'load_smoke_artifact_lock_release_failed')
    } catch {
      throw loadSmokeError('load_smoke_artifact_lock_release_failed')
    }
  }
  if (evidenceError) throw evidenceError

  const receipt = buildReceipt({
    preflight,
    postflightArtifact: {
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
    },
    environmentContract,
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
    try { await rmImpl(temporary, { force: true }) } catch { /* fail closed */ }
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
    expect(CLI_VALUE_OPTIONS.has(key) || CLI_BOOLEAN_OPTIONS.has(key),
      'load_smoke_options_invalid')
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
    process.stdout.write('Usage: node scripts/hf-qualification-qwen2_5-load-smoke.mjs [options]\n\n'
      + '  --binary <path>       Frozen release-fat-LTO Camelid binary\n'
      + '  --artifact <path>     Ignored exact Qwen2.5 0.5B Q8_0 GGUF\n'
      + '  --cwd <path>          Dedicated candidate-free run directory\n'
      + '  --models-dir <path>   Dedicated candidate-free models directory\n'
      + '  --root <path>         Clean tracked Camelid source root (default: .)\n'
      + '  --binary-profile <id> Provenance label (default: release-fat-lto)\n'
      + '  --out <path>          Atomically write the privacy-safe sealed receipt\n')
    return
  }
  const root = resolve(args.get('root') || '.')
  const defaultBinary = process.platform === 'win32'
    ? 'target/model-qualification/bin/camelid-qwen2_5-release-fat-lto.exe'
    : 'target/model-qualification/bin/camelid-qwen2_5-release-fat-lto'
  const receipt = await runQwen25LoadSmoke({
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
  DOES_NOT_PROVE,
  ERROR_CONTRACTS,
  EXACT_ROW,
  EXECUTION_PLAN_EXACT_MODEL_ROW,
  LEGACY_STORAGE_LABELS,
  LIMITS,
  Qwen25LoadSmokeError,
  RAW_REQUEST,
  RECEIPT_SCHEMA,
  ROW_ID,
  SAFE_CAMELID_ENV,
  SERVER_ADDR,
  STEP_CONTRACT,
  assertRawMemory,
  assertValidLoadSmokeReceipt,
  buildChildEnv,
  buildReceipt,
  buildServeArgs,
  classifyQwen25LoadSmokeError,
  describeChildEnvironment,
  httpJson,
  inspectExactArtifactIdentity,
  normalizeGeneration,
  normalizeHealth,
  parseArgs,
  receiptCommand,
  runPostflight,
  runPreflight,
  runQwen25LoadSmoke,
  validateLoadSmokeReceipt,
  writeReceiptAtomic,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    const failure = classifyQwen25LoadSmokeError(error)
    console.error(`${failure.error_code}: ${failure.reason}`)
    process.exit(1)
  })
}
