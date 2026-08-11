#!/usr/bin/env node

import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { EventEmitter } from 'node:events'
import { mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { PassThrough, Writable } from 'node:stream'
import { setTimeout as delay } from 'node:timers/promises'
import {
  CHAT_REQUEST,
  DOES_NOT_PROVE,
  EXACT_ROW,
  EXPECTED_TEMPLATE_CAPS,
  LEGACY_STORAGE_LABELS,
  LIMITS,
  RAW_REQUEST,
  RECEIPT_SCHEMA,
  SAFE_CAMELID_ENV,
  SERVER_ADDR,
  STEP_CONTRACT,
  SmolLM3LoadSmokeError,
  acquireWindowsArtifactReadLock,
  assertAutoSelectRootsEmpty,
  buildChildEnv,
  buildServeArgs,
  canonicalJson,
  classifySmolLM3LoadSmokeError,
  createResourceGuard,
  describeChildEnvironment,
  httpJson,
  inspectExactArtifactIdentity,
  inspectProvenance,
  parseArgs,
  receiptCommand,
  runPostflight,
  runSmolLM3LoadSmoke,
  sealReceipt,
  terminateSpawnedChild,
  validateLoadSmokeReceipt,
} from './hf-qualification-smollm3-load-smoke.mjs'

const root = resolve('.')
const clone = (value) => structuredClone(value)
const runtimeHead = `12345678${'9'.repeat(32)}`
const sourceDescribe = 'v0.6.1-1-g12345678'
const binaryVersion = `camelid ${sourceDescribe}`
const healthBuild = sourceDescribe
const binarySha256 = 'a'.repeat(64)
const templatePack = JSON.parse(await readFile(
  resolve(root, 'qa/prompt-packs/smollm3-chat-template-shapes-v1.json'),
  'utf8',
))
const template = templatePack.source_template.text
const durableReceiptBytes = await readFile(resolve(
  root,
  'qa/model-qualification/smollm3-3b-q8-windows-cpu-load-smoke.json',
))
const durableReceipt = JSON.parse(durableReceiptBytes)
const fileSha256 = bytes => createHash('sha256').update(bytes).digest('hex')

assert.equal(Buffer.byteLength(template, 'utf8'), 5_493)
assert.equal(EXACT_ROW.source.size_bytes, 3_275_574_624)
assert.equal(EXACT_ROW.source.sha256, '8aa8cc74656137174a1988d993b00828e65a86fd68773412b632a75aa1373248')
assert.equal(RECEIPT_SCHEMA, 'camelid.model-qualification.load-smoke/v2')
assert.equal(SERVER_ADDR, '127.0.0.1:8297')
assert.equal(
  fileSha256(durableReceiptBytes),
  '4c156199ef4395188aa64210401bb3bfa40e8ef8acdb58c4e4908cc583257b17',
)
assert.equal(
  durableReceipt.receipt_id,
  '7d5a31a30609db49847790d69fd809612d579000f9fa7f4857f0b753dd4a5aa4',
)
assert.equal(durableReceipt.schema, 'camelid.model-qualification.load-smoke/v1')
const durableReceiptErrors = validateLoadSmokeReceipt(durableReceipt)
assert.ok(durableReceiptErrors.some((error) => error.includes('runtime_contract.environment')),
  'the immutable pre-hardening receipt must be rejected for omitting inherited OS env evidence')
assert.equal(durableReceipt.provenance.runtime_head, '0634334c0f912e5ab71710dc7542af7be3b97263')
assert.equal(durableReceipt.provenance.source_describe, 'v0.6.1-49-g0634334c')
assert.deepEqual(durableReceipt.provenance.binary, {
  profile: 'release-fat-lto',
  sha256: 'e3e7e08609e132785c7ddf4fc00bdab001d288b8c23fe782c757acf22fdcef3e',
  version: 'camelid v0.6.1-49-g0634334c',
  health_build: 'v0.6.1-49-g0634334c',
  built_from_clean_tracked_head: true,
})
assert.equal(durableReceipt.provenance.tracked_files_clean, true)
assert.equal(durableReceipt.provenance.untracked_files_excluded, true)
assert.equal(durableReceipt.row.id, EXACT_ROW.id)
assert.deepEqual(durableReceipt.row.source, EXACT_ROW.source)
assert.deepEqual(durableReceipt.steps.map(({ name }) => name), STEP_CONTRACT.map(([name]) => name))
const durableSteps = Object.fromEntries(durableReceipt.steps.map(({ name, evidence }) => [name, evidence]))
const durableRaw = durableSteps.raw_first_forward
const durableChat = durableSteps.chat_followup
assert.equal(durableRaw.timings.weight_cache_hit, false)
assert.equal(durableRaw.timings.forward_total, 2847.006)
assert.equal(durableRaw.generated_token_ids.length, 1)
assert.equal(durableRaw.logits.greedy_top.token_id, durableRaw.generated_token_ids[0])
assert.equal(Number.isFinite(durableRaw.logits.greedy_top.logit), true)
assert.equal(durableChat.lane, 'experimental')
assert.equal(durableChat.timings.weight_cache_hit, true)
assert.equal(durableChat.timings.forward_total, 16855.325)
assert.equal(durableChat.generated_token_ids.length, 1)
assert.equal(durableChat.logits.greedy_top.token_id, durableChat.generated_token_ids[0])
assert.equal(Number.isFinite(durableChat.logits.greedy_top.logit), true)
const durableMemoryPhases = [...durableRaw.memory_phases, ...durableChat.memory_phases]
assert.equal(durableMemoryPhases.length, 6)
for (const phaseEvidence of durableMemoryPhases) {
  const materialization = phaseEvidence.materialization
  const reads = phaseEvidence.q8_file_reads
  assert.equal(materialization.q8_0_source_tensor_count, 254)
  assert.equal(materialization.q8_0_file_backed_tensor_count, 254)
  assert.equal(materialization.q8_0_f32_materialized_tensor_count, 0)
  assert.equal(materialization.q8_0_f32_materialized_bytes, 0)
  assert.equal(materialization.q8_0_retained_block_tensor_count, 0)
  assert.equal(materialization.q8_0_retained_block_bytes, 0)
  assert.equal(materialization.has_lazy_q8_0_file_backing, true)
  assert.equal(materialization.has_q8_0_f32_materialization, false)
  assert.equal(materialization.has_retained_q8_0_blocks, false)
  assert.ok(reads.read_calls > 0)
  assert.ok(reads.read_bytes > 0)
  for (const key of [
    'cache_hits', 'cache_hit_bytes', 'cache_misses', 'cache_miss_bytes',
    'cache_inserts', 'cache_insert_bytes', 'cache_evictions', 'cache_evicted_bytes',
    'cache_merges', 'cache_merged_bytes', 'cache_decoded_scale_hits',
    'cache_decoded_scale_hit_blocks', 'cache_entries', 'cache_bytes', 'cache_capacity_bytes',
  ]) assert.equal(reads[key], 0, `${key} must remain zero in the durable receipt`)
}
assert.deepEqual(durableSteps.loaded_health.execution_plan, {
  profile: 'safe',
  operating_system: 'windows',
  architecture: 'x86_64',
  model_family: 'smollm3',
  quant_type: 'Q8_0',
  exact_model_row: 'SmolLM3-Q8_0.gguf',
  support_level: 'unknown_or_unvalidated',
  selected_backend: 'cpu_reference',
  selected_q8_path: 'safe_dense_or_q8_cpu',
  diagnostics_status: 'operator-requested RSS timings enabled; performance claims disabled',
  cuda_resident_active: false,
  legacy_storage_labels_excluded: [...LEGACY_STORAGE_LABELS],
})
for (const name of ['baseline_gpu', 'post_raw_gpu', 'final_gpu']) {
  assert.equal(durableSteps[name].enabled, false)
  assert.equal(durableSteps[name].run_count, 0)
}
assert.ok(durableReceipt.resource_observations.monitor_samples > 0)
assert.ok(durableReceipt.resource_observations.minimum_available_physical_bytes > 0)
assert.ok(durableReceipt.resource_observations.peak_child_working_set_bytes > 0)
assert.equal(durableReceipt.resource_observations.thresholds_tripped, false)
assert.deepEqual(durableReceipt.gate_decision, {
  load_smoke: 'pass',
  support_claim: false,
  disposition: 'hold',
  target_tier: 'experimental_exact_row',
  authorized_roster_scope: ['gates.load_smoke'],
  other_gates_unchanged: true,
})
assert.deepEqual(durableReceipt.does_not_prove, DOES_NOT_PROVE)
const durableSerialized = JSON.stringify(durableReceipt)
assert.doesNotMatch(durableSerialized, /(?:[A-Za-z]:\\|file:\/\/|\\\\[^\\]|\/Users\/|\/home\/|\/tmp\/)/i)
assert.doesNotMatch(durableSerialized, /(?:bearer\s+|basic\s+|hf_[A-Za-z0-9]{8,}|(?:token|password|secret)\s*[:=])/i)
assert.deepEqual(STEP_CONTRACT.map(([name]) => name), [
  'baseline_health',
  'baseline_gpu',
  'load',
  'verify_identity',
  'loaded_health',
  'props',
  'raw_first_forward',
  'post_raw_health',
  'post_raw_gpu',
  'chat_followup',
  'final_health',
  'final_gpu',
])

const inherited = {
  Path: 'C:\\Windows\\System32',
  PATHEXT: '.COM;.EXE;.BAT;.CMD',
  SystemRoot: 'C:\\Windows',
  ComSpec: 'C:\\Windows\\System32\\cmd.exe',
  TEMP: 'C:\\Temp',
  TMP: 42,
  CAMELID_MODEL: 'C:\\private\\startup.gguf',
  camelid_gpu: 'on',
  CAMELID_RETAIN_Q8_BLOCKS: '1',
  CAMELID_API_KEY: 'private',
  HF_TOKEN: 'private-hf-credential',
  GH_TOKEN: 'github-private-credential',
  AWS_ACCESS_KEY_ID: 'AKIA_PRIVATE',
  AWS_SECRET_ACCESS_KEY: 'aws-private-credential',
  NODE_OPTIONS: '--require=C:\\private\\inject.cjs',
}
const childEnv = buildChildEnv(inherited)
assert.equal(Object.isFrozen(childEnv), true)
assert.equal(childEnv.PATH, inherited.Path)
assert.equal(childEnv.PATHEXT, inherited.PATHEXT)
assert.equal(childEnv.SYSTEMROOT, inherited.SystemRoot)
assert.equal(childEnv.COMSPEC, inherited.ComSpec)
assert.equal(childEnv.TEMP, inherited.TEMP)
assert.equal(Object.hasOwn(childEnv, 'TMP'), false, 'non-string inherited values must be omitted')
assert.deepEqual(
  Object.fromEntries(Object.entries(childEnv).filter(([key]) => key.toUpperCase().startsWith('CAMELID_'))),
  SAFE_CAMELID_ENV,
)
assert.equal(Object.hasOwn(childEnv, 'CAMELID_MODEL'), false)
assert.equal(Object.hasOwn(childEnv, 'camelid_gpu'), false)
assert.equal(Object.hasOwn(childEnv, 'CAMELID_RETAIN_Q8_BLOCKS'), false)
assert.equal(Object.hasOwn(childEnv, 'CAMELID_API_KEY'), false)
for (const secret of [
  'HF_TOKEN', 'GH_TOKEN', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'NODE_OPTIONS',
]) assert.equal(Object.hasOwn(childEnv, secret), false, `${secret} must not reach the child`)
assert.throws(() => { childEnv.PATH = 'mutated' }, TypeError)
const childEnvironmentDescriptor = describeChildEnvironment(childEnv)
const expectedInheritedAllowlisted = {
  COMSPEC: inherited.ComSpec,
  PATH: inherited.Path,
  PATHEXT: inherited.PATHEXT,
  SYSTEMROOT: inherited.SystemRoot,
  TEMP: inherited.TEMP,
}
const environmentCommitmentDomain = 'camelid.model-qualification.child-environment/v1'
const expectedEnvironmentCommitment = createHash('sha256')
  .update(`${environmentCommitmentDomain}\0${canonicalJson(expectedInheritedAllowlisted)}`)
  .digest('hex')
assert.deepEqual(childEnvironmentDescriptor, {
  model_overrides: Object.fromEntries(Object.entries(SAFE_CAMELID_ENV)
    .sort(([left], [right]) => left.localeCompare(right))),
  windows_allowlist: [
    'COMSPEC', 'PATH', 'PATHEXT', 'SYSTEMDRIVE', 'SYSTEMROOT', 'TEMP', 'TMP', 'WINDIR',
  ],
  present_os_keys: ['COMSPEC', 'PATH', 'PATHEXT', 'SYSTEMROOT', 'TEMP'],
  inherited_values_redacted: true,
  inherited_allowlisted_values_commitment: {
    domain: environmentCommitmentDomain,
    algorithm: 'sha256',
    digest: expectedEnvironmentCommitment,
  },
  effective_keys: [...new Set([
    'COMSPEC', 'PATH', 'PATHEXT', 'SYSTEMROOT', 'TEMP', ...Object.keys(SAFE_CAMELID_ENV),
  ])].sort(),
})
const serializedEnvironmentDescriptor = JSON.stringify(childEnvironmentDescriptor)
for (const value of Object.values(expectedInheritedAllowlisted)) {
  assert.equal(serializedEnvironmentDescriptor.includes(value), false,
    'allowlisted inherited values must remain committed but redacted')
}
for (const value of [
  inherited.HF_TOKEN,
  inherited.GH_TOKEN,
  inherited.AWS_SECRET_ACCESS_KEY,
  inherited.NODE_OPTIONS,
]) assert.equal(serializedEnvironmentDescriptor.includes(value), false)
assert.throws(
  () => buildChildEnv({ Path: 'first', PATH: 'second' }),
  (error) => error.code === 'load_smoke_options_invalid',
  'case-colliding allowlisted keys must fail closed',
)

let streamedReadIndex = 0
const streamedChunks = [Buffer.from('{"ok":'), Buffer.from('true}')]
assert.deepEqual(await httpJson({
  method: 'GET', endpoint: '/v1/health', timeoutMs: 1_000,
  fetchImpl: async () => ({
    status: 200,
    headers: { get: () => String(Buffer.byteLength('{"ok":true}')) },
    body: { getReader: () => ({
      read: async () => streamedReadIndex < streamedChunks.length
        ? { done: false, value: streamedChunks[streamedReadIndex++] }
        : { done: true },
      releaseLock() {},
    }) },
  }),
}), { status: 200, body: { ok: true } })

const modelsDir = resolve('target/model-qualification/phase1/synthetic-empty-models')
const serveArgs = buildServeArgs(modelsDir)
assert.deepEqual(serveArgs, [
  'serve', '--addr', SERVER_ADDR,
  '--models-dir', modelsDir,
  '--threads', '4',
  '--gpu', 'off',
  '--deterministic',
  '--kv-quant', 'f16',
  '--no-open',
  '--max-prompt-tokens', '1024',
  '--max-generation-tokens', '1',
])
assert.equal(serveArgs.includes('--model'), false)
assert.deepEqual(receiptCommand(), [
  '<camelid>',
  'serve', '--addr', SERVER_ADDR,
  '--models-dir', '<empty-models-dir>',
  '--threads', '4',
  '--gpu', 'off',
  '--deterministic',
  '--kv-quant', 'f16',
  '--no-open',
  '--max-prompt-tokens', '1024',
  '--max-generation-tokens', '1',
])

assert.deepEqual([...parseArgs(['--help']).entries()], [['help', true]])
assert.deepEqual([...parseArgs([
  '--root=.', '--binary', 'camelid.exe', '--artifact=model.gguf', '--cwd', 'run',
  '--models-dir=models', '--binary-profile', 'release-fat-lto', '--out=receipt.json',
]).entries()], [
  ['root', '.'],
  ['binary', 'camelid.exe'],
  ['artifact', 'model.gguf'],
  ['cwd', 'run'],
  ['models-dir', 'models'],
  ['binary-profile', 'release-fat-lto'],
  ['out', 'receipt.json'],
])
for (const argv of [
  [],
  ['positional'],
  ['-h'],
  ['--unknown'],
  ['--root'],
  ['--root='],
  ['--artifact=   ', '--cwd', 'run', '--models-dir', 'models'],
  ['--artifact', 'model.gguf', '--cwd', '   ', '--models-dir', 'models'],
  ['--artifact', 'model.gguf', '--cwd', 'run', '--models-dir', '\t'],
  ['--root', '--binary', 'camelid.exe'],
  ['--root=one', '--root=two'],
  ['--help=true'],
  ['--help', '--root', '.'],
  ['--artifact', 'model.gguf', '--cwd', 'run'],
]) {
  assert.throws(
    () => parseArgs(argv),
    (error) => error.code === 'load_smoke_options_invalid',
    `strict CLI rejection: ${JSON.stringify(argv)}`,
  )
}

const frozenBinary = resolve('C:\\qualification\\bin\\camelid.exe')
function provenanceExec({ describe = sourceDescribe, version = binaryVersion } = {}) {
  return async (command, args) => {
    if (command === 'git' && args.includes('rev-parse')) return { stdout: `${runtimeHead}\n` }
    if (command === 'git' && args.includes('status')) return { stdout: '' }
    if (command === 'git' && args.includes('describe')) return { stdout: `${describe}\n` }
    if (command === frozenBinary && args.length === 1 && args[0] === '--version') {
      return { stdout: `${version}\n` }
    }
    throw new Error('unexpected provenance command')
  }
}

assert.deepEqual(await inspectProvenance({
  root,
  binary: frozenBinary,
  binaryProfile: 'release-fat-lto',
}, {
  execFileImpl: provenanceExec(),
  sha256FileImpl: async () => binarySha256,
}), {
  runtime_head: runtimeHead,
  source_describe: sourceDescribe,
  tracked_files_clean: true,
  untracked_files_excluded: true,
  binary_profile: 'release-fat-lto',
  binary_sha256: binarySha256,
  binary_version: binaryVersion,
})
const bareDescribe = runtimeHead.slice(0, 12)
assert.equal((await inspectProvenance({ root, binary: frozenBinary }, {
  execFileImpl: provenanceExec({ describe: bareDescribe, version: `camelid ${bareDescribe}` }),
  sha256FileImpl: async () => binarySha256,
})).source_describe, bareDescribe)
await assert.rejects(
  inspectProvenance({ root, binary: frozenBinary }, {
    execFileImpl: provenanceExec({
      describe: `${sourceDescribe}-dirty`,
      version: `camelid ${sourceDescribe}-dirty`,
    }),
    sha256FileImpl: async () => binarySha256,
  }),
  (error) => error.code === 'load_smoke_source_dirty',
)
await assert.rejects(
  inspectProvenance({ root, binary: frozenBinary }, {
    execFileImpl: provenanceExec({ version: 'camelid stale-build-gdeadbee' }),
    sha256FileImpl: async () => binarySha256,
  }),
  (error) => error.code === 'load_smoke_binary_stale',
)
await assert.rejects(
  inspectProvenance({ root, binary: frozenBinary }, {
    execFileImpl: provenanceExec({
      describe: 'v0.6.1-1-gdeadbee',
      version: 'camelid v0.6.1-1-gdeadbee',
    }),
    sha256FileImpl: async () => binarySha256,
  }),
  (error) => error.code === 'load_smoke_source_dirty',
)

for (const signal of [
  (() => {
    const controller = new AbortController()
    controller.abort(new SmolLM3LoadSmokeError('load_smoke_resource_abort'))
    return controller.signal
  })(),
  {
    aborted: false,
    reason: new SmolLM3LoadSmokeError('load_smoke_resource_abort'),
    addEventListener(_name, listener) {
      this.aborted = true
      listener()
    },
    removeEventListener() {},
  },
]) {
  let fetchCalls = 0
  await assert.rejects(
    httpJson({
      method: 'GET',
      endpoint: '/v1/health',
      timeoutMs: 1_000,
      signal,
      fetchImpl: async () => { fetchCalls += 1; throw new Error('must not fetch') },
    }),
    (error) => error.code === 'load_smoke_resource_abort',
  )
  assert.equal(fetchCalls, 0, 'an already-observed resource abort must preempt fetch')
}

let declaredOverflowCancelled = 0
let declaredOverflowReaders = 0
await assert.rejects(
  httpJson({
    method: 'GET', endpoint: '/v1/health', timeoutMs: 1_000,
    fetchImpl: async () => ({
      status: 200,
      headers: { get: (name) => name === 'content-length' ? String(LIMITS.max_response_bytes + 1) : null },
      body: {
        async cancel() { declaredOverflowCancelled += 1 },
        getReader() { declaredOverflowReaders += 1; throw new Error('must not allocate a reader') },
      },
    }),
  }),
  (error) => error.code === 'load_smoke_http_failed',
)
assert.equal(declaredOverflowCancelled, 1)
assert.equal(declaredOverflowReaders, 0,
  'an oversized declared body must be rejected before acquiring its reader')

let overflowReads = 0
let overflowCancelled = 0
const overflowChunks = [Buffer.alloc(LIMITS.max_response_bytes), Buffer.from('x')]
await assert.rejects(
  httpJson({
    method: 'GET', endpoint: '/v1/health', timeoutMs: 1_000,
    fetchImpl: async () => ({
      status: 200,
      headers: { get: () => null },
      body: { getReader: () => ({
        read: async () => overflowReads < overflowChunks.length
          ? { done: false, value: overflowChunks[overflowReads++] }
          : { done: true },
        async cancel() { overflowCancelled += 1 },
        releaseLock() {},
      }) },
    }),
  }),
  (error) => error.code === 'load_smoke_http_failed',
)
assert.equal(overflowReads, 2)
assert.equal(overflowCancelled, 1,
  'crossing the 16 MiB streaming bound must cancel the reader')

let pendingReadStarted
const pendingRead = new Promise((resolvePromise) => { pendingReadStarted = resolvePromise })
let abortedReaderCancelled = 0
const requestController = new AbortController()
const cancelledRequest = httpJson({
  method: 'GET', endpoint: '/v1/health', timeoutMs: 1_000, signal: requestController.signal,
  fetchImpl: async () => ({
    status: 200,
    headers: { get: () => null },
    body: { getReader: () => ({
      read: async () => { pendingReadStarted(); return new Promise(() => {}) },
      async cancel() { abortedReaderCancelled += 1 },
      releaseLock() {},
    }) },
  }),
})
await pendingRead
requestController.abort(new SmolLM3LoadSmokeError('load_smoke_resource_abort'))
await assert.rejects(cancelledRequest, (error) => error.code === 'load_smoke_resource_abort')
assert.equal(abortedReaderCancelled, 1,
  'an external resource abort must cancel an in-flight body reader')

if (process.platform === 'win32') {
  const lockTemp = await mkdtemp(join(tmpdir(), 'camelid-smollm3-read-lock-'))
  const artifact = join(lockTemp, 'artifact.gguf')
  const renamed = join(lockTemp, 'renamed.gguf')
  let artifactLock
  let released = false
  let helperArgv = []
  try {
    await writeFile(artifact, 'immutable while held')
    artifactLock = await acquireWindowsArtifactReadLock(artifact, {
      spawnImpl(command, args, options) {
        helperArgv = [command, ...args]
        return spawn(command, args, options)
      },
    })
    assert.equal(helperArgv.some((argument) => String(argument).includes(artifact)), false,
      'the artifact path must travel over stdin, never helper argv')
    assert.equal(Object.hasOwn(artifactLock, 'pid'), false)
    assert.equal(Object.hasOwn(artifactLock, 'path'), false)
    assert.equal(await readFile(artifact, 'utf8'), 'immutable while held')
    await assert.rejects(writeFile(artifact, 'mutation must fail'))
    await assert.rejects(rename(artifact, renamed))
    await assert.rejects(rm(artifact))
    const release = await artifactLock.release()
    released = true
    assert.deepEqual(release, {
      observed: true,
      released_token_observed: true,
      exit_code: 0,
    })
    await writeFile(artifact, 'mutable after release')
    await rename(artifact, renamed)
    assert.equal(await readFile(renamed, 'utf8'), 'mutable after release')
  } finally {
    if (artifactLock && !released && !artifactLock.isExited()) {
      try { await artifactLock.release() } catch { /* the test assertion remains primary */ }
    }
    await rm(lockTemp, { recursive: true, force: true })
  }

  function fakeLockSpawn(behavior) {
    return () => {
      const child = new EventEmitter()
      child.stdout = new PassThrough()
      child.stderr = new PassThrough()
      let finished = false
      let nonce = null
      let writes = 0
      const closeAfterDrain = (code, signal = null) => {
        if (finished) return
        finished = true
        child.emit('exit', code, signal)
        setImmediate(() => {
          child.stdout.end()
          child.stderr.end()
          child.emit('close', code, signal)
        })
      }
      child.kill = (signal) => {
        closeAfterDrain(null, signal)
        return true
      }
      child.stdin = new Writable({
        write(chunk, _encoding, callback) {
          writes += 1
          const line = String(chunk).trim()
          if (writes === 1) {
            nonce = JSON.parse(line).nonce
            if (behavior === 'nonce_mismatch') child.stdout.write(`LOCKED:${'0'.repeat(32)}\n`)
            else if (behavior === 'released_before_locked') child.stdout.write(`RELEASED:${nonce}\n`)
            else if (behavior === 'duplicate_locked') {
              child.stdout.write(`LOCKED:${nonce}\nLOCKED:${nonce}\n`)
            } else if (behavior === 'extra_stdout') {
              child.stdout.write(`LOCKED:${nonce}\nEXTRA\n`)
            } else {
              child.stdout.write(`LOCKED:${nonce}\n`)
              if (behavior === 'unexpected_exit_zero') setImmediate(() => closeAfterDrain(0))
            }
          } else if (behavior === 'duplicate_released') {
            child.stdout.write(`RELEASED:${nonce}\nRELEASED:${nonce}\n`)
            closeAfterDrain(0)
          } else if (behavior === 'exit_before_release_drain') {
            if (!finished) {
              finished = true
              child.emit('exit', 0, null)
              setImmediate(() => {
                child.stdout.write(`RELEASED:${nonce}\n`)
                child.stdout.end()
                child.stderr.end()
                child.emit('close', 0, null)
              })
            }
          } else {
            child.stdout.write(`RELEASED:${nonce}\n`)
            closeAfterDrain(0)
          }
          callback()
        },
      })
      return child
    }
  }

  for (const behavior of [
    'nonce_mismatch', 'released_before_locked', 'duplicate_locked', 'extra_stdout',
  ]) {
    await assert.rejects(
      acquireWindowsArtifactReadLock(resolve('C:\\qualification\\artifact.gguf'), {
        spawnImpl: fakeLockSpawn(behavior),
      }),
      (error) => error.code === 'load_smoke_artifact_lock_failed',
      `malformed lock protocol must fail: ${behavior}`,
    )
  }

  const unexpectedLockExit = await acquireWindowsArtifactReadLock(
    resolve('C:\\qualification\\artifact.gguf'),
    { spawnImpl: fakeLockSpawn('unexpected_exit_zero') },
  )
  await unexpectedLockExit.exited
  assert.throws(
    () => unexpectedLockExit.assertHeld(),
    (error) => error.code === 'load_smoke_artifact_lock_lost',
  )

  const duplicateRelease = await acquireWindowsArtifactReadLock(
    resolve('C:\\qualification\\artifact.gguf'),
    { spawnImpl: fakeLockSpawn('duplicate_released') },
  )
  await assert.rejects(
    duplicateRelease.release(),
    (error) => error.code === 'load_smoke_artifact_lock_release_failed',
  )

  const exitBeforeDrain = await acquireWindowsArtifactReadLock(
    resolve('C:\\qualification\\artifact.gguf'),
    { spawnImpl: fakeLockSpawn('exit_before_release_drain') },
  )
  assert.equal((await exitBeforeDrain.release()).observed, true,
    'release must wait for nonce stdout to drain even when exit fires first')
}

for (const { name, linkStats, identityStats, expectedStatCalls } of [
  {
    name: 'symlink',
    linkStats: { isFile: () => true, isSymbolicLink: () => true },
    identityStats: { isFile: () => true, size: EXACT_ROW.source.size_bytes },
    expectedStatCalls: 0,
  },
  {
    name: 'non-file',
    linkStats: { isFile: () => false, isSymbolicLink: () => false },
    identityStats: { isFile: () => true, size: EXACT_ROW.source.size_bytes },
    expectedStatCalls: 0,
  },
  {
    name: 'wrong-size',
    linkStats: { isFile: () => true, isSymbolicLink: () => false },
    identityStats: { isFile: () => true, size: EXACT_ROW.source.size_bytes - 1 },
    expectedStatCalls: 1,
  },
]) {
  let statCalls = 0
  let hashCalls = 0
  await assert.rejects(
    inspectExactArtifactIdentity(resolve('C:\\qualification\\artifact.gguf'), {
      lstatImpl: async () => linkStats,
      statImpl: async () => { statCalls += 1; return identityStats },
      sha256FileImpl: async () => { hashCalls += 1; return EXACT_ROW.source.sha256 },
    }),
    (error) => error.code === 'load_smoke_artifact_identity_mismatch',
  )
  assert.equal(statCalls, expectedStatCalls, `${name} stat ordering`)
  assert.equal(hashCalls, 0, `${name} must be rejected before artifact bytes are read`)
}

const rootsTemp = await mkdtemp(join(tmpdir(), 'camelid-smollm3-load-roots-'))
try {
  const binary = join(rootsTemp, 'bin', 'camelid.exe')
  const cwd = join(rootsTemp, 'run')
  const emptyModels = join(rootsTemp, 'empty-models')
  await Promise.all([
    mkdir(dirname(binary), { recursive: true }),
    mkdir(join(dirname(binary), 'models'), { recursive: true }),
    mkdir(join(cwd, 'models'), { recursive: true }),
    mkdir(emptyModels, { recursive: true }),
  ])
  await writeFile(binary, 'offline fake executable')
  const roots = await assertAutoSelectRootsEmpty({ binary, cwd, modelsDir: emptyModels })
  assert.equal(roots.length, 5)
  assert.deepEqual(roots.map((entry) => entry.kind), [
    'configured_models_dir',
    'executable_models_dir',
    'executable_dir',
    'cwd_models_dir',
    'cwd',
  ])
  assert.ok(roots.every((entry) => entry.path_redacted && entry.gguf_candidates === 0))

  const hiddenCandidate = join(cwd, 'unexpected.GGUF')
  await writeFile(hiddenCandidate, 'not a model')
  await assert.rejects(
    assertAutoSelectRootsEmpty({ binary, cwd, modelsDir: emptyModels }),
    (error) => error.code === 'load_smoke_auto_select_candidate_present',
  )
  await rm(hiddenCandidate)
  const selector = join(emptyModels, '.camelid-default-model')
  await writeFile(selector, 'some.gguf')
  await assert.rejects(
    assertAutoSelectRootsEmpty({ binary, cwd, modelsDir: emptyModels }),
    (error) => error.code === 'load_smoke_auto_select_candidate_present',
  )
  await rm(selector)
} finally {
  await rm(rootsTemp, { recursive: true, force: true })
}

function q8Runtime() {
  return {
    policy: 'forced_lazy_file_backed_q8',
    lazy_q8_linear: true,
    retain_q8_blocks: false,
    file_cache_bytes: 0,
    note: 'forced lazy test policy',
  }
}

function executionPlan() {
  return {
    profile: 'safe',
    operating_system: 'windows',
    architecture: 'x86_64',
    platform_label: 'windows x86_64',
    cpu_model: 'redacted',
    cpu_features: ['avx2'],
    model_family: 'smollm3',
    quant_type: 'Q8_0',
    exact_model_row: 'SmolLM3-Q8_0.gguf',
    support_level: 'unknown_or_unvalidated',
    selected_backend: 'cpu_reference',
    selected_q8_path: 'safe_dense_or_q8_cpu',
    prefill_path: 'safe_cpu_prefill',
    prefill_runtime_policy: 'always_retained_reference_path',
    decode_path: 'safe_cpu_decode',
    thread_count: 4,
    diagnostics_status: 'RSS timings enabled',
    fallback_path: 'retained_q8_reference_path',
    cuda_resident_active: false,
    reasons: ['synthetic offline fixture'],
  }
}

function health(loaded) {
  return {
    ok: true,
    engine: 'camelid',
    version: '0.6.1',
    build: healthBuild,
    loaded_now: loaded,
    generation_ready: loaded,
    vision_ready: false,
    active_model_id: loaded ? EXACT_ROW.id : null,
    q8_runtime: q8Runtime(),
    execution_plan: loaded ? executionPlan() : null,
    backend: loaded ? 'llama' : 'none',
    model_family: loaded ? 'llama-family' : null,
    gemma4_available: false,
    engine_queue_depth: 0,
    engine_queued_tasks: 0,
    engine_active_task_id: null,
    engine_active_generated_tokens: 0,
    engine_active_elapsed_seconds: 0,
    engine_stalled_seconds: 0,
    continuous_batch_slots: 1,
    executable: 'C:\\private\\frozen\\camelid.exe',
    listen_addr: SERVER_ADDR,
  }
}

function gpu() {
  return { available: true, enabled: false, device: 'redacted', backend: 'cuda', run_count: 0 }
}

function loadResponse() {
  return {
    data: {
      id: EXACT_ROW.id,
      path: null,
      status: { value: 'loaded', args: [] },
      architecture: { input_modalities: ['text'], output_modalities: ['text'] },
      camelid: { generation_ready: true, model_path_redacted: true },
    },
    camelid: {
      compatibility: 'partial_llama_server_models_load_local_path',
      scope: 'single_local_model_load_alias',
      model_path_redacted: true,
      unsupported: ['models_reload'],
    },
  }
}

function verifyResponse() {
  return {
    model_id: EXACT_ROW.id,
    gguf_sha256: EXACT_ROW.source.sha256,
    eligible: false,
    profile_id: null,
    report: null,
  }
}

function propsResponse() {
  return {
    default_generation_settings: {
      is_processing: false,
      next_token: { has_next_token: true },
    },
    total_slots: 1,
    model_path: null,
    model_id: EXACT_ROW.id,
    chat_template: template,
    chat_template_caps: clone(EXPECTED_TEMPLATE_CAPS),
    modalities: { vision: false },
    build_info: 'camelid',
    is_sleeping: false,
    camelid: { generation_ready: true, model_path_redacted: true },
  }
}

function q8Reads(active) {
  return {
    read_calls: active ? 12 : 0,
    read_bytes: active ? 4_096 : 0,
    cache_hits: 0,
    cache_hit_bytes: 0,
    cache_misses: 0,
    cache_miss_bytes: 0,
    cache_inserts: 0,
    cache_insert_bytes: 0,
    cache_evictions: 0,
    cache_evicted_bytes: 0,
    cache_merges: 0,
    cache_merged_bytes: 0,
    cache_decoded_scale_hits: 0,
    cache_decoded_scale_hit_blocks: 0,
    cache_entries: 0,
    cache_bytes: 0,
    cache_capacity_bytes: 0,
  }
}

function materialization() {
  return {
    tensor_count: 12,
    dense_f32_tensor_count: 2,
    dense_f32_bytes: 512,
    q8_0_source_tensor_count: 10,
    q8_0_f32_materialized_tensor_count: 0,
    q8_0_f32_materialized_bytes: 0,
    q8_0_file_backed_tensor_count: 10,
    q8_0_file_backed_storage_bytes: 4_096,
    q8_0_file_backed_f32_bytes_avoided: 16_384,
    q8_0_file_backed_retained_block_bytes_if_enabled: 4_096,
    q8_0_file_handle_cached_count: 1,
    q8_0_retained_block_tensor_count: 0,
    q8_0_retained_block_bytes: 0,
    has_q8_0_f32_materialization: false,
    has_lazy_q8_0_file_backing: true,
    has_retained_q8_0_blocks: false,
  }
}

function memory(active = true) {
  return {
    forward_passes: 1,
    materialization: materialization(),
    q8_file_reads: q8Reads(active),
    q8_file_read_phases: [],
    start: {},
    end: {},
    peak_rss_kib: null,
    layers: [],
  }
}

function phase(forwardTotal) {
  return {
    forward_total: forwardTotal,
    embedding: 1,
    layers_total: Math.max(0, forwardTotal - 2),
    final_norm: 0.5,
    logits: 0.5,
    sample: 0,
  }
}

function generationResponse(chat) {
  const tokenId = chat ? 456 : 123
  const text = chat ? 'Sure' : ' Paris'
  return {
    id: 'response-id-must-not-persist',
    object: chat ? 'chat.completion' : 'text_completion',
    created: 123,
    model: EXACT_ROW.id,
    choices: chat
      ? [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'length' }]
      : [{ index: 0, text, finish_reason: 'length', logprobs: null }],
    usage: { prompt_tokens: chat ? 42 : 6, completion_tokens: 1, total_tokens: chat ? 43 : 7 },
    camelid: {
      prompt_token_ids: chat ? [1, 2, 3, 4] : [10, 11, 12],
      generated_token_ids: [tokenId],
      dense_metadata: {},
      top_logits: [
        { token_id: tokenId, logit: 9.25, probability: 0.75, rank: 1, selected: false, text },
        { token_id: tokenId + 1, logit: 8.5, probability: 0.25, rank: 2, selected: false, text: 'x' },
      ],
      output_projection: [],
      timings_ms: {
        tokenize: 1,
        weight_load: chat ? 3 : 250,
        weight_cache_hit: chat,
        prompt_cache_hit: false,
        session_create: 1,
        generate: 10,
        generation: phase(1),
        prompt_evaluation: {
          prompt_token_count: chat ? 42 : 6,
          prefill_token_count: chat ? 41 : 5,
          first_token_evaluated: true,
          prefill: phase(10),
          first_token: phase(2),
          prefill_layers: [],
          first_token_layers: [],
          prefill_memory: chat ? undefined : memory(true),
          first_token_memory: null,
        },
        layers: [],
        memory: null,
      },
    },
    ...(chat ? { lane: 'experimental' } : {}),
  }
}

const responseFactories = {
  baseline_health: () => health(false),
  baseline_gpu: gpu,
  load: loadResponse,
  verify_identity: verifyResponse,
  loaded_health: () => health(true),
  props: propsResponse,
  raw_first_forward: () => generationResponse(false),
  post_raw_health: () => health(true),
  post_raw_gpu: gpu,
  chat_followup: () => generationResponse(true),
  final_health: () => health(true),
  final_gpu: gpu,
}

function autoRoots() {
  return [
    'configured_models_dir',
    'executable_models_dir',
    'executable_dir',
    'cwd_models_dir',
    'cwd',
  ].map((kind) => ({
    kind,
    exists: true,
    path_redacted: true,
    gguf_candidates: 0,
    default_preference_present: false,
  }))
}

function preflight() {
  return {
    platform: 'windows-x86_64',
    artifact: {
      size_bytes: EXACT_ROW.source.size_bytes,
      expected_sha256: EXACT_ROW.source.sha256,
      hash_recomputed: false,
      ignored: true,
      path_redacted: true,
    },
    provenance: {
      runtime_head: runtimeHead,
      source_describe: sourceDescribe,
      tracked_files_clean: true,
      untracked_files_excluded: true,
      binary_profile: 'release-fat-lto',
      binary_sha256: binarySha256,
      binary_version: binaryVersion,
    },
    auto_select_roots: autoRoots(),
    available_physical_bytes: 6 * 1024 ** 3,
    available_disk_bytes: 8 * 1024 ** 3,
    qualification_port_unbound: true,
    llama_server_absent: true,
  }
}

function makeHarness({
  mutateResponse,
  warmup = false,
  warmupDuringTermination = false,
  stopFails = false,
  guardStartFails = false,
  terminationFails = false,
  terminationObserved = true,
  postflightArtifactSha256 = EXACT_ROW.source.sha256,
  exitDuringStep = null,
  lateExitAfterHttpStep = null,
  exitAtTermination = false,
  lockAcquireFails = false,
  lockReleaseFails = false,
  lockExitDuringStep = null,
  preloadArtifactSha256 = EXACT_ROW.source.sha256,
  peakChildWorkingSetBytes = 900 * 1024 ** 2,
} = {}) {
  const observed = []
  const killed = []
  const lifecycle = []
  let postflightCalls = 0
  let index = 0
  let exitResolve
  let exitStatus = null
  const exited = new Promise((resolvePromise) => { exitResolve = resolvePromise })
  const observeExit = (status) => {
    if (exitStatus !== null) return
    exitStatus = status
    exitResolve(status)
  }
  let closeResolve
  let closeStatus = null
  const closed = new Promise((resolvePromise) => { closeResolve = resolvePromise })
  const observeClose = (status) => {
    if (closeStatus !== null) return
    closeStatus = status
    closeResolve(status)
  }
  const observeStopped = (status) => {
    observeExit(status)
    observeClose({ code: status.code, signal: status.signal })
  }
  let lockExitResolve
  let lockExitStatus = null
  let lockReleased = false
  const lockExited = new Promise((resolvePromise) => { lockExitResolve = resolvePromise })
  const observeLockExit = (status) => {
    if (lockExitStatus !== null) return
    lockExitStatus = status
    lockExitResolve(status)
  }
  const artifactLock = {
    acquired: true,
    exited: lockExited,
    closed: lockExited,
    isExited: () => lockExitStatus !== null,
    exitStatus: () => lockExitStatus,
    assertHeld() {
      if (lockExitStatus !== null || lockReleased) {
        throw new SmolLM3LoadSmokeError('load_smoke_artifact_lock_lost')
      }
    },
    async release() {
      lifecycle.push('lock_release')
      if (lockReleaseFails) throw new Error('offline lock release failure')
      this.assertHeld()
      lockReleased = true
      observeLockExit({ error: false, code: 0, signal: null })
      return { observed: true, released_token_observed: true, exit_code: 0 }
    },
  }
  let warmupSeen = warmup
  const handle = {
    pid: 4_321,
    exited,
    closed,
    isExited: () => exitStatus !== null,
    isClosed: () => closeStatus !== null,
    exitStatus: () => exitStatus,
    kill(signal) {
      killed.push(signal)
      observeStopped({ error: false, code: null, signal })
      return true
    },
    logMarkers: () => ({
      warming_up_seen: warmupSeen,
      generation_warmup_complete_seen: warmupSeen,
      raw_output_persisted: false,
    }),
  }
  const fixturePreflight = preflight()
  let clock = 0
  let spawnedEnv = null
  const deps = {
    inheritedEnv: inherited,
    preflightImpl: async () => clone(fixturePreflight),
    acquireArtifactLockImpl: async (path) => {
      lifecycle.push('lock_acquire')
      assert.equal(path, resolve('C:\\qualification\\artifact.gguf'))
      if (lockAcquireFails) throw new Error('offline lock acquisition failure')
      return artifactLock
    },
    preloadArtifactIdentityImpl: async (path) => {
      lifecycle.push('preload_identity')
      artifactLock.assertHeld()
      assert.equal(path, resolve('C:\\qualification\\artifact.gguf'))
      return { size_bytes: EXACT_ROW.source.size_bytes, sha256: preloadArtifactSha256 }
    },
    postflightImpl: async () => {
      lifecycle.push('postflight')
      postflightCalls += 1
      artifactLock.assertHeld()
      return {
        provenance: clone(fixturePreflight.provenance),
        auto_select_roots: clone(fixturePreflight.auto_select_roots),
        artifact: {
          size_bytes: EXACT_ROW.source.size_bytes,
          sha256: postflightArtifactSha256,
          verified_after_generation: true,
          path_redacted: true,
        },
      }
    },
    startProcessImpl: async ({ args, cwd, env }) => {
      lifecycle.push('start')
      spawnedEnv = env
      assert.equal(Object.isFrozen(env), true)
      assert.equal(args.includes('--model'), false)
      assert.equal(cwd, resolve('C:\\qualification\\run'))
      assert.deepEqual(
        Object.fromEntries(Object.entries(env).filter(([key]) => key.toUpperCase().startsWith('CAMELID_'))),
        SAFE_CAMELID_ENV,
      )
      return handle
    },
    createResourceGuardImpl: async (candidate) => {
      lifecycle.push('guard_start')
      assert.equal(candidate, handle)
      if (guardStartFails) throw new Error('offline guard startup failure')
      const controller = new AbortController()
      return {
        signal: controller.signal,
        throwIfAborted() {},
        async stop() {
          lifecycle.push('guard_stop')
          if (stopFails) throw new Error('offline stop failure')
          return { observed: true }
        },
        summary: () => ({
          samples: 3,
          minimum_available_physical_bytes: 3 * 1024 ** 3,
          peak_child_working_set_bytes: peakChildWorkingSetBytes,
          thresholds_tripped: false,
        }),
      }
    },
    terminateChildImpl: async (candidate) => {
      lifecycle.push('terminate')
      assert.equal(candidate, handle)
      if (terminationFails) throw new Error('offline termination failure')
      if (exitAtTermination && !candidate.isExited()) {
        observeStopped({ error: false, code: 71, signal: null })
      }
      const alreadyExited = candidate.isExited()
      if (!alreadyExited) candidate.kill('SIGTERM')
      await Promise.all([candidate.exited, candidate.closed])
      if (warmupDuringTermination) warmupSeen = true
      return {
        observed: terminationObserved,
        already_exited: alreadyExited,
        termination_requested: !alreadyExited,
      }
    },
    httpJsonImpl: async ({ method, endpoint, body }) => {
      const [name, expectedMethod, expectedEndpoint] = STEP_CONTRACT[index++]
      observed.push({ name, method, endpoint, body: clone(body) })
      assert.equal(method, expectedMethod)
      assert.equal(endpoint, expectedEndpoint)
      if (name === 'load') {
        assert.deepEqual(body, { path: resolve('C:\\qualification\\artifact.gguf'), id: EXACT_ROW.id })
        assert.equal(Object.hasOwn(body, 'replace'), false)
      } else if (name === 'raw_first_forward') {
        assert.deepEqual(body, RAW_REQUEST)
        assert.equal(Object.hasOwn(body, 'camelid_receipt'), false)
      } else if (name === 'chat_followup') {
        assert.deepEqual(body, CHAT_REQUEST)
        assert.equal(Object.hasOwn(body, 'camelid_receipt'), false)
        assert.equal(Object.hasOwn(body, 'camelid_enable_thinking'), false)
      } else {
        assert.equal(body, undefined)
      }
      if (exitDuringStep === name) {
        observeStopped({ error: false, code: 70, signal: null })
        throw new Error('secondary HTTP symptom after child exit')
      }
      if (lockExitDuringStep === name) {
        observeLockExit({ error: false, code: 72, signal: null })
        throw new Error('secondary HTTP symptom after lock exit')
      }
      let response = { status: 200, body: responseFactories[name]() }
      if (lateExitAfterHttpStep === name) response = { status: 500, body: { error: 'offline' } }
      if (mutateResponse) response = mutateResponse(name, response) || response
      return response
    },
    nowMsImpl: () => { clock += 5; return clock },
    nowIsoImpl: () => '2026-08-10T12:34:56.000Z',
    sleepImpl: async () => {},
    yieldImpl: async () => {
      if (lateExitAfterHttpStep && !handle.isExited()) {
        observeStopped({ error: false, code: 73, signal: null })
      }
    },
  }
  return {
    deps,
    observed,
    killed,
    lifecycle,
    postflightCalls: () => postflightCalls,
    lockReleased: () => lockReleased,
    spawnedEnv: () => spawnedEnv,
  }
}

const runOptions = {
  root,
  binary: resolve('C:\\qualification\\bin\\camelid.exe'),
  artifact: resolve('C:\\qualification\\artifact.gguf'),
  cwd: resolve('C:\\qualification\\run'),
  modelsDir: resolve('C:\\qualification\\empty-models'),
  binaryProfile: 'release-fat-lto',
}

const success = makeHarness()
const receipt = await runSmolLM3LoadSmoke(runOptions, success.deps)
assert.deepEqual(validateLoadSmokeReceipt(receipt), [])
assert.deepEqual(success.observed.map(({ name }) => name), STEP_CONTRACT.map(([name]) => name))
assert.deepEqual(success.killed, ['SIGTERM'])
assert.deepEqual(success.lifecycle.slice(-4), ['guard_stop', 'terminate', 'postflight', 'lock_release'])
assert.equal(success.postflightCalls(), 1)
assert.equal(success.lockReleased(), true)
assert.equal(receipt.provenance.artifact.mutation_guard.mechanism, 'windows_file_stream_share_read')
assert.equal(receipt.provenance.artifact.mutation_guard.artifact_path_in_helper_argv, false)
assert.equal(receipt.gate_decision.support_claim, false)
assert.deepEqual(receipt.gate_decision.authorized_roster_scope, ['gates.load_smoke'])
assert.deepEqual(receipt.runtime_contract.environment,
  describeChildEnvironment(success.spawnedEnv()),
  'the receipt environment descriptor must bind the exact frozen spawn environment')
assert.equal(receipt.steps[6].name, 'raw_first_forward')
assert.equal(receipt.steps[6].evidence.timings.weight_cache_hit, false)
assert.equal(receipt.steps[6].evidence.timings.forward_total, 1,
  'raw receipt must copy the API aggregate rather than add its component phases again')
assert.equal(receipt.steps[6].evidence.memory_phases[0].peak_rss_kib, null)
assert.equal(receipt.steps[6].evidence.logits.greedy_top.rank, 1)
assert.equal(receipt.steps[6].evidence.logits.greedy_top.token_id,
  receipt.steps[6].evidence.generated_token_ids[0])
assert.equal(receipt.steps[9].name, 'chat_followup')
assert.equal(receipt.steps[9].evidence.timings.weight_cache_hit, true)
assert.equal(receipt.steps[9].evidence.timings.weight_load, 3)
assert.equal(receipt.steps[9].evidence.timings.forward_total, 1,
  'chat receipt must copy the API aggregate rather than add its component phases again')
assert.equal(receipt.provenance.source_describe, sourceDescribe)
assert.equal(receipt.provenance.binary.version, binaryVersion)
assert.equal(JSON.stringify(receipt).includes('C:\\qualification'), false)
for (const privateValue of [
  inherited.Path,
  inherited.SystemRoot,
  inherited.ComSpec,
  inherited.HF_TOKEN,
  inherited.GH_TOKEN,
  inherited.AWS_SECRET_ACCESS_KEY,
  inherited.NODE_OPTIONS,
]) assert.equal(JSON.stringify(receipt).includes(privateValue), false)
assert.equal(JSON.stringify(receipt).includes('response-id-must-not-persist'), false)
assert.equal(JSON.stringify(receipt).includes(template.slice(0, 256)), false)
assert.equal(JSON.stringify(receipt).includes('camelid_receipt'), true, 'the receipt records the false request contract')
assert.equal(receipt.runtime_contract.requests.camelid_receipt_requested, false)
assert.deepEqual(receipt.runtime_contract.excluded_legacy_storage_labels, LEGACY_STORAGE_LABELS)

function reseal(mutator) {
  const body = clone(receipt)
  delete body.receipt_id
  mutator(body)
  return sealReceipt(body)
}

assert.ok(validateLoadSmokeReceipt({ ...clone(receipt), receipt_id: 'f'.repeat(64) })
  .some((error) => error.includes('seal')))
for (const [description, mutator, fragment] of [
  ['support claim', (value) => { value.gate_decision.support_claim = true }, 'without support'],
  ['raw/chat order', (value) => { [value.steps[6], value.steps[9]] = [value.steps[9], value.steps[6]] }, 'step 7'],
  ['raw cache state', (value) => { value.steps[6].evidence.timings.weight_cache_hit = true }, 'first materializing'],
  ['raw f32 materialization', (value) => {
    value.steps[6].evidence.memory_phases[0].materialization.q8_0_f32_materialized_tensor_count = 1
    value.steps[6].evidence.memory_phases[0].materialization.has_q8_0_f32_materialization = true
  }, 'lazy-Q8'],
  ['hidden startup model', (value) => { value.runtime_contract.command.push('--model', '<redacted>') }, 'serve command'],
  ['receipt request', (value) => { value.runtime_contract.requests.raw_first_forward.camelid_receipt = true }, 'raw first-forward'],
  ['environment model override', (value) => {
    value.runtime_contract.environment.model_overrides.CAMELID_PROFILE = 'experimental'
  }, 'child environment'],
  ['environment Windows allowlist', (value) => {
    value.runtime_contract.environment.windows_allowlist.pop()
  }, 'child environment'],
  ['environment present key order', (value) => {
    value.runtime_contract.environment.present_os_keys.reverse()
  }, 'child environment'],
  ['environment commitment', (value) => {
    value.runtime_contract.environment.inherited_allowlisted_values_commitment.digest = '0'
  }, 'child environment'],
  ['environment effective keys', (value) => {
    value.runtime_contract.environment.effective_keys.pop()
  }, 'child environment'],
  ['environment unknown field', (value) => {
    value.runtime_contract.environment.unexpected = true
  }, 'runtime_contract.environment'],
  ['Q8 policy', (value) => { value.steps[4].evidence.q8_runtime.policy = 'legacy' }, 'forced-lazy'],
  ['baseline ok', (value) => { value.steps[0].evidence.ok = false }, 'baseline_health scalar'],
  ['baseline engine', (value) => { value.steps[0].evidence.engine = 'forged' }, 'baseline_health scalar'],
  ['baseline queue', (value) => { value.steps[0].evidence.queue.depth = 1 }, 'baseline_health scalar'],
  ['baseline Q8 cache', (value) => { value.steps[0].evidence.q8_runtime.file_cache_bytes = 1 }, 'baseline_health scalar'],
  ['loaded CPU backend', (value) => { value.steps[4].evidence.backend = 'runnable-runtime' }, 'unsupported SmolLM3 CPU'],
  ['loaded selected backend', (value) => { value.steps[4].evidence.execution_plan.selected_backend = 'cuda' }, 'unsupported SmolLM3 CPU'],
  ['loaded selected Q8 path', (value) => { value.steps[4].evidence.execution_plan.selected_q8_path = 'safe_q8_0_block_dot' }, 'unsupported SmolLM3 CPU'],
  ['loaded exact row', (value) => { value.steps[4].evidence.execution_plan.exact_model_row = null }, 'unsupported SmolLM3 CPU'],
  ['load ID', (value) => { value.steps[2].evidence.id = 'forged' }, 'load evidence scalars'],
  ['props ID', (value) => { value.steps[5].evidence.model_id = 'forged' }, 'props scalars'],
  ['raw support semantics', (value) => { value.steps[6].evidence.lane = 'supported' }, 'explicitly unsupported'],
  ['chat cached load', (value) => { value.steps[9].evidence.timings.weight_load = -1 }, 'internally consistent'],
  ['created timestamp', (value) => { value.created_utc = '2026-08-10' }, 'ISO timestamp'],
  ['memory phase name', (value) => { value.steps[6].evidence.memory_phases[0].phase = 'forged' }, 'canonical order'],
  ['memory phase forward count', (value) => { value.steps[6].evidence.memory_phases[0].forward_passes = -1 }, 'counters'],
  ['memory materialization count', (value) => { value.steps[6].evidence.memory_phases[0].materialization.tensor_count = -1 }, 'materialization telemetry'],
  ['memory materialization flag', (value) => { value.steps[6].evidence.memory_phases[0].materialization.has_lazy_q8_0_file_backing = 'yes' }, 'materialization telemetry'],
  ['Q8 read count', (value) => { value.steps[6].evidence.memory_phases[0].q8_file_reads.cache_hits = -1 }, 'Q8 read telemetry'],
  ['API RSS shape', (value) => { value.steps[6].evidence.memory_phases[0].peak_rss_kib = -1 }, 'counters'],
  ['greedy probability', (value) => { value.steps[6].evidence.logits.greedy_top.probability = 2 }, 'internally consistent'],
  ['greedy rank', (value) => { value.steps[6].evidence.logits.greedy_top.rank = 2 }, 'internally consistent'],
  ['greedy token', (value) => { value.steps[6].evidence.logits.greedy_top.token_id += 1 }, 'internally consistent'],
  ['binary profile', (value) => { value.provenance.binary.profile = ' ' }, 'binary provenance'],
  ['stale binary version', (value) => { value.provenance.binary.version = 'camelid stale-build-gdeadbee' }, 'binary provenance'],
  ['dirty paired provenance', (value) => {
    const dirty = `${sourceDescribe}-dirty`
    value.provenance.source_describe = dirty
    value.provenance.binary.version = `camelid ${dirty}`
    value.provenance.binary.health_build = dirty
    for (const index of [0, 4, 7, 10]) value.steps[index].evidence.build = dirty
  }, 'source describe'],
  ['unrelated paired provenance', (value) => {
    const unrelated = 'v0.6.1-1-gdeadbee'
    value.provenance.source_describe = unrelated
    value.provenance.binary.version = `camelid ${unrelated}`
    value.provenance.binary.health_build = unrelated
    for (const index of [0, 4, 7, 10]) value.steps[index].evidence.build = unrelated
  }, 'runtime HEAD'],
  ['binary health build equality', (value) => {
    value.provenance.binary.health_build = `different-build-${runtimeHead.slice(0, 8)}`
  }, 'every health observation'],
  ['artifact postflight hash', (value) => { value.provenance.artifact.sha256 = 'f'.repeat(64) }, 'artifact identity'],
  ['artifact mutation guard', (value) => {
    value.provenance.artifact.mutation_guard.write_access = 'allowed'
  }, 'artifact identity'],
  ['resource guard peak', (value) => { value.resource_observations.peak_child_working_set_bytes = 0 }, 'resource telemetry'],
  ['path leak', (value) => {
    value.provenance.private_note = ['C:', 'Users', 'private', 'model.gguf'].join('\\')
  }, 'absolute local path'],
  ['credential assignment', (value) => { value.provenance.binary.profile = 'token:SECRET' }, 'credential-like'],
  ['legacy label omission', (value) => { value.runtime_contract.excluded_legacy_storage_labels.pop() }, 'legacy'],
]) {
  assert.ok(
    validateLoadSmokeReceipt(reseal(mutator)).some((error) => error.includes(fragment)),
    description,
  )
}

function nestedObjectPaths(value) {
  const paths = []
  const stack = [{ value, path: [] }]
  while (stack.length) {
    const current = stack.pop()
    if (!current.value || typeof current.value !== 'object') continue
    if (Array.isArray(current.value)) {
      current.value.forEach((item, index) => stack.push({ value: item, path: [...current.path, index] }))
      continue
    }
    paths.push(current.path)
    for (const [key, item] of Object.entries(current.value)) {
      stack.push({ value: item, path: [...current.path, key] })
    }
  }
  return paths
}

function nestedArrayPaths(value) {
  const paths = []
  const stack = [{ value, path: [] }]
  while (stack.length) {
    const current = stack.pop()
    if (!current.value || typeof current.value !== 'object') continue
    if (Array.isArray(current.value)) paths.push(current.path)
    for (const [key, item] of Object.entries(current.value)) {
      stack.push({ value: item, path: [...current.path, key] })
    }
  }
  return paths
}

function atPath(value, path) {
  return path.reduce((node, segment) => node[segment], value)
}

const receiptObjectPaths = nestedObjectPaths(receipt)
assert.ok(receiptObjectPaths.length > 50, 'the key-closure matrix must reach deeply nested evidence')
for (const path of receiptObjectPaths) {
  const withUnknown = clone(receipt)
  delete withUnknown.receipt_id
  atPath(withUnknown, path).unrecognized_proof = true
  const unknownErrors = validateLoadSmokeReceipt(sealReceipt(withUnknown))
  assert.ok(unknownErrors.length > 0, `unknown nested key passed at ${path.join('.') || '<root>'}`)

  const sourceObject = atPath(receipt, path)
  for (const key of Object.keys(sourceObject)) {
    if (path.length === 0 && key === 'receipt_id') continue
    const missingRequired = clone(receipt)
    delete missingRequired.receipt_id
    delete atPath(missingRequired, path)[key]
    const missingErrors = validateLoadSmokeReceipt(sealReceipt(missingRequired))
    assert.ok(missingErrors.length > 0, `missing nested key passed at ${[...path, key].join('.')}`)
  }
}

const receiptArrayPaths = nestedArrayPaths(receipt)
assert.ok(receiptArrayPaths.length > 20, 'the shape-closure matrix must reach nested arrays')
for (const path of receiptArrayPaths) {
  const withNamedProperty = clone(receipt)
  delete withNamedProperty.receipt_id
  atPath(withNamedProperty, path).unrecognized_proof = true
  const namedPropertyErrors = validateLoadSmokeReceipt(sealReceipt(withNamedProperty))
  assert.ok(namedPropertyErrors.length > 0,
    `named array property passed at ${path.join('.') || '<root>'}`)

  const sourceArray = atPath(receipt, path)
  if (sourceArray.length > 0) {
    const withSparseElement = clone(receipt)
    delete withSparseElement.receipt_id
    delete atPath(withSparseElement, path)[0]
    const sparseErrors = validateLoadSmokeReceipt(sealReceipt(withSparseElement))
    assert.ok(sparseErrors.length > 0, `sparse array passed at ${path.join('.')}`)
  }
}

const specificallyReportedHole = clone(receipt)
delete specificallyReportedHole.receipt_id
specificallyReportedHole.steps[0].evidence.unrecognized_proof = true
assert.ok(validateLoadSmokeReceipt(sealReceipt(specificallyReportedHole))
  .some((error) => error.includes('baseline_health') || error.includes('keys must be exact')))

const tooDeep = clone(receipt)
let deepCursor = tooDeep.provenance
for (let index = 0; index < 160; index += 1) {
  deepCursor.unrecognized_proof = {}
  deepCursor = deepCursor.unrecognized_proof
}
assert.doesNotThrow(() => validateLoadSmokeReceipt(tooDeep))
assert.ok(validateLoadSmokeReceipt(tooDeep)
  .some((error) => error.includes('bounded privacy depth') || error.includes('safely inspect')))

const cyclic = clone(receipt)
cyclic.provenance.unrecognized_proof = cyclic
assert.doesNotThrow(() => validateLoadSmokeReceipt(cyclic))
assert.ok(validateLoadSmokeReceipt(cyclic).length > 0)

const accessor = clone(receipt)
Object.defineProperty(accessor.provenance, 'unrecognized_proof', {
  enumerable: true,
  get() { throw new Error('must not escape validator') },
})
assert.doesNotThrow(() => validateLoadSmokeReceipt(accessor))
assert.ok(validateLoadSmokeReceipt(accessor).length > 0)

const rawFailure = makeHarness({
  mutateResponse(name, response) {
    if (name === 'raw_first_forward') return { status: 500, body: { error: 'private details not persisted' } }
    return response
  },
})
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, rawFailure.deps),
  (error) => error.code === 'load_smoke_http_failed',
)
assert.equal(rawFailure.observed.some(({ name }) => name === 'chat_followup'), false)
assert.deepEqual(rawFailure.killed, ['SIGTERM'])

for (const [description, mutate] of [
  ['diagnostic selection flag', (body) => { body.camelid.top_logits[0].selected = true }],
  ['duplicate rank-one logit', (body) => { body.camelid.top_logits[1].rank = 1 }],
  ['greedy token mismatch', (body) => { body.camelid.top_logits[0].token_id += 10 }],
]) {
  const invalidLogits = makeHarness({
    mutateResponse(name, response) {
      if (name === 'raw_first_forward') mutate(response.body)
      return response
    },
  })
  await assert.rejects(
    runSmolLM3LoadSmoke(runOptions, invalidLogits.deps),
    (error) => error.code === 'load_smoke_raw_invalid',
    description,
  )
  assert.equal(invalidLogits.observed.some(({ name }) => name === 'chat_followup'), false)
}

const selectedChatDiagnostic = makeHarness({
  mutateResponse(name, response) {
    if (name === 'chat_followup') response.body.camelid.top_logits[0].selected = true
    return response
  },
})
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, selectedChatDiagnostic.deps),
  (error) => error.code === 'load_smoke_chat_invalid',
)
assert.equal(selectedChatDiagnostic.postflightCalls(), 0)

const childExitDuringRequest = makeHarness({ exitDuringStep: 'loaded_health' })
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, childExitDuringRequest.deps),
  (error) => error.code === 'load_smoke_process_exited',
)
assert.deepEqual(
  childExitDuringRequest.observed.map(({ name }) => name),
  ['baseline_health', 'baseline_gpu', 'load', 'verify_identity', 'loaded_health'],
)
assert.deepEqual(childExitDuringRequest.killed, [], 'an already-exited exact child is only observed')
assert.deepEqual(childExitDuringRequest.lifecycle.slice(-3), ['guard_stop', 'terminate', 'lock_release'])
assert.equal(childExitDuringRequest.postflightCalls(), 0)

const lateChildExitAfterHttp = makeHarness({ lateExitAfterHttpStep: 'loaded_health' })
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, lateChildExitAfterHttp.deps),
  (error) => error.code === 'load_smoke_process_exited',
)
assert.deepEqual(lateChildExitAfterHttp.killed, [],
  'a natural exit observed on the cleanup turn must not be relabeled as requested teardown')
assert.equal(lateChildExitAfterHttp.lockReleased(), true)
assert.equal(lateChildExitAfterHttp.postflightCalls(), 0)

const childExitAtTerminationBoundary = makeHarness({ exitAtTermination: true })
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, childExitAtTerminationBoundary.deps),
  (error) => error.code === 'load_smoke_process_exited',
)
assert.deepEqual(childExitAtTerminationBoundary.killed, [])
assert.equal(childExitAtTerminationBoundary.lockReleased(), true)
assert.equal(childExitAtTerminationBoundary.postflightCalls(), 0)

const lockExitDuringRequest = makeHarness({ lockExitDuringStep: 'loaded_health' })
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, lockExitDuringRequest.deps),
  (error) => error.code === 'load_smoke_artifact_lock_lost',
)
assert.deepEqual(lockExitDuringRequest.killed, ['SIGTERM'])
assert.equal(lockExitDuringRequest.lockReleased(), false)
assert.equal(lockExitDuringRequest.postflightCalls(), 0)

const lockAcquisitionFailure = makeHarness({ lockAcquireFails: true })
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, lockAcquisitionFailure.deps),
  (error) => error.code === 'load_smoke_artifact_lock_failed',
)
assert.deepEqual(lockAcquisitionFailure.lifecycle, ['lock_acquire'])

const preloadArtifactMutation = makeHarness({ preloadArtifactSha256: 'f'.repeat(64) })
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, preloadArtifactMutation.deps),
  (error) => error.code === 'load_smoke_artifact_identity_mismatch',
)
assert.deepEqual(preloadArtifactMutation.lifecycle, [
  'lock_acquire', 'preload_identity', 'lock_release',
])
assert.equal(preloadArtifactMutation.lockReleased(), true)

const warmupFailure = makeHarness({ warmup: true })
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, warmupFailure.deps),
  (error) => error.code === 'load_smoke_warmup_detected',
)
assert.deepEqual(warmupFailure.killed, ['SIGTERM'])
assert.deepEqual(warmupFailure.observed.map(({ name }) => name), ['baseline_health'])
assert.equal(warmupFailure.postflightCalls(), 0)

const lateWarmupDuringTermination = makeHarness({ warmupDuringTermination: true })
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, lateWarmupDuringTermination.deps),
  (error) => error.code === 'load_smoke_warmup_detected',
)
assert.deepEqual(
  lateWarmupDuringTermination.observed.map(({ name }) => name),
  STEP_CONTRACT.map(([name]) => name),
  'the post-close marker regression must reach normal teardown',
)
assert.equal(lateWarmupDuringTermination.postflightCalls(), 0)
assert.equal(lateWarmupDuringTermination.lockReleased(), true)

const zeroWorkingSetPeak = makeHarness({ peakChildWorkingSetBytes: 0 })
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, zeroWorkingSetPeak.deps),
  (error) => error.code === 'load_smoke_resource_telemetry_unavailable',
)
assert.equal(zeroWorkingSetPeak.postflightCalls(), 0)

const requestAndTerminationFailure = makeHarness({
  terminationFails: true,
  mutateResponse(name, response) {
    if (name === 'raw_first_forward') return { status: 500, body: { error: 'offline' } }
    return response
  },
})
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, requestAndTerminationFailure.deps),
  (error) => error.code === 'load_smoke_termination_failed',
)
assert.equal(requestAndTerminationFailure.postflightCalls(), 0)
assert.ok(requestAndTerminationFailure.lifecycle.includes('guard_stop'))
assert.ok(requestAndTerminationFailure.lifecycle.includes('terminate'))

const requestAndLockReleaseFailure = makeHarness({
  lockReleaseFails: true,
  mutateResponse(name, response) {
    if (name === 'raw_first_forward') return { status: 500, body: { error: 'offline' } }
    return response
  },
})
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, requestAndLockReleaseFailure.deps),
  (error) => error.code === 'load_smoke_artifact_lock_release_failed',
)
assert.equal(requestAndLockReleaseFailure.postflightCalls(), 0)
assert.ok(requestAndLockReleaseFailure.lifecycle.includes('lock_release'))

const requestAndGuardCleanupFailure = makeHarness({
  stopFails: true,
  mutateResponse(name, response) {
    if (name === 'raw_first_forward') return { status: 500, body: { error: 'offline' } }
    return response
  },
})
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, requestAndGuardCleanupFailure.deps),
  (error) => error.code === 'load_smoke_resource_telemetry_unavailable',
)
assert.deepEqual(requestAndGuardCleanupFailure.killed, ['SIGTERM'])
assert.equal(requestAndGuardCleanupFailure.postflightCalls(), 0)

const unobservedTermination = makeHarness({ terminationObserved: false })
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, unobservedTermination.deps),
  (error) => error.code === 'load_smoke_termination_failed',
)
assert.equal(unobservedTermination.postflightCalls(), 0)

const guardStartupFailure = makeHarness({ guardStartFails: true })
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, guardStartupFailure.deps),
  (error) => error.code === 'load_smoke_resource_telemetry_unavailable',
)
assert.deepEqual(guardStartupFailure.killed, ['SIGTERM'])
assert.deepEqual(guardStartupFailure.lifecycle, [
  'lock_acquire', 'preload_identity', 'start', 'guard_start', 'terminate', 'lock_release',
])
assert.equal(guardStartupFailure.postflightCalls(), 0)

const sameSizeArtifactMutation = makeHarness({ postflightArtifactSha256: 'f'.repeat(64) })
await assert.rejects(
  runSmolLM3LoadSmoke(runOptions, sameSizeArtifactMutation.deps),
  (error) => error.code === 'load_smoke_artifact_identity_mismatch',
)
assert.deepEqual(
  sameSizeArtifactMutation.observed.map(({ name }) => name),
  STEP_CONTRACT.map(([name]) => name),
  'artifact identity must be rechecked only after raw and chat completed',
)
assert.deepEqual(sameSizeArtifactMutation.lifecycle.slice(-4), [
  'guard_stop', 'terminate', 'postflight', 'lock_release',
])
assert.equal(sameSizeArtifactMutation.postflightCalls(), 1)

let postflightHashCalls = 0
await assert.rejects(
  runPostflight(runOptions, preflight(), {
    inspectProvenanceImpl: async () => clone(preflight().provenance),
    readdirImpl: async () => [],
    lstatImpl: async () => ({ isFile: () => true, isSymbolicLink: () => false }),
    statImpl: async () => ({ isFile: () => true, size: EXACT_ROW.source.size_bytes }),
    sha256FileImpl: async (path) => {
      postflightHashCalls += 1
      assert.equal(path, runOptions.artifact)
      return 'f'.repeat(64)
    },
  }),
  (error) => error.code === 'load_smoke_artifact_identity_mismatch',
)
assert.equal(postflightHashCalls, 1, 'same-size post-generation artifact drift must be hashed once')

const driftedProvenance = clone(preflight().provenance)
driftedProvenance.source_describe = 'v0.6.1-1-g1234567'
driftedProvenance.binary_version = 'camelid v0.6.1-1-g1234567'
await assert.rejects(
  runPostflight(runOptions, preflight(), {
    inspectProvenanceImpl: async () => clone(driftedProvenance),
    readdirImpl: async () => [],
    lstatImpl: async () => ({ isFile: () => true, isSymbolicLink: () => false }),
    statImpl: async () => ({ isFile: () => true, size: EXACT_ROW.source.size_bytes }),
    sha256FileImpl: async () => EXACT_ROW.source.sha256,
  }),
  (error) => error.code === 'load_smoke_source_changed',
)

let resourceSamples = 0
const resourceHandle = { pid: 99 }
const resourceGuard = createResourceGuard(resourceHandle, {
  sampleImpl: async () => {
    resourceSamples += 1
    return {
      available_physical_bytes: LIMITS.low_memory_abort_bytes - 1,
      child_working_set_bytes: 100,
    }
  },
  sleepImpl: async () => {},
})
for (let index = 0; index < 20 && resourceSamples < LIMITS.consecutive_abort_samples; index += 1) {
  await delay(0)
}
assert.ok(resourceSamples >= LIMITS.consecutive_abort_samples)
assert.throws(
  () => resourceGuard.throwIfAborted(),
  (error) => error.code === 'load_smoke_resource_abort',
)
await resourceGuard.stop()
assert.equal(resourceGuard.summary().thresholds_tripped, true)

if (process.platform === 'win32') {
  const liveResourceGuard = createResourceGuard({ pid: process.pid }, {
    limits: {
      ...LIMITS,
      low_memory_abort_bytes: 0,
      child_working_set_abort_bytes: Number.MAX_SAFE_INTEGER,
    },
  })
  for (let index = 0; index < 100 && liveResourceGuard.summary().samples === 0; index += 1) {
    await delay(10)
  }
  await liveResourceGuard.stop()
  const liveResources = liveResourceGuard.summary()
  assert.ok(liveResources.samples >= 1, 'the real Windows resource sampler must return a sample')
  assert.ok(liveResources.minimum_available_physical_bytes > 0)
  assert.ok(liveResources.peak_child_working_set_bytes > 0)
  assert.equal(liveResources.thresholds_tripped, false)
}

const signals = []
let terminateResolve
let closeResolve
let terminated = false
let closed = false
const terminationHandle = {
  exited: new Promise((resolvePromise) => { terminateResolve = resolvePromise }),
  closed: new Promise((resolvePromise) => { closeResolve = resolvePromise }),
  isExited: () => terminated,
  isClosed: () => closed,
  kill(signal) {
    signals.push(signal)
    terminated = true
    closed = true
    terminateResolve({ code: null, signal })
    closeResolve({ code: null, signal })
    return true
  },
}
await terminateSpawnedChild(terminationHandle, { sleepImpl: () => new Promise(() => {}) })
assert.deepEqual(signals, ['SIGTERM'])

const classified = classifySmolLM3LoadSmokeError(Object.assign(
  new Error('C:\\private\\secret.gguf bearer private'),
  { code: 'load_smoke_raw_invalid' },
))
assert.deepEqual(classified, {
  status: 'blocked',
  error_code: 'load_smoke_http_failed',
  reason: 'an isolated loopback request failed or timed out',
})
assert.equal(canonicalJson({ z: 1, a: { y: 2, x: 3 } }), '{"a":{"x":3,"y":2},"z":1}')

console.log('SmolLM3 Windows CPU load-smoke harness foundation tests passed')
