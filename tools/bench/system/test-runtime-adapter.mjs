#!/usr/bin/env node
import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { isolatedRuntimeEnv, runRuntimeCampaign } from './adapters/runtime-camelid.mjs'
import { sha256Bytes } from './lib/digest.mjs'

const root = resolve(fileURLToPath(new URL('.', import.meta.url)))
const fixturePlan = JSON.parse(await readFile(resolve(root, 'fixtures/schemas/valid-plan.json'), 'utf8'))
const temp = await mkdtemp(join(tmpdir(), 'camelid-runtime-adapter-'))

try {
  const plan = await materializePlan(fixturePlan, temp)
  const prepared = preparedArms(plan)
  const prompts = []
  const valid = await runRuntimeCampaign(plan, prepared, {
    outputDir: join(temp, 'valid'),
    verifyFile: async () => {},
    execute: fakeExecute({ prompts }),
  })
  assert.equal(valid.samples.length, 4)
  assert.deepEqual(valid.samples.map((sample) => sample.arm_id), ['base', 'head', 'head', 'base'])
  assert.ok(valid.samples.every((sample) => sample.validity === 'valid'))
  assert.ok(valid.samples.every((sample) => sample.backend.observed === 'cpu_deterministic'))
  assert.equal(prompts[0], prompts[1], 'paired arms must receive the same block marker')
  assert.notEqual(prompts[1], prompts[2], 'later process blocks must receive a fresh front marker')
  assert.match(prompts[0], /^CAMELID-BENCHMARK-MARKER:/)

  const divergent = await runRuntimeCampaign(plan, prepared, {
    outputDir: join(temp, 'divergent'),
    verifyFile: async () => {},
    execute: fakeExecute({ divergentHead: true }),
  })
  assert.ok(divergent.samples.every((sample) => sample.validity === 'invalid_correctness'))
  assert.ok(divergent.samples.every((sample) => sample.correctness.parity_passed === false))

  const wrongBackend = await runRuntimeCampaign(plan, prepared, {
    outputDir: join(temp, 'backend'),
    verifyFile: async () => {},
    execute: fakeExecute({ gpuHead: true }),
  })
  const badBackendSamples = wrongBackend.samples.filter((sample) => sample.arm_id === 'head')
  assert.ok(badBackendSamples.every((sample) => sample.validity === 'invalid_backend'))
  assert.ok(badBackendSamples.every((sample) => sample.backend.observed === 'gpu_resident'))

  const timedOut = await runRuntimeCampaign(plan, prepared, {
    outputDir: join(temp, 'timeout'),
    verifyFile: async () => {},
    execute: fakeExecute({ timeoutHead: true }),
  })
  const timeoutSamples = timedOut.samples.filter((sample) => sample.arm_id === 'head')
  const timeoutPeers = timedOut.samples.filter((sample) => sample.arm_id === 'base')
  assert.ok(timeoutSamples.every((sample) => sample.validity === 'invalid_timeout'))
  assert.ok(timeoutSamples.every((sample) => sample.metrics === null))
  assert.ok(timeoutSamples.every((sample) => sample.correctness.output_token_ids_sha256 === null))
  assert.ok(timeoutPeers.every((sample) => sample.validity === 'invalid_environment'))
  assert.ok(timeoutPeers.every((sample) => sample.correctness.parity_passed === null))
  assert.ok(timeoutPeers.every((sample) => /paired arm/.test(sample.correctness.parity_unavailable_reason)))

  const isolated = isolatedRuntimeEnv('2'.repeat(40), {
    PATH: '<path>',
    CAMELID_GPU: 'on',
    CAMELID_DETERMINISTIC: '0',
    CUDA_VISIBLE_DEVICES: '0',
    RAYON_NUM_THREADS: '99',
    RUST_LOG: 'trace',
    OMP_NUM_THREADS: '99',
  })
  assert.deepEqual(isolated, { PATH: '<path>', CAMELID_COMMIT: '2'.repeat(40) })

  const stalePrepared = structuredClone(prepared)
  stalePrepared[0].source_sha = '9999999999999999999999999999999999999999'
  await assert.rejects(
    () => runRuntimeCampaign(plan, stalePrepared, {
      outputDir: join(temp, 'stale'),
      verifyFile: async () => {},
      execute: assert.fail,
    }),
    /source SHA does not match/,
  )
} finally {
  await rm(temp, { recursive: true, force: true })
}

console.log('benchmark Phase 1 runtime adapter and validity gates: PASS')

async function materializePlan(source, temp) {
  const plan = structuredClone(source)
  plan.repository_root = temp
  plan.source_arms[0].source_dir = join(temp, 'base')
  plan.source_arms[1].source_dir = join(temp, 'head')
  plan.source_arms[0].build.binary_path = join(temp, 'base', 'camelid.exe')
  plan.source_arms[1].build.binary_path = join(temp, 'head', 'camelid.exe')
  plan.models[0].artifact_path = join(temp, 'model.gguf')
  const promptPath = join(temp, 'prompt.txt')
  const prompt = 'Explain why exact output checks belong beside performance measurements.\n'
  await writeFile(promptPath, prompt, 'utf8')
  plan.workloads[0].prompt_file = promptPath
  plan.workloads[0].prompt_sha256 = sha256Bytes(Buffer.from(prompt, 'utf8'))
  plan.workloads[0].max_tokens = 4
  return plan
}

function preparedArms(plan) {
  return plan.source_arms.map((arm, index) => ({
    arm_id: arm.id,
    source_sha: arm.git_sha,
    binary_path: arm.build.binary_path,
    binary_sha256: String(index + 1).repeat(64),
    binary_size_bytes: 42,
    reported_version: 'camelid 0.6.1',
  }))
}

function fakeExecute(options = {}) {
  return async (run) => {
    const armId = run.env.CAMELID_COMMIT.startsWith('1') ? 'base' : 'head'
    const promptPath = argumentAfter(run.args, '--prompt-file')
    const prompt = await readFile(promptPath, 'utf8')
    options.prompts?.push(prompt.split('\n', 1)[0])
    if (options.timeoutHead && armId === 'head') return processResult({ timedOut: true })
    const tokenIds = options.divergentHead && armId === 'head' ? [1, 700, 999, 2] : [1, 700, 701, 2]
    const model = run.args[1]
    const record = {
      runtime: 'camelid',
      commit: run.env.CAMELID_COMMIT,
      model,
      quantization: 'Q8_0',
      iteration: 0,
      prompt_tokens: 21,
      generated_tokens: tokenIds.length,
      load_ms: 100,
      prefill_ms: 50,
      ttft_ms: 55,
      decode_ms: 120,
      tokens_per_second: 25,
      peak_memory_bytes: 1000000,
      output_text: 'fixture',
      output_token_ids: tokenIds,
    }
    if (options.gpuHead && armId === 'head') {
      record.offload = {
        total_layers: 28,
        layers_resident: 28,
        layers_offloaded: 0,
        per_layer_bytes: 0,
        free_vram_bytes: 6000000000,
        pcie_gbps: null,
        source: 'none',
      }
    }
    return processResult({ stdout: `${JSON.stringify(record)}\n` })
  }
}

function processResult(options = {}) {
  const timedOut = options.timedOut ?? false
  const stdout = options.stdout ?? ''
  return {
    state: timedOut ? 'timed_out' : 'exited',
    exitCode: timedOut ? 1 : 0,
    signal: null,
    timedOut,
    durationMs: 10,
    cleanupPassed: true,
    cleanupDetail: null,
    error: null,
    stdout: output(stdout),
    stderr: output(''),
  }
}

function output(text) {
  return {
    preview: text,
    totalBytes: Buffer.byteLength(text),
    capturedBytes: Buffer.byteLength(text),
    truncated: false,
  }
}

function argumentAfter(args, name) {
  const index = args.indexOf(name)
  assert.notEqual(index, -1, `missing argument ${name}`)
  return args[index + 1]
}
