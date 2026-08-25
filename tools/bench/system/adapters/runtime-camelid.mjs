import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'

import { parseBenchGenerateJsonl, BenchGenerateParseError } from '../lib/bench-generate.mjs'
import { validatePlan, validateRuntimeSample } from '../lib/contracts.mjs'
import { sha256Bytes, sha256File } from '../lib/digest.mjs'
import { runProcess } from '../process/runner.mjs'

const STDOUT_CAPTURE_BYTES = 4 * 1024 * 1024

export async function runRuntimeCampaign(plan, preparedArms, options = {}) {
  validatePlan(plan)
  const execute = options.execute ?? runProcess
  const verifyFile = options.verifyFile ?? defaultVerifyFile
  const outputDir = resolve(options.outputDir ?? join(plan.repository_root, 'target', 'benchmark-runs', plan.campaign_id))
  const preparedById = preparedMap(plan, preparedArms)
  const modelById = new Map(plan.models.map((model) => [model.id, model]))
  await mkdir(outputDir, { recursive: true })

  for (const arm of plan.source_arms) {
    const prepared = preparedById.get(arm.id)
    await verifyFile(prepared.binary_path, prepared.binary_sha256, 'binary', arm.id)
  }
  for (const model of plan.models) {
    await verifyFile(model.artifact_path, model.artifact_sha256, 'model', model.id)
  }

  const samples = []
  const executions = []
  for (const workload of plan.workloads) {
    const model = modelById.get(workload.model_id)
    const template = await readPromptTemplate(workload)
    const armCount = plan.source_arms.length
    const promptByBlock = new Map()
    for (let orderIndex = 0; orderIndex < workload.order.length; orderIndex += 1) {
      const armId = workload.order[orderIndex]
      const processBlock = Math.floor(orderIndex / armCount)
      let prompt = promptByBlock.get(processBlock)
      if (!prompt) {
        prompt = await materializePrompt(outputDir, plan.campaign_id, workload.id, processBlock, template)
        promptByBlock.set(processBlock, prompt)
      }
      const arm = plan.source_arms.find((candidate) => candidate.id === armId)
      const prepared = preparedById.get(armId)
      const stem = `${String(orderIndex).padStart(3, '0')}-${armId}`
      const rawDir = join(outputDir, 'raw', workload.id)
      const stdoutFile = join(rawDir, `${stem}.stdout.log`)
      const stderrFile = join(rawDir, `${stem}.stderr.log`)
      const args = benchArgs(model.artifact_path, prompt.path, workload)
      const execution = await execute({
        file: prepared.binary_path,
        args,
        cwd: arm.source_dir,
        env: isolatedRuntimeEnv(arm.git_sha, process.env),
        timeoutMs: workload.timeout_ms,
        maxCaptureBytes: STDOUT_CAPTURE_BYTES,
        stdoutFile,
        stderrFile,
      })
      executions.push({
        workload_id: workload.id,
        arm_id: armId,
        process_block: processBlock,
        order_index: orderIndex,
        args,
        stdout_file: stdoutFile,
        stderr_file: stderrFile,
        process: execution,
      })
      samples.push(normalizeExecution({
        plan,
        workload,
        arm,
        prepared,
        model,
        prompt,
        processBlock,
        execution,
      }))
    }
  }

  enforceParity(samples, plan)
  samples.forEach(validateRuntimeSample)
  return { outputDir, samples, executions }
}

export function benchArgs(modelPath, promptPath, workload) {
  const args = [
    'bench-generate',
    modelPath,
    '--prompt-file',
    promptPath,
    '--max-tokens',
    String(workload.max_tokens),
    '--temperature',
    '0',
    '--iterations',
    '1',
    '--json',
  ]
  if (workload.warmup) args.push('--warmup')
  if (workload.deterministic) args.push('--deterministic')
  if (workload.threads !== null) args.push('--threads', String(workload.threads))
  return args
}

export function isolatedRuntimeEnv(sourceSha, sourceEnv) {
  const blocked = /^(?:CAMELID_|CUDA_VISIBLE_DEVICES$|CUDA_DEVICE_ORDER$|RAYON_NUM_THREADS$|RUST_LOG$|RUST_BACKTRACE$|OMP_NUM_THREADS$|MKL_NUM_THREADS$|OPENBLAS_NUM_THREADS$|NUMEXPR_NUM_THREADS$)/i
  const env = Object.fromEntries(Object.entries(sourceEnv).filter(([key]) => !blocked.test(key)))
  env.CAMELID_COMMIT = sourceSha
  return env
}

function normalizeExecution(context) {
  const { plan, workload, arm, prepared, model, prompt, processBlock, execution } = context
  const base = {
    schema: 'camelid.benchmark.runtime-sample/v1',
    campaign_id: plan.campaign_id,
    workload_id: workload.id,
    arm_id: arm.id,
    process_block: processBlock,
    request_index: 0,
    identity: {
      source_sha: arm.git_sha,
      binary_sha256: prepared.binary_sha256,
      model_sha256: model.artifact_sha256,
      prompt_sha256: prompt.sha256,
    },
  }

  const process = processRecord(execution)
  const processFailure = classifyProcessFailure(execution)
  if (processFailure) {
    return invalidSample(base, workload, process, processFailure.validity, processFailure.reason)
  }
  if (execution.stdout.truncated) {
    return invalidSample(base, workload, process, 'invalid_parse', 'bench-generate stdout exceeded the capture bound')
  }

  let record
  try {
    const records = parseBenchGenerateJsonl(execution.stdout.preview, {
      expectedCommit: arm.git_sha,
      expectedModel: model.artifact_path,
    })
    if (records.length !== 1) {
      return invalidSample(base, workload, process, 'invalid_parse', `expected one bench-generate record, found ${records.length}`)
    }
    record = records[0]
  } catch (error) {
    const reason = error instanceof BenchGenerateParseError ? `${error.code}: ${error.message}` : error.message
    return invalidSample(base, workload, process, 'invalid_parse', reason)
  }

  const backend = assertBackend(record, workload)
  const validity = backend.assertion_passed ? 'valid' : 'invalid_backend'
  const invalidReason = backend.assertion_passed ? null : backend.reason
  return {
    ...base,
    validity,
    invalid_reason: invalidReason,
    backend: {
      requested: workload.backend.requested,
      observed: backend.observed,
      assertion_passed: backend.assertion_passed,
    },
    metrics: {
      load_ms: record.load_ms,
      prefill_ms: record.prefill_ms,
      ttft_ms: record.ttft_ms,
      decode_ms: record.decode_ms,
      tokens_per_second: record.tokens_per_second,
      prompt_tokens: record.prompt_tokens,
      generated_tokens: record.generated_tokens,
      peak_rss_bytes: record.peak_memory_bytes,
      peak_vram_bytes: null,
      peak_vram_unavailable_reason: 'not reported by bench-generate',
    },
    metrics_unavailable_reason: null,
    correctness: {
      output_token_ids_sha256: tokenIdsDigest(record.output_token_ids),
      parity_required: true,
      parity_passed: true,
      unavailable_reason: null,
      parity_unavailable_reason: null,
    },
    process,
  }
}

function invalidSample(base, workload, process, validity, reason) {
  return {
    ...base,
    validity,
    invalid_reason: reason,
    backend: {
      requested: workload.backend.requested,
      observed: null,
      assertion_passed: false,
    },
    metrics: null,
    metrics_unavailable_reason: 'process did not produce one valid bench-generate record',
    correctness: {
      output_token_ids_sha256: null,
      parity_required: true,
      parity_passed: null,
      unavailable_reason: 'process did not produce validated output token IDs',
      parity_unavailable_reason: 'process did not produce validated output token IDs',
    },
    process,
  }
}

function assertBackend(record, workload) {
  if (workload.backend.assertion !== 'deterministic_no_offload') {
    return { observed: null, assertion_passed: false, reason: `unsupported backend assertion ${workload.backend.assertion}` }
  }
  if (!workload.deterministic) {
    return { observed: null, assertion_passed: false, reason: 'deterministic backend assertion requires --deterministic' }
  }
  if (Object.hasOwn(record, 'offload')) {
    const source = record.offload.source
    const observed = record.offload.layers_offloaded > 0 ? `gpu_offload_${source}` : 'gpu_resident'
    return { observed, assertion_passed: false, reason: `expected no GPU offload status, observed ${observed}` }
  }
  return {
    observed: 'cpu_deterministic',
    assertion_passed: workload.backend.requested === 'cpu_deterministic',
    reason: null,
  }
}

function enforceParity(samples, plan) {
  for (const workload of plan.workloads) {
    const blocks = new Map()
    for (const sample of samples.filter((candidate) => candidate.workload_id === workload.id)) {
      const block = blocks.get(sample.process_block) ?? []
      block.push(sample)
      blocks.set(sample.process_block, block)
    }
    for (const [blockId, block] of blocks) {
      const digests = block.map((sample) => sample.correctness.output_token_ids_sha256)
      const complete = block.length === plan.source_arms.length && digests.every((digest) => digest !== null)
      const matching = complete && new Set(digests).size === 1
      if (matching) continue
      for (const sample of block) {
        if (sample.correctness.output_token_ids_sha256 === null) continue
        if (complete) {
          sample.correctness.parity_passed = false
          sample.correctness.parity_unavailable_reason = null
          if (sample.validity === 'valid') {
            sample.validity = 'invalid_correctness'
            sample.invalid_reason = `output token IDs diverged in process block ${blockId}`
          }
        } else {
          sample.correctness.parity_passed = null
          sample.correctness.parity_unavailable_reason = `a paired arm did not produce validated output token IDs in process block ${blockId}`
          if (sample.validity === 'valid') {
            sample.validity = 'invalid_environment'
            sample.invalid_reason = `output token parity was unavailable in process block ${blockId}`
          }
        }
      }
    }
  }
}

async function readPromptTemplate(workload) {
  const bytes = await readFile(workload.prompt_file)
  const digest = sha256Bytes(bytes)
  if (digest !== workload.prompt_sha256) {
    throw new Error(`prompt ${workload.id} resolved to ${digest}, expected ${workload.prompt_sha256}`)
  }
  let text
  try {
    text = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch (error) {
    throw new Error(`prompt ${workload.id} is not valid UTF-8: ${error.message}`)
  }
  return text
}

async function materializePrompt(outputDir, campaignId, workloadId, block, template) {
  const marker = `CAMELID-BENCHMARK-MARKER:${campaignId}/${workloadId}/${String(block).padStart(6, '0')}`
  const text = `${marker}\n${template}`
  const path = join(outputDir, 'prompts', workloadId, `block-${String(block).padStart(3, '0')}.txt`)
  await mkdir(resolve(path, '..'), { recursive: true })
  await writeFile(path, text, 'utf8')
  return { path, sha256: sha256Bytes(Buffer.from(text, 'utf8')), marker }
}

function preparedMap(plan, preparedArms) {
  if (!Array.isArray(preparedArms)) throw new TypeError('preparedArms must be an array')
  const map = new Map()
  for (const prepared of preparedArms) {
    if (map.has(prepared.arm_id)) throw new Error(`duplicate prepared arm ${prepared.arm_id}`)
    map.set(prepared.arm_id, prepared)
  }
  for (const arm of plan.source_arms) {
    const prepared = map.get(arm.id)
    if (!prepared) throw new Error(`missing prepared arm ${arm.id}`)
    if (prepared.source_sha !== arm.git_sha) throw new Error(`prepared arm ${arm.id} source SHA does not match the plan`)
  }
  if (map.size !== plan.source_arms.length) throw new Error('prepared arms include an arm that is not in the plan')
  return map
}

function classifyProcessFailure(result) {
  if (result.cleanupPassed !== true) return { validity: 'invalid_cleanup', reason: result.cleanupDetail || 'process cleanup failed' }
  if (result.timedOut) return { validity: 'invalid_timeout', reason: `bench-generate exceeded its process timeout` }
  if (result.state !== 'exited' || result.exitCode !== 0) {
    return { validity: 'invalid_environment', reason: result.error || result.stderr.preview.trim() || `${result.state} exit ${result.exitCode}` }
  }
  return null
}

function processRecord(result) {
  return {
    state: result.state,
    exit_code: result.exitCode,
    timed_out: result.timedOut,
    cleanup_passed: result.cleanupPassed,
  }
}

function tokenIdsDigest(tokenIds) {
  return sha256Bytes(Buffer.from(JSON.stringify(tokenIds), 'utf8'))
}

async function defaultVerifyFile(path, expectedSha256, kind, id) {
  const info = await stat(path)
  if (!info.isFile()) throw new Error(`${kind} ${id} is not a file: ${path}`)
  const actual = await sha256File(path)
  if (actual !== expectedSha256) throw new Error(`${kind} ${id} resolved to ${actual}, expected ${expectedSha256}`)
}
