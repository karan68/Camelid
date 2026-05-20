#!/usr/bin/env node

import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const scriptDir = dirname(fileURLToPath(import.meta.url))
const summaryScript = join(scriptDir, 'summarize-same-host-stream-timing.mjs')
const tempDir = await mkdtemp(join(tmpdir(), 'camelid-same-host-stream-summary-'))

try {
  const inputPath = join(tempDir, 'same-host.json')
  const outPath = join(tempDir, 'summary.json')
  await writeFile(inputPath, `${JSON.stringify(fixture(), null, 2)}\n`)

  const { stdout } = await execFileAsync(process.execPath, [
    summaryScript,
    '--input', inputPath,
    '--out', outPath,
  ], { cwd: resolve(scriptDir, '..') })

  assert.match(stdout, /schema=camelid\.same_host_stream_timing_summary\.v1/)
  assert.match(stdout, /runs=3/)
  assert.match(stdout, /client_first_byte_mean_ms=25\.000/)
  assert.match(stdout, /backend_first_content_mean_ms=210\.000/)
  assert.match(stdout, /first_content_minus_first_byte_mean_ms=195\.000/)
  assert.match(stdout, /backend_generate_minus_first_content_mean_ms=590\.000/)
  assert.match(stdout, /client_minus_backend_first_content_mean_ms=10\.000/)
  assert.match(stdout, /q8_total_gemm_mean_ms=126\.000/)
  assert.match(stdout, /q8_fused_gate_up_calls_mean=28\.000/)
  assert.match(stdout, /1\. generation\.ffn_down mean=92\.000ms/)
  assert.match(stdout, /1\. ffn_down gemm_mean=70\.000ms pack_mean=7\.000ms calls_mean=28\.000/)

  const report = JSON.parse(await readFile(outPath, 'utf8'))
  assert.equal(report.schema, 'camelid.same_host_stream_timing_summary.v1')
  assert.equal(report.aggregate.camelid_client_first_byte_ms.mean, 25)
  assert.equal(report.aggregate.camelid_client_ttft_ms.mean, 220)
  assert.equal(report.aggregate.llama_cpp_ttft_ms.mean, 150)
  assert.equal(report.aggregate.first_content_minus_first_byte_ms.mean, 195)
  assert.equal(report.aggregate.backend_generate_minus_first_byte_ms.mean, 775)
  assert.equal(report.aggregate.backend_generate_minus_first_content_ms.mean, 590)
  assert.equal(report.aggregate.backend_first_content_delta_vs_llama_cpp_ttft_ms.mean, 60)
  assert.equal(report.aggregate.q8_calls.mean, 140)
  assert.equal(report.aggregate.q8_total_gemm_us.mean, 126000)
  assert.equal(report.aggregate.q8_total_pack_us.mean, 12600)
  assert.equal(report.aggregate.q8_fused_gate_up_calls.mean, 28)
  assert.equal(report.aggregate.prompt_cache_hits, 0)
  assert.equal(report.aggregate.weight_cache_hits, 3)
  assert.equal(report.stages.prefill.ffn_down.mean, 40)
  assert.equal(report.stages.first_token.logits.mean, 5)
  assert.equal(report.ranked_roles[0].stage, 'generation')
  assert.equal(report.ranked_roles[0].role, 'ffn_down')
  assert.equal(report.q8_role_work[0].role, 'ffn_down')
  assert.equal(report.q8_role_work[0].gemm_ms_mean, 70)
  assert.equal(report.q8_role_work[1].role, 'attention_output')
  assert.equal(report.outliers[0].label, 'camelid-measure-3')
  assert.equal(report.outliers[0].top_role_deltas[0].stage, 'prefill')
  assert.equal(report.outliers[0].top_role_deltas[0].role, 'ffn_down')
  assert.equal(report.runs[1].client_first_byte_ms, 25)
  assert.equal(report.runs[1].first_content_minus_first_byte_ms, 195)
  assert.equal(report.runs[1].backend_generate_minus_first_content_ms, 590)
  assert.equal(report.runs[1].backend_first_content_delta_vs_llama_cpp_ms, 50)
  assert.equal(report.runs[1].q8_total_gemm_us, 126000)
  assert.equal(report.runs[1].q8_fused_gate_up_calls, 28)

  const missingPath = join(tempDir, 'missing.json')
  await writeFile(missingPath, '{}\n')
  let failed = false
  try {
    await execFileAsync(process.execPath, [summaryScript, missingPath], { cwd: resolve(scriptDir, '..') })
  } catch (err) {
    failed = true
    assert.match(err.stderr, /missing same-host camelid\.runs array/)
  }
  assert.equal(failed, true)

  console.log('summarize-same-host-stream-timing self-test passed')
} finally {
  await rm(tempDir, { recursive: true, force: true })
}

function fixture() {
  return {
    model: { row_id: 'llama32_3b_instruct_q8_0_mac', model_id: 'llama32-3b-q8' },
    method: { warmup: 2, repeats: 3, max_tokens: 8, unique_prompt: true, require_marker: true },
    camelid: {
      runs: [
        run('camelid-measure-1', 20, 190, 200, 780, 10),
        run('camelid-measure-2', 25, 210, 220, 800, 10),
        run('camelid-measure-3', 30, 230, 240, 820, 10),
      ],
    },
    llama_cpp: {
      runs: [
        { label: 'llama-measure-1', first_content_ms: 140 },
        { label: 'llama-measure-2', first_content_ms: 160 },
        { label: 'llama-measure-3', first_content_ms: 150 },
      ],
    },
  }
}

function run(label, firstByte, backendFirstContent, clientTtft, generate, residual) {
  const index = Number(label.match(/(\d+)$/)?.[1] ?? 1)
  return {
    label,
    first_byte_ms: firstByte,
    first_content_ms: backendFirstContent + residual,
    total_elapsed_ms: generate + 300,
    backend_first_content_ms: backendFirstContent,
    backend_generate_ms: generate,
    backend_q8_calls: 140,
    backend_timing: {
      q8_schedule: {
        i8mm_single_projection_calls: 140,
        i8mm_fused_gate_up_calls: 28,
        q8_gemm_compute_us: 126000,
        activation_quantize_pack_us: 12600,
        i8mm_single_projection_by_role: {
          attention_q: { calls: 28, pack_us: 1000, gemm_us: 5000, rows: 2000 },
          attention_k: { calls: 28, pack_us: 1100, gemm_us: 6000, rows: 2000 },
          attention_v: { calls: 28, pack_us: 1200, gemm_us: 7000, rows: 2000 },
          attention_output: { calls: 28, pack_us: 3000, gemm_us: 38000, rows: 2000 },
          ffn_down: { calls: 28, pack_us: 7000, gemm_us: 70000, rows: 2000 },
        },
      },
      timings_ms: {
        prompt_cache_hit: false,
        weight_cache_hit: true,
        first_content: backendFirstContent,
        generate,
        prefill_role_timings: {
          attention_context: 6,
          attention_output: 14,
          ffn_gate: 30,
          ffn_up: 31,
          ffn_down: 36 + (index * 2),
          logits: 0,
        },
        first_token_role_timings: {
          attention_context: 2,
          attention_output: 3,
          ffn_gate: 4,
          ffn_up: 4,
          ffn_down: 8,
          logits: 5,
        },
        generation_role_timings: {
          attention_context: 12,
          attention_output: 24,
          ffn_gate: 60,
          ffn_up: 61,
          ffn_down: 92,
          logits: 15,
        },
      },
    },
  }
}
