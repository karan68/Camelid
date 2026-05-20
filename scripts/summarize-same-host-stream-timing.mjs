#!/usr/bin/env node

import { readFile, writeFile } from 'node:fs/promises'
import { basename, resolve } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const inputPath = args.get('input') || args.get('in') || args.positionals[0]
const outPath = args.get('out')

if (args.has('help') || args.has('h') || !inputPath) {
  console.log(usage())
  process.exit(inputPath ? 0 : 1)
}

const input = JSON.parse(await readFile(inputPath, 'utf8'))
const report = summarizeSameHostStreamTiming(input, inputPath)

const text = humanSummary(report)
console.log(text)
if (outPath) {
  await writeFile(resolve(outPath), `${JSON.stringify(report, null, 2)}\n`)
}

export function summarizeSameHostStreamTiming(input, inputPath = 'same-host.json') {
  const camelidRuns = input?.camelid?.runs
  if (!Array.isArray(camelidRuns) || camelidRuns.length === 0) {
    throw new Error('missing same-host camelid.runs array')
  }
  const llamaRuns = Array.isArray(input?.llama_cpp?.runs) ? input.llama_cpp.runs : []
  const analyzedRuns = camelidRuns.map((run, index) => analyzeCamelidRun(run, index))
  const llamaTtfts = llamaRuns.map((run) => finite(run.first_content_ms ?? run.first_byte_ms)).filter((value) => value !== null)

  const roleKeys = [
    'attention_context',
    'attention_output',
    'ffn_gate',
    'ffn_up',
    'ffn_down',
    'logits',
  ]
  const stageKeys = ['prefill', 'first_token', 'generation']
  const stages = Object.fromEntries(stageKeys.map((stage) => [
    stage,
    Object.fromEntries(roleKeys.map((role) => [role, stats(analyzedRuns.map((run) => run.roles?.[stage]?.[role]))])),
  ]))

  const firstByteStats = stats(analyzedRuns.map((run) => run.client_first_byte_ms))
  const firstContentStats = stats(analyzedRuns.map((run) => run.backend_first_content_ms))
  const ttftStats = stats(analyzedRuns.map((run) => run.client_ttft_ms))
  const generateStats = stats(analyzedRuns.map((run) => run.backend_generate_ms))
  const totalStats = stats(analyzedRuns.map((run) => run.client_total_ms))
  const llamaTtftStats = stats(llamaTtfts)
  const residuals = analyzedRuns.map((run) => run.client_minus_backend_first_content_ms).filter((value) => value !== null)
  const firstContentMinusFirstByte = analyzedRuns.map((run) => run.client_first_content_minus_first_byte_ms).filter((value) => value !== null)
  const backendGenerateMinusFirstByte = analyzedRuns.map((run) => run.backend_generate_minus_first_byte_ms).filter((value) => value !== null)
  const backendGenerateMinusFirstContent = analyzedRuns.map((run) => run.backend_generate_minus_first_content_ms).filter((value) => value !== null)

  const rankedRoles = rankRoles(stages)
  const outliers = rankOutliers(analyzedRuns, firstContentStats, stages)
  const q8RoleWork = summarizeQ8RoleWork(analyzedRuns)
  const runDeltas = analyzedRuns.map((run, index) => {
    const llamaTtft = llamaTtfts[index] ?? null
    return {
      label: run.label,
      client_first_byte_ms: run.client_first_byte_ms,
      client_ttft_ms: run.client_ttft_ms,
      backend_first_content_ms: run.backend_first_content_ms,
      backend_generate_ms: run.backend_generate_ms,
      first_content_minus_first_byte_ms: run.client_first_content_minus_first_byte_ms,
      backend_generate_minus_first_byte_ms: run.backend_generate_minus_first_byte_ms,
      backend_generate_minus_first_content_ms: run.backend_generate_minus_first_content_ms,
      client_minus_backend_first_content_ms: run.client_minus_backend_first_content_ms,
      llama_cpp_ttft_ms: llamaTtft,
      backend_first_content_delta_vs_llama_cpp_ms: delta(run.backend_first_content_ms, llamaTtft),
      q8_calls: run.q8_calls,
      q8_total_gemm_us: run.q8_total_gemm_us,
      q8_total_pack_us: run.q8_total_pack_us,
      q8_fused_gate_up_calls: run.q8_fused_gate_up_calls,
      prompt_cache_hit: run.prompt_cache_hit,
      weight_cache_hit: run.weight_cache_hit,
    }
  })

  return {
    schema: 'camelid.same_host_stream_timing_summary.v1',
    input: basename(inputPath),
    generated_utc: new Date().toISOString(),
    method: {
      row_id: input?.model?.row_id ?? null,
      model_id: input?.model?.model_id ?? null,
      warmup: input?.method?.warmup ?? null,
      repeats: input?.method?.repeats ?? camelidRuns.length,
      max_tokens: input?.method?.max_tokens ?? null,
      unique_prompt: input?.method?.unique_prompt ?? null,
      require_marker: input?.method?.require_marker ?? null,
    },
    aggregate: {
      camelid_client_first_byte_ms: firstByteStats,
      camelid_client_ttft_ms: ttftStats,
      camelid_backend_first_content_ms: firstContentStats,
      camelid_backend_generate_ms: generateStats,
      camelid_client_total_ms: totalStats,
      llama_cpp_ttft_ms: llamaTtftStats,
      first_content_minus_first_byte_ms: stats(firstContentMinusFirstByte),
      backend_generate_minus_first_byte_ms: stats(backendGenerateMinusFirstByte),
      backend_generate_minus_first_content_ms: stats(backendGenerateMinusFirstContent),
      client_minus_backend_first_content_ms: stats(residuals),
      backend_first_content_delta_vs_llama_cpp_ttft_ms: stats(runDeltas.map((run) => run.backend_first_content_delta_vs_llama_cpp_ms)),
      q8_calls: stats(analyzedRuns.map((run) => run.q8_calls)),
      q8_total_gemm_us: stats(analyzedRuns.map((run) => run.q8_total_gemm_us)),
      q8_total_pack_us: stats(analyzedRuns.map((run) => run.q8_total_pack_us)),
      q8_fused_gate_up_calls: stats(analyzedRuns.map((run) => run.q8_fused_gate_up_calls)),
      prompt_cache_hits: analyzedRuns.filter((run) => run.prompt_cache_hit === true).length,
      weight_cache_hits: analyzedRuns.filter((run) => run.weight_cache_hit === true).length,
    },
    stages,
    ranked_roles: rankedRoles,
    q8_role_work: q8RoleWork,
    outliers,
    runs: runDeltas,
  }
}

function analyzeCamelidRun(run, index) {
  const timings = run?.backend_timing?.timings_ms ?? {}
  const q8Schedule = run?.backend_timing?.q8_schedule ?? {}
  const roles = {
    prefill: timings.prefill_role_timings ?? {},
    first_token: timings.first_token_role_timings ?? {},
    generation: timings.generation_role_timings ?? {},
  }
  const clientFirstByte = finite(run.first_byte_ms)
  const clientTtft = finite(run.first_content_ms ?? run.ttft_ms)
  const backendFirstContent = finite(run.backend_first_content_ms ?? timings.first_content)
  const backendGenerate = finite(run.backend_generate_ms ?? timings.generate)
  return {
    label: run.label ?? `camelid-run-${index + 1}`,
    client_first_byte_ms: clientFirstByte,
    client_ttft_ms: clientTtft,
    client_total_ms: finite(run.total_elapsed_ms),
    backend_generate_ms: backendGenerate,
    backend_first_content_ms: backendFirstContent,
    client_first_content_minus_first_byte_ms: delta(clientTtft, clientFirstByte),
    backend_generate_minus_first_byte_ms: delta(backendGenerate, clientFirstByte),
    backend_generate_minus_first_content_ms: delta(backendGenerate, backendFirstContent),
    client_minus_backend_first_content_ms: delta(clientTtft, backendFirstContent),
    q8_calls: finite(run.backend_q8_calls ?? run?.backend_timing?.q8_schedule?.i8mm_single_projection_calls),
    q8_total_gemm_us: finite(run.backend_q8_gemm_compute_us ?? q8Schedule.q8_gemm_compute_us),
    q8_total_pack_us: finite(run.backend_q8_pack_us ?? q8Schedule.activation_quantize_pack_us),
    q8_fused_gate_up_calls: finite(q8Schedule.i8mm_fused_gate_up_calls),
    q8_roles: q8Schedule.i8mm_single_projection_by_role ?? {},
    prompt_cache_hit: timings.prompt_cache_hit ?? null,
    weight_cache_hit: timings.weight_cache_hit ?? null,
    roles,
  }
}

function rankRoles(stages) {
  const rows = []
  for (const [stage, roles] of Object.entries(stages)) {
    for (const [role, stat] of Object.entries(roles)) {
      if (stat.count > 0) {
        rows.push({ stage, role, mean_ms: stat.mean, p95_ms: stat.p95, max_ms: stat.max })
      }
    }
  }
  rows.sort((left, right) => (right.mean_ms ?? 0) - (left.mean_ms ?? 0))
  return rows
}

function rankOutliers(runs, baselineStats, stages) {
  const mean = baselineStats.mean ?? 0
  return runs
    .map((run) => ({
      label: run.label,
      backend_first_content_ms: run.backend_first_content_ms,
      client_ttft_ms: run.client_ttft_ms,
      over_mean_ms: delta(run.backend_first_content_ms, mean),
      q8_calls: run.q8_calls,
      q8_total_gemm_us: run.q8_total_gemm_us,
      q8_total_pack_us: run.q8_total_pack_us,
      top_role_deltas: topRoleDeltas(run, stages),
      prompt_cache_hit: run.prompt_cache_hit,
    }))
    .filter((run) => run.over_mean_ms !== null)
    .sort((left, right) => right.over_mean_ms - left.over_mean_ms)
    .slice(0, 5)
}

function topRoleDeltas(run, stages) {
  const rows = []
  for (const [stage, roles] of Object.entries(run.roles ?? {})) {
    for (const [role, value] of Object.entries(roles ?? {})) {
      const valueMs = finite(value)
      const meanMs = finite(stages?.[stage]?.[role]?.mean)
      if (valueMs !== null && meanMs !== null) {
        rows.push({
          stage,
          role,
          value_ms: valueMs,
          over_role_mean_ms: delta(valueMs, meanMs),
        })
      }
    }
  }
  return rows
    .filter((row) => row.over_role_mean_ms !== null)
    .sort((left, right) => right.over_role_mean_ms - left.over_role_mean_ms)
    .slice(0, 5)
}

function summarizeQ8RoleWork(runs) {
  const roleNames = new Set()
  for (const run of runs) {
    for (const role of Object.keys(run.q8_roles ?? {})) roleNames.add(role)
  }
  return [...roleNames].sort().map((role) => {
    const samples = runs.map((run) => run.q8_roles?.[role] ?? {})
    const calls = stats(samples.map((sample) => sample.calls))
    const packUs = stats(samples.map((sample) => sample.pack_us))
    const gemmUs = stats(samples.map((sample) => sample.gemm_us))
    const rows = stats(samples.map((sample) => sample.rows))
    return {
      role,
      calls,
      pack_us: packUs,
      gemm_us: gemmUs,
      rows,
      gemm_ms_mean: gemmUs.mean === null ? null : round(gemmUs.mean / 1000),
      pack_ms_mean: packUs.mean === null ? null : round(packUs.mean / 1000),
    }
  }).sort((left, right) => (right.gemm_us.mean ?? 0) - (left.gemm_us.mean ?? 0))
}

function stats(values) {
  const xs = values.map(finite).filter((value) => value !== null).sort((a, b) => a - b)
  if (xs.length === 0) {
    return { count: 0, min: null, p50: null, mean: null, p95: null, max: null }
  }
  const sum = xs.reduce((acc, value) => acc + value, 0)
  return {
    count: xs.length,
    min: round(xs[0]),
    p50: round(percentile(xs, 0.5)),
    mean: round(sum / xs.length),
    p95: round(percentile(xs, 0.95)),
    max: round(xs[xs.length - 1]),
  }
}

function percentile(sorted, p) {
  if (sorted.length === 1) return sorted[0]
  const index = (sorted.length - 1) * p
  const lower = Math.floor(index)
  const upper = Math.ceil(index)
  if (lower === upper) return sorted[lower]
  const weight = index - lower
  return sorted[lower] * (1 - weight) + sorted[upper] * weight
}

function humanSummary(report) {
  const top = report.ranked_roles.slice(0, 6)
    .map((row, index) => `${index + 1}. ${row.stage}.${row.role} mean=${fmt(row.mean_ms)}ms p95=${fmt(row.p95_ms)}ms`)
    .join('\n')
  const outliers = report.outliers.slice(0, 3)
    .map((run) => {
      const deltas = run.top_role_deltas?.slice(0, 3)
        .map((role) => `${role.stage}.${role.role}+${fmt(role.over_role_mean_ms)}ms`)
        .join(', ')
      const suffix = deltas ? ` role_deltas=[${deltas}]` : ''
      return `${run.label}: backend_first_content=${fmt(run.backend_first_content_ms)}ms over_mean=${fmt(run.over_mean_ms)}ms q8_calls=${run.q8_calls}${suffix}`
    })
    .join('\n')
  const q8Roles = report.q8_role_work.slice(0, 6)
    .map((row, index) => `${index + 1}. ${row.role} gemm_mean=${fmt(row.gemm_ms_mean)}ms pack_mean=${fmt(row.pack_ms_mean)}ms calls_mean=${fmt(row.calls.mean)}`)
    .join('\n')
  return [
    `schema=${report.schema}`,
    `input=${report.input}`,
    `runs=${report.aggregate.camelid_client_ttft_ms.count}`,
    `client_first_byte_mean_ms=${fmt(report.aggregate.camelid_client_first_byte_ms.mean)}`,
    `camelid_ttft_mean_ms=${fmt(report.aggregate.camelid_client_ttft_ms.mean)}`,
    `backend_first_content_mean_ms=${fmt(report.aggregate.camelid_backend_first_content_ms.mean)}`,
    `backend_generate_mean_ms=${fmt(report.aggregate.camelid_backend_generate_ms.mean)}`,
    `first_content_minus_first_byte_mean_ms=${fmt(report.aggregate.first_content_minus_first_byte_ms.mean)}`,
    `backend_generate_minus_first_byte_mean_ms=${fmt(report.aggregate.backend_generate_minus_first_byte_ms.mean)}`,
    `backend_generate_minus_first_content_mean_ms=${fmt(report.aggregate.backend_generate_minus_first_content_ms.mean)}`,
    `llama_cpp_ttft_mean_ms=${fmt(report.aggregate.llama_cpp_ttft_ms.mean)}`,
    `client_minus_backend_first_content_mean_ms=${fmt(report.aggregate.client_minus_backend_first_content_ms.mean)}`,
    `backend_first_content_delta_vs_llama_cpp_mean_ms=${fmt(report.aggregate.backend_first_content_delta_vs_llama_cpp_ttft_ms.mean)}`,
    `q8_calls_mean=${fmt(report.aggregate.q8_calls.mean)}`,
    `q8_total_gemm_mean_ms=${fmt(report.aggregate.q8_total_gemm_us.mean === null ? null : report.aggregate.q8_total_gemm_us.mean / 1000)}`,
    `q8_total_pack_mean_ms=${fmt(report.aggregate.q8_total_pack_us.mean === null ? null : report.aggregate.q8_total_pack_us.mean / 1000)}`,
    `q8_fused_gate_up_calls_mean=${fmt(report.aggregate.q8_fused_gate_up_calls.mean)}`,
    'top_roles:',
    top,
    'top_q8_gemm_roles:',
    q8Roles,
    'top_backend_first_content_outliers:',
    outliers,
  ].filter(Boolean).join('\n')
}

function parseArgs(argv) {
  const map = new Map()
  map.positionals = []
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (!arg.startsWith('--')) {
      map.positionals.push(arg)
      continue
    }
    const key = arg.slice(2)
    if (key.includes('=')) {
      const [name, ...rest] = key.split('=')
      map.set(name, rest.join('='))
    } else if (i + 1 < argv.length && !argv[i + 1].startsWith('--')) {
      map.set(key, argv[++i])
    } else {
      map.set(key, true)
    }
  }
  return map
}

function finite(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function delta(left, right) {
  left = finite(left)
  right = finite(right)
  return left === null || right === null ? null : round(left - right)
}

function round(value) {
  return Number.isFinite(value) ? Math.round(value * 1000) / 1000 : null
}

function fmt(value) {
  return value === null || value === undefined ? 'null' : Number(value).toFixed(3)
}

function usage() {
  return `Usage: node scripts/summarize-same-host-stream-timing.mjs --input same-host.json [--out summary.json]\n\nSummarizes Camelid same-host streaming diagnostics, first-byte vs backend generate/first-content gaps, backend first-content residuals, role timing hot spots, and Q8 scheduler work by role.`
}
