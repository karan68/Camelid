#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'

import { renderExpectedPrompt, resolveReferenceContext } from './lib/chat-parity-harness.mjs'

const args = parseArgs(process.argv.slice(2))
const backendBase = (args.get('backend') || process.env.CAMELID_API_BASE || 'http://127.0.0.1:8181').replace(/\/$/, '')
const llamaBase = (args.get('llama-url') || process.env.LLAMA3_LLAMA_SERVER_URL || 'http://127.0.0.1:8183').replace(/\/$/, '')
const modelPath = resolve(args.get('model') || process.env.LLAMA3_GGUF || 'models/Llama-3.2-3B-Instruct-Q8_0.gguf')
const modelId = args.get('model-id') || process.env.LLAMA3_MODEL_ID || 'llama32-3b-q8'
const rowId = args.get('row-id') || process.env.CAMELID_BENCH_ROW_ID || 'llama32_3b_instruct_q8_0'
const backendBin = resolve(args.get('backend-bin') || process.env.CAMELID_BIN || 'target/release/camelid')
const llamaServerBin = resolve(args.get('llama-server') || process.env.LLAMA3_LLAMA_SERVER || 'target/reference/llama.cpp/build/bin/llama-server')
const renderMode = args.get('render-mode') || process.env.LLAMA3_CHAT_RENDER_MODE || 'compact'
const maxTokens = parsePositiveInt(args.get('max-tokens') || process.env.CAMELID_BENCH_MAX_TOKENS || '16', 'max-tokens')
const repeats = parsePositiveInt(args.get('repeats') || process.env.CAMELID_BENCH_REPEATS || '3', 'repeats')
const warmup = parseNonNegativeInt(args.get('warmup') || process.env.CAMELID_BENCH_WARMUP || '1', 'warmup')
const out = args.get('out') || process.env.CAMELID_SAME_HOST_BENCH_OUT
const startBackend = args.get('start-backend') !== 'false'
const startLlamaServer = args.get('start-llama-server') !== 'false'
const waitMs = parsePositiveInt(args.get('wait-ms') || process.env.CAMELID_BENCH_WAIT_MS || '600000', 'wait-ms')
const explicitLlamaContext = parseOptionalPositiveInt(args.get('llama-context') || process.env.LLAMA3_LLAMA_CONTEXT, 'llama-context')
const threads = parseOptionalPositiveInt(args.get('threads') || process.env.CAMELID_BENCH_THREADS, 'threads')

if (args.has('help') || args.has('h')) {
  console.log(usage())
  process.exit(0)
}

const benchmarkMessages = [
  { role: 'system', content: 'You are Camelid benchmark mode. Reply with concise factual text only.' },
  { role: 'user', content: 'Give a short three-line summary of why exact-row parity matters for local inference, and end the last line with the marker CMLD-BENCH.' },
]

const expectedPrompt = renderExpectedPrompt(benchmarkMessages, renderMode)
const estimatedPromptTokens = estimatePromptTokens(expectedPrompt)
const llamaContext = resolveReferenceContext({
  promptTokenCount: estimatedPromptTokens,
  maxTokens,
  explicitContext: explicitLlamaContext,
})

if (args.has('print-plan')) {
  const plan = buildPlan()
  printPlan(plan)
  if (out) {
    const outPath = resolve(out)
    await mkdir(dirname(outPath), { recursive: true })
    await writeFile(outPath, `${JSON.stringify(plan, null, 2)}\n`)
    console.log(`json_out=${outPath}`)
  }
  process.exit(0)
}

let backendChild = null
let llamaChild = null
let backendSpawnError = null
let llamaSpawnError = null

try {
  if (startBackend) {
    const url = new URL(backendBase)
    backendChild = spawn(backendBin, ['serve', '--addr', `${url.hostname}:${url.port || '8181'}`], { stdio: ['ignore', 'pipe', 'pipe'] })
    backendChild.once('error', (err) => { backendSpawnError = err })
    backendChild.stdout.on('data', (chunk) => process.stderr.write(`[camelid] ${chunk}`))
    backendChild.stderr.on('data', (chunk) => process.stderr.write(`[camelid] ${chunk}`))
  }

  if (startLlamaServer) {
    const url = new URL(llamaBase)
    const llamaArgs = ['--host', url.hostname, '--port', url.port || '8183', '-m', modelPath, '-ngl', '0', '-c', String(llamaContext), '--no-warmup']
    if (threads) llamaArgs.push('-t', String(threads))
    llamaChild = spawn(llamaServerBin, llamaArgs, { stdio: ['ignore', 'pipe', 'pipe'] })
    llamaChild.once('error', (err) => { llamaSpawnError = err })
    llamaChild.stdout.on('data', (chunk) => process.stderr.write(`[llama-server] ${chunk}`))
    llamaChild.stderr.on('data', (chunk) => process.stderr.write(`[llama-server] ${chunk}`))
  }

  await waitForJson(`${backendBase}/v1/health`, {}, 'camelid', waitMs)
  await waitForJson(`${llamaBase}/health`, {}, 'llama-server', waitMs).catch((err) => {
    if (llamaSpawnError?.code === 'ENOENT') {
      throw new Error(`could not start llama-server binary ${JSON.stringify(llamaServerBin)}`)
    }
    throw err
  })
  if (backendSpawnError?.code === 'ENOENT') {
    throw new Error(`could not start Camelid binary ${JSON.stringify(backendBin)}`)
  }

  await fetchJson(`${backendBase}/api/models/load`, {
    method: 'POST',
    body: JSON.stringify({ path: modelPath, id: modelId }),
  })

  const camelidWarmups = []
  const llamaWarmups = []
  for (let i = 0; i < warmup; i += 1) {
    camelidWarmups.push(await runCamelidStream(i, 'warmup'))
    llamaWarmups.push(await runLlamaStream(i, 'warmup'))
  }

  const camelidRuns = []
  const llamaRuns = []
  for (let i = 0; i < repeats; i += 1) {
    camelidRuns.push(await runCamelidStream(i, 'measure'))
    llamaRuns.push(await runLlamaStream(i, 'measure'))
  }

  const report = {
    schema: 'camelid.same_host_llama3_benchmark.v1',
    generated_utc: new Date().toISOString(),
    model: {
      row_id: rowId,
      model_path: modelPath,
      model_id: modelId,
      render_mode: renderMode,
    },
    method: {
      warmup,
      repeats,
      max_tokens: maxTokens,
      benchmark_messages: benchmarkMessages,
      estimated_prompt_tokens: estimatedPromptTokens,
      llama_context: llamaContext,
      threads: threads ?? null,
      commands: buildPlan().commands,
      outputs: buildPlan().outputs,
      bounded_metrics: boundedMetrics(),
      note: 'Same-host comparison using streaming requests to Camelid /v1/chat/completions and llama.cpp /completion. TTFT is first non-empty streamed content chunk; token throughput is estimated from streamed content chunks, not tokenizer-ground-truth completion tokens.',
    },
    camelid: {
      base_url: backendBase,
      warmups: camelidWarmups,
      runs: camelidRuns,
      summary: summarizeRuns(camelidRuns),
    },
    llama_cpp: {
      base_url: llamaBase,
      warmups: llamaWarmups,
      runs: llamaRuns,
      summary: summarizeRuns(llamaRuns),
      binary: llamaServerBin,
    },
    comparison: compareSummaries(summarizeRuns(camelidRuns), summarizeRuns(llamaRuns)),
    claim_boundary: claimBoundary(),
  }

  printHumanSummary(report)
  if (out) {
    const outPath = resolve(out)
    await mkdir(dirname(outPath), { recursive: true })
    await writeFile(outPath, `${JSON.stringify(report, null, 2)}\n`)
    console.log(`json_out=${outPath}`)
  }
} finally {
  backendChild?.kill('SIGTERM')
  llamaChild?.kill('SIGTERM')
}

function buildPlan() {
  const backendUrl = new URL(backendBase)
  const llamaUrl = new URL(llamaBase)
  const llamaArgs = ['--host', llamaUrl.hostname, '--port', llamaUrl.port || '8183', '-m', modelPath, '-ngl', '0', '-c', String(llamaContext), '--no-warmup']
  if (threads) llamaArgs.push('-t', String(threads))
  return {
    schema: 'camelid.same_host_llama3_benchmark_plan.v1',
    generated_utc: new Date().toISOString(),
    model: {
      row_id: rowId,
      model_path: modelPath,
      model_id: modelId,
      render_mode: renderMode,
    },
    method: {
      warmup,
      repeats,
      max_tokens: maxTokens,
      estimated_prompt_tokens: estimatedPromptTokens,
      llama_context: llamaContext,
      threads: threads ?? null,
      benchmark_messages: benchmarkMessages,
      bounded_metrics: boundedMetrics(),
    },
    commands: {
      harness: `node scripts/bench-llama3-same-host.mjs --model ${shellQuote(modelPath)} --model-id ${shellQuote(modelId)} --row-id ${shellQuote(rowId)} --max-tokens ${maxTokens} --warmup ${warmup} --repeats ${repeats}${threads ? ` --threads ${threads}` : ''}${explicitLlamaContext ? ` --llama-context ${explicitLlamaContext}` : ''}${out ? ` --out ${shellQuote(resolve(out))}` : ''}`,
      camelid_serve: startBackend ? `${shellQuote(backendBin)} serve --addr ${shellQuote(`${backendUrl.hostname}:${backendUrl.port || '8181'}`)}` : 'not started by harness (--start-backend=false)',
      llama_server: startLlamaServer ? [shellQuote(llamaServerBin), ...llamaArgs.map(shellQuote)].join(' ') : 'not started by harness (--start-llama-server=false)',
      camelid_load_request: `POST ${backendBase}/api/models/load {"path":${JSON.stringify(modelPath)},"id":${JSON.stringify(modelId)}}`,
      camelid_measure_request: `POST ${backendBase}/v1/chat/completions stream=true max_tokens=${maxTokens} temperature=0`,
      llama_cpp_measure_request: `POST ${llamaBase}/completion stream=true n_predict=${maxTokens} temperature=0 cache_prompt=false`,
    },
    outputs: {
      stdout: [
        'camelid_ttft_ms=<mean first non-empty streamed content chunk over measured runs>',
        'camelid_decode_tok_s=<mean estimated streamed chunks per second after first content>',
        'camelid_ms_tok=<mean estimated milliseconds per streamed content chunk after first content>',
        'llama_cpp_ttft_ms=<same metric for llama.cpp>',
        'llama_cpp_decode_tok_s=<same metric for llama.cpp>',
        'llama_cpp_ms_tok=<same metric for llama.cpp>',
        'json_out=<absolute path when --out is set>',
      ],
      json: 'Full machine-readable report at --out, schema camelid.same_host_llama3_benchmark.v1.',
    },
    claim_boundary: claimBoundary(),
  }
}

function boundedMetrics() {
  return [
    'first_byte_ms: first network byte from each streaming response',
    'first_event_ms: first parsed SSE event',
    'first_content_ms / TTFT: first non-empty streamed content chunk',
    'total_elapsed_ms: full streaming response wall time',
    'completion_tokens_estimate: count of non-empty streamed content chunks, not tokenizer-ground-truth tokens',
    'decode_tok_per_s and ms_per_token_after_first: derived from completion_tokens_estimate after first content',
  ]
}

function claimBoundary() {
  return `Same-host benchmark snapshot only for exact row ${rowId} with the provided GGUF, prompt, max-token budget, host, binaries, and thread settings. It is bounded timing evidence only: it does not widen support, portability, production-throughput, model-native/larger-context, neighboring-row, broad Llama-family, 1B, or Mixtral claims unless a separate row-specific evidence bundle records those exact conditions.`
}

function printPlan(plan) {
  console.log(`schema=${plan.schema}`)
  console.log(`row_id=${plan.model.row_id}`)
  console.log(`harness_command=${plan.commands.harness}`)
  console.log(`camelid_serve=${plan.commands.camelid_serve}`)
  console.log(`llama_server=${plan.commands.llama_server}`)
  console.log(`outputs=${Object.keys(plan.outputs).join(',')}`)
  console.log(`claim_boundary=${plan.claim_boundary}`)
}

function usage() {
  return `Usage: node scripts/bench-llama3-same-host.mjs --model <GGUF> --model-id <id> --row-id <compat-row> [options]

Purpose:
  Repeatable same-host Camelid vs llama.cpp streaming benchmark for one exact Llama-family row.

Key options:
  --backend <url>                 Camelid API base. Default: CAMELID_API_BASE or http://127.0.0.1:8181
  --llama-url <url>               llama-server base. Default: LLAMA3_LLAMA_SERVER_URL or http://127.0.0.1:8183
  --model <path>                  GGUF path. Default: LLAMA3_GGUF or models/Llama-3.2-3B-Instruct-Q8_0.gguf
  --model-id <id>                 API model id. Default: LLAMA3_MODEL_ID or llama32-3b-q8
  --row-id <compat-row>           Compatibility row recorded in output. Default: llama32_3b_instruct_q8_0
  --max-tokens <n>                Completion budget. Default: 16
  --warmup <n>                    Warmup runs per engine. Default: 1
  --repeats <n>                   Measured runs per engine. Default: 3
  --threads <n>                   Optional llama-server CPU threads.
  --llama-context <n>             Optional llama-server context; otherwise bounded from prompt + max tokens.
  --start-backend=false           Reuse an already-running Camelid server.
  --start-llama-server=false      Reuse an already-running llama-server.
  --out <path>                    Write the JSON report or --print-plan JSON.
  --print-plan                    Print exact commands/outputs/metric bounds without starting servers.

Example:
  CAMELID_BIN=target/release/camelid \\
  LLAMA3_LLAMA_SERVER=target/reference/llama.cpp/build/bin/llama-server \\
  node scripts/bench-llama3-same-host.mjs \\
    --model /home/ubuntu/models/Llama-3.2-3B-Instruct-Q8_0.gguf \\
    --model-id llama32-3b-q8-throughput \\
    --row-id llama32_3b_instruct_q8_0 \\
    --max-tokens 16 --warmup 1 --repeats 3 --threads 8 \\
    --out target/bench-llama32-3b-same-host.json

Outputs:
  stdout summary keys: camelid_ttft_ms, camelid_decode_tok_s, camelid_ms_tok,
  llama_cpp_ttft_ms, llama_cpp_decode_tok_s, llama_cpp_ms_tok, json_out.
  JSON report schema: camelid.same_host_llama3_benchmark.v1.

Claim boundary:
  Bounded same-host timing evidence only. This does not promote production throughput,
  portability, 1B, Mixtral, neighboring-row, or broad-family support without separate
  row-specific evidence.`
}

async function runCamelidStream(idx, phase) {
  const started = performance.now()
  const response = await fetch(`${backendBase}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: modelId,
      messages: benchmarkMessages,
      max_tokens: maxTokens,
      stream: true,
      temperature: 0,
    }),
  })
  return consumeSseResponse({ response, started, label: `camelid-${phase}-${idx + 1}` })
}

async function runLlamaStream(idx, phase) {
  const started = performance.now()
  const response = await fetch(`${llamaBase}/completion`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      prompt: expectedPrompt,
      n_predict: maxTokens,
      temperature: 0,
      stream: true,
      cache_prompt: false,
    }),
  })
  return consumeSseResponse({ response, started, label: `llama-${phase}-${idx + 1}` })
}

async function consumeSseResponse({ response, started, label }) {
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`${label} failed with HTTP ${response.status}: ${text.slice(0, 400)}`)
  }
  const reader = response.body?.getReader()
  if (!reader) throw new Error(`${label} returned no body`)
  const decoder = new TextDecoder()
  let buffer = ''
  let content = ''
  let firstByteMs = null
  let firstEventMs = null
  let firstContentMs = null
  let doneAtMs = null
  let chunkCount = 0
  let completionTokens = 0

  const nowMs = () => performance.now() - started

  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    if (firstByteMs === null) firstByteMs = nowMs()
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() || ''
    for (const eventText of parts) {
      if (!String(eventText).trim()) continue
      if (firstEventMs === null) firstEventMs = nowMs()
      const dataLines = String(eventText).split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart())
      for (const data of dataLines) {
        if (!data || data === '[DONE]') continue
        let payload
        try {
          payload = JSON.parse(data)
        } catch {
          continue
        }
        const delta = payload?.choices?.[0]?.delta?.content
          ?? payload?.content
          ?? payload?.choices?.[0]?.text
          ?? ''
        if (delta) {
          if (firstContentMs === null) firstContentMs = nowMs()
          content += delta
          chunkCount += 1
          completionTokens += 1
        }
      }
    }
  }
  doneAtMs = nowMs()
  const decodeWindowMs = firstContentMs === null ? null : Math.max(doneAtMs - firstContentMs, 0)
  return {
    text: content,
    first_byte_ms: round(firstByteMs),
    first_event_ms: round(firstEventMs),
    first_content_ms: round(firstContentMs),
    total_elapsed_ms: round(doneAtMs),
    completion_tokens_estimate: completionTokens,
    chunk_count: chunkCount,
    decode_tok_per_s: decodeWindowMs && completionTokens > 0 ? round((completionTokens / decodeWindowMs) * 1000) : null,
    ms_per_token_after_first: decodeWindowMs && completionTokens > 0 ? round(decodeWindowMs / completionTokens) : null,
  }
}

function summarizeRuns(runs) {
  const avg = (field) => round(average(runs.map((run) => run[field]).filter(Number.isFinite)))
  return {
    count: runs.length,
    avg_first_byte_ms: avg('first_byte_ms'),
    avg_first_event_ms: avg('first_event_ms'),
    avg_ttft_ms: avg('first_content_ms'),
    avg_total_elapsed_ms: avg('total_elapsed_ms'),
    avg_decode_tok_per_s: avg('decode_tok_per_s'),
    avg_ms_per_token_after_first: avg('ms_per_token_after_first'),
    avg_completion_tokens_estimate: avg('completion_tokens_estimate'),
  }
}

function compareSummaries(camelid, llama) {
  const pct = (a, b) => Number.isFinite(a) && Number.isFinite(b) && b !== 0 ? round(((a - b) / b) * 100) : null
  return {
    ttft_delta_pct_vs_llama_cpp: pct(camelid.avg_ttft_ms, llama.avg_ttft_ms),
    total_elapsed_delta_pct_vs_llama_cpp: pct(camelid.avg_total_elapsed_ms, llama.avg_total_elapsed_ms),
    decode_tok_per_s_delta_pct_vs_llama_cpp: pct(camelid.avg_decode_tok_per_s, llama.avg_decode_tok_per_s),
    ms_per_token_after_first_delta_pct_vs_llama_cpp: pct(camelid.avg_ms_per_token_after_first, llama.avg_ms_per_token_after_first),
  }
}

function printHumanSummary(report) {
  const c = report.camelid.summary
  const l = report.llama_cpp.summary
  console.log(`camelid_ttft_ms=${c.avg_ttft_ms}`)
  console.log(`camelid_decode_tok_s=${c.avg_decode_tok_per_s}`)
  console.log(`camelid_ms_tok=${c.avg_ms_per_token_after_first}`)
  console.log(`llama_cpp_ttft_ms=${l.avg_ttft_ms}`)
  console.log(`llama_cpp_decode_tok_s=${l.avg_decode_tok_per_s}`)
  console.log(`llama_cpp_ms_tok=${l.avg_ms_per_token_after_first}`)
}

async function waitForJson(url, init, label, timeoutMs) {
  const started = Date.now()
  for (;;) {
    try {
      return await fetchJson(url, init)
    } catch (error) {
      if (Date.now() - started >= timeoutMs) throw new Error(`${label} did not become ready within ${timeoutMs}ms: ${error.message}`)
      await sleep(500)
    }
  }
}

async function fetchJson(url, init) {
  const response = await fetch(url, {
    ...init,
    headers: {
      'content-type': 'application/json',
      ...(init?.headers || {}),
    },
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`${url} failed with HTTP ${response.status}: ${text.slice(0, 400)}`)
  }
  return response.json()
}

function estimatePromptTokens(text) {
  const normalized = String(text || '').trim()
  if (!normalized) return 0
  const pieces = normalized.match(/[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu) || []
  return Math.max(1, Math.round(Math.max(pieces.length, normalized.length / 4)))
}

function average(values) {
  if (!values.length) return null
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function round(value) {
  return Number.isFinite(value) ? Math.round(value * 100) / 100 : null
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

function parsePositiveInt(value, label) {
  const parsed = Number.parseInt(String(value), 10)
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`--${label} must be a positive integer, got ${value}`)
  return parsed
}

function parseNonNegativeInt(value, label) {
  const parsed = Number.parseInt(String(value), 10)
  if (!Number.isInteger(parsed) || parsed < 0) throw new Error(`--${label} must be a non-negative integer, got ${value}`)
  return parsed
}

function parseOptionalPositiveInt(value, label) {
  if (value === undefined || value === null || value === '') return null
  return parsePositiveInt(value, label)
}

function sleep(ms) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, ms))
}

function shellQuote(value) {
  const text = String(value)
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(text)) return text
  return `'${text.replaceAll("'", `'"'"'`)}'`
}
