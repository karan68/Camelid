#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const backendBase = (args.get('backend') || process.env.BACKENDINFERENCE_API_BASE || 'http://127.0.0.1:8181').replace(/\/$/, '')
const llamaBase = (args.get('llama-url') || process.env.LLAMA3_LLAMA_SERVER_URL || 'http://127.0.0.1:8183').replace(/\/$/, '')
const modelPath = resolve(args.get('model') || process.env.LLAMA3_GGUF || '$CAMELID_MODEL_DIR/Llama-3.2-1B-Instruct-Q8_0.gguf')
const modelId = args.get('model-id') || process.env.LLAMA3_MODEL_ID || 'llama3-small-q8'
const userMessage = args.get('message') ?? process.env.LLAMA3_CHAT_MESSAGE ?? 'hello'
const maxTokens = Number.parseInt(args.get('max-tokens') || process.env.LLAMA3_CHAT_MAX_TOKENS || '1', 10)
const llamaServerBin = resolve(args.get('llama-server') || process.env.LLAMA3_LLAMA_SERVER || 'target/reference/llama.cpp/build/bin/llama-server')
const llamaTokenizeBin = resolve(args.get('llama-tokenize') || process.env.LLAMA3_LLAMA_TOKENIZE || 'target/reference/llama.cpp/build/bin/llama-tokenize')
const startLlamaServer = args.has('start-llama-server') || process.env.LLAMA3_START_LLAMA_SERVER === '1'
const diagnosticsOut = args.get('diagnostics-out') || process.env.LLAMA3_CHAT_DIAGNOSTICS_OUT
const requirePromptMatch = args.has('require-prompt-match') || process.env.LLAMA3_CHAT_REQUIRE_PROMPT_MATCH === '1'
const requireGeneratedMatch = args.has('require-generated-match') || process.env.LLAMA3_CHAT_REQUIRE_GENERATED_MATCH === '1'
const waitMs = Number.parseInt(args.get('wait-ms') || process.env.LLAMA3_WAIT_MS || '120000', 10)

if (!Number.isInteger(maxTokens) || maxTokens < 1) {
  throw new Error(`--max-tokens must be a positive integer, got ${args.get('max-tokens')}`)
}
if (!Number.isInteger(waitMs) || waitMs < 1) {
  throw new Error(`--wait-ms must be a positive integer, got ${args.get('wait-ms')}`)
}

const messages = [{ role: 'user', content: userMessage }]
// Camelid's current Llama 3 support intentionally renders the compact header/eot
// form used by the checked-in tokenizer fixtures, then asks the tokenizer to add
// BOS. Use llama-server /completion with this exact prompt instead of /v1/chat,
// because newer Llama 3.2 GGUF chat templates inject a default dated system
// header that Camelid does not implement yet.
const expectedPrompt = `<|start_header_id|>user<|end_header_id|>\n\n${userMessage}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`
let child
let childSpawnError
try {
  if (startLlamaServer) {
    const url = new URL(llamaBase)
    child = spawn(llamaServerBin, [
      '--host', url.hostname,
      '--port', url.port || '8183',
      '-m', modelPath,
      '-ngl', '0',
      '-c', '512',
      '--no-warmup',
    ], { stdio: ['ignore', 'pipe', 'pipe'] })
    child.once('error', err => { childSpawnError = err })
    child.stdout.on('data', chunk => process.stderr.write(`[llama-server] ${chunk}`))
    child.stderr.on('data', chunk => process.stderr.write(`[llama-server] ${chunk}`))
  }

  await waitForJson(`${backendBase}/v1/health`, {}, 'backend', waitMs)
  try {
    await waitForJson(`${llamaBase}/health`, {}, 'llama-server', waitMs)
  } catch (err) {
    if (childSpawnError?.code === 'ENOENT') {
      throw new Error(`could not start llama-server binary ${JSON.stringify(llamaServerBin)}; pass --llama-server or set LLAMA3_LLAMA_SERVER to an executable path`)
    }
    throw err
  }

  await fetchJson(`${backendBase}/api/models/load`, {
    method: 'POST',
    body: JSON.stringify({ path: modelPath, id: modelId }),
  })

  const referencePromptTokens = await tokenizeExpectedPrompt()
  const chatPayload = {
    model: modelId,
    messages,
    max_tokens: maxTokens,
    stream: false,
    temperature: 0,
  }
  const llamaCompletion = await fetchJson(`${llamaBase}/completion`, {
    method: 'POST',
    body: JSON.stringify({
      prompt: expectedPrompt,
      n_predict: maxTokens,
      temperature: 0,
      cache_prompt: false,
      n_probs: 20,
    }),
  })
  const llamaText = llamaCompletion.content ?? ''
  const llamaLogprobContent = llamaCompletion.completion_probabilities ?? []
  const llamaGeneratedTokens = llamaLogprobContent
    .map(item => Number.isInteger(item?.id) ? item.id : null)
    .filter(token => token !== null)
  const llamaTopLogprobs = llamaLogprobContent.flatMap(item => item?.top_logprobs ?? [])
  const diagnosticTokenIds = uniqueTokenIds([
    ...llamaGeneratedTokens,
    ...llamaTopLogprobs.map(item => item?.id),
  ]).slice(0, 16)

  const backendChat = await fetchJson(`${backendBase}/v1/chat/completions`, {
    method: 'POST',
    body: JSON.stringify({
      ...chatPayload,
      backendinference_logit_token_ids: diagnosticTokenIds,
    }),
  })

  const backendPromptTokens = backendChat.backendinference?.prompt_token_ids || []
  const backendGeneratedTokens = backendChat.backendinference?.generated_token_ids || []
  const promptMatch = JSON.stringify(backendPromptTokens) === JSON.stringify(referencePromptTokens)
  const generatedTokensMatch = JSON.stringify(backendGeneratedTokens) === JSON.stringify(llamaGeneratedTokens)
  const backendText = backendChat.choices?.[0]?.message?.content ?? ''
  const textMatch = backendText === llamaText

  const report = {
    backend: backendBase,
    llama_server: llamaBase,
    model: modelPath,
    model_id: modelId,
    message: userMessage,
    expected_prompt: expectedPrompt,
    prompt_tokens_match: promptMatch,
    generated_tokens_match: generatedTokensMatch,
    generated_text_match: textMatch,
    first_generated_text_diff_index: firstStringDifference(backendText, llamaText),
    backend_prompt_tokens: backendPromptTokens,
    reference_prompt_tokens: referencePromptTokens,
    backend_generated_tokens: backendGeneratedTokens,
    llama_generated_tokens: llamaGeneratedTokens,
    llama_top_logprobs: llamaTopLogprobs,
    backend_diagnostic_token_ids: diagnosticTokenIds,
    backend_text: backendText,
    llama_text: llamaText,
    backend_usage: backendChat.usage,
    llama_usage: llamaCompletion.timings,
    backendinference: backendChat.backendinference,
  }

  console.log(`backend=${backendBase}`)
  console.log(`llama_server=${llamaBase}`)
  console.log(`model=${modelPath}`)
  console.log(`message=${JSON.stringify(userMessage)}`)
  console.log(`expected_prompt=${JSON.stringify(expectedPrompt)}`)
  console.log(`backend_prompt_tokens=${JSON.stringify(backendPromptTokens)}`)
  console.log(`reference_prompt_tokens=${JSON.stringify(referencePromptTokens)}`)
  console.log(`prompt_tokens_match=${promptMatch}`)
  console.log(`backend_generated_tokens=${JSON.stringify(backendGeneratedTokens)}`)
  console.log(`llama_generated_tokens=${JSON.stringify(llamaGeneratedTokens)}`)
  console.log(`generated_tokens_match=${generatedTokensMatch}`)
  console.log(`backend_text=${JSON.stringify(backendText)}`)
  console.log(`llama_text=${JSON.stringify(llamaText)}`)
  console.log(`generated_text_match=${textMatch}`)
  console.log(`backend_usage=${JSON.stringify(backendChat.usage)}`)
  console.log(`llama_usage=${JSON.stringify(llamaCompletion.timings)}`)

  if (diagnosticsOut) {
    const diagnosticsPath = resolve(diagnosticsOut)
    await mkdir(dirname(diagnosticsPath), { recursive: true })
    await writeFile(diagnosticsPath, `${JSON.stringify(report, null, 2)}\n`)
    console.log(`diagnostics_out=${diagnosticsPath}`)
  }

  if (requirePromptMatch && !promptMatch) process.exitCode = 1
  if (requireGeneratedMatch && !generatedTokensMatch) process.exitCode = 1
} finally {
  if (child) child.kill('SIGTERM')
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

async function tokenizeExpectedPrompt() {
  const { stdout } = await run(llamaTokenizeBin, [
    '-m', modelPath,
    '--ids',
    '--log-disable',
    '-p', expectedPrompt,
  ])
  return JSON.parse(stdout.trim())
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

async function waitForJson(url, options, label, waitMs) {
  const deadline = Date.now() + waitMs
  let lastError
  while (Date.now() < deadline) {
    try {
      return await fetchJson(url, options)
    } catch (err) {
      lastError = err
      await new Promise(resolve => setTimeout(resolve, 500))
    }
  }
  throw new Error(`${label} did not become reachable at ${url} within ${waitMs} ms: ${lastError?.message}`)
}

async function run(command, commandArgs) {
  return new Promise((resolvePromise, reject) => {
    const childProcess = spawn(command, commandArgs, { stdio: ['ignore', 'pipe', 'pipe'] })
    let stdout = ''
    let stderr = ''
    childProcess.stdout.on('data', chunk => { stdout += chunk })
    childProcess.stderr.on('data', chunk => { stderr += chunk })
    childProcess.once('error', reject)
    childProcess.once('close', code => {
      if (code === 0) {
        resolvePromise({ stdout, stderr })
      } else {
        reject(new Error(`${command} exited ${code}: ${stderr || stdout}`))
      }
    })
  })
}

function uniqueTokenIds(ids) {
  const out = []
  for (const id of ids) {
    if (Number.isInteger(id) && !out.includes(id)) out.push(id)
  }
  return out
}

function firstStringDifference(left, right) {
  const max = Math.max(left.length, right.length)
  for (let i = 0; i < max; i += 1) {
    if (left[i] !== right[i]) return i
  }
  return -1
}
