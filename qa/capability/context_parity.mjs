#!/usr/bin/env node
// Capability harness for `context.full_length` / `context.rope_scaling` (oracle class D)
// on Windows CPU. Drives a long prompt through Camelid, emits a native sealed
// camelid.parity-receipt/v1, and asserts bit-exact generation parity vs llama.cpp with
// the receipt's prompt token IDs pinned.
//
// SAFETY (conductor section 9 predict-and-abort): this harness projects f32 CPU KV
// bytes before starting the model and checks the exact rendered token count again before
// accepting evidence. The user-supplied KV coefficient must describe the selected row.
//
// Dense/raw example:
//   node qa/capability/context_parity.mjs --gguf models/Llama-3.2-1B-Instruct-Q8_0.gguf \
//     --row llama-3.2-1b-instruct-q8_0 --target-tokens 16000 \
//     --kv-bytes-per-token 65536 --llama-ctx 16384
//
// Runnable-only rows use their native chat bridge. `target-tokens` is only a character
// heuristic, so an exact context bucket must pin the rendered count:
//   node qa/capability/context_parity.mjs --request-mode chat \
//     --gguf models/LFM2.5-2.6B-Q8_0.gguf --row lfm2_5_2_6b_q8_0 \
//     --target-tokens 681 --expect-actual-tokens 512 --kv-bytes-per-token 32768 \
//     --verify-mode reference-only
import { execFile, spawn } from 'node:child_process'
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { promisify } from 'node:util'

import {
  actualTokenGate,
  buildGenerationRequest,
  buildLongPrompt,
  buildMessages,
  oracleConfig,
  parseArgs,
  positiveInt,
  requireNativeReceipt,
  requestMode,
  verifierArgs,
} from './context_parity_lib.mjs'

const execFileAsync = promisify(execFile)
const args = parseArgs(process.argv.slice(2))
const need = key => args.get(key) ?? (() => { throw new Error(`--${key} required`) })()
const gguf = resolve(need('gguf'))
const row = need('row')
const label = args.get('label') || row
const mode = requestMode(args)
const targetTokens = positiveInt(need('target-tokens'), '--target-tokens')
const kvBytesPerToken = positiveInt(need('kv-bytes-per-token'), '--kv-bytes-per-token')
const llamaCtx = positiveInt(args.get('llama-ctx') || String(targetTokens + 64), '--llama-ctx')
const maxGen = positiveInt(args.get('max-gen') || '8', '--max-gen')
const camelidExe = args.get('camelid-exe') || 'target/release/camelid.exe'
const llamaServer = args.get('llama-server') || 'llama-server'
const camelidPort = positiveInt(args.get('camelid-port') || '8231', '--camelid-port')
const llamaPort = positiveInt(args.get('llama-port') || '8233', '--llama-port')
const out = resolve(args.get('out') || `qa/capability/ctx_out/${row}`)
const verifyMode = args.get('verify-mode') || (mode === 'chat' ? 'reference-only' : 'full')
if (!['full', 'reference-only', 'self-only'].includes(verifyMode)) {
  throw new Error(`--verify-mode must be full, reference-only, or self-only, got ${JSON.stringify(verifyMode)}`)
}
if (mode === 'chat' && verifyMode !== 'reference-only') {
  throw new Error('chat mode requires --verify-mode reference-only until runnable chat self-replay exists')
}
if (mode === 'chat' && !args.has('expect-actual-tokens')) {
  throw new Error('chat mode requires --expect-actual-tokens so a heuristic target cannot promote the wrong context bucket')
}
const oracle = oracleConfig(args)
const safeBudget = positiveInt(
  args.get('safe-kv-budget-bytes') || String(Math.floor(3.5 * 1024 ** 3)),
  '--safe-kv-budget-bytes',
)
const camelidBase = `http://127.0.0.1:${camelidPort}`
const env = { ...process.env, CUDA_VISIBLE_DEVICES: '-1' }

const preflightProjectedKv = targetTokens * kvBytesPerToken
const gib = value => `${(value / 1024 ** 3).toFixed(2)} GiB`
console.log(
  `row=${row} request_mode=${mode} target_tokens=${targetTokens} ` +
  `kv/token=${kvBytesPerToken}B projected_KV=${gib(preflightProjectedKv)} safe_budget=${gib(safeBudget)}`,
)
if (preflightProjectedKv > safeBudget) {
  console.log(
    `ABORT (predict-and-abort): projected KV ${gib(preflightProjectedKv)} exceeds ` +
    `safe budget ${gib(safeBudget)} on this host - not requested.`,
  )
  process.exitCode = 2
  process.exit()
}

// Heavy long-prompt POSTs go through curl: CPU prefill can outrun undici's headers timeout.
async function curlJson(url, body) {
  const dir = await mkdtemp(join(tmpdir(), 'ctxreq-'))
  const reqPath = join(dir, 'req.json')
  await writeFile(reqPath, JSON.stringify(body))
  const { stdout } = await execFileAsync(
    'curl',
    ['-s', '--max-time', '1800', '-H', 'content-type: application/json', '-d', `@${reqPath}`, url],
    { maxBuffer: 64 * 1024 * 1024 },
  )
  return stdout ? JSON.parse(stdout) : null
}

async function waitUrl(url, ms = 120000) {
  const deadline = Date.now() + ms
  let last
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url)
      if (response.ok) return
      last = new Error(`HTTP ${response.status}`)
    } catch (error) {
      last = error
    }
    await new Promise(resolvePromise => setTimeout(resolvePromise, 500))
  }
  throw new Error(`not reachable at ${url}: ${last?.message}`)
}

const prompt = buildLongPrompt(targetTokens)
const messages = buildMessages(prompt, args.get('system-message'))
const endpoint = mode === 'chat' ? '/v1/chat/completions' : '/v1/completions'
const requestBody = buildGenerationRequest({ mode, prompt, messages, maxGen })

await mkdir(out, { recursive: true })
const camelid = spawn(
  camelidExe,
  [
    'serve', '--addr', `127.0.0.1:${camelidPort}`, '--model', gguf,
    '--gpu', 'off', '--deterministic', '--no-open',
  ],
  { stdio: 'ignore', env },
)
let parityReceipt
let promptTokens
let generatedText
let gate
try {
  await waitUrl(`${camelidBase}/api/models/current`)
  const response = await curlJson(`${camelidBase}${endpoint}`, requestBody)
  const native = requireNativeReceipt(response, mode)
  parityReceipt = native.receipt
  const promptIds = native.promptIds
  promptTokens = promptIds.length
  gate = actualTokenGate(promptTokens, args.get('expect-actual-tokens'))
  const actualProjectedKv = promptTokens * kvBytesPerToken
  if (actualProjectedKv > safeBudget) {
    throw new Error(
      `actual rendered prompt projects ${actualProjectedKv} KV bytes, above safe budget ${safeBudget}`,
    )
  }
  if (gate.enabled && !gate.pass) {
    const failed = {
      row,
      label,
      gguf,
      request_mode: mode,
      target_tokens: targetTokens,
      actual_prompt_tokens: promptTokens,
      actual_prompt_tokens_gate: gate,
      preflight_projected_kv_bytes: preflightProjectedKv,
      projected_kv_bytes: actualProjectedKv,
      generation_parity_pass: false,
      failure: `expected exactly ${gate.expected} rendered prompt tokens, got ${gate.actual}`,
    }
    await writeFile(resolve(out, 'summary.json'), `${JSON.stringify(failed, null, 2)}\n`)
    throw new Error(failed.failure)
  }
  generatedText = native.generatedText
} finally {
  camelid.kill('SIGTERM')
  await new Promise(resolvePromise => setTimeout(resolvePromise, 800))
}

const receiptPath = resolve(out, 'parity-receipt.json')
await writeFile(receiptPath, `${JSON.stringify(parityReceipt, null, 2)}\n`)
const ctx = Math.max(llamaCtx, promptTokens + maxGen + 8)
console.log(
  `actual_prompt_tokens=${promptTokens} positions_exceed_8192=${promptTokens > 8192} llama_ctx=${ctx}`,
)

let verifyOut = ''
const verifyCommandArgs = verifierArgs({
  receiptPath,
  gguf,
  llamaServer,
  llamaCtx: ctx,
  llamaPort: llamaPort + 10,
  verifyMode,
  oracle,
})
try {
  const { stdout } = await execFileAsync(camelidExe, verifyCommandArgs, {
    env,
    maxBuffer: 32 * 1024 * 1024,
  })
  verifyOut = stdout
} catch (error) {
  verifyOut = (error.stdout || '') + (error.stderr || '') + String(error)
}
const pass = verifyMode === 'reference-only'
  ? /PASS reference-rerun/.test(verifyOut)
  : verifyMode === 'self-only'
    ? /PASS camelid-rerun/.test(verifyOut)
    : /RECEIPT VERIFIED/.test(verifyOut) && /PASS reference-rerun/.test(verifyOut)
await writeFile(resolve(out, 'verify.log'), verifyOut)
const summary = {
  row,
  label,
  gguf,
  request_mode: mode,
  endpoint,
  target_tokens: targetTokens,
  actual_prompt_tokens: promptTokens,
  actual_prompt_tokens_gate: gate,
  positions_exceed_8192: promptTokens > 8192,
  preflight_projected_kv_bytes: preflightProjectedKv,
  projected_kv_bytes: promptTokens * kvBytesPerToken,
  safe_kv_budget_bytes: safeBudget,
  verify_mode: verifyMode,
  verification_scope: verifyMode === 'reference-only'
    ? 'llama.cpp reference rerun only; Camelid self-replay intentionally skipped'
    : verifyMode,
  llama_oracle: {
    gpu_layers: 0,
    cache_type_k: oracle.cacheTypeK,
    cache_type_v: oracle.cacheTypeV,
    flash_attention: oracle.flashAttn,
    repack: oracle.repack,
  },
  generation_parity_pass: pass,
  evidence: (verifyOut.match(/PASS reference-rerun:.*/)?.[0] || '').trim(),
  generated_text: generatedText,
}
await writeFile(resolve(out, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`)
console.log(`generation_parity_pass=${pass}  (${summary.evidence})`)
console.log(`receipt=${receiptPath}`)
if (!pass) {
  console.log('FAIL: generation parity did not verify')
  process.exitCode = 1
} else {
  console.log('OK')
}
