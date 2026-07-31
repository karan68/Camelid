#!/usr/bin/env node
// gemma3 chat-parity harness — TWO-PHASE (MUSTER M-A1), modeled on
// scripts/chat-parity-qwen3-twophase.mjs.
//
// Parity contract: greedy chat via a camelid serve lane (originally the runnable
// serve lane, CAMELID_RUNNABLE_SERVE=1; the served lane actually exercised is
// recorded by --lane-label), prompt-token + generated-token + generated-text
// parity at 1/5/50 against the pinned llama.cpp reference. The rendered prompt
// carries NO BOS string (byte-identical to the oracle's /apply-template output,
// locked by qa/prompt-packs/gemma3-chat-template-shapes-v1.json); BOTH engines
// add BOS at token level (llama-server /completion + /tokenize add_special, and
// camelid's runnable bridge encode(add_special=true)). Prompt-token parity is
// CROSS-ENGINE: llama /tokenize captured in phase 1, camelid
// /api/models/tokenizer/encode compared in phase 2 (the runnable lane has no
// dense-diagnostics prompt echo).
//
//   Phase 1 (ONLY llama-server running):
//     node scripts/chat-parity-gemma3.mjs --mode capture \
//       --llama http://127.0.0.1:8090 --oracle <oracle.json> \
//       [--prompts-file qa/prompt-packs/gemma3-chat-gate-pack-v1.json] [--token-counts 1,5,50]
//
//   ... stop llama-server, start camelid serve (CAMELID_RUNNABLE_SERVE=1) ...
//
//   Phase 2 (ONLY camelid running):
//     node scripts/chat-parity-gemma3.mjs --mode compare \
//       --camelid http://127.0.0.1:8185 --oracle <oracle.json> \
//       --model-id "<served id>" [--row-id gemma_3_1b_it_q8_0] \
//       --display-name "Gemma 3 1B-It Q8_0" --comparator "llama.cpp 9632 ..." \
//       [--lane-label gemma3_marker_chat_greedy_metal_resident_serve] \
//       [--request-timeout-ms 3600000] --out <parity.json>

import { writeFile, readFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import http from 'node:http'

const args = parseArgs(process.argv.slice(2))
const mode = args.get('mode')
const llamaBase = (args.get('llama') || 'http://127.0.0.1:8090').replace(/\/$/, '')
const camelidBase = (args.get('camelid') || process.env.CAMELID_API_BASE || 'http://127.0.0.1:8185').replace(/\/$/, '')
const modelId = args.get('model-id') || 'Gemma 3 1B It'
// The compatibility row id is `gemma_3_1b_it_q8_0` (src/api/mod.rs); the older
// `gemma3_1b_it_q8_0` spelling matches no row, so a receipt generated without an
// explicit --row-id used to carry an id that resolved to nothing.
const rowId = args.get('row-id') || 'gemma_3_1b_it_q8_0'
const displayName = args.get('display-name') || 'Gemma 3 1B-It Q8_0'
// Which served lane produced the camelid side. Defaults to the runnable serve lane
// this harness was written for; pass --lane-label when driving another served lane
// so the emitted receipt does not mislabel the lane it actually certified.
const laneLabel = args.get('lane-label') || 'gemma3_marker_chat_greedy_runnable_serve'
const comparatorLabel =
  args.get('comparator') || 'llama.cpp /completion (gemma3 turn markers parsed, BOS via add_special), -ngl 0 -ctk f32 -ctv f32 -fa off --no-repack'
const oraclePath = args.get('oracle')
const outPath = args.get('out') || null
const tokenCounts = (args.get('token-counts') || '1,5,50').split(',').map((s) => Number.parseInt(s.trim(), 10))
// Socket idle timeout for both engines (default 30 min); see postJson below.
const requestTimeoutMs = Number.parseInt(args.get('request-timeout-ms') || '1800000', 10)
// gemma3 EOG ids in this row's vocab: <eos>=1, <end_of_turn>=106.
const STOP = new Set([1, 106])
let PROMPTS
if (args.get('prompts-file')) {
  const pack = JSON.parse(await readFile(args.get('prompts-file'), 'utf8'))
  PROMPTS = Array.isArray(pack) ? pack : pack.prompts
} else if (args.get('prompts-json')) {
  PROMPTS = JSON.parse(args.get('prompts-json'))
} else {
  PROMPTS = ['What is the capital of France?', 'Say hello in one short sentence.', 'What is 2+2?']
}

// Single-user-turn gemma3 render — must byte-match render_gemma3_prompt in
// src/api/mod.rs (locked by the shapes pack) for [{role:"user"}] input.
function renderGemma3(userContent) {
  return `<start_of_turn>user\n${userContent.trim()}<end_of_turn>\n<start_of_turn>model\n`
}

// node:http, NOT the global fetch: undici's ~5-minute headersTimeout fires while a
// slow lane holds the connection open during a long prefill (a >=512-token prompt on
// the f32 CPU runnable lane takes far longer than that), and the client then reports
// UND_ERR_HEADERS_TIMEOUT for a request the server is still serving correctly. Same
// fix, same flag name, as scripts/raw-decode-parity.mjs.
function postJson(base, path, body) {
  return new Promise((resolve, reject) => {
    const u = new URL(`${base}${path}`)
    const data = JSON.stringify(body)
    const req = http.request(
      {
        hostname: u.hostname,
        port: u.port,
        path: u.pathname + u.search,
        method: 'POST',
        headers: { 'content-type': 'application/json', 'content-length': Buffer.byteLength(data) },
      },
      (res) => {
        let buf = ''
        res.setEncoding('utf8')
        res.on('data', (c) => (buf += c))
        res.on('end', () => {
          if (res.statusCode < 200 || res.statusCode >= 300) {
            reject(new Error(`${base}${path} -> HTTP ${res.statusCode}: ${buf.slice(0, 300)}`))
          } else {
            try {
              resolve(JSON.parse(buf))
            } catch {
              reject(new Error(`${base}${path} -> non-JSON response: ${buf.slice(0, 200)}`))
            }
          }
        })
      },
    )
    req.setTimeout(requestTimeoutMs, () => req.destroy(new Error(`${base}${path} -> idle timeout after ${requestTimeoutMs}ms`)))
    req.on('error', reject)
    req.write(data)
    req.end()
  })
}

async function referenceCompletion(promptText, nPredict) {
  const r = await postJson(llamaBase, '/completion', {
    prompt: promptText,
    n_predict: nPredict,
    temperature: 0,
    top_k: 1,
    seed: 0,
    cache_prompt: false,
    samplers: ['top_k'],
    return_tokens: true,
  })
  return { text: r.content, tokens: r.tokens }
}

async function referenceTokenize(promptText) {
  const r = await postJson(llamaBase, '/tokenize', { content: promptText, add_special: true })
  return r.tokens
}

async function encodeCamelid(text, addSpecial, parseSpecial) {
  const r = await postJson(camelidBase, '/api/models/tokenizer/encode', {
    text,
    add_special: addSpecial,
    parse_special: parseSpecial,
  })
  return r.tokens
}

async function camelidChat(userContent, maxTokens) {
  const r = await postJson(camelidBase, '/v1/chat/completions', {
    model: modelId,
    messages: [{ role: 'user', content: userContent }],
    max_tokens: maxTokens,
    temperature: 0,
    top_k: 1,
    seed: 0,
    stream: false,
  })
  return { text: r.choices[0].message.content, promptTokens: r.usage?.prompt_tokens ?? null }
}

function arraysEqual(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => v === b[i])
}

async function capture() {
  const captured = []
  for (const userContent of PROMPTS) {
    const rendered = renderGemma3(userContent)
    const promptTokens = await referenceTokenize(rendered)
    const perCount = {}
    for (const n of tokenCounts) {
      const ref = await referenceCompletion(rendered, n)
      perCount[n] = { reference_text: ref.text, reference_tokens: ref.tokens }
      process.stderr.write(`captured ${JSON.stringify(userContent)} n=${n}: ${JSON.stringify(ref.text)}\n`)
    }
    captured.push({ prompt: userContent, rendered, reference_prompt_tokens: promptTokens, generations: perCount })
  }
  const oracle = {
    schema: 'camelid.gemma3.chat_oracle.v1',
    comparator: comparatorLabel,
    llama_base: llamaBase,
    token_counts: tokenCounts,
    prompts: PROMPTS,
    captured,
  }
  await mkdir(dirname(oraclePath), { recursive: true })
  await writeFile(oraclePath, JSON.stringify(oracle, null, 2))
  process.stderr.write(`\nwrote oracle ${oraclePath}\n`)
}

async function compare() {
  const oracle = JSON.parse(await readFile(oraclePath, 'utf8'))
  const results = []
  let allPass = true
  for (const cap of oracle.captured) {
    const userContent = cap.prompt
    const camelidPromptTokens = await encodeCamelid(cap.rendered, true, true)
    const promptMatch = arraysEqual(cap.reference_prompt_tokens, camelidPromptTokens)

    const perCount = {}
    let usagePromptTokens = null
    for (const n of oracle.token_counts) {
      const ref = cap.generations[n]
      const cam = await camelidChat(userContent, n)
      usagePromptTokens = cam.promptTokens
      const camTokens = await encodeCamelid(cam.text, false, false)
      const refContentTokens = [...ref.reference_tokens]
      while (refContentTokens.length && STOP.has(refContentTokens[refContentTokens.length - 1])) refContentTokens.pop()
      const textMatch = ref.reference_text === cam.text
      const tokenMatch = arraysEqual(refContentTokens, camTokens)
      perCount[n] = {
        reference_text: ref.reference_text,
        reference_tokens: ref.reference_tokens,
        reference_content_tokens: refContentTokens,
        camelid_text: cam.text,
        camelid_content_tokens: camTokens,
        text_match: textMatch,
        token_match: tokenMatch,
        stopped_early_at_eos: ref.reference_tokens.length < n,
      }
      if (!textMatch || !tokenMatch) allPass = false
    }
    if (!promptMatch) allPass = false
    results.push({
      prompt: userContent,
      rendered: cap.rendered,
      reference_prompt_tokens: cap.reference_prompt_tokens,
      camelid_prompt_tokens: camelidPromptTokens,
      camelid_usage_prompt_tokens: usagePromptTokens,
      prompt_token_match: promptMatch,
      generations: perCount,
    })
  }

  const report = {
    schema: 'camelid.gemma3.chat_parity.v1',
    row_id: rowId,
    display_name: displayName,
    mode: laneLabel,
    capture_method: 'two_phase_oracle',
    comparator: oracle.comparator || comparatorLabel,
    camelid_base: camelidBase,
    llama_base: oracle.llama_base,
    token_counts: oracle.token_counts,
    all_pass: allPass,
    results,
  }
  const json = JSON.stringify(report, null, 2)
  if (outPath) {
    await mkdir(dirname(outPath), { recursive: true })
    await writeFile(outPath, json)
    process.stderr.write(`wrote ${outPath}\n`)
  }
  for (const r of results) {
    process.stderr.write(`\n=== ${JSON.stringify(r.prompt)} ===\n`)
    process.stderr.write(`  prompt-token parity (cross-engine): ${r.prompt_token_match ? 'PASS' : 'FAIL'}\n`)
    for (const n of oracle.token_counts) {
      const g = r.generations[n]
      process.stderr.write(
        `  n=${n}: text ${g.text_match ? 'PASS' : 'FAIL'} | tokens ${g.token_match ? 'PASS' : 'FAIL'} | camelid=${JSON.stringify(g.camelid_text)}\n`,
      )
    }
  }
  process.stderr.write(`\nALL_PASS: ${allPass}\n`)
  process.stdout.write(json)
  process.exitCode = allPass ? 0 : 1
}

function parseArgs(argv) {
  const map = new Map()
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a.startsWith('--')) {
      const key = a.slice(2)
      const next = argv[i + 1]
      if (next === undefined || next.startsWith('--')) map.set(key, 'true')
      else {
        map.set(key, next)
        i++
      }
    }
  }
  return map
}

if (!oraclePath) {
  process.stderr.write('error: --oracle <path> is required\n')
  process.exitCode = 2
} else if (mode === 'capture') {
  await capture()
} else if (mode === 'compare') {
  await compare()
} else {
  process.stderr.write('error: --mode capture|compare is required\n')
  process.exitCode = 2
}
