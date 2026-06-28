#!/usr/bin/env node
// Template-agnostic raw-prompt token-level DECODE parity harness (K-quant lane).
//
// Certifies the camelid GPU-resident CUDA decode kernels (q4k_gemv / q6k_gemv)
// directly, with NO chat template in the way: for each raw prompt it greedily
// generates from both engines and checks token-AND-text parity.
//
//   camelid: POST /v1/completions   (raw prompt, GPU-resident decode)
//   llama:   POST /completion        (raw prompt, -ngl 0 CPU reference)
//
// This avoids the chat-template / BOS confounds AND the camelid_dense_diagnostics
// f32 path (which 503s on wire-only K-quant tensors). The reference's generated
// tokens may carry a trailing stop token stripped from the detokenized text, so we
// compare against the reference's CONTENT tokens (trailing stops removed) and the
// camelid text re-encoded by the camelid tokenizer — same discipline as
// chat-parity-qwen3.mjs.
//
// First prompt also asserts BOS alignment implicitly: if the engines disagreed on
// BOS the first generated token would differ and the run FAILs loudly.
//
// Usage:
//   node scripts/raw-decode-parity.mjs --camelid http://127.0.0.1:8185 \
//     --llama http://127.0.0.1:8090 --model-id "<id>" --row-id <id> \
//     --display-name "..." --comparator "..." --stop "128009,128001" \
//     --prompts-file qa/speed/prompts.json --out <bundle>/parity.json

import { writeFile, mkdir, readFile } from 'node:fs/promises'
import { dirname } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const camelidBase = (args.get('camelid') || 'http://127.0.0.1:8185').replace(/\/$/, '')
const llamaBase = (args.get('llama') || 'http://127.0.0.1:8090').replace(/\/$/, '')
const modelId = args.get('model-id') || 'model'
const rowId = args.get('row-id') || 'row'
const displayName = args.get('display-name') || modelId
const comparatorLabel = args.get('comparator') || 'llama.cpp /completion'
const outPath = args.get('out') || null
const tokenCounts = (args.get('token-counts') || '1,5,50').split(',').map((s) => Number.parseInt(s.trim(), 10))
const STOP = new Set((args.get('stop') || '128009,128001,128000').split(',').map((s) => Number.parseInt(s.trim(), 10)))

let PROMPTS
if (args.get('prompts-file')) {
  const raw = JSON.parse(await readFile(args.get('prompts-file'), 'utf8'))
  PROMPTS = Array.isArray(raw) ? raw.map((p) => (typeof p === 'string' ? p : p.prompt ?? p.text)) : raw.prompts
} else {
  PROMPTS = JSON.parse(
    args.get('prompts-json') ||
      JSON.stringify([
        'The capital of France is',
        'Q: What is 2+2? A:',
        'Once upon a time,',
        'def fibonacci(n):',
      ]),
  )
}
PROMPTS = PROMPTS.filter(Boolean)

async function postJson(base, path, body) {
  const res = await fetch(`${base}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${base}${path} -> HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`)
  return res.json()
}

async function encodeCamelid(text) {
  const r = await postJson(camelidBase, '/api/models/tokenizer/encode', { text, add_special: false, parse_special: false })
  return r.tokens
}

async function camelidCompletion(prompt, maxTokens) {
  const r = await postJson(camelidBase, '/v1/completions', {
    model: modelId,
    prompt,
    max_tokens: maxTokens,
    temperature: 0,
    top_k: 1,
    seed: 0,
    stream: false,
  })
  return r.choices[0].text
}

async function referenceCompletion(prompt, nPredict) {
  const r = await postJson(llamaBase, '/completion', {
    prompt,
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

function arraysEqual(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => v === b[i])
}

async function main() {
  const results = []
  let allPass = true
  for (const prompt of PROMPTS) {
    const perCount = {}
    for (const n of tokenCounts) {
      const ref = await referenceCompletion(prompt, n)
      const camText = await camelidCompletion(prompt, n)
      const camTokens = await encodeCamelid(camText)
      const refContentTokens = [...ref.tokens]
      while (refContentTokens.length && STOP.has(refContentTokens[refContentTokens.length - 1])) refContentTokens.pop()
      const textMatch = ref.text === camText
      const tokenMatch = arraysEqual(refContentTokens, camTokens)
      perCount[n] = {
        reference_text: ref.text,
        reference_tokens: ref.tokens,
        reference_content_tokens: refContentTokens,
        camelid_text: camText,
        camelid_content_tokens: camTokens,
        text_match: textMatch,
        token_match: tokenMatch,
        stopped_early_at_eos: ref.tokens.length < n,
      }
      if (!textMatch || !tokenMatch) allPass = false
    }
    results.push({ prompt, generations: perCount })
  }

  const report = {
    schema: 'camelid.raw_decode_parity.v1',
    variant: 'kquant_gpu_resident',
    row_id: rowId,
    display_name: displayName,
    mode: 'raw_completion_greedy',
    comparator: comparatorLabel,
    proof_chain: 'camelid GPU-resident CUDA decode (q4k_gemv/q6k_gemv) == llama.cpp, raw-prompt token+text parity. No chat template, no f32 diagnostics. K-quant has no camelid CPU decode path yet (Phase 2), so llama.cpp is the direct reference.',
    camelid_base: camelidBase,
    llama_base: llamaBase,
    token_counts: tokenCounts,
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
    for (const n of tokenCounts) {
      const g = r.generations[n]
      process.stderr.write(`  n=${n}: text ${g.text_match ? 'PASS' : 'FAIL'} | tokens ${g.token_match ? 'PASS' : 'FAIL'} | cam=${JSON.stringify(g.camelid_text)}\n`)
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

await main()
