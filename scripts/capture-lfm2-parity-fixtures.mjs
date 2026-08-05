#!/usr/bin/env node
// Capture the frozen llama.cpp reference fixtures for the LFM2 runnable-lane
// parity gate (`tests/lfm2_parity.rs`).
//
// Both sides of that gate are fed the SAME prompt token ids, so the tokenizer is
// deliberately not a variable: this script asks llama-server to tokenize each
// prompt, then to greedily continue it, and records both id lists verbatim.
//
// The reference must be the CPU, full-precision-KV configuration so the fixture
// is a property of the graph rather than of a kernel/offload choice:
//
//   llama-server -m models/LFM2.5-2.6B-Q8_0.gguf --port 8090 \
//     -ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 4096
//
// Usage:
//   node scripts/capture-lfm2-parity-fixtures.mjs \
//     --llama http://127.0.0.1:8090 \
//     --gguf LFM2.5-2.6B-Q8_0.gguf --gguf-sha256 <sha> \
//     --reference "llama.cpp b9632 (acd79d603) ..." \
//     --max-new 24 --out tests/fixtures/lfm2_parity/lfm2.json

import { writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const llamaBase = (args.get('llama') || 'http://127.0.0.1:8090').replace(/\/$/, '')
const gguf = args.get('gguf') || 'LFM2.5-2.6B-Q8_0.gguf'
const ggufSha = args.get('gguf-sha256') || ''
const reference = args.get('reference') || 'llama.cpp /completion (CPU, -ngl 0, f32 KV, greedy)'
const maxNew = Number.parseInt(args.get('max-new') || '24', 10)
const outPath = args.get('out') || 'tests/fixtures/lfm2_parity/lfm2.json'

// Raw completion prompts — no chat template, so this gate measures the forward
// graph and nothing else. Deliberately short and factual so greedy decode is
// stable rather than wandering.
const PROMPTS = JSON.parse(
  args.get('prompts-json') ||
    JSON.stringify([
      'The capital of France is',
      'Water boils at a temperature of',
      'One two three four',
      'The quick brown fox',
    ]),
)

if (!ggufSha) {
  console.error('ERROR: --gguf-sha256 is required (the receipt pins the exact bytes)')
  process.exit(2)
}

async function postJson(path, body) {
  const res = await fetch(`${llamaBase}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}: ${await res.text()}`)
  return res.json()
}

const fixtures = []
for (const prompt_text of PROMPTS) {
  // 1) Tokenize through the reference so prompt ids are the reference's own.
  const tok = await postJson('/tokenize', { content: prompt_text })
  const prompt_ids = tok.tokens
  if (!Array.isArray(prompt_ids) || prompt_ids.length === 0) {
    throw new Error(`tokenize returned nothing for ${JSON.stringify(prompt_text)}`)
  }

  // 2) Greedy continuation FROM THE IDS (not the text), with sampling fully
  //    pinned to argmax and prompt caching off so each fixture is independent.
  const comp = await postJson('/completion', {
    prompt: prompt_ids,
    n_predict: maxNew,
    temperature: 0,
    top_k: 1,
    seed: 0,
    cache_prompt: false,
    return_tokens: true,
  })
  const greedy_ids = comp.tokens ?? []
  if (greedy_ids.length === 0) {
    throw new Error(`no tokens returned for ${JSON.stringify(prompt_text)}; is return_tokens supported?`)
  }

  console.error(
    `  ${JSON.stringify(prompt_text)}\n    prompt_ids(${prompt_ids.length}) = ${JSON.stringify(prompt_ids)}\n    greedy_ids(${greedy_ids.length}) = ${JSON.stringify(greedy_ids)}\n    text = ${JSON.stringify(comp.content)}`,
  )
  fixtures.push({ prompt_text, prompt_ids, greedy_ids, reference_text: comp.content })
}

const doc = { reference, gguf, gguf_sha256: ggufSha, max_new: maxNew, fixtures }
await mkdir(dirname(outPath), { recursive: true })
await writeFile(outPath, `${JSON.stringify(doc, null, 2)}\n`)
console.error(`wrote ${outPath} (${fixtures.length} fixtures)`)

function parseArgs(argv) {
  const map = new Map()
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i]
    if (a.startsWith('--')) {
      const key = a.slice(2)
      const next = argv[i + 1]
      if (next && !next.startsWith('--')) {
        map.set(key, next)
        i += 1
      } else {
        map.set(key, 'true')
      }
    }
  }
  return map
}
