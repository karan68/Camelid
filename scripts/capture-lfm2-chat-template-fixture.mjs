#!/usr/bin/env node
// Capture llama.cpp's own application of the LFM2.5 chat template, so Camelid's
// `render_lfm2_chatml_prompt` can be gated against it offline
// (`tests/lfm2_chat_template.rs`).
//
// The renderer is hand-written Rust; the GGUF ships a Jinja template. The only
// honest way to claim the renderer is faithful is to compare it against the
// template actually applied by a reference engine, at the TOKEN level.
//
// Requires llama-server on the LFM2.5 row:
//   llama-server -m models/LFM2.5-2.6B-Q8_0.gguf --port 8090 \
//     -ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 4096
//
// Usage:
//   node scripts/capture-lfm2-chat-template-fixture.mjs \
//     --llama http://127.0.0.1:8090 --gguf-sha256 <sha> \
//     --out tests/fixtures/lfm2_parity/lfm2-chat-template.json

import { writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const llamaBase = (args.get('llama') || 'http://127.0.0.1:8090').replace(/\/$/, '')
const ggufSha = args.get('gguf-sha256') || ''
const outPath = args.get('out') || 'tests/fixtures/lfm2_parity/lfm2-chat-template.json'

if (!ggufSha) {
  console.error('ERROR: --gguf-sha256 is required')
  process.exit(2)
}

// Message shapes the bridge actually produces: a bare user turn, a system+user
// pair, and a multi-turn conversation with a prior assistant answer.
const CASES = [
  { name: 'single_user', messages: [{ role: 'user', content: 'What is the capital of France?' }] },
  {
    name: 'system_user',
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Name a primary color.' },
    ],
  },
  {
    name: 'multi_turn',
    messages: [
      { role: 'user', content: 'Say hello.' },
      { role: 'assistant', content: 'Hello!' },
      { role: 'user', content: 'Say it again.' },
    ],
  },
]

async function postJson(path, body) {
  const res = await fetch(`${llamaBase}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}: ${await res.text()}`)
  return res.json()
}

const cases = []
for (const c of CASES) {
  // llama.cpp applies the GGUF's own Jinja template.
  const applied = await postJson('/apply-template', { messages: c.messages })
  const prompt = applied.prompt ?? applied.content
  if (typeof prompt !== 'string') {
    throw new Error(`/apply-template returned no prompt for ${c.name}: ${JSON.stringify(applied)}`)
  }
  // Token-level truth, WITH specials — this is what the forward actually sees.
  const tok = await postJson('/tokenize', { content: prompt, add_special: true })
  console.error(`  ${c.name}:\n    applied = ${JSON.stringify(prompt)}\n    ids(${tok.tokens.length}) = ${JSON.stringify(tok.tokens)}`)
  cases.push({ name: c.name, messages: c.messages, applied_prompt: prompt, prompt_ids: tok.tokens })
}

const doc = {
  reference: 'llama.cpp b9632 (acd79d603) /apply-template + /tokenize(add_special=true)',
  gguf: 'LFM2.5-2.6B-Q8_0.gguf',
  gguf_sha256: ggufSha,
  note: 'applied_prompt INCLUDES the template bos_token; Camelid emits BOS via tokenizer add_special, not as renderer text, so the Rust gate compares TOKEN IDS.',
  cases,
}
await mkdir(dirname(outPath), { recursive: true })
await writeFile(outPath, `${JSON.stringify(doc, null, 2)}\n`)
console.error(`wrote ${outPath} (${cases.length} cases)`)

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
