#!/usr/bin/env node
// Two-phase Prism/Qwen3.5 chat + vision parity harness.
//
// Capture the vendor oracle while PrismML-Eng/Bonsai-demo's pinned llama-server
// is the only loaded engine, then stop it and compare Camelid in a second run:
//
//   node scripts/prism-chat-vision-parity.mjs --mode capture --base http://127.0.0.1:8114 \
//     --model Bonsai-27B-Q1_0.gguf --image target/prism-windows-test.png \
//     --out qa/evidence-bundles/.../bonsai-27b-chat-vision-oracle.json
//
//   node scripts/prism-chat-vision-parity.mjs --mode compare --base http://127.0.0.1:8185 \
//     --model Bonsai-27B --image target/prism-windows-test.png \
//     --reference qa/evidence-bundles/.../bonsai-27b-chat-vision-oracle.json \
//     --out qa/evidence-bundles/.../bonsai-27b-chat-vision-parity.json

import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const mode = required('mode')
if (mode !== 'capture' && mode !== 'compare') throw new Error('--mode must be capture or compare')

const base = required('base').replace(/\/$/, '')
const model = required('model')
const outPath = resolve(required('out'))
const referencePath = args.get('reference') ? resolve(args.get('reference')) : null
const imagePath = args.get('image') ? resolve(args.get('image')) : null
const maxTokens = Number.parseInt(args.get('max-tokens') || '16', 10)
const timeoutMs = Number.parseInt(args.get('timeout-ms') || '180000', 10)
const thinking = args.get('thinking') || 'enabled'
if (thinking !== 'enabled' && thinking !== 'disabled') {
  throw new Error('--thinking must be enabled or disabled')
}

const cases = [
  { id: 'greeting', text: 'Hello.' },
  { id: 'factual', text: 'What is the capital of France? Answer with the city only.' },
  { id: 'arithmetic', text: 'What is 2+2? Answer with the number only.' },
  { id: 'code', text: 'Write the first line of a Python fibonacci function.' },
]

if (imagePath) {
  const bytes = await readFile(imagePath)
  cases.push({
    id: 'vision_png',
    text: 'What colors are visible in this image? Answer in a short phrase.',
    image_url: `data:image/png;base64,${bytes.toString('base64')}`,
  })
}
const selectedCases = args.get('only')
  ? cases.filter((item) => item.id === args.get('only'))
  : cases
if (selectedCases.length === 0) throw new Error(`--only did not match a case: ${args.get('only')}`)

const reference = mode === 'compare'
  ? JSON.parse(await readFile(requiredPath(referencePath, '--reference'), 'utf8'))
  : null
const results = []

for (const item of selectedCases) {
  const messages = [{ role: 'user', content: item.image_url ? [
    { type: 'text', text: item.text },
    { type: 'image_url', image_url: { url: item.image_url } },
  ] : item.text }]

  // The vendor demo enables 27B thinking by default. Camelid needs the explicit
  // request flag so the comparison does not depend on the server-wide default.
  // A disabled vendor capture requires a llama-server started with the matching
  // reasoning/template flags; the harness intentionally does not invent those.
  const engineThinkingControl = mode === 'compare'
    ? { camelid_enable_thinking: thinking === 'enabled' }
    : {}
  const body = {
    model,
    messages,
    max_tokens: maxTokens,
    temperature: 0,
    top_k: 1,
    seed: 0,
    stream: false,
    ...engineThinkingControl,
  }

  const started = performance.now()
  const response = await postJson('/v1/chat/completions', body)
  const elapsedMs = performance.now() - started
  const message = response.choices?.[0]?.message || {}
  const reasoning = message.reasoning_content || ''
  const content = message.content || ''
  const generatedText = reasoning + content
  const generatedTokens = await tokenize(generatedText)
  const observed = {
    id: item.id,
    prompt: item.text,
    has_image: Boolean(item.image_url),
    generated_text: generatedText,
    generated_token_ids: generatedTokens,
    prompt_tokens: response.usage?.prompt_tokens ?? null,
    completion_tokens: response.usage?.completion_tokens ?? null,
    finish_reason: response.choices?.[0]?.finish_reason ?? null,
    response_channels: {
      reasoning_content: reasoning,
      content,
    },
    elapsed_ms: Number(elapsedMs.toFixed(3)),
  }

  if (mode === 'compare') {
    const expected = reference.results.find((candidate) => candidate.id === item.id)
    if (!expected) throw new Error(`reference is missing case ${item.id}`)
    observed.text_match = generatedText === expected.generated_text
    observed.token_match = arraysEqual(generatedTokens, expected.generated_token_ids)
    observed.prompt_token_count_match = observed.prompt_tokens === expected.prompt_tokens
    observed.first_divergent_generated_token_index = firstDivergence(
      generatedTokens,
      expected.generated_token_ids,
    )
  }
  results.push(observed)
  process.stderr.write(`${mode} ${item.id}: ${generatedTokens.length} tokens, ${elapsedMs.toFixed(1)} ms\n`)
}

const allPass = mode === 'capture' || results.every((item) =>
  item.text_match && item.token_match && item.prompt_token_count_match)
const report = {
  schema: 'camelid.prism_chat_vision_parity/v1',
  mode,
  base,
  model,
  thinking,
  sampling: { temperature: 0, top_k: 1, seed: 0, max_tokens: maxTokens },
  reference: referencePath,
  image: imagePath,
  all_pass: allPass,
  results,
}

await mkdir(dirname(outPath), { recursive: true })
await writeFile(outPath, `${JSON.stringify(report, null, 2)}\n`)
console.log(`wrote ${outPath}`)
console.log(`ALL_PASS: ${allPass}`)
if (!allPass) process.exitCode = 1

async function tokenize(text) {
  if (!text) return []
  if (mode === 'capture') {
    const response = await postJson('/tokenize', {
      content: text,
      add_special: false,
      parse_special: true,
    })
    return response.tokens
  }
  const response = await postJson('/api/models/tokenizer/encode', {
    text,
    add_special: false,
    parse_special: true,
  })
  return response.tokens
}

async function postJson(path, body) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(`${base}${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    const text = await response.text()
    if (!response.ok) throw new Error(`${path} -> HTTP ${response.status}: ${text.slice(0, 500)}`)
    return JSON.parse(text)
  } finally {
    clearTimeout(timer)
  }
}

function parseArgs(argv) {
  const parsed = new Map()
  for (let i = 0; i < argv.length; i += 2) {
    if (!argv[i]?.startsWith('--') || argv[i + 1] === undefined) {
      throw new Error(`expected --name value arguments; got ${argv[i] || '<end>'}`)
    }
    parsed.set(argv[i].slice(2), argv[i + 1])
  }
  return parsed
}

function required(name) {
  const value = args.get(name)
  if (!value) throw new Error(`--${name} is required`)
  return value
}

function requiredPath(value, name) {
  if (!value) throw new Error(`${name} is required in compare mode`)
  return value
}

function arraysEqual(left, right) {
  return Array.isArray(left) && Array.isArray(right) &&
    left.length === right.length && left.every((value, index) => value === right[index])
}

function firstDivergence(left, right) {
  const length = Math.max(left?.length || 0, right?.length || 0)
  for (let index = 0; index < length; index += 1) {
    if (left?.[index] !== right?.[index]) return index
  }
  return -1
}
