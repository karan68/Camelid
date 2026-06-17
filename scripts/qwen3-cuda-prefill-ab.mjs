#!/usr/bin/env node
// A/B parity probe for the batched GPU prefill vs the serial GPU prefill.
//
// Greedy /v1/chat/completions (ChatML, thinking disabled — the parity-locked mode).
// Run twice against the same build: once with batched prefill (default) and once with
// CAMELID_CUDA_RESIDENT_PREFILL_BATCHED=0 (serial). The serial path is the one the
// committed CUDA-resident bundles validated token-identical to llama.cpp, so
// batched == serial over a 50-token completion ⇒ batched == llama.cpp transitively.
//
// Prints JSON { prompt: completionText } to stdout for a caller to diff.
//
// Usage: node scripts/qwen3-cuda-prefill-ab.mjs --base http://127.0.0.1:8185

const args = parseArgs(process.argv.slice(2))
const base = (args.get('base') || 'http://127.0.0.1:8185').replace(/\/$/, '')
const maxTokens = Number.parseInt(args.get('max-tokens') || '50', 10)

// The three fixed bundle prompts (transitive to llama.cpp via the serial-prefill
// bundle) plus one long prompt that forces many MAX_VERIFY_K prefill chunks.
const longPrompt =
  'Read the following passage carefully and then answer. ' +
  'The quick brown fox jumps over the lazy dog near the riverbank. '.repeat(12) +
  'In one word, what animal jumps?'
const PROMPTS = [
  'What is the capital of France?',
  'Name a primary color.',
  'Say hello.',
  longPrompt,
]

async function chat(userContent) {
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      messages: [{ role: 'user', content: userContent }],
      max_tokens: maxTokens,
      temperature: 0,
      top_k: 1,
      camelid_enable_thinking: false,
    }),
  })
  if (!res.ok) throw new Error(`chat -> HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`)
  const body = await res.json()
  return body.choices[0].message.content
}

async function main() {
  const out = {}
  for (const p of PROMPTS) out[p] = await chat(p)
  console.log(JSON.stringify(out, null, 2))
}

function parseArgs(argv) {
  const m = new Map()
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) {
      const key = argv[i].slice(2)
      const val = i + 1 < argv.length && !argv[i + 1].startsWith('--') ? argv[++i] : 'true'
      m.set(key, val)
    }
  }
  return m
}

main().catch((e) => {
  console.error(e.stack || String(e))
  process.exit(1)
})
