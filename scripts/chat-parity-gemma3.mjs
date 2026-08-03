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
// TOKEN IDENTITY IS READ FROM THE ENGINE, NOT RE-DERIVED FROM TEXT.
// `token_match` compares the oracle's generated ids against camelid's OWN
// `camelid.generated_token_ids` (the same field scripts/chat-parity-llama3.mjs
// reads). It used to compare against camelid's output STRING re-encoded by
// camelid's tokenizer, which is lossy in BOTH directions on this 262k SPM vocab
// and therefore both manufactured and masked divergences:
//   * manufactured — a run of spaces re-encodes as single-space tokens (236743)
//     where the engine emitted the merged whitespace tokens (138 = two spaces,
//     140 = four), so an identical token stream reported token_match:false (the
//     Phase 4 bundle had to add camelid-raw-probe.json to unpick exactly this);
//   * masked — any two distinct id sequences that decode to the same string, or
//     that the tokenizer re-merges to the SAME canonical ids, compare EQUAL. A
//     batched-prefill defect that changes an id but not the rendered text is
//     invisible to a text round-trip, and that is the defect class this campaign
//     is built to catch.
// The old re-encode is kept as `text_reencode_token_match`, a DIAGNOSTIC only —
// it is reported, never scored, and a mismatch between it and `token_match`
// localizes a tokenizer round-trip artifact instead of hiding one.
//
// MARGINS (--top-logprobs N, default 0 = off): when armed, phase 1 asks the
// oracle for `n_probs: N` and phase 2 asks camelid for `logprobs:true,
// top_logprobs:N`, and every generated position records its top-2 gap in nats.
// Default OFF on purpose: arming it changes the request both engines answer, and
// a harness must not silently alter what it measures. Arm it explicitly for a
// receipt run — a parity result with no recorded margin cannot have its
// sensitivity estimated after the fact.
//
//   Phase 1 (ONLY llama-server running):
//     node scripts/chat-parity-gemma3.mjs --mode capture \
//       --llama http://127.0.0.1:8090 --oracle <oracle.json> \
//       [--prompts-file qa/prompt-packs/gemma3-chat-gate-pack-v1.json] [--token-counts 1,5,50] \
//       [--top-logprobs 2]
//
//   ... stop llama-server, start camelid serve (CAMELID_RUNNABLE_SERVE=1) ...
//
//   Phase 2 (ONLY camelid running):
//     node scripts/chat-parity-gemma3.mjs --mode compare \
//       --camelid http://127.0.0.1:8185 --oracle <oracle.json> \
//       --model-id "<served id>" [--row-id gemma_3_1b_it_q8_0] \
//       --display-name "Gemma 3 1B-It Q8_0" --comparator "llama.cpp 9632 ..." \
//       [--lane-label gemma3_marker_chat_greedy_metal_resident_serve] \
//       [--request-timeout-ms 3600000] [--top-logprobs 2] --out <parity.json>
//
// A pack file may be a bare JSON array, or an object with `prompts` (strings) —
// the window-edge pack additionally carries per-item metadata, which the harness
// ignores and the mutation harness reads.

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
// Which served lane produced the camelid side. The harness CANNOT observe this —
// it only speaks HTTP — so the default must not name a lane. It used to default to
// `..._runnable_serve`, which was correct only while that was gemma3's only lane;
// since the Metal resident lane became the default on a Metal host, that default
// would stamp a resident run with a runnable label. Fails to "unspecified" instead:
// a receipt that declines to name its lane is auditable, one that names the wrong
// lane is not. Pass --lane-label to record the lane you actually drove.
const laneLabel = args.get('lane-label') || 'gemma3_marker_chat_greedy_serve_lane_unspecified'
const comparatorLabel =
  args.get('comparator') || 'llama.cpp /completion (gemma3 turn markers parsed, BOS via add_special), -ngl 0 -ctk f32 -ctv f32 -fa off --no-repack'
const oraclePath = args.get('oracle')
const outPath = args.get('out') || null
const tokenCounts = (args.get('token-counts') || '1,5,50').split(',').map((s) => Number.parseInt(s.trim(), 10))
// Socket idle timeout for both engines (default 30 min); see postJson below.
const requestTimeoutMs = Number.parseInt(args.get('request-timeout-ms') || '1800000', 10)
// Top-N logprob capture on BOTH engines. 0 = off (the request shape the frozen
// bundles were captured with); >=2 records per-position top-2 margins.
const topLogprobs = Number.parseInt(args.get('top-logprobs') || '0', 10)
// gemma3 EOG ids in this row's vocab: <eos>=1, <end_of_turn>=106.
const STOP = new Set([1, 106])
let PROMPTS
if (args.get('prompts-file')) {
  const pack = JSON.parse(await readFile(args.get('prompts-file'), 'utf8'))
  const raw = Array.isArray(pack) ? pack : pack.prompts || pack.items
  // A pack entry is either a bare prompt string (every pack before v1 of the
  // window-edge pack) or an item object that carries its user turn plus the
  // metadata documenting what the item targets.
  PROMPTS = raw.map((p) => (typeof p === 'string' ? p : p.user_content))
  if (PROMPTS.some((p) => typeof p !== 'string')) {
    throw new Error('prompts-file: every entry must be a string or carry a `user_content` string')
  }
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
    ...(topLogprobs > 0 ? { n_probs: topLogprobs, post_sampling_probs: false } : {}),
  })
  return {
    text: r.content,
    tokens: r.tokens,
    margins: topLogprobs > 0 ? marginsFromLlamaProbs(r.completion_probabilities) : null,
  }
}

// Per-position top-2 gap in nats from llama-server's `completion_probabilities`
// (each entry carries the chosen token plus its `top_logprobs` list). A null
// entry means the server returned fewer than 2 candidates at that position.
function marginsFromLlamaProbs(probs) {
  if (!Array.isArray(probs)) return null
  return probs.map((p) => {
    const list = p?.top_logprobs || p?.probs || []
    const lp = list
      .map((e) => (typeof e.logprob === 'number' ? e.logprob : Math.log(e.prob ?? 0)))
      .filter((v) => Number.isFinite(v))
      .sort((a, b) => b - a)
    return lp.length >= 2 ? lp[0] - lp[1] : null
  })
}

// Same shape from an OpenAI-style `choices[0].logprobs.content[]`.
function marginsFromOpenAiLogprobs(logprobs) {
  const content = logprobs?.content
  if (!Array.isArray(content)) return null
  return content.map((step) => {
    const lp = (step.top_logprobs || [])
      .map((e) => e.logprob)
      .filter((v) => Number.isFinite(v))
      .sort((a, b) => b - a)
    return lp.length >= 2 ? lp[0] - lp[1] : null
  })
}

function minFinite(values) {
  const finite = (values || []).filter((v) => Number.isFinite(v))
  return finite.length ? Math.min(...finite) : null
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
    ...(topLogprobs > 0 ? { logprobs: true, top_logprobs: topLogprobs } : {}),
  })
  return {
    text: r.choices[0].message.content,
    promptTokens: r.usage?.prompt_tokens ?? null,
    // The ENGINE's own emitted ids. Absent only on a lane that ships no
    // diagnostics block; the harness fails loudly rather than falling back to a
    // text round-trip, because a silent fallback is how the old comparison lost
    // its power in the first place.
    generatedTokenIds: r.camelid?.generated_token_ids ?? null,
    promptTokenIds: r.camelid?.prompt_token_ids ?? null,
    margins: topLogprobs > 0 ? marginsFromOpenAiLogprobs(r.choices[0].logprobs) : null,
  }
}

function arraysEqual(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => v === b[i])
}

// Drop trailing EOG ids so the two engines' generated streams are compared on
// content only: llama-server strips EOG from `content` but returns it in
// `tokens`, and camelid's diagnostics echo may or may not carry it depending on
// the stop path. Applied IDENTICALLY to both sides, and both raw arrays are kept
// in the receipt so the stripping is auditable.
function stripTrailingStops(tokens) {
  const out = [...(tokens || [])]
  let stripped = 0
  while (out.length && STOP.has(out[out.length - 1])) {
    out.pop()
    stripped++
  }
  return { tokens: out, stripped }
}

async function capture() {
  const captured = []
  for (const userContent of PROMPTS) {
    const rendered = renderGemma3(userContent)
    const promptTokens = await referenceTokenize(rendered)
    const perCount = {}
    for (const n of tokenCounts) {
      const ref = await referenceCompletion(rendered, n)
      perCount[n] = {
        reference_text: ref.text,
        reference_tokens: ref.tokens,
        ...(ref.margins ? { reference_top2_margins_nats: ref.margins } : {}),
      }
      process.stderr.write(`captured ${JSON.stringify(userContent)} n=${n}: ${JSON.stringify(ref.text)}\n`)
    }
    captured.push({ prompt: userContent, rendered, reference_prompt_tokens: promptTokens, generations: perCount })
  }
  const oracle = {
    schema: 'camelid.gemma3.chat_oracle.v1',
    comparator: comparatorLabel,
    llama_base: llamaBase,
    token_counts: tokenCounts,
    top_logprobs: topLogprobs,
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
      if (!Array.isArray(cam.generatedTokenIds)) {
        throw new Error(
          `camelid returned no camelid.generated_token_ids for ${JSON.stringify(userContent)} n=${n}. ` +
            'This harness scores TOKEN identity on the engine\'s own ids and will not fall back to ' +
            're-encoding the output text (lossy on this vocab in both directions). Drive a lane ' +
            'that emits the diagnostics block.',
        )
      }
      // Diagnostic only — the old text round-trip, kept so a tokenizer artifact
      // is visible and attributable instead of silently scored.
      const reencoded = await encodeCamelid(cam.text, false, false)
      const refStripped = stripTrailingStops(ref.reference_tokens)
      const camStripped = stripTrailingStops(cam.generatedTokenIds)
      const textMatch = ref.reference_text === cam.text
      const tokenMatch = arraysEqual(refStripped.tokens, camStripped.tokens)
      const reencodeMatch = arraysEqual(refStripped.tokens, reencoded)
      // First index at which the two id streams part company (-1 = never).
      let firstDivergence = -1
      const lim = Math.max(refStripped.tokens.length, camStripped.tokens.length)
      for (let i = 0; i < lim; i++) {
        if (refStripped.tokens[i] !== camStripped.tokens[i]) {
          firstDivergence = i
          break
        }
      }
      perCount[n] = {
        reference_text: ref.reference_text,
        reference_tokens: ref.reference_tokens,
        reference_content_tokens: refStripped.tokens,
        camelid_text: cam.text,
        // The ENGINE's ids — what token_match is scored on.
        camelid_generated_token_ids: cam.generatedTokenIds,
        camelid_content_tokens: camStripped.tokens,
        camelid_prompt_token_ids: cam.promptTokenIds,
        // The old text->tokenizer round-trip, reported and never scored.
        camelid_text_reencoded_tokens: reencoded,
        text_reencode_token_match: reencodeMatch,
        text_reencode_artifact: tokenMatch && !reencodeMatch,
        stop_tokens_stripped: { reference: refStripped.stripped, camelid: camStripped.stripped },
        text_match: textMatch,
        token_match: tokenMatch,
        first_divergence_index: firstDivergence,
        stopped_early_at_eos: ref.reference_tokens.length < n,
        ...(oracle.top_logprobs > 0 || topLogprobs > 0
          ? {
              reference_top2_margins_nats: ref.reference_top2_margins_nats ?? null,
              camelid_top2_margins_nats: cam.margins,
              reference_min_top2_margin_nats: minFinite(ref.reference_top2_margins_nats),
              camelid_min_top2_margin_nats: minFinite(cam.margins),
            }
          : {}),
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
    // How token identity was scored, stated in the receipt so a reader never has
    // to guess whether a `token_match` came from the engine or from a re-encode.
    token_identity_source: 'camelid.generated_token_ids (engine-emitted)',
    text_reencode_scored: false,
    top_logprobs: { oracle: oracle.top_logprobs ?? 0, camelid: topLogprobs },
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
      const margin =
        typeof g.camelid_min_top2_margin_nats === 'number'
          ? ` | min top-2 margin ${g.camelid_min_top2_margin_nats.toFixed(4)} nat`
          : ''
      const artifact = g.text_reencode_artifact ? ' | text-reencode artifact (diagnostic only)' : ''
      const at = g.first_divergence_index >= 0 ? ` @${g.first_divergence_index}` : ''
      process.stderr.write(
        `  n=${n}: text ${g.text_match ? 'PASS' : 'FAIL'} | tokens ${g.token_match ? 'PASS' : `FAIL${at}`}${margin}${artifact} | camelid=${JSON.stringify(g.camelid_text)}\n`,
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
