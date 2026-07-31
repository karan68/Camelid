#!/usr/bin/env node
// Build qa/prompt-packs/gemma3-window-edge-pack-v1.json — a windowed-context pack
// with real power against WINDOW-EDGE errors, for the gemma3 long-prompt TTFT
// campaign (Phase 1).
//
// WHY A SECOND WINDOWED PACK EXISTS
//
// qa/prompt-packs/gemma3-windowed-context-pack-v1.json is a strong test of
// GLOBAL-layer reach and almost no test of the 22 sliding layers' window edge:
//   * its three prompts are built from a pool of 30 unique sentences cycled up to
//     8x (211 sentences, 30 distinct, at N=2403), so any boundary error moves the
//     window edge across VERBATIM-DUPLICATE content and the same information is
//     still reachable from the duplicate;
//   * its one load-bearing fact ("Willow") sits at character 48, ~token 12 — inside
//     the window of no query past position 523, i.e. reachable at N=2403 only
//     through the 4 global layers, which no window mutation touches;
//   * its lengths (606/1205/2403) are 30/53/35 mod 64 and 94/53/35 mod 128 — no
//     length is a tile multiple, a tile edge, or a window edge.
//
// WHAT THIS PACK DOES INSTEAD
//
//  1. Non-repeating body text. Every filler sentence in every prompt is unique
//     across the whole pack (asserted before writing), so no wrong mask can
//     recover a fact from a duplicate.
//  2. Anchors placed by TOKEN POSITION relative to the query position q (the last
//     prompt token, whose logits produce the first generated token), at q-510
//     (comfortably inside), q-511 (the OLDEST position a 512-window sees), q-512
//     (the FIRST position outside) and q-513. The q-511/q-512 pair differ by a
//     single word, so an off-by-one is a visible answer change and not a near-tie.
//  3. Multi-depth items: several anchors at distinct depths in one prompt, so a
//     single continuation exercises several edges at once.
//  4. A length ladder that straddles the window (511/512/513, 1023/1024/1025), the
//     64-wide NR0/NR1 attention tiles (63/64/65, 127/128/129, 255/256/257) and the
//     n_pad = next_multiple_of(128) boundary (2400/2432/2433).
//
// HONEST LIMIT, RECORDED IN THE PACK ITSELF: a prompt whose LAST position has not
// saturated the window cannot exercise the window bound at all — below 512
// positions `filled.saturating_sub(w)` is 0 for every w. The short ladder items
// are there for the Tier B TILE geometry, not for the mask, and each says so in
// its own `intent`. See `window_power` on every item.
//
// USAGE (tokenizer source is the PINNED oracle, alone on the machine):
//   llama-server -m <gemma-3-1b-it-Q8_0.gguf> -c 4096 --port 8090 --host 127.0.0.1 \
//     -ngl 0 -ctk f32 -ctv f32 -fa off --no-repack
//   node scripts/build-gemma3-window-edge-pack.mjs \
//     --tokenizer http://127.0.0.1:8090 \
//     --out qa/prompt-packs/gemma3-window-edge-pack-v1.json
//
// The pack carries each item's `prompt_token_ids` (from the oracle's /tokenize with
// add_special=true, i.e. exactly what both engines see) so the in-src mutation
// harness needs no tokenizer, and so the measured positions in the pack are
// checkable by anyone without re-running this script.

import { writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import http from 'node:http'

const args = parseArgs(process.argv.slice(2))
const tokenizerBase = (args.get('tokenizer') || 'http://127.0.0.1:8090').replace(/\/$/, '')
const outPath = args.get('out') || 'qa/prompt-packs/gemma3-window-edge-pack-v1.json'
const WINDOW = Number.parseInt(args.get('window') || '512', 10)

function parseArgs(argv) {
  const map = new Map()
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (!a.startsWith('--')) continue
    const key = a.slice(2)
    const next = argv[i + 1]
    if (next === undefined || next.startsWith('--')) map.set(key, 'true')
    else {
      map.set(key, next)
      i++
    }
  }
  return map
}

function postJson(base, path, body) {
  return new Promise((resolve, reject) => {
    const u = new URL(`${base}${path}`)
    const data = JSON.stringify(body)
    const req = http.request(
      {
        hostname: u.hostname,
        port: u.port,
        path: u.pathname,
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
              reject(new Error(`${base}${path} -> non-JSON: ${buf.slice(0, 200)}`))
            }
          }
        })
      },
    )
    req.setTimeout(600000, () => req.destroy(new Error('tokenize idle timeout')))
    req.on('error', reject)
    req.write(data)
    req.end()
  })
}

const tokenizeCache = new Map()
async function tokenize(text) {
  if (tokenizeCache.has(text)) return tokenizeCache.get(text)
  const r = await postJson(tokenizerBase, '/tokenize', { content: text, add_special: true })
  tokenizeCache.set(text, r.tokens)
  return r.tokens
}

// Must byte-match render_gemma3_prompt in src/api/mod.rs for a single user turn
// (locked by qa/prompt-packs/gemma3-chat-template-shapes-v1.json), and match
// scripts/chat-parity-gemma3.mjs's renderGemma3.
function render(userContent) {
  return `<start_of_turn>user\n${userContent.trim()}<end_of_turn>\n<start_of_turn>model\n`
}

// ---------------------------------------------------------------------------
// Content generation. Plain English, no emoji, no repeated spaces, no rare
// vocabulary (the SPM merge-order discipline the gemma3 gate pack states), but
// UNIQUE: every sentence is generated once and never repeated anywhere in the
// pack. Uniqueness is asserted before the pack is written.
// ---------------------------------------------------------------------------

const SUBJECT = [
  'surveyor', 'archivist', 'ferryman', 'glassblower', 'cartwright', 'beekeeper', 'shepherd',
  'ropemaker', 'stonecutter', 'weaver', 'cooper', 'thatcher', 'miller', 'tanner', 'saddler',
  'chandler', 'fletcher', 'brewer', 'potter', 'joiner', 'mason', 'carter', 'drover', 'reeve',
  'warden', 'steward', 'huntsman', 'forester', 'gardener', 'clerk',
]
const VERB = [
  'measured', 'recorded', 'counted', 'sorted', 'labelled', 'weighed', 'stacked', 'wrapped',
  'listed', 'checked', 'moved', 'stored', 'cleaned', 'repaired', 'painted', 'carried',
  'delivered', 'collected', 'sealed', 'opened',
]
const NOUN = [
  'crates', 'ledgers', 'lanterns', 'baskets', 'anchors', 'saddles', 'kettles', 'chisels',
  'barrels', 'pulleys', 'hinges', 'mirrors', 'shutters', 'benches', 'satchels', 'buckets',
  'candles', 'ropes', 'nets', 'ladders', 'spades', 'hammers', 'files', 'clamps', 'awnings',
  'planks', 'tiles', 'bricks', 'jars', 'flasks', 'spools', 'reels', 'hooks', 'chains',
  'wedges', 'mallets', 'sieves', 'trowels', 'scythes', 'harrows',
]
const PLACE = [
  'north cellar', 'old mill', 'lower yard', 'stone quay', 'river gate', 'east barn',
  'copper works', 'salt house', 'tide pool', 'clock tower', 'weigh station', 'rope walk',
  'timber shed', 'coal wharf', 'apple loft', 'wool store', 'iron bridge', 'grain silo',
  'boat house', 'kiln field', 'oak avenue', 'chalk pit', 'ferry stair', 'lamp room',
  'drying green', 'fish market', 'well court', 'peat store', 'brick lane', 'reed bank',
]
const TIME = [
  'before dawn', 'at first light', 'late in the morning', 'just after noon', 'in the afternoon',
  'towards evening', 'after supper', 'at dusk', 'on the second shift', 'during the low tide',
  'while the rain held off', 'once the wind dropped',
]

// Distinct anchor vocabulary. Every gate name and every colour is used at most
// once in the entire pack, so an answer can never be recovered from a duplicate.
const GATE = [
  'Alder', 'Bramble', 'Cinder', 'Dovecote', 'Elmshaw', 'Fernwick', 'Gorsely', 'Harrow',
  'Ivybank', 'Junewell', 'Kelpstone', 'Larkfield', 'Marlow', 'Netherby', 'Oakhanger',
  'Pinfold', 'Quarrend', 'Rushmere', 'Saltgate', 'Thornby', 'Undercliff', 'Verney',
  'Wexford', 'Yarrow', 'Zennor', 'Ashcombe', 'Bexley', 'Cranmoor', 'Dunfield', 'Eastmark',
  'Fallowby', 'Greenhow', 'Hollinsea', 'Inglewood', 'Jarrowmere', 'Kingsthorn', 'Lyndhurst',
  'Merribourne', 'Northgate', 'Oldstead', 'Priorswood', 'Quenby',
]
const COLOUR = [
  'crimson', 'amber', 'indigo', 'emerald', 'scarlet', 'violet', 'copper', 'silver',
  'golden', 'russet', 'azure', 'olive', 'maroon', 'saffron', 'teal', 'ivory',
  'charcoal', 'bronze', 'coral', 'plum', 'slate', 'ochre', 'lilac', 'sable',
  'cobalt', 'jade', 'sepia', 'garnet', 'pewter', 'chestnut',
  'magenta', 'turquoise', 'burgundy', 'lavender', 'mustard', 'seagreen', 'oxblood', 'flaxen',
]
// Single-word fine-tuning ladder, appended one at a time to a trailing note
// sentence. Each is a common noun so the tokenizer keeps it short, and each is
// used at most once per prompt.
const PAD = [
  'oak', 'elm', 'birch', 'cedar', 'rowan', 'holly', 'hazel', 'maple', 'willow', 'alder',
  'beech', 'poplar', 'spruce', 'larch', 'yew', 'ash', 'pine', 'fir', 'lime', 'plane',
  'thorn', 'briar', 'furze', 'sedge', 'reed', 'rush', 'moss', 'fern', 'ivy', 'vine',
  'clover', 'thistle', 'nettle', 'sorrel', 'yarrow', 'teasel', 'mallow', 'comfrey',
  'borage', 'chicory', 'burdock', 'tansy', 'agrimony', 'betony', 'vervain', 'hyssop',
]

// An anchor's answer word must occur EXACTLY ONCE in its prompt, or its position
// cannot be measured and a mask error could be recovered from the other occurrence.
// Filter the colour and gate pools against every other vocabulary the filler draws
// from, rather than hand-auditing them (a hand audit already missed "copper", which
// also appears in the PLACE list as "copper works").
const FILLER_WORDS = new Set(
  [...SUBJECT, ...VERB, ...NOUN, ...PLACE, ...TIME, ...PAD]
    .flatMap((p) => p.split(/\s+/))
    .map((w) => w.toLowerCase()),
)
for (const pool of [COLOUR, GATE]) {
  const clashes = pool.filter((w) => FILLER_WORDS.has(w.toLowerCase()))
  for (const c of clashes) pool.splice(pool.indexOf(c), 1)
  if (clashes.length) process.stderr.write(`dropped clashing anchor words: ${clashes.join(', ')}\n`)
}

// Deterministic unique-sentence stream. Coprime strides across five lists give
// 30*20*40*30*12 distinct tuples before any repeat is possible; a global counter
// plus a used-set guarantees uniqueness in fact, not just in principle.
const usedSentences = new Set()
let sentenceCounter = 0
function nextSentence() {
  for (let attempt = 0; attempt < 100000; attempt++) {
    const n = sentenceCounter++
    const s = SUBJECT[(n * 7) % SUBJECT.length]
    const v = VERB[(n * 3) % VERB.length]
    const o = NOUN[(n * 11) % NOUN.length]
    const p = PLACE[(n * 13) % PLACE.length]
    const t = TIME[(n * 5) % TIME.length]
    const k = n % 4
    let sentence
    if (k === 0) sentence = `The ${s} ${v} ${(n % 47) + 3} ${o} at the ${p} ${t}.`
    else if (k === 1) sentence = `${t.charAt(0).toUpperCase()}${t.slice(1)} the ${s} ${v} ${o} near the ${p}.`
    else if (k === 2) sentence = `Entry ${n + 100} shows that the ${s} ${v} ${(n % 31) + 2} ${o} in the ${p}.`
    else sentence = `At the ${p} the ${s} ${v} ${o} and noted the ${(n % 23) + 1} that remained.`
    if (usedSentences.has(sentence)) continue
    usedSentences.add(sentence)
    return sentence
  }
  throw new Error('sentence pool exhausted')
}

let gateCursor = 0
let colourCursor = 0
function nextGate() {
  if (gateCursor >= GATE.length) throw new Error('gate pool exhausted')
  return GATE[gateCursor++]
}
function nextColour() {
  if (colourCursor >= COLOUR.length) throw new Error('colour pool exhausted')
  return COLOUR[colourCursor++]
}

// The anchor sentence. The ANSWER WORD is last, so anchoring it by position puts
// the load-bearing token exactly where the design says.
function anchorSentence(gate, colour) {
  return `The seal at gate ${gate} is ${colour}.`
}

// Fine-tuning sentence. `slot` is a STABLE id (item + knob), never a call counter:
// the solver rebuilds the prompt many times and the text must be a pure function
// of the knob values or it will not converge. The slot also makes every pad
// sentence unique across the pack, so the uniqueness audit covers them too.
//
// `count` is ALWAYS >= 1 so the sentence is either absent (count 0, structure
// changes) or present with a continuous 1-token-per-word ladder. PAD is calibrated
// against the live tokenizer at startup and any word that does not cost exactly one
// token is dropped, which is what makes the ladder exact.
let PAD_CALIBRATED = PAD
function padSentence(count, slot) {
  if (count <= 0) return ''
  let h = 0
  for (const ch of slot) h = (h * 31 + ch.charCodeAt(0)) % 100003
  const words = []
  for (let i = 0; i < count; i++) words.push(PAD_CALIBRATED[(h + i) % PAD_CALIBRATED.length])
  return `Note ${slot}: nearby grow ${words.join(' ')}.`
}

// Keep only pad words that cost EXACTLY one token when appended to a pad sentence.
// Measured against the live tokenizer, never assumed: SPM merges are context
// sensitive and a two-token pad word would let the fine solver step over its target.
async function calibratePad() {
  const base = await tokenize('Note cal: nearby grow oak.')
  const good = []
  for (const w of PAD) {
    if (w === 'oak') continue
    const t = await tokenize(`Note cal: nearby grow oak ${w}.`)
    if (t.length - base.length === 1) good.push(w)
  }
  good.unshift('oak')
  if (good.length < 20) throw new Error(`pad calibration kept only ${good.length} words`)
  PAD_CALIBRATED = good
  process.stderr.write(`pad calibration: ${good.length}/${PAD.length} words cost exactly 1 token\n`)
}

// ---------------------------------------------------------------------------
// Position solving. Two knobs, applied in order:
//   * `tailPad` — words appended AFTER the anchor. Moves the anchor's distance
//     from the end (and the total length) together.
//   * `headPad` — words appended BEFORE the anchor. Moves the total length only.
// So solve the anchor offset first, then the total length. Both are solved
// EMPIRICALLY (build -> tokenize -> measure -> adjust), never predicted: SPM
// merges are context sensitive and a predicted token count is a guess.
// ---------------------------------------------------------------------------

async function locate(tokens, needleTokens) {
  const hits = []
  outer: for (let i = 0; i + needleTokens.length <= tokens.length; i++) {
    for (let j = 0; j < needleTokens.length; j++) {
      if (tokens[i + j] !== needleTokens[j]) continue outer
    }
    hits.push(i)
  }
  return hits
}

// Build one item. `anchors` is a list of { offset, gate, colour } sorted by
// DESCENDING offset (deepest first). Each anchor's answer token must end up at
// q - offset. `targetTokens` is the exact rendered prompt length, or null for
// "whatever it comes out as".
async function buildItem(spec) {
  const { id, intent, anchors, targetTokens, question } = spec
  // Layout: [head sentences] [head pad] [anchor0] [gap0 sentences] [gap0 pad]
  //         [anchor1] [gap1 sentences] [gap1 pad] ... [question]
  // `sorted` is DEEPEST-first, so gap i sits between anchor i and anchor i+1.
  const sorted = [...anchors].sort((a, b) => b.offset - a.offset)
  const head = []
  const gapSentences = sorted.map(() => [])
  const gapPad = sorted.map(() => 1)
  let headPad = 1

  const compose = () => {
    const parts = []
    if (head.length) parts.push(head.join(' '))
    if (headPad > 0) parts.push(padSentence(headPad, `${id}-h`))
    for (let i = 0; i < sorted.length; i++) {
      parts.push(anchorSentence(sorted[i].gate, sorted[i].colour))
      if (gapSentences[i].length) parts.push(gapSentences[i].join(' '))
      if (gapPad[i] > 0) parts.push(padSentence(gapPad[i], `${id}-g${i}`))
    }
    parts.push(question)
    return parts.filter(Boolean).join(' ')
  }

  const colourNeedles = new Map()
  const measure = async () => {
    const content = compose()
    const renderedText = render(content)
    const tokens = await tokenize(renderedText)
    const q = tokens.length - 1
    const offsets = []
    for (const a of sorted) {
      if (!colourNeedles.has(a.colour)) {
        // ` colour` tokenized standalone carries BOS; drop it, then trim from the
        // front until the remaining run occurs EXACTLY once in the prompt.
        colourNeedles.set(a.colour, (await tokenize(` ${a.colour}`)).slice(1))
      }
      let needleTail = colourNeedles.get(a.colour)
      let hits = await locate(tokens, needleTail)
      // 0 hits means the leading-space token did not materialize in context, so
      // trim the needle. >1 hits is a HARD error: an answer word that occurs twice
      // is exactly the defect this pack exists to avoid.
      while (hits.length === 0 && needleTail.length > 1) {
        needleTail = needleTail.slice(1)
        hits = await locate(tokens, needleTail)
      }
      if (hits.length !== 1) {
        throw new Error(`${id}: anchor colour ${a.colour} is not uniquely locatable (${hits.length} hits)`)
      }
      offsets.push(q - (hits[0] + needleTail.length - 1))
    }
    return { content, rendered: renderedText, tokens, q, offsets }
  }

  // Coarse-then-fine solver, driven entirely by MEASUREMENT — nothing about the
  // token count is predicted, because SPM merges are context sensitive.
  //   coarse: whole sentences (~15 tokens each) until the shortfall fits the pad
  //           ladder;
  //   fine:   pad words, calibrated to cost EXACTLY one token each, so the last
  //           step is exact rather than an iteration that can straddle the target.
  // `need(m)` is always "tokens still to ADD at this knob".
  const AVG = 15
  const solve = async (label, need, addSentence, dropSentence, setPad) => {
    const LADDER = PAD_CALIBRATED.length - 4
    for (let round = 0; round < 40; round++) {
      setPad(1)
      let m = await measure()
      let n = need(m)
      // Coarse: land the shortfall inside [padFloor, padFloor + LADDER).
      for (let i = 0; i < 400 && (n < 0 || n > LADDER); i++) {
        if (n < 0) {
          if (!dropSentence()) break
        } else {
          const k = Math.max(1, Math.round((n - LADDER / 2) / AVG))
          for (let j = 0; j < k; j++) addSentence()
        }
        m = await measure()
        n = need(m)
      }
      if (n < 0) {
        // Cannot shrink further at this knob; give the caller a clear failure
        // rather than a silently wrong offset.
        throw new Error(`${id}: ${label} overshoots by ${-n} tokens with no content left to drop`)
      }
      // Fine: one calibrated token per pad word.
      let pad = 1 + n
      for (let i = 0; i < 12; i++) {
        setPad(pad)
        const mm = await measure()
        const nn = need(mm)
        if (nn === 0) return
        pad += nn
        if (pad < 1) break
      }
      // The ladder straddled the target (a pad word that did not cost 1 token in
      // this context). Perturb the coarse level and try again.
      addSentence()
    }
    throw new Error(`${id}: ${label} did not converge`)
  }

  // 1. Anchor offsets, SHALLOWEST first: gap i only moves anchors 0..i, so fixing
  //    a shallow anchor first and a deeper one after never disturbs the fix.
  for (let i = sorted.length - 1; i >= 0; i--) {
    await solve(
      `anchor ${i} offset`,
      (m) => sorted[i].offset - m.offsets[i], // >0: anchor too shallow, add after it
      () => gapSentences[i].push(nextSentence()),
      () => (gapSentences[i].length ? (gapSentences[i].pop(), true) : false),
      (p) => (gapPad[i] = p),
    )
  }

  // 2. Total length, via content BEFORE every anchor — offsets are untouched.
  if (targetTokens !== null) {
    await solve(
      'total length',
      (m) => targetTokens - m.tokens.length,
      () => head.push(nextSentence()),
      () => (head.length ? (head.pop(), true) : false),
      (p) => (headPad = p),
    )
  }

  const m = await measure()
  // Re-verify everything after the final compose: a late length fix must not have
  // moved an anchor.
  for (let i = 0; i < sorted.length; i++) {
    if (m.offsets[i] !== sorted[i].offset) {
      throw new Error(`${id}: anchor ${i} landed at q-${m.offsets[i]}, wanted q-${sorted[i].offset}`)
    }
  }
  if (targetTokens !== null && m.tokens.length !== targetTokens) {
    throw new Error(`${id}: landed at ${m.tokens.length} tokens, wanted ${targetTokens}`)
  }
  return {
    id,
    intent,
    user_content: m.content,
    rendered: m.rendered,
    prompt_tokens: m.tokens.length,
    prompt_token_ids: m.tokens,
    query_position: m.q,
    anchors: sorted.map((a, i) => ({
      gate: a.gate,
      colour: a.colour,
      target_offset_from_query: a.offset,
      measured_offset_from_query: m.offsets[i],
      absolute_position: m.q - m.offsets[i],
      inside_window_at_final_query: m.offsets[i] <= WINDOW - 1,
      role: a.role,
    })),
    ...spec.extra,
  }
}

async function main() {
  await calibratePad()
  const items = []
  const mutationSubset = []

  // Question text is SHARED by design (the items must differ in anchor placement,
  // not in what is asked), so its sentences are registered and excluded from the
  // body-uniqueness audit rather than being allowed to fail it.
  const questionFragments = new Set()
  const registerQuestion = (q) => {
    for (const frag of q.split(/(?<=\.|\?)\s+/)) questionFragments.add(frag.trim())
    return q
  }
  const askColour = (gate) =>
    registerQuestion(`Question: what colour is the seal at gate ${gate}? Answer in one short sentence.`)
  const askGeneric = () =>
    registerQuestion('Question: name one item mentioned above. Answer in one short sentence.')

  // ---- Group W: the window edge, one anchor each, same length, same question
  // shape. These four differ ONLY in where the answer word sits relative to q.
  const edgeSpecs = [
    [510, 'inside', 'Anchor one position INSIDE the oldest visible slot: a -1 window error still sees it, a -2 does not.'],
    [511, 'edge_inside', 'Anchor at the OLDEST position a 512-window sees. window-1 drops it; the correct mask keeps it. This is the single highest-power item for a lower-bound off-by-one.'],
    [512, 'edge_outside', 'Anchor at the FIRST position outside the window. window+1 admits it; the correct mask must not. Pairs with edge_inside: the two differ by one token of placement and by one word of answer.'],
    [513, 'outside', 'Anchor two positions outside: a control for edge_outside — window+1 must NOT reach it, so a catch here means the bound is off by more than one.'],
  ]
  for (const [offset, role, why] of edgeSpecs) {
    const gate = nextGate()
    const colour = nextColour()
    const item = await buildItem({
      id: `w-edge-q-${offset}`,
      intent: `Window-edge probe at q-${offset} (window ${WINDOW}). ${why}`,
      anchors: [{ offset, gate, colour, role }],
      targetTokens: 1024,
      question: askColour(gate),
      extra: {
        window_power: 'high',
        expected_answer_word: colour,
        mutation_subset: true,
      },
    })
    items.push(item)
    mutationSubset.push(item.id)
  }

  // ---- Group M: several depths in ONE prompt, so one continuation exercises
  // several edges.
  {
    const gates = [nextGate(), nextGate(), nextGate(), nextGate()]
    const colours = [nextColour(), nextColour(), nextColour(), nextColour()]
    const item = await buildItem({
      id: 'w-multi-1536',
      intent:
        'Four anchors at four depths in one 1536-token prompt: q-1024 (deep — sliding layers reach it only by stacking), q-552 (outside the window), q-511 (the oldest position inside it) and q-64 (inside the final 64-wide attention tile). One greedy continuation exercises all four; the question asks about the q-511 anchor, whose in/out status flips under a one-position window error. Consecutive anchors are kept >=40 tokens apart because an anchor sentence is ~9 tokens long — the exact q-511/q-512 pair is carried by the dedicated single-anchor items, which is the only way to place them one token apart.',
      anchors: [
        { offset: 1024, gate: gates[0], colour: colours[0], role: 'deep_stacked' },
        { offset: 552, gate: gates[1], colour: colours[1], role: 'outside' },
        { offset: 511, gate: gates[2], colour: colours[2], role: 'edge_inside' },
        { offset: 64, gate: gates[3], colour: colours[3], role: 'final_tile' },
      ],
      targetTokens: 1536,
      question: askColour(gates[2]),
      extra: { window_power: 'high', expected_answer_word: colours[2], mutation_subset: true },
    })
    items.push(item)
    mutationSubset.push(item.id)
  }
  {
    const gates = [nextGate(), nextGate(), nextGate(), nextGate(), nextGate()]
    const colours = [nextColour(), nextColour(), nextColour(), nextColour(), nextColour()]
    const item = await buildItem({
      id: 'w-multi-2400',
      intent:
        'The campaign headline length (2400 prompt tokens) with five anchors: q-2048, q-1024, q-552, q-511 and q-64. At this length 1889 of the 2400 query positions have a saturated window, so a lower-bound error perturbs the majority of the sequence; the anchors make the perturbation legible in the answer rather than only in the logits.',
      anchors: [
        { offset: 2048, gate: gates[0], colour: colours[0], role: 'deep_stacked' },
        { offset: 1024, gate: gates[1], colour: colours[1], role: 'deep_stacked' },
        { offset: 552, gate: gates[2], colour: colours[2], role: 'outside' },
        { offset: 511, gate: gates[3], colour: colours[3], role: 'edge_inside' },
        { offset: 64, gate: gates[4], colour: colours[4], role: 'final_tile' },
      ],
      targetTokens: 2400,
      question: askColour(gates[3]),
      extra: { window_power: 'high', expected_answer_word: colours[3], mutation_subset: true },
    })
    items.push(item)
    mutationSubset.push(item.id)
  }

  // ---- Group L: the length ladder. Each carries ONE anchor at q-511 where the
  // length allows it, so the ladder is not just a length sweep.
  const ladder = [
    [63, 'NR0/NR1 = 64 attention-tile edge (one below).', 'none'],
    [64, 'Exactly one 64-wide tile.', 'none'],
    [65, 'One token into the second tile — the ragged-row guard.', 'none'],
    [127, 'One below the n_pad = next_multiple_of(128) boundary.', 'none'],
    [128, 'Exactly one n_pad block.', 'none'],
    [129, 'One token into the second n_pad block.', 'none'],
    [255, 'One below a 256 boundary (attn_qb floor).', 'none'],
    [256, 'The attn_qb minimum.', 'none'],
    [257, 'One past it.', 'none'],
    [511, 'One position BELOW window saturation: the last length at which the window bound is provably a no-op at the final query.', 'none'],
    [512, 'Exactly window-saturated at the final query: window_start is 0 for w=512 and 1 for w=511, so this is the FIRST length at which a -1 error is observable during prefill, and it is observable at exactly one position. NO CONTENT ANCHOR IS POSSIBLE HERE: at N=512 the position that moves in or out is 0, which is BOS — the window edge lands on the template prefix, not on text. That is not a defect of the pack, it is a property of the length, and it makes this item a probe of the attention-sink tokens rather than of a planted fact.', 'none'],
    [513, 'One position past saturation: two query positions clip during prefill, and the boundary token is the <start_of_turn> marker at position 1 — again a template token, not content. The hardest real catch in the pack: the smallest perturbation a lower-bound error can produce.', 'none'],
    [1023, 'One below the second window multiple.', 'q511'],
    [1024, 'Exactly two windows.', 'q511'],
    [1025, 'One past two windows.', 'q511'],
    [2400, 'The campaign headline length; n_pad = 2432.', 'q511'],
    [2432, 'Exactly n_pad — next_multiple_of(128) is the identity here.', 'q511'],
    [2433, 'One past, so n_pad jumps to 2560.', 'q511'],
  ]
  for (const [n, why, anchorMode] of ladder) {
    let anchors = []
    let question = ''
    let colour = null
    if (anchorMode === 'q511') {
      const gate = nextGate()
      colour = nextColour()
      anchors = [{ offset: 511, gate, colour, role: 'edge_inside' }]
      question = askColour(gate)
    } else {
      // No anchor: a plain unique-filler passage plus a question about content it
      // does contain, sized to the target. Anchoring at q-511 is impossible below
      // 512 tokens and pointless where the window never saturates.
      anchors = []
      question = askGeneric()
    }
    const power =
      n <= WINDOW - 1
        ? 'none'
        : n <= WINDOW + 8
          ? 'minimal'
          : 'high'
    const note =
      power === 'none'
        ? `NO WINDOW POWER BY CONSTRUCTION: with ${n} prompt tokens the final query is at position ${n - 1} < ${WINDOW}, so filled.saturating_sub(w) is 0 for every w and the bound cannot be exercised until generation pushes past it. This item exists for the Tier B TILE geometry.`
        : power === 'minimal'
          ? `MINIMAL WINDOW POWER: only ${n - WINDOW + 1} query position(s) clip during prefill. Detecting a one-position error here is the strictest test in the pack, and a miss here is expected to be recoverable by generating past the boundary.`
          : `${n - WINDOW + 1} of ${n} query positions clip during prefill.`
    const item = await buildItem({
      id: `w-len-${n}`,
      intent: `Length-ladder item at exactly ${n} rendered prompt tokens. ${why} ${note}`,
      anchors,
      targetTokens: n,
      question,
      extra: {
        window_power: power,
        ...(colour ? { expected_answer_word: colour } : {}),
        mutation_subset: [513, 1024, 2400].includes(n),
      },
    })
    items.push(item)
    if (item.mutation_subset) mutationSubset.push(item.id)
  }

  // ---- Uniqueness audit: no filler sentence may appear twice anywhere in the
  // pack, and no anchor gate or colour may be reused. This is the property the
  // old pack lacked, so it is asserted rather than assumed.
  const seen = new Map()
  for (const item of items) {
    for (const s of item.user_content.split(/(?<=\.)\s+/)) {
      const t = s.trim()
      if (!t || questionFragments.has(t)) continue
      if (seen.has(t)) {
        throw new Error(`duplicate sentence across items ${seen.get(t)} and ${item.id}: ${JSON.stringify(t)}`)
      }
      seen.set(t, item.id)
    }
  }
  const gatesUsed = new Set()
  const coloursUsed = new Set()
  for (const item of items) {
    for (const a of item.anchors) {
      if (gatesUsed.has(a.gate)) throw new Error(`gate ${a.gate} reused`)
      if (coloursUsed.has(a.colour)) throw new Error(`colour ${a.colour} reused`)
      gatesUsed.add(a.gate)
      coloursUsed.add(a.colour)
    }
  }

  const pack = {
    schema: 'camelid.gemma3.window_edge_prompt_pack/v1',
    pack_id: 'gemma3-window-edge-pack-v1',
    generated_by: 'scripts/build-gemma3-window-edge-pack.mjs',
    window_tokens: WINDOW,
    tokenizer:
      'llama.cpp llama-server /tokenize, add_special=true, pinned build 9632 (acd79d603), CPU backend — the same tokenization both engines are compared on',
    purpose:
      'Window-edge parity pack for the exact row gemma-3-1b-it-Q8_0.gguf, built for the gemma3 long-prompt TTFT campaign (batched prefill, Tier A/B). It replaces qa/prompt-packs/gemma3-windowed-context-pack-v1.json as the WINDOW-BOUND probe: that pack cycles ~30 unique sentences up to 8x and puts its only load-bearing fact at ~token 12, reachable through the 4 GLOBAL layers alone, so it tests global reach and not the 22 sliding layers\' window edge. Here every filler sentence is unique across the whole pack (asserted at build time), every anchor gate and colour is used at most once, and each anchor\'s ANSWER WORD is placed at a measured token offset from the query position q (the last prompt token). The four q-510/511/512/513 items differ only in that placement. Lengths straddle the 512 window, the 64-wide NR0/NR1 attention tiles and the n_pad=next_multiple_of(128) boundary.',
    conventions: {
      query_position: 'q = index of the LAST rendered prompt token; its logits produce the first generated token.',
      window_semantics:
        'A sliding layer at position p attends [max(0, p+1-w) ..= p] — the window INCLUDES the current position (src/window_ref.rs is the pinned reference). So q-511 is the OLDEST visible position and q-512 is the FIRST invisible one at w=512.',
      offsets: 'anchors[].measured_offset_from_query is q minus the absolute position of the anchor answer word\'s LAST token.',
      limits:
        'A sliding-window model can still recall a fact from outside a single layer\'s window by STACKING: 22 sliding layers give an effective receptive field of ~22x511 positions. This pack therefore does NOT claim that a correct implementation fails to recall a q-512 fact. Its power comes from OUTPUT SENSITIVITY to the mask — the generated token stream must change when the bound moves — scored against a reference run, not against a notion of the right answer.',
    },
    mutation_subset: mutationSubset,
    mutation_subset_note:
      'Items the in-src mutation harness (src/metal.rs gemma3_real_row_window_mutation_harness) runs under every mutant schedule. The full pack is used for the external-oracle receipts; the subset bounds the mutation run to one process-hour on a 16 GB host.',
    items,
  }
  await mkdir(dirname(outPath), { recursive: true })
  await writeFile(outPath, `${JSON.stringify(pack, null, 2)}\n`)
  process.stderr.write(`\nwrote ${outPath}: ${items.length} items\n`)
  for (const item of items) {
    const anchors = item.anchors.map((a) => `q-${a.measured_offset_from_query}=${a.colour}${a.inside_window_at_final_query ? '' : ' (outside)'}`)
    process.stderr.write(
      `  ${item.id.padEnd(16)} ${String(item.prompt_tokens).padStart(5)} tok  power=${item.window_power.padEnd(7)} ${anchors.join(' ')}\n`,
    )
  }
}

await main()
