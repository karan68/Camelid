#!/usr/bin/env node
/* The context meter and the send-budget check must not tell two stories.
 *
 * A user complaint that started this: "the context window seems really very
 * small compared to Ollama". Camelid reads the model's native window straight
 * from GGUF metadata, so the window was never small -- but the UI only ever
 * surfaced context as a *caution* ("beyond the verified row's tested N-token
 * context"), never as a quantity. A tested envelope rendered as the only number
 * on screen reads as a ceiling.
 *
 * The meter fixes that by showing the real denominator, so its arithmetic now
 * has to match the admission rules the backend actually applies
 * (`src/api/mod.rs`, prompt admission):
 *
 *   1. A response limit is an UPPER BOUND, clamped to the room left in the
 *      window. An oversized reservation is not an overflow.
 *   2. The only hard failure is a prompt that already fills the window.
 *
 * If a future edit makes the meter disagree with `validateSendBudget`, one of
 * the two is lying to the user. This fails here instead.
 *
 * Pure module test -- no browser, no dist build required.
 */
import assert from 'node:assert/strict'
import {
  composeContextBudget,
  formatTokenCount,
  formatPercent,
} from '../src/lib/contextBudget.js'
import { validateSendBudget } from '../src/lib/responseLimits.js'

let checks = 0
function check(label, fn) {
  fn()
  checks += 1
  console.log(`  ok  ${label}`)
}

console.log('context meter arithmetic')

check('an unreadable window renders no meter rather than a guessed one', () => {
  assert.equal(composeContextBudget({ contextLength: null, promptTokens: 10 }), null)
  assert.equal(composeContextBudget({ contextLength: 0, promptTokens: 10 }), null)
})

check('rule 1: an oversized reservation is clamped, never an overflow', () => {
  // The common real case: an 8192 response limit on an 8192-context model.
  const b = composeContextBudget({ contextLength: 8192, promptTokens: 2000, reservedTokens: 8192 })
  assert.equal(b.reservedTokens, 6192, 'reservation should take exactly the remaining room')
  assert.equal(b.freeTokens, 0)
  assert.equal(b.reserveClamped, true)
  assert.equal(b.level, 'notice')
  assert.ok(b.usedPercent + b.reservedPercent + b.freePercent <= 100.0001)
})

check('rule 2: only a prompt that fills the window is a hard error', () => {
  const nearly = composeContextBudget({ contextLength: 4096, promptTokens: 4095, reservedTokens: 512 })
  assert.equal(nearly.level, 'notice', '1 token of room is still not an error')
  const full = composeContextBudget({ contextLength: 4096, promptTokens: 4096, reservedTokens: 512 })
  assert.equal(full.level, 'error')
  assert.equal(full.reservedTokens, 0)
})

check('the meter and validateSendBudget agree on every boundary', () => {
  const contextLength = 8192
  const cases = [
    { promptTokens: 1, maxTokens: 256 },
    { promptTokens: 4000, maxTokens: 4192 },
    { promptTokens: 4000, maxTokens: 8192 },
    { promptTokens: 8191, maxTokens: 1 },
    { promptTokens: 8192, maxTokens: 1 },
    { promptTokens: 99999, maxTokens: 512 },
  ]
  for (const { promptTokens, maxTokens } of cases) {
    const meter = composeContextBudget({ contextLength, promptTokens, reservedTokens: maxTokens })
    const gate = validateSendBudget({ promptTokens, maxTokens, contextLength })
    const gateLevel = gate.level === 'ok' ? 'ok' : gate.level
    assert.equal(
      meter.level,
      gateLevel,
      `disagreement at prompt=${promptTokens} max=${maxTokens}: meter=${meter.level} gate=${gateLevel}`,
    )
  }
})

check('percentages never exceed the bar', () => {
  const b = composeContextBudget({ contextLength: 1000, promptTokens: 5000, reservedTokens: 5000 })
  assert.ok(b.usedPercent <= 100)
  assert.ok(b.filledPercent <= 100)
  assert.equal(b.usedTokens, 1000, 'a prompt larger than the window is clamped to it')
})

check('segments are subsets of the prompt and never invent tokens', () => {
  const b = composeContextBudget({
    contextLength: 10000,
    promptTokens: 1000,
    systemTokens: 400,
    imageTokens: 300,
    reservedTokens: 500,
  })
  const byKey = Object.fromEntries(b.segments.map((s) => [s.key, s.tokens]))
  assert.equal(byKey.system + byKey.images + byKey.messages, 1000, 'segments must sum to the prompt')
  const total = b.segments.reduce((sum, s) => sum + s.tokens, 0)
  assert.equal(total, 10000, 'every token in the window is accounted for exactly once')
})

check('empty segments are dropped instead of rendering as zero rows', () => {
  const b = composeContextBudget({ contextLength: 4096, promptTokens: 100, reservedTokens: 256 })
  assert.ok(!b.segments.some((s) => s.tokens === 0))
  assert.ok(!b.segments.some((s) => s.key === 'images'), 'no images means no image row')
})

console.log('verified bound is a marker, not a ceiling')

check('the marker is hidden when the whole window is verified', () => {
  const b = composeContextBudget({ contextLength: 8192, promptTokens: 10, reservedTokens: 10, verifiedBound: 8192 })
  assert.equal(b.showVerifiedMarker, false, 'a fully verified window needs no marker')
  assert.equal(b.verifiedPercent, null)
})

check('a tested envelope smaller than the window is drawn on the bar', () => {
  // The exact shape of the complaint: 8k tested inside a 40k window.
  const b = composeContextBudget({ contextLength: 40960, promptTokens: 1000, reservedTokens: 8192, verifiedBound: 8192 })
  assert.equal(b.showVerifiedMarker, true)
  assert.ok(b.verifiedPercent > 19 && b.verifiedPercent < 21, 'marker sits at ~20% of a 40,960 window')
  assert.equal(b.beyondVerified, false)
  assert.ok(b.freeTokens > 0, 'context past the tested envelope stays usable')
  assert.equal(b.level, 'ok', 'being under the tested bound is not a warning state')
})

check('passing the tested envelope is reported without becoming an error', () => {
  const b = composeContextBudget({ contextLength: 40960, promptTokens: 12000, reservedTokens: 4096, verifiedBound: 8192 })
  assert.equal(b.beyondVerified, true)
  assert.equal(b.level, 'ok', 'untested is not unsupported')
})

console.log('formatting')

check('token counts stay short enough for a chip', () => {
  assert.equal(formatTokenCount(0), '0')
  assert.equal(formatTokenCount(512), '512')
  assert.equal(formatTokenCount(8192), '8.2K')
  assert.equal(formatTokenCount(490432), '490K')
  assert.equal(formatTokenCount(1048576), '1.0M')
  for (const n of [0, 1, 999, 1000, 40960, 131072, 1048576, 10000000]) {
    assert.ok(formatTokenCount(n).length <= 6, `${n} formats too wide: ${formatTokenCount(n)}`)
  }
})

check('a non-empty share never rounds away to 0%', () => {
  assert.equal(formatPercent(0), '0%')
  assert.equal(formatPercent(0.2), '<1%')
  assert.equal(formatPercent(49.4), '49%')
})

console.log(`\ncontext-meter smoke: ${checks} checks passed`)
