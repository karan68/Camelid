#!/usr/bin/env node
import assert from 'node:assert/strict'

import { analyzePairedSamples, median } from './stats/comparison.mjs'

assert.equal(median([3]), 3)
assert.equal(median([4, 1, 3, 2]), 2.5)
assert.throws(() => median([]), /non-empty/)
assert.throws(() => median([1, Number.NaN]), /finite/)

const aa = analyzePairedSamples([
  { base: 10, head: 10 },
  { base: 20, head: 20 },
  { base: 30, head: 30 },
  { base: 40, head: 40 },
])
assert.equal(aa.validPairs, 4)
assert.equal(aa.medianRatioHeadOverBase, 1)
assert.deepEqual(aa.bootstrapCi, [1, 1])
assert.equal(aa.observedDirection, 'no_clear_direction')
assert.equal(aa.verdict, 'INCONCLUSIVE_NOISE')

const improvement = analyzePairedSamples([
  { base: 100, head: 110 },
  { base: 90, head: 99 },
  { base: 120, head: 132 },
  { base: 80, head: 88 },
], { seed: 7 })
close(improvement.medianRatioHeadOverBase, 1.1)
assert.equal(improvement.observedDirection, 'head_faster')
assert.equal(improvement.verdict, 'INCONCLUSIVE_NOISE')

const slowdown = analyzePairedSamples([
  { base: 100, head: 90 },
  { base: 90, head: 81 },
  { base: 120, head: 108 },
  { base: 80, head: 72 },
], { seed: 9 })
close(slowdown.medianRatioHeadOverBase, 0.9)
assert.equal(slowdown.observedDirection, 'head_slower')

const lowerIsBetter = analyzePairedSamples([
  { base: 100, head: 90 },
  { base: 110, head: 99 },
  { base: 120, head: 108 },
], { higherIsBetter: false, seed: 11 })
assert.equal(lowerIsBetter.observedDirection, 'head_faster')

const noisyPairs = [
  { base: 100, head: 101 },
  { base: 100, head: 99 },
  { base: 100, head: 102 },
  { base: 100, head: 98 },
  { base: 100, head: 100 },
]
const first = analyzePairedSamples(noisyPairs, { seed: 1234, bootstrapSamples: 2000 })
const second = analyzePairedSamples([...noisyPairs].reverse(), { seed: 1234, bootstrapSamples: 2000 })
assert.deepEqual(first, second)
assert.equal(first.observedDirection, 'no_clear_direction')

const excluded = analyzePairedSamples([
  { base: 10, head: 11 },
  { base: 0, head: 2 },
  { base: 2, head: Number.NaN },
])
assert.equal(excluded.validPairs, 1)
assert.equal(excluded.excludedPairs.length, 2)
assert.equal(excluded.observedDirection, 'insufficient_data')
assert.equal(excluded.verdict, 'INCONCLUSIVE_NOISE')

const none = analyzePairedSamples([{ base: -1, head: 2 }])
assert.equal(none.validPairs, 0)
assert.equal(none.medianRatioHeadOverBase, null)
assert.equal(none.bootstrapCi, null)
assert.equal(none.observedDirection, 'insufficient_data')

assert.throws(() => analyzePairedSamples([], { confidence: 1 }), /confidence/)
assert.throws(() => analyzePairedSamples([], { bootstrapSamples: 0 }), /bootstrapSamples/)

console.log('benchmark Phase 1 paired statistics: PASS')

function close(actual, expected) {
  assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`)
}
