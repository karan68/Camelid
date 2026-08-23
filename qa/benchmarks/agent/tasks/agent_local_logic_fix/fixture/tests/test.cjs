'use strict'

const assert = require('node:assert/strict')
const { discountedTotal } = require('../src/pricing.cjs')

assert.equal(discountedTotal(9999, 10), 9999)
assert.equal(discountedTotal(10001, 10), 9001)
assert.throws(() => discountedTotal(-1, 10), /subtotalCents/)
assert.throws(() => discountedTotal(10001, 101), /discountPercent/)