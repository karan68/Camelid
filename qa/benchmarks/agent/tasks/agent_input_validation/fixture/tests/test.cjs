'use strict'

const assert = require('node:assert/strict')
const { parsePort } = require('../src/port.cjs')

assert.equal(parsePort('1'), 1)
assert.equal(parsePort('8080'), 8080)
assert.equal(parsePort('65535'), 65535)
assert.throws(() => parsePort('0'), /port/)
assert.throws(() => parsePort('65536'), /port/)