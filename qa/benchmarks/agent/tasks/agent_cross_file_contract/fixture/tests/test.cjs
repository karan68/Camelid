'use strict'

const assert = require('node:assert/strict')
const { auditProfile } = require('../src/audit.cjs')
const { renderProfile } = require('../src/render.cjs')

assert.equal(renderProfile(' Ada '), 'User: Ada')
assert.equal(auditProfile(' Ada '), 'profile=ada')