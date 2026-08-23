'use strict'

const { publicProfile } = require('./profile.cjs')

function auditProfile(name) {
  return `profile=${publicProfile(name).displayName.toLowerCase()}`
}

module.exports = { auditProfile }