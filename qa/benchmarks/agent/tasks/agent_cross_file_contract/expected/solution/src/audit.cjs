'use strict'

const { publicProfile } = require('./profile.cjs')

function auditProfile(name) {
  return `profile=${publicProfile(name).display_name.toLowerCase()}`
}

module.exports = { auditProfile }