'use strict'

const { publicProfile } = require('./profile.cjs')

function renderProfile(name) {
  return `User: ${publicProfile(name).displayName}`
}

module.exports = { renderProfile }