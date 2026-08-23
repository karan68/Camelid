'use strict'

const { publicProfile } = require('./profile.cjs')

function renderProfile(name) {
  return `User: ${publicProfile(name).display_name}`
}

module.exports = { renderProfile }