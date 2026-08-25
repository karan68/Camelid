'use strict'

function publicProfile(name) {
  return { displayName: name.trim() }
}

module.exports = { publicProfile }