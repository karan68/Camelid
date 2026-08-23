'use strict'

function publicProfile(name) {
  const displayName = name.trim()
  return { displayName, display_name: displayName }
}

module.exports = { publicProfile }