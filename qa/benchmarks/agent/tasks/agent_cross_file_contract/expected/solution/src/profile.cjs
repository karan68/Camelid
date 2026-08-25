'use strict'

function publicProfile(name) {
  return { display_name: name.trim() }
}

module.exports = { publicProfile }