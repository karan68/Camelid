'use strict'

function parsePort(value) {
  const port = Number(value)
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new RangeError('port must be from 1 through 65535')
  return port
}

module.exports = { parsePort }