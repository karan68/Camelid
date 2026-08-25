'use strict'

function parsePort(value) {
  if (typeof value !== 'string' || !/^[0-9]+$/.test(value.trim())) throw new RangeError('port must be a base-10 digit string')
  const port = Number(value.trim())
  if (!Number.isSafeInteger(port) || port < 1 || port > 65535) throw new RangeError('port must be from 1 through 65535')
  return port
}

module.exports = { parsePort }