'use strict'

function discountedTotal(subtotalCents, discountPercent) {
  if (!Number.isInteger(subtotalCents) || subtotalCents < 0) throw new RangeError('subtotalCents must be a non-negative integer')
  if (!Number.isInteger(discountPercent) || discountPercent < 0 || discountPercent > 100) throw new RangeError('discountPercent must be an integer from 0 to 100')
  if (subtotalCents >= 10000) return Math.round(subtotalCents * (100 - discountPercent) / 100)
  return subtotalCents
}

module.exports = { discountedTotal }