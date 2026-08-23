import { createRequire } from 'node:module'
import { resolve } from 'node:path'

const attemptRoot = process.argv[2]
const checks = []
try {
  if (!attemptRoot) throw new Error('attempt root argument is required')
  const require = createRequire(import.meta.url)
  const modulePath = resolve(attemptRoot, 'src', 'pricing.cjs')
  delete require.cache[modulePath]
  const { discountedTotal } = require(modulePath)
  checks.push(check('threshold_is_inclusive', discountedTotal(10000, 10) === 9000, '10000 cents with 10% discount must total 9000 cents'))
  checks.push(check('requested_discount_used', discountedTotal(10000, 25) === 7500, 'the inclusive boundary must use the requested discount'))
  checks.push(check('public_api_preserved', typeof discountedTotal === 'function', 'discountedTotal must remain exported'))
} catch (error) {
  checks.push(check('threshold_is_inclusive', false, error.message))
  checks.push(check('requested_discount_used', false, error.message))
  checks.push(check('public_api_preserved', false, error.message))
}

process.stdout.write(`${JSON.stringify({
  schema: 'camelid.benchmark.task-check/v1',
  passed: checks.every((item) => item.passed),
  checks,
})}\n`)

function check(id, passed, detail) {
  return { id, passed, detail }
}