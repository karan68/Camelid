import { createRequire } from 'node:module'
import { resolve } from 'node:path'

const attemptRoot = process.argv[2]
const checks = []
try {
  if (!attemptRoot) throw new Error('attempt root argument is required')
  const require = createRequire(import.meta.url)
  const modulePath = resolve(attemptRoot, 'src', 'port.cjs')
  delete require.cache[modulePath]
  const { parsePort } = require(modulePath)
  checks.push(check('valid_decimal', parsePort('443') === 443, 'ordinary decimal ports must parse'))
  for (const value of [' 80', '80 ', '+80', '-1', '80.5', '0x50', '1e2', '', 'abc']) {
    checks.push(check(`reject_${encode(value)}`, throws(() => parsePort(value)), `${JSON.stringify(value)} must be rejected`))
  }
} catch (error) {
  checks.push(check('valid_decimal', false, error.message))
  for (const value of [' 80', '80 ', '+80', '-1', '80.5', '0x50', '1e2', '', 'abc']) {
    checks.push(check(`reject_${encode(value)}`, false, error.message))
  }
}

process.stdout.write(`${JSON.stringify({
  schema: 'camelid.benchmark.task-check/v1',
  passed: checks.every((item) => item.passed),
  checks,
})}\n`)

function check(id, passed, detail) {
  return { id, passed, detail }
}

function throws(action) {
  try {
    action()
    return false
  } catch {
    return true
  }
}

function encode(value) {
  if (value === '') return 'empty'
  return Buffer.from(value).toString('hex')
}