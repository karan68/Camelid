import { readFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { resolve } from 'node:path'

const attemptRoot = process.argv[2]
const checks = []
try {
  if (!attemptRoot) throw new Error('attempt root argument is required')
  const require = createRequire(import.meta.url)
  const profilePath = resolve(attemptRoot, 'src', 'profile.cjs')
  const renderPath = resolve(attemptRoot, 'src', 'render.cjs')
  const auditPath = resolve(attemptRoot, 'src', 'audit.cjs')
  for (const path of [profilePath, renderPath, auditPath]) delete require.cache[path]
  const { publicProfile } = require(profilePath)
  const { renderProfile } = require(renderPath)
  const { auditProfile } = require(auditPath)
  const profile = publicProfile(' Ada ')
  checks.push(check('new_contract_shape', JSON.stringify(profile) === JSON.stringify({ display_name: 'Ada' }), 'publicProfile must expose only display_name'))
  checks.push(check('render_caller_updated', renderProfile(' Ada ') === 'User: Ada', 'rendered output must remain stable'))
  checks.push(check('audit_caller_updated', auditProfile(' Ada ') === 'profile=ada', 'audit output must remain stable'))
  const source = await Promise.all([profilePath, renderPath, auditPath].map((path) => readFile(path, 'utf8')))
  checks.push(check('no_stale_symbol', source.every((text) => !text.includes('displayName')), 'displayName must not remain in source callers'))
} catch (error) {
  for (const id of ['new_contract_shape', 'render_caller_updated', 'audit_caller_updated', 'no_stale_symbol']) {
    checks.push(check(id, false, error.message))
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