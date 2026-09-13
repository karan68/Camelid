// Explicit live-model smoke. Run against an isolated local engine store.
// Accepts only the known fixture edit and test command; never blanket-approves.
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
const base = process.env.CAMELID_CODING_LIVE_URL
assert.ok(base, 'Set CAMELID_CODING_LIVE_URL to an isolated loopback engine URL.')
const url = new URL(base)
assert.ok(['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname), 'Live smoke requires loopback.')
const out = resolve(process.env.CAMELID_CODING_LIVE_OUT || '../target/coding-live')
mkdirSync(out, { recursive: true })
const workspace = mkdtempSync(join(tmpdir(), 'camelid-coding-live-'))
const before = 'def greet(name):\n    return "Hello"\n'
const after = 'def greet(name):\n    return f"Hello, {name}!"\n'
writeFileSync(join(workspace, 'greet.py'), before)
writeFileSync(join(workspace, 'test_greet.py'), 'import unittest\nfrom greet import greet\n\nclass GreetingTests(unittest.TestCase):\n    def test_greeting(self):\n        self.assertEqual(greet("Camelid"), "Hello, Camelid!")\n')
async function request(path, method = 'GET', body, extra = {}) {
 const response = await fetch(base + path, { method, headers: { 'Content-Type': 'application/json', Origin: base, ...extra }, ...(body ? { body: JSON.stringify(body) } : {}), signal: AbortSignal.timeout(10000) })
 const data = await response.json()
 assert.ok(response.ok, `${response.status}: ${JSON.stringify(data)}`)
 return data
}
const health = await request('/v1/health')
assert.ok(health.loaded_now && health.generation_ready, 'Load a certified tool-capable model first.')
const goal = 'Fix greet.py so greet(name) returns f"Hello, {name}!". First use spawn_subagent to assign explorer to read greet.py and reviewer to read test_greet.py. Collect both findings with check_subagent_status. Use edit_file to fix greet.py; do not change test_greet.py. Then request exactly the shell command python3 -m unittest -q. Report only the test result you observed. This is a small fixture; keep each response brief.'
const creation = { workspace, goal, message_id: randomUUID().replaceAll('-', ''), model_id: health.active_model_id, allow_commands: true, max_steps: 32, max_tokens: 768 }
const route = '/api/agent/coding/sessions'
let session = await request(route, 'POST', creation)
const id = session.id, decisions = [], started = Date.now()
console.log(JSON.stringify({ session: id, workspace, model: health.active_model_id }))
let last = ''
try {
 while (['running', 'paused', 'waiting_approval', 'stopping'].includes(session.phase)) {
  const status = JSON.stringify({ phase: session.phase, agents: Object.fromEntries(Object.entries(session.agents).map(([id,a]) => [id,{status:a.status,action:a.action}])) })
  if (status !== last) { console.log(status); last = status }
  if (session.approval && !decisions.some(d => d.id === session.approval.id)) {
   const a = session.approval, review = a.detail.review
   const approved = review
    ? review.path === 'greet.py' && [after.trim(), after.trim().replaceAll('"', "'")].includes(review.after.trim())
    : a.tool === 'run_shell' && ['run_shell(python3 -m unittest -q)', 'python3 -m unittest -q'].includes(a.detail.command.trim())
   decisions.push({ id: a.id, tool: a.tool, approved, path: review?.path, command: a.detail.command })
   console.log(JSON.stringify({ approval: decisions.at(-1) }))
   await request(`${route}/${id}/approvals/${a.id}`, 'POST', { approved })
  }
  assert.ok(Date.now() - started < 15 * 60 * 1000, 'Live run exceeded 15 minutes.')
  await new Promise(resolve => setTimeout(resolve, 1000))
  session = await request(`${route}/${id}`)
 }
 writeFileSync(join(out, 'session.json'), JSON.stringify(session, null, 2))
 writeFileSync(join(out, 'decisions.json'), JSON.stringify(decisions, null, 2))
 assert.equal(session.phase, 'completed', `Live run: ${session.error}`)
 assert.ok(decisions.some(d => ['edit_file','write_file'].includes(d.tool) && d.approved), 'Model did not obtain a fixture edit approval.')
 assert.ok(decisions.some(d => d.tool === 'run_shell' && d.approved), 'Model did not obtain a test-command approval.')
 assert.equal(readFileSync(join(workspace, 'greet.py'), 'utf8').replaceAll("'", '"').trim(), after.trim())
 assert.ok(session.events.some(e => e.kind === 'tool.result' && e.detail.tool === 'run_shell' && e.detail.ok && e.detail.content.includes('OK')), 'No successful test result observed.')
 assert.ok(Object.values(session.agents).filter(a => a.parent_id === 'lead' && a.status === 'done').length >= 2, 'The lead did not collect two completed helpers.')
 const retry = await request(route, 'POST', creation)
 assert.equal(retry.id, id, 'Create retry must return the original run.')
 assert.equal(retry.run_id, session.run_id, 'Create retry must not replay the run.')
 const review = session.reviews.find(r => r.status === 'applied')
 assert.ok(review, 'No applied journal review.')
 await request(`/api/changes/${review.id}/undo`, 'POST', {}, { 'X-Camelid-Changes': '1' })
 assert.equal(readFileSync(join(workspace, 'greet.py'), 'utf8'), before, 'Undo did not restore the fixture.')
 console.log('Live coding smoke passed: two actual helpers, exact reviewed edit, approved command/test output, idempotent retry, and durable Undo.')
} catch (error) {
 if (['running','paused','waiting_approval'].includes(session.phase)) await request(`${route}/${id}/control`, 'POST', { action: 'stop' }).catch(() => {})
 writeFileSync(join(out, 'session.json'), JSON.stringify(session, null, 2))
 writeFileSync(join(out, 'decisions.json'), JSON.stringify(decisions, null, 2))
 throw error
}
