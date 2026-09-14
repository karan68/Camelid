// Real-model orchestration fixture. Run only on the designated test host with
// a separate engine data store. All mutations are confined to a fresh temp project.
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, readFileSync, mkdirSync, realpathSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { resolve, join } from 'node:path'
import { randomUUID } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { codingActive } from '../src/lib/codingSessions.js'

const base = process.env.CAMELID_CODING_LIVE_URL
assert.ok(base && ['127.0.0.1', 'localhost', '[::1]'].includes(new URL(base).hostname), 'Use an isolated loopback engine.')
const out = resolve(process.env.CAMELID_CODING_LIVE_OUT || '../target/coding-reliability-live')
mkdirSync(out, { recursive: true })
const workspace = realpathSync(mkdtempSync(join(tmpdir(), 'camelid-reliability-')))
const original = 'def count_open(tasks):\n    return len(tasks)\n'
const tests = `import unittest
from task_counter import count_open
class CounterTests(unittest.TestCase):
    def test_empty(self): self.assertEqual(count_open([]), 0)
    def test_mixed(self): self.assertEqual(count_open([{'done': True}, {'done': False}, {}]), 2)
    def test_all_done(self): self.assertEqual(count_open([{'done': True}]), 0)
if __name__ == '__main__': unittest.main()
`
writeFileSync(join(workspace, 'task_counter.py'), original)
writeFileSync(join(workspace, 'test_task_counter.py'), tests)
const python = process.platform === 'win32' ? 'python' : 'python3'
const command = `${python} -m unittest -v`
const runTests = () => spawnSync(python, ['-m', 'unittest', '-v'], { cwd: workspace, encoding: 'utf8', timeout: 15000 })
const baseline = runTests()
assert.ifError(baseline.error)
assert.equal(baseline.status, 1, 'The original counter must fail the real unit tests')
assert.match(baseline.stderr, /FAILED \(failures=2\)/, 'Both counter defects must be detected before any model edit')
writeFileSync(join(out, 'baseline.txt'), baseline.stdout + baseline.stderr)
const messageId = () => randomUUID().replaceAll('-', '')
const route = '/api/agent/coding/sessions'
async function request(path, method = 'GET', body, expected = 200) {
  const response = await fetch(base + path, { method, headers: { Origin: base, 'Content-Type': 'application/json' }, ...(body ? { body: JSON.stringify(body) } : {}), signal: AbortSignal.timeout(15000) })
  const data = await response.json()
  assert.equal(response.status, expected, JSON.stringify(data))
  return data
}
const health = await request('/v1/health')
const listing = await request(route)
assert.ok(listing.execution_engine?.id, 'Engine binding must be advertised')
let session = await request(route, 'POST', {
  workspace, model_id: health.active_model_id, message_id: messageId(), allow_commands: true,
  goal: 'Inspect task_counter.py and test_task_counter.py. Assign one read-only helper to inspect the tests with spawn_subagent. Fix count_open(tasks) to count tasks whose done flag is false or missing. Only edit task_counter.py, preserving the public API. Use verify_project for the configured tests. Collect the helper findings before finishing and report the observed check result.',
  max_steps: 32, max_tokens: 2048,
  project: { engine_id: listing.execution_engine.id, verification_command: command, browser_check: false, max_run_seconds: 900 },
})
const sessionId = session.id, firstRun = session.run_id, decisions = []
const active = () => codingActive(session.phase)
const started = Date.now()
let seq = -1
try {
  session = await request(`${route}/${sessionId}/messages`, 'POST', { mode: 'steer', run_id: firstRun, message_id: messageId(), message: 'Also handle an empty task list. Keep the exact count_open(tasks) function signature.' })
  session = await request(`${route}/${sessionId}/messages`, 'POST', { mode: 'queue', run_id: firstRun, message_id: messageId(), message: 'Summarize the recorded verification result without changing any files or running more tools.' })
  console.log(JSON.stringify({ sessionId, firstRun, workspace, model: health.active_model_id }))
  while (active()) {
    if (seq !== session.seq) {
      console.log(JSON.stringify({ seq: session.seq, run: session.run_id, phase: session.phase, action: session.agents.lead?.action, checks: session.checks?.map(c => c.status), helpers: session.helper_results?.map(r => ({ outcome: r.outcome, delivered: r.delivered })) }))
      seq = session.seq
    }
    const approval = session.approval
    if (approval && !decisions.some(d => d.id === approval.id)) {
      const review = approval.detail?.review
      const approved = Boolean(
        (review?.path === 'task_counter.py' && typeof review.after === 'string' && review.after.length < 8192 && session.run_id === firstRun)
        || (approval.tool === 'verify_project' && approval.detail.command === command && session.run_id === firstRun)
      )
      decisions.push({ id: approval.id, tool: approval.tool, approved })
      await request(`${route}/${sessionId}/approvals/${approval.id}`, 'POST', { approved })
    }
    assert.ok(Date.now() - started < 16 * 60 * 1000, 'Live fixture exceeded its time budget')
    await new Promise(done => setTimeout(done, 1000))
    session = await request(`${route}/${sessionId}`)
  }
  assert.equal(session.phase, 'completed', session.error)
  assert.equal(session.turns.length, 2, 'Queued follow-up must start once')
  assert.ok(session.incoming.some(m => m.mode === 'steer' && m.status === 'consumed'))
  assert.ok(session.incoming.some(m => m.mode === 'queue' && m.status === 'started'))
  assert.ok(session.helper_results.some(r => r.parent_run_id === firstRun && r.outcome === 'done' && r.delivered), 'Helper findings must be delivered')
  assert.ok(session.checks.some(c => c.status === 'passed' && c.run_id === firstRun && c.command === command), 'Exact approved checks must pass')
  assert.equal(readFileSync(join(workspace, 'test_task_counter.py'), 'utf8'), tests, 'Tests must stay unchanged')
  const independent = runTests()
  assert.ifError(independent.error)
  writeFileSync(join(out, 'independent-tests.txt'), independent.stdout + independent.stderr)
  assert.equal(independent.status, 0, independent.stderr)
  await request(`${route}/${sessionId}/project`, 'POST', { action: 'save_workflow', name: 'counter-check', notes: 'Run the existing counter unit tests after edits; preserve count_open(tasks).' })
  const checkpoint = session.checkpoints.find(c => c.run_id === firstRun)
  assert.ok(checkpoint?.review_ids.length)
  const generated = readFileSync(join(workspace, 'task_counter.py'), 'utf8')
  writeFileSync(join(workspace, 'task_counter.py'), '# independent manual edit\n')
  const stale = await request(`${route}/${sessionId}`)
  assert.ok(stale.checks.every(c => c.status !== 'passed'), 'External edits invalidate the displayed evidence')
  const stream = await fetch(base + `${route}/${sessionId}/events`, { headers: { Origin: base }, signal: AbortSignal.timeout(10000) })
  assert.equal(stream.status, 200)
  const frame = (await stream.text()).split('\n').find(line => line.startsWith('data:'))
  assert.ok(frame, 'A finished session must provide its terminal snapshot')
  assert.ok(JSON.parse(frame.slice(5)).checks.every(c => c.status !== 'passed'), 'Reconnect must not revive stale verification')
  await request(`${route}/${sessionId}/project`, 'POST', { action: 'restore_checkpoint', run_id: session.run_id, checkpoint_id: checkpoint.id }, 409)
  assert.equal(readFileSync(join(workspace, 'task_counter.py'), 'utf8'), '# independent manual edit\n')
  writeFileSync(join(workspace, 'task_counter.py'), generated)
  session = await request(`${route}/${sessionId}/project`, 'POST', { action: 'restore_checkpoint', run_id: session.run_id, checkpoint_id: checkpoint.id })
  assert.equal(readFileSync(join(workspace, 'task_counter.py'), 'utf8'), original)
  console.log('Live reliability fixture passed: correction, queued follow-up, automatic helper delivery, real approved edit and verification, independent unchanged tests, saved workflow, and conflict-preserving grouped Undo.')
} finally {
  if (active()) await request(`${route}/${sessionId}/control`, 'POST', { action: 'stop' }).catch(() => {})
  writeFileSync(join(out, 'session.json'), JSON.stringify(session, null, 2))
  writeFileSync(join(out, 'decisions.json'), JSON.stringify(decisions, null, 2))
}
