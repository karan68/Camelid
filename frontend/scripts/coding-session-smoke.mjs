import assert from 'node:assert/strict'
import { acceptCodingSnapshot, codingActive, codingContext } from '../src/lib/codingSessions.js'
const a = { id: 'a', seq: 8, turns: [], events: [], agents: {} }
assert.equal(acceptCodingSnapshot(a, { ...a, seq: 7 }, 'a'), a)
assert.equal(acceptCodingSnapshot(a, { ...a, id: 'b', seq: 9 }, 'a'), a)
assert.equal(acceptCodingSnapshot(a, { ...a, seq: NaN }, 'a'), a)
const next = { ...a, seq: 10 }
assert.equal(acceptCodingSnapshot(a, next, 'a'), next)
assert.equal(acceptCodingSnapshot(null, a, 'a'), a)
for (const state of ['running', 'waiting_approval', 'waiting_helpers', 'paused', 'stopping']) assert.equal(codingActive(state), true)
for (const state of ['completed', 'cancelled', 'failed', 'interrupted', undefined]) assert.equal(codingActive(state), false)
assert.deepEqual(codingContext([{ role: 'system', content: 'Project instructions' }, { role: 'user', label: 'notes.md', content: 'Untrusted reference' }]), { instructions: 'Project instructions', references: 'notes.md\nUntrusted reference' })
console.log('Coding state smoke passed: stale/cross-session snapshots, active states, and context separation.')

const { describeCodingMessage } = await import('../src/lib/codingPresentation.js')
for (const code of ['```js\nconst secret = 1\n```', '~~~js\nconst secret = 1\n~~~', '```js\nconst secret = 1', '    const secret = 1', '<tool_call>\nconst secret = 1\n</tool_call>']) {
 const result = describeCodingMessage('Changed the task list.\n' + code)
 assert.equal(result.description, 'Changed the task list.')
 assert.match(result.snippets[0].content, /const secret/)
}
assert.equal(describeCodingMessage('Updated `app.js` and checked the links.').description, 'Updated `app.js` and checked the links.')

// Closing a file review while its body is arriving must remain cancellation.
const { changeRequest } = await import('../src/lib/changeReviews.js')
const originalFetch = globalThis.fetch
try {
 globalThis.fetch = async () => ({ ok: true, json: async () => { throw new DOMException('Closed review', 'AbortError') } })
 await assert.rejects(changeRequest('', '/test'), { name: 'AbortError' })
 globalThis.fetch = async () => ({ ok: true, json: async () => { throw new SyntaxError('Incomplete JSON') } })
 await assert.rejects(changeRequest('', '/test'), /unreadable/)
} finally { globalThis.fetch = originalFetch }
