import assert from 'node:assert/strict'
import { acceptCodingSnapshot, codingActive, codingContext } from '../src/lib/codingSessions.js'
const a = { id: 'a', seq: 8, turns: [], events: [], agents: {} }
assert.equal(acceptCodingSnapshot(a, { ...a, seq: 7 }, 'a'), a)
assert.equal(acceptCodingSnapshot(a, { ...a, id: 'b', seq: 9 }, 'a'), a)
assert.equal(acceptCodingSnapshot(a, { ...a, seq: NaN }, 'a'), a)
const next = { ...a, seq: 10 }
assert.equal(acceptCodingSnapshot(a, next, 'a'), next)
assert.equal(acceptCodingSnapshot(null, a, 'a'), a)
for (const state of ['running', 'waiting_approval', 'paused', 'stopping']) assert.equal(codingActive(state), true)
for (const state of ['completed', 'cancelled', 'failed', 'interrupted', undefined]) assert.equal(codingActive(state), false)
assert.deepEqual(codingContext([{ role: 'system', content: 'Project instructions' }, { role: 'user', label: 'notes.md', content: 'Untrusted reference' }]), { instructions: 'Project instructions', references: 'notes.md\nUntrusted reference' })
console.log('Coding state smoke passed: stale/cross-session snapshots, active states, and context separation.')
