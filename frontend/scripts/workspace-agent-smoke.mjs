import assert from 'node:assert/strict'
import { reduceWorkspaceEvent, WORKSPACE_IDLE_STATE, workspaceEndpoint, workspaceModelsEndpoint, workspaceBrowseEndpoint, workspaceThreadsEndpoint, workspaceCompactionEndpoint } from '../src/lib/workspaceAgent.js'

let state = { ...WORKSPACE_IDLE_STATE, events: [] }
state = reduceWorkspaceEvent(state, { event: 'session.started', model_id: 'tool-model', sequence: 1 })
state = reduceWorkspaceEvent(state, { event: 'model.delta', content: '<tool_', sequence: 2 })
state = reduceWorkspaceEvent(state, { event: 'model.delta', content: 'call>', sequence: 3 })
assert.equal(state.events.at(-1).event, 'model.live')
assert.equal(state.events.at(-1).content, '<tool_call>')

state = reduceWorkspaceEvent(state, { event: 'tool.call', detail: 'read_file(a.txt)', sequence: 4 })
assert.equal(state.events.at(-1).event, 'tool.call')
assert.equal(state.events.some((event) => event.event === 'model.live'), false, 'raw streamed tool syntax must be removed')

state = reduceWorkspaceEvent(state, { event: 'model.delta', content: 'Done', sequence: 6 })
state = reduceWorkspaceEvent(state, { event: 'model.answer', content: 'Done', sequence: 7 })
assert.equal(state.events.at(-1).event, 'model.answer')
assert.equal(state.events.filter((event) => event.event === 'model.live').length, 0)

state = reduceWorkspaceEvent(state, {
  event: 'memory.compacted', compacted_through_turn: 3, archived_turns: 4,
  compaction_count: 1, trigger_tokens: 3072, budget_total: 4096, sequence: 8,
})
assert.equal(state.events.at(-1).event, 'memory.compacted')
assert.equal(state.events.at(-1).archived_turns, 4)
assert.equal(state.pendingApproval, null, 'read-only Workspace must never acquire write approval state')

state = reduceWorkspaceEvent(state, { event: 'session.finished', outcome: 'answered', sequence: 9 })
assert.equal(state.phase, 'finished')

state = reduceWorkspaceEvent(state, { event: 'session.reset' })
assert.deepEqual(state, { ...WORKSPACE_IDLE_STATE, events: [] })
state = reduceWorkspaceEvent(state, {
  event: 'thread.restored',
  turns: [{ user_text: 'Where is login?', assistant_text: 'In src/auth.rs.' }],
})
assert.equal(state.events[0].event, 'turn.user')
assert.equal(state.events[1].event, 'model.answer')
state = reduceWorkspaceEvent(state, { event: 'turn.starting' })
assert.equal(state.phase, 'starting')
assert.equal(state.events.length, 2, 'starting a follow-up must preserve restored turns')
assert.equal(workspaceEndpoint('http://127.0.0.1:8181/', '/abc/events'), 'http://127.0.0.1:8181/api/agent/workspace/sessions/abc/events')
assert.equal(workspaceModelsEndpoint('http://127.0.0.1:8181/'), 'http://127.0.0.1:8181/api/agent/workspace/models')
assert.equal(workspaceBrowseEndpoint('http://127.0.0.1:8181/'), 'http://127.0.0.1:8181/api/agent/workspace/browse')
assert.equal(workspaceBrowseEndpoint('http://127.0.0.1:8181/', 'C:/data'), 'http://127.0.0.1:8181/api/agent/workspace/browse?path=C%3A%2Fdata')
assert.equal(workspaceThreadsEndpoint('http://127.0.0.1:8181/', 'C:/data', 'thread/1'), 'http://127.0.0.1:8181/api/agent/workspace/threads/thread%2F1?workspace=C%3A%2Fdata')
assert.equal(workspaceCompactionEndpoint('http://127.0.0.1:8181/', 'C:/data', 'thread/1'), 'http://127.0.0.1:8181/api/agent/workspace/threads/thread%2F1/compact?workspace=C%3A%2Fdata')

console.log('workspace-agent-smoke: PASS')
