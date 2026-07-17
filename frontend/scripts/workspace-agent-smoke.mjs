import assert from 'node:assert/strict'
import { reduceWorkspaceEvent, WORKSPACE_IDLE_STATE, workspaceEndpoint } from '../src/lib/workspaceAgent.js'

let state = { ...WORKSPACE_IDLE_STATE, events: [] }
state = reduceWorkspaceEvent(state, { event: 'session.started', model_id: 'tool-model', sequence: 1 })
state = reduceWorkspaceEvent(state, { event: 'model.delta', content: '<tool_', sequence: 2 })
state = reduceWorkspaceEvent(state, { event: 'model.delta', content: 'call>', sequence: 3 })
assert.equal(state.events.at(-1).event, 'model.live')
assert.equal(state.events.at(-1).content, '<tool_call>')

state = reduceWorkspaceEvent(state, { event: 'tool.call', detail: 'read_file(a.txt)', sequence: 4 })
assert.equal(state.events.at(-1).event, 'tool.call')
assert.equal(state.events.some((event) => event.event === 'model.live'), false, 'raw streamed tool syntax must be removed')

state = reduceWorkspaceEvent(state, {
  event: 'approval.required',
  approval_id: 'approval-1',
  tool: 'write_file',
  detail: 'write_file -> result.txt',
  sequence: 5,
})
assert.equal(state.phase, 'awaiting_approval')
assert.equal(state.pendingApproval.approval_id, 'approval-1')

state = reduceWorkspaceEvent(state, { event: 'approval.resolved' })
assert.equal(state.phase, 'running')
assert.equal(state.pendingApproval, null)

state = reduceWorkspaceEvent(state, { event: 'model.delta', content: 'Done', sequence: 6 })
state = reduceWorkspaceEvent(state, { event: 'model.answer', content: 'Done', sequence: 7 })
assert.equal(state.events.at(-1).event, 'model.answer')
assert.equal(state.events.filter((event) => event.event === 'model.live').length, 0)

state = reduceWorkspaceEvent(state, { event: 'session.finished', outcome: 'answered', sequence: 8 })
assert.equal(state.phase, 'finished')

state = reduceWorkspaceEvent(state, { event: 'session.reset' })
assert.deepEqual(state, { ...WORKSPACE_IDLE_STATE, events: [] })
assert.equal(workspaceEndpoint('http://127.0.0.1:8181/', '/abc/events'), 'http://127.0.0.1:8181/api/agent/workspace/sessions/abc/events')

console.log('workspace-agent-smoke: PASS')
