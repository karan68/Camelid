export const WORKSPACE_IDLE_STATE = Object.freeze({
  phase: 'idle',
  events: [],
  pendingApproval: null,
  error: '',
})

export function reduceWorkspaceEvent(state, envelope) {
  const event = envelope?.event
  if (!event) return state
  if (event === 'session.reset') return { ...WORKSPACE_IDLE_STATE, events: [] }
  if (event === 'session.starting') return { ...state, phase: 'starting', error: '' }
  if (event === 'approval.resolved') return { ...state, phase: 'running', pendingApproval: null }
  const events = [...state.events]
  const withoutLiveTail = () => {
    if (events.at(-1)?.event === 'model.live') events.pop()
  }

  if (event === 'model.delta') {
    const content = String(envelope.content || '')
    if (!content) return state
    const tail = events.at(-1)
    if (tail?.event === 'model.live') {
      events[events.length - 1] = { ...tail, content: `${tail.content}${content}` }
    } else {
      events.push({ ...envelope, event: 'model.live', content })
    }
    return { ...state, phase: 'running', events }
  }

  if (event === 'tool.call' || event === 'model.answer') withoutLiveTail()
  events.push(envelope)

  if (event === 'approval.required') {
    return { ...state, phase: 'awaiting_approval', events, pendingApproval: envelope }
  }
  if (event === 'tool.result') {
    return { ...state, phase: 'running', events, pendingApproval: null }
  }
  if (event === 'session.finished') {
    return { ...state, phase: envelope.outcome === 'answered' ? 'finished' : envelope.outcome, events, pendingApproval: null }
  }
  if (event === 'session.error') {
    return { ...state, phase: 'error', events, pendingApproval: null, error: String(envelope.message || 'Workspace stopped.') }
  }
  return { ...state, phase: event === 'session.started' ? 'running' : state.phase, events }
}

export function workspaceEndpoint(apiBase, suffix = '') {
  const base = String(apiBase || '').replace(/\/$/, '')
  return `${base}/api/agent/workspace/sessions${suffix}`
}

async function readError(response, fallback) {
  try {
    const payload = await response.json()
    return payload?.error?.message || fallback
  } catch {
    return fallback
  }
}

export async function createWorkspaceSession(apiBase, input) {
  const response = await fetch(workspaceEndpoint(apiBase), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
  if (!response.ok) throw new Error(await readError(response, `Workspace start failed (${response.status}).`))
  return response.json()
}

export async function decideWorkspaceApproval(apiBase, sessionId, approvalId, decision) {
  const response = await fetch(workspaceEndpoint(apiBase, `/${encodeURIComponent(sessionId)}/decisions`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approval_id: approvalId, decision }),
  })
  if (!response.ok) throw new Error(await readError(response, `Approval failed (${response.status}).`))
}

export async function cancelWorkspaceSession(apiBase, sessionId) {
  const response = await fetch(workspaceEndpoint(apiBase, `/${encodeURIComponent(sessionId)}`), { method: 'DELETE' })
  if (!response.ok && response.status !== 404) throw new Error(await readError(response, `Stop failed (${response.status}).`))
}
