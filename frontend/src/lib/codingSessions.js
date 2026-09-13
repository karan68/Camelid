// The server owns execution. A reconnect replaces the view from a versioned
// snapshot; it never dispatches a saved tool call or an approval.
export const codingActive = phase => ['running', 'paused', 'waiting_approval', 'stopping'].includes(phase)
export const codingMessageId = () => crypto.randomUUID().replaceAll('-', '')
export function acceptCodingSnapshot(current, next, selectedId) {
  if (!next || next.id !== selectedId || !Number.isSafeInteger(next.seq) || next.seq < 0 || !Array.isArray(next.turns) || !Array.isArray(next.events) || !next.agents) return current
  if (current?.id === next.id && current.seq > next.seq) return current
  return next
}
export function codingEndpoint(apiBase, suffix = '') {
  return String(apiBase || '').replace(/\/$/, '') + '/api/agent/coding/sessions' + suffix
}
export async function codingRequest(apiBase, suffix = '', { method = 'GET', body, signal } = {}) {
  const response = await fetch(codingEndpoint(apiBase, suffix), {
    method, signal, headers: { 'Content-Type': 'application/json' },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })
  const data = await response.json().catch(() => null)
  if (!response.ok) throw new Error(data?.error?.message || `Coding request failed (${response.status}).`)
  if (!data) throw new Error('This engine did not return coding-session data. Update the engine and try again.')
  return data
}
export function codingContext(sources = []) {
  return {
    instructions: sources.filter(s => s.role === 'system').map(s => s.content).join('\n\n'),
    references: sources.filter(s => s.role !== 'system').map(s => `${s.label || 'Reference'}\n${s.content}`).join('\n\n'),
  }
}
