export async function changeRequest(apiBase, path = '', { method = 'GET', body, signal } = {}) {
  const response = await fetch(String(apiBase || '').replace(/\/$/, '') + '/api/changes' + path, {
    method, signal, headers: { 'Content-Type': 'application/json', 'X-Camelid-Changes': '1' },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })
  const result = await response.json().catch(error => {
    // A review can be closed after headers arrive but before its body is read.
    // Preserve cancellation so callers cannot publish a null review as success.
    if (error.name === 'AbortError') throw error
    return null
  })
  if (!response.ok) throw new Error(result?.error?.message || 'Could not load the file review (' + response.status + ').')
  if (!result) throw new Error('The file review response was unreadable. Try opening it again.')
  return result
}
