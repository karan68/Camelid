function defaultEstimateTokenCount(value) {
  const text = String(value || '').trim()
  if (!text) return 0
  const wordPieces = text.match(/[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu) || []
  return Math.max(1, Math.round(Math.max(wordPieces.length, text.length / 4)))
}

export function extractSseEvents(buffer) {
  const normalized = String(buffer || '').replace(/\r\n/g, '\n')
  const parts = normalized.split('\n\n')
  return {
    events: parts.slice(0, -1),
    remainder: parts.at(-1) || '',
  }
}

function readSseDataLines(eventText) {
  return String(eventText || '')
    .replace(/\r\n/g, '\n')
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
}

export function readChatCompletionJsonPayload(payload, { estimateTokenCount = defaultEstimateTokenCount } = {}) {
  const choice = payload?.choices?.[0]
  const content = choice?.message?.content ?? choice?.text ?? ''
  return {
    content,
    finishReason: choice?.finish_reason ?? null,
    completionTokens: payload?.usage?.completion_tokens ?? estimateTokenCount(content),
    firstContentMs: null,
    usage: payload?.usage || null,
  }
}

export async function readStreamingChatCompletion(response, onDelta, { estimateTokenCount = defaultEstimateTokenCount } = {}) {
  if (!response.ok) {
    let detail = null
    try {
      detail = await response.json()
    } catch {
      // Fall through to generic response status below.
    }
    const message = detail?.error?.message || detail?.message || `Request failed with HTTP ${response.status}`
    const error = new Error(message)
    error.payload = detail
    throw error
  }

  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    const payload = await response.json()
    const parsed = readChatCompletionJsonPayload(payload, { estimateTokenCount })
    if (parsed.content) onDelta(parsed.content, parsed.content, { completionTokens: parsed.completionTokens, elapsedMs: 0, firstContentMs: null })
    return parsed
  }

  const reader = response.body?.getReader()
  if (!reader) return { content: '', finishReason: null, completionTokens: 0, firstContentMs: null, usage: null }
  const decoder = new TextDecoder()
  let buffer = ''
  let content = ''
  let finishReason = null
  let completionTokens = 0
  const streamStartedAt = performance.now()
  let firstContentMs = null

  const consumeEvent = (eventText) => {
    const dataLines = readSseDataLines(eventText)
    for (const data of dataLines) {
      if (!data || data === '[DONE]') continue
      let chunk = null
      try {
        chunk = JSON.parse(data)
      } catch {
        continue
      }
      const choice = chunk?.choices?.[0]
      const delta = choice?.delta?.content ?? choice?.text ?? ''
      if (delta) {
        completionTokens += 1
        if (firstContentMs === null) firstContentMs = performance.now() - streamStartedAt
        content += delta
        onDelta(delta, content, {
          completionTokens,
          elapsedMs: performance.now() - streamStartedAt,
          firstContentMs,
        })
      }
      if (choice?.finish_reason) finishReason = choice.finish_reason
    }
  }

  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const { events, remainder } = extractSseEvents(buffer)
    events.forEach(consumeEvent)
    buffer = remainder
  }
  buffer += decoder.decode()
  if (buffer.trim()) consumeEvent(buffer.replace(/\r\n/g, '\n'))
  return { content, finishReason, completionTokens, firstContentMs, usage: null }
}
