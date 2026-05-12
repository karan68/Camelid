#!/usr/bin/env node
import assert from 'node:assert/strict'

import { extractSseEvents, readChatCompletionJsonPayload, readStreamingChatCompletion } from '../src/lib/chatCompletionStream.js'

const partial = 'data: {"choices":[{"delta":{"content":"hel"}}]}\r\n\r\ndata: {"choices":[{"delta":{"content":"lo"}}]}'
const firstPass = extractSseEvents(partial)
assert.equal(firstPass.events.length, 1, 'complete SSE events should flush while partial backend chunks stay buffered')
assert.match(firstPass.remainder, /"lo"/, 'partial SSE data should remain buffered until the blank-line event boundary arrives')
const secondPass = extractSseEvents(`${firstPass.remainder}\n\ndata: [DONE]\n\n`)
assert.equal(secondPass.events.length, 2, 'the remaining partial SSE event should flush after its boundary arrives')

const jsonPayload = readChatCompletionJsonPayload({
  choices: [{ message: { content: 'json reply' }, finish_reason: 'stop' }],
  usage: { completion_tokens: 2 },
})
assert.equal(jsonPayload.content, 'json reply', 'non-streaming JSON fallback should preserve assistant content')
assert.equal(jsonPayload.completionTokens, 2, 'JSON usage should remain exact when the backend provides it')

function streamFromChunks(chunks) {
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
}

const response = new Response(streamFromChunks([
  'data: {"choices":[{"delta":{"content":"```js\\nconst"}}]}\n\n',
  'data: {"choices":[{"delta":{"content":" answer = 42"}}]}\n',
  '\n',
  'data: {"choices":[{"finish_reason":"stop"}]}\n\n',
  'data: [DONE]\n\n',
]), {
  status: 200,
  headers: { 'content-type': 'text/event-stream' },
})

const deltas = []
const streamed = await readStreamingChatCompletion(response, (delta, fullContent, metrics) => {
  deltas.push({ delta, fullContent, completionTokens: metrics.completionTokens })
})

assert.equal(streamed.content, '```js\nconst answer = 42', 'stream parser should preserve incomplete fenced code content safely for live rendering')
assert.equal(streamed.finishReason, 'stop', 'stream parser should preserve finish_reason from the terminal chunk')
assert.deepEqual(deltas.map((item) => item.fullContent), ['```js\nconst', '```js\nconst answer = 42'], 'stream deltas should update visible content before backend completion')
assert.deepEqual(deltas.map((item) => item.completionTokens), [1, 2], 'stream metrics should advance while generation is active')

console.log('Streaming parser smoke passed')
