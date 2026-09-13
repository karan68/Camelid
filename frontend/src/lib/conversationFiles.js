import { codeFences } from './codeFences.js'
import { textOutput, toolOutputs } from './outputFiles.js'

// Derived from all saved turns, including those outside the visible history window.
export function conversationFiles(messages = []) {
  return messages.flatMap(message => {
    const outputs = message.role === 'tool' ? toolOutputs(message.content)
      : message.role === 'assistant' && !message.streaming && ['json', 'text'].includes(message.output_format) ? [textOutput(message.content, message.output_format)]
      : message.role === 'assistant' ? codeFences(message.content)
        .filter(fence => !message.streaming || !fence.incomplete)
        .map(fence => textOutput(fence.code, fence.language)) : []
    return outputs.map((output, index) => ({ ...output, id: `${message.id}:${index}`, messageId: message.id }))
  })
}
