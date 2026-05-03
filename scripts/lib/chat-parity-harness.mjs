export const SUPPORTED_RENDER_MODES = ['compact', 'tinyllama-marker']

export function renderExpectedPrompt(messages, renderMode) {
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new Error('messages must contain at least one entry')
  }
  switch (renderMode) {
    case 'compact':
      return renderCompactLlama3Prompt(messages)
    case 'tinyllama-marker':
      return renderTinyLlamaMarkerPrompt(messages)
    default:
      throw new Error(`unsupported --render-mode ${JSON.stringify(renderMode)}; supported modes: ${SUPPORTED_RENDER_MODES.join(', ')}`)
  }
}

export function resolveReferenceContext({ promptTokenCount, maxTokens, explicitContext = null, minimumContext = 512, headroom = 16 }) {
  const required = promptTokenCount + maxTokens + headroom
  if (Number.isInteger(explicitContext) && explicitContext > 0) {
    if (explicitContext < required) {
      throw new Error(`--llama-context ${explicitContext} is too small for ${promptTokenCount} prompt tokens + ${maxTokens} generated tokens + ${headroom} token headroom`)
    }
    return explicitContext
  }
  return Math.max(minimumContext, required)
}

function renderCompactLlama3Prompt(messages) {
  let prompt = ''
  for (const message of messages) {
    prompt += `<|start_header_id|>${message.role.trim()}<|end_header_id|>\n\n${message.content}<|eot_id|>`
  }
  if (messages.at(-1)?.role.trim() !== 'assistant') {
    prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n'
  }
  return prompt
}

function renderTinyLlamaMarkerPrompt(messages) {
  let prompt = ''
  for (const message of messages) {
    prompt += `<|${message.role.trim()}|>\n${message.content}</s>\n`
  }
  if (messages.at(-1)?.role.trim() !== 'assistant') {
    prompt += '<|assistant|>\n'
  }
  return prompt
}
