// Call IDs are only unique within an assistant turn. Pair results with that
// turn's adjacent tool messages so reused IDs cannot show another turn's result.
export function toolActivityGroups(messages) {
  const groups = new Map(), pairedResults = new Set()
  messages.forEach((message, index) => {
    if (!message.mcp_managed || !message.tool_calls?.length) return
    const results = []
    for (let i = index + 1; i < messages.length && messages[i].role === 'tool'; i++) results.push(messages[i])
    const ids = message.tool_calls.map(call => call.id)
    const unique = new Set(ids).size === ids.length
    groups.set(message.id, message.tool_calls.map(call => {
      const matches = unique ? results.filter(result => result.tool_call_id === call.id) : []
      const result = matches.length === 1 ? matches[0] : null
      if (result) pairedResults.add(result.id)
      return { call, result }
    }))
  })
  return { groups, pairedResults }
}
