// Keep source and tool output in explicit sidebar disclosures. Also handles an
// unfinished fence, so streaming or truncated code cannot leak into the thread.
export function describeCodingMessage(value) {
  const lines = String(value || '').split('\n')
  const prose = [], snippets = []
  let block = null
  for (const line of lines) {
    const fence = line.match(/^\s{0,3}(`{3,}|~{3,})(.*)$/)
    if (block?.envelope) {
      block.content += line + '\n'
      if (line.includes('</tool_call>')) { snippets.push(block); block = null }
    } else if (block?.fence) {
      if (fence && fence[1][0] === block.fence[0] && fence[1].length >= block.fence.length && !fence[2].trim()) {
        snippets.push(block); block = null
      } else block.content += line + '\n'
    } else if (line.trimStart().startsWith('<tool_call>')) {
      if (block) snippets.push(block)
      block = { envelope: true, language: 'Tool request', content: line + '\n' }
      if (line.includes('</tool_call>')) { snippets.push(block); block = null }
    } else if (fence) {
      if (block) snippets.push(block)
      block = { fence: fence[1], language: fence[2].trim(), content: '' }
    } else if (/^( {4}|\t)\S/.test(line) || (block && /^( {4}|\t)/.test(line))) {
      block ||= { language: '', content: '' }
      block.content += line.replace(/^( {4}|\t)/, '') + '\n'
    } else {
      if (block) { snippets.push(block); block = null }
      prose.push(line)
    }
  }
  if (block) snippets.push(block)
  return { description: prose.join('\n').trim(), snippets }
}

const actions = {
  list_dir: 'Exploring the project', read_file: 'Reading project files', search: 'Searching the project',
  write_file: 'Preparing a new file', edit_file: 'Preparing a file change', run_shell: 'Running an approved command',
  update_plan: 'Updating the work plan', spawn_subagent: 'Assigning an investigation', check_subagent_status: 'Collecting helper findings',
}
export function describeCodingAction(value, status) {
  if (status === 'waiting_approval') return 'Waiting for your review'
  if (status === 'done') return 'Finished this assignment'
  if (status === 'failed') return 'Stopped with an error'
  if (status === 'cancelled') return 'Stopped'
  const tool = String(value || '').match(/^([a-z_]+)\(/)?.[1]
  return actions[tool] || (tool ? 'Using project tools' : 'Planning the next step')
}
export function describeCodingEvent(event) {
  if (event.kind === 'tool.result') return event.detail?.ok ? ({ read_file: 'Read a project file', list_dir: 'Explored the project', search: 'Finished a search', write_file: 'Applied a new file', edit_file: 'Applied a file change', run_shell: 'Finished an approved command', update_plan: 'Updated the work plan', spawn_subagent: 'Assigned an investigation', check_subagent_status: 'Checked helper progress' })[event.detail?.tool] || 'Finished a project action' : 'An action needs attention'
  if (event.kind === 'tool.call') return describeCodingAction(event.detail?.detail)
  if (event.kind === 'review.updated') return ({ pending: 'Prepared a file for review', applied: 'Applied an approved change', rejected: 'Respected a denied change', undone: 'Restored a file' })[event.detail?.status] || 'Updated a file review'
  return ({ 'agent.file': 'Working with a project file', 'agent.assigned': 'Assigned an investigation', 'agent.working': 'Planning the next step', 'agent.failed': 'Assignment stopped with an error', 'agent.notice': 'Recorded a work update', 'run.started': 'Started this turn', 'run.control': 'Updated run controls', 'review.error': 'A file review needs attention', 'plan.updated': 'Updated the work plan', 'model.progress': 'Continuing the requested work', 'model.answer': 'Shared a result', 'run.finished': 'Turn finished', 'run.stalled': 'Work stopped before completion', 'approval.required': 'Requested your approval', 'approval.decided': 'Recorded your decision', 'agent.started': 'Started an assignment', 'agent.finished': 'Finished an assignment' })[event.kind] || 'Updated the session'
}
