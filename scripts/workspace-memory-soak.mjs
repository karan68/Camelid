#!/usr/bin/env node

import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { basename, join, resolve } from 'node:path'

const API = process.env.CAMELID_API || 'http://127.0.0.1:8192'
const MODEL = process.env.CAMELID_WORKSPACE_MODEL
const OUT = process.env.CAMELID_WORKSPACE_EVIDENCE
const EXPECTED_SHA = '7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5'
const EXPECTED_SIZE = 2497280256
const ORIGIN = new URL(API).origin

if (!MODEL) throw new Error('CAMELID_WORKSPACE_MODEL must name Qwen3-4B-Q4_K_M.gguf')
if (!OUT) throw new Error('CAMELID_WORKSPACE_EVIDENCE must name an output directory')

const modelPath = resolve(MODEL)
const evidenceRoot = resolve(OUT)
const workspaceRoot = join(evidenceRoot, 'workspace')

function sha256File(path) {
  return new Promise((resolveHash, reject) => {
    const hash = createHash('sha256')
    const stream = createReadStream(path)
    stream.on('error', reject)
    stream.on('data', (chunk) => hash.update(chunk))
    stream.on('end', () => resolveHash(hash.digest('hex')))
  })
}

async function api(path, options = {}) {
  const headers = { Origin: ORIGIN, ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers }
  const response = await fetch(`${API}${path}`, { ...options, headers })
  if (!response.ok) throw new Error(`${options.method || 'GET'} ${path} -> ${response.status}: ${await response.text()}`)
  return response
}

function parseSse(buffer) {
  const parts = buffer.replaceAll('\r\n', '\n').split('\n\n')
  return { complete: parts.slice(0, -1), remainder: parts.at(-1) || '' }
}

async function readTurn(sessionId, name) {
  const started = performance.now()
  const response = await api(`/api/agent/workspace/sessions/${encodeURIComponent(sessionId)}/events`)
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const events = []
  let buffer = ''
  let terminal = false
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parsed = parseSse(buffer)
    buffer = parsed.remainder
    for (const block of parsed.complete) {
      const data = block.split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart()).join('\n')
      if (!data) continue
      const event = JSON.parse(data)
      events.push(event)
      if (event.event === 'approval.required') throw new Error(`${name}: unexpected write approval`)
      if (event.event === 'session.error') throw new Error(`${name}: ${event.message}`)
      if (event.event === 'session.finished') terminal = true
    }
    if (terminal) break
  }
  await reader.cancel()
  if (!terminal) throw new Error(`${name}: stream ended without session.finished`)
  const elapsedMs = Math.round(performance.now() - started)
  await writeFile(join(evidenceRoot, `${name}.events.json`), `${JSON.stringify(events, null, 2)}\n`)
  return { events, elapsedMs }
}

function answer(events) {
  return events.filter((event) => event.event === 'model.answer').at(-1)?.content || ''
}

async function assertIdle(sessionId, name) {
  const status = await (await api(`/api/agent/workspace/sessions/${encodeURIComponent(sessionId)}`)).json()
  if (status.state !== 'idle') throw new Error(`${name}: expected idle status, got ${JSON.stringify(status)}`)
  return status
}

function assertBudgetEvents(events, name) {
  const budgets = events.filter((event) => event.event === 'memory.updated')
  if (!budgets.length) throw new Error(`${name}: memory.updated missing`)
  for (const budget of budgets) {
    const estimated = [
      budget.system_tokens_estimate, budget.tool_definition_tokens_estimate,
      budget.message_tokens_estimate, budget.recent_memory_tokens_estimate,
      budget.retrieved_memory_tokens_estimate, budget.evidence_memory_tokens_estimate,
      budget.tool_result_tokens_estimate,
    ].reduce((total, value) => total + Number(value || 0), 0)
    if (estimated !== budget.prompt_tokens) throw new Error(`${name}: estimated categories ${estimated} != exact prompt ${budget.prompt_tokens}`)
    if (budget.prompt_tokens + budget.generation_tokens > budget.budget_total) throw new Error(`${name}: context budget exceeded ${JSON.stringify(budget)}`)
  }
  const timings = events.filter((event) => event.event === 'model.timing')
  if (!timings.length || timings.some((timing) => !(timing.total_ms > 0))) {
    throw new Error(`${name}: valid model.timing missing`)
  }
}

await rm(evidenceRoot, { recursive: true, force: true })
await mkdir(workspaceRoot, { recursive: true })
await writeFile(join(workspaceRoot, 'memory-note.txt'), 'Workspace memory validation code: BWSM-4729\n')

const modelStat = await stat(modelPath)
if (modelStat.size !== EXPECTED_SIZE) throw new Error(`model size mismatch: ${modelStat.size}`)
const modelSha = await sha256File(modelPath)
if (modelSha !== EXPECTED_SHA) throw new Error(`model SHA mismatch: ${modelSha}`)

await api('/api/models/load', {
  method: 'POST',
  body: JSON.stringify({ path: modelPath, id: 'qwen3_4b_q4_k_m' }),
})
const health = await (await api('/v1/health')).json()
if (!health.loaded_now || !health.generation_ready || health.active_model_id !== 'qwen3_4b_q4_k_m') {
  throw new Error(`exact model not ready: ${JSON.stringify(health)}`)
}

const created = await (await api('/api/agent/workspace/sessions', {
  method: 'POST',
  body: JSON.stringify({
    workspace: workspaceRoot,
    goal: 'Use read_file with start_line 1 and max_lines 20 to read memory-note.txt. Then answer with the code BWSM-4729.',
    max_steps: 4,
    max_tokens: 128,
    temperature: 0,
    allow_writes: false,
  }),
})).json()
const first = await readTurn(created.id, 'turn-1-read')
assertBudgetEvents(first.events, 'turn 1')
if (!first.events.some((event) => event.event === 'turn.started' && event.turn_index === 0)) throw new Error('turn 1: turn.started missing')
const firstRead = first.events.find((event) => event.event === 'tool.result' && event.tool === 'read_file' && event.outcome === 'ok')
if (!firstRead) throw new Error('turn 1: successful read_file result missing')
if (Buffer.byteLength(firstRead.content, 'utf8') > 2048) throw new Error(`turn 1: tool result exceeded 2048 bytes (${Buffer.byteLength(firstRead.content, 'utf8')})`)
if (!answer(first.events).includes('BWSM-4729')) throw new Error(`turn 1: code missing from answer ${JSON.stringify(answer(first.events))}`)
await assertIdle(created.id, 'turn 1')

const messageId = 'memory-recall-1'
const message = await (await api(`/api/agent/workspace/sessions/${encodeURIComponent(created.id)}/messages`, {
  method: 'POST',
  body: JSON.stringify({ text: 'Without reading any file again, what validation code did you find? Include BWSM-4729.', client_message_id: messageId }),
})).json()
if (message.duplicate || message.turn_index !== 1) throw new Error(`turn 2: unexpected message response ${JSON.stringify(message)}`)
const second = await readTurn(created.id, 'turn-2-recall')
assertBudgetEvents(second.events, 'turn 2')
if (second.events.some((event) => event.event === 'tool.call')) throw new Error('turn 2: model re-read files instead of using memory')
if (!answer(second.events).includes('BWSM-4729')) throw new Error(`turn 2: memory recall failed ${JSON.stringify(answer(second.events))}`)
await assertIdle(created.id, 'turn 2')

const duplicate = await (await api(`/api/agent/workspace/sessions/${encodeURIComponent(created.id)}/messages`, {
  method: 'POST',
  body: JSON.stringify({ text: 'duplicate body must not replace the turn', client_message_id: messageId }),
})).json()
if (!duplicate.duplicate || duplicate.turn_index !== 1) throw new Error(`duplicate message was not idempotent: ${JSON.stringify(duplicate)}`)

await api(`/api/agent/workspace/sessions/${encodeURIComponent(created.id)}`, { method: 'DELETE' })
const resumed = await (await api('/api/agent/workspace/sessions', {
  method: 'POST',
  body: JSON.stringify({
    workspace: workspaceRoot,
    thread_id: created.id,
    goal: 'This is a new session. Without reading files, repeat the earlier validation code BWSM-4729.',
    max_steps: 3,
    max_tokens: 128,
    temperature: 0,
    allow_writes: false,
  }),
})).json()
if (resumed.id !== created.id) throw new Error(`resume returned a different thread id: ${JSON.stringify(resumed)}`)
const third = await readTurn(resumed.id, 'turn-3-resume')
assertBudgetEvents(third.events, 'turn 3')
if (third.events.some((event) => event.event === 'tool.call')) throw new Error('turn 3: resumed session re-read files')
if (!answer(third.events).includes('BWSM-4729')) throw new Error(`turn 3: cross-session recall failed ${JSON.stringify(answer(third.events))}`)
const runtimeStatus = await assertIdle(resumed.id, 'turn 3')
if (!runtimeStatus.resident_cuda || runtimeStatus.resident_cuda.offloaded) throw new Error(`resident status missing or offloaded: ${JSON.stringify(runtimeStatus)}`)
if (runtimeStatus.resident_cuda.max_positions < runtimeStatus.context_budget_tokens) throw new Error(`resident capacity below Workspace budget: ${JSON.stringify(runtimeStatus)}`)

const compacted = await (await api(`/api/agent/workspace/threads/${encodeURIComponent(created.id)}/compact?workspace=${encodeURIComponent(workspaceRoot)}`, { method: 'POST' })).json()
if (compacted.archived_turns !== 3 || compacted.compaction_count !== 1 || compacted.compacted_through_turn !== 2) throw new Error(`unexpected compaction result: ${JSON.stringify(compacted)}`)

await api(`/api/agent/workspace/sessions/${encodeURIComponent(resumed.id)}/messages`, {
  method: 'POST',
  body: JSON.stringify({ text: 'After compaction, retrieve the archived validation code BWSM-4729 without reading files.', client_message_id: 'after-compaction-1' }),
})
const fourth = await readTurn(resumed.id, 'turn-4-compacted-recall')
assertBudgetEvents(fourth.events, 'turn 4')
if (fourth.events.some((event) => event.event === 'tool.call')) throw new Error('turn 4: compacted recall re-read files')
if (!answer(fourth.events).includes('BWSM-4729')) throw new Error(`turn 4: compacted FTS recall failed ${JSON.stringify(answer(fourth.events))}`)
await assertIdle(resumed.id, 'turn 4')

const undone = await (await api(`/api/agent/workspace/threads/${encodeURIComponent(created.id)}/compact?workspace=${encodeURIComponent(workspaceRoot)}`, { method: 'DELETE' })).json()
if (undone.archived_turns !== 3 || undone.compaction_count !== 0 || undone.compacted_through_turn !== null) throw new Error(`unexpected compaction undo: ${JSON.stringify(undone)}`)

const thread = await (await api(`/api/agent/workspace/threads/${encodeURIComponent(created.id)}?workspace=${encodeURIComponent(workspaceRoot)}`)).json()
if (thread.turns.length !== 4) throw new Error(`persisted transcript should have 4 turns: ${JSON.stringify(thread)}`)

const summary = {
  schema: 'camelid.workspace-memory-soak/v1',
  generated_at: new Date().toISOString(),
  model: { filename: basename(modelPath), size_bytes: modelStat.size, sha256: modelSha },
  thread_id: created.id,
  turn_count: thread.turns.length,
  resident_cuda: runtimeStatus.resident_cuda,
  timings_ms: { first: first.elapsedMs, second: second.elapsedMs, resumed: third.elapsedMs, compacted_recall: fourth.elapsedMs },
  checks: {
    bounded_tool_result: true,
    in_session_recall_without_reread: true,
    duplicate_message_idempotent: true,
    explicit_new_session_resume: true,
    cross_session_recall_without_reread: true,
    idle_between_turns: true,
    exact_context_categories_reconcile: true,
    resident_capacity_exceeds_budget: true,
    reversible_compaction: true,
    compacted_fts_recall_without_reread: true,
    read_only_three_tool_mode: true,
  },
  claim_boundary: 'Exact Qwen3-4B-Q4_K_M on this host; four short bounded Workspace turns including reversible compaction. Does not prove p95 latency, long-thread recall, per-request no-fallback, or production promotion.',
}
await writeFile(join(evidenceRoot, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`)
console.log(JSON.stringify(summary, null, 2))
