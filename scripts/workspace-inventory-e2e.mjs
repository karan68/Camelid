#!/usr/bin/env node

import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, rm, stat, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'

const API = process.env.CAMELID_API || 'http://127.0.0.1:8192'
const MODEL = resolve(process.env.CAMELID_WORKSPACE_MODEL || '')
const OUT = resolve(process.env.CAMELID_WORKSPACE_EVIDENCE || '')
const ORIGIN = new URL(API).origin
const EXPECTED_SHA = '7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5'
const EXPECTED_SIZE = 2497280256
const FILES = [
  'CONFIGURATION.md',
  'CONFORMANCE.md',
  'CONTEXT.md',
  'CONTRIBUTOR_QUICKSTART.md',
  'TELEMETRY.md',
  'VALIDATION_MATRIX.md',
  'WAR_ROOM_EVIDENCE_INDEX.md',
  'gemma4-cuda-port-plan.md',
  'gemma4-cuda-q4_0-plan.md',
  'gemma4-engine-status.md',
  'gemma4-gpu-port-plan.md',
  'gemma4-row-audit-2026-06-09.md',
  'gemma4-two-mac-cluster.md',
  'housekeeping-check.md',
]
const EXPECTED_FILES = [...FILES].sort((left, right) => {
  const a = left.toLowerCase()
  const b = right.toLowerCase()
  return a < b ? -1 : a > b ? 1 : 0
})
if (!process.env.CAMELID_WORKSPACE_MODEL || !process.env.CAMELID_WORKSPACE_EVIDENCE) {
  throw new Error('CAMELID_WORKSPACE_MODEL and CAMELID_WORKSPACE_EVIDENCE are required')
}

function sha256File(path) {
  return new Promise((resolveHash, reject) => {
    const hash = createHash('sha256')
    const stream = createReadStream(path)
    stream.on('error', reject)
    stream.on('data', (chunk) => hash.update(chunk))
    stream.on('end', () => resolveHash(hash.digest('hex')))
  })
}

async function api(path, options = {}, expectedStatus = null) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      Origin: ORIGIN,
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...options.headers,
    },
  })
  if (expectedStatus !== null) {
    if (response.status !== expectedStatus) {
      throw new Error(`${options.method || 'GET'} ${path} -> ${response.status}, expected ${expectedStatus}: ${await response.text()}`)
    }
  } else if (!response.ok) {
    throw new Error(`${options.method || 'GET'} ${path} -> ${response.status}: ${await response.text()}`)
  }
  return response
}

function parseSse(buffer) {
  const parts = buffer.replaceAll('\r\n', '\n').split('\n\n')
  return { complete: parts.slice(0, -1), remainder: parts.at(-1) || '' }
}

async function readTurn(sessionId) {
  const response = await api(`/api/agent/workspace/sessions/${encodeURIComponent(sessionId)}/events`)
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const events = []
  let buffer = ''
  let terminal = false
  while (!terminal) {
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
      if (event.event === 'approval.required') throw new Error('read-only inventory requested write approval')
      if (event.event === 'session.error') throw new Error(event.message)
      if (event.event === 'session.finished') terminal = true
    }
  }
  await reader.cancel()
  if (!terminal) throw new Error('inventory stream ended without session.finished')
  return events
}

await rm(OUT, { recursive: true, force: true })
const workspace = join(OUT, 'workspace')
await mkdir(join(workspace, 'architecture'), { recursive: true })
await mkdir(join(workspace, 'archive'), { recursive: true })
await Promise.all(FILES.map((file) => writeFile(join(workspace, file), `# ${file}\nGrounded inventory fixture.\n`)))
await writeFile(join(workspace, 'notes.txt'), 'not markdown\n')

const modelStat = await stat(MODEL)
const modelSha = await sha256File(MODEL)
if (modelStat.size !== EXPECTED_SIZE || modelSha !== EXPECTED_SHA) {
  throw new Error(`exact model identity mismatch: size=${modelStat.size} sha=${modelSha}`)
}
await api('/api/models/load', {
  method: 'POST',
  body: JSON.stringify({ path: MODEL, id: 'qwen3_4b_q4_k_m' }),
})
const health = await (await api('/v1/health')).json()
if (!health.loaded_now || !health.generation_ready || health.active_model_id !== 'qwen3_4b_q4_k_m') {
  throw new Error(`exact model is not ready: ${JSON.stringify(health)}`)
}

const rejected = await api('/api/agent/workspace/sessions', {
  method: 'POST',
  body: JSON.stringify({ workspace, goal: 'edit files', allow_writes: true }),
}, 400)
const rejection = await rejected.json()
if (rejection?.error?.code !== 'workspace_read_only') {
  throw new Error(`write mode did not fail closed: ${JSON.stringify(rejection)}`)
}

const created = await (await api('/api/agent/workspace/sessions', {
  method: 'POST',
  body: JSON.stringify({
    workspace,
    goal: 'check all the md files in this folder',
    max_steps: 4,
    max_tokens: 512,
    temperature: 0,
    allow_writes: false,
  }),
})).json()
const events = await readTurn(created.id)
const listCalls = events.filter((event) => event.event === 'tool.call' && String(event.detail || '').startsWith('list_dir('))
const successfulListings = events.filter((event) => event.event === 'tool.result' && event.tool === 'list_dir' && event.outcome === 'ok')
const prohibitedTools = events.filter((event) => event.event === 'tool.call' && /^(write_file|edit_file)\(/.test(String(event.detail || '')))
if (!listCalls.length || !successfulListings.length) throw new Error('successful list_dir evidence is missing')
if (prohibitedTools.length) throw new Error(`prohibited write tools were called: ${JSON.stringify(prohibitedTools)}`)

const answer = events.filter((event) => event.event === 'model.answer').at(-1)?.content || ''
if (!answer.startsWith(`Found ${FILES.length} Markdown files in the selected folder:`)) {
  throw new Error(`canonical inventory heading mismatch: ${JSON.stringify(answer)}`)
}
for (const file of EXPECTED_FILES) {
  if (!answer.includes(`- \`${file}\``)) throw new Error(`inventory omitted ${file}: ${JSON.stringify(answer)}`)
}
if (!answer.includes('Nested folders were not searched.')) throw new Error('inventory scope disclosure is missing')
const answerFiles = [...answer.matchAll(/^- `([^`]+)`$/gm)].map((match) => match[1])
if (JSON.stringify(answerFiles) !== JSON.stringify(EXPECTED_FILES)) {
  throw new Error(`inventory exact-set mismatch: ${JSON.stringify(answerFiles)}`)
}

const thread = await (await api(`/api/agent/workspace/threads/${encodeURIComponent(created.id)}?workspace=${encodeURIComponent(workspace)}`)).json()
if (thread.turns.length !== 1 || thread.turns[0].assistant_text !== answer) {
  throw new Error(`persisted transcript mismatch: ${JSON.stringify(thread)}`)
}
if (!events.some((event) => event.event === 'memory.updated' && event.prompt_tokens + event.generation_tokens <= event.budget_total)) {
  throw new Error('bounded context event is missing')
}
const status = await (await api(`/api/agent/workspace/sessions/${encodeURIComponent(created.id)}`)).json()
if (status.state !== 'idle' || status.allow_writes !== false) throw new Error(`terminal read-only state mismatch: ${JSON.stringify(status)}`)

const summary = {
  schema: 'camelid.workspace-inventory-e2e/v1',
  generated_at: new Date().toISOString(),
  model: { size_bytes: modelStat.size, sha256: modelSha },
  thread_id: created.id,
  expected_files: EXPECTED_FILES,
  tool_calls: events.filter((event) => event.event === 'tool.call').map((event) => event.detail),
  answer,
  checks: {
    write_mode_rejected: true,
    successful_list_dir_required: true,
    deterministic_file_count: true,
    directories_excluded: true,
    nonmatching_files_excluded: true,
    scope_disclosed: true,
    transcript_persisted: true,
    context_bounded: true,
    terminal_read_only_idle: true,
  },
  claim_boundary: 'One exact-model immediate-folder Markdown inventory run. Recursive discovery and content review are separate request classes.',
}
await writeFile(join(OUT, 'events.json'), `${JSON.stringify(events, null, 2)}\n`)
await writeFile(join(OUT, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`)
console.log(JSON.stringify(summary, null, 2))
