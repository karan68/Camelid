#!/usr/bin/env node

import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, rm, stat, writeFile } from 'node:fs/promises'
import { resolve, join } from 'node:path'

const API = process.env.CAMELID_API || 'http://127.0.0.1:8192'
const MODEL = resolve(process.env.CAMELID_WORKSPACE_MODEL || '')
const OUT = resolve(process.env.CAMELID_WORKSPACE_EVIDENCE || '')
const EXPECTED_SHA = '7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5'
const EXPECTED_SIZE = 2497280256
const ORIGIN = new URL(API).origin
if (!process.env.CAMELID_WORKSPACE_MODEL || !process.env.CAMELID_WORKSPACE_EVIDENCE) throw new Error('model and evidence env vars are required')

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
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { Origin: ORIGIN, ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
  })
  if (!response.ok) throw new Error(`${options.method || 'GET'} ${path} -> ${response.status}: ${await response.text()}`)
  return response
}

function parseSse(buffer) {
  const parts = buffer.replaceAll('\r\n', '\n').split('\n\n')
  return { complete: parts.slice(0, -1), remainder: parts.at(-1) || '' }
}

async function readTurn(sessionId) {
  const started = performance.now()
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
      if (event.event === 'approval.required') throw new Error('unexpected approval in read-only sustained run')
      if (event.event === 'session.error') throw new Error(event.message)
      if (event.event === 'session.finished') terminal = true
    }
  }
  await reader.cancel()
  if (!terminal) throw new Error('stream ended without terminal event')
  return { events, elapsed_ms: Math.round(performance.now() - started) }
}

function percentile(values, fraction) {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)]
}

await rm(OUT, { recursive: true, force: true })
const workspace = join(OUT, 'workspace')
await mkdir(workspace, { recursive: true })
await writeFile(join(workspace, 'memory-note.txt'), 'Sustained memory code: LONG-9081\n')
const modelStat = await stat(MODEL)
if (modelStat.size !== EXPECTED_SIZE || await sha256File(MODEL) !== EXPECTED_SHA) throw new Error('exact model identity mismatch')
await api('/api/models/load', { method: 'POST', body: JSON.stringify({ path: MODEL, id: 'qwen3_4b_q4_k_m' }) })

const created = await (await api('/api/agent/workspace/sessions', {
  method: 'POST',
  body: JSON.stringify({
    workspace,
    goal: 'Read memory-note.txt with read_file and answer with LONG-9081.',
    max_steps: 3,
    max_tokens: 96,
    temperature: 0,
    allow_writes: false,
  }),
})).json()
const initial = await readTurn(created.id)
if (!initial.events.some((event) => event.event === 'tool.result' && event.tool === 'read_file')) throw new Error('initial read missing')
if (!initial.events.some((event) => event.event === 'model.answer' && event.content.includes('LONG-9081'))) throw new Error('initial answer missing code')

const turns = []
for (let index = 1; index <= 10; index += 1) {
  const text = index % 2
    ? `Memory check ${index}: reply with LONG-9081 and do not read files.`
    : `Unique follow-up ${index}: what sustained code was established earlier? Include LONG-9081 without tools.`
  await api(`/api/agent/workspace/sessions/${encodeURIComponent(created.id)}/messages`, {
    method: 'POST',
    body: JSON.stringify({ text, client_message_id: `sustained-${index}` }),
  })
  const turn = await readTurn(created.id)
  if (turn.events.some((event) => event.event === 'tool.call')) throw new Error(`turn ${index}: unexpected tool call`)
  const answer = turn.events.findLast((event) => event.event === 'model.answer')?.content || ''
  if (!answer.includes('LONG-9081')) throw new Error(`turn ${index}: recall failed: ${answer}`)
  const budget = turn.events.findLast((event) => event.event === 'memory.updated')
  if (!budget || budget.prompt_tokens + budget.generation_tokens > budget.budget_total) throw new Error(`turn ${index}: budget invariant failed`)
  const timing = turn.events.findLast((event) => event.event === 'model.timing')
  if (!timing || !(timing.total_ms > 0)) throw new Error(`turn ${index}: timing missing`)
  const status = await (await api(`/api/agent/workspace/sessions/${encodeURIComponent(created.id)}`)).json()
  if (status.state !== 'idle') throw new Error(`turn ${index}: not idle`)
  turns.push({ index, elapsed_ms: turn.elapsed_ms, model_total_ms: timing.total_ms, ttft_ms: timing.ttft_ms, prompt_tokens: budget.prompt_tokens })
}

const thread = await (await api(`/api/agent/workspace/threads/${encodeURIComponent(created.id)}?workspace=${encodeURIComponent(workspace)}`)).json()
if (thread.turns.length !== 11) throw new Error(`expected 11 persisted turns, got ${thread.turns.length}`)
const elapsed = turns.map((turn) => turn.elapsed_ms)
const model = turns.map((turn) => turn.model_total_ms)
const ttft = turns.map((turn) => turn.ttft_ms).filter((value) => value != null)
const summary = {
  schema: 'camelid.workspace-memory-sustained/v1',
  generated_at: new Date().toISOString(),
  model: { size_bytes: modelStat.size, sha256: EXPECTED_SHA },
  thread_id: created.id,
  persisted_turns: thread.turns.length,
  followup_turns: turns,
  metrics: {
    elapsed_ms: { p50: percentile(elapsed, 0.5), p95: percentile(elapsed, 0.95), max: Math.max(...elapsed) },
    model_total_ms: { p50: percentile(model, 0.5), p95: percentile(model, 0.95), max: Math.max(...model) },
    ttft_ms: { p50: percentile(ttft, 0.5), p95: percentile(ttft, 0.95), max: Math.max(...ttft) },
    prompt_tokens: { min: Math.min(...turns.map((turn) => turn.prompt_tokens)), max: Math.max(...turns.map((turn) => turn.prompt_tokens)) },
  },
  checks: { ten_followups_recalled: true, no_followup_tools: true, all_idle: true, budgets_bounded: true, transcript_11_turns: true },
  claim_boundary: 'One exact-model 10-followup run on this host; not a population-level hardware SLA or per-request no-fallback proof.',
}
await writeFile(join(OUT, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`)
console.log(JSON.stringify(summary, null, 2))
