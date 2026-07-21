#!/usr/bin/env node

import { mkdir, rm, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'

const API = process.env.CAMELID_API || 'http://127.0.0.1:8192'
const MODEL = resolve(process.env.CAMELID_WORKSPACE_MODEL || '')
const OUT = resolve(process.env.CAMELID_WORKSPACE_EVIDENCE || '')
const ORIGIN = new URL(API).origin
if (!process.env.CAMELID_WORKSPACE_MODEL || !process.env.CAMELID_WORKSPACE_EVIDENCE) {
  throw new Error('model and evidence env vars are required')
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

await rm(OUT, { recursive: true, force: true })
const workspace = join(OUT, 'workspace')
await mkdir(workspace, { recursive: true })
await api('/api/models/load', {
  method: 'POST',
  body: JSON.stringify({ path: MODEL, id: 'qwen3_4b_q4_k_m' }),
})
const health = await (await api('/v1/health')).json()
if (!health.generation_ready || health.active_model_id !== 'qwen3_4b_q4_k_m') {
  throw new Error(`exact model not ready: ${JSON.stringify(health)}`)
}

const created = await (await api('/api/agent/workspace/sessions', {
  method: 'POST',
  body: JSON.stringify({
    workspace,
    goal: 'Do not use tools. Produce a numbered list from 1 through 1000 with one complete sentence per item.',
    max_steps: 1,
    max_tokens: 1024,
    temperature: 0,
    allow_writes: false,
  }),
})).json()

const response = await api(`/api/agent/workspace/sessions/${encodeURIComponent(created.id)}/events`)
const reader = response.body.getReader()
const decoder = new TextDecoder()
const events = []
let buffer = ''
let cancelStarted
let cancelElapsedMs
let terminal
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
    if (event.event === 'model.delta' && cancelStarted === undefined) {
      await new Promise((resolveDelay) => setTimeout(resolveDelay, 750))
      cancelStarted = performance.now()
      await api(`/api/agent/workspace/sessions/${encodeURIComponent(created.id)}`, { method: 'DELETE' })
    }
    if (event.event === 'session.error') throw new Error(event.message)
    if (event.event === 'session.finished') {
      cancelElapsedMs = Math.round(performance.now() - cancelStarted)
      terminal = event
      break
    }
  }
  if (terminal) break
}
await reader.cancel()

if (cancelStarted === undefined) throw new Error('model.delta was not observed before cancellation')
if (terminal?.outcome !== 'aborted') throw new Error(`expected aborted terminal event: ${JSON.stringify(terminal)}`)
if (cancelElapsedMs > 5000) throw new Error(`active cancellation took ${cancelElapsedMs} ms`)
const status = await (await api(`/api/agent/workspace/sessions/${encodeURIComponent(created.id)}`)).json()
if (status.state !== 'cancelled') throw new Error(`expected cancelled state: ${JSON.stringify(status)}`)
const thread = await (await api(`/api/agent/workspace/threads/${encodeURIComponent(created.id)}?workspace=${encodeURIComponent(workspace)}`)).json()
if (thread.turns.length !== 0) throw new Error(`partial cancelled turn was persisted: ${JSON.stringify(thread.turns)}`)

const preclaim = await (await api('/api/agent/workspace/sessions', {
  method: 'POST',
  body: JSON.stringify({
    workspace,
    goal: 'This turn must be cancelled before its event stream is claimed.',
    max_steps: 1,
    max_tokens: 64,
    temperature: 0,
    allow_writes: false,
  }),
})).json()
const preclaimCancelStarted = performance.now()
await api(`/api/agent/workspace/sessions/${encodeURIComponent(preclaim.id)}`, { method: 'DELETE' })
const preclaimResponse = await api(`/api/agent/workspace/sessions/${encodeURIComponent(preclaim.id)}/events`)
const preclaimReader = preclaimResponse.body.getReader()
const preclaimDecoder = new TextDecoder()
const preclaimEvents = []
let preclaimBuffer = ''
let preclaimTerminal
for (;;) {
  const { value, done } = await preclaimReader.read()
  if (done) break
  preclaimBuffer += preclaimDecoder.decode(value, { stream: true })
  const parsed = parseSse(preclaimBuffer)
  preclaimBuffer = parsed.remainder
  for (const block of parsed.complete) {
    const data = block.split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart()).join('\n')
    if (!data) continue
    const event = JSON.parse(data)
    preclaimEvents.push(event)
    if (event.event === 'session.error') throw new Error(event.message)
    if (event.event === 'session.finished') {
      preclaimTerminal = event
      break
    }
  }
  if (preclaimTerminal) break
}
await preclaimReader.cancel()
const preclaimCancelElapsedMs = Math.round(performance.now() - preclaimCancelStarted)
if (preclaimTerminal?.outcome !== 'aborted') throw new Error(`preclaim cancel did not abort: ${JSON.stringify(preclaimTerminal)}`)
if (preclaimCancelElapsedMs > 5000) throw new Error(`preclaim cancellation took ${preclaimCancelElapsedMs} ms`)
const preclaimStatus = await (await api(`/api/agent/workspace/sessions/${encodeURIComponent(preclaim.id)}`)).json()
if (preclaimStatus.state !== 'cancelled') throw new Error(`preclaim state was overwritten: ${JSON.stringify(preclaimStatus)}`)
const preclaimThread = await (await api(`/api/agent/workspace/threads/${encodeURIComponent(preclaim.id)}?workspace=${encodeURIComponent(workspace)}`)).json()
if (preclaimThread.turns.length !== 0) throw new Error(`preclaim cancelled turn was persisted: ${JSON.stringify(preclaimThread.turns)}`)

const summary = {
  schema: 'camelid.workspace-memory-cancel/v1',
  generated_at: new Date().toISOString(),
  thread_id: created.id,
  cancel_to_finish_ms: cancelElapsedMs,
  terminal_outcome: terminal.outcome,
  terminal_state: status.state,
  persisted_turns: thread.turns.length,
  preclaim_cancel_to_finish_ms: preclaimCancelElapsedMs,
  preclaim_terminal_state: preclaimStatus.state,
  preclaim_persisted_turns: preclaimThread.turns.length,
  resident_cuda: status.resident_cuda,
  checks: {
    cancelled_after_model_output_started: true,
    active_stream_finished_within_five_seconds: true,
    terminal_state_cancelled: true,
    no_partial_turn_persisted: true,
    cancel_before_event_claim_finished_within_five_seconds: true,
    cancel_before_event_claim_remained_cancelled: true,
    cancel_before_event_claim_persisted_nothing: true,
  },
  claim_boundary: 'One exact-model active-output cancellation plus one cancel-before-event-claim race on this host; the separate socket regression covers the model-step deadline before HTTP headers.',
}
await writeFile(join(OUT, 'events.json'), `${JSON.stringify(events, null, 2)}\n`)
await writeFile(join(OUT, 'preclaim-events.json'), `${JSON.stringify(preclaimEvents, null, 2)}\n`)
await writeFile(join(OUT, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`)
console.log(JSON.stringify(summary, null, 2))