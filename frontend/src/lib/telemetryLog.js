/* Session telemetry store (Phase 6).

   Records flow in ONLY from real traffic: the chat send path, workbench
   try-it runs, and the live health poll. There is no seeding, no synthetic
   data, and no persistence — this is operational telemetry for the current
   session, never compatibility evidence (I4), and a smoke bars the obvious
   fabrication routes.

   Prompt content is held in memory so the request log's per-session reveal
   toggle can work, but the export path is a field WHITELIST that can never
   include content or filesystem paths (I7). */

const MAX_REQUEST_RECORDS = 500
const MAX_HEALTH_RECORDS = 240

const state = {
  requests: [],
  health: [],
  counter: 0,
}
const listeners = new Set()

function emit() {
  for (const listener of listeners) listener()
}

export function subscribeTelemetry(listener) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

function pushRequest(record) {
  state.counter += 1
  state.requests.push({ id: `req-${state.counter}`, at: Date.now(), ...record })
  if (state.requests.length > MAX_REQUEST_RECORDS) state.requests.splice(0, state.requests.length - MAX_REQUEST_RECORDS)
  emit()
}

/* Chat generation: called from the real sendMessage path on completion,
   interruption, or error. */
export function recordChatGeneration({ modelId, durationMs, ttftMs, promptTokens, completionTokens, tokensPerSec, usageSource, outcome, promptText }) {
  pushRequest({
    kind: 'chat',
    endpoint: '/v1/chat/completions',
    modelId: modelId || null,
    outcome: outcome || 'ok',
    durationMs: Number.isFinite(durationMs) ? durationMs : null,
    ttftMs: Number.isFinite(ttftMs) ? ttftMs : null,
    promptTokens: Number.isFinite(promptTokens) ? promptTokens : null,
    completionTokens: Number.isFinite(completionTokens) ? completionTokens : null,
    tokensPerSec: Number.isFinite(tokensPerSec) ? tokensPerSec : null,
    usageSource: usageSource || null,
    promptText: typeof promptText === 'string' ? promptText : null,
  })
}

/* Workbench try-it: called from the real ApiWorkbench runner. */
export function recordWorkbenchRun({ endpoint, modelId, durationMs, headersMs, httpStatus, outcome }) {
  pushRequest({
    kind: 'workbench',
    endpoint: endpoint || null,
    modelId: modelId || null,
    outcome: outcome || 'ok',
    httpStatus: httpStatus || null,
    durationMs: Number.isFinite(durationMs) ? durationMs : null,
    ttftMs: Number.isFinite(headersMs) ? headersMs : null,
    promptTokens: null,
    completionTokens: null,
    tokensPerSec: null,
    promptText: null,
  })
}

/* Health poll outcome: called from the real dashboard refresh loop. */
export function recordHealthPoll({ ok, latencyMs }) {
  state.health.push({ at: Date.now(), ok: Boolean(ok), latencyMs: Number.isFinite(latencyMs) ? latencyMs : null })
  if (state.health.length > MAX_HEALTH_RECORDS) state.health.splice(0, state.health.length - MAX_HEALTH_RECORDS)
  emit()
}

export function getTelemetrySnapshot() {
  return { requests: state.requests.slice(), health: state.health.slice() }
}

/* Export whitelist (I7): time, endpoint, model, outcome, duration, token
   counts. promptText and any path-like field can never appear here. */
const EXPORT_FIELDS = ['at', 'kind', 'endpoint', 'modelId', 'outcome', 'httpStatus', 'durationMs', 'ttftMs', 'promptTokens', 'completionTokens', 'tokensPerSec', 'usageSource']

export function exportTelemetryJson() {
  const rows = state.requests.map((record) => {
    const out = {}
    for (const field of EXPORT_FIELDS) {
      if (record[field] !== undefined && record[field] !== null) out[field] = record[field]
    }
    return out
  })
  return JSON.stringify({
    format: 'camelid.telemetry-log/v1',
    note: 'Operational telemetry from one local browser session. Not compatibility or support evidence. Prompt content and file paths are excluded by construction.',
    requests: rows,
  }, null, 2)
}

/* Summaries for the dashboard tiles/sparklines — plain math over real records. */
function median(values) {
  const sorted = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b)
  if (!sorted.length) return null
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

export function summarizeTelemetry(requests) {
  const total = requests.length
  const errors = requests.filter((r) => r.outcome !== 'ok').length
  return {
    total,
    errors,
    errorRate: total ? errors / total : null,
    medianTtftMs: median(requests.map((r) => r.ttftMs)),
    medianDurationMs: median(requests.map((r) => r.durationMs)),
    medianTokensPerSec: median(requests.map((r) => r.tokensPerSec)),
  }
}

export function perModelBreakdown(requests) {
  const byModel = new Map()
  for (const record of requests) {
    if (!record.modelId) continue
    if (!byModel.has(record.modelId)) byModel.set(record.modelId, [])
    byModel.get(record.modelId).push(record)
  }
  return [...byModel.entries()].map(([modelId, records]) => ({ modelId, ...summarizeTelemetry(records) }))
}
