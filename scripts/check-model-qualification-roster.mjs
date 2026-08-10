#!/usr/bin/env node
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const REQUIRED_GATES = [
  'source',
  'metadata',
  'tokenizer',
  'template',
  'load_smoke',
  'parity',
  'api_webui',
  'context',
]
const GATE_STATUSES = new Set(['pass', 'fail', 'blocked', 'pending'])
const DISPOSITIONS = new Set(['active_validation', 'hold', 'backlog', 'blocked', 'promotion_candidate'])
const TARGET_TIERS = new Set(['experimental_exact_row'])
const SHA256_RE = /^[0-9a-f]{64}$/
const REVISION_RE = /^[0-9a-f]{40}$/
const ID_RE = /^[a-z0-9][a-z0-9_]*$/

function validateRoster(roster, label = 'roster') {
  const errors = []
  const fail = (path, message) => errors.push(`${label}.${path}: ${message}`)

  if (roster?.schema !== 'camelid.model-qualification-roster/v1') {
    fail('schema', 'expected camelid.model-qualification-roster/v1')
  }
  if (roster?.phase?.id !== 1) fail('phase.id', 'expected integer 1')
  if (roster?.phase?.status !== 'active') fail('phase.status', 'expected active')
  if (roster?.defaults?.models_dir_env !== 'CAMELID_MODELS_DIR') {
    fail('defaults.models_dir_env', 'use the runtime-standard plural CAMELID_MODELS_DIR')
  }
  if (!Array.isArray(roster?.gate_order) || JSON.stringify(roster.gate_order) !== JSON.stringify(REQUIRED_GATES)) {
    fail('gate_order', `expected exact fail-closed order ${REQUIRED_GATES.join(', ')}`)
  }
  if (!Array.isArray(roster?.rows) || roster.rows.length === 0) {
    fail('rows', 'expected at least one row')
    return errors
  }

  const ids = new Set()
  const artifacts = new Set()
  const priorities = []

  roster.rows.forEach((row, index) => {
    const base = `rows[${index}]`
    if (!Number.isInteger(row.priority) || row.priority < 1) fail(`${base}.priority`, 'expected a positive integer')
    else priorities.push(row.priority)
    if (typeof row.id !== 'string' || !ID_RE.test(row.id)) fail(`${base}.id`, 'expected snake_case id')
    else if (ids.has(row.id)) fail(`${base}.id`, `duplicate id ${row.id}`)
    else ids.add(row.id)

    const identity = row.identity || {}
    for (const field of ['family', 'architecture', 'quantization']) {
      if (typeof identity[field] !== 'string' || !identity[field]) fail(`${base}.identity.${field}`, 'expected non-empty string')
    }
    if (identity.gguf_filename !== null && (typeof identity.gguf_filename !== 'string' || !identity.gguf_filename.toLowerCase().endsWith('.gguf'))) {
      fail(`${base}.identity.gguf_filename`, 'expected a .gguf filename or null')
    }
    if (identity.size_bytes !== null && (!Number.isInteger(identity.size_bytes) || identity.size_bytes <= 0)) {
      fail(`${base}.identity.size_bytes`, 'expected positive integer or null')
    }
    if (identity.sha256 !== null && (typeof identity.sha256 !== 'string' || !SHA256_RE.test(identity.sha256))) {
      fail(`${base}.identity.sha256`, 'expected lowercase 64-character SHA-256 or null')
    }
    if (identity.gguf_filename) {
      const artifactKey = `${identity.architecture}/${identity.quantization}/${identity.gguf_filename.toLowerCase()}`
      if (artifacts.has(artifactKey)) fail(`${base}.identity`, `duplicate exact artifact ${artifactKey}`)
      artifacts.add(artifactKey)
    }

    const source = row.source || {}
    if (source.revision !== null && (typeof source.revision !== 'string' || !REVISION_RE.test(source.revision))) {
      fail(`${base}.source.revision`, 'expected lowercase 40-character git revision or null')
    }
    if (source.file !== null && typeof source.file !== 'string') fail(`${base}.source.file`, 'expected string or null')
    if (source.repo !== null && typeof source.repo !== 'string') fail(`${base}.source.repo`, 'expected string or null')

    if (!TARGET_TIERS.has(row.target_tier)) fail(`${base}.target_tier`, `unsupported target tier ${JSON.stringify(row.target_tier)}`)
    if (!DISPOSITIONS.has(row.disposition)) fail(`${base}.disposition`, `unsupported disposition ${JSON.stringify(row.disposition)}`)

    const gates = row.gates || {}
    const gateKeys = Object.keys(gates)
    for (const name of REQUIRED_GATES) {
      const gate = gates[name]
      if (!gate || typeof gate !== 'object' || Array.isArray(gate)) {
        fail(`${base}.gates.${name}`, 'missing gate object')
        continue
      }
      if (typeof gate.required !== 'boolean') fail(`${base}.gates.${name}.required`, 'expected boolean')
      if (!GATE_STATUSES.has(gate.status)) fail(`${base}.gates.${name}.status`, `unsupported status ${JSON.stringify(gate.status)}`)
      const evidenceIsValid = Array.isArray(gate.evidence)
        && gate.evidence.every((item) => typeof item === 'string' && item)
      if (!evidenceIsValid) {
        fail(`${base}.gates.${name}.evidence`, 'expected an array of non-empty strings')
      }
      if (gate.status === 'pass' && evidenceIsValid && gate.evidence.length === 0) {
        fail(`${base}.gates.${name}`, 'a passing gate needs durable evidence')
      }
      if ((gate.status === 'fail' || gate.status === 'blocked') && (typeof gate.reason !== 'string' || !gate.reason.trim())) {
        fail(`${base}.gates.${name}.reason`, `${gate.status} gates need a concrete reason`)
      }
    }
    for (const extra of gateKeys.filter((key) => !REQUIRED_GATES.includes(key))) {
      fail(`${base}.gates.${extra}`, 'unexpected gate')
    }

    if (gates.source?.status === 'pass') {
      for (const field of ['repo', 'file', 'revision', 'license']) {
        if (typeof source[field] !== 'string' || !source[field]) fail(`${base}.source.${field}`, 'source-pass rows must pin this field')
      }
      if (!identity.sha256 || !identity.size_bytes) fail(`${base}.identity`, 'source-pass rows must pin size_bytes and sha256')
    }

    if (row.disposition === 'promotion_candidate') {
      const incomplete = REQUIRED_GATES.filter((name) => gates[name]?.required && gates[name]?.status !== 'pass')
      if (incomplete.length) fail(`${base}.disposition`, `promotion_candidate has incomplete gates: ${incomplete.join(', ')}`)
    }
  })

  const expectedPriorities = Array.from({ length: roster.rows.length }, (_, i) => i + 1)
  if (JSON.stringify([...priorities].sort((a, b) => a - b)) !== JSON.stringify(expectedPriorities)) {
    fail('rows', `priorities must be unique and contiguous 1..${roster.rows.length}`)
  }
  return errors
}

function summarizeRoster(roster) {
  return roster.rows.map((row) => {
    const counts = { pass: 0, fail: 0, blocked: 0, pending: 0 }
    for (const gate of Object.values(row.gates)) counts[gate.status] += 1
    return {
      priority: row.priority,
      id: row.id,
      disposition: row.disposition,
      gates: counts,
      next_gate: roster.gate_order.find((name) => row.gates[name].status !== 'pass') || null,
    }
  })
}

async function main() {
  const manifestPath = resolve(process.argv[2] || 'qa/model-qualification/phase1-roster.json')
  let roster
  try {
    roster = JSON.parse(await readFile(manifestPath, 'utf8'))
  } catch (error) {
    console.error(`model qualification roster check FAILED: ${manifestPath}: ${error.message}`)
    process.exit(1)
  }
  const errors = validateRoster(roster, manifestPath)
  if (errors.length) {
    console.error(`model qualification roster check FAILED (${errors.length}):`)
    for (const error of errors) console.error(`  - ${error}`)
    process.exit(1)
  }
  console.log(JSON.stringify({
    schema: roster.schema,
    phase: roster.phase,
    rows: summarizeRoster(roster),
  }, null, 2))
}

export { REQUIRED_GATES, validateRoster, summarizeRoster }

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
