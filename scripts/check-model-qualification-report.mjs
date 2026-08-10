#!/usr/bin/env node
import { readdir, readFile, stat } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const REQUIRED_STAGES = [
  'source',
  'artifact',
  'metadata',
  'tokenizer',
  'template',
  'load_smoke',
  'parity',
  'api_webui',
  'context',
]
const STATUSES = new Set(['pass', 'fail', 'blocked', 'skipped'])
const WINDOWS_ABSOLUTE_RE = /(?:^|[\s"'=([{,:])(?:[a-zA-Z]:[\\/]|\\\\)/
const UNIX_ABSOLUTE_RE = /(?:^|[\s"'=([{,:])(\/+[^\s"'<>]*)/g
const HTTP_URL_RE = /https?:\/\/[^\s"'<>]+/gi
const FILE_URI_RE = /\bfile:\/\/[^\s"'<>]*/i
const PUBLIC_ROUTE_LITERALS = new Set([
  '/apply-template',
  '/completion',
  '/health',
  '/tokenize',
  '/api/capabilities',
  '/api/models',
  '/api/models/current',
  '/api/models/inspect',
  '/api/models/load',
  '/api/models/local',
  '/v1/chat/completions',
  '/v1/completions',
  '/v1/embeddings',
  '/v1/models',
  '/v1/rerank',
])

function routeField(path) {
  return path.some((part) => /^(?:public_)?(?:route|routes|endpoint|endpoints)$/.test(String(part)))
}

function routeLiteralContext(path) {
  return routeField(path)
    || path.some((part) => /^(?:command|commands)$/.test(String(part)))
}

function containsAbsoluteLocalPath(value, path) {
  // Source/revision URLs are intentionally public evidence. Remove only HTTP(S)
  // spans before examining the remaining command/prose tokens; file:// remains
  // forbidden and is caught as a slash-absolute value.
  if (FILE_URI_RE.test(value)) return true
  const scrubbed = value.replace(HTTP_URL_RE, '<public-url>')
  if (WINDOWS_ABSOLUTE_RE.test(scrubbed)) return true

  UNIX_ABSOLUTE_RE.lastIndex = 0
  for (const match of scrubbed.matchAll(UNIX_ABSOLUTE_RE)) {
    const candidate = match[1].replace(/[),.;:\]}]+$/g, '')
    if (!candidate || /^\/+$/u.test(candidate)) continue
    if (PUBLIC_ROUTE_LITERALS.has(candidate) && routeLiteralContext(path)) continue
    if (routeField(path)
      && /^\/(?:api|v1)(?:\/[A-Za-z0-9_.:{}-]+)+\/?(?:\?[^\s]*)?$/.test(candidate)
      && !candidate.includes('..')) {
      continue
    }
    return true
  }
  return false
}

function reportContainsAbsoluteLocalPath(value, path = []) {
  if (typeof value === 'string') return containsAbsoluteLocalPath(value, path)
  if (Array.isArray(value)) {
    return value.some((item, index) => reportContainsAbsoluteLocalPath(item, [...path, index]))
  }
  if (value && typeof value === 'object') {
    return Object.entries(value)
      .some(([key, item]) => reportContainsAbsoluteLocalPath(item, [...path, key]))
  }
  return false
}

function expectedOverall(stages) {
  const required = REQUIRED_STAGES.map((name) => stages[name]).filter((stage) => stage?.required !== false)
  if (required.some((stage) => stage?.status === 'fail')) return 'fail'
  if (required.some((stage) => stage?.status === 'blocked' || stage?.status === 'skipped')) return 'blocked'
  return 'pass'
}

function validateQualificationReport(report, label = 'report') {
  const errors = []
  const fail = (path, message) => errors.push(`${label}.${path}: ${message}`)
  if (report?.schema !== 'camelid.model-qualification-report/v1') {
    fail('schema', 'expected camelid.model-qualification-report/v1')
  }
  if (typeof report?.row_id !== 'string' || !report.row_id) fail('row_id', 'expected a non-empty row id')
  if (report?.source_dirty !== null && typeof report?.source_dirty !== 'boolean') {
    fail('source_dirty', 'expected boolean or null (unknown); failed inspection must not look clean')
  }
  if (report?.host?.hostname_redacted !== true) {
    fail('host.hostname_redacted', 'host identity must be redacted')
  }
  if (Object.hasOwn(report?.host || {}, 'hostname')) fail('host.hostname', 'raw hostname is forbidden')

  const stages = report?.stages || {}
  for (const name of REQUIRED_STAGES) {
    const stage = stages[name]
    if (!stage || typeof stage !== 'object' || Array.isArray(stage)) {
      fail(`stages.${name}`, 'missing stage object')
    } else if (!STATUSES.has(stage.status)) {
      fail(`stages.${name}.status`, `unsupported status ${JSON.stringify(stage.status)}`)
    } else if ((stage.status === 'fail' || stage.status === 'blocked')
      && typeof stage.reason !== 'string'
      && !Object.hasOwn(stage, 'prompts')) {
      fail(`stages.${name}.reason`, `${stage.status} stages need a reason or detailed probe evidence`)
    }
  }
  for (const extra of Object.keys(stages).filter((name) => !REQUIRED_STAGES.includes(name))) {
    fail(`stages.${extra}`, 'unexpected stage')
  }
  if (STATUSES.has(report?.overall_status)) {
    const expected = expectedOverall(stages)
    if (report.overall_status !== expected) {
      fail('overall_status', `${report.overall_status} does not match fail-closed stage result ${expected}`)
    }
  } else {
    fail('overall_status', `unsupported status ${JSON.stringify(report?.overall_status)}`)
  }

  if (reportContainsAbsoluteLocalPath(report)) {
    fail('privacy', 'report contains an absolute local path')
  }
  return errors
}

async function collectReportPaths(inputs) {
  const requested = inputs.length ? inputs.map((path) => resolve(path)) : [resolve('qa/model-qualification')]
  const paths = []
  for (const path of requested) {
    const info = await stat(path)
    if (info.isDirectory()) {
      const entries = await readdir(path)
      paths.push(...entries.filter((entry) => entry.endsWith('-report.json')).map((entry) => resolve(path, entry)))
    } else {
      paths.push(path)
    }
  }
  return paths.sort()
}

async function main() {
  const paths = await collectReportPaths(process.argv.slice(2))
  if (!paths.length) throw new Error('no qualification reports found')
  const errors = []
  for (const path of paths) {
    const report = JSON.parse(await readFile(path, 'utf8'))
    errors.push(...validateQualificationReport(report, path))
  }
  if (errors.length) {
    console.error(`model qualification report check FAILED (${errors.length}):`)
    for (const error of errors) console.error(`  - ${error}`)
    process.exit(1)
  }
  console.log(JSON.stringify({ checked: paths.length, reports: paths }, null, 2))
}

export {
  REQUIRED_STAGES,
  containsAbsoluteLocalPath,
  expectedOverall,
  reportContainsAbsoluteLocalPath,
  validateQualificationReport,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
