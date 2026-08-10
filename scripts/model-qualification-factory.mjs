#!/usr/bin/env node
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { basename, isAbsolute, relative, resolve, sep } from 'node:path'
import { pathToFileURL } from 'node:url'
import { validateRoster } from './check-model-qualification-roster.mjs'
import { resolveHfSource, validateLockAgainstSelection } from './hf-qualification-source.mjs'
import { deriveOverall, qualify, redactLocalPaths } from './model-qualification-runner.mjs'

function parseArgs(argv) {
  const args = new Map()
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    if (!arg.startsWith('--')) continue
    const [key, inline] = arg.slice(2).split('=', 2)
    const next = argv[index + 1]
    const value = inline ?? (next && !next.startsWith('--') ? argv[++index] : 'true')
    args.set(key, value)
  }
  return args
}

function selectRows(roster, requested) {
  if (!requested?.length) return [...roster.rows].sort((a, b) => a.priority - b.priority)
  const requestedSet = new Set(requested)
  const unknown = [...requestedSet].filter((id) => !roster.rows.some((row) => row.id === id))
  if (unknown.length) throw new Error(`unknown qualification rows: ${unknown.join(', ')}`)
  return roster.rows
    .filter((row) => requestedSet.has(row.id))
    .sort((a, b) => a.priority - b.priority)
}

function artifactForRow(row, modelsDir, explicitArtifact = null) {
  if (explicitArtifact) return resolve(explicitArtifact)
  const filename = row.identity.gguf_filename || row.source.file
  if (!modelsDir || !filename) return null
  return resolve(modelsDir, basename(filename))
}

function firstUnresolvedStage(report, gateOrder) {
  if (report.stages.artifact?.status !== 'pass') return 'artifact'
  return gateOrder.find((name) => report.stages[name]?.status !== 'pass') || null
}

function publicRosterLabel(root, rosterPath) {
  const candidate = relative(root, rosterPath)
  if (!candidate || candidate === '.') return '<workspace-roster>'
  if (isAbsolute(candidate) || candidate === '..' || candidate.startsWith('..' + sep)) {
    return '<external-roster>'
  }
  return candidate.split(sep).join('/')
}

function sourceSelectionForRow(row) {
  return {
    row_id: row.id,
    repo: row.source.repo,
    file: row.source.file,
    revision: row.source.revision,
    expected: {
      size_bytes: row.identity.size_bytes,
      sha256: row.identity.sha256,
      license: row.source.license,
    },
  }
}

function sourceLookupErrorCode(error) {
  const name = typeof error?.name === 'string' ? error.name.toLowerCase() : ''
  const code = typeof error?.code === 'string' ? error.code.toLowerCase() : ''
  const message = error instanceof Error ? error.message : String(error)
  if (name === 'aborterror' || name === 'timeouterror' || ['etimedout', 'esockettimedout'].includes(code)) {
    return 'timeout'
  }
  const httpStatus = /(?:request failed|http)\D*(\d{3})/i.exec(message)?.[1]
  if (httpStatus) return `http_${httpStatus}`
  if (/immutable 40-character revision/i.test(message)) return 'missing_immutable_revision'
  if (/file .* is absent from/i.test(message)) return 'file_not_found_at_revision'
  if (/positive byte size/i.test(message)) return 'missing_file_size'
  if (/LFS SHA-256/i.test(message)) return 'missing_lfs_sha256'
  if (['econnreset', 'econnrefused', 'enotfound', 'eai_again'].includes(code)) return 'network_error'
  return 'source_lookup_error'
}

async function resolveSourceStage(row, {
  resolver = resolveHfSource,
  token = process.env.HF_TOKEN || null,
} = {}) {
  const selected = sourceSelectionForRow(row)
  const missing = [
    ['repo', selected.repo],
    ['file', selected.file],
    ['revision', selected.revision],
    ['size_bytes', selected.expected.size_bytes],
    ['sha256', selected.expected.sha256],
    ['license', selected.expected.license],
  ].filter(([, value]) => value === null || value === undefined || value === '')
    .map(([field]) => field)
  if (missing.length) {
    return {
      status: 'blocked',
      resolution: 'live_huggingface',
      reason: `roster source selector is not fully pinned: ${missing.join(', ')}`,
    }
  }

  let lock
  try {
    lock = await resolver({
      repo: selected.repo,
      file: selected.file,
      revision: selected.revision,
      token,
    })
  } catch (error) {
    const errorCode = sourceLookupErrorCode(error)
    return {
      status: 'blocked',
      resolution: 'live_huggingface',
      error_code: errorCode,
      reason: `live Hugging Face source resolution could not complete (${errorCode})`,
    }
  }
  if (!lock || typeof lock !== 'object' || Array.isArray(lock)) {
    return {
      status: 'blocked',
      resolution: 'live_huggingface',
      reason: 'live Hugging Face source resolution returned no source-lock object',
    }
  }

  const access = {
    gated: Boolean(lock.access?.gated),
    private: Boolean(lock.access?.private),
    disabled: Boolean(lock.access?.disabled),
  }

  try {
    validateLockAgainstSelection(lock, selected)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return {
      status: 'fail',
      resolution: 'live_huggingface',
      reason: message,
      resolved: {
        repo: lock.repo,
        file: lock.file,
        revision: lock.revision,
        size_bytes: lock.size_bytes,
        sha256: lock.sha256,
        license: lock.license,
        access,
      },
    }
  }

  return {
    status: 'pass',
    resolution: 'live_huggingface',
    repo: lock.repo,
    file: lock.file,
    revision: lock.revision,
    size_bytes: lock.size_bytes,
    sha256: lock.sha256,
    license: lock.license,
    access,
  }
}

function summarizeSourceResolution(reports) {
  const counts = { pass: 0, fail: 0, blocked: 0 }
  for (const { report } of reports) {
    const status = report.stages.source?.status
    counts[status] = (counts[status] || 0) + 1
  }
  return { mode: 'live_huggingface', counts }
}

function summarizeReports(reports, gateOrder, { sourcePreflight = false } = {}) {
  const counts = { pass: 0, fail: 0, blocked: 0 }
  const rows = reports.map(({ row, report, reportFile }) => {
    counts[report.overall_status] = (counts[report.overall_status] || 0) + 1
    return {
      priority: row.priority,
      row_id: row.id,
      disposition: row.disposition,
      overall_status: report.overall_status,
      first_unresolved_stage: sourcePreflight && report.stages.source?.status !== 'pass'
        ? 'source'
        : firstUnresolvedStage(report, gateOrder),
      report_file: reportFile,
    }
  })
  return { counts, rows }
}

async function runFactory(options) {
  const root = resolve(options.root || '.')
  const rosterPath = resolve(root, options.roster || 'qa/model-qualification/phase1-roster.json')
  const roster = JSON.parse(await readFile(rosterPath, 'utf8'))
  const rosterErrors = validateRoster(roster, rosterPath)
  if (rosterErrors.length) throw new Error(`roster is invalid:\n${rosterErrors.join('\n')}`)

  const rows = selectRows(roster, options.rows)
  if (options.artifact && rows.length !== 1) {
    throw new Error('--artifact is only valid when exactly one row is selected')
  }

  const configuredModelsDir = options.modelsDir
    || process.env[roster.defaults.models_dir_env]
    || null
  const modelsDir = configuredModelsDir ? resolve(root, configuredModelsDir) : null
  const outDir = resolve(root, options.outDir || 'target/model-qualification/factory')
  await mkdir(outDir, { recursive: true })

  const reports = []
  for (const row of rows) {
    const artifact = artifactForRow(row, modelsDir, options.artifact)
    const resolvedSource = options.resolveSource
      ? await resolveSourceStage(row, { resolver: options.sourceResolver || resolveHfSource })
      : null
    const sourcePassed = !resolvedSource || resolvedSource.status === 'pass'
    let report = await qualify({
      root,
      roster: options.roster,
      row: row.id,
      artifact: sourcePassed ? artifact : null,
      camelid: options.camelid,
      runSmoke: sourcePassed && options.runSmoke,
      runGeneration: sourcePassed && options.runGeneration,
      promptLimit: options.promptLimit,
    })
    if (resolvedSource) {
      report.stages.source = resolvedSource
      if (!sourcePassed && artifact) {
        report.stages.artifact = {
          status: 'blocked',
          reason: 'live source preflight did not pass; the local artifact was not inspected',
        }
      }
      report.overall_status = deriveOverall(report.stages)
      report = redactLocalPaths(report, [
        [artifact, '<artifact>'],
        [root, '<repo>'],
      ])
    }
    const reportFile = `${row.id}.json`
    await writeFile(resolve(outDir, reportFile), `${JSON.stringify(report, null, 2)}\n`)
    reports.push({ row, report, reportFile })
  }

  const summary = summarizeReports(reports, roster.gate_order, {
    sourcePreflight: Boolean(options.resolveSource),
  })
  const sourceDirtyValues = reports.map(({ report }) => report.source_dirty)
  const index = {
    schema: 'camelid.model-qualification-index/v1',
    generated_at: new Date().toISOString(),
    roster: publicRosterLabel(root, rosterPath),
    roster_schema: roster.schema,
    models_dir_env: roster.defaults.models_dir_env,
    models_dir_configured: Boolean(modelsDir),
    source_head: reports[0]?.report.source_head ?? null,
    source_dirty: sourceDirtyValues.some((value) => value === null)
      ? null
      : sourceDirtyValues.some(Boolean),
    ...(options.resolveSource
      ? { source_resolution: summarizeSourceResolution(reports) }
      : {}),
    ...summary,
  }
  await writeFile(resolve(outDir, 'index.json'), `${JSON.stringify(index, null, 2)}\n`)
  return index
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  if (args.has('help')) {
    console.log(`Usage: node scripts/model-qualification-factory.mjs [options]

Options:
  --roster <path>          Qualification roster (default: Phase 1)
  --rows <id,id>           Run only these rows, preserving roster priority
  --models-dir <path>      Artifact directory (default: roster env variable)
  --artifact <path>        Exact artifact override; requires one selected row
  --camelid <path>         Camelid binary (default: target/release/camelid)
  --out-dir <path>         Scrubbed reports/index output directory
  --resolve-source         Live-resolve each pinned HF selector before local probes
  --run-smoke              Execute configured smoke probes
  --run-generation         Execute pinned greedy parity probes
  --prompt-limit <n>       Deliberately partial parity run (remains blocked)
`)
    return
  }
  const requestedRows = args.get('rows')
    ? args.get('rows').split(',').map((id) => id.trim()).filter(Boolean)
    : []
  const promptLimitRaw = args.get('prompt-limit')
  const promptLimit = promptLimitRaw ? Number.parseInt(promptLimitRaw, 10) : undefined
  if (promptLimitRaw && (!Number.isInteger(promptLimit) || promptLimit < 1)) {
    throw new Error('--prompt-limit must be a positive integer')
  }
  const index = await runFactory({
    root: '.',
    roster: args.get('roster'),
    rows: requestedRows,
    modelsDir: args.get('models-dir'),
    artifact: args.get('artifact'),
    camelid: args.get('camelid'),
    outDir: args.get('out-dir'),
    resolveSource: args.has('resolve-source'),
    runSmoke: args.has('run-smoke'),
    runGeneration: args.has('run-generation'),
    promptLimit,
  })
  process.stdout.write(`${JSON.stringify(index, null, 2)}\n`)
}

export {
  artifactForRow,
  firstUnresolvedStage,
  publicRosterLabel,
  resolveSourceStage,
  runFactory,
  selectRows,
  sourceLookupErrorCode,
  sourceSelectionForRow,
  summarizeSourceResolution,
  summarizeReports,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
