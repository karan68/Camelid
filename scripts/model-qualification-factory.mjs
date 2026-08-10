#!/usr/bin/env node
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { basename, isAbsolute, relative, resolve, sep } from 'node:path'
import { pathToFileURL } from 'node:url'
import { validateRoster } from './check-model-qualification-roster.mjs'
import { qualify } from './model-qualification-runner.mjs'

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

function summarizeReports(reports, gateOrder) {
  const counts = { pass: 0, fail: 0, blocked: 0 }
  const rows = reports.map(({ row, report, reportFile }) => {
    counts[report.overall_status] = (counts[report.overall_status] || 0) + 1
    return {
      priority: row.priority,
      row_id: row.id,
      disposition: row.disposition,
      overall_status: report.overall_status,
      first_unresolved_stage: firstUnresolvedStage(report, gateOrder),
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
    const report = await qualify({
      root,
      roster: options.roster,
      row: row.id,
      artifact,
      camelid: options.camelid,
      runSmoke: options.runSmoke,
      runGeneration: options.runGeneration,
      promptLimit: options.promptLimit,
    })
    const reportFile = `${row.id}.json`
    await writeFile(resolve(outDir, reportFile), `${JSON.stringify(report, null, 2)}\n`)
    reports.push({ row, report, reportFile })
  }

  const summary = summarizeReports(reports, roster.gate_order)
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
  runFactory,
  selectRows,
  summarizeReports,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
