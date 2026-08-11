#!/usr/bin/env node
import { createHash } from 'node:crypto'
import { spawn } from 'node:child_process'
import { createReadStream } from 'node:fs'
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import { arch, cpus, platform, release } from 'node:os'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { validateRoster } from './check-model-qualification-roster.mjs'

const REPORT_STATUSES = new Set(['pass', 'fail', 'blocked', 'skipped'])
const COMMAND_TIMEOUT_MS = 300_000

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

function compareIds(actual, expected) {
  const limit = Math.min(actual.length, expected.length)
  let firstDivergence = null
  for (let index = 0; index < limit; index += 1) {
    if (actual[index] !== expected[index]) {
      firstDivergence = index
      break
    }
  }
  if (firstDivergence === null && actual.length !== expected.length) firstDivergence = limit
  return {
    match: firstDivergence === null,
    first_divergence: firstDivergence,
    actual_length: actual.length,
    expected_length: expected.length,
  }
}

function headlineQuant(tensors) {
  const counts = new Map()
  for (const tensor of tensors || []) {
    const type = tensor.tensor_type
    if (typeof type !== 'string' || ['F32', 'F16', 'BF16'].includes(type)) continue
    counts.set(type, (counts.get(type) || 0) + 1)
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]?.[0] || null
}

function deriveOverall(stages) {
  const required = Object.values(stages).filter((stage) => stage.required !== false)
  for (const stage of required) {
    if (!REPORT_STATUSES.has(stage.status)) throw new Error(`invalid report stage status ${stage.status}`)
  }
  if (required.some((stage) => stage.status === 'fail')) return 'fail'
  if (required.some((stage) => stage.status === 'blocked')) return 'blocked'
  if (required.some((stage) => stage.status === 'skipped')) return 'blocked'
  return 'pass'
}

function stage(status, details = {}) {
  if (!REPORT_STATUSES.has(status)) throw new Error(`invalid stage status ${status}`)
  return { status, ...details }
}

function redactLocalPaths(value, replacements) {
  if (typeof value === 'string') {
    return replacements.reduce(
      (text, [localPath, replacement]) => localPath ? text.split(localPath).join(replacement) : text,
      value,
    )
  }
  if (Array.isArray(value)) return value.map((item) => redactLocalPaths(item, replacements))
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, redactLocalPaths(item, replacements)]),
    )
  }
  return value
}

function validateOracleFixture(fixture, row, defaults = {}) {
  const errors = []
  if (!fixture || typeof fixture !== 'object' || Array.isArray(fixture)) {
    return ['fixture must be a JSON object']
  }
  if (fixture.schema !== 'camelid.model-qualification-oracle/v1') {
    errors.push('schema must be camelid.model-qualification-oracle/v1')
  }
  if (fixture.row_id !== row.id) errors.push(`row_id ${JSON.stringify(fixture.row_id)} != ${JSON.stringify(row.id)}`)

  const artifact = fixture.artifact || {}
  const expectedArtifact = {
    repo: row.source.repo,
    revision: row.source.revision,
    file: row.source.file,
    size_bytes: row.identity.size_bytes,
    sha256: row.identity.sha256,
  }
  for (const [field, expected] of Object.entries(expectedArtifact)) {
    if (expected !== null && expected !== undefined && artifact[field] !== expected) {
      errors.push(`artifact.${field} ${JSON.stringify(artifact[field])} != ${JSON.stringify(expected)}`)
    }
  }

  const expectedOracle = defaults.llama_cpp || {}
  for (const field of ['build', 'revision']) {
    const expected = expectedOracle[field]
    if (expected && fixture.oracle?.[field] !== expected) {
      errors.push(`oracle.${field} ${JSON.stringify(fixture.oracle?.[field])} != ${JSON.stringify(expected)}`)
    }
  }
  return errors
}

function gitDirtyState(statusPorcelain) {
  return statusPorcelain === null ? null : statusPorcelain.length > 0
}

async function sha256File(path) {
  const hash = createHash('sha256')
  await new Promise((resolvePromise, reject) => {
    const input = createReadStream(path)
    input.on('data', (chunk) => hash.update(chunk))
    input.once('error', reject)
    input.once('end', resolvePromise)
  })
  return hash.digest('hex')
}

async function run(command, commandArgs, options = {}) {
  return new Promise((resolvePromise) => {
    const child = spawn(command, commandArgs, {
      cwd: options.cwd,
      env: { ...process.env, ...(options.env || {}) },
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    })
    let stdout = ''
    let stderr = ''
    let settled = false
    let timedOut = false
    const timeoutMs = options.timeoutMs ?? COMMAND_TIMEOUT_MS
    const finish = (result) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolvePromise({ stdout, stderr, timeout_ms: timeoutMs, ...result })
    }
    const timer = setTimeout(() => {
      timedOut = true
      child.kill('SIGKILL')
    }, timeoutMs)
    child.stdout.on('data', (chunk) => { stdout += chunk })
    child.stderr.on('data', (chunk) => { stderr += chunk })
    child.once('error', (error) => finish({
      code: null,
      timed_out: timedOut,
      spawn_error: timedOut ? null : error.message,
    }))
    child.once('close', (code) => finish({ code: code ?? 1, timed_out: timedOut, spawn_error: null }))
  })
}

function commandInfrastructureProblem(result, label) {
  if (result.timed_out) return `${label} timed out after ${result.timeout_ms} ms`
  if (result.spawn_error) return `${label} could not start: ${result.spawn_error}`
  return null
}

function parseSingleJson(stdout) {
  const lines = stdout.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    try { return JSON.parse(lines[index]) } catch {}
  }
  return JSON.parse(stdout)
}

async function gitValue(root, args) {
  const result = await run('git', args, { cwd: root, timeoutMs: 10_000 })
  return result.code === 0 ? result.stdout.trim() : null
}

async function inspectMetadata({ binary, artifact, row, root }) {
  const command = [binary, 'inspect', artifact]
  const result = await run(binary, ['inspect', artifact], { cwd: root, env: row.lane.env })
  const infrastructureProblem = commandInfrastructureProblem(result, 'camelid inspect')
  if (infrastructureProblem) return stage('blocked', { command, reason: infrastructureProblem })
  if (result.code !== 0) {
    return stage('fail', { command, reason: `camelid inspect exited ${result.code}`, stderr: result.stderr.trim() })
  }
  let inspect
  try { inspect = JSON.parse(result.stdout) }
  catch (error) { return stage('fail', { command, reason: `inspect did not emit JSON: ${error.message}` }) }
  const metadata = inspect.metadata || {}
  const observed = {
    architecture: metadata['general.architecture'] ?? null,
    tokenizer_model: metadata['tokenizer.ggml.model'] ?? null,
    tokenizer_pre: metadata['tokenizer.ggml.pre'] ?? null,
    headline_quant: headlineQuant(inspect.tensors),
    tensor_count: inspect.tensor_count ?? inspect.tensors?.length ?? null,
  }
  const mismatches = []
  if (observed.architecture !== row.identity.architecture) mismatches.push(`architecture ${observed.architecture} != ${row.identity.architecture}`)
  if (observed.tokenizer_model !== row.expected.tokenizer_model) mismatches.push(`tokenizer model ${observed.tokenizer_model} != ${row.expected.tokenizer_model}`)
  if (row.expected.tokenizer_pre !== null && observed.tokenizer_pre !== row.expected.tokenizer_pre) mismatches.push(`tokenizer pre ${observed.tokenizer_pre} != ${row.expected.tokenizer_pre}`)
  if (observed.headline_quant !== row.identity.quantization) mismatches.push(`headline quant ${observed.headline_quant} != ${row.identity.quantization}`)
  return mismatches.length
    ? stage('fail', { command, observed, reason: mismatches.join('; ') })
    : stage('pass', { command, observed })
}

async function checkTokenizer({ binary, artifact, row, fixture, root }) {
  if (!fixture?.raw_prompts?.length) return stage('blocked', { reason: 'required independent raw-prompt fixture is absent' })
  const probes = []
  for (const prompt of fixture.raw_prompts) {
    const args = ['tokenize', '--model', artifact, '--prompt', prompt.text]
    const result = await run(binary, args, { cwd: root, env: row.lane.env })
    const infrastructureProblem = commandInfrastructureProblem(result, 'camelid tokenize')
    if (infrastructureProblem) {
      return stage('blocked', { command: [binary, ...args], reason: infrastructureProblem, probes })
    }
    if (result.code !== 0) {
      return stage('fail', { command: [binary, ...args], reason: `tokenize exited ${result.code}`, stderr: result.stderr.trim(), probes })
    }
    let output
    try { output = parseSingleJson(result.stdout) }
    catch (error) { return stage('fail', { command: [binary, ...args], reason: `tokenize did not emit JSON: ${error.message}`, probes }) }
    const comparison = compareIds(output.ids || [], prompt.prompt_ids || [])
    probes.push({ text: prompt.text, ...comparison, actual_ids: output.ids, expected_ids: prompt.prompt_ids })
  }

  const chatProbes = []
  for (const template of fixture.chat_templates || []) {
    const args = ['tokenize', '--model', artifact, '--prompt', template.rendered, '--parse-special', '--no-add-special']
    const result = await run(binary, args, { cwd: root, env: row.lane.env })
    const infrastructureProblem = commandInfrastructureProblem(result, 'camelid tokenize')
    if (infrastructureProblem) {
      return stage('blocked', { command: [binary, ...args], reason: infrastructureProblem, probes, chat_probes: chatProbes })
    }
    if (result.code !== 0) {
      return stage('fail', { command: [binary, ...args], reason: `chat-template tokenization exited ${result.code}`, probes, chat_probes: chatProbes, stderr: result.stderr.trim() })
    }
    let output
    try { output = parseSingleJson(result.stdout) }
    catch (error) {
      return stage('fail', { command: [binary, ...args], reason: `chat-template tokenize did not emit JSON: ${error.message}`, probes, chat_probes: chatProbes })
    }
    const comparison = compareIds(output.ids || [], template.prompt_ids || [])
    chatProbes.push({ name: template.name, ...comparison, actual_ids: output.ids, expected_ids: template.prompt_ids })
  }

  const allMatch = [...probes, ...chatProbes].every((probe) => probe.match)
  return allMatch
    ? stage('pass', { oracle_fixture: row.probes.oracle_fixture, probes, chat_probes: chatProbes })
    : stage('fail', { oracle_fixture: row.probes.oracle_fixture, probes, chat_probes: chatProbes, reason: 'one or more prompt token sequences diverged from the pinned oracle' })
}

async function checkSmoke({ binary, artifact, row, root, enabled }) {
  if (!enabled) return stage('blocked', { reason: 'smoke execution was not requested; pass --run-smoke' })
  if (row.probes.smoke !== 'runnable_smoke') return stage('blocked', { reason: `runner does not yet orchestrate smoke kind ${row.probes.smoke}` })
  const args = ['runnable-smoke', artifact]
  const result = await run(binary, args, { cwd: root, env: row.lane.env })
  const infrastructureProblem = commandInfrastructureProblem(result, 'camelid runnable-smoke')
  if (infrastructureProblem) return stage('blocked', { command: [binary, ...args], reason: infrastructureProblem })
  if (result.code === 0) {
    try {
      return stage('pass', { command: [binary, ...args], receipt: JSON.parse(result.stdout), stderr: result.stderr.trim() })
    } catch (error) {
      return stage('fail', { command: [binary, ...args], reason: `runnable-smoke did not emit JSON: ${error.message}` })
    }
  }
  const reason = result.stderr.trim() || result.stdout.trim() || `runnable-smoke exited ${result.code}`
  if (/combo not yet anchored/i.test(reason)) {
    return stage('blocked', { command: [binary, ...args], reason })
  }
  return stage('fail', { command: [binary, ...args], reason })
}

async function checkParity({ binary, artifact, row, fixture, root, enabled, promptLimit }) {
  if (!enabled) return stage('blocked', { reason: 'greedy generation parity was not requested; pass --run-generation' })
  if (!fixture?.raw_prompts?.length) return stage('blocked', { reason: 'required independent greedy fixture is absent' })
  const selected = fixture.raw_prompts.slice(0, promptLimit ?? fixture.raw_prompts.length)
  const probes = []
  for (const prompt of selected) {
    const args = [
      'bench-generate', artifact,
      '--prompt', prompt.text,
      '--max-tokens', String(prompt.generated_ids.length),
      '--temperature', '0',
      '--iterations', '1',
      '--deterministic',
    ]
    const result = await run(binary, args, { cwd: root, env: row.lane.env })
    const infrastructureProblem = commandInfrastructureProblem(result, 'camelid bench-generate')
    if (infrastructureProblem) {
      return stage('blocked', { command: [binary, ...args], reason: infrastructureProblem, probes })
    }
    if (result.code !== 0) {
      return stage('fail', { command: [binary, ...args], reason: `bench-generate exited ${result.code}`, stderr: result.stderr.trim(), probes })
    }
    let output
    try { output = parseSingleJson(result.stdout) }
    catch (error) { return stage('fail', { command: [binary, ...args], reason: `bench-generate did not emit JSON: ${error.message}`, probes }) }
    const comparison = compareIds(output.output_token_ids || [], prompt.generated_ids || [])
    probes.push({
      text: prompt.text,
      ...comparison,
      actual_ids: output.output_token_ids,
      expected_ids: prompt.generated_ids,
      actual_text: output.output_text,
      expected_text: prompt.generated_text,
      metrics: {
        prompt_tokens: output.prompt_tokens,
        load_ms: output.load_ms,
        ttft_ms: output.ttft_ms,
        decode_ms: output.decode_ms,
        tokens_per_second: output.tokens_per_second,
        peak_memory_bytes: output.peak_memory_bytes,
      },
    })
  }
  if (selected.length !== fixture.raw_prompts.length) {
    return stage('blocked', { probes, reason: `partial parity probe ran ${selected.length}/${fixture.raw_prompts.length} prompts` })
  }
  const allMatch = probes.every((probe) => probe.match)
  return allMatch
    ? stage('pass', { oracle_fixture: row.probes.oracle_fixture, probes })
    : stage('fail', { oracle_fixture: row.probes.oracle_fixture, probes, reason: 'greedy token parity diverged from the pinned oracle' })
}

async function qualify(options) {
  const root = resolve(options.root || '.')
  const rosterPath = resolve(root, options.roster || 'qa/model-qualification/phase1-roster.json')
  const roster = JSON.parse(await readFile(rosterPath, 'utf8'))
  const rosterErrors = validateRoster(roster, rosterPath)
  if (rosterErrors.length) throw new Error(`roster is invalid:\n${rosterErrors.join('\n')}`)
  const row = roster.rows.find((candidate) => candidate.id === options.row)
  if (!row) throw new Error(`unknown --row ${JSON.stringify(options.row)}`)

  const artifact = options.artifact ? resolve(root, options.artifact) : null
  const binary = resolve(root, options.camelid || 'target/release/camelid')
  const fixturePath = row.probes.oracle_fixture ? resolve(root, row.probes.oracle_fixture) : null
  let fixture = null
  let fixtureReadError = null
  if (fixturePath) {
    try { fixture = JSON.parse(await readFile(fixturePath, 'utf8')) }
    catch (error) { fixtureReadError = `oracle fixture could not be read: ${error.message}` }
  }
  const fixtureErrors = fixture
    ? validateOracleFixture(fixture, row, roster.defaults)
    : [fixtureReadError || 'no oracle fixture is configured for this row']
  const sourceHead = await gitValue(root, ['rev-parse', 'HEAD'])
  const sourceStatus = await gitValue(root, ['status', '--porcelain'])

  const report = {
    schema: 'camelid.model-qualification-report/v1',
    generated_at: new Date().toISOString(),
    row_id: row.id,
    source_head: sourceHead,
    source_dirty: gitDirtyState(sourceStatus),
    source_inspection: sourceHead === null || sourceStatus === null ? 'unknown' : 'observed',
    host: {
      hostname_redacted: true,
      platform: platform(),
      release: release(),
      arch: arch(),
      logical_cpus: cpus().length,
    },
    backend: row.lane,
    artifact: { path: artifact },
    oracle_fixture: fixtureErrors.length
      ? { path: row.probes.oracle_fixture, status: 'blocked', reason: fixtureErrors.join('; ') }
      : { path: row.probes.oracle_fixture, status: 'pass' },
    stages: {},
    overall_status: 'blocked',
  }

  report.stages.source = row.gates.source.status === 'pass'
    ? stage('pass', { repo: row.source.repo, revision: row.source.revision, file: row.source.file, license: row.source.license })
    : stage('blocked', { reason: row.gates.source.reason || 'source gate is not pinned in the roster' })

  let artifactStat
  if (!artifact) {
    report.stages.artifact = stage('blocked', {
      reason: 'artifact path is unresolved; provide --artifact or configure the roster factory models directory',
    })
  } else {
    try { artifactStat = await stat(artifact) }
    catch (error) {
      report.stages.artifact = stage('blocked', { reason: `artifact is absent: ${error.code || error.message}` })
    }
  }
  if (artifactStat) {
    const sha256 = await sha256File(artifact)
    report.artifact = { path: artifact, size_bytes: artifactStat.size, sha256 }
    const mismatches = []
    if (row.identity.size_bytes !== null && artifactStat.size !== row.identity.size_bytes) mismatches.push(`size ${artifactStat.size} != ${row.identity.size_bytes}`)
    if (row.identity.sha256 !== null && sha256 !== row.identity.sha256) mismatches.push(`sha256 ${sha256} != ${row.identity.sha256}`)
    report.stages.artifact = mismatches.length
      ? stage('fail', { reason: mismatches.join('; ') })
      : stage('pass', { size_bytes: artifactStat.size, sha256 })
  }

  if (report.stages.artifact.status === 'pass') {
    report.stages.metadata = await inspectMetadata({ binary, artifact, row, root })
    if (fixtureErrors.length) {
      const reason = `oracle fixture identity gate did not pass: ${fixtureErrors.join('; ')}`
      report.stages.tokenizer = stage('blocked', { reason })
      report.stages.template = stage('blocked', { reason })
    } else {
      report.stages.tokenizer = await checkTokenizer({ binary, artifact, row, fixture, root })
      const chatTokenProbePassed = report.stages.tokenizer.status === 'pass' && (fixture?.chat_templates?.length || 0) > 0
      report.stages.template = chatTokenProbePassed
        ? stage('blocked', { oracle_fixture: row.probes.oracle_fixture, reason: 'the rendered oracle bytes tokenize identically, but the Camelid API renderer has not yet been compared byte-for-byte in this report' })
        : stage('blocked', { reason: 'template rendering is downstream of the tokenizer fixture gate' })
    }
    report.stages.load_smoke = await checkSmoke({ binary, artifact, row, root, enabled: options.runSmoke })
    report.stages.parity = fixtureErrors.length
      ? stage('blocked', { reason: `oracle fixture identity gate did not pass: ${fixtureErrors.join('; ')}` })
      : await checkParity({ binary, artifact, row, fixture, root, enabled: options.runGeneration, promptLimit: options.promptLimit })
  } else {
    for (const name of ['metadata', 'tokenizer', 'template', 'load_smoke', 'parity']) {
      report.stages[name] = stage('blocked', { reason: 'artifact identity gate did not pass' })
    }
  }

  report.stages.api_webui = stage('blocked', { reason: 'API/WebUI promotion smoke requires an explicitly started current-head server and frontend; this local file runner does not infer it' })
  report.stages.context = stage('blocked', { reason: `no preflighted context receipt captured; required buckets: ${row.probes.context_buckets.join(', ')}` })
  report.overall_status = deriveOverall(report.stages)
  report.artifact.path_redacted = true
  return redactLocalPaths(report, [
    [artifact, '<artifact>'],
    [binary, '<camelid>'],
    [root, '<repo>'],
  ])
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  if (args.has('help') || !args.get('row') || !args.get('artifact')) {
    console.log(`Usage: node scripts/model-qualification-runner.mjs --row <id> --artifact <gguf> [options]

Options:
  --roster <path>          Roster path (default: qa/model-qualification/phase1-roster.json)
  --camelid <path>         Camelid binary (default: target/release/camelid)
  --out <path>             Write the generated report as JSON
  --run-smoke              Execute runnable-smoke when configured
  --run-generation         Execute greedy token parity against the pinned fixture
  --prompt-limit <n>       Deliberately run only the first n parity prompts (report stays blocked)
`)
    process.exit(args.has('help') ? 0 : 1)
  }
  const promptLimitRaw = args.get('prompt-limit')
  const promptLimit = promptLimitRaw ? Number.parseInt(promptLimitRaw, 10) : undefined
  if (promptLimitRaw && (!Number.isInteger(promptLimit) || promptLimit < 1)) throw new Error('--prompt-limit must be a positive integer')
  const report = await qualify({
    root: '.',
    roster: args.get('roster'),
    row: args.get('row'),
    artifact: args.get('artifact'),
    camelid: args.get('camelid'),
    out: args.get('out'),
    runSmoke: args.has('run-smoke'),
    runGeneration: args.has('run-generation'),
    promptLimit,
  })
  const rendered = `${JSON.stringify(report, null, 2)}\n`
  if (args.get('out')) {
    const outPath = resolve(args.get('out'))
    await mkdir(dirname(outPath), { recursive: true })
    await writeFile(outPath, rendered)
  }
  process.stdout.write(rendered)
}

export {
  compareIds,
  deriveOverall,
  gitDirtyState,
  headlineQuant,
  qualify,
  redactLocalPaths,
  validateOracleFixture,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
