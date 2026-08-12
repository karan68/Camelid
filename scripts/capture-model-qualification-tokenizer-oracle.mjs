#!/usr/bin/env node
import { createHash } from 'node:crypto'
import { execFile } from 'node:child_process'
import { createReadStream } from 'node:fs'
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import { arch, platform, release } from 'node:os'
import { dirname, resolve } from 'node:path'
import { promisify } from 'node:util'
import { pathToFileURL } from 'node:url'
import { validateRoster } from './check-model-qualification-roster.mjs'

const execFileAsync = promisify(execFile)
const DEFAULT_PROMPTS = [
  'Hello',
  'The capital of France is',
  'Once upon a time',
  'The quick brown fox',
  '2 + 2 =',
  "don't stop 12345\nUnicode café 東京 🙂",
]

function parseArgs(argv) {
  const args = new Map()
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    if (!arg.startsWith('--')) continue
    const [key, inline] = arg.slice(2).split('=', 2)
    const next = argv[index + 1]
    args.set(key, inline ?? (next && !next.startsWith('--') ? argv[++index] : 'true'))
  }
  return args
}

async function sha256File(path) {
  const hash = createHash('sha256')
  await new Promise((resolvePromise, reject) => {
    const stream = createReadStream(path)
    stream.on('data', (chunk) => hash.update(chunk))
    stream.once('error', reject)
    stream.once('end', resolvePromise)
  })
  return hash.digest('hex')
}

function parseIdArray(stdout) {
  const line = String(stdout)
    .split(/\r?\n/)
    .map((candidate) => candidate.trim())
    .findLast((candidate) => candidate.startsWith('[') && candidate.endsWith(']'))
  if (!line) throw new Error('llama-tokenize did not emit an ID array')
  const ids = JSON.parse(line)
  if (!Array.isArray(ids) || ids.length === 0
    || ids.some((id) => !Number.isSafeInteger(id) || id < 0)) {
    throw new Error('llama-tokenize emitted an invalid ID array')
  }
  return ids
}

function parseVersion(output) {
  const match = /version:\s*(\d+)\s*\(([0-9a-f]{7,40})\)/i.exec(String(output))
  if (!match) throw new Error('llama-server did not report a parseable build identity')
  return { build: `b${match[1]}`, revision: match[2].toLowerCase() }
}

async function tokenize(binary, artifact, text) {
  const { stdout } = await execFileAsync(binary, [
    '-m', artifact,
    '-p', text,
    '--ids',
    '--no-escape',
    '--log-disable',
  ], { maxBuffer: 16 * 1024 * 1024, timeout: 120_000, windowsHide: true })
  return parseIdArray(stdout)
}

async function capture(options) {
  const root = resolve(options.root || '.')
  const rosterPath = resolve(root, options.roster)
  const roster = JSON.parse(await readFile(rosterPath, 'utf8'))
  const rosterErrors = validateRoster(roster, rosterPath)
  if (rosterErrors.length) throw new Error(`roster is invalid:\n${rosterErrors.join('\n')}`)
  const row = roster.rows.find((candidate) => candidate.id === options.row)
  if (!row) throw new Error(`unknown --row ${JSON.stringify(options.row)}`)

  const artifact = resolve(options.artifact)
  const artifactStat = await stat(artifact)
  const artifactSha256 = await sha256File(artifact)
  if (artifactStat.size !== row.identity.size_bytes || artifactSha256 !== row.identity.sha256) {
    throw new Error(`artifact identity mismatch for ${row.id}`)
  }

  const llamaTokenize = resolve(options.llamaTokenize)
  const llamaServer = resolve(options.llamaServer)
  const { stdout: versionStdout, stderr: versionStderr } = await execFileAsync(
    llamaServer,
    ['--version'],
    { timeout: 30_000, windowsHide: true },
  )
  const oracle = parseVersion(`${versionStdout}\n${versionStderr}`)
  const expectedOracle = roster.defaults.llama_cpp
  if (oracle.build !== expectedOracle.build || oracle.revision !== expectedOracle.revision) {
    throw new Error(`llama.cpp identity ${oracle.build}/${oracle.revision} does not match ${expectedOracle.build}/${expectedOracle.revision}`)
  }

  const rawPrompts = []
  for (const text of options.prompts || DEFAULT_PROMPTS) {
    rawPrompts.push({ text, prompt_ids: await tokenize(llamaTokenize, artifact, text) })
  }

  return {
    schema: 'camelid.model-qualification-oracle/v1',
    generated_at: new Date().toISOString(),
    row_id: row.id,
    scope: 'tokenizer_only',
    artifact: {
      repo: row.source.repo,
      revision: row.source.revision,
      file: row.source.file,
      size_bytes: row.identity.size_bytes,
      sha256: row.identity.sha256,
    },
    oracle: {
      engine: 'llama.cpp',
      build: oracle.build,
      revision: oracle.revision,
      platform: `${platform()}-${arch()}`,
      release: release(),
      llama_tokenize_sha256: await sha256File(llamaTokenize),
      llama_server_sha256: await sha256File(llamaServer),
      command: ['<llama-tokenize>', '-m', '<artifact>', '-p', '<prompt>', '--ids', '--no-escape', '--log-disable'],
    },
    raw_prompts: rawPrompts,
    chat_templates: [],
    does_not_prove: [
      'chat-template rendering or prompt-token parity',
      'weight load, logits, generation, or greedy-token parity',
      'API, WebUI, context, performance, GPU execution, or support',
    ],
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  if (args.has('help') || !args.get('row') || !args.get('artifact')
    || !args.get('llama-tokenize') || !args.get('llama-server') || !args.get('out')) {
    console.log(`Usage:
  node scripts/capture-model-qualification-tokenizer-oracle.mjs \\
    --roster <path> --row <id> --artifact <gguf> \\
    --llama-tokenize <b9632-binary> --llama-server <b9632-binary> --out <json>`)
    process.exit(args.has('help') ? 0 : 1)
  }
  const receipt = await capture({
    root: '.',
    roster: args.get('roster'),
    row: args.get('row'),
    artifact: args.get('artifact'),
    llamaTokenize: args.get('llama-tokenize'),
    llamaServer: args.get('llama-server'),
  })
  const out = resolve(args.get('out'))
  await mkdir(dirname(out), { recursive: true })
  await writeFile(out, `${JSON.stringify(receipt, null, 2)}\n`)
  process.stdout.write(`${JSON.stringify({ row_id: receipt.row_id, out, prompts: receipt.raw_prompts.length })}\n`)
}

export { capture, parseIdArray, parseVersion }

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
