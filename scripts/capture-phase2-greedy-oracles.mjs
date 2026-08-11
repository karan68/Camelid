#!/usr/bin/env node
// Sequentially enrich Phase 2 tokenizer fixtures with exact-row greedy decode
// captures from the pinned llama.cpp server. Only one model is resident at a
// time, and Camelid is not started by this script; the later qualification
// runner replays these IDs through Camelid in a separate phase.

import { spawn } from 'node:child_process'
import { readFile, stat, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const root = resolve(args.get('root') || '.')
const rosterPath = resolve(root, args.get('roster') || 'qa/model-qualification/phase2-roster.json')
const modelsDir = resolve(args.get('models-dir') || '/Volumes/Untitled/models/phase2')
const server = resolve(args.get('llama-server') || '/Volumes/Untitled/llama.cpp-metal-parity/build-metal/bin/llama-server')
const maxNew = positiveInteger(args.get('max-new') || '5', '--max-new')
const port = positiveInteger(args.get('port') || '8391', '--port')
const selectedRows = new Set((args.get('rows') || '').split(',').map((row) => row.trim()).filter(Boolean))
const force = args.has('force')

const roster = JSON.parse(await readFile(rosterPath, 'utf8'))
const rows = selectedRows.size ? roster.rows.filter((row) => selectedRows.has(row.id)) : roster.rows
const missing = [...selectedRows].filter((id) => !rows.some((row) => row.id === id))
if (missing.length) throw new Error(`unknown row(s): ${missing.join(', ')}`)

for (const row of rows) {
  const fixturePath = resolve(root, row.probes.oracle_fixture)
  const fixture = JSON.parse(await readFile(fixturePath, 'utf8'))
  if (!force && fixture.raw_prompts.every((prompt) => Array.isArray(prompt.generated_ids) && prompt.generated_ids.length > 0)) {
    process.stderr.write(`skip ${row.id}: greedy IDs already present\n`)
    continue
  }
  const artifact = resolve(modelsDir, row.identity.gguf_filename)
  const artifactStat = await stat(artifact)
  if (artifactStat.size !== row.identity.size_bytes || fixture.artifact?.sha256 !== row.identity.sha256) {
    throw new Error(`${row.id}: artifact size or fixture SHA identity does not match the roster`)
  }

  process.stderr.write(`load ${row.id} (${artifactStat.size} bytes)\n`)
  const child = spawn(server, [
    '-m', artifact,
    '--host', '127.0.0.1',
    '--port', String(port),
    '-c', '256',
    '-ngl', '0',
    '-ctk', 'f32',
    '-ctv', 'f32',
    '-fa', 'off',
    '--no-repack',
    '--no-warmup',
    '--log-disable',
  ], { stdio: ['ignore', 'ignore', 'pipe'] })
  let stderr = ''
  child.stderr.on('data', (chunk) => { stderr = `${stderr}${chunk}`.slice(-8000) })
  try {
    await waitForHealth(port, child)
    for (const prompt of fixture.raw_prompts) {
      const completion = await postJson(port, '/completion', {
        prompt: prompt.prompt_ids,
        n_predict: maxNew,
        temperature: 0,
        top_k: 1,
        seed: 0,
        cache_prompt: false,
        samplers: ['top_k'],
        return_tokens: true,
      })
      if (!Array.isArray(completion.tokens) || completion.tokens.length === 0) {
        throw new Error(`${row.id}: oracle emitted no token IDs for ${JSON.stringify(prompt.text)}`)
      }
      prompt.generated_ids = completion.tokens
      prompt.generated_text = completion.content
      process.stderr.write(`  ${JSON.stringify(prompt.text)} -> ${JSON.stringify(completion.tokens)}\n`)
    }
  } catch (error) {
    throw new Error(`${error.message}\nllama-server stderr tail:\n${stderr}`)
  } finally {
    await stopChild(child)
  }

  fixture.scope = 'tokenizer_and_raw_greedy_generation'
  fixture.greedy_capture = {
    captured_at: new Date().toISOString(),
    endpoint: '/completion',
    prompt_input: 'pinned_prompt_ids',
    max_new_tokens: maxNew,
    temperature: 0,
    top_k: 1,
    seed: 0,
    cache_prompt: false,
    samplers: ['top_k'],
    backend: 'cpu_reference_f32_kv_no_repack',
    llama_server_command: ['<llama-server>', '-m', '<artifact>', '-c', '256', '-ngl', '0', '-ctk', 'f32', '-ctv', 'f32', '-fa', 'off', '--no-repack', '--no-warmup', '--log-disable'],
  }
  fixture.does_not_prove = [
    'Camelid weight load, logits, generation, or greedy-token parity until the independent replay phase passes',
    'chat-template rendering or prompt-token parity unless a separate exact-row shape pack is cited',
    'API, WebUI, bounded context, performance, GPU execution, portability, neighboring rows, or broad-family support',
  ]
  await writeFile(fixturePath, `${JSON.stringify(fixture, null, 2)}\n`)
  process.stderr.write(`wrote ${row.probes.oracle_fixture}\n`)
}

async function waitForHealth(portNumber, child) {
  const deadline = Date.now() + 10 * 60_000
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error(`llama-server exited ${child.exitCode} while loading`)
    try {
      const response = await fetch(`http://127.0.0.1:${portNumber}/health`)
      if (response.ok) return
    } catch {}
    await delay(500)
  }
  throw new Error('llama-server health check timed out after 10 minutes')
}

async function postJson(portNumber, path, body) {
  const response = await fetch(`http://127.0.0.1:${portNumber}${path}`, {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error(`${path} -> HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`)
  return response.json()
}

async function stopChild(child) {
  if (child.exitCode !== null) return
  child.kill('SIGINT')
  const exited = await Promise.race([
    new Promise((done) => child.once('exit', () => done(true))),
    delay(5000).then(() => false),
  ])
  if (!exited && child.exitCode === null) child.kill('SIGKILL')
}

function delay(milliseconds) {
  return new Promise((done) => setTimeout(done, milliseconds))
}

function positiveInteger(value, option) {
  if (!/^[1-9][0-9]*$/.test(value)) throw new Error(`${option} must be a positive integer`)
  return Number(value)
}

function parseArgs(argv) {
  const parsed = new Map()
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]
    if (!argument.startsWith('--')) throw new Error(`unexpected argument ${argument}`)
    const key = argument.slice(2)
    if (key === 'force') {
      parsed.set(key, 'true')
      continue
    }
    const value = argv[++index]
    if (!value || value.startsWith('--')) throw new Error(`--${key} requires a value`)
    parsed.set(key, value)
  }
  return parsed
}
