#!/usr/bin/env node
// smoke-release-artifact.mjs — boot the SHIPPED binary out of an extracted
// release archive and prove it actually serves.
//
// Why this exists. Nothing in release.yml has ever executed an artifact. Every
// existing gate is either a source-tree test (`cargo test`) or a staging-
// directory inspection (scripts/package-*), so an entire class of defect —
// the kind that only appears once the bytes are packaged and unpacked
// somewhere else — reaches users first. v0.4.1 through v0.4.4 were all that
// class.
//
// What this specifically regression-guards, on a stock CI runner that has NO
// NVIDIA driver and NO CUDA toolkit:
//
//   * `fix(cuda): a missing NVRTC must fall back to CPU, not abort the process`
//     (02f6a8b04). A driverless host is exactly the environment that fix
//     addresses, so booting the shipped artifact there exercises it directly.
//   * The embedded web UI. If `npm run build` ever silently produces nothing,
//     rust-embed happily bakes an empty UI into a binary that still compiles,
//     still passes every test, and ships a blank app.
//   * Tag/binary version agreement. `VERSION` (src/main.rs) comes from
//     CAMELID_GIT_DESCRIBE, falling back to CARGO_PKG_VERSION, and Cargo.toml
//     is bumped by a separate commit from the one that gets tagged.
//
// The run is deliberately MODEL-LESS. There is no GGUF on a CI runner, so this
// proves boot, health, capability reporting, UI embedding and clean shutdown —
// it makes no inference or parity claim whatsoever. Those stay with the
// existing exact-row evidence lanes.
//
// Usage:
//   node scripts/smoke-release-artifact.mjs --dir <extracted-payload> --label <label>
//        [--expect-version <tag>] [--out <receipt.json>] [--timeout-ms <n>]
//
// Exit 0 = the artifact booted and served. Exit 1 = it did not.
import { spawn } from 'node:child_process'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { request } from 'node:http'
import { createServer } from 'node:net'
import { tmpdir } from 'node:os'
import { basename, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

import { ARTIFACT_SPECS, LABELS, compareVersionToTag, parseVersionOutput } from './check-release-artifact.mjs'

// Mirror camelid-desktop/src/engine.rs (HEALTH_TIMEOUT / HEALTH_POLL_INTERVAL)
// on purpose: the desktop app and CI should agree on one definition of "this
// engine failed to come up", so a regression that pushes startup past the
// budget fails here instead of in a user's splash screen.
export const HEALTH_TIMEOUT_MS = 40_000
export const HEALTH_POLL_INTERVAL_MS = 350

/** Version probe budget. `--version` parses the clap tree and exits. */
export const VERSION_TIMEOUT_MS = 30_000

// ---------------------------------------------------------------------------
// Pure assertions (unit-tested directly)
// ---------------------------------------------------------------------------

/**
 * Arguments for a deterministic, model-less `serve`.
 *
 * `--models-dir` is pointed at an empty directory on purpose. Without it,
 * `auto_select_model` (src/main.rs) scans `<exe dir>/models`, the exe dir,
 * `./models` and `.` for the first *.gguf and loads it — so a stray model
 * anywhere near the runner's working directory would change what this smoke
 * actually tested.
 */
export function buildServeArgs({ port, modelsDir }) {
  if (!Number.isInteger(port) || port <= 0 || port > 65535) throw new RangeError(`invalid port ${port}`)
  if (!modelsDir) throw new Error('modelsDir is required so the boot stays model-less')
  return ['serve', '--addr', `127.0.0.1:${port}`, '--no-open', '--models-dir', modelsDir]
}

function parseJson(bodyText, what, failures) {
  try {
    return JSON.parse(bodyText)
  } catch (error) {
    failures.push(`${what}: response is not valid JSON — ${error.message}`)
    return null
  }
}

/**
 * `/v1/health` on a model-less engine.
 * Field names are HealthResponse in src/api/mod.rs.
 */
export function assertHealthPayload(status, bodyText) {
  const failures = []
  if (status !== 200) failures.push(`/v1/health: expected 200, got ${status}`)
  const body = parseJson(bodyText, '/v1/health', failures)
  if (!body) return failures
  if (body.ok !== true) failures.push(`/v1/health: ok=${JSON.stringify(body.ok)}, expected true`)
  if (body.loaded_now !== false) {
    failures.push(`/v1/health: loaded_now=${JSON.stringify(body.loaded_now)}, expected false for a model-less boot`)
  }
  if (body.generation_ready !== false) {
    failures.push(
      `/v1/health: generation_ready=${JSON.stringify(body.generation_ready)}, expected false for a model-less boot`,
    )
  }
  if (body.active_model_id !== null && body.active_model_id !== undefined) {
    failures.push(`/v1/health: active_model_id=${JSON.stringify(body.active_model_id)}, expected null`)
  }
  if (typeof body.backend !== 'string') failures.push('/v1/health: backend must be a string')
  if (body.engine_queue_depth !== 0) {
    failures.push(`/v1/health: engine_queue_depth=${JSON.stringify(body.engine_queue_depth)}, expected 0 at rest`)
  }
  return failures
}

/** `/api/capabilities` must be reachable and parse; its CONTENT is the ledger's business. */
export function assertCapabilitiesPayload(status, bodyText) {
  const failures = []
  if (status !== 200) failures.push(`/api/capabilities: expected 200, got ${status}`)
  const body = parseJson(bodyText, '/api/capabilities', failures)
  if (!body) return failures
  if (typeof body !== 'object' || Array.isArray(body)) failures.push('/api/capabilities: expected a JSON object')
  return failures
}

/**
 * `GET /` must serve the EMBEDDED web UI, not merely 200.
 *
 * Requiring both the app title and a hashed module asset means an empty or
 * placeholder embed fails, which a bare status check would wave through.
 */
export function assertWebUi(status, headers, bodyText) {
  const failures = []
  if (status !== 200) failures.push(`GET /: expected 200, got ${status}`)
  const contentType = String(headers?.['content-type'] ?? '')
  if (!contentType.includes('text/html')) failures.push(`GET /: expected an HTML content-type, got ${JSON.stringify(contentType)}`)
  const body = String(bodyText ?? '')
  if (!/<title>\s*Camelid\s*<\/title>/i.test(body)) failures.push('GET /: embedded UI is missing the Camelid app shell title')
  if (!/src="\/assets\/[^"]+\.js"/.test(body)) {
    failures.push('GET /: embedded UI references no hashed /assets/*.js bundle — the web UI build did not make it into the binary')
  }
  return failures
}

/** Engine stderr must be free of crash markers even when the run succeeds. */
export function assertNoCrashMarkers(stderr) {
  const failures = []
  const text = String(stderr ?? '')
  const markers = [
    { re: /panicked at/i, why: 'rust panic' },
    { re: /stack overflow/i, why: 'stack overflow' },
    { re: /segmentation fault|SIGSEGV/i, why: 'segfault' },
    { re: /Aborted \(core dumped\)/i, why: 'abort' },
  ]
  for (const { re, why } of markers) {
    if (re.test(text)) failures.push(`engine stderr contains a ${why} marker`)
  }
  return failures
}

// ---------------------------------------------------------------------------
// I/O shell
// ---------------------------------------------------------------------------

/** Ask the OS for an unused loopback port. */
export function pickFreePort() {
  return new Promise((resolvePort, rejectPort) => {
    const server = createServer()
    server.on('error', rejectPort)
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address()
      server.close((error) => (error ? rejectPort(error) : resolvePort(port)))
    })
  })
}

/** Dependency-free loopback GET. Resolves even on non-2xx; rejects on transport failure. */
export function httpGet(port, path, { timeoutMs = 5000, host = '127.0.0.1' } = {}) {
  return new Promise((resolveGet, rejectGet) => {
    const req = request({ host, port, path, method: 'GET', timeout: timeoutMs }, (res) => {
      const chunks = []
      res.on('data', (chunk) => chunks.push(chunk))
      res.on('end', () => resolveGet({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks).toString('utf8') }))
    })
    req.on('timeout', () => req.destroy(new Error(`GET ${path} timed out after ${timeoutMs}ms`)))
    req.on('error', rejectGet)
    req.end()
  })
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

/**
 * Poll `/v1/health` until it answers, the process dies, or the budget elapses.
 * Dependencies are injectable so the self-test can drive every branch without
 * a real binary.
 */
export async function waitForHealth({
  port,
  isAlive = () => true,
  timeoutMs = HEALTH_TIMEOUT_MS,
  intervalMs = HEALTH_POLL_INTERVAL_MS,
  get = httpGet,
  now = () => Date.now(),
  wait = sleep,
} = {}) {
  const deadline = now() + timeoutMs
  let lastError = 'no attempt was made'
  for (;;) {
    if (!isAlive()) return { ok: false, reason: `the engine exited before /v1/health answered (last transport state: ${lastError})` }
    try {
      const response = await get(port, '/v1/health', { timeoutMs: Math.min(intervalMs * 4, 5000) })
      if (response.status === 200) return { ok: true, response }
      lastError = `HTTP ${response.status}`
    } catch (error) {
      // Connection refused is the normal pre-listen state, not a failure.
      lastError = error.code || error.message
    }
    if (now() >= deadline) {
      return { ok: false, reason: `/v1/health did not return 200 within ${timeoutMs}ms (last transport state: ${lastError})` }
    }
    await wait(intervalMs)
  }
}

/** Run all three endpoint probes against a live server. */
export async function probeServer(port, get = httpGet) {
  const failures = []
  const health = await get(port, '/v1/health')
  failures.push(...assertHealthPayload(health.status, health.body))
  const capabilities = await get(port, '/api/capabilities')
  failures.push(...assertCapabilitiesPayload(capabilities.status, capabilities.body))
  const index = await get(port, '/')
  failures.push(...assertWebUi(index.status, index.headers, index.body))
  return { failures, health: health.body }
}

/** Terminate a child and its tree, then wait for the exit to be observed. */
async function terminate(child) {
  if (child.exitCode !== null || child.signalCode !== null) return
  if (process.platform === 'win32') {
    // A bare kill() leaves the process tree behind on Windows and the runner
    // then hangs waiting on inherited pipes.
    await new Promise((done) => {
      const killer = spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' })
      killer.on('exit', done)
      killer.on('error', done)
    })
  } else {
    child.kill('SIGTERM')
  }
  const exited = new Promise((done) => child.once('exit', done))
  const timedOut = await Promise.race([exited.then(() => false), sleep(10_000).then(() => true)])
  if (timedOut) child.kill('SIGKILL')
}

/** Run `<binary> --version` and compare it to the expected tag. */
async function checkVersion(binaryPath, expectedTag) {
  const result = await new Promise((done) => {
    const child = spawn(binaryPath, ['--version'], { stdio: ['ignore', 'pipe', 'pipe'] })
    let stdout = ''
    let stderr = ''
    const timer = setTimeout(() => child.kill('SIGKILL'), VERSION_TIMEOUT_MS)
    child.stdout.on('data', (d) => { stdout += d })
    child.stderr.on('data', (d) => { stderr += d })
    child.on('error', (error) => { clearTimeout(timer); done({ code: -1, stdout, stderr: `${stderr}${error.message}` }) })
    child.on('exit', (code) => { clearTimeout(timer); done({ code, stdout, stderr }) })
  })
  const failures = []
  if (result.code !== 0) failures.push(`--version exited ${result.code}: ${result.stderr.trim() || '(no stderr)'}`)
  const reported = parseVersionOutput(result.stdout)
  if (!reported) {
    failures.push(`could not parse a version from --version output: ${JSON.stringify(result.stdout.trim())}`)
  } else if (expectedTag) {
    const mismatch = compareVersionToTag(reported, expectedTag)
    if (mismatch) failures.push(mismatch)
  }
  return { failures, reported }
}

function parseArgs(argv) {
  const args = {}
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    const next = () => {
      const value = argv[i + 1]
      if (value === undefined || value.startsWith('--')) throw new Error(`${arg} requires a value`)
      i += 1
      return value
    }
    switch (arg) {
      case '--dir': args.dir = next(); break
      case '--label': args.label = next(); break
      case '--expect-version': args.expectVersion = next(); break
      case '--out': args.out = next(); break
      case '--timeout-ms': args.timeoutMs = Number.parseInt(next(), 10); break
      case '--help': case '-h': args.help = true; break
      default: throw new Error(`unknown argument ${arg}`)
    }
  }
  return args
}

const USAGE = `Usage:
  node scripts/smoke-release-artifact.mjs --dir <extracted-payload> --label <label> [options]

Options:
  --dir <path>              payload directory (the one that holds the binary)
  --label <label>           ${LABELS.join(' | ')}
  --expect-version <tag>    release tag the binary must report (e.g. v0.4.4)
  --timeout-ms <n>          health budget (default ${HEALTH_TIMEOUT_MS})
  --out <path>              write a JSON receipt
`

async function main() {
  let args
  try {
    args = parseArgs(process.argv.slice(2))
  } catch (error) {
    console.error(`smoke-release-artifact: ${error.message}\n\n${USAGE}`)
    process.exit(2)
  }
  if (args.help) {
    console.log(USAGE)
    return
  }
  if (!args.dir || !args.label || !ARTIFACT_SPECS[args.label]) {
    console.error(`smoke-release-artifact: --dir and a valid --label are required\n\n${USAGE}`)
    process.exit(2)
  }

  const spec = ARTIFACT_SPECS[args.label]
  // The desktop portable layout ships the Tauri GUI as its primary binary, but
  // the thing that must SERVE is the sidecar. Launching a GUI app headless on a
  // CI runner proves nothing and hangs; boot the engine it bundles instead.
  const serveBinary = args.label === 'desktop-windows-x64' ? 'camelid.exe' : spec.binary
  const binaryPath = resolve(join(args.dir, serveBinary))
  const failures = []
  const timeoutMs = Number.isInteger(args.timeoutMs) && args.timeoutMs > 0 ? args.timeoutMs : HEALTH_TIMEOUT_MS

  console.log(`smoke-release-artifact: label=${args.label} binary=${serveBinary}`)

  // 1. version / tag agreement --------------------------------------------
  const version = await checkVersion(binaryPath, args.expectVersion)
  failures.push(...version.failures)
  if (version.reported) console.log(`  version: ${version.reported}${args.expectVersion ? ` (tag ${args.expectVersion})` : ''}`)

  // 2. boot, serve, shut down ---------------------------------------------
  const emptyModelsDir = await mkdtemp(join(tmpdir(), 'camelid-smoke-models-'))
  const port = await pickFreePort()
  let stderr = ''
  let stdout = ''
  let child = null
  let healthBody = null
  try {
    child = spawn(binaryPath, buildServeArgs({ port, modelsDir: emptyModelsDir }), {
      cwd: args.dir,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    child.stdout.on('data', (d) => { stdout += d })
    child.stderr.on('data', (d) => { stderr += d })
    let spawnError = null
    child.on('error', (error) => { spawnError = error })

    const health = await waitForHealth({
      port,
      timeoutMs,
      isAlive: () => spawnError === null && child.exitCode === null && child.signalCode === null,
    })
    if (!health.ok) {
      failures.push(health.reason)
      if (spawnError) failures.push(`spawn failed: ${spawnError.message}`)
    } else {
      console.log(`  /v1/health answered on 127.0.0.1:${port}`)
      const probe = await probeServer(port)
      failures.push(...probe.failures)
      healthBody = probe.health
      if (!probe.failures.length) console.log('  /v1/health, /api/capabilities and the embedded web UI all served')
    }
  } finally {
    if (child) await terminate(child)
    await rm(emptyModelsDir, { recursive: true, force: true })
  }

  // 3. crash markers -------------------------------------------------------
  // A driverless runner is exactly the missing-NVRTC environment, so a clean
  // stderr here is the standing guard for "fall back to CPU, do not abort".
  failures.push(...assertNoCrashMarkers(stderr))

  if (args.out) {
    const receipt = {
      schema: 'camelid.release-artifact-smoke/v1',
      checked_at: new Date().toISOString(),
      label: args.label,
      binary: basename(binaryPath),
      reported_version: version.reported ?? null,
      expected_version: args.expectVersion ?? null,
      health_timeout_ms: timeoutMs,
      health: healthBody ? JSON.parse(healthBody) : null,
      failures,
      passed: failures.length === 0,
    }
    await writeFile(resolve(args.out), `${JSON.stringify(receipt, null, 2)}\n`, 'utf8')
    console.log(`  receipt: ${basename(args.out)}`)
  }

  if (failures.length) {
    console.error(`\nsmoke-release-artifact FAILED (${failures.length}):`)
    for (const failure of failures) console.error(`  FAIL: ${failure}`)
    if (stderr.trim()) console.error(`\n--- engine stderr ---\n${stderr.trim()}`)
    if (stdout.trim()) console.error(`\n--- engine stdout ---\n${stdout.trim()}`)
    process.exit(1)
  }
  console.log(`\nsmoke-release-artifact passed: ${args.label} boots and serves`)
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
