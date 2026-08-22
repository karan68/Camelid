import { spawn, spawnSync } from 'node:child_process'
import { createWriteStream } from 'node:fs'
import { mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import { performance } from 'node:perf_hooks'

const DEFAULT_CAPTURE_BYTES = 64 * 1024

export async function runProcess(options) {
  const normalized = validateOptions(options)
  await prepareLog(normalized.stdoutFile)
  await prepareLog(normalized.stderrFile)
  const stdoutWriter = openWriter(normalized.stdoutFile)
  const stderrWriter = openWriter(normalized.stderrFile)
  const stdoutCapture = capture(normalized.maxCaptureBytes)
  const stderrCapture = capture(normalized.maxCaptureBytes)
  const started = performance.now()
  let child
  let timedOut = false
  let spawnError = null
  let killResult = null

  const outcome = await new Promise((resolveOutcome) => {
    try {
      child = spawn(normalized.file, normalized.args, {
        cwd: normalized.cwd,
        env: normalized.env,
        shell: false,
        windowsHide: true,
        detached: process.platform !== 'win32',
        stdio: ['ignore', 'pipe', 'pipe'],
      })
    } catch (error) {
      spawnError = error
      resolveOutcome({ code: null, signal: null })
      return
    }

    child.stdout.on('data', (chunk) => {
      stdoutCapture.append(chunk)
      stdoutWriter?.stream.write(chunk)
    })
    child.stderr.on('data', (chunk) => {
      stderrCapture.append(chunk)
      stderrWriter?.stream.write(chunk)
    })
    child.once('error', (error) => {
      spawnError = error
    })

    const timer = setTimeout(() => {
      timedOut = true
      killResult = killProcessTree(child.pid)
    }, normalized.timeoutMs)
    timer.unref()

    child.once('close', (code, signal) => {
      clearTimeout(timer)
      resolveOutcome({ code, signal })
    })
  })

  await Promise.all([closeWriter(stdoutWriter), closeWriter(stderrWriter)])
  const durationMs = performance.now() - started
  const state = spawnError
    ? 'spawn_failed'
    : timedOut
      ? 'timed_out'
      : 'exited'
  const cleanupPassed = !timedOut || killResult?.ok === true

  return {
    state,
    exitCode: outcome.code,
    signal: outcome.signal,
    timedOut,
    durationMs,
    cleanupPassed,
    cleanupDetail: killResult?.detail ?? null,
    error: spawnError?.message ?? null,
    stdout: stdoutCapture.result(),
    stderr: stderrCapture.result(),
  }
}

export function killProcessTree(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) {
    return { ok: false, detail: `invalid pid ${pid}` }
  }
  if (process.platform === 'win32') {
    const result = spawnSync('taskkill.exe', ['/PID', String(pid), '/T', '/F'], {
      encoding: 'utf8',
      windowsHide: true,
    })
    const detail = `${result.stdout || ''}${result.stderr || ''}`.trim()
    // A process may exit between the timeout firing and taskkill inspecting it.
    const absent = /not found|no running instance|not exist/i.test(detail)
    return { ok: result.status === 0 || absent, detail: detail || `taskkill exit ${result.status}` }
  }
  try {
    process.kill(-pid, 'SIGKILL')
    return { ok: true, detail: `killed process group ${pid}` }
  } catch (error) {
    if (error?.code === 'ESRCH') return { ok: true, detail: `process group ${pid} already exited` }
    return { ok: false, detail: error.message }
  }
}

function validateOptions(options) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('process options must be an object')
  }
  if (typeof options.file !== 'string' || options.file.length === 0) {
    throw new TypeError('process file must be a non-empty string')
  }
  const args = options.args ?? []
  if (!Array.isArray(args) || args.some((arg) => typeof arg !== 'string')) {
    throw new TypeError('process args must be an array of strings')
  }
  const timeoutMs = options.timeoutMs
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1) {
    throw new RangeError('process timeoutMs must be a positive safe integer')
  }
  const maxCaptureBytes = options.maxCaptureBytes ?? DEFAULT_CAPTURE_BYTES
  if (!Number.isSafeInteger(maxCaptureBytes) || maxCaptureBytes < 1) {
    throw new RangeError('process maxCaptureBytes must be a positive safe integer')
  }
  return {
    file: options.file,
    args: [...args],
    cwd: options.cwd,
    env: options.env ?? process.env,
    timeoutMs,
    maxCaptureBytes,
    stdoutFile: options.stdoutFile ?? null,
    stderrFile: options.stderrFile ?? null,
  }
}

function capture(limit) {
  const chunks = []
  let captured = 0
  let total = 0
  return {
    append(chunk) {
      const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
      total += bytes.length
      if (captured >= limit) return
      const kept = bytes.subarray(0, limit - captured)
      chunks.push(kept)
      captured += kept.length
    },
    result() {
      return {
        preview: Buffer.concat(chunks).toString('utf8'),
        totalBytes: total,
        capturedBytes: captured,
        truncated: total > captured,
      }
    },
  }
}

async function prepareLog(path) {
  if (path) await mkdir(dirname(path), { recursive: true })
}

function openWriter(path) {
  if (!path) return null
  const state = { stream: createWriteStream(path), error: null }
  // Attach immediately so ENOSPC/permission errors cannot become unhandled
  // events while the child is still running. The campaign fails after the
  // child closes and the writer reaches a terminal state.
  state.stream.on('error', (error) => {
    state.error ??= error
  })
  return state
}

async function closeWriter(state) {
  if (!state) return
  if (!state.stream.closed) {
    await new Promise((resolveClose) => {
      state.stream.once('close', resolveClose)
      state.stream.end()
    })
  }
  if (state.error) throw state.error
}
