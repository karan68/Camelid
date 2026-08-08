/* Shared Chrome launch for every puppeteer-driven script in this directory.
 *
 * Booting Chrome is the one step in these smokes that can fail for reasons that
 * have nothing to do with the product, and when it does the job log is a
 * `TimeoutError` from `ChromeLauncher.launch` with no assertion ever having run.
 * That is indistinguishable from a real regression at a glance, and it reds the
 * release gate. Three things made it likelier than it needed to be:
 *
 *   - puppeteer's default launch timeout is 30s, while the whole first-run-card
 *     smoke takes ~25s wall-clock on a healthy runner. There was almost no
 *     headroom for a cold or loaded runner.
 *   - a single slow boot was fatal; nothing retried it.
 *   - the CI smokes passed no `args` at all, so Chrome tried to use its sandbox
 *     on runners where an AppArmor profile restricts unprivileged user
 *     namespaces, and tried to use a container-sized /dev/shm.
 *
 * So the launch -- and ONLY the launch -- gets headroom, a bounded retry and the
 * standard CI args. Nothing here retries a navigation, a selector or an
 * assertion: once this returns a browser, every later failure is the product's
 * and still fails loudly on the first occurrence.
 */
import { existsSync } from 'node:fs'
import puppeteer from 'puppeteer-core'

/* Superset of the per-script lists this replaced, most specific first. The two
 * env vars come first so a runner or a developer can always force the choice. */
const CHROME_PATHS = [
  process.env.PUPPETEER_EXECUTABLE_PATH,
  process.env.CHROME_PATH,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
]

/* 4x puppeteer's default. Long enough that only a genuinely stuck Chrome hits
 * it, short enough that a stuck one is still reported inside a job's lifetime. */
const LAUNCH_TIMEOUT_MS = Number(process.env.CAMELID_LAUNCH_TIMEOUT_MS || 120000)
const LAUNCH_ATTEMPTS = Number(process.env.CAMELID_LAUNCH_ATTEMPTS || 3)

/* Only under CI. Locally the sandbox stays on, because there it both works and
 * is worth having. */
const CI_ARGS = ['--no-sandbox', '--disable-dev-shm-usage']

const sleep = (ms) => new Promise((done) => setTimeout(done, ms))

/** Resolve the browser binary, or throw naming the smoke that needed it. */
export function resolveChromeExecutable(purpose = 'this smoke') {
  const executablePath = CHROME_PATHS.filter(Boolean).find(existsSync)
  if (!executablePath) throw new Error(`Chrome or Edge is required for ${purpose}`)
  return executablePath
}

/**
 * Launch Chrome for a smoke. Accepts everything `puppeteer.launch` does; the
 * caller's `args` are kept and the CI args are added to them.
 *
 * Retries the launch itself up to LAUNCH_ATTEMPTS times. If every attempt
 * fails the last error is rethrown unchanged, so a browser that genuinely
 * cannot start still fails the job.
 */
export async function launchBrowser({ purpose = 'this smoke', args = [], ...options } = {}) {
  const executablePath = options.executablePath || resolveChromeExecutable(purpose)
  const launchArgs = [...args]
  if (process.env.CI) for (const arg of CI_ARGS) if (!launchArgs.includes(arg)) launchArgs.push(arg)

  let lastError
  for (let attempt = 1; attempt <= LAUNCH_ATTEMPTS; attempt += 1) {
    try {
      return await puppeteer.launch({
        timeout: LAUNCH_TIMEOUT_MS,
        ...options,
        executablePath,
        args: launchArgs,
      })
    } catch (error) {
      lastError = error
      if (attempt === LAUNCH_ATTEMPTS) break
      /* Announce it: a retried launch is still a slow runner worth noticing, and
       * a green job that needed two attempts should say so in the log. */
      console.warn(`browser launch attempt ${attempt}/${LAUNCH_ATTEMPTS} failed (${error.message}); retrying`)
      await sleep(1000 * attempt)
    }
  }
  throw lastError
}
