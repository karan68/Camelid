#!/usr/bin/env node

import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const scriptDir = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(scriptDir, '..')
const publisher = join(scriptDir, 'publish-public-evidence-bundle.mjs')
const privacyAudit = join(scriptDir, 'audit-evidence-bundle-privacy.mjs')
const tempDir = await mkdtemp(join(tmpdir(), 'camelid-publish-evidence-test-'))
const sourceDir = join(tempDir, 'source')
const outputDir = join(tempDir, 'public')

try {
  await mkdir(sourceDir, { recursive: true })
  const macHome = ['', 'Users', 'private-operator'].join('/')
  const volumeRoot = ['', 'Volumes', 'private-volume'].join('/')
  const evidenceRoot = ['', 'private', 'tmp', 'camelid-lfm2-mac-handoff'].join('/')
  const paths = {
    workspace: `${macHome}/Documents/Camelid/qa/output.json`,
    frontend_script: `${macHome}/Documents/Camelid/frontend/scripts/smoke.mjs`,
    state: `${macHome}/.local/state/Camelid/logs/camelid.log`,
    model: `${volumeRoot}/models/lfm2/LFM2.5-2.6B-Q8_0.gguf`,
    binary: `${volumeRoot}/cargo-targets/global/release/camelid`,
    comparator: `${volumeRoot}/llama.cpp-metal-parity/build-metal/bin/llama-server`,
    evidence: `${evidenceRoot}/lfm2-context-512`,
    node: '/opt/homebrew/Cellar/node@22/22.22.2_2/bin/node',
    home_fallback: `${macHome}/Downloads/receipt.json`,
    volume_fallback: `${volumeRoot}/other/receipt.json`,
    loopback_api: 'http://127.0.0.1:8251/v1/health',
    private_api: 'http://10.0.0.42:8251/v1/health',
  }
  await writeFile(join(sourceDir, 'paths.json'), `${JSON.stringify(paths, null, 2)}\n`)
  await writeFile(join(sourceDir, 'paths.log'), `${Object.values(paths).join('\n')}\n`)

  await execFileAsync(process.execPath, [publisher, '--src', sourceDir, '--dst', outputDir], { cwd: repoRoot })
  await execFileAsync(process.execPath, [privacyAudit, '--root', outputDir, '--strict'], { cwd: repoRoot })

  const published = JSON.parse(await readFile(join(outputDir, 'paths.json'), 'utf8'))
  assert.equal(published.workspace, '$CAMELID_WORKTREE/qa/output.json')
  assert.equal(published.frontend_script, 'frontend/scripts/smoke.mjs')
  assert.equal(published.state, '$CAMELID_STATE_DIR/logs/camelid.log')
  assert.equal(published.model, '$CAMELID_MODEL_ROOT/lfm2/LFM2.5-2.6B-Q8_0.gguf')
  assert.equal(published.binary, '$CAMELID_BIN')
  assert.equal(published.comparator, '$CAMELID_LLAMA_CPP_BIN/llama-server')
  assert.equal(published.evidence, '$CAMELID_EVIDENCE_TMP/lfm2-context-512')
  assert.equal(published.node, 'node')
  assert.equal(published.home_fallback, '$HOME/Downloads/receipt.json')
  assert.equal(published.volume_fallback, '$CAMELID_EXTERNAL_VOLUME/other/receipt.json')
  assert.equal(published.loopback_api, 'http://127.0.0.1:8251/v1/health')
  assert.equal(published.private_api, 'http://canonical-private-ubuntu-validation-host:8251/v1/health')

  const publishedLog = await readFile(join(outputDir, 'paths.log'), 'utf8')
  assert.doesNotMatch(publishedLog, /private-operator|private-volume|\/Users\/|\/Volumes\/|\/private\/tmp\//)
  const checksumLines = (await readFile(join(outputDir, 'SHA256SUMS'), 'utf8')).trim().split('\n')
  assert.equal(checksumLines.length, 2)
  for (const line of checksumLines) {
    const match = /^([a-f0-9]{64})  (.+)$/.exec(line)
    assert.ok(match, `invalid checksum line: ${line}`)
    const actual = createHash('sha256').update(await readFile(join(outputDir, match[2]))).digest('hex')
    assert.equal(actual, match[1], `${match[2]} checksum`)
  }
  console.log('public evidence publisher self-test passed')
} finally {
  await rm(tempDir, { recursive: true, force: true })
}
