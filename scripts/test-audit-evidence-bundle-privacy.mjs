#!/usr/bin/env node
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const tempRoot = await mkdtemp(join(tmpdir(), 'camelid-evidence-privacy-'))
const safeRoot = join(tempRoot, 'safe')
const leakRoot = join(tempRoot, 'leak')
const rawWindowsHomePath = 'C:\\Users\\alice\\models\\Bonsai-27B-Q1_0.gguf'
const jsonEscapedWindowsHomePath = rawWindowsHomePath.replaceAll('\\', '\\\\')

await mkdir(join(safeRoot, 'llama32-3b-local-smoke'), { recursive: true })
await writeFile(
  join(safeRoot, 'llama32-3b-local-smoke', 'summary.json'),
  `${JSON.stringify(
    {
      schema: 'camelid.local_smoke.v1',
      host: 'local-only mac smoke',
      model_path: '$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf',
      health_endpoint: '127.0.0.1',
      elapsed_seconds: '0.03.255.467',
      validation_status:
        'Remote Linux x86_64 validation was not attempted in this run; no fresh same-host timing/parity claim is made.',
    },
    null,
    2,
  )}\n`,
)

await mkdir(join(leakRoot, 'llama32-3b-local-smoke'), { recursive: true })
await writeFile(
  join(leakRoot, 'llama32-3b-local-smoke', 'summary.json'),
  `${JSON.stringify(
    {
      schema: 'camelid.local_smoke.v1',
      host: 'local-only mac smoke',
      model_path: '/Volumes/SSK Drive/Camelid/models/llama-3.2-3b-instruct/Llama-3.2-3B-Instruct-Q8_0.gguf',
      remote_attempt: 'ssh -i /tmp/operator.pem -o BatchMode=yes ubuntu@203.0.113.7 uname -sm',
      remote_stderr: 'ssh: connect to host validation.example port 22: Operation timed out',
      remote_rc: 'rc=255',
      windows_model_path: rawWindowsHomePath,
    },
    null,
    2,
  )}\n`,
)
await writeFile(
  join(leakRoot, 'llama32-3b-local-smoke', 'raw-windows-path.log'),
  'model_path=' + rawWindowsHomePath + '\n',
)
await writeFile(
  join(leakRoot, 'llama32-3b-local-smoke', 'SHA256SUMS'),
  [
    'abcd1234  /Users/timtoole/.openclaw/workspace/projects/Camelid/target/private-artifact.json',
    'ef567890  file:///home/tim/.cache/camelid/model.gguf',
    '',
  ].join('\n'),
)

const safe = spawnAudit(safeRoot)
assert.equal(safe.status, 0, safe.stderr || safe.stdout)
const safeReport = JSON.parse(safe.stdout)
assert.equal(safeReport.finding_count, 0)

const leaked = spawnAudit(leakRoot)
assert.notEqual(leaked.status, 0, 'mounted-volume paths in evidence bundles must fail strict privacy audit')
const leakedReport = JSON.parse(leaked.stdout)
assert.ok(leakedReport.finding_count >= 7)
const leakedPatterns = leakedReport.bundles[0].findings.map((finding) => finding.pattern).sort()
assert.ok(leakedReport.bundles[0].findings.some((finding) => finding.file.endsWith('/SHA256SUMS')))
assert.ok(leakedReport.bundles[0].findings.some((finding) => /file:\/\/\/home\/tim\/\.cache\/camelid\/model\.gguf/.test(finding.sample)))
assert.ok(leakedReport.bundles[0].findings.some((finding) => /Llama-3\.2-3B-Instruct-Q8_0\.gguf/.test(finding.sample)))
assert.ok(leakedPatterns.includes('mac_mounted_volume_path'))
assert.ok(leakedPatterns.includes('mac_home_path'))
assert.ok(leakedPatterns.includes('linux_home_path'))
assert.ok(leakedPatterns.includes('windows_home_path'))
assert.ok(leakedPatterns.includes('ipv4_literal'))
assert.ok(leakedReport.bundles[0].findings.some((finding) => finding.pattern === 'raw_ssh_command'))
assert.ok(leakedReport.bundles[0].findings.some((finding) => finding.pattern === 'raw_ssh_timeout'))
assert.ok(leakedReport.bundles[0].findings.some((finding) => finding.pattern === 'raw_ssh_rc_255'))
const windowsHomeFindings = leakedReport.bundles[0].findings.filter(
  (finding) => finding.pattern === 'windows_home_path',
)
assert.equal(windowsHomeFindings.length, 2)
assert.ok(windowsHomeFindings.some((finding) => finding.sample === rawWindowsHomePath))
assert.ok(windowsHomeFindings.some((finding) => finding.sample === jsonEscapedWindowsHomePath))

function spawnAudit(root) {
  return spawnSync(process.execPath, ['scripts/audit-evidence-bundle-privacy.mjs', '--root', root, '--strict'], {
    cwd: process.cwd(),
    encoding: 'utf8',
  })
}
