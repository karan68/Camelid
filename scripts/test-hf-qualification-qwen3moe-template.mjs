#!/usr/bin/env node

import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { canonicalGitTextBytes } from './lib/canonical-git-text.mjs'
import {
  COMMITTED_GROUNDING_INSPECTOR,
  COMPLETED_DIFF_SECTIONS,
  EXPECTED_NORMALIZED_TRANSCRIPT,
  HISTORY_PROMPT_SHA256,
  LOCAL_FILE_LIMITS,
  NORMALIZED_TRANSCRIPT_SHA256,
  NORMALIZED_TRANSCRIPT_UTF8_BYTES,
  PREFIX_BYTES,
  PREFIX_SHA256,
  Qwen3MoeTemplateQualificationError,
  SINGLE_USER_PROMPT_SHA256,
  TEMPLATE_SHA256,
  TEMPLATE_UTF8_BYTES,
  UNREACHED_DIFF_SECTIONS,
  WINDOWS_CHILD_ENV_ALLOWLIST,
  assertExactTemplate,
  buildWindowsChildEnv,
  classifyQwen3MoeTemplateQualificationError,
  expectedPromptForMessages,
  inspectCamelid,
  inspectFileIdentity,
  inspectLlamaPackage,
  inspectOracle,
  normalizeAnalyzerTranscript,
  normalizePrefixBytes,
  parseArgs,
  parsePartialAnalyzerTranscript,
  qualifyQwen3MoeTemplate,
  qwen3MoeTemplatePackAvailable,
  qwen3MoeTemplatePrefixBytesForRow,
  readGroundingSnapshot,
  runTemplateAnalyzer,
  validateShapePack,
} from './hf-qualification-qwen3moe-template.mjs'
import { HeaderInspectionError } from './hf-qualification-header.mjs'

const root = resolve('.')
const packPath = resolve(root, 'qa/prompt-packs/qwen3-moe-chat-template-shapes-v1.json')
const analyzerPath = resolve(root, 'target/reference/llama.cpp-b9632/bin/llama-template-analysis.exe')
const camelidPath = resolve(root, 'target/model-qualification/bin/camelid-ded8e95b-clean.exe')
const clone = (value) => structuredClone(value)
const sha256 = (value) => createHash('sha256').update(value).digest('hex')

const pack = JSON.parse(await readFile(packPath, 'utf8'))
assert.deepEqual(validateShapePack(pack), [])
assert.equal(Buffer.byteLength(EXPECTED_NORMALIZED_TRANSCRIPT), NORMALIZED_TRANSCRIPT_UTF8_BYTES)
assert.equal(sha256(EXPECTED_NORMALIZED_TRANSCRIPT), NORMALIZED_TRANSCRIPT_SHA256)
assert.equal(Buffer.byteLength(pack.source_template.text), TEMPLATE_UTF8_BYTES)
assert.equal(assertExactTemplate(pack.source_template.text), pack.source_template.text)
assert.equal(pack.source_template.sha256, TEMPLATE_SHA256)
for (const [pathField, hashField] of [
  ['header_receipt', 'header_receipt_sha256'],
  ['tokenizer_receipt', 'tokenizer_receipt_sha256'],
  ['template_fixture', 'template_fixture_sha256'],
]) {
  assert.equal(
    sha256(canonicalGitTextBytes(await readFile(resolve(root, pack.grounding[pathField])))),
    pack.grounding[hashField],
  )
}

const parsed = parsePartialAnalyzerTranscript(EXPECTED_NORMALIZED_TRANSCRIPT)
assert.deepEqual(parsed.completed_diff_sections, COMPLETED_DIFF_SECTIONS)
assert.deepEqual(parsed.unreached_diff_sections, UNREACHED_DIFF_SECTIONS)
assert.equal(parsed.result, 'known_partial_failure_after_four_of_nine_diff_sections')
assert.equal(parsed.analyzer_exit_code, 0)
assert.equal(parsed.analyzer_completed_banner, true)
assert.equal(Buffer.byteLength(parsed.single_user_prompt), 72)
assert.equal(sha256(parsed.single_user_prompt), SINGLE_USER_PROMPT_SHA256)
assert.equal(Buffer.byteLength(parsed.user_assistant_user_prompt), 168)
assert.equal(sha256(parsed.user_assistant_user_prompt), HISTORY_PROMPT_SHA256)
assert.equal(parsed.single_user_prompt, expectedPromptForMessages([
  { role: 'user', content: 'Hello, please help me.' },
]))
assert.equal(parsed.user_assistant_user_prompt, expectedPromptForMessages([
  { role: 'user', content: 'Hello, please help me.' },
  { role: 'assistant', content: 'I can help you with that.' },
  { role: 'user', content: 'Thank you.' },
]))
assert.equal(pack.contract.full_analyzer_success, false)
assert.equal(pack.contract.runtime_renderer_qualified, false)
assert.equal(pack.template_gate.status, 'blocked')
assert.equal(pack.support_decision, 'no_roster_change_template_preparation_only')

assert.equal(normalizePrefixBytes(), PREFIX_BYTES)
assert.equal(normalizePrefixBytes(String(PREFIX_BYTES)), PREFIX_BYTES)
assert.equal(qwen3MoeTemplatePackAvailable('qwen3_30b_a3b_q8_0'), true)
assert.equal(qwen3MoeTemplatePackAvailable('qwen3_4b_instruct_q8_0'), false)
assert.equal(qwen3MoeTemplatePrefixBytesForRow('qwen3_30b_a3b_q8_0'), PREFIX_BYTES)
assert.equal(qwen3MoeTemplatePrefixBytesForRow('unsupported'), null)
for (const value of [0, PREFIX_BYTES - 1, PREFIX_BYTES + 1, '33554431', 'nope']) {
  assert.throws(
    () => normalizePrefixBytes(value),
    (error) => error.code === 'qwen3moe_template_prefix_budget_invalid',
  )
}

const hostileInheritedEnv = {
  Path: 'kept',
  PATHEXT: '.EXE',
  SYSTEMROOT: 'C:\\Windows',
  HF_TOKEN: 'hf_private',
  GH_TOKEN: 'ghp_private',
  AWS_ACCESS_KEY_ID: 'private',
  AWS_SECRET_ACCESS_KEY: 'private',
}
assert.deepEqual(WINDOWS_CHILD_ENV_ALLOWLIST, [
  'COMSPEC', 'PATH', 'PATHEXT', 'SYSTEMDRIVE', 'SYSTEMROOT', 'TEMP', 'TMP', 'WINDIR',
])
assert.deepEqual(buildWindowsChildEnv(hostileInheritedEnv), {
  PATH: 'kept',
  PATHEXT: '.EXE',
  SYSTEMROOT: 'C:\\Windows',
})
const regularStats = (size) => ({
  size,
  isFile: () => true,
  isSymbolicLink: () => false,
})
let rejectedLocalStatCalls = 0
let rejectedLocalHashCalls = 0
await assert.rejects(
  inspectFileIdentity('symlink.exe', {
    lstatImpl: async () => ({
      size: 1,
      isFile: () => false,
      isSymbolicLink: () => true,
    }),
    statImpl: async () => { rejectedLocalStatCalls += 1; return regularStats(1) },
    sha256FileImpl: async () => { rejectedLocalHashCalls += 1; return '0'.repeat(64) },
    unavailableCode: 'qwen3moe_template_oracle_unavailable',
    mismatchCode: 'qwen3moe_template_oracle_identity_mismatch',
  }),
  (error) => error.code === 'qwen3moe_template_oracle_identity_mismatch',
)
assert.equal(rejectedLocalStatCalls, 0, 'a symlink must fail before target stat')
assert.equal(rejectedLocalHashCalls, 0, 'a symlink must fail before hashing')
await assert.rejects(
  inspectFileIdentity('wrong-size.exe', {
    lstatImpl: async () => regularStats(9),
    statImpl: async () => regularStats(9),
    sha256FileImpl: async () => { rejectedLocalHashCalls += 1; return '0'.repeat(64) },
    expectedSize: 10,
    maxBytes: 10,
    unavailableCode: 'qwen3moe_template_oracle_unavailable',
    mismatchCode: 'qwen3moe_template_oracle_identity_mismatch',
  }),
  (error) => error.code === 'qwen3moe_template_oracle_identity_mismatch',
)
assert.equal(rejectedLocalHashCalls, 0, 'a wrong pinned size must fail before hashing')

const fakeOracleNames = [
  'llama-template-analysis.exe',
  'llama-cli.exe',
  ...Array.from({ length: 49 }, (_, index) => `bounded-${index}.dll`),
]
let aggregateOracleHashCalls = 0
await assert.rejects(
  inspectLlamaPackage(analyzerPath, {
    platformInfo: () => ({ platform: 'win32', arch: 'x64' }),
    readdirImpl: async () => fakeOracleNames.map((name) => ({ name, isFile: () => true })),
    lstatImpl: async () => regularStats(32 * 1024 ** 2),
    statImpl: async () => regularStats(32 * 1024 ** 2),
    sha256FileImpl: async () => { aggregateOracleHashCalls += 1; return '0'.repeat(64) },
  }),
  (error) => error.code === 'qwen3moe_template_oracle_identity_mismatch',
)
assert.ok(aggregateOracleHashCalls > 0 && aggregateOracleHashCalls < fakeOracleNames.length,
  'the aggregate oracle-package cap must stop hashing at the crossing member')
let wrongArchiveHashCalls = 0
await assert.rejects(
  inspectLlamaPackage(analyzerPath, {
    platformInfo: () => ({ platform: 'win32', arch: 'x64' }),
    readdirImpl: async () => fakeOracleNames.map((name) => ({ name, isFile: () => true })),
    lstatImpl: async () => regularStats(1),
    statImpl: async (path) => regularStats(path.endsWith('.zip')
      ? pack.analyzer.archive_size_bytes - 1 : 1),
    sha256FileImpl: async (path) => {
      if (path.endsWith('.zip')) wrongArchiveHashCalls += 1
      return '0'.repeat(64)
    },
  }),
  (error) => error.code === 'qwen3moe_template_oracle_identity_mismatch',
)
assert.equal(wrongArchiveHashCalls, 0,
  'a wrong pinned archive size must fail before the archive is hashed')

let childExecOptions
assert.deepEqual(await inspectCamelid('camelid.exe', {
  inheritedEnv: hostileInheritedEnv,
  inspectFileIdentityImpl: async () => ({
    executable: 'camelid.exe',
    size_bytes: 1,
    binary_sha256: COMMITTED_GROUNDING_INSPECTOR.binary_sha256,
  }),
  execImpl: async (_binary, _args, options) => {
    childExecOptions = options
    return { stdout: COMMITTED_GROUNDING_INSPECTOR.version, stderr: '' }
  },
}), COMMITTED_GROUNDING_INSPECTOR)
assert.equal(childExecOptions.env.PATH, 'kept')
for (const secret of ['HF_TOKEN', 'GH_TOKEN', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']) {
  assert.equal(childExecOptions.env[secret], undefined, `inspector child must not inherit ${secret}`)
}

assert.deepEqual(parseArgs([
  '--root', '.', '--prefix-bytes=33554432', '--out', 'pack.json',
]), new Map([
  ['root', '.'], ['prefix-bytes', '33554432'], ['out', 'pack.json'],
]))
for (const argv of [
  ['positional'],
  ['--unknown', 'x'],
  ['--root'],
  ['--root', '.', '--root', '.'],
]) {
  assert.throws(
    () => parseArgs(argv),
    (error) => error.code === 'qwen3moe_template_qualification_error',
  )
}

const typed = new Qwen3MoeTemplateQualificationError('qwen3moe_template_oracle_unavailable')
typed.code = 'qwen3moe_template_source_identity_mismatch'
typed.status = 'fail'
typed.message = 'C:\\private\\secret.gguf hf_secret'
assert.deepEqual(classifyQwen3MoeTemplateQualificationError(typed), {
  status: 'blocked',
  error_code: 'qwen3moe_template_oracle_unavailable',
  reason: 'the pinned llama.cpp template analyzer package is unavailable',
})
assert.deepEqual(
  classifyQwen3MoeTemplateQualificationError(new HeaderInspectionError('header_range_invalid')),
  {
    status: 'fail',
    error_code: 'qwen3moe_template_range_invalid',
    reason: 'the bounded immutable Qwen3-MoE prefix response is invalid',
  },
)

function expectTranscriptFailure(mutator, label) {
  assert.throws(
    () => parsePartialAnalyzerTranscript(mutator(EXPECTED_NORMALIZED_TRANSCRIPT)),
    (error) => error.code === 'qwen3moe_template_transcript_mismatch',
    label,
  )
}
expectTranscriptFailure((value) => value.replace('Analysis failed:', 'Analysis succeeded:'), 'failure marker')
expectTranscriptFailure((value) => value.replace('line 34, column 31', 'line 35, column 31'), 'failure location')
expectTranscriptFailure((value) => value.replace('Cannot perform operation on null values', 'null'), 'failure text')
expectTranscriptFailure((value) => value.replace(COMPLETED_DIFF_SECTIONS[1], COMPLETED_DIFF_SECTIONS[2]), 'section order')
expectTranscriptFailure((value) => value.replace('ANALYSIS COMPLETE', 'ANALYSIS INCOMPLETE'), 'completion banner')
expectTranscriptFailure((value) => `${value}\n=== ${UNREACHED_DIFF_SECTIONS[0]} ===\n`, 'unreached section')
expectTranscriptFailure((value) => value.replace('Hello, please help me.', 'Hello, please help me!'), 'prompt bytes')

const fakeTemplatePath = 'C:\\Temp\\camelid-qwen3moe-test\\template.jinja'
const rawTranscript = EXPECTED_NORMALIZED_TRANSCRIPT
  .replace('<template.jinja>', fakeTemplatePath)
  .replace(/\n/g, '\r\n')
assert.equal(normalizeAnalyzerTranscript(rawTranscript, fakeTemplatePath), EXPECTED_NORMALIZED_TRANSCRIPT)
for (const badRaw of [
  rawTranscript.replace(fakeTemplatePath, 'C:\\Temp\\other\\template.jinja'),
  rawTranscript.replace('Analysis failed:', 'Analysis failed twice:'),
  rawTranscript.replace('\r\n', '\u001b[2J\r\n'),
  `${rawTranscript}\r`,
]) {
  assert.throws(
    () => normalizeAnalyzerTranscript(badRaw, fakeTemplatePath),
    (error) => error.code === 'qwen3moe_template_transcript_mismatch',
  )
}

function expectPackTamper(mutator, label) {
  const tampered = clone(pack)
  mutator(tampered)
  assert.notDeepEqual(validateShapePack(tampered), [], label)
}

expectPackTamper((value) => { value.unknown = true }, 'unknown top-level field')
for (const field of ['repo', 'file', 'revision', 'size_bytes', 'sha256', 'license']) {
  expectPackTamper((value) => { value.source[field] = 'forged' }, `source ${field}`)
}
for (const field of ['requested_bytes', 'received_bytes', 'prefix_sha256']) {
  expectPackTamper((value) => { value.bounded_prefix[field] = 0 }, `prefix ${field}`)
}
for (const field of ['start', 'end', 'total']) {
  expectPackTamper((value) => { value.bounded_prefix.content_range[field] = -1 }, `range ${field}`)
}
for (const field of Object.keys(pack.grounding)) {
  expectPackTamper((value) => { value.grounding[field] = 'forged' }, `grounding ${field}`)
}
expectPackTamper((value) => { value.source_template.text += ' ' }, 'template text')
expectPackTamper((value) => { value.source_template.extra = true }, 'unknown template field')
for (const field of Object.keys(pack.inspector)) {
  expectPackTamper((value) => { value.inspector[field] = 'forged' }, `inspector ${field}`)
}
for (const field of Object.keys(pack.analyzer)) {
  expectPackTamper((value) => { value.analyzer[field] = 'forged' }, `analyzer ${field}`)
}
for (const field of Object.keys(pack.analyzer.normalized_transcript)) {
  expectPackTamper((value) => { value.analyzer.normalized_transcript[field] = 'forged' }, `transcript ${field}`)
}
for (const field of Object.keys(pack.analyzer.failure)) {
  expectPackTamper((value) => { value.analyzer.failure[field] = 'forged' }, `failure ${field}`)
}
expectPackTamper((value) => { value.cases.reverse() }, 'case order')
expectPackTamper((value) => { value.cases[0].prompt += ' ' }, 'prompt text')
expectPackTamper((value) => { value.cases[0].prompt_sha256 = 'f'.repeat(64) }, 'prompt hash')
expectPackTamper((value) => { value.cases[0].messages[0].role = 'system' }, 'message role')
expectPackTamper((value) => { value.cases[0].messages[0].private = 'secret' }, 'message field')
for (const field of Object.keys(pack.contract)) {
  expectPackTamper((value) => { value.contract[field] = 'forged' }, `contract ${field}`)
}
expectPackTamper((value) => { value.typed_hold_branches.pop() }, 'typed HOLD list')
expectPackTamper((value) => { value.does_not_prove.pop() }, 'scope exclusions')
expectPackTamper((value) => { value.template_gate.status = 'pass' }, 'template gate')
expectPackTamper((value) => { value.support_decision = 'supported' }, 'support decision')
expectPackTamper((value) => {
  value.private_path = ['C:', 'Users', 'private', 'secret.gguf'].join('\\')
}, 'private path')
expectPackTamper((value) => { value.analyzer.secret = 'Bearer secret' }, 'unknown secret field')

const grounding = await readGroundingSnapshot(root)
assert.equal(grounding.template, pack.source_template.text)
assert.equal(grounding.identity.length, 3)
const groundingBytes = new Map(await Promise.all(grounding.identity.map(async (entry) => [
  resolve(root, entry.path),
  await readFile(resolve(root, entry.path)),
])))
for (const entry of grounding.identity) {
  await assert.rejects(
    readGroundingSnapshot(root, async (path) => {
      const bytes = Buffer.from(groundingBytes.get(path))
      if (path === resolve(root, entry.path)) bytes[0] ^= 1
      return bytes
    }),
    (error) => error.code === 'qwen3moe_template_grounding_mismatch',
  )
}
await assert.rejects(
  readGroundingSnapshot(root, async () => { throw new Error('offline C:\\private\\receipt') }),
  (error) => error.code === 'qwen3moe_template_grounding_unavailable',
)

if (existsSync(analyzerPath)) {
  assert.deepEqual(await inspectLlamaPackage(analyzerPath), {
    platform: 'win32-x64',
    package_file_count: 51,
    package_manifest_bytes: 4_682,
    package_manifest_sha256: 'd70bbe8beb7848396d0993ee533062c200350fd9961e2b92c799b24f94a33e93',
    archive_size_bytes: 16_899_258,
    archive_sha256: 'b835d5c5155dd2a5ed748a0351debf2ede0dc9f808757e0429f8700a11832dcd',
  })
  const liveOracle = await inspectOracle(analyzerPath)
  assert.equal(liveOracle.revision, 'acd79d603cb2e1c84c0886137b80f1ad649b6857')
  assert.equal(liveOracle.reported_revision, 'acd79d603')
  const liveAnalysis = await runTemplateAnalyzer(analyzerPath, pack.source_template.text)
  assert.equal(liveAnalysis.single_user_prompt, pack.cases[0].prompt)
  assert.equal(liveAnalysis.user_assistant_user_prompt, pack.cases[1].prompt)

  let retryAttempts = 0
  const retryAnalysis = await runTemplateAnalyzer(analyzerPath, pack.source_template.text, {
    execImpl: async (_binary, args) => {
      retryAttempts += 1
      const exact = rawForTemplatePath(args[1])
      return {
        stdout: '',
        stderr: retryAttempts === 1
          ? exact.slice(0, exact.indexOf('Analysis failed:'))
          : retryAttempts === 2 ? exact.slice(0, -81) : exact,
      }
    },
  })
  assert.equal(retryAttempts, 3, 'only a strict known transcript-prefix truncation may retry')
  assert.equal(retryAnalysis.single_user_prompt, pack.cases[0].prompt)
  let exhaustedAttempts = 0
  await assert.rejects(
    runTemplateAnalyzer(analyzerPath, pack.source_template.text, {
      execImpl: async (_binary, args) => {
        exhaustedAttempts += 1
        return { stdout: '', stderr: rawForTemplatePath(args[1]).slice(0, -81) }
      },
    }),
    (error) => error.code === 'qwen3moe_template_oracle_unavailable',
  )
  assert.equal(exhaustedAttempts, 16, 'bounded transcript recapture must fail closed')
  let invalidTranscriptAttempts = 0
  await assert.rejects(
    runTemplateAnalyzer(analyzerPath, pack.source_template.text, {
      execImpl: async (_binary, args) => {
        invalidTranscriptAttempts += 1
        return {
          stdout: '',
          stderr: rawForTemplatePath(args[1]).replace('line 34, column 31', 'line 35, column 31'),
        }
      },
    }),
    (error) => error.code === 'qwen3moe_template_transcript_mismatch',
  )
  assert.equal(invalidTranscriptAttempts, 1, 'an altered transcript must never retry')

  async function expectRunError(execImpl, code, extra = {}) {
    await assert.rejects(
      runTemplateAnalyzer(analyzerPath, pack.source_template.text, { execImpl, ...extra }),
      (error) => error.code === code,
    )
  }
  await expectRunError(async () => { throw new Error('nonzero with valid-looking transcript') }, 'qwen3moe_template_oracle_unavailable')
  await expectRunError(async () => ({ stdout: 'unexpected', stderr: '' }), 'qwen3moe_template_transcript_mismatch')
  await expectRunError(async (_binary, args) => {
    await writeFile(args[1], `${pack.source_template.text} `)
    return { stdout: '', stderr: rawForTemplatePath(args[1]) }
  }, 'qwen3moe_template_identity_mismatch')

  for (const primaryFails of [false, true]) {
    const tempRoot = await mkdtemp(join(tmpdir(), 'camelid-qwen3moe-analysis-cleanup-test-'))
    try {
      await assert.rejects(
        runTemplateAnalyzer(analyzerPath, pack.source_template.text, {
          mkdtempImpl: async () => tempRoot,
          execImpl: async (_binary, args) => {
            if (primaryFails) throw new Error('primary failure')
            return { stdout: '', stderr: rawForTemplatePath(args[1]) }
          },
          rmImpl: async () => { throw new Error('cleanup failure') },
        }),
        (error) => error.code === 'qwen3moe_template_cleanup_failed',
      )
    } finally {
      await rm(tempRoot, { recursive: true, force: true })
    }
  }
}

if (existsSync(camelidPath)) {
  assert.deepEqual(await inspectCamelid(camelidPath), COMMITTED_GROUNDING_INSPECTOR)
  await assert.rejects(
    inspectCamelid(camelidPath, {
      execImpl: async () => ({ stdout: 'camelid v9.9.9-1-g11111111', stderr: '' }),
    }),
    (error) => error.code === 'qwen3moe_template_inspector_identity_mismatch',
  )
}

function rawForTemplatePath(templatePath) {
  return EXPECTED_NORMALIZED_TRANSCRIPT
    .replace('<template.jinja>', templatePath)
    .replace(/\n/g, '\r\n')
}

const roster = JSON.parse(await readFile(resolve(root, 'qa/model-qualification/phase1-roster.json'), 'utf8'))
const row = roster.rows.find((candidate) => candidate.id === 'qwen3_30b_a3b_q8_0')
const fakePrefix = Buffer.alloc(PREFIX_BYTES, 0xa5)
const lock = {
  schema: 'camelid.hf-source-lock/v1',
  repo: row.source.repo,
  file: row.source.file,
  revision: row.source.revision,
  size_bytes: row.identity.size_bytes,
  sha256: row.identity.sha256,
  license: row.source.license,
  download_url: `https://huggingface.co/${row.source.repo}/resolve/${row.source.revision}/${row.source.file}?download=true`,
  access: { gated: false, private: false, disabled: false },
}
const range = {
  bytes: fakePrefix,
  requested_bytes: PREFIX_BYTES,
  prefix_sha256: PREFIX_SHA256,
  content_range: { start: 0, end: PREFIX_BYTES - 1, total: row.identity.size_bytes },
}
const oracleIdentity = {
  project: pack.analyzer.project,
  build: pack.analyzer.build,
  revision: pack.analyzer.revision,
  reported_revision: pack.analyzer.reported_revision,
  analyzer_executable: pack.analyzer.analyzer_executable,
  analyzer_binary_sha256: pack.analyzer.analyzer_binary_sha256,
  companion_executable: pack.analyzer.companion_executable,
  companion_binary_sha256: pack.analyzer.companion_binary_sha256,
  platform: pack.analyzer.platform,
  package_file_count: pack.analyzer.package_file_count,
  package_manifest_bytes: pack.analyzer.package_manifest_bytes,
  package_manifest_sha256: pack.analyzer.package_manifest_sha256,
  archive_size_bytes: pack.analyzer.archive_size_bytes,
  archive_sha256: pack.analyzer.archive_sha256,
}
const baseDeps = {
  roster,
  readGrounding: async () => clone(grounding),
  inspectCamelid: async () => clone(COMMITTED_GROUNDING_INSPECTOR),
  inspectOracle: async () => clone(oracleIdentity),
  resolveSource: async () => clone(lock),
  fetchPrefix: async () => range,
  writeFileImpl: async () => {},
  inspectPrefixFile: async () => ({ size_bytes: PREFIX_BYTES, binary_sha256: PREFIX_SHA256 }),
  inspectPrefix: async () => ({ metadata: { 'tokenizer.chat_template': pack.source_template.text } }),
  assertMetadata: () => ({
    chat_template_utf8_bytes: TEMPLATE_UTF8_BYTES,
    chat_template_sha256: TEMPLATE_SHA256,
  }),
  runAnalyzer: async () => clone(parsed),
  sha256Impl: (bytes) => bytes.length === PREFIX_BYTES && bytes[0] === 0xa5
    ? PREFIX_SHA256
    : sha256(bytes),
}
const qualificationOptions = {
  root,
  binary: 'redacted-camelid',
  analyzer: 'redacted-analyzer',
  prefixBytes: PREFIX_BYTES,
}
assert.deepEqual(validateShapePack(await qualifyQwen3MoeTemplate(qualificationOptions, baseDeps)), [])

let reusedLockResolves = 0
assert.deepEqual(validateShapePack(await qualifyQwen3MoeTemplate(
  { ...qualificationOptions, initialLock: clone(lock) },
  {
    ...baseDeps,
    resolveSource: async () => {
      reusedLockResolves += 1
      return clone(lock)
    },
  },
)), [])
assert.equal(reusedLockResolves, 1, 'an injected initial lock must still be re-resolved postflight')

let preflightCalls = 0
await assert.rejects(
  qualifyQwen3MoeTemplate(
    { ...qualificationOptions, prefixBytes: PREFIX_BYTES - 1 },
    {
      ...baseDeps,
      readGrounding: async () => { preflightCalls += 1; return clone(grounding) },
    },
  ),
  (error) => error.code === 'qwen3moe_template_prefix_budget_invalid',
)
assert.equal(preflightCalls, 0, 'invalid prefix budget must fail before disk or network access')

await assert.rejects(
  qualifyQwen3MoeTemplate(
    { ...qualificationOptions, initialLock: { ...clone(lock), sha256: 'f'.repeat(64) } },
    baseDeps,
  ),
  (error) => error.code === 'qwen3moe_template_source_identity_mismatch',
)
await assert.rejects(
  qualifyQwen3MoeTemplate(qualificationOptions, {
    ...baseDeps,
    resolveSource: async () => { throw new Error('offline hf_secret C:\\private') },
  }),
  (error) => error.code === 'qwen3moe_template_source_unavailable',
)

for (const [headerCode, expectedCode] of [
  ['header_request_failed', 'qwen3moe_template_range_unavailable'],
  ['header_range_invalid', 'qwen3moe_template_range_invalid'],
]) {
  await assert.rejects(
    qualifyQwen3MoeTemplate(qualificationOptions, {
      ...baseDeps,
      fetchPrefix: async () => { throw new HeaderInspectionError(headerCode) },
    }),
    (error) => classifyQwen3MoeTemplateQualificationError(error).error_code === expectedCode,
  )
}
for (const mutateRange of [
  (value) => { value.requested_bytes -= 1 },
  (value) => { value.prefix_sha256 = 'f'.repeat(64) },
  (value) => { value.content_range.end -= 1 },
]) {
  await assert.rejects(
    qualifyQwen3MoeTemplate(qualificationOptions, {
      ...baseDeps,
      fetchPrefix: async () => {
        const value = { ...range, content_range: { ...range.content_range } }
        mutateRange(value)
        return value
      },
    }),
    (error) => error.code === 'qwen3moe_template_prefix_identity_mismatch',
  )
}

let inspectorReads = 0
await assert.rejects(
  qualifyQwen3MoeTemplate(qualificationOptions, {
    ...baseDeps,
    inspectCamelid: async () => {
      inspectorReads += 1
      return inspectorReads === 1
        ? clone(COMMITTED_GROUNDING_INSPECTOR)
        : { ...clone(COMMITTED_GROUNDING_INSPECTOR), binary_sha256: 'f'.repeat(64) }
    },
  }),
  (error) => error.code === 'qwen3moe_template_inspector_changed',
)
let oracleReads = 0
await assert.rejects(
  qualifyQwen3MoeTemplate(qualificationOptions, {
    ...baseDeps,
    inspectOracle: async () => {
      oracleReads += 1
      return oracleReads === 1
        ? clone(oracleIdentity)
        : { ...clone(oracleIdentity), archive_sha256: 'f'.repeat(64) }
    },
  }),
  (error) => error.code === 'qwen3moe_template_oracle_changed',
)
let lockReads = 0
await assert.rejects(
  qualifyQwen3MoeTemplate(qualificationOptions, {
    ...baseDeps,
    resolveSource: async () => {
      lockReads += 1
      return lockReads === 1 ? clone(lock) : { ...clone(lock), download_url: `${lock.download_url}?changed=1` }
    },
  }),
  (error) => error.code === 'qwen3moe_template_source_identity_mismatch',
)
let groundingReads = 0
await assert.rejects(
  qualifyQwen3MoeTemplate(qualificationOptions, {
    ...baseDeps,
    readGrounding: async () => {
      groundingReads += 1
      const snapshot = clone(grounding)
      if (groundingReads > 1) snapshot.identity[0].sha256 = 'f'.repeat(64)
      return snapshot
    },
  }),
  (error) => error.code === 'qwen3moe_template_source_changed',
)
let prefixReads = 0
await assert.rejects(
  qualifyQwen3MoeTemplate(qualificationOptions, {
    ...baseDeps,
    inspectPrefixFile: async () => {
      prefixReads += 1
      return prefixReads === 1
        ? { size_bytes: PREFIX_BYTES, binary_sha256: PREFIX_SHA256 }
        : { size_bytes: PREFIX_BYTES, binary_sha256: 'f'.repeat(64) }
    },
  }),
  (error) => error.code === 'qwen3moe_template_prefix_identity_mismatch',
)

for (const primaryFails of [false, true]) {
  const tempRoot = await mkdtemp(join(tmpdir(), 'camelid-qwen3moe-prefix-cleanup-test-'))
  try {
    await assert.rejects(
      qualifyQwen3MoeTemplate(qualificationOptions, {
        ...baseDeps,
        mkdtempImpl: async () => tempRoot,
        inspectPrefix: primaryFails
          ? async () => { throw new Error('primary failure') }
          : baseDeps.inspectPrefix,
        rmImpl: async () => { throw new Error('cleanup failure') },
      }),
      (error) => error.code === 'qwen3moe_template_cleanup_failed',
    )
  } finally {
    await rm(tempRoot, { recursive: true, force: true })
  }
}

console.log('Qwen3-MoE bounded partial template preparation tests passed')
