#!/usr/bin/env node

import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import {
  DATE_PLACEHOLDER,
  DEFAULT_THINKING_INSTRUCTION,
  PREFIX_BYTES,
  PREFIX_SHA256,
  SmolLM3TemplateQualificationError,
  TEMPLATE_SHA256,
  TEMPLATE_UTF8_BYTES,
  assertExactTemplate,
  buildShapePack,
  classifySmolLM3TemplateQualificationError,
  expectedDefaultPrompt,
  expectedPromptForMessages,
  inspectLlamaPackage,
  inspectOracle,
  normalizeOracleDate,
  normalizePrefixBytes,
  parseAnalyzerDefaultPrompt,
  qualifySmolLM3Template,
  runTemplateAnalyzer,
  smollm3TemplatePackAvailable,
  smollm3TemplatePrefixBytesForRow,
  validateShapePack,
} from './hf-qualification-smollm3-template.mjs'
import { HeaderInspectionError } from './hf-qualification-header.mjs'

const root = resolve('.')
const packPath = resolve(root, 'qa/prompt-packs/smollm3-chat-template-shapes-v1.json')
const analyzerPath = resolve(
  root,
  'target/reference/llama.cpp-b9632/bin/llama-template-analysis.exe',
)
const clone = (value) => structuredClone(value)
const sha256 = (value) => createHash('sha256').update(value).digest('hex')

const singleWithoutGeneration = expectedPromptForMessages(
  '10 August 2026',
  [{ role: 'user', content: 'Hello, please help me.' }],
  false,
)
const singleWithGeneration = expectedDefaultPrompt('10 August 2026')
const userAssistant = expectedPromptForMessages('10 August 2026', [
  { role: 'user', content: 'Hello, please help me.' },
  { role: 'assistant', content: 'I can help you with that.' },
], false)
const userAssistantUser = expectedPromptForMessages('10 August 2026', [
  { role: 'user', content: 'Hello, please help me.' },
  { role: 'assistant', content: 'I can help you with that.' },
  { role: 'user', content: 'Thank you.' },
], false)

const titles = [
  'Diff: With vs Without Tools (single user message)',
  'Diff: With vs Without add_generation_prompt (single user message)',
  'Diff: With vs Without reasoning_content (user, assistant)',
  'Diff: With vs Without reasoning_content (user, assistant, user)',
  'Diff: With vs Without tool call (user, assistant)',
  'Diff: With vs Without tool call (user, assistant, user)',
  'Diff: One vs Two tool calls (user, assistant)',
  'Diff: One vs Two tool calls (user, assistant, user)',
  'Diff: Tool call with vs without reasoning_content (user, assistant)',
]

function diff(title, prefix, suffix = '', left = '', right = '') {
  return `=== ${title} ===\n`
    + `Common Prefix: '${prefix}'\n`
    + `Common Suffix: '${suffix}'\n`
    + `Left (difference): '${left}'\n`
    + `Right (difference): '${right}'\n`
}

function syntheticAnalyzerTranscript() {
  const sections = [
    diff(titles[0], singleWithoutGeneration),
    diff(titles[1], singleWithoutGeneration, '', '', '<|im_start|>assistant\n'),
    diff(titles[2], userAssistant),
    diff(titles[3], userAssistantUser),
    // An apostrophe inside a quoted multiline field proves parsing is bounded
    // by the exact following label, not by the first quote character.
    diff(titles[4], singleWithoutGeneration, '', "can't", ''),
    diff(titles[5], singleWithoutGeneration),
    diff(titles[6], singleWithoutGeneration),
    diff(titles[7], singleWithoutGeneration),
    diff(titles[8], singleWithoutGeneration),
  ]
  return '\u001b[1m\u001b[38;5;126mTEMPLATE ANALYSIS TOOL\u001b[0m\n'
    + 'ANALYZING TEMPLATE: C:\\redacted\\template.jinja\n'
    + '=== Template Capabilities (from jinja::caps) ===\n'
    + 'supports_tools: false\n'
    + 'supports_tool_calls: false\n'
    + 'supports_system_role: true\n'
    + 'supports_parallel_tool_calls: false\n'
    + 'supports_typed_content: false\n'
    + 'supports_string_content: true\n\n'
    + sections.join('\n')
    + '\n=== Checking Reasoning Variables ===\n'
    + 'No reasoning/thinking-related variables were queried by the template\n\n'
    + 'ANALYSIS COMPLETE\n'
}

const syntheticAnalysis = parseAnalyzerDefaultPrompt(syntheticAnalyzerTranscript())
assert.equal(syntheticAnalysis.normalized_single_user_prompt, expectedDefaultPrompt(DATE_PLACEHOLDER))
assert.equal(syntheticAnalysis.standard_tools_inert_for_analyzer_case, true)

if (process.env.UPDATE_SMOLLM3_TEMPLATE_PACK === '1') {
  assert.ok(existsSync(analyzerPath), 'fixture refresh requires the pinned local analyzer package')
  const [roster, headerReceipt, rawInspection, oracle] = await Promise.all([
    readFile(resolve(root, 'qa/model-qualification/phase1-roster.json'), 'utf8').then(JSON.parse),
    readFile(resolve(root, 'qa/model-qualification/smollm3-3b-q8-header-inspection.json'), 'utf8').then(JSON.parse),
    readFile(resolve(root, 'target/model-qualification/smollm3-header-inspection.json'), 'utf8').then(JSON.parse),
    inspectOracle(analyzerPath),
  ])
  const row = roster.rows.find((candidate) => candidate.id === 'smollm3_3b_q8_0')
  const template = assertExactTemplate(rawInspection.inspection.metadata['tokenizer.chat_template'])
  const analysis = await runTemplateAnalyzer(analyzerPath, template)
  const inspector = {
    version: headerReceipt.inspector.version,
    binary_sha256: headerReceipt.inspector.binary_sha256,
    provenance: {
      source_head: headerReceipt.inspector.source_head,
      source_tracked_dirty: false,
      binary_commit_abbrev: headerReceipt.inspector.binary_commit_abbrev,
      binary_reports_dirty: headerReceipt.inspector.binary_reports_dirty,
      binary_matches_source_head: headerReceipt.inspector.binary_matches_source_head,
      clean_current_head: headerReceipt.inspector.clean_current_head,
    },
  }
  const range = {
    bytes: Buffer.alloc(PREFIX_BYTES),
    requested_bytes: PREFIX_BYTES,
    prefix_sha256: PREFIX_SHA256,
    content_range: headerReceipt.range.content_range,
  }
  const refreshed = buildShapePack({ row, range, template, oracle, inspector, analysis })
  assert.deepEqual(validateShapePack(refreshed), [])
  await writeFile(packPath, `${JSON.stringify(refreshed, null, 2)}\n`)
}

const pack = JSON.parse(await readFile(packPath, 'utf8'))
assert.deepEqual(validateShapePack(pack), [])
assert.equal(
  sha256(await readFile(resolve(root, pack.grounding.header_receipt))),
  pack.grounding.header_receipt_sha256,
)
assert.equal(
  sha256(await readFile(resolve(root, pack.grounding.tokenizer_receipt))),
  pack.grounding.tokenizer_receipt_sha256,
)
assert.equal(Buffer.byteLength(pack.source_template.text), TEMPLATE_UTF8_BYTES)
assert.equal(assertExactTemplate(pack.source_template.text), pack.source_template.text)
assert.equal(pack.source_template.sha256, TEMPLATE_SHA256)
assert.equal(
  pack.cases[0].normalized_prompt.includes(
    `${DEFAULT_THINKING_INSTRUCTION}\n\n<|im_start|>user`,
  ),
  true,
)
assert.equal(
  pack.cases[0].normalized_prompt.includes(
    `${DEFAULT_THINKING_INSTRUCTION}\n\n<|im_end|>`,
  ),
  false,
  'the source template intentionally omits the ordinary synthetic-system terminator',
)

assert.equal(normalizePrefixBytes(), PREFIX_BYTES)
assert.equal(normalizePrefixBytes(String(PREFIX_BYTES)), PREFIX_BYTES)
assert.equal(smollm3TemplatePackAvailable('smollm3_3b_q8_0'), true)
assert.equal(smollm3TemplatePackAvailable('qwen2_5_0_5b_instruct_q8_0'), false)
assert.equal(smollm3TemplatePrefixBytesForRow('smollm3_3b_q8_0'), PREFIX_BYTES)
assert.equal(smollm3TemplatePrefixBytesForRow('unsupported'), null)
for (const value of [0, PREFIX_BYTES - 1, PREFIX_BYTES + 1, '33554431', 'nope']) {
  assert.throws(
    () => normalizePrefixBytes(value),
    (error) => error.code === 'smollm3_template_prefix_budget_invalid',
  )
}

const typed = new SmolLM3TemplateQualificationError('smollm3_template_oracle_unavailable')
typed.code = 'smollm3_template_source_identity_mismatch'
typed.status = 'fail'
typed.message = 'C:\\private\\secret.gguf hf_secret'
assert.deepEqual(classifySmolLM3TemplateQualificationError(typed), {
  status: 'blocked',
  error_code: 'smollm3_template_oracle_unavailable',
  reason: 'the pinned llama.cpp template analyzer package is unavailable',
})

for (const value of [
  'Today Date: 10 August 2026',
  singleWithGeneration.replace('10 August 2026', '00 August 2026'),
  singleWithGeneration.replace('10 August 2026', '31 February 2026'),
  singleWithGeneration.replace('10 August 2026', '10 August 0000'),
  singleWithGeneration.replace('10 August 2026', '10 Smarch 2026'),
  `${singleWithGeneration}\nKnowledge Cutoff Date: June 2025\nToday Date: 10 August 2026\nReasoning Mode: /think\n`,
  singleWithGeneration.replace('10 August 2026', DATE_PLACEHOLDER),
]) {
  assert.throws(
    () => normalizeOracleDate(value),
    (error) => error.code === 'smollm3_template_oracle_output_invalid',
  )
}

function expectTranscriptFailure(mutator) {
  const transcript = mutator(syntheticAnalyzerTranscript())
  assert.throws(
    () => parseAnalyzerDefaultPrompt(transcript),
    (error) => error.code === 'smollm3_template_oracle_output_invalid',
  )
}
expectTranscriptFailure((value) => value.replace(titles[5], 'missing section'))
expectTranscriptFailure((value) => `${value}\n${diff(titles[5], singleWithoutGeneration)}`)
expectTranscriptFailure((value) => value.replace(
  `=== ${titles[1]} ===`,
  `=== ${titles[3]} ===`,
))
expectTranscriptFailure((value) => value.replace('ANALYSIS COMPLETE', 'Analysis failed: secret path'))
expectTranscriptFailure((value) => value.replace('\u001b[1m', '\u001b[2J'))
expectTranscriptFailure((value) => value.replace('10 August 2026', '11 August 2026'))

async function expectRunError(execImpl, code, extra = {}) {
  await assert.rejects(
    runTemplateAnalyzer(analyzerPath, pack.source_template.text, { execImpl, ...extra }),
    (error) => error.code === code,
  )
}
await expectRunError(
  async () => {
    const error = new Error('nonzero')
    error.stderr = syntheticAnalyzerTranscript()
    throw error
  },
  'smollm3_template_oracle_unavailable',
)
await expectRunError(
  async (_binary, args) => {
    await writeFile(args[1], `${pack.source_template.text} `)
    return { stdout: '', stderr: syntheticAnalyzerTranscript() }
  },
  'smollm3_template_identity_mismatch',
)

for (const primaryFails of [false, true]) {
  const tempRoot = await mkdtemp(join(tmpdir(), 'camelid-smollm3-cleanup-test-'))
  try {
    await assert.rejects(
      runTemplateAnalyzer(analyzerPath, pack.source_template.text, {
        mkdtempImpl: async () => tempRoot,
        execImpl: async () => {
          if (primaryFails) throw new Error('primary failure')
          return { stdout: '', stderr: syntheticAnalyzerTranscript() }
        },
        rmImpl: async () => { throw new Error('cleanup failure') },
      }),
      (error) => error.code === 'smollm3_template_cleanup_failed',
    )
  } finally {
    await rm(tempRoot, { recursive: true, force: true })
  }
}

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
  const exitOneOracle = await inspectOracle(analyzerPath, {
    execImpl: async () => {
      const error = new Error('historical Windows launcher exit')
      error.code = 1
      error.stdout = 'version: 9632 (acd79d603)\n'
      error.stderr = ''
      throw error
    },
  })
  assert.equal(exitOneOracle.reported_revision, 'acd79d603')
  for (const partialFailure of [
    { code: 1, killed: true },
    { code: 1, signal: 'SIGTERM' },
    { code: 'ETIMEDOUT' },
    { code: 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER' },
    { code: 'ENOENT' },
  ]) {
    await assert.rejects(
      inspectOracle(analyzerPath, {
        execImpl: async () => {
          const error = Object.assign(new Error('partial version must not certify'), partialFailure)
          error.stdout = 'version: 9632 (acd79d603)\n'
          throw error
        },
      }),
      (error) => error.code === 'smollm3_template_oracle_unavailable',
    )
  }
  const liveAnalysis = await runTemplateAnalyzer(analyzerPath, pack.source_template.text)
  assert.equal(liveAnalysis.normalized_single_user_prompt, pack.cases[0].normalized_prompt)
}

function expectPackTamper(mutator, message) {
  const tampered = clone(pack)
  mutator(tampered)
  assert.notDeepEqual(validateShapePack(tampered), [], message)
}

for (const field of ['repo', 'file', 'revision', 'size_bytes', 'sha256', 'license']) {
  expectPackTamper((value) => { value.source[field] = 'forged' }, `source ${field}`)
}
for (const field of ['requested_bytes', 'received_bytes', 'prefix_sha256']) {
  expectPackTamper((value) => { value.bounded_prefix[field] = 0 }, `range ${field}`)
}
for (const field of ['start', 'end', 'total']) {
  expectPackTamper((value) => { value.bounded_prefix.content_range[field] = -1 }, `Content-Range ${field}`)
}
expectPackTamper((value) => { value.source_template.text += ' ' }, 'template text')
expectPackTamper((value) => { value.grounding.header_receipt_sha256 = 'f'.repeat(64) }, 'header grounding')
expectPackTamper((value) => { value.grounding.tokenizer_receipt = 'forged.json' }, 'tokenizer grounding')
for (const field of [
  'project', 'build', 'revision', 'reported_revision', 'analyzer_executable',
  'analyzer_binary_sha256', 'companion_executable', 'companion_binary_sha256',
  'platform', 'package_file_count', 'package_manifest_bytes', 'package_manifest_sha256',
  'archive_size_bytes', 'archive_sha256', 'mode', 'executable_paths_redacted',
  'default_case_only', 'analyzer_date_value_redacted',
]) {
  expectPackTamper((value) => { value.oracle[field] = 'forged' }, `oracle ${field}`)
}
for (const field of [
  'version', 'binary_sha256', 'source_head', 'source_tracked_dirty',
  'binary_commit_abbrev', 'binary_reports_dirty', 'binary_matches_source_head',
  'clean_current_head', 'binary_path_redacted',
]) {
  expectPackTamper((value) => { value.inspector[field] = 'forged' }, `inspector ${field}`)
}
expectPackTamper((value) => {
  value.inspector = {
    version: 'camelid v9.9.9-99-g11111111',
    binary_sha256: '2'.repeat(64),
    source_head: '1'.repeat(40),
    source_tracked_dirty: false,
    binary_commit_abbrev: '11111111',
    binary_reports_dirty: false,
    binary_matches_source_head: true,
    clean_current_head: true,
    binary_path_redacted: true,
  }
}, 'coherent forged inspector projection')
expectPackTamper((value) => { value.cases.reverse() }, 'case order')
expectPackTamper((value) => { value.cases[0].messages[0].role = 'system' }, 'case messages')
expectPackTamper((value) => { value.cases[0].enable_thinking = false }, 'case thinking')
expectPackTamper((value) => { value.cases[0].add_generation_prompt = false }, 'case generation prompt')
expectPackTamper((value) => { value.dynamic_date.format = '%Y-%m-%d' }, 'date contract')
expectPackTamper((value) => { value.contract.allowed_roles.push('system') }, 'allowed roles')
for (const field of [
  'message_input_stage', 'text_only_content_parts', 'exact_role_bytes_required',
  'messages_nonempty', 'content_nonempty',
  'history_starts_with', 'history_must_end_with', 'history_strictly_alternating',
  'injected_date',
]) {
  expectPackTamper((value) => { value.contract[field] = 'forged' }, `contract ${field}`)
}
expectPackTamper((value) => { value.contract.exact_adjacency = '<|im_end|>' }, 'adjacency')
expectPackTamper((value) => { value.typed_hold_branches.splice(3, 1) }, 'typed HOLD list')
expectPackTamper((value) => { value.does_not_prove.pop() }, 'scope exclusions')
expectPackTamper((value) => { value.private_path = 'C:\\Users\\private\\secret.gguf' }, 'privacy')
expectPackTamper((value) => { value.runtime_chat_enabled = true }, 'unknown runtime claim')
expectPackTamper((value) => { value.source.download_url = 'relative-secret' }, 'unknown source field')
expectPackTamper((value) => { value.oracle.secret = 'SUPERSECRET' }, 'unknown oracle field')
expectPackTamper((value) => { value.cases[0].messages[0].name = 'forged' }, 'unknown message field')

const roster = JSON.parse(await readFile(resolve(root, 'qa/model-qualification/phase1-roster.json'), 'utf8'))
const row = roster.rows.find((candidate) => candidate.id === 'smollm3_3b_q8_0')
const fakePrefix = Buffer.alloc(PREFIX_BYTES)
const sourceState = { head: pack.inspector.source_head, tracked_dirty: false }
const inspectorIdentity = {
  version: pack.inspector.version,
  binary_sha256: pack.inspector.binary_sha256,
  executable: 'camelid.exe',
  provenance: {
    source_head: pack.inspector.source_head,
    source_tracked_dirty: false,
    binary_commit_abbrev: pack.inspector.binary_commit_abbrev,
    binary_reports_dirty: false,
    binary_matches_source_head: true,
    clean_current_head: true,
  },
}
const oracleIdentity = {
  project: 'llama.cpp',
  build: 9_632,
  revision: pack.oracle.revision,
  reported_revision: pack.oracle.reported_revision,
  analyzer_executable: pack.oracle.analyzer_executable,
  analyzer_binary_sha256: pack.oracle.analyzer_binary_sha256,
  companion_executable: pack.oracle.companion_executable,
  companion_binary_sha256: pack.oracle.companion_binary_sha256,
  platform: pack.oracle.platform,
  package_file_count: pack.oracle.package_file_count,
  package_manifest_bytes: pack.oracle.package_manifest_bytes,
  package_manifest_sha256: pack.oracle.package_manifest_sha256,
  archive_size_bytes: pack.oracle.archive_size_bytes,
  archive_sha256: pack.oracle.archive_sha256,
}
const lock = {
  repo: row.source.repo,
  file: row.source.file,
  revision: row.source.revision,
  size_bytes: row.identity.size_bytes,
  sha256: row.identity.sha256,
  license: row.source.license,
  download_url: `https://huggingface.co/${row.source.repo}/resolve/${row.source.revision}/${row.source.file}`,
  access: { gated: false, private: false, disabled: false },
}
const range = {
  bytes: fakePrefix,
  requested_bytes: PREFIX_BYTES,
  prefix_sha256: PREFIX_SHA256,
  content_range: { start: 0, end: PREFIX_BYTES - 1, total: row.identity.size_bytes },
}
const baseDeps = {
  roster,
  readSourceState: async () => clone(sourceState),
  inspectCamelid: async () => clone(inspectorIdentity),
  inspectOracle: async () => clone(oracleIdentity),
  resolveSource: async () => clone(lock),
  fetchPrefix: async () => range,
  writeFileImpl: async () => {},
  readFileImpl: async () => fakePrefix,
  inspectPrefix: async () => ({ metadata: { 'tokenizer.chat_template': pack.source_template.text } }),
  assertMetadata: () => ({
    chat_template_utf8_bytes: TEMPLATE_UTF8_BYTES,
    chat_template_sha256: TEMPLATE_SHA256,
  }),
  runAnalyzer: async () => clone(syntheticAnalysis),
  sha256Impl: () => PREFIX_SHA256,
}
const qualificationOptions = {
  root,
  binary: 'redacted-camelid',
  analyzer: 'redacted-analyzer',
  prefixBytes: PREFIX_BYTES,
}
assert.deepEqual(
  validateShapePack(await qualifySmolLM3Template(qualificationOptions, baseDeps)),
  [],
)
let reusedLockResolverCalls = 0
assert.deepEqual(
  validateShapePack(await qualifySmolLM3Template(
    { ...qualificationOptions, initialLock: clone(lock) },
    {
      ...baseDeps,
      resolveSource: async () => {
        reusedLockResolverCalls += 1
        return clone(lock)
      },
    },
  )),
  [],
)
assert.equal(
  reusedLockResolverCalls,
  1,
  'an injected preflight lock must be reused initially while preserving the post-probe re-resolution',
)
await assert.rejects(
  qualifySmolLM3Template(
    { ...qualificationOptions, initialLock: { ...clone(lock), sha256: 'f'.repeat(64) } },
    baseDeps,
  ),
  (error) => error.code === 'smollm3_template_source_identity_mismatch',
)
let trackedDirtyFetchCalls = 0
await assert.rejects(
  qualifySmolLM3Template(qualificationOptions, {
    ...baseDeps,
    readSourceState: async () => ({ ...clone(sourceState), tracked_dirty: true }),
    inspectCamelid: undefined,
    execImpl: async () => ({ stdout: inspectorIdentity.version, stderr: '' }),
    readFileImpl: async () => Buffer.from('test binary'),
    fetchPrefix: async () => {
      trackedDirtyFetchCalls += 1
      return range
    },
  }),
  (error) => error.code === 'smollm3_template_inspector_unavailable',
)
assert.equal(
  trackedDirtyFetchCalls,
  0,
  'the authoritative tracked-dirty check must fail before the bounded prefix fetch',
)

for (const primaryFails of [false, true]) {
  const tempRoot = await mkdtemp(join(tmpdir(), 'camelid-smollm3-prefix-cleanup-test-'))
  try {
    await assert.rejects(
      qualifySmolLM3Template(qualificationOptions, {
        ...baseDeps,
        mkdtempImpl: async () => tempRoot,
        inspectPrefix: primaryFails
          ? async () => { throw new Error('primary prefix-probe failure') }
          : baseDeps.inspectPrefix,
        rmImpl: async () => { throw new Error('prefix cleanup failure') },
      }),
      (error) => error.code === 'smollm3_template_cleanup_failed',
    )
  } finally {
    await rm(tempRoot, { recursive: true, force: true })
  }
}

let sourceResolveCalls = 0
await assert.rejects(
  qualifySmolLM3Template(
    { ...qualificationOptions, prefixBytes: PREFIX_BYTES - 1 },
    { ...baseDeps, resolveSource: async () => { sourceResolveCalls += 1; return clone(lock) } },
  ),
  (error) => error.code === 'smollm3_template_prefix_budget_invalid',
)
assert.equal(sourceResolveCalls, 0, 'invalid byte budget must fail before source/network lookup')

await assert.rejects(
  qualifySmolLM3Template(qualificationOptions, {
    ...baseDeps,
    resolveSource: async () => { throw new Error('offline with C:\\private\\token') },
  }),
  (error) => error.code === 'smollm3_template_source_unavailable'
    && classifySmolLM3TemplateQualificationError(error).status === 'blocked',
)
await assert.rejects(
  qualifySmolLM3Template(qualificationOptions, {
    ...baseDeps,
    resolveSource: async () => ({ ...clone(lock), sha256: 'f'.repeat(64) }),
  }),
  (error) => error.code === 'smollm3_template_source_identity_mismatch'
    && classifySmolLM3TemplateQualificationError(error).status === 'fail',
)
let offlinePostflightResolves = 0
await assert.rejects(
  qualifySmolLM3Template(qualificationOptions, {
    ...baseDeps,
    resolveSource: async () => {
      offlinePostflightResolves += 1
      if (offlinePostflightResolves === 1) return clone(lock)
      throw new Error('offline after probe')
    },
  }),
  (error) => error.code === 'smollm3_template_source_unavailable',
)

let sourceReads = 0
await assert.rejects(
  qualifySmolLM3Template(qualificationOptions, {
    ...baseDeps,
    readSourceState: async () => {
      sourceReads += 1
      return sourceReads === 1 ? clone(sourceState) : { ...sourceState, head: 'f'.repeat(40) }
    },
  }),
  (error) => error.code === 'smollm3_template_source_changed',
)

let inspectorReads = 0
await assert.rejects(
  qualifySmolLM3Template(qualificationOptions, {
    ...baseDeps,
    inspectCamelid: async () => {
      inspectorReads += 1
      return inspectorReads === 1
        ? clone(inspectorIdentity)
        : { ...clone(inspectorIdentity), binary_sha256: 'f'.repeat(64) }
    },
  }),
  (error) => error.code === 'smollm3_template_inspector_changed',
)

for (const [headerCode, qualificationCode] of [
  ['header_inspector_timeout', 'smollm3_template_inspector_unavailable'],
  ['header_inspector_unavailable', 'smollm3_template_inspector_unavailable'],
  ['header_parse_failed', 'smollm3_template_metadata_mismatch'],
  ['header_inspector_output_invalid', 'smollm3_template_metadata_mismatch'],
]) {
  await assert.rejects(
    qualifySmolLM3Template(qualificationOptions, {
      ...baseDeps,
      inspectPrefix: async () => { throw new HeaderInspectionError(headerCode) },
    }),
    (error) => error.code === qualificationCode,
  )
}
await assert.rejects(
  qualifySmolLM3Template(qualificationOptions, {
    ...baseDeps,
    inspectPrefix: async () => { throw new Error('private unknown failure') },
  }),
  (error) => error.code === 'smollm3_template_qualification_error'
    && classifySmolLM3TemplateQualificationError(error).status === 'blocked',
)

let oracleReads = 0
await assert.rejects(
  qualifySmolLM3Template(qualificationOptions, {
    ...baseDeps,
    inspectOracle: async () => {
      oracleReads += 1
      return oracleReads === 1
        ? clone(oracleIdentity)
        : { ...clone(oracleIdentity), archive_sha256: 'f'.repeat(64) }
    },
  }),
  (error) => error.code === 'smollm3_template_oracle_changed',
)

let lockReads = 0
await assert.rejects(
  qualifySmolLM3Template(qualificationOptions, {
    ...baseDeps,
    resolveSource: async () => {
      lockReads += 1
      return lockReads === 1 ? clone(lock) : { ...clone(lock), download_url: `${lock.download_url}?drift=1` }
    },
  }),
  (error) => error.code === 'smollm3_template_source_changed',
)

const mutatingLock = clone(lock)
let mutatingLockReads = 0
await assert.rejects(
  qualifySmolLM3Template(qualificationOptions, {
    ...baseDeps,
    resolveSource: async () => {
      mutatingLockReads += 1
      if (mutatingLockReads === 2) {
        mutatingLock.download_url = `${mutatingLock.download_url}?mutated=1`
        mutatingLock.access.private = true
      }
      return mutatingLock
    },
  }),
  (error) => error.code === 'smollm3_template_source_changed',
)

let prefixReads = 0
await assert.rejects(
  qualifySmolLM3Template(qualificationOptions, {
    ...baseDeps,
    readFileImpl: async () => {
      prefixReads += 1
      return prefixReads === 1 ? fakePrefix : Buffer.alloc(PREFIX_BYTES, 1)
    },
    sha256Impl: (bytes) => bytes === fakePrefix ? PREFIX_SHA256 : 'f'.repeat(64),
  }),
  (error) => error.code === 'smollm3_template_prefix_identity_mismatch',
)

console.log('SmolLM3 bounded template qualification tests passed')
