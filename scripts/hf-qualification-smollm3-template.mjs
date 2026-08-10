#!/usr/bin/env node

import { createHash } from 'node:crypto'
import { execFile } from 'node:child_process'
import { mkdtemp, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import { validateRoster } from './check-model-qualification-roster.mjs'
import {
  HeaderInspectionError,
  classifyHeaderInspectionError,
  fetchHeaderPrefix,
  inspectPrefix,
} from './hf-qualification-header.mjs'
import {
  assertSmolLM3TokenizerMetadata,
  classifyCamelidProvenance,
  parseLlamaVersionOutput,
  sourceSelectionForRow,
} from './hf-qualification-tokenizer.mjs'
import {
  resolveHfSource,
  validateLockAgainstSelection,
} from './hf-qualification-source.mjs'

const execFileAsync = promisify(execFile)
const ROW_ID = 'smollm3_3b_q8_0'
const PREFIX_BYTES = 32 * 1024 * 1024
const PREFIX_SHA256 = '2d043b2114b89100c7ba464e57375a6f32c06c04729542d54ed684b5e8c5016e'
const TEMPLATE_UTF8_BYTES = 5_493
const TEMPLATE_SHA256 = 'b9b66f04c64fbb8695cf5b35c37780efd0b8e0829fbfe3e30fafb9f469b7d30e'
const LLAMA_BUILD = 9_632
const LLAMA_REVISION = 'acd79d603cb2e1c84c0886137b80f1ad649b6857'
const LLAMA_ANALYZER_SHA256 = '3ee4a64a2cc3f71cb07f0fe7357b779cab03468fc603f76637bb9d5364a9216c'
const LLAMA_CLI_SHA256 = '2ec09da0b81d0201ce5b21810caefb4e77fd108f383b30c15ca493c5a70f7731'
const LLAMA_PACKAGE_FILE_COUNT = 51
const LLAMA_PACKAGE_MANIFEST_BYTES = 4_682
const LLAMA_PACKAGE_MANIFEST_SHA256 = 'd70bbe8beb7848396d0993ee533062c200350fd9961e2b92c799b24f94a33e93'
const LLAMA_ARCHIVE_SIZE_BYTES = 16_899_258
const LLAMA_ARCHIVE_SHA256 = 'b835d5c5155dd2a5ed748a0351debf2ede0dc9f808757e0429f8700a11832dcd'
const DATE_PLACEHOLDER = '{{CURRENT_DATE_DD_MONTH_YYYY}}'
const HEADER_GROUNDING_RECEIPT = 'qa/model-qualification/smollm3-3b-q8-header-inspection.json'
const HEADER_GROUNDING_RECEIPT_SHA256 = '6faa8cee4a70b5821f485e2debc8a9263d02ebbf5c346138221e9f78a46c9dae'
const TOKENIZER_GROUNDING_RECEIPT = 'qa/model-qualification/smollm3-3b-q8-header-tokenizer-parity.json'
const TOKENIZER_GROUNDING_RECEIPT_SHA256 = '4e3f3b74346b4b005a462ab888a50d9bacdd01cc5a20d05e57682d70eff9afe4'
const ORACLE_USER_TEXT = 'Hello, please help me.'
const ORACLE_ASSISTANT_TEXT = 'I can help you with that.'
const ORACLE_FOLLOWUP_TEXT = 'Thank you.'
const EXACT_SOURCE = Object.freeze({
  repo: 'ggml-org/SmolLM3-3B-GGUF',
  file: 'SmolLM3-Q8_0.gguf',
  revision: '4965cb60b150737b68a0408c36aeefb65078f894',
  size_bytes: 3_275_574_624,
  sha256: '8aa8cc74656137174a1988d993b00828e65a86fd68773412b632a75aa1373248',
  license: 'apache-2.0',
})
const COMMITTED_GROUNDING_INSPECTOR = Object.freeze({
  version: 'camelid v0.6.1-28-gfcb3e022',
  binary_sha256: '17c60bcad794934416a1ae29de99ce0316d77b68207ac1e5d1e7d90189f5b9c3',
  source_head: 'fcb3e022e24efecdd0a554ff7fd218a6ee106dfc',
  source_tracked_dirty: false,
  binary_commit_abbrev: 'fcb3e022',
  binary_reports_dirty: false,
  binary_matches_source_head: true,
  clean_current_head: true,
  binary_path_redacted: true,
})

const DEFAULT_THINKING_INSTRUCTION = 'You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.'

const TEMPLATE_ERROR_CONTRACTS = Object.freeze({
  smollm3_template_prefix_budget_invalid: ['fail', 'SmolLM3 template qualification requires exactly the pinned 32 MiB prefix'],
  smollm3_template_source_unavailable: ['blocked', 'the immutable SmolLM3 source lock could not be resolved'],
  smollm3_template_source_identity_mismatch: ['fail', 'the immutable source lock does not match the exact SmolLM3 roster row'],
  smollm3_template_inspector_unavailable: ['blocked', 'a clean current-head Camelid prefix inspector is unavailable'],
  smollm3_template_inspector_changed: ['blocked', 'the Camelid prefix inspector changed during qualification'],
  smollm3_template_oracle_unavailable: ['blocked', 'the pinned llama.cpp template analyzer package is unavailable'],
  smollm3_template_oracle_identity_mismatch: ['fail', 'the llama.cpp template analyzer package does not match the pin'],
  smollm3_template_oracle_changed: ['blocked', 'the llama.cpp template analyzer package changed during qualification'],
  smollm3_template_range_unavailable: ['blocked', 'the bounded immutable prefix range is unavailable'],
  smollm3_template_range_invalid: ['fail', 'the bounded immutable prefix range response is invalid'],
  smollm3_template_prefix_identity_mismatch: ['fail', 'the bounded prefix does not match its exact-row SHA-256'],
  smollm3_template_metadata_mismatch: ['fail', 'the bounded prefix metadata does not match the exact SmolLM3 row'],
  smollm3_template_identity_mismatch: ['fail', 'tokenizer.chat_template does not match the exact 5,493-byte SmolLM3 template'],
  smollm3_template_oracle_output_invalid: ['fail', 'the pinned template analyzer output is incomplete or inconsistent'],
  smollm3_template_source_changed: ['blocked', 'source HEAD or tracked state changed during template qualification'],
  smollm3_template_cleanup_failed: ['blocked', 'temporary template qualification files could not be removed'],
  smollm3_template_qualification_error: ['blocked', 'bounded SmolLM3 template qualification could not complete'],
})
const FALLBACK_ERROR = TEMPLATE_ERROR_CONTRACTS.smollm3_template_qualification_error
const ERROR_CODES = new WeakMap()

class SmolLM3TemplateQualificationError extends Error {
  constructor(code) {
    const known = typeof code === 'string' && Object.hasOwn(TEMPLATE_ERROR_CONTRACTS, code)
    const canonical = known ? code : 'smollm3_template_qualification_error'
    super(TEMPLATE_ERROR_CONTRACTS[canonical][1])
    this.name = 'SmolLM3TemplateQualificationError'
    this.code = canonical
    this.status = TEMPLATE_ERROR_CONTRACTS[canonical][0]
    ERROR_CODES.set(this, canonical)
  }
}

function templateError(code) {
  return new SmolLM3TemplateQualificationError(code)
}

function classifySmolLM3TemplateQualificationError(error) {
  if (error instanceof SmolLM3TemplateQualificationError) {
    const canonical = ERROR_CODES.get(error)
    const known = typeof canonical === 'string' && Object.hasOwn(TEMPLATE_ERROR_CONTRACTS, canonical)
    const code = known ? canonical : 'smollm3_template_qualification_error'
    const contract = known ? TEMPLATE_ERROR_CONTRACTS[code] : FALLBACK_ERROR
    return { status: contract[0], error_code: code, reason: contract[1] }
  }
  if (error instanceof HeaderInspectionError) {
    const header = classifyHeaderInspectionError(error)
    const code = header.status === 'fail'
      ? 'smollm3_template_range_invalid'
      : 'smollm3_template_range_unavailable'
    const contract = TEMPLATE_ERROR_CONTRACTS[code]
    return { status: contract[0], error_code: code, reason: contract[1] }
  }
  return {
    status: FALLBACK_ERROR[0],
    error_code: 'smollm3_template_qualification_error',
    reason: FALLBACK_ERROR[1],
  }
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

function normalizePrefixBytes(value = PREFIX_BYTES) {
  const parsed = typeof value === 'string' && /^\d+$/.test(value) ? Number(value) : value
  if (!Number.isSafeInteger(parsed) || parsed !== PREFIX_BYTES) {
    throw templateError('smollm3_template_prefix_budget_invalid')
  }
  return parsed
}

function assertExactTemplate(template) {
  if (typeof template !== 'string'
    || Buffer.byteLength(template) !== TEMPLATE_UTF8_BYTES
    || sha256(Buffer.from(template, 'utf8')) !== TEMPLATE_SHA256) {
    throw templateError('smollm3_template_identity_mismatch')
  }
  return template
}

function expectedPromptForMessages(today, messages, addGenerationPrompt = true) {
  let prompt = '<|im_start|>system\n'
    + '## Metadata\n\n'
    + 'Knowledge Cutoff Date: June 2025\n'
    + `Today Date: ${today}\n`
    + 'Reasoning Mode: /think\n\n'
    + '## Custom Instructions\n\n'
    + `${DEFAULT_THINKING_INSTRUCTION}\n\n`
  // The pinned template intentionally omits the synthetic system terminator
  // on its ordinary no-tools branch. Preserve the resulting adjacency.
  for (const message of messages) {
    if (message.role === 'user') {
      prompt += `<|im_start|>user\n${message.content}<|im_end|>\n`
    } else if (message.role === 'assistant') {
      prompt += `<|im_start|>assistant\n${message.content.replace(/^\n+/, '')}<|im_end|>\n`
    } else {
      throw templateError('smollm3_template_oracle_output_invalid')
    }
  }
  if (addGenerationPrompt) prompt += '<|im_start|>assistant\n'
  return prompt
}

function expectedDefaultPrompt(today, userText = ORACLE_USER_TEXT) {
  return expectedPromptForMessages(today, [{ role: 'user', content: userText }], true)
}

function stripPinnedAnalyzerSgr(value) {
  const text = String(value)
  const withoutKnown = text.replace(/\u001b\[(?:0|1|38;5;(?:[0-9]{1,3}))m/g, '')
  if (/[\u001b\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(withoutKnown)) {
    throw templateError('smollm3_template_oracle_output_invalid')
  }
  return withoutKnown
}

function normalizeOracleDate(prompt) {
  if (typeof prompt !== 'string' || prompt.includes(DATE_PLACEHOLDER)) {
    throw templateError('smollm3_template_oracle_output_invalid')
  }
  const matches = [...prompt.matchAll(
    /Knowledge Cutoff Date: June 2025\nToday Date: (\d{2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4})\nReasoning Mode: \/think\n/g,
  )]
  if (matches.length !== 1) {
    throw templateError('smollm3_template_oracle_output_invalid')
  }
  const captured = matches[0][1]
  const [dayText, month, yearText] = captured.split(' ')
  const day = Number(dayText)
  const year = Number(yearText)
  const monthDays = {
    January: 31,
    February: year % 400 === 0 || (year % 4 === 0 && year % 100 !== 0) ? 29 : 28,
    March: 31,
    April: 30,
    May: 31,
    June: 30,
    July: 31,
    August: 31,
    September: 30,
    October: 31,
    November: 30,
    December: 31,
  }
  if (!Number.isSafeInteger(year) || year < 1
    || !Number.isSafeInteger(day) || day < 1 || day > monthDays[month]) {
    throw templateError('smollm3_template_oracle_output_invalid')
  }
  return {
    captured,
    normalized: prompt.replace(`Today Date: ${captured}`, `Today Date: ${DATE_PLACEHOLDER}`),
  }
}

function captureExactDiff(clean, title) {
  const escaped = title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const expression = new RegExp(
    `^=== ${escaped} ===\\n`
      + "Common Prefix: '([\\s\\S]*?)'\\n"
      + "Common Suffix: '([\\s\\S]*?)'\\n"
      + "Left \\(difference\\): '([\\s\\S]*?)'\\n"
      + "Right \\(difference\\): '([\\s\\S]*?)'(?:\\n|$)",
    'gm',
  )
  const matches = [...clean.matchAll(expression)]
  if (matches.length !== 1) throw templateError('smollm3_template_oracle_output_invalid')
  return {
    common_prefix: matches[0][1],
    common_suffix: matches[0][2],
    left: matches[0][3],
    right: matches[0][4],
  }
}

function normalizeAnalyzedPrompt(prompt) {
  if (prompt.includes(DATE_PLACEHOLDER)) {
    throw templateError('smollm3_template_oracle_output_invalid')
  }
  return normalizeOracleDate(prompt).normalized
}

function parseAnalyzerDefaultPrompt(stdout) {
  const clean = stripPinnedAnalyzerSgr(stdout).replace(/\r\n/g, '\n')
  if (/Analysis failed|Error checking|error:/i.test(clean)
    || (clean.match(/TEMPLATE ANALYSIS TOOL/g) || []).length !== 1
    || (clean.match(/ANALYZING TEMPLATE:/g) || []).length !== 1
    || (clean.match(/ANALYSIS COMPLETE/g) || []).length !== 1) {
    throw templateError('smollm3_template_oracle_output_invalid')
  }
  const requiredTitles = [
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
  let priorTitleIndex = -1
  const parsedDiffs = new Map()
  for (const title of requiredTitles) {
    const marker = `=== ${title} ===`
    const indices = [...clean.matchAll(new RegExp(marker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'))]
      .map((match) => match.index)
    if (indices.length !== 1 || indices[0] <= priorTitleIndex) {
      throw templateError('smollm3_template_oracle_output_invalid')
    }
    priorTitleIndex = indices[0]
    parsedDiffs.set(title, captureExactDiff(clean, title))
  }
  if ((clean.match(/^=== Diff:/gm) || []).length !== requiredTitles.length) {
    throw templateError('smollm3_template_oracle_output_invalid')
  }
  const exactTail = '=== Checking Reasoning Variables ===\n'
    + 'No reasoning/thinking-related variables were queried by the template\n\n'
  const tailIndex = clean.indexOf(exactTail)
  if (tailIndex <= priorTitleIndex || clean.indexOf(exactTail, tailIndex + 1) >= 0) {
    throw templateError('smollm3_template_oracle_output_invalid')
  }
  const completionIndex = clean.indexOf('ANALYSIS COMPLETE')
  if (completionIndex <= tailIndex) throw templateError('smollm3_template_oracle_output_invalid')
  const capabilityMatch = /=== Template Capabilities \(from jinja::caps\) ===\n([\s\S]*?)\n\n=== Diff:/.exec(clean)
  if (!capabilityMatch) throw templateError('smollm3_template_oracle_output_invalid')
  const capabilityLines = capabilityMatch[1].split('\n')
  const expectedCapabilities = [
    'supports_tools: false',
    'supports_tool_calls: false',
    'supports_system_role: true',
    'supports_parallel_tool_calls: false',
    'supports_typed_content: false',
    'supports_string_content: true',
  ]
  if (!sameJson(capabilityLines, expectedCapabilities)) {
    throw templateError('smollm3_template_oracle_output_invalid')
  }

  const tools = parsedDiffs.get('Diff: With vs Without Tools (single user message)')
  const generation = parsedDiffs.get('Diff: With vs Without add_generation_prompt (single user message)')
  const userAssistant = parsedDiffs.get('Diff: With vs Without reasoning_content (user, assistant)')
  const userAssistantUser = parsedDiffs.get('Diff: With vs Without reasoning_content (user, assistant, user)')
  const toolsLeft = `${tools.common_prefix}${tools.left}${tools.common_suffix}`
  const toolsRight = `${tools.common_prefix}${tools.right}${tools.common_suffix}`
  const singleWithoutGeneration = `${generation.common_prefix}${generation.left}${generation.common_suffix}`
  const singleWithGeneration = `${generation.common_prefix}${generation.right}${generation.common_suffix}`
  const assistantHistoryLeft = `${userAssistant.common_prefix}${userAssistant.left}${userAssistant.common_suffix}`
  const assistantHistoryRight = `${userAssistant.common_prefix}${userAssistant.right}${userAssistant.common_suffix}`
  const endingUserLeft = `${userAssistantUser.common_prefix}${userAssistantUser.left}${userAssistantUser.common_suffix}`
  const endingUserRight = `${userAssistantUser.common_prefix}${userAssistantUser.right}${userAssistantUser.common_suffix}`
  const normalized = {
    tools_left: normalizeAnalyzedPrompt(toolsLeft),
    tools_right: normalizeAnalyzedPrompt(toolsRight),
    single_without_generation: normalizeAnalyzedPrompt(singleWithoutGeneration),
    single_with_generation: normalizeAnalyzedPrompt(singleWithGeneration),
    user_assistant_left: normalizeAnalyzedPrompt(assistantHistoryLeft),
    user_assistant_right: normalizeAnalyzedPrompt(assistantHistoryRight),
    user_assistant_user_left: normalizeAnalyzedPrompt(endingUserLeft),
    user_assistant_user_right: normalizeAnalyzedPrompt(endingUserRight),
  }
  const reconstructedPrompts = [...parsedDiffs.values()].flatMap((diff) => [
    `${diff.common_prefix}${diff.left}${diff.common_suffix}`,
    `${diff.common_prefix}${diff.right}${diff.common_suffix}`,
  ])
  const rawDates = reconstructedPrompts.map((prompt) => normalizeOracleDate(prompt).captured)
  if (new Set(rawDates).size !== 1) {
    throw templateError('smollm3_template_oracle_output_invalid')
  }
  const expectedSingleWithout = expectedPromptForMessages(
    DATE_PLACEHOLDER,
    [{ role: 'user', content: ORACLE_USER_TEXT }],
    false,
  )
  const expectedSingleWith = expectedDefaultPrompt(DATE_PLACEHOLDER)
  const expectedAssistantHistory = expectedPromptForMessages(DATE_PLACEHOLDER, [
    { role: 'user', content: ORACLE_USER_TEXT },
    { role: 'assistant', content: ORACLE_ASSISTANT_TEXT },
  ], false)
  const expectedEndingUser = expectedPromptForMessages(DATE_PLACEHOLDER, [
    { role: 'user', content: ORACLE_USER_TEXT },
    { role: 'assistant', content: ORACLE_ASSISTANT_TEXT },
    { role: 'user', content: ORACLE_FOLLOWUP_TEXT },
  ], false)
  if (normalized.tools_left !== expectedSingleWithout
    || normalized.tools_right !== expectedSingleWithout
    || normalized.single_without_generation !== expectedSingleWithout
    || normalized.single_with_generation !== expectedSingleWith
    || normalized.user_assistant_left !== expectedAssistantHistory
    || normalized.user_assistant_right !== expectedAssistantHistory
    || normalized.user_assistant_user_left !== expectedEndingUser
    || normalized.user_assistant_user_right !== expectedEndingUser
    || normalized.single_with_generation.includes(`${DEFAULT_THINKING_INSTRUCTION}<|im_end|>`)) {
    throw templateError('smollm3_template_oracle_output_invalid')
  }
  return {
    captured_date_redacted: true,
    date_format: '%d %B %Y',
    date_placeholder: DATE_PLACEHOLDER,
    capabilities: {
      supports_tools: false,
      supports_tool_calls: false,
      supports_system_role: true,
      supports_parallel_tool_calls: false,
      supports_typed_content: false,
      supports_string_content: true,
    },
    standard_tools_inert_for_analyzer_case: normalized.tools_left === normalized.tools_right,
    normalized_single_user_prompt: normalized.single_with_generation,
    normalized_user_assistant_core: normalized.user_assistant_left,
    normalized_user_assistant_user_core: normalized.user_assistant_user_left,
  }
}

async function runTemplateAnalyzer(analyzer, template, {
  execImpl = execFileAsync,
  mkdtempImpl = mkdtemp,
  readFileImpl = readFile,
  rmImpl = rm,
  writeFileImpl = writeFile,
} = {}) {
  const tempRoot = await mkdtempImpl(join(tmpdir(), 'camelid-smollm3-template-analysis-'))
  const templatePath = join(tempRoot, 'template.jinja')
  try {
    await writeFileImpl(templatePath, template, { encoding: 'utf8', flag: 'wx' })
    let result
    try {
      result = await execImpl(analyzer, ['--template-file', templatePath], {
        timeout: 30_000,
        maxBuffer: 16 * 1024 * 1024,
        windowsHide: true,
      })
    } catch {
      throw templateError('smollm3_template_oracle_unavailable')
    }
    if (String(result.stdout || '').trim()) {
      throw templateError('smollm3_template_oracle_output_invalid')
    }
    const templateAfter = await readFileImpl(templatePath)
      .catch(() => { throw templateError('smollm3_template_identity_mismatch') })
    if (!templateAfter.equals(Buffer.from(template, 'utf8'))
      || templateAfter.length !== TEMPLATE_UTF8_BYTES
      || sha256(templateAfter) !== TEMPLATE_SHA256) {
      throw templateError('smollm3_template_identity_mismatch')
    }
    return parseAnalyzerDefaultPrompt(String(result.stderr || ''))
  } finally {
    try { await rmImpl(tempRoot, { recursive: true, force: true }) }
    catch { throw templateError('smollm3_template_cleanup_failed') }
  }
}

async function readSourceState(root, { execImpl = execFileAsync } = {}) {
  const [{ stdout: head }, { stdout: status }] = await Promise.all([
    execImpl('git', ['rev-parse', 'HEAD'], { cwd: root, timeout: 10_000, windowsHide: true }),
    execImpl('git', ['status', '--porcelain', '--untracked-files=no'], {
      cwd: root,
      timeout: 10_000,
      windowsHide: true,
    }),
  ])
  return { head: String(head).trim().toLowerCase(), tracked_dirty: Boolean(String(status).trim()) }
}

async function inspectFileIdentity(path, readFileImpl = readFile) {
  return { executable: basename(path), binary_sha256: sha256(await readFileImpl(path)) }
}

async function inspectLlamaPackage(analyzer, {
  readFileImpl = readFile,
  readdirImpl = readdir,
} = {}) {
  if (process.platform !== 'win32' || process.arch !== 'x64') {
    throw templateError('smollm3_template_oracle_identity_mismatch')
  }
  const binDir = dirname(analyzer)
  let dirents
  try { dirents = await readdirImpl(binDir, { withFileTypes: true }) }
  catch { throw templateError('smollm3_template_oracle_unavailable') }
  if (!Array.isArray(dirents) || dirents.some((entry) => !entry?.isFile?.())) {
    throw templateError('smollm3_template_oracle_identity_mismatch')
  }
  const names = dirents.map((entry) => entry.name).sort()
  if (names.length !== LLAMA_PACKAGE_FILE_COUNT
    || names.some((name) => !/^[a-z0-9._+-]+$/.test(name))) {
    throw templateError('smollm3_template_oracle_identity_mismatch')
  }
  const lines = []
  for (const name of names) {
    const bytes = await readFileImpl(join(binDir, name))
      .catch(() => { throw templateError('smollm3_template_oracle_unavailable') })
    lines.push(`${name}\t${bytes.length}\t${sha256(bytes)}\n`)
  }
  const archivePath = join(dirname(binDir), 'llama-b9632-bin-win-cpu-x64.zip')
  const archive = await readFileImpl(archivePath)
    .catch(() => { throw templateError('smollm3_template_oracle_unavailable') })
  const manifest = Buffer.from(lines.join(''), 'utf8')
  const identity = {
    platform: 'win32-x64',
    package_file_count: names.length,
    package_manifest_bytes: manifest.length,
    package_manifest_sha256: sha256(manifest),
    archive_size_bytes: archive.length,
    archive_sha256: sha256(archive),
  }
  if (identity.package_manifest_bytes !== LLAMA_PACKAGE_MANIFEST_BYTES
    || identity.package_manifest_sha256 !== LLAMA_PACKAGE_MANIFEST_SHA256
    || identity.archive_size_bytes !== LLAMA_ARCHIVE_SIZE_BYTES
    || identity.archive_sha256 !== LLAMA_ARCHIVE_SHA256) {
    throw templateError('smollm3_template_oracle_identity_mismatch')
  }
  return identity
}

async function inspectCamelid(binary, sourceState, {
  execImpl = execFileAsync,
  readFileImpl = readFile,
} = {}) {
  let version
  let identity
  try {
    const result = await execImpl(binary, ['--version'], {
      timeout: 10_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
    })
    version = String(result.stdout || '').trim()
    identity = await inspectFileIdentity(binary, readFileImpl)
  } catch {
    throw templateError('smollm3_template_inspector_unavailable')
  }
  const provenance = classifyCamelidProvenance({
    version,
    sourceHead: sourceState.head,
    sourceTrackedDirty: sourceState.tracked_dirty,
  })
  if (!provenance.clean_current_head) {
    throw templateError('smollm3_template_inspector_unavailable')
  }
  return { version, ...identity, provenance }
}

async function inspectOracle(analyzer, {
  execImpl = execFileAsync,
  readFileImpl = readFile,
} = {}) {
  const companion = join(dirname(analyzer), process.platform === 'win32' ? 'llama-cli.exe' : 'llama-cli')
  let output = ''
  try {
    const result = await execImpl(companion, ['--version'], { timeout: 10_000, windowsHide: true })
    output = `${result.stdout || ''}\n${result.stderr || ''}`
  } catch (error) {
    if (error?.killed || error?.signal || error?.code !== 1) {
      throw templateError('smollm3_template_oracle_unavailable')
    }
    output = `${error?.stdout || ''}\n${error?.stderr || ''}`
    if (!output.trim()) throw templateError('smollm3_template_oracle_unavailable')
  }
  let parsed
  try { parsed = parseLlamaVersionOutput(output) }
  catch { throw templateError('smollm3_template_oracle_identity_mismatch') }
  let analyzerIdentity
  let companionIdentity
  let packageIdentity
  try {
    [analyzerIdentity, companionIdentity, packageIdentity] = await Promise.all([
      inspectFileIdentity(analyzer, readFileImpl),
      inspectFileIdentity(companion, readFileImpl),
      inspectLlamaPackage(analyzer, { readFileImpl }),
    ])
  } catch (error) {
    if (error instanceof SmolLM3TemplateQualificationError) throw error
    throw templateError('smollm3_template_oracle_unavailable')
  }
  if (parsed.build !== LLAMA_BUILD
    || analyzerIdentity.executable !== 'llama-template-analysis.exe'
    || companionIdentity.executable !== 'llama-cli.exe'
    || analyzerIdentity.binary_sha256 !== LLAMA_ANALYZER_SHA256
    || companionIdentity.binary_sha256 !== LLAMA_CLI_SHA256) {
    throw templateError('smollm3_template_oracle_identity_mismatch')
  }
  return {
    project: 'llama.cpp',
    build: parsed.build,
    revision: LLAMA_REVISION,
    reported_revision: parsed.revision,
    analyzer_executable: analyzerIdentity.executable,
    analyzer_binary_sha256: analyzerIdentity.binary_sha256,
    companion_executable: companionIdentity.executable,
    companion_binary_sha256: companionIdentity.binary_sha256,
    ...packageIdentity,
  }
}

function sameJson(left, right) {
  return JSON.stringify(left) === JSON.stringify(right)
}

function containsPrivateMaterial(value) {
  if (typeof value === 'string') {
    return /(?:[A-Z]:\\(?:Users|Documents and Settings|Windows\\Temp|Temp|private)\\|\/(?:Users|home)\/|Bearer\s|hf_[A-Za-z0-9]|https?:\/\/)/.test(value)
  }
  if (Array.isArray(value)) return value.some(containsPrivateMaterial)
  if (value && typeof value === 'object') {
    return Object.values(value).some(containsPrivateMaterial)
  }
  return false
}

function exactObjectKeys(value, keys) {
  return value && typeof value === 'object' && !Array.isArray(value)
    && sameJson(Object.keys(value).sort(), [...keys].sort())
}

function buildShapePack({ row, range, template, oracle, inspector, analysis }) {
  const singleMessages = [{ role: 'user', content: ORACLE_USER_TEXT }]
  const historyMessages = [
    { role: 'user', content: ORACLE_USER_TEXT },
    { role: 'assistant', content: ORACLE_ASSISTANT_TEXT },
    { role: 'user', content: ORACLE_FOLLOWUP_TEXT },
  ]
  const expectedPrompt = expectedPromptForMessages(DATE_PLACEHOLDER, singleMessages, true)
  const expectedHistoryPrompt = expectedPromptForMessages(DATE_PLACEHOLDER, historyMessages, true)
  return {
    schema: 'camelid.smollm3-chat-template-shapes/v1',
    pack_id: 'smollm3-chat-template-shapes-v1',
    row_id: ROW_ID,
    support_scope: 'preparation_only_runtime_chat_remains_architecture_wide_typed_hold',
    grounding: {
      header_receipt: HEADER_GROUNDING_RECEIPT,
      header_receipt_sha256: HEADER_GROUNDING_RECEIPT_SHA256,
      tokenizer_receipt: TOKENIZER_GROUNDING_RECEIPT,
      tokenizer_receipt_sha256: TOKENIZER_GROUNDING_RECEIPT_SHA256,
    },
    source: {
      repo: row.source.repo,
      file: row.source.file,
      revision: row.source.revision,
      size_bytes: row.identity.size_bytes,
      sha256: row.identity.sha256,
      license: row.source.license,
    },
    bounded_prefix: {
      requested_bytes: PREFIX_BYTES,
      received_bytes: range.bytes.length,
      prefix_sha256: range.prefix_sha256,
      content_range: {
        start: range.content_range.start,
        end: range.content_range.end,
        total: range.content_range.total,
      },
      full_weights_downloaded: false,
    },
    source_template: {
      utf8_bytes: TEMPLATE_UTF8_BYTES,
      sha256: TEMPLATE_SHA256,
      text: template,
    },
    inspector: {
      version: inspector.version,
      binary_sha256: inspector.binary_sha256,
      source_head: inspector.provenance.source_head,
      source_tracked_dirty: inspector.provenance.source_tracked_dirty,
      binary_commit_abbrev: inspector.provenance.binary_commit_abbrev,
      binary_reports_dirty: inspector.provenance.binary_reports_dirty,
      binary_matches_source_head: inspector.provenance.binary_matches_source_head,
      clean_current_head: true,
      binary_path_redacted: true,
    },
    oracle: {
      project: 'llama.cpp',
      build: oracle.build,
      revision: oracle.revision,
      reported_revision: oracle.reported_revision,
      analyzer_executable: oracle.analyzer_executable,
      analyzer_binary_sha256: oracle.analyzer_binary_sha256,
      companion_executable: oracle.companion_executable,
      companion_binary_sha256: oracle.companion_binary_sha256,
      platform: oracle.platform,
      package_file_count: oracle.package_file_count,
      package_manifest_bytes: oracle.package_manifest_bytes,
      package_manifest_sha256: oracle.package_manifest_sha256,
      archive_size_bytes: oracle.archive_size_bytes,
      archive_sha256: oracle.archive_sha256,
      executable_paths_redacted: true,
      mode: 'template_core_analysis_not_http_apply_template',
      default_case_only: true,
      analyzer_date_value_redacted: analysis.captured_date_redacted,
      capabilities: analysis.capabilities,
      standard_tools_inert_for_analyzer_case: analysis.standard_tools_inert_for_analyzer_case,
    },
    dynamic_date: {
      injected_by_preparation_renderer: true,
      format: analysis.date_format,
      placeholder: analysis.date_placeholder,
      normalized_occurrences: 1,
    },
    cases: [
      {
        id: 'default_think_single_user_generation_prompt',
        messages: singleMessages,
        enable_thinking: true,
        add_generation_prompt: true,
        normalized_prompt: expectedPrompt,
        normalized_prompt_utf8_bytes: Buffer.byteLength(expectedPrompt),
        normalized_prompt_sha256: sha256(Buffer.from(expectedPrompt, 'utf8')),
        oracle_grounding: 'exact add_generation_prompt diff right side',
        oracle_exact_match_after_date_normalization: analysis.normalized_single_user_prompt === expectedPrompt,
      },
      {
        id: 'default_think_user_assistant_user_generation_prompt',
        messages: historyMessages,
        enable_thinking: true,
        add_generation_prompt: true,
        normalized_prompt: expectedHistoryPrompt,
        normalized_prompt_utf8_bytes: Buffer.byteLength(expectedHistoryPrompt),
        normalized_prompt_sha256: sha256(Buffer.from(expectedHistoryPrompt, 'utf8')),
        oracle_grounding: 'exact user-assistant-user core plus isolated exact generation suffix',
        oracle_core_exact_match_after_date_normalization:
          analysis.normalized_user_assistant_user_core
            === expectedPromptForMessages(DATE_PLACEHOLDER, historyMessages, false),
      },
    ],
    contract: {
      allowed_roles: ['user', 'assistant'],
      message_input_stage: 'post_canonicalized_chat_messages',
      text_only_content_parts: 'canonicalized_to_string_before_helper',
      exact_role_bytes_required: true,
      messages_nonempty: true,
      content_nonempty: true,
      history_starts_with: 'user',
      history_must_end_with: 'user',
      history_strictly_alternating: true,
      thinking: ['omitted_defaults_true', 'explicit_true'],
      add_generation_prompt: true,
      injected_date: 'calendar_valid_DD_EnglishMonth_YYYY',
      parse_special: true,
      add_special: false,
      synthetic_system_terminator_present: false,
      exact_adjacency: `${DEFAULT_THINKING_INSTRUCTION.slice(-64)}\n\n<|im_start|>user`,
    },
    typed_hold_branches: [
      'empty_messages',
      'empty_content',
      'non_exact_role_bytes',
      'unsupported_roles',
      'non_alternating_history',
      'invalid_injected_date',
      'explicit_enable_thinking_false',
      'system_message',
      'system_override',
      'custom_system_instructions',
      'xml_tools',
      'python_tools',
      'tool_role_history',
      'multimodal_content',
      'unsupported_nontext_content_parts',
      'history_ending_assistant',
      'add_generation_prompt_false',
    ],
    does_not_prove: [
      'runtime_chat_enablement',
      'full_weight_download_or_hash',
      'model_load',
      'generation',
      'http_apply_template_parity',
      'tool_or_system_branches',
      'false_thinking_branch',
      'wire_content_parts_representation_parity',
    ],
    support_decision: 'no_roster_change_template_preparation_only',
  }
}

function validateShapePack(pack, { expectedInspector = COMMITTED_GROUNDING_INSPECTOR } = {}) {
  const errors = []
  const check = (condition, message) => { if (!condition) errors.push(message) }
  check(exactObjectKeys(pack, [
    'schema', 'pack_id', 'row_id', 'support_scope', 'grounding', 'source', 'bounded_prefix',
    'source_template', 'inspector', 'oracle', 'dynamic_date', 'cases', 'contract',
    'typed_hold_branches', 'does_not_prove', 'support_decision',
  ]), 'top-level fields mismatch')
  check(pack?.schema === 'camelid.smollm3-chat-template-shapes/v1', 'schema mismatch')
  check(pack?.pack_id === 'smollm3-chat-template-shapes-v1', 'pack_id mismatch')
  check(pack?.row_id === ROW_ID, 'row_id mismatch')
  check(pack?.support_scope === 'preparation_only_runtime_chat_remains_architecture_wide_typed_hold', 'support scope mismatch')
  check(sameJson(pack?.grounding, {
    header_receipt: HEADER_GROUNDING_RECEIPT,
    header_receipt_sha256: HEADER_GROUNDING_RECEIPT_SHA256,
    tokenizer_receipt: TOKENIZER_GROUNDING_RECEIPT,
    tokenizer_receipt_sha256: TOKENIZER_GROUNDING_RECEIPT_SHA256,
  }), 'grounding receipts mismatch')
  for (const [field, expected] of Object.entries(EXACT_SOURCE)) {
    check(pack?.source?.[field] === expected, `source ${field} mismatch`)
  }
  check(exactObjectKeys(pack?.source, Object.keys(EXACT_SOURCE)), 'source fields mismatch')
  check(exactObjectKeys(pack?.bounded_prefix, [
    'requested_bytes', 'received_bytes', 'prefix_sha256', 'content_range',
    'full_weights_downloaded',
  ]), 'bounded-prefix fields mismatch')
  check(exactObjectKeys(pack?.bounded_prefix?.content_range, ['start', 'end', 'total']), 'Content-Range fields mismatch')
  check(pack?.bounded_prefix?.requested_bytes === PREFIX_BYTES, 'prefix byte budget mismatch')
  check(pack?.bounded_prefix?.received_bytes === PREFIX_BYTES, 'received prefix length mismatch')
  check(pack?.bounded_prefix?.prefix_sha256 === PREFIX_SHA256, 'prefix SHA-256 mismatch')
  check(pack?.bounded_prefix?.full_weights_downloaded === false, 'full-weights scope mismatch')
  check(pack?.bounded_prefix?.content_range?.start === 0, 'Content-Range start mismatch')
  check(pack?.bounded_prefix?.content_range?.end === PREFIX_BYTES - 1, 'Content-Range end mismatch')
  check(pack?.bounded_prefix?.content_range?.total === EXACT_SOURCE.size_bytes, 'Content-Range total mismatch')
  check(pack?.source_template?.utf8_bytes === TEMPLATE_UTF8_BYTES, 'template byte count mismatch')
  check(pack?.source_template?.sha256 === TEMPLATE_SHA256, 'template SHA-256 declaration mismatch')
  check(exactObjectKeys(pack?.source_template, ['utf8_bytes', 'sha256', 'text']), 'source-template fields mismatch')
  if (typeof pack?.source_template?.text === 'string') {
    check(Buffer.byteLength(pack.source_template.text) === TEMPLATE_UTF8_BYTES, 'template text byte count mismatch')
    check(sha256(Buffer.from(pack.source_template.text)) === TEMPLATE_SHA256, 'template text SHA-256 mismatch')
  } else {
    check(false, 'template text missing')
  }
  check(pack?.oracle?.build === LLAMA_BUILD, 'oracle build mismatch')
  check(pack?.oracle?.project === 'llama.cpp', 'oracle project mismatch')
  check(pack?.oracle?.revision === LLAMA_REVISION, 'oracle revision mismatch')
  check(pack?.oracle?.reported_revision === 'acd79d603', 'oracle reported revision mismatch')
  check(pack?.oracle?.analyzer_executable === 'llama-template-analysis.exe', 'oracle analyzer basename mismatch')
  check(pack?.oracle?.companion_executable === 'llama-cli.exe', 'oracle companion basename mismatch')
  check(pack?.oracle?.analyzer_binary_sha256 === LLAMA_ANALYZER_SHA256, 'oracle analyzer SHA-256 mismatch')
  check(pack?.oracle?.companion_binary_sha256 === LLAMA_CLI_SHA256, 'oracle companion SHA-256 mismatch')
  check(pack?.oracle?.platform === 'win32-x64', 'oracle platform mismatch')
  check(pack?.oracle?.package_file_count === LLAMA_PACKAGE_FILE_COUNT, 'oracle package file count mismatch')
  check(pack?.oracle?.package_manifest_bytes === LLAMA_PACKAGE_MANIFEST_BYTES, 'oracle package manifest byte count mismatch')
  check(pack?.oracle?.package_manifest_sha256 === LLAMA_PACKAGE_MANIFEST_SHA256, 'oracle package manifest mismatch')
  check(pack?.oracle?.archive_size_bytes === LLAMA_ARCHIVE_SIZE_BYTES, 'oracle archive size mismatch')
  check(pack?.oracle?.archive_sha256 === LLAMA_ARCHIVE_SHA256, 'oracle archive SHA-256 mismatch')
  check(exactObjectKeys(pack?.oracle, [
    'project', 'build', 'revision', 'reported_revision', 'analyzer_executable',
    'analyzer_binary_sha256', 'companion_executable', 'companion_binary_sha256',
    'platform', 'package_file_count', 'package_manifest_bytes', 'package_manifest_sha256',
    'archive_size_bytes', 'archive_sha256', 'executable_paths_redacted', 'mode',
    'default_case_only', 'analyzer_date_value_redacted', 'capabilities',
    'standard_tools_inert_for_analyzer_case',
  ]), 'oracle fields mismatch')
  check(exactObjectKeys(pack?.oracle?.capabilities, [
    'supports_tools', 'supports_tool_calls', 'supports_system_role',
    'supports_parallel_tool_calls', 'supports_typed_content', 'supports_string_content',
  ]), 'oracle capability fields mismatch')
  check(pack?.oracle?.mode === 'template_core_analysis_not_http_apply_template', 'oracle scope mismatch')
  check(pack?.oracle?.executable_paths_redacted === true, 'oracle path redaction mismatch')
  check(pack?.oracle?.default_case_only === true, 'oracle default-case scope mismatch')
  check(pack?.oracle?.analyzer_date_value_redacted === true, 'oracle date redaction mismatch')
  check(sameJson(pack?.oracle?.capabilities, {
    supports_tools: false,
    supports_tool_calls: false,
    supports_system_role: true,
    supports_parallel_tool_calls: false,
    supports_typed_content: false,
    supports_string_content: true,
  }), 'oracle capabilities mismatch')
  check(pack?.oracle?.standard_tools_inert_for_analyzer_case === true, 'standard-tools analyzer result mismatch')
  const inspectorVersion = /^camelid [A-Za-z0-9._+()-]+-g([0-9a-f]{7,40})$/.exec(pack?.inspector?.version || '')
  check(Boolean(inspectorVersion), 'inspector version mismatch')
  check(/^[0-9a-f]{64}$/.test(pack?.inspector?.binary_sha256 || ''), 'inspector binary SHA-256 mismatch')
  check(/^[0-9a-f]{40}$/.test(pack?.inspector?.source_head || ''), 'inspector source HEAD mismatch')
  check(pack?.inspector?.source_tracked_dirty === false, 'inspector source tracked state mismatch')
  check(pack?.inspector?.binary_reports_dirty === false, 'inspector reports dirty')
  check(typeof pack?.inspector?.binary_commit_abbrev === 'string'
    && pack.inspector.source_head?.startsWith(pack.inspector.binary_commit_abbrev), 'inspector commit provenance mismatch')
  check(inspectorVersion?.[1] === pack?.inspector?.binary_commit_abbrev, 'inspector version/commit mismatch')
  check(pack?.inspector?.binary_matches_source_head === true, 'inspector/source HEAD mismatch')
  check(pack?.inspector?.clean_current_head === true, 'inspector is not clean current-head')
  check(pack?.inspector?.binary_path_redacted === true && !Object.hasOwn(pack?.inspector || {}, 'binary_path'), 'inspector path privacy mismatch')
  check(exactObjectKeys(pack?.inspector, [
    'version', 'binary_sha256', 'source_head', 'source_tracked_dirty',
    'binary_commit_abbrev', 'binary_reports_dirty', 'binary_matches_source_head',
    'clean_current_head', 'binary_path_redacted',
  ]), 'inspector fields mismatch')
  check(sameJson(pack?.inspector, expectedInspector), 'inspector does not match its grounding projection')
  check(pack?.contract?.synthetic_system_terminator_present === false, 'synthetic system terminator contract mismatch')
  check(pack?.contract?.add_generation_prompt === true, 'generation prompt contract mismatch')
  check(pack?.contract?.parse_special === true && pack?.contract?.add_special === false, 'tokenization flags mismatch')
  const testCase = Array.isArray(pack?.cases) ? pack.cases.find((candidate) => candidate?.id === 'default_think_single_user_generation_prompt') : null
  check(Array.isArray(pack?.cases) && pack.cases.length === 2, 'case count mismatch')
  check(pack?.cases?.[0]?.id === 'default_think_single_user_generation_prompt'
    && pack?.cases?.[1]?.id === 'default_think_user_assistant_user_generation_prompt', 'case order mismatch')
  check(exactObjectKeys(pack?.cases?.[0], [
    'id', 'messages', 'enable_thinking', 'add_generation_prompt', 'normalized_prompt',
    'normalized_prompt_utf8_bytes', 'normalized_prompt_sha256', 'oracle_grounding',
    'oracle_exact_match_after_date_normalization',
  ]), 'default case fields mismatch')
  check(Array.isArray(pack?.cases?.[0]?.messages)
    && pack.cases[0].messages.every((message) => exactObjectKeys(message, ['role', 'content'])), 'default message fields mismatch')
  const expected = expectedDefaultPrompt(DATE_PLACEHOLDER)
  check(sameJson(testCase?.messages, [{ role: 'user', content: ORACLE_USER_TEXT }]), 'default case messages mismatch')
  check(testCase?.enable_thinking === true, 'default case thinking mismatch')
  check(testCase?.add_generation_prompt === true, 'default case generation flag mismatch')
  check(testCase?.oracle_grounding === 'exact add_generation_prompt diff right side', 'default case grounding mismatch')
  check(testCase?.normalized_prompt === expected, 'default prompt mismatch')
  check(testCase?.normalized_prompt_utf8_bytes === Buffer.byteLength(expected), 'default prompt byte count mismatch')
  check(testCase?.normalized_prompt_sha256 === sha256(Buffer.from(expected)), 'default prompt SHA-256 mismatch')
  check(testCase?.oracle_exact_match_after_date_normalization === true, 'default prompt oracle comparison did not pass')
  check(!expected.includes(`${DEFAULT_THINKING_INSTRUCTION}<|im_end|>`), 'fixture inserted a synthetic system terminator')
  const historyCase = Array.isArray(pack?.cases) ? pack.cases.find((candidate) => candidate?.id === 'default_think_user_assistant_user_generation_prompt') : null
  const historyMessages = [
    { role: 'user', content: ORACLE_USER_TEXT },
    { role: 'assistant', content: ORACLE_ASSISTANT_TEXT },
    { role: 'user', content: ORACLE_FOLLOWUP_TEXT },
  ]
  const expectedHistory = expectedPromptForMessages(DATE_PLACEHOLDER, historyMessages, true)
  check(exactObjectKeys(pack?.cases?.[1], [
    'id', 'messages', 'enable_thinking', 'add_generation_prompt', 'normalized_prompt',
    'normalized_prompt_utf8_bytes', 'normalized_prompt_sha256', 'oracle_grounding',
    'oracle_core_exact_match_after_date_normalization',
  ]), 'history case fields mismatch')
  check(Array.isArray(pack?.cases?.[1]?.messages)
    && pack.cases[1].messages.every((message) => exactObjectKeys(message, ['role', 'content'])), 'history message fields mismatch')
  check(sameJson(historyCase?.messages, historyMessages), 'history case messages mismatch')
  check(historyCase?.enable_thinking === true, 'history case thinking mismatch')
  check(historyCase?.add_generation_prompt === true, 'history case generation flag mismatch')
  check(historyCase?.oracle_grounding === 'exact user-assistant-user core plus isolated exact generation suffix', 'history case grounding mismatch')
  check(historyCase?.normalized_prompt === expectedHistory, 'history prompt mismatch')
  check(historyCase?.normalized_prompt_utf8_bytes === Buffer.byteLength(expectedHistory), 'history prompt byte count mismatch')
  check(historyCase?.normalized_prompt_sha256 === sha256(Buffer.from(expectedHistory)), 'history prompt SHA-256 mismatch')
  check(historyCase?.oracle_core_exact_match_after_date_normalization === true, 'history core oracle comparison did not pass')
  check(sameJson(pack?.dynamic_date, {
    injected_by_preparation_renderer: true,
    format: '%d %B %Y',
    placeholder: DATE_PLACEHOLDER,
    normalized_occurrences: 1,
  }), 'dynamic date contract mismatch')
  check(exactObjectKeys(pack?.dynamic_date, [
    'injected_by_preparation_renderer', 'format', 'placeholder', 'normalized_occurrences',
  ]), 'dynamic date fields mismatch')
  check(exactObjectKeys(pack?.contract, [
    'allowed_roles', 'message_input_stage', 'text_only_content_parts',
    'exact_role_bytes_required', 'messages_nonempty', 'content_nonempty',
    'history_starts_with', 'history_must_end_with', 'history_strictly_alternating',
    'thinking', 'add_generation_prompt', 'injected_date', 'parse_special', 'add_special',
    'synthetic_system_terminator_present', 'exact_adjacency',
  ]), 'contract fields mismatch')
  check(sameJson(pack?.contract?.allowed_roles, ['user', 'assistant']), 'allowed roles mismatch')
  check(pack?.contract?.message_input_stage === 'post_canonicalized_chat_messages', 'message input-stage contract mismatch')
  check(pack?.contract?.text_only_content_parts === 'canonicalized_to_string_before_helper', 'text-only content-parts contract mismatch')
  check(pack?.contract?.exact_role_bytes_required === true, 'exact role-byte contract mismatch')
  check(pack?.contract?.messages_nonempty === true, 'nonempty messages contract mismatch')
  check(pack?.contract?.content_nonempty === true, 'nonempty content contract mismatch')
  check(pack?.contract?.history_starts_with === 'user', 'history start contract mismatch')
  check(pack?.contract?.history_must_end_with === 'user', 'history ending contract mismatch')
  check(pack?.contract?.history_strictly_alternating === true, 'history alternation contract mismatch')
  check(sameJson(pack?.contract?.thinking, ['omitted_defaults_true', 'explicit_true']), 'thinking contract mismatch')
  check(pack?.contract?.injected_date === 'calendar_valid_DD_EnglishMonth_YYYY', 'injected date contract mismatch')
  check(pack?.contract?.exact_adjacency === `${DEFAULT_THINKING_INSTRUCTION.slice(-64)}\n\n<|im_start|>user`, 'system/user adjacency contract mismatch')
  const exactTypedHolds = [
    'empty_messages', 'empty_content', 'non_exact_role_bytes', 'unsupported_roles',
    'non_alternating_history', 'invalid_injected_date',
    'explicit_enable_thinking_false', 'system_message', 'system_override',
    'custom_system_instructions', 'xml_tools', 'python_tools', 'tool_role_history', 'multimodal_content',
    'unsupported_nontext_content_parts', 'history_ending_assistant',
    'add_generation_prompt_false',
  ]
  for (const branch of exactTypedHolds) {
    check(Array.isArray(pack?.typed_hold_branches) && pack.typed_hold_branches.includes(branch), `typed HOLD branch missing: ${branch}`)
  }
  check(sameJson(pack?.typed_hold_branches, exactTypedHolds), 'typed HOLD list mismatch')
  const exactExclusions = [
    'runtime_chat_enablement',
    'full_weight_download_or_hash',
    'model_load',
    'generation',
    'http_apply_template_parity',
    'tool_or_system_branches',
    'false_thinking_branch',
    'wire_content_parts_representation_parity',
  ]
  for (const exclusion of exactExclusions) {
    check(Array.isArray(pack?.does_not_prove) && pack.does_not_prove.includes(exclusion), `scope exclusion missing: ${exclusion}`)
  }
  check(sameJson(pack?.does_not_prove, exactExclusions), 'scope exclusion list mismatch')
  check(pack?.support_decision === 'no_roster_change_template_preparation_only', 'support decision mismatch')
  check(!containsPrivateMaterial(pack), 'pack contains a path, token, or URL')
  return errors
}

function stableLockIdentity(lock) {
  return {
    repo: lock?.repo,
    file: lock?.file,
    revision: lock?.revision,
    size_bytes: lock?.size_bytes,
    sha256: lock?.sha256,
    license: lock?.license,
    download_url: lock?.download_url,
    access: lock?.access,
  }
}

async function qualifySmolLM3Template({ root = resolve('.'), rosterPath = 'qa/model-qualification/phase1-roster.json', binary, analyzer, prefixBytes = PREFIX_BYTES, token = null }, deps = {}) {
  normalizePrefixBytes(prefixBytes)
  const sha256Impl = deps.sha256Impl || sha256
  const readFileImpl = deps.readFileImpl || readFile
  const execImpl = deps.execImpl || execFileAsync
  const roster = deps.roster || JSON.parse(await readFileImpl(resolve(root, rosterPath), 'utf8'))
  const rosterErrors = validateRoster(roster, resolve(root, rosterPath))
  if (rosterErrors.length) throw templateError('smollm3_template_source_identity_mismatch')
  const row = roster.rows.find((candidate) => candidate?.id === ROW_ID)
  if (!row) throw templateError('smollm3_template_source_identity_mismatch')
  if (row.source.repo !== EXACT_SOURCE.repo
    || row.source.file !== EXACT_SOURCE.file
    || row.source.revision !== EXACT_SOURCE.revision
    || row.source.license !== EXACT_SOURCE.license
    || row.identity.size_bytes !== EXACT_SOURCE.size_bytes
    || row.identity.sha256 !== EXACT_SOURCE.sha256) {
    throw templateError('smollm3_template_source_identity_mismatch')
  }
  const selection = sourceSelectionForRow(row)
  const resolveSelectedSource = async () => {
    let candidate
    try {
      candidate = deps.resolveSource
        ? await deps.resolveSource(selection)
        : await resolveHfSource({
          repo: selection.repo,
          file: selection.file,
          revision: selection.revision,
          token,
        })
    } catch {
      throw templateError('smollm3_template_source_unavailable')
    }
    try { validateLockAgainstSelection(candidate, selection) }
    catch { throw templateError('smollm3_template_source_identity_mismatch') }
    return candidate
  }
  const sourceStateBefore = deps.readSourceState
    ? await deps.readSourceState(root)
    : await readSourceState(root, { execImpl })
  const inspectorBefore = deps.inspectCamelid
    ? await deps.inspectCamelid(binary, sourceStateBefore)
    : await inspectCamelid(binary, sourceStateBefore, { execImpl, readFileImpl })
  const oracleBefore = deps.inspectOracle
    ? await deps.inspectOracle(analyzer)
    : await inspectOracle(analyzer, { execImpl, readFileImpl })
  const lock = await resolveSelectedSource()
  const range = deps.fetchPrefix
    ? await deps.fetchPrefix(lock, { prefixBytes: PREFIX_BYTES, token })
    : await fetchHeaderPrefix(lock, { prefixBytes: PREFIX_BYTES, token })
  if (!Buffer.isBuffer(range?.bytes)
    || range.bytes.length !== PREFIX_BYTES
    || range.prefix_sha256 !== PREFIX_SHA256
    || sha256Impl(range.bytes) !== PREFIX_SHA256
    || range.requested_bytes !== PREFIX_BYTES
    || range.content_range?.start !== 0
    || range.content_range?.end !== PREFIX_BYTES - 1
    || range.content_range?.total !== EXACT_SOURCE.size_bytes) {
    throw templateError('smollm3_template_prefix_identity_mismatch')
  }

  const tempRoot = await (deps.mkdtempImpl || mkdtemp)(join(tmpdir(), 'camelid-smollm3-template-'))
  const prefixPath = join(tempRoot, 'prefix.gguf')
  try {
    await (deps.writeFileImpl || writeFile)(prefixPath, range.bytes, { flag: 'wx' })
    const prefixBefore = await (deps.readFileImpl || readFile)(prefixPath)
    if (!Buffer.isBuffer(prefixBefore)
      || prefixBefore.length !== PREFIX_BYTES
      || sha256Impl(prefixBefore) !== PREFIX_SHA256
      || !prefixBefore.equals(range.bytes)) {
      throw templateError('smollm3_template_prefix_identity_mismatch')
    }
    let inspection
    try {
      inspection = deps.inspectPrefix
        ? await deps.inspectPrefix(binary, prefixPath, lock.size_bytes)
        : await inspectPrefix(binary, prefixPath, lock.size_bytes, { execImpl })
    } catch (error) {
      if (error instanceof HeaderInspectionError) {
        const failure = classifyHeaderInspectionError(error)
        throw templateError(failure.status === 'fail'
          ? 'smollm3_template_metadata_mismatch'
          : 'smollm3_template_inspector_unavailable')
      }
      throw templateError('smollm3_template_qualification_error')
    }
    let summary
    try { summary = (deps.assertMetadata || assertSmolLM3TokenizerMetadata)(inspection) }
    catch { throw templateError('smollm3_template_metadata_mismatch') }
    const template = assertExactTemplate(inspection.metadata['tokenizer.chat_template'])
    if (summary.chat_template_utf8_bytes !== TEMPLATE_UTF8_BYTES
      || summary.chat_template_sha256 !== TEMPLATE_SHA256) {
      throw templateError('smollm3_template_identity_mismatch')
    }
    const analysis = deps.runAnalyzer
      ? await deps.runAnalyzer(analyzer, template)
      : await runTemplateAnalyzer(analyzer, template, { execImpl })
    const sourceStateAfter = deps.readSourceState
      ? await deps.readSourceState(root)
      : await readSourceState(root, { execImpl })
    if (!sameJson(sourceStateAfter, sourceStateBefore)) {
      throw templateError('smollm3_template_source_changed')
    }
    const [inspectorAfter, oracleAfter, lockAfter, prefixAfter] = await Promise.all([
      deps.inspectCamelid
        ? deps.inspectCamelid(binary, sourceStateAfter)
        : inspectCamelid(binary, sourceStateAfter, { execImpl, readFileImpl }),
      deps.inspectOracle ? deps.inspectOracle(analyzer) : inspectOracle(analyzer, { execImpl, readFileImpl }),
      resolveSelectedSource(),
      (deps.readFileImpl || readFile)(prefixPath),
    ])
    if (!sameJson(inspectorAfter, inspectorBefore)) throw templateError('smollm3_template_inspector_changed')
    if (!sameJson(oracleAfter, oracleBefore)) throw templateError('smollm3_template_oracle_changed')
    if (!sameJson(stableLockIdentity(lockAfter), stableLockIdentity(lock))) {
      throw templateError('smollm3_template_source_changed')
    }
    if (!Buffer.isBuffer(prefixAfter)
      || prefixAfter.length !== PREFIX_BYTES
      || sha256Impl(prefixAfter) !== PREFIX_SHA256
      || !prefixAfter.equals(range.bytes)) {
      throw templateError('smollm3_template_prefix_identity_mismatch')
    }
    const pack = buildShapePack({ row, range, template, oracle: oracleBefore, inspector: inspectorBefore, analysis })
    if (validateShapePack(pack, { expectedInspector: pack.inspector }).length) {
      throw templateError('smollm3_template_oracle_output_invalid')
    }
    return pack
  } finally {
    try { await (deps.rmImpl || rm)(tempRoot, { recursive: true, force: true }) }
    catch { throw templateError('smollm3_template_cleanup_failed') }
  }
}

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

async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv)
  const root = resolve(args.get('root') || '.')
  const prefixBytes = normalizePrefixBytes(args.get('prefix-bytes') || PREFIX_BYTES)
  const defaultBinary = process.platform === 'win32' ? 'target/debug/camelid.exe' : 'target/debug/camelid'
  const defaultAnalyzer = process.platform === 'win32'
    ? 'target/reference/llama.cpp-b9632/bin/llama-template-analysis.exe'
    : 'target/reference/llama.cpp-b9632/bin/llama-template-analysis'
  const pack = await qualifySmolLM3Template({
    root,
    rosterPath: args.get('roster') || 'qa/model-qualification/phase1-roster.json',
    binary: resolve(root, args.get('binary') || defaultBinary),
    analyzer: resolve(root, args.get('analyzer') || defaultAnalyzer),
    prefixBytes,
    token: process.env.HF_TOKEN || null,
  })
  const rendered = `${JSON.stringify(pack, null, 2)}\n`
  if (args.get('out')) {
    const out = resolve(root, args.get('out'))
    await mkdir(dirname(out), { recursive: true })
    await writeFile(out, rendered)
  }
  process.stdout.write(rendered)
}

export {
  DATE_PLACEHOLDER,
  DEFAULT_THINKING_INSTRUCTION,
  LLAMA_ANALYZER_SHA256,
  LLAMA_CLI_SHA256,
  LLAMA_REVISION,
  PREFIX_BYTES,
  PREFIX_SHA256,
  ROW_ID,
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
  validateShapePack,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    const failure = classifySmolLM3TemplateQualificationError(error)
    console.error(`${failure.error_code}: ${failure.reason}`)
    process.exit(1)
  })
}
