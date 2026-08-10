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
  assertQwen3MoeTokenizerMetadata,
  parseLlamaVersionOutput,
  sourceSelectionForRow,
} from './hf-qualification-tokenizer.mjs'
import {
  resolveHfSource,
  validateLockAgainstSelection,
} from './hf-qualification-source.mjs'

const execFileAsync = promisify(execFile)
const ROW_ID = 'qwen3_30b_a3b_q8_0'
const PREFIX_BYTES = 32 * 1024 * 1024
const PREFIX_SHA256 = '55c565264523c5862247d983f857b9034c04d762ee14fecfd68a827cdbb2d566'
const TEMPLATE_UTF8_BYTES = 4_100
const TEMPLATE_SHA256 = '57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361'
const LLAMA_BUILD = 9_632
const LLAMA_REVISION = 'acd79d603cb2e1c84c0886137b80f1ad649b6857'
const LLAMA_ANALYZER_SHA256 = '3ee4a64a2cc3f71cb07f0fe7357b779cab03468fc603f76637bb9d5364a9216c'
const LLAMA_CLI_SHA256 = '2ec09da0b81d0201ce5b21810caefb4e77fd108f383b30c15ca493c5a70f7731'
const LLAMA_PACKAGE_FILE_COUNT = 51
const LLAMA_PACKAGE_MANIFEST_BYTES = 4_682
const LLAMA_PACKAGE_MANIFEST_SHA256 = 'd70bbe8beb7848396d0993ee533062c200350fd9961e2b92c799b24f94a33e93'
const LLAMA_ARCHIVE_SIZE_BYTES = 16_899_258
const LLAMA_ARCHIVE_SHA256 = 'b835d5c5155dd2a5ed748a0351debf2ede0dc9f808757e0429f8700a11832dcd'
const NORMALIZED_TRANSCRIPT_UTF8_BYTES = 2_931
const NORMALIZED_TRANSCRIPT_SHA256 = '9d213ca1db68779fd424aa3b75eb0b60d5957e5e4f47737a6247e20980873de9'
const SINGLE_USER_PROMPT_SHA256 = 'd1247d1ee2c7a1804cfa96ffca35c4d7470b47d7d898d77a531b2899076b0745'
const HISTORY_PROMPT_SHA256 = '5bb69176d20519c59dabf2d9dddad85821298960aa047848761deeedf6a46c82'
const HEADER_GROUNDING_RECEIPT = 'qa/model-qualification/qwen3-30b-a3b-q8-header-inspection.json'
const HEADER_GROUNDING_RECEIPT_SHA256 = '293f8dd99f4f31478a0a6a7b3fc9c3e6a1c224a9df0b1dc3253e619d93a2dc33'
const TOKENIZER_GROUNDING_RECEIPT = 'qa/model-qualification/qwen3-30b-a3b-q8-header-tokenizer-parity.json'
const TOKENIZER_GROUNDING_RECEIPT_SHA256 = '021dbe0b4f6a94f7140daa8e02969106dab941e205d184ee60f683d58f13ea37'
const TEMPLATE_GROUNDING_FIXTURE = 'qa/model-qualification/fixtures/qwen3-moe-chat-template-v1.json'
const TEMPLATE_GROUNDING_FIXTURE_SHA256 = 'd56522fa07d1e5b597d3edd708dfaca6087bc5f14e216702d5a27671f304f427'
const ORACLE_USER_TEXT = 'Hello, please help me.'
const ORACLE_ASSISTANT_TEXT = 'I can help you with that.'
const ORACLE_FOLLOWUP_TEXT = 'Thank you.'

const EXACT_SOURCE = Object.freeze({
  repo: 'Qwen/Qwen3-30B-A3B-GGUF',
  file: 'Qwen3-30B-A3B-Q8_0.gguf',
  revision: 'e4d4bafdfb96a411a163846265362aceb0b9c63a',
  size_bytes: 32_483_931_648,
  sha256: '4ad960d180b16f56024f5b704697e5dd5b0837167c2e515ef0569abfc599743c',
  license: 'apache-2.0',
})

// This is the clean exact binary which produced the committed Qwen3-MoE
// tokenizer receipt. The preparation gate deliberately pins it instead of
// silently accepting whatever happens to be in target/debug.
const COMMITTED_GROUNDING_INSPECTOR = Object.freeze({
  version: 'camelid v0.6.1-35-gded8e95b',
  binary_sha256: 'f55721a6fbd66035725f3f67309112e68c9a98cb646a1da9c74a620a4ec2d1f1',
  source_head: 'ded8e95b95fadbe2e7ab6a03d48e4d1e9a2c32d6',
  binary_commit_abbrev: 'ded8e95b',
  binary_reports_dirty: false,
  binary_matches_source_head: true,
  clean_current_head_at_grounding: true,
  binary_path_redacted: true,
})

const COMPLETED_DIFF_SECTIONS = Object.freeze([
  'Diff: With vs Without Tools (single user message)',
  'Diff: With vs Without add_generation_prompt (single user message)',
  'Diff: With vs Without reasoning_content (user, assistant)',
  'Diff: With vs Without reasoning_content (user, assistant, user)',
])

const UNREACHED_DIFF_SECTIONS = Object.freeze([
  'Diff: With vs Without tool call (user, assistant)',
  'Diff: With vs Without tool call (user, assistant, user)',
  'Diff: One vs Two tool calls (user, assistant)',
  'Diff: One vs Two tool calls (user, assistant, user)',
  'Diff: Tool call with vs without reasoning_content (user, assistant)',
])

// b9632 exits zero and prints ANALYSIS COMPLETE even though this exact template
// aborts after four of nine diffs. Pinning the entire normalized transcript is
// what makes this a narrow, reproducible partial observation rather than an
// oracle-success claim.
const EXPECTED_NORMALIZED_TRANSCRIPT = `
================================================================================
                      TEMPLATE ANALYSIS TOOL
================================================================================
Analyzing 1 template(s)

================================================================================
                    ANALYZING TEMPLATE: <template.jinja>
================================================================================

=== Template Capabilities (from jinja::caps) ===
supports_tools: true
supports_tool_calls: true
supports_system_role: true
supports_parallel_tool_calls: true
supports_typed_content: false
supports_string_content: true

=== Diff: With vs Without Tools (single user message) ===
Common Prefix: '<|im_start|>'
Common Suffix: 'user
Hello, please help me.<|im_end|>
'
Left (difference): ''
Right (difference): 'system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "test_function_name", "description": "A test function for debugging", "parameters": {"type": "object", "properties": {"param1": {"type": "string", "description": "First parameter"}, "param2": {"type": "string", "description": "Second parameter"}}, "required": ["param1", "param2"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>'

=== Diff: With vs Without add_generation_prompt (single user message) ===
Common Prefix: '<|im_start|>user
Hello, please help me.<|im_end|>
'
Common Suffix: ''
Left (difference): ''
Right (difference): '<|im_start|>assistant
'

=== Diff: With vs Without reasoning_content (user, assistant) ===
Common Prefix: '<|im_start|>user
Hello, please help me.<|im_end|>
<|im_start|>assistant
<think>
'
Common Suffix: '
</think>

I can help you with that.<|im_end|>
'
Left (difference): ''
Right (difference): 'The user is asking for help. I should respond positively.'

=== Diff: With vs Without reasoning_content (user, assistant, user) ===
Common Prefix: '<|im_start|>user
Hello, please help me.<|im_end|>
<|im_start|>assistant
I can help you with that.<|im_end|>
<|im_start|>user
Thank you.<|im_end|>
'
Common Suffix: ''
Left (difference): ''
Right (difference): ''
Analysis failed:${' '}
------------
While executing BinaryExpression at line 34, column 31 in source:
...- else %}↵            {%- if '</think>' in message.content %}↵                {%...
                                           ^
Error: Cannot perform operation on null values

================================================================================
                      ANALYSIS COMPLETE
================================================================================
`

const ERROR_CONTRACTS = Object.freeze({
  qwen3moe_template_prefix_budget_invalid: ['fail', 'Qwen3-MoE template preparation requires exactly the pinned 32 MiB prefix'],
  qwen3moe_template_grounding_unavailable: ['blocked', 'a committed Qwen3-MoE grounding receipt is unavailable'],
  qwen3moe_template_grounding_mismatch: ['fail', 'a committed Qwen3-MoE grounding receipt does not match its pin'],
  qwen3moe_template_source_unavailable: ['blocked', 'the immutable Qwen3-MoE source lock could not be resolved'],
  qwen3moe_template_source_identity_mismatch: ['fail', 'the immutable source lock does not match the exact Qwen3-MoE row'],
  qwen3moe_template_inspector_unavailable: ['blocked', 'the pinned Camelid prefix inspector is unavailable'],
  qwen3moe_template_inspector_identity_mismatch: ['fail', 'the Camelid prefix inspector does not match the tokenizer-receipt pin'],
  qwen3moe_template_inspector_changed: ['blocked', 'the pinned Camelid prefix inspector changed during preparation'],
  qwen3moe_template_oracle_unavailable: ['blocked', 'the pinned llama.cpp template analyzer package is unavailable'],
  qwen3moe_template_oracle_identity_mismatch: ['fail', 'the llama.cpp template analyzer package does not match the b9632 pin'],
  qwen3moe_template_oracle_changed: ['blocked', 'the llama.cpp template analyzer package changed during preparation'],
  qwen3moe_template_range_unavailable: ['blocked', 'the bounded immutable Qwen3-MoE prefix range is unavailable'],
  qwen3moe_template_range_invalid: ['fail', 'the bounded immutable Qwen3-MoE prefix response is invalid'],
  qwen3moe_template_prefix_identity_mismatch: ['fail', 'the bounded Qwen3-MoE prefix does not match its exact-row SHA-256'],
  qwen3moe_template_metadata_mismatch: ['fail', 'bounded-prefix metadata does not match the exact Qwen3-MoE row'],
  qwen3moe_template_identity_mismatch: ['fail', 'tokenizer.chat_template does not match the exact 4,100-byte Qwen3-MoE template'],
  qwen3moe_template_transcript_mismatch: ['fail', 'the pinned analyzer did not reproduce the exact known partial-failure transcript'],
  qwen3moe_template_source_changed: ['blocked', 'the immutable source lock or grounding inputs changed during preparation'],
  qwen3moe_template_cleanup_failed: ['blocked', 'temporary Qwen3-MoE preparation files could not be removed'],
  qwen3moe_template_qualification_error: ['blocked', 'bounded Qwen3-MoE template preparation could not complete'],
})
const FALLBACK_ERROR = ERROR_CONTRACTS.qwen3moe_template_qualification_error
const ERROR_CODES = new WeakMap()

class Qwen3MoeTemplateQualificationError extends Error {
  constructor(code) {
    const known = typeof code === 'string' && Object.hasOwn(ERROR_CONTRACTS, code)
    const canonical = known ? code : 'qwen3moe_template_qualification_error'
    super(ERROR_CONTRACTS[canonical][1])
    this.name = 'Qwen3MoeTemplateQualificationError'
    this.code = canonical
    this.status = ERROR_CONTRACTS[canonical][0]
    ERROR_CODES.set(this, canonical)
  }
}

function templateError(code) {
  return new Qwen3MoeTemplateQualificationError(code)
}

function classifyQwen3MoeTemplateQualificationError(error) {
  if (error instanceof Qwen3MoeTemplateQualificationError) {
    const canonical = ERROR_CODES.get(error)
    const known = typeof canonical === 'string' && Object.hasOwn(ERROR_CONTRACTS, canonical)
    const code = known ? canonical : 'qwen3moe_template_qualification_error'
    const contract = known ? ERROR_CONTRACTS[code] : FALLBACK_ERROR
    return { status: contract[0], error_code: code, reason: contract[1] }
  }
  if (error instanceof HeaderInspectionError) {
    const failure = classifyHeaderInspectionError(error)
    const code = failure.status === 'fail'
      ? 'qwen3moe_template_range_invalid'
      : 'qwen3moe_template_range_unavailable'
    return { status: ERROR_CONTRACTS[code][0], error_code: code, reason: ERROR_CONTRACTS[code][1] }
  }
  return {
    status: FALLBACK_ERROR[0],
    error_code: 'qwen3moe_template_qualification_error',
    reason: FALLBACK_ERROR[1],
  }
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

function sameJson(left, right) {
  return JSON.stringify(left) === JSON.stringify(right)
}

function exactObjectKeys(value, keys) {
  return value && typeof value === 'object' && !Array.isArray(value)
    && sameJson(Object.keys(value).sort(), [...keys].sort())
}

function containsPrivateMaterial(value) {
  if (typeof value === 'string') {
    return /(?:[A-Z]:\\(?:Users|Documents and Settings|Windows\\Temp|Temp|private)\\|\/(?:Users|home)\/|Bearer\s|hf_[A-Za-z0-9]|https?:\/\/)/.test(value)
  }
  if (Array.isArray(value)) return value.some(containsPrivateMaterial)
  if (value && typeof value === 'object') return Object.values(value).some(containsPrivateMaterial)
  return false
}

function normalizePrefixBytes(value = PREFIX_BYTES) {
  const parsed = typeof value === 'string' && /^\d+$/.test(value) ? Number(value) : value
  if (!Number.isSafeInteger(parsed) || parsed !== PREFIX_BYTES) {
    throw templateError('qwen3moe_template_prefix_budget_invalid')
  }
  return parsed
}

function qwen3MoeTemplatePackAvailable(rowId) {
  return rowId === ROW_ID
}

function qwen3MoeTemplatePrefixBytesForRow(rowId) {
  return qwen3MoeTemplatePackAvailable(rowId) ? PREFIX_BYTES : null
}

function assertExactTemplate(template) {
  if (typeof template !== 'string'
    || Buffer.byteLength(template) !== TEMPLATE_UTF8_BYTES
    || sha256(Buffer.from(template, 'utf8')) !== TEMPLATE_SHA256) {
    throw templateError('qwen3moe_template_identity_mismatch')
  }
  return template
}

function expectedPromptForMessages(messages, addGenerationPrompt = true) {
  let prompt = ''
  for (const message of messages) {
    if (!message || (message.role !== 'user' && message.role !== 'assistant')
      || typeof message.content !== 'string') {
      throw templateError('qwen3moe_template_transcript_mismatch')
    }
    prompt += `<|im_start|>${message.role}\n${message.content}<|im_end|>\n`
  }
  if (addGenerationPrompt) prompt += '<|im_start|>assistant\n'
  return prompt
}

function stripPinnedAnalyzerSgr(value) {
  const text = String(value)
  const clean = text.replace(/\u001b\[(?:0|1|38;5;(?:[0-9]{1,3}))m/g, '')
  if (/[\u001b\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(clean)) {
    throw templateError('qwen3moe_template_transcript_mismatch')
  }
  return clean
}

function normalizeAnalyzerTranscriptCandidate(stderr, templatePath) {
  let normalized = stripPinnedAnalyzerSgr(stderr).replace(/\r\n/g, '\n')
  if (normalized.includes('\r')) throw templateError('qwen3moe_template_transcript_mismatch')
  const escaped = String(templatePath).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const pathLine = new RegExp(`^                    ANALYZING TEMPLATE: ${escaped}$`, 'gm')
  const matches = [...normalized.matchAll(pathLine)]
  if (matches.length !== 1) throw templateError('qwen3moe_template_transcript_mismatch')
  normalized = normalized.replace(pathLine, '                    ANALYZING TEMPLATE: <template.jinja>')
  return normalized
}

function normalizeAnalyzerTranscript(stderr, templatePath) {
  const normalized = normalizeAnalyzerTranscriptCandidate(stderr, templatePath)
  if (Buffer.byteLength(normalized) !== NORMALIZED_TRANSCRIPT_UTF8_BYTES
    || sha256(Buffer.from(normalized, 'utf8')) !== NORMALIZED_TRANSCRIPT_SHA256
    || normalized !== EXPECTED_NORMALIZED_TRANSCRIPT) {
    throw templateError('qwen3moe_template_transcript_mismatch')
  }
  return normalized
}

function isRetryableTruncatedTranscript(stderr, templatePath) {
  let normalized
  try { normalized = normalizeAnalyzerTranscriptCandidate(stderr, templatePath) }
  catch { return false }
  return normalized !== EXPECTED_NORMALIZED_TRANSCRIPT
    && EXPECTED_NORMALIZED_TRANSCRIPT.startsWith(normalized)
    && COMPLETED_DIFF_SECTIONS.every((title) => normalized.includes(`=== ${title} ===`))
    && normalized.length >= EXPECTED_NORMALIZED_TRANSCRIPT.indexOf('Analysis failed:')
}

function captureExactDiff(transcript, title) {
  const escaped = title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const expression = new RegExp(
    `^=== ${escaped} ===\\n`
      + "Common Prefix: '([\\s\\S]*?)'\\n"
      + "Common Suffix: '([\\s\\S]*?)'\\n"
      + "Left \\(difference\\): '([\\s\\S]*?)'\\n"
      + "Right \\(difference\\): '([\\s\\S]*?)'(?:\\n|$)",
    'gm',
  )
  const matches = [...transcript.matchAll(expression)]
  if (matches.length !== 1) throw templateError('qwen3moe_template_transcript_mismatch')
  return {
    common_prefix: matches[0][1],
    common_suffix: matches[0][2],
    left: matches[0][3],
    right: matches[0][4],
  }
}

function parsePartialAnalyzerTranscript(transcript) {
  if (transcript !== EXPECTED_NORMALIZED_TRANSCRIPT
    || Buffer.byteLength(transcript) !== NORMALIZED_TRANSCRIPT_UTF8_BYTES
    || sha256(Buffer.from(transcript, 'utf8')) !== NORMALIZED_TRANSCRIPT_SHA256) {
    throw templateError('qwen3moe_template_transcript_mismatch')
  }
  let priorIndex = -1
  const diffs = new Map()
  for (const title of COMPLETED_DIFF_SECTIONS) {
    const marker = `=== ${title} ===`
    const indices = [...transcript.matchAll(new RegExp(marker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'))]
      .map((match) => match.index)
    if (indices.length !== 1 || indices[0] <= priorIndex) {
      throw templateError('qwen3moe_template_transcript_mismatch')
    }
    priorIndex = indices[0]
    diffs.set(title, captureExactDiff(transcript, title))
  }
  if ((transcript.match(/^=== Diff:/gm) || []).length !== COMPLETED_DIFF_SECTIONS.length
    || UNREACHED_DIFF_SECTIONS.some((title) => transcript.includes(`=== ${title} ===`))) {
    throw templateError('qwen3moe_template_transcript_mismatch')
  }
  const exactFailure = 'Analysis failed: \n'
    + '------------\n'
    + 'While executing BinaryExpression at line 34, column 31 in source:\n'
    + "...- else %}↵            {%- if '</think>' in message.content %}↵                {%...\n"
    + '                                           ^\n'
    + 'Error: Cannot perform operation on null values\n'
  const failureIndex = transcript.indexOf(exactFailure)
  const completionIndex = transcript.indexOf('                      ANALYSIS COMPLETE\n')
  if (failureIndex <= priorIndex || completionIndex <= failureIndex
    || transcript.indexOf(exactFailure, failureIndex + 1) !== -1
    || (transcript.match(/ANALYSIS COMPLETE/g) || []).length !== 1) {
    throw templateError('qwen3moe_template_transcript_mismatch')
  }
  const capabilityMatch = /=== Template Capabilities \(from jinja::caps\) ===\n([\s\S]*?)\n\n=== Diff:/.exec(transcript)
  const capabilities = capabilityMatch?.[1]?.split('\n')
  if (!sameJson(capabilities, [
    'supports_tools: true',
    'supports_tool_calls: true',
    'supports_system_role: true',
    'supports_parallel_tool_calls: true',
    'supports_typed_content: false',
    'supports_string_content: true',
  ])) {
    throw templateError('qwen3moe_template_transcript_mismatch')
  }

  const generation = diffs.get(COMPLETED_DIFF_SECTIONS[1])
  const history = diffs.get(COMPLETED_DIFF_SECTIONS[3])
  const singlePrompt = `${generation.common_prefix}${generation.right}${generation.common_suffix}`
  const historyCore = `${history.common_prefix}${history.left}${history.common_suffix}`
  const historyPrompt = `${historyCore}${generation.right}`
  const expectedSingle = expectedPromptForMessages([{ role: 'user', content: ORACLE_USER_TEXT }], true)
  const expectedHistory = expectedPromptForMessages([
    { role: 'user', content: ORACLE_USER_TEXT },
    { role: 'assistant', content: ORACLE_ASSISTANT_TEXT },
    { role: 'user', content: ORACLE_FOLLOWUP_TEXT },
  ], true)
  if (singlePrompt !== expectedSingle
    || historyPrompt !== expectedHistory
    || Buffer.byteLength(singlePrompt) !== 72
    || Buffer.byteLength(historyPrompt) !== 168
    || sha256(Buffer.from(singlePrompt, 'utf8')) !== SINGLE_USER_PROMPT_SHA256
    || sha256(Buffer.from(historyPrompt, 'utf8')) !== HISTORY_PROMPT_SHA256) {
    throw templateError('qwen3moe_template_transcript_mismatch')
  }
  return {
    result: 'known_partial_failure_after_four_of_nine_diff_sections',
    analyzer_exit_code: 0,
    analyzer_completed_banner: true,
    expected_diff_section_count: 9,
    completed_diff_sections: [...COMPLETED_DIFF_SECTIONS],
    unreached_diff_sections: [...UNREACHED_DIFF_SECTIONS],
    capabilities: {
      supports_tools: true,
      supports_tool_calls: true,
      supports_system_role: true,
      supports_parallel_tool_calls: true,
      supports_typed_content: false,
      supports_string_content: true,
    },
    failure: {
      marker: 'Analysis failed:',
      expression_line: 34,
      expression_column: 31,
      error: 'Cannot perform operation on null values',
    },
    single_user_prompt: singlePrompt,
    user_assistant_user_prompt: historyPrompt,
  }
}

async function runTemplateAnalyzer(analyzer, template, {
  execImpl = execFileAsync,
  mkdtempImpl = mkdtemp,
  readFileImpl = readFile,
  rmImpl = rm,
  writeFileImpl = writeFile,
} = {}) {
  const tempRoot = await mkdtempImpl(join(tmpdir(), 'camelid-qwen3moe-template-analysis-'))
  const templatePath = join(tempRoot, 'template.jinja')
  try {
    const analyzerBefore = await readFileImpl(analyzer)
      .catch(() => { throw templateError('qwen3moe_template_oracle_unavailable') })
    if (!Buffer.isBuffer(analyzerBefore) || sha256(analyzerBefore) !== LLAMA_ANALYZER_SHA256) {
      throw templateError('qwen3moe_template_oracle_identity_mismatch')
    }
    await writeFileImpl(templatePath, template, { encoding: 'utf8', flag: 'wx' })
    let analysis = null
    for (let attempt = 0; attempt < 16; attempt += 1) {
      let result
      try {
        result = await execImpl(analyzer, ['--template-file', templatePath], {
          timeout: 30_000,
          maxBuffer: 16 * 1024 * 1024,
          windowsHide: true,
        })
      } catch {
        throw templateError('qwen3moe_template_oracle_unavailable')
      }
      if (String(result.stdout || '') !== '') {
        throw templateError('qwen3moe_template_transcript_mismatch')
      }
      try {
        analysis = parsePartialAnalyzerTranscript(normalizeAnalyzerTranscript(result.stderr || '', templatePath))
        break
      } catch (error) {
        if (error instanceof Qwen3MoeTemplateQualificationError
          && error.code === 'qwen3moe_template_transcript_mismatch'
          && isRetryableTruncatedTranscript(result.stderr || '', templatePath)) {
          continue
        }
        throw error
      }
    }
    if (analysis === null) throw templateError('qwen3moe_template_oracle_unavailable')
    const [templateAfter, analyzerAfter] = await Promise.all([
      readFileImpl(templatePath),
      readFileImpl(analyzer),
    ]).catch(() => { throw templateError('qwen3moe_template_oracle_unavailable') })
    if (!Buffer.isBuffer(templateAfter)
      || !templateAfter.equals(Buffer.from(template, 'utf8'))
      || templateAfter.length !== TEMPLATE_UTF8_BYTES
      || sha256(templateAfter) !== TEMPLATE_SHA256) {
      throw templateError('qwen3moe_template_identity_mismatch')
    }
    if (!Buffer.isBuffer(analyzerAfter)
      || !analyzerAfter.equals(analyzerBefore)
      || sha256(analyzerAfter) !== LLAMA_ANALYZER_SHA256) {
      throw templateError('qwen3moe_template_oracle_changed')
    }
    return analysis
  } finally {
    try { await rmImpl(tempRoot, { recursive: true, force: true }) }
    catch { throw templateError('qwen3moe_template_cleanup_failed') }
  }
}

async function inspectFileIdentity(path, readFileImpl = readFile) {
  const bytes = await readFileImpl(path)
  if (!Buffer.isBuffer(bytes)) throw new TypeError('binary read was not a Buffer')
  return { executable: basename(path), binary_sha256: sha256(bytes) }
}

async function inspectCamelid(binary, {
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
    throw templateError('qwen3moe_template_inspector_unavailable')
  }
  if (version !== COMMITTED_GROUNDING_INSPECTOR.version
    || identity.binary_sha256 !== COMMITTED_GROUNDING_INSPECTOR.binary_sha256) {
    throw templateError('qwen3moe_template_inspector_identity_mismatch')
  }
  return { ...COMMITTED_GROUNDING_INSPECTOR }
}

async function inspectLlamaPackage(analyzer, {
  readFileImpl = readFile,
  readdirImpl = readdir,
} = {}) {
  if (process.platform !== 'win32' || process.arch !== 'x64') {
    throw templateError('qwen3moe_template_oracle_identity_mismatch')
  }
  const binDir = dirname(analyzer)
  let dirents
  try { dirents = await readdirImpl(binDir, { withFileTypes: true }) }
  catch { throw templateError('qwen3moe_template_oracle_unavailable') }
  if (!Array.isArray(dirents) || dirents.some((entry) => !entry?.isFile?.())) {
    throw templateError('qwen3moe_template_oracle_identity_mismatch')
  }
  const names = dirents.map((entry) => entry.name).sort()
  if (names.length !== LLAMA_PACKAGE_FILE_COUNT
    || names.some((name) => !/^[a-z0-9._+-]+$/.test(name))) {
    throw templateError('qwen3moe_template_oracle_identity_mismatch')
  }
  const lines = []
  for (const name of names) {
    const bytes = await readFileImpl(join(binDir, name))
      .catch(() => { throw templateError('qwen3moe_template_oracle_unavailable') })
    lines.push(`${name}\t${bytes.length}\t${sha256(bytes)}\n`)
  }
  const archivePath = join(dirname(binDir), 'llama-b9632-bin-win-cpu-x64.zip')
  const archive = await readFileImpl(archivePath)
    .catch(() => { throw templateError('qwen3moe_template_oracle_unavailable') })
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
    throw templateError('qwen3moe_template_oracle_identity_mismatch')
  }
  return identity
}

async function inspectOracle(analyzer, {
  execImpl = execFileAsync,
  readFileImpl = readFile,
} = {}) {
  const companion = join(dirname(analyzer), 'llama-cli.exe')
  let output = ''
  try {
    const result = await execImpl(companion, ['--version'], { timeout: 10_000, windowsHide: true })
    output = `${result.stdout || ''}\n${result.stderr || ''}`
  } catch (error) {
    if (error?.killed || error?.signal || error?.code !== 1) {
      throw templateError('qwen3moe_template_oracle_unavailable')
    }
    output = `${error?.stdout || ''}\n${error?.stderr || ''}`
    if (!output.trim()) throw templateError('qwen3moe_template_oracle_unavailable')
  }
  let parsed
  try { parsed = parseLlamaVersionOutput(output) }
  catch { throw templateError('qwen3moe_template_oracle_identity_mismatch') }
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
    if (error instanceof Qwen3MoeTemplateQualificationError) throw error
    throw templateError('qwen3moe_template_oracle_unavailable')
  }
  if (parsed.build !== LLAMA_BUILD
    || analyzerIdentity.executable !== 'llama-template-analysis.exe'
    || companionIdentity.executable !== 'llama-cli.exe'
    || analyzerIdentity.binary_sha256 !== LLAMA_ANALYZER_SHA256
    || companionIdentity.binary_sha256 !== LLAMA_CLI_SHA256) {
    throw templateError('qwen3moe_template_oracle_identity_mismatch')
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

function sourceProjection(value, includeLicense = false) {
  const projection = {
    repo: value?.repo,
    file: value?.file,
    revision: value?.revision,
    size_bytes: value?.size_bytes,
    sha256: value?.sha256,
  }
  if (includeLicense) projection.license = value?.license
  return projection
}

async function readGroundingSnapshot(root, readFileImpl = readFile) {
  let entries
  try {
    entries = await Promise.all([
      [HEADER_GROUNDING_RECEIPT, HEADER_GROUNDING_RECEIPT_SHA256],
      [TOKENIZER_GROUNDING_RECEIPT, TOKENIZER_GROUNDING_RECEIPT_SHA256],
      [TEMPLATE_GROUNDING_FIXTURE, TEMPLATE_GROUNDING_FIXTURE_SHA256],
    ].map(async ([relativePath, expectedHash]) => {
      const bytes = await readFileImpl(resolve(root, relativePath))
      if (!Buffer.isBuffer(bytes)) throw new TypeError('grounding read was not a Buffer')
      const digest = sha256(bytes)
      if (digest !== expectedHash) throw templateError('qwen3moe_template_grounding_mismatch')
      return { relativePath, bytes, digest, parsed: JSON.parse(bytes.toString('utf8')) }
    }))
  } catch (error) {
    if (error instanceof Qwen3MoeTemplateQualificationError) throw error
    throw templateError('qwen3moe_template_grounding_unavailable')
  }
  const [headerEntry, tokenizerEntry, templateEntry] = entries
  const header = headerEntry.parsed
  const tokenizer = tokenizerEntry.parsed
  const fixture = templateEntry.parsed
  const exactWithoutLicense = sourceProjection(EXACT_SOURCE)
  if (header?.schema !== 'camelid.remote-gguf-header-inspection/v1'
    || header?.row_id !== ROW_ID
    || !sameJson(sourceProjection(header.source), exactWithoutLicense)
    || header?.range?.requested_bytes !== PREFIX_BYTES
    || header?.range?.received_bytes !== PREFIX_BYTES
    || header?.range?.prefix_sha256 !== PREFIX_SHA256
    || !sameJson(header?.range?.content_range, { start: 0, end: PREFIX_BYTES - 1, total: EXACT_SOURCE.size_bytes })
    || header?.inspection?.observed?.architecture !== 'qwen3moe'
    || header?.inspection?.observed?.tokenizer_pre !== 'qwen2') {
    throw templateError('qwen3moe_template_grounding_mismatch')
  }
  if (tokenizer?.schema !== 'camelid.header-tokenizer-parity/v1'
    || tokenizer?.row_id !== ROW_ID
    || !sameJson(sourceProjection(tokenizer.source, true), EXACT_SOURCE)
    || tokenizer?.bounded_fetch?.requested_bytes !== PREFIX_BYTES
    || tokenizer?.bounded_fetch?.received_bytes !== PREFIX_BYTES
    || tokenizer?.bounded_fetch?.prefix_sha256 !== PREFIX_SHA256
    || tokenizer?.tokenizer_metadata?.chat_template_utf8_bytes !== TEMPLATE_UTF8_BYTES
    || tokenizer?.tokenizer_metadata?.chat_template_sha256 !== TEMPLATE_SHA256
    || tokenizer?.result?.all_token_ids_match !== true
    || tokenizer?.result?.support_decision !== 'qwen3_moe_exact_row_tokenizer_gate_only') {
    throw templateError('qwen3moe_template_grounding_mismatch')
  }
  if (fixture?.schema !== 'camelid.qwen3_moe_chat_template_fixture.v1'
    || !sameJson(sourceProjection(fixture.source), exactWithoutLicense)
    || fixture?.grounding?.header_receipt !== HEADER_GROUNDING_RECEIPT
    || fixture?.grounding?.header_receipt_sha256 !== HEADER_GROUNDING_RECEIPT_SHA256
    || fixture?.grounding?.receipt_prefix_bytes !== PREFIX_BYTES
    || fixture?.grounding?.receipt_prefix_sha256 !== PREFIX_SHA256
    || fixture?.template?.utf8_bytes !== TEMPLATE_UTF8_BYTES
    || fixture?.template?.sha256 !== TEMPLATE_SHA256) {
    throw templateError('qwen3moe_template_grounding_mismatch')
  }
  const template = assertExactTemplate(fixture.template.jinja)
  return {
    identity: entries.map(({ relativePath, bytes, digest }) => ({
      path: relativePath,
      utf8_bytes: bytes.length,
      sha256: digest,
    })),
    template,
  }
}

function stableLockIdentity(lock) {
  return {
    schema: lock?.schema,
    repo: lock?.repo,
    file: lock?.file,
    revision: lock?.revision,
    size_bytes: lock?.size_bytes,
    sha256: lock?.sha256,
    license: lock?.license,
    download_url: lock?.download_url,
    access: lock?.access && typeof lock.access === 'object'
      ? {
        gated: lock.access.gated,
        private: lock.access.private,
        disabled: lock.access.disabled,
      }
      : lock?.access,
  }
}

function validateExactResolvedLock(lock, selection) {
  try { validateLockAgainstSelection(lock, selection) }
  catch { throw templateError('qwen3moe_template_source_identity_mismatch') }
  const expectedDownloadUrl = `https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/${EXACT_SOURCE.revision}/Qwen3-30B-A3B-Q8_0.gguf?download=true`
  if (!exactObjectKeys(lock, [
    'schema', 'repo', 'file', 'revision', 'size_bytes', 'sha256', 'license',
    'access', 'download_url',
  ])
    || lock.schema !== 'camelid.hf-source-lock/v1'
    || !exactObjectKeys(lock.access, ['gated', 'private', 'disabled'])
    || !sameJson(lock.access, { gated: false, private: false, disabled: false })
    || lock.download_url !== expectedDownloadUrl) {
    throw templateError('qwen3moe_template_source_identity_mismatch')
  }
  return lock
}

function buildShapePack({ range, template, oracle, inspector, analysis }) {
  const singleMessages = [{ role: 'user', content: ORACLE_USER_TEXT }]
  const historyMessages = [
    { role: 'user', content: ORACLE_USER_TEXT },
    { role: 'assistant', content: ORACLE_ASSISTANT_TEXT },
    { role: 'user', content: ORACLE_FOLLOWUP_TEXT },
  ]
  return {
    schema: 'camelid.qwen3-moe-chat-template-shapes/v1',
    pack_id: 'qwen3-moe-chat-template-shapes-v1',
    row_id: ROW_ID,
    support_scope: 'preparation_only_template_gate_blocked_known_partial_analyzer_evidence',
    grounding: {
      header_receipt: HEADER_GROUNDING_RECEIPT,
      header_receipt_sha256: HEADER_GROUNDING_RECEIPT_SHA256,
      tokenizer_receipt: TOKENIZER_GROUNDING_RECEIPT,
      tokenizer_receipt_sha256: TOKENIZER_GROUNDING_RECEIPT_SHA256,
      template_fixture: TEMPLATE_GROUNDING_FIXTURE,
      template_fixture_sha256: TEMPLATE_GROUNDING_FIXTURE_SHA256,
    },
    source: { ...EXACT_SOURCE },
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
    inspector: { ...inspector },
    analyzer: {
      project: oracle.project,
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
      mode: 'known_partial_failure_not_oracle_success',
      result: analysis.result,
      exit_code: analysis.analyzer_exit_code,
      completed_banner_observed: analysis.analyzer_completed_banner,
      expected_diff_section_count: analysis.expected_diff_section_count,
      completed_diff_sections: analysis.completed_diff_sections,
      unreached_diff_sections: analysis.unreached_diff_sections,
      normalized_transcript: {
        normalization: 'strip_pinned_sgr_crlf_to_lf_redact_exact_template_path',
        capture_requirement: 'exact_full_match_after_bounded_strict_prefix_only_retries',
        utf8_bytes: NORMALIZED_TRANSCRIPT_UTF8_BYTES,
        sha256: NORMALIZED_TRANSCRIPT_SHA256,
        persisted_in_pack: false,
      },
      failure: analysis.failure,
      reported_capabilities_before_failure: analysis.capabilities,
    },
    cases: [
      {
        id: 'omitted_thinking_single_user_generation_prompt',
        messages: singleMessages,
        tools_present: false,
        enable_thinking_input: 'omitted_by_pinned_analyzer_case',
        add_generation_prompt: true,
        prompt: analysis.single_user_prompt,
        prompt_utf8_bytes: Buffer.byteLength(analysis.single_user_prompt),
        prompt_sha256: sha256(Buffer.from(analysis.single_user_prompt, 'utf8')),
        grounding: 'exact completed add_generation_prompt diff right reconstruction',
      },
      {
        id: 'omitted_thinking_user_assistant_user_generation_prompt',
        messages: historyMessages,
        tools_present: false,
        enable_thinking_input: 'omitted_by_pinned_analyzer_case',
        add_generation_prompt: true,
        prompt: analysis.user_assistant_user_prompt,
        prompt_utf8_bytes: Buffer.byteLength(analysis.user_assistant_user_prompt),
        prompt_sha256: sha256(Buffer.from(analysis.user_assistant_user_prompt, 'utf8')),
        grounding: 'exact completed user-assistant-user core plus exact isolated generation suffix',
      },
    ],
    contract: {
      evidence_status: 'partial_analyzer_failure_expected_and_pinned',
      qualified_cases: [
        'omitted_thinking_single_user_generation_prompt',
        'omitted_thinking_user_assistant_user_generation_prompt',
      ],
      qualification_granularity: 'these_two_exact_no_tools_text_shapes_only',
      roles_observed: ['user', 'assistant'],
      content_observed: 'nonempty_strings_only',
      tools_observed: false,
      enable_thinking_observed: 'omitted_only',
      add_generation_prompt_observed: true,
      parse_special_for_future_tokenization: true,
      add_special_for_future_tokenization: false,
      full_analyzer_success: false,
      runtime_renderer_qualified: false,
    },
    typed_hold_branches: [
      'all_message_shapes_other_than_the_two_exact_cases',
      'system_messages',
      'tools_and_tool_calls',
      'tool_role_history',
      'reasoning_content',
      'explicit_enable_thinking_true',
      'explicit_enable_thinking_false',
      'add_generation_prompt_false',
      'empty_or_nontext_content',
      'multimodal_content',
      'runtime_apply_template_and_chat_surfaces',
    ],
    template_gate: {
      status: 'blocked',
      reason: 'b9632 exits zero but reports Analysis failed after four of nine diff sections; only two exact pre-failure no-tools omitted-thinking text shapes are retained',
    },
    does_not_prove: [
      'full_template_oracle_success',
      'runtime_chat_enablement',
      'http_apply_template_parity',
      'full_weight_download_or_hash',
      'model_load',
      'generation',
      'tools_or_tool_calls',
      'system_messages',
      'reasoning_content',
      'explicit_thinking_modes',
      'tokenization_of_rendered_prompts',
      'API_SSE_Models_page_WebUI_or_context_readiness',
      'neighboring_rows_or_broad_Qwen3_MoE_support',
    ],
    support_decision: 'no_roster_change_template_preparation_only',
  }
}

function validateShapePack(pack, { expectedInspector = COMMITTED_GROUNDING_INSPECTOR } = {}) {
  const errors = []
  const check = (condition, message) => { if (!condition) errors.push(message) }
  check(exactObjectKeys(pack, [
    'schema', 'pack_id', 'row_id', 'support_scope', 'grounding', 'source',
    'bounded_prefix', 'source_template', 'inspector', 'analyzer', 'cases', 'contract',
    'typed_hold_branches', 'template_gate', 'does_not_prove', 'support_decision',
  ]), 'top-level fields mismatch')
  check(pack?.schema === 'camelid.qwen3-moe-chat-template-shapes/v1', 'schema mismatch')
  check(pack?.pack_id === 'qwen3-moe-chat-template-shapes-v1', 'pack_id mismatch')
  check(pack?.row_id === ROW_ID, 'row_id mismatch')
  check(pack?.support_scope === 'preparation_only_template_gate_blocked_known_partial_analyzer_evidence', 'support scope mismatch')
  check(exactObjectKeys(pack?.grounding, [
    'header_receipt', 'header_receipt_sha256', 'tokenizer_receipt',
    'tokenizer_receipt_sha256', 'template_fixture', 'template_fixture_sha256',
  ]), 'grounding fields mismatch')
  check(sameJson(pack?.grounding, {
    header_receipt: HEADER_GROUNDING_RECEIPT,
    header_receipt_sha256: HEADER_GROUNDING_RECEIPT_SHA256,
    tokenizer_receipt: TOKENIZER_GROUNDING_RECEIPT,
    tokenizer_receipt_sha256: TOKENIZER_GROUNDING_RECEIPT_SHA256,
    template_fixture: TEMPLATE_GROUNDING_FIXTURE,
    template_fixture_sha256: TEMPLATE_GROUNDING_FIXTURE_SHA256,
  }), 'grounding mismatch')
  check(exactObjectKeys(pack?.source, Object.keys(EXACT_SOURCE)), 'source fields mismatch')
  check(sameJson(pack?.source, EXACT_SOURCE), 'source mismatch')
  check(exactObjectKeys(pack?.bounded_prefix, [
    'requested_bytes', 'received_bytes', 'prefix_sha256', 'content_range',
    'full_weights_downloaded',
  ]), 'bounded-prefix fields mismatch')
  check(exactObjectKeys(pack?.bounded_prefix?.content_range, ['start', 'end', 'total']), 'Content-Range fields mismatch')
  check(pack?.bounded_prefix?.requested_bytes === PREFIX_BYTES, 'prefix budget mismatch')
  check(pack?.bounded_prefix?.received_bytes === PREFIX_BYTES, 'received prefix mismatch')
  check(pack?.bounded_prefix?.prefix_sha256 === PREFIX_SHA256, 'prefix SHA-256 mismatch')
  check(sameJson(pack?.bounded_prefix?.content_range, {
    start: 0,
    end: PREFIX_BYTES - 1,
    total: EXACT_SOURCE.size_bytes,
  }), 'Content-Range mismatch')
  check(pack?.bounded_prefix?.full_weights_downloaded === false, 'full-weight scope mismatch')
  check(exactObjectKeys(pack?.source_template, ['utf8_bytes', 'sha256', 'text']), 'template fields mismatch')
  check(pack?.source_template?.utf8_bytes === TEMPLATE_UTF8_BYTES, 'template byte declaration mismatch')
  check(pack?.source_template?.sha256 === TEMPLATE_SHA256, 'template SHA declaration mismatch')
  if (typeof pack?.source_template?.text === 'string') {
    check(Buffer.byteLength(pack.source_template.text) === TEMPLATE_UTF8_BYTES, 'template text byte count mismatch')
    check(sha256(Buffer.from(pack.source_template.text, 'utf8')) === TEMPLATE_SHA256, 'template text SHA mismatch')
  } else {
    check(false, 'template text missing')
  }
  check(exactObjectKeys(pack?.inspector, Object.keys(COMMITTED_GROUNDING_INSPECTOR)), 'inspector fields mismatch')
  check(sameJson(pack?.inspector, expectedInspector), 'inspector identity mismatch')
  check(pack?.inspector?.binary_path_redacted === true && !Object.hasOwn(pack?.inspector || {}, 'binary_path'), 'inspector path privacy mismatch')

  const analyzerKeys = [
    'project', 'build', 'revision', 'reported_revision', 'analyzer_executable',
    'analyzer_binary_sha256', 'companion_executable', 'companion_binary_sha256',
    'platform', 'package_file_count', 'package_manifest_bytes', 'package_manifest_sha256',
    'archive_size_bytes', 'archive_sha256', 'executable_paths_redacted', 'mode',
    'result', 'exit_code', 'completed_banner_observed', 'expected_diff_section_count',
    'completed_diff_sections', 'unreached_diff_sections', 'normalized_transcript',
    'failure', 'reported_capabilities_before_failure',
  ]
  check(exactObjectKeys(pack?.analyzer, analyzerKeys), 'analyzer fields mismatch')
  check(pack?.analyzer?.project === 'llama.cpp', 'analyzer project mismatch')
  check(pack?.analyzer?.build === LLAMA_BUILD, 'analyzer build mismatch')
  check(pack?.analyzer?.revision === LLAMA_REVISION, 'analyzer revision mismatch')
  check(pack?.analyzer?.reported_revision === 'acd79d603', 'analyzer reported revision mismatch')
  check(pack?.analyzer?.analyzer_executable === 'llama-template-analysis.exe', 'analyzer basename mismatch')
  check(pack?.analyzer?.analyzer_binary_sha256 === LLAMA_ANALYZER_SHA256, 'analyzer binary mismatch')
  check(pack?.analyzer?.companion_executable === 'llama-cli.exe', 'companion basename mismatch')
  check(pack?.analyzer?.companion_binary_sha256 === LLAMA_CLI_SHA256, 'companion binary mismatch')
  check(pack?.analyzer?.platform === 'win32-x64', 'analyzer platform mismatch')
  check(pack?.analyzer?.package_file_count === LLAMA_PACKAGE_FILE_COUNT, 'package file count mismatch')
  check(pack?.analyzer?.package_manifest_bytes === LLAMA_PACKAGE_MANIFEST_BYTES, 'package manifest byte count mismatch')
  check(pack?.analyzer?.package_manifest_sha256 === LLAMA_PACKAGE_MANIFEST_SHA256, 'package manifest mismatch')
  check(pack?.analyzer?.archive_size_bytes === LLAMA_ARCHIVE_SIZE_BYTES, 'archive size mismatch')
  check(pack?.analyzer?.archive_sha256 === LLAMA_ARCHIVE_SHA256, 'archive SHA mismatch')
  check(pack?.analyzer?.executable_paths_redacted === true, 'analyzer path redaction mismatch')
  check(pack?.analyzer?.mode === 'known_partial_failure_not_oracle_success', 'analyzer mode mismatch')
  check(pack?.analyzer?.result === 'known_partial_failure_after_four_of_nine_diff_sections', 'analyzer result mismatch')
  check(pack?.analyzer?.exit_code === 0, 'analyzer exit mismatch')
  check(pack?.analyzer?.completed_banner_observed === true, 'analyzer banner mismatch')
  check(pack?.analyzer?.expected_diff_section_count === 9, 'expected section count mismatch')
  check(sameJson(pack?.analyzer?.completed_diff_sections, COMPLETED_DIFF_SECTIONS), 'completed sections mismatch')
  check(sameJson(pack?.analyzer?.unreached_diff_sections, UNREACHED_DIFF_SECTIONS), 'unreached sections mismatch')
  check(exactObjectKeys(pack?.analyzer?.normalized_transcript, [
    'normalization', 'capture_requirement', 'utf8_bytes', 'sha256', 'persisted_in_pack',
  ]), 'transcript fields mismatch')
  check(sameJson(pack?.analyzer?.normalized_transcript, {
    normalization: 'strip_pinned_sgr_crlf_to_lf_redact_exact_template_path',
    capture_requirement: 'exact_full_match_after_bounded_strict_prefix_only_retries',
    utf8_bytes: NORMALIZED_TRANSCRIPT_UTF8_BYTES,
    sha256: NORMALIZED_TRANSCRIPT_SHA256,
    persisted_in_pack: false,
  }), 'transcript identity mismatch')
  check(exactObjectKeys(pack?.analyzer?.failure, ['marker', 'expression_line', 'expression_column', 'error']), 'failure fields mismatch')
  check(sameJson(pack?.analyzer?.failure, {
    marker: 'Analysis failed:',
    expression_line: 34,
    expression_column: 31,
    error: 'Cannot perform operation on null values',
  }), 'failure boundary mismatch')
  check(exactObjectKeys(pack?.analyzer?.reported_capabilities_before_failure, [
    'supports_tools', 'supports_tool_calls', 'supports_system_role',
    'supports_parallel_tool_calls', 'supports_typed_content', 'supports_string_content',
  ]), 'capability fields mismatch')
  check(sameJson(pack?.analyzer?.reported_capabilities_before_failure, {
    supports_tools: true,
    supports_tool_calls: true,
    supports_system_role: true,
    supports_parallel_tool_calls: true,
    supports_typed_content: false,
    supports_string_content: true,
  }), 'reported capabilities mismatch')

  const singleMessages = [{ role: 'user', content: ORACLE_USER_TEXT }]
  const historyMessages = [
    { role: 'user', content: ORACLE_USER_TEXT },
    { role: 'assistant', content: ORACLE_ASSISTANT_TEXT },
    { role: 'user', content: ORACLE_FOLLOWUP_TEXT },
  ]
  const expectedCases = [
    {
      id: 'omitted_thinking_single_user_generation_prompt',
      messages: singleMessages,
      prompt: expectedPromptForMessages(singleMessages, true),
      bytes: 72,
      digest: SINGLE_USER_PROMPT_SHA256,
      grounding: 'exact completed add_generation_prompt diff right reconstruction',
    },
    {
      id: 'omitted_thinking_user_assistant_user_generation_prompt',
      messages: historyMessages,
      prompt: expectedPromptForMessages(historyMessages, true),
      bytes: 168,
      digest: HISTORY_PROMPT_SHA256,
      grounding: 'exact completed user-assistant-user core plus exact isolated generation suffix',
    },
  ]
  check(Array.isArray(pack?.cases) && pack.cases.length === 2, 'case count mismatch')
  for (let index = 0; index < expectedCases.length; index += 1) {
    const candidate = pack?.cases?.[index]
    const expected = expectedCases[index]
    check(exactObjectKeys(candidate, [
      'id', 'messages', 'tools_present', 'enable_thinking_input',
      'add_generation_prompt', 'prompt', 'prompt_utf8_bytes', 'prompt_sha256', 'grounding',
    ]), `case ${index} fields mismatch`)
    check(candidate?.id === expected.id, `case ${index} id mismatch`)
    check(Array.isArray(candidate?.messages)
      && candidate.messages.every((message) => exactObjectKeys(message, ['role', 'content'])), `case ${index} message fields mismatch`)
    check(sameJson(candidate?.messages, expected.messages), `case ${index} messages mismatch`)
    check(candidate?.tools_present === false, `case ${index} tools scope mismatch`)
    check(candidate?.enable_thinking_input === 'omitted_by_pinned_analyzer_case', `case ${index} thinking scope mismatch`)
    check(candidate?.add_generation_prompt === true, `case ${index} generation flag mismatch`)
    check(candidate?.prompt === expected.prompt, `case ${index} prompt mismatch`)
    check(candidate?.prompt_utf8_bytes === expected.bytes, `case ${index} prompt bytes mismatch`)
    check(candidate?.prompt_sha256 === expected.digest, `case ${index} prompt SHA mismatch`)
    check(candidate?.grounding === expected.grounding, `case ${index} grounding mismatch`)
  }

  const expectedContract = {
    evidence_status: 'partial_analyzer_failure_expected_and_pinned',
    qualified_cases: expectedCases.map((entry) => entry.id),
    qualification_granularity: 'these_two_exact_no_tools_text_shapes_only',
    roles_observed: ['user', 'assistant'],
    content_observed: 'nonempty_strings_only',
    tools_observed: false,
    enable_thinking_observed: 'omitted_only',
    add_generation_prompt_observed: true,
    parse_special_for_future_tokenization: true,
    add_special_for_future_tokenization: false,
    full_analyzer_success: false,
    runtime_renderer_qualified: false,
  }
  check(exactObjectKeys(pack?.contract, Object.keys(expectedContract)), 'contract fields mismatch')
  check(sameJson(pack?.contract, expectedContract), 'contract mismatch')
  const expectedHolds = [
    'all_message_shapes_other_than_the_two_exact_cases',
    'system_messages',
    'tools_and_tool_calls',
    'tool_role_history',
    'reasoning_content',
    'explicit_enable_thinking_true',
    'explicit_enable_thinking_false',
    'add_generation_prompt_false',
    'empty_or_nontext_content',
    'multimodal_content',
    'runtime_apply_template_and_chat_surfaces',
  ]
  check(sameJson(pack?.typed_hold_branches, expectedHolds), 'typed HOLD list mismatch')
  check(exactObjectKeys(pack?.template_gate, ['status', 'reason']), 'template gate fields mismatch')
  check(sameJson(pack?.template_gate, {
    status: 'blocked',
    reason: 'b9632 exits zero but reports Analysis failed after four of nine diff sections; only two exact pre-failure no-tools omitted-thinking text shapes are retained',
  }), 'template gate mismatch')
  const expectedExclusions = [
    'full_template_oracle_success',
    'runtime_chat_enablement',
    'http_apply_template_parity',
    'full_weight_download_or_hash',
    'model_load',
    'generation',
    'tools_or_tool_calls',
    'system_messages',
    'reasoning_content',
    'explicit_thinking_modes',
    'tokenization_of_rendered_prompts',
    'API_SSE_Models_page_WebUI_or_context_readiness',
    'neighboring_rows_or_broad_Qwen3_MoE_support',
  ]
  check(sameJson(pack?.does_not_prove, expectedExclusions), 'scope exclusions mismatch')
  check(pack?.support_decision === 'no_roster_change_template_preparation_only', 'support decision mismatch')
  check(!containsPrivateMaterial(pack), 'pack contains a path, token, or URL')
  return errors
}

async function qualifyQwen3MoeTemplate({
  root = resolve('.'),
  rosterPath = 'qa/model-qualification/phase1-roster.json',
  binary,
  analyzer,
  prefixBytes = PREFIX_BYTES,
  token = null,
  initialLock = null,
  sourceResolver = resolveHfSource,
}, deps = {}) {
  normalizePrefixBytes(prefixBytes)
  const sha256Impl = deps.sha256Impl || sha256
  const readFileImpl = deps.readFileImpl || readFile
  const execImpl = deps.execImpl || execFileAsync
  let roster
  try {
    roster = deps.roster || JSON.parse(await readFileImpl(resolve(root, rosterPath), 'utf8'))
  } catch {
    throw templateError('qwen3moe_template_source_identity_mismatch')
  }
  const rosterErrors = validateRoster(roster, resolve(root, rosterPath))
  if (rosterErrors.length) throw templateError('qwen3moe_template_source_identity_mismatch')
  const row = roster.rows.find((candidate) => candidate?.id === ROW_ID)
  if (!row
    || !sameJson({
      repo: row.source.repo,
      file: row.source.file,
      revision: row.source.revision,
      size_bytes: row.identity.size_bytes,
      sha256: row.identity.sha256,
      license: row.source.license,
    }, EXACT_SOURCE)) {
    throw templateError('qwen3moe_template_source_identity_mismatch')
  }
  const selection = sourceSelectionForRow(row)
  const resolveSelectedSource = async () => {
    let candidate
    try {
      candidate = deps.resolveSource
        ? await deps.resolveSource(selection)
        : await sourceResolver({
          repo: selection.repo,
          file: selection.file,
          revision: selection.revision,
          token,
        })
    } catch {
      throw templateError('qwen3moe_template_source_unavailable')
    }
    return validateExactResolvedLock(candidate, selection)
  }

  const groundingBefore = deps.readGrounding
    ? await deps.readGrounding(root)
    : await readGroundingSnapshot(root, readFileImpl)
  const [inspectorBefore, oracleBefore] = await Promise.all([
    deps.inspectCamelid ? deps.inspectCamelid(binary) : inspectCamelid(binary, { execImpl, readFileImpl }),
    deps.inspectOracle ? deps.inspectOracle(analyzer) : inspectOracle(analyzer, { execImpl, readFileImpl }),
  ])
  let lock
  if (initialLock === null) {
    lock = await resolveSelectedSource()
  } else {
    lock = validateExactResolvedLock(initialLock, selection)
  }
  const lockIdentityBefore = stableLockIdentity(lock)
  let range
  try {
    range = deps.fetchPrefix
      ? await deps.fetchPrefix(lock, { prefixBytes: PREFIX_BYTES, token })
      : await fetchHeaderPrefix(lock, { prefixBytes: PREFIX_BYTES, token })
  } catch (error) {
    if (error instanceof HeaderInspectionError) throw error
    throw templateError('qwen3moe_template_range_unavailable')
  }
  if (!Buffer.isBuffer(range?.bytes)
    || range.bytes.length !== PREFIX_BYTES
    || range.prefix_sha256 !== PREFIX_SHA256
    || sha256Impl(range.bytes) !== PREFIX_SHA256
    || range.requested_bytes !== PREFIX_BYTES
    || !sameJson(range.content_range, { start: 0, end: PREFIX_BYTES - 1, total: EXACT_SOURCE.size_bytes })) {
    throw templateError('qwen3moe_template_prefix_identity_mismatch')
  }

  const tempRoot = await (deps.mkdtempImpl || mkdtemp)(join(tmpdir(), 'camelid-qwen3moe-template-'))
  const prefixPath = join(tempRoot, 'prefix.gguf')
  try {
    await (deps.writeFileImpl || writeFile)(prefixPath, range.bytes, { flag: 'wx' })
    const prefixBefore = await readFileImpl(prefixPath)
    if (!Buffer.isBuffer(prefixBefore)
      || prefixBefore.length !== PREFIX_BYTES
      || sha256Impl(prefixBefore) !== PREFIX_SHA256
      || !prefixBefore.equals(range.bytes)) {
      throw templateError('qwen3moe_template_prefix_identity_mismatch')
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
          ? 'qwen3moe_template_metadata_mismatch'
          : 'qwen3moe_template_inspector_unavailable')
      }
      throw templateError('qwen3moe_template_qualification_error')
    }
    let summary
    try { summary = (deps.assertMetadata || assertQwen3MoeTokenizerMetadata)(inspection) }
    catch { throw templateError('qwen3moe_template_metadata_mismatch') }
    const template = assertExactTemplate(inspection?.metadata?.['tokenizer.chat_template'])
    if (template !== groundingBefore.template
      || summary.chat_template_utf8_bytes !== TEMPLATE_UTF8_BYTES
      || summary.chat_template_sha256 !== TEMPLATE_SHA256) {
      throw templateError('qwen3moe_template_identity_mismatch')
    }
    const analysis = deps.runAnalyzer
      ? await deps.runAnalyzer(analyzer, template)
      : await runTemplateAnalyzer(analyzer, template, { execImpl, readFileImpl })

    const [inspectorAfter, oracleAfter, lockAfter, groundingAfter, prefixAfter] = await Promise.all([
      deps.inspectCamelid ? deps.inspectCamelid(binary) : inspectCamelid(binary, { execImpl, readFileImpl }),
      deps.inspectOracle ? deps.inspectOracle(analyzer) : inspectOracle(analyzer, { execImpl, readFileImpl }),
      resolveSelectedSource(),
      deps.readGrounding ? deps.readGrounding(root) : readGroundingSnapshot(root, readFileImpl),
      readFileImpl(prefixPath),
    ])
    if (!sameJson(inspectorAfter, inspectorBefore)) throw templateError('qwen3moe_template_inspector_changed')
    if (!sameJson(oracleAfter, oracleBefore)) throw templateError('qwen3moe_template_oracle_changed')
    if (!sameJson(stableLockIdentity(lockAfter), lockIdentityBefore)
      || !sameJson(groundingAfter.identity, groundingBefore.identity)
      || groundingAfter.template !== groundingBefore.template) {
      throw templateError('qwen3moe_template_source_changed')
    }
    if (!Buffer.isBuffer(prefixAfter)
      || prefixAfter.length !== PREFIX_BYTES
      || sha256Impl(prefixAfter) !== PREFIX_SHA256
      || !prefixAfter.equals(range.bytes)) {
      throw templateError('qwen3moe_template_prefix_identity_mismatch')
    }
    const pack = buildShapePack({
      range,
      template,
      oracle: oracleBefore,
      inspector: inspectorBefore,
      analysis,
    })
    if (validateShapePack(pack).length) {
      throw templateError('qwen3moe_template_transcript_mismatch')
    }
    return pack
  } finally {
    try { await (deps.rmImpl || rm)(tempRoot, { recursive: true, force: true }) }
    catch { throw templateError('qwen3moe_template_cleanup_failed') }
  }
}

function parseArgs(argv) {
  const allowed = new Set(['root', 'roster', 'binary', 'analyzer', 'prefix-bytes', 'out'])
  const args = new Map()
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    if (!arg.startsWith('--')) throw templateError('qwen3moe_template_qualification_error')
    const [key, inline] = arg.slice(2).split('=', 2)
    if (!allowed.has(key) || args.has(key)) throw templateError('qwen3moe_template_qualification_error')
    const next = argv[index + 1]
    const value = inline ?? (next && !next.startsWith('--') ? argv[++index] : null)
    if (value === null || value === '') throw templateError('qwen3moe_template_qualification_error')
    args.set(key, value)
  }
  return args
}

async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv)
  const root = resolve(args.get('root') || '.')
  const prefixBytes = normalizePrefixBytes(args.get('prefix-bytes') || PREFIX_BYTES)
  const pack = await qualifyQwen3MoeTemplate({
    root,
    rosterPath: args.get('roster') || 'qa/model-qualification/phase1-roster.json',
    binary: resolve(root, args.get('binary') || 'target/debug/camelid.exe'),
    analyzer: resolve(root, args.get('analyzer') || 'target/reference/llama.cpp-b9632/bin/llama-template-analysis.exe'),
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
  COMMITTED_GROUNDING_INSPECTOR,
  COMPLETED_DIFF_SECTIONS,
  EXPECTED_NORMALIZED_TRANSCRIPT,
  HISTORY_PROMPT_SHA256,
  LLAMA_ANALYZER_SHA256,
  LLAMA_CLI_SHA256,
  LLAMA_REVISION,
  NORMALIZED_TRANSCRIPT_SHA256,
  NORMALIZED_TRANSCRIPT_UTF8_BYTES,
  PREFIX_BYTES,
  PREFIX_SHA256,
  Qwen3MoeTemplateQualificationError,
  ROW_ID,
  SINGLE_USER_PROMPT_SHA256,
  TEMPLATE_SHA256,
  TEMPLATE_UTF8_BYTES,
  UNREACHED_DIFF_SECTIONS,
  assertExactTemplate,
  buildShapePack,
  classifyQwen3MoeTemplateQualificationError,
  expectedPromptForMessages,
  inspectCamelid,
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
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    const failure = classifyQwen3MoeTemplateQualificationError(error)
    console.error(`${failure.error_code}: ${failure.reason}`)
    process.exit(1)
  })
}
