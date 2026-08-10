#!/usr/bin/env node

import { createHash } from 'node:crypto'
import { execFile } from 'node:child_process'
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import { validateRoster } from './check-model-qualification-roster.mjs'
import {
  HeaderInspectionError,
  classifyHeaderInspectionError,
  fetchHeaderPrefix,
} from './hf-qualification-header.mjs'
import {
  resolveHfSource,
  validateLockAgainstSelection,
} from './hf-qualification-source.mjs'

const execFileAsync = promisify(execFile)
const DEFAULT_PREFIX_BYTES = 32 * 1024 * 1024
const PINNED_LLAMA_REVISION = 'acd79d603'
const PINNED_LLAMA_TOKENIZE_SHA256 = 'a44a4d7e1445d22a4cffb0d38f6efa8f1d81e84ae2c3d481af857c5e331b8c7a'
const PINNED_LLAMA_CLI_SHA256 = '2ec09da0b81d0201ce5b21810caefb4e77fd108f383b30c15ca493c5a70f7731'
const GEMMA2_TEMPLATE_SHA256 = 'ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6'
const SMOLLM3_TEMPLATE_SHA256 = 'b9b66f04c64fbb8695cf5b35c37780efd0b8e0829fbfe3e30fafb9f469b7d30e'
const QWEN3_MOE_TEMPLATE_SHA256 = '57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361'
const TOKENIZER_SCOPE_NOTE = 'the prefix hash includes opaque initial tensor payload bytes after data_start_offset; it is not a full payload or full artifact hash'

const TOKENIZER_ERROR_CONTRACTS = Object.freeze({
  tokenizer_pack_unavailable: ['blocked', 'no bounded tokenizer pack is defined for this exact row'],
  tokenizer_prefix_budget_invalid: ['fail', 'bounded tokenizer qualification requires the exact pack byte budget'],
  tokenizer_source_identity_mismatch: ['fail', 'bounded tokenizer source identity does not match the roster row'],
  tokenizer_inspector_unavailable: ['blocked', 'the Camelid tokenizer inspector is unavailable'],
  tokenizer_inspector_not_clean_current_head: ['blocked', 'the Camelid tokenizer inspector is not a clean build of current source HEAD'],
  tokenizer_inspector_changed: ['blocked', 'the Camelid tokenizer inspector changed during qualification'],
  tokenizer_oracle_unavailable: ['blocked', 'the pinned llama.cpp tokenizer oracle is unavailable'],
  tokenizer_oracle_identity_mismatch: ['fail', 'the llama.cpp tokenizer oracle does not match the pinned package'],
  tokenizer_oracle_changed: ['blocked', 'the pinned llama.cpp tokenizer oracle changed during qualification'],
  tokenizer_range_unavailable: ['blocked', 'the bounded tokenizer range fetch is unavailable'],
  tokenizer_range_invalid: ['fail', 'the bounded tokenizer range response is invalid'],
  tokenizer_prefix_identity_mismatch: ['fail', 'the bounded tokenizer prefix does not match the grounded exact-row hash'],
  tokenizer_metadata_mismatch: ['fail', 'the bounded tokenizer metadata does not match the exact-row pack'],
  tokenizer_probe_failed: ['fail', 'an exact-row tokenizer probe failed to produce a valid result'],
  tokenizer_source_changed: ['blocked', 'source HEAD or tracked state changed during tokenizer qualification'],
  tokenizer_receipt_time_invalid: ['fail', 'the tokenizer receipt time could not be recorded'],
  tokenizer_cleanup_failed: ['blocked', 'temporary tokenizer qualification files could not be removed'],
  tokenizer_qualification_error: ['blocked', 'bounded tokenizer qualification could not complete'],
})
const FALLBACK_TOKENIZER_ERROR = TOKENIZER_ERROR_CONTRACTS.tokenizer_qualification_error
const TOKENIZER_ERROR_CODES = new WeakMap()

class TokenizerQualificationError extends Error {
  constructor(code, _status, _message, _details = {}) {
    const knownCode = typeof code === 'string' && Object.hasOwn(TOKENIZER_ERROR_CONTRACTS, code)
    const contract = knownCode ? TOKENIZER_ERROR_CONTRACTS[code] : FALLBACK_TOKENIZER_ERROR
    super(contract[1])
    this.name = 'TokenizerQualificationError'
    this.code = knownCode ? code : 'tokenizer_qualification_error'
    this.status = contract[0]
    TOKENIZER_ERROR_CODES.set(this, this.code)
  }
}

function tokenizerError(code) {
  return new TokenizerQualificationError(code)
}

function classifyTokenizerQualificationError(error) {
  if (error instanceof TokenizerQualificationError) {
    const canonicalCode = TOKENIZER_ERROR_CODES.get(error)
    const knownCode = typeof canonicalCode === 'string'
      && Object.hasOwn(TOKENIZER_ERROR_CONTRACTS, canonicalCode)
    const code = knownCode ? canonicalCode : 'tokenizer_qualification_error'
    const contract = knownCode ? TOKENIZER_ERROR_CONTRACTS[code] : FALLBACK_TOKENIZER_ERROR
    return { status: contract[0], error_code: code, reason: contract[1] }
  }
  if (error instanceof HeaderInspectionError) {
    const failure = classifyHeaderInspectionError(error)
    const code = failure.status === 'fail'
      ? 'tokenizer_range_invalid'
      : 'tokenizer_range_unavailable'
    const contract = TOKENIZER_ERROR_CONTRACTS[code]
    return { status: contract[0], error_code: code, reason: contract[1] }
  }
  return {
    status: FALLBACK_TOKENIZER_ERROR[0],
    error_code: 'tokenizer_qualification_error',
    reason: FALLBACK_TOKENIZER_ERROR[1],
  }
}

const GEMMA2_CASES = [
  {
    id: 'plain_ascii_with_bos',
    text: 'Hello',
    add_special: true,
    parse_special: false,
  },
  {
    id: 'unicode_and_leading_spaces_with_bos',
    text: '  naïve café 中文 123',
    add_special: true,
    parse_special: false,
  },
  {
    id: 'punctuation_and_newline_with_bos',
    text: 'Line one.\nLine two!?',
    add_special: true,
    parse_special: false,
  },
  {
    id: 'plain_ascii_without_bos',
    text: 'Hello',
    add_special: false,
    parse_special: false,
  },
  {
    id: 'single_user_chat_controls',
    text: '<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n',
    add_special: true,
    parse_special: true,
  },
  {
    id: 'multi_turn_chat_controls',
    text: '<start_of_turn>user\nFirst<end_of_turn>\n<start_of_turn>model\nOne<end_of_turn>\n<start_of_turn>user\nSecond<end_of_turn>\n<start_of_turn>model\n',
    add_special: true,
    parse_special: true,
  },
  {
    id: 'chat_markers_as_ordinary_text',
    text: '<start_of_turn>user\nHello<end_of_turn>\n',
    add_special: true,
    parse_special: false,
  },
]

const SMOLLM3_CASES = [
  {
    id: 'empty_with_add_special',
    text: '',
    add_special: true,
    parse_special: false,
  },
  {
    id: 'plain_ascii_with_add_special',
    text: 'Hello',
    add_special: true,
    parse_special: false,
  },
  {
    id: 'plain_ascii_without_add_special',
    text: 'Hello',
    add_special: false,
    parse_special: false,
  },
  {
    id: 'smaug_unicode_spacing_and_contractions',
    text: "  na\u00efve caf\u00e9 \u4e2d\u6587\uff11\uff12\uff13 can't I'M we'd",
    add_special: true,
    parse_special: false,
  },
  {
    id: 'smaug_digits_punctuation_newlines_and_case',
    text: "x  y 1234 a/b\nCAN'T we're!!!",
    add_special: true,
    parse_special: false,
  },
  {
    id: 'single_user_chat_controls',
    text: '<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n',
    add_special: true,
    parse_special: true,
  },
  {
    id: 'multi_turn_chat_controls',
    text: '<|im_start|>user\nFirst<|im_end|>\n<|im_start|>assistant\nOne<|im_end|>\n<|im_start|>user\nSecond<|im_end|>\n<|im_start|>assistant\n',
    add_special: true,
    parse_special: true,
  },
  {
    id: 'chat_controls_as_ordinary_text',
    text: '<|im_start|>user\nHello<|im_end|>\n',
    add_special: true,
    parse_special: false,
  },
  {
    id: 'user_defined_tool_tags_with_parse_special',
    text: '<tool_call>{"name":"weather"}</tool_call>',
    add_special: true,
    parse_special: true,
  },
  {
    id: 'user_defined_tool_tags_without_parse_special',
    text: '<tool_call>{"name":"weather"}</tool_call>',
    add_special: true,
    parse_special: false,
  },
]

const QWEN3_MOE_CASES = [
  {
    id: 'empty_with_add_special',
    text: '',
    add_special: true,
    parse_special: false,
    expected_ids: [],
  },
  {
    id: 'plain_ascii_with_add_special',
    text: 'Hello',
    add_special: true,
    parse_special: false,
    expected_ids: [9_707],
  },
  {
    id: 'plain_ascii_without_add_special',
    text: 'Hello',
    add_special: false,
    parse_special: false,
    expected_ids: [9_707],
  },
  {
    id: 'qwen2_unicode_spacing_and_contractions',
    text: "  na\u00efve caf\u00e9 \u4e2d\u6587\uff11\uff12\uff13 can't I'M we'd",
    add_special: true,
    parse_special: false,
    expected_ids: [
      220, 94_880, 586, 51_950, 72_858, 16_744, 20_109, 24_918,
      33_517, 646, 944, 358, 27_603, 582, 4_172,
    ],
  },
  {
    id: 'qwen2_digits_punctuation_newlines_and_case',
    text: "x  y 1234 a/b\nCAN'T we're!!!",
    add_special: true,
    parse_special: false,
    expected_ids: [
      87, 220, 379, 220, 16, 17, 18, 19, 264, 3_470, 198,
      41_955, 17_323, 582, 2_299, 12_069,
    ],
  },
  {
    id: 'single_user_chat_controls',
    text: '<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n',
    add_special: true,
    parse_special: true,
    expected_ids: [151_644, 872, 198, 9_707, 151_645, 198, 151_644, 77_091, 198],
  },
  {
    id: 'multi_turn_chat_controls',
    text: '<|im_start|>user\nFirst<|im_end|>\n<|im_start|>assistant\nOne<|im_end|>\n<|im_start|>user\nSecond<|im_end|>\n<|im_start|>assistant\n',
    add_special: true,
    parse_special: true,
    expected_ids: [
      151_644, 872, 198, 5_338, 151_645, 198, 151_644, 77_091, 198,
      3_966, 151_645, 198, 151_644, 872, 198, 15_666, 151_645, 198,
      151_644, 77_091, 198,
    ],
  },
  {
    id: 'chat_controls_as_ordinary_text',
    text: '<|im_start|>user\nHello<|im_end|>\n',
    add_special: true,
    parse_special: false,
    expected_ids: [
      27, 91, 318, 4_906, 91, 29, 872, 198, 9_707,
      27, 91, 318, 6_213, 91, 397,
    ],
  },
  {
    id: 'user_defined_tool_tags_with_parse_special',
    text: '<tool_call>{"name":"weather"}</tool_call>',
    add_special: true,
    parse_special: true,
    expected_ids: [151_657, 4_913, 606, 3_252, 15_206, 9_207, 151_658],
  },
  {
    id: 'user_defined_tool_tags_without_parse_special',
    text: '<tool_call>{"name":"weather"}</tool_call>',
    add_special: true,
    parse_special: false,
    expected_ids: [151_657, 4_913, 606, 3_252, 15_206, 9_207, 151_658],
  },
  {
    id: 'user_defined_think_tags_with_parse_special',
    text: '<think>reason</think>',
    add_special: true,
    parse_special: true,
    expected_ids: [151_667, 19_895, 151_668],
  },
  {
    id: 'unused_pad_with_parse_special',
    text: '[PAD151669]',
    add_special: true,
    parse_special: true,
    expected_ids: [42_347, 1_808, 16, 20, 16, 21, 21, 24, 60],
  },
  {
    id: 'unused_pad_without_parse_special',
    text: '[PAD151669]',
    add_special: true,
    parse_special: false,
    expected_ids: [42_347, 1_808, 16, 20, 16, 21, 21, 24, 60],
  },
]

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

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex')
}

function exactObjectKeys(value, expectedKeys) {
  return value !== null
    && typeof value === 'object'
    && !Array.isArray(value)
    && JSON.stringify(Object.keys(value).sort()) === JSON.stringify([...expectedKeys].sort())
}

function tokenizerDoesNotProve(pack) {
  return [
    'full artifact integrity or presence on this host',
    'weight load, logits, generation, or greedy-token parity',
    'API, SSE, Models page, WebUI, or context readiness',
    `sampling, tools, GPU execution, performance, neighboring rows, or broad ${pack.family} support`,
  ]
}

function makeVocabOnlyGguf(prefix) {
  if (!Buffer.isBuffer(prefix) || prefix.length < 24) {
    throw new Error('GGUF prefix is too short to contain the fixed header')
  }
  if (!prefix.subarray(0, 4).equals(Buffer.from('GGUF'))) {
    throw new Error('GGUF prefix has invalid magic')
  }
  const version = prefix.readUInt32LE(4)
  if (version !== 2 && version !== 3) {
    throw new Error(`GGUF prefix has unsupported version ${version}`)
  }
  const tensorCount = prefix.readBigInt64LE(8)
  const metadataCount = prefix.readBigInt64LE(16)
  if (tensorCount <= 0n || metadataCount <= 0n) {
    throw new Error('GGUF prefix must declare positive tensor and metadata counts')
  }

  // llama-tokenize loads only vocabulary metadata, but its ordinary GGUF reader
  // still validates the physical file size. A disposable copy with tensor_count
  // set to zero lets the pinned reference consume the UNMODIFIED metadata bytes
  // without extending a 32 MiB prefix into a multi-gigabyte sparse model. This
  // derivative is never treated as the source artifact and is always deleted.
  const derived = Buffer.from(prefix)
  derived.writeBigInt64LE(0n, 8)
  return {
    bytes: derived,
    original_tensor_count: Number(tensorCount),
    metadata_count: Number(metadataCount),
    patched_offset: 8,
  }
}

function parseLlamaIds(stdout) {
  const line = String(stdout)
    .split(/\r?\n/)
    .map((candidate) => candidate.trim())
    .filter(Boolean)
    .findLast((candidate) => candidate.startsWith('[') && candidate.endsWith(']'))
  if (!line) throw new Error('llama-tokenize did not emit an ID array')
  const ids = JSON.parse(line)
  if (!Array.isArray(ids) || ids.some((id) => !Number.isSafeInteger(id) || id < 0)) {
    throw new Error('llama-tokenize emitted an invalid ID array')
  }
  return ids
}

function parseLlamaVersionOutput(output, expectedRevision = PINNED_LLAMA_REVISION) {
  const match = /version:\s*(\d+)\s*\(([0-9a-f]{7,40})\)/i.exec(String(output))
  if (!match) throw new Error('llama.cpp companion did not emit a parseable build revision')
  const revision = match[2].toLowerCase()
  if (revision !== expectedRevision.toLowerCase()) {
    throw new Error(`llama.cpp companion revision ${revision} does not match pin ${expectedRevision}`)
  }
  return { build: Number(match[1]), revision }
}

function classifyCamelidProvenance({ version, sourceHead, sourceTrackedDirty }) {
  const match = /^camelid [A-Za-z0-9._+()-]+-g([0-9a-f]{7,40})(-dirty)?$/i.exec(String(version))
  const binaryCommit = match?.[1]?.toLowerCase() || null
  const binaryReportsDirty = Boolean(match?.[2])
  const normalizedHead = String(sourceHead || '').trim().toLowerCase()
  const binaryMatchesSourceHead = Boolean(
    binaryCommit
      && /^[0-9a-f]{40}$/.test(normalizedHead)
      && normalizedHead.startsWith(binaryCommit),
  )
  const cleanCurrentHead = binaryMatchesSourceHead
    && !binaryReportsDirty
    && sourceTrackedDirty === false
  return {
    status: cleanCurrentHead
      ? 'clean_current_head_receipt'
      : 'preparatory_requires_clean_current_head_rerun',
    gate_requires_clean_current_head: true,
    source_head: normalizedHead || null,
    source_tracked_dirty: sourceTrackedDirty,
    binary_commit_abbrev: binaryCommit,
    binary_reports_dirty: binaryReportsDirty,
    binary_matches_source_head: binaryMatchesSourceHead,
    clean_current_head: cleanCurrentHead,
  }
}

async function readSourceProvenance(root) {
  const [{ stdout: head }, { stdout: status }] = await Promise.all([
    execFileAsync('git', ['rev-parse', 'HEAD'], { cwd: root, timeout: 10_000, windowsHide: true }),
    execFileAsync('git', ['status', '--porcelain', '--untracked-files=no'], {
      cwd: root,
      timeout: 10_000,
      windowsHide: true,
    }),
  ])
  return {
    sourceHead: head.trim(),
    sourceTrackedDirty: Boolean(status.trim()),
  }
}

async function verifyLlamaCppPackage(llamaTokenize, {
  execImpl = execFileAsync,
  readFileImpl = readFile,
  platform = process.platform,
} = {}) {
  const companion = join(
    dirname(llamaTokenize),
    platform === 'win32' ? 'llama-cli.exe' : 'llama-cli',
  )
  let output = ''
  try {
    const result = await execImpl(companion, ['--version'], {
      timeout: 10_000,
      windowsHide: true,
    })
    output = `${result.stdout || ''}\n${result.stderr || ''}`
  } catch (error) {
    // The pinned Windows b9632 launcher reports its version and exits 1. Accept
    // that historical quirk only when the output still carries the exact pin.
    const knownWindowsExitOne = platform === 'win32'
      && error?.code === 1
      && error?.killed !== true
      && (error?.signal === null || error?.signal === undefined)
    if (!knownWindowsExitOne) throw error
    output = `${error.stdout || ''}\n${error.stderr || ''}`
    if (!output.trim()) throw error
  }
  const version = parseLlamaVersionOutput(output)
  return {
    ...version,
    companion_executable: basename(companion),
    companion_binary_sha256: sha256(await readFileImpl(companion)),
    executable: basename(llamaTokenize),
    binary_sha256: sha256(await readFileImpl(llamaTokenize)),
  }
}

async function inspectCamelidTokenizerIdentity(binary, {
  sourceProvenance,
  execImpl = execFileAsync,
  readFileImpl = readFile,
} = {}) {
  if (typeof binary !== 'string' || !binary.trim()) {
    throw tokenizerError('tokenizer_inspector_unavailable')
  }
  let version
  let binarySha256
  try {
    const result = await execImpl(binary, ['--version'], {
      timeout: 10_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
    })
    version = String(result.stdout || '').trim()
    binarySha256 = sha256(await readFileImpl(binary))
  } catch {
    throw tokenizerError('tokenizer_inspector_unavailable')
  }
  const provenance = classifyCamelidProvenance({
    version,
    ...sourceProvenance,
  })
  if (!provenance.clean_current_head) {
    throw tokenizerError('tokenizer_inspector_not_clean_current_head')
  }
  return { version, binary_sha256: binarySha256, provenance }
}

function buildCamelidArgs({ prefixPath, declaredLength, inputPath, addSpecial, parseSpecial }) {
  const args = [
    'tokenize',
    '--model', prefixPath,
    '--declared-len', String(declaredLength),
    '--file', inputPath,
  ]
  if (parseSpecial) args.push('--parse-special')
  if (!addSpecial) args.push('--no-add-special')
  return args
}

function buildLlamaArgs({ modelPath, promptPath, addSpecial, parseSpecial }) {
  const args = [
    '--model', modelPath,
    '--file', promptPath,
    '--ids',
    '--no-escape',
    '--log-disable',
  ]
  if (!addSpecial) args.push('--no-bos')
  if (!parseSpecial) args.push('--no-parse-special')
  return args
}

function assertGemma2TokenizerMetadata(inspection) {
  const metadata = inspection?.metadata
  if (!metadata || typeof metadata !== 'object') {
    throw new Error('prefix inspection did not return metadata')
  }
  const exact = [
    ['general.architecture', 'gemma2'],
    ['tokenizer.ggml.model', 'llama'],
    ['tokenizer.ggml.pre', 'default'],
    ['tokenizer.ggml.bos_token_id', 2],
    ['tokenizer.ggml.eos_token_id', 1],
    ['tokenizer.ggml.unknown_token_id', 3],
    ['tokenizer.ggml.add_bos_token', true],
    ['tokenizer.ggml.add_eos_token', false],
    ['tokenizer.ggml.add_space_prefix', false],
  ]
  for (const [key, expected] of exact) {
    if (metadata[key] !== expected) {
      throw new Error(`Gemma2 tokenizer metadata ${key} mismatch`)
    }
  }
  const tokens = metadata['tokenizer.ggml.tokens']
  const scores = metadata['tokenizer.ggml.scores']
  const types = metadata['tokenizer.ggml.token_type']
  if (!Array.isArray(tokens) || tokens.length !== 256_000) {
    throw new Error('Gemma2 tokenizer token array is missing or not 256000 entries')
  }
  if (!Array.isArray(scores) || scores.length !== tokens.length) {
    throw new Error('Gemma2 tokenizer score array is missing or length-mismatched')
  }
  if (!Array.isArray(types) || types.length !== tokens.length) {
    throw new Error('Gemma2 tokenizer type array is missing or length-mismatched')
  }
  const template = metadata['tokenizer.chat_template']
  if (typeof template !== 'string' || sha256(Buffer.from(template)) !== GEMMA2_TEMPLATE_SHA256) {
    throw new Error('Gemma2 tokenizer chat template does not match the pinned exact-row hash')
  }
  return {
    token_count: tokens.length,
    score_count: scores.length,
    token_type_count: types.length,
    chat_template_utf8_bytes: Buffer.byteLength(template),
    chat_template_sha256: sha256(Buffer.from(template)),
  }
}

function assertSmolLM3TokenizerMetadata(inspection) {
  const metadata = inspection?.metadata
  if (!metadata || typeof metadata !== 'object') {
    throw new Error('prefix inspection did not return metadata')
  }
  const exact = [
    ['general.architecture', 'smollm3'],
    ['general.license', 'apache-2.0'],
    ['smollm3.vocab_size', 128_256],
    ['tokenizer.ggml.model', 'gpt2'],
    ['tokenizer.ggml.pre', 'smaug-bpe'],
    ['tokenizer.ggml.bos_token_id', 128_000],
    ['tokenizer.ggml.eos_token_id', 128_012],
    ['tokenizer.ggml.padding_token_id', 128_012],
  ]
  for (const [key, expected] of exact) {
    if (metadata[key] !== expected) {
      throw new Error(`SmolLM3 tokenizer metadata ${key} mismatch`)
    }
  }
  for (const key of [
    'tokenizer.ggml.add_bos_token',
    'tokenizer.ggml.add_eos_token',
    'tokenizer.ggml.add_space_prefix',
  ]) {
    if (Object.hasOwn(metadata, key)) {
      throw new Error(`SmolLM3 exact row unexpectedly declares ${key}`)
    }
  }

  const tokens = metadata['tokenizer.ggml.tokens']
  const merges = metadata['tokenizer.ggml.merges']
  const types = metadata['tokenizer.ggml.token_type']
  if (!Array.isArray(tokens) || tokens.length !== 128_256) {
    throw new Error('SmolLM3 tokenizer token array is missing or not 128256 entries')
  }
  if (!Array.isArray(merges) || merges.length !== 280_147) {
    throw new Error('SmolLM3 tokenizer merge array is missing or not 280147 entries')
  }
  if (!Array.isArray(types) || types.length !== tokens.length) {
    throw new Error('SmolLM3 tokenizer type array is missing or length-mismatched')
  }
  if (!types.every((type) => Number.isSafeInteger(type))) {
    throw new Error('SmolLM3 tokenizer type array contains a non-integer entry')
  }

  const tokenTypeCounts = Object.create(null)
  for (const type of types) tokenTypeCounts[type] = (tokenTypeCounts[type] || 0) + 1
  const normalTokenCount = tokenTypeCounts[1] || 0
  const controlTokenCount = tokenTypeCounts[3] || 0
  const userDefinedTokenCount = tokenTypeCounts[4] || 0
  const specialTokenCount = tokens.length - normalTokenCount
  if (normalTokenCount !== 128_000
    || controlTokenCount !== 248
    || userDefinedTokenCount !== 8
    || specialTokenCount !== 256
    || Object.keys(tokenTypeCounts).some((type) => !['1', '3', '4'].includes(type))) {
    throw new Error('SmolLM3 tokenizer special-token type counts mismatch')
  }

  for (const [id, expected, expectedType] of [
    [128_000, '<|begin_of_text|>', 3],
    [128_001, '<|end_of_text|>', 3],
    [128_002, '<think>', 4],
    [128_003, '</think>', 4],
    [128_006, '<|start_header_id|>', 3],
    [128_007, '<|end_header_id|>', 3],
    [128_008, '<|eom_id|>', 3],
    [128_009, '<|eot_id|>', 3],
    [128_010, '<|python_tag|>', 3],
    [128_011, '<|im_start|>', 3],
    [128_012, '<|im_end|>', 3],
    [128_013, '<tool_response>', 4],
    [128_014, '</tool_response>', 4],
    [128_015, '<tool_call>', 4],
    [128_016, '</tool_call>', 4],
    [128_017, '<code>', 4],
    [128_018, '</code>', 4],
  ]) {
    if (tokens[id] !== expected || types[id] !== expectedType) {
      throw new Error(`SmolLM3 tokenizer special token ${id} mismatch`)
    }
  }

  const template = metadata['tokenizer.chat_template']
  if (typeof template !== 'string'
    || Buffer.byteLength(template) !== 5_493
    || sha256(Buffer.from(template)) !== SMOLLM3_TEMPLATE_SHA256) {
    throw new Error('SmolLM3 tokenizer chat template does not match the pinned exact-row hash')
  }
  return {
    token_count: tokens.length,
    merge_count: merges.length,
    token_type_count: types.length,
    normal_token_count: normalTokenCount,
    special_token_count: specialTokenCount,
    control_token_count: controlTokenCount,
    user_defined_token_count: userDefinedTokenCount,
    bos_token_id: 128_000,
    eos_token_id: 128_012,
    padding_token_id: 128_012,
    declared_add_bos_token: 'absent',
    declared_add_eos_token: 'absent',
    declared_add_space_prefix: 'absent',
    oracle_resolved_add_bos_token: false,
    oracle_resolved_add_eos_token: false,
    chat_control_token_ids: {
      im_start: 128_011,
      im_end: 128_012,
    },
    chat_template_utf8_bytes: Buffer.byteLength(template),
    chat_template_sha256: sha256(Buffer.from(template)),
  }
}

function assertQwen3MoeTokenizerMetadata(inspection) {
  const metadata = inspection?.metadata
  if (!metadata || typeof metadata !== 'object') {
    throw new Error('prefix inspection did not return metadata')
  }
  const exact = [
    ['general.architecture', 'qwen3moe'],
    ['tokenizer.ggml.model', 'gpt2'],
    ['tokenizer.ggml.pre', 'qwen2'],
    ['tokenizer.ggml.bos_token_id', 151_643],
    ['tokenizer.ggml.eos_token_id', 151_645],
    ['tokenizer.ggml.padding_token_id', 151_643],
    ['tokenizer.ggml.add_bos_token', false],
  ]
  for (const [key, expected] of exact) {
    if (metadata[key] !== expected) {
      throw new Error(`Qwen3 MoE tokenizer metadata ${key} mismatch`)
    }
  }
  for (const key of [
    'tokenizer.ggml.add_eos_token',
    'tokenizer.ggml.add_space_prefix',
  ]) {
    if (Object.hasOwn(metadata, key)) {
      throw new Error(`Qwen3 MoE exact row unexpectedly declares ${key}`)
    }
  }

  const tokens = metadata['tokenizer.ggml.tokens']
  const merges = metadata['tokenizer.ggml.merges']
  const types = metadata['tokenizer.ggml.token_type']
  if (!Array.isArray(tokens) || tokens.length !== 151_936) {
    throw new Error('Qwen3 MoE tokenizer token array is missing or not 151936 entries')
  }
  if (!Array.isArray(merges) || merges.length !== 151_387) {
    throw new Error('Qwen3 MoE tokenizer merge array is missing or not 151387 entries')
  }
  if (!Array.isArray(types) || types.length !== tokens.length) {
    throw new Error('Qwen3 MoE tokenizer type array is missing or length-mismatched')
  }
  if (!types.every((type) => Number.isSafeInteger(type))) {
    throw new Error('Qwen3 MoE tokenizer type array contains a non-integer entry')
  }

  const tokenTypeCounts = Object.create(null)
  for (const type of types) tokenTypeCounts[type] = (tokenTypeCounts[type] || 0) + 1
  const normalTokenCount = tokenTypeCounts[1] || 0
  const controlTokenCount = tokenTypeCounts[3] || 0
  const userDefinedTokenCount = tokenTypeCounts[4] || 0
  const unusedTokenCount = tokenTypeCounts[5] || 0
  const specialTokenCount = tokens.length - normalTokenCount
  if (normalTokenCount !== 151_643
    || controlTokenCount !== 20
    || userDefinedTokenCount !== 6
    || unusedTokenCount !== 267
    || specialTokenCount !== 293
    || Object.keys(tokenTypeCounts).some((type) => !['1', '3', '4', '5'].includes(type))) {
    throw new Error('Qwen3 MoE tokenizer special-token type counts mismatch')
  }

  for (const [id, expected, expectedType] of [
    [151_643, '<|endoftext|>', 3],
    [151_644, '<|im_start|>', 3],
    [151_645, '<|im_end|>', 3],
    [151_657, '<tool_call>', 4],
    [151_658, '</tool_call>', 4],
    [151_665, '<tool_response>', 4],
    [151_666, '</tool_response>', 4],
    [151_667, '<think>', 4],
    [151_668, '</think>', 4],
  ]) {
    if (tokens[id] !== expected || types[id] !== expectedType) {
      throw new Error(`Qwen3 MoE tokenizer special token ${id} mismatch`)
    }
  }
  for (let id = 151_669; id < 151_936; id += 1) {
    if (tokens[id] !== `[PAD${id}]` || types[id] !== 5) {
      throw new Error(`Qwen3 MoE tokenizer unused token ${id} mismatch`)
    }
  }

  const template = metadata['tokenizer.chat_template']
  if (typeof template !== 'string'
    || Buffer.byteLength(template) !== 4_100
    || sha256(Buffer.from(template)) !== QWEN3_MOE_TEMPLATE_SHA256) {
    throw new Error('Qwen3 MoE tokenizer chat template does not match the pinned exact-row hash')
  }
  return {
    token_count: tokens.length,
    merge_count: merges.length,
    token_type_count: types.length,
    normal_token_count: normalTokenCount,
    special_token_count: specialTokenCount,
    control_token_count: controlTokenCount,
    user_defined_token_count: userDefinedTokenCount,
    unused_token_count: unusedTokenCount,
    bos_token_id: 151_643,
    eos_token_id: 151_645,
    padding_token_id: 151_643,
    declared_add_bos_token: false,
    declared_add_eos_token: 'absent',
    declared_add_space_prefix: 'absent',
    oracle_resolved_add_bos_token: false,
    oracle_resolved_add_eos_token: false,
    special_token_ids: {
      endoftext: 151_643,
      im_start: 151_644,
      im_end: 151_645,
      tool_call_start: 151_657,
      tool_call_end: 151_658,
      tool_response_start: 151_665,
      tool_response_end: 151_666,
      think_start: 151_667,
      think_end: 151_668,
      unused_first: 151_669,
      unused_last: 151_935,
    },
    chat_template_utf8_bytes: Buffer.byteLength(template),
    chat_template_sha256: sha256(Buffer.from(template)),
  }
}

const TOKENIZER_PACKS = Object.freeze({
  gemma2_9b_it_q8_0: Object.freeze({
    family: 'Gemma',
    cases: GEMMA2_CASES,
    assertMetadata: assertGemma2TokenizerMetadata,
    tensorCount: 464,
    metadataCount: 26,
    prefixBytes: DEFAULT_PREFIX_BYTES,
    prefixSha256: 'b2bcc601c188ffc7c306f0011944a7a5492bfde490c34ddc390b69424c09a5e5',
    supportDecision: 'no_change_header_tokenizer_evidence_only',
    requireReceiptLicense: false,
    grounding: null,
    metadataSummary: Object.freeze({
      token_count: 256_000,
      score_count: 256_000,
      token_type_count: 256_000,
      chat_template_utf8_bytes: 591,
      chat_template_sha256: GEMMA2_TEMPLATE_SHA256,
    }),
  }),
  smollm3_3b_q8_0: Object.freeze({
    family: 'SmolLM3',
    cases: SMOLLM3_CASES,
    assertMetadata: assertSmolLM3TokenizerMetadata,
    tensorCount: 326,
    metadataCount: 26,
    prefixBytes: DEFAULT_PREFIX_BYTES,
    prefixSha256: '2d043b2114b89100c7ba464e57375a6f32c06c04729542d54ed684b5e8c5016e',
    supportDecision: 'smollm3_exact_row_tokenizer_gate_only',
    requireReceiptLicense: true,
    grounding: Object.freeze({
      header_receipt: 'qa/model-qualification/smollm3-3b-q8-header-inspection.json',
      tokenizer_pre_fixture: 'qa/model-qualification/fixtures/smollm3-tokenizer-pre-v1.json',
    }),
    metadataSummary: Object.freeze({
      token_count: 128_256,
      merge_count: 280_147,
      token_type_count: 128_256,
      normal_token_count: 128_000,
      special_token_count: 256,
      control_token_count: 248,
      user_defined_token_count: 8,
      bos_token_id: 128_000,
      eos_token_id: 128_012,
      padding_token_id: 128_012,
      declared_add_bos_token: 'absent',
      declared_add_eos_token: 'absent',
      declared_add_space_prefix: 'absent',
      oracle_resolved_add_bos_token: false,
      oracle_resolved_add_eos_token: false,
      chat_control_token_ids: Object.freeze({
        im_start: 128_011,
        im_end: 128_012,
      }),
      chat_template_utf8_bytes: 5_493,
      chat_template_sha256: SMOLLM3_TEMPLATE_SHA256,
    }),
  }),
  qwen3_30b_a3b_q8_0: Object.freeze({
    family: 'Qwen3 MoE',
    cases: QWEN3_MOE_CASES,
    assertMetadata: assertQwen3MoeTokenizerMetadata,
    tensorCount: 579,
    metadataCount: 31,
    prefixBytes: DEFAULT_PREFIX_BYTES,
    prefixSha256: '55c565264523c5862247d983f857b9034c04d762ee14fecfd68a827cdbb2d566',
    derivativeSha256: '39c4ed3e1ec5dbf8b1582bef982b97436e2d83709cf02195255d7908c595a54d',
    supportDecision: 'qwen3_moe_exact_row_tokenizer_gate_only',
    requireReceiptLicense: true,
    grounding: Object.freeze({
      header_receipt: 'qa/model-qualification/qwen3-30b-a3b-q8-header-inspection.json',
      header_receipt_sha256: '293f8dd99f4f31478a0a6a7b3fc9c3e6a1c224a9df0b1dc3253e619d93a2dc33',
    }),
    metadataSummary: Object.freeze({
      token_count: 151_936,
      merge_count: 151_387,
      token_type_count: 151_936,
      normal_token_count: 151_643,
      special_token_count: 293,
      control_token_count: 20,
      user_defined_token_count: 6,
      unused_token_count: 267,
      bos_token_id: 151_643,
      eos_token_id: 151_645,
      padding_token_id: 151_643,
      declared_add_bos_token: false,
      declared_add_eos_token: 'absent',
      declared_add_space_prefix: 'absent',
      oracle_resolved_add_bos_token: false,
      oracle_resolved_add_eos_token: false,
      special_token_ids: Object.freeze({
        endoftext: 151_643,
        im_start: 151_644,
        im_end: 151_645,
        tool_call_start: 151_657,
        tool_call_end: 151_658,
        tool_response_start: 151_665,
        tool_response_end: 151_666,
        think_start: 151_667,
        think_end: 151_668,
        unused_first: 151_669,
        unused_last: 151_935,
      }),
      chat_template_utf8_bytes: 4_100,
      chat_template_sha256: QWEN3_MOE_TEMPLATE_SHA256,
    }),
  }),
})

function tokenizerPackForRow(rowId) {
  const pack = TOKENIZER_PACKS[rowId]
  if (!pack) {
    throw new Error(`bounded tokenizer pack is not defined for row ${JSON.stringify(rowId)}`)
  }
  return pack
}

function tokenizerPackAvailable(rowId) {
  return typeof rowId === 'string' && Object.hasOwn(TOKENIZER_PACKS, rowId)
}

function tokenizerPrefixBytesForRow(rowId) {
  return tokenizerPackAvailable(rowId) ? TOKENIZER_PACKS[rowId].prefixBytes : null
}

function normalizeTokenizerPrefixBytes(rowId, value = DEFAULT_PREFIX_BYTES) {
  const pack = tokenizerPackForRow(rowId)
  const prefixBytes = typeof value === 'number' ? value : Number(value)
  if (!Number.isSafeInteger(prefixBytes) || prefixBytes !== pack.prefixBytes) {
    throw new Error(`${rowId} tokenizer evidence requires exactly ${pack.prefixBytes} prefix bytes`)
  }
  return prefixBytes
}

async function selectRow(root, rosterPath, rowId) {
  const absolute = resolve(root, rosterPath)
  const roster = JSON.parse(await readFile(absolute, 'utf8'))
  const errors = validateRoster(roster, absolute)
  if (errors.length) throw new Error(`roster is invalid:\n${errors.join('\n')}`)
  const row = roster.rows.find((candidate) => candidate.id === rowId)
  if (!row) throw new Error(`unknown --row ${JSON.stringify(rowId)}`)
  tokenizerPackForRow(row.id)
  return row
}

function sourceSelectionForRow(row) {
  return {
    row_id: row.id,
    repo: row.source.repo,
    file: row.source.file,
    revision: row.source.revision,
    expected: {
      size_bytes: row.identity.size_bytes,
      sha256: row.identity.sha256,
      license: row.source.license,
    },
  }
}

function validIdArray(value, tokenCount) {
  return Array.isArray(value)
    && Number.isSafeInteger(tokenCount)
    && tokenCount > 0
    && value.every((id) => Number.isSafeInteger(id) && id >= 0 && id < tokenCount)
}

function assessTokenizerReceiptForPack(receipt, row, defaults, pack, {
  expectedSourceHead = null,
} = {}) {
  const errors = []
  const parityErrors = []
  const check = (condition, message) => { if (!condition) errors.push(message) }
  const parityCheck = (condition, message) => { if (!condition) parityErrors.push(message) }
  const shaRe = /^[0-9a-f]{64}$/
  const expectedTopLevelKeys = [
    'schema',
    'generated_at',
    'provenance',
    'row_id',
    'host',
    'source',
    'bounded_fetch',
    ...(pack.grounding ? ['grounding'] : []),
    'tokenizer_metadata',
    'camelid',
    'oracle',
    'cases',
    'result',
    'does_not_prove',
  ]
  check(row?.id && TOKENIZER_PACKS[row.id] === pack, 'row is not bound to this tokenizer pack')
  if (row?.id === 'qwen3_30b_a3b_q8_0') {
    check(exactObjectKeys(receipt, expectedTopLevelKeys), 'receipt top-level fields mismatch')
  }
  check(receipt?.schema === 'camelid.header-tokenizer-parity/v1', 'schema mismatch')
  check(receipt?.row_id === row.id, 'row_id mismatch')
  check(typeof receipt?.generated_at === 'string'
    && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(receipt.generated_at)
    && !Number.isNaN(Date.parse(receipt.generated_at)), 'generated_at is invalid')
  check(receipt?.host?.hostname_redacted === true, 'hostname is not redacted')
  check(!Object.hasOwn(receipt?.host || {}, 'hostname'), 'raw hostname is present')
  check(typeof receipt?.host?.platform === 'string'
    && /^[A-Za-z0-9_.:+-]{1,128}$/.test(receipt.host.platform), 'host platform is invalid')

  for (const [field, expected] of [
    ['repo', row.source.repo],
    ['file', row.source.file],
    ['revision', row.source.revision],
    ['size_bytes', row.identity.size_bytes],
    ['sha256', row.identity.sha256],
  ]) {
    check(receipt?.source?.[field] === expected, `source.${field} mismatch`)
  }
  if (pack.requireReceiptLicense) {
    check(receipt?.source?.license === row.source.license, 'source.license mismatch')
  } else if (Object.hasOwn(receipt?.source || {}, 'license')) {
    check(receipt.source.license === row.source.license, 'source.license mismatch')
  }

  const provenance = receipt?.provenance || {}
  const recomputedProvenance = classifyCamelidProvenance({
    version: receipt?.camelid?.version,
    sourceHead: provenance.source_head,
    sourceTrackedDirty: provenance.source_tracked_dirty,
  })
  check(provenance.status === 'clean_current_head_receipt', 'receipt is not clean-current-head evidence')
  check(provenance.gate_requires_clean_current_head === true, 'clean-current-head gate is not recorded')
  check(provenance.source_tracked_dirty === false, 'source was tracked-dirty')
  check(provenance.binary_reports_dirty === false, 'Camelid binary reports dirty')
  check(provenance.binary_matches_source_head === true, 'Camelid binary does not match source head')
  check(provenance.clean_current_head === true, 'clean_current_head is not true')
  check(expectedSourceHead === null
    || (/^[0-9a-f]{40}$/.test(expectedSourceHead)
      && provenance.source_head === expectedSourceHead), 'provenance source_head does not match expected source HEAD')
  for (const [field, expected] of Object.entries(recomputedProvenance)) {
    check(provenance[field] === expected, `provenance.${field} is not derivable from Camelid version and source head`)
  }
  check(shaRe.test(receipt?.camelid?.binary_sha256 || ''), 'Camelid binary SHA-256 is invalid')
  check(receipt?.camelid?.prefix_mode === 'tokenize --declared-len', 'Camelid prefix mode mismatch')

  const bounded = receipt?.bounded_fetch || {}
  const contentRange = bounded.content_range || {}
  check(bounded.requested_bytes === pack.prefixBytes, 'bounded request size does not match the exact pack')
  check(bounded.received_bytes === bounded.requested_bytes, 'bounded received byte count mismatch')
  check(contentRange.start === 0, 'Content-Range must start at zero')
  check(contentRange.end + 1 === bounded.received_bytes, 'Content-Range end does not match received bytes')
  check(contentRange.total === row.identity.size_bytes, 'Content-Range total does not match row size')
  check(bounded.prefix_sha256 === pack.prefixSha256, 'prefix SHA-256 does not match the grounded exact-row prefix')
  check(bounded.temporary_paths_redacted === true, 'temporary paths are not redacted')
  check(bounded.temporary_files_deleted === true, 'temporary files were not confirmed deleted')
  if (row?.id === 'qwen3_30b_a3b_q8_0') {
    check(bounded.scope_note === TOKENIZER_SCOPE_NOTE, 'bounded scope note mismatch')
  }
  if (pack.grounding) {
    for (const [field, expected] of Object.entries(pack.grounding)) {
      check(receipt?.grounding?.[field] === expected, `grounding ${field} mismatch`)
    }
  }

  const metadata = receipt?.tokenizer_metadata || {}
  for (const [field, expected] of Object.entries(pack.metadataSummary)) {
    const observed = metadata[field]
    const equal = expected && typeof expected === 'object'
      ? JSON.stringify(observed) === JSON.stringify(expected)
      : observed === expected
    check(equal, `tokenizer metadata ${field} mismatch`)
  }

  const expectedBuild = Number(String(defaults?.llama_cpp?.build || '').replace(/^b/, ''))
  check(receipt?.oracle?.project === 'ggml-org/llama.cpp', 'llama.cpp project mismatch')
  check(receipt?.oracle?.revision === defaults?.llama_cpp?.revision, 'llama.cpp revision mismatch')
  check(receipt?.oracle?.build === expectedBuild, 'llama.cpp build mismatch')
  check(receipt?.oracle?.binary_sha256 === PINNED_LLAMA_TOKENIZE_SHA256, 'llama-tokenize SHA-256 mismatch')
  check(receipt?.oracle?.companion_binary_sha256 === PINNED_LLAMA_CLI_SHA256, 'llama.cpp companion SHA-256 mismatch')
  check(receipt?.oracle?.derivative?.persisted === false, 'vocabulary-only derivative was persisted')
  check(receipt?.oracle?.derivative?.original_tensor_count === pack.tensorCount, 'derivative tensor count mismatch')
  check(receipt?.oracle?.derivative?.metadata_count === pack.metadataCount, 'derivative metadata count mismatch')
  check(receipt?.oracle?.derivative?.patch_offset === 8, 'derivative patch offset mismatch')
  if (pack.derivativeSha256) {
    check(receipt?.oracle?.derivative?.sha256 === pack.derivativeSha256,
      'derivative SHA-256 does not match the deterministic exact-row derivative')
  } else {
    check(shaRe.test(receipt?.oracle?.derivative?.sha256 || ''), 'derivative SHA-256 is invalid')
  }

  const tokenCount = pack.metadataSummary.token_count
  const cases = Array.isArray(receipt?.cases) ? receipt.cases : []
  check(cases.length === pack.cases.length, 'case count mismatch')
  let exactMatches = 0
  for (let index = 0; index < pack.cases.length; index += 1) {
    const expected = pack.cases[index]
    const observed = cases[index] || {}
    const idsValid = validIdArray(observed.camelid_ids, tokenCount)
      && validIdArray(observed.llama_cpp_ids, tokenCount)
    const idsMatch = idsValid
      && JSON.stringify(observed.camelid_ids) === JSON.stringify(observed.llama_cpp_ids)
    if (idsMatch) exactMatches += 1
    check(observed.id === expected.id, `case ${expected.id} id/order mismatch`)
    check(observed.text_utf8_bytes === Buffer.byteLength(expected.text), `case ${expected.id} byte count mismatch`)
    check(observed.text_sha256 === sha256(Buffer.from(expected.text)), `case ${expected.id} text SHA-256 mismatch`)
    check(observed.add_special === expected.add_special, `case ${expected.id} add_special mismatch`)
    check(observed.parse_special === expected.parse_special, `case ${expected.id} parse_special mismatch`)
    check(idsValid, `case ${expected.id} has invalid token IDs`)
    if (Array.isArray(expected.expected_ids)) {
      check(JSON.stringify(observed.camelid_ids) === JSON.stringify(expected.expected_ids),
        `case ${expected.id} Camelid IDs do not match the pinned exact array`)
      check(JSON.stringify(observed.llama_cpp_ids) === JSON.stringify(expected.expected_ids),
        `case ${expected.id} llama.cpp IDs do not match the pinned exact array`)
    }
    check(observed.exact_match === idsMatch, `case ${expected.id} exact_match is not derived from token IDs`)
    parityCheck(idsMatch, `case ${expected.id} token IDs diverge`)
    check(shaRe.test(observed.camelid_decoded_sha256 || ''), `case ${expected.id} decoded SHA-256 is invalid`)
  }
  check(receipt?.result?.case_count === cases.length, 'result case_count mismatch')
  check(receipt?.result?.exact_match_count === exactMatches, 'result exact_match_count mismatch')
  check(receipt?.result?.all_token_ids_match === (exactMatches === pack.cases.length), 'result all_token_ids_match mismatch')
  check(receipt?.result?.support_decision === pack.supportDecision, 'support decision widened unexpectedly')
  if (row?.id === 'qwen3_30b_a3b_q8_0') {
    check(exactObjectKeys(receipt?.result, [
      'case_count',
      'exact_match_count',
      'all_token_ids_match',
      'support_decision',
    ]), 'result fields mismatch')
    check(JSON.stringify(receipt?.does_not_prove) === JSON.stringify(tokenizerDoesNotProve(pack)),
      'does_not_prove exclusions mismatch')
  }

  if (row.id === 'smollm3_3b_q8_0') {
    const emptyWithSpecial = cases.find((testCase) => testCase?.id === 'empty_with_add_special')
    const helloWithSpecial = cases.find((testCase) => testCase?.id === 'plain_ascii_with_add_special')
    const helloWithoutSpecial = cases.find((testCase) => testCase?.id === 'plain_ascii_without_add_special')
    check(Boolean(emptyWithSpecial)
      && validIdArray(emptyWithSpecial.camelid_ids, tokenCount)
      && validIdArray(emptyWithSpecial.llama_cpp_ids, tokenCount)
      && emptyWithSpecial.camelid_ids.length === 0
      && emptyWithSpecial.llama_cpp_ids.length === 0,
    'SmolLM3 absent add_bos metadata did not resolve false for empty input')
    check(Boolean(helloWithSpecial && helloWithoutSpecial)
      && validIdArray(helloWithSpecial.camelid_ids, tokenCount)
      && validIdArray(helloWithSpecial.llama_cpp_ids, tokenCount)
      && validIdArray(helloWithoutSpecial.camelid_ids, tokenCount)
      && validIdArray(helloWithoutSpecial.llama_cpp_ids, tokenCount)
      && JSON.stringify(helloWithSpecial.camelid_ids) === '[9906]'
      && JSON.stringify(helloWithSpecial.llama_cpp_ids) === '[9906]'
      && JSON.stringify(helloWithoutSpecial.camelid_ids) === '[9906]'
      && JSON.stringify(helloWithoutSpecial.llama_cpp_ids) === '[9906]',
    'SmolLM3 absent add_bos/add_eos metadata did not preserve exact Hello token IDs')

    const parsedChat = cases.find((testCase) => testCase?.id === 'single_user_chat_controls')
    const ordinaryChat = cases.find((testCase) => testCase?.id === 'chat_controls_as_ordinary_text')
    check(Boolean(parsedChat)
      && validIdArray(parsedChat.camelid_ids, tokenCount)
      && validIdArray(parsedChat.llama_cpp_ids, tokenCount)
      && parsedChat.camelid_ids.includes(128_011)
      && parsedChat.camelid_ids.includes(128_012)
      && parsedChat.llama_cpp_ids.includes(128_011)
      && parsedChat.llama_cpp_ids.includes(128_012),
    'SmolLM3 CONTROL chat markers were not parsed to their exact IDs')
    check(Boolean(ordinaryChat)
      && validIdArray(ordinaryChat.camelid_ids, tokenCount)
      && validIdArray(ordinaryChat.llama_cpp_ids, tokenCount)
      && !ordinaryChat.camelid_ids.includes(128_011)
      && !ordinaryChat.camelid_ids.includes(128_012)
      && !ordinaryChat.llama_cpp_ids.includes(128_011)
      && !ordinaryChat.llama_cpp_ids.includes(128_012),
    'SmolLM3 CONTROL chat markers parsed despite parse_special=false')

    const withParse = cases.find((testCase) => testCase?.id === 'user_defined_tool_tags_with_parse_special')
    const withoutParse = cases.find((testCase) => testCase?.id === 'user_defined_tool_tags_without_parse_special')
    check(Boolean(withParse && withoutParse)
      && validIdArray(withParse.camelid_ids, tokenCount)
      && validIdArray(withParse.llama_cpp_ids, tokenCount)
      && validIdArray(withoutParse.camelid_ids, tokenCount)
      && validIdArray(withoutParse.llama_cpp_ids, tokenCount)
      && JSON.stringify(withParse.camelid_ids) === JSON.stringify(withoutParse.camelid_ids)
      && JSON.stringify(withParse.llama_cpp_ids) === JSON.stringify(withoutParse.llama_cpp_ids)
      && withParse.camelid_ids[0] === 128_015
      && withParse.camelid_ids.at(-1) === 128_016
      && withParse.llama_cpp_ids[0] === 128_015
      && withParse.llama_cpp_ids.at(-1) === 128_016
      && withoutParse.camelid_ids[0] === 128_015
      && withoutParse.camelid_ids.at(-1) === 128_016
      && withoutParse.llama_cpp_ids[0] === 128_015
      && withoutParse.llama_cpp_ids.at(-1) === 128_016,
    'SmolLM3 USER_DEFINED tool tags changed with parse_special or lost exact boundary IDs')
  }

  if (row.id === 'qwen3_30b_a3b_q8_0') {
    const emptyWithSpecial = cases.find((testCase) => testCase?.id === 'empty_with_add_special')
    const helloWithSpecial = cases.find((testCase) => testCase?.id === 'plain_ascii_with_add_special')
    const helloWithoutSpecial = cases.find((testCase) => testCase?.id === 'plain_ascii_without_add_special')
    check(Boolean(emptyWithSpecial)
      && validIdArray(emptyWithSpecial.camelid_ids, tokenCount)
      && validIdArray(emptyWithSpecial.llama_cpp_ids, tokenCount)
      && emptyWithSpecial.camelid_ids.length === 0
      && emptyWithSpecial.llama_cpp_ids.length === 0,
    'Qwen3 MoE add_bos=false did not preserve empty add-special output')
    check(Boolean(helloWithSpecial && helloWithoutSpecial)
      && JSON.stringify(helloWithSpecial.camelid_ids) === '[9707]'
      && JSON.stringify(helloWithSpecial.llama_cpp_ids) === '[9707]'
      && JSON.stringify(helloWithoutSpecial.camelid_ids) === '[9707]'
      && JSON.stringify(helloWithoutSpecial.llama_cpp_ids) === '[9707]',
    'Qwen3 MoE add_bos=false did not preserve exact Hello token IDs')

    const parsedChat = cases.find((testCase) => testCase?.id === 'single_user_chat_controls')
    const ordinaryChat = cases.find((testCase) => testCase?.id === 'chat_controls_as_ordinary_text')
    check(Boolean(parsedChat)
      && validIdArray(parsedChat.camelid_ids, tokenCount)
      && validIdArray(parsedChat.llama_cpp_ids, tokenCount)
      && parsedChat.camelid_ids.includes(151_644)
      && parsedChat.camelid_ids.includes(151_645)
      && parsedChat.llama_cpp_ids.includes(151_644)
      && parsedChat.llama_cpp_ids.includes(151_645),
    'Qwen3 MoE CONTROL ChatML markers were not parsed to their exact IDs')
    check(Boolean(ordinaryChat)
      && validIdArray(ordinaryChat.camelid_ids, tokenCount)
      && validIdArray(ordinaryChat.llama_cpp_ids, tokenCount)
      && !ordinaryChat.camelid_ids.includes(151_644)
      && !ordinaryChat.camelid_ids.includes(151_645)
      && !ordinaryChat.llama_cpp_ids.includes(151_644)
      && !ordinaryChat.llama_cpp_ids.includes(151_645),
    'Qwen3 MoE CONTROL ChatML markers parsed despite parse_special=false')

    const toolWithParse = cases.find((testCase) => testCase?.id === 'user_defined_tool_tags_with_parse_special')
    const toolWithoutParse = cases.find((testCase) => testCase?.id === 'user_defined_tool_tags_without_parse_special')
    check(Boolean(toolWithParse && toolWithoutParse)
      && JSON.stringify(toolWithParse.camelid_ids) === JSON.stringify(toolWithoutParse.camelid_ids)
      && JSON.stringify(toolWithParse.llama_cpp_ids) === JSON.stringify(toolWithoutParse.llama_cpp_ids)
      && toolWithParse.camelid_ids?.[0] === 151_657
      && toolWithParse.camelid_ids?.at?.(-1) === 151_658
      && toolWithParse.llama_cpp_ids?.[0] === 151_657
      && toolWithParse.llama_cpp_ids?.at?.(-1) === 151_658,
    'Qwen3 MoE USER_DEFINED tool tags changed with parse_special or lost exact boundary IDs')

    const parsedThink = cases.find((testCase) => testCase?.id === 'user_defined_think_tags_with_parse_special')
    check(Boolean(parsedThink)
      && JSON.stringify(parsedThink.camelid_ids) === '[151667,19895,151668]'
      && JSON.stringify(parsedThink.llama_cpp_ids) === '[151667,19895,151668]',
    'Qwen3 MoE USER_DEFINED think tags lost their exact boundary IDs')

    const padWithParse = cases.find((testCase) => testCase?.id === 'unused_pad_with_parse_special')
    const padWithoutParse = cases.find((testCase) => testCase?.id === 'unused_pad_without_parse_special')
    check(Boolean(padWithParse && padWithoutParse)
      && JSON.stringify(padWithParse.camelid_ids) === JSON.stringify(padWithoutParse.camelid_ids)
      && JSON.stringify(padWithParse.llama_cpp_ids) === JSON.stringify(padWithoutParse.llama_cpp_ids)
      && !padWithParse.camelid_ids?.includes?.(151_669)
      && !padWithParse.llama_cpp_ids?.includes?.(151_669),
    'Qwen3 MoE UNUSED padding token was parsed as a special token')
  }

  const serialized = JSON.stringify(receipt)
  check(!/[A-Za-z]:[\\/]/.test(serialized), 'receipt exposes an absolute Windows path')
  check(!/(?:\/Users\/|\/home\/|\/tmp\/)/i.test(serialized), 'receipt exposes an absolute local path')
  check(!/(?:Bearer\s+|hf_[A-Za-z0-9]{8,})/i.test(serialized), 'receipt exposes an access token')
  const compactMetadata = Object.fromEntries(
    Object.keys(pack.metadataSummary).map((field) => [field, structuredClone(metadata[field])]),
  )
  return {
    errors,
    parity_errors: parityErrors,
    case_count: cases.length,
    exact_match_count: exactMatches,
    all_token_ids_match: exactMatches === pack.cases.length,
    tokenizer_metadata: compactMetadata,
    prefix_bytes: pack.prefixBytes,
  }
}

function validateGemma2TokenizerReceipt(receipt, row, defaults) {
  const assessed = assessTokenizerReceiptForPack(
    receipt,
    row,
    defaults,
    TOKENIZER_PACKS.gemma2_9b_it_q8_0,
  )
  return [...assessed.errors, ...assessed.parity_errors]
}

function validateSmolLM3TokenizerReceipt(receipt, row, defaults) {
  const assessed = assessTokenizerReceiptForPack(
    receipt,
    row,
    defaults,
    TOKENIZER_PACKS.smollm3_3b_q8_0,
  )
  return [...assessed.errors, ...assessed.parity_errors]
}

function validateQwen3MoeTokenizerReceipt(receipt, row, defaults) {
  const assessed = assessTokenizerReceiptForPack(
    receipt,
    row,
    defaults,
    TOKENIZER_PACKS.qwen3_30b_a3b_q8_0,
  )
  return [...assessed.errors, ...assessed.parity_errors]
}

function assessTokenizerReceipt(receipt, row, defaults, options = {}) {
  if (!tokenizerPackAvailable(row?.id)) {
    return {
      errors: ['bounded tokenizer pack is unavailable for this row'],
      parity_errors: [],
      case_count: 0,
      exact_match_count: 0,
      all_token_ids_match: false,
      tokenizer_metadata: {},
      prefix_bytes: null,
    }
  }
  return assessTokenizerReceiptForPack(
    receipt,
    row,
    defaults,
    TOKENIZER_PACKS[row.id],
    options,
  )
}

async function inspectPrefix(binary, prefixPath, declaredLength) {
  const { stdout } = await execFileAsync(binary, [
    'inspect-prefix', prefixPath, '--declared-len', String(declaredLength),
  ], { timeout: 90_000, maxBuffer: 256 * 1024 * 1024, windowsHide: true })
  return JSON.parse(stdout)
}

async function runCamelidCase(binary, prefixPath, declaredLength, temporary, testCase) {
  const inputPath = join(temporary, `${testCase.id}.camelid.json`)
  await writeFile(inputPath, `${JSON.stringify([testCase.text])}\n`)
  const { stdout } = await execFileAsync(binary, buildCamelidArgs({
    prefixPath,
    declaredLength,
    inputPath,
    addSpecial: testCase.add_special,
    parseSpecial: testCase.parse_special,
  }), { timeout: 90_000, maxBuffer: 16 * 1024 * 1024, windowsHide: true })
  const lines = stdout.split(/\r?\n/).filter((line) => line.trim())
  if (lines.length !== 1) throw new Error(`Camelid emitted ${lines.length} rows for ${testCase.id}`)
  const parsed = JSON.parse(lines[0])
  if (!Array.isArray(parsed.ids)) throw new Error(`Camelid omitted IDs for ${testCase.id}`)
  return parsed
}

async function runLlamaCase(binary, vocabOnlyPath, temporary, testCase) {
  const promptPath = join(temporary, `${testCase.id}.prompt.txt`)
  await writeFile(promptPath, testCase.text)
  const { stdout } = await execFileAsync(binary, buildLlamaArgs({
    modelPath: vocabOnlyPath,
    promptPath,
    addSpecial: testCase.add_special,
    parseSpecial: testCase.parse_special,
  }), { timeout: 90_000, maxBuffer: 16 * 1024 * 1024, windowsHide: true })
  return parseLlamaIds(stdout)
}

function validSourceProvenance(value) {
  return value
    && typeof value === 'object'
    && /^[0-9a-f]{40}$/.test(value.sourceHead || '')
    && value.sourceTrackedDirty === false
}

function validateCamelidTokenizerIdentity(identity, sourceProvenance) {
  const recomputed = classifyCamelidProvenance({
    version: identity?.version,
    ...sourceProvenance,
  })
  const provenanceMatches = Object.entries(recomputed)
    .every(([field, expected]) => identity?.provenance?.[field] === expected)
  if (!identity
    || typeof identity !== 'object'
    || !/^[0-9a-f]{64}$/.test(identity.binary_sha256 || '')
    || !recomputed.clean_current_head
    || !provenanceMatches) {
    throw tokenizerError('tokenizer_inspector_not_clean_current_head')
  }
  return { version: identity.version, binary_sha256: identity.binary_sha256, provenance: recomputed }
}

function validateLlamaPackageIdentity(llamaPackage, defaults) {
  const expectedBuild = Number(String(defaults?.llama_cpp?.build || '').replace(/^b/, ''))
  if (!llamaPackage
    || typeof llamaPackage !== 'object'
    || llamaPackage.revision !== PINNED_LLAMA_REVISION
    || llamaPackage.revision !== defaults?.llama_cpp?.revision
    || llamaPackage.build !== expectedBuild
    || llamaPackage.binary_sha256 !== PINNED_LLAMA_TOKENIZE_SHA256
    || llamaPackage.companion_binary_sha256 !== PINNED_LLAMA_CLI_SHA256) {
    throw tokenizerError('tokenizer_oracle_identity_mismatch')
  }
  return llamaPackage
}

async function inspectRemoteTokenizer(lock, {
  row,
  defaults,
  binary,
  llamaTokenize,
  sourceRoot = resolve('.'),
  prefixBytes = DEFAULT_PREFIX_BYTES,
  token = null,
  fetchImpl = fetch,
  fetchPrefixImpl = fetchHeaderPrefix,
  sourceProvenanceImpl = readSourceProvenance,
  camelidIdentityImpl = inspectCamelidTokenizerIdentity,
  llamaPackageImpl = verifyLlamaCppPackage,
  inspectImpl = inspectPrefix,
  metadataValidatorImpl = (selectedPack, inspection) => selectedPack.assertMetadata(inspection),
  deriveImpl = makeVocabOnlyGguf,
  camelidCaseImpl = runCamelidCase,
  llamaCaseImpl = runLlamaCase,
  mkdtempImpl = mkdtemp,
  writeFileImpl = writeFile,
  rmImpl = rm,
  prefixSha256Impl = sha256,
  derivativeSha256Impl = sha256,
  now = () => new Date(),
} = {}) {
  if (!tokenizerPackAvailable(row?.id)) {
    throw tokenizerError('tokenizer_pack_unavailable')
  }
  const pack = TOKENIZER_PACKS[row.id]
  try {
    normalizeTokenizerPrefixBytes(row.id, prefixBytes)
  } catch {
    throw tokenizerError('tokenizer_prefix_budget_invalid')
  }
  try {
    validateLockAgainstSelection(lock, sourceSelectionForRow(row))
  } catch {
    throw tokenizerError('tokenizer_source_identity_mismatch')
  }

  let sourceBefore
  try { sourceBefore = await sourceProvenanceImpl(sourceRoot) }
  catch { throw tokenizerError('tokenizer_inspector_unavailable') }
  if (!validSourceProvenance(sourceBefore)) {
    throw tokenizerError('tokenizer_inspector_not_clean_current_head')
  }

  let camelidIdentity
  try {
    camelidIdentity = validateCamelidTokenizerIdentity(
      await camelidIdentityImpl(binary, { sourceProvenance: sourceBefore }),
      sourceBefore,
    )
  } catch (error) {
    if (error instanceof TokenizerQualificationError) throw error
    throw tokenizerError('tokenizer_inspector_unavailable')
  }

  let llamaPackage
  try {
    llamaPackage = validateLlamaPackageIdentity(
      await llamaPackageImpl(llamaTokenize),
      defaults,
    )
  } catch (error) {
    if (error instanceof TokenizerQualificationError) throw error
    throw tokenizerError('tokenizer_oracle_unavailable')
  }

  let ranged
  try {
    ranged = await fetchPrefixImpl(lock, { prefixBytes: pack.prefixBytes, token, fetchImpl })
  } catch (error) {
    if (error instanceof HeaderInspectionError) {
      const classified = classifyHeaderInspectionError(error)
      throw tokenizerError(classified.status === 'fail'
        ? 'tokenizer_range_invalid'
        : 'tokenizer_range_unavailable')
    }
    throw tokenizerError('tokenizer_range_unavailable')
  }
  const contentRange = ranged?.content_range
  if (!Buffer.isBuffer(ranged?.bytes)
    || ranged.requested_bytes !== pack.prefixBytes
    || ranged.bytes.length !== pack.prefixBytes
    || contentRange?.start !== 0
    || contentRange?.end + 1 !== pack.prefixBytes
    || contentRange?.total !== row.identity.size_bytes) {
    throw tokenizerError('tokenizer_range_invalid')
  }
  if (ranged.prefix_sha256 !== pack.prefixSha256
    || prefixSha256Impl(ranged.bytes) !== pack.prefixSha256) {
    throw tokenizerError('tokenizer_prefix_identity_mismatch')
  }

  let temporary
  try { temporary = await mkdtempImpl(join(tmpdir(), 'camelid-hf-tokenizer-')) }
  catch { throw tokenizerError('tokenizer_qualification_error') }
  let report
  try {
    const prefixPath = join(temporary, 'header.gguf')
    const vocabOnlyPath = join(temporary, 'vocab-only.gguf')
    await writeFileImpl(prefixPath, ranged.bytes)
    let inspection
    let tokenizerMetadata
    let derived
    try {
      inspection = await inspectImpl(binary, prefixPath, lock.size_bytes)
      tokenizerMetadata = metadataValidatorImpl(pack, inspection)
      derived = deriveImpl(ranged.bytes)
      if (derived.original_tensor_count !== inspection.tensor_count
        || derived.original_tensor_count !== pack.tensorCount
        || derived.metadata_count !== pack.metadataCount) {
        throw new Error('descriptor counts drifted')
      }
    } catch {
      throw tokenizerError('tokenizer_metadata_mismatch')
    }
    await writeFileImpl(vocabOnlyPath, derived.bytes)

    const cases = []
    try {
      for (const testCase of pack.cases) {
        const camelidResult = await camelidCaseImpl(
          binary,
          prefixPath,
          lock.size_bytes,
          temporary,
          testCase,
        )
        const llamaIds = await llamaCaseImpl(
          llamaTokenize,
          vocabOnlyPath,
          temporary,
          testCase,
        )
        if (!validIdArray(camelidResult?.ids, pack.metadataSummary.token_count)
          || !validIdArray(llamaIds, pack.metadataSummary.token_count)) {
          throw new Error('invalid token IDs')
        }
        cases.push({
          id: testCase.id,
          text_utf8_bytes: Buffer.byteLength(testCase.text),
          text_sha256: sha256(Buffer.from(testCase.text)),
          add_special: testCase.add_special,
          parse_special: testCase.parse_special,
          camelid_ids: camelidResult.ids,
          llama_cpp_ids: llamaIds,
          exact_match: JSON.stringify(camelidResult.ids) === JSON.stringify(llamaIds),
          camelid_decoded_sha256: sha256(Buffer.from(String(camelidResult.decoded ?? ''))),
        })
      }
    } catch (error) {
      if (error instanceof TokenizerQualificationError) throw error
      throw tokenizerError('tokenizer_probe_failed')
    }

    let sourceAfter
    try { sourceAfter = await sourceProvenanceImpl(sourceRoot) }
    catch { throw tokenizerError('tokenizer_source_changed') }
    if (!validSourceProvenance(sourceAfter)
      || sourceAfter.sourceHead !== sourceBefore.sourceHead) {
      throw tokenizerError('tokenizer_source_changed')
    }

    let camelidAfter
    try {
      camelidAfter = validateCamelidTokenizerIdentity(
        await camelidIdentityImpl(binary, { sourceProvenance: sourceAfter }),
        sourceAfter,
      )
    } catch {
      throw tokenizerError('tokenizer_inspector_changed')
    }
    if (camelidAfter.version !== camelidIdentity.version
      || camelidAfter.binary_sha256 !== camelidIdentity.binary_sha256
      || JSON.stringify(camelidAfter.provenance) !== JSON.stringify(camelidIdentity.provenance)) {
      throw tokenizerError('tokenizer_inspector_changed')
    }

    let llamaPackageAfter
    try {
      llamaPackageAfter = validateLlamaPackageIdentity(
        await llamaPackageImpl(llamaTokenize),
        defaults,
      )
    } catch {
      throw tokenizerError('tokenizer_oracle_changed')
    }
    if (llamaPackageAfter.revision !== llamaPackage.revision
      || llamaPackageAfter.build !== llamaPackage.build
      || llamaPackageAfter.binary_sha256 !== llamaPackage.binary_sha256
      || llamaPackageAfter.companion_binary_sha256 !== llamaPackage.companion_binary_sha256
      || llamaPackageAfter.executable !== llamaPackage.executable
      || llamaPackageAfter.companion_executable !== llamaPackage.companion_executable) {
      throw tokenizerError('tokenizer_oracle_changed')
    }
    let generatedAt
    try { generatedAt = now().toISOString() }
    catch { throw tokenizerError('tokenizer_receipt_time_invalid') }
    if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(generatedAt)) {
      throw tokenizerError('tokenizer_receipt_time_invalid')
    }

    const allMatch = cases.every((testCase) => testCase.exact_match)
    report = {
      schema: 'camelid.header-tokenizer-parity/v1',
      generated_at: generatedAt,
      provenance: camelidIdentity.provenance,
      row_id: row.id,
      host: {
        platform: `${process.platform}-${process.arch}`,
        hostname_redacted: true,
      },
      source: {
        repo: lock.repo,
        file: lock.file,
        revision: lock.revision,
        size_bytes: lock.size_bytes,
        sha256: lock.sha256,
        license: lock.license,
      },
      bounded_fetch: {
        requested_bytes: ranged.requested_bytes,
        received_bytes: ranged.bytes.length,
        content_range: ranged.content_range,
        prefix_sha256: ranged.prefix_sha256,
        temporary_paths_redacted: true,
        temporary_files_deleted: true,
        scope_note: TOKENIZER_SCOPE_NOTE,
      },
      ...(pack.grounding ? { grounding: pack.grounding } : {}),
      tokenizer_metadata: tokenizerMetadata,
      camelid: {
        version: camelidIdentity.version,
        binary_sha256: camelidIdentity.binary_sha256,
        prefix_mode: 'tokenize --declared-len',
      },
      oracle: {
        project: 'ggml-org/llama.cpp',
        revision: llamaPackage.revision,
        build: llamaPackage.build,
        revision_verification: 'parsed from the sibling llama-cli --version output in the same binary package',
        companion_executable: llamaPackage.companion_executable,
        companion_binary_sha256: llamaPackage.companion_binary_sha256,
        executable: llamaPackage.executable,
        binary_sha256: llamaPackage.binary_sha256,
        input: 'disposable vocabulary-only derivative of the same immutable GGUF prefix; tensor_count is zeroed so llama-tokenize consumes unchanged tokenizer metadata and ignores the original descriptor region plus opaque partial payload bytes',
        derivative: {
          original_tensor_count: derived.original_tensor_count,
          metadata_count: derived.metadata_count,
          patch: 'set fixed-header tensor_count at byte offset 8 to zero; all tokenizer metadata bytes remain unchanged',
          patch_offset: derived.patched_offset,
          sha256: derivativeSha256Impl(derived.bytes),
          persisted: false,
        },
      },
      cases,
      result: {
        case_count: cases.length,
        exact_match_count: cases.filter((testCase) => testCase.exact_match).length,
        all_token_ids_match: allMatch,
        support_decision: pack.supportDecision,
      },
      does_not_prove: tokenizerDoesNotProve(pack),
    }
  } catch (error) {
    if (error instanceof TokenizerQualificationError) throw error
    throw tokenizerError('tokenizer_qualification_error')
  } finally {
    try { await rmImpl(temporary, { recursive: true, force: true }) }
    catch { throw tokenizerError('tokenizer_cleanup_failed') }
  }
  const selfAssessment = assessTokenizerReceiptForPack(report, row, defaults, pack, {
    expectedSourceHead: sourceBefore.sourceHead,
  })
  if (selfAssessment.errors.length > 0) {
    throw tokenizerError('tokenizer_probe_failed')
  }
  return report
}

async function runTokenizerCli(argv = process.argv.slice(2), {
  sourceResolver = resolveHfSource,
} = {}) {
  const args = parseArgs(argv)
  if (args.has('help') || !args.get('row')) {
    console.log(`Usage:
  node scripts/hf-qualification-tokenizer.mjs --row <gemma2_9b_it_q8_0|smollm3_3b_q8_0|qwen3_30b_a3b_q8_0> [options]

Options:
  --roster <path>          Roster path (default: Phase 1)
  --camelid <path>         Camelid binary (default: target/debug/camelid)
  --llama-tokenize <path>  Pinned llama-tokenize binary
  --prefix-bytes <n>       Exact range budget; only 32 MiB is accepted
  --out <path>             Write the scrubbed parity receipt
  HF_TOKEN                 Optional token for gated/private rows
`)
    process.exit(args.has('help') ? 0 : 1)
  }

  const root = resolve('.')
  const row = await selectRow(
    root,
    args.get('roster') || 'qa/model-qualification/phase1-roster.json',
    args.get('row'),
  )
  let prefixBytes
  try {
    prefixBytes = normalizeTokenizerPrefixBytes(
      row.id,
      args.get('prefix-bytes') || DEFAULT_PREFIX_BYTES,
    )
  } catch {
    throw tokenizerError('tokenizer_prefix_budget_invalid')
  }
  const lock = await sourceResolver({
    repo: row.source.repo,
    file: row.source.file,
    revision: row.source.revision,
    token: process.env.HF_TOKEN || null,
  })
  const defaultCamelid = process.platform === 'win32'
    ? 'target/debug/camelid.exe'
    : 'target/debug/camelid'
  const defaultLlama = process.platform === 'win32'
    ? 'target/reference/llama.cpp-b9632/bin/llama-tokenize.exe'
    : 'target/reference/llama.cpp-b9632/bin/llama-tokenize'
  const camelid = resolve(args.get('camelid') || process.env.CAMELID_BIN || defaultCamelid)
  const llamaTokenize = resolve(args.get('llama-tokenize') || defaultLlama)
  const report = await inspectRemoteTokenizer(lock, {
    row,
    defaults: JSON.parse(await readFile(
      resolve(root, args.get('roster') || 'qa/model-qualification/phase1-roster.json'),
      'utf8',
    )).defaults,
    binary: camelid,
    llamaTokenize,
    sourceRoot: root,
    prefixBytes,
    token: process.env.HF_TOKEN || null,
  })

  // Emit durable evidence only after the temporary prefix, derivative model,
  // and plaintext probe files have been removed successfully. This makes the
  // receipt's cleanup claims observed facts rather than intentions.
  const rendered = `${JSON.stringify(report, null, 2)}\n`
  if (args.get('out')) {
    const out = resolve(args.get('out'))
    await mkdir(dirname(out), { recursive: true })
    await writeFile(out, rendered)
  }
  process.stdout.write(rendered)
  if (!report.result.all_token_ids_match) process.exitCode = 2
}

export {
  DEFAULT_PREFIX_BYTES as DEFAULT_TOKENIZER_PREFIX_BYTES,
  GEMMA2_CASES,
  QWEN3_MOE_CASES,
  SMOLLM3_CASES,
  TokenizerQualificationError,
  assessTokenizerReceipt,
  assertGemma2TokenizerMetadata,
  assertQwen3MoeTokenizerMetadata,
  assertSmolLM3TokenizerMetadata,
  buildCamelidArgs,
  buildLlamaArgs,
  classifyCamelidProvenance,
  classifyTokenizerQualificationError,
  inspectRemoteTokenizer,
  makeVocabOnlyGguf,
  normalizeTokenizerPrefixBytes,
  parseLlamaIds,
  parseLlamaVersionOutput,
  runTokenizerCli,
  sourceSelectionForRow,
  tokenizerPackAvailable,
  tokenizerPrefixBytesForRow,
  validateGemma2TokenizerReceipt,
  validateQwen3MoeTokenizerReceipt,
  validateSmolLM3TokenizerReceipt,
  verifyLlamaCppPackage,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  runTokenizerCli().catch((error) => {
    const failure = classifyTokenizerQualificationError(error)
    console.error(`${failure.error_code}: ${failure.reason}`)
    process.exit(1)
  })
}
