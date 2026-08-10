#!/usr/bin/env node

import { createHash } from 'node:crypto'
import { execFile } from 'node:child_process'
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import { validateRoster } from './check-model-qualification-roster.mjs'
import { fetchHeaderPrefix } from './hf-qualification-header.mjs'
import {
  resolveHfSource,
  validateLockAgainstSelection,
} from './hf-qualification-source.mjs'

const execFileAsync = promisify(execFile)
const DEFAULT_PREFIX_BYTES = 32 * 1024 * 1024
const MAX_PREFIX_BYTES = 64 * 1024 * 1024
const PINNED_LLAMA_REVISION = 'acd79d603'
const PINNED_LLAMA_TOKENIZE_SHA256 = 'a44a4d7e1445d22a4cffb0d38f6efa8f1d81e84ae2c3d481af857c5e331b8c7a'
const PINNED_LLAMA_CLI_SHA256 = '2ec09da0b81d0201ce5b21810caefb4e77fd108f383b30c15ca493c5a70f7731'
const GEMMA2_TEMPLATE_SHA256 = 'ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6'

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
  const match = /(?:^|-)g([0-9a-f]{7,40})(-dirty)?$/i.exec(String(version))
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

async function verifyLlamaCppPackage(llamaTokenize) {
  const companion = join(
    dirname(llamaTokenize),
    process.platform === 'win32' ? 'llama-cli.exe' : 'llama-cli',
  )
  let output = ''
  try {
    const result = await execFileAsync(companion, ['--version'], {
      timeout: 10_000,
      windowsHide: true,
    })
    output = `${result.stdout || ''}\n${result.stderr || ''}`
  } catch (error) {
    // The pinned Windows b9632 launcher reports its version and exits 1. Accept
    // that historical quirk only when the output still carries the exact pin.
    output = `${error.stdout || ''}\n${error.stderr || ''}`
    if (!output.trim()) throw error
  }
  const version = parseLlamaVersionOutput(output)
  return {
    ...version,
    companion_executable: basename(companion),
    companion_binary_sha256: sha256(await readFile(companion)),
  }
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

async function selectRow(root, rosterPath, rowId) {
  const absolute = resolve(root, rosterPath)
  const roster = JSON.parse(await readFile(absolute, 'utf8'))
  const errors = validateRoster(roster, absolute)
  if (errors.length) throw new Error(`roster is invalid:\n${errors.join('\n')}`)
  const row = roster.rows.find((candidate) => candidate.id === rowId)
  if (!row) throw new Error(`unknown --row ${JSON.stringify(rowId)}`)
  if (row.id !== 'gemma2_9b_it_q8_0') {
    throw new Error('this bounded tokenizer pack is pinned only to gemma2_9b_it_q8_0')
  }
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

function validIdArray(value) {
  return Array.isArray(value)
    && value.every((id) => Number.isSafeInteger(id) && id >= 0)
}

function validateGemma2TokenizerReceipt(receipt, row, defaults) {
  const errors = []
  const check = (condition, message) => { if (!condition) errors.push(message) }
  const shaRe = /^[0-9a-f]{64}$/
  check(receipt?.schema === 'camelid.header-tokenizer-parity/v1', 'schema mismatch')
  check(receipt?.row_id === row.id, 'row_id mismatch')

  for (const [field, expected] of [
    ['repo', row.source.repo],
    ['file', row.source.file],
    ['revision', row.source.revision],
    ['size_bytes', row.identity.size_bytes],
    ['sha256', row.identity.sha256],
  ]) {
    check(receipt?.source?.[field] === expected, `source.${field} mismatch`)
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
  for (const [field, expected] of Object.entries(recomputedProvenance)) {
    check(provenance[field] === expected, `provenance.${field} is not derivable from Camelid version and source head`)
  }
  check(shaRe.test(receipt?.camelid?.binary_sha256 || ''), 'Camelid binary SHA-256 is invalid')
  check(receipt?.camelid?.prefix_mode === 'tokenize --declared-len', 'Camelid prefix mode mismatch')

  const bounded = receipt?.bounded_fetch || {}
  const contentRange = bounded.content_range || {}
  check(Number.isSafeInteger(bounded.requested_bytes) && bounded.requested_bytes > 0 && bounded.requested_bytes <= MAX_PREFIX_BYTES, 'bounded request size is invalid')
  check(bounded.received_bytes === bounded.requested_bytes, 'bounded received byte count mismatch')
  check(contentRange.start === 0, 'Content-Range must start at zero')
  check(contentRange.end + 1 === bounded.received_bytes, 'Content-Range end does not match received bytes')
  check(contentRange.total === row.identity.size_bytes, 'Content-Range total does not match row size')
  check(shaRe.test(bounded.prefix_sha256 || ''), 'prefix SHA-256 is invalid')
  check(bounded.temporary_paths_redacted === true, 'temporary paths are not redacted')
  check(bounded.temporary_files_deleted === true, 'temporary files were not confirmed deleted')

  const metadata = receipt?.tokenizer_metadata || {}
  check(metadata.token_count === 256_000, 'token count mismatch')
  check(metadata.score_count === metadata.token_count, 'score count mismatch')
  check(metadata.token_type_count === metadata.token_count, 'token type count mismatch')
  check(metadata.chat_template_sha256 === GEMMA2_TEMPLATE_SHA256, 'chat template SHA-256 mismatch')

  const expectedBuild = Number(String(defaults?.llama_cpp?.build || '').replace(/^b/, ''))
  check(receipt?.oracle?.revision === defaults?.llama_cpp?.revision, 'llama.cpp revision mismatch')
  check(receipt?.oracle?.build === expectedBuild, 'llama.cpp build mismatch')
  check(receipt?.oracle?.binary_sha256 === PINNED_LLAMA_TOKENIZE_SHA256, 'llama-tokenize SHA-256 mismatch')
  check(receipt?.oracle?.companion_binary_sha256 === PINNED_LLAMA_CLI_SHA256, 'llama.cpp companion SHA-256 mismatch')
  check(receipt?.oracle?.derivative?.persisted === false, 'vocabulary-only derivative was persisted')
  check(receipt?.oracle?.derivative?.original_tensor_count === 464, 'derivative tensor count mismatch')
  check(receipt?.oracle?.derivative?.metadata_count === 26, 'derivative metadata count mismatch')
  check(receipt?.oracle?.derivative?.patch_offset === 8, 'derivative patch offset mismatch')
  check(shaRe.test(receipt?.oracle?.derivative?.sha256 || ''), 'derivative SHA-256 is invalid')

  const cases = Array.isArray(receipt?.cases) ? receipt.cases : []
  check(cases.length === GEMMA2_CASES.length, 'case count mismatch')
  let exactMatches = 0
  for (let index = 0; index < GEMMA2_CASES.length; index += 1) {
    const expected = GEMMA2_CASES[index]
    const observed = cases[index] || {}
    const idsValid = validIdArray(observed.camelid_ids) && validIdArray(observed.llama_cpp_ids)
    const idsMatch = idsValid
      && JSON.stringify(observed.camelid_ids) === JSON.stringify(observed.llama_cpp_ids)
    if (idsMatch) exactMatches += 1
    check(observed.id === expected.id, `case ${expected.id} id/order mismatch`)
    check(observed.text_utf8_bytes === Buffer.byteLength(expected.text), `case ${expected.id} byte count mismatch`)
    check(observed.text_sha256 === sha256(Buffer.from(expected.text)), `case ${expected.id} text SHA-256 mismatch`)
    check(observed.add_special === expected.add_special, `case ${expected.id} add_special mismatch`)
    check(observed.parse_special === expected.parse_special, `case ${expected.id} parse_special mismatch`)
    check(idsValid, `case ${expected.id} has invalid token IDs`)
    check(observed.exact_match === idsMatch, `case ${expected.id} exact_match is not derived from token IDs`)
    check(idsMatch, `case ${expected.id} token IDs diverge`)
    check(shaRe.test(observed.camelid_decoded_sha256 || ''), `case ${expected.id} decoded SHA-256 is invalid`)
  }
  check(receipt?.result?.case_count === cases.length, 'result case_count mismatch')
  check(receipt?.result?.exact_match_count === exactMatches, 'result exact_match_count mismatch')
  check(receipt?.result?.all_token_ids_match === (exactMatches === GEMMA2_CASES.length), 'result all_token_ids_match mismatch')
  check(receipt?.result?.support_decision === 'no_change_header_tokenizer_evidence_only', 'support decision widened unexpectedly')
  return errors
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

async function main() {
  const args = parseArgs(process.argv.slice(2))
  if (args.has('help') || !args.get('row')) {
    console.log(`Usage:
  node scripts/hf-qualification-tokenizer.mjs --row gemma2_9b_it_q8_0 [options]

Options:
  --roster <path>          Roster path (default: Phase 1)
  --camelid <path>         Camelid binary (default: target/debug/camelid)
  --llama-tokenize <path>  Pinned llama-tokenize binary
  --prefix-bytes <n>       Range budget, max 64 MiB (default: 32 MiB)
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
  const lock = await resolveHfSource({
    repo: row.source.repo,
    file: row.source.file,
    revision: row.source.revision,
    token: process.env.HF_TOKEN || null,
  })
  // A successful immutable Hub lookup is insufficient by itself: the bytes must
  // also agree with the roster's exact revision, size, SHA-256, and license.
  // Otherwise a row-labelled receipt could silently qualify a different file.
  validateLockAgainstSelection(lock, sourceSelectionForRow(row))
  const prefixBytes = Number(args.get('prefix-bytes') || DEFAULT_PREFIX_BYTES)
  if (!Number.isSafeInteger(prefixBytes) || prefixBytes <= 0 || prefixBytes > MAX_PREFIX_BYTES) {
    throw new Error(`prefix byte budget must be between 1 and ${MAX_PREFIX_BYTES}`)
  }
  const ranged = await fetchHeaderPrefix(lock, {
    prefixBytes,
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
  const llamaPackage = await verifyLlamaCppPackage(llamaTokenize)
  const temporary = await mkdtemp(join(tmpdir(), 'camelid-hf-tokenizer-'))
  let report
  let allMatch = false

  try {
    const prefixPath = join(temporary, 'header.gguf')
    const vocabOnlyPath = join(temporary, 'vocab-only.gguf')
    await writeFile(prefixPath, ranged.bytes)
    const inspection = await inspectPrefix(camelid, prefixPath, lock.size_bytes)
    const tokenizerMetadata = assertGemma2TokenizerMetadata(inspection)
    const derived = makeVocabOnlyGguf(ranged.bytes)
    if (derived.original_tensor_count !== inspection.tensor_count) {
      throw new Error('vocab-only derivative tensor count does not match prefix inspection')
    }
    await writeFile(vocabOnlyPath, derived.bytes)

    const cases = []
    for (const testCase of GEMMA2_CASES) {
      const camelidResult = await runCamelidCase(
        camelid,
        prefixPath,
        lock.size_bytes,
        temporary,
        testCase,
      )
      const llamaIds = await runLlamaCase(llamaTokenize, vocabOnlyPath, temporary, testCase)
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

    const camelidVersion = (await execFileAsync(camelid, ['--version'], {
      timeout: 10_000,
      windowsHide: true,
    })).stdout.trim()
    const provenance = classifyCamelidProvenance({
      version: camelidVersion,
      ...await readSourceProvenance(root),
    })
    const camelidBytes = await readFile(camelid)
    const llamaBytes = await readFile(llamaTokenize)
    allMatch = cases.every((testCase) => testCase.exact_match)
    report = {
      schema: 'camelid.header-tokenizer-parity/v1',
      generated_at: new Date().toISOString(),
      provenance,
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
      },
      bounded_fetch: {
        requested_bytes: ranged.requested_bytes,
        received_bytes: ranged.bytes.length,
        content_range: ranged.content_range,
        prefix_sha256: sha256(ranged.bytes),
        temporary_paths_redacted: true,
        temporary_files_deleted: true,
        scope_note: 'the prefix hash includes opaque initial tensor payload bytes after data_start_offset; it is not a full payload or full artifact hash',
      },
      tokenizer_metadata: tokenizerMetadata,
      camelid: {
        version: camelidVersion,
        binary_sha256: sha256(camelidBytes),
        prefix_mode: 'tokenize --declared-len',
      },
      oracle: {
        project: 'ggml-org/llama.cpp',
        revision: llamaPackage.revision,
        build: llamaPackage.build,
        revision_verification: 'parsed from the sibling llama-cli --version output in the same binary package',
        companion_executable: llamaPackage.companion_executable,
        companion_binary_sha256: llamaPackage.companion_binary_sha256,
        executable: basename(llamaTokenize),
        binary_sha256: sha256(llamaBytes),
        input: 'disposable vocabulary-only derivative of the same immutable GGUF prefix; tensor_count is zeroed so llama-tokenize consumes unchanged tokenizer metadata and ignores the original descriptor region plus opaque partial payload bytes',
        derivative: {
          original_tensor_count: derived.original_tensor_count,
          metadata_count: derived.metadata_count,
          patch: 'set fixed-header tensor_count at byte offset 8 to zero; all tokenizer metadata bytes remain unchanged',
          patch_offset: derived.patched_offset,
          sha256: sha256(derived.bytes),
          persisted: false,
        },
      },
      cases,
      result: {
        case_count: cases.length,
        exact_match_count: cases.filter((testCase) => testCase.exact_match).length,
        all_token_ids_match: allMatch,
        support_decision: 'no_change_header_tokenizer_evidence_only',
      },
      does_not_prove: [
        'full artifact integrity or presence on this host',
        'weight load, logits, generation, or greedy-token parity',
        'API, SSE, Models page, WebUI, or context readiness',
        'sampling, tools, GPU execution, performance, neighboring rows, or broad Gemma support',
      ],
    }
  } finally {
    await rm(temporary, { recursive: true, force: true })
  }

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
  if (!allMatch) process.exitCode = 2
}

export {
  GEMMA2_CASES,
  assertGemma2TokenizerMetadata,
  buildCamelidArgs,
  buildLlamaArgs,
  classifyCamelidProvenance,
  makeVocabOnlyGguf,
  parseLlamaIds,
  parseLlamaVersionOutput,
  sourceSelectionForRow,
  validateGemma2TokenizerReceipt,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
