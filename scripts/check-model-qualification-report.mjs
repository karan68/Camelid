#!/usr/bin/env node
import { createHash } from 'node:crypto'
import { readdir, readFile, stat } from 'node:fs/promises'
import { resolve } from 'node:path'
import { URL, pathToFileURL } from 'node:url'

const REQUIRED_STAGES = [
  'source',
  'artifact',
  'metadata',
  'tokenizer',
  'template',
  'load_smoke',
  'parity',
  'api_webui',
  'context',
]
const STATUSES = new Set(['pass', 'fail', 'blocked', 'skipped'])
const WINDOWS_ABSOLUTE_RE = /(?:^|[\s"'=([{,:])(?:[a-zA-Z]:[\\/]|\\\\)/
const UNIX_ABSOLUTE_RE = /(?:^|[\s"'=([{,:])(\/+[^\s"'<>]*)/g
const HTTP_URL_RE = /https?:\/\/[^\s"'<>]+/gi
const FILE_URI_RE = /\bfile:\/\/[^\s"'<>]*/i
const PUBLIC_ROUTE_LITERALS = new Set([
  '/apply-template',
  '/completion',
  '/health',
  '/tokenize',
  '/api/capabilities',
  '/api/models',
  '/api/models/current',
  '/api/models/inspect',
  '/api/models/load',
  '/api/models/local',
  '/v1/chat/completions',
  '/v1/completions',
  '/v1/embeddings',
  '/v1/models',
  '/v1/rerank',
])
const CANDIDATE_QUALIFICATION_MODE = 'unrostered_hf_selector'
const LEGACY_PROMOTION_MODE = 'legacy_exact_row_promotion'
const LEGACY_PROMOTION_CANONICAL_SHA256 = 'aeedbd186a27ec3c8164a6381102ebf109c94790ed12efa57f9f07357cc3587a'
const LEGACY_PROMOTION_ROW_ID = 'lfm2_5_2_6b_q8_0'
const LEGACY_PROMOTION_SOURCE_HEAD = '2cd9fa78f8dcaa78e011914076ecd1473ae99d0a'
const LEGACY_PROMOTION_RUNTIME_HEAD = '15ab4ddac5a21ac0f98f6ccba5e3ddff4b51a9b5'
const LEGACY_PROMOTION_ARTIFACT_SHA256 = '36587fdf27bdfc69caf2637273679a0870ec155162161bde6fd16e8c70bdb757'
const LEGACY_PROMOTION_SUPPORT_DECISION = 'promote_supported_exact_row_smoke'
const CANDIDATE_ID_RE = /^hf_selector_([0-9a-f]{24})$/
const CANDIDATE_RUN_ID_RE = /^hf_candidate_run_([0-9a-f]{32})$/
const UUID_RE = /^([0-9a-f]{8})-([0-9a-f]{4})-(4[0-9a-f]{3})-([89ab][0-9a-f]{3})-([0-9a-f]{12})$/
const CANDIDATE_REPO_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/
const CANDIDATE_FILE_SEGMENT_RE = /^[A-Za-z0-9][A-Za-z0-9._+() -]{0,255}$/
const CANDIDATE_LICENSE_RE = /^[A-Za-z0-9][A-Za-z0-9 ._+()-]{0,127}$/
const REVISION_RE = /^[0-9a-f]{40}$/
const SHA256_RE = /^[0-9a-f]{64}$/
const CANDIDATE_REPORT_LABEL_RE = /(?:^|[\\/])hf_(?:selector|candidate_run)_[^\\/]+-report\.json$/i
const CANDIDATE_HTTP_URL_VALUE_RE = /https?:\/\/[^\s"'<>]+/i
const AUTHORIZATION_VALUE_RE = /\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+/i
const HF_TOKEN_VALUE_RE = /\bhf_[A-Za-z0-9]{8,}\b/i
const CREDENTIAL_ASSIGNMENT_VALUE_RE = /(?:^|[?&#;,\s])(?:access[_-]?token|auth[_-]?token|authorization|bearer[_-]?token|client[_-]?secret|credential|download[_-]?(?:url|uri)|hf[_-]?token|id[_-]?token|api[_-]?key|password|private[_-]?key|refresh[_-]?token|secret|signed[_-]?(?:url|uri)|token)\s*[:=]\s*[^\s&#;,]+/i
const PRIVACY_MAX_DEPTH = 512
const PRIVACY_MAX_NODES = 100_000
const PRIVACY_MAX_STRING_CHARS = 4 * 1024 * 1024
const ROUTE_FIELD_RE = /^(?:public_)?(?:route|routes|endpoint|endpoints)$/
const COMMAND_FIELD_RE = /^(?:command|commands)$/

function containsAbsoluteLocalPath(value, context = []) {
  if (typeof value !== 'string') return false
  const routeContext = Array.isArray(context)
    ? context.some((part) => ROUTE_FIELD_RE.test(String(part)))
    : Boolean(context.routeContext)
  const commandContext = Array.isArray(context)
    ? context.some((part) => COMMAND_FIELD_RE.test(String(part)))
    : Boolean(context.commandContext)
  // Source/revision URLs are intentionally public evidence. Remove only HTTP(S)
  // spans before examining the remaining command/prose tokens; file:// remains
  // forbidden and is caught as a slash-absolute value.
  if (FILE_URI_RE.test(value)) return true
  const scrubbed = value.replace(HTTP_URL_RE, '<public-url>')
  if (WINDOWS_ABSOLUTE_RE.test(scrubbed)) return true

  UNIX_ABSOLUTE_RE.lastIndex = 0
  for (const match of scrubbed.matchAll(UNIX_ABSOLUTE_RE)) {
    const candidate = match[1].replace(/[),.;:\]}]+$/g, '')
    if (!candidate || /^\/+$/u.test(candidate)) continue
    if (PUBLIC_ROUTE_LITERALS.has(candidate) && (routeContext || commandContext)) continue
    if (routeContext
      && /^\/(?:api|v1)(?:\/[A-Za-z0-9_.:{}-]+)+\/?(?:\?[^\s]*)?$/.test(candidate)
      && !candidate.includes('..')) {
      continue
    }
    return true
  }
  return false
}

const CANDIDATE_DOWNSTREAM_CODES = Object.freeze({
  artifact: 'full_artifact_not_downloaded',
  tokenizer: 'candidate_tokenizer_pack_unavailable',
  template: 'candidate_template_pack_unavailable',
  load_smoke: 'candidate_full_artifact_unavailable',
  parity: 'candidate_oracle_pack_unavailable',
  api_webui: 'candidate_runtime_unqualified',
  context: 'candidate_context_unqualified',
})
const CANDIDATE_STAGE_NAMES = new Set([
  'source',
  'artifact',
  'metadata',
  'tokenizer',
  'template',
  'load_smoke',
  'parity',
  'api_webui',
  'context',
])
const CANDIDATE_SOURCE_ERROR_STATUSES = Object.freeze({
  source_identity_invalid: 'fail',
  source_license_unavailable: 'blocked',
  private_source_not_persisted: 'blocked',
  source_disabled: 'blocked',
  timeout: 'blocked',
  missing_immutable_revision: 'blocked',
  file_not_found_at_revision: 'blocked',
  missing_file_size: 'blocked',
  missing_lfs_sha256: 'blocked',
  network_error: 'blocked',
  source_lookup_error: 'blocked',
})
const CANDIDATE_METADATA_ERROR_STATUSES = Object.freeze({
  header_inspector_identity_invalid: 'fail',
  header_inspector_not_clean_current_head: 'blocked',
  header_inspector_unavailable: 'blocked',
  source_lock_invalid: 'blocked',
  header_fetch_unavailable: 'blocked',
  header_range_unavailable: 'blocked',
  header_range_invalid: 'fail',
  header_range_identity_mismatch: 'fail',
  header_range_incomplete: 'blocked',
  header_body_budget_exceeded: 'blocked',
  header_body_missing: 'blocked',
  header_body_unavailable: 'blocked',
  header_body_length_mismatch: 'fail',
  header_inspection_contract_invalid: 'fail',
  header_inspector_timeout: 'blocked',
  header_inspector_output_budget: 'blocked',
  header_parse_failed: 'fail',
  header_inspector_output_invalid: 'fail',
  header_row_id_invalid: 'fail',
  header_inspection_error: 'blocked',
  header_receipt_time_invalid: 'fail',
  header_receipt_invalid: 'fail',
  header_source_identity_mismatch: 'fail',
  header_descriptor_invariants_invalid: 'fail',
  header_full_artifact_forbidden: 'fail',
  header_source_preflight_blocked: 'blocked',
  header_source_head_unavailable: 'blocked',
  header_source_tracked_dirty: 'blocked',
  header_partial_range_unavailable: 'blocked',
})
const CREDENTIAL_KEY_LITERALS = new Set([
  'authorization',
  'proxyauthorization',
  'cookie',
  'setcookie',
  'password',
  'passwd',
  'secret',
  'clientsecret',
  'secretkey',
  'privatekey',
  'accesstoken',
  'accesskey',
  'refreshtoken',
  'idtoken',
  'hftoken',
  'authtoken',
  'bearertoken',
  'authentication',
  'apikey',
  'downloadurl',
  'downloaduri',
  'signedurl',
  'signeduri',
  'credential',
  'credentials',
  'token',
  'tokens',
])
const GENERIC_ROOT_KEY_VARIANTS = [
  [
    'artifact', 'backend', 'generated_at', 'host', 'oracle_fixture', 'overall_status',
    'row_id', 'schema', 'source_dirty', 'source_head', 'source_inspection', 'stages',
  ],
  [
    'artifact', 'does_not_prove', 'generated_at', 'host', 'overall_status', 'phase',
    'row_id', 'runtime_head', 'schema', 'source_dirty', 'source_evidence', 'source_head',
    'stages', 'support_decision',
  ],
  [
    'artifact', 'does_not_prove', 'generated_at', 'host', 'oracle', 'overall_status',
    'phase', 'row_id', 'schema', 'source_dirty', 'source_head', 'stages',
    'support_decision',
  ],
  [
    'artifact', 'backend', 'does_not_prove', 'generated_at', 'host', 'overall_status',
    'phase', 'row_id', 'schema', 'source_dirty', 'source_head', 'source_inspection',
    'stages', 'support_decision',
  ],
  [
    'artifact', 'backend', 'does_not_prove', 'generated_at', 'hold', 'host',
    'implementation_scope', 'overall_status', 'phase', 'row_id', 'schema', 'source_dirty',
    'source_evidence', 'source_head', 'source_inspection', 'stages', 'support_decision',
  ],
].map((keys) => new Set(keys))

const keyVariants = (...variants) => variants.map((keys) => new Set(keys))
const GENERIC_STAGE_KEY_VARIANTS = Object.freeze({
  source: Object.freeze({
    pass: keyVariants(
      ['status', 'evidence'],
      ['status', 'repo', 'revision', 'file', 'license'],
      ['status', 'resolution', 'repo', 'file', 'revision', 'size_bytes', 'sha256', 'license', 'access'],
    ),
    blocked: keyVariants(
      ['status', 'reason'],
      ['status', 'resolution', 'reason'],
      ['status', 'resolution', 'error_code', 'reason'],
    ),
    fail: keyVariants(
      ['status', 'resolution', 'reason', 'expected'],
      ['status', 'resolution', 'error_code', 'reason'],
    ),
    skipped: keyVariants(['status', 'reason', 'required']),
  }),
  artifact: Object.freeze({
    pass: keyVariants(
      ['status', 'evidence'],
      ['status', 'size_bytes', 'sha256'],
      ['status', 'observed_size_bytes', 'observed_sha256'],
    ),
    blocked: keyVariants(['status', 'reason']),
    fail: keyVariants(['status', 'reason']),
    skipped: keyVariants(['status', 'reason', 'required']),
  }),
  metadata: Object.freeze({
    pass: keyVariants(
      ['status', 'evidence'],
      ['status', 'evidence', 'note'],
      ['status', 'binary_profile', 'observed'],
      ['status', 'command', 'observed'],
      ['status', 'mode', 'inspection_generated_at', 'host', 'inspector', 'source', 'range', 'observed', 'scope'],
    ),
    blocked: keyVariants(
      ['status', 'reason'],
      ['status', 'command', 'reason'],
      ['status', 'mode', 'error_code', 'reason'],
    ),
    fail: keyVariants(
      ['status', 'command', 'reason'],
      ['status', 'command', 'reason', 'stderr'],
      ['status', 'command', 'observed', 'reason'],
      ['status', 'mode', 'error_code', 'reason'],
      ['status', 'mode', 'inspection_generated_at', 'host', 'inspector', 'source', 'range', 'observed', 'scope', 'reason'],
    ),
    skipped: keyVariants(['status', 'reason', 'required']),
  }),
  tokenizer: Object.freeze({
    pass: keyVariants(
      ['status', 'evidence'],
      ['status', 'evidence', 'note'],
      ['status', 'fixture', 'raw_prompt_sequences_matched', 'raw_prompt_sequences_total', 'chat_prompt_sequences_matched', 'chat_prompt_sequences_total'],
      ['status', 'oracle_fixture', 'probes', 'chat_probes'],
      ['status', 'mode', 'inspection_generated_at', 'host', 'inspector', 'source', 'range', 'oracle', 'observed', 'result', 'scope'],
    ),
    blocked: keyVariants(
      ['status', 'reason'],
      ['status', 'command', 'reason', 'probes'],
      ['status', 'command', 'reason', 'probes', 'chat_probes'],
      ['status', 'mode', 'error_code', 'reason'],
    ),
    fail: keyVariants(
      ['status', 'command', 'reason', 'probes'],
      ['status', 'command', 'reason', 'stderr', 'probes'],
      ['status', 'command', 'reason', 'probes', 'chat_probes'],
      ['status', 'command', 'reason', 'stderr', 'probes', 'chat_probes'],
      ['status', 'oracle_fixture', 'probes', 'chat_probes', 'reason'],
      ['status', 'mode', 'error_code', 'reason'],
      ['status', 'mode', 'inspection_generated_at', 'host', 'inspector', 'source', 'range', 'oracle', 'observed', 'result', 'scope', 'error_code', 'reason'],
    ),
    skipped: keyVariants(['status', 'reason', 'required']),
  }),
  template: Object.freeze({
    pass: keyVariants(
      ['status', 'evidence'],
      ['status', 'fixture', 'guard', 'prompt_token_ids_match', 'rendered_bytes_match', 'shape'],
    ),
    blocked: keyVariants(
      ['status', 'reason'],
      ['status', 'oracle_fixture', 'reason'],
      ['status', 'mode', 'error_code', 'reason', 'preparation'],
      ['status', 'mode', 'error_code', 'reason', 'preparation', 'source', 'range', 'inspector', 'oracle', 'template', 'cases', 'scope'],
    ),
    fail: keyVariants(
      ['status', 'reason'],
      ['status', 'mode', 'error_code', 'reason', 'preparation'],
    ),
    skipped: keyVariants(['status', 'reason', 'required']),
  }),
  load_smoke: Object.freeze({
    pass: keyVariants(
      ['status', 'evidence'],
      ['status', 'command', 'receipt', 'stderr'],
    ),
    blocked: keyVariants(
      ['status', 'reason'],
      ['status', 'load_and_generation_ready', 'runnable_smoke_refusal', 'selected_backend', 'reason'],
      ['status', 'command', 'reason'],
    ),
    fail: keyVariants(['status', 'reason'], ['status', 'command', 'reason']),
    skipped: keyVariants(['status', 'reason', 'required']),
  }),
  parity: Object.freeze({
    pass: keyVariants(
      ['status', 'evidence'],
      ['status', 'evidence', 'note'],
      ['status', 'oracle_fixture', 'probes'],
    ),
    blocked: keyVariants(
      ['status', 'reason'],
      ['status', 'probes', 'reason'],
      ['status', 'command', 'reason', 'probes'],
    ),
    fail: keyVariants(
      ['status', 'matched_prompts', 'mode', 'prompts', 'total_prompts'],
      ['status', 'oracle_fixture', 'probes', 'reason'],
      ['status', 'command', 'reason', 'probes'],
      ['status', 'command', 'reason', 'stderr', 'probes'],
    ),
    skipped: keyVariants(['status', 'reason', 'required']),
  }),
  api_webui: Object.freeze({
    pass: keyVariants(['status', 'evidence'], ['status', 'evidence', 'note']),
    blocked: keyVariants(['status', 'reason'], ['status', 'api_observations', 'reason']),
    fail: keyVariants(['status', 'reason']),
    skipped: keyVariants(['status', 'reason', 'required']),
  }),
  context: Object.freeze({
    pass: keyVariants(['status', 'evidence'], ['status', 'evidence', 'note']),
    blocked: keyVariants(['status', 'reason']),
    fail: keyVariants(['status', 'reason']),
    skipped: keyVariants(['status', 'reason', 'required']),
  }),
})

function forbiddenCredentialKey(key) {
  const raw = String(key).toLowerCase()
  const normalized = raw.replace(/[^a-z0-9]/g, '')
  if (CREDENTIAL_KEY_LITERALS.has(normalized)) return true
  if (['secret', 'secrets', 'password', 'passwords', 'credential', 'credentials']
    .some((suffix) => normalized.endsWith(suffix))) return true
  if (/tokens?$/.test(normalized)
    && !/(?:bos|eos|prompt|completion|input|output|generated|total|cached|reasoning|max)tokens?$/.test(normalized)) {
    return true
  }
  return ['authorization', 'downloadurl', 'downloaduri', 'signedurl', 'signeduri', 'apikey', 'clientsecret']
    .some((fragment) => normalized.includes(fragment))
    || /(?:^|[_.-])(?:access|api|auth|bearer|hf|id|refresh|session)[_.-]?token(?:$|[_.-])/.test(raw)
}

function stringContainsCredentialMaterial(value) {
  return AUTHORIZATION_VALUE_RE.test(value)
    || HF_TOKEN_VALUE_RE.test(value)
    || CREDENTIAL_ASSIGNMENT_VALUE_RE.test(value)
}

const URL_COMPONENT_MAX_CHARS = 1_024
const URL_COMPONENT_MAX_DECODE_PASSES = 16

function repeatedlyDecodeUrlComponent(value) {
  let decoded = String(value)
  if (decoded.length > URL_COMPONENT_MAX_CHARS) return null
  for (let attempt = 0; attempt < URL_COMPONENT_MAX_DECODE_PASSES; attempt += 1) {
    if (!decoded.includes('%')) return decoded
    let next
    try { next = decodeURIComponent(decoded) } catch { return null }
    if (next.length > URL_COMPONENT_MAX_CHARS || next === decoded) return null
    decoded = next
  }
  return decoded.includes('%') ? null : decoded
}

function forbiddenCredentialQueryKey(key) {
  const decoded = repeatedlyDecodeUrlComponent(key)
  if (decoded === null) return true
  const normalized = decoded.toLowerCase().replace(/[^a-z0-9]/g, '')
  return forbiddenCredentialKey(decoded)
    || [
      'auth',
      'authentication',
      'authorization',
      'credential',
      'credentials',
      'googleaccessid',
      'keypairid',
      'sig',
      'signature',
      'signed',
    ].includes(normalized)
    || normalized.includes('signature')
    || normalized.endsWith('accesskeyid')
}

function stringContainsCredentialBearingUrl(value) {
  HTTP_URL_RE.lastIndex = 0
  for (const match of value.matchAll(HTTP_URL_RE)) {
    const rawUrl = match[0].replace(/[),.;:\]}]+$/g, '')
    let parsed
    try { parsed = new URL(rawUrl) } catch { continue }
    if (!['http:', 'https:'].includes(parsed.protocol)) continue
    if (parsed.username !== '' || parsed.password !== '') return true
    for (const key of parsed.searchParams.keys()) {
      if (forbiddenCredentialQueryKey(key)) return true
    }
  }
  return false
}

function scanReportPrivacy(root) {
  const result = {
    absolute_local_path: false,
    credential_key: false,
    credential_value: false,
    candidate_url_value: false,
    budget_exceeded: false,
  }
  const stack = [{
    value: root,
    depth: 0,
    routeContext: false,
    commandContext: false,
  }]
  const seen = new WeakSet()
  let nodes = 0
  let stringChars = 0
  while (stack.length) {
    const item = stack.pop()
    nodes += 1
    if (nodes > PRIVACY_MAX_NODES || item.depth > PRIVACY_MAX_DEPTH) {
      result.budget_exceeded = true
      break
    }
    if (typeof item.value === 'string') {
      stringChars += item.value.length
      if (stringChars > PRIVACY_MAX_STRING_CHARS) {
        result.budget_exceeded = true
        break
      }
      if (!result.absolute_local_path && containsAbsoluteLocalPath(item.value, item)) {
        result.absolute_local_path = true
      }
      if (!result.credential_value
        && (stringContainsCredentialMaterial(item.value)
          || stringContainsCredentialBearingUrl(item.value))) {
        result.credential_value = true
      }
      if (!result.candidate_url_value
        && (CANDIDATE_HTTP_URL_VALUE_RE.test(item.value) || FILE_URI_RE.test(item.value))) {
        result.candidate_url_value = true
      }
      continue
    }
    if (!item.value || typeof item.value !== 'object') continue
    if (seen.has(item.value)) {
      result.budget_exceeded = true
      break
    }
    seen.add(item.value)
    if (Array.isArray(item.value)) {
      if (item.value.length > PRIVACY_MAX_NODES - nodes - stack.length) {
        result.budget_exceeded = true
        break
      }
      for (let index = item.value.length - 1; index >= 0; index -= 1) {
        stack.push({
          value: item.value[index],
          depth: item.depth + 1,
          routeContext: item.routeContext,
          commandContext: item.commandContext,
        })
      }
      continue
    }
    const keys = Object.keys(item.value)
    if (keys.length > PRIVACY_MAX_NODES - nodes - stack.length) {
      result.budget_exceeded = true
      break
    }
    for (let index = keys.length - 1; index >= 0; index -= 1) {
      const key = keys[index]
      if (forbiddenCredentialKey(key)) result.credential_key = true
      stack.push({
        value: item.value[key],
        depth: item.depth + 1,
        routeContext: item.routeContext || ROUTE_FIELD_RE.test(key),
        commandContext: item.commandContext || COMMAND_FIELD_RE.test(key),
      })
    }
  }
  return result
}

function reportContainsAbsoluteLocalPath(value) {
  const scan = scanReportPrivacy(value)
  return scan.absolute_local_path || scan.budget_exceeded
}

function validCandidateFile(value) {
  if (typeof value !== 'string' || !value || value.length > 1024
    || value.startsWith('/') || value.endsWith('/') || value.includes('\\')
    || !value.toLowerCase().endsWith('.gguf')) return false
  return value.split('/').every((segment) => CANDIDATE_FILE_SEGMENT_RE.test(segment))
}

function rejectUnexpectedKeys(value, allowed, path, fail) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    fail(path, 'expected a compact object')
    return
  }
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) fail(`${path}.${key}`, 'unexpected field in compact candidate report')
  }
}

function validIsoTimestamp(value) {
  if (typeof value !== 'string') return false
  const time = Date.parse(value)
  return Number.isFinite(time) && new Date(time).toISOString() === value
}

function validCompactToken(value) {
  return typeof value === 'string' && /^[A-Za-z0-9_.:+()-]{1,128}$/.test(value)
}

function validFailureText(value) {
  return typeof value === 'string' && value.length > 0 && value.length <= 512
    && !/[\r\n\0]/.test(value)
}

function exactStringArray(value, expected) {
  return Array.isArray(value)
    && value.length === expected.length
    && value.every((item, index) => typeof item === 'string' && item === expected[index])
}

function compactValueDescription(value) {
  if (value === null || ['string', 'number', 'boolean', 'undefined'].includes(typeof value)) {
    return String(value)
  }
  return Array.isArray(value) ? '<array>' : '<object>'
}

function isPowerOfTwo(value) {
  if (!Number.isSafeInteger(value) || value <= 0) return false
  const candidate = BigInt(value)
  return (candidate & (candidate - 1n)) === 0n
}

function validateCandidateAccess(access, path, fail) {
  rejectUnexpectedKeys(access, new Set(['gated', 'private', 'disabled']), path, fail)
  if (!access || !['gated', 'private', 'disabled']
    .every((field) => typeof access[field] === 'boolean')) {
    fail(path, 'expected the exact gated/private/disabled boolean projection')
    return false
  }
  return true
}

function validateBlockedCandidateStage(stage, path, expectedCode, fail) {
  rejectUnexpectedKeys(stage, new Set(['status', 'error_code', 'reason']), path, fail)
  if (stage?.status !== 'blocked') fail(`${path}.status`, 'expected blocked')
  if (stage?.error_code !== expectedCode) fail(`${path}.error_code`, `expected ${expectedCode}`)
  if (!validFailureText(stage?.reason)) fail(`${path}.reason`, 'expected compact non-empty reason')
}

function publicCandidateDigest(source) {
  return createHash('sha256').update(JSON.stringify({
    repo: source.repo,
    file: source.file,
    revision: source.requested_revision,
  })).digest('hex')
}

function canonicalJson(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map((item) => canonicalJson(item)).join(',')}]`
  return `{${Object.keys(value).sort().map((key) => (
    `${JSON.stringify(key)}:${canonicalJson(value[key])}`
  )).join(',')}}`
}

function validateLegacyPromotion(report, fail) {
  if (report.phase !== 1
    || report.row_id !== LEGACY_PROMOTION_ROW_ID
    || report.source_head !== LEGACY_PROMOTION_SOURCE_HEAD
    || report.runtime_head !== LEGACY_PROMOTION_RUNTIME_HEAD
    || report.artifact?.sha256 !== LEGACY_PROMOTION_ARTIFACT_SHA256
    || report.support_decision !== LEGACY_PROMOTION_SUPPORT_DECISION
    || report.overall_status !== 'pass') {
    fail('legacy_promotion', 'legacy promotion identity does not match the one pinned exact-row exception')
  }
  const digest = createHash('sha256').update(canonicalJson(report)).digest('hex')
  if (digest !== LEGACY_PROMOTION_CANONICAL_SHA256) {
    fail('legacy_promotion', 'legacy promotion report does not match its canonical pinned evidence contract')
  }
}

function validateCandidateQualification(report, fail) {
  rejectUnexpectedKeys(report, new Set([
    'schema',
    'generated_at',
    'phase',
    'qualification_mode',
    'row_id',
    'candidate',
    'source_head',
    'source_dirty',
    'source_tracked_dirty',
    'source_inspection',
    'host',
    'backend',
    'artifact',
    'stages',
    'overall_status',
    'support_decision',
    'support_claim',
    'scope',
  ]), 'report', fail)
  if (report.phase !== 2) fail('phase', 'selector candidate reports must be Phase 2')
  if (!validIsoTimestamp(report.generated_at)) fail('generated_at', 'expected a canonical ISO timestamp')
  if (report.support_decision !== 'hold_unrostered_header_candidate') {
    fail('support_decision', 'selector candidates must remain on the unrostered HOLD')
  }
  if (report.support_claim !== false) fail('support_claim', 'selector candidates cannot make a support claim')
  if (report.overall_status !== 'blocked' && report.overall_status !== 'fail') {
    fail('overall_status', 'selector candidates can only be blocked or fail')
  }
  if (report.source_inspection === 'observed') {
    if (!REVISION_RE.test(report.source_head || '')) fail('source_head', 'observed provenance requires a 40-hex source HEAD')
    if (typeof report.source_dirty !== 'boolean') fail('source_dirty', 'observed provenance requires a boolean dirty state')
    if (typeof report.source_tracked_dirty !== 'boolean') {
      fail('source_tracked_dirty', 'observed provenance requires a boolean tracked-dirty state')
    } else if (report.source_tracked_dirty && report.source_dirty !== true) {
      fail('source_tracked_dirty', 'tracked-dirty provenance must also report the workspace dirty')
    }
  } else if (report.source_inspection === 'unknown') {
    if (report.source_head !== null || report.source_dirty !== null || report.source_tracked_dirty !== null) {
      fail('source_inspection', 'unknown provenance must fail closed with null HEAD and dirty states')
    }
  } else {
    fail('source_inspection', 'expected observed or unknown')
  }
  if (report.artifact?.local_path_redacted !== true
    || report.artifact?.full_artifact_download !== 'not_run'
    || report.artifact?.full_artifact_sha256 !== 'not_run'
    || Object.hasOwn(report.artifact || {}, 'path')) {
    fail('artifact', 'selector candidates must not contain or imply a local/full artifact')
  }
  rejectUnexpectedKeys(report.artifact, new Set([
    'local_path_redacted',
    'full_artifact_download',
    'full_artifact_sha256',
  ]), 'artifact', fail)
  rejectUnexpectedKeys(report.host, new Set([
    'hostname_redacted',
    'platform',
    'release',
    'arch',
  ]), 'host', fail)
  if (report.host?.hostname_redacted !== true
    || !validCompactToken(report.host?.platform)
    || !validCompactToken(report.host?.release)
    || !validCompactToken(report.host?.arch)) {
    fail('host', 'expected a scrubbed compact host projection')
  }
  rejectUnexpectedKeys(report.backend, new Set(['serve', 'env']), 'backend', fail)
  if (report.backend?.serve !== 'not_run'
    || !report.backend?.env
    || typeof report.backend.env !== 'object'
    || Array.isArray(report.backend.env)
    || Object.keys(report.backend.env).length !== 0) {
    fail('backend', 'selector candidates cannot configure or imply a runtime lane')
  }
  rejectUnexpectedKeys(report.scope, new Set([
    'source_resolution',
    'header_inspection',
    'full_artifact_download',
    'full_artifact_sha256',
    'tensor_payload_interpretation',
    'tokenizer',
    'template',
    'load',
    'generation',
    'api_webui',
    'context',
    'support_claim',
  ]), 'scope', fail)
  for (const [field, expected] of [
    ['source_resolution', 'live_huggingface'],
    ['header_inspection', 'bounded_partial_range'],
    ['full_artifact_download', 'not_run'],
    ['full_artifact_sha256', 'not_run'],
    ['tensor_payload_interpretation', 'not_run'],
    ['tokenizer', 'not_run'],
    ['template', 'not_run'],
    ['load', 'not_run'],
    ['generation', 'not_run'],
    ['api_webui', 'not_run'],
    ['context', 'not_run'],
  ]) {
    if (report.scope?.[field] !== expected) fail(`scope.${field}`, `expected ${expected}`)
  }
  if (report.scope?.support_claim !== false) fail('scope.support_claim', 'expected false')

  const stages = report.stages || {}
  rejectUnexpectedKeys(stages, CANDIDATE_STAGE_NAMES, 'stages', fail)
  if (!['pass', 'fail', 'blocked'].includes(stages.source?.status)) {
    fail('stages.source.status', 'selector source resolution cannot be skipped')
  }
  if (!['pass', 'fail', 'blocked'].includes(stages.metadata?.status)) {
    fail('stages.metadata.status', 'selector header inspection cannot be skipped')
  }
  for (const [name, code] of Object.entries(CANDIDATE_DOWNSTREAM_CODES)) {
    validateBlockedCandidateStage(stages[name], `stages.${name}`, code, fail)
  }

  const source = stages.source
  const candidate = report.candidate || {}
  if (source?.status === 'pass') {
    rejectUnexpectedKeys(source, new Set([
      'status',
      'resolution',
      'repo',
      'file',
      'requested_revision',
      'revision',
      'size_bytes',
      'sha256',
      'license',
      'access',
    ]), 'stages.source', fail)
    if (source.resolution !== 'live_huggingface') fail('stages.source.resolution', 'expected live_huggingface')
    if (!CANDIDATE_REPO_RE.test(source.repo || '')) fail('stages.source.repo', 'invalid public Hugging Face repo id')
    if (!validCandidateFile(source.file)) fail('stages.source.file', 'invalid public GGUF file selector')
    if (!REVISION_RE.test(source.revision || '')) fail('stages.source.revision', 'expected immutable revision')
    if (candidate.requested_revision_pinned) {
      if (!REVISION_RE.test(source.requested_revision || '') || source.requested_revision !== source.revision) {
        fail('stages.source.requested_revision', 'pinned request must equal the resolved revision')
      }
    } else if (source.requested_revision !== null) {
      fail('stages.source.requested_revision', 'an unpinned selector must record null before the immutable resolution')
    }
    if (!Number.isSafeInteger(source.size_bytes) || source.size_bytes <= 0) {
      fail('stages.source.size_bytes', 'candidate source must have a positive exact byte size')
    }
    if (!SHA256_RE.test(source.sha256 || '')) fail('stages.source.sha256', 'expected LFS SHA-256')
    if (!CANDIDATE_LICENSE_RE.test(source.license || '')) fail('stages.source.license', 'invalid license identity')
    if (validateCandidateAccess(source.access, 'stages.source.access', fail)
      && (source.access.private || source.access.disabled)) {
      fail('stages.source.access', 'private or disabled sources cannot pass selector qualification')
    }
  } else if (source?.status === 'blocked' || source?.status === 'fail') {
    const accessBearing = source.error_code === 'private_source_not_persisted'
      || source.error_code === 'source_disabled'
    rejectUnexpectedKeys(source, new Set([
      'status',
      'resolution',
      'error_code',
      'reason',
      ...(accessBearing ? ['access', 'selector_redacted'] : []),
    ]), 'stages.source', fail)
    if (source.resolution !== 'live_huggingface') fail('stages.source.resolution', 'expected live_huggingface')
    if (!/^[a-z][a-z0-9_]*(?:_[0-9]{3})?$/.test(source.error_code || '')) {
      fail('stages.source.error_code', 'expected a compact source error code')
    }
    const expectedSourceStatus = /^http_[0-9]{3}$/.test(source.error_code || '')
      ? 'blocked'
      : CANDIDATE_SOURCE_ERROR_STATUSES[source.error_code]
    if (!expectedSourceStatus) {
      fail('stages.source.error_code', 'unsupported selector source error code')
    } else if (source.status !== expectedSourceStatus) {
      fail('stages.source.status', `${source.error_code} requires ${expectedSourceStatus}`)
    }
    if (!validFailureText(source.reason)) fail('stages.source.reason', 'expected compact non-empty reason')
    if (accessBearing) {
      const accessValid = validateCandidateAccess(source.access, 'stages.source.access', fail)
      if (source.selector_redacted !== true) fail('stages.source.selector_redacted', 'expected true')
      if (accessValid && source.error_code === 'private_source_not_persisted' && !source.access.private) {
        fail('stages.source.access.private', 'private-source reports must project private=true')
      }
      if (accessValid && source.error_code === 'source_disabled'
        && (source.access.private || !source.access.disabled)) {
        fail('stages.source.access', 'disabled-source reports require private=false and disabled=true')
      }
    }
  }

  if (source?.status === 'pass') {
    const idMatch = CANDIDATE_ID_RE.exec(report.row_id || '')
    if (!idMatch) fail('row_id', 'a public passing selector requires a safe selector digest id')
    rejectUnexpectedKeys(candidate, new Set([
      'identity_mode',
      'selector_id',
      'selector_sha256',
      'selector_redacted',
      'requested_revision_pinned',
      'roster_membership',
    ]), 'candidate', fail)
    if (candidate.identity_mode !== 'public_selector_digest') {
      fail('candidate.identity_mode', 'a public passing source requires public_selector_digest')
    }
    const digestibleSource = CANDIDATE_REPO_RE.test(source.repo || '')
      && validCandidateFile(source.file)
      && (source.requested_revision === null || REVISION_RE.test(source.requested_revision || ''))
    if (digestibleSource) {
      const expectedDigest = publicCandidateDigest(source)
      const expectedId = `hf_selector_${expectedDigest.slice(0, 24)}`
      if (candidate.selector_sha256 !== expectedDigest) {
        fail('candidate.selector_sha256', 'must equal the digest recomputed from the public source selector')
      }
      if (candidate.selector_id !== expectedId || report.row_id !== expectedId) {
        fail('candidate.selector_id', 'candidate id and row id must bind to the recomputed selector digest')
      }
    } else {
      fail('candidate.selector_sha256', 'cannot derive a selector digest from an invalid public source identity')
    }
    if (candidate.requested_revision_pinned !== (source.requested_revision !== null)) {
      fail('candidate.requested_revision_pinned', 'must match requested_revision presence')
    }
  } else {
    const rowMatch = CANDIDATE_RUN_ID_RE.exec(report.row_id || '')
    const uuidMatch = UUID_RE.exec(candidate.run_id || '')
    rejectUnexpectedKeys(candidate, new Set([
      'identity_mode',
      'run_id',
      'selector_redacted',
      'roster_membership',
    ]), 'candidate', fail)
    if (candidate.identity_mode !== 'opaque_run') {
      fail('candidate.identity_mode', 'nonpassing or nonpublic sources require opaque_run identity')
    }
    if (!rowMatch) fail('row_id', 'nonpassing or nonpublic sources require an opaque run id')
    if (!uuidMatch) {
      fail('candidate.run_id', 'expected a lowercase RFC 4122 version-4 UUID')
    } else if (rowMatch && uuidMatch.slice(1).join('') !== rowMatch[1]) {
      fail('candidate.run_id', 'opaque row id must bind exactly to the run UUID')
    }
  }
  if (candidate.selector_redacted !== true) {
    fail('candidate.selector_redacted', 'selector inputs must remain redacted in candidate metadata')
  }
  if (candidate.roster_membership !== 'absent_from_phase1_roster') {
    fail('candidate.roster_membership', 'candidate mode is only valid outside the Phase 1 roster')
  }

  const metadata = stages.metadata
  if (metadata?.status === 'pass') {
    rejectUnexpectedKeys(metadata, new Set([
      'status',
      'mode',
      'assessment',
      'inspection_generated_at',
      'host',
      'inspector',
      'source',
      'range',
      'observed',
      'scope',
    ]), 'stages.metadata', fail)
    if (source?.status !== 'pass') fail('stages.metadata', 'metadata cannot pass without a passing source stage')
    if (!validIsoTimestamp(metadata.inspection_generated_at)) {
      fail('stages.metadata.inspection_generated_at', 'expected a canonical ISO timestamp')
    }
    if (metadata.mode !== 'remote_immutable_prefix'
      || metadata.assessment !== 'bounded_header_descriptor_inspection_only') {
      fail('stages.metadata.assessment', 'metadata pass must remain bounded descriptor-only evidence')
    }
    for (const field of ['repo', 'file', 'revision', 'size_bytes', 'sha256']) {
      if (metadata.source?.[field] !== source?.[field]) {
        fail(`stages.metadata.source.${field}`, 'must equal the passing source identity')
      }
    }
    rejectUnexpectedKeys(metadata.host, new Set([
      'hostname_redacted',
      'platform',
      'release',
      'arch',
    ]), 'stages.metadata.host', fail)
    rejectUnexpectedKeys(metadata.source, new Set([
      'repo',
      'file',
      'revision',
      'size_bytes',
      'sha256',
    ]), 'stages.metadata.source', fail)
    rejectUnexpectedKeys(metadata.observed, new Set([
      'gguf_version',
      'architecture',
      'tokenizer_model',
      'tokenizer_pre',
      'headline_quant',
      'tensor_count',
      'metadata_count',
      'alignment',
      'data_start_offset',
      'tensor_payload_n_bytes',
      'tensor_inventory_sha256',
      'tensor_type_counts',
    ]), 'stages.metadata.observed', fail)
    rejectUnexpectedKeys(metadata.scope, new Set([
      'prefix_sha256',
      'tensor_payload',
      'opaque_tensor_payload_prefix_bytes',
      'full_artifact_sha256',
      'tensor_payload_interpretation',
      'load',
      'generation',
      'runtime_compatibility',
      'support_claim',
    ]), 'stages.metadata.scope', fail)
    if (metadata.host?.hostname_redacted !== true
      || !validCompactToken(metadata.host?.platform)
      || !validCompactToken(metadata.host?.release)
      || !validCompactToken(metadata.host?.arch)) {
      fail('stages.metadata.host', 'metadata host identity must be a scrubbed compact projection')
    }
    const observed = metadata.observed || {}
    const safeObserved = (value) => value === null
      || (typeof value === 'string' && /^[A-Za-z0-9_.:+-]{1,128}$/.test(value))
    if (![2, 3].includes(observed.gguf_version)
      || !safeObserved(observed.architecture)
      || !safeObserved(observed.tokenizer_model)
      || !safeObserved(observed.tokenizer_pre)
      || !safeObserved(observed.headline_quant)
      || !Number.isSafeInteger(observed.tensor_count) || observed.tensor_count < 0
      || !Number.isSafeInteger(observed.metadata_count) || observed.metadata_count < 0
      || !isPowerOfTwo(observed.alignment)
      || !Number.isSafeInteger(observed.data_start_offset) || observed.data_start_offset < 24
      || observed.data_start_offset % observed.alignment !== 0
      || !Number.isSafeInteger(observed.tensor_payload_n_bytes) || observed.tensor_payload_n_bytes < 0
      || !SHA256_RE.test(observed.tensor_inventory_sha256 || '')) {
      fail('stages.metadata.observed', 'expected compact, scrubbed GGUF descriptor observations')
    }
    const sourceSize = Number.isSafeInteger(source?.size_bytes) ? source.size_bytes : -1
    const range = metadata.range || {}
    const contentRange = range.content_range || {}
    rejectUnexpectedKeys(range, new Set([
      'requested_bytes',
      'received_bytes',
      'content_range',
      'prefix_sha256',
    ]), 'stages.metadata.range', fail)
    rejectUnexpectedKeys(contentRange, new Set(['start', 'end', 'total']), 'stages.metadata.range.content_range', fail)
    if (!Number.isSafeInteger(range.requested_bytes) || range.requested_bytes <= 0
      || range.requested_bytes > 64 * 1024 * 1024
      || range.received_bytes !== range.requested_bytes
      || range.received_bytes >= sourceSize
      || contentRange.start !== 0
      || contentRange.end + 1 !== range.received_bytes
      || contentRange.total !== sourceSize
      || !SHA256_RE.test(range.prefix_sha256 || '')) {
      fail('stages.metadata.range', 'must be an exact strict partial range bound to the source size')
    }
    const payloadEnd = observed.data_start_offset + observed.tensor_payload_n_bytes
    if (observed.data_start_offset >= sourceSize
      || observed.data_start_offset >= range.received_bytes
      || !Number.isSafeInteger(payloadEnd)
      || payloadEnd !== sourceSize) {
      fail('stages.metadata.observed', 'descriptor offsets and payload bytes must close exactly over the bounded source identity')
    }
    if (!Object.hasOwn(observed, 'tensor_type_counts')) {
      fail('stages.metadata.observed.tensor_type_counts', 'candidate receipts require exact tensor-type counts')
    } else {
      const counts = observed.tensor_type_counts
      if (!counts || typeof counts !== 'object' || Array.isArray(counts)) {
        fail('stages.metadata.observed.tensor_type_counts', 'expected a compact tensor-type count object')
      } else {
        let total = 0
        for (const [name, count] of Object.entries(counts)) {
          if (!/^[A-Za-z][A-Za-z0-9_]{0,31}$/.test(name)
            || !Number.isSafeInteger(count) || count <= 0) {
            fail(`stages.metadata.observed.tensor_type_counts.${name}`, 'expected a positive safe count')
          } else {
            total += count
          }
        }
        if (!Number.isSafeInteger(total) || total !== observed.tensor_count) {
          fail('stages.metadata.observed.tensor_type_counts', 'type counts must sum exactly to tensor_count')
        }
      }
    }
    const inspector = metadata.inspector || {}
    rejectUnexpectedKeys(inspector, new Set([
      'version',
      'binary_sha256',
      'source_head',
      'binary_commit_abbrev',
      'binary_reports_dirty',
      'binary_matches_source_head',
      'clean_current_head',
      'binary_path_redacted',
      'command',
    ]), 'stages.metadata.inspector', fail)
    const versionMatch = /^camelid [A-Za-z0-9._+()-]+-g([0-9a-f]{7,40})$/.exec(inspector.version || '')
    if (!versionMatch
      || !SHA256_RE.test(inspector.binary_sha256 || '')
      || !REVISION_RE.test(inspector.source_head || '')
      || inspector.source_head !== report.source_head
      || inspector.binary_commit_abbrev !== versionMatch?.[1]
      || !inspector.source_head?.startsWith(versionMatch?.[1] || '<invalid>')
      || inspector.binary_reports_dirty !== false
      || inspector.binary_matches_source_head !== true
      || inspector.clean_current_head !== true
      || inspector.binary_path_redacted !== true
      || !exactStringArray(inspector.command, [
        '<camelid>',
        'inspect-prefix',
        '<remote-gguf-prefix>',
        '--declared-len',
        String(sourceSize),
      ])) {
      fail('stages.metadata.inspector', 'must be a clean current-HEAD inspector identity')
    }
    if (report.source_inspection !== 'observed'
      || !REVISION_RE.test(report.source_head || '')
      || report.source_tracked_dirty !== false) {
      fail('source_tracked_dirty', 'metadata PASS requires observed 40-hex, tracked-clean factory provenance')
    }
    for (const [field, expected] of [
      ['prefix_sha256', 'all_received_prefix_bytes'],
      ['tensor_payload', 'partially_range_fetched_opaque'],
      ['full_artifact_sha256', 'not_run'],
      ['tensor_payload_interpretation', 'not_run'],
      ['load', 'not_run'],
      ['generation', 'not_run'],
      ['runtime_compatibility', 'not_run'],
    ]) {
      if (metadata.scope?.[field] !== expected) {
        fail(`stages.metadata.scope.${field}`, `expected ${expected}`)
      }
    }
    const expectedOpaqueBytes = Math.max(0, range.received_bytes - observed.data_start_offset)
    if (expectedOpaqueBytes <= 0
      || metadata.scope?.opaque_tensor_payload_prefix_bytes !== expectedOpaqueBytes) {
      fail('stages.metadata.scope.opaque_tensor_payload_prefix_bytes', 'must equal a positive count of received opaque bytes after data_start_offset')
    }
    if (metadata.scope?.support_claim !== false) fail('stages.metadata.scope.support_claim', 'expected false')
  } else if (metadata?.status === 'blocked' || metadata?.status === 'fail') {
    rejectUnexpectedKeys(metadata, new Set([
      'status',
      'mode',
      'error_code',
      'reason',
    ]), 'stages.metadata', fail)
    if (metadata.mode !== 'remote_immutable_prefix') fail('stages.metadata.mode', 'expected remote_immutable_prefix')
    if (!/^[a-z][a-z0-9_]*$/.test(metadata.error_code || '')) {
      fail('stages.metadata.error_code', 'expected a compact header error code')
    }
    const expectedMetadataStatus = CANDIDATE_METADATA_ERROR_STATUSES[metadata.error_code]
    if (!expectedMetadataStatus) {
      fail('stages.metadata.error_code', 'unsupported selector header error code')
    } else if (metadata.status !== expectedMetadataStatus) {
      fail('stages.metadata.status', `${metadata.error_code} requires ${expectedMetadataStatus}`)
    }
    if (!validFailureText(metadata.reason)) fail('stages.metadata.reason', 'expected compact non-empty reason')
    if (source?.status !== 'pass'
      && (metadata.status !== 'blocked' || metadata.error_code !== 'header_source_preflight_blocked')) {
      fail('stages.metadata', 'a nonpassing source must close metadata with header_source_preflight_blocked')
    }
  }
}

function expectedOverall(stages) {
  const required = REQUIRED_STAGES.map((name) => stages[name]).filter((stage) => stage?.required !== false)
  if (required.some((stage) => stage?.status === 'fail')) return 'fail'
  if (required.some((stage) => stage?.status === 'blocked' || stage?.status === 'skipped')) return 'blocked'
  if (!required.some((stage) => stage?.status === 'pass')) return 'blocked'
  return 'pass'
}

function hasExactKeys(value, expected) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const keys = Object.keys(value)
  return keys.length === expected.size && keys.every((key) => expected.has(key))
}

function validateNonCandidateQualification(report, fail) {
  const stages = report?.stages || {}
  if (!GENERIC_ROOT_KEY_VARIANTS.some((variant) => hasExactKeys(report, variant))) {
    fail('report', 'generic v1 root fields do not match a closed qualification-report shape')
  }
  if (report?.overall_status === 'pass') {
    fail('overall_status', 'generic v1 PASS is unsupported; promotion requires a future stricter qualification mode or schema')
  }
  if (Object.hasOwn(report || {}, 'phase')) {
    if (!Number.isSafeInteger(report.phase) || report.phase < 1 || report.phase === 2) {
      fail('phase', 'phase-tagged generic reports require an explicit non-candidate phase')
    }
    if (typeof report.support_decision !== 'string'
      || !/^[a-z][a-z0-9_]{0,127}$/.test(report.support_decision)) {
      fail('support_decision', 'phase-tagged generic reports require a compact explicit support decision')
    }
    if (!Array.isArray(report.does_not_prove)
      || report.does_not_prove.length === 0
      || report.does_not_prove.some((item) => !validFailureText(item))) {
      fail('does_not_prove', 'phase-tagged generic reports require a non-empty compact evidence-limit list')
    }
  }
  for (const name of REQUIRED_STAGES) {
    const stage = stages[name]
    const variants = GENERIC_STAGE_KEY_VARIANTS[name]?.[stage?.status]
    if (variants && !variants.some((variant) => hasExactKeys(stage, variant))) {
      fail(`stages.${name}`, `fields do not match a closed generic ${stage.status} ${name} evidence shape`)
    }
    if (stage && Object.hasOwn(stage, 'evidence')
      && (!Array.isArray(stage.evidence)
        || stage.evidence.length === 0
        || stage.evidence.some((item) => !validFailureText(item)))) {
      fail(`stages.${name}.evidence`, 'generic evidence must be a non-empty compact string list')
    }
    if (stage && Object.hasOwn(stage, 'note')
      && !validFailureText(stage.note)) {
      fail(`stages.${name}.note`, 'generic evidence notes must be compact and non-empty')
    }
    if (stage?.status === 'skipped'
      && (stage.required !== false || !validFailureText(stage.reason))) {
      fail(`stages.${name}`, 'generic skipped stages require required=false and a compact non-empty reason')
    }
  }
}

function validateQualificationReport(report, label = 'report') {
  const errors = []
  const fail = (path, message) => errors.push(`${label}.${path}: ${message}`)
  const privacy = scanReportPrivacy(report)
  if (privacy.budget_exceeded) {
    fail('privacy', 'report privacy scan exceeded its depth, node, string, or acyclic-structure budget')
    return errors
  }
  if (report?.schema !== 'camelid.model-qualification-report/v1') {
    fail('schema', 'expected camelid.model-qualification-report/v1')
  }
  if (typeof report?.row_id !== 'string' || !report.row_id) fail('row_id', 'expected a non-empty row id')
  if (report?.source_dirty !== null && typeof report?.source_dirty !== 'boolean') {
    fail('source_dirty', 'expected boolean or null (unknown); failed inspection must not look clean')
  }
  if (Object.hasOwn(report || {}, 'source_tracked_dirty')
    && report?.source_tracked_dirty !== null
    && typeof report?.source_tracked_dirty !== 'boolean') {
    fail('source_tracked_dirty', 'expected boolean or null (unknown)')
  }
  if (report?.host?.hostname_redacted !== true) {
    fail('host.hostname_redacted', 'host identity must be redacted')
  }
  if (Object.hasOwn(report?.host || {}, 'hostname')) fail('host.hostname', 'raw hostname is forbidden')

  const stages = report?.stages || {}
  for (const name of REQUIRED_STAGES) {
    const stage = stages[name]
    if (!stage || typeof stage !== 'object' || Array.isArray(stage)) {
      fail(`stages.${name}`, 'missing stage object')
    } else if (!STATUSES.has(stage.status)) {
      fail(`stages.${name}.status`, `unsupported status ${compactValueDescription(stage.status)}`)
    } else if ((stage.status === 'fail' || stage.status === 'blocked')
      && typeof stage.reason !== 'string'
      && !Object.hasOwn(stage, 'prompts')) {
      fail(`stages.${name}.reason`, `${stage.status} stages need a reason or detailed probe evidence`)
    } else if (stage.status === 'skipped' && !validFailureText(stage.reason)) {
      fail(`stages.${name}.reason`, 'skipped stages need a compact non-empty reason')
    }
  }
  for (const extra of Object.keys(stages).filter((name) => !REQUIRED_STAGES.includes(name))) {
    fail(`stages.${extra}`, 'unexpected stage')
  }
  if (STATUSES.has(report?.overall_status)) {
    const expected = expectedOverall(stages)
    if (report.overall_status !== expected) {
      fail('overall_status', `${report.overall_status} does not match fail-closed stage result ${expected}`)
    }
  } else {
    fail('overall_status', `unsupported status ${compactValueDescription(report?.overall_status)}`)
  }

  if (privacy.absolute_local_path) {
    fail('privacy', 'report contains an absolute local path')
  }
  if (privacy.credential_value) {
    fail('privacy', 'report contains a forbidden credential or download value')
  }
  if (privacy.credential_key) {
    fail('privacy', 'report contains a forbidden credential or download field')
  }
  const legacyPromotion = report?.qualification_mode === LEGACY_PROMOTION_MODE
  const candidateShaped = !legacyPromotion && (CANDIDATE_REPORT_LABEL_RE.test(String(label))
    || Object.hasOwn(report || {}, 'qualification_mode')
    || CANDIDATE_ID_RE.test(report?.row_id || '')
    || CANDIDATE_RUN_ID_RE.test(report?.row_id || '')
    || Object.hasOwn(report || {}, 'candidate')
    || report?.phase === 2
    || report?.support_decision === 'hold_unrostered_header_candidate'
    || Object.hasOwn(report || {}, 'support_claim')
    || report?.scope?.header_inspection === 'bounded_partial_range'
    || report?.artifact?.full_artifact_download === 'not_run'
    || report?.stages?.artifact?.error_code === 'full_artifact_not_downloaded'
    || report?.stages?.metadata?.assessment === 'bounded_header_descriptor_inspection_only'
    || Object.values(report?.stages || {}).some((stage) => (
      typeof stage?.error_code === 'string' && stage.error_code.startsWith('candidate_')
    ))
    || (report?.stages?.source?.resolution === 'live_huggingface'
      && Object.hasOwn(report?.stages?.source || {}, 'requested_revision')))
  if (legacyPromotion) {
    validateLegacyPromotion(report, fail)
  } else if (candidateShaped) {
    if (privacy.candidate_url_value) {
      fail('privacy', 'selector candidate report contains a forbidden credential or download field')
    }
    if (report?.qualification_mode !== CANDIDATE_QUALIFICATION_MODE) {
      fail('qualification_mode', `candidate-shaped reports must declare ${CANDIDATE_QUALIFICATION_MODE}`)
    }
    validateCandidateQualification(report, fail)
  } else {
    validateNonCandidateQualification(report, fail)
  }
  return errors
}

async function collectReportPaths(inputs) {
  const requested = inputs.length ? inputs.map((path) => resolve(path)) : [resolve('qa/model-qualification')]
  const paths = []
  for (const path of requested) {
    const info = await stat(path)
    if (info.isDirectory()) {
      const entries = await readdir(path)
      paths.push(...entries.filter((entry) => entry.endsWith('-report.json')).map((entry) => resolve(path, entry)))
    } else {
      paths.push(path)
    }
  }
  return paths.sort()
}

async function main() {
  const paths = await collectReportPaths(process.argv.slice(2))
  if (!paths.length) throw new Error('no qualification reports found')
  const errors = []
  for (const path of paths) {
    const report = JSON.parse(await readFile(path, 'utf8'))
    errors.push(...validateQualificationReport(report, path))
  }
  if (errors.length) {
    console.error(`model qualification report check FAILED (${errors.length}):`)
    for (const error of errors) console.error(`  - ${error}`)
    process.exit(1)
  }
  console.log(JSON.stringify({ checked: paths.length, reports: paths }, null, 2))
}

export {
  REQUIRED_STAGES,
  containsAbsoluteLocalPath,
  expectedOverall,
  reportContainsAbsoluteLocalPath,
  validateQualificationReport,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
