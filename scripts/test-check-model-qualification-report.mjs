#!/usr/bin/env node
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readdir, readFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  expectedOverall,
  reportContainsAbsoluteLocalPath,
  validateQualificationReport,
} from './check-model-qualification-report.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const path = resolve(root, 'qa/model-qualification/qwen2.5-0.5b-q8-bootstrap-report.json')
const report = JSON.parse(await readFile(path, 'utf8'))
const mutated = (change) => {
  const candidate = structuredClone(report)
  change(candidate)
  return validateQualificationReport(candidate, 'test')
}

const reportDir = resolve(root, 'qa/model-qualification')
const committedReports = (await readdir(reportDir)).filter((name) => name.endsWith('-report.json'))
assert.ok(committedReports.length >= 2, 'the committed Phase 1 source/bootstrap reports must stay under validation')
for (const name of committedReports) {
  const candidate = JSON.parse(await readFile(resolve(reportDir, name), 'utf8'))
  assert.deepEqual(validateQualificationReport(candidate, name), [], `${name} must satisfy its closed report contract`)
}
const legacyPromotionReport = JSON.parse(await readFile(
  resolve(reportDir, 'lfm2.5-2.6b-q8-phase1-report.json'),
  'utf8',
))
assert.equal(legacyPromotionReport.qualification_mode, 'legacy_exact_row_promotion')
const genericPromotionForgery = structuredClone(legacyPromotionReport)
delete genericPromotionForgery.qualification_mode
assert.ok(
  validateQualificationReport(genericPromotionForgery, 'generic-promotion-forgery')
    .some((error) => error.includes('generic v1 PASS is unsupported')),
  'removing the exact legacy mode must restore the generic-v1 PASS prohibition',
)
const fakeLegacyMode = structuredClone(legacyPromotionReport)
fakeLegacyMode.qualification_mode = 'legacy_exact_row_promotion_fake'
assert.ok(
  validateQualificationReport(fakeLegacyMode, 'fake-legacy-mode')
    .some((error) => error.includes('candidate-shaped reports must declare')),
  'nearby or invented legacy mode names must fail closed',
)
for (const [label, mutateLegacy] of [
  ['support decision', (value) => { value.support_decision = 'promote_supported_exact_row_smoke_changed' }],
  ['stage evidence', (value) => { value.stages.source.evidence[0] += ' changed' }],
  ['stage note', (value) => { value.stages.parity.note += ' changed' }],
  ['artifact identity', (value) => { value.artifact.sha256 = '0'.repeat(64) }],
]) {
  const changed = structuredClone(legacyPromotionReport)
  mutateLegacy(changed)
  assert.ok(
    validateQualificationReport(changed, `legacy-${label}`)
      .some((error) => error.includes('canonical pinned evidence contract')
        || error.includes('pinned exact-row exception')),
    `a one-field ${label} mutation must invalidate the exact legacy promotion exception`,
  )
}
assert.equal(expectedOverall(report.stages), 'fail')
const allSkippedReport = structuredClone(report)
allSkippedReport.stages = Object.fromEntries(Object.keys(report.stages).map((name) => [name, {
  status: 'skipped',
  reason: '',
  required: false,
}]))
allSkippedReport.overall_status = 'pass'
assert.equal(
  expectedOverall(allSkippedReport.stages),
  'blocked',
  'zero required or passing gates must never derive PASS',
)
const allSkippedErrors = validateQualificationReport(allSkippedReport, 'all-skipped')
assert.ok(allSkippedErrors.some((error) => error.includes('generic v1 PASS is unsupported')))
assert.ok(allSkippedErrors.some((error) => error.includes('non-empty reason')))
const explainedSkippedReport = structuredClone(allSkippedReport)
for (const stage of Object.values(explainedSkippedReport.stages)) stage.reason = 'gate was deliberately not run'
assert.equal(expectedOverall(explainedSkippedReport.stages), 'blocked')
assert.ok(
  validateQualificationReport(explainedSkippedReport, 'explained-all-skipped')
    .some((error) => error.includes('does not match fail-closed stage result blocked')),
  'even explained all-skipped gates cannot produce overall PASS',
)
assert.ok(mutated((candidate) => { candidate.host.hostname = 'private-machine' }).some((error) => error.includes('raw hostname')))
const windowsPrivatePath = ['C:', 'Users', 'private', 'model.gguf'].join('\\')
assert.ok(mutated((candidate) => { candidate.artifact.path = windowsPrivatePath }).some((error) => error.includes('absolute local path')))
for (const privatePath of [
  '/tmp/private/model.gguf',
  '//server/private/model.gguf',
  '///tmp/private/model.gguf',
  '/var/tmp/private/model.gguf',
  '/mnt/models/private.gguf',
  'file://localhost/tmp/private/model.gguf',
]) {
  assert.ok(
    mutated((candidate) => { candidate.artifact.path = privatePath }).some((error) => error.includes('absolute local path')),
    'Unix local absolute paths must be rejected',
  )
}
assert.ok(mutated((candidate) => { candidate.stages.metadata.command = ['<camelid>', 'inspect', '/workspace/models/private.gguf'] }).some((error) => error.includes('absolute local path')))
for (const assignedPath of [
  '--model=/tmp/private/model.gguf',
  `--model=${windowsPrivatePath}`,
  '--model=file:///tmp/private/model.gguf',
]) {
  assert.ok(
    mutated((candidate) => { candidate.stages.metadata.command = assignedPath }).some((error) => error.includes('absolute local path')),
    'absolute command-assignment paths must be rejected',
  )
}
assert.ok(
  mutated((candidate) => { candidate.artifact.path = '/api/private/model.gguf' }).some((error) => error.includes('absolute local path')),
  'an artifact path must not bypass privacy checks by starting with an API-looking directory',
)
assert.ok(
  mutated((candidate) => { candidate.artifact.path = '/completion' }).some((error) => error.includes('absolute local path')),
  'a known route literal is safe only in a route/endpoint/command field',
)
assert.equal(
  reportContainsAbsoluteLocalPath({
    stages: {
      api_webui: {
        public_routes: [
      '/api/models',
      '/api/models/example-id/status',
      '/v1/chat/completions',
        ],
        command: [
      '/completion',
      '/health',
      '/apply-template',
      '/tokenize',
        ],
      },
    },
    source_url: 'https://huggingface.co/org/repo',
  }),
  false,
  'public URLs and API route literals must not be mistaken for host paths',
)
assert.ok(mutated((candidate) => { candidate.source_dirty = false; candidate.overall_status = 'pass' }).some((error) => error.includes('fail-closed')))
assert.ok(mutated((candidate) => { delete candidate.stages.context }).some((error) => error.includes('missing stage object')))
assert.ok(mutated((candidate) => {
  candidate.stages.api_webui.api_observations.Authorization = 'redacted'
}).some((error) => error.includes('forbidden credential or download field')),
'generic reports must reject nested Authorization keys even when their values are innocuous')
assert.ok(mutated((candidate) => {
  candidate.stages.api_webui.api_observations.nested = {
    access_token: 'redacted',
    nested_token: 'also-redacted',
  }
}).some((error) => error.includes('forbidden credential or download field')),
'generic reports must reject nested token-like key families even when their values are innocuous')
const percentEncodeAscii = (value) => [...value]
  .map((character) => `%${character.charCodeAt(0).toString(16).padStart(2, '0')}`)
  .join('')
const encodeQueryKeyLayers = (value, layers) => {
  let encoded = percentEncodeAscii(value)
  for (let layer = 1; layer < layers; layer += 1) encoded = encodeURIComponent(encoded)
  return encoded
}
const layeredCredentialUrls = Array.from({ length: 12 }, (_, index) => (
  `https://example.com/model?${encodeQueryKeyLayers('token', index + 1)}=SUPERSECRET123`
))
for (const credentialUrl of [
  'https://user:password@huggingface.co/example/model',
  'https://%75ser:%70assword@huggingface.co/example/model',
  'https://huggingface.co/example/model?%74%6f%6b%65%6e=SUPERSECRET123',
  'https://huggingface.co/example/model?%61%75%74%68=SUPERSECRET123',
  'https://huggingface.co/example/model?X-Amz-%53ignature=SUPERSECRET123',
  'https://example.com/model?%2525252574%252525256f%252525256b%2525252565%252525256e=SUPERSECRET123',
  'https://example.com/model?%ZZ=public',
  `https://example.com/model?${encodeQueryKeyLayers('revision', 20)}=main`,
  ...layeredCredentialUrls,
]) {
  assert.ok(
    mutated((candidate) => { candidate.stages.api_webui.reason = `public evidence: ${credentialUrl}` })
      .some((error) => error.includes('forbidden credential or download value')),
    `generic reports must reject credential-bearing URL ${credentialUrl}`,
  )
}
assert.deepEqual(
  mutated((candidate) => {
    candidate.stages.api_webui.reason = 'public evidence: https://huggingface.co/example/model%20card?rev%69sion=refs%2Fheads%2Fmain&download=true'
  }),
  [],
  'generic reports must continue to allow public HTTP URLs without userinfo or credential query parameters',
)

const publicSelector = {
  repo: 'example/model',
  file: 'weights/model-Q8_0.gguf',
  revision: null,
}
const selectorSha256 = createHash('sha256').update(JSON.stringify(publicSelector)).digest('hex')
const candidateId = `hf_selector_${selectorSha256.slice(0, 24)}`
const sourceHead = `12345678${'9'.repeat(32)}`
const blocked = (errorCode, reason) => ({ status: 'blocked', error_code: errorCode, reason })
const candidateReport = {
  schema: 'camelid.model-qualification-report/v1',
  generated_at: '2026-08-10T12:34:56.000Z',
  phase: 2,
  qualification_mode: 'unrostered_hf_selector',
  row_id: candidateId,
  candidate: {
    identity_mode: 'public_selector_digest',
    selector_id: candidateId,
    selector_sha256: selectorSha256,
    selector_redacted: true,
    requested_revision_pinned: false,
    roster_membership: 'absent_from_phase1_roster',
  },
  source_head: sourceHead,
  source_dirty: false,
  source_tracked_dirty: false,
  source_inspection: 'observed',
  host: { hostname_redacted: true, platform: 'win32', release: '10.0', arch: 'x64' },
  backend: { serve: 'not_run', env: {} },
  artifact: {
    local_path_redacted: true,
    full_artifact_download: 'not_run',
    full_artifact_sha256: 'not_run',
  },
  stages: {
    source: {
      status: 'pass',
      resolution: 'live_huggingface',
      repo: publicSelector.repo,
      file: publicSelector.file,
      requested_revision: null,
      revision: 'b'.repeat(40),
      size_bytes: 100,
      sha256: 'c'.repeat(64),
      license: 'apache-2.0',
      access: { gated: false, private: false, disabled: false },
    },
    artifact: blocked('full_artifact_not_downloaded', 'the full artifact was not downloaded'),
    metadata: {
      status: 'pass',
      mode: 'remote_immutable_prefix',
      assessment: 'bounded_header_descriptor_inspection_only',
      inspection_generated_at: '2026-08-10T12:34:56.000Z',
      host: { hostname_redacted: true, platform: 'win32', release: '10.0', arch: 'x64' },
      inspector: {
        version: 'camelid v0.6.1-27-g12345678',
        binary_sha256: 'd'.repeat(64),
        source_head: sourceHead,
        source_tracked_dirty: false,
        binary_commit_abbrev: '12345678',
        binary_reports_dirty: false,
        binary_matches_source_head: true,
        clean_current_head: true,
        binary_path_redacted: true,
        command: ['<camelid>', 'inspect-prefix', '<remote-gguf-prefix>', '--declared-len', '100'],
      },
      source: {
        repo: publicSelector.repo,
        file: publicSelector.file,
        revision: 'b'.repeat(40),
        size_bytes: 100,
        sha256: 'c'.repeat(64),
      },
      range: {
        requested_bytes: 99,
        received_bytes: 99,
        content_range: { start: 0, end: 98, total: 100 },
        prefix_sha256: 'e'.repeat(64),
      },
      observed: {
        gguf_version: 3,
        architecture: 'qwen2',
        tokenizer_model: 'gpt2',
        tokenizer_pre: 'qwen2',
        general_file_type: 7,
        declared_quantization: 'Q8_0',
        headline_quant: 'Q8_0',
        tensor_count: 4,
        metadata_count: 8,
        alignment: 32,
        data_start_offset: 96,
        tensor_payload_n_bytes: 4,
        tensor_inventory_sha256: 'f'.repeat(64),
        tensor_type_counts: { F32: 1, Q8_0: 3 },
      },
      scope: {
        prefix_sha256: 'all_received_prefix_bytes',
        tensor_payload: 'partially_range_fetched_opaque',
        opaque_tensor_payload_prefix_bytes: 3,
        full_artifact_sha256: 'not_run',
        tensor_payload_interpretation: 'not_run',
        load: 'not_run',
        generation: 'not_run',
        runtime_compatibility: 'not_run',
        support_claim: false,
      },
    },
    tokenizer: blocked('candidate_tokenizer_pack_unavailable', 'no exact tokenizer pack'),
    template: blocked('candidate_template_pack_unavailable', 'no exact template pack'),
    load_smoke: blocked('candidate_full_artifact_unavailable', 'no full artifact'),
    parity: blocked('candidate_oracle_pack_unavailable', 'no exact oracle pack'),
    api_webui: blocked('candidate_runtime_unqualified', 'runtime is unqualified'),
    context: blocked('candidate_context_unqualified', 'context is unqualified'),
  },
  overall_status: 'blocked',
  support_decision: 'hold_unrostered_header_candidate',
  support_claim: false,
  scope: {
    source_resolution: 'live_huggingface',
    header_inspection: 'bounded_partial_range',
    full_artifact_download: 'not_run',
    full_artifact_sha256: 'not_run',
    tensor_payload_interpretation: 'not_run',
    tokenizer: 'not_run',
    template: 'not_run',
    load: 'not_run',
    generation: 'not_run',
    api_webui: 'not_run',
    context: 'not_run',
    support_claim: false,
  },
}
const candidateErrors = (change) => {
  const value = structuredClone(candidateReport)
  change(value)
  return validateQualificationReport(value, 'candidate')
}
assert.deepEqual(validateQualificationReport(candidateReport, 'candidate'), [])
assert.ok(candidateErrors((value) => { value.support_claim = true }).some((error) => error.includes('support_claim')))
assert.ok(candidateErrors((value) => { value.stages.artifact.status = 'pass'; value.overall_status = 'pass' }).some((error) => error.includes('expected blocked')))
assert.ok(candidateErrors((value) => { value.stages.tokenizer.status = 'pass' }).some((error) => error.includes('expected blocked')))
assert.ok(candidateErrors((value) => { value.stages.source.access.private = true }).some((error) => error.includes('private or disabled')))
assert.ok(candidateErrors((value) => {
  value.stages.metadata.range.received_bytes = 100
  value.stages.metadata.range.requested_bytes = 100
  value.stages.metadata.range.content_range.end = 99
}).some((error) => error.includes('strict partial range')))
assert.ok(candidateErrors((value) => { value.stages.metadata.inspector.source_head = 'f'.repeat(40) }).some((error) => error.includes('current-HEAD')))
assert.ok(candidateErrors((value) => { value.stages.metadata.inspector.source_tracked_dirty = true }).some((error) => error.includes('current-HEAD')))
assert.ok(candidateErrors((value) => { value.stages.source.download_url = 'https://huggingface.co/example/model' }).some((error) => error.includes('forbidden credential or download')))
assert.ok(candidateErrors((value) => {
  const forged = 'a'.repeat(64)
  value.row_id = `hf_selector_${forged.slice(0, 24)}`
  value.candidate.selector_id = value.row_id
  value.candidate.selector_sha256 = forged
}).some((error) => error.includes('recomputed selector digest')))
assert.ok(candidateErrors((value) => { delete value.qualification_mode }).some((error) => error.includes('candidate-shaped reports must declare')))
assert.ok(candidateErrors((value) => { value.candidate.raw_selector = 'private/model' }).some((error) => error.includes('unexpected field')))
assert.ok(candidateErrors((value) => { value.stages.metadata.observed.tokens = ['secret'] }).some((error) => error.includes('unexpected field')))

assert.ok(candidateErrors((value) => {
  delete value.qualification_mode
  delete value.candidate
  value.row_id = 'generic-looking-row'
}).some((error) => error.includes('candidate-shaped reports must declare')),
'Phase 2 must remain a non-launderable candidate sentinel')
assert.ok(candidateErrors((value) => {
  value.phase = 1
  delete value.qualification_mode
  delete value.candidate
  value.row_id = 'generic-looking-row'
}).some((error) => error.includes('candidate-shaped reports must declare')),
'candidate-only HOLD/scope sentinels must remain non-launderable even if primary markers are stripped')
const fullyLaunderedCandidate = structuredClone(candidateReport)
fullyLaunderedCandidate.phase = 1
delete fullyLaunderedCandidate.qualification_mode
delete fullyLaunderedCandidate.candidate
delete fullyLaunderedCandidate.artifact
delete fullyLaunderedCandidate.scope
delete fullyLaunderedCandidate.support_decision
delete fullyLaunderedCandidate.support_claim
fullyLaunderedCandidate.row_id = 'generic-looking-row'
fullyLaunderedCandidate.stages = Object.fromEntries(
  Object.keys(fullyLaunderedCandidate.stages).map((name) => [name, { status: 'pass' }]),
)
fullyLaunderedCandidate.overall_status = 'pass'
assert.ok(
  validateQualificationReport(
    fullyLaunderedCandidate,
    resolve('target/audit', `${candidateId}-report.json`),
  ).some((error) => error.includes('candidate-shaped reports must declare')),
  'a selector report filename must remain a non-launderable candidate sentinel after every in-report marker is stripped',
)
assert.ok(
  validateQualificationReport(fullyLaunderedCandidate, 'renamed-generic-report.json')
    .some((error) => error.includes('candidate-shaped reports must declare')
      || error.includes('generic v1 PASS is unsupported')
      || error.includes('generic v1 root fields')),
  'renaming a fully stripped selector file must not satisfy the minimum non-candidate v1 contract',
)
const evidenceLaunderedCandidate = structuredClone(fullyLaunderedCandidate)
evidenceLaunderedCandidate.stages.source.evidence = 'x'
assert.ok(
  validateQualificationReport(evidenceLaunderedCandidate, 'renamed-generic-report.json')
    .some((error) => error.includes('candidate-shaped reports must declare')
      || error.includes('generic evidence')),
  'arbitrary evidence cannot launder a stripped and renamed selector candidate into a generic PASS report',
)
const markerFreeCandidate = structuredClone(fullyLaunderedCandidate)
delete markerFreeCandidate.phase
delete markerFreeCandidate.backend
delete markerFreeCandidate.source_tracked_dirty
for (const [label, rewrite] of [
  ['arbitrary notes', (_name, _index) => ({ status: 'pass', note: 'x' })],
  ['trivial evidence arrays', (_name, _index) => ({ status: 'pass', evidence: ['x'] })],
  ['distinct proof fields', (_name, index) => ({ status: 'pass', [`proof_${index}`]: true })],
]) {
  const forged = structuredClone(markerFreeCandidate)
  forged.stages = Object.fromEntries(
    Object.keys(forged.stages).map((name, index) => [name, rewrite(name, index)]),
  )
  const strippedErrors = validateQualificationReport(forged, 'renamed-generic-report.json')
  assert.ok(
    strippedErrors.some((error) => error.includes('generic v1 root fields')),
    `${label} must not satisfy the closed generic-v1 root and stage contracts`,
  )
  const legitimateRootWithForgedStages = structuredClone(report)
  legitimateRootWithForgedStages.stages = structuredClone(forged.stages)
  legitimateRootWithForgedStages.overall_status = 'pass'
  assert.ok(
    validateQualificationReport(legitimateRootWithForgedStages, 'closed-stage-probe')
      .some((error) => error.includes('.stages.')
        || error.includes('generic v1 PASS is unsupported')),
    `${label} must independently fail the closed stage contract or the durable generic-PASS boundary`,
  )
}

const deeplyNestedReport = structuredClone(report)
let nestedCursor = []
deeplyNestedReport.deeply_nested_extension = nestedCursor
for (let depth = 0; depth < 10_000; depth += 1) {
  const child = []
  nestedCursor.push(child)
  nestedCursor = child
}
let deeplyNestedErrors
assert.doesNotThrow(() => {
  deeplyNestedErrors = validateQualificationReport(deeplyNestedReport, 'deeply-nested-report')
}, 'privacy validation must be iterative and total for deeply nested valid JSON values')
assert.ok(
  deeplyNestedErrors.some((error) => error.includes('privacy scan exceeded')),
  'deeply nested reports must fail closed on the explicit privacy scan budget',
)

for (const key of [
  'accessToken',
  'access_key',
  'client_secret',
  'signed-url',
  'download_uri',
  'Authorization',
  'hf.token',
]) {
  assert.ok(candidateErrors((value) => {
    value.stages.source.access[key] = 'test-secret-token'
  }).some((error) => error.includes('forbidden credential or download')), `${key} must be rejected as a credential-key family`)
}
for (const [label, change] of [
  ['Bearer credential value', (value) => { value.stages.artifact.reason = 'Bearer hf_SUPERSECRET123' }],
  ['HF token in an allowed host field', (value) => { value.host.platform = 'hf_SUPERSECRET123' }],
  ['credential assignment in an allowed host field', (value) => { value.host.platform = 'token:SUPERSECRET123' }],
  ['download URL in an allowed reason field', (value) => {
    value.stages.artifact.reason = 'see https://huggingface.co/example/model/resolve/main/model.gguf?token=SUPERSECRET123'
  }],
]) {
  assert.ok(
    candidateErrors(change).some((error) => error.includes('forbidden credential or download')),
    `${label} must be rejected even when its field name is allowed`,
  )
}

for (const [label, change] of [
  ['non-power-of-two alignment', (value) => { value.stages.metadata.observed.alignment = 24 }],
  ['unaligned data start', (value) => { value.stages.metadata.observed.data_start_offset = 95 }],
  ['implausibly small data start', (value) => {
    value.stages.metadata.observed.alignment = 16
    value.stages.metadata.observed.data_start_offset = 16
  }],
  ['data start outside received prefix', (value) => { value.stages.metadata.range.received_bytes = 64 }],
  ['no opaque payload byte fetched', (value) => {
    value.stages.metadata.range.requested_bytes = 96
    value.stages.metadata.range.received_bytes = 96
    value.stages.metadata.range.content_range.end = 95
    value.stages.metadata.scope.opaque_tensor_payload_prefix_bytes = 0
  }],
  ['payload outside source', (value) => { value.stages.metadata.observed.tensor_payload_n_bytes = 5 }],
  ['unsafe payload arithmetic', (value) => { value.stages.metadata.observed.tensor_payload_n_bytes = Number.MAX_SAFE_INTEGER }],
  ['wrong opaque byte count', (value) => { value.stages.metadata.scope.opaque_tensor_payload_prefix_bytes = 2 }],
  ['wrong prefix scope', (value) => { value.stages.metadata.scope.prefix_sha256 = 'header_only' }],
  ['wrong tensor scope', (value) => { value.stages.metadata.scope.tensor_payload = 'decoded' }],
  ['wrong tensor-type sum', (value) => { value.stages.metadata.observed.tensor_type_counts.Q8_0 = 2 }],
  ['missing tensor-type counts', (value) => { delete value.stages.metadata.observed.tensor_type_counts }],
  ['descriptor payload does not close source', (value) => { value.stages.metadata.observed.tensor_payload_n_bytes = 3 }],
]) {
  assert.ok(candidateErrors(change).length > 0, `${label} must fail candidate report validation`)
}
assert.ok(candidateErrors((value) => {
  value.source_tracked_dirty = true
}).some((error) => error.includes('tracked-clean')), 'metadata PASS requires explicitly tracked-clean provenance')
const untrackedOnlyReport = structuredClone(candidateReport)
untrackedOnlyReport.source_dirty = true
assert.deepEqual(
  validateQualificationReport(untrackedOnlyReport, 'untracked-only'),
  [],
  'the provenance policy deliberately permits untracked-only dirt while requiring tracked-clean source',
)
assert.ok(candidateErrors((value) => {
  value.source_head = 'short'
}).some((error) => error.includes('40-hex')), 'metadata PASS requires observed full-commit provenance')
assert.ok(candidateErrors((value) => {
  value.stages.source.access.extra = false
}).some((error) => error.includes('unexpected field')), 'public access projection must be closed')
assert.ok(candidateErrors((value) => {
  value.stages.metadata.required = false
}).some((error) => error.includes('unexpected field')), 'metadata status shapes must reject required=false laundering')
assert.ok(candidateErrors((value) => {
  value.stages.metadata.status = 'blocked'
  value.stages.metadata.error_code = 'header_descriptor_invariants_invalid'
  delete value.stages.metadata.assessment
  delete value.stages.metadata.inspection_generated_at
  delete value.stages.metadata.host
  delete value.stages.metadata.inspector
  delete value.stages.metadata.source
  delete value.stages.metadata.range
  delete value.stages.metadata.observed
  delete value.stages.metadata.scope
}).some((error) => error.includes('requires fail')), 'metadata failure codes must bind to their fail-closed status')
assert.ok(candidateErrors((value) => {
  value.stages.tokenizer.required = false
}).some((error) => error.includes('unexpected field')), 'downstream status shapes must be closed')
const missingSourceReport = structuredClone(candidateReport)
delete missingSourceReport.stages.source
assert.doesNotThrow(() => validateQualificationReport(missingSourceReport, 'missing-source'))
assert.ok(validateQualificationReport(missingSourceReport, 'missing-source').length > 0)

const opaqueUuid = '01234567-89ab-4cde-8fab-0123456789ab'
const opaqueId = 'hf_candidate_run_0123456789ab4cde8fab0123456789ab'
const privateCandidateReport = structuredClone(candidateReport)
privateCandidateReport.row_id = opaqueId
privateCandidateReport.candidate = {
  identity_mode: 'opaque_run',
  run_id: opaqueUuid,
  selector_redacted: true,
  roster_membership: 'absent_from_phase1_roster',
}
privateCandidateReport.stages.source = {
  status: 'blocked',
  resolution: 'live_huggingface',
  error_code: 'private_source_not_persisted',
  reason: 'private Hugging Face source identities are not persisted by selector qualification',
  access: { gated: true, private: true, disabled: false },
  selector_redacted: true,
}
privateCandidateReport.stages.metadata = {
  status: 'blocked',
  mode: 'remote_immutable_prefix',
  error_code: 'header_source_preflight_blocked',
  reason: 'bounded header inspection is downstream of a passing public immutable source lock',
}
assert.deepEqual(validateQualificationReport(privateCandidateReport, 'private-candidate'), [])
const privateErrors = (change) => {
  const value = structuredClone(privateCandidateReport)
  change(value)
  return validateQualificationReport(value, 'private-candidate')
}
assert.ok(privateErrors((value) => {
  value.row_id = candidateId
  value.candidate = structuredClone(candidateReport.candidate)
}).some((error) => error.includes('opaque_run')), 'nonpublic reports cannot persist selector-derived identities')
assert.ok(privateErrors((value) => {
  value.stages.source.repo = publicSelector.repo
}).some((error) => error.includes('unexpected field')), 'nonpublic source shapes cannot persist raw selectors')
assert.ok(privateErrors((value) => {
  value.stages.source.status = 'fail'
  value.overall_status = 'fail'
}).some((error) => error.includes('requires blocked')), 'source failure codes must bind to their fail-closed status')
assert.ok(privateErrors((value) => {
  value.stages.source.access.accessToken = 'secret'
}).some((error) => error.includes('forbidden credential or download')), 'nonpublic access projection must remain exact and scrubbed')
assert.ok(privateErrors((value) => {
  value.stages.metadata.receipt = { raw_error: 'secret' }
}).some((error) => error.includes('unexpected field')), 'nonpass metadata shapes cannot carry raw receipts or errors')
assert.ok(privateErrors((value) => {
  value.candidate.run_id = 'fedcba98-7654-4321-8fed-cba987654321'
}).some((error) => error.includes('bind exactly')), 'opaque filenames must bind to their injected run identity')

console.log('test-check-model-qualification-report: all checks passed')
