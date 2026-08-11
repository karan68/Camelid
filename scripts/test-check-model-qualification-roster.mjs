#!/usr/bin/env node
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { SUPPORTED_MODELS } from '../frontend/src/lib/supportedModels.js'
import { summarizeRoster, validateRoster } from './check-model-qualification-roster.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const path = join(root, 'qa', 'model-qualification', 'phase1-roster.json')
const roster = JSON.parse(await readFile(path, 'utf8'))
const phase2Path = join(root, 'qa', 'model-qualification', 'phase2-roster.json')
const fileSha256 = bytes => createHash('sha256').update(bytes).digest('hex')
const gitBlobSha1 = bytes => createHash('sha1')
  .update(Buffer.from(`blob ${bytes.length}\0`))
  .update(bytes)
  .digest('hex')
const smolRuntimeFixturePath = join(
  root,
  'qa',
  'model-qualification',
  'fixtures',
  'smollm3-default-thinking-runtime-envelope-v1.json',
)
const smolRuntimeFixture = JSON.parse(await readFile(smolRuntimeFixturePath, 'utf8'))
const smolPreparationPackBytes = await readFile(join(
  root,
  'qa',
  'prompt-packs',
  'smollm3-chat-template-shapes-v1.json',
))
const smolHeaderReceiptBytes = await readFile(join(
  root,
  'qa',
  'model-qualification',
  'smollm3-3b-q8-header-inspection.json',
))
const smolTokenizerReceiptBytes = await readFile(join(
  root,
  'qa',
  'model-qualification',
  'smollm3-3b-q8-header-tokenizer-parity.json',
))
const apiSourceBytes = await readFile(join(root, 'src', 'api', 'mod.rs'))
const apiSourceText = apiSourceBytes.toString('utf8')
const exactKeys = (value, keys, label) => assert.deepEqual(
  Object.keys(value).sort(),
  [...keys].sort(),
  `${label} must stay a compact closed contract`,
)
const clone = () => structuredClone(roster)
const errorsAfter = (mutate) => {
  const candidate = clone()
  mutate(candidate)
  return validateRoster(candidate, 'test')
}

assert.deepEqual(validateRoster(roster, 'phase1'), [], 'the committed Phase 1 roster must validate')
assert.equal(summarizeRoster(roster).length, 6, 'all six Phase 1 graph families must remain visible')
assert.equal(summarizeRoster(roster)[0].next_gate, null, 'LFM2 closes every Phase 1 gate on the exact promoted row')
assert.equal(roster.rows[0].disposition, 'promotion_candidate', 'all-green LFM2 is ready for exact-row promotion')
const qwenRow = roster.rows[1]
const qwenSummary = summarizeRoster(roster)[1]
assert.equal(qwenSummary.next_gate, 'load_smoke',
  'Qwen2.5 retains the load gate until a hardened environment-sealed rerun')
assert.deepEqual(qwenSummary.gates, { pass: 4, fail: 1, blocked: 3, pending: 0 })
assert.equal(qwenRow.gates.load_smoke.status, 'blocked')
assert.deepEqual(qwenRow.gates.load_smoke.evidence, [])
assert.match(qwenRow.gates.load_smoke.reason, /pre-hardening receipt/)
assert.match(qwenRow.gates.load_smoke.reason, /host environment variables/)
assert.equal(qwenRow.gates.parity.status, 'fail')
assert.equal(qwenRow.gates.api_webui.status, 'blocked')
assert.equal(qwenRow.gates.context.status, 'blocked')
assert.match(qwenRow.gates.context.reason, /verifier transcript is absent/)
assert.equal(qwenRow.disposition, 'active_validation')
assert.equal(qwenRow.target_tier, 'experimental_exact_row')
assert.equal(qwenRow.gates.context.status, 'blocked', 'Qwen2.5 context stays blocked until its tracked verifier transcript is present')
assert.equal(summarizeRoster(roster)[3].next_gate, 'load_smoke', 'Gemma2 advances through exact-row metadata, tokenizer, and template evidence')
assert.equal(summarizeRoster(roster)[4].next_gate, 'template', 'SmolLM3 advances through exact-row tokenizer evidence while preserving its dynamic-template HOLD')
const smolRow = roster.rows[4]
const smolSummary = summarizeRoster(roster)[4]
assert.equal(smolRow.gates.load_smoke.status, 'blocked')
assert.deepEqual(smolRow.gates.load_smoke.evidence, [])
assert.match(smolRow.gates.load_smoke.reason, /pre-hardening receipt/)
assert.match(smolRow.gates.load_smoke.reason, /host environment variables/)
assert.deepEqual(smolSummary.gates, { pass: 3, fail: 0, blocked: 3, pending: 2 })
assert.equal(smolSummary.next_gate, 'template')
assert.equal(smolRow.gates.template.status, 'blocked')
assert.equal(smolRow.gates.parity.status, 'blocked')
assert.equal(smolRow.gates.api_webui.status, 'pending')
assert.equal(smolRow.gates.context.status, 'pending')
assert.equal(smolRow.disposition, 'hold')
assert.equal(summarizeRoster(roster)[5].next_gate, 'metadata', 'Qwen3 MoE retains its metadata provenance blocker while recording the independent tokenizer receipt')
assert.equal(roster.rows[5].gates.tokenizer.status, 'pass', 'Qwen3 tokenizer evidence is independently clean even though metadata remains blocked')
exactKeys(
  smolRuntimeFixture,
  ['does_not_prove', 'fixture_id', 'gate_decision', 'grounding', 'implementation', 'qualified_envelope', 'row_id', 'schema', 'source', 'status', 'template', 'typed_refusals'].sort(),
  'the SmolLM3 runtime fixture',
)
exactKeys(smolRuntimeFixture.grounding, ['preparation_pack', 'preparation_pack_sha256', 'header_receipt', 'header_receipt_sha256', 'tokenizer_receipt', 'tokenizer_receipt_sha256'], 'SmolLM3 runtime grounding')
exactKeys(smolRuntimeFixture.source, ['repo', 'file', 'revision', 'size_bytes', 'sha256', 'license'], 'SmolLM3 runtime source')
exactKeys(smolRuntimeFixture.template, ['utf8_bytes', 'sha256', 'identity_required', 'substituted_or_missing_template'], 'SmolLM3 runtime template identity')
exactKeys(smolRuntimeFixture.implementation, ['source_file', 'source_git_blob_sha1', 'architecture', 'renderer', 'public_chokepoints', 'tests'], 'SmolLM3 runtime implementation')
exactKeys(smolRuntimeFixture.qualified_envelope, ['message_input_stage', 'content', 'messages_nonempty', 'content_nonempty', 'exact_role_bytes_required', 'roles', 'history', 'thinking', 'add_generation_prompt', 'today_date', 'public_surfaces'], 'SmolLM3 qualified envelope')
exactKeys(smolRuntimeFixture.qualified_envelope.public_surfaces, ['/apply-template', '/v1/chat/completions'], 'SmolLM3 public surfaces')
exactKeys(smolRuntimeFixture.typed_refusals, ['renderer_entry', 'public_surface_preflight'], 'SmolLM3 typed refusals')
exactKeys(smolRuntimeFixture.typed_refusals.renderer_entry, ['unsupported_chat_template', 'unsupported_tools', 'unsupported_parameter'], 'SmolLM3 renderer-entry refusals')
exactKeys(smolRuntimeFixture.typed_refusals.public_surface_preflight, ['/apply-template', '/v1/chat/completions'], 'SmolLM3 public preflight refusals')
exactKeys(smolRuntimeFixture.typed_refusals.public_surface_preflight['/apply-template'], ['empty_chat_messages', 'invalid_message_content', 'unsupported_parameter'], 'SmolLM3 apply-template preflight refusals')
exactKeys(smolRuntimeFixture.typed_refusals.public_surface_preflight['/v1/chat/completions'], ['missing_generation_input', 'invalid_message_content', 'unsupported_multimodal_content', 'vision_model_required'], 'SmolLM3 chat preflight refusals')
exactKeys(smolRuntimeFixture.gate_decision, ['template_gate', 'api_webui_gate', 'support_claim', 'disposition', 'reason'], 'SmolLM3 gate decision')
assert.equal(smolRuntimeFixture.schema, 'camelid.smollm3-runtime-envelope/v1')
assert.equal(smolRuntimeFixture.fixture_id, 'smollm3-default-thinking-runtime-envelope-v1')
assert.equal(smolRuntimeFixture.row_id, roster.rows[4].id)
assert.equal(smolRuntimeFixture.status, 'partial_renderer_qualified_template_gate_blocked')
assert.deepEqual(smolRuntimeFixture.grounding, {
  preparation_pack: 'qa/prompt-packs/smollm3-chat-template-shapes-v1.json',
  preparation_pack_sha256: fileSha256(smolPreparationPackBytes),
  header_receipt: 'qa/model-qualification/smollm3-3b-q8-header-inspection.json',
  header_receipt_sha256: fileSha256(smolHeaderReceiptBytes),
  tokenizer_receipt: 'qa/model-qualification/smollm3-3b-q8-header-tokenizer-parity.json',
  tokenizer_receipt_sha256: fileSha256(smolTokenizerReceiptBytes),
})
assert.deepEqual(smolRuntimeFixture.source, {
  repo: roster.rows[4].source.repo,
  file: roster.rows[4].source.file,
  revision: roster.rows[4].source.revision,
  size_bytes: roster.rows[4].identity.size_bytes,
  sha256: roster.rows[4].identity.sha256,
  license: roster.rows[4].source.license,
})
assert.equal(fileSha256(smolPreparationPackBytes), smolRuntimeFixture.grounding.preparation_pack_sha256)
assert.equal(fileSha256(smolHeaderReceiptBytes), smolRuntimeFixture.grounding.header_receipt_sha256)
assert.equal(fileSha256(smolTokenizerReceiptBytes), smolRuntimeFixture.grounding.tokenizer_receipt_sha256)
assert.equal(gitBlobSha1(apiSourceBytes), smolRuntimeFixture.implementation.source_git_blob_sha1)
assert.deepEqual(smolRuntimeFixture.implementation, {
  source_file: 'src/api/mod.rs',
  source_git_blob_sha1: gitBlobSha1(apiSourceBytes),
  architecture: 'smollm3',
  renderer: 'render_smollm3_production_chat_prompt',
  public_chokepoints: ['/apply-template', '/v1/chat/completions'],
  tests: [
    'smollm3_exact_renderer_is_architecture_scoped_and_not_generic_qwen3',
    'smollm3_exact_renderer_matches_bounded_oracle_shapes',
    'smollm3_exact_renderer_typed_blocks_every_unqualified_branch',
    'smollm3_props_caps_are_exact_template_and_envelope_aware',
    'smollm3_props_caps_refuse_exact_template_on_foreign_architecture',
    'smollm3_apply_template_renders_the_exact_default_thinking_shape',
    'smollm3_apply_template_rejects_substituted_and_missing_templates',
    'smollm3_exact_chat_opens_before_tokenization_for_stream_and_nonstream',
    'smollm3_unqualified_chat_branches_fail_typed_for_stream_and_nonstream',
    'smollm3_tools_fail_closed_independent_of_template_identity_and_stream_mode',
    'smollm3_chat_rejects_nonexact_templates_for_stream_and_nonstream',
  ],
})
for (const testName of smolRuntimeFixture.implementation.tests) {
  assert.match(apiSourceText, new RegExp(`(?:async\\s+)?fn\\s+${testName}\\s*\\(`), `missing implementation test ${testName}`)
}
assert.equal(smolRuntimeFixture.template.utf8_bytes, 5493)
assert.equal(smolRuntimeFixture.template.sha256, 'b9b66f04c64fbb8695cf5b35c37780efd0b8e0829fbfe3e30fafb9f469b7d30e')
assert.deepEqual(smolRuntimeFixture.template, {
  utf8_bytes: 5493,
  sha256: 'b9b66f04c64fbb8695cf5b35c37780efd0b8e0829fbfe3e30fafb9f469b7d30e',
  identity_required: true,
  substituted_or_missing_template: 'typed_refusal',
})
assert.deepEqual(smolRuntimeFixture.qualified_envelope, {
  message_input_stage: 'post_canonicalized_chat_messages',
  content: 'text_only',
  messages_nonempty: true,
  content_nonempty: true,
  exact_role_bytes_required: true,
  roles: ['user', 'assistant'],
  history: 'strict_alternation_starting_and_ending_user',
  thinking: ['omitted_defaults_true', 'explicit_true'],
  add_generation_prompt: true,
  today_date: 'one_system_local_calendar_read_formatted_dd_english_month_yyyy',
  public_surfaces: {
    '/apply-template': {
      thinking: 'omitted_effective_true',
      result: 'exact_rendered_prompt',
    },
    '/v1/chat/completions': {
      thinking: ['omitted_defaults_true', 'explicit_true'],
      streaming: [false, true],
      result: 'renderer_opens_before_tokenization',
    },
  },
})
assert.deepEqual(smolRuntimeFixture.typed_refusals, {
  renderer_entry: {
    unsupported_chat_template: [
      'substituted_template',
      'missing_template',
      'empty_messages',
      'empty_content',
      'non_exact_role_bytes',
      'system_message',
      'custom_instructions',
      'system_override',
      'tool_role_history',
      'invalid_roles',
      'non_alternating_history',
      'history_not_ending_user',
      'multimodal_content',
      'non_text_content',
    ],
    unsupported_tools: [
      'openai_tools_with_exact_template',
      'openai_tools_with_substituted_template',
      'openai_tools_with_missing_template',
    ],
    unsupported_parameter: ['camelid_enable_thinking_false'],
  },
  public_surface_preflight: {
    '/apply-template': {
      empty_chat_messages: ['empty_messages'],
      invalid_message_content: ['empty_content'],
      unsupported_parameter: ['arbitrary_template_kwargs_including_explicit_thinking'],
    },
    '/v1/chat/completions': {
      missing_generation_input: ['empty_messages'],
      invalid_message_content: ['empty_content'],
      unsupported_multimodal_content: ['unsupported_nontext_content'],
      vision_model_required: ['image_content'],
    },
  },
})
assert.deepEqual(smolRuntimeFixture.gate_decision, {
  template_gate: 'blocked',
  api_webui_gate: 'pending',
  support_claim: false,
  disposition: 'hold',
  reason: "Only the exact default-thinking text envelope is qualified. The source template's system/custom-instruction/system-override, tool, tool-history, no-think, and nontext branches remain unqualified, and no full real-artifact HTTP prompt-token or generation receipt exists.",
})
assert.deepEqual(
  smolRuntimeFixture.does_not_prove,
  [
    'full_template_contract',
    'full_artifact_download_or_hash',
    'model_load',
    'real_artifact_http_prompt_token_parity',
    'generation',
    'api_webui_readiness',
    'context_parity',
    'tools_or_system_branches',
    'thinking_disabled_branch',
    'support',
  ],
)
assert.equal(roster.rows[4].gates.template.status, 'blocked')
assert.equal(roster.rows[4].disposition, 'hold')
assert.ok(
  roster.rows[4].gates.template.evidence.includes('qa/model-qualification/fixtures/smollm3-default-thinking-runtime-envelope-v1.json'),
  'the still-blocked SmolLM3 template gate must cite the narrow runtime envelope',
)
const smolRuntimeSerialized = JSON.stringify(smolRuntimeFixture)
assert.doesNotMatch(smolRuntimeSerialized, /(?:[A-Za-z]:\\|file:\/\/|\\\\[^\\]|\/Users\/|\/home\/|\/tmp\/)/i)
assert.doesNotMatch(smolRuntimeSerialized, /(?:bearer\s+|hf_[A-Za-z0-9]{8,})/i)
assert.ok(
  errorsAfter((candidate) => { candidate.defaults.models_dir_env = 'CAMELID_MODEL_DIR' }).some((error) => error.includes('plural CAMELID_MODELS_DIR')),
  'the stale singular model-dir env spelling must be rejected',
)
assert.ok(
  errorsAfter((candidate) => { candidate.rows[0].gates.parity.status = 'pass'; candidate.rows[0].gates.parity.evidence = [] }).some((error) => error.includes('passing gate needs durable evidence')),
  'a pass without evidence must be rejected',
)
assert.ok(
  errorsAfter((candidate) => { delete candidate.rows[0].gates.parity.evidence }).some((error) => error.includes('array of non-empty strings')),
  'missing evidence must be reported without crashing the validator',
)
assert.ok(
  errorsAfter((candidate) => { candidate.rows[1].identity.sha256 = 'not-a-hash' }).some((error) => error.includes('64-character SHA-256')),
  'malformed artifact identities must be rejected',
)
assert.ok(
  errorsAfter((candidate) => { candidate.rows[1].disposition = 'promotion_candidate' }).some((error) => error.includes('incomplete gates')),
  'promotion must fail closed until every required gate passes',
)
assert.ok(
  errorsAfter((candidate) => { candidate.rows[5].gates.source.status = 'blocked'; delete candidate.rows[5].gates.source.reason }).some((error) => error.includes('need a concrete reason')),
  'blocked gates must name the blocker',
)
assert.ok(
  errorsAfter((candidate) => { candidate.rows[5].priority = 2 }).some((error) => error.includes('unique and contiguous')),
  'priorities must be deterministic',
)

const phase2Roster = JSON.parse(await readFile(phase2Path, 'utf8'))
assert.deepEqual(validateRoster(phase2Roster, 'phase2'), [], 'the committed Phase 2 roster must validate')
assert.equal(summarizeRoster(phase2Roster).length, 20, 'all twenty selected exact rows must remain visible')
assert.deepEqual(
  phase2Roster.rows.map((row) => row.priority),
  Array.from({ length: 20 }, (_, index) => index + 1),
  'Phase 2 priorities must retain the selected qualification order',
)
assert.equal(new Set(phase2Roster.rows.map((row) => row.id)).size, 20, 'Phase 2 row ids must remain unique')
const frontendPhase2 = new Map(SUPPORTED_MODELS.map((row) => [row.catalog_id, row]))
for (const row of phase2Roster.rows) {
  const frontend = frontendPhase2.get(row.id)
  assert.ok(frontend, `${row.id} must remain in the frontend fallback catalog`)
  assert.equal(frontend.repo_id, row.source.repo, `${row.id} frontend repo must match the source lock`)
  assert.equal(frontend.filename, row.identity.gguf_filename, `${row.id} frontend filename must match the source lock`)
  assert.equal(frontend.size_bytes, row.identity.size_bytes, `${row.id} frontend size must match the source lock`)
  assert.equal(frontend.quant, row.identity.quantization, `${row.id} frontend quant must match the source lock`)
}
const invalidPhase2 = structuredClone(phase2Roster)
invalidPhase2.phase.id = 3
assert.ok(
  validateRoster(invalidPhase2, 'phase2').some((error) => error.includes('expected integer 1 or 2')),
  'unrecognized qualification phases must fail closed',
)

console.log('test-check-model-qualification-roster: all checks passed')
