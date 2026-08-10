#!/usr/bin/env node
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { summarizeRoster, validateRoster } from './check-model-qualification-roster.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const path = join(root, 'qa', 'model-qualification', 'phase1-roster.json')
const roster = JSON.parse(await readFile(path, 'utf8'))
const qwenContextBundle = join(
  root,
  'qa',
  'evidence-bundles',
  'qwen2.5-0.5b-q8-context-512-20260810',
)
const qwenContext = JSON.parse(await readFile(
  join(qwenContextBundle, 'context-summary.json'),
  'utf8',
))
const qwenReceiptBytes = await readFile(join(qwenContextBundle, 'parity-receipt.json'))
const qwenReceipt = JSON.parse(qwenReceiptBytes)
const qwenVerifyBytes = await readFile(join(qwenContextBundle, 'verify.log'))
const qwenVerifyLog = qwenVerifyBytes.toString('utf8')
const qwenContextManifest = JSON.parse(await readFile(join(qwenContextBundle, 'manifest.json'), 'utf8'))
const qwenContextSums = await readFile(join(qwenContextBundle, 'SHA256SUMS'), 'utf8')
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
assert.equal(summarizeRoster(roster)[1].next_gate, 'load_smoke', 'Qwen2.5 preserves its first post-bootstrap blocker')
assert.equal(roster.rows[1].gates.context.status, 'pass', 'Qwen2.5 closes the independent exact-512 context gate without weakening strict short-prompt parity')
assert.equal(qwenContext.schema, 'camelid.context-qualification-summary/v1')
assert.equal(qwenContext.row_id, roster.rows[1].id)
assert.equal(qwenContext.artifact.sha256, roster.rows[1].identity.sha256)
assert.equal(qwenContext.oracle.revision, roster.defaults.llama_cpp.revision)
assert.match(qwenContext.provenance.source_head, /^[0-9a-f]{40}$/)
assert.match(qwenContext.provenance.camelid_binary_sha256, /^[0-9a-f]{64}$/)
assert.equal(qwenContext.provenance.binary_built_from_clean_head, true)
assert.match(qwenContext.provenance.camelid_version, new RegExp(qwenContext.provenance.source_head.slice(0, 8)))
assert.equal(qwenContext.request.actual_prompt_tokens, 512)
assert.equal(qwenContext.request.actual_prompt_token_gate_pass, true)
assert.equal(qwenContext.result.strict_token_and_text_parity, true)
assert.equal(qwenContext.result.camelid_self_replay_pass, true)
assert.equal(qwenContext.result.llama_cpp_reference_replay_pass, true)
assert.equal(qwenContext.result.first_divergent_token_index, -1)
assert.equal(qwenContext.result.generated_token_ids.length, 8)
assert.match(qwenContext.does_not_prove.join(' '), /short-context parity gate.*remains failed/)
assert.equal(qwenContext.oracle.binary_sha256, '6c787bf07ac1d7e1bbaa1ee176c3ef0df58ea86494c8c1b1d2d9f4a9176b19ae')
assert.equal(qwenContext.oracle.package_archive_sha256, 'b835d5c5155dd2a5ed748a0351debf2ede0dc9f808757e0429f8700a11832dcd')
assert.equal(fileSha256(qwenReceiptBytes), qwenContext.receipt.file_sha256)
assert.equal(fileSha256(qwenVerifyBytes), qwenContext.receipt.verify_log_sha256)
assert.equal(qwenContextManifest.model.row_id, roster.rows[1].id)
assert.equal(qwenContextManifest.model.sha256, roster.rows[1].identity.sha256)
assert.equal(qwenContextManifest.runtime_head, qwenContext.provenance.source_head)
for (const filename of qwenContextManifest.artifacts.filter((name) => name !== 'SHA256SUMS')) {
  const bytes = await readFile(join(qwenContextBundle, filename))
  assert.match(qwenContextSums, new RegExp(`^${fileSha256(bytes)}  ${filename}$`, 'm'))
}
assert.equal(qwenReceipt.receipt_id, qwenContext.receipt.receipt_id)
assert.equal(qwenReceipt.lane.gguf_sha256, roster.rows[1].identity.sha256)
assert.equal(qwenReceipt.lane.camelid_commit, qwenContext.provenance.source_head)
assert.equal(qwenReceipt.result.prompt_token_ids.length, 512)
assert.deepEqual(qwenReceipt.result.generated_token_ids, qwenContext.result.generated_token_ids)
assert.equal(qwenReceipt.result.generated_text, qwenContext.result.generated_text)
assert.equal(qwenReceipt.execution_trace.digest, qwenContext.receipt.execution_trace.digest)
assert.equal(qwenReceipt.execution_trace.fold_count, 200)
assert.match(qwenVerifyLog, /PASS camelid-rerun: prompt tokens \(512\), generated tokens \(8\)/)
assert.match(qwenVerifyLog, /PASS execution-trace:.*200 checkpoints/)
assert.match(qwenVerifyLog, /INFO reference-rerun: llama-server version "version: 9632 \(acd79d603\)"/)
assert.match(qwenVerifyLog, /PASS reference-rerun: generated tokens \(8\) and text match llama\.cpp \(first_divergent_token_index=-1\)/)
assert.match(qwenVerifyLog, /RECEIPT VERIFIED \(self-digest, lane identity, Camelid replay, and llama\.cpp reference re-run all passed/)
assert.equal(summarizeRoster(roster)[3].next_gate, 'load_smoke', 'Gemma2 advances through exact-row metadata, tokenizer, and template evidence')
assert.equal(summarizeRoster(roster)[4].next_gate, 'template', 'SmolLM3 advances through exact-row tokenizer evidence while preserving its dynamic-template HOLD')
assert.equal(summarizeRoster(roster)[5].next_gate, 'template', 'Qwen3 MoE advances through exact-row tokenizer evidence while preserving every downstream HOLD')
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

console.log('test-check-model-qualification-roster: all checks passed')
