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
assert.equal(summarizeRoster(roster)[4].next_gate, 'tokenizer', 'SmolLM3 advances metadata while preserving its tokenizer and dynamic-template HOLDs')

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
