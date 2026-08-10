#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { summarizeRoster, validateRoster } from './check-model-qualification-roster.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const path = join(root, 'qa', 'model-qualification', 'phase1-roster.json')
const roster = JSON.parse(await readFile(path, 'utf8'))
const clone = () => structuredClone(roster)
const errorsAfter = (mutate) => {
  const candidate = clone()
  mutate(candidate)
  return validateRoster(candidate, 'test')
}

assert.deepEqual(validateRoster(roster, 'phase1'), [], 'the committed Phase 1 roster must validate')
assert.equal(summarizeRoster(roster).length, 6, 'all six Phase 1 graph families must remain visible')
assert.equal(summarizeRoster(roster)[0].next_gate, 'api_webui', 'LFM2 proceeds to API/WebUI after immutable provenance closes')
assert.equal(summarizeRoster(roster)[1].next_gate, 'load_smoke', 'Qwen2.5 preserves its first post-bootstrap blocker')

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
