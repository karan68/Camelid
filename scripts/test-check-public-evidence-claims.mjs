#!/usr/bin/env node
import assert from 'node:assert/strict'
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawnSync } from 'node:child_process'

const tempRoot = await mkdtemp(join(tmpdir(), 'camelid-evidence-claims-'))
const goodRoot = join(tempRoot, 'good')
const badRoot = join(tempRoot, 'bad')

await writeBundle(goodRoot, { mutate: false })
await writeSingleRowContextBundle(goodRoot, { mutate: false })
await writeEightBContextBundle(goodRoot, { mutate: false })
await writeLegacyPublicContextBundle(goodRoot, { mutate: false })
await writeBundle(badRoot, { mutate: true })
await writeSingleRowContextBundle(badRoot, { mutate: true })
await writeEightBContextBundle(badRoot, { mutate: true })
await writeLegacyPublicContextBundle(badRoot, { mutate: true })

const good = spawnSync(process.execPath, ['scripts/check-public-evidence-claims.mjs', '--root', goodRoot], {
  cwd: process.cwd(),
  encoding: 'utf8',
})
assert.equal(good.status, 0, good.stderr || good.stdout)
assert.match(good.stdout, /public evidence claim check passed/)

const bad = spawnSync(process.execPath, ['scripts/check-public-evidence-claims.mjs', '--root', badRoot], {
  cwd: process.cwd(),
  encoding: 'utf8',
})
assert.notEqual(bad.status, 0, 'invalid context evidence should fail')
assert.match(bad.stderr, /generated_tokens_match must be true/)
assert.match(bad.stderr, /source_prompt_pack must be qa\/prompt-packs\/llama3-context-1024-smoke\.json/)
assert.match(bad.stderr, /backend_generated_tokens must stay \[34,2735,35,12,7854\]/)

async function writeBundle(root, { mutate }) {
  const dir = join(root, 'four-row-context-512-test')
  await mkdir(dir, { recursive: true })
  const boundary = 'Closes only the first bounded 512-context pack. It does not promote neighboring rows, other quantizations, larger contexts, broader chat-template behavior, or full Llama-family support.'
  const rows = [
    row('tinyllama_1_1b_chat_q8_0', 291),
    row('llama32_1b_instruct_q8_0', 245),
    row('llama32_3b_instruct_q8_0', 245),
    row('llama3_8b_instruct_q8_0', 245),
  ]
  if (mutate) rows[3].generated_tokens_match = false
  const manifest = {
    schema: 'camelid.four_row_context_512_public_evidence.v1',
    passed: true,
    checkout_clean: true,
    pack: {
      target_context_window: 512,
      max_tokens: 5,
      source_prompt_pack: 'qa/prompt-packs/llama3-context-512-smoke.json',
    },
    rows,
    claim_boundary: boundary,
  }
  const summary = {
    schema: 'camelid.four_row_context_512_public_summary.v1',
    passed: true,
    checks: {
      checkout_clean: true,
      prompt_tokens_all_match: true,
      generated_tokens_all_match: true,
      generated_text_all_match: true,
      all_rows_have_bounded_rss: true,
    },
    rows: rows.map((item) => ({
      row_id: item.row_id,
      context_window: item.context_window,
      reference_prompt_token_count: item.reference_prompt_token_count,
      max_tokens: item.max_tokens,
      max_resident_set_kib: item.max_resident_set_kib,
      passed: item.prompt_tokens_match && item.generated_tokens_match && item.generated_text_match,
    })),
    claim_boundary: boundary,
  }
  await writeFile(join(dir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`)
  await writeFile(join(dir, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`)
}

async function writeLegacyPublicContextBundle(root, { mutate }) {
  const dir = join(root, 'llama32-1b-context-2048-legacy-test')
  await mkdir(dir, { recursive: true })
  const generatedTokens = mutate ? [34, 2735, 35, 12, 4278] : [34, 2735, 35, 12, 7854]
  const manifest = {
    schema: 'camelid.public-evidence-bundle.v1',
    id: 'llama32-1b-context-2048-legacy-test',
    source_head: '62f8cbc',
    created_at_utc: '2026-05-06T01:05:00Z',
    model_row: 'llama32_1b_instruct_q8_0',
    model: '$CAMELID_MODEL_DIR/Llama-3.2-1B-Instruct-Q8_0.gguf',
    pack_id: 'llama3-context-2048-smoke-v1',
    target_context_window: 2048,
    reference_prompt_token_count: 1910,
    max_tokens: 5,
    result: {
      passed: true,
      prompt_tokens_all_match: true,
      generated_tokens_all_match: true,
      generated_text_all_match: true,
      backend_generated_tokens: generatedTokens,
      reference_generated_tokens: [34, 2735, 35, 12, 7854],
      backend_text: 'CMLD-204',
      reference_text: 'CMLD-204',
    },
    boundary:
      'Closes only the third bounded 2048-context pack for the exact Llama 3.2 1B row. It does not promote neighboring rows, other quantizations, model-native/larger context buckets, arbitrary templates, broad/full Llama-family support, production throughput, or portability support.',
    primary_artifacts: [
      'pack/summary.json',
      'pack/llama32-1b-q8-roughly-2048-token-recall/report.json',
    ],
  }
  await writeFile(join(dir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`)
}

async function writeSingleRowContextBundle(root, { mutate }) {
  const dir = join(root, 'llama32-1b-context-1024-test')
  await mkdir(dir, { recursive: true })
  const rowItem = contextRow({
    rowId: 'llama32_1b_instruct_q8_0',
    contextWindow: 1024,
    promptId: 'roughly-1024-token-recall',
    generatedText: 'CMLD-102',
    rawArtifact: 'target/llama32-1b-context-1024-test/summary.json',
  })
  const manifest = {
    schema: 'camelid.llama32_1b_context_1024_public_evidence.v1',
    passed: true,
    checkout_clean: true,
    pack: {
      target_context_window: 1024,
      max_tokens: 5,
      source_prompt_pack: mutate ? 'qa/prompt-packs/llama3-context-512-smoke.json' : 'qa/prompt-packs/llama3-context-1024-smoke.json',
      prompt_count: 1,
    },
    rows: [rowItem],
    claim_boundary:
      'Closes only the second bounded 1024-context pack for the exact Llama 3.2 1B row. It does not promote neighboring rows, other quantizations, model-native/larger context buckets, arbitrary templates, broad/full Llama-family support, production throughput, or portability support.',
  }
  await writeFile(join(dir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`)
}

async function writeEightBContextBundle(root, { mutate }) {
  const dir = join(root, 'llama3-8b-context-2048-test')
  await mkdir(dir, { recursive: true })
  const rowItem = contextRow({
    rowId: 'llama3_8b_instruct_q8_0',
    contextWindow: 2048,
    promptId: 'roughly-2048-token-recall',
    generatedText: mutate ? 'CMLD-102' : 'CMLD-204',
    rawArtifact: 'target/llama3-8b-context-2048-test/summary.json',
  })
  const manifest = {
    schema: 'camelid.llama3_8b_context_2048_public_evidence.v1',
    passed: true,
    checkout_clean: true,
    pack: {
      target_context_window: 2048,
      max_tokens: 5,
      source_prompt_pack: 'qa/prompt-packs/llama3-context-2048-smoke.json',
      prompt_count: 1,
    },
    rows: [rowItem],
    claim_boundary:
      'Closes only the third bounded 2048-context pack for the exact Llama 3 8B row. It does not promote neighboring rows, other quantizations, model-native/larger context buckets, arbitrary templates, broad/full Llama-family support, production throughput, or portability support.',
  }
  await writeFile(join(dir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`)
}

function contextRow({ rowId, contextWindow, promptId, generatedText, rawArtifact }) {
  return {
    row_id: rowId,
    context_window: contextWindow,
    max_tokens: 5,
    prompt_id: promptId,
    reference_prompt_token_count: contextWindow === 2048 ? 1910 : 881,
    prompt_tokens_match: true,
    generated_tokens_match: true,
    generated_text_match: true,
    first_generated_token_diff_index: -1,
    generated_text: generatedText,
    max_resident_set_kib: 2897852,
    model_sha256: 'b'.repeat(64),
    raw_artifact: rawArtifact,
    passed: true,
  }
}

function row(rowId, tokenCount) {
  return {
    row_id: rowId,
    context_window: 512,
    max_tokens: 5,
    reference_prompt_token_count: tokenCount,
    prompt_tokens_match: true,
    generated_tokens_match: true,
    generated_text_match: true,
    first_generated_token_diff_index: -1,
    max_resident_set_kib: 1024,
    model_sha256: 'a'.repeat(64),
    raw_artifact: `target/${rowId}/summary.json`,
  }
}
