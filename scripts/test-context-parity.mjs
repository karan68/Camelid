#!/usr/bin/env node
import assert from 'node:assert/strict'

import {
  actualTokenGate,
  buildGenerationRequest,
  buildMessages,
  camelidLaunchConfig,
  oracleConfig,
  parseArgs,
  requireNativeReceipt,
  requestMode,
  verifierArgs,
} from '../qa/capability/context_parity_lib.mjs'

const chatArgs = parseArgs([
  '--request-mode', 'chat',
  '--expect-actual-tokens=512',
  '--llama-cache-type-k', 'f32',
  '--llama-cache-type-v', 'f32',
  '--llama-flash-attn', 'off',
  '--llama-repack', 'off',
])
assert.equal(requestMode(chatArgs), 'chat')
assert.deepEqual(oracleConfig(chatArgs), {
  cacheTypeK: 'f32',
  cacheTypeV: 'f32',
  flashAttn: 'off',
  repack: 'off',
})

assert.deepEqual(camelidLaunchConfig(chatArgs, { KEEP: 'yes' }), {
  lane: 'cpu',
  env: { KEEP: 'yes', CUDA_VISIBLE_DEVICES: '-1', CAMELID_LFM2_METAL: '0' },
  serveArgs: ['--gpu', 'off', '--deterministic'],
})
assert.deepEqual(
  camelidLaunchConfig(parseArgs(['--camelid-lane', 'metal']), { KEEP: 'yes' }),
  {
    lane: 'metal',
    env: { KEEP: 'yes', CUDA_VISIBLE_DEVICES: '-1', CAMELID_LFM2_METAL: '1' },
    serveArgs: ['--gpu', 'on'],
  },
)

const messages = buildMessages('long prompt', 'system rule')
assert.deepEqual(messages, [
  { role: 'system', content: 'system rule' },
  { role: 'user', content: 'long prompt' },
])
assert.deepEqual(buildGenerationRequest({ mode: 'chat', prompt: 'ignored', messages, maxGen: 8 }), {
  messages,
  max_tokens: 8,
  temperature: 0,
  stream: false,
  camelid_receipt: true,
})
assert.deepEqual(buildGenerationRequest({ mode: 'raw', prompt: 'raw prompt', messages, maxGen: 8 }), {
  prompt: 'raw prompt',
  max_tokens: 8,
  temperature: 0,
  stream: false,
  camelid_receipt: true,
})

assert.deepEqual(actualTokenGate(512, '512'), {
  enabled: true,
  expected: 512,
  actual: 512,
  pass: true,
})
assert.equal(actualTokenGate(511, '512').pass, false)
assert.equal(actualTokenGate(512, undefined).enabled, false)

const native = requireNativeReceipt({
  camelid_receipt: {
    result: {
      prompt_token_ids: [1, 2, 3],
      generated_token_ids: [4],
      generated_text: 'answer',
    },
  },
}, 'chat')
assert.deepEqual(native.promptIds, [1, 2, 3])
assert.equal(native.generatedText, 'answer')
assert.throws(
  () => requireNativeReceipt({ camelid: { prompt_token_ids: [1, 2, 3] } }, 'chat'),
  /do not synthesize one/,
)
assert.throws(
  () => requireNativeReceipt({ camelid_receipt: { result: { prompt_token_ids: [] } } }, 'chat'),
  /no result.prompt_token_ids/,
)

const verify = verifierArgs({
  receiptPath: 'receipt.json',
  gguf: 'model.gguf',
  llamaServer: 'llama-server.exe',
  llamaCtx: 640,
  llamaPort: 8243,
  verifyMode: 'reference-only',
  oracle: oracleConfig(chatArgs),
})
assert.deepEqual(verify.slice(-8), [
  '--llama-cache-type-k', 'f32',
  '--llama-cache-type-v', 'f32',
  '--llama-flash-attn', 'off',
  '--llama-no-repack',
  '--reference-only',
])

assert.throws(() => requestMode(parseArgs(['--request-mode', 'other'])), /raw or chat/)
assert.throws(() => oracleConfig(parseArgs(['--llama-flash-attn', 'maybe'])), /on, off, or auto/)
assert.throws(() => actualTokenGate(512, '512x'), /positive integer/)
assert.throws(
  () => camelidLaunchConfig(parseArgs(['--camelid-lane', 'cuda'])),
  /cpu or metal/,
)

console.log('context parity helper tests passed')
