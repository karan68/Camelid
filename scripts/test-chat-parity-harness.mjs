#!/usr/bin/env node
import assert from 'node:assert/strict'

import { renderExpectedPrompt, resolveReferenceContext } from './lib/chat-parity-harness.mjs'

assert.equal(
  renderExpectedPrompt([
    { role: 'system', content: 'Answer briefly.' },
    { role: 'user', content: 'Say alpha.' },
    { role: 'assistant', content: 'alpha' },
    { role: 'user', content: 'Now say beta.' },
  ], 'compact'),
  '<|start_header_id|>system<|end_header_id|>\n\nAnswer briefly.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSay alpha.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nalpha<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nNow say beta.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
)

assert.equal(
  renderExpectedPrompt([
    { role: 'user', content: 'Complete cam' },
    { role: 'assistant', content: 'elid' },
  ], 'compact'),
  '<|start_header_id|>user<|end_header_id|>\n\nComplete cam<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nelid<|eot_id|>',
)

assert.equal(
  renderExpectedPrompt([
    { role: 'system', content: 'Answer briefly.' },
    { role: 'user', content: 'Name one primary color.' },
  ], 'tinyllama-marker'),
  '<|system|>\nAnswer briefly.</s>\n<|user|>\nName one primary color.</s>\n<|assistant|>\n',
)

assert.equal(
  renderExpectedPrompt([
    { role: 'user', content: 'Complete cam' },
    { role: 'assistant', content: 'elid' },
  ], 'tinyllama-marker'),
  '<|user|>\nComplete cam</s>\n<|assistant|>\nelid</s>\n',
)

assert.equal(resolveReferenceContext({ promptTokenCount: 120, maxTokens: 5 }), 512)
assert.equal(resolveReferenceContext({ promptTokenCount: 520, maxTokens: 5 }), 541)
assert.equal(resolveReferenceContext({ promptTokenCount: 520, maxTokens: 5, explicitContext: 600 }), 600)
assert.throws(
  () => resolveReferenceContext({ promptTokenCount: 520, maxTokens: 5, explicitContext: 530 }),
  /too small/,
)

console.log('chat-parity-harness self-test passed')
