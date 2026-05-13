#!/usr/bin/env node
import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(scriptDir, '..')

const server = await createServer({
  root: frontendRoot,
  appType: 'custom',
  logLevel: 'silent',
  server: { middlewareMode: true },
})

try {
  const { default: ChatWorkspace } = await server.ssrLoadModule('/src/views/ChatWorkspace.jsx')
  const { default: ApiView } = await server.ssrLoadModule('/src/views/ApiView.jsx')

  const noop = () => {}
  const readyRuntime = {
    api_base: 'http://127.0.0.1:8181',
    loaded_now: true,
    generation_ready: true,
    active_model_id: 'llama32_3b_instruct_q8_0',
  }
  const selectedModel = {
    id: 'llama32_3b_instruct_q8_0',
    name: 'Llama 3.2 3B Instruct Q8_0',
    provider_kind: 'local',
    loaded_now: true,
    generation_ready: true,
    status: 'ready',
    quant: 'Q8_0',
  }
  const capabilities = {
    support_contract: {
      current_gate: 'supported_current_gate',
      support_policy: 'Only exact rows unlock chat.',
      unsupported_policy: 'Everything else remains guarded.',
    },
    supported_model_families: [{ id: 'broad_family_trap', status: 'supported' }],
    supported_quantization: [{ id: 'broad_quant_trap', status: 'supported' }],
    model_compatibility: [
      {
        id: 'llama32_3b_instruct_q8_0',
        status: 'supported_current_gate',
        family: 'llama_bpe_decoder',
        quantization: 'Q8_0',
        support_scope: 'exact row only',
        frontend_readiness_gate: 'loaded_now + generation_ready + active_model_id + exact row',
        latest_checked_bucket: 'current_head',
        latest_checked_result: 'pass',
        latest_checked_output: 'exact row fixture output',
        full_support_status: 'guarded_by_exact_row',
        full_support_blockers: 'No broad-family inheritance.',
        evidence: 'Exact row evidence bundle.',
        metadata_parses: 'validated',
        tokenizer_works: 'validated',
        tensors_load: 'validated',
        generation_runs: 'validated',
        frontend_load_path_verified: 'validated',
        chat_template_shape_pack: 'validated',
        bounded_context_512_pack: 'validated',
        bounded_context_1024_pack: 'validated',
        bounded_context_2048_pack: 'validated',
        performance_measured: 'measured',
        next_step: 'Keep exact-row scoped.',
      },
      {
        id: 'other_future_row_q8_0',
        status: 'planned',
        family: 'future_decoder',
        quantization: 'Q8_0',
        next_step: 'Do not unlock selected chat.',
      },
    ],
    api_features: [
      { id: 'streaming_chat_completions', status: 'supported_current_gate', notes: 'Streaming stays enabled.' },
      { id: 'future_batch_endpoint', status: 'planned', notes: 'Guarded feature row.' },
    ],
  }

  const streamingMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: {
      id: 'conversation-streaming-code',
      title: 'Streaming code',
      updated_at: '2026-05-13T04:21:00.000Z',
      messages: [
        { id: 'user-1', role: 'user', content: 'Create one self-contained HTML page', created_at: '2026-05-13T04:21:00.000Z' },
        { id: 'assistant-1', role: 'assistant', content: '```html\n<!doctype html>\n<button id="go">Go</button>', streaming: true, streaming_phase: 'streaming', created_at: '2026-05-13T04:21:01.000Z' },
      ],
    },
    selectedModel,
    selectedModelId: selectedModel.id,
    setSelectedModelId: noop,
    models: [selectedModel],
    runtime: readyRuntime,
    capabilities,
    pendingConversation: null,
    composer: '',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable: true,
    setTab: noop,
  }))

  assert.match(streamingMarkup, /data-streaming-state="active"/, 'streaming assistant rows should render an active streaming state')
  assert.match(streamingMarkup, /data-streaming-code-state="open"/, 'open streaming fences should expose the active code state')
  assert.match(streamingMarkup, /Still generating — code block incomplete/, 'open streaming code should visibly say it is incomplete')
  assert.match(streamingMarkup, /Streaming code response/, 'streaming code rows should keep an active live-generation label')
  assert.match(streamingMarkup, /aria-busy="true"/, 'streaming rows and code cards should be marked busy while backend generation is active')
  assert.match(streamingMarkup, /message-code-card is-generating/, 'open streaming code should render as the real ForgeLocal-derived code card, not fallback prose')

  const preTokenMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: {
      id: 'conversation-pre-token',
      title: 'Pre-token',
      updated_at: '2026-05-13T04:21:00.000Z',
      messages: [
        { id: 'user-2', role: 'user', content: 'Say hello', created_at: '2026-05-13T04:21:00.000Z' },
        { id: 'assistant-2', role: 'assistant', content: '', streaming: true, streaming_phase: 'generating', created_at: '2026-05-13T04:21:01.000Z' },
      ],
    },
    selectedModel,
    selectedModelId: selectedModel.id,
    setSelectedModelId: noop,
    models: [selectedModel],
    runtime: readyRuntime,
    capabilities,
    pendingConversation: null,
    composer: '',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable: true,
    setTab: noop,
  }))

  assert.match(preTokenMarkup, /data-streaming-state="active"/, 'pre-token assistant rows should remain visibly active while the backend is generating')
  assert.match(preTokenMarkup, /Waiting for first token/, 'pre-token streaming should render the first-token live status')
  assert.match(preTokenMarkup, /pacman-loader-mouth/, 'pre-token streaming should render the active loader, not a static placeholder')

  const preTokenSendingMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: {
      id: 'conversation-pre-token-active-send',
      title: 'Pre-token active send',
      updated_at: '2026-05-13T04:21:00.000Z',
      messages: [
        { id: 'user-3', role: 'user', content: 'Say hello', created_at: '2026-05-13T04:21:00.000Z' },
        { id: 'assistant-3', role: 'assistant', content: '', streaming: true, streaming_phase: 'generating', created_at: '2026-05-13T04:21:01.000Z' },
      ],
    },
    selectedModel,
    selectedModelId: selectedModel.id,
    setSelectedModelId: noop,
    models: [selectedModel],
    runtime: readyRuntime,
    capabilities,
    pendingConversation: null,
    composer: '',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: true,
    selectedModelRunnable: true,
    setTab: noop,
  }))

  assert.equal((preTokenSendingMarkup.match(/data-streaming-state="active"/g) || []).length, 1, 'active send with an inserted pre-token assistant row should not render a duplicate pending assistant loader')
  assert.equal((preTokenSendingMarkup.match(/pacman-loader-track/g) || []).length, 1, 'pre-token active send should keep exactly one visible live loader for the backend generation')

  const exactReadyMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: readyRuntime,
    selectedModel,
    capabilities,
  }))

  assert.match(exactReadyMarkup, /Selected exact row ready/, 'API readiness should turn green only for a matching loaded exact row')
  assert.match(exactReadyMarkup, /llama32_3b_instruct_q8_0/, 'API view should render the selected exact compatibility row id')
  assert.match(exactReadyMarkup, /Exact row evidence bundle\./, 'API view should render exact-row evidence text')
  assert.match(exactReadyMarkup, /exact row fixture output/, 'API view should render latest exact-row output evidence')
  assert.doesNotMatch(exactReadyMarkup, /broad_family_trap|broad_quant_trap/, 'API view must not promote broad family or quant lists as support evidence')

  const mismatchedRuntimeMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: { ...readyRuntime, active_model_id: 'different-loaded-model' },
    selectedModel,
    capabilities,
  }))

  assert.match(mismatchedRuntimeMarkup, /Different loaded model is ready/, 'API readiness should fail closed when active_model_id differs from the selected exact row')
  assert.match(mismatchedRuntimeMarkup, /Blocked for UX chat until selected exact row evidence and runtime readiness both match/, 'API curl should stay blocked until exact row and runtime readiness both match')
  assert.doesNotMatch(mismatchedRuntimeMarkup, /Selected exact row ready/, 'mismatched runtime must not claim selected exact-row readiness')

  console.log('Frontend integration smoke passed')
} finally {
  await server.close()
}
