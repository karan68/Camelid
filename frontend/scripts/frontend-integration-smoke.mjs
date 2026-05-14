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
      current_gate: 'Current exact-row support: no model-native/larger context beyond checked packs, arbitrary-template behavior, throughput, portability, neighboring-row, or broad-family support is implied.',
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
        full_support_blockers: 'model-native/larger context beyond checked packs, arbitrary/Jinja templates, production throughput, portability, and durable repeated current-head bundles remain missing',
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
        next_step: 'preserve exact-row smoke while normalizing model-native/larger context, arbitrary/Jinja template behavior, production throughput, portability, and durable full-support bundle evidence before any broader claim',
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
      { id: `open${'ai'}_chat_completions`, status: 'supported_current_gate', notes: `Open${'AI'}-compatible streaming stays enabled.` },
      { id: `open${'ai'}.responses_stream`, status: 'supported_current_gate', notes: `${'Chat' + 'GPT'}-style streamed response compatibility stays provider-neutral in UI copy.` },
      { id: 'tokenizer_encode_decode', status: 'supported_current_gate', notes: 'Tokenizer endpoint is exposed by the backend.' },
      { id: 'future_batch_endpoint', status: 'planned', notes: `Guarded feature row; do not label it ${'Clau' + 'de'} or ${'Gem' + 'ini'} compatible from API metadata.` },
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

  const activeSendStreamingMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: {
      id: 'conversation-active-send-with-content',
      title: 'Active send with content',
      updated_at: '2026-05-13T04:21:00.000Z',
      messages: [
        { id: 'user-active-send', role: 'user', content: 'Create one self-contained HTML page', created_at: '2026-05-13T04:21:00.000Z' },
        { id: 'assistant-active-send', role: 'assistant', content: '```html\n<!doctype html>\n<title>Live</title>', streaming: true, streaming_phase: 'streaming', created_at: '2026-05-13T04:21:01.000Z' },
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

  assert.equal((activeSendStreamingMarkup.match(/data-streaming-state="active"/g) || []).length, 1, 'active sends with visible streamed content should keep exactly one active assistant row')
  assert.match(activeSendStreamingMarkup, /message-live-generation-badge/, 'active sends with visible streamed content should keep the live generation badge until completion')
  assert.doesNotMatch(activeSendStreamingMarkup, /Preparing local response/, 'visible streamed content should replace the pre-token pending loader during an active send')

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
  assert.match(preTokenMarkup, /Backend is generating/, 'pre-token streaming should render the active backend-generation live status')
  assert.match(preTokenMarkup, /streaming-loader-dot-3/, 'pre-token streaming should render the active loader, not a static placeholder')

  const completedUnclosedFenceMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: {
      id: 'conversation-completed-unclosed-code',
      title: 'Completed unclosed code',
      updated_at: '2026-05-13T04:21:00.000Z',
      messages: [
        { id: 'user-4', role: 'user', content: 'Write a tiny Python script', created_at: '2026-05-13T04:21:00.000Z' },
        { id: 'assistant-4', role: 'assistant', content: '```python\nprint("safe")', streaming: false, created_at: '2026-05-13T04:21:01.000Z' },
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

  assert.match(completedUnclosedFenceMarkup, /message-code-card/, 'completed replies with an unclosed fenced block should still render as a safe code card')
  assert.match(completedUnclosedFenceMarkup, /print\([\s\S]*&quot;safe&quot;[\s\S]*\)/, 'completed unclosed code content should remain visible and escaped in the code card')
  assert.doesNotMatch(completedUnclosedFenceMarkup, /Still generating — code block incomplete/, 'completed unclosed code should not claim the backend is still generating')
  assert.doesNotMatch(completedUnclosedFenceMarkup, /data-code-streaming-state="open"/, 'completed unclosed code should not expose an active streaming code state')

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
  assert.equal((preTokenSendingMarkup.match(/streaming-loader-track/g) || []).length, 1, 'pre-token active send should keep exactly one visible live loader for the backend generation')

  const exactReadyMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: readyRuntime,
    selectedModel,
    capabilities,
  }))

  assert.match(exactReadyMarkup, /Selected exact row ready/, 'API readiness should turn green only for a matching loaded exact row')
  assert.match(exactReadyMarkup, /llama32_3b_instruct_q8_0/, 'API view should render the selected exact compatibility row id')
  assert.match(exactReadyMarkup, /Exact row evidence bundle\./, 'API view should render exact-row evidence text')
  assert.match(exactReadyMarkup, /exact row fixture output/, 'API view should render latest exact-row output evidence')
  assert.match(exactReadyMarkup, /Template\/Jinja readiness[\s\S]*Template readiness is green for this supported exact row/, 'API view should show resolved template/Jinja as a green exact-row readiness lane')
  assert.match(exactReadyMarkup, /Throughput readiness[\s\S]*Production-throughput readiness is green for this supported exact row/, 'API view should show resolved production-throughput as a green exact-row readiness lane')
  assert.match(exactReadyMarkup, /Remaining support boundary:<\/b> model-native\/larger context beyond checked packs; portability; and durable repeated current-head bundles remain missing/, 'API view should keep unresolved row blockers while filtering resolved template/Jinja and throughput caveats')
  assert.doesNotMatch(exactReadyMarkup, /arbitrary-template behavior|arbitrary\/Jinja templates, production throughput|throughput, portability, neighboring-row/, 'API support surface should not repeat resolved template/Jinja or throughput caveats as generic blockers')
  assert.doesNotMatch(exactReadyMarkup, /normalizing model-native\/larger context; arbitrary\/Jinja template behavior; production throughput/, 'API compatibility list next-step copy should filter resolved template/Jinja and production-throughput caveats')
  assert.match(exactReadyMarkup, /Supported API feature rows/, 'API view should render supported feature rows from /api/capabilities')
  assert.match(exactReadyMarkup, /chat completions/, 'API view should display provider-scoped feature ids as neutral capability names')
  assert.match(exactReadyMarkup, /standard-compatible streaming stays enabled\./, 'API view should sanitize provider-specific feature notes before rendering')
  assert.match(exactReadyMarkup, /responses stream/, 'API view should normalize provider-scoped dotted feature ids before rendering')
  assert.match(exactReadyMarkup, /hosted model-style streamed response compatibility stays provider-neutral/, 'API view should neutralize hosted-brand feature notes before rendering')
  assert.match(exactReadyMarkup, /Guarded feature row; do not label it hosted model or hosted model compatible from API metadata\./, 'API view should also neutralize guarded feature metadata before rendering')
  assert.doesNotMatch(exactReadyMarkup, /openai|OpenAI|ChatGPT|Claude|Gemini|broad_family_trap|broad_quant_trap/, 'API view must not promote broad family/quant lists or raw provider-scoped/hosted-brand feature labels as support evidence')

  const mismatchedRuntimeMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: { ...readyRuntime, active_model_id: 'different-loaded-model' },
    selectedModel,
    capabilities,
  }))

  assert.match(mismatchedRuntimeMarkup, /Different loaded model is ready/, 'API readiness should fail closed when active_model_id differs from the selected exact row')
  assert.match(mismatchedRuntimeMarkup, /Blocked for UX chat until selected exact row evidence and runtime readiness both match/, 'API curl should stay blocked until exact row and runtime readiness both match')
  assert.doesNotMatch(mismatchedRuntimeMarkup, /Selected exact row ready/, 'mismatched runtime must not claim selected exact-row readiness')

  const plannedExactModel = {
    id: 'mistral-7b-instruct-v0.3-q8_0',
    name: 'Mistral 7B Instruct v0.3 Q8_0',
    provider_kind: 'local',
    status: 'ready',
    loaded_now: true,
    generation_ready: true,
    quant: 'Q8_0',
    model_path: '/models/mistral-7b-instruct-v0.3-q8_0.gguf',
  }
  const plannedExactCapabilities = {
    ...capabilities,
    model_compatibility: [
      ...capabilities.model_compatibility,
      {
        id: 'mistral_7b_instruct_v0_3_q8_0',
        status: 'planned',
        family: 'mistral',
        quantization: 'Q8_0',
        support_scope: 'exact row only once validated',
        frontend_readiness_gate: 'must stay blocked until supported',
        latest_checked_bucket: 'not_started',
        latest_checked_result: 'not_started',
        latest_checked_output: 'no validated output yet',
        full_support_status: 'not_supported',
        full_support_blockers: 'generation evidence missing',
        evidence: 'Planned exact-row placeholder, not runnable support.',
        next_step: 'Collect exact-row evidence before unlocking chat.',
      },
    ],
  }
  const plannedExactMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: { ...readyRuntime, active_model_id: plannedExactModel.id },
    selectedModel: plannedExactModel,
    capabilities: plannedExactCapabilities,
  }))

  assert.match(plannedExactMarkup, /mistral_7b_instruct_v0_3_q8_0/, 'API view should show selected planned exact-row evidence by row id')
  assert.match(plannedExactMarkup, /Generation ready; exact row required/, 'API readiness should stay guarded when the selected exact row is not supported')
  assert.match(plannedExactMarkup, /Planned exact-row placeholder, not runnable support\./, 'API view should render the exact row evidence without broad-family inference')
  assert.doesNotMatch(plannedExactMarkup, /Selected exact row ready/, 'planned exact rows must not claim selected exact-row readiness even when runtime health is green')

  const genericExactModel = {
    id: 'custom-exact-row-q8-0',
    name: 'Custom exact row Q8_0',
    provider_kind: 'local',
    status: 'ready',
    loaded_now: true,
    generation_ready: true,
    quant: 'Q8_0',
  }
  const genericExactCapabilities = {
    support_contract: capabilities.support_contract,
    model_compatibility: [
      {
        id: 'custom_exact_row_q8_0',
        status: 'supported_exact_row_smoke',
        family: 'custom_decoder',
        quantization: 'Q8_0',
        support_scope: 'exact custom row only',
        frontend_readiness_gate: 'green only when this exact custom row is selected and loaded',
        latest_checked_bucket: 'frontend_fixture',
        latest_checked_result: 'pass',
        latest_checked_output: 'custom exact row fixture output',
        full_support_status: 'blocked_pending_normalized_full_support',
        full_support_blockers: 'no neighboring custom rows inherit support',
        evidence: 'Custom exact row evidence from /api/capabilities.',
        next_step: 'Keep row-id scoped.',
      },
    ],
    api_features: [],
  }
  const genericExactMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: { ...readyRuntime, active_model_id: genericExactModel.id },
    selectedModel: genericExactModel,
    capabilities: genericExactCapabilities,
  }))

  assert.match(genericExactMarkup, /Selected exact row ready/, 'API view should support generic exact compatibility row ids without family-specific frontend matchers')
  assert.match(genericExactMarkup, /custom_exact_row_q8_0/, 'API view should render generic selected exact-row ids from capabilities')
  assert.match(genericExactMarkup, /Custom exact row evidence from \/api\/capabilities\./, 'API view should render generic exact-row evidence text')
  assert.doesNotMatch(genericExactMarkup, /No selected model exact row matched/, 'generic exact row-id matches should not fall through to broad or missing support copy')

  console.log('Frontend integration smoke passed')
} finally {
  await server.close()
}
