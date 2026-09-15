import { createServer } from 'vite'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import assert from 'node:assert/strict'

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' })
try {
  const { default: ModelsView } = await server.ssrLoadModule('/src/views/ModelsView.jsx')
  const { default: ChatWorkspace } = await server.ssrLoadModule('/src/views/ChatWorkspace.jsx')
  const progress = [{ filename: 'test-model.gguf', bytes_read: 512, total_bytes: 1024 }]
  const runtime = { status: 'online', model_load_progress: progress }
  const modelsHtml = renderToStaticMarkup(createElement(ModelsView, {
    runtime, capabilities: {}, registerForm: {}, setRegisterForm: () => {},
  }))
  assert.ok(modelsHtml.includes('Checking test-model.gguf: 512 B of 1.0 KB read.'))
  const chatProps = { runtime, models: [], composer: '', capabilities: {} }
  const progressHtml = renderToStaticMarkup(createElement(ChatWorkspace, chatProps))
  assert.ok(progressHtml.includes('Checking test-model.gguf: 50% read.'))
  const model = { id: 'test-model.gguf', name: 'Test model', model_path: '/models/test-model.gguf', loaded_now: true, status: 'registered' }
  const blockedHtml = renderToStaticMarkup(createElement(ChatWorkspace, {
    ...chatProps, models: [model], selectedModel: model, selectedModelId: model.id,
    runtime: { status: 'online', loaded_now: true, active_model_id: model.id, generation_ready: false,
      generation_readiness_reason: 'Weight storage exceeds the configured budget.' },
  }))
  assert.ok(blockedHtml.includes('Weight storage exceeds the configured budget.'))
  assert.ok(!blockedHtml.includes('this build cannot run it for Chat'))
  assert.ok(!blockedHtml.includes('not runnable for Chat in this build'))
  const unknownBlockerHtml = renderToStaticMarkup(createElement(ChatWorkspace, {
    ...chatProps, models: [model], selectedModel: model, selectedModelId: model.id,
    runtime: { status: 'online', loaded_now: true, active_model_id: model.id, generation_ready: false },
  }))
  assert.ok(unknownBlockerHtml.includes('Chat is unavailable. Check Models for details.'))
  assert.ok(!unknownBlockerHtml.includes('waiting for runtime readiness'))
  console.log('Model loading status smoke: Models progress, Chat progress, and actual blocker rendered')
} finally { await server.close() }
