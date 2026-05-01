import { useEffect, useMemo, useState } from 'react'
import { compatibilityHintCopy, compatibilityHintLabel, findCompatibilityHint, isCompatibilitySupportedForModel, quantLabelFromGgufFileType } from '../lib/capabilities'
import { isExternalModel, isRunnableInCurrentRuntime, isRunnableModel } from '../lib/modelState'

const TAB_STORAGE_KEY = 'camelid.activeTab'
const SELECTED_CONVERSATION_STORAGE_KEY = 'camelid.selectedConversationId'
const SELECTED_MODEL_STORAGE_KEY = 'camelid.selectedModelId'
const LOCAL_MODELS_STORAGE_KEY = 'camelid.localModels'
const CONVERSATIONS_STORAGE_KEY = 'backendinference.conversations'
const MEMORIES_STORAGE_KEY = 'backendinference.memories'
const API_BASE_STORAGE_KEY = 'backendinference.apiBase'
const VALID_TABS = new Set(['chat', 'library', 'api', 'analytics', 'history', 'memory', 'system'])
const NEW_CHAT_SENTINEL = '__new__'
const DEFAULT_API_BASE = import.meta.env.VITE_BACKENDINFERENCE_API_BASE || 'http://127.0.0.1:8181'
const LOCAL_CHAT_DEMO_MAX_TOKENS = 16

function getInitialTab() {
  if (typeof window === 'undefined') return 'chat'
  const saved = window.localStorage.getItem(TAB_STORAGE_KEY)
  return saved && VALID_TABS.has(saved) ? saved : 'chat'
}

function getInitialConversationId() {
  if (typeof window === 'undefined') return null
  return window.localStorage.getItem(SELECTED_CONVERSATION_STORAGE_KEY) || null
}

function getInitialModelId() {
  if (typeof window === 'undefined') return ''
  return window.localStorage.getItem(SELECTED_MODEL_STORAGE_KEY) || ''
}

function getApiBase() {
  if (typeof window === 'undefined') return DEFAULT_API_BASE
  return window.localStorage.getItem(API_BASE_STORAGE_KEY) || DEFAULT_API_BASE
}

function normalizeApiBase(value) {
  return (value || DEFAULT_API_BASE).trim().replace(/\/$/, '')
}

function readJsonStorage(key, fallback) {
  if (typeof window === 'undefined') return fallback
  try {
    const saved = window.localStorage.getItem(key)
    return saved ? JSON.parse(saved) : fallback
  } catch {
    window.localStorage.removeItem(key)
    return fallback
  }
}

function writeJsonStorage(key, value) {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(key, JSON.stringify(value))
}

function normalizeSortText(value) {
  return (value || '').toString().trim().toLowerCase()
}

function compareModelsByName(left, right) {
  return normalizeSortText(left.name).localeCompare(normalizeSortText(right.name), undefined, {
    numeric: true,
    sensitivity: 'base',
  }) || normalizeSortText(left.id).localeCompare(normalizeSortText(right.id), undefined, {
    numeric: true,
    sensitivity: 'base',
  })
}

function getModelPath(model) {
  return typeof model?.path === 'string' ? model.path : ''
}

function getLoadedModelFileType(model) {
  const metadata = model?.gguf?.metadata || {}
  return metadata?.general?.file_type ?? metadata?.['general.file_type'] ?? null
}

function getLoadedModelQuantLabel(model) {
  const fileType = getLoadedModelFileType(model)
  if (fileType === null || fileType === undefined) return null
  return quantLabelFromGgufFileType(fileType) || `file_type ${fileType}`
}

function finiteNumber(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function topFiveLogitProbabilities(response) {
  return (response?.backendinference?.top_logits || [])
    .slice(0, 5)
    .map((entry) => ({
      token_id: entry.token_id,
      rank: entry.rank,
      text: entry.text || `#${entry.token_id}`,
      logit: finiteNumber(entry.logit),
      probability: finiteNumber(entry.probability),
      selected: Boolean(entry.selected),
    }))
}

function completionTokensPerSecond(response, elapsedMs) {
  const completionTokens = finiteNumber(response?.usage?.completion_tokens)
  if (!completionTokens || completionTokens <= 0) return null
  const generationMs = finiteNumber(response?.backendinference?.timings_ms?.generate)
  const denominatorMs = generationMs && generationMs > 0 ? generationMs : finiteNumber(elapsedMs)
  if (!denominatorMs || denominatorMs <= 0) return null
  return completionTokens / (denominatorMs / 1000)
}

function isLoadedModelGenerationReady(model) {
  return Boolean(model?.llama_config && model?.llama_tensors && model?.tokenizer?.status === 'available')
}

function fallbackModelName(id, modelPath) {
  if (id) return id
  const fileName = modelPath?.split('/').filter(Boolean).pop() || ''
  return fileName.replace(/\.gguf$/i, '') || 'Local GGUF model'
}

function optionalString(value) {
  if (typeof value !== 'string') return null
  const trimmed = value.trim()
  return trimmed || null
}

function normalizeLocalModelStatus(status) {
  return status === 'ready' || status === 'registered' || status === 'failed' ? status : 'registered'
}

function normalizeLocalModelRecord(record) {
  if (!record || typeof record !== 'object') return null
  const modelPath = String(record.model_path || record.path || '').trim()
  const id = String(record.id || record.runtime_model_name || fallbackModelName('', modelPath)).trim()
  if (!id || !modelPath) return null
  return {
    id,
    name: String(record.name || fallbackModelName(id, modelPath)).trim(),
    provider_kind: 'local',
    status: normalizeLocalModelStatus(record.status),
    model_path: modelPath,
    runtime_model_name: String(record.runtime_model_name || id).trim(),
    source: record.source || 'Local GGUF file',
    engine: record.engine || 'backendinference',
    quant: record.quant || null,
    size_gb: record.size_gb || null,
    api_base: record.api_base || null,
    api_key_configured: false,
    install_error: optionalString(record.install_error),
    load_error: optionalString(record.load_error),
    last_load_attempt_at: optionalString(record.last_load_attempt_at),
    last_loaded_at: optionalString(record.last_loaded_at),
    loaded_now: false,
    generation_ready: false,
    backendinference: {
      active: false,
      loaded_now: false,
      generation_ready: false,
      tokenizer_status: null,
      tokenizer_model: null,
      tensor_ready: false,
      config_ready: false,
    },
    updated_at: record.updated_at || nowIso(),
  }
}

function upsertLocalModelRecord(records, record) {
  const normalized = normalizeLocalModelRecord(record)
  if (!normalized) return records
  return [normalized, ...records.filter((item) => item.id !== normalized.id)].sort(compareModelsByName)
}

function modelReadinessFromCurrent(currentModel, active, generationReady) {
  return {
    active,
    loaded_now: active,
    generation_ready: generationReady,
    tokenizer_status: active ? currentModel?.tokenizer?.status || null : null,
    tokenizer_model: active ? currentModel?.tokenizer?.model || null : null,
    tensor_ready: active ? Boolean(currentModel?.llama_tensors) : false,
    config_ready: active ? Boolean(currentModel?.llama_config) : false,
  }
}

function modelFromLocalRecord(record, health, currentModel, apiBase) {
  const active = health?.active_model_id === record.id
  const generationReady = active && Boolean(health?.generation_ready)
  return {
    ...record,
    status: generationReady ? 'ready' : record.status,
    model_path: active ? getModelPath(currentModel) || record.model_path : record.model_path,
    api_base: apiBase,
    install_error: active ? null : record.install_error,
    load_error: active ? null : record.load_error,
    loaded_now: active,
    generation_ready: generationReady,
    backendinference: modelReadinessFromCurrent(currentModel, active, generationReady),
  }
}

function modelFromBackend(item, health, currentModel, localRecord, apiBase) {
  const active = health?.active_model_id === item.id
  const generationReady = active && Boolean(health?.generation_ready)
  const tokenizer = active ? currentModel?.tokenizer : null
  const quantLabel = active ? getLoadedModelQuantLabel(currentModel) : null
  const modelPath = active ? getModelPath(currentModel) || localRecord?.model_path || '' : localRecord?.model_path || ''

  return {
    id: item.id,
    name: localRecord?.name || item.id,
    provider_kind: 'local',
    status: generationReady ? 'ready' : localRecord?.status || 'registered',
    model_path: modelPath,
    runtime_model_name: item.id,
    source: localRecord?.source || 'Camelid local runtime',
    engine: 'backendinference',
    quant: quantLabel || localRecord?.quant || null,
    size_gb: localRecord?.size_gb || null,
    api_base: apiBase,
    api_key_configured: false,
    install_error: active ? null : localRecord?.install_error || null,
    load_error: active ? null : localRecord?.load_error || null,
    last_load_attempt_at: localRecord?.last_load_attempt_at || null,
    last_loaded_at: localRecord?.last_loaded_at || null,
    loaded_now: active,
    generation_ready: generationReady,
    backendinference: modelReadinessFromCurrent(currentModel, active, generationReady),
  }
}

function mergeModelLists({ modelItems, health, currentModel, localModels, apiBase }) {
  const localRecords = localModels.map(normalizeLocalModelRecord).filter(Boolean)
  const byId = new Map()
  localRecords.forEach((record) => {
    byId.set(record.id, modelFromLocalRecord(record, health, currentModel, apiBase))
  })
  modelItems.forEach((item) => {
    const localRecord = localRecords.find((record) => record.id === item.id) || null
    byId.set(item.id, modelFromBackend(item, health, currentModel, localRecord, apiBase))
  })
  return [...byId.values()].sort(compareModelsByName)
}

function nowIso() {
  return new Date().toISOString()
}

function makeId(prefix) {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return `${prefix}-${crypto.randomUUID()}`
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function getErrorMessage(error, fallback = 'Request failed.') {
  if (!error) return fallback
  if (typeof error === 'string') return error
  return error?.body?.error?.message || error?.error?.message || error?.message || fallback
}

function getBackendErrorCode(error) {
  return error?.body?.error?.code || error?.error?.code || ''
}

function isTypedUnsupportedBackendError(code, message) {
  const normalized = `${code} ${message}`.toLowerCase()
  return normalized.includes('unsupported')
    || normalized.includes('not_supported')
    || normalized.includes('cpu_weight_materialization_exceeds_budget')
    || normalized.includes('exceeds_budget')
}

function getGuardrailErrorMessage(error, fallback = 'Request failed.') {
  const message = getErrorMessage(error, fallback)
  const code = getBackendErrorCode(error)
  if (!isTypedUnsupportedBackendError(code, message)) return message
  const codeCopy = code ? ` (${code})` : ''
  return `Camelid refused this with a typed guardrail${codeCopy}: ${message}. This is not chat-ready support yet; check /api/capabilities and COMPATIBILITY.md before retrying.`
}

async function fetchJson(pathOrUrl, options = {}) {
  const response = await fetch(pathOrUrl, {
    ...options,
    headers: {
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...(options.headers || {}),
    },
  })
  const text = await response.text()
  let body = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = text
    }
  }
  if (!response.ok) {
    const error = new Error(typeof body === 'string' ? body : getErrorMessage(body, response.statusText))
    error.status = response.status
    error.body = body
    throw error
  }
  return body
}

function makeDashboard({ health, models, currentModel, capabilities, conversations, memories, apiBase }) {
  return {
    app: 'backendinference',
    api_base: apiBase,
    health,
    capabilities,
    conversations,
    memories,
    models,
    runtime: {
      engine: health?.engine || 'backendinference',
      loaded_now: Boolean(health?.active_model_id),
      active_model_id: health?.active_model_id || null,
      generation_ready: Boolean(health?.generation_ready),
      status: health?.ok ? 'online' : 'offline',
      api_base: apiBase,
      current_model: currentModel || null,
    },
    stats: {
      conversation_count: conversations.length,
      memory_count: memories.length,
      model_count: models.length,
    },
  }
}

export function useDashboardData({ showNotice, clearNotice }) {
  const [dashboard, setDashboard] = useState(null)
  const [apiBase, setApiBaseState] = useState(getApiBase)
  const [tab, setTab] = useState(getInitialTab)
  const [selectedConversationId, setSelectedConversationId] = useState(getInitialConversationId)
  const [selectedModelId, setSelectedModelId] = useState(getInitialModelId)
  const [search, setSearch] = useState('')
  const [memorySearch, setMemorySearch] = useState('')
  const [composer, setComposer] = useState('')
  const [newChatTitle, setNewChatTitle] = useState('')
  const [sending, setSending] = useState(false)
  const [loadingModelId, setLoadingModelId] = useState('')
  const [pendingChat, setPendingChat] = useState(null)
  const [registerForm, setRegisterForm] = useState({ id: '', name: '', model_path: '', runtime_model_name: '' })
  const [externalForm, setExternalForm] = useState({ id: '', name: '', source: 'OpenAI', api_base: 'https://api.openai.com/v1', api_key: '', model_name: '' })
  const [localModels, setLocalModels] = useState(() => readJsonStorage(LOCAL_MODELS_STORAGE_KEY, []).map(normalizeLocalModelRecord).filter(Boolean))
  const [localConversations, setLocalConversations] = useState(() => readJsonStorage(CONVERSATIONS_STORAGE_KEY, []))
  const [localMemories, setLocalMemories] = useState(() => readJsonStorage(MEMORIES_STORAGE_KEY, []))

  const normalizedApiBase = normalizeApiBase(apiBase)

  const persistConversations = (updater) => {
    setLocalConversations((current) => {
      const next = typeof updater === 'function' ? updater(current) : updater
      writeJsonStorage(CONVERSATIONS_STORAGE_KEY, next)
      return next
    })
  }

  const persistMemories = (updater) => {
    setLocalMemories((current) => {
      const next = typeof updater === 'function' ? updater(current) : updater
      writeJsonStorage(MEMORIES_STORAGE_KEY, next)
      return next
    })
  }

  const persistLocalModels = (updater) => {
    const nextModels = (typeof updater === 'function' ? updater(localModels) : updater)
      .map(normalizeLocalModelRecord)
      .filter(Boolean)
      .sort(compareModelsByName)
    writeJsonStorage(LOCAL_MODELS_STORAGE_KEY, nextModels)
    setLocalModels(nextModels)
    return nextModels
  }

  const loadDashboard = async ({ silent = false, localModelsOverride = null } = {}) => {
    try {
      const [health, modelList, capabilities] = await Promise.all([
        fetchJson(`${normalizedApiBase}/v1/health`),
        fetchJson(`${normalizedApiBase}/v1/models`),
        fetchJson(`${normalizedApiBase}/api/capabilities`).catch(() => null),
      ])
      const currentModel = health?.active_model_id
        ? await fetchJson(`${normalizedApiBase}/api/models/current`).catch(() => null)
        : null
      const modelItems = Array.isArray(modelList?.data) ? modelList.data : []
      const nextModels = mergeModelLists({
        modelItems,
        health,
        currentModel,
        localModels: localModelsOverride || localModels,
        apiBase: normalizedApiBase,
      })
      const nextDashboard = makeDashboard({
        health,
        models: nextModels,
        currentModel,
        capabilities,
        conversations: localConversations,
        memories: localMemories,
        apiBase: normalizedApiBase,
      })
      setDashboard(nextDashboard)
      if (!silent) clearNotice()
      setSelectedConversationId((current) => {
        if (current === NEW_CHAT_SENTINEL) return current
        if (!localConversations.length) return null
        if (current && localConversations.some((conversation) => conversation.id === current)) return current
        return localConversations[0]?.id || null
      })
      setSelectedModelId((current) => {
        if (!nextModels.length) return ''
        const currentModel = current ? nextModels.find((model) => model.id === current) : null
        const activeModel = health?.active_model_id ? nextModels.find((model) => model.id === health.active_model_id) : null
        const runnableModel = nextModels.find((model) => isRunnableModel(model)) || null

        if (currentModel && isRunnableModel(currentModel)) return current
        if (activeModel && currentModel?.id !== activeModel.id) return activeModel.id
        if (currentModel) return current
        return runnableModel?.id || activeModel?.id || nextModels[0]?.id || ''
      })
    } catch (error) {
      const fallbackDashboard = makeDashboard({
        health: { ok: false, engine: 'backendinference', generation_ready: false, active_model_id: null },
        models: mergeModelLists({
          modelItems: [],
          health: { ok: false, engine: 'backendinference', generation_ready: false, active_model_id: null },
          currentModel: null,
          localModels: localModelsOverride || localModels,
          apiBase: normalizedApiBase,
        }),
        currentModel: null,
        capabilities: null,
        conversations: localConversations,
        memories: localMemories,
        apiBase: normalizedApiBase,
      })
      setDashboard(fallbackDashboard)
      if (!silent) showNotice(`Could not reach Camelid at ${normalizedApiBase}: ${getErrorMessage(error)}`, 'error')
    }
  }

  useEffect(() => {
    loadDashboard()
    const interval = setInterval(() => loadDashboard({ silent: true }), 2500)
    return () => clearInterval(interval)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [normalizedApiBase, localConversations, localMemories, localModels])

  useEffect(() => {
    if (typeof window === 'undefined' || !VALID_TABS.has(tab)) return
    window.localStorage.setItem(TAB_STORAGE_KEY, tab)
  }, [tab])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (!selectedConversationId) window.localStorage.removeItem(SELECTED_CONVERSATION_STORAGE_KEY)
    else window.localStorage.setItem(SELECTED_CONVERSATION_STORAGE_KEY, selectedConversationId)
  }, [selectedConversationId])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (!selectedModelId) window.localStorage.removeItem(SELECTED_MODEL_STORAGE_KEY)
    else window.localStorage.setItem(SELECTED_MODEL_STORAGE_KEY, selectedModelId)
  }, [selectedModelId])

  const conversations = dashboard?.conversations || localConversations
  const memories = dashboard?.memories || localMemories
  const models = dashboard?.models || []
  const runtime = dashboard?.runtime

  const selectedConversation = useMemo(() => {
    if (selectedConversationId === NEW_CHAT_SENTINEL) return null
    return conversations.find((conversation) => conversation.id === selectedConversationId) || conversations[0] || null
  }, [conversations, selectedConversationId])

  const selectedModel = useMemo(() => models.find((model) => model.id === selectedModelId) || models[0], [models, selectedModelId])
  const selectedModelRuntimeReady = isRunnableInCurrentRuntime(selectedModel, runtime)
  const selectedModelCapabilitySupported = isCompatibilitySupportedForModel(dashboard?.capabilities, selectedModel)
  const selectedModelRunnable = selectedModelRuntimeReady && selectedModelCapabilitySupported
  const pendingConversation = pendingChat?.conversationId
    && (selectedConversation?.id === pendingChat.conversationId || selectedConversationId === pendingChat.conversationId)
    ? pendingChat
    : null

  const filteredConversations = useMemo(() => {
    if (!search.trim()) return conversations
    const q = search.toLowerCase()
    return conversations.filter((conversation) =>
      conversation.title.toLowerCase().includes(q)
      || conversation.messages.some((message) => message.content.toLowerCase().includes(q)),
    )
  }, [conversations, search])

  const filteredMemories = useMemo(() => {
    if (!memorySearch.trim()) return memories
    const q = memorySearch.toLowerCase()
    return memories.filter((memory) =>
      memory.title.toLowerCase().includes(q)
      || memory.body.toLowerCase().includes(q)
      || memory.scope.toLowerCase().includes(q),
    )
  }, [memories, memorySearch])

  const latestAssistantMessage = useMemo(
    () => [...(selectedConversation?.messages || [])].reverse().find((message) => message.role === 'assistant'),
    [selectedConversation],
  )

  const createConversationRecord = async ({ manualTitle = '', silent = false } = {}) => {
    const conversation = {
      id: makeId('conversation'),
      title: manualTitle || 'New conversation',
      model_id: selectedModelId || models[0]?.id || null,
      messages: manualTitle ? [] : [{ id: makeId('message'), role: 'assistant', content: 'Conversation created. Load a Camelid model and send a prompt when ready.', created_at: nowIso() }],
      created_at: nowIso(),
      updated_at: nowIso(),
    }
    persistConversations((current) => [conversation, ...current])
    setSelectedConversationId(conversation.id)
    setTab('chat')
    setNewChatTitle('')
    if (!silent) showNotice(manualTitle ? 'Conversation created locally.' : 'Conversation created locally.', 'success')
    return conversation
  }

  const createConversation = async () => {
    try {
      await createConversationRecord({ manualTitle: newChatTitle.trim() })
    } catch (error) {
      showNotice(error.message || 'Could not create the conversation.', 'error')
    }
  }

  const ensureConversation = async () => selectedConversation || createConversationRecord({ silent: true })

  const sendMessage = async () => {
    if (!composer.trim()) return
    if (!selectedModelRunnable) {
      if (selectedModelRuntimeReady && !selectedModelCapabilitySupported) {
        const hint = findCompatibilityHint(dashboard?.capabilities, selectedModel)
        showNotice(`${compatibilityHintLabel(hint, 'No matching COMPATIBILITY.md row')}: ${compatibilityHintCopy(hint)} Chat is blocked until /api/capabilities marks this exact model/quant as supported.`, 'error')
      } else {
        showNotice('Camelid is not generation-ready for the selected model yet.', 'error')
      }
      return
    }

    const messageContent = composer.trim()
    setSending(true)
    showNotice(`Running Camelid local chat completion (${LOCAL_CHAT_DEMO_MAX_TOKENS} token demo cap)…`, 'info')

    try {
      const conversation = await ensureConversation()
      const userMessage = { id: makeId('message'), role: 'user', content: messageContent, model_id: selectedModelId, created_at: nowIso() }
      setPendingChat({ conversationId: conversation.id, content: messageContent, modelId: selectedModelId })
      setComposer('')

      const history = [...(conversation.messages || []), userMessage]
        .filter((message) => message.role === 'user' || message.role === 'assistant')
        .filter((message) => !message.content.startsWith('Conversation created.'))
        .map(({ role, content }) => ({ role, content }))

      persistConversations((current) => current.map((item) => (
        item.id === conversation.id
          ? { ...item, model_id: selectedModelId, messages: [...(item.messages || []), userMessage], updated_at: nowIso() }
          : item
      )))

      const requestStartedAt = performance.now()
      const response = await fetchJson(`${normalizedApiBase}/v1/chat/completions`, {
        method: 'POST',
        body: JSON.stringify({ model: selectedModelId, messages: history, max_tokens: LOCAL_CHAT_DEMO_MAX_TOKENS, temperature: 0, stream: false }),
      })
      const elapsedMs = performance.now() - requestStartedAt
      const assistantContent = response?.choices?.[0]?.message?.content || ''
      const assistantMessage = {
        id: response?.id || makeId('message'),
        role: 'assistant',
        content: assistantContent || '(empty response)',
        model_id: selectedModelId,
        model_name: selectedModel?.name || selectedModelId,
        created_at: nowIso(),
        tokens_in_per_sec: null,
        tokens_out_per_sec: completionTokensPerSecond(response, elapsedMs),
        top_logits: topFiveLogitProbabilities(response),
        generated_token_ids: Array.isArray(response?.backendinference?.generated_token_ids) ? response.backendinference.generated_token_ids : [],
        demo_token_cap: LOCAL_CHAT_DEMO_MAX_TOKENS,
        timings_ms: response?.backendinference?.timings_ms || null,
        usage: response?.usage || null,
      }
      persistConversations((current) => current.map((item) => (
        item.id === conversation.id
          ? { ...item, title: item.title === 'New conversation' ? messageContent.slice(0, 64) : item.title, messages: [...(item.messages || []), assistantMessage], updated_at: nowIso() }
          : item
      )))
      setPendingChat(null)
      setSelectedConversationId(conversation.id)
      showNotice(`Camelid returned a raw local reply with the ${LOCAL_CHAT_DEMO_MAX_TOKENS}-token demo cap. Inspect it before treating longer generation as polished.`, 'success')
    } catch (error) {
      setPendingChat(null)
      showNotice(getGuardrailErrorMessage(error, 'Local inference failed.'), 'error')
    } finally {
      setSending(false)
      await loadDashboard({ silent: true })
    }
  }

  const renameConversation = async (id, nextTitle) => {
    const trimmedTitle = nextTitle.trim()
    if (!trimmedTitle) {
      showNotice('Conversation title cannot be empty.', 'error')
      return false
    }
    persistConversations((current) => current.map((conversation) => conversation.id === id ? { ...conversation, title: trimmedTitle, updated_at: nowIso() } : conversation))
    showNotice('Conversation title updated.', 'success')
    return true
  }

  const deleteConversation = async (id) => {
    persistConversations((current) => current.filter((conversation) => conversation.id !== id))
    if (selectedConversationId === id) setSelectedConversationId(null)
    showNotice('Conversation deleted locally.', 'success')
    return true
  }

  const showNewChatLanding = () => {
    setTab('chat')
    setSelectedConversationId(NEW_CHAT_SENTINEL)
    setComposer('')
    setPendingChat(null)
  }

  const createMemory = async ({ title, body, scope = 'General' }) => {
    const memory = { id: makeId('memory'), title, body, scope, created_at: nowIso(), updated_at: nowIso() }
    persistMemories((current) => [memory, ...current])
    setTab('memory')
    showNotice('Memory saved in browser storage for this Camelid UI session.', 'success')
    return true
  }

  const updateMemory = async (id, changes, { successMessage = 'Memory updated.' } = {}) => {
    persistMemories((current) => current.map((memory) => memory.id === id ? { ...memory, ...changes, updated_at: nowIso() } : memory))
    if (successMessage) showNotice(successMessage, 'success')
    return true
  }

  const deleteMemory = async (id, { successMessage = 'Memory deleted.' } = {}) => {
    persistMemories((current) => current.filter((memory) => memory.id !== id))
    if (successMessage) showNotice(successMessage, 'success')
    return true
  }

  const saveToMemory = async () => {
    const latestAssistant = [...(selectedConversation?.messages || [])].reverse().find((message) => message.role === 'assistant')
    if (!latestAssistant) {
      showNotice('There is no assistant reply to save yet.', 'error')
      return
    }
    await createMemory({ title: `Saved from ${selectedConversation?.title?.trim() || 'Current chat'}`, body: latestAssistant.content, scope: 'Conversation' })
  }

  const installModel = async () => {
    showNotice('Model catalog downloads are not wired in Camelid yet. Camelid currently loads local GGUF paths.', 'error')
  }

  const installCatalogModel = async () => {
    showNotice('Hugging Face catalog install is not wired to Camelid yet.', 'error')
    return false
  }

  const cancelModelDownload = async () => {
    showNotice('No Camelid download is running from this UI.', 'info')
    return false
  }

  const activateModel = async (id) => {
    const model = models.find((item) => item.id === id) || localModels.find((item) => item.id === id)
    setSelectedModelId(id)

    if (!model) {
      showNotice('Choose a saved local model before loading it.', 'error')
      return
    }
    if (isExternalModel(model)) {
      showNotice('Hosted API chat routing is planned but not wired yet. Keep using local GGUF loading for now.', 'info')
      return
    }
    if (!model.model_path) {
      showNotice('This model needs a local GGUF path before Camelid can load it.', 'error')
      return
    }
    if (runtime?.active_model_id === id && runtime?.generation_ready) {
      showNotice('That model is already loaded and generation-ready.', 'success')
      return
    }

    setLoadingModelId(id)
    showNotice(`Loading ${model.name || id} into Camelid…`, 'info')
    try {
      const loaded = await fetchJson(`${normalizedApiBase}/api/models/load`, {
        method: 'POST',
        body: JSON.stringify({ id, path: model.model_path }),
      })
      const loadedId = loaded?.id || id
      const loadedPath = getModelPath(loaded) || model.model_path
      const ready = isLoadedModelGenerationReady(loaded)
      const fileType = getLoadedModelFileType(loaded)
      const quantLabel = getLoadedModelQuantLabel(loaded) || (fileType !== null && fileType !== undefined ? `file_type ${fileType}` : model.quant)
      const loadedRecord = {
        ...model,
        id: loadedId,
        model_path: loadedPath,
        status: ready ? 'ready' : 'registered',
        quant: quantLabel,
        install_error: null,
        load_error: null,
        last_load_attempt_at: nowIso(),
        last_loaded_at: nowIso(),
        updated_at: nowIso(),
      }
      const supportedByContract = isCompatibilitySupportedForModel(dashboard?.capabilities, loadedRecord)
      const nextLocalModels = persistLocalModels((current) => upsertLocalModelRecord(current, loadedRecord))
      setSelectedModelId(loadedId)
      await loadDashboard({ silent: true, localModelsOverride: nextLocalModels })
      showNotice(
        ready
          ? supportedByContract
            ? 'Model loaded. Camelid reports generation-ready and the /api/capabilities support contract matches this model/quant.'
            : 'Model loaded and generation-ready, but chat stays guarded until /api/capabilities has an exact supported COMPATIBILITY.md row for this model/quant.'
          : 'Model loaded, but Camelid does not report generation-ready yet. Check tokenizer/config/tensor readiness.',
        ready && supportedByContract ? 'success' : 'info',
      )
    } catch (error) {
      const message = getGuardrailErrorMessage(error, 'Could not load that local GGUF into Camelid.')
      const nextLocalModels = persistLocalModels((current) => upsertLocalModelRecord(current, {
        ...model,
        id,
        status: 'registered',
        install_error: null,
        load_error: message,
        last_load_attempt_at: nowIso(),
        updated_at: nowIso(),
      }))
      await loadDashboard({ silent: true, localModelsOverride: nextLocalModels })
      showNotice(message, 'error')
    } finally {
      setLoadingModelId((current) => current === id ? '' : current)
    }
  }

  const unloadCurrentModel = async () => {
    const activeModelId = runtime?.active_model_id
    if (!activeModelId) {
      showNotice('No model is loaded in Camelid right now.', 'info')
      return false
    }

    setLoadingModelId(activeModelId)
    showNotice(`Unloading ${activeModelId} from Camelid…`, 'info')
    try {
      await fetchJson(`${normalizedApiBase}/api/models/unload`, { method: 'POST' })
      await loadDashboard({ silent: true })
      showNotice('Camelid unloaded the current model. Local saved paths are unchanged.', 'success')
      return true
    } catch (error) {
      showNotice(getErrorMessage(error, 'Could not unload the current model.'), 'error')
      return false
    } finally {
      setLoadingModelId((current) => current === activeModelId ? '' : current)
    }
  }

  const connectExternalModel = async () => {
    showNotice('Hosted-provider setup is intentionally disabled until Camelid wires API routing.', 'info')
  }

  const registerModel = async () => {
    const name = registerForm.name.trim()
    const modelPath = registerForm.model_path.trim()
    const derivedId = registerForm.id.trim() || registerForm.runtime_model_name.trim() || name || modelPath.split('/').pop()?.replace(/\.gguf$/i, '') || ''
    if (!modelPath || !derivedId) {
      showNotice('Add a local GGUF path and model name before loading it into Camelid.', 'error')
      return
    }
    setLoadingModelId(derivedId)
    showNotice(`Loading ${name || derivedId} from the local GGUF path…`, 'info')
    try {
      const loaded = await fetchJson(`${normalizedApiBase}/api/models/load`, {
        method: 'POST',
        body: JSON.stringify({ id: derivedId, path: modelPath }),
      })
      const loadedId = loaded?.id || derivedId
      const loadedPath = getModelPath(loaded) || modelPath
      const ready = isLoadedModelGenerationReady(loaded)
      const fileType = getLoadedModelFileType(loaded)
      const quantLabel = getLoadedModelQuantLabel(loaded) || (fileType !== null && fileType !== undefined ? `file_type ${fileType}` : null)
      const loadedRecord = {
        id: loadedId,
        name: name || loadedId,
        model_path: loadedPath,
        runtime_model_name: registerForm.runtime_model_name.trim() || loadedId,
        status: ready ? 'ready' : 'registered',
        quant: quantLabel,
        install_error: null,
        load_error: null,
        last_load_attempt_at: nowIso(),
        last_loaded_at: nowIso(),
        updated_at: nowIso(),
      }
      const supportedByContract = isCompatibilitySupportedForModel(dashboard?.capabilities, loadedRecord)
      const nextLocalModels = persistLocalModels((current) => upsertLocalModelRecord(current, loadedRecord))
      setSelectedModelId(loadedId)
      setRegisterForm({ id: '', name: '', model_path: '', runtime_model_name: '' })
      await loadDashboard({ silent: true, localModelsOverride: nextLocalModels })
      showNotice(
        ready
          ? supportedByContract
            ? 'Local model saved, loaded, generation-ready, and matched to a supported /api/capabilities row.'
            : 'Local model saved and generation-ready, but chat stays guarded until COMPATIBILITY.md and /api/capabilities explicitly support this model/quant.'
          : 'Local model saved and loaded, but Camelid still reports generation is not ready.',
        ready && supportedByContract ? 'success' : 'info',
      )
    } catch (error) {
      const message = getGuardrailErrorMessage(error, 'Could not load that local GGUF.')
      const nextLocalModels = persistLocalModels((current) => upsertLocalModelRecord(current, {
        id: derivedId,
        name: name || derivedId,
        model_path: modelPath,
        runtime_model_name: registerForm.runtime_model_name.trim() || derivedId,
        status: 'registered',
        install_error: null,
        load_error: message,
        last_load_attempt_at: nowIso(),
        updated_at: nowIso(),
      }))
      setSelectedModelId(derivedId)
      await loadDashboard({ silent: true, localModelsOverride: nextLocalModels })
      showNotice(message, 'error')
    } finally {
      setLoadingModelId((current) => current === derivedId ? '' : current)
    }
  }

  return {
    dashboard,
    tab,
    setTab,
    selectedConversationId,
    setSelectedConversationId,
    selectedModelId,
    setSelectedModelId,
    search,
    setSearch,
    memorySearch,
    setMemorySearch,
    composer,
    setComposer,
    newChatTitle,
    setNewChatTitle,
    sending,
    loadingModelId,
    registerForm,
    setRegisterForm,
    externalForm,
    setExternalForm,
    conversations,
    memories,
    models,
    runtime,
    selectedConversation,
    selectedModel,
    selectedModelRunnable,
    filteredConversations,
    filteredMemories,
    latestAssistantMessage,
    pendingConversation,
    createConversation,
    showNewChatLanding,
    sendMessage,
    saveToMemory,
    createMemory,
    updateMemory,
    deleteMemory,
    renameConversation,
    deleteConversation,
    installModel,
    installCatalogModel,
    cancelModelDownload,
    activateModel,
    unloadCurrentModel,
    registerModel,
    connectExternalModel,
    loadDashboard,
    apiBase,
    setApiBase: (value) => {
      const next = normalizeApiBase(value)
      setApiBaseState(next)
      if (typeof window !== 'undefined') window.localStorage.setItem(API_BASE_STORAGE_KEY, next)
    },
  }
}
