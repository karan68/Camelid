/* Model task/capability helpers.

   `embedding_capable` and `generation_capable` come from Camelid's header-only
   inspect/local-model APIs and are the authoritative signals. The fallbacks keep
   the UI honest while talking to an older backend build that predates those
   booleans; they are deliberately exact-artifact/runtime checks, never broad
   architecture guesses (the Microsoft encoders reuse qwen3/gemma3). */

const LEGACY_EMBEDDING_IDENTITIES = new Set([
  'nomic-embed-text-v1.5.q8_0.gguf',
  'nomic-embed-text-v1.5',
  'bitnet-embeddings-0.6b-bf16-i2_s.gguf',
  'bitnet-embeddings-0.6b',
  'bitnet-embeddings-270m-bf16-i2_s.gguf',
  'bitnet-embeddings-270m',
])

function normalized(value) {
  return String(value || '').trim().toLowerCase()
}

function modelIdentities(model) {
  return [
    model?.id,
    model?.runtime_model_name,
    model?.filename,
    model?.model_path,
    model?.path,
    model?.name,
  ]
    .flatMap((value) => {
      const identity = normalized(value)
      if (!identity) return []
      const filename = identity.split(/[\\/]/).filter(Boolean).pop() || identity
      return identity === filename ? [identity] : [identity, filename]
    })
}

function modelMatchesActiveRuntime(model, runtime) {
  const activeId = normalized(runtime?.active_model_id)
  if (!activeId || !model) return false
  return modelIdentities(model).some((identity) => identity === activeId)
}

function hasEmbeddingTaskTag(model) {
  const tags = Array.isArray(model?.task_tags) ? model.task_tags : []
  return tags.some((tag) => ['embedding', 'embeddings', 'retrieval', 'rerank', 'reranking'].includes(normalized(tag)))
}

export function isEmbeddingOnlyModel(model, runtime = null) {
  if (!model) return false

  // These paired booleans are authoritative when present. In particular,
  // `chat_capable=false` is NOT equivalent: a causal base model may simply lack
  // a chat template while remaining generation-capable.
  if (model?.generation_capable === true) return false
  if (model?.embedding_capable === true && model?.generation_capable === false) return true

  const activeRuntime = modelMatchesActiveRuntime(model, runtime)
  if (activeRuntime && normalized(runtime?.model_family) === 'embedding') return true
  if (
    activeRuntime
    && runtime?.current_model?.unsupported_runtime?.code === 'model_not_generation_capable'
  ) return true

  if (normalized(model?.task_kind) === 'embedding' || hasEmbeddingTaskTag(model)) return true
  if (normalized(model?.architecture) === 'nomic-bert') return true
  return modelIdentities(model).some((identity) => LEGACY_EMBEDDING_IDENTITIES.has(identity))
}

export function modelTaskKind(model, runtime = null) {
  if (isEmbeddingOnlyModel(model, runtime)) return 'embedding'
  if (model?.generation_capable === true) return 'generation'
  if (model?.generation_capable === false && model?.embedding_capable !== true) return 'companion'
  const explicit = normalized(model?.task_kind)
  if (explicit === 'generation' || explicit === 'chat') return 'generation'
  return 'unknown'
}

export function isGenerationCapableModel(model, runtime = null) {
  if (!model || isEmbeddingOnlyModel(model, runtime)) return false
  return model.generation_capable !== false
}

export function modelCapabilityFields(model, runtime = null) {
  const embeddingOnly = isEmbeddingOnlyModel(model, runtime)
  return {
    embedding_capable: model?.embedding_capable === true || embeddingOnly,
    generation_capable: typeof model?.generation_capable === 'boolean'
      ? model.generation_capable
      : embeddingOnly
        ? false
        : null,
    task_kind: embeddingOnly ? 'embedding' : modelTaskKind(model, runtime),
  }
}

function filenameFromValue(value) {
  return String(value || '').split(/[\\/]/).filter(Boolean).pop() || ''
}

function localRecordFilename(record) {
  return filenameFromValue(record?.model_path || record?.path || record?.filename || '')
}

function residentItemFilename(item) {
  const explicit = item?.filename
    || item?.model_path
    || item?.path
    || item?.meta?.filename
    || item?.meta?.model_path
    || item?.meta?.path
  if (explicit) return filenameFromValue(explicit)
  const id = String(item?.id || '')
  return id.toLowerCase().endsWith('.gguf') ? filenameFromValue(id) : ''
}

/* Join each resident `/v1/models` item to at most one saved local record.

   Direct runtime ids win. Exact filenames cover non-active sidecars, whose
   runtime id is the filename while the saved record may retain a catalog id.
   The active current-model filename is the final fallback for startup loads
   whose runtime id came from GGUF `general.name`. Consuming each record once
   prevents two runtime aliases from collapsing onto the same saved row. */
export function matchResidentItemsToLocalRecords({
  items = [],
  records = [],
  activeModelId = '',
  activeFilename = '',
} = {}) {
  const claimed = new Set()
  return items.map((item) => {
    const available = (record) => !claimed.has(record)
    let match = records.find((record) => available(record)
      && (item?.id === record?.id || item?.id === record?.runtime_model_name))
    const itemFilename = residentItemFilename(item)
    if (!match && itemFilename) {
      match = records.find((record) => available(record) && localRecordFilename(record) === itemFilename)
    }
    if (!match && item?.id === activeModelId && activeFilename) {
      match = records.find((record) => available(record) && localRecordFilename(record) === activeFilename)
    }
    if (match) claimed.add(match)
    return match || null
  })
}
