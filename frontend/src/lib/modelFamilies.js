/* Presentation-only model-family grouping for the Models page.

   Support and load routing continue to use exact catalog rows and authoritative
   GGUF metadata. These labels only make long lists navigable, so it is safe to
   use product names (for example Bonsai) before falling back to the lower-level
   `general.architecture` value. */

const FAMILY_RULES = [
  ['Bonsai', /\bbonsai\b/i],
  ['Ornith', /\bornith\b/i],
  ['BitNet', /\bbitnet\b/i],
  ['DeepSeek', /\bdeepseek\b/i],
  ['Command R', /\bcommand[-_. ]?r\b|\bc4ai\b/i],
  ['Nomic Embed', /\bnomic[-_. ]?(?:embed|bert)\b/i],
  ['TinyLlama', /\btiny[-_. ]?llama\b/i],
  ['Llama', /\bllama\b/i],
  ['Qwen', /\bqwen\b/i],
  ['Gemma', /\bgemma\b/i],
  ['Mixtral', /\bmixtral\b/i],
  ['Mistral', /\bmistral\b/i],
  ['SmolLM', /\bsmol[-_. ]?lm(?:\d+)?\b/i],
  ['Phi', /\bphi[-_. ]?\d/i],
  ['LFM', /\blfm[-_. ]?\d/i],
]

const ARCHITECTURE_LABELS = new Map([
  ['llama', 'Llama'],
  ['mistral', 'Mistral'],
  ['qwen2', 'Qwen'],
  ['qwen25', 'Qwen'],
  ['qwen3', 'Qwen'],
  ['qwen3moe', 'Qwen'],
  ['qwen35', 'Qwen'],
  ['gemma2', 'Gemma'],
  ['gemma3', 'Gemma'],
  ['gemma4', 'Gemma'],
  ['phi3', 'Phi'],
  ['lfm2', 'LFM'],
  ['smollm3', 'SmolLM'],
  ['bitnet-b1.58', 'BitNet'],
  ['nomic-bert', 'Nomic Embed'],
  ['command-r', 'Command R'],
])

function searchableIdentity(item = {}) {
  return [
    item.name,
    item.filename,
    item.repo_id,
    item.repoId,
    item.title,
    item.catalog_id,
  ]
    .filter(Boolean)
    .join(' ')
}

export function modelFamily(item = {}) {
  const identity = searchableIdentity(item)
  for (const [label, pattern] of FAMILY_RULES) {
    if (pattern.test(identity)) return label
  }

  const architecture = String(item.architecture || '').trim().toLowerCase()
  if (ARCHITECTURE_LABELS.has(architecture)) return ARCHITECTURE_LABELS.get(architecture)
  if (architecture) return architecture
  return 'Other'
}

/* Preserve first appearance: catalog order is relevance/recommendation order and
   local order comes from the backend scan. Alphabetizing here would silently
   replace both rankings. */
export function groupByModelFamily(items = []) {
  const groups = []
  const byFamily = new Map()
  for (const item of items) {
    const family = modelFamily(item)
    let group = byFamily.get(family)
    if (!group) {
      group = { family, items: [] }
      byFamily.set(family, group)
      groups.push(group)
    }
    group.items.push(item)
  }
  return groups
}

/* Keep the text users can see in a family disclosure searchable even when a
   local file has been renamed. Local scan rows usually have no display `name`,
   but their architecture can still place them under a visible family label. */
export function modelSearchText(item = {}) {
  return `${searchableIdentity(item)} ${item.architecture || ''} ${modelFamily(item)}`
    .trim()
    .toLowerCase()
}

/* Search results must not be hidden behind a closed disclosure. Outside search,
   open the family containing any resident model, including an embedding
   sidecar that is loaded without being the active chat model. */
export function shouldOpenModelFamily(
  group,
  { filtering = false, activeFilename = '', loadedModelIds = new Set() } = {},
) {
  if (filtering) return true
  return (group?.items || []).some(
    (model) => model.filename === activeFilename || loadedModelIds.has(model.filename),
  )
}
