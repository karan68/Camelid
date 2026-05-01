export function formatCapabilityStatus(value) {
  return (value || '').toString().replace(/_/g, ' ')
}

const GGUF_FILE_TYPE_QUANT_LABELS = {
  0: 'F32',
  1: 'F16',
  2: 'Q4_0',
  3: 'Q4_1',
  7: 'Q8_0',
  8: 'Q5_0',
  9: 'Q5_1',
  10: 'Q2_K',
  11: 'Q3_K_S',
  12: 'Q3_K_M',
  13: 'Q3_K_L',
  14: 'Q4_K_S',
  15: 'Q4_K_M',
  16: 'Q5_K_S',
  17: 'Q5_K_M',
  18: 'Q6_K',
  19: 'IQ2_XXS',
  20: 'IQ2_XS',
  21: 'Q2_K_S',
  22: 'IQ3_XS',
  23: 'IQ3_XXS',
  24: 'IQ1_S',
  25: 'IQ4_NL',
  26: 'IQ3_S',
  27: 'IQ3_M',
  28: 'IQ2_S',
  29: 'IQ2_M',
  30: 'IQ4_XS',
  31: 'IQ1_M',
  32: 'BF16',
  36: 'TQ1_0',
  37: 'TQ2_0',
  38: 'MXFP4_MOE',
  39: 'NVFP4',
  40: 'Q1_0',
}

export function quantLabelFromGgufFileType(fileType) {
  const value = Number(fileType)
  if (!Number.isInteger(value)) return null
  return GGUF_FILE_TYPE_QUANT_LABELS[value] || null
}

function normalizeCapabilityKey(value) {
  return (value || '').toString().trim().toUpperCase().replace(/[^A-Z0-9]+/g, '')
}

function splitCapabilityKeys(value) {
  return (value || '').toString().split('/').map(normalizeCapabilityKey).filter(Boolean)
}

function extractQuantKey(model, catalogItem, subject) {
  const explicitLabel = model?.quant || catalogItem?.quant
  const explicitFileType = explicitLabel?.toString().match(/\bfile[_\s-]*type\s*(\d+)\b/i)?.[1]
  const explicit = normalizeCapabilityKey(explicitFileType ? quantLabelFromGgufFileType(explicitFileType) : explicitLabel)
  if (explicit) return explicit

  const text = subject || ''
  const match = text.match(/\b(q\d(?:_k_[ms]|_\d)|bf16|f16|f32)\b/i)
  return normalizeCapabilityKey(match?.[1])
}

function targetMatchesQuant(target, quantKey) {
  if (!quantKey) return true
  return splitCapabilityKeys(target?.quantization).includes(quantKey)
}

function findCompatibilityRowForQuant(rows, family, quantKey) {
  if (!quantKey) return null
  return rows.find((row) => row.family === family && targetMatchesQuant(row, quantKey)) || null
}

function detectLlamaBpeTarget(subject) {
  if (!/llama[\s._-]*3|meta[\s._-]*llama[\s._-]*3/.test(subject)) return null
  const sizeMatch = subject.match(/(?:^|[^a-z0-9])([138])\s*b(?:[^a-z0-9]|$)/i)
  return {
    family: 'llama_bpe_decoder',
    sizeKey: sizeMatch ? `${sizeMatch[1]}B` : null,
  }
}

function rowMatchesModelSize(row, sizeKey) {
  if (!sizeKey) return true
  return normalizeCapabilityKey(row?.id).includes(normalizeCapabilityKey(sizeKey))
}

function findPlannedFamilyForLlamaBpe(plannedFamilies) {
  return plannedFamilies.find((item) => item.id.includes('larger_llama') || item.id.includes('llama3') || item.id.includes('llama')) || null
}

function findLlamaBpeCompatibilityHint(rows, plannedFamilies, quantKey, identity) {
  const familyRows = rows.filter((row) => row.family === identity.family)
  const exactSizeTarget = familyRows.find((row) => rowMatchesModelSize(row, identity.sizeKey)) || null
  if (exactSizeTarget && targetMatchesQuant(exactSizeTarget, quantKey)) {
    return { kind: 'compatibility', target: exactSizeTarget, confidence: quantKey ? 'exact model-size + quant match' : 'exact model-size match' }
  }
  if (exactSizeTarget) {
    return { kind: 'quant_mismatch', target: exactSizeTarget, observedQuant: quantKey, confidence: 'exact model-size match with different quant' }
  }

  if (!identity.sizeKey) {
    const genericTarget = findCompatibilityRowForQuant(rows, identity.family, quantKey)
    if (genericTarget) return { kind: 'compatibility', target: genericTarget, confidence: 'family + quant match' }
  }

  const plannedFamily = findPlannedFamilyForLlamaBpe(plannedFamilies)
  if (plannedFamily) return { kind: 'family', target: plannedFamily, confidence: identity.sizeKey ? `Llama BPE ${identity.sizeKey} family match without exact row` : 'Llama BPE family match' }
  return null
}

export function isSupportedCapabilityStatus(status = '') {
  const value = status.toLowerCase()
  return value === 'supported' || value.startsWith('supported_') || value === 'validated' || value === 'measured'
}

export function isGuardedCapabilityStatus(status = '') {
  return !isSupportedCapabilityStatus(status)
}

export function capabilityStatusTone(status = '') {
  const value = status.toLowerCase()
  if (isSupportedCapabilityStatus(value)) return 'ready'
  if (
    value.includes('planned')
    || value.includes('partial')
    || value.includes('guarded')
    || value.includes('blocked')
    || value.includes('unsupported')
    || value.includes('future')
    || value.includes('not_started')
  ) return 'warm'
  return ''
}

export function summarizeCapabilityItems(items = [], fallback = 'Not advertised by this backend yet.') {
  if (!items.length) return fallback
  return items.map((item) => `${item.id}: ${formatCapabilityStatus(item.status)}`).join(' · ')
}

export function guardedCapabilityCopy(item, subject = 'UI controls') {
  const notes = item?.notes ? `${item.notes}. ` : ''
  return `${notes}${subject} should stay disabled, labeled planned/unsupported, or require an explicit verification path until /api/capabilities reports this row as supported; callers should expect typed backend refusals and surface them directly, not silently drop parameters, downgrade behavior, or infer broad compatibility.`
}

export function getCurrentCompatibilityTarget(capabilities) {
  const targets = capabilities?.model_compatibility || []
  return targets.find((target) => target.status === 'supported_current_gate') || targets[0] || null
}

function getModelCapabilitySubject(model, catalogItem) {
  return [
    model?.id,
    model?.name,
    model?.runtime_model_name,
    model?.hf_repo,
    model?.hf_filename,
    model?.model_path,
    catalogItem?.name,
    catalogItem?.repo_id,
    catalogItem?.filename,
  ].filter(Boolean).join(' ').toLowerCase()
}

export function findCompatibilityHint(capabilities, model, catalogItem) {
  const subject = getModelCapabilitySubject(model, catalogItem)
  if (!subject) return null
  const rows = capabilities?.model_compatibility || []
  const plannedFamilies = capabilities?.planned_model_families || []
  const quantKey = extractQuantKey(model, catalogItem, subject)

  const findRow = (predicate) => rows.find(predicate) || null
  const findFamily = (predicate) => plannedFamilies.find(predicate) || null

  if (subject.includes('tinyllama')) {
    const target = findRow((row) => row.id.includes('tinyllama'))
    if (target && targetMatchesQuant(target, quantKey)) return { kind: 'compatibility', target, confidence: quantKey ? 'name/path + quant match' : 'name/path match' }
    const quantSpecificTarget = findCompatibilityRowForQuant(rows, 'llama_spm_decoder', quantKey)
    if (quantSpecificTarget) return { kind: 'compatibility', target: quantSpecificTarget, confidence: 'family + quant match' }
    if (target) return { kind: 'quant_mismatch', target, observedQuant: model?.quant || catalogItem?.quant || quantKey, confidence: 'name/path match with different quant' }
  }

  const llamaBpeIdentity = detectLlamaBpeTarget(subject)
  if (llamaBpeIdentity) {
    const hint = findLlamaBpeCompatibilityHint(rows, plannedFamilies, quantKey, llamaBpeIdentity)
    if (hint) return hint.kind === 'quant_mismatch' ? { ...hint, observedQuant: model?.quant || catalogItem?.quant || quantKey } : hint
  }

  if (subject.includes('mistral')) {
    const target = findRow((row) => row.family === 'mistral' || row.id.includes('mistral'))
    if (target) return { kind: 'compatibility', target, confidence: 'name/path match' }
    const family = findFamily((item) => item.id.includes('mistral'))
    if (family) return { kind: 'family', target: family, confidence: 'family name match' }
  }

  const futureFamily = findFamily((item) => item.id.includes('qwen_gemma_phi_falcon_mamba') && /(qwen|gemma|phi|falcon|mamba)/.test(subject))
  if (futureFamily) return { kind: 'family', target: futureFamily, confidence: 'future family name match' }

  return null
}

export function compatibilityHintLabel(hint, fallback = 'No matching compatibility row') {
  if (!hint) return fallback
  if (hint.kind === 'quant_mismatch') return `${hint.target.id}: quant mismatch`
  return `${hint.target.id}: ${formatCapabilityStatus(hint.target.status)}`
}

export function compatibilityHintCopy(hint) {
  if (!hint) return 'No exact COMPATIBILITY.md row matched this model name/path, so the UI will not infer model-family support; load results and typed backend errors remain the source of truth.'
  if (hint.kind === 'family') return `${hint.target.notes}. This is only a ${hint.confidence}; it is not chat-ready support until a concrete compatibility row is validated.`
  if (hint.kind === 'quant_mismatch') return `${hint.target.id} is scoped to ${hint.target.quantization}, but this entry appears to be ${hint.observedQuant || 'a different quantization'}. Do not inherit the supported gate from a same-family row; wait for an exact COMPATIBILITY.md row plus generation_ready=true.`
  return `${hint.target.family} · ${hint.target.quantization} · ${hint.target.evidence || hint.target.next_step}. Match source: ${hint.confidence}; runtime generation still requires loaded_now=true and generation_ready=true.`
}

export function isCompatibilitySupportedForModel(capabilities, model, catalogItem) {
  const hint = findCompatibilityHint(capabilities, model, catalogItem)
  return Boolean(
    hint?.kind === 'compatibility'
    && isSupportedCapabilityStatus(hint.target?.status)
    && hint.confidence !== 'name/path match',
  )
}
