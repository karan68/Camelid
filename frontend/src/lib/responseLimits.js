/* Response-length limits (Phase 9). Pure helpers; every number traces to a
   real source: model context from /v1/models meta.n_ctx_train (descriptive
   metadata, not a support claim — I2), the verified bound from validated
   bounded-context packs on the exact /api/capabilities row. Memory and
   KV-cost inputs do not exist on the backend yet (BACKEND_ASKS.md #3) — the
   UI renders those indicators ABSENT, never estimated client-side. */

import { findCompatibilityHint, isExactCompatibilityHint } from './capabilities.js'

export const MAX_RESPONSE_TOKENS = 1000000
export const MIN_RESPONSE_TOKENS = 1
export const DETENTS = [256, 1000, 4000, 16000, 64000, 256000, 1000000]

export function modelContextLength(model) {
  const value = Number(model?.meta?.n_ctx_train)
  return Number.isFinite(value) && value > 0 ? value : null
}

/* Highest bounded-context window whose pack status is validated on the exact
   matched row. Family/quant resemblance never produces a bound. */
export function verifiedContextBound(capabilities, model) {
  const hint = findCompatibilityHint(capabilities, model)
  if (!isExactCompatibilityHint(hint) || !hint.target) return null
  const row = hint.target
  let bound = null
  for (const key of Object.keys(row)) {
    const match = key.match(/^bounded_context_(\d+)_pack$/)
    if (match && String(row[key]).startsWith('validated')) {
      const window = Number(row[`bounded_context_${match[1]}_window`] ?? match[1])
      if (Number.isFinite(window)) bound = Math.max(bound ?? 0, window)
    }
  }
  return bound
}

/* Log-scale slider mapping: position 0..1 over [MIN, MAX]. */
const LOG_MIN = Math.log(MIN_RESPONSE_TOKENS)
const LOG_MAX = Math.log(MAX_RESPONSE_TOKENS)

export function tokensToSlider(value) {
  const clamped = Math.min(Math.max(value, MIN_RESPONSE_TOKENS), MAX_RESPONSE_TOKENS)
  return (Math.log(clamped) - LOG_MIN) / (LOG_MAX - LOG_MIN)
}

export function sliderToTokens(position) {
  const value = Math.round(Math.exp(LOG_MIN + (LOG_MAX - LOG_MIN) * Math.min(Math.max(position, 0), 1)))
  // light detent snap: within 2% of track distance
  for (const detent of DETENTS) {
    if (Math.abs(tokensToSlider(detent) - position) < 0.012) return detent
  }
  return value
}

/* Validation states, priority-ordered. Red = the request will fail (the
   backend returns typed context_length_exceeded, it does not clamp); amber =
   allowed but beyond the verified row's tested context; slate stays for
   support states elsewhere. */
export function validateResponseLength({ value, contextLength = null, verifiedBound = null, modelName = 'the loaded model' }) {
  if (contextLength !== null && value > contextLength) {
    return {
      level: 'error',
      code: 'over_model_context',
      message: `Exceeds ${modelName}’s ${contextLength.toLocaleString()}-token context — the backend rejects this with a typed context_length_exceeded error. Lower the limit below ${contextLength.toLocaleString()}, or load a longer-context model.`,
    }
  }
  if (verifiedBound !== null && value > verifiedBound) {
    return {
      level: 'caution',
      code: 'over_verified_bound',
      message: `Beyond the verified row’s tested ${verifiedBound.toLocaleString()}-token context — allowed, untested. Evidence covers the checked packs only.`,
    }
  }
  return { level: 'ok', code: 'ok', message: '' }
}

/* Send-time check mirrors the backend’s actual rule:
   prompt_tokens + max_tokens must fit the context (typed error otherwise). */
export function validateSendBudget({ promptTokens, maxTokens, contextLength }) {
  if (contextLength === null || !Number.isFinite(promptTokens)) return { level: 'ok' }
  if (promptTokens + maxTokens > contextLength) {
    return {
      level: 'error',
      code: 'context_length_exceeded',
      message: `Prompt (~${promptTokens.toLocaleString()} tokens, estimated) plus the ${maxTokens.toLocaleString()}-token response limit exceeds the ${contextLength.toLocaleString()}-token context. Lower the response limit in Settings or shorten the prompt.`,
    }
  }
  return { level: 'ok' }
}

const MAX_TOKENS_KEY = 'camelid.maxTokens'

export function getConfiguredMaxTokens(modelId = '') {
  if (typeof window === 'undefined') return 8192
  const perModel = modelId ? Number.parseInt(window.localStorage.getItem(`${MAX_TOKENS_KEY}.${modelId}`) || '', 10) : NaN
  if (Number.isFinite(perModel) && perModel >= MIN_RESPONSE_TOKENS) return perModel
  const legacy = Number.parseInt(window.localStorage.getItem(MAX_TOKENS_KEY) || '', 10)
  return Number.isFinite(legacy) && legacy >= 256 ? legacy : 8192
}

export function setConfiguredMaxTokens(modelId, value) {
  if (typeof window === 'undefined') return
  const clamped = Math.min(Math.max(Math.round(value), MIN_RESPONSE_TOKENS), MAX_RESPONSE_TOKENS)
  if (modelId) window.localStorage.setItem(`${MAX_TOKENS_KEY}.${modelId}`, String(clamped))
  else window.localStorage.setItem(MAX_TOKENS_KEY, String(clamped))
}
