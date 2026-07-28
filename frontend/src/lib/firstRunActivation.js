/* First-run activation: deciding whether this is a fresh install, and which single
   model to offer.

   Both answers are DERIVED from live server state on every render — never from a
   localStorage "has onboarded" flag. A stored flag is wrong in both directions (it
   survives a wiped models folder, and a new browser profile resurrects the card for
   an established install), and this is precisely the surface where a stale profile
   has hidden a first-load bug before.

   The recommendation is derived too: it is whatever the catalog and the support
   contract say right now, so promoting or demoting a row cannot leave a hard-coded
   model id behind in the onboarding path. */

import { isCompatibilitySupportedForModel } from './capabilities.js'
import { isRefusingFit } from './catalogBrowse.js'
import { hasLocalModelPath } from './modelState.js'

/* A fresh install: the engine is answering, it is holding no model, and there is no
   GGUF on disk to hold.

   All three matter. Offline is the backend banner's job, not onboarding's. A loaded
   model means the user is already through the funnel. And a machine that HAS models
   but has none loaded is not a fresh install — it is someone who unloaded, or whose
   only local model failed to load; sending them at a download would be wrong, and
   the Models page already owns that case.

   `models` is the dashboard's merged model list, which is reconciled against the
   live `/api/models/local` disk scan, so a record whose file is gone cannot keep the
   card hidden. */
export function isFirstRunHost({ runtime, models = [] } = {}) {
  if (runtime?.status !== 'online') return false
  if (runtime?.loaded_now || runtime?.active_model_id) return false
  return !models.some((model) => hasLocalModelPath(model))
}

/* Ascending by download size, with the catalog id as a deterministic tie-break so
   two equally sized rows cannot reorder between renders. */
function bySmallestDownload(a, b) {
  if (a.size_bytes !== b.size_bytes) return a.size_bytes - b.size_bytes
  return String(a.catalog_id).localeCompare(String(b.catalog_id))
}

/* The one model a fresh install is offered.

   Rules, in order, and each is a refusal rather than a fallback:

   1. Only `supported_*` contract rows. The first thing a new user runs must be a row
      Camelid has cross-validated — never an experimental or merely-runnable one. The
      test suite pins this, because "just take the smallest curated row" would have
      quietly started offering an unverified model the day one was added.
   2. Only rows this host can actually load. A row the load-time fit guard would
      refuse with a 422 must never be offered; that turns one click into a dead end.
   3. Smallest first. The cost of the first token is the download, so the shortest
      honest path wins.

   Returns a tagged result rather than `null`, because the three empty cases need
   three different sentences:
     - `recommended`     — `item` is the offer.
     - `no_fitting_row`  — supported rows exist, none fits here. `smallest` is the
                           closest one, so the UI can say what it would have offered.
     - `no_supported_row`— the catalog carries no supported row at all (an unreachable
                           backend, or a contract that advertises none). */
export function recommendFirstRunModel(items = [], capabilities = null) {
  const supported = (items || []).filter(
    (item) => item?.group === 'curated' && isCompatibilitySupportedForModel(capabilities, null, item),
  )
  if (!supported.length) return { kind: 'no_supported_row', item: null, smallest: null }

  const ordered = [...supported].sort(bySmallestDownload)
  const fitting = ordered.filter((item) => !isRefusingFit(item.fit))
  if (!fitting.length) {
    return { kind: 'no_fitting_row', item: null, smallest: ordered[0] }
  }
  return { kind: 'recommended', item: fitting[0], smallest: ordered[0] }
}

/* Whether a failed activation is worth a retry button.

   A host that is too small stays too small, so offering "Try again" there is a
   button that cannot work. Memory pressure and every untyped/transport failure are
   retryable. */
const PERMANENT_REFUSAL_CODES = new Set(['model_too_large_for_host', 'unsupported_model_architecture'])

export function firstRunFailureIsRetryable(code = '') {
  return !PERMANENT_REFUSAL_CODES.has(String(code || ''))
}
