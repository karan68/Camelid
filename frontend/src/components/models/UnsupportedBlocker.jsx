import { EvidenceChip } from '../ui/EvidenceChip'

/* Fail-closed blocker surface for a GGUF whose architecture Camelid does not
   implement (or whose metadata is invalid). Shows the EXACT typed reason verbatim
   and states that chat is disabled — never a "try anyway" that would route to a
   different inference path. When the backend's reason names a dedicated lane
   (e.g. DiffusionGemma's `camelid diffusion-gemma-chat`), that command is pulled
   out as a copyable redirect. `blocker` is `{ code, message }`. */

const DEDICATED_LANE_COMMAND = /(camelid\s+diffusion-gemma-chat[^\n.]*)/i

export function blockerNoteFor(blocker) {
  if (blocker?.code === 'model_io_error') {
    return 'Camelid could not open this model from the configured storage location, so chat stays disabled until the path is corrected.'
  }
  if (blocker?.code === 'model_too_large_for_host') {
    return 'This model does not fit the current host safety budget, so chat stays disabled instead of risking an unstable load.'
  }
  if (blocker?.code === 'unsupported_model_architecture') {
    return 'This architecture is not implemented, so chat stays disabled — Camelid fails closed rather than emit plausible-but-wrong tokens on a different code path.'
  }
  return 'This model did not pass Camelid’s runtime admission checks, so chat stays disabled rather than using an unverified fallback path.'
}

export function UnsupportedBlocker({ blocker, className = '' }) {
  if (!blocker?.message) return null
  const redirect = blocker.message.match(DEDICATED_LANE_COMMAND)?.[1]?.trim() || null

  return (
    /* views.css colors this surface's edge with --color-danger, a token
       tokens.css never defines (its fallback hex was rendering instead). Bind it
       to the real error token here so the edge tracks the theme. */
    <div
      className={`unsupported-blocker ${className}`.trim()}
      role="alert"
      style={{ '--color-danger': 'var(--color-error)' }}
    >
      <div className="unsupported-blocker__head">
        <EvidenceChip state="unsupported" asText size="sm">Fail-closed</EvidenceChip>
        {blocker.code ? <code className="unsupported-blocker__code">{blocker.code}</code> : null}
      </div>
      <p className="unsupported-blocker__message">{blocker.message}</p>
      <p className="unsupported-blocker__note">{blockerNoteFor(blocker)}</p>
      {redirect ? (
        <p className="unsupported-blocker__redirect">
          Dedicated lane: <code>{redirect}</code>
        </p>
      ) : null}
    </div>
  )
}

export default UnsupportedBlocker
