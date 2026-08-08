/* Streaming status indicators — extracted verbatim from ChatWorkspace.
   The class names (streaming-loader-track, streaming-loader-dot-N,
   message-live-generation-badge) are asserted by the CI smokes. */

export const PREPARING_STREAMING_LABEL = 'Preparing local response'
export const FIRST_TOKEN_STREAMING_LABEL = 'Generating response'
export const LONG_FIRST_TOKEN_STREAMING_LABEL = 'Local response is taking a while'
export const ACTIVE_STREAMING_LABEL = 'Streaming response'
export const OPEN_CODE_STREAMING_LABEL = 'Streaming code response'

export const streamingStatusLabel = (phase, elapsedSeconds, isOpenCode = false) => {
  if (phase === 'preparing') return PREPARING_STREAMING_LABEL
  if (phase === 'streaming') return isOpenCode ? OPEN_CODE_STREAMING_LABEL : ACTIVE_STREAMING_LABEL
  if (elapsedSeconds >= 20) return LONG_FIRST_TOKEN_STREAMING_LABEL
  return FIRST_TOKEN_STREAMING_LABEL
}

/* The aria-label carries only the phase label: the live region should announce
   phase changes (preparing → generating → streaming), not re-announce a ticking
   seconds counter every second. */
export function StreamingLoader({ label = ACTIVE_STREAMING_LABEL, compact = false }) {
  return (
    <div className={`streaming-loader ${compact ? 'streaming-loader-compact' : ''}`} role="status" aria-live="polite" aria-label={label}>
      <div className="streaming-loader-track" aria-hidden="true">
        <span className="streaming-loader-dot streaming-loader-dot-1" />
        <span className="streaming-loader-dot streaming-loader-dot-2" />
        <span className="streaming-loader-dot streaming-loader-dot-3" />
      </div>
    </div>
  )
}

const formatLiveRate = (value) => {
  const rate = Number(value)
  if (!Number.isFinite(rate) || rate <= 0) return null
  return rate >= 10 ? String(Math.round(rate)) : rate.toFixed(1)
}

/* Per-second values (elapsed time, live tok/s) are aria-hidden so the polite
   live region announces only phase-label changes, never the ticking counters. */
export function LiveGenerationBadge({ elapsedSeconds, label = ACTIVE_STREAMING_LABEL, tokensPerSec = null }) {
  const liveRate = formatLiveRate(tokensPerSec)
  return (
    <div className="message-live-generation-badge" role="status" aria-live="polite" data-live-status="active">
      <span className="message-live-dot" aria-hidden="true" />
      <span>{label}</span>
      <span aria-hidden="true">{elapsedSeconds}s</span>
      {liveRate && (
        <span className="message-live-tps" aria-hidden="true">
          <span className="message-live-tps__value">{liveRate}</span>
          <span className="message-live-tps__unit">tok/s</span>
        </span>
      )}
    </div>
  )
}
