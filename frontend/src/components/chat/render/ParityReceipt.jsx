import { useState } from 'react'
import { copyText } from '../../../lib/markdown'
import { IconCheck, IconClose } from '../../ui/icons'

/* Parity receipt card — extracted verbatim from ChatWorkspace.
   Copy rule (mirrors the release ledger boundary): the card may say a match was
   verified for THIS request, and must never imply the lane itself is supported. */

const downloadJson = (filename, value) => {
  try {
    const blob = new Blob([`${JSON.stringify(value, null, 2)}\n`], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = filename
    anchor.click()
    URL.revokeObjectURL(url)
  } catch {
    // Download is best-effort outside full browser contexts.
  }
}

export function ParityReceiptCard({ receipt }) {
  const [copiedCommand, setCopiedCommand] = useState(false)
  const [copiedId, setCopiedId] = useState(false)
  if (!receipt?.receipt_id) return null
  const lane = receipt.lane || {}
  const parity = receipt.parity || {}
  // The generic runnable lane attests deterministic, HF-anchored execution — NOT a
  // supported parity contract. It must be unmistakable from a supported receipt and
  // is never copper.
  const isRunnable = receipt.execution_lane === 'runnable'
  const shortHash = String(lane.gguf_sha256 || '').slice(0, 12)
  const shortId = String(receipt.receipt_id || '').slice(0, 12)
  const downloadName = `camelid-parity-receipt-${shortId}.json`
  const verifyCommand = `camelid verify-receipt ${downloadName} --gguf <path-to-${lane.gguf_filename || 'model.gguf'}>`
  const compared = parity.compared_against_reference === true
  const allMatch = compared
    && parity.prompt_tokens_match === true
    && parity.generated_tokens_match === true
    && parity.generated_text_match === true
  const matchMark = (value) => {
    if (value === true) {
      return <span className="parity-receipt-mark is-match"><IconCheck size={12} /><span className="sr-only">match</span></span>
    }
    if (value === false) {
      return <span className="parity-receipt-mark is-miss"><IconClose size={12} /><span className="sr-only">mismatch</span></span>
    }
    return <span aria-hidden="true">—</span>
  }
  const statusLabel = !receipt.reproducible
    ? 'Not verifiable — this reply used random sampling'
    : compared
      ? (allMatch ? 'Matches the reference output for this response' : 'Differs from the reference output for this response')
      : 'Not yet verified — run the command below to check'
  const statusTone = !receipt.reproducible ? 'sampled' : compared ? (allMatch ? 'match' : 'diverged') : 'claim'

  /* Only confirm "Copied" when the text actually reached the clipboard. */
  const handleCopyCommand = async () => {
    if (!(await copyText(verifyCommand))) return
    setCopiedCommand(true)
    window.setTimeout(() => setCopiedCommand(false), 1600)
  }
  const handleCopyId = async () => {
    if (!(await copyText(receipt.receipt_id))) return
    setCopiedId(true)
    window.setTimeout(() => setCopiedId(false), 1600)
  }

  return (
    <div
      className={`parity-receipt-card${isRunnable ? ' is-runnable' : ''}`}
      aria-label="Verification receipt for this response"
    >
      <div className="parity-receipt-header">
        <span className="parity-receipt-title">Verification receipt</span>
        {isRunnable && (
          <span className="parity-receipt-lane-badge" title="This model isn't officially supported. The receipt still applies to this single response.">
            Not officially supported
          </span>
        )}
        <span className={`parity-receipt-badge is-${receipt.reproducible ? 'reproducible' : 'sampled'}`}>
          {receipt.reproducible ? 'Reproducible' : 'Not reproducible'}
        </span>
      </div>
      <div className="parity-receipt-lane">
        {lane.model_id || 'Unknown model'} · {lane.quantization || '?'} · file {shortHash || '?'}
      </div>
      <div className={`parity-receipt-status is-${statusTone}`}>{statusLabel}</div>
      {compared && (
        <ul className="parity-receipt-matches">
          <li>prompt tokens {matchMark(parity.prompt_tokens_match)}</li>
          <li>generated tokens {matchMark(parity.generated_tokens_match)}</li>
          <li>generated text {matchMark(parity.generated_text_match)}</li>
          <li>first difference at token: {parity.first_divergent_token_index ?? '—'}</li>
        </ul>
      )}
      <div className="parity-receipt-id" title={receipt.receipt_id}>
        <span>Receipt {shortId}…</span>
        <button type="button" className="message-action-button" onClick={handleCopyId}>
          {copiedId ? 'Copied' : 'Copy ID'}
        </button>
      </div>
      <div className="parity-receipt-actions">
        <button type="button" className="message-action-button" onClick={() => downloadJson(downloadName, receipt)}>
          Download receipt
        </button>
        <button type="button" className="message-action-button" onClick={handleCopyCommand}>
          {copiedCommand ? 'Copied' : 'Copy verify command'}
        </button>
      </div>
      <p className="parity-receipt-note">
        {isRunnable
          ? 'This receipt verifies this single response against the reference model. It does not mean this model is officially supported.'
          : 'This receipt covers this single response from this exact model file. It is not a statement about overall model support.'}
      </p>
    </div>
  )
}
