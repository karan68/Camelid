import { frontendSupportContractCopy, isSupportedCapabilityStatus } from '../../lib/capabilities'
import { CanonicalStatement } from '../ui/CanonicalStatement'
import { IconChevronRight } from '../ui/icons'

/* SupportContractSummary — the single source for the support-contract summary
   card shown on System and API. Row-level evidence (per-row quant, family,
   context packs) lives only in the Compatibility ledger; this card carries the
   counts and policies plus a deep link, so the copy cannot drift per view. */
export function SupportContractSummary({ capabilities }) {
  const supportContract = capabilities?.support_contract
  const targets = capabilities?.model_compatibility || []
  const supportedCount = targets.filter((target) => isSupportedCapabilityStatus(target.status)).length
  const currentGate = frontendSupportContractCopy(capabilities)

  const openLedger = () => {
    window.dispatchEvent(new CustomEvent('camelid:open-ledger', { detail: {} }))
  }

  return (
    <div className="cxv-card cxv-card--flat sys-evidence">
      <strong>Current gate</strong>
      {supportContract ? (
        <>
          <div className="sys-contract-overview">
            <span><b>{supportedCount}</b> verified models</span>
            <span><b>{targets.length - supportedCount}</b> guarded or unclaimed rows</span>
          </div>
          <details className="sys-evidence-details sys-evidence-details--canonical">
            <summary>Read the complete current-gate statement</summary>
            <CanonicalStatement text={currentGate} />
          </details>
          <dl className="sys-policy-list">
            <div><dt className="cx-caption">Support policy</dt><dd>{supportContract.support_policy}</dd></div>
            <div><dt className="cx-caption">Unsupported policy</dt><dd>{supportContract.unsupported_policy}</dd></div>
          </dl>
        </>
      ) : (
        <p>/api/capabilities is unavailable, so the UI falls back to runtime health only and does not infer broader support.</p>
      )}
      <button type="button" className="ev-pop__ledger-link" onClick={openLedger}>
        View full row-level evidence in Compatibility <IconChevronRight size={12} />
      </button>
    </div>
  )
}

export default SupportContractSummary
