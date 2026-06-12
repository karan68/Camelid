import { useEffect, useMemo, useRef, useState } from 'react'
import { displayCapabilityCopy, displayCapabilityId, formatCapabilityStatus, isSupportedCapabilityStatus } from '../lib/capabilities'
import { EvidenceChip } from '../components/ui/EvidenceChip'
import { EmptyState } from '../components/ui/EmptyState'
import { IconReceipt } from '../components/ui/icons'

/* Compatibility & evidence explorer (Phase 4) — the signature view.

   This is the release ledger rendered live: every row, status, evidence field,
   pack id, blocker, and policy string on this screen comes from
   /api/capabilities at render time. The view contains ZERO hardcoded support
   claims — if the contract and any copy could disagree, the contract is the
   only voice here. The restraint (the not-claimed column at equal weight) is
   the product. */

/* Evidence checklist: field → human framing. Labels describe the CATEGORY;
   every status value comes from the row itself. */
const EVIDENCE_TRACKS = [
  { field: 'metadata_parses', label: 'metadata parses' },
  { field: 'tokenizer_works', label: 'tokenizer' },
  { field: 'tensors_load', label: 'tensors load' },
  { field: 'generation_runs', label: 'generation runs' },
  { field: 'parity_audited', label: 'prompt-token parity' },
  { field: 'frontend_load_path_verified', label: 'frontend load path' },
  { field: 'chat_template_shape_pack', label: 'template-shape pack', packIdField: 'chat_template_shape_pack_id' },
  { field: 'bounded_context_512_pack', label: 'bounded 512 context', packIdField: 'bounded_context_512_pack_id' },
  { field: 'bounded_context_1024_pack', label: 'bounded 1024 context', packIdField: 'bounded_context_1024_pack_id' },
  { field: 'bounded_context_2048_pack', label: 'bounded 2048 context', packIdField: 'bounded_context_2048_pack_id' },
  { field: 'bounded_context_4096_pack', label: 'bounded 4096 context', packIdField: 'bounded_context_4096_pack_id' },
  { field: 'bounded_context_8192_pack', label: 'bounded 8192 context', packIdField: 'bounded_context_8192_pack_id' },
  { field: 'performance_measured', label: 'perf / RSS (bounded)' },
]

function evidenceTracksForRow(row) {
  return EVIDENCE_TRACKS
    .filter((track) => row?.[track.field] !== undefined && row?.[track.field] !== null)
    .map((track) => {
      const packId = track.packIdField ? row[track.packIdField] : null
      return {
        ...track,
        status: row[track.field],
        packId: packId && packId !== 'not_selected' ? packId : null,
      }
    })
}

function LedgerRow({ row, focused, registerRef }) {
  const [open, setOpen] = useState(false)
  const supported = isSupportedCapabilityStatus(row.status)
  const tracks = evidenceTracksForRow(row)

  useEffect(() => {
    if (focused) setOpen(true)
  }, [focused])

  return (
    <article
      ref={(node) => registerRef(row.id, node)}
      className={`ledger-row ${supported ? 'ledger-row--supported' : ''} ${focused ? 'is-focused' : ''}`}
      data-row-id={row.id}
    >
      <header className="ledger-row__head" onClick={() => setOpen((v) => !v)}>
        <div className="ledger-row__identity">
          <code className="ledger-row__id">{row.id}</code>
          <span className="ledger-row__family">{row.family} · {row.quantization}</span>
        </div>
        <div className="ledger-row__posture">
          <EvidenceChip
            status={row.status}
            source={{ rowId: row.id, detail: row.support_scope ? displayCapabilityCopy(row.support_scope) : undefined }}
            size="sm"
          />
          <button type="button" className="ledger-row__toggle" aria-expanded={open} onClick={(event) => { event.stopPropagation(); setOpen((v) => !v) }}>
            {open ? 'Collapse' : 'Evidence'}
          </button>
        </div>
      </header>

      <div className="ledger-row__columns">
        <div className="ledger-row__col">
          <h3 className="ledger-row__col-title">Proven</h3>
          <p className="ledger-row__copy">{displayCapabilityCopy(row.evidence) || 'No evidence copy advertised for this row.'}</p>
          {row.tested_context && <p className="ledger-row__meta">tested context: <code>{row.tested_context}</code></p>}
        </div>
        <div className="ledger-row__col ledger-row__col--not-claimed">
          <h3 className="ledger-row__col-title">Not claimed</h3>
          <p className="ledger-row__copy">{displayCapabilityCopy(row.full_support_blockers) || 'This row advertises no explicit boundary copy; nothing beyond the proven column is claimed.'}</p>
          {row.support_scope && <p className="ledger-row__meta">scope: <code>{row.support_scope}</code></p>}
        </div>
      </div>

      {open && (
        <div className="ledger-row__drill">
          <h3 className="ledger-row__col-title">Evidence checklist</h3>
          <ul className="ledger-row__tracks">
            {tracks.map((track) => (
              <li key={track.field} className="ledger-row__track">
                <span className="ledger-row__track-label">{track.label}</span>
                <EvidenceChip
                  status={track.status}
                  source={{
                    rowId: row.id,
                    detail: `${track.label} — field ${track.field}`,
                    note: track.packId ? `Evidence bundle: ${track.packId}` : 'No evidence-bundle id advertised for this lane.',
                  }}
                  size="sm"
                />
                {track.packId && <code className="ledger-row__pack">{track.packId}</code>}
              </li>
            ))}
          </ul>

          {(row.latest_checked_bucket || row.latest_checked_result) && (
            <p className="ledger-row__meta">
              latest checked: <code>{row.latest_checked_bucket || '—'}</code> → <code>{row.latest_checked_result || '—'}</code>
              {row.latest_checked_output && <> · output starts <code>{String(row.latest_checked_output).slice(0, 60)}</code></>}
            </p>
          )}
          {row.frontend_readiness_gate && (
            <p className="ledger-row__meta">readiness gate: {displayCapabilityCopy(row.frontend_readiness_gate)}</p>
          )}

          {!supported && row.next_step && (
            <div className="ledger-row__promotion">
              <h3 className="ledger-row__col-title">Promotion path</h3>
              <p className="ledger-row__copy">{displayCapabilityCopy(row.next_step)}</p>
              <p className="ledger-row__meta">An honest checklist, not a promise — this row moves only when the evidence above does.</p>
            </div>
          )}
        </div>
      )}
    </article>
  )
}

export default function CompatibilityView({ capabilities, focusRowId = null, onFocusConsumed = null }) {
  const rows = capabilities?.model_compatibility || []
  const apiFeatures = capabilities?.api_features || []
  const supportContract = capabilities?.support_contract
  const rowRefs = useRef(new Map())
  const registerRef = (id, node) => {
    if (node) rowRefs.current.set(id, node)
    else rowRefs.current.delete(id)
  }

  const supportedCount = useMemo(() => rows.filter((row) => isSupportedCapabilityStatus(row.status)).length, [rows])

  useEffect(() => {
    if (!focusRowId) return undefined
    const node = rowRefs.current.get(focusRowId)
    if (node) node.scrollIntoView({ block: 'start', behavior: 'smooth' })
    const timer = window.setTimeout(() => onFocusConsumed?.(), 2400)
    return () => window.clearTimeout(timer)
  }, [focusRowId, onFocusConsumed, rows.length])

  return (
    <section className="compatibility-view cxv">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconReceipt size={14} /> Compatibility</p>
          <h1>The evidence ledger</h1>
          <p className="cxv-sub">
            Every row below is the live /api/capabilities contract — nothing on this screen is
            written by hand. Support is exact-row: one artifact, one quant, one set of checked
            evidence. Resemblance is not evidence; a family name, a filename, or a neighboring
            size proves nothing here, and the “not claimed” column carries the same weight as
            the proven one.
          </p>
        </div>
      </header>

      {supportContract && (
        <div className="cxv-card cxv-card--flat ledger-contract">
          <strong>Support contract</strong>
          {supportContract.current_gate && <p className="ledger-contract__line"><span className="ledger-contract__key">current gate</span>{displayCapabilityCopy(supportContract.current_gate)}</p>}
          {supportContract.support_policy && <p className="ledger-contract__line"><span className="ledger-contract__key">support policy</span>{displayCapabilityCopy(supportContract.support_policy)}</p>}
          {supportContract.unsupported_policy && <p className="ledger-contract__line"><span className="ledger-contract__key">unsupported policy</span>{displayCapabilityCopy(supportContract.unsupported_policy)}</p>}
        </div>
      )}

      {rows.length === 0 ? (
        <EmptyState
          className="cx-empty--inline"
          icon={<IconReceipt size={22} />}
          title="Ledger unavailable"
          description="No compatibility rows were read from /api/capabilities. The ledger renders only the live contract — start the backend (cargo run -- serve) or fix the API base in Settings; nothing is shown from memory or assumption."
        />
      ) : (
        <>
          <div className="cxv-stat-grid">
            <div className="cxv-stat"><span>Rows</span><strong>{rows.length}</strong><small>exact lanes tracked</small></div>
            <div className="cxv-stat"><span>Supported</span><strong>{supportedCount}</strong><small>exact rows, bounded envelopes</small></div>
            <div className="cxv-stat"><span>Everything else</span><strong>{rows.length - supportedCount}</strong><small>tracked, honestly not claimed</small></div>
          </div>

          <div className="ledger-rows">
            {rows.map((row) => (
              <LedgerRow key={row.id} row={row} focused={focusRowId === row.id} registerRef={registerRef} />
            ))}
          </div>
        </>
      )}

      {apiFeatures.length > 0 && (
        <section className="cxv-card cxv-panel ledger-features">
          <div className="cxv-section__head"><h2>API feature rows</h2><span className="cxv-section__count">{apiFeatures.length} advertised</span></div>
          <p className="cxv-sub">Feature lanes from the same contract. They gate API affordances and never widen any model row above.</p>
          <ul className="ledger-features__list">
            {apiFeatures.map((feature) => (
              <li key={feature.id} className="ledger-features__item" ref={(node) => registerRef(feature.id, node)} data-row-id={feature.id}>
                <span className={`ledger-features__id ${focusRowId === feature.id ? 'is-focused' : ''}`}>{displayCapabilityId(feature.id)}</span>
                <EvidenceChip status={feature.status} source={{ rowId: feature.id, note: displayCapabilityCopy(feature.notes) }} size="sm" />
              </li>
            ))}
          </ul>
        </section>
      )}

      <footer className="ledger-explainer">
        <h3>How to read this ledger</h3>
        <p>
          A <b>supported</b> row means the exact artifact named by the row passed the checks the
          row lists — and only those. Bounded context packs cover their window, not the model’s
          native maximum; perf numbers are bounded measurements, not throughput promises. A row
          that is not supported is a normal, honest state: the promotion path says what evidence
          is still missing, and {formatCapabilityStatus('planned')} rows do not run at all.
          When any other screen in this app makes a claim, its chip cites a row here.
        </p>
      </footer>
    </section>
  )
}
