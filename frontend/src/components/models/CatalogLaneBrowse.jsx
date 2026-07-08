import { useCallback, useEffect, useRef, useState } from 'react'
import { isCompatibilitySupportedForModel } from '../../lib/capabilities'
import { SUPPORTED_MODELS } from '../../lib/supportedModels'
import { EvidenceChip } from '../ui/EvidenceChip'

/* Zone 5 — Get models. Curated picks first, then live Hugging Face GGUF search
   (>= 2 chars). Each row shows which lane it WOULD land in (derived: supported
   contract match, oracle-qualified runnable, or not-yet-anchored). Download is
   user-initiated and explicitly confirmed (filename + HF repo + size); no
   background/auto pulls. Live progress renders in the global Downloads zone —
   rows here only reflect their own acquisition state, read from the shared
   downloads poll + the live /api/models/local scan (never localStorage). After a
   download lands, smoke-admission runs for oracle-qualified combos and the model
   appears in its derived local section. */

const GB = 1024 * 1024 * 1024
function prettySize(bytes) {
  if (!bytes) return ''
  if (bytes >= GB) return `${(bytes / GB).toFixed(bytes >= 10 * GB ? 0 : 1)} GB`
  return `${Math.round(bytes / (1024 * 1024))} MB`
}

/* Curated download suggestions (blurbs, "Recommended") may DECORATE catalog rows,
   never place them: lane membership and outcome chips stay derived. */
const CURATED_DECORATION = new Map(SUPPORTED_MODELS.map((item) => [item.catalog_id, item]))

/* Predicted lane for a catalog entry — derived, never a hand-authored label. */
function predictedLane(item, capabilities) {
  // Experimental (live Hugging Face) rows are advisory only: their architecture/quant
  // are filename guesses, so they can never anchor a lane or imply support — even when
  // the filename happens to coincide with a supported contract row. Always not-anchored.
  if (item.group === 'experimental') return 'not_anchored'
  if (isCompatibilitySupportedForModel(capabilities, null, item)) return 'supported'
  if (item.oracle_qualified) return 'compatible'
  return 'not_anchored'
}

function laneChip(lane) {
  if (lane === 'supported') return <EvidenceChip status="supported" asText>Lands in Supported</EvidenceChip>
  if (lane === 'compatible') return <EvidenceChip state="runnable" asText>Experimental · runnable</EvidenceChip>
  return <EvidenceChip state="unsupported" asText>Experimental · unverified</EvidenceChip>
}

/* Capacity advisory for THIS host (fit axis, NOT a support claim — kept on its own
   line, never merged into the lane/support chip). `item.fit` is the backend
   FitVerdict; `unknown`/missing (e.g. unprobed host, experimental rows) shows
   nothing rather than guessing. */
function fitLabel(fit) {
  switch (fit) {
    case 'fits_resident':
      return 'Fits your machine'
    case 'fits_with_offload':
      return 'Fits (GPU + RAM offload)'
    case 'cpu_only_ok':
      return 'Fits (CPU)'
    case 'wont_fit':
      return 'Too big for this machine'
    default:
      return null
  }
}

/* A small CPU/chip glyph so the capacity chip reads as "your hardware" — distinct
   from the support/lane chips. A check (fits) or cross (too big) sits in the die. */
function FitIcon({ bad }) {
  const stroke = 'currentColor'
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <rect x="4.5" y="4.5" width="7" height="7" rx="1.2" stroke={stroke} strokeWidth="1.3" />
      {bad ? (
        <path d="M6.4 6.4l3.2 3.2M9.6 6.4l-3.2 3.2" stroke={stroke} strokeWidth="1.3" strokeLinecap="round" />
      ) : (
        <path d="M6 8.2l1.4 1.4L10 6.6" stroke={stroke} strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
      )}
      <path
        d="M6.5 2.6v1.9M9.5 2.6v1.9M6.5 11.5v1.9M9.5 11.5v1.9M2.6 6.5h1.9M2.6 9.5h1.9M11.5 6.5h1.9M11.5 9.5h1.9"
        stroke={stroke}
        strokeWidth="1.1"
        strokeLinecap="round"
      />
    </svg>
  )
}

function CatalogRow({
  item,
  capabilities,
  installed,
  downloading,
  apiBase,
  installAvailable,
  installBlockedReason,
  onInstallStarted,
  onAcquired,
}) {
  // phase: idle | confirm | starting | waiting | smoking | done
  const [phase, setPhase] = useState('idle')
  const [message, setMessage] = useState('')
  const [isError, setIsError] = useState(false)
  const sawDownloadRef = useRef(false)
  const startedAtRef = useRef(0)
  const lane = predictedLane(item, capabilities)
  const decoration = item.group === 'experimental' ? null : CURATED_DECORATION.get(item.catalog_id)

  const finishLanded = useCallback(async () => {
    // After download: smoke-admission only applies to oracle-qualified combos. For
    // everything else the file just lands on disk — a machine with the right
    // support lane can still run it; we don't gate the download on local hardware.
    if (item.oracle_qualified) {
      setPhase('smoking')
      try {
        const smoke = await fetch(`${apiBase}/api/models/runnable-smoke`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: item.filename }),
        })
        const body = await smoke.json().catch(() => ({}))
        setMessage(
          smoke.ok && body.passed
            ? 'Downloaded and smoke-admitted — see it above in its section.'
            : body?.error?.message
              ? `Downloaded. Smoke-admission did not pass here: ${body.error.message}`
              : 'Downloaded. Smoke-admission did not pass on this machine — the file is on disk.',
        )
      } catch (err) {
        setMessage(`Downloaded. Smoke-admission could not run: ${String(err?.message || err)}`)
      }
    } else {
      setMessage('Downloaded — see it above in its section.')
    }
    setPhase('done')
    setIsError(false)
    onAcquired?.()
  }, [apiBase, item.filename, item.oracle_qualified, onAcquired])

  // The row watches the SHARED downloads poll + local scan instead of polling
  // itself: downloading -> (gone + on disk) = landed; (gone + not on disk after
  // having been seen) = failed or canceled.
  useEffect(() => {
    if (phase !== 'waiting') return
    if (downloading) {
      sawDownloadRef.current = true
      return
    }
    if (installed) {
      finishLanded()
      return
    }
    const waitedMs = Date.now() - startedAtRef.current
    if (sawDownloadRef.current || waitedMs > 20000) {
      setPhase('idle')
      setIsError(true)
      setMessage('Download did not complete (canceled or failed). It can be retried.')
    }
  }, [phase, downloading, installed, finishLanded])

  const confirmDownload = async () => {
    setPhase('starting')
    setMessage('')
    setIsError(false)
    sawDownloadRef.current = false
    startedAtRef.current = Date.now()
    try {
      const res = await fetch(`${apiBase}/api/models/catalog/install`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          catalog_id: item.catalog_id,
          repo_id: item.repo_id,
          filename: item.filename,
          size_bytes: item.size_bytes,
        }),
      })
      if (!res.ok && res.status !== 409) {
        const text = await res.text()
        throw new Error(text || `download failed (HTTP ${res.status})`)
      }
      setPhase('waiting')
      onInstallStarted?.()
    } catch (err) {
      setPhase('idle')
      setIsError(true)
      setMessage(String(err?.message || err))
    }
  }

  return (
    <article className={`catalog-row${lane === 'not_anchored' ? ' catalog-row--advisory' : ''}`}>
      <div className="catalog-row-head">
        <div className="catalog-row-id">
          <span className="catalog-row-name">
            {item.name}
            {decoration?.recommended ? <span className="catalog-row-recommended">Recommended</span> : null}
          </span>
          <span className="catalog-row-meta">
            {item.repo_id} · {item.filename} · {prettySize(item.size_bytes)}
            {item.architecture ? ` · ${item.architecture}` : ''}
          </span>
        </div>
        {laneChip(lane)}
      </div>
      {item.group !== 'experimental' && fitLabel(item.fit) ? (
        <div className="catalog-fit-row">
          <span
            className={`catalog-fit-chip catalog-fit-chip--${item.fit === 'wont_fit' ? 'bad' : 'good'}`}
            title={
              item.fit_confidence === 'exact'
                ? "Sized from the model's real dimensions (KV cache computed exactly)"
                : 'Estimate — upgrades to exact once the model header has been read'
            }
          >
            <FitIcon bad={item.fit === 'wont_fit'} />
            {item.fit_confidence === 'approx' ? '~ ' : ''}
            {fitLabel(item.fit)}
          </span>
          {Array.isArray(item.task_tags) && item.task_tags.length ? (
            <span className="catalog-fit-tags">
              <span className="catalog-fit-tags-label">best for</span>
              {item.task_tags.map((tag) => (
                <span key={tag} className="catalog-fit-tag">
                  {tag}
                </span>
              ))}
            </span>
          ) : null}
        </div>
      ) : null}
      {decoration?.blurb ? <p className="catalog-row-blurb">{decoration.blurb}</p> : null}

      {installed ? (
        <p className="catalog-row-faint">Already on disk — shown in its section above.</p>
      ) : phase === 'idle' ? (
        <>
          {item.group === 'experimental' ? (
            <p className="catalog-row-faint">
              From Hugging Face — unverified, no parity claim. Architecture/quant
              {item.architecture || item.quant ? ` (guessed ${[item.architecture, item.quant].filter(Boolean).join(' / ')})` : ''}{' '}
              are read from the filename, not the model; the real lane is only known after it loads.
            </p>
          ) : lane === 'not_anchored' ? (
            <p className="catalog-row-faint">
              Its {item.architecture}/{item.quant} combo is not yet in the runnable lane — still
              downloadable; it lands in Experimental and loads through the experimental chat path.
            </p>
          ) : null}
          {message ? <p className={isError ? 'catalog-row-error' : 'catalog-row-faint'}>{message}</p> : null}
          {installAvailable ? (
            <button type="button" className="catalog-row-action" onClick={() => setPhase('confirm')}>
              Download…
            </button>
          ) : (
            <>
              <button type="button" className="catalog-row-action" disabled>
                Download unavailable
              </button>
              <p className="catalog-row-faint">{installBlockedReason}</p>
            </>
          )}
        </>
      ) : phase === 'confirm' ? (
        <div className="catalog-confirm">
          <p>
            Download <strong>{item.filename}</strong> from <code>{item.repo_id}</code> (
            {prettySize(item.size_bytes)})? This pulls from HuggingFace into your local models folder.
          </p>
          <div className="catalog-confirm-actions">
            <button type="button" className="catalog-row-action" onClick={confirmDownload}>
              Confirm download
            </button>
            <button type="button" className="catalog-row-cancel" onClick={() => setPhase('idle')}>
              Cancel
            </button>
          </div>
        </div>
      ) : phase === 'starting' || phase === 'waiting' ? (
        <p className="catalog-row-faint">
          {downloading ? 'Downloading — live progress in Downloads above.' : 'Starting download…'}
        </p>
      ) : phase === 'smoking' ? (
        <p className="catalog-row-faint">Download complete — running smoke-admission…</p>
      ) : (
        <p className={isError ? 'catalog-row-error' : 'catalog-row-faint'}>{message}</p>
      )}
    </article>
  )
}

/* Persistent, non-dismissible marker for the experimental group. Reuses the
   unsupported EvidenceChip so it can never read as an endorsement. */
function ExperimentalMarker() {
  return (
    <span className="catalog-experimental-marker">
      <EvidenceChip state="unsupported" asText>Experimental — unverified, no parity claim</EvidenceChip>
    </span>
  )
}

function CatalogGroup({ title, marker, items, emptyText, renderRow }) {
  return (
    <section className="catalog-group">
      <div className="catalog-group-head">
        <h3>{title}</h3>
        {marker}
      </div>
      <div className="catalog-list">
        {items.map(renderRow)}
        {items.length === 0 ? <p className="lane-empty">{emptyText}</p> : null}
      </div>
    </section>
  )
}

export function CatalogLaneBrowse({
  apiBase = '',
  capabilities,
  localFilenames = new Set(),
  downloads = [],
  installAvailable = true,
  installBlockedReason = '',
  onInstallStarted,
  onAcquired,
}) {
  const base = (apiBase || '').replace(/\/$/, '')
  const [items, setItems] = useState(null)
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [nextCursor, setNextCursor] = useState(null)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState('')

  // Debounce the query so each keystroke doesn't fire a live Hugging Face search.
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query.trim()), 350)
    return () => clearTimeout(t)
  }, [query])

  const load = useCallback(async () => {
    setError('')
    try {
      const params = debouncedQuery ? `?query=${encodeURIComponent(debouncedQuery)}` : ''
      const res = await fetch(`${base}/api/models/catalog${params}`)
      if (!res.ok) throw new Error(`catalog HTTP ${res.status}`)
      const body = await res.json()
      setItems(body.items || [])
      setNextCursor(body.next_cursor || null)
    } catch (err) {
      setError(String(err?.message || err))
    }
  }, [base, debouncedQuery])

  useEffect(() => {
    load()
  }, [load])

  // Append the next page of experimental (Hugging Face) results.
  const loadMore = useCallback(async () => {
    if (!nextCursor || !debouncedQuery) return
    setLoadingMore(true)
    try {
      const params = `?query=${encodeURIComponent(debouncedQuery)}&cursor=${encodeURIComponent(nextCursor)}`
      const res = await fetch(`${base}/api/models/catalog${params}`)
      if (!res.ok) throw new Error(`catalog HTTP ${res.status}`)
      const body = await res.json()
      const more = (body.items || []).filter((it) => it.group === 'experimental')
      setItems((prev) => {
        const seen = new Set((prev || []).map((it) => it.catalog_id))
        return [...(prev || []), ...more.filter((it) => !seen.has(it.catalog_id))]
      })
      setNextCursor(body.next_cursor || null)
    } catch (err) {
      setError(String(err?.message || err))
    } finally {
      setLoadingMore(false)
    }
  }, [base, debouncedQuery, nextCursor])

  const downloadingNames = new Set(
    downloads.filter((d) => d.status === 'downloading').map((d) => d.filename),
  )
  const renderRow = (item) => (
    <CatalogRow
      key={item.catalog_id}
      item={item}
      capabilities={capabilities}
      installed={localFilenames.has(item.filename)}
      downloading={downloadingNames.has(item.filename)}
      apiBase={base}
      installAvailable={installAvailable}
      installBlockedReason={installBlockedReason}
      onInstallStarted={onInstallStarted}
      onAcquired={onAcquired}
    />
  )

  const curated = (items || []).filter((it) => it.group !== 'experimental')
  const experimental = (items || []).filter((it) => it.group === 'experimental')
  const searching = debouncedQuery.length >= 2

  return (
    <div className="catalog-lane-browse">
      <div className="local-lane-head">
        <h2>Get models</h2>
      </div>
      <p className="local-lane-intro">
        Curated picks are pinned and known-good. Searching also browses live Hugging Face GGUFs as an
        experimental group — those are unverified and carry no parity claim. Downloads are explicit
        and confirmed; progress appears in Downloads above, and the model joins its derived section
        when the file lands.
      </p>
      <input
        className="catalog-search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search curated picks and live Hugging Face GGUFs (name, repo, filename)"
      />
      {error ? (
        <p className="lane-error">
          {items === null ? `Catalog unavailable: ${error}` : error}
        </p>
      ) : null}
      {items === null && !error ? <p className="lane-empty">Loading catalog…</p> : null}

      {items !== null || error ? (
        <CatalogGroup
          title="Curated"
          marker={null}
          items={curated}
          emptyText={debouncedQuery ? 'No curated entries match.' : 'No curated entries available.'}
          renderRow={renderRow}
        />
      ) : null}

      {searching && items !== null ? (
        <>
          <CatalogGroup
            title="Experimental (Hugging Face)"
            marker={<ExperimentalMarker />}
            items={experimental}
            emptyText="No live Hugging Face GGUFs match (or the Hub is unreachable)."
            renderRow={renderRow}
          />
          {nextCursor ? (
            <button
              type="button"
              className="catalog-row-action"
              onClick={loadMore}
              disabled={loadingMore}
            >
              {loadingMore ? 'Loading…' : 'Load more from Hugging Face'}
            </button>
          ) : null}
        </>
      ) : null}
    </div>
  )
}
