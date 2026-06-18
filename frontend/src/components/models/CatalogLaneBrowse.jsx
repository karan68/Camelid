import { useCallback, useEffect, useRef, useState } from 'react'
import { isCompatibilitySupportedForModel } from '../../lib/capabilities'
import { EvidenceChip } from '../ui/EvidenceChip'

/* Acquire known GGUFs from HuggingFace. Each entry shows which lane it WOULD land in
   (derived: supported contract match, oracle-qualified runnable, or not-yet-anchored).
   Download is user-initiated and explicitly confirmed (filename + HF repo + size); no
   background/auto pulls. After a download completes we run smoke-admission, and the
   model then appears in its derived local section. */

const GB = 1024 * 1024 * 1024
function prettySize(bytes) {
  if (!bytes) return ''
  if (bytes >= GB) return `${(bytes / GB).toFixed(bytes >= 10 * GB ? 0 : 1)} GB`
  return `${Math.round(bytes / (1024 * 1024))} MB`
}

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
  if (lane === 'supported') return <EvidenceChip status="supported" asText>Supported lane</EvidenceChip>
  if (lane === 'compatible') return <EvidenceChip state="runnable" asText>Runnable lane</EvidenceChip>
  return <EvidenceChip state="unsupported" asText>Not yet in a lane</EvidenceChip>
}

function CatalogRow({ item, capabilities, installed, apiBase, onAcquired }) {
  // phase: idle | confirm | installing | smoking | done | error
  const [phase, setPhase] = useState('idle')
  const [progress, setProgress] = useState(null) // { bytes, total }
  const [message, setMessage] = useState('')
  const pollRef = useRef(null)
  const lane = predictedLane(item, capabilities)

  useEffect(() => () => clearInterval(pollRef.current), [])

  const pollUntilDone = useCallback(async () => {
    return new Promise((resolve) => {
      pollRef.current = setInterval(async () => {
        try {
          const res = await fetch(`${apiBase}/api/models/catalog/downloads`)
          const list = res.ok ? await res.json() : []
          const dl = list.find((d) => d.filename === item.filename)
          if (dl) {
            setProgress({ bytes: dl.bytes_downloaded, total: dl.total_bytes || item.size_bytes })
            if (dl.status === 'failed') {
              clearInterval(pollRef.current)
              resolve(false)
            }
          } else {
            // No longer downloading — confirm the file actually landed on disk.
            clearInterval(pollRef.current)
            const localRes = await fetch(`${apiBase}/api/models/local`)
            const local = localRes.ok ? await localRes.json() : { models: [] }
            resolve(local.models.some((m) => m.filename === item.filename))
          }
        } catch {
          /* transient; keep polling */
        }
      }, 1500)
    })
  }, [apiBase, item.filename, item.size_bytes])

  const confirmDownload = async () => {
    setPhase('installing')
    setMessage('')
    setProgress({ bytes: 0, total: item.size_bytes })
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
      const completed = await pollUntilDone()
      if (!completed) throw new Error('download did not complete')

      // After download: smoke-admission only applies to oracle-qualified combos. For
      // everything else the file just lands on disk — a machine with the right
      // support lane can still run it; we don't gate the download on local hardware.
      if (item.oracle_qualified) {
        setPhase('smoking')
        const smoke = await fetch(`${apiBase}/api/models/runnable-smoke`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: item.filename }),
        })
        const body = await smoke.json().catch(() => ({}))
        setPhase('done')
        setMessage(
          smoke.ok && body.passed
            ? 'Downloaded and smoke-admitted — see it above in its lane section.'
            : body?.error?.message
              ? `Downloaded. Smoke-admission did not pass here: ${body.error.message}`
              : 'Downloaded. Smoke-admission did not pass on this machine — the file is on disk.',
        )
      } else {
        setPhase('done')
        setMessage('Downloaded — on disk now. Not in the runnable lane; a machine with the right support lane can run it.')
      }
      onAcquired?.()
    } catch (err) {
      setPhase('error')
      setMessage(String(err?.message || err))
    }
  }

  const pct = progress && progress.total ? Math.min(100, Math.round((progress.bytes / progress.total) * 100)) : 0

  return (
    <article className={`catalog-row${lane === 'not_anchored' ? ' catalog-row--advisory' : ''}`}>
      <div className="catalog-row-head">
        <div className="catalog-row-id">
          <span className="catalog-row-name">{item.name}</span>
          <span className="catalog-row-meta">
            {item.repo_id} · {item.filename} · {prettySize(item.size_bytes)}
            {item.architecture ? ` · ${item.architecture}` : ''}
          </span>
        </div>
        {laneChip(lane)}
      </div>

      {installed ? (
        <p className="catalog-row-faint">Already on disk — shown in its lane section above.</p>
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
              downloadable; a machine with the right support lane can run it.
            </p>
          ) : null}
          <button type="button" className="catalog-row-action" onClick={() => setPhase('confirm')}>
            Download…
          </button>
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
      ) : phase === 'installing' ? (
        <button
          type="button"
          className={`catalog-row-action catalog-row-action--progress${pct === 0 ? ' is-indeterminate' : ''}`}
          disabled
          aria-label={`Downloading, ${pct} percent`}
          aria-busy="true"
        >
          <span className="catalog-row-action__fill" style={{ width: `${pct}%` }} aria-hidden="true" />
          <span className="catalog-row-action__label">
            Downloading {pct}% · {prettySize(progress?.bytes)} / {prettySize(progress?.total)}
          </span>
        </button>
      ) : phase === 'smoking' ? (
        <p className="catalog-row-faint">Download complete — running smoke-admission…</p>
      ) : (
        <p className={phase === 'error' ? 'catalog-row-error' : 'catalog-row-faint'}>{message}</p>
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

function CatalogGroup({ title, marker, items, capabilities, localNames, base, onAcquired, emptyText }) {
  return (
    <section className="catalog-group">
      <div className="catalog-group-head">
        <h3>{title}</h3>
        {marker}
      </div>
      <div className="catalog-list">
        {items.map((item) => (
          <CatalogRow
            key={item.catalog_id}
            item={item}
            capabilities={capabilities}
            installed={localNames.has(item.filename)}
            apiBase={base}
            onAcquired={onAcquired}
          />
        ))}
        {items.length === 0 ? <p className="lane-empty">{emptyText}</p> : null}
      </div>
    </section>
  )
}

export function CatalogLaneBrowse({ apiBase = '', capabilities, onAcquired }) {
  const base = (apiBase || '').replace(/\/$/, '')
  const [items, setItems] = useState(null)
  const [localNames, setLocalNames] = useState(new Set())
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
      const [cat, local] = await Promise.all([
        fetch(`${base}/api/models/catalog${params}`),
        fetch(`${base}/api/models/local`),
      ])
      if (!cat.ok) throw new Error(`catalog HTTP ${cat.status}`)
      const catBody = await cat.json()
      setItems(catBody.items || [])
      setNextCursor(catBody.next_cursor || null)
      if (local.ok) {
        const lb = await local.json()
        setLocalNames(new Set((lb.models || []).map((m) => m.filename)))
      }
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

  if (items === null && !error) return <p className="lane-empty">Loading catalog…</p>

  const curated = (items || []).filter((it) => it.group !== 'experimental')
  const experimental = (items || []).filter((it) => it.group === 'experimental')
  const searching = debouncedQuery.length >= 2

  return (
    <div className="catalog-lane-browse">
      <div className="local-lane-head">
        <h2>Catalog — acquire from HuggingFace</h2>
      </div>
      <p className="local-lane-intro">
        Curated rows are pinned and known-good. Searching also browses live Hugging Face GGUFs as an
        experimental group — those are unverified and carry no parity claim. Downloads are explicit
        and confirmed; after a download we run smoke-admission and the model joins its lane section
        above.
      </p>
      <input
        className="catalog-search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search curated rows and live Hugging Face GGUFs (name, repo, filename)"
      />
      {error ? <p className="lane-error">{error}</p> : null}

      <CatalogGroup
        title="Curated"
        marker={null}
        items={curated}
        capabilities={capabilities}
        localNames={localNames}
        base={base}
        onAcquired={onAcquired}
        emptyText="No curated entries match."
      />

      {searching ? (
        <>
          <CatalogGroup
            title="Experimental (Hugging Face)"
            marker={<ExperimentalMarker />}
            items={experimental}
            capabilities={capabilities}
            localNames={localNames}
            base={base}
            onAcquired={onAcquired}
            emptyText="No live Hugging Face GGUFs match (or the Hub is unreachable)."
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
