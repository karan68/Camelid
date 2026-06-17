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
          {lane === 'not_anchored' ? (
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
        <div className="catalog-progress">
          <div className="catalog-progress-bar">
            <span style={{ width: `${pct}%` }} />
          </div>
          <small>
            Downloading {prettySize(progress?.bytes)} / {prettySize(progress?.total)} ({pct}%)
          </small>
        </div>
      ) : phase === 'smoking' ? (
        <p className="catalog-row-faint">Download complete — running smoke-admission…</p>
      ) : (
        <p className={phase === 'error' ? 'catalog-row-error' : 'catalog-row-faint'}>{message}</p>
      )}
    </article>
  )
}

export function CatalogLaneBrowse({ apiBase = '', capabilities, onAcquired }) {
  const base = (apiBase || '').replace(/\/$/, '')
  const [items, setItems] = useState(null)
  const [localNames, setLocalNames] = useState(new Set())
  const [query, setQuery] = useState('')
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    setError('')
    try {
      const params = query ? `?query=${encodeURIComponent(query)}` : ''
      const [cat, local] = await Promise.all([
        fetch(`${base}/api/models/catalog${params}`),
        fetch(`${base}/api/models/local`),
      ])
      if (!cat.ok) throw new Error(`catalog HTTP ${cat.status}`)
      const catBody = await cat.json()
      setItems(catBody.items || [])
      if (local.ok) {
        const lb = await local.json()
        setLocalNames(new Set((lb.models || []).map((m) => m.filename)))
      }
    } catch (err) {
      setError(String(err?.message || err))
    }
  }, [base, query])

  useEffect(() => {
    load()
  }, [load])

  if (items === null && !error) return <p className="lane-empty">Loading catalog…</p>

  return (
    <div className="catalog-lane-browse">
      <div className="local-lane-head">
        <h2>Catalog — acquire from HuggingFace</h2>
      </div>
      <p className="local-lane-intro">
        Each entry shows which lane it would land in. Downloads are explicit and confirmed; after a
        download we run smoke-admission and the model joins its lane section above.
      </p>
      <input
        className="catalog-search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Filter by name, repo, or filename"
      />
      {error ? <p className="lane-error">{error}</p> : null}
      <div className="catalog-list">
        {(items || []).map((item) => (
          <CatalogRow
            key={item.catalog_id}
            item={item}
            capabilities={capabilities}
            installed={localNames.has(item.filename)}
            apiBase={base}
            onAcquired={onAcquired}
          />
        ))}
        {items && items.length === 0 ? <p className="lane-empty">No catalog entries match.</p> : null}
      </div>
    </div>
  )
}
