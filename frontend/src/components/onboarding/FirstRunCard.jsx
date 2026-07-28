import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  beginCatalogSettlement,
  catalogDownloadSettlement,
  completeCatalogAcquisition,
} from '../../lib/catalogActivation'
import { findCompatibilityHint } from '../../lib/capabilities'
import { firstRunFailureIsRetryable, recommendFirstRunModel } from '../../lib/firstRunActivation'
import { formatBytes } from '../../lib/formatters'
import { loadLocalModelForChat, warmGenerationPath } from '../../lib/modelActivation'
import { EvidenceChip } from '../ui/EvidenceChip'

/* First-run activation — one click from a fresh install to a streaming token.

   A new install starts with an empty models folder and no loaded model, so the chat
   surface it lands on can do nothing at all. Everything needed to fix that already
   existed (a curated catalog with per-host fit verdicts, a background downloader, a
   load protocol); what was missing was one place that puts them behind one sentence
   and one button.

   This card owns exactly that path and nothing else:
     download → land → check → load → warm → Chat.
   The download stays cancellable throughout, the rest of the app stays usable, and
   the offer itself is derived (see lib/firstRunActivation.js) so it can never drift
   into recommending an unverified row. */

/* One combined poll drives settlement: `/api/models/catalog/downloads` says whether
   bytes are still moving, `/api/models/local` says whether the file actually landed.
   Neither alone is enough — the download record disappears the moment the fetcher
   exits, and the file is only loadable once it is promoted off its `.part` name.
   Same cadence as the Models page so the two can never disagree about one download. */
const SETTLEMENT_POLL_MS = 1000

const STAGE_COPY = {
  starting: 'Starting the download…',
  downloading: 'Downloading…',
  checking: 'Downloaded — checking the model…',
  loading: 'Check passed — loading the model…',
  warming: 'Loaded — warming up so your first message is fast…',
}

const STEPS = ['Download', 'Check', 'Load']
const IN_FLIGHT_PHASES = new Set(['starting', 'downloading', 'checking', 'loading', 'warming'])
const CANCELLABLE_PHASES = new Set(['starting', 'downloading'])

function stepIndex(phase) {
  if (phase === 'checking') return 1
  if (phase === 'loading' || phase === 'warming') return 2
  return 0
}

function percentOf(download) {
  if (!download?.total_bytes) return null
  const value = Math.round((download.bytes_downloaded / download.total_bytes) * 100)
  return Math.max(0, Math.min(100, value))
}

export function FirstRunCard({ apiBase = '', capabilities = null, onActivated, onBusyChange, onOpenModels }) {
  const base = String(apiBase || '').replace(/\/$/, '')
  // phase: idle | starting | downloading | checking | loading | warming | failed | done
  const [phase, setPhase] = useState('idle')
  const [catalogItems, setCatalogItems] = useState(null)
  const [catalogFailed, setCatalogFailed] = useState(false)
  const [download, setDownload] = useState(null)
  const [failure, setFailure] = useState(null)
  const [canceling, setCanceling] = useState(false)
  const [settlementTick, setSettlementTick] = useState(0)

  const itemRef = useRef(null)
  const sawDownloadRef = useRef(false)
  const startedAtRef = useRef(0)
  const settledAtRef = useRef(0)
  const settlementInFlightRef = useRef(false)
  const installedRef = useRef(false)
  const loadResultRef = useRef(null)

  /* Fetch the catalog once per backend. Keyed on `base` only: `capabilities` is a
     fresh object on every dashboard refresh, so depending on it here would re-fetch
     the catalog on a timer. The offer is derived from both below instead. */
  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false
    setCatalogItems(null)
    setCatalogFailed(false)
    ;(async () => {
      try {
        const res = await fetch(`${base}/api/models/catalog`, { signal: controller.signal })
        if (!res.ok) throw new Error(`catalog HTTP ${res.status}`)
        const body = await res.json()
        if (cancelled) return
        setCatalogItems(body?.items || [])
      } catch (error) {
        if (cancelled || error?.name === 'AbortError') return
        setCatalogItems([])
        setCatalogFailed(true)
      }
    })()
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [base])

  const recommendation = useMemo(
    () => (catalogItems === null ? null : recommendFirstRunModel(catalogItems, capabilities)),
    [catalogItems, capabilities],
  )
  const item = recommendation?.kind === 'recommended' ? recommendation.item : null

  /* Report in-flight work upward so the owner keeps this card mounted even once the
     downloaded file makes the host stop looking like a fresh install. Dropping the
     component mid-activation would abandon the download's only watcher. */
  const busy = IN_FLIGHT_PHASES.has(phase)
  useEffect(() => {
    onBusyChange?.(busy)
  }, [busy, onBusyChange])
  useEffect(() => () => onBusyChange?.(false), [onBusyChange])

  /* Rejoin a download this backend is already running for the recommended row.
     A reload, or a first render that lands after the install request, must not leave
     a real download with nobody watching it — the file would arrive and never load. */
  useEffect(() => {
    if (!item || phase !== 'idle') return undefined
    let cancelled = false
    ;(async () => {
      const downloads = await fetch(`${base}/api/models/catalog/downloads`)
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null)
      if (cancelled || !Array.isArray(downloads)) return
      const existing = downloads.find((entry) => entry.id === item.catalog_id)
      if (!existing || (existing.status !== 'downloading' && existing.status !== 'completed')) return
      itemRef.current = item
      sawDownloadRef.current = true
      startedAtRef.current = Date.now()
      settledAtRef.current = 0
      installedRef.current = false
      settlementInFlightRef.current = false
      setDownload(existing)
      setPhase('downloading')
    })()
    return () => {
      cancelled = true
    }
  }, [base, item, phase])

  // Watch the shared download list + the disk scan while a download is in flight.
  useEffect(() => {
    if (phase !== 'downloading') return undefined
    let cancelled = false
    const tick = async () => {
      const [downloads, local] = await Promise.all([
        fetch(`${base}/api/models/catalog/downloads`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
        fetch(`${base}/api/models/local`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
      ])
      if (cancelled) return
      const target = itemRef.current
      setDownload(Array.isArray(downloads) ? downloads.find((entry) => entry.id === target?.catalog_id) || null : null)
      if (local?.models) {
        installedRef.current = local.models.some((model) => model.filename === target?.filename)
      }
      setSettlementTick((value) => value + 1)
    }
    tick()
    const timer = setInterval(tick, SETTLEMENT_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [base, phase])

  const fail = useCallback((message, code = '') => {
    setFailure({ message, code })
    setPhase('failed')
  }, [])

  const activate = useCallback(async () => {
    /* Re-entrancy is real: the settlement effect re-runs on every poll tick and a
       landed file satisfies `landed` on all of them. */
    if (!beginCatalogSettlement(settlementInFlightRef)) return
    loadResultRef.current = null
    const target = itemRef.current
    const result = await completeCatalogAcquisition({
      item: target,
      mode: 'start',
      apiBase: base,
      loadModelForChat: async (filename, options) => {
        const loaded = await loadLocalModelForChat({ apiBase: base, filename, ...options })
        loadResultRef.current = loaded
        return loaded
      },
      onStage: setPhase,
    })
    if (!result.ok) {
      settlementInFlightRef.current = false
      fail(result.message, result.code || loadResultRef.current?.code || '')
      return
    }

    /* Retire the download record. The model is already usable, so this failure is
       reported without pretending the activation itself failed. */
    try {
      const ack = await fetch(`${base}/api/models/catalog/ack`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: target.catalog_id }),
      })
      if (!ack.ok) throw new Error(`acknowledgement failed (HTTP ${ack.status})`)
    } catch (error) {
      settlementInFlightRef.current = false
      fail(`${target.filename} is loaded and ready, but Camelid could not finalize the download state: ${String(error?.message || error)}`)
      return
    }

    // Pay the one-time engine build here rather than on the user's first message.
    setPhase('warming')
    await warmGenerationPath({ apiBase: base, modelId: target.filename })
    setPhase('done')
    await onActivated?.(target)
  }, [base, fail, onActivated])

  // Settle: landed → activate; gone without landing → a clean, retryable failure.
  useEffect(() => {
    if (phase !== 'downloading') return
    const settlement = catalogDownloadSettlement({
      downloading: download?.status === 'downloading',
      installed: installedRef.current,
      sawDownload: sawDownloadRef.current,
      settledAt: settledAtRef.current,
      startedAt: startedAtRef.current,
    })
    sawDownloadRef.current = settlement.sawDownload
    settledAtRef.current = settlement.settledAt
    if (settlement.action === 'landed') {
      activate()
      return
    }
    if (settlement.action === 'failed') {
      fail('The download did not complete. It can be started again.')
    }
  }, [activate, download, fail, phase, settlementTick])

  const startDownload = async () => {
    if (!item) return
    itemRef.current = item
    sawDownloadRef.current = false
    startedAtRef.current = Date.now()
    settledAtRef.current = 0
    installedRef.current = false
    settlementInFlightRef.current = false
    setFailure(null)
    setDownload(null)
    setPhase('starting')
    try {
      const res = await fetch(`${base}/api/models/catalog/install`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          catalog_id: item.catalog_id,
          repo_id: item.repo_id,
          filename: item.filename,
          size_bytes: item.size_bytes,
          continuation_mode: 'start',
        }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        // Rejoining a download that is already running is success, not an error.
        if (res.status !== 409 || body?.error?.code !== 'download_already_running') {
          throw new Error(body?.error?.message || `download failed (HTTP ${res.status})`)
        }
      }
      setPhase('downloading')
    } catch (error) {
      fail(String(error?.message || error))
    }
  }

  const cancelDownload = async () => {
    if (!itemRef.current || canceling) return
    setCanceling(true)
    try {
      await fetch(`${base}/api/models/catalog/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: itemRef.current.catalog_id }),
      })
    } catch {
      /* Best-effort: the settlement watcher lands on the same clean, retryable
         state whether or not the cancel request itself was delivered. */
    } finally {
      setCanceling(false)
      /* The file can land while the cancel is in flight. Activation has then already
         claimed the settlement, and reporting "canceled" over a model that is being
         loaded would be a lie — leave that path alone. */
      if (!settlementInFlightRef.current) {
        setDownload(null)
        fail('Download canceled. Nothing was installed — you can start again.')
      }
    }
  }

  // Nothing is claimed until the catalog answers, and the card retires once it wins.
  if (recommendation === null || phase === 'done') return null

  if (!item) {
    const smallest = recommendation.smallest
    return (
      <section className="cxfirstrun cxfirstrun--blocked" aria-labelledby="cxfirstrun-title">
        <p className="cxfirstrun__kicker">Get started</p>
        <h2 className="cxfirstrun__title" id="cxfirstrun-title">Camelid needs a model</h2>
        <p className="cxfirstrun__summary">
          {catalogFailed
            ? 'Camelid could not read its model catalog, so it cannot recommend a first model. The full catalog is in Models.'
            : recommendation.kind === 'no_fitting_row' && smallest
              ? `Every validated model is larger than this machine can load right now — the smallest is ${smallest.name} at ${formatBytes(smallest.size_bytes)}. Close some applications and reload, or browse the full catalog in Models.`
              : recommendation.kind === 'no_fitting_row'
                ? 'No validated model fits this machine right now. Close some applications and reload, or browse the full catalog in Models.'
                : 'This backend advertises no validated model to start from. The full catalog is in Models.'}
        </p>
        <div className="cxfirstrun__actions">
          <button type="button" className="cxfirstrun__secondary" onClick={onOpenModels}>Open Models</button>
        </div>
      </section>
    )
  }

  const hint = findCompatibilityHint(capabilities, null, item)
  const active = stepIndex(phase)
  const percent = percentOf(download)
  const retryable = !failure || firstRunFailureIsRetryable(failure.code)
  const size = formatBytes(item.size_bytes)

  return (
    <section className="cxfirstrun" aria-labelledby="cxfirstrun-title">
      <div className="cxfirstrun__head">
        <div className="cxfirstrun__id">
          <p className="cxfirstrun__kicker">Get started</p>
          <h2 className="cxfirstrun__title" id="cxfirstrun-title">
            Download {item.name} and start chatting
          </h2>
        </div>
        <EvidenceChip
          status={hint?.target?.status || ''}
          state={hint?.target?.status ? null : 'unsupported'}
          label={`${size} download`}
          source={{
            rowId: hint?.target?.id,
            note: 'A validated exact row: Camelid cross-checks this exact file against the reference implementation.',
          }}
          size="sm"
        />
      </div>

      <p className="cxfirstrun__summary">
        Camelid runs models on your own machine, so it needs one on disk first. This is the
        smallest model Camelid has validated that fits here — one {size} download, then it
        loads and Chat is ready. Nothing else to choose.
      </p>
      <p className="cxfirstrun__meta">{item.filename} · {item.repo_id} · {item.license}</p>

      {busy ? (
        <div className="cxfirstrun__progress" role="status" aria-live="polite">
          <ol className="cxfirstrun__steps" aria-label="Setup progress">
            {STEPS.map((label, index) => (
              <li key={label} className={index < active ? 'is-done' : index === active ? 'is-active' : ''}>
                <span aria-hidden="true">{index < active ? '✓' : index + 1}</span>
                {label}
              </li>
            ))}
          </ol>
          {percent === null ? null : (
            <div className="cxfirstrun__bar" role="progressbar" aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100}>
              <span style={{ width: `${percent}%` }} />
            </div>
          )}
          <p className="cxfirstrun__stage">
            {phase === 'downloading' && percent !== null
              ? `Downloading — ${percent}% of ${size}`
              : STAGE_COPY[phase] || 'Working…'}
          </p>
          {CANCELLABLE_PHASES.has(phase) ? (
            <button type="button" className="cxfirstrun__secondary" onClick={cancelDownload} disabled={canceling}>
              {canceling ? 'Canceling…' : 'Cancel download'}
            </button>
          ) : null}
        </div>
      ) : (
        <>
          {failure ? <p className="cxfirstrun__error" role="alert">{failure.message}</p> : null}
          <div className="cxfirstrun__actions">
            {retryable ? (
              <button type="button" className="cxfirstrun__primary" onClick={startDownload}>
                {failure ? 'Try again' : `Download ${size} and start`}
              </button>
            ) : null}
            <button type="button" className="cxfirstrun__secondary" onClick={onOpenModels}>
              {retryable ? 'Choose a different model' : 'Open Models'}
            </button>
          </div>
          <p className="cxfirstrun__fineprint">
            The download comes from Hugging Face into your local models folder. Nothing is
            downloaded until you press the button, and you can cancel at any point.
          </p>
        </>
      )}
    </section>
  )
}
