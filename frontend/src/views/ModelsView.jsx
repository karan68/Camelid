import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ModelInspector } from '../components/models/ModelInspector'
import { TokenizerPlayground } from '../components/models/TokenizerPlayground'
import { ActiveModelBar } from '../components/models/ActiveModelBar'
import { CatalogLaneBrowse } from '../components/models/CatalogLaneBrowse'
import { DownloadsPanel } from '../components/models/DownloadsPanel'
import { UnsupportedBlocker } from '../components/models/UnsupportedBlocker'
import { Section, SupportedRow, CompatibleRow, EligibleRow, NotAnchoredRow } from '../components/models/LaneRows'
import { Button } from '../components/ui/Button'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import { Notice } from '../components/ui/Notice'
import { useModelsPageData } from '../hooks/useModelsPageData'
import { formatBytes } from '../lib/formatters'
import { bucketByLane } from '../lib/modelLanes'
import { loadLocalModelForChat, modelFilenameFromPath } from '../lib/modelActivation'
import { modelDeleteBlockedReason } from '../lib/modelDeletion'
import { IconClose, IconModels, IconRefresh, IconSearch } from '../components/ui/icons'

/* The Models page: one scroll, five zones.
     1. Active model bar — what is loaded now, with Unload.
     2. Supported — local GGUFs matching an exact supported /api/capabilities row.
     3. Experimental — every other local GGUF, honestly labeled by evidence state.
     4. Downloads — one global live-progress area with cancel.
     5. Get models — curated picks + live Hugging Face search, confirmed downloads.
   Membership everywhere is DERIVED at render time from /api/models/local +
   /api/capabilities (lib/modelLanes); no hand-authored arrays place models, no
   localStorage records claim "downloaded". Diagnostics (tokenizer playground,
   metadata inspector, import-by-path) live in a collapsed disclosure at the end. */

export default function ModelsView({
  runtime,
  capabilities,
  refreshDashboard,
  onOpenChat,
  unloadCurrentModel,
  loadingModelId,
  registerForm,
  setRegisterForm,
  registerModel,
  apiBase = '',
}) {
  const catalogApiBase = (runtime?.api_base || '').replace(/\/$/, '')
  const runtimeOnline = runtime?.status === 'online'
  const catalogInstallAvailable = Boolean(
    capabilities?.model_catalog_install || capabilities?.model_downloads || capabilities?.hf_catalog_install,
  )

  /* Single data spine: /api/models/local + /api/models/current + downloads. */
  const spine = useModelsPageData({ apiBase: catalogApiBase || apiBase })
  const [receipts, setReceipts] = useState({})
  const [smokeBusy, setSmokeBusy] = useState({})
  const [usingFilename, setUsingFilename] = useState('')
  const [unloading, setUnloading] = useState(false)
  const [verification, setVerification] = useState(null)
  const [verificationBusy, setVerificationBusy] = useState(false)
  const verificationRequestRef = useRef(0)
  // Typed fail-closed blocker from a pre-load inspect ({ code, message }), shown
  // verbatim instead of attempting a multi-GB load that cannot run.
  const [blocker, setBlocker] = useState(null)
  const [laneError, setLaneError] = useState('')
  const [cancelingDownloads, setCancelingDownloads] = useState(new Set())
  const [canceledCatalogIds, setCanceledCatalogIds] = useState(new Set())
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const [importing, setImporting] = useState(false)
  const [pendingDeleteEntry, setPendingDeleteEntry] = useState(null)
  const [deletingFilename, setDeletingFilename] = useState('')
  const [defaultingFilename, setDefaultingFilename] = useState('')
  const [deleteNotice, setDeleteNotice] = useState('')
  const [catalogOperations, setCatalogOperations] = useState(new Set())
  const [modelQuery, setModelQuery] = useState('')
  /* Seeds the catalog's own search when a local search comes up empty. */
  const [catalogSeedQuery, setCatalogSeedQuery] = useState('')
  const loadInFlightRef = useRef('')

  const laneBuckets = useMemo(
    () => (spine.local ? bucketByLane(spine.local.models, capabilities) : null),
    [spine.local, capabilities],
  )
  const activeEntry = useMemo(
    () => spine.local?.models.find((m) => m.filename === spine.activeFilename) || null,
    [spine.local, spine.activeFilename],
  )
  const ghostMoePreparedFilenames = useMemo(
    () => new Set((spine.local?.models || []).filter((model) => model.ghost_moe_prepared).map((model) => model.filename)),
    [spine.local],
  )
  const allExperimentalRows = laneBuckets
    ? [...laneBuckets.compatible, ...laneBuckets.eligible, ...laneBuckets.not_anchored]
    : []

  /* Name filter for the models already on this machine. It searches the display
     name AND the filename, because a GGUF's file name is often the only place
     the quantization appears — someone looking for "q4" means the file, not the
     model family. The catalog below keeps its own search: that one reaches the
     network to find models you do NOT have, which is a different question. */
  const matchesModelQuery = (model) => {
    const needle = modelQuery.trim().toLowerCase()
    if (!needle) return true
    return `${model?.name || ''} ${model?.filename || ''}`.toLowerCase().includes(needle)
  }
  const supportedRows = (laneBuckets ? laneBuckets.supported : []).filter(matchesModelQuery)
  const experimentalRows = allExperimentalRows.filter(matchesModelQuery)
  const filteringModels = Boolean(modelQuery.trim())
  const localMatchCount = supportedRows.length + experimentalRows.length
  const deleteBlockedReason = modelDeleteBlockedReason({
    activeFilename: spine.activeFilename,
    downloads: spine.downloads,
    loading: Boolean(usingFilename || loadingModelId || importing || unloading),
    smoking: Object.values(smokeBusy).some(Boolean) || catalogOperations.size > 0,
  })

  const setCatalogOperationBusy = useCallback((catalogId, busy) => {
    setCatalogOperations((current) => {
      const next = new Set(current)
      if (busy) next.add(catalogId)
      else next.delete(catalogId)
      return next
    })
  }, [])

  // Load a local model into the chat backend. The HTTP sequence itself (header-only
  // inspect -> authoritative load -> identity + readiness confirmation) lives in
  // lib/modelActivation so this page and the first-run card cannot drift; what stays
  // here is the page's own state wiring. The spine's `/api/models/current` refresh
  // answers the identity check, so the confirmation costs no extra request.
  const loadModelForChat = async (filename, { onStage } = {}) => {
    if (loadInFlightRef.current) {
      const message = loadInFlightRef.current === filename
        ? `${filename} is already loading.`
        : `Wait for ${loadInFlightRef.current} to finish loading, then retry.`
      setLaneError(message)
      return { ok: false, stage: 'loading', message }
    }
    loadInFlightRef.current = filename
    setUsingFilename(filename)
    setLaneError('')
    setBlocker(null)
    try {
      const result = await loadLocalModelForChat({
        apiBase: spine.base,
        filename,
        onStage,
        readActiveFilename: async () => modelFilenameFromPath((await spine.refreshCurrent())?.path),
      })
      if (!result.ok) {
        if (result.blocker) setBlocker(result.blocker)
        setLaneError(result.message)
        return result
      }
      await refreshDashboard?.({ silent: true })
      return result
    } finally {
      if (loadInFlightRef.current === filename) loadInFlightRef.current = ''
      setUsingFilename('')
    }
  }

  const handleUnload = async () => {
    setUnloading(true)
    try {
      await unloadCurrentModel()
      await spine.refreshCurrent()
    } finally {
      setUnloading(false)
    }
  }

  const refreshVerification = async () => {
    const requestId = ++verificationRequestRef.current
    if (!spine.activeFilename) {
      setVerification(null)
      return
    }
    try {
      const res = await fetch(`${spine.base}/api/models/verify`)
      const next = res.ok ? await res.json() : null
      if (requestId === verificationRequestRef.current) setVerification(next)
    } catch {
      if (requestId === verificationRequestRef.current) setVerification(null)
    }
  }

  useEffect(() => {
    refreshVerification()
  }, [spine.activeFilename, spine.base])

  const runVerification = async () => {
    setVerificationBusy(true)
    setLaneError('')
    try {
      const res = await fetch(`${spine.base}/api/models/verify`, { method: 'POST' })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(body?.error?.message || `verification failed (HTTP ${res.status})`)
      await refreshVerification()
    } catch (err) {
      setLaneError(String(err?.message || err))
    } finally {
      setVerificationBusy(false)
    }
  }

  const cancelDownloadById = async (id) => {
    setCancelingDownloads((s) => new Set([...s, id]))
    try {
      const canceled = await spine.cancelDownload(id)
      if (canceled) {
        setCanceledCatalogIds((current) => new Set([...current, id]))
      }
    } finally {
      setCancelingDownloads((s) => {
        const next = new Set(s)
        next.delete(id)
        return next
      })
    }
  }

  const clearCanceledDownload = (catalogId) => {
    setCanceledCatalogIds((current) => {
      if (!current.has(catalogId)) return current
      const next = new Set(current)
      next.delete(catalogId)
      return next
    })
  }

  const requestDeleteModel = (entry) => {
    if (deleteBlockedReason) {
      setLaneError(deleteBlockedReason)
      return
    }
    setLaneError('')
    setDeleteNotice('')
    setPendingDeleteEntry(entry)
  }

  const makeDefaultModel = async (filename) => {
    setDefaultingFilename(filename)
    setLaneError('')
    setDeleteNotice('')
    try {
      await spine.setDefaultModel(filename)
      setDeleteNotice(`${filename} will load automatically when Camelid starts.`)
    } catch (error) {
      setLaneError(String(error?.message || error))
    } finally {
      setDefaultingFilename('')
    }
  }

  useEffect(() => {
    if (pendingDeleteEntry && deleteBlockedReason && !deletingFilename) {
      setPendingDeleteEntry(null)
      setLaneError(deleteBlockedReason)
    }
  }, [deleteBlockedReason, deletingFilename, pendingDeleteEntry])

  const deleteModelFromDisk = async () => {
    if (!pendingDeleteEntry || deletingFilename) return
    if (deleteBlockedReason) {
      setPendingDeleteEntry(null)
      setLaneError(deleteBlockedReason)
      return
    }
    const entry = pendingDeleteEntry
    setDeletingFilename(entry.filename)
    setLaneError('')
    try {
      const result = await spine.deleteLocalModel(entry)
      setReceipts((current) => {
        const next = { ...current }
        delete next[entry.filename]
        return next
      })
      setPendingDeleteEntry(null)
      setDeleteNotice(result.bytes_freed
        ? `Deleted ${entry.filename} and freed ${formatBytes(result.bytes_freed)}.`
        : `Deleted ${entry.filename}.`)
    } catch (error) {
      setPendingDeleteEntry(null)
      setLaneError(String(error?.message || error))
    } finally {
      setDeletingFilename('')
    }
  }

  const runSmoke = async (filename) => {
    setSmokeBusy((b) => ({ ...b, [filename]: true }))
    setLaneError('')
    try {
      const res = await fetch(`${spine.base}/api/models/runnable-smoke`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      })
      const body = await res.json()
      if (res.ok && body.passed) {
        setReceipts((r) => ({ ...r, [filename]: body.receipt }))
        await spine.refreshLocal()
      } else {
        setLaneError(body?.error?.message || `The quick test did not pass for ${filename}.`)
      }
    } catch (err) {
      setLaneError(String(err?.message || err))
    } finally {
      setSmokeBusy((b) => ({ ...b, [filename]: false }))
    }
  }

  // Pull the runnable receipt for each Compatible model (those that passed smoke).
  useEffect(() => {
    if (!spine.local) return
    spine.local.models
      .filter((m) => m.runnable_receipt_present && !receipts[m.filename])
      .forEach(async (m) => {
        try {
          const res = await fetch(
            `${spine.base}/api/models/runnable-receipt?filename=${encodeURIComponent(m.filename)}`,
          )
          if (res.ok) {
            const receipt = await res.json()
            setReceipts((r) => ({ ...r, [m.filename]: receipt }))
          }
        } catch {
          /* receipt is best-effort; the row still renders */
        }
      })
  }, [spine.local, spine.base, receipts])

  const importFromPath = async () => {
    setImporting(true)
    try {
      await registerModel()
      await spine.refreshAll()
    } finally {
      setImporting(false)
    }
  }

  return (
    <section className="models-view cxv">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconModels size={14} /> Model support</p>
          <h1>Models</h1>
          <p className="cxv-sub">
            Load, download, and manage the models on this machine.
          </p>
        </div>
        <div className="cxv-head__actions models-head-actions">
          <label className="cxv-search models-head-search">
            <IconSearch size={17} />
            <input
              value={modelQuery}
              onChange={(event) => setModelQuery(event.target.value)}
              placeholder="Search models by name"
              aria-label="Search models on this machine by name"
              type="search"
            />
            {modelQuery && (
              <button
                type="button"
                className="cxv-search__clear"
                aria-label="Clear model search"
                onClick={() => setModelQuery('')}
              >
                <IconClose size={14} />
              </button>
            )}
          </label>
          <Button
            variant="outline"
            size="sm"
            icon={<IconRefresh size={16} />}
            onClick={() => spine.refreshAll()}
            disabled={spine.localLoading}
          >
            {spine.localLoading ? 'Refreshing…' : 'Refresh'}
          </Button>
        </div>
      </header>

      {/* One result line for the whole page rather than a changed empty state in
          each section: the filter spans Supported and Experimental, so saying it
          once is clearer than saying it twice. A search that matches nothing
          locally is the moment someone is most likely looking for a model they
          do not have yet, so it hands the term to the catalog instead of just
          reporting failure. */}
      {filteringModels && (
        <div className="models-filter-summary" role="status">
          {localMatchCount > 0 ? (
            <span>
              {localMatchCount} model{localMatchCount === 1 ? '' : 's'} on this machine match{localMatchCount === 1 ? 'es' : ''}{' '}
              <strong>{modelQuery.trim()}</strong>
            </span>
          ) : (
            <span>No model on this machine matches <strong>{modelQuery.trim()}</strong>.</span>
          )}
          <div className="models-filter-summary__actions">
            {localMatchCount === 0 && (
              <Button
                variant="tonal"
                size="sm"
                icon={<IconSearch size={15} />}
                onClick={() => {
                  setCatalogSeedQuery(modelQuery.trim())
                  document.querySelector('.catalog-lane-browse')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                }}
              >
                Look for it in Get models
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={() => setModelQuery('')}>Clear</Button>
          </div>
        </div>
      )}

      {/* Zone 1 — active model bar */}
      <ActiveModelBar
        runtime={runtime}
        activeFilename={spine.activeFilename}
        activeEntry={activeEntry}
        capabilities={capabilities}
        busy={unloading || verificationBusy || Boolean(loadingModelId)}
        unloading={unloading}
        verification={verification}
        verificationBusy={verificationBusy}
        onVerify={runVerification}
        onUnload={handleUnload}
      />
      <Notice notice={laneError} tone="error" onDismiss={() => setLaneError('')} />
      <Notice notice={deleteNotice} tone="success" onDismiss={() => setDeleteNotice('')} />
      {deleteBlockedReason ? (
        <p className="lane-delete-guard" id="model-delete-guard">{deleteBlockedReason}</p>
      ) : null}
      {spine.localError && !spine.local ? (
        <Notice notice={`Could not list local models: ${spine.localError}`} tone="error" />
      ) : null}

      {/* Zone 2 — supported local models (derived membership only) */}
      <Section
        title="Supported"
        count={laneBuckets ? supportedRows.length : undefined}
        subtitle="Verified to run correctly here."
      >
        {!laneBuckets ? (
          <p className="lane-empty">
            {spine.localLoading ? 'Scanning local models…' : runtimeOnline ? 'Local model scan unavailable.' : 'Runtime offline — the local scan resumes when the backend is back.'}
          </p>
        ) : supportedRows.length ? (
          supportedRows.map((m) => (
            <SupportedRow
              key={m.filename}
              entry={m}
              active={m.filename === spine.activeFilename}
              busy={usingFilename === m.filename}
              deleteBusy={deletingFilename === m.filename}
              defaultBusy={defaultingFilename === m.filename}
              isDefault={spine.defaultFilename === m.filename}
              blockedReason={deleteBlockedReason}
              onUse={() => loadModelForChat(m.filename)}
              onDelete={requestDeleteModel}
              onMakeDefault={makeDefaultModel}
            />
          ))
        ) : (
          <p className="lane-empty">{filteringModels ? 'No verified model matches this search.' : 'No verified models on this machine yet — download one below in “Get models”.'}</p>
        )}
      </Section>

      {/* Zone 3 — everything else local, honestly labeled by evidence state */}
      <Section
        title="Experimental"
        count={laneBuckets ? experimentalRows.length : undefined}
        subtitle="These run, but their output isn't verified."
      >
        {blocker ? <UnsupportedBlocker blocker={blocker} className="local-lane-blocker" /> : null}
        {!laneBuckets ? (
          <p className="lane-empty">
            {spine.localLoading ? 'Scanning local models…' : runtimeOnline ? 'Local model scan unavailable.' : 'Runtime offline — the local scan resumes when the backend is back.'}
          </p>
        ) : experimentalRows.length ? (
          <>
            {laneBuckets.compatible.map((m) => (
              <CompatibleRow
                key={m.filename}
                entry={m}
                receipt={receipts[m.filename]}
                busy={usingFilename === m.filename}
                deleteBusy={deletingFilename === m.filename}
                defaultBusy={defaultingFilename === m.filename}
                isDefault={spine.defaultFilename === m.filename}
                blockedReason={deleteBlockedReason}
                onUse={() => loadModelForChat(m.filename)}
                onDelete={requestDeleteModel}
                onMakeDefault={makeDefaultModel}
              />
            ))}
            {laneBuckets.eligible.map((m) => (
              <EligibleRow
                key={m.filename}
                entry={m}
                busy={Boolean(smokeBusy[m.filename])}
                deleteBusy={deletingFilename === m.filename}
                blockedReason={deleteBlockedReason}
                onRun={() => runSmoke(m.filename)}
                onDelete={requestDeleteModel}
              />
            ))}
            {laneBuckets.not_anchored.map((m) => (
              <NotAnchoredRow
                key={m.filename}
                entry={m}
                busy={usingFilename === m.filename}
                deleteBusy={deletingFilename === m.filename}
                defaultBusy={defaultingFilename === m.filename}
                isDefault={spine.defaultFilename === m.filename}
                blockedReason={deleteBlockedReason}
                onUse={() => loadModelForChat(m.filename)}
                onDelete={requestDeleteModel}
                onMakeDefault={makeDefaultModel}
              />
            ))}
          </>
        ) : (
          <p className="lane-empty">{filteringModels ? 'No experimental model matches this search.' : 'Nothing experimental on this machine — every downloaded model is verified.'}</p>
        )}
      </Section>

      {/* Zone 4 — downloads in progress (global; hidden while idle) */}
      <DownloadsPanel
        downloads={spine.downloads}
        cancelingIds={cancelingDownloads}
        onCancel={cancelDownloadById}
      />

      {/* Zone 5 — get models: curated picks + live Hugging Face search */}
      <CatalogLaneBrowse
        seedQuery={catalogSeedQuery}
        apiBase={catalogApiBase || apiBase}
        capabilities={capabilities}
        localFilenames={spine.localFilenames}
        ghostMoePreparedFilenames={ghostMoePreparedFilenames}
        downloads={spine.downloads}
        installAvailable={runtimeOnline && catalogInstallAvailable}
        installBlockedReason={
          !runtimeOnline
            ? 'The runtime is offline — start the Camelid backend to download models.'
            : 'This backend does not support model downloads.'
        }
        onInstallStarted={spine.kickDownloadsPoll}
        onDownloadAcknowledged={spine.refreshDownloads}
        onAcquired={spine.refreshLocalAndDefault}
        canceledCatalogIds={canceledCatalogIds}
        onDownloadRetry={clearCanceledDownload}
        onStartModel={loadModelForChat}
        onModelStarted={onOpenChat}
        onOperationBusy={setCatalogOperationBusy}
      />

      {/* Diagnostics — operator tools, collapsed by default. Import-by-path lives
          here because it is the only way to load a GGUF stored outside models/. */}
      <details className="models-diagnostics">
        <summary>Diagnostics</summary>
        <div className="models-diagnostics__body">
          <div className="models-diagnostics__tools">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setInspectorOpen(true)}
              title="View the loaded model's metadata, tokenizer, and tensors"
            >
              Inspect loaded model metadata
            </Button>
          </div>

          <div className="models-diagnostics__import">
            <h3>Import a GGUF by path</h3>
            <p className="lane-empty">
              Models inside the <code>models/</code> folder appear above automatically. Use this only
              for a model file stored elsewhere; it loads immediately and is verified the same way as
              the models above.
            </p>
            <div className="models-diagnostics__import-grid">
              <input
                value={registerForm.name}
                onChange={(e) => setRegisterForm((form) => ({ ...form, name: e.target.value }))}
                placeholder="Model name"
              />
              <input
                value={registerForm.model_path}
                onChange={(e) => setRegisterForm((form) => ({ ...form, model_path: e.target.value }))}
                placeholder="/path/to/your-model.gguf"
              />
              <Button
                variant="tonal"
                size="sm"
                onClick={importFromPath}
                loading={importing || Boolean(loadingModelId)}
                disabled={importing || Boolean(loadingModelId)}
              >
                Import and load
              </Button>
            </div>
          </div>

          <TokenizerPlayground apiBase={catalogApiBase || apiBase} />
        </div>
      </details>

      {inspectorOpen && (
        <ModelInspector apiBase={catalogApiBase || apiBase} onClose={() => setInspectorOpen(false)} />
      )}

      <ConfirmDialog
        open={Boolean(pendingDeleteEntry)}
        title={pendingDeleteEntry ? `Delete ${pendingDeleteEntry.filename}?` : 'Delete model?'}
        detail={pendingDeleteEntry
          ? `This permanently removes ${pendingDeleteEntry.size_bytes ? formatBytes(pendingDeleteEntry.size_bytes) : 'this file'}${pendingDeleteEntry.ghost_moe_prepared ? ' and its Ghost MoE expert pack' : ''} from disk. This cannot be undone.`
          : ''}
        confirmLabel="Delete model"
        busy={Boolean(deletingFilename)}
        onCancel={() => { if (!deletingFilename) setPendingDeleteEntry(null) }}
        onConfirm={deleteModelFromDisk}
      />
    </section>
  )
}
